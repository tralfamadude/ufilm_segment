import json

import ray
from ray.util.queue import Queue
import features
import extract_ocr
import post_model as pm
import ia_util as util
import time
import random
import sys
import os
import cv2
import numpy as np
from imageio import imread, imsave
from dh_segment.io import PAGE
from dh_segment.post_processing import boxes_detection, binarization
import traceback
import hocr
import excerpt as ex

label_bins = []  # used for histogram

# label colors
label_colors = [[0, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [1, 255, 254],
                [255, 166, 254],
                [255, 219, 102],
                [0, 100, 1]]

np.set_printoptions(precision=5)

@ray.remote
class PostProcess:  # actor, CONSUMER of queue
    def __init__(self, work_queue: Queue, finishing_queue: Queue, output_dir: str, ground_file: str, post_model_path: str,
                 post_model_training_data_path: str,
                 enable_debug: bool, production_mode: bool, nclasses: int, hocr_file: str = ""):
        self.work_queue = work_queue
        self.finishing_queue = finishing_queue
        self.output_dir = output_dir
        self.ground_file = ground_file
        self.post_model_path = post_model_path
        self.post_model_training_data_path = post_model_training_data_path
        self.enable_debug = enable_debug
        self.production_mode = production_mode
        self.nclasses = nclasses
        self.start_time_sec = time.time()
        self.count = 0
        self.post_model = None
        self.feat = None
        self.counter = 0
        self.cleaner = util.TextUtil()
        if production_mode:
            # production means using the post model, so we load it here
            # (post_model_training_data_path is not used since it is already inherent in the post model joblib)
            self.feat = features.Features(None, "dummy.data")  # no post model training
            self.post_model = pm.PostModel(post_model_path)
            if len(hocr_file) == 0:
                print("PostProcess must have an hocr file in production mode")
                raise NameError("PostProcess must have an hocr file in production mode")
            self.extractor = extract_ocr.ExtractOCR(hocr_file)
        else:
            # training mode for post model
            self.feat = features.Features(ground_file, post_model_training_data_path)
            self.extractor = None
        for i in range(0, nclasses + 1):
            label_bins.append(i)  # [0] is background, not a real label
        if not production_mode:
            #   make a png to show which colors map to which labels (top bar is background)
            label_color_demo = np.zeros([20 * (nclasses + 1), 200, 3], np.uint8)
            for labeli in range(0, nclasses + 1):
                c = self.label_val_to_color(labeli)
                for h in range(labeli * 20, 20 + (labeli * 20)):
                    for w in range(0, 200):
                        label_color_demo[h, w, 0] = c[0]
                        label_color_demo[h, w, 1] = c[1]
                        label_color_demo[h, w, 2] = c[2]
            imsave(os.path.join(output_dir, "label_demo.png"), label_color_demo)
        # Save txt file
        # with open(os.path.join(output_dir, 'pages.txt'), 'w') as f:
        #    f.write(txt_coordinates)
        self.results_log_path = os.path.join(output_dir, f"info{random.randint(1, 1000)}.log")
        self.results_log = open(self.results_log_path, "a")

    def close(self):
        self.feat.close()
        self.results_log.flush()
        self.results_log.close()

    def _put_results_log(self, s):
        self.results_log.write(f"{s}\n")
        self.results_log.flush()  # ToDo: perhaps not flush for performance reasons

    def get_count(self):
        """
        :return: count of files processed.
        """
        return self.counter

    def get_uptime(self):
        """
        :return: uptime in seconds, as a float.
        """
        return time.time() - self.start_time_sec

    def unpack_bbox(self, info_dict: dict):
        return 2 * info_dict["tag_rect_x0"], 2 * info_dict["tag_rect_y0"], 2 * info_dict["tag_rect_x1"], 2 * info_dict[
            "tag_rect_y1"]

    def label_val_to_color(self, labelv):
        """
        Map a label number to a color.
        :param labelv: label value, 0-Nclasses inclusive where 0 is background.
        :return: [r,g,b]
        """
        return label_colors[labelv]

    def page_make_binary_mask(self, probs: np.ndarray, threshold: float = -1) -> np.ndarray:
        """
        Computes the binary mask of the detected Page from the probabilities output by network
        :param probs: array with values in range [0, 1]
        :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
        :return: binary mask
        """

        mask = binarization.thresholding(probs, threshold)
        mask = binarization.cleaning_binary(mask, kernel_size=5)
        return mask

    def format_quad_to_string(self, quad):
        """
        Formats the corner points into a string.
        :param quad: coordinates of the quadrilateral
        :return:
        """
        s = ''
        for corner in quad:
            s += '{},{},'.format(corner[0], corner[1])
        return s[:-1]

    def emit_debug_summary(self, basename: str, prediction_summary_txt: str):
        #   if debug, emit a txt summary
        if self.enable_debug:  ##############################
            if len(prediction_summary_txt) > 0:
                debug_per_image = os.path.join(self.output_dir, f"{basename}.txt")
                with open(debug_per_image, "a") as f:
                    f.write(f"{prediction_summary_txt}\n")

    def gen_blank_page_msg(self, basename: str) -> None:
        jid, iid, page_number = self.cleaner.parse_page_id(basename)
        excerpt = ex.Excerpt(jid, basename, page_number, page_type=0)
        g = self.finishing_queue.group(jid, iid, "continue", self.output_dir, excerpt, -1)
        self.finishing_queue.push(g)

    async def run(self):
        while True:
            try:
                #
                #                  get page inference off work queue, process it
                #                  get page inference off work queue, process it
                #
                start_wait = time.time()
                g = self.work_queue.pop()
                finish_wait = time.time()
                self.counter += 1
                labels_all, probs_all, filename, original_shape, inference_time_sec, page_number = self.work_queue.ungroup(g)
                basename = os.path.basename(filename).split('.')[0]
                self.feat.start(basename)
                if self.enable_debug:
                    # write out an image of the per pixel labels
                    label_viz = np.zeros((labels_all.shape[0], labels_all.shape[1], 3), np.uint8)
                    at_least_one = False
                    for h in range(0, labels_all.shape[0]):
                        for w in range(0, labels_all.shape[1]):
                            c = self.label_val_to_color(labels_all[h, w])
                            at_least_one = True
                            label_viz[h, w, 0] = c[0]
                            label_viz[h, w, 1] = c[1]
                            label_viz[h, w, 2] = c[2]
                    if at_least_one:
                        imsave(os.path.join(self.output_dir, f"{basename}_label_viz.png"), label_viz)
                    del label_viz
                # what pixel labels do we have?
                hist_label_counts = np.bincount(labels_all.flatten()).tolist()
                while len(hist_label_counts) < max(label_bins) + 1:
                    hist_label_counts.append(0)
                # now hist_label_counts contains counts of pixel labels

                self._put_results_log(f"processing: file={filename} histogram={hist_label_counts}  "
                                      f"infer_timing={inference_time_sec} original_shape={original_shape}")

                original_img = imread(filename, pilmode='RGB')
                if self.enable_debug:
                    original_img_box_viz = np.array(original_img)
                    original_img_box_viz_modified = False

                #
                #             handle rectangles here!
                #  process rects with CV to convert messy blobs to nice bboxes, putting content into one Features instance.
                #   Do start() of page, many put()s, finish() page on Features instance
                #   so that feat holds all the rects for a page.
                for label_slice in label_bins:
                    if label_slice == 0:
                        continue  # skip background
                    color_tuple = self.label_val_to_color(label_slice)
                    #  area of all the pixel labels for a particular class, might be multiple regions
                    area = hist_label_counts[label_slice]
                    if area < 500:  # minimum size
                        # reject small label areas
                        continue

                    probs = probs_all[:, :, label_slice]

                    #        make an image showing probability map for this label before postprocessing
                    #            (it can include multiple blobs)
                    if self.enable_debug:
                        prob_img = np.zeros((probs.shape[0], probs.shape[1], 3), np.uint8)
                        for h in range(0, probs.shape[0]):
                            for w in range(0, probs.shape[1]):
                                c = probs[h, w] * 255
                                prob_img[h, w, 0] = c
                                prob_img[h, w, 1] = c
                                prob_img[h, w, 2] = c
                        imsave(os.path.join(self.output_dir, f"{basename}_{label_slice}_label_prob.png"), prob_img)

                    # Binarize the predictions
                    page_bin = self.page_make_binary_mask(probs)

                    # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
                    bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                              tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)
                    # upscale probs the same way so we can calculate confidence later
                    probs_upscaled = cv2.resize(probs.astype(np.float32, casting='same_kind'),
                                                tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)

                    # Find quadrilateral(s) enclosing the label area(s).
                    #  allow more than reasonable number of boxes so we can use spurious boxes as a reject signal.
                    #  The n largest are returned. n_max_boxes must be equal to max ith minus 1 in Features class.
                    pred_region_coords_list = boxes_detection.find_boxes(bin_upscaled.astype(np.uint8, copy=False),
                                                                         mode='rectangle', min_area=0.001, n_max_boxes=4)

                    # coord is [[a,b], [c,b], [c,d], [a,d]]  (a path for drawing a polygon, clockwise)
                    #  origin is upper left [x,y]:
                    #  [a,b]         [c,b]
                    #       rectangle
                    #  [a,d]         [c,d]
                    # which means a<c and b<d

                    if pred_region_coords_list is not None:
                        # Draw region box on original image and export it. Add also box coordinates to the txt file
                        region_count = len(pred_region_coords_list)
                        count = 0
                        for pred_region_coords in pred_region_coords_list:
                            #  cut out rectangle for region based on original image size
                            a = pred_region_coords[0, 0]
                            b = pred_region_coords[0, 1]
                            c = pred_region_coords[1, 0]
                            d = pred_region_coords[2, 1]
                            probs_rectangle = probs_upscaled[b:d + 1, a:c + 1]  # values are in range [0,1]
                            overall_confidence = (sum(sum(probs_rectangle))) / ((c - a) * (d - b))
                            aspect_ratio = (c - a) / (d - b)  # w/h
                            page_width_fraction = (c - a) / original_shape[1]
                            page_height_fraction = (d - b) / original_shape[0]
                            normalized_x = a / original_shape[1]
                            normalized_y = b / original_shape[0]
                            region_size = page_width_fraction * page_height_fraction
                            cmts = f"Prediction {a},{b},{c},{d} confidence={overall_confidence} aspect={aspect_ratio} widthfrac={page_width_fraction} heightfrac={page_height_fraction} normalized_x={normalized_x} normalized_y={normalized_y} dimensions={c - a}x{d - b} spec={basename}_{label_slice}-{count}"
                            self._put_results_log(cmts)
                            img_rectangle = original_img[b:d + 1, a:c + 1]
                            tag_rect_x0 = a
                            tag_rect_y0 = b
                            tag_rect_x1 = c
                            tag_rect_y1 = d
                            if self.enable_debug:
                                # draw box to visualize rectangle
                                cv2.polylines(original_img_box_viz, [pred_region_coords[:, None, :]], True,
                                              (color_tuple[0], color_tuple[1], color_tuple[2]), thickness=5)
                                original_img_box_viz_modified = True
                                imsave(os.path.join(self.output_dir,
                                                    f"{basename}_{label_slice}-{count}_{overall_confidence}_rect.jpg"),
                                       img_rectangle)
                            # Write corners points into a .txt file
                            # txt_coordinates += '{},{}\n'.format(filename, self.format_quad_to_string(pred_region_coords))

                            # store info on area for use after all areas in image are gathered
                            self.feat.put(label_slice, count, region_size, overall_confidence, aspect_ratio,
                                          page_width_fraction, page_height_fraction,
                                          normalized_x, normalized_y,
                                          tag_rect_x0, tag_rect_y0, tag_rect_x1, tag_rect_y1,
                                          img_rectangle, cmts)

                            # Create page region and XML file
                            page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(pred_region_coords[:, None, :]))

                            count += 1
                    else:
                        # No box found for label
                        continue
                if self.enable_debug:
                    # boxes for all labels, using mask colors
                    if original_img_box_viz_modified:
                        imsave(os.path.join(self.output_dir, f"{basename}__boxes.jpg"), original_img_box_viz)

                self.feat.finish()  # finish image, in non-production this saves feature vector for post model training
                page_prediction_msg = ""
                prediction_summary_txt = ""
                if self.production_mode:
                    #
                    #    apply post-model to determine page type
                    #
                    v = np.zeros((1, self.feat.vec_length()))
                    v[0] = self.feat.get_post_model_vec()
                    y = self.post_model.predict(v)
                    page_type = int(y[0])
                    page_prediction_msg = f"PagePrediction: {basename} type={page_type} "
                    prediction_summary_txt = f"type={page_type} "

                    #
                    #     impose basic qualifications
                    #

                    # find feature with max confidence
                    best_label_code, best_ith, best_confidence = self.feat.find_max_confidence_label()
                    # look for disagreement between best confidence and post model predicted page type
                    found_disagreement = False
                    if best_confidence > 0.3:  # generous lower bound
                        if page_type == 0:
                            if best_confidence > 0.45:
                                found_disagreement = True
                        elif page_type == 1:
                            if not (best_label_code == 1 or best_label_code == 2):
                                # article_start, but other features higher confidence
                                found_disagreement = True
                        elif page_type == 2:
                            if best_label_code != 3:
                                # refs page, but other features had higher confidence
                                found_disagreement = True
                        else:
                            if best_label_code != 4:
                                # toc page, but other features had higher confidence
                                found_disagreement = True
                    if found_disagreement:
                        page_prediction_msg += f" DISAGREEMENT with best label={best_label_code} conf={best_confidence}"

                    #           override
                    # look for page_type==1 that should be page_type==3
                    # "Contents" is a title that can throw off a toc
                    if page_type == 1 and best_label_code == 4:
                        # check toc feature minimal width, page_number
                        toc_info = self.feat.get_label_instance(best_label_code, best_ith)
                        width = toc_info["page_width_fraction"]
                        if width > 0.6 and page_number < 10 and toc_info["confidence"] > 0.48:
                            page_prediction_msg += f" OVERRIDE as toc was page_type={page_type} width={width}"
                            page_type = 3  # force toc treatment
                    if page_type == 3 and page_number > 20:
                        # unlikely to be correct
                        page_prediction_msg += f" OVERRIDE as blank, was toc at page={page_number} "
                        page_type = 0

                    #
                    #    take actions
                    #
                    if page_type == 0:                           # other page, skip
                        self.gen_blank_page_msg(basename)
                    elif page_type == 1:
                        #
                        #                   start page of article
                        #
                        #         title
                        #
                        title_info_list = self.feat.get_label_instances_confidence_descending(1)
                        title_info = title_info_list[0]
                        # initial qualification, a low bar
                        try:
                            title_confidence = title_info["confidence"]
                        except KeyError:
                            prediction_summary_txt = f"REJECT {basename} as start_article, reason: missing title"
                            self._put_results_log(prediction_summary_txt)
                            self.emit_debug_summary(basename, prediction_summary_txt)
                            self.gen_blank_page_msg(basename)
                            continue  # done with this page
                        if title_confidence < .5:
                            prediction_summary_txt = f"REJECT {basename} as start_article, reason: confidence too low {title_confidence}"
                            self._put_results_log(prediction_summary_txt)
                            self.emit_debug_summary(basename, prediction_summary_txt)
                            self.gen_blank_page_msg(basename)
                            continue  # done with this page
                        title_rect_x0, title_rect_y0, title_rect_x1, title_rect_y1 = self.unpack_bbox(title_info)
                        title_normalized_y = title_info["normalized_y"]
                        if title_normalized_y > 0.5:
                            msg = f"  REJECT: {basename} title appears in lower half of page "
                            self._put_results_log(msg)
                            prediction_summary_txt += msg
                            self.emit_debug_summary(basename, prediction_summary_txt)
                            self.gen_blank_page_msg(basename)
                            continue  # done with this page
                        title = self.extractor.find_bbox_text(page_number, title_rect_x0, title_rect_y0, title_rect_x1, title_rect_y1)
                        title = self.cleaner.one_line(title)

                        #
                        #           authors
                        #
                        # examine author instances ordered from top of page toward lower
                        author_info_list = self.feat.get_label_instances_y_ascending(2)
                        authors = ""  # accumulated authors block, possibly from multiple instances
                        for j in range(0, 4):
                            author_info = author_info_list[j]
                            author_rect_x0, author_rect_y0, author_rect_x1, author_rect_y1 = self.unpack_bbox(author_info)
                            author_normalized_y = author_info["normalized_y"]
                            author_confidence = author_info["confidence"]
                            #     qualifications
                            if author_confidence == 0.0:  # empty
                                continue  # skip this author_info
                            if author_confidence < 0.4:
                                prediction_summary_txt = f"{basename} SKIP an author bbox reason: confidence {author_confidence}"
                                self._put_results_log(prediction_summary_txt)
                                self.emit_debug_summary(basename, prediction_summary_txt)
                                continue  # skip this author_info
                            if author_normalized_y < title_normalized_y:
                                # author appears above title, abby normal
                                prediction_summary_txt = f"{basename} SKIP an author bbox reason: appears above title"
                                self._put_results_log(prediction_summary_txt)
                                self.emit_debug_summary(basename, prediction_summary_txt)
                                continue  # skip this author_info
                            authors_tmp = self.extractor.find_bbox_text(page_number, author_rect_x0, author_rect_y0,
                                                                        author_rect_x1, author_rect_y1)
                            authors_tmp = self.cleaner.cleanAuthors(authors_tmp)
                            if len(authors) > 0:
                                authors += "\\n"  # escaped EOL char removed in Finisher
                            authors += authors_tmp
                        #
                        #       queue msg to Finisher
                        #
                        jid, iid, page_number = self.cleaner.parse_page_id(basename)
                        excerpt = ex.Excerpt(jid, str(basename), page_number, page_type)
                        excerpt.set_title(title)
                        excerpt.set_authors(authors)
                        g = self.finishing_queue.group(jid, iid, "continue", self.output_dir, excerpt, -1)
                        self.finishing_queue.push(g)
                        #    logging
                        smsg = f"{basename}: page={page_number} TITLE={title} AUTHORS={authors}"
                        self._put_results_log(smsg)
                        prediction_summary_txt += smsg
                        self.emit_debug_summary(basename, prediction_summary_txt)
                    elif page_type == 2:
                        #
                        #                         references page
                        #
                        #   get refs, left to right priority.
                        ref_info_list = self.feat.get_label_instances_x_ascending(3)
                        ref_info_list_cleaned = []
                        #     qualification
                        # reject if normalized width too small
                        smsg = ""
                        max_width = 0.0
                        for ref_peek in ref_info_list:
                            width = 0.0
                            ith = -1
                            conf = 0.0
                            try:
                                width = ref_peek["page_width_fraction"]
                                ith = ref_peek["ith"]
                                conf = ref_peek["confidence"]
                            except KeyError:
                                self._put_results_log(f"in {basename}  page_type=2 KeyError")
                                continue
                            max_width = max(max_width, width)
                            if width < .35 or conf < 0.4:
                                if conf == 0.0:
                                    continue  # skip dummy
                                smsg += f"REJECT ref block {ith} because page_width_fraction={width} conf={conf}"
                                self._put_results_log(smsg)
                                prediction_summary_txt += smsg + "\n"
                            else:
                                ref_info_list_cleaned.append(ref_peek)
                        ref_info_list = ref_info_list_cleaned
                        # we assume 2 cols of refs max
                        ref_1_confidence = 0.0
                        ref_2_confidence = 0.0
                        if len(ref_info_list) > 0:
                            ref_info_1 = ref_info_list[0]
                            ref_1_confidence = ref_info_1["confidence"]
                        if len(ref_info_list) > 1:
                            ref_info_2 = ref_info_list[1]
                            ref_2_confidence = ref_info_2["confidence"]

                        #    combine multiple if present, qualified by confidence
                        refs = ""  # accumulate text here
                        if ref_1_confidence > 0.48:
                            ref_rect_x0, ref_rect_y0, ref_rect_x1, ref_rect_y1 = self.unpack_bbox(ref_info_1)
                            refs += self.extractor.find_bbox_text(page_number, ref_rect_x0, ref_rect_y0,
                                                                  ref_rect_x1, ref_rect_y1)
                            if ref_2_confidence > 0.48:
                                refs += "\n"
                                ref_rect_x0, ref_rect_y0, ref_rect_x1, ref_rect_y1 = self.unpack_bbox(ref_info_2)
                                refs += self.extractor.find_bbox_text(page_number, ref_rect_x0, ref_rect_y0,
                                                                      ref_rect_x1, ref_rect_y1)
                        #
                        #         queue msg to Finisher
                        #
                        if len(refs) > 0:
                            jid, iid, page_number = self.cleaner.parse_page_id(basename)
                            excerpt = ex.Excerpt(jid, basename, page_number, page_type)
                            excerpt.set_refs(refs)
                            g = self.finishing_queue.group(jid, iid, "continue", self.output_dir, excerpt, -1)
                            self.finishing_queue.push(g)
                            smsg += f"ref page {basename} text len={len(refs)}"
                            self._put_results_log(smsg)
                            prediction_summary_txt += smsg + "\n"
                        else:
                            smsg += f"NO TEXT FOUND or was rejected for ref page {basename}"
                            self._put_results_log(smsg)
                            prediction_summary_txt += smsg + "\n"
                            self.gen_blank_page_msg(basename)
                        self.emit_debug_summary(basename, prediction_summary_txt)
                    else:
                        # page_type == 3
                        #                       toc page
                        #
                        toc_info_list = self.feat.get_label_instances_confidence_descending(4)
                        toc_info = toc_info_list[0]  # max confidence toc bbox
                        # qualification
                        width = toc_info["page_width_fraction"]
                        confidence = toc_info["confidence"]
                        if width > 0.6 and confidence > 0.5:
                            toc_rect_x0, toc_rect_y0, toc_rect_x1, toc_rect_y1 = self.unpack_bbox(toc_info)
                            toc = self.extractor.find_bbox_text(page_number, toc_rect_x0, toc_rect_y0,
                                                                  toc_rect_x1, toc_rect_y1)
                            #  queue msg to Finisher
                            jid, iid, page_number = self.cleaner.parse_page_id(basename)
                            excerpt = ex.Excerpt(jid, basename, page_number, page_type)
                            excerpt.set_toc(toc)
                            g = self.finishing_queue.group(jid, iid, "continue", self.output_dir, excerpt, -1)
                            self.finishing_queue.push(g)
                            smsg = f"toc page {basename} text len={len(toc)}"
                            self._put_results_log(smsg)
                            prediction_summary_txt += smsg + "\n"
                        else:
                            # reject
                            smsg = f"REJECT confidence={confidence} width={width} for toc page {basename}"
                            self._put_results_log(smsg)
                            prediction_summary_txt += smsg + "\n"
                            self.gen_blank_page_msg(basename)
                        self.emit_debug_summary(basename, prediction_summary_txt)
                else:  # mode for gathering of training data for post model
                    pass  # anything to do here?
                finish_post = time.time()
                self._put_results_log(
                    f"TIMING: wait={finish_wait - start_wait} post={finish_post - finish_wait} {page_prediction_msg}")
            except Exception:
                # we catch in order to keep running
                trace = traceback.format_exc(limit=4)
                self._put_results_log(
                    f"Unexpected exception: {sys.exc_info()[0]} {sys.exc_info()[1]}:  {trace}")



