
import numpy as np
import pandas as pd
import sys
import ufilm_constants as uc
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier  # also try this

"""
This organizes the features and training data. 
The features are resolution-independent values reflecting what dhSegment predicts about an image. 
The goal is to use classic ML to associate sets of label areas with page types so that postprocessing
steps can be trained (the post-model). 

ToDo: Think about about how to use this for page processing, not just for preparing a dataset for learning.  
"""

class Features:
    """
    Feature management for post-dhSegment ML used to infer which page type is being processed. This will help
    determine actions to take during postprocessing.

    Usage has two modes: post training and production. In both cases, prediction results are put() here to
    collect information together. We say "post training" to distinguish dhSegment training from training
    the postprocessing stage. During post-training, a finite number of instances are put() here and then the
    training vectors are save()'ed for input to an sklearn model in post_model.py.

    For production, there is no ground.csv and no need to save the training vectors and
    an indefinite number of start(), put(), and finish() calls can be made. In this case, the _fgrid
    array holds the working dicts.

    If ground_page_type_path is None when __init__() is called, then

    Create an instance, providing the path to the ground.csv file which provides the ground truth
        as to what kind of page it is.

    Start processing images by using the dhSegment model to predict label/class areas.

    After inference predictions are made for an image, call start().

    Then for each predicted region/rectangle, put() computed values.

    Call finish() to complete one feature vector corresponding to all features from a single image.

    When done processing all images, close() to commit the post feature vectors to disk along with more general json
        values for downstream processing. The post feature vectors are post_feature.csv where each row is a training
        case for sklearn and the last col. is the y value to be predicted (obtained from ground.csv).
        A post_details.json file is saved which holds more information than the training vectors, such as
        absolute coordinates of rectangle in original image, time to do predictions, etc.

    Afterwards, the post_feature.csv is used by is_post_model.py to train an sklearn model.
    """

    def __init__(self, ground_page_type_path: str, training_set_out_path: str, max_rects_per_label: int = 4):
        """

        :param ground_page_type_path:  if None, then production mode, otherwise, path to ground.csv which maps
            file basename (string) to page type (int 0-3); a header is expected "file_basename,page_type"
        :param training_set_out_path: where to save training data (not used if ground_page_type_path is None.
        :param max_rects_per_label: defaults to 4, this limits how mny instances per label for one image.
        """
        self._ground_page_type_path = ground_page_type_path
        self._max_rects_per_label = max_rects_per_label # max number of instances (ith) of a label
        #  next, not including background value 0, corresponds to how many types of labels
        self._nclasses = 4
        self._values_per_instance = 8  # vector values per instance of a labeled region (rectangle)
        self._basename = "notSet"  # working state when assembling training vectors
        self._fvec = None # working area for one page/image
        self._fgrid = None # working area for one page/image organized by label_int x ith holding a map of values.
        self._ground_dict = {}
        self._training_out_fd = None
        if ground_page_type_path is not None:
            self._training_out_fd = open(training_set_out_path, "w")
            self._ground_df = pd.read_csv(ground_page_type_path, header=0, sep=',', quotechar='"',
                dtype={'file_basename': str, 'page_type': int})
            # load it into a dict
            for index, row in self._ground_df.iterrows():
                self._ground_dict[row["file_basename"]] = row["page_type"]
        else:
            self._ground_df = None


    def is_production(self):
        return self._ground_df is None

    def vec_length(self):
        return self._nclasses * self._max_rects_per_label * self._values_per_instance


    def start(self, basename: str):
        """
        start collecting info on one image.
        Resets _fvec, _fgrid
        :param basename: the image file basename. This is used for traceability and finding predicted page type
            during post training.
        :param gpu_time_sec: floating point time to process image in seconds.
        :return: None
        """
        self._basename = basename

        # The training/inference vector is assembled as values are put().
        # Big enough to hold uc.max_rects_per_class instances of _nclasses (labels).
        #  Like   class_1_instance_0, class_1_instance_1, class_1_instance_2, class_1_instance_3, class_2_instance_0,...
        #      ending with  y value indicating page type (an int).
        #  Where  class_a_instance_b is info from one rectangle (self._values_per_instance values).
        if self.is_production():
            # production, vec will not have y value
            vec_length = self.vec_length()
            self._fvec = np.zeros((vec_length))
        else:
            #  post training model, vec length is 1 more for 'y' value
            vec_length = self.vec_length() + 1
            self._fvec = np.zeros((vec_length))
            #  get the y to predict
            try:
                ptype = self._ground_dict[basename]
            except KeyError:
                ptype = None
            if ptype is None:
                print(f"ERROR: missing {basename} in {self._ground_page_type_path}, using 0 as page_type")
                ptype = 0
            self._fvec[-1] = ptype
        #  grid of maps, row is class/label, cols are instances (ith)
        self._fgrid = [[None for i in range(self._max_rects_per_label)] for j in range(self._nclasses)]


    #  label, region_size, confidence, aspect_ratio, normalized_position, original(x,y,w,h)
    # params that start with tag_ are tag-alongs that are not features, they are for traceability or
    #   for deriving other features later.  The param ith is not a feature, it indicates the ith rectangle
    #   predicted for a particular label/class (example: if 2 title regions are found, this method is called
    #   twice, first with ith=0 and then with ith=1. ith must not exceed uc.max_rects_per_class-1 and must
    #   uc.max_rects_per_class must equal n_max_boxes in postprocessing.
    def put(self, predicted_label: int, ith: int, region_size: float, confidence: float, aspect_ratio: float,
            page_width_fraction: float, page_height_fraction: float,
            normalized_x: float, normalized_y: float, tag_rect_x0: int, tag_rect_y0: int,
            tag_rect_x1: int, tag_rect_y1: int, tag_rect_image: np.ndarray, tag_comments: str) -> None:
        """
        The tag_rect* params are needed for extracting OCR text from the hOCR file and not for training.

        :param predicted_label: 0=background, 1=title, 2=authors, 3=refs, 4=toc (0 is never used here)
        :param ith: 0 thru uc.max_rects_per_class-1 indicating which instance of a label/class this is
        :param region_size: fraction of area of entire page (0.0 to 1.0)
        :param confidence:  (0.0 to 1.0)
        :param aspect_ratio: w/h
        :param page_width_fraction: fraction of image width that rectangle covers (0.0 to 1.0)
        :param page_height_fraction: fraction of image height that rectangle covers (0.0 to 1.0)
        :param normalized_x: normalized x of upper-left corner of rectangle (0.0 to 1.0) in original image
        :param normalized_y: normalized y of upper-left corner of rectangle (0.0 to 1.0) in original image
        :param tag_rect_x0: x upper left corner of rectangle within original image
        :param tag_rect_y0: y upper left corner of rectangle within original image
        :param tag_rect_x1: x lower right corner of rectangle within original image
        :param tag_rect_y1: y lower right corner of rectangle within original image
        :param tag_rect_image: np image array (w,h,3)
        :param tag_comments: info stats as text
        :return:
        """
        if predicted_label > self._nclasses or predicted_label <= 0:
            print(f"ERROR: predicted_label={predicted_label} is out of bounds")
            sys.exit(3)
        if ith > uc.max_rects_per_class - 1:
            print(f"ERROR: ith={ith} is out of bounds")
            sys.exit(3)   # inconsistency

        offset = (predicted_label - 1) * self._max_rects_per_label * self._values_per_instance
        offset += ith * self._values_per_instance
        # set vec value, self._values_per_instance slots
        self._fvec[offset + 0] = predicted_label
        self._fvec[offset + 1] = region_size
        self._fvec[offset + 2] = confidence
        self._fvec[offset + 3] = aspect_ratio
        self._fvec[offset + 4] = normalized_x
        self._fvec[offset + 5] = normalized_y
        self._fvec[offset + 6] = page_width_fraction
        self._fvec[offset + 7] = page_height_fraction

        # generated name for debug or logging
        item_name = f"{self._basename}_label{predicted_label}-{ith}"
        m = {"predicted_label": predicted_label, "ith": ith,
            "region_size": region_size, "confidence": confidence, "aspect_ratio": aspect_ratio,
             "normalized_x": normalized_x, "normalized_y": normalized_y,
             "page_width_fraction": page_width_fraction, "page_height_fraction": page_height_fraction,
             "tag_rect_x0": tag_rect_x0,
             "tag_rect_y0": tag_rect_y0, "tag_rect_x1": tag_rect_x1, "tag_rect_y1": tag_rect_y1,
             "name": item_name, "image_array": tag_rect_image, "comments": tag_comments}
        self._fgrid[predicted_label - 1][ith] = m


    def _get_dummy_instance(self):
        return {"predicted_label": 0, "ith": 0,
                "region_size": 0, "confidence": 0.0, "aspect_ratio": 0.0,
                "normalized_x": 0.0, "normalized_y": 0.0, "page_width_fraction": 0, "page_height_fraction": 0,
                "tag_rect_x0": 0, "tag_rect_y0": 0, "tag_rect_x1": 0, "tag_rect_y1": 0,
                "name": "unknown", "comments": ""}

    def _pick_keys_confidence(self, arr, j) -> (float, float):
        k_left = arr[j]["confidence"]
        k_right = arr[j + 1]["confidence"]
        return k_left, k_right

    def _pick_keys_x(self, arr, j) -> (int, int):
        k_left = arr[j]["normalized_x"]
        k_right = arr[j + 1]["normalized_x"]
        return k_left, k_right

    def _pick_keys_y(self, arr, j) -> (int, int):
        k_left = arr[j]["normalized_y"]
        k_right = arr[j + 1]["normalized_y"]
        return k_left, k_right

    def _bubble_sort(self, arr: [], pick_keys_f, ascending: bool):
        """
        Sort list of dict, using func ref to compare.
        Bubble sort is fine for short lists.
        :param arr: array of dict
        :param pick_keys_f:  _pick_keys_confidence for example
        :param ascending: true to sort ascending, else descending.
        :return: None, arr is altered in-place.
        """
        n = len(arr)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                k_left, k_right = pick_keys_f(arr, j)
                if ascending:
                    if k_left > k_right:
                        # swap
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]
                else:  # descending
                    if k_left < k_right:
                        # swap
                        arr[j], arr[j + 1] = arr[j + 1], arr[j]

    def get_label_instance(self, label_code: int, ith: int) -> dict:
        """

        :param label_code:
        :param ith:
        :return: a dict with info put() here about a labeled region/rectangle, a dummy instance is returned if
            the slot was not filled.
        """
        r = self._fgrid[label_code - 1][ith]
        if r is None:
            r = self._get_dummy_instance()
        return r

    def _get_label_instance_all_ith(self, label_code: int) -> []:
        arr_dict = []
        for i in range(0, uc.max_rects_per_class):
            arr_dict.append(self.get_label_instance(label_code, i))
        return arr_dict

    def get_label_instances_confidence_descending(self, label_code: int) -> [dict]:
        arr_dict = self._get_label_instance_all_ith(label_code)
        self._bubble_sort(arr_dict, self._pick_keys_confidence, False)
        return arr_dict

    def get_label_instances_x_ascending(self, label_code: int) -> [dict]:
        arr_dict = self._get_label_instance_all_ith(label_code)
        self._bubble_sort(arr_dict, self._pick_keys_x, True)
        return arr_dict

    def get_label_instances_y_ascending(self, label_code: int) -> [dict]:
        arr_dict = self._get_label_instance_all_ith(label_code)
        self._bubble_sort(arr_dict, self._pick_keys_y, True)
        return arr_dict

    def find_max_confidence_label(self):
        """
        Find the label and ith that has the max confidence across all labels.
        :return: label_int, ith, confidence
        """
        feats = []
        max_confidence = 0.0
        label_int = 0
        ith = 0
        feats.append(self.get_label_instances_confidence_descending(1)[0])
        feats.append(self.get_label_instances_confidence_descending(2)[0])
        feats.append(self.get_label_instances_confidence_descending(3)[0])
        feats.append(self.get_label_instances_confidence_descending(4)[0])
        for feat in feats:
            confidence = feat["confidence"]
            if confidence > max_confidence:
                max_confidence = confidence
                label_int = feat["predicted_label"]
                ith = feat["ith"]
        return label_int, ith, max_confidence

    def get_post_model_vec(self) -> np.ndarray:
        """
        In production (is_production() is True), this vector is what the sklearn model uses as input to
        infer what type of page an image represents.
        :return: current feature vector as assembled with put().
        """
        return self._fvec


    def finish(self):
        """
        Finish one image example. During training, this will write out the training vector + y value in last col.
        :return:
        """
        if self.is_production():
            # production
            pass
        else:
            #  post training mode, save the vector
            line = ""
            n = len(self._fvec)
            for i in range(n):
                line += f"{self._fvec[i]}"
                if i == n - 1:
                    line += "\n"
                else:
                    line += ", "
            self._training_out_fd.write(line)


    def close(self):
        if self.is_production():
            # not used in production
            pass
        else:
            #  post training mode, save the vector
            self._training_out_fd.close()


#
#   ToDo: The main() will print some stats about an existing feature set.
#
if __name__ == "__main__":
    n_arguments = len(sys.argv) - 1
    if n_arguments != 2:
        print("usage: path_to_ground_page_type_file_csv path_to_training_set.csv")
        print("example usage: ground.csv  post_training_set.csv")
        sys.exit(1)
    # ToDo: load csv
    # ToDo:
