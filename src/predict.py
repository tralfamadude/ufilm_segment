
# derived from demo.py
# Evaluate images by doing prediction.

import os
from glob import glob
import sys
from tqdm import tqdm
import time
import argparse
import work_queue as wq
import finishing_queue as fq
import finisher
import postprocess as pp
import ia_util as util
import ray

from dh_segment.io import PAGE
from dh_segment.inference import LoadedModel

post_model_path = "postmodel.joblib"
enable_debug = True
Nclasses = 4  # number of labels or classes, not counting background
# when False, then use ground.csv to collect post model training data.
# When True, then use the post model to guide actions to take
production_mode = False


if __name__ == '__main__':
    FLAGS = None
    # init the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', '-m',
        type=str,
        default='my_example/output/export',
        help='path to saved model'
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        default='my_example/images',
        help='path to dir of jpg files to process'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='my_example/processed_images',
        help='dir to put output files'
    )
    parser.add_argument(
        '--ground_file', '-g',
        type=str,
        default='my_example/ground.csv',
        help='ground labels for pages needed for training the post model (non-production)'
    )
    parser.add_argument(
        '--post_model_path', '-p',
        type=str,
        default='postmodel.joblib',
        help='path to post model file, ends with .joblib'
    )
    parser.add_argument(
        '--hocr_path',
        type=str,
        default="",
        help='base path to find hocr files, needed for production, subdirs will be searched'
    )
    parser.add_argument(
        '--debug',
        action="store_true",
        help='use this to enable debug'
    )
    parser.add_argument(
        '--production',
        action="store_true",
        help='use this to to enable production mode, otherwise post-model training mode'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f"  Unknown args ignored: {unparsed}")
        sys.exit(5)
    model_dir = FLAGS.model_path  # dhSegment model ("saved model" format)
    input_dir = FLAGS.input_dir  # dir of jpgs or pngs to process
    # glob for input_files
    input_files1 = glob(input_dir + "/*.jpg")
    input_files2 = glob(input_dir + "/*.png")
    input_file_list = input_files1 + input_files2
    print(f"    {len(input_file_list)} files to process")
    input_file_list.sort()
    output_dir = FLAGS.output_dir
    ground_file = FLAGS.ground_file  # required for post model training
    post_model_path = FLAGS.post_model_path  # required
    enable_debug = FLAGS.debug
    hocr_file = FLAGS.hocr_path
    production_mode = FLAGS.production
    # validation
    if len(input_file_list) == 0:
        print("Zero files to process!")
        sys.exit(3)
    if not os.path.exists(model_dir):
        print(f"model dir does not exist: {model_dir}")
        sys.exit(2)
    if production_mode and not os.path.exists(hocr_file):
        print(f"hocr file(s) required in production mode, missing or nonexistent: {hocr_file}")
        sys.exit(2)
    # next is used if non-production (post model training data gen)
    post_model_training_data_path = os.path.dirname(post_model_path) + "/postmodel_training.data"

    os.makedirs(output_dir, exist_ok=True)

    #np.set_printoptions(precision=5)

    #
    #   start Ray, a Queue actor, and post-processing actor(s)
    #
    ray.init(dashboard_host="0.0.0.0", num_cpus=6, num_gpus=1)
    work_queue = wq.WorkQueue()
    finishing_queue = fq.FinishingQueue()
    post_process1 = pp.PostProcess.options(name="PostProcess1").remote(work_queue, finishing_queue, output_dir, ground_file, post_model_path,
                                                                    post_model_training_data_path, enable_debug,
                                                                       production_mode, Nclasses, hocr_file)
    if production_mode:
        post_process2 = pp.PostProcess.options(name="PostProcess2").remote(work_queue, finishing_queue, output_dir, ground_file, post_model_path,
                                                                        post_model_training_data_path, enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
        post_process3 = pp.PostProcess.options(name="PostProcess3").remote(work_queue, finishing_queue, output_dir, ground_file, post_model_path,
                                                                        post_model_training_data_path, enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
        post_process4 = pp.PostProcess.options(name="PostProcess4").remote(work_queue, finishing_queue, output_dir, ground_file, post_model_path,
                                                                        post_model_training_data_path, enable_debug,
                                                                           production_mode, Nclasses, hocr_file)
    post_process1.run.remote()
    if production_mode:
        post_process2.run.remote()
        post_process3.run.remote()
        post_process4.run.remote()
    finisher = finisher.Finisher.options(name="Finisher").remote(finishing_queue, enable_debug, production_mode)
    finisher.run.remote()

    import tensorflow as tf

    # check that GPU is present
    if not tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("        WARNING:  GPU not available")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    pages_processed = 0
    start_predict_time = time.time()

    with tf.Session():  # Start a tensorflow session
        # Load the model
        tf_model = LoadedModel(model_dir, predict_mode='filename')
        print("")
        print(f"          PROCESSING {len(input_file_list)} files from {input_dir}")
        if production_mode:
            print("")
            print(f"                   P R O D U C T I O N")
        else:
            print("")
            print("                    Post-Model Training")
            print("")
            print(f" training data for post-mode will be put in {post_model_training_data_path}")
        print("")
        #               send first msg to finishing_queue
        one_file = input_file_list[0]
        tu = util.TextUtil()
        page_id = tu.image_file_to_page_id(one_file)
        journal_id, issue_id, _ = tu.parse_page_id(page_id)
        working_dir = output_dir + "/" + journal_id
        g = finishing_queue.group(journal_id, issue_id, "start", working_dir, None, len(input_file_list))
        finishing_queue.push(g)

        #  For each image
        for filename in tqdm(input_file_list, desc='Processed files'):
            basename = os.path.basename(filename).split('.')[0]
            page_number = -1
            try:
                page_number = int(basename.split("_")[-1])
            except Exception:
                print(f"ERROR: problem parsing page number in {basename} ; skipping")
                continue
            start_time_sec = time.time()
            #
            #       predict each pixel's label
            #
            prediction_outputs = tf_model.predict(filename)
            finish_time_sec = time.time()
            # labels_all has shape (h,w) which is like (976, 737)
            labels_all = prediction_outputs['labels'][0]
            probs_all = prediction_outputs['probs'][0]
            # probs_all have shape like (976, 737, 4) corresponding to (H, W, Nclasses)
            original_shape = prediction_outputs['original_shape']
            g = work_queue.group(labels_all, probs_all, filename, original_shape, finish_time_sec - start_time_sec,
                                 page_number)
            # give work to the post-processing actor
            work_queue.push(g)
            pages_processed += 1
    print("")
    print(f"          {pages_processed} files processed, results in {output_dir}")
    print("")
    # tell Finisher that this issue is almost complete (pages will probably still be in postprocessing state)
    g = finishing_queue.group(journal_id, issue_id, "finish", working_dir, None, len(input_file_list))
    finishing_queue.push(g)

    #
    #        shutdown procedure
    #
    # wait for the queues to be empty
    while not work_queue.empty():
        time.sleep(1.0)
    while not finishing_queue.empty():
        time.sleep(1.0)
    time.sleep(6.0)
    if enable_debug:
        time.sleep(1.0)
    # finisher could still be working on whole issue
    print("Starting shutdown...")
    g = finishing_queue.group("", "", "close", working_dir, None, 0)
    finishing_queue.push(g)
    time.sleep(9.0)
    post_process1.close.remote()
    if production_mode:
        post_process2.close.remote()
        post_process3.close.remote()
        post_process4.close.remote()
    time.sleep(2.0)
    print(f" Run Duration: {time.time() - start_predict_time} sec")
