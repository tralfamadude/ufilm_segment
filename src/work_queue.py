import numpy as np
import ray
from ray.util.queue import Queue


"""
Ray Queue manager/wrapper for holding a work queue where prediction generates groups of args to post-process and 
postprocessing consumes these.

Queue(actor_options={"num_cpus": 1})
"""

class WorkQueue:
    def __init__(self, max_depth: int = 8):
        self._queue = Queue(maxsize=max_depth)

    def get_queue(self):
        """
        :return: Ray Queue actor, needed by the consumers.
        """
        return self._queue

    def empty(self):
        """
        :return: Ray Queue actor, needed by the consumers.
        """
        return self._queue.empty()

    def group(self, labels_all: np.ndarray, probs_all: np.ndarray, filename: str, original_shape: tuple,
              inference_time_sec: float, page_number: int) -> dict:
        return {"labels_all": labels_all, "probs_all": probs_all, "filename": filename, "original_shape": original_shape, "inference_time_sec": inference_time_sec, "page_number": page_number}

    def ungroup(self, dictionary):
        """
        use this like: labels_all, probs_all, filename, original_shape = ungroup(d)
        :param dictionary:
        :return:
        """
        return dictionary["labels_all"], dictionary["probs_all"], dictionary["filename"], dictionary["original_shape"], dictionary["inference_time_sec"], dictionary["page_number"]

    def push(self, dictionary):
        """
        Push dictionary of params to post-process. Blocks if queue is full for flow-control and proceeds when
        queue has enough space.
        :param dictionary: a dictionary created with group() method.
        :return: None
        """
        # put in object store
        ref = ray.put(dictionary)
        # put ref in queue
        self._queue.put(ref)
        return None

    def pop(self):
        """
        :return: a dictionary created with group() method, use ungroup() to unpack or lookup individually.
        """
        return self._queue.get()
