import numpy as np
import ray
from ray.util.queue import Queue


"""
Ray Queue manager/wrapper for holding issue IDs to work on. This enables issue IDs to be added after starting
the process running under Ray by running a utility that connects to the ensemble. 

ToDo: consider what else to add as params. Model directory? working dir? Should the whole ensemble be 
restarted when switching journals?  

Queue(actor_options={"num_cpus": 1})
"""

class IssueQueue:
    def __init__(self, max_depth: int = 1e5):
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

    def group(self, journal_id: str, issue_id: str) -> dict:
        return {"journal_id": journal_id, "issue_id": issue_id}

    def ungroup(self, dictionary):
        """
        use this like: journal_id, issue_id = ungroup(d)
        :param dictionary:
        :return:
        """
        return dictionary["journal_id"], dictionary["issue_id"]

    def push(self, dictionary):
        """
        Push dictionary of journal issue to process. Blocks if queue is full for flow-control and proceeds when
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
