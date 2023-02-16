import numpy as np
import ray
from ray.util.queue import Queue
import excerpt as ex

"""
Ray Queue manager/wrapper for serializing coordination of finishing steps when processing a journal issue. 

The worker processes that do post-processing are not in a position to do final rollup of an issue because
pages are processed in an async fashion and seeing the last page is not proof all pages are done because 
earlier pages might not yet be finished. The idea here is to post information to this queue when the pages of an 
issue are finished.  The one finisher task that pops items off this queue will be able to determine when the 
issue is done and perform finishing work on the issue. 

Analysis of workload:  Can the finisher task keep up? It should not be a bottleneck because finishing work is 
only done once per issue while normal page post-processing occurs 100x more often, although that work has
4 workers. The finisher actor must do the work in about 10 seconds to keep up (about 10 images per sec. are
processed by the GPU and issues have usually about 100 pages). If the finisher gets behind, what happens? 
The predict loop and workers do not stop for the finisher, they will continue to add to the finish_queue 
after the predict loop posts the finish state message. 



The predict loop posts 2 messages to this:
1. state='start'  when starting on an issue, along with working_dir, journal_id, no Excerpt.
2. state='finish' after finishing an issue, along with working_dir, journal_id, page_count, no Excerpt

The post-processing workers post to this:
1. state='continue' after finishing processing of a page, along with working_dir, Excerpt instance. 

The finisher task is in a position to evaluate the whole issue. Is the page count wrong? (Post-processing 
workers could have failed.) Are there any articles? (An index issue could be processed by accident.) It can gather
up the outputs from post-processing to make a combined toc.json and store the combined citations per article. 

Queue(actor_options={"num_cpus": 1})
"""

class FinishingQueue:
    def __init__(self, max_depth: int = 150):
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

    def group(self, journal_id: str, issue_id: str, state: str, working_dir: str, excerpt: ex.Excerpt, page_count: int) -> dict:
        """

        :param journal_id: the journal ID as known to ia tool
        :param issue_id: the issue ID as know to ia tool
        :param state: one of 'start' (start new issue), 'finish' (finish issue), 'continue' add page, 'close' (shutdown)
        :param working_dir: path to output directory
        :param excerpt: results from post-processing, None for when predict loop posts start/finish state messages.
        :param page_count: provide number of pages in issue with state "finish".
        :return:
        """
        return {"journal_id": journal_id, "issue_id": issue_id, "state": state, "working_dir": working_dir, "excerpt": excerpt, "page_count": page_count}

    def ungroup(self, dictionary):
        """
        use this like: journal_id, issue_id, state, working_dir, excerpt, page_count = ungroup(d)
        :param dictionary:
        :return:
        """
        return dictionary["journal_id"], dictionary["issue_id"], dictionary["state"], dictionary["working_dir"], dictionary["excerpt"], dictionary["page_count"]

    def push(self, dictionary):
        """
        Push dictionary of params to finisher task. Blocks if queue is full for flow-control and proceeds when
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
