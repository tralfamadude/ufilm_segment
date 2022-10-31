import json

import ray
from ray.util.queue import Queue
import excerpt as ex
import process_issue as pi
import time
import os
import threading as th
import logging
import pathlib
import ufilm_constants
import traceback
import sys


def _delayed_close(target: pi.ProcessIssue) -> None:
    logging.debug("delayed_close() triggered")
    target.close()

@ray.remote
class Finisher:  # actor, consumer of finishing_queue
    def __init__(self, finishing_queue: Queue, enable_debug: bool, production_mode: bool):
        self.finishing_queue = finishing_queue
        self.enable_debug = enable_debug
        self.production_mode = production_mode
        self.pages_processed = 0  #  count over multiple issues
        self.start_time_sec = time.time()
        self.results_log_path = "finisher_info.log"
        self.results_log = open(self.results_log_path, "a")
        # we keep up to 2 active ProcessIssue instances active
        self.processing_state_a = None
        self.processing_state_b = None
        # keep track of when issues are started/finished; accumulates indefinitely;
        #    use it to avoid accidental restart an issue due to straggler page (which could mass up a completed issue)
        self.issues_started = {}
        self.issues_finished = {}
        self.timer_a = None
        self.timer_b = None
        logging_dest_dir = os.environ.get("PWD") + "/" + ufilm_constants.logging_dest_dir
        pathlib.Path(logging_dest_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=f"{logging_dest_dir}/finisher.log",
                            format='%(asctime)s %(message)s',
                            level=logging.DEBUG)
        logging.info(f"STARTING debug={enable_debug} production_mode={production_mode}")

    def stats_page_count(self) -> int:
        """
        :return: count of pages processed, across multiple issues.
        """
        return self.pages_processed

    def stats_issue_count(self) -> int:
        return len(self.issues_started)

    def stats_uptime(self) -> float:
        """
        :return: uptime in seconds, as a float.
        """
        return time.time() - self.start_time_sec

    def find_processing_state(self, journal_id: str, issue_id: str, working_dir: str) -> pi.ProcessIssue:
        """
        find active ProcessIssue instance for given journal ID or make a new one.
        We keep up to 2 instances, like an a/b buffer strategy because there is overlap between issues
        during processing.
        As a side effect, we also close instances if they are ready.
        :param journal_id:
        :param issue_id:
        :param working_dir:
        :return: ProcessIssue instance matching given vars, or None if capacity at max.
        """
        # is it A?
        if self.processing_state_a is not None:
            if self.processing_state_a.is_match(journal_id, issue_id):
                return self.processing_state_a
        # is it B?
        if self.processing_state_b is not None:
            if self.processing_state_b.is_match(journal_id, issue_id):
                return self.processing_state_b

        # existing instance for given issue not found, so create one...

        try:
            # was the given issue previously processed and closed? we do not want to start an issue accidentally
            #  due to a straggler page. Better to drop a straggler than to mess up previous work.
            if self.issues_finished[issue_id]:
                logging.error(f"rejecting straggler from restarting ProcessIssue for issue_id={issue_id}")
                return None
        except KeyError:
            pass

        # target is in neither a nor b, can we make a new one to replace a completed one?
        if self.processing_state_a is None or self.processing_state_a.is_complete():
            self.processing_state_a = pi.ProcessIssue(journal_id, issue_id, working_dir)
            return self.processing_state_a
        if self.processing_state_b is None or self.processing_state_b.is_complete():
            self.processing_state_b = pi.ProcessIssue(journal_id, issue_id, working_dir)
            return self.processing_state_b

        # OR can we close one now?
        if self.processing_state_a.ready_to_close():
            self.processing_state_a.close()
            self.processing_state_a = pi.ProcessIssue(journal_id, issue_id, working_dir)
            return self.processing_state_a
        if self.processing_state_b.ready_to_close():
            self.processing_state_b.close()
            self.processing_state_b = pi.ProcessIssue(journal_id, issue_id, working_dir)
            return self.processing_state_b
        # pile up
        logging.error(f"Unable to find ProcessIssue instance for {issue_id}")
        return None

    def close(self) -> None:
        """
        Close all ProcessIssue instances now.
        :return: None
        """
        logging.info("Finisher.close() begin")
        if self.processing_state_a is not None:
            self.processing_state_a.close()
            self.processing_state_a = None
        if self.processing_state_b is not None:
            self.processing_state_b.close()
            self.processing_state_b = None
        logging.info("Finisher.close(): Shutdown done")

    async def run(self):
        while True:
            try:
                #
                #   get item off work queue
                #
                start_wait = time.time()
                g = self.finishing_queue.pop()
                finish_wait = time.time()
                journal_id, issue_id, state, working_dir, excerpt, page_count = self.finishing_queue.ungroup(g)
                if state == "start":
                    self.issues_started[issue_id] = time.time()
                    logging.info(f"start issue {issue_id} working_dir={working_dir}")
                elif state == "finish":
                    logging.info(f"finish issue msg received from finisher queue!  page_count={page_count}")
                    self.issues_finished[issue_id] = time.time() # inexact finish of issue, close enough to avoid stragglers
                    logging.info(f"finish imminent for issue_id={issue_id} page_count={page_count} working_dir={working_dir}")
                    # Prepare for finishing up corresponding ProcessIssue instance,
                    #   start timer for timeout-based flushing, set page count target
                    delay_secs = ufilm_constants.issue_close_latency_secs
                    if self.processing_state_a  and  self.processing_state_a.is_match(journal_id, issue_id):
                        self.processing_state_a.put_page_count_target(page_count)
                        self.processing_state_a.finish_up()  # start timeout to finish up
                        # in delay_secs sec, have a timer close processing state for issue_id
                        self.timer_a = th.Timer(delay_secs, _delayed_close, [self.processing_state_a])
                        self.timer_a.start()
                    elif self.processing_state_b  and  self.processing_state_b.is_match(journal_id, issue_id):
                        self.processing_state_b.put_page_count_target(page_count)
                        self.processing_state_b.finish_up()  # start timeout to finish up
                        # in delay_secs, have a timer close processing state for issue_id
                        self.timer_b = th.Timer(delay_secs, _delayed_close, [self.processing_state_b])
                        self.timer_b.start()
                elif state == "continue":
                    self.pages_processed += 1
                    pi = self.find_processing_state(journal_id, issue_id, working_dir)
                    if pi:
                        #logging.debug(f"received page {excerpt.get_page_number()}")
                        pi.put(excerpt)
                        # if number of pages matches total given by predictor in finish msg, then close it
                        if pi.ready_to_close():
                            pi.close()
                    else:
                        logging.error(f"rejecting {excerpt.page_id} as straggler or due to congestion")
                elif state == "close":
                    logging.info("close msg on finisher queue, shutting down!")
                    # shutdown due to all work completed.
                    #  predictor should send this only after sleeping a few seconds when it is ready to shutdown
                    self.close()
                else:
                    logging.error(f"ERROR: unknown state {state}")
                    continue
            except Exception:
                # we catch in order to keep running
                trace = traceback.format_exc(limit=6)
                logging.error(f"Unexpected exception  {sys.exc_info()[0]} {sys.exc_info()[1]}:  {trace}")
