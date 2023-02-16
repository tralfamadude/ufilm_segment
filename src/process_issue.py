import time
import excerpt as ex
import pathlib
import time
import json
import ia_util
import logging
import ufilm_constants

close_latency = ufilm_constants.issue_close_latency_secs

"""
A place to keep state about one issue and perform processing. This makes it easy to keep track of 2 issues 
simultaneously when the finishing_queue makes a transition from one issue to another. 
"""
class ProcessIssue:
    def __init__(self, journal_id, issue_id, working_dir):
        """

        :param journal_id: the journal this issue is a part of
        :param issue_id: unique issue ID
        :param working_dir: where to write out results.
        """
        self.journal_id = journal_id
        self.issue_id = issue_id
        self.working_dir = working_dir
        self.start_time = time.time()
        self.finish_time = None  # marked to be ready to finish after a timeout
        self.completed_time = None  # when actually finished
        self.start_close_time = None
        self.issue_excerpts = ex.IssueExcerpts()
        self.last_activity = time.time()
        self.page_count_target = 0
        self.start_time = time.time()  # when instance created
        self.util = ia_util.TextUtil()

    def duration_secs(self) -> float:
        return self.completed_time - self.start_time

    def is_match(self, journal_id: str, issue_id: str) -> bool:
        """
        :param journal_id: query if this journal id matches
        :return: True if journal_id and issue_id matches.
        """
        return self.journal_id == journal_id and self.issue_id == issue_id

    def finish_up(self) -> None:
        """
        Call this to start clock ticking that this issue needs to be finished.
        The predictor puts a "finish" state message on the finishing_queue which
        tells the finisher that the issue is almost done. That is when this
        method should be called. Reason: need to give post-processing workers
        time to finish up pages for an issue that predict loop indicated is finished.
        :return: None
        """
        self.finish_time = time.time()
        return None

    def ready_to_close(self) -> bool:
        """
        :return: True if finish_up() has been called and some leeway seconds have passed, meaning it
            is safe to write out and close() this. The leeway allows postprocessing workers to
            work on last few pages of an issue because "finish" msg from predict loop is not serialized
            to come after all pages are processed by postprocess workers.
        """
        if self.page_count_target == self.get_page_count():
            logging.info("process_issue.ready_to_close: page count triggered")
            return True
        if not self.finish_time:
            return False
        delta_sec = time.time() - self.finish_time
        if delta_sec > close_latency:
            logging.info("process_issue.ready_to_close: time triggered")
            return True
        else:
            return False

    def is_complete(self) -> bool:
        """
        :return: True if all processing and saving of results is completed.
        """
        return self.completed_time is not None

    def get_page_count(self) -> int:
        return self.issue_excerpts.get_page_count()

    def put_page_count_target(self, page_count_target: int) -> None:
        self.page_count_target = page_count_target

    def put(self, excerpt: ex.Excerpt) -> None:
        """
        :param excerpt: add given excerpt instance to this issue.
        :return:  None
        """
        if self.is_complete():
            logging.warning(f"process_issue: already complete but received a put for page {excerpt.get_page_number()}")
        self.last_activity = time.time()
        self.issue_excerpts.put(excerpt)
        return None

    def close_duration_sec(self) -> float:
        """
        :return: time in seconds (float) of duration of the close() action.
        """
        if self.is_complete():
            return self.completed_time - self.start_close_time
        else:
            return None

    def close(self) -> None:
        """
        Process whole issue to coalesce excerpts into whole-issue json.
        Write out final state and close up this instance.
        After this, is_compelete() returns True.
        Idempotent, safe to call multiple times.
        :return:
        """
        if self.is_complete():
            logging.debug("already finished or started finishing!")
            return None
        self.start_close_time = time.time()

        toc = ""  # gathered TOC text

        #
        #        write out excerpt results
        # (helpful for debugging, but could be avoided for production)
        #
        dest_dir = self.working_dir
        apath = pathlib.Path(dest_dir)
        apath.mkdir(parents=True, exist_ok=True)
        self.issue_excerpts.save(dest_dir + "/" + self.issue_id + ".jsonl")

        #
        #       locate toc pages, skip non-contiguous ones, combine
        #
        self.issue_excerpts.is_contiguous()
        toc_ex_candidate_list = []
        page_index = self.issue_excerpts.get_ordered_pages()
        for page_number in page_index:
            ex = self.issue_excerpts.get(page_number)
            if ex.is_toc():
                toc_ex_candidate_list.append(ex)
        if len(toc_ex_candidate_list) > 0:
            toc_ex_good_list = []
            # check contiguousness
            curr_page = -1
            for ex in toc_ex_candidate_list:
                if curr_page < 0:
                    curr_page = ex.page_number
                    toc_ex_good_list.append(ex)
                elif curr_page + 1 == ex.page_number:
                    # good
                    curr_page = ex.page_number
                    toc_ex_good_list.append(ex)
                else:
                    # non-contiguous, msg and zap this excerpt
                    logging.warn(f"REJECT toc on page {ex.page_number} as non-contiguous noise, {ex.page_id}")
            if len(toc_ex_good_list) > 0:
                for ex in toc_ex_good_list:
                    if len(toc) > 0:
                        toc += '\n'
                    toc += ex.toc
        # toc contains text of concatenated TOCs or is zero length string if none found
        logging.debug(f"TOC GATHERED={self.util.escape_eol(toc)}")  # DEBUG

        #
        #        Find articles and references/citations
        #
        # results of this passage as 2 parallel lists
        article_list = []  #  value is ex of article start
        refs_list = []  #  value is list of ex of refs (could be empty list)
        # state for scanning
        curr_article_ex = None  # tmp as we scan
        curr_ref_ex_list = []   # tmp as we scan
        for page_number in page_index:
            ex = self.issue_excerpts.get(page_number)
            if ex.is_article():
                if curr_article_ex is not None:
                    # put one article to results lists
                    article_list.append(curr_article_ex)
                    refs_list.append(curr_ref_ex_list)
                curr_article_ex = ex
                curr_ref_ex_list = []
            elif ex.is_refs():
                if curr_article_ex is None:
                    # refs without article start!
                    logging.warning(f"REJECT: refs without article page {ex.page_number}, {ex.page_id}"
                                    + f" for issue_id={self.issue_id}")
                else:
                    curr_ref_ex_list.append(ex)
            elif ex.is_blank() and curr_article_ex is not None and len(curr_ref_ex_list) > 0:
                # allow blank page to signal end of article, only if there is a refs page.
                # put one article to results lists
                article_list.append(curr_article_ex)
                refs_list.append(curr_ref_ex_list)
                curr_article_ex = None
                curr_ref_ex_list = []
        if curr_article_ex is not None:
            article_list.append(curr_article_ex)
            refs_list.append(curr_ref_ex_list)

        # article_list and refs_list hold articles,refs at the same index.
        #  emit some stat info
        n_articles_with_refs = 0  # calc how many articles have refs (editorials usually do not)
        for i in range(0, len(article_list)):
            if len(refs_list[i]) > 0:
                n_articles_with_refs += 1
            #       clean up authors
            # ToDo: use toc to validate tokens, final authors string should have no EOL chars
            article_ex = article_list[i]
            authors_tmp = article_ex.authors
            authors_tmp = authors_tmp.replace('&', " ")
            # ToDo: split into tokens
            # ToDo: if token appears in toc, then accept, else reject
            # ToDo: authors on separate lines need comma sep when combined into one line
            authors_tmp = authors_tmp.replace("\n", ", ")
            # make excerpt reflect the clean info
            article_ex.set_authors(authors_tmp)

        #
        logging.info(f"FOUND {len(article_list)} articles, {n_articles_with_refs} have refs," +
                     f" for issue_id={self.issue_id}")

        # ToDo: write out toc.json (does this combine articles with refs?)
        json_out_path = dest_dir + "/" + self.issue_id + ".extract.json"
        pretty_out_path = dest_dir + "/" + self.issue_id + ".toc.txt"
        toc_list = []  # value is dict per toc entry with keys title, authors, page, page_id, refs
        with open(pretty_out_path, "w") as f:
            for i in range(0, len(article_list)):
                article_ex = article_list[i]
                article_refs = ""
                for refs_ex in refs_list[i]:
                    if len(article_refs) > 0:
                        article_refs += "\n"
                    article_refs += refs_ex.refs
                f.write(f"'{article_ex.title}' BY {article_ex.authors}  PAGE {article_ex.page_number}\n")
                item = {}
                item["title"] = article_ex.title
                item["authors"] = article_ex.authors
                item["page"] = article_ex.page_number
                item["page_id"] = article_ex.page_id
                item["refs"] = article_refs  # ToDo: need eol chars preserved, escape them?
                toc_list.append(item)
        # write out toc_list as json
        json_dict = {}
        json_dict["articles"] = toc_list
        json_dict["toc_ocr"] = toc
        json_dict["issue_id"] = self.issue_id
        json_dict["journal_id"] = self.journal_id
        json_txt = json.dumps(json_dict)
        with open(json_out_path, "w") as f:
            f.write(f"{json_txt}\n")

        # mark self as done
        self.completed_time = time.time()

        # log duration
        print(f"COMPLETED: {self.issue_id} finished, page_count={self.get_page_count()} duration={self.duration_secs()}")
        return None

