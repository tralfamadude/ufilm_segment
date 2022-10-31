import sys
import re
import json
import argparse
import Levenshtein

"""
A way to organize results of page post-processing so it can be passed to the finisher task. 
The data here also serves as rows that summarize what was extracted so that different runs
of the extraction process can be compared. Each row is json and the concatenated rows
are a jsonl file. 

Excerpt is used to hold info about a single page and is passed to the finisher task via the finishing_queue.

IssueExcerpts is used by the finisher task to organize the collection of excerpts
"""

class AbsExcerpt:
    """
    abstract baseclass to enable a tiny bit of type help in IDE.
    """
    def __init__(self, journal_id: str = "", page_id: str = "", page_number: int = -1, page_type: int = -1):
        pass

    def set_title(self, title: str):
        pass

    def set_authors(self, authors: str):
        pass

    def set_refs(self, refs: str):
        pass

    def set_toc(self, toc: str):
        pass

    def set_explanation(self, explanation: str):
        pass

    def compare(self, other) -> float:
        return 0.0


class Excerpt(AbsExcerpt):
    def __init__(self, journal_id: str = "", page_id: str = "", page_number: int = -1, page_type: int = -1):

        """
        :param journal_id: as known to IA.
        :param page_id: as known to IA.
        :param page_number: page number from the image filename, ignored for state start/finish
        :param page_type: int, same as ground.csv where 0=no annotations, 1=start_article, 2=references, 3=toc.
        """
        self.journal_id = journal_id
        self.page_id = page_id
        self.page_type = page_type
        self.page_number = page_number
        self.title = ""
        self.authors = ""
        self.refs = ""
        self.toc = ""
        self.explanation = ""

    def get_page_number(self) -> int:
        return self.page_number

    def get_page_type(self) -> int:
        return self.page_type

    def escape_eol(self, s: str) -> str:
        return s.replace("\n", "\\n")

    def remove_eols(self, s: str) -> str:
        """
        Remove newline chars and consecutive spaces are collapsed into one space.
        Trailing space is removed.
        :param s:
        :return:
        """
        s = s.replace('\n', ' ')
        s = re.sub('\s+',' ', s)
        while s[-1:] == ' ':
            s = s[:-1]
        return s

    def set_title(self, title: str) -> AbsExcerpt:
        """
        :param title: text without EOL chars
        :return: self
        """
        self.title = self.remove_eols(title)

    def set_authors(self, authors: str) -> AbsExcerpt:
        self.authors = authors

    def set_refs(self, refs: str) -> AbsExcerpt:
        self.refs = refs

    def set_toc(self, toc: str) -> AbsExcerpt:
        self.toc = toc

    def set_explanation(self, explanation: str) -> AbsExcerpt:
        self.explanation = explanation

    def pretty(self) -> str:
        """
        format excerpt for deubg/diag viewing.
        :return: formatted string, or None if not worth printing.
        """
        ptype = ""
        operand1 = ""
        operand2 = ""
        if self.is_blank():
            return None
        elif self.is_article():
            ptype = "ARTICLE"
            operand1 = f"\"{self.title}\" by \"{self.authors}\""
            operand2 = self.escape_eol(self.explanation)
        elif self.is_refs():
            ptype = "REFS"
            operand1 = self.escape_eol(self.refs)
            operand2 = self.escape_eol(self.explanation)
        elif self.is_toc():
            ptype = "TOC"
            operand1 =  self.escape_eol(self.toc)
            operand2 = self.escape_eol(self.explanation)
        else:
            ptype = f"UNKNOWN({self.page_type})"
        sout = f"{self.page_id}: {ptype} page {self.page_number}: {operand2}\n    {operand1}"
        return sout

    def to_json(self) -> str:
        """
        :return: json for excerpt as one line, EOL chars in string types are escaped, suitable for concatenating
            into an "*.jsonl" file.
        """
        dict = {"journal_id": self.journal_id, "page_id": self.page_id, "page_number": self.page_number,
                "page_type": self.page_type,  "title": self.title, "authors": self.authors, "refs": self.refs,
                "toc": self.toc, "explanation": self.explanation}
        json_out = json.dumps(dict)
        return json_out

    def load_from_json(self, json_str: str):
        """
        Load values into this object from a json string.
        :param json_str:
        :return:
        """
        dict = json.loads(json_str)
        # print(f" load_from_json: {dict}")
        self.journal_id = dict["journal_id"]
        self.page_id = dict["page_id"]
        self.page_type = dict["page_type"]
        self.page_number = dict["page_number"]
        self.title = dict["title"]
        self.authors = dict["authors"]
        self.refs = dict["refs"]
        self.toc = dict["toc"]
        self.explanation = dict["explanation"]

    def same_page(self, other) -> bool:
        if self.journal_id != other.journal_id:
            return False
        if self.page_id != other.page_id:
            return False
        return self.page_number == other.page_number

    def same_type(self, other) -> bool:
        return self.page_type == other.page_type

    def is_blank(self):
        return self.page_type == 0

    def is_article(self):
        return self.page_type == 1

    def is_refs(self):
        return self.page_type == 2

    def is_toc(self):
        return self.page_type == 3

    def compare(self, other: AbsExcerpt) -> float:
        """
        Compare this to other instance.
        :param other: compare this to the other.
        :return: a score, the similarity of the extracted text, if the two instances
            are the same page type. If types differ, then score is 0
        """
        if not self.same_page(other):
            raise KeyError("cannot compare excerpts from different pages")
        if not self.same_type(other):
            return 0.0
        if self.is_blank():
            return 1.0
        elif self.is_article():
            m1 = Levenshtein.ratio(self.title, other.title)
            m2 = Levenshtein.ratio(self.authors, other.authors)
            return m1 * m2
        elif self.is_refs():
            return Levenshtein.ratio(self.refs, other.refs)
        elif self.is_toc():
            return Levenshtein.ratio(self.toc, other.toc)
        else:
            raise IndexError("unknown page type")


class IssueExcerpts:
    """
    A collection of Excerpts for a single issue. It can be saved, loaded, compared, and dumped to stdout.
    """
    def __init__(self):
        self.max_page = -1       # page number max
        self.page_count = 0
        self.excerpt_list = []   # partially ordered Excerpts
        self.excerpt_index = {}  # index by page number to get Excerpt instance

    def put(self, row: Excerpt) -> int:
        """
        Add an excerpt to this collection for a single issue. If put() again, it will replace
        previous if the same page.
        :param row: an Excerpt instance to store.
        :return: total number of pages
        """
        # notice if already exists (we are replacing)
        replacement = False
        prev = self.get(row.get_page_number())
        if prev:
            replacement = True
        self.excerpt_list.append(row)
        self.excerpt_index[row.get_page_number()] = row
        if not replacement:
            self.page_count += 1
            self.max_page = max(self.max_page, row.get_page_number())
        return self.page_count

    def get(self, page_number: int) -> Excerpt:
        try:
            return self.excerpt_index[page_number]
        except KeyError:
            return None

    def get_page_count(self) -> int:
        return self.page_count

    def is_contiguous(self) -> bool:
        """
        Check whether the IssueExcerpts has no missing pages
        :return: True if no missing pages.
        """
        if self.max_page+1 != self.page_count:
            return False
        # check that all the pages exist
        for ipage in range(0, self.max_page+1):
            ex = self.get(ipage)
            if ex is None:
                return False
        return True

    def save(self, dest: str) -> None:
        """
        Save as a jsonl file (one json per line), in page order.
        :param dest:
        :return:
        """
        with open(dest, 'w',  encoding='utf-8') as f:
            for ipage in range(0, self.max_page + 1):
                ex = self.get(ipage)
                if ex:
                    f.write(ex.to_json() + "\n")

    def load(self, src: str):
        """
        Load an IssueExcerpts that was saved as a jsonl file, one json per line.
        :param src: file location to read.
        :return: self (IssueExcerpts)
        """
        with open(src, "r", encoding='utf-8') as f:
            while True:
                json_str = f.readline()
                if not json_str:
                    break
                ex = Excerpt("","",-1,-1)
                ex.load_from_json(json_str)
                self.max_page = max(self.max_page, ex.page_number)
                self.page_count += 1
                self.excerpt_list.append(ex)
                self.excerpt_index[ex.page_number] = ex
        return self

    def dump(self):
        """
        dump to stdout, pretty printed.
        :return: None
        """
        continued_line_prefix = "    "
        for k in self.get_ordered_pages():
            ex = self.get(k)
            if ex:
                spretty = ex.pretty()
                if spretty:
                    spretty = spretty.replace("\\n", f"\n{continued_line_prefix}")
                    sys.stdout.write(f"{spretty}")
        sys.stdout.write('\n')

    def get_ordered_pages(self):
        """
        :return: list of the pages ordered by page number.
        """
        ordered_list = []
        for k in self.excerpt_index.keys():
            ordered_list.append(k)
        ordered_list.sort()
        return ordered_list

    def compare(self, other) -> (float, str):
        """
        Compare results.

        :param other: compare this to the other.
        :return: (score, explanation)
        """
        scores = []
        missing_counterparts = 0
        explanation = ""
        count_blank = 0
        count_article = 0
        count_refs = 0
        count_toc = 0
        my_pages = self.get_ordered_pages()
        my_pages_len = len(my_pages)
        other_pages = other.get_ordered_pages()
        if my_pages_len != len(other_pages):
            explanation += " differing page count; "
        if self.is_contiguous() != other.is_contiguous():
            explanation += " only one file non-contiguous; "
        other_pages_mask = {}
        for i in other_pages:
            other_pages_mask[i] = 1
        for i in range(0, my_pages_len):
            page_no = my_pages[i]
            my_pages[i] = None # mark page as checked
            my_ex = self.get(page_no)
            my_type = my_ex.get_page_type()
            if my_type == 0:
                count_blank += 1
            elif my_type == 1:
                count_article += 1
            elif my_type == 2:
                count_refs += 1
            elif my_type == 3:
                count_toc += 1
            other_ex = other.get(page_no)
            if other_ex:
                other_pages_mask[page_no] = 0  # mark as done
                # compare the page
                score = my_ex.compare(other_ex)
                scores.append(score)
            else:
                # other page does not exist
                scores.append(0.0)
                missing_counterparts += 1
        # check on pages in other that are not in this
        for j in range(0, sum(other_pages_mask.values())):
            scores.append(0.0)
            missing_counterparts += 1
        if missing_counterparts > 0:
            explanation += f" {missing_counterparts} missing counterpart pages"
        # overall score
        overall_score = sum(scores) / len(scores)
        if len(explanation) == 0:
            explanation = f"{my_pages_len} pages; "
            explanation += f" articles={count_article} refs={count_refs} toc={count_toc} blank={count_blank}"
        return overall_score, explanation



if __name__ == '__main__':
    FLAGS = None
    # init the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dump', '-d',
        type=str,
        default='',
        help='path of excerpt jsonl file to dump, pretty printed'
    )
    parser.add_argument(
        '--golden', '-g',
        type=str,
        default='',
        help='path to golden excerpt jsonl file to compare'
    )
    parser.add_argument(
        '--check', '-c',
        type=str,
        default='',
        help='path to excerpt jsonl file to compare to golden'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f"  Unknown args ignored: {unparsed}")
        parser.print_usage()
        sys.exit(5)
    dump_path = FLAGS.dump
    golden_path = FLAGS.golden
    check_path = FLAGS.check

    if dump_path != '':
        ie = IssueExcerpts()
        ie.load(dump_path)
        ie.dump()
        sys.exit(0)
    if  golden_path == '' or check_path == '':
        parser.print_usage()
        sys.exit(1)
    ie_golden = IssueExcerpts()
    ie_golden.load(golden_path)
    ie_check = IssueExcerpts()
    ie_check.load(check_path)
    score, explanation = ie_golden.compare(ie_check)
    print(f"{score}  explanation: {explanation}")
