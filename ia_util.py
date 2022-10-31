import numpy as np
import os
from string import punctuation

class TextUtil:
    def __init__(self):
        # prep chars to remove except single quote and comma
        self.charsToRemove = punctuation.replace("'", "").replace(",", "").replace(".", "")
        #  and add some other chars to remove
        self.charsToRemove += "®“”"
        self.charsToRemoveMap = np.full((65536), False)
        for i in range(len(self.charsToRemove)):
            c = self.charsToRemove[i]
            self.charsToRemoveMap[ord(c)] = True

    def extract_journal_id(self, issue_id: str):
        if not issue_id.startswith("sim_"):
            return None
        without_prefix = issue_id[4:]
        i = without_prefix.index("_")
        if i > 0:
            jid = without_prefix[:i]
            return "sim_" + jid
        else:
            return None

    def parse_page_id(self, page_id: str):
        """
        :param page_id: the basename of an image with the sim_ prefix (no .png)
        :return: (journal_id, issue_id, page_number)
        """
        # example: sim_journal-of-thought_2005_fall_40_3_0000
        if not page_id.startswith("sim_"):
            return None, None
        jid = self.extract_journal_id(page_id)
        b = page_id.rfind("_")
        if b < 0:
            return None, None
        iid = page_id[:b]  # remove page number
        page_str = page_id[b+1:]
        return jid, iid, int(page_str)

    def image_file_to_page_id(self, image_file):
        # remove directory and extension like .png or .jpg
        page_id = os.path.basename(image_file).split('.')[0]
        return page_id

    def removeIt(self, c):
        """
        :param c:  char to test
        :return: True of char should be removed.
        """
        return self.charsToRemoveMap[ord(c)]

    def cleanAuthors(self, authors):
        """
        Clean up authors string which will contain mis-recognized superscripts, but keep single quote
        char for names like O'Reilly.
        :param authors: ocr chars from authors block
        :return: cleaned string
        """
        result = ""
        offset = 0
        n = len(authors)
        try:
            while offset < n:
                c = authors[offset]
                offset += 1
                if c.isalpha() or c == ' ' or c == '-' or c == '.':
                    result += c
                    continue
                if c == ',':
                    result += c  #  keep comma
                    result += ' '  # space after comma
                    offset += 1
                    if offset >= n:
                        break  #  unlikely to see comma at end
                    c = authors[offset]
                    while self.removeIt(c):  #  skip chars
                        offset += 1 # skip
                        if offset >= n:  #  safety
                            break
                        c = authors[offset]
                    # now we are probably have c==' '
                    if c == ' ':
                        continue
                    # now we are looking to remove non-alpha chars until we see an alpha
                    while not c.isalpha() and c == "'" and offset < n:
                        offset += 1
                        continue
        except Exception:  # just in case
            print(f"Exception occurred cleaning:  {authors}")
        result = result.replace("\n", " ")  # convert EOL chars to space
        result = " ".join(result.split())   # remove consecutive spaces
        return result

    def one_line(self, s):
        s = s.replace("\n", " ")  # convert EOL chars to space
        s = s.replace("\\n", " ")  # convert escaped EOLs to space
        s = " ".join(s.split())   # remove consecutive spaces
        return s

    def escape_eol(self, s: str) -> str:
        return s.replace("\n", "\\n")


if __name__ == '__main__':
    test_example = 'A Conde-Agudelo,* AT Papageorghiou,"* SH Kennedy,” J Villar®“'
    test_result = 'A Conde-Agudelo, AT Papageorghiou, SH Kennedy, J Villar'
    cleaner = TextUtil()
    print(f"{cleaner.charsToRemoveMap}")
    r = cleaner.cleanAuthors(test_example)
    good = test_result == r
    print(f" {good}\n  {test_example}\n  {test_result}\n  {r}")
