import hocr
from glob import glob
import shutil
import time



class ExtractOCR:
    """
    Extract text from the hocr file given a bounding box.
    """
    def __init__(self, hocr_file):
        self.hocr_file = hocr_file
        self.page_iter = hocr.parse.hocr_page_iterator(self.hocr_file)
        self.page_offset = 0
        self.iter_value = self.page_iter.__next__()

    def next_page(self):
        self.iter_value = self.page_iter.__next__()
        self.page_offset += 1

    def seek_page(self, offset):
        if self.page_offset > offset:
            raise IndexError(f"cannot go backwards, current {self.page_offset} > requested {offset}")  # already past desired page
        while self.page_offset < offset:
            self.next_page()

    def find_bbox_text(self, offset, x0, y0, x1, y1):
        """

        :param offset: page offset (nth page in hocr file) from which to extract text
        :param x0: left edge
        :param y0: top edge
        :param x1: right edge where x0<x1
        :param y1: top edge where y0<y1
        :return: text found in bounding box.
        """
        ret = ""
        self.seek_page(offset)
        #w, h = hocr.parse.hocr_page_get_dimensions(self.iter_value)
        #print(f" page {offset} is ({w}, {h})")
        word_data = hocr.parse.hocr_page_to_word_data(self.iter_value)
        for paragraph in word_data:
            for line in paragraph['lines']:
                for word in line['words']:
                    if x0 <= word['bbox'][0] and x1 >= word['bbox'][2] and y0 <= word['bbox'][1] and y1 >= word['bbox'][3]:
                        ret += word['text']
                        ret += " "
                ret += "\n"
        return ret.strip()


class FindHocr:
    """
    Find hocr and chocr files, compressed or not, under a base directory. This will decompress and hand
    back a path to the decompressed file on demand. An hocr file can be found with issue ID.
    """
    def __init__(self, hocr_path):
        self.hocr_path = hocr_path
        # hocr file map:  issue ID -> file
        # if html.gz file is found, we uncompress it as needed
        self.all_hocr_files = glob(f"{self.hocr_path}/**/*hocr.html", recursive=True)
        self.all_hocr_gz_files = glob(f"{self.hocr_path}/**/*hocr.html.gz", recursive=True)
        self.hocr_file_map = self.make_file_map(self.all_hocr_files)
        self.hocr_gz_file_map = self.make_file_map(self.all_hocr_gz_files)

    def dump(self, count_only = False):
        """
        print hocr, chocr files found, along with issue ID.
        :param count_only: do not print, just return the count.
        :return: count of hocr files found
        """
        issue_id_dict = {}
        if not count_only:
            print(f"hocr_path={self.hocr_path}")
            print("  uncompressed hocr files:")
        for h in self.hocr_file_map.keys():
            v = self.hocr_file_map[h]
            iid = self.extract_issue_id(v)
            rel_path = v[len(self.hocr_path):]
            issue_id_dict[iid] = 1
            if not count_only:
                print(f"{iid} : {rel_path}")
        if not count_only:
            print("  compressed hocr files:")
        for h in self.hocr_gz_file_map.keys():
            v = self.hocr_gz_file_map[h]
            iid = self.extract_issue_id(v)
            issue_id_dict[iid] = 1
            rel_path = v[len(self.hocr_path):]
            if not count_only:
                print(f"{iid} : {rel_path}")
        # calc the count
        count = 0
        for j in issue_id_dict.keys():
            count += 1
        if not count_only:
            print(f"{count} unique issue IDs found")
        return count

    def extract_issue_id(self, path) -> str:
        for token in path.split("/"):
            if len(token) < 6:
                continue
            if token.endswith("chocr.html.gz"):
                return token[:-14]
            elif token.endswith("hocr.html.gz"):
                return token[:-13]
            elif token.endswith("chocr.html"):
                return token[:-11]
            elif token.endswith("hocr.html"):
                return token[:-10]

    def make_file_map(self, file_list) -> {}:
        dict = {}
        print(f"make_file_map: file_list={file_list}")
        for apath in file_list:
            issue_id = self.extract_issue_id(apath)
            dict[issue_id] = apath
        return dict

    def find_hocr_file(self, issue_id: str) -> str:
        """
        Find hocr file via issue ID.
        :param issue_id: find the hocr file for this issue_id.
        :return: path to hocr.html file or none if not found.
        """
        result = None
        try:
            result = self.hocr_file_map[issue_id]
            return result
        except KeyError:
            pass
        try:
            gz_file = self.hocr_gz_file_map[issue_id]
            result = gz_file[:-3]
            # uncompress
            import gzip
            with gzip.open(gz_file, 'rb') as f_in:
                with open(result, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # remember that we uncompressed it
            self.hocr_file_map[issue_id] = result
        except KeyError:
            pass
        return result


"""
ad hoc test/demonstration:
"""
if __name__ == '__main__':
    hfinder = FindHocr("/Volumes/pebproject/InternetArchive/Microfilm/ws")
    hfinder.dump()

    eocr = ExtractOCR("/Users/peb/Downloads/sim_bjog_2013-05_120_6_hocr.html")
    # dump entire page 15 (starts from 0)
    #text = eocr.find_bbox_text(15, 0., 0.0, 3322.0, 4300.0)  #  entire page

    # page 8 has TOC, 748,173,1368,1120    and   109,173,721,1127  (need to double these values)
    #text = eocr.find_bbox_text(8, 1496,  346, 2736, 2240)
    #print(f"{text}")
    #text = eocr.find_bbox_text(8, 218,  346, 1442, 2254)
    #print(f"{text}")

    # page 38 has first page of article
    #   title: [402,  500, 2596,  882]
    #   authors: [410,  900, 2038,  994]
    text = eocr.find_bbox_text(38, 402,  500, 2596,  882)
    print(f" TITLE:  {text}")
    text = eocr.find_bbox_text(38, 410,  900, 2038,  994)
    print(f" AUTHORS:  {text}")

