import http.server
import socketserver
import argparse
import sys
from collections import OrderedDict
import ia_util
import os
from glob import glob
import json
from urllib.parse import urlparse
from urllib.parse import parse_qs

### NOTE: run this where the current working directory is a prefix of the -b base dir pattern
### so that images can be found.

generated_page = ""
page_id_2_image_path = {}
issue_id_2_json_path = OrderedDict()
cwd = os.environ.get("PWD")
tu = ia_util.TextUtil()

def gen_page_prelude(title: str):
    return f"""<!DOCTYPE html>
    <html>
    <head>
      <title>{title}</title>
      <style>
      table, th, td """ + \
           '{' "border: 1px solid gray; border-collapse: collapse;  padding: 2px;" + '}' + """
      </style>
    </head>
    <body>
    """


def gen_page_postlude():
    return "</body>\n</html>\n"

# specific page
# https://archive.org/details/sim_family-practice-research-journal_winter-1990_10_2/page/133/mode/1up?admin=1
# first page:
# https://archive.org/details/sim_family-practice-research-journal_winter-1990_10_2/?admin=1

def gen_table_prelude(issue_id: str):
    issue_url = f"https://archive.org/details/{issue_id}/?admin=1"
    issue_link = f"<a href=\"{issue_url}\">{issue_id}</a>"

    return f"""
<h2>{issue_link}</h2>
<table style="width:100%">
  <tr>
    <th bgcolor="#e0e0f0">Title</th>
    <th bgcolor="#e0e0f0">Authors</th>
    <th bgcolor="#e0e0f0">Page</th>
    <th bgcolor="#e0e0f0">Bbox</th>
  </tr>
"""


def gen_table_row(page_id: str, image_path: str, title: str, authors: str, page_no: int):
    if len(title) > 150:
        title = title[:150] + "..."
    _, iid, page_number = tu.parse_page_id(page_id)

    url = f"https://archive.org/details/{iid}/page/n{page_no}/mode/1up?admin=1"
    tsv_out.write(f"{iid}\t{title}\t{authors}\t{page}\t{url}\n")
    page_link = f"<a href=\"{url}\">{page_no}</a>"
    url = f"{image_path}"
    bbox_link = f"<a href=\"{url}\">bbox</a>"
    return f"  <tr><td>{title}</td><td>{authors}</td><td>{page_link}</td><td>{bbox_link}</td></tr>\n"


def gen_table_postlude():
    return "</table>\n"


def find_files(base_dir: str):
    #   find files and put in dicts:
    # page_id_2_image_path = {}
    # issue_id_2_json_path = OrderedDict()
    #
    file_list = glob(f"{base_dir}/**/*__boxes.jpg", recursive=True)
    for apath in file_list:
        s = os.path.basename(apath)
        i = s.index("__boxes.jpg")
        pid = s[0:i]
        page_id_2_image_path[pid] = apath

    file_list = glob(f"{base_dir}/**/*.extract.json", recursive=True)
    for apath in file_list:
        s = os.path.basename(apath)
        i = s.index(".extract.json")
        iid = s[0:i]
        issue_id_2_json_path[iid] = apath


def page_id_to_issue_id(page_id):
    jid, iid, page_number = (tu.parse_page_id(page_id))
    return iid


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(generated_page, "utf8"))
        elif self.path.endswith(".jpg"):
            print(f" path={self.path}")  # DEBUG
            if self.path.startswith(cwd):
                self.path = self.path[len(cwd):]
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        return


if __name__ == '__main__':
    FLAGS = None
    # init the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base', '-b',
        type=str,
        help='base directory patterns for finding __boxes.jpg files and extract.json files '
             'from a predict.py run with debug enabled; '
             'example: \'/home/peb/image_downloads/*/out_run_pub_journal-of-thought_v2c/\'  '
             '(~ does not work, MUST have trailing slash, use single quotes around pattern)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=9999,
        help='optional listening port, default is 9999'
    )
    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print(f"  Unknown args: {unparsed}")
        sys.exit(1)
    port = FLAGS.port
    base = FLAGS.base

    print(f"Locating content under {base}...")
    find_files(base)
    print(f"Found {len(issue_id_2_json_path)} jsons, and {len(page_id_2_image_path.keys())} page images")
    if len(issue_id_2_json_path) > 0:
        vlist = list(issue_id_2_json_path.values())
        print("  EXAMPLE json: ")
        print(f"    {vlist[0]}")
    if len(page_id_2_image_path) > 0:
        vlist = list(page_id_2_image_path.values())
        print(f"  EXAMPLE image:")
        print(f"    {vlist[0]}")

    # while True:
    #     answer = input("Okay to proceed (Y/N)?: ")
    #     if answer.startswith("Y") or answer.startswith("y"):
    #         break
    #     elif answer.startswith("N") or answer.startswith("n"):
    #         sys.exit(0)
    #     else:
    #         continue

    tsv_out = open("results.tsv", "w")
    tsv_out.write("issue\ttitle\tauthors\tpage\turl\n")

    generated_page = gen_page_prelude("Extraction Results")
    for iid in issue_id_2_json_path.keys():
        json_path = issue_id_2_json_path[iid]
        json_str = ""
        with open(json_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        extract_dict = json.loads(json_str)
        generated_page += gen_table_prelude(iid)
        article_list = extract_dict["articles"]
        for article in article_list:
            title = article["title"]
            authors = article["authors"]
            page = article["page"]
            page_id = article["page_id"]
            refs = article["refs"]
            if page_id in page_id_2_image_path:
                image_file_path = page_id_2_image_path[page_id]
            else:
                image_file_path = "NotFound"
            generated_page += gen_table_row(page_id,image_file_path, title, authors, page)
        generated_page += gen_table_postlude()

    generated_page += gen_page_postlude()
    tsv_out.flush()
    tsv_out.close()

    # Create an object of the above class
    handler_object = MyHttpRequestHandler

    my_server = socketserver.TCPServer(("", port), MyHttpRequestHandler)

    # Star the server
    my_server.serve_forever()

