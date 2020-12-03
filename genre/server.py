import argparse
import json
import pickle
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from genre import GENRE
from genre.utils import get_entity_spans_fairseq


def load_redirections(lowercase=False):
    redirections = dict()
    with open("data/wiki_redirects.txt") as fin:
        redirections_errors = 0
        for line in fin:
            line = line.rstrip()
            try:
                old_title, new_title = line.split("\t")
                if lowercase:
                    old_title, new_title = old_title.lower(), new_title.lower()
                redirections[old_title] = new_title
            except ValueError:
                redirections_errors += 1

    print("redirections_errors: ", redirections_errors)
    return redirections


# https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
# http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


class GetHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()

        data = json.loads(post_data.decode("utf-8"))
        print("received data:", data)
        text = data["text"]

        response = get_entity_spans_fairseq(
            model,
            [text],
            mention_trie=mention_trie,
            mention_to_candidates_dict=mention_to_candidates_dict,
            redirections=redirections,
        )[0]

        print("response in server.py code:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def terminate():
    tee.close()


if __name__ == "__main__":

    print("Loading model")
    model = (
        GENRE.from_pretrained("models/fairseq_e2e_entity_linking_wiki_abs")
        .eval()
        .to("cuda:0")
    )

    print("Loading mention_trie")
    with open("data/mention_trie_gerbil.pkl", "rb") as f:
        mention_trie = pickle.load(f)

    print("Loading mention_to_candidates_dict")
    with open("data/mention_to_candidates_dict_gerbil.pkl", "rb") as f:
        mention_to_candidates_dict = pickle.load(f)

    print("Loading redirections")
    redirections = load_redirections()

    server = HTTPServer(("localhost", 55555), GetHandler)
    print("Starting server at http://localhost:55555")

    tee = Tee("server.txt", "w")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)
