import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from genere.global_genere import GeNeRe


class DummyModel:
    def __init__(self, fname):
        with open(fname, "r") as f:
            self.data = json.load(f)

    def get_prediction(self, text):
        return self.data.get(text, [])


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

        text, spans = read_json(post_data)
        response = model.get_prediction(text)

        print("response in server.py code:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    print("received data:", data)
    text = data["text"]
    spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    return text, spans


def terminate():
    tee.close()


if __name__ == "__main__":

    model = GeNeRe(
        model_path="/checkpoint/ndecao/2020-08-27/new_fairseq_globalel_wiki_abs_aidayago.bart_large.ls0.1.mt2048.uf4.mu10000.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm500.fp16.ngpu8",
        checkpoint_file="checkpoint130.pt",
        device="cuda:0",
    )

    #     model = DummyModel("dummy_oke16.json")

    server = HTTPServer(("localhost", 55555), GetHandler)
    print("Starting server at http://localhost:55555")

    tee = Tee("server.txt", "w")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)
