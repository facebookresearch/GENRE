#!/usr/bin/env python
# coding: utf-8


import requests
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from urllib.parse import unquote
from tqdm.auto import tqdm, trange
from kilt.knowledge_source import KnowledgeSource


with open("/checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/id_title.json") as f:
    id2title = json.load(f)
    title2id = {v: k for k, v in id2title.items()}

    
def get_id_title(anchor, title2id):

    if "http" in anchor:
        return {"wikipedia_title": None, "wikipedia_id": None}

    unquoted = unquote(anchor).split("#")[0].replace("_", " ")
    if unquoted == "":
        return {"wikipedia_title": None, "wikipedia_id": None}

    unquoted = unquoted[0].upper() + unquoted[1:]

    if unquoted in title2id:
        wikipedia_title = unquoted
        wikipedia_id = title2id[unquoted]
        return {"wikipedia_title": wikipedia_title, "wikipedia_id": wikipedia_id}
    else:
        wikipedia_title = requests.head("https://en.wikipedia.org/wiki/{}".format(a),
                                        allow_redirects=True).url.split("/")[-1].split("#")[0].replace("_", " ")
        if wikipedia_title is not None:
            wikipedia_id = title2id.get(wikipedia_title, None)
            if wikipedia_id is not None:
                return {
                    "wikipedia_title": wikipedia_title,
                    "wikipedia_id": wikipedia_id,
                }

    return {"wikipedia_title": None, "wikipedia_id": None}



with open("all_kilt_anchors.pkl", "rb") as f:
    anchors = pickle.load(f)


num_threads = 32
with ThreadPoolExecutor(max_workers=num_threads) as executor:

    futures = {
        executor.submit(get_id_title, anchor, title2id): anchor
        for anchor in anchors
    }

    iter_ = tqdm(as_completed(futures), total=len(futures), smoothing=0)
    results = {futures[future]: future.result() for future in iter_}

