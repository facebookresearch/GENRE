#!/usr/bin/env python
# coding: utf-8


import argparse
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

import requests
from kilt.knowledge_source import KnowledgeSource
from tqdm.auto import tqdm, trange


def batch_iter(obj, batch_size=1):
    out = []
    for item in obj:
        if len(out) == batch_size:
            yield out
            out = []
        out.append(item)

    if len(out):
        yield out

def _read_url(anchor, lang="en"):
    url = "https://{}.wikipedia.org/wiki/{}".format(lang, anchor.replace(" ", "_"))
    res = requests.get(url, allow_redirects=True)
    content = res.content.decode('utf-8')
    title = html.unescape(
        content[content.index("<title>") + len("<title>"):content.index("</title>")][:-12]
    )
    title = title[0].upper() + title[1:]

    return title

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
        attepts = 0
        while attepts < 3:
            try:
                wikipedia_title = _read_url(anchor)
                attepts = 3
            except e:
                attepts += 1
                print("error")
                time.sleep(1)

        if wikipedia_title is not None:
            wikipedia_id = title2id.get(wikipedia_title, None)
            if wikipedia_id is not None:
                return {
                    "wikipedia_title": wikipedia_title,
                    "wikipedia_id": wikipedia_id,
                }

    return {"wikipedia_title": None, "wikipedia_id": None}




# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--rank", default=0, type=int, help="rank in a distributed execution",
# )

# args = parser.parse_args()

ks = KnowledgeSource()

# anchors = []
# iter_ = tqdm(ks.get_all_pages_cursor(), total=ks.get_num_pages())
# for page in iter_:
#     anchors += [a['href'] for a in page["anchors"]]
#     iter_.set_postfix(anchors=len(anchors), refresh=False)

# with open("/checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/id_title.json") as f:
#     id2title = json.load(f)
#     title2id = {v: k for k, v in id2title.items()}

# def get_id_title(anchor, title2id):

#     if "http" in anchor:
#         return {"wikipedia_title": None, "wikipedia_id": None}

#     unquoted = unquote(anchor).split("#")[0].replace("_", " ")
#     if unquoted == "":
#         return {"wikipedia_title": None, "wikipedia_id": None}

#     unquoted = unquoted[0].upper() + unquoted[1:]

#     if unquoted in title2id:
#         wikipedia_title = unquoted
#         wikipedia_id = title2id[unquoted]
#         return {"wikipedia_title": wikipedia_title, "wikipedia_id": wikipedia_id}
#     else:
#         attepts = 0
#         while attepts < 3:
#             try:
#                 wikipedia_title = requests.head("https://en.wikipedia.org/wiki/{}".format(anchor),
#                                                 allow_redirects=True).url.split("/")[-1].split("#")[0].replace("_", " ")
#                 attepts = 3
#             except:
#                 attepts += 1
#                 print("error")
#                 time.sleep(1)

#         if wikipedia_title is not None:
#             wikipedia_id = title2id.get(wikipedia_title, None)
#             if wikipedia_id is not None:
#                 return {
#                     "wikipedia_title": wikipedia_title,
#                     "wikipedia_id": wikipedia_id,
#                 }

#     return {"wikipedia_title": None, "wikipedia_id": None}


# with open("all_kilt_anchors_{}.pkl".format(args.rank), "rb") as f:
#     anchors = pickle.load(f)

# num_threads = 32
# with ThreadPoolExecutor(max_workers=num_threads) as executor:

#     futures = {
#         executor.submit(get_id_title, anchor, title2id): anchor
#         for anchor in tqdm(anchors)
#     }

#     iter_ = tqdm(as_completed(futures), total=len(futures), smoothing=0)
#     results = {futures[future]: future.result() for future in iter_}

# with open("all_kilt_anchors_map_{}.pkl".format(args.rank), "wb") as f:
#     pickle.dump(results, f)


with open("all_kilt_anchors_map.pkl", "rb") as f:
    results = pickle.load(f)
print(len(results))

# for page in tqdm(ks.get_all_pages_cursor(), total=ks.get_num_pages(), smoothing=0):
#     anchors = page["anchors"]
#     for anchor in (anchors):
# #         if anchor["href"] in results:
#         anchor["wikipedia_title"] = results[anchor["href"]]["wikipedia_title"]
#         anchor["wikipedia_id"] = results[anchor["href"]]["wikipedia_id"]
# #     break
#     ks.db.find_one_and_update(
#         {"_id": page["wikipedia_id"]}, {"$set": {"anchors": anchors}}, upsert=True,
#     )


mention_entitiy_table = defaultdict(lambda: defaultdict(int))
for page in tqdm(ks.get_all_pages_cursor(), total=ks.get_num_pages()):
    for anchor in page["anchors"]:
        if anchor["wikipedia_title"]:
            mention_entitiy_table[anchor["text"]][anchor["wikipedia_title"]] += 1

mention_entitiy_table = {k: dict(v) for k, v in mention_entitiy_table.items()}
with open("mention_entitiy_table.pkl", "wb") as f:
    pickle.dump(mention_entitiy_table, f)
