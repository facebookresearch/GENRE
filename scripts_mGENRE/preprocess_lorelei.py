import argparse
import logging
import os
import pickle
import re

import pandas

import jsonlines
from mgenre.utils import chunk_it, get_wikidata_ids
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/checkpoint/ndecao/lorelei",
    )
    parser.add_argument(
        "--base_wikipedia",
        type=str,
        default="/checkpoint/ndecao/wikipedia",
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        default="/checkpoint/ndecao/wikidata",
    )
    parser.add_argument(
        "--base_lorelei",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/data/KILT_format",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)

    filename = os.path.join(args.base_wikidata, "lang_title2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    with open(filename, "rb") as f:
        lang_title2wikidataID = pickle.load(f)

    filename = os.path.join(args.base_wikidata, "lang_redirect2title.pkl")
    logging.info("Loading {}".format(filename))
    with open(filename, "rb") as f:
        lang_redirect2title = pickle.load(f)

    filename = os.path.join(args.base_wikidata, "label_or_alias2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    with open(filename, "rb") as f:
        label_or_alias2wikidataID = pickle.load(f)

    filename = os.path.join(args.base_wikidata, "mention2wikidataID.pkl")
    logging.info("Loading {}".format(filename))
    with open(filename, "rb") as f:
        mention2wikidataID = pickle.load(f)

    for lang in [e for e in os.listdir(args.base_lorelei) if "jsonl" in e]:
        filename = os.path.join(args.base_lorelei, lang)
        logging.info("Loading {}".format(filename))
        with jsonlines.open(filename) as f:
            data = list(f)

        kilt_dataset = []
        for d in data:
            wikidataIDs = set()
            for o in d["output"]:
                if "wikipedia_url" in o:
                    for link in o["wikipedia_url"].split("|"):
                        if "wikipedia" in link:
                            for e in get_wikidata_ids(
                                re.findall(r"wiki/(.*)", link.strip())[0].replace(
                                    "_", " "
                                ),
                                re.findall(r"http://(.*?).wikipedia", link.strip())[0],
                                lang_title2wikidataID,
                                lang_redirect2title,
                                label_or_alias2wikidataID,
                            )[0]:
                                wikidataIDs.add(e)

            meta = {**d["meta"], "kb_id": d["output"][0]["kb_id"]}
            item = {
                "id": d["id"],
                "input": (
                    meta["left_context"]
                    + " [START] "
                    + meta["mention"]
                    + " [END] "
                    + meta["right_context"]
                ),
                "output": [{"answer": list(wikidataIDs)}],
                "meta": meta,
                "candidates": (
                    [
                        e[0]
                        for e in sorted(
                            mention2wikidataID[meta["mention"]].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:100]
                    ]
                    if meta["mention"] in mention2wikidataID
                    else []
                ),
            }
            kilt_dataset.append(item)

        filename = os.path.join(args.output_dir, lang)
        logging.info("Saving {}".format(filename))
        with jsonlines.open(filename, "w") as f:
            f.write_all(kilt_dataset)
