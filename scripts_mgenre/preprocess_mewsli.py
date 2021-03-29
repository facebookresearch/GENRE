# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import re

import jsonlines
import pandas
from mgenre.utils import chunk_it, get_wikidata_ids
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_dir",
        type=str,
    )
    parser.add_argument(
        "--base_wikipedia",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "--base_mewsli",
        type=str,
        help="Base folder with mewsli-9 dataset.",
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

    logging.basicConfig(level=args.loglevel)

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

    for lang in os.listdir(args.base_mewsli):
        logging.info("Loading {}".format(os.path.join(args.base_mewsli, lang)))
        docs = pandas.read_csv(
            os.path.join(args.base_mewsli, lang, "docs.tsv"), sep="\t"
        )
        mentions = pandas.read_csv(
            os.path.join(args.base_mewsli, lang, "mentions.tsv"), sep="\t"
        )
        merge = pandas.merge(docs, mentions, on="docid", how="outer")

        kilt_dataset = []
        for _, row in tqdm(merge.iterrows(), total=len(merge)):
            with open(os.path.join(args.base_mewsli, lang, "text", row["docid"])) as f:
                doc = "".join(f.readlines())
                wikidataIDs = get_wikidata_ids(
                    re.findall(r"wiki/(.*)", row["url_y"])[0].replace("_", " "),
                    re.findall(r"http://(.*?).wikipedia", row["url_y"])[0],
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )[0]
                meta = {
                    "left_context": doc[: row["position"]].strip(),
                    "mention": doc[
                        row["position"] : row["position"] + row["length"]
                    ].strip(),
                    "right_context": doc[row["position"] + row["length"] :].strip(),
                }
                item = {
                    "id": "mewsli-9-{}-{}-{}".format(
                        row["lang"], row["docid"], row["position"]
                    ),
                    "original_link": row["url_y"],
                    "input": (
                        meta["left_context"]
                        + " [START] "
                        + meta["mention"]
                        + " [END] "
                        + meta["right_context"]
                    ),
                    "output": [{"answer": list(wikidataIDs)}],
                    "meta": meta,
                }
                kilt_dataset.append(item)

        filename = os.path.join(args.output_dir, "{}-kilt-test.jsonl".format(lang))
        logging.info("Saving {}".format(filename))
        with jsonlines.open(filename, "w") as f:
            f.write_all(kilt_dataset)
