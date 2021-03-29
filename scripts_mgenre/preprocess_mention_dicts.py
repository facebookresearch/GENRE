# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines
import numpy as np
from hanziconv import HanziConv
from mgenre.utils import create_input
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--langs",
        type=str,
        help="Pipe (|) separated list of language ID to process.",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=10e20,
        help="Max number of datapoints to process.",
    )
    parser.add_argument(
        "--abstracts",
        action="store_true",
        help="Process abstracts only.",
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

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    mention2wikidataID = defaultdict(lambda: defaultdict(int))
    wikidataID2mention = defaultdict(lambda: defaultdict(int))
    wikidataID2lang = defaultdict(lambda: defaultdict(int))

    for lang in args.langs.split("|"):
        filename = os.path.join(args.base_wikipedia, "{0}/{0}wiki.pkl".format(lang))
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wiki = pickle.load(f)

        count = 0
        for page in tqdm(wiki.values()):

            max_paragraph_id = 0 if args.abstracts else len(page["paragraphs"])
            while (
                max_paragraph_id < len(page["paragraphs"])
                and "Section::::" not in page["paragraphs"][max_paragraph_id]
            ):
                max_paragraph_id += 1

            for anchor in page["anchors"]:
                if anchor["paragraph_id"] < max_paragraph_id and count < args.max_count:
                    count += 1
                    for id_ in anchor["wikidata_ids"]:
                        mention = anchor["text"]
                        mention = (
                            unicodedata.normalize(
                                "NFKD", HanziConv.toSimplified(mention)
                            )
                            .replace("•", "·")
                            .replace("．", "·")
                        )

                        mention2wikidataID[mention][id_] += 1
                        wikidataID2mention[id_][mention] += 1
                        wikidataID2lang[id_][lang] += 1

    wikidataID2lang_total = defaultdict(int)
    for v in wikidataID2lang.values():
        for lang, count in v.items():
            wikidataID2lang_total[lang] += count

    wikidataID2lang_priority = {}
    for wikidata_id in tqdm(wikidataID2lang):
        wikidataID2lang_priority[wikidata_id] = [
            e[0]
            for e in sorted(
                [
                    (lang2, count, wikidataID2lang_total[lang2])
                    for lang2, count in wikidataID2lang[wikidata_id].items()
                ],
                key=lambda x: x[1:],
                reverse=True,
            )
        ]

    for data, name in zip(
        (
            mention2wikidataID,
            wikidataID2mention,
            wikidataID2lang,
            wikidataID2lang_priority,
        ),
        (
            "mention2wikidataID.pkl",
            "wikidataID2mention_.pkl",
            "wikidataID2lang.pkl",
            "wikidataID2lang_priority.pkl",
        ),
    ):

        data = {k: dict(v) if not isinstance(v, list) else v for k, v in data.items()}
        filename = os.path.join(args.base_wikidata, name)
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(data, f)
