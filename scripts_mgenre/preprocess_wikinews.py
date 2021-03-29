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
from collections import defaultdict

import jsonlines
import numpy as np
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
        "--base_wikinews",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default="ar|bg|bs|ca|cs|de|el|en|eo|es|fa|fi|fr|he|hu|it|ja|ko|nl|no|pl|pt|ro|ru|sd|sq|sr|sv|ta|th|tr|uk|zh",
        help="Pipe (|) separated list of language ID to use.",
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

    for lang in args.langs.split("|"):
        filename = os.path.join(args.base_wikinews, lang, "{}wiki.pkl".format(lang))
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wiki = pickle.load(f)

        kilt_dataset = []
        for doc in tqdm(wiki.values()):
            for i, anchor in enumerate(doc["anchors"]):
                if len(anchor["wikidata_ids"]) == 1:
                    meta = {
                        "left_context": (
                            "".join(doc["paragraphs"][: anchor["paragraph_id"]])
                            + doc["paragraphs"][anchor["paragraph_id"]][
                                : anchor["start"]
                            ]
                        ),
                        "mention": (
                            doc["paragraphs"][anchor["paragraph_id"]][
                                anchor["start"] : anchor["end"]
                            ]
                        ),
                        "right_context": (
                            doc["paragraphs"][anchor["paragraph_id"]][anchor["end"] :]
                            + "".join(doc["paragraphs"][anchor["paragraph_id"] :])
                        ),
                    }
                    item = {
                        "id": "wikinews-{}-{}-{}".format(lang, doc["id"], i),
                        "input": (
                            meta["left_context"]
                            + " [START] "
                            + meta["mention"]
                            + " [END] "
                            + meta["right_context"]
                        ),
                        "output": [{"answer": anchor["wikidata_ids"]}],
                        "meta": meta,
                    }
                    kilt_dataset.append(item)

        filename = os.path.join(args.output_dir, "{}-kilt-all.jsonl".format(lang))
        logging.info("Saving {}".format(filename))
        with jsonlines.open(filename, "w") as f:
            f.write_all(kilt_dataset)

        if len(kilt_dataset) >= 10000:
            wiki_dict = defaultdict(list)
            for doc in kilt_dataset:
                wiki_dict[doc["id"].split("-")[2]].append(doc)

            test_set = []
            dev_set = []
            train_set = []

            np.random.seed(0)
            for docs in np.random.permutation(list(wiki_dict.values())):
                if len(test_set) < len(kilt_dataset) // 10:
                    test_set += docs
                elif len(dev_set) < len(kilt_dataset) // 10:
                    dev_set += docs
                else:
                    train_set += docs

            for split_name, split in zip(
                ("test", "dev", "train"), (test_set, dev_set, train_set)
            ):
                filename = os.path.join(
                    args.output_dir, "{}-kilt-{}.jsonl".format(lang, split_name)
                )
                logging.info("Saving {}".format(filename))
                with jsonlines.open(filename, "w") as f:
                    f.write_all(split)
