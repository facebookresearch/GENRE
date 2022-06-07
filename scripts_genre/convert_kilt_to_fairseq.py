# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import jsonlines
from tqdm import tqdm

from genre.utils import create_input


def convert_kilt_to_fairseq(dataset):

    source = []
    target = []
    for doc in tqdm(dataset, desc="Processing"):
        for title in set(
            prov["title"]
            for out in doc["output"]
            if "provenance" in out
            for prov in out["provenance"]
            if prov.get("bleu_score", 1) > 0.5
        ):
            source.append(
                create_input(
                    doc,
                    max_length=384,
                    start_delimiter="[START_ENT]",
                    end_delimiter="[END_ENT]",
                )
            )
            target.append(title)
            if "meta" in doc and "template_questions" in doc["meta"]:
                for template_question in doc["meta"]["template_questions"]:
                    source.append(template_question)
                    target.append(title)

    return source, target


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_filename",
        type=str,
        help="Filename of the KILT dataset",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the converted dataset",
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

    logging.info("Loading {}".format(args.input_filename))
    with jsonlines.open(args.input_filename) as f:
        dataset = [e for e in f]
    split_name = os.path.basename(args.input_filename).split("-")[1]

    source, target = convert_kilt_to_fairseq(
        dataset,
    )

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    for type_name, data in (("source", source), ("target", target)):

        with open(
            os.path.join(
                args.output_path,
                "{}.{}".format(split_name, type_name),
            ),
            "w",
        ) as f:
            f.writelines(
                [doc.replace("\r", ">>").replace("\n", ">>") + "\n" for doc in data]
            )
