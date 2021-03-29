# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from collections import defaultdict

import jsonlines
from prettytable import PrettyTable
from tqdm.auto import tqdm


def evaluate(guess_path, gold_path):
    results = defaultdict(lambda: defaultdict(list))
    for fname in tqdm(sorted(os.listdir(guess_path))):

        with jsonlines.open(os.path.join(guess_path, fname)) as f:
            pred = [e for e in f]

        with jsonlines.open(os.path.join(gold_path, fname)) as f:
            gold = [e for e in f]

        recalls = []
        for dg, dp in zip(gold, pred):
            assert dg["id"] == dp["id"]
            dp["predictions"] = [
                e["answer"][0] if isinstance(e["answer"], list) else e
                for e in dp["predictions"]
            ]
            dp["predictions"] = [e for e in dp["predictions"] if e != None]

            if len(dg["output"][0]["answer"]) != 1:
                recalls.append(math.inf)
            else:
                recalls.append(
                    1
                    + min(
                        [
                            i
                            for i, e in enumerate(dp["predictions"])
                            if e in dg["output"][0]["answer"]
                        ]
                        + [math.inf]
                    )
                )

        lang = fname[:2]
        results["R@1"][lang] = [sum(e <= 1 for e in recalls) / len(recalls)]
        results["R@10"][lang] = [sum(e <= 10 for e in recalls) / len(recalls)]

        results["R@1"]["micro-avg"] += [e <= 1 for e in recalls]
        results["R@1"]["macro-avg"] += results["R@1"][lang]

        results["R@10"]["micro-avg"] += [e <= 10 for e in recalls]
        results["R@10"]["macro-avg"] += results["R@10"][lang]

    results_final = defaultdict(dict)
    for k1, v1 in results.items():
        for k2, v2 in v1.items():
            results_final[k2][k1] = sum(v2) / len(v2)

    return results_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--guess_path",
        type=str,
        help="Path to guess datasets",
    )
    parser.add_argument(
        "--gold_path",
        type=str,
        help="Path to gold datasets",
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

    results_final = evaluate(args.guess_path, args.gold_path)

    keys = sorted(results_final, key=lambda x: x if "-avg" not in x else "zz")

    results = PrettyTable()
    results.field_names = [
        "Lang",
        "R@1",
        "R@10",
    ]
    for k in keys:
        results.add_row(
            [k]
            + [
                "{:.2f}".format(100 * e)
                for e in (results_final[k]["R@1"], results_final[k]["R@10"])
            ]
        )

    print(results)
