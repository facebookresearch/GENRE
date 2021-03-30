# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import pickle
from copy import deepcopy

import jsonlines
from kilt.eval_retrieval import compute
from prettytable import PrettyTable
from tqdm.auto import tqdm

from genre.fairseq_model import GENRE
from genre.trie import Trie
from genre.utils import batch_it, create_input


def evaluate_kilt_dataset(
    model,
    dataset,
    batch_size=4,
    beams=10,
    max_len_a=384,
    max_len_b=15,
    candidates=False,
    trie=None,
    title2id={},
    free_generation=False,
    test=False,
):

    dataset_original = deepcopy(dataset)

    gold = []
    pred = []

    iter_ = tqdm(dataset, desc="Evaluating")
    for docs in batch_it(iter_, batch_size):

        if not free_generation:
            batch_trie = {
                i: (
                    (
                        Trie(
                            [
                                [2] + model.encode(e).tolist()[1:]
                                for e in doc["candidates"]
                            ]
                        )
                        if doc["candidates"]
                        else Trie([[2] + model.encode("NIL").tolist()[1:]])
                    )
                    if candidates
                    else trie
                )
                for i, doc in enumerate(docs)
            }

            def prefix_allowed_tokens_fn(batch_id, sent):
                return batch_trie[batch_id].get(sent.tolist())

        outputs = model.sample(
            [
                create_input(
                    doc,
                    max_len_a,
                    start_delimiter="[START_ENT]",
                    end_delimiter="[END_ENT]",
                )
                for doc in docs
            ],
            beam=beams,
            max_len_b=max_len_b,
            prefix_allowed_tokens_fn=None
            if free_generation
            else prefix_allowed_tokens_fn,
        )

        for doc, out in zip(docs, outputs):
            if not test:
                gold.append(doc["output"][0]["answer"])
                try:
                    pred.append(out[0]["text"])
                except Exception as e:
                    pred.append("NIL")
                    print(doc)
                    print(e)

            doc["output"] = [
                {
                    "answer": "",
                    "provenance": [
                        {
                            "wikipedia_id": title2id.get(prov["text"], None),
                            "title": prov["text"],
                            "score": prov["score"].item(),
                        }
                        for prov in out
                    ],
                }
            ]

        if not test:
            true_pos = 0
            for g, p in zip(gold, pred):
                if g == p and p != "NIL":
                    true_pos += 1

            precision = (
                (true_pos / len([p for p in pred if p != "NIL"]))
                if len([p for p in pred if p != "NIL"])
                else 0
            )
            recall = (true_pos / len(gold)) if len(gold) else 0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if precision + recall
                else 0
            )

            iter_.set_postfix(f1=f1, prec=precision, rec=recall)

    if not test:
        kilt_dict = compute(dataset_original, dataset, ks=[1, 5], rank_keys=["title"])
        return dataset, f1, precision, recall, kilt_dict["Rprec"], kilt_dict["recall@5"]
    else:
        return dataset, 0, 0, 0, 0, 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="model.pt",
        help="Checkpoint file",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path where to load the dataset(s)",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where to save the prediction(s)",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--beams",
        default=10,
        type=int,
        help="Number of beams",
    )
    parser.add_argument(
        "--max_len_a",
        default=384,
        type=int,
        help="Max input length",
    )
    parser.add_argument(
        "--max_len_b",
        default=15,
        type=int,
        help="Max output length",
    )
    parser.add_argument(
        "--trie",
        type=str,
        help="Trie pickle file",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Enables the use of provided candidates",
    )
    parser.add_argument(
        "--free_generation",
        action="store_true",
        help="Disables constrained decoding",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="CPU/GPU device",
    )
    parser.add_argument(
        "--id_title",
        type=str,
        help="ID to title map json file",
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
    parser.add_argument(
        "--test",
        help="Run tests (no evaluation)",
        action="store_true",
    )

    args = parser.parse_args()

    assert (os.path.isdir(args.input_path) and os.path.isdir(args.output_path)) or (
        not os.path.isdir(args.input_path) and not os.path.isdir(args.output_path)
    ), "`input_path` and `output_path` have either to be both files or folders"

    logging.basicConfig(level=args.loglevel)

    logging.info("Loading model")
    if "cuda" not in args.device and torch.cuda.is_available():
        logging.warning(
            "CUDA is available but running on CPU. Set --device cuda:<ID> for running on GPU."
        )

    model = (
        GENRE.from_pretrained(args.model_path, checkpoint_file=args.checkpoint_file)
        .eval()
        .to(args.device)
    )

    if not args.candidates and not args.free_generation:
        logging.info("Loading Trie from {}".format(args.trie))
        with open(args.trie, "rb") as f:
            trie = Trie.load_from_dict(pickle.load(f))
    else:
        trie = None

    if args.id_title is not None:
        logging.info("Loading ID to title map from {}".format(args.id_title))
        with open(args.id_title) as f:
            id2title = json.load(f)
            title2id = {v: k for k, v in id2title.items()}
    else:
        title2id = {}

    results = PrettyTable()
    results.field_names = [
        "Dataset",
        "F1",
        "Precision",
        "Recall",
        "R-precision",
        "Recall@5",
    ]

    datasets_filenames = (
        [os.path.join(args.input_path, fname) for fname in os.listdir(args.input_path)]
        if os.path.isdir(args.input_path)
        else [args.input_path]
    )

    for dataset_filename in datasets_filenames:

        logging.info("Loading {}".format(dataset_filename))
        with jsonlines.open(dataset_filename) as f:
            dataset = [e for e in f]

        dataset, f1, precision, recall, rprec, recall_at_5 = evaluate_kilt_dataset(
            model,
            dataset,
            args.batch_size,
            args.beams,
            args.max_len_a,
            args.max_len_b,
            args.candidates,
            trie,
            title2id,
            args.free_generation,
            args.test,
        )

        results.add_row(
            [
                os.path.splitext(os.path.basename(dataset_filename))[0],
            ]
            + [
                "{:.2f}".format(100 * e)
                for e in (f1, precision, recall, rprec, recall_at_5)
            ]
        )

        output_filename = (
            os.path.join(args.output_path, os.path.basename(dataset_filename))
            if os.path.isdir(args.output_path)
            else args.output_path
        )
        logging.info("Saving dataset in {}".format(output_filename))
        with jsonlines.open(output_filename, "w") as f:
            f.write_all(dataset)

    print(results)
