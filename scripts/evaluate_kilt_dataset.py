import os
import re
import math
import json
import string
import torch
import pickle
import jsonlines
import logging
import argparse

from tqdm import tqdm
from copy import deepcopy
from genre.trie import Trie
from prettytable import PrettyTable
from genre.utils import add_to_trie, chunk_it
from genre.base_model import GENRE
from fairseq.models.bart import BARTModel
from kilt.eval_retrieval import compute


def evaluate_kilt_dataset(
    model,
    dataset,
    batch_size=4,
    beams=10,
    max_len_b=15,
    candidates=False,
    trie=None,
    title2id={},
):

    dataset_original = deepcopy(dataset)

    gold = []
    pred = []

    iter_ = tqdm(chunk_it(dataset, len(dataset) // batch_size), desc="Evaluating")
    for docs in iter_:

        batch_trie = {
            i: (
                (
                    Trie(
                        [[2] + model.encode(e).tolist()[1:] for e in doc["candidates"]]
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
            [doc["input"] for doc in docs],
            beam=beams,
            max_len_b=max_len_b,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        for doc, out in zip(docs, outputs):
            gold.append(doc["output"][0]["answer"])
            pred.append(out[0]["text"])

            doc["output"] = [
                {
                    "answer": "",
                    "provenance": [
                        {
                            "wikipedia_id": title2id.get(prov["text"], None),
                            "title": prov["text"],
                            "score": prov["logprob"],
                        }
                        for prov in out
                    ],
                }
            ]

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
            (2 * precision * recall / (precision + recall)) if precision + recall else 0
        )

        iter_.set_postfix(f1=f1, prec=precision, rec=recall)

    kilt_dict = compute(dataset_original, dataset, ks=[1, 5], rank_keys=["title"])

    return dataset, f1, precision, recall, kilt_dict["Rprec"], kilt_dict["recall@5"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "checkpoint_file",
        type=str,
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
        "--max_len_b",
        default=15,
        type=int,
        help="Max output length",
    )
    parser.add_argument(
        "--trie",
        default="data/kilt/trie.pkl",
        type=str,
        help="Trie pickle file",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Enables the use of provided candidates",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="CPU/GPU device",
    )
    parser.add_argument(
        "--id_title",
        default="data/kilt/id_title.json",
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

    args = parser.parse_args()

    assert (os.path.isdir(args.input_path) and os.path.isdir(args.output_path)) or (
        not os.path.isdir(args.input_path) and not os.path.isdir(args.output_path)
    ), "`input_path` and `output_path` has either to be both files or folders"

    logging.basicConfig(level=args.loglevel)

    logging.info("Loading model")
    model = (
        GENRE.from_pretrained(args.model_path, checkpoint_file=args.checkpoint_file)
        .eval()
        .to(args.device)
    )

    logging.info("Loading Trie")
    if not args.candidates:
        with open(args.trie, "rb") as f:
            trie = pickle.load(f)
    else:
        trie = None

    if args.id_title is not None:
        logging.info("Loading ID to title map")
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
            args.max_len_b,
            args.candidates,
            trie,
            title2id,
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

        logging.info("Saving dataset")
        output_filename = (
            os.path.join(args.output_path, os.path.basename(dataset_filename))
            if os.path.isdir(args.output_path)
            else args.output_path
        )
        with jsonlines.open(output_filename, "w") as f:
            f.write_all(dataset)

    print(results)
