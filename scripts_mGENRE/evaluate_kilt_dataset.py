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
import pprint
import re
import string
import unicodedata
from collections import defaultdict
from copy import deepcopy

import numpy as np
import requests

import jsonlines
from hanziconv import HanziConv
from mgenre import mGENRE
from mgenre.utils import add_to_trie, batch_it, create_input, get_from_trie_dict
from prettytable import PrettyTable
from tqdm.auto import tqdm


# class ServerDict:
#     def __init__(self, name):
#         self.name = name

#     def get(self, key, value=None):
#         return json.loads(
#             requests.post(
#                 "http://100.97.69.169:5555",
#                 json={"dict": self.name, "action": "get", "key": key, "value": value},
#             ).content
#         )

#     def __getitem__(self, key):
#         return json.loads(
#             requests.post(
#                 "http://100.97.69.169:5555",
#                 json={"dict": self.name, "action": "__getitem__", "key": key},
#             ).content
#         )


#     def get_from_trie_dict(indices):
#         return json.loads(
#             requests.post(
#                 "http://100.97.69.169:5555",
#                 json={"indices": indices, "action": "get_from_trie_dict"}
#             ).content
#         )


def evaluate_kilt_dataset(
    model,
    dataset,
    batch_size=4,
    beams=10,
    max_len_a=128,
    max_len_b=32,
    lenpen=1,
    trie=None,
    lang_title2wikidataID={},
    wikidataID2lang_title={},
    canonical_lang_title2wikidataID={},
    wikidataID2canonical_lang_title={},
    order="title_lang",
    canonical=False,
    free_generation=False,
    mention2wikidataID={},
    candidates_lowercase=False,
    allowed_langs=[],
    desc=None,
    max_candidates=None,
    only_en_candidates=False,
    only_freebase_candidates=False,
    wikidataID2freebaseID={},
):
    gold = []
    pred = []

    iter_ = tqdm(dataset, desc="Evaluating {}".format(desc if desc else ""))

    for docs in batch_it(iter_, batch_size):

        if not free_generation:
            batch_trie = {}
            for i, doc in enumerate(docs):
                mention = (
                    unicodedata.normalize(
                        "NFKD", HanziConv.toSimplified(doc["meta"]["mention"])
                    )
                    .replace("•", "·")
                    .replace("．", "·")
                )

                candidates = list(mention2wikidataID.get(mention, {}).items())

                if candidates_lowercase:
                    candidates += list(
                        mention2wikidataID.get(mention.lower(), {}).items()
                    )

                candidates_tmp = defaultdict(int)
                for k, v in candidates:
                    candidates_tmp[k] += v

                candidates = [
                    e[0]
                    for e in sorted(
                        candidates_tmp.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ]

                if only_en_candidates:
                    candidates = [
                        cand
                        for cand in candidates
                        if "en" in dict(wikidataID2lang_title[cand])
                    ]
                if only_freebase_candidates:
                    candidates = [
                        cand for cand in candidates if cand in wikidataID2freebaseID
                    ]

                candidates = candidates[:max_candidates]

                if mention2wikidataID and candidates:
                    if canonical:
                        batch_bpes = [
                            [2]
                            + model.encode(
                                "{} >> {}".format(
                                    *tuple(
                                        reversed(wikidataID2canonical_lang_title[cand])
                                    )
                                    if order == "title_lang"
                                    else "{} >> {}".format(
                                        *wikidataID2canonical_lang_title[cand]
                                    )
                                )
                            ).tolist()[1:]
                            for cand in candidates
                            if cand in wikidataID2canonical_lang_title
                        ]
                    else:
                        batch_bpes = [
                            [2]
                            + model.encode(
                                "{} >> {}".format(title, lang)
                                if order == "title_lang"
                                else "{} >> {}".format(lang, title)
                            ).tolist()[1:]
                            for cand in candidates
                            #                             if cand in wikidataID2lang_title
                            for lang, title in wikidataID2lang_title.get(cand, [])
                            if lang in allowed_langs
                        ]

                    if batch_bpes:
                        batch_trie[i] = {}
                        for e in batch_bpes:
                            add_to_trie(e, batch_trie[i])

                    else:
                        batch_trie[i] = trie

                else:
                    batch_trie[i] = trie

            def prefix_allowed_tokens_fn(batch_id, sent):
                return [
                    e
                    for e in get_from_trie_dict(sent.tolist(), batch_trie[batch_id])
                    if e < len(model.task.target_dictionary)
                ]

        outputs = model.sample(
            [create_input(doc, max_len_a) for doc in docs],
            beam=beams,
            lenpen=lenpen,
            max_len_b=max_len_b,
            prefix_allowed_tokens_fn=None
            if free_generation
            else prefix_allowed_tokens_fn,
        )

        for doc, out in zip(docs, outputs):

            doc["predictions"] = [
                {
                    "answer": list(
                        [
                            canonical_lang_title2wikidataID.get(
                                tuple(
                                    reversed(o["text"].split(" >> "))
                                    if order == "title_lang"
                                    else o["text"].split(" >> ")
                                ),
                                None,
                            )
                        ]
                        if canonical
                        else lang_title2wikidataID.get(
                            tuple(
                                reversed(o["text"].split(" >> "))
                                if order == "title_lang"
                                else o["text"].split(" >> ")
                            ),
                            [None],
                        )
                    ),
                    "text": o["text"],
                    "logprob": o["logprob"].item(),
                }
                for o in out
            ]

            gold.append(doc["output"][0]["answer"])

            try:
                pred.append(doc["predictions"][0]["answer"])
            except Exception as e:
                pred.append([None])

        true_pos = 0
        for g, p in zip(gold, pred):
            if set(g).intersection(set(p)) and p != [None]:
                true_pos += 1

        precision = (
            (true_pos / len([p for p in pred if p != [None]]))
            if len([p for p in pred if p != [None]])
            else 0
        )
        recall = (true_pos / len(gold)) if len(gold) else 0
        f1 = (
            (2 * precision * recall / (precision + recall)) if precision + recall else 0
        )
        accuracy = [
            (set(g).intersection(set(p)) and p != [None]) or (g == [] and p == [None])
            for g, p in zip(gold, pred)
        ]
        accuracy = sum(accuracy) / len(accuracy)

        iter_.set_postfix(f1=f1, prec=precision, rec=recall, acc=accuracy)

    return dataset, f1, precision, recall, accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        help="Model path",
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
        "--checkpoint_file",
        type=str,
        default="model.pt",
        help="Checkpoint file",
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
        default=128,
        type=int,
        help="Max input length",
    )
    parser.add_argument(
        "--max_len_b",
        default=32,
        type=int,
        help="Max output length",
    )
    parser.add_argument(
        "--lenpen",
        default=1,
        type=int,
        help="Length penalty",
    )
    parser.add_argument(
        "--trie",
        type=str,
        help="Trie pickle file",
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
        "--lang_title2wikidataID",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/lang_title2wikidataID-normalized.pkl",
    )
    parser.add_argument(
        "--wikidataID2lang_title",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/wikidataID2lang_title-normalized.pkl",
    )
    parser.add_argument(
        "--canonical_lang_title2wikidataID",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/canonical_lang_title2wikidataID.pkl",
    )
    parser.add_argument(
        "--wikidataID2canonical_lang_title",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/wikidataID2canonical_lang_title.pkl",
    )
    parser.add_argument(
        "--wikidataID2freebaseID",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/wikidataID2freebaseID.pkl",
    )
    parser.add_argument(
        "--candidates",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--candidates_lowercase",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--only_en_candidates",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--only_freebase_candidates",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mention2wikidataID",
        type=str,
        default="/checkpoint/fabiopetroni/mGENRE/wikidata/mention2wikidataID_with_titles_label_alias.pkl",
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["title_lang", "lang_title"],
    )
    parser.add_argument(
        "--server_dict",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--canonical",
        help="Whether to use provided canidates",
        action="store_true",
    )
    parser.add_argument(
        "--allowed_langs",
        type=str,
        default="af|am|ar|as|az|be|bg|bm|bn|br|bs|ca|cs|cy|da|de|el|en|eo|es|et|eu|fa|ff|fi|fr|fy|ga|gd|gl|gn|gu|ha|he|hi|hr|ht|hu|hy|id|ig|is|it|ja|jv|ka|kg|kk|km|kn|ko|ku|ky|la|lg|ln|lo|lt|lv|mg|mk|ml|mn|mr|ms|my|ne|nl|no|om|or|pa|pl|ps|pt|qu|ro|ru|sa|sd|si|sk|sl|so|sq|sr|ss|su|sv|sw|ta|te|th|ti|tl|tn|tr|uk|ur|uz|vi|wo|xh|yo|zh",
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

    args, _ = parser.parse_known_args()

    assert (os.path.isdir(args.input_path) and os.path.isdir(args.output_path)) or (
        not os.path.isdir(args.input_path) and not os.path.isdir(args.output_path)
    ), "`input_path` and `output_path` have either to be both files or folders"

    if not args.free_generation:
        assert args.canonical or args.order is not None

    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.info(pprint.pformat(args.__dict__))

    args.allowed_langs = set(args.allowed_langs.split("|"))

    logging.info("Loading model")
    if "cuda" not in args.device and torch.cuda.is_available():
        logging.warning(
            "CUDA is available but running on CPU. Set --device cuda:<ID> for running on GPU."
        )

    model = (
        mGENRE.from_pretrained(
            args.model_path,
            checkpoint_file=args.checkpoint_file,
            bpe="sentencepiece",
            layernorm_embedding=True,
            sentencepiece_model="/private/home/ndecao/mGENRE/models/mbart.cc100/spm_256000.model",
        )
        .eval()
        .to(args.device)
    )

    if not args.free_generation:
        logging.info("Loading Trie from {}".format(args.trie))

        with open(args.trie, "rb") as f:
            trie = pickle.load(f)

    else:
        trie = None

    canonical_lang_title2wikidataID = {}
    wikidataID2canonical_lang_title = {}
    lang_title2wikidataID = {}
    wikidataID2lang_title = {}
    wikidataID2freebaseID = {}
    mention2wikidataID = {}

    if args.canonical:
        if args.canonical_lang_title2wikidataID is not None:
            logging.info(
                "Loading canonical title to wikidataID map from {}".format(
                    args.canonical_lang_title2wikidataID
                )
            )
            with open(args.canonical_lang_title2wikidataID, "rb") as f:
                canonical_lang_title2wikidataID = pickle.load(f)

        if args.wikidataID2canonical_lang_title is not None:
            logging.info(
                "Loading wikidataID to canonical title map from {}".format(
                    args.wikidataID2canonical_lang_title
                )
            )
            with open(args.wikidataID2canonical_lang_title, "rb") as f:
                wikidataID2canonical_lang_title = pickle.load(f)

    else:
        if args.server_dict:
            lang_title2wikidataID = ServerDict("lang_title2wikidataID")
        elif args.lang_title2wikidataID is not None:
            logging.info(
                "Loading <lang, title> to wikidataID map from {}".format(
                    args.lang_title2wikidataID
                )
            )
            with open(args.lang_title2wikidataID, "rb") as f:
                lang_title2wikidataID = pickle.load(f)

        if args.server_dict:
            wikidataID2lang_title = ServerDict("wikidataID2lang_title")
        elif args.wikidataID2lang_title is not None:
            logging.info(
                "Loading wikidataID to <lang, title> map from {}".format(
                    args.wikidataID2lang_title
                )
            )
            with open(args.wikidataID2lang_title, "rb") as f:
                wikidataID2lang_title = pickle.load(f)

    if args.only_freebase_candidates:
        if args.server_dict:
            wikidataID2freebaseID = ServerDict("wikidataID2freebaseID")
        elif args.wikidataID2freebaseID is not None:
            logging.info(
                "Loading wikidataID to freebaseID from {}".format(
                    args.wikidataID2freebaseID
                )
            )
            with open(args.wikidataID2freebaseID, "rb") as f:
                wikidataID2freebaseID = pickle.load(f)

    if args.candidates:
        if args.server_dict:
            mention2wikidataID = ServerDict("mention2wikidataID")
        elif args.mention2wikidataID is not None:
            logging.info(
                "Loading mention to wikidataID map from {}".format(
                    args.mention2wikidataID
                )
            )
            with open(args.mention2wikidataID, "rb") as f:
                mention2wikidataID = pickle.load(f)

    results = PrettyTable()
    results.field_names = [
        "Dataset",
        "F1",
        "Precision",
        "Recall",
        "Accuracy",
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

        dataset, f1, precision, recall, accuracy = evaluate_kilt_dataset(
            model,
            np.random.permutation(dataset),
            args.batch_size,
            args.beams,
            args.max_len_a,
            args.max_len_b,
            args.lenpen,
            trie,
            lang_title2wikidataID,
            wikidataID2lang_title,
            canonical_lang_title2wikidataID,
            wikidataID2canonical_lang_title,
            args.order,
            args.canonical,
            args.free_generation,
            mention2wikidataID,
            args.candidates_lowercase,
            args.allowed_langs,
            os.path.basename(dataset_filename),
            args.max_candidates,
            args.only_en_candidates,
            args.only_freebase_candidates,
            wikidataID2freebaseID,
        )

        results.add_row(
            [
                os.path.splitext(os.path.basename(dataset_filename))[0],
            ]
            + ["{:.2f}".format(100 * e) for e in (f1, precision, recall, accuracy)]
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
