# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

import jsonlines
from mgenre.base_model import mGENRE
from mgenre.utils import chunk_it, extract_pages
from tqdm.auto import tqdm, trange

NOPAGE = [
    "Q4167836",
    "Q24046192",
    "Q20010800",
    "Q11266439",
    "Q11753321",
    "Q19842659",
    "Q21528878",
    "Q17362920",
    "Q14204246",
    "Q21025364",
    "Q17442446",
    "Q26267864",
    "Q4663903",
    "Q15184295",
    #     "Q4167410",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "step",
        type=str,
        choices=["compress", "normalize", "dicts", "redirects", "freebase"],
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
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

    if args.step == "compress":
        wikidata = 0
        with open(
            os.path.join(args.base_wikidata, "wikidata-all.json"), "r"
        ) as fi, jsonlines.open(
            os.path.join(args.base_wikidata, "wikidata-all-compressed.jsonl"), "w"
        ) as fo:

            iter_ = tqdm(fi)
            for i, line in enumerate(iter_):
                iter_.set_postfix(wikidata=wikidata, refresh=False)

                line = line.strip()
                if line[-1] == ",":
                    line = line[:-1]

                if line == "[" or line == "]":
                    continue

                line = json.loads(line)
                if line["type"] == "item":

                    if any(
                        e["mainsnak"]["datavalue"]["value"]["id"] in NOPAGE
                        for e in line["claims"].get("P31", {})
                        if "datavalue" in e["mainsnak"]
                    ):
                        continue
                    if any(
                        e["mainsnak"]["datavalue"]["value"]["id"] in NOPAGE
                        for e in line["claims"].get("P279", {})
                        if "datavalue" in e["mainsnak"]
                    ):
                        continue

                    line["sitelinks"] = {
                        k[:-4]: v["title"]
                        for k, v in line["sitelinks"].items()
                        if k.endswith("wiki")
                    }
                    if len(line["sitelinks"]) == 0:
                        continue

                    line["labels"] = {k: v["value"] for k, v in line["labels"].items()}
                    line["descriptions"] = {
                        k: v["value"] for k, v in line["descriptions"].items()
                    }
                    line["aliases"] = {
                        k: [e["value"] for e in v] for k, v in line["aliases"].items()
                    }

                    for e in ("claims", "lastrevid", "type"):
                        del line[e]

                    fo.write(line)
                    wikidata += 1

    elif args.step == "normalize":

        mgenre = mGENRE.from_pretrained(
            "models/mbart.cc100",
            checkpoint_file="model.pt",
            bpe="sentencepiece",
            layernorm_embedding=True,
            sentencepiece_model="models/spm_256000.model",
        ).eval()

        with jsonlines.open(
            os.path.join(args.base_wikidata, "wikidata-all-compressed.jsonl"), "r"
        ) as fi, jsonlines.open(
            os.path.join(
                args.base_wikidata, "wikidata-all-compressed-normalized.jsonl"
            ),
            "w",
        ) as fo:

            for item in tqdm(fi):
                item["sitelinks"] = {
                    lang: mgenre.decode(mgenre.encode(title))
                    for lang, title in item["sitelinks"].items()
                }
                item["labels"] = {
                    lang: mgenre.decode(mgenre.encode(label))
                    for lang, label in item["labels"].items()
                }
                item["descriptions"] = {
                    lang: mgenre.decode(mgenre.encode(desc))
                    for lang, desc in item["descriptions"].items()
                }
                item["aliases"] = {
                    lang: [mgenre.decode(mgenre.encode(alias)) for alias in aliases]
                    for lang, aliases in item["aliases"].items()
                }
                fo.write(item)

    elif args.step == "dicts":

        lang_title2wikidataID = defaultdict(set)
        wikidataID2lang_title = defaultdict(set)
        label_desc2wikidataID = defaultdict(set)
        wikidataID2label_desc_lang = defaultdict(set)
        label_or_alias2wikidataID = defaultdict(set)
        wikidataID2label_or_alias = defaultdict(set)
        wikidataID2lang2label_or_alias = defaultdict(lambda: defaultdict(set))

        filename = os.path.join(
            args.base_wikidata,
            "wikidata-all-compressed{}.jsonl".format(
                "-normalized" if args.normalized else ""
            ),
        )
        logging.info("Processing {}".format(filename))
        with jsonlines.open(filename, "r") as f:

            for item in tqdm(f):
                for lang, title in item["sitelinks"].items():
                    lang_title2wikidataID[(lang, title)].add(item["id"])
                    wikidataID2lang_title[item["id"]].add((lang, title))

                for lang, label in item["labels"].items():
                    if lang in item["descriptions"]:
                        label_desc2wikidataID[(label, item["descriptions"][lang])].add(
                            item["id"]
                        )
                        wikidataID2label_desc_lang[item["id"]].add(
                            (label, item["descriptions"][lang], lang)
                        )

                for lang, aliases in item["aliases"].items():
                    for alias in aliases:
                        label_or_alias2wikidataID[alias.lower()].add(item["id"])
                        wikidataID2label_or_alias[item["id"]].add(alias)
                        wikidataID2lang2label_or_alias[item["id"]][lang].add(alias)

                for lang, label in item["labels"].items():
                    label_or_alias2wikidataID[label.lower()].add(item["id"])
                    wikidataID2label_or_alias[item["id"]].add(label)
                    wikidataID2lang2label_or_alias[item["id"]][lang].add(label)

        wikidataID2lang2label_or_alias = {
            wikidataID: dict(lang2label_or_alias)
            for wikidataID, lang2label_or_alias in wikidataID2lang2label_or_alias.items()
        }

        for data, name in zip(
            (
                lang_title2wikidataID,
                wikidataID2lang_title,
                label_desc2wikidataID,
                wikidataID2label_desc_lang,
                label_or_alias2wikidataID,
                wikidataID2label_or_alias,
                wikidataID2lang2label_or_alias,
            ),
            (
                "lang_title2wikidataID",
                "wikidataID2lang_title",
                "label_desc2wikidataID",
                "wikidataID2label_desc_lang",
                "label_or_alias2wikidataID",
                "wikidataID2label_or_alias",
                "wikidataID2lang2label_or_alias",
            ),
        ):

            filename = os.path.join(
                args.base_wikidata,
                "{}{}.pkl".format(name, "-normalized" if args.normalized else ""),
            )
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(dict(data), f)

    elif args.step == "redirects":

        lang_redirect2title = {}
        for lang in set(wiki_langs).intersection(set(mbart100_langs)):
            with open(
                "wikipedia_redirect/target/{}wiki-redirects.txt".format(lang)
            ) as f:
                for row in tqdm(csv.reader(f, delimiter="\t"), desc=lang):
                    title = unquote(row[1]).split("#")[0].replace("_", " ")
                    if title:
                        title = title[0].upper() + title[1:]
                        assert (lang, row[0]) not in lang_redirect2title
                        lang_redirect2title[(lang, row[0])] = title

        filename = os.path.join(args.base_wikidata, "lang_redirect2title.pkl")
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(lang_redirect2title, f)

    elif args.step == "freebase":

        wikidataID2freebaseID = defaultdict(list)
        freebaseID2wikidataID = defaultdict(list)

        with open(os.path.join(args.base_wikidata, "wikidata-all.json"), "r") as fi:

            iter_ = tqdm(fi)
            for i, line in enumerate(iter_):
                line = line.strip()
                if line[-1] == ",":
                    line = line[:-1]

                if line == "[" or line == "]":
                    continue

                line = json.loads(line)

                if line["type"] == "item":

                    if any(
                        e["mainsnak"]["datavalue"]["value"]["id"] in NOPAGE
                        for e in line["claims"].get("P31", {})
                        if "datavalue" in e["mainsnak"]
                    ):
                        continue
                    if any(
                        e["mainsnak"]["datavalue"]["value"]["id"] in NOPAGE
                        for e in line["claims"].get("P279", {})
                        if "datavalue" in e["mainsnak"]
                    ):
                        continue

                    line["sitelinks"] = {
                        k[:-4]: v["title"]
                        for k, v in line["sitelinks"].items()
                        if k.endswith("wiki")
                    }
                    if len(line["sitelinks"]) == 0:
                        continue

                    for freebaseID in [
                        e["mainsnak"]["datavalue"]["value"]
                        for e in line["claims"].get("P646", {})
                        if "datavalue" in e["mainsnak"]
                    ]:
                        wikidataID2freebaseID[line["id"]].append(freebaseID)
                        freebaseID2wikidataID[freebaseID].append(line["id"])

        wikidataID2freebaseID = dict(wikidataID2freebaseID)
        filename = os.path.join(args.base_wikidata, "wikidataID2freebaseID.pkl")
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(wikidataID2freebaseID, f)

        freebaseID2wikidataID = dict(freebaseID2wikidataID)
        filename = os.path.join(args.base_wikidata, "freebaseID2wikidataID.pkl")
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(freebaseID2wikidataID, f)
