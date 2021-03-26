import argparse
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import jsonlines
from mgenre.utils import create_input
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        type=str,
        choices=[
            "titles_lang",
            "lang_titles",
            "canonical_title",
            "marginal",
        ],  # "titles_lang_v2"],
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
        "--langs",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--allowed_langs",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--random_n",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--abstracts",
        action="store_true",
    )
    parser.add_argument(
        "--target_switching",
        action="store_true",
    )
    parser.add_argument(
        "--target_switching_prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--monolingual",
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

    args.allowed_langs = set(args.allowed_langs.split("|"))

    assert not (args.monolingual and args.target_switching)

    wikidataID2canonical_lang_title = {}
    wikidataID2lang_title = {}
    if args.action == "canonical_title":
        filename = os.path.join(
            args.base_wikidata, "wikidataID2canonical_lang_title.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2canonical_lang_title = pickle.load(f)
    else:
        filename = os.path.join(
            args.base_wikidata, "wikidataID2lang_title-normalized.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2lang_title = pickle.load(f)

    for lang in args.langs.split("|"):
        filename = os.path.join(args.base_wikipedia, "{0}/{0}wiki.pkl".format(lang))
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wiki = pickle.load(f)

        flag = False
        for page in tqdm(wiki.values(), desc=lang):
            for a in page["anchors"]:
                if (
                    page["paragraphs"][a["paragraph_id"]][a["start"] : a["end"]]
                    != a["text"]
                ):
                    a["paragraph_id"] -= 1
                    flag = True
                assert (
                    page["paragraphs"][a["paragraph_id"]][a["start"] : a["end"]]
                    == a["text"]
                )

        if flag:
            filename = os.path.join(args.base_wikipedia, "{0}/{0}wiki.pkl".format(lang))
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(wiki, f)

        fs_name = os.path.join(args.base_wikipedia, "{}/{}{}{}.source").format(
            lang,
            args.action,
            "_abstract" if args.abstracts else "",
            "_target_switching" if args.target_switching else "",
            "_monolingual" if args.monolingual else "",
        )

        ft_name = os.path.join(args.base_wikipedia, "{}/{}{}{}.target").format(
            lang,
            args.action,
            "_abstract" if args.abstracts else "",
            "_target_switching" if args.target_switching else "",
            "_monolingual" if args.monolingual else "",
        )

        logging.info("Creating {}".format(fs_name))
        logging.info("Creating {}".format(ft_name))
        with open(fs_name, "w") as fs, open(ft_name, "w") as ft:

            for page in tqdm(wiki.values()):

                max_paragraph_id = 0 if args.abstracts else len(page["paragraphs"])
                while (
                    max_paragraph_id < len(page["paragraphs"])
                    and "Section::::" not in page["paragraphs"][max_paragraph_id]
                ):
                    max_paragraph_id += 1

                for anchor in page["anchors"]:
                    if (
                        len(anchor["wikidata_ids"]) == 1
                        and anchor["paragraph_id"] < max_paragraph_id
                        and (
                            anchor["wikidata_ids"][0] in wikidataID2lang_title
                            or anchor["wikidata_ids"][0]
                            in wikidataID2canonical_lang_title
                        )
                    ):
                        left_context = page["paragraphs"][anchor["paragraph_id"]][
                            : anchor["start"]
                        ].strip()
                        mention = page["paragraphs"][anchor["paragraph_id"]][
                            anchor["start"] : anchor["end"]
                        ].strip()
                        right_context = page["paragraphs"][anchor["paragraph_id"]][
                            anchor["end"] :
                        ].strip()

                        if mention == "":
                            continue

                        input_ = (
                            create_input(
                                {
                                    "input": "{} [START] {} [END] {}".format(
                                        left_context, mention, right_context
                                    ).strip(),
                                    "meta": {
                                        "left_context": left_context,
                                        "mention": mention,
                                        "right_context": right_context,
                                    },
                                },
                                128,
                            )
                            .replace("\n", ">>")
                            .replace("\r", ">>")
                        )

                        if args.action == "titles_lang" or args.action == "lang_titles":

                            tmp_dict = dict(
                                wikidataID2lang_title[anchor["wikidata_ids"][0]]
                            )
                            title = tmp_dict.get(lang, None)

                            if title and (
                                args.monolingual
                                or (not args.target_switching)
                                or (
                                    args.target_switching
                                    and np.random.rand() > args.target_switching_prob
                                )
                            ):
                                if args.action == "titles_lang":
                                    output_ = "{} >> {}".format(title, lang)
                                else:
                                    output_ = "{} >> {}".format(lang, title)

                                fs.write(input_ + "\n")
                                ft.write(output_ + "\n")

                            elif not args.monolingual:
                                if args.action == "titles_lang":
                                    choices = [
                                        "{} >> {}".format(title, lang2)
                                        for lang2, title in tmp_dict.items()
                                        if lang2 in args.allowed_langs and lang2 != lang
                                    ]
                                else:
                                    choices = [
                                        "{} >> {}".format(lang2, title)
                                        for lang2, title in tmp_dict.items()
                                        if lang2 in args.allowed_langs and lang2 != lang
                                    ]

                                for output_ in np.random.choice(
                                    choices,
                                    min(len(choices), args.random_n),
                                    replace=False,
                                ):
                                    fs.write(input_ + "\n")
                                    ft.write(output_ + "\n")

                        elif args.action == "canonical_title":

                            if (
                                anchor["wikidata_ids"][0]
                                in wikidataID2canonical_lang_title
                            ):
                                lang, title = wikidataID2canonical_lang_title[
                                    anchor["wikidata_ids"][0]
                                ]
                                fs.write(input_ + "\n")
                                ft.write("{} >> {}".format(title, lang) + "\n")

                        elif args.action == "marginal":
                            output_ = " || ".join(
                                "{} >> {}".format(lang2, title)
                                for lang2, title in wikidataID2lang_title[
                                    anchor["wikidata_ids"][0]
                                ]
                                if lang2 in args.allowed_langs
                            )
                            if output_:
                                fs.write(input_ + "\n")
                                ft.write(output_ + "\n")
