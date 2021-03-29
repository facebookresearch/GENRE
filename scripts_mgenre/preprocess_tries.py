import argparse
import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonlines
from mgenre.base_model import mGENRE
from mgenre.utils import add_to_trie, chunk_it, extract_pages
from tqdm.auto import tqdm, trange

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        type=str,
        choices=[
            "titles",
            "titles_lang_trie_append",
            "lang_titles_trie_append",
            "canonical",
        ],
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "--allowed_langs",
        type=str,
        default="af|am|ar|as|az|be|bg|bm|bn|br|bs|ca|cs|cy|da|de|el|en|eo|es|et|eu|fa|ff|fi|fr|fy|ga|gd|gl|gn|gu|ha|he|hi|hr|ht|hu|hy|id|ig|is|it|ja|jv|ka|kg|kk|km|kn|ko|ku|ky|la|lg|ln|lo|lt|lv|mg|mk|ml|mn|mr|ms|my|ne|nl|no|om|or|pa|pl|ps|pt|qu|ro|ru|sa|sd|si|sk|sl|so|sq|sr|ss|su|sv|sw|ta|te|th|ti|tl|tn|tr|uk|ur|uz|vi|wo|xh|yo|zh",
        help="Pipe (|) separated list of allowed language ID to use.",
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

    mgenre = mGENRE.from_pretrained(
        "models/mbart.cc100",
        checkpoint_file="model.pt",
        bpe="sentencepiece",
        layernorm_embedding=True,
        sentencepiece_model="models/mbart.cc100/spm_256000.model",
    ).eval()

    if args.action == "titles":

        filename = os.path.join(
            args.base_wikidata, "lang_title2wikidataID-normalized.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            lang_title2wikidataID = pickle.load(f)

        lang_titles2bpes = {
            (lang, title): mgenre.encode(title).tolist()
            for lang, title in tqdm(lang_title2wikidataID.keys())
        }

        filename = os.path.join(args.base_wikidata, "lang_titles2bpes-normalized.pkl")
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(lang_titles2bpes, f)

    elif (
        args.action == "titles_lang_trie_append"
        or args.action == "lang_titles_trie_append"
    ):

        filename = os.path.join(args.base_wikidata, "lang_titles2bpes-normalized.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            lang_titles2bpes = pickle.load(f)

        lang2titles2bpes = defaultdict(dict)
        lang_codes = {}
        if args.action == "titles_lang_trie_append":
            for (lang, title), bpes in tqdm(lang_titles2bpes.items()):
                if lang not in lang_codes:
                    lang_codes[lang] = mgenre.encode(" >> {}".format(lang)).tolist()
                lang2titles2bpes[lang][title] = [2] + bpes[1:-1] + lang_codes[lang][1:]
        elif args.action == "lang_titles_trie_append":
            for (lang, title), bpes in tqdm(lang_titles2bpes.items()):
                if lang not in lang_codes:
                    lang_codes[lang] = mgenre.encode("{} >>".format(lang)).tolist()
                lang2titles2bpes[lang][title] = [2] + lang_codes[lang][1:-1] + bpes[1:]

        trie = {}
        for lang in sorted(args.allowed_langs):
            for sequence in tqdm(lang2titles2bpes[lang].values(), desc=lang):
                add_to_trie(sequence, trie)

        if args.action == "titles_lang_trie_append":
            filename = os.path.join(args.base_wikidata, "titles_lang_all105_trie.pkl")
        elif args.action == "lang_titles_trie_append":
            filename = os.path.join(args.base_wikidata, "lang_titles_all105_trie.pkl")

        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(trie, f)

    elif args.action == "canonical":

        filename = os.path.join(args.base_wikidata, "lang_titles2bpes-normalized.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            lang_titles2bpes = pickle.load(f)

        filename = os.path.join(
            args.base_wikidata, "wikidataID2lang_title-normalized.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2lang_title = pickle.load(f)

        filename = os.path.join(args.base_wikidata, "wikidataID2lang_priority.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2lang_priority = pickle.load(f)

        all_priority = Counter(
            [x for e in wikidataID2lang_priority.values() for x in e]
        )
        all_priority = [
            e[0] for e in sorted(all_priority.items(), key=lambda x: x[1], reverse=True)
        ]

        trie = {}
        wikidataID2canonical_lang_title = {}
        canonical_lang_title2wikidataID = {}
        lang_codes = {}
        for wikidata_id in tqdm(wikidataID2lang_title):
            sorted_langs = [
                e
                for e in wikidataID2lang_priority.get(wikidata_id, all_priority)
                if e in args.allowed_langs
            ]

            i = 0
            while i < len(sorted_langs):
                title = dict(wikidataID2lang_title[wikidata_id]).get(
                    sorted_langs[i], None
                )

                if title:
                    wikidataID2canonical_lang_title[wikidata_id] = (
                        sorted_langs[i],
                        title,
                    )
                    canonical_lang_title2wikidataID[
                        (sorted_langs[i], title)
                    ] = wikidata_id

                    if sorted_langs[i] not in lang_codes:
                        lang_codes[sorted_langs[i]] = mgenre.encode(
                            " >> {}".format(sorted_langs[i])
                        ).tolist()

                    add_to_trie(
                        [2]
                        + lang_titles2bpes[(sorted_langs[i], title)][1:-1]
                        + lang_codes[sorted_langs[i]][1:],
                        trie,
                    )
                    break
                else:
                    i += 1

        filename = os.path.join(args.base_wikidata, "canonical_trie.pkl")
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(trie, f)

        filename = os.path.join(
            args.base_wikidata, "wikidataID2canonical_lang_title.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(wikidataID2canonical_lang_title, f)

        filename = os.path.join(
            args.base_wikidata, "canonical_lang_title2wikidataID.pkl"
        )
        logging.info("Saving {}".format(filename))
        with open(filename, "wb") as f:
            pickle.dump(canonical_lang_title2wikidataID, f)
