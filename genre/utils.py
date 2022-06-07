# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import html
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import unquote

from bs4 import BeautifulSoup

from genre.entity_linking import (
    get_end_to_end_prefix_allowed_tokens_fn_fairseq,
    get_end_to_end_prefix_allowed_tokens_fn_hf,
)


def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def batch_it(seq, num=1):
    out = []
    for item in seq:
        if len(out) == num:
            yield out
            out = []
        out.append(item)

    if len(out):
        yield out


def create_input(doc, max_length, start_delimiter, end_delimiter):
    if "meta" in doc and all(
        e in doc["meta"] for e in ("left_context", "mention", "right_context")
    ):
        if len(doc["input"].split(" ")) <= max_length:
            input_ = (
                doc["meta"]["left_context"]
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + doc["meta"]["right_context"]
            )
        elif len(doc["meta"]["left_context"].split(" ")) <= max_length // 2:
            input_ = (
                doc["meta"]["left_context"]
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + " ".join(
                    doc["meta"]["right_context"].split(" ")[
                        : max_length - len(doc["meta"]["left_context"].split(" "))
                    ]
                )
            )
        elif len(doc["meta"]["right_context"].split(" ")) <= max_length // 2:
            input_ = (
                " ".join(
                    doc["meta"]["left_context"].split(" ")[
                        len(doc["meta"]["right_context"].split(" ")) - max_length :
                    ]
                )
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + doc["meta"]["right_context"]
            )
        else:
            input_ = (
                " ".join(doc["meta"]["left_context"].split(" ")[-max_length // 2 :])
                + " {} ".format(start_delimiter)
                + doc["meta"]["mention"]
                + " {} ".format(end_delimiter)
                + " ".join(doc["meta"]["right_context"].split(" ")[: max_length // 2])
            )
    else:
        input_ = doc["input"]

    input_ = html.unescape(input_)

    return input_


def get_entity_spans_pre_processing(sentences):
    return [
        (
            " {} ".format(sent)
            .replace("\xa0", " ")
            .replace("{", "(")
            .replace("}", ")")
            .replace("[", "(")
            .replace("]", ")")
        )
        for sent in sentences
    ]


def get_entity_spans_post_processing(sentences):
    outputs = []
    for sent in sentences:
        sent = re.sub(r"{.*?", "{ ", sent)
        sent = re.sub(r"}.*?", "} ", sent)
        sent = re.sub(r"\].*?", "] ", sent)
        sent = re.sub(r"\[.*?", "[ ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        sent = re.sub(r"\. \. \} \[ (.*?) \]", r". } [ \1 ] .", sent)
        sent = re.sub(r"\, \} \[ (.*?) \]", r" } [ \1 ] ,", sent)
        sent = re.sub(r"\; \} \[ (.*?) \]", r" } [ \1 ] ;", sent)
        sent = sent.replace("{ ", "{").replace(" } [ ", "}[").replace(" ]", "]")
        outputs.append(sent)

    return outputs


def _get_entity_spans(
    model, input_sentences, prefix_allowed_tokens_fn, redirections=None,
):
    output_sentences = model.sample(
        get_entity_spans_pre_processing(input_sentences),
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    output_sentences = get_entity_spans_post_processing(
        [e[0]["text"] for e in output_sentences]
    )

    return get_entity_spans_finalize(
        input_sentences, output_sentences, redirections=redirections
    )


def get_entity_spans_fairseq(
    model,
    input_sentences,
    mention_trie=None,
    candidates_trie=None,
    mention_to_candidates_dict=None,
    redirections=None,
):
    return _get_entity_spans(
        model,
        input_sentences,
        prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_fairseq(
            model,
            get_entity_spans_pre_processing(input_sentences),
            mention_trie=mention_trie,
            candidates_trie=candidates_trie,
            mention_to_candidates_dict=mention_to_candidates_dict,
        ),
        redirections=redirections,
    )


def get_entity_spans_hf(
    model,
    input_sentences,
    mention_trie=None,
    candidates_trie=None,
    mention_to_candidates_dict=None,
    redirections=None,
):
    return _get_entity_spans(
        model,
        input_sentences,
        prefix_allowed_tokens_fn=get_end_to_end_prefix_allowed_tokens_fn_hf(
            model,
            get_entity_spans_pre_processing(input_sentences),
            mention_trie=mention_trie,
            candidates_trie=candidates_trie,
            mention_to_candidates_dict=mention_to_candidates_dict,
        ),
        redirections=redirections,
    )


def get_entity_spans_finalize(input_sentences, output_sentences, redirections=None):

    return_outputs = []
    for input_, output_ in zip(input_sentences, output_sentences):
        input_ = input_.replace("\xa0", " ") + "  -"
        output_ = output_.replace("\xa0", " ") + "  -"

        entities = []
        status = "o"
        i = 0
        j = 0
        while j < len(output_) and i < len(input_):

            if status == "o":
                if input_[i] == output_[j] or (
                    output_[j] in "()" and input_[i] in "[]{}"
                ):
                    i += 1
                    j += 1
                elif output_[j] == " ":
                    j += 1
                elif input_[i] == " ":
                    i += 1
                elif output_[j] == "{":
                    entities.append([i, 0, ""])
                    j += 1
                    status = "m"
                else:
                    raise RuntimeError

            elif status == "m":
                if input_[i] == output_[j]:
                    i += 1
                    j += 1
                    entities[-1][1] += 1
                elif output_[j] == " ":
                    j += 1
                elif input_[i] == " ":
                    i += 1
                elif output_[j] == "}":
                    j += 1
                    status = "e"
                else:
                    raise RuntimeError

            elif status == "e":
                if output_[j] == "[":
                    j += 1
                elif output_[j] != "]":
                    entities[-1][2] += output_[j]
                    j += 1
                elif output_[j] == "]":
                    entities[-1][2] = entities[-1][2].replace(" ", "_")
                    if len(entities[-1][2]) <= 1:
                        del entities[-1]
                    elif entities[-1][2] == "NIL":
                        del entities[-1]
                    elif redirections is not None and entities[-1][2] in redirections:
                        entities[-1][2] = redirections[entities[-1][2]]

                    if len(entities) > 0:
                        entities[-1] = tuple(entities[-1])

                    status = "o"
                    j += 1
                else:
                    raise RuntimeError

        return_outputs.append(entities)

    return return_outputs


def get_markdown(sentences, entity_spans):
    return_outputs = []
    for sent, entities in zip(sentences, entity_spans):
        text = ""
        last_end = 0
        for begin, length, href in entities:
            text += sent[last_end:begin]
            text += "[{}](https://en.wikipedia.org/wiki/{})".format(
                sent[begin : begin + length], href
            )
            last_end = begin + length

        text += sent[last_end:]
        return_outputs.append(text)

    return return_outputs


def strong_tp(guess_entities, gold_entities):
    return len(gold_entities.intersection(guess_entities))


def weak_tp(guess_entities, gold_entities):
    tp = 0
    for pred in guess_entities:
        for gold in gold_entities:
            if (
                pred[0] == gold[0]
                and (
                    gold[1] <= pred[1] <= gold[1] + gold[2]
                    or gold[1] <= pred[1] + pred[2] <= gold[1] + gold[2]
                )
                and pred[3] == gold[3]
            ):
                tp += 1

    return tp


def get_micro_precision(guess_entities, gold_entities, mode="strong"):
    guess_entities = set(guess_entities)
    gold_entities = set(gold_entities)

    if mode == "strong":
        return (
            (strong_tp(guess_entities, gold_entities) / len(guess_entities))
            if len(guess_entities)
            else 0
        )
    elif mode == "weak":
        return (
            (weak_tp(guess_entities, gold_entities) / len(guess_entities))
            if len(guess_entities)
            else 0
        )


def get_micro_recall(guess_entities, gold_entities, mode="strong"):
    guess_entities = set(guess_entities)
    gold_entities = set(gold_entities)

    if mode == "strong":
        return (
            (strong_tp(guess_entities, gold_entities) / len(gold_entities))
            if len(gold_entities)
            else 0
        )
    elif mode == "weak":
        return (
            (weak_tp(guess_entities, gold_entities) / len(gold_entities))
            if len(gold_entities)
            else 0
        )


def get_micro_f1(guess_entities, gold_entities, mode="strong"):
    precision = get_micro_precision(guess_entities, gold_entities, mode)
    recall = get_micro_recall(guess_entities, gold_entities, mode)
    return (
        (2 * (precision * recall) / (precision + recall)) if precision + recall else 0
    )


def get_doc_level_guess_gold_entities(guess_entities, gold_entities):
    new_guess_entities = defaultdict(list)
    for e in guess_entities:
        new_guess_entities[e[0]].append(e)

    new_gold_entities = defaultdict(list)
    for e in gold_entities:
        new_gold_entities[e[0]].append(e)

    return new_guess_entities, new_gold_entities


def get_macro_precision(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_precision(guess_entities[k], gold_entities[k], mode)
        for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0


def get_macro_recall(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_recall(guess_entities[k], gold_entities[k], mode)
        for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0


def get_macro_f1(guess_entities, gold_entities, mode="strong"):
    guess_entities, gold_entities = get_doc_level_guess_gold_entities(
        guess_entities, gold_entities
    )
    all_scores = [
        get_micro_f1(guess_entities[k], gold_entities[k], mode) for k in guess_entities
    ]
    return (sum(all_scores) / len(all_scores)) if len(all_scores) else 0


def extract_pages(filename):
    docs = {}
    with open(filename) as f:
        for line in f:
            # CASE 1: beginning of the document
            if line.startswith("<doc id="):
                doc = ET.fromstring("{}{}".format(line, "</doc>")).attrib
                doc["paragraphs"] = []
                doc["anchors"] = []

            # CASE 2: end of the document
            elif line.startswith("</doc>"):
                assert doc["id"] not in docs, "{} ({}) already in dict as {}".format(
                    doc["id"], doc["title"], docs[doc["id"]]["title"]
                )
                docs[doc["id"]] = doc

            # CASE 3: in the document
            else:
                doc["paragraphs"].append("")
                try:
                    line = BeautifulSoup(line, "html.parser")
                except:
                    print("error line `{}`".format(line))
                    line = [line]

                for span in line:
                    if isinstance(span, bs4.element.Tag):
                        if span.get("href", None):
                            doc["anchors"].append(
                                {
                                    "text": span.get_text(),
                                    "href": span["href"],
                                    "paragraph_id": len(doc["paragraphs"]) - 1,
                                    "start": len(doc["paragraphs"][-1]),
                                    "end": len(doc["paragraphs"][-1])
                                    + len(span.get_text()),
                                }
                            )
                        doc["paragraphs"][-1] += span.get_text()
                    else:
                        doc["paragraphs"][-1] += str(span)

    return docs


def search_simple(anchor, lang, lang_title2wikidataID):
    if "http" in anchor:
        return True, []

    unquoted = unquote(anchor).split("#")[0].replace("_", " ")
    if unquoted == "":
        return True, []

    unquoted = unquoted[0].upper() + unquoted[1:]
    if (lang, unquoted) in lang_title2wikidataID:
        return True, lang_title2wikidataID[(lang, unquoted)]
    else:
        return False, unquoted


def search_wikipedia(title, lang, lang_title2wikidataID, lang_redirect2title):

    max_redirects = 10
    while (lang, title) in lang_redirect2title and max_redirects > 0:
        title = lang_redirect2title[(lang, title)]
        max_redirects -= 1

    if (lang, title) in lang_title2wikidataID:
        return True, lang_title2wikidataID[(lang, title)]
    else:
        return False, title


def search_wikidata(query, label_alias2wikidataID):
    return list(set(label_alias2wikidataID.get(query.lower(), [])))


def get_wikidata_ids(
    anchor, lang, lang_title2wikidataID, lang_redirect2title, label_or_alias2wikidataID,
):
    success, result = search_simple(anchor, lang, label_or_alias2wikidataID)
    if success:
        return result, "simple"
    else:
        success, result = search_wikipedia(
            result, lang, lang_title2wikidataID, lang_redirect2title
        )
        if success:
            return result, "wikipedia"
        else:
            return search_wikidata(result, label_or_alias2wikidataID), "wikidata"


def post_process_wikidata(outputs, text_to_id=False, marginalize=False):

    if text_to_id:
        outputs = [
            [{**hypo, "id": text_to_id(hypo["text"])} for hypo in hypos]
            for hypos in outputs
        ]

        if marginalize:
            for (i, hypos), hypos_tok in zip(enumerate(outputs), batched_hypos):
                outputs_dict = defaultdict(list)
                for hypo, hypo_tok in zip(hypos, hypos_tok):
                    outputs_dict[hypo["id"]].append(
                        {**hypo, "len": len(hypo_tok["tokens"])}
                    )

                outputs[i] = sorted(
                    [
                        {
                            "id": _id,
                            "texts": [hypo["text"] for hypo in hypos],
                            "scores": torch.stack([hypo["score"] for hypo in hypos]),
                            "score": torch.stack(
                                [
                                    hypo["score"]
                                    * hypo["len"]
                                    / (hypo["len"] ** marginalize_lenpen)
                                    for hypo in hypos
                                ]
                            ).logsumexp(-1),
                        }
                        for _id, hypos in outputs_dict.items()
                    ],
                    key=lambda x: x["score"],
                    reverse=True,
                )

    return outputs


tr2016_langs = ["ar", "de", "es", "fr", "he", "it", "ta", "th", "tl", "tr", "ur", "zh"]

news_langs = [
    "ar",
    "bg",
    "bs",
    "ca",
    "cs",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "fa",
    "fi",
    "fr",
    "he",
    "hu",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sd",
    "sq",
    "sr",
    "sv",
    "ta",
    "th",
    "tr",
    "uk",
    "zh",
]

mewsli_langs = ["ar", "de", "en", "es", "fa", "ja", "sr", "ta", "tr"]

mbart25_langs = [
    "ar",
    "cs",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "gu",
    "hi",
    "it",
    "ja",
    "kk",
    "ko",
    "lt",
    "lv",
    "my",
    "ne",
    "nl",
    "ro",
    "ru",
    "si",
    "tr",
    "vi",
    "zh",
]

mbart100_langs = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bm",
    "bn",
    "br",
    "bs",
    "ca",
    "cb",
    "ci",
    "cs",
    "cx",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gn",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kg",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "om",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "q2",
    "q3",
    "qa",
    "qd",
    "qf",
    "qh",
    "qi",
    "qj",
    "ql",
    "qm",
    "qp",
    "qq",
    "qu",
    "qw",
    "qx",
    "qy",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "ti",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yo",
    "zh",
    "zu",
]

el100_langs = [
    "af",
    "an",
    "ar",
    "ar",
    "ast",
    "az",
    "azb",
    "ba",
    "bar",
    "be",
    "bg",
    "bn",
    "bpy",
    "br",
    "bs",
    "ca",
    "ce",
    "ceb",
    "cs",
    "cv",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gl",
    "gu",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "io",
    "is",
    "it",
    "iw",
    "ja",
    "jv",
    "ka",
    "kk",
    "kn",
    "ko",
    "ky",
    "la",
    "lah",
    "lb",
    "lmo",
    "lt",
    "lv",
    "mg",
    "min",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "nds",
    "ne",
    "new",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "pms",
    "pt",
    "ro",
    "ru",
    "scn",
    "sco",
    "sk",
    "sl",
    "sq",
    "sr",
    "sr-Latn",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "vo",
    "yo",
    "zh",
    "zh-TW",
]

wiki_langs = [
    "aa",
    "ab",
    "ace",
    "ady",
    "af",
    "ak",
    "als",
    "am",
    "an",
    "ang",
    "ar",
    "arc",
    "arz",
    "as",
    "ast",
    "atj",
    "av",
    "ay",
    "az",
    "azb",
    "ba",
    "bar",
    "bcl",
    "be",
    "bg",
    "bh",
    "bi",
    "bjn",
    "bm",
    "bn",
    "bo",
    "bpy",
    "br",
    "bs",
    "bug",
    "bxr",
    "ca",
    "cdo",
    "ce",
    "ceb",
    "ch",
    "cho",
    "chr",
    "chy",
    "ckb",
    "co",
    "cr",
    "crh",
    "cs",
    "csb",
    "cu",
    "cv",
    "cy",
    "da",
    "de",
    "din",
    "diq",
    "dsb",
    "dty",
    "dv",
    "dz",
    "ee",
    "el",
    "eml",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "ext",
    "fa",
    "ff",
    "fi",
    "fj",
    "fo",
    "fr",
    "frp",
    "frr",
    "fur",
    "fy",
    "ga",
    "gag",
    "gan",
    "gd",
    "gl",
    "glk",
    "gn",
    "gom",
    "gor",
    "got",
    "gu",
    "gv",
    "ha",
    "hak",
    "haw",
    "he",
    "hi",
    "hif",
    "ho",
    "hr",
    "hsb",
    "ht",
    "hu",
    "hy",
    "hyw",
    "hz",
    "ia",
    "id",
    "ie",
    "ig",
    "ii",
    "ik",
    "ilo",
    "inh",
    "io",
    "is",
    "it",
    "iu",
    "ja",
    "jam",
    "jbo",
    "jv",
    "ka",
    "kaa",
    "kab",
    "kbd",
    "kbp",
    "kg",
    "ki",
    "kj",
    "kk",
    "kl",
    "km",
    "kn",
    "ko",
    "koi",
    "kr",
    "krc",
    "ks",
    "ksh",
    "ku",
    "kv",
    "kw",
    "ky",
    "la",
    "lad",
    "lb",
    "lbe",
    "lez",
    "lfn",
    "lg",
    "li",
    "lij",
    "lmo",
    "ln",
    "lo",
    "lrc",
    "lt",
    "ltg",
    "lv",
    "mai",
    "mdf",
    "mg",
    "mh",
    "mhr",
    "mi",
    "min",
    "mk",
    "ml",
    "mn",
    "mr",
    "mrj",
    "ms",
    "mt",
    "mus",
    "mwl",
    "my",
    "myv",
    "mzn",
    "na",
    "nah",
    "nap",
    "nds",
    "ne",
    "new",
    "ng",
    "nl",
    "nn",
    "no",
    "nov",
    "nqo",
    "nrm",
    "nso",
    "nv",
    "ny",
    "oc",
    "olo",
    "om",
    "or",
    "os",
    "pa",
    "pag",
    "pam",
    "pap",
    "pcd",
    "pdc",
    "pfl",
    "pi",
    "pih",
    "pl",
    "pms",
    "pnb",
    "pnt",
    "ps",
    "pt",
    "qu",
    "rm",
    "rmy",
    "rn",
    "ro",
    "ru",
    "rue",
    "rw",
    "sa",
    "sah",
    "sat",
    "sc",
    "scn",
    "sco",
    "sd",
    "se",
    "sg",
    "sh",
    "shn",
    "si",
    "simple",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "srn",
    "ss",
    "st",
    "stq",
    "su",
    "sv",
    "sw",
    "szl",
    "ta",
    "tcy",
    "te",
    "tet",
    "tg",
    "th",
    "ti",
    "tk",
    "tl",
    "tn",
    "to",
    "tpi",
    "tr",
    "ts",
    "tt",
    "tum",
    "tw",
    "ty",
    "tyv",
    "udm",
    "ug",
    "uk",
    "ur",
    "uz",
    "ve",
    "vec",
    "vep",
    "vi",
    "vls",
    "vo",
    "wa",
    "war",
    "wo",
    "wuu",
    "xal",
    "xh",
    "xmf",
    "yi",
    "yo",
    "za",
    "zea",
    "zh",
]

our105_langs = sorted(set(mbart100_langs).intersection(set(wiki_langs)))

our105_langs_to_name = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "as": "Assamese",
    "az": "Azerbaijani",
    "bm": "Bambara",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "br": "Breton",
    "bg": "Bulgarian",
    "my": "Burmese",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ff": "Fulah",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gn": "Guarani",
    "gu": "Gujarati",
    "ht": "Haitian",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "ga": "Irish",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "ky": "Kyrgyz",
    "kg": "Kongo",
    "ko": "Korean",
    "ku": "Kurdish",
    "la": "Latin",
    "lg": "Ganda",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "no": "Norwegian",
    "om": "Oromo",
    "or": "Oriya",
    "pa": "Panjabi",
    "fa": "Persian",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "sr": "Serbian",
    "gd": "Gaelic,",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "ss": "Swati",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "ti": "Tigrinya",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "wo": "Wolof",
    "fy": "Frysk",
    "xh": "Xhosa",
    "yo": "Yoruba",
}
