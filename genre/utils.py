# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import html
import re
from collections import defaultdict

from genre.entity_linking import (
    get_end_to_end_prefix_allowed_tokens_fn_fairseq,
    get_end_to_end_prefix_allowed_tokens_fn_hf)


def batch_it(seq, num=1):
    out = []
    for item in seq:
        if len(out) == num:
            yield out
            out = []
        out.append(item)

    if len(out):
        yield out


def create_input(doc, max_length):
    if "meta" in doc and all(
        e in doc["meta"] for e in ("left_context", "mention", "right_context")
    ):
        if len(doc["input"].split(" ")) <= max_length:
            input_ = (
                doc["meta"]["left_context"]
                + " [START_ENT] "
                + doc["meta"]["mention"]
                + " [END_ENT] "
                + doc["meta"]["right_context"]
            )
        elif len(doc["meta"]["left_context"].split(" ")) <= max_length // 2:
            input_ = (
                doc["meta"]["left_context"]
                + " [START_ENT] "
                + doc["meta"]["mention"]
                + " [END_ENT] "
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
                + " [START_ENT] "
                + doc["meta"]["mention"]
                + " [END_ENT] "
                + doc["meta"]["right_context"]
            )
        else:
            input_ = (
                " ".join(doc["meta"]["left_context"].split(" ")[-max_length // 2 :])
                + " [START_ENT] "
                + doc["meta"]["mention"]
                + " [END_ENT] "
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


def get_entity_spans_fairseq(
    model,
    input_sentences,
    mention_trie=None,
    candidates_trie=None,
    mention_to_candidates_dict=None,
    redirections=None,
):

    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_fairseq(
        model,
        get_entity_spans_pre_processing(input_sentences),
        mention_trie=mention_trie,
        candidates_trie=candidates_trie,
        mention_to_candidates_dict=mention_to_candidates_dict,
    )

    output_sentences = model.sample(
        get_entity_spans_pre_processing(input_sentences),
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        max_len_a=1024,
        max_len_b=1024,
    )

    output_sentences = get_entity_spans_post_processing(
        [e[0]["text"] for e in output_sentences]
    )

    return get_entity_spans_finalize(
        input_sentences, output_sentences, redirections=redirections
    )


def get_entity_spans_hf(
    model,
    tokenizer,
    input_sentences,
    mention_trie=None,
    candidates_trie=None,
    mention_to_candidates_dict=None,
    redirections=None,
):

    input_args = {
        k: v.to(model.device)
        for k, v in tokenizer.batch_encode_plus(
            get_entity_spans_pre_processing(input_sentences), return_tensors="pt"
        ).items()
    }

    prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
        tokenizer,
        get_entity_spans_pre_processing(input_sentences),
        mention_trie=mention_trie,
        candidates_trie=candidates_trie,
        mention_to_candidates_dict=mention_to_candidates_dict,
    )

    output_sentences = tokenizer.batch_decode(
        model.generate(
            **input_args,
            min_length=0,
            num_beams=5,
            num_return_sequences=1,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        ),
        skip_special_tokens=True,
    )

    output_sentences = get_entity_spans_post_processing(output_sentences)

    return get_entity_spans_finalize(
        input_sentences, output_sentences, redirections=redirections
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
