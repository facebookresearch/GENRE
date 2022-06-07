# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch

from genre.trie import DummyTrieEntity, DummyTrieMention, Trie


def get_end_to_end_prefix_allowed_tokens_fn_hf(
    model,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x)),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        len(model.tokenizer) - 1,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        mention_to_candidates_dict,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id

    if mention_trie is None:
        mention_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i not in (bos_token_id, pad_token_id,)
            ]
        )

    if candidates_trie is None and mention_to_candidates_dict is None:
        candidates_trie = DummyTrieEntity(
            [
                i
                for i in range(vocabulary_length)
                if i not in (bos_token_id, pad_token_id,)
            ],
            codes,
        )

    sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]

    def prefix_allowed_tokens_fn(batch_id, sent):

        sent = sent.tolist()
        status = get_status(sent)
        sent_orig = sent_origs[batch_id]

        if status == "o":
            trie_out = get_trie_outside(sent, sent_orig)
        elif status == "m":
            trie_out = get_trie_mention(sent, sent_orig)
        elif status == "e":
            trie_out = get_trie_entity(sent, sent_orig)
            if trie_out == codes["EOS"]:
                trie_out = get_trie_outside(sent, sent_orig)
        else:
            raise RuntimeError

        return trie_out

    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            )
        ]
        status = sum(e in c for e in sent) % 4

        if status == 0:
            return "o"
        elif status == 1:
            return "m"
        else:
            return "e"

    def get_trie_outside(sent, sent_orig):
        pointer_end = get_pointer_end(sent, sent_orig)

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"] and sent_orig[
                pointer_end
            ] in mention_trie.get([]):
                return [sent_orig[pointer_end], codes["start_mention_token"]]
            else:
                return [sent_orig[pointer_end]]
        else:
            return []

    def get_pointer_end(sent, sent_orig):
        i = 0
        j = 0
        while i < len(sent):
            if sent[i] == sent_orig[j]:
                i += 1
                j += 1
            elif (
                sent[i] == codes["start_mention_token"]
                or sent[i] == codes["end_mention_token"]
            ):
                i += 1
            elif sent[i] == codes["start_entity_token"]:
                i += 1
                while sent[i] != codes["end_entity_token"]:
                    i += 1
                i += 1
            else:
                return None

        return j if j != len(sent_orig) else None

    def get_trie_mention(sent, sent_orig):

        pointer_start, _ = get_pointer_mention(sent)
        if pointer_start + 1 < len(sent):
            ment_next = mention_trie.get(sent[pointer_start + 1 :])
        else:
            ment_next = mention_trie.get([])

        pointer_end = get_pointer_end(sent, sent_orig)

        if pointer_end:
            if sent_orig[pointer_end] != codes["EOS"]:
                if sent_orig[pointer_end] in ment_next:
                    if codes["EOS"] in ment_next:
                        return [sent_orig[pointer_end], codes["end_mention_token"]]
                    else:
                        return [sent_orig[pointer_end]]
                elif codes["EOS"] in ment_next:
                    return [codes["end_mention_token"]]
                else:
                    return []
            else:
                return [codes["end_mention_token"]]
        else:
            return []

    def get_pointer_mention(sent):
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    def get_trie_entity(sent, sent_orig):
        pointer_start, pointer_end = get_pointer_mention(sent)

        if pointer_start + 1 != pointer_end:
            mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()

            if candidates_trie is not None:
                candidates_trie_tmp = candidates_trie
            elif mention_to_candidates_dict is not None:
                candidates_trie_tmp = Trie(
                    [
                        encode_fn(
                            " {} {} {} {}".format(
                                end_mention_token,
                                start_entity_token,
                                e,
                                end_entity_token,
                            )
                        )[1:]
                        for e in mention_to_candidates_dict.get(mention, ["NIL"])
                    ]
                )
            else:
                raise RuntimeError()

            return candidates_trie_tmp.get(sent[pointer_end:])

        return []

    return prefix_allowed_tokens_fn
