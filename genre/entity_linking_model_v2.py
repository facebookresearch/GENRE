import pickle

import torch

from genre.base_model import GENRE
from genre.trie import Trie


class GENREForEndToEndEntityLinking(object):
    def __init__(self, genre_model, mention_trie_file, candidates_dict_file):

        self.bart = genre_model

        with open(mention_trie_file, "rb") as f:
            self.mention_trie = pickle.load(f)

        with open(candidates_dict_file, "rb") as f:
            self.candidates_dict = pickle.load(f)

        self.codes = {
            k: self.bart.encode(" {}".format(k))[1].item() for k in ("{", "}", "[", "]")
        }
        self.codes["EOS"] = 2

    @classmethod
    def from_pretrained(self, *args, mention_trie_file, candidates_dict_file, **kwargs):
        return GENREForEndToEndEntityLinking(
            GENRE.from_pretrained(*args, **kwargs),
            mention_trie_file,
            candidates_dict_file,
        )

    def sample(self, inputs, beam=6, max_len_b=1024):
        def prefix_allowed_tokens_fn(batch_i, sent):
            sent = sent.tolist()
            status = get_status(sent)
            sent_orig = sent_origs[batch_i]

            if status == "o":
                trie_out = get_trie_outside(sent, sent_orig)
            elif status == "m":
                trie_out = get_trie_mention(sent, sent_orig)
            elif status == "e":
                trie_out = get_trie_entity(sent, sent_orig)
                if trie_out == 2:
                    trie_out = get_trie_outside(sent, sent_orig)
            else:
                raise RuntimeError

            return trie_out

        def get_status(sent):
            c = [self.codes[e] for e in "{}[]"]
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
                if sent_orig[pointer_end] != self.codes["EOS"] and sent_orig[
                    pointer_end
                ] in self.mention_trie.get([]):
                    return [sent_orig[pointer_end], self.codes["{"]]
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
                elif sent[i] == self.codes["{"] or sent[i] == self.codes["}"]:
                    i += 1
                elif sent[i] == self.codes["["]:
                    i += 1
                    while sent[i] != self.codes["]"]:
                        i += 1
                    i += 1
                else:
                    return None

            return j if j != len(sent_orig) else None

        def get_trie_mention(sent, sent_orig):

            pointer_start, _ = get_pointer_mention(sent)
            if pointer_start + 1 < len(sent):
                ment_next = self.mention_trie.get(sent[pointer_start + 1 :])
            else:
                ment_next = self.mention_trie.get([])

            pointer_end = get_pointer_end(sent, sent_orig)

            if pointer_end:
                if sent_orig[pointer_end] != self.codes["EOS"]:
                    if sent_orig[pointer_end] in ment_next:
                        if self.codes["EOS"] in ment_next:
                            return [sent_orig[pointer_end], self.codes["}"]]
                        else:
                            return [sent_orig[pointer_end]]
                    elif self.codes["EOS"] in ment_next:
                        return [self.codes["}"]]
                    else:
                        return []
                else:
                    return [self.codes["}"]]
            else:
                return []

        def get_pointer_mention(sent):
            pointer_end = -1
            for i, e in enumerate(sent):
                if e == self.codes["{"]:
                    pointer_start = i
                elif e == self.codes["}"]:
                    pointer_end = i

            return pointer_start, pointer_end

        def get_trie_entity(sent, sent_orig):
            pointer_start, pointer_end = get_pointer_mention(sent)

            if pointer_start + 1 != pointer_end:
                mention = self.bart.decode(
                    torch.tensor(sent[pointer_start + 1 : pointer_end])
                ).strip()
                candidates = get_candidates(mention)

                return Trie(
                    [
                        self.bart.encode(" }} [ {} ]".format(e)).tolist()[1:]
                        for e in candidates
                    ]
                ).get(sent[pointer_end:])

            return []

        def get_candidates(mention):
            return self.candidates_dict.get(mention, [0, ["NIL"]])[1]

        #             # Kolitas used:
        #             title = mention.title()
        #             title_freq = self.candidates_dict.get(title, [0])[0]
        #             mention_freq = self.candidates_dict.get(mention, [0])[0]

        #             if title_freq == 0 and mention_freq == 0 or mention_freq > title_freq:
        #                 return self.candidates_dict.get(mention, self.candidates_dict.get(title, [0, ["NIL"]]))[1]
        #             else:
        #                 return self.candidates_dict.get(title, self.candidates_dict.get(mention, [0, ["NIL"]]))[1]

        sent_origs = [[2] + self.bart.encode(e).tolist()[1:] for e in inputs]

        outputs = self.bart.sample(
            inputs,
            beam=beam,
            max_len_b=max_len_b,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        return outputs
