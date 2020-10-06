
import jsonlines
import argparse
import json
import os
import string
import kilt.kilt_utils as utils
from fairseq.models.bart import BARTModel
from tqdm import tqdm
from fairseq.data.data_utils import collate_tokens
from genre.trie import Trie
import os
import re
import shutil
import json
import jsonlines
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
import kilt.kilt_utils as utils
from kilt.eval_retrieval import compute
from collections import defaultdict
import pickle
from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import collate_tokens
import torch
from pprint import pprint
import math
import random
import string
from pprint import pprint
import html
import csv
import unicodedata
from IPython.display import display, Markdown, Latex
import re
import regex
from genre.my_fetcher import FetchCandidateEntities, FetchFilteredCoreferencedCandEntities, load_wiki_name_id_map, load_redirections, load_disambiguations, load_disambiguations2
from types import SimpleNamespace
from nltk.corpus import stopwords
from genre.base_model import GENRE


class GeNeRe(object):
    def __init__(
        self,
        model_path,
        checkpoint_file="checkpoint_best.pt",
        device="cuda:0",
    ):

                
        with open("/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/data/el/global_el_mention_trie2.pkl", "rb") as f:
            self.mention_trie = pickle.load(f)
        
        print("loading model", os.path.join(model_path, checkpoint_file))
        self.bart = (
            GENRE.from_pretrained(model_path, checkpoint_file=checkpoint_file)
            .eval()
            .to(device)
        )

        self.codes = {
            k: [
                self.bart.encode(" {}".format(k))[1].item(),
            ]
            for k in ("{", "}", "[", "]")
        }
        self.codes["EOS"] = [2]

        args = SimpleNamespace(lowercase_spans=False, lowercase_p_e_m=False, cand_ent_num=100,
                               persons_coreference=True, persons_coreference_merge=True)

        self.fetcher = FetchFilteredCoreferencedCandEntities(args)
        _, self.wiki_id_name_map = load_wiki_name_id_map(lowercase=False)

        entities_universe = []

        for fname in (
            "data/el/entities/extension_entities/entities_universe.txt",
            "data/el/entities/entities_universe.txt",
            "data/el/entities/extension_entities/extension_entities.txt",
        ):
            with open(fname) as f:
                entities_universe += f.readlines()

        entities_universe = [e.split("\t")[1].strip() for e in entities_universe]
        entities_universe = list(set(entities_universe))

        self.entities_universe = {
            e.replace("_", " "): e
            for e in entities_universe if e.replace("_", " ")
        }

        self.redirections = load_redirections(lowercase=False)

        with open("data/el/aida_means.pkl", "rb") as f:
            self.aida_means = pickle.load(f)
            
        self.stopwords = set(stopwords.words('english'))


    def get_candidates(self, mention):
        if mention in string.punctuation or mention.lower() in self.stopwords or mention.strip() == "":
            return ["NIL"]

        output = list(self.fetcher.fetchCandidateEntities.process(mention))[0]
        output = [self.wiki_id_name_map[e] for e in (output if output else [])]

        if mention in self.aida_means:
            output += [
                html.unescape(unicodedata.normalize("NFC", 
                    e.encode('ascii').decode('unicode-escape'))).replace("_", " ")
                for e in self.aida_means[mention]
            ]

        output = [e for e in set(output) if e in self.entities_universe and "given name" not in e]

        if not output:
            return ["NIL"]

        return output

    def get_predictions(self, inputs):

#         def apply_mask_fn(x, prev_output_tokens, original_batch_idxs):
#             beam_size = x.shape[0] // original_batch_idxs.shape[0]
#             original_batch_idxs = (
#                 original_batch_idxs.unsqueeze(-1)
#                 .repeat((1, beam_size))
#                 .flatten()
#                 .tolist()
#             )

#             mask = torch.full_like(x, -math.inf)
#             for sent_i, (sent, batch_i) in enumerate(
#                 zip(prev_output_tokens, original_batch_idxs)
#             ):
#                 sent = sent.tolist()
#                 status = get_status(sent)
#                 mask[
#                     sent_i, :, get_allowed_tokens(
#                         sent, 
#                         sent_origs[batch_i], 
#                         status
#                     )
#                 ] = 0
#             return mask

        def get_status(sent):
            c = [
                f
                for e in (
                    self.codes["{"],
                    self.codes["}"],
                    self.codes["["],
                    self.codes["]"],
                )
                for f in e
            ]
            status = (
                sum(
                    e
                    in c
                    for e in sent
                )
                % 4
            )

            if status == 0:
                return "o"
            elif status == 1:
                return "m"
            else:
                return "e"

#         def get_allowed_tokens(sent, sent_orig, status):
        def get_allowed_tokens(batch_i, sent):
            sent = sent.tolist()
            status = get_status(sent)
            sent_orig = sent_origs[batch_i]

#             print(self.bart.decode(torch.tensor([e for e in sent if e != 1])))
            if status == "o":
                trie_out = get_trie_outside(sent, sent_orig)
            elif status == "m":
                trie_out = get_trie_mention(sent, sent_orig)
            elif status == "e":
                trie_out = get_trie_entity(sent, sent_orig)
                if trie_out == [2]:
                    trie_out = get_trie_outside(sent, sent_orig)
            else:
                raise RuntimeError
                
#             print(status, trie_out)
#             if trie_out == None:
#                 raise RuntimeError
            
            return trie_out

        def get_pointer_end(sent, sent_orig):
            i = 0
            j = 0
            while i < len(sent):
                if sent[i] == sent_orig[j]:
                    i += 1
                    j += 1
                elif sent[i] in self.codes["{"] or sent[i] in self.codes["}"]:
                    i += 1
                elif sent[i] in self.codes["["]:
                    i += 1
                    while sent[i] not in self.codes["]"]:
                        i += 1
                    i += 1
                else:
                    return None

            return j if j != len(sent_orig) else None

        def get_pointer_mention(sent):
            pointer_end = -1
            for i, e in enumerate(sent):
                if e in self.codes["{"]:
                    pointer_start = i
                elif e in self.codes["}"]:
                    pointer_end = i

            return pointer_start, pointer_end

        def get_trie_outside(sent, sent_orig):
            pointer_end = get_pointer_end(sent, sent_orig)

            if pointer_end:
                if sent_orig[pointer_end] not in self.codes["EOS"] and sent_orig[pointer_end] in self.mention_trie.get([]):
                    return [sent_orig[pointer_end]] + self.codes["{"]
                else:
                    return [sent_orig[pointer_end]]
            else:
                return []

        def get_trie_mention(sent, sent_orig):

            pointer_start, _ = get_pointer_mention(sent)
            if pointer_start + 1 < len(sent):
                ment_next = self.mention_trie.get(sent[pointer_start+1:])
            else:
                ment_next = self.mention_trie.get([])

            pointer_end = get_pointer_end(sent, sent_orig)
            
            if pointer_end:
                if sent_orig[pointer_end] not in self.codes["EOS"]:
                    if sent_orig[pointer_end] in ment_next:
                        if 2 in ment_next:
                            return [sent_orig[pointer_end]] + self.codes["}"]
                        else:
                            return [sent_orig[pointer_end]]
                    elif 2 in ment_next:
                        return self.codes["}"]
                    else:
                        return []
                else:
                    return self.codes["}"]
            else:
                return []

        def get_trie_entity(sent, sent_orig):
            pointer_start, pointer_end = get_pointer_mention(sent)

            if pointer_start + 1 != pointer_end:
#                 print(pointer_start+1,pointer_end, sent)
                mention = self.bart.decode(torch.tensor(sent[pointer_start+1:pointer_end])).strip()
                candidates = self.get_candidates(mention)

                if candidates:
                    return Trie([
                        self.bart.encode(" }} [ {} ]".format(e)).tolist()[1:] 
                        for e in candidates
                    ]).get(sent[pointer_end:])
            
            return []

        def decode(tokens):
            tokens = self.bart.decode(tokens)
            tokens = re.sub(r"{.*?", "{ ", tokens)
            tokens = re.sub(r"}.*?", "} ", tokens)
            tokens = re.sub(r"\].*?", "] ", tokens)
            tokens = re.sub(r"\[.*?", "[ ", tokens)
            tokens = re.sub(r"\s{2,}", " ", tokens)
            return tokens

        inputs = [(
            " {} ".format(e)
            .replace(u'\xa0',' ')
            .replace("{", "(")
            .replace("}", ")")
            .replace("[", "(")
            .replace("]", ")")

        ) for e in inputs]
        
#         new_inputs = []
#         for s in inputs:
#             sent = Sentence(s)
#             self.tagger.predict(sent)
#             last_end = 0
#             new_s = ""
#             for entity in sent.get_spans('ner'):
#                 new_s += s[last_end:entity.start_pos] + " { " + s[entity.start_pos:entity.end_pos] + " } "
#                 last_end = entity.end_pos
#             new_s += s[last_end:]
#             new_s = new_s.replace(", }", " } ,").replace(". }", " } .").replace("; }", " } ;").replace(": }", " } :")
#             new_inputs.append(new_s)

#         inputs = new_inputs
        
        sent_origs = [[2] + self.bart.encode(e).tolist()[1:] for e in inputs]

        outputs = self.bart.sample(
            inputs,
            beam=6,
            max_len_b=1024,
            prefix_allowed_tokens_fn=get_allowed_tokens,
        )

        outputs = [
            [[hyp["text"], hyp["logprob"]] for hyp in sent]
            for sent in outputs
        ]

        return outputs

    def get_entities(self, input_, output_):

#         if hasattr(self, "dataset_coref"):
#             coref = self.dataset_coref[input_]
        
        input_ = input_.replace(u'\xa0',' ') + "  -"
        output_ = output_.replace("{ ", "{").replace(" } [ ", "}[").replace(" ]", "]") + "  -"

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
#                         entities[-1][2] = input_[entities[-1][0]:entities[-1][0]+entities[-1][1]].replace(" ", "_")
                        del entities[-1]
                    elif entities[-1][2] in self.redirections:
                        entities[-1][2] = self.redirections[entities[-1][2]]

                    status = "o"
                    j += 1
                else:
                    raise RuntimeError

#         if hasattr(self, "dataset_coref"):
#             return entities, coref
                    
        return entities

    def get_prediction(self, input_, max_length=384):

        if max_length < 32:
            return input_

        i = 0
        inputs = []
        for le in [len(e) for e in utils.chunk_it(
            input_.split(" "), len(input_.split(" ")) // max_length
        )] if len(input_.split(" ")) > max_length else [len(input_.split(" "))]:
            inputs.append(" ".join(input_.split(" ")[i:i+le]))
            i += le

        outputs = self.get_predictions(inputs)

        if any(e == [] for e in outputs):
            print("failed with max_length={}".format(max_length))
            return self.get_prediction(input_, max_length=max_length // 2)

        combined_outputs = outputs.pop(0)
        while len(outputs) > 0:
            combined_outputs = [
                [tp + " " + t, sp + s]
                for t, s in outputs.pop(0)
                for tp, sp in combined_outputs
            ]

        outputs = combined_outputs
        
        print("outputs:")
        pprint(outputs)
        
        output_ = re.sub(r"\s{2,}", " ", outputs[0][0])
        output_ = re.sub(r"\. \. \} \[ (.*?) \]", r". } [ \1 ] .", output_)
        output_ = re.sub(r"\, \} \[ (.*?) \]", r" } [ \1 ] ,", output_)
        output_ = re.sub(r"\; \} \[ (.*?) \]", r" } [ \1 ] ;", output_)
        
        print("output:", output_)
        return self.get_entities(input_, output_)
    
    def get_markdown(self, sent, spans):
        text = ""
        last_end = 0
        for begin, length, href in spans:
            text += sent[last_end:begin]
            text += "[{}](https://en.wikipedia.org/wiki/{})".format(sent[begin:begin + length], href)
            last_end = begin + length

        text += sent[last_end:]

        return Markdown(text)
