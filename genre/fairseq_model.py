# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import List, Dict

import torch
from fairseq.models.bart import BARTHubInterface, BARTModel

from genre.utils import post_process_wikidata

logger = logging.getLogger(__name__)


class _GENREHubInterface:
    def sample(
        self,
        sentences: List[str],
        beam: int = 5,
        verbose: bool = False,
        text_to_id=None,
        marginalize=False,
        marginalize_lenpen=0.5,
        max_len_a=1024,
        max_len_b=1024,
        **kwargs,
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]

        batched_hypos = self.generate(
            tokenized_sentences,
            beam,
            verbose,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            **kwargs,
        )

        outputs = [
            [
                {"text": self.decode(hypo["tokens"]), "score": hypo["score"]}
                for hypo in hypos
            ]
            for hypos in batched_hypos
        ]

        outputs = post_process_wikidata(
            outputs, text_to_id=text_to_id, marginalize=marginalize
        )

        return outputs

    def generate(self, *args, **kwargs) -> List[List[Dict[str, torch.Tensor]]]:
        return super(BARTHubInterface, self).generate(*args, **kwargs)

    def encode(self, sentence) -> torch.LongTensor:
        tokens = super(BARTHubInterface, self).encode(sentence)
        tokens[
            tokens >= len(self.task.target_dictionary)
        ] = self.task.target_dictionary.unk_index
        if tokens[0] != self.task.target_dictionary.bos_index:
            return torch.cat(
                (torch.tensor([self.task.target_dictionary.bos_index]), tokens)
            )
        else:
            return tokens

class GENREHubInterface(_GENREHubInterface, BARTHubInterface):
    pass
    
class mGENREHubInterface(_GENREHubInterface, BARTHubInterface):
    pass

class GENRE(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GENREHubInterface(x["args"], x["task"], x["models"][0])

class mGENRE(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        sentencepiece_model="spm_256000.model",
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        layernorm_embedding=True,
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sentencepiece_model=os.path.join(model_name_or_path, sentencepiece_model),
            **kwargs,
        )
        return mGENREHubInterface(x["args"], x["task"], x["models"][0])
