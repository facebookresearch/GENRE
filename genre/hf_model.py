# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List

import torch
from genre.utils import chunk_it
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


class GENREHubInterface(BartForConditionalGeneration):
    def sample(
        self, sentences: List[str], num_beams: int = 5, num_return_sequences=5, **kwargs
    ) -> List[str]:

        input_args = {
            k: v.to(self.device)
            for k, v in self.tokenizer.batch_encode_plus(
                sentences, padding=True, return_tensors="pt"
            ).items()
        }

        outputs = self.generate(
            **input_args,
            min_length=0,
            max_length=1024,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )

        return chunk_it(
            [
                {
                    "text": text,
                    "logprob": score,
                }
                for text, score in zip(
                    self.tokenizer.batch_decode(
                        outputs.sequences, skip_special_tokens=True
                    ),
                    outputs.sequences_scores,
                )
            ],
            num_return_sequences,
        )

    def encode(self, sentence):
        return self.tokenizer.encode(sentence, return_tensors="pt")[0]


class GENRE(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        model = GENREHubInterface.from_pretrained(model_name_or_path)
        model.tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        return model
