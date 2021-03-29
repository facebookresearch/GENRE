# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
from collections import defaultdict
from typing import Dict, List

import torch
from fairseq import search, utils
from fairseq.models.bart import BARTHubInterface, BARTModel
from omegaconf import open_dict

logger = logging.getLogger(__name__)


class GENREHubInterface(BARTHubInterface):
    def sample(
        self,
        sentences: List[str],
        beam: int = 5,
        verbose: bool = False,
        text_to_id=None,
        marginalize=False,
        marginalize_lenpen=0.5,
        **kwargs,
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        outputs = [
            [
                {"text": self.decode(hypo["tokens"]), "score": hypo["score"]}
                for hypo in hypos
            ]
            for hypos in batched_hypos
        ]
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
                                "scores": torch.stack(
                                    [hypo["score"] for hypo in hypos]
                                ),
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

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        *args,
        inference_step_args=None,
        skip_invalid_size_inputs=False,
        **kwargs,
    ) -> List[List[Dict[str, torch.Tensor]]]:
        inference_step_args = inference_step_args or {}
        if "prefix_tokens" in inference_step_args:
            raise NotImplementedError("prefix generation not implemented for BART")
        res = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            src_tokens = batch["net_input"]["src_tokens"]
            results = super(BARTHubInterface, self).generate(
                src_tokens,
                *args,
                inference_step_args=inference_step_args,
                skip_invalid_size_inputs=skip_invalid_size_inputs,
                **kwargs,
            )
            for id, hypos in zip(batch["id"].tolist(), results):
                res.append((id, hypos))

        # sort output to match input order
        res = [hypos for _, hypos in sorted(res, key=lambda x: x[0])]
        return res


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
        return GENREHubInterface(x["args"], x["task"], x["models"][0])
