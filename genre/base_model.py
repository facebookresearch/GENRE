import torch
import copy
from typing import List
from fairseq.models.bart import BARTHubInterface


class GENRE(BARTHubInterface):
    def sample(
        self, sentences: List[str], beam: int = 10, verbose: bool = False, **kwargs
    ) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [
            [
                {"text": self.decode(x["tokens"]), "logprob": x["score"].item()}
                for x in h
            ]
            for h in hypos
        ]

    def generate(
        self,
        tokens: List[torch.LongTensor],
        beam: int = 10,
        verbose: bool = False,
        **kwargs,
    ) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator, [self.model], sample, prefix_tokens=None
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info("S\t{}".format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x for x in translations]
        hypos = [v for _, v in sorted(zip(sample["id"].tolist(), hypos))]
        return hypos

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

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
        return GENRE(x["args"], x["task"], x["models"][0])
