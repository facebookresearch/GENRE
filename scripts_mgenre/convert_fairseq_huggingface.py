import os

import torch
from transformers import (
    FlaxMBartForConditionalGeneration,
    MBartConfig,
    MBartForConditionalGeneration,
    TFMBartForConditionalGeneration,
    XLMRobertaTokenizer,
)

from genre.fairseq_model import GENRE, mGENRE


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = torch.nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Load

fairseq_path = "../models/fairseq_multilingual_entity_disambiguation"
hf_path = "../models/hf_multilingual_entity_disambiguation"

fairseq_model = mGENRE.from_pretrained(fairseq_path).eval()
config = MBartConfig(vocab_size=256001, scale_embedding=True)
hf_model = MBartForConditionalGeneration(config).eval()
hf_tokenizer = XLMRobertaTokenizer(os.path.join(fairseq_path, "spm_256000.model"))
hf_tokenizer.save_pretrained(hf_path)

# Convert pytorch

state_dict = fairseq_model.model.state_dict()
remove_ignore_keys_(state_dict)
state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
hf_model.model.load_state_dict(state_dict)
hf_model.lm_head = make_linear_from_emb(hf_model.model.shared)
hf_model.save_pretrained(hf_path)

# Convert flax

hf_model = FlaxMBartForConditionalGeneration.from_pretrained(hf_path, from_pt=True)
hf_model.save_pretrained(hf_path)

# Convert tensorflow

hf_model = TFMBartForConditionalGeneration.from_pretrained(hf_path, from_pt=True)
hf_model.save_pretrained(hf_path)
