import os
import torch
from genre.fairseq_model import GENRE, mGENRE
from transformers import (
    BartConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    TFBartForConditionalGeneration,
    MBartConfig,
    XLMRobertaTokenizer,
    MBartForConditionalGeneration,
    TFMBartForConditionalGeneration,
    load_pytorch_model_in_tf2_model,
)


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


# Load GENRE

# fairseq_path = "../models/fairseq_entity_disambiguation_aidayago"
# hf_path = "../models/hf_entity_disambiguation_aidayago"

# fairseq_model = GENRE.from_pretrained(fairseq_path).eval()
# config = BartConfig(vocab_size=50264)
# hf_model = BartForConditionalGeneration(config).eval()
# hf_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

# Load mGENRE

fairseq_path = "../models/fairseq_multilingual_entity_disambiguation"
hf_path = "../models/hf_multilingual_entity_disambiguation"

fairseq_model = mGENRE.from_pretrained(fairseq_path).eval()
config = MBartConfig(vocab_size=256001, scale_embedding=True)
hf_model = MBartForConditionalGeneration(config).eval()
hf_tokenizer = XLMRobertaTokenizer(os.path.join(fairseq_path, "spm_256000.model"))

# Convert model

state_dict = fairseq_model.model.state_dict()
remove_ignore_keys_(state_dict)
state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
hf_model.model.load_state_dict(state_dict)
hf_model.lm_head = make_linear_from_emb(hf_model.model.shared)

# Save

hf_tokenizer.save_pretrained(hf_path)
hf_model.save_pretrained(hf_path)

# Convert TF GENRE

# hf_model = load_pytorch_model_in_tf2_model(
#     TFBartForConditionalGeneration(
#         config
#     ),
#     hf_model,
# )

# Convert TF mGENRE

hf_model = load_pytorch_model_in_tf2_model(
    TFMBartForConditionalGeneration(config),
    hf_model,
)

# Save

hf_model.save_pretrained(hf_path)
