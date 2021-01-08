# GENRE (Generarive ENtity REtrieval)

The GENRE system as presented in [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) implemented in pytorch.

```bibtex
@article{de2020autoregressive,
  title={Autoregressive Entity Retrieval},
  author={De Cao, Nicola and Izacard, Gautier and Riedel, Sebastian and Petroni, Fabio},
  journal={arXiv preprint arXiv:2010.00904},
  year={2020}
}
```

In a nutshell, GENRE uses a sequence-to-sequence approach to entity retrieval (e.g., linking), based on fine-tuned [BART](https://arxiv.org/abs/1910.13461) architecture. GENRE performs retrieval generating the unique entity name conditioned on the input text using constrained beam search to only generate valid identifiers. Here an example of generation for Wikipedia page retrieval for open-domain question answering:

![](GENRE-animation-QA.gif)

For end-to-end entity linking GENRE re-generates the input text annoted with a markup:

![](GENRE-animation-EL.gif)

GENRE achieves state-of-the-art results on multiple datasets.

## Main dependencies
* python>=3.7
* pytorch>=1.6
* fariseq>=0.10 (for training -- optional for inference)
* transformers>=4.0 (optional for inference)

## Usage

See examples on how to use GENRE for both pytorch fairseq and huggingface transformers:
* For [pytorch/fairseq](https://github.com/facebookresearch/GENRE/blob/main/examples/fairseq.md)
* For [huggingface/transformers](https://github.com/facebookresearch/GENRE/blob/main/examples/transformers.md)

Generally, after importing and loading the model, you would generate predictions (in this example for Entity Disambiguation) with a simple call like:

```python
model.sample(
    sentences=[
        "[START_ENT] Armstrong [END_ENT] was the first man on the Moon."
    ]
)
```




    [[{'text': 'Neil Armstrong', 'logprob': tensor(-0.1443)},
      {'text': 'William Armstrong', 'logprob': tensor(-1.4650)},
      {'text': 'Scott Armstrong', 'logprob': tensor(-1.7311)},
      {'text': 'Arthur Armstrong', 'logprob': tensor(-1.7356)},
      {'text': 'Rob Armstrong', 'logprob': tensor(-1.7426)}]]





**NOTE: we used fairseq for all experiments in the paper. The huggingface/transformers models are obtained with a conversion script similar to [this](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py). Therefore results might differ.**

## Models

Use the link above to download models in `.tar.gz` format and then `tar -zxvf <FILENAME>` do uncompress.  As an alternative use [this](https://github.com/facebookresearch/GENRE/blob/main/scripts/download_all_models.sh) script to dowload all of them.

### Entity Disambiguation
| Training Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| BLINK | [fairseq_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz)|[hf_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz)|
| BLINK + AidaYago2 | [fairseq_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz)|[hf_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)|

### End-to-End Entity Linking
| Training Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| WIKIPEDIA | [fairseq_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz)|[hf_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz)|
| WIKIPEDIA + AidaYago2 | [fairseq_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz)|[hf_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz)|

### Document Retieval
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| KILT | [fairseq_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz)|[hf_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz)|

See [here](https://github.com/facebookresearch/GENRE/blob/main/examples) examples to load the models and make inference.

## Dataset

Use the link above to download datasets. As an alternative use [this](https://github.com/facebookresearch/GENRE/blob/main/scripts/download_all_datasets.sh) script to dowload all of them. These dataset (except BLINK data) are a pre-processed version of [Phong Le and Ivan Titov (2018)](https://arxiv.org/pdf/1804.10637.pdf) data availabe [here](https://github.com/lephong/mulrel-nel). BLINK data taken from [here](https://github.com/facebookresearch/KILT).

### Entity Disambiguation (train / dev)
- [BLINK train](http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl) (9,000,000 lines, 11GiB)
- [BLINK dev](http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl) (10,000 lines, 13MiB)
- [AIDA-YAGO2 train](http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl) (18,448 lines, 56MiB)
- [AIDA-YAGO2 dev](http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl) (4,791 lines, 15MiB)

### Entity Disambiguation (test)
- [ACE2004](http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl) (257 lines, 850KiB)
- [AQUAINT](http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl) (727 lines, 2.0MiB)
- [AIDA-YAGO2](http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl) (4,485 lines, 14MiB)
- [MSNBC](http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl) (656 lines, 1.9MiB)
- [WNED-CWEB](http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl) (11,154 lines, 38MiB)
- [WNED-WIKI](http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl) (6,821 lines, 19MiB)

### Document Retieval
- KILT for the these datasets please follow the download instruction on the [KILT](https://github.com/facebookresearch/KILT) repository.

### Pre-processing
To pre-process a KILT formatted dataset into source and target files as expected from `fairseq` use 
```bash
python scripts/convert_kilt_to_fairseq.py $INPUT_FILENAME $OUTPUT_FOLDER
```
Then, to tokenize and binarize them as expected from `fairseq` use 
```bash
./preprocess_fairseq.sh $DATASET_PATH $MODEL_PATH
```
note that this requires to have `fairseq` source code downloaded in the same folder as the `genre` repository (see [here](https://github.com/facebookresearch/GENRE/blob/main/scripts/preprocess_fairseq.sh#L14)).

### Trie from KILT Wikipedia titles
We also release the BPE prefix tree (trie) from KILT Wikipedia titles ([kilt_titles_trie.pkl](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie.pkl)) that is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format [here](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2).
The trie contains ~5M titles and it is used to generate entites for all the KILT experiments.

## Troubleshooting
If the module cannot be found, preface the python command with `PYTHONPATH=.`

## Licence
LAMA is licensed under the CC-BY-NC 4.0 license. The text of the license can be found [here](https://github.com/facebookresearch/GENRE/blob/main/LICENSE).
