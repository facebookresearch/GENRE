# GENRE for fairseq

First make sure that you have [fairseq](https://github.com/pytorch/fairseq) installed.
Since `fairseq` is going through breaking changes please install it from [this](https://github.com/nicola-decao/fairseq/tree/fixing_prefix_allowed_tokens_fn) fork using: 
```bash
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq
pip install --editable ./
``` 
as described in the [fairseq repository](https://github.com/pytorch/fairseq#requirements-and-installation) since `pip install fairseq` has issues. 

# GENRE for transformers

First make sure that you have [transformers](https://github.com/huggingface/transformers) >=4.2.0 installed. 
**NOTE: we used fairseq for all experiments in the paper. The huggingface/transformers models are obtained with a [conversion script](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/convert_bart_original_pytorch_checkpoint_to_pytorch.py).**

<hr>

# Datasets

Use the links below to download datasets. As an alternative use [this](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh) script to dowload all of them. These dataset (except BLINK data) are a pre-processed version of [Phong Le and Ivan Titov (2018)](https://arxiv.org/pdf/1804.10637.pdf) data availabe [here](https://github.com/lephong/mulrel-nel). BLINK data taken from [here](https://github.com/facebookresearch/KILT).

## Entity Disambiguation (train / dev)
- [BLINK train](http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl) (9,000,000 lines, 11GiB)
- [BLINK dev](http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl) (10,000 lines, 13MiB)
- [AIDA-YAGO2 train](http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl) (18,448 lines, 56MiB)
- [AIDA-YAGO2 dev](http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl) (4,791 lines, 15MiB)

## Entity Disambiguation (test)
- [ACE2004](http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl) (257 lines, 850KiB)
- [AQUAINT](http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl) (727 lines, 2.0MiB)
- [AIDA-YAGO2](http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl) (4,485 lines, 14MiB)
- [MSNBC](http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl) (656 lines, 1.9MiB)
- [WNED-CWEB](http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl) (11,154 lines, 38MiB)
- [WNED-WIKI](http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl) (6,821 lines, 19MiB)

## Entity Linking (train)
- [WIKI-ABSTRACTS](http://dl.fbaipublicfiles.com/GENRE/train_data_e2eEL.tar.gz) (6,221,563 lines, 5.1GiB)

## Document Retieval
- KILT for the these datasets please follow the download instruction on the [KILT](https://github.com/facebookresearch/KILT) repository.

## Pre-processing
To pre-process a KILT formatted dataset into source and target files as expected from `fairseq` use 
```bash
python scripts/convert_kilt_to_fairseq.py $INPUT_FILENAME $OUTPUT_FOLDER
```
Then, to tokenize and binarize them as expected from `fairseq` use 
```bash
./preprocess_fairseq.sh $DATASET_PATH $MODEL_PATH
```
note that this requires to have `fairseq` source code downloaded in the same folder as the `genre` repository (see [here](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/preprocess_fairseq.sh#L14)).

## Trie from KILT Wikipedia titles

We also release the BPE prefix tree (trie) from KILT Wikipedia titles ([kilt_titles_trie_dict.pkl](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl)) that is based on the 2019/08/01 Wikipedia dump, downloadable in its raw format [here](http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2).
The trie contains ~5M titles and it is used to generate entites for all the KILT experiments.

<hr>

# Example: Entity Disambiguation
Download one of the pre-trained models:

| Training Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| BLINK | [fairseq_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz)|[hf_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz)|
| BLINK + AidaYago2 | [fairseq_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz)|[hf_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)|

as well as the prefix tree from KILT Wikipedia titles ([kilt_titles_trie_dict.pkl](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl)).

Then load the trie and define the function to apply the constraints with the entities trie


```python
# OPTIONAL:
import sys
sys.path.append("../")
```


```python
import pickle
from genre.trie import Trie

# load the prefix tree (trie)
with open("../data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))
```

Then, load the model


```python
# for pytorch/fairseq
from genre.fairseq_model import GENRE
model = GENRE.from_pretrained("../models/fairseq_entity_disambiguation_aidayago").eval()

# for huggingface/transformers
# from genre.hf_model import GENRE
# model = GENRE.from_pretrained("../models/hf_entity_disambiguation_aidayago").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["Einstein was a [START_ENT] German [END_ENT] physicist."]

model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
```




    [[{'text': 'Germany', 'score': tensor(-0.1856)},
      {'text': 'Germans', 'score': tensor(-0.5461)},
      {'text': 'German Empire', 'score': tensor(-2.1858)},
      {'text': 'Nazi Germany', 'score': tensor(-2.4682)},
      {'text': 'France', 'score': tensor(-4.2070)}]]



# Example: Document Retieval
Download one of the pre-trained models:

| Training Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| [KILT](https://github.com/facebookresearch/KILT) | [fairseq_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz)|[hf_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz)|

Then, load the model


```python
# for pytorch/fairseq
from genre.fairseq_model import GENRE
model = GENRE.from_pretrained("../models/fairseq_wikipage_retrieval").eval()

# for huggingface/transformers
# from genre.hf_model import GENRE
# model = GENRE.from_pretrained("../models/hf_wikipage_retrieval").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["Einstein was a German physicist."]

model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
```




    [[{'text': 'Albert Einstein', 'score': tensor(-0.0708)},
      {'text': 'Werner Bruschke', 'score': tensor(-1.5357)},
      {'text': 'Werner von Habsburg', 'score': tensor(-1.8696)},
      {'text': 'Werner von Moltke', 'score': tensor(-2.2318)},
      {'text': 'Werner von Eichstedt', 'score': tensor(-3.0177)}]]



# Example: End-to-End Entity Linking

Download one of the pre-trained models:

| Training Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| WIKIPEDIA | [fairseq_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz)|[hf_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz)|
| WIKIPEDIA + AidaYago2 | [fairseq_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz)|[hf_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz)|

Then, load the model


```python
# for pytorch/fairseq
from genre.fairseq_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_fairseq as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_fairseq as get_entity_spans
model = GENRE.from_pretrained("../models/fairseq_e2e_entity_linking_aidayago").eval()

# for huggingface/transformers
# from genre.hf_model import GENRE
# from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
# from genre.utils import get_entity_spans_hf as get_entity_spans
# model = GENRE.from_pretrained("../models/hf_e2e_entity_linking_aidayago").eval()
```

and 
1. get the `prefix_allowed_tokens_fn` with the only constraints to annotate the original sentence (i.e., no other constrains on mention nor candidates)
2. use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["In 1921, Einstein received a Nobel Prize."]

prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ].',
       'score': tensor(-0.9068)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Nobel Prize in Physiology or Medicine ]',
       'score': tensor(-0.9301)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Albert Einstein ]',
       'score': tensor(-0.9943)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Physiology ].',
       'score': tensor(-1.0778)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Ernest Einstein ]',
       'score': tensor(-1.1164)}]]



You can constrain the mentions with a prefix tree (no constrains on candidates)


```python
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(e)[1:].tolist()
        for e in [" Einstein"]
    ])
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
       'score': tensor(-1.4009)},
      {'text': 'In 1921, { Einstein } [ Einstein (crater) ] received a Nobel Prize.',
       'score': tensor(-1.6665)},
      {'text': 'In 1921, { Einstein } [ Albert Albert Einstein ] received a Nobel Prize.',
       'score': tensor(-1.7498)},
      {'text': 'In 1921, { Einstein } [ Ernest Einstein ] received a Nobel Prize.',
       'score': tensor(-1.8327)},
      {'text': 'In 1921, { Einstein } [ Max Einstein ] received a Nobel Prize.',
       'score': tensor(-1.8757)}]]



You can constrain the candidates with a prefix tree (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    candidates_trie=Trie([
        model.encode(" }} [ {} ]".format(e))[1:].tolist()
        for e in ["Albert Einstein", "Nobel Prize in Physics", "NIL"]
    ])
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'score': tensor(-0.8925)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize. } [ Nobel Prize in Physics ]',
       'score': tensor(-0.8990)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ Nobel Prize in Physics ].',
       'score': tensor(-0.9330)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ Nobel Prize in Physics ]',
       'score': tensor(-0.9781)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ Albert Einstein ]',
       'score': tensor(-0.9854)}]]



You can constrain the candidate sets given a mention (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_to_candidates_dict={
        "Einstein": ["Einstein"],
        "Nobel": ["Nobel Prize in Physics"],
    }
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.',
       'score': tensor(-1.5417)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a Nobel Prize.',
       'score': tensor(-2.1319)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ NIL ].',
       'score': tensor(-2.2816)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize. } [ NIL ]',
       'score': tensor(-2.3914)},
      {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel Prize. } [ NIL ]',
       'score': tensor(-2.9078)}]]



A combiation of these constraints is also possible


```python
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```




    [[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'score': tensor(-0.8925)},
      {'text': 'In 1921, { Einstein } [ Einstein (surname) ] received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'score': tensor(-1.3275)},
      {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
       'score': tensor(-1.4009)},
      {'text': 'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Physics ].',
       'score': tensor(-1.8266)},
      {'text': 'In 1921, Einstein received a Nobel Prize.',
       'score': tensor(-3.4495)}]]



Finally, you can also use some functions from `genre.utils` that wraps pre- and post-processing of strings (e.g., normalization and outputs the character offsets and length of the mentions)


```python
get_entity_spans(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)
```




    [[(9, 8, 'Albert_Einstein'), (29, 11, 'Nobel_Prize_in_Physics')]]



and with the `entity_spans` generate Markdown with clickable links


```python
from genre.utils import get_markdown
from IPython.display import Markdown

entity_spans = get_entity_spans(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

Markdown(get_markdown(sentences, entity_spans)[0])
```




In 1921, [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) received a [Nobel Prize](https://en.wikipedia.org/wiki/Nobel_Prize_in_Physics).



## Custom End-to-End Entity Linking evaluation

We have some useful function to evaluate End-to-End Entity Linking predictions. Let's suppose we have a `Dict[str, str]` with document IDs and text as well as the gold entites spans as a `List[Tuple[str, int, int, str]]` containing documentID, start offset, length and entity title respectively.


```python
documents = {
    "id_0": "In 1921, Einstein received a Nobel Prize.",
    "id_1": "Armstrong was the first man on the Moon.",
}

gold_entities = [
    ("id_0", 3, 4, "1921"),
    ("id_0", 9, 8, 'Albert_Einstein'),
    ("id_0", 29, 11, 'Nobel_Prize_in_Physics'),
    ("id_1", 0, 9, 'Neil_Armstrong'),
    ("id_1", 35, 4, 'Moon'),
]
```

Then we can get preditions and using `get_entity_spans_fairseq` to have spans. `guess_entities` is then a `List[List[Tuple[int, int, str]]]` containing for each document, a list of entity spans (without the document ID). We further need to add documentIDs to `guess_entities` and remove the nested list to be compatible with `gold_entities`.


```python
guess_entities = get_entity_spans(
    model,
    list(documents.values()),
)

guess_entities = [
    (k,) + x
    for k, e in zip(documents.keys(), guess_entities)
    for x in e
]
```

Finally, we can import all functions from `genre.utils` to compute scores.


```python
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

micro_p = get_micro_precision(guess_entities, gold_entities)
micro_r = get_micro_recall(guess_entities, gold_entities)
micro_f1 = get_micro_f1(guess_entities, gold_entities)
macro_p = get_macro_precision(guess_entities, gold_entities)
macro_r = get_macro_recall(guess_entities, gold_entities)
macro_f1 = get_macro_f1(guess_entities, gold_entities)

print(
   "micro_p={:.4f} micro_r={:.4f}, micro_f1={:.4f}, macro_p={:.4f}, macro_r={:.4f}, macro_f1={:.4f}".format(
       micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1
   )
)
```

    micro_p=0.2500 micro_r=0.4000, micro_f1=0.3077, macro_p=0.2500, macro_r=0.4167, macro_f1=0.3095



```python
assert 2 / 8 == micro_p
assert 2 / 5 == micro_r
assert 2 * (2 / 8 * 2 / 5) / (2 / 8 + 2 / 5) == micro_f1
assert (1 / 4 + 1 / 4) / 2 == macro_p
assert (1 / 3 + 1 / 2) / 2 == macro_r
assert (2 * (1 / 4 * 1 / 3) / (1 / 4 + 1 / 3) + 2 * (1 / 4 * 1 / 2) / (1 / 4 + 1 / 2)) / 2 == macro_f1
```
