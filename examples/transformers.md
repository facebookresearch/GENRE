# GENRE for fairseq

First make sure that you have [transformers](https://github.com/huggingface/transformers) >=4.0.0 installed. 


## Entity Disambiguation
Download one of the pre-trained models:

| Training Dataset | Model |
| -------- | -------- |
| BLINK | [hf_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz)|
| BLINK + AidaYago2 | [hf_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)|

as well as the prefix tree from KILT Wikipedia titles ([kilt_titles_trie.pkl](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie.pkl)).

Then load the trie and define the function to apply the constraints with the entities trie


```python
import pickle

with open("data/kilt_titles_trie.pkl", "rb") as f:
    trie = pickle.load(f)

def prefix_allowed_tokens_fn(batch_id, sent):
    return trie.get(sent.tolist())
```

Then, load the model


```python
from transformers import BartForConditionalGeneration, BartTokenizer
tokenizer = BartTokenizer.from_pretrained("models/hf_entity_disambiguation_aidayago")
model = BartForConditionalGeneration.from_pretrained("models/hf_entity_disambiguation_aidayago").eval()
```

and simply use `.generate` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["[START_ENT] Armstrong [END_ENT] was the first man on the Moon."]

input_args = {
    k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
        sentences,
        padding=True,
        return_tensors="pt"
    ).items()
}

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['Neil Armstrong',
     'William Armstrong',
     'Scott Armstrong',
     'Arthur Armstrong',
     'Rob Armstrong']



## Document Retieval
Download one of the pre-trained models:

| Training Dataset | Models |
| -------- | -------- |
| KILT | [hf_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz)|

Then, load the model


```python
tokenizer = BartTokenizer.from_pretrained("models/hf_wikipage_retrieval")
model = BartForConditionalGeneration.from_pretrained("models/hf_wikipage_retrieval").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["Armstrong was the first man on the Moon."]

input_args = {
    k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt"
    ).items()
}

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['Neil Armstrong', 'Apollo 11', 'Astronaut', 'Buzz Aldrin', 'Apollo 17']



## End-to-End Entity Linking

Download one of the pre-trained models:

| Training Dataset | Models |
| -------- | -------- |
| WIKIPEDIA | [hf_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz)|
| WIKIPEDIA + AidaYago2 | [hf_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz)|

Then, load the model


```python
tokenizer = BartTokenizer.from_pretrained("models/hf_e2e_entity_linking_wiki_abs")
model = BartForConditionalGeneration.from_pretrained("models/hf_e2e_entity_linking_wiki_abs").eval()
```

and 
1. get the `prefix_allowed_tokens_fn` with the only constraints to annotate the original sentence (i.e., no other constrains on mention nor candidates)
2. use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf

sentences = ["In 1921, Einstein received a Nobel Prize."]

input_args = {
    k: v.to(model.device) for k, v in tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt"
    ).items()
}

prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(tokenizer, sentences)

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['In { 1921 } [ List of Nobel laureates in Physiology or Medicine by year of appointment ], { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.',
     'In { 1921 } [ List of Nobel laureates in Physiology or Medicine by year of appointment ], { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ Einstein (crater) ]',
     'In { 1921 } [ List of Nobel laureates in Physiology or Medicine by year of appointment ], { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ Einstein (crater) ]',
     'In { 1921 } [ List of Nobel laureates in Physiology or Medicine by year of appointment ], { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ Doctor of Philosophy ]',
     'In { 1921 } [ List of Nobel laureates in Physiology or Medicine by year of appointment ], { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ Max Einstein ]']



You can constrain the mentions with a prefix tree (no constrains on candidates)


```python
from genre.trie import Trie

prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
    tokenizer,
    sentences,
    mention_trie=Trie([
        tokenizer.encode(" {}".format(e))[1:]
        for e in ["Einstein"]
    ]),
)

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
     'In 1921, { Einstein } [ Einstein (crater) ] received a Nobel Prize.',
     'In 1921, { Einstein } [ Albert Albert Einstein ] received a Nobel Prize.',
     'In 1921, { Einstein } [Albert Einstein ] received a Nobel Prize.',
     'In 1921, { Einstein } [ Max Einstein ] received a Nobel Prize.']



You can constrain the candidates with a prefix tree (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
    tokenizer,
    sentences,
    candidates_trie=Trie([
        tokenizer.encode(" }} [ {} ]".format(e))[1:]
        for e in ["Albert Einstein", "Nobel Prize in Physics", "NIL"]
    ])
)

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize. } [ Nobel Prize in Physics ]',
     'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.',
     'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ NIL ]',
     'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ NIL ]',
     'In 1921, { Einstein } [ Albert Einstein ] { received } [ Nobel Prize in Physics ] a { Nobel Prize. } [ NIL ]']



You can constrain the candidate sets given a mention (no constrains on mentions)


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
    tokenizer,
    sentences,
    mention_to_candidates_dict={
        "Einstein": ["Einstein"],
        "Nobel": ["Nobel Prize in Physics"],
    }
)

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['In 1921, Einstein received a { Nobel } [ Nobel Prize in Physics ] Prize.',
     'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.',
     'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ NIL ]',
     'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize. } [ NIL ]',
     'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ NIL ].']



A combiation of these constraints is also possible


```python
prefix_allowed_tokens_fn = get_end_to_end_prefix_allowed_tokens_fn_hf(
    tokenizer,
    sentences,
    mention_trie=Trie([
        tokenizer.encode(" {}".format(e))[1:]
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

tokenizer.batch_decode(
    model.generate(
        **input_args,
        min_length=0,
        num_beams=5,
        num_return_sequences=5,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    ),
    skip_special_tokens=True
)
```




    ['In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.',
     'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Physics ].',
     'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Medicine ].',
     'In 1921, Einstein received a Nobel Prize.',
     'In 1921, Einstein received a Nobel Prize.']



Finally, you can also use some functions from `genre.utils` that wraps pre- and post-processing of strings (e.g., normalization and outputs the character offsets and length of the mentions):


```python
from genre.utils import get_entity_spans_hf

get_entity_spans_hf(
    model,
    tokenizer,
    sentences,
    mention_trie=Trie([
        tokenizer.encode(" {}".format(e))[1:]
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)
```




    [[[9, 8, 'Albert_Einstein']]]




```python
from genre.utils import get_markdown
from IPython.display import Markdown

entity_spans = get_entity_spans_hf(
    model,
    tokenizer,
    sentences,
    mention_trie=Trie([
        tokenizer.encode(" {}".format(e))[1:]
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

Markdown(get_markdown(sentences, entity_spans)[0])
```




> In 1921, [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) received a Nobel Prize.


