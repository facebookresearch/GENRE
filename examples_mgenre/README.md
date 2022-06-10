# mGENRE for fairseq

First make sure that you have [fairseq](https://github.com/pytorch/fairseq) installed.
Since `fairseq` is going through breaking changes please install it from [this](https://github.com/nicola-decao/fairseq/tree/fixing_prefix_allowed_tokens_fn) fork using: 
```bash
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq
pip install --editable ./
``` 
as described in the [fairseq repository](https://github.com/pytorch/fairseq#requirements-and-installation) since `pip install fairseq` has issues. 

### Download
* the pre-trained **model** [fairseq_multilingual_entity_disambiguation](https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz);
* the **prefix tree (trie)** from Wikipedia titles [titles_lang_all105_trie_with_redirect.pkl](http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_trie_with_redirect.pkl)---this is fast but memory inefficient prefix tree implemented with nested python `dict`. As an alternative, we have a prefix tree implemented with `marisa_trie` that is much more memory efficient but a little slower [titles_lang_all105_marisa_trie_with_redirect.pkl](http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl);
* the dictionary to map the generated strings to **Wikidata identifiers** [lang_title2wikidataID-normalized_with_redirect](https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl) (the inverse mapping is availabe here [wikidataID2lang_title-normalized_with_redirect](https://dl.fbaipublicfiles.com/GENRE/wikidataID2lang_title-normalized_with_redirect.pkl));
* optionally, we can use a **mention table** to restrict the search space to a number of candidates [mention2wikidataID_with_titles_label_alias_redirect](https://dl.fbaipublicfiles.com/GENRE/mention2wikidataID_with_titles_label_alias_redirect.pkl).


# mGENRE for transformers

First make sure that you have [transformers](https://github.com/huggingface/transformers) >=4.2.0 installed. 
**NOTE: we used fairseq for all experiments in the paper. The huggingface/transformers models are obtained with a [conversion script](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/convert_bart_original_pytorch_checkpoint_to_pytorch.py).**

Then load the trie and define the function to apply the constraints with the entities trie


```python
# OPTIONAL:
import sys
sys.path.append("../")
```


```python
import pickle
from genre.trie import Trie, MarisaTrie

with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

# fast but memory inefficient prefix tree (trie) -- it is implemented with nested python `dict`
# NOTE: loading this map may take up to 10 minutes and occupy a lot of RAM!
# with open("../data/titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

# memory efficient but slower prefix tree (trie) -- it is implemented with `marisa_trie`
with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)
```

Then, load the model


```python
# for pytorch/fairseq
from genre.fairseq_model import mGENRE
model = mGENRE.from_pretrained("../models/fairseq_multilingual_entity_disambiguation").eval()

# for huggingface/transformers
# from genre.hf_model import mGENRE
# model = mGENRE.from_pretrained("../models/hf_multilingual_entity_disambiguation").eval()
```

and simply use `.sample` to make predictions constraining using `prefix_allowed_tokens_fn`


```python
sentences = ["[START] Einstein [END] era un fisico tedesco."]
# Italian for "[START] Einstein [END] was a German physicist."

model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist())
        if e < len(model.task.target_dictionary)
        # for huggingface/transformers
        # if e < len(model2.tokenizer) - 1
    ],
)
```




    [[{'text': 'Albert Einstein >> it', 'score': tensor(-0.0808)},
      {'text': 'Albert Einstein (disambiguation) >> en', 'score': tensor(-1.0998)},
      {'text': 'Alfred Einstein >> it', 'score': tensor(-1.4337)},
      {'text': 'Alberto Einstein >> it', 'score': tensor(-1.4619)},
      {'text': 'Einstein >> it', 'score': tensor(-1.5765)}]]



Additionally, we can use the `lang_title2wikidataID` dictionary to map the generated strings to Wikidata identifiers via the function `text_to_id`. The boolean parameter `marginalise` enables the aggregation of scores by entity ID


```python
model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist())
        if e < len(model.task.target_dictionary)
        # for huggingface/transformers
        # if e < len(model2.tokenizer) - 1
    ],
    text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
    marginalize=True,
)
```




    [[{'id': 'Q937',
       'texts': ['Albert Einstein >> it',
        'Alberto Einstein >> it',
        'Einstein >> it'],
       'scores': tensor([-0.0808, -1.4619, -1.5765]),
       'score': tensor(-0.0884)},
      {'id': 'Q60197',
       'texts': ['Alfred Einstein >> it'],
       'scores': tensor([-1.4337]),
       'score': tensor(-3.2058)},
      {'id': 'Q15990626',
       'texts': ['Albert Einstein (disambiguation) >> en'],
       'scores': tensor([-1.0998]),
       'score': tensor(-3.6478)}]]



Similar to `GENRE` we can use a mention table to restrict the search space to a number of candidates. We need fist two addinional dictionaries:


```python
# mapping between mentions and Wikidata IDs and number of times they appear on Wikipedia
with open("../data/mention2wikidataID_with_titles_label_alias_redirect.pkl", "rb") as f:
    mention2wikidataID = pickle.load(f)
    
# mapping between wikidataIDs and (lang, title) in all languages
with open("../data/wikidataID2lang_title-normalized_with_redirect.pkl", "rb") as f:
    wikidataID2lang_title = pickle.load(f)
```

then let's build the temporary trie for the mention and run inference


```python
sentences = ["[START] Einstein [END] era un fisico tedesco."]
# Italian for "[START] Einstein [END] was a German physicist."

# building a temporary trie for the mention (to the purpose of
# demonstraing the use of the mention table, let's restrict the
# prediction to only candidates in Italian!)
trie_of_mention = Trie([
    [2] + model.encode(f"{name} >> {lang}")[1:].tolist()
    for qid in mention2wikidataID["Einstein"]
    for lang, name in wikidataID2lang_title.get(qid, [])
    if lang == "it"
])

# getting predictions
model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie_of_mention.get(sent.tolist())
        if e < len(model.task.target_dictionary)
        # for huggingface/transformers
        # if e < len(model2.tokenizer) - 1
    ],
    text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
    marginalize=True,
)
```




    [[{'id': 'Q937',
       'texts': ['Albert Einstein >> it',
        'Alberto Einstein >> it',
        'Einstein >> it'],
       'scores': tensor([-0.0808, -1.4619, -1.5765]),
       'score': tensor(-0.0884)},
      {'id': 'Q60197',
       'texts': ['Alfred Einstein >> it'],
       'scores': tensor([-1.4337]),
       'score': tensor(-3.2058)},
      {'id': 'Q13426745',
       'texts': ['Albert Einstein (album) >> it'],
       'scores': tensor([-2.0844]),
       'score': tensor(-5.8956)}]]


