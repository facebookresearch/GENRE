![](Genre-TwoColor-Light-BG.png)

The GENRE (Generative ENtity REtrieval) system as presented in [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) implemented in pytorch.

```bibtex
@inproceedings{decao2021autoregressive,
  author    = {Nicola {De Cao} and
               Gautier Izacard and
               Sebastian Riedel and
               Fabio Petroni},
  title     = {Autoregressive Entity Retrieval},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=5k8F6UU39V},
}
```

![](mGenre-TwoColor-Light-BG.png)

The mGENRE system as presented in [Multilingual Autoregressive Entity Linking](https://arxiv.org/abs/2103.12528)

```bibtex
@article{de-cao-etal-2022-multilingual,
    title = "Multilingual Autoregressive Entity Linking",
    author = "De Cao, Nicola  and
      Wu, Ledell  and
      Popat, Kashyap  and
      Artetxe, Mikel  and
      Goyal, Naman  and
      Plekhanov, Mikhail  and
      Zettlemoyer, Luke  and
      Cancedda, Nicola  and
      Riedel, Sebastian  and
      Petroni, Fabio",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.16",
    doi = "10.1162/tacl_a_00460",
    pages = "274--290",
}
```

**Please consider citing our works if you use code from this repository.**

In a nutshell, (m)GENRE uses a sequence-to-sequence approach to entity retrieval (e.g., linking), based on fine-tuned [BART](https://arxiv.org/abs/1910.13461) architecture or [mBART](https://arxiv.org/abs/2001.08210) (for multilingual). (m)GENRE performs retrieval generating the unique entity name conditioned on the input text using constrained beam search to only generate valid identifiers. Here an example of generation for Wikipedia page retrieval for open-domain question answering:

![](GENRE-animation-QA.gif)

For end-to-end entity linking GENRE re-generates the input text annotated with a markup:

![](GENRE-animation-EL.gif)

GENRE achieves state-of-the-art results on multiple datasets.

mGENRE performs multilingual entity linking in 100+ languages treating language as latent variables and marginalizing over them:

![](mGENRE-animation-EL.gif)

## Main dependencies
* python>=3.7
* pytorch>=1.6
* fairseq>=0.10 (optional for training GENRE) **NOTE: fairseq is going though changing without backward compatibility. Install `fairseq` from source and use [this](https://github.com/nicola-decao/fairseq/tree/fixing_prefix_allowed_tokens_fn) commit for reproducibilty. See [here](https://github.com/pytorch/fairseq/pull/3276) for the current PR that should fix `fairseq/master`.**
* transformers>=4.2 (optional for inference of GENRE)

## Examples & Usage

For a full review of (m)GENRE API see:
* [examples for GENRE](https://github.com/facebookresearch/GENRE/blob/main/examples_genre) on how to use GENRE for both pytorch fairseq and huggingface transformers;
* [examples for mGENRE](https://github.com/facebookresearch/GENRE/blob/main/examples_mgenre) on how to use mGENRE.

### GENRE
After importing and loading the model and a prefix tree (trie), you would generate predictions (in this example for Entity Disambiguation) with a simple call like:

```python
import pickle

from genre.fairseq_model import GENRE
from genre.trie import Trie

# load the prefix tree (trie)
with open("../data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

# load the model
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# generate Wikipedia titles
model.sample(
    sentences=["Einstein was a [START_ENT] German [END_ENT] physicist."],
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
```




    [[{'text': 'Germany', 'score': tensor(-0.1856)},
      {'text': 'Germans', 'score': tensor(-0.5461)},
      {'text': 'German Empire', 'score': tensor(-2.1858)}]


### mGENRE
Making predictions with mGENRE is very similar, but we additionally need to map `(title, language_ID)` to Wikidata IDs and (optionally) marginalize over predictions of the same entity:

```python
import pickle

from genre.fairseq_model import mGENRE
from genre.trie import MarisaTrie, Trie

with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

# memory efficient prefix tree (trie) implemented with `marisa_trie`
with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

# generate Wikipedia titles and language IDs
model = mGENRE.from_pretrained("../models/fairseq_multilingual_entity_disambiguation").eval()

model.sample(
    sentences=["[START] Einstein [END] era un fisico tedesco."],
    # Italian for "[START] Einstein [END] was a German physicist."
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
    ],
    text_to_id=lambda x: max(lang_title2wikidataID[
        tuple(reversed(x.split(" >> ")))
    ], key=lambda y: int(y[1:])),
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



## Models & Datasets

For **GENRE** use [this](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_models.sh) script to download all models and [this](https://github.com/facebookresearch/GENRE/blob/main/scripts_genre/download_all_datasets.sh) to download all datasets. See [here](https://github.com/facebookresearch/GENRE/blob/main/examples_genre) the list of all individual models for each task and for both pytorch fairseq and huggingface transformers. See the [example](https://github.com/facebookresearch/GENRE/blob/main/examples_genre) on how to download additional optional files like the prefix tree (trie) for KILT Wikipedia.

For **mGENRE** we only have a model available [here](https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz). See the [example](https://github.com/facebookresearch/GENRE/blob/main/examples_mgenre) on how to download additional optional files like the prefix tree (trie) for Wikipedia in all languages and the mapping between titles and Wikidata IDs.

Pre-trained **mBART** model on 125 languages available [here](https://dl.fbaipublicfiles.com/GENRE/mbart.cc100.tar.gz).

## Troubleshooting
If the module cannot be found, preface the python command with `PYTHONPATH=.`

## Licence
GENRE is licensed under the CC-BY-NC 4.0 license. The text of the license can be found [here](https://github.com/facebookresearch/GENRE/blob/main/LICENSE).
