# Models
You need to download dictionaries in the same folder of the model 
- [http://dl.fbaipublicfiles.com/GENRE/dict.source.txt](http://dl.fbaipublicfiles.com/GENRE/dict.source.txt)
- [http://dl.fbaipublicfiles.com/GENRE/dict.target.txt](http://dl.fbaipublicfiles.com/GENRE/dict.target.txt)
- Entity Disambiguation: [http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt](http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt)

- model_path = "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models//kilt"
- checkpoint_file = "checkpoint.pt"

## End-to-End Entity Linking

- model_path="/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/el"
- checkpoint_file='checkpoint_aidayago.pt'

# Models (work in progress)

## Entity Disambiguation
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| AidaYago2 | [fairseq_entity_disambiguation_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz)|[hf_entity_disambiguation_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)|
| Wikipedia | [fairseq_entity_disambiguation_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_wiki.tar.gz)|[hf_entity_disambiguation_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_wiki.tar.gz)|

## End-to-End Entity Linking
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| AidaYago2 | [fairseq_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_aidayago.tar.gz)|[hf_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_aidayago.tar.gz)|
| Wikipedia | [fairseq_entity_linking_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_wiki.tar.gz)|[hf_entity_linking_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_wiki.tar.gz)|

## Document Retieval
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| KILT | [fairseq_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_document_retrieval_kilt.tar.gz)|[hf_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_document_retrieval_kilt.tar.gz)|


# Dataset (work in progress)
## Entity Disambiguation
- [ACE2004](http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl)
- [AIDA-YAGO2 train set](http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl)
- [AIDA-YAGO2 dev set](http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl)
- [AIDA-YAGO2 test set](http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl)
- [AQUAINT](http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl)
- [MSNBC](http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl)
- [WNED-CWEB](http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl)
- [WNED-WIKI](http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl)

## End-to-End Entity Linking

## Document Retieval

# Create your trie
```python
from genre.trie import Trie
bpes = [[2] + model.encode(e)[1:].tolist() for e in candidates]
trie = Trie(bpes)
# you can then dump/load the trie with pickle
```
