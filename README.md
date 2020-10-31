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


# Models for [pytorch / fairseq](https://github.com/pytorch/fairseq) (work in progress)
## Entity Disambiguation
- [fairseq_entity_disambiguation_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz)
- [fairseq_entity_disambiguation_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_wiki.tar.gz)

## End-to-End Entity Linking
- [fairseq_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_aidayago.tar.gz)
- [fairseq_entity_linking_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_wiki.tar.gz)

## Document Retieval
- [fairseq_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_document_retrieval_kilt.tar.gz)

# Models for [huggingface / transformers](https://github.com/huggingface/transformers) (work in progress)
## Entity Disambiguation
- [hf_entity_disambiguation_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)
- [hf_entity_disambiguation_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_wiki.tar.gz)

## End-to-End Entity Linking
- [hf_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_aidayago.tar.gz)
- [hf_entity_linking_wiki.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_wiki.tar.gz)

## Document Retieval
- [hf_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_document_retrieval_kilt.tar.gz)

# Create your trie
```python
from genre.trie import Trie
bpes = [[2] + model.encode(e)[1:].tolist() for e in candidates]
trie = Trie(bpes)
# you can then dump/load the trie with pickle
```
