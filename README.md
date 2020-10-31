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
| BLINK | [fairseq_entity_disambiguation_blink.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz)|[hf_entity_disambiguation_blink.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz)|

## End-to-End Entity Linking
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| AidaYago2 | [fairseq_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_aidayago.tar.gz)|[hf_entity_linking_aidayago.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_aidayago.tar.gz)|
| BLINK | [fairseq_entity_linking_blink.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_linking_blink.tar.gz)|[hf_entity_linking_blink.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_entity_linking_blink.tar.gz)|

## Document Retieval
| Dataset | [pytorch / fairseq](https://github.com/pytorch/fairseq)   | [huggingface / transformers](https://github.com/huggingface/transformers) |
| -------- | -------- | ----------- |
| KILT | [fairseq_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/fairseq_document_retrieval_kilt.tar.gz)|[hf_document_retrieval_kilt.tar.gz](http://dl.fbaipublicfiles.com/GENRE/hf_document_retrieval_kilt.tar.gz)|


# Dataset (work in progress)

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

## End-to-End Entity Linking
- [GERBIL](https://github.com/dice-group/gerbil)

## Document Retieval
- [KILT](https://github.com/facebookresearch/KILT)

# Create your trie
```python
from genre.trie import Trie
bpes = [[2] + model.encode(e)[1:].tolist() for e in candidates]
trie = Trie(bpes)
# you can then dump/load the trie with pickle
```
