# Models
You need to download dictionaries in the same folder of the model 
- [http://dl.fbaipublicfiles.com/GENRE/dict.source.txt](http://dl.fbaipublicfiles.com/GENRE/dict.source.txt)
- [http://dl.fbaipublicfiles.com/GENRE/dict.target.txt](http://dl.fbaipublicfiles.com/GENRE/dict.target.txt)
- Entity Disambiguation: [http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt](http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt)

## Document Retieval

- model_path = "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models//kilt"
- checkpoint_file = "checkpoint.pt"

## End-to-End Entity Linking

- model_path="/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/el"
- checkpoint_file='checkpoint_aidayago.pt'

# Create your trie
```
from genre.trie import Trie
bpes = [[2] + model.encode(e)[1:].tolist() for e in candidates]
trie = Trie(bpes)
# you can dump/load the trie with pickle
```
