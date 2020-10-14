use fairseq from https://github.com/nicola-decao/fairseq on the `add_PrefixConstrainedBeamSearch` branch

# Models
You need to download vocabularies in the same folder of the model 
- [http://dl.fbaipublicfiles.com/GENRE/dict.source.txt](http://dl.fbaipublicfiles.com/GENRE/dict.source.txt)
- [http://dl.fbaipublicfiles.com/GENRE/dict.target.txt](http://dl.fbaipublicfiles.com/GENRE/dict.target.txt)
- Entity Disambiguation: [http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt](http://dl.fbaipublicfiles.com/GENRE/entity_disambiguation.pt)

## Document Retieval

- model_path = "/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models//kilt"
- checkpoint_file = "checkpoint.pt"

## End-to-End Entity Linking

- model_path="/checkpoint/fabiopetroni/GENRE/home/GeNeRe/__GENRE/models/el"
- checkpoint_file='checkpoint_aidayago.pt'
