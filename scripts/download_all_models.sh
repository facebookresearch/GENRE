#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir models
cd models

### Entity Disambiguation
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_aidayago.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_entity_disambiguation_blink.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz

tar -zxvf fairseq_entity_disambiguation_aidayago.tar.gz
tar -zxvf hf_entity_disambiguation_aidayago.tar.gz
tar -zxvf fairseq_entity_disambiguation_blink.tar.gz
tar -zxvf hf_entity_disambiguation_blink.tar.gz

### End-to-End Entity Linking
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_aidayago.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_e2e_entity_linking_wiki_abs.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz

tar -zxvf fairseq_e2e_entity_linking_aidayago.tar.gz
tar -zxvf hf_e2e_entity_linking_aidayago.tar.gz
tar -zxvf fairseq_e2e_entity_linking_wiki_abs.tar.gz
tar -zxvf hf_e2e_entity_linking_wiki_abs.tar.gz

### Document Retieval
wget http://dl.fbaipublicfiles.com/GENRE/fairseq_wikipage_retrieval.tar.gz
wget http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz

tar -zxvf fairseq_wikipage_retrieval.tar.gz
tar -zxvf hf_wikipage_retrieval.tar.gz

cd ..

mkdir data
cd data

### KILT prefix tree
wget http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl

cd ..
