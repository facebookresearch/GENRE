#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mkdir datasets
cd datasets

### Entity Disambiguation (train / dev)
wget http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl
wget http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl

### Entity Disambiguation (test)
wget http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl

### Entity Linking (train)
wget http://dl.fbaipublicfiles.com/GENRE/train_data_e2eEL.tar.gz

cd ..
