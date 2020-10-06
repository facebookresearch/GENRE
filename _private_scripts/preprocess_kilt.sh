#!/bin/bash

###### DOWNLOADING KILT datasets ######
cd KILT
mkdir data
python scripts/donwload_all_kilt_data.py
python scripts/get_triviaqa_input.py
cd data
wget http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl
wget http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl
cd ../..
##############################

###### CONVERTING KILT datasets to source/target for fairseq ######
mkdir data
mkdir data/kilt
python scripts/convert_kilt_to_fairseq.py KILT/data data/kilt
mkdir data/kilt/all
cd data/kilt
cat aidayago2/train.source wow/train.source fever/train.source nq/train.source hotpotqa/train.source triviaqa/train.source blink/train.source trex/train.source structured_zeroshot/train.source >> all/train.source
cat aidayago2/train.target wow/train.target fever/train.target nq/train.target hotpotqa/train.target triviaqa/train.target blink/train.target trex/train.target structured_zeroshot/train.target >> all/train.target
cat aidayago2/dev.source wow/dev.source fever/dev.source nq/dev.source wned/dev.source hotpotqa/dev.source triviaqa/dev.source blink/dev.source trex/dev.source eli5/dev.source cweb/dev.source structured_zeroshot/dev.source >> all/dev.source
cat aidayago2/dev.target wow/dev.target fever/dev.target nq/dev.target wned/dev.target hotpotqa/dev.target triviaqa/dev.target blink/dev.target trex/dev.target eli5/dev.target cweb/dev.target structured_zeroshot/dev.target >> all/dev.target
cd ../..
##############################

######  BINARIZING source/target for fairseq ###### 
./scripts/preprocess_fairseq.sh $(dirname $(realpath $0))/../data/kilt/all
##############################