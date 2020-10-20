#!/bin/bash

KILT_DATA=/home/ndecao/KILT/data
FAIRSEQ_PATH=/datastore/shared/kilt/fairseq/query2answer
for dataset in fever nq hotpotqa triviaqa eli5 trex structured_zeroshot aidayago2 wow
do
    for split in train dev
    do
        python scripts/convert_kilt_to_fairseq.py $KILT_DATA/$dataset-$split-kilt.jsonl $FAIRSEQ_PATH/$dataset -v --mode answer
    done
done
