#!/bin/bash

# rm *0.pt *1.pt *2.pt *3.pt *4.pt *5.pt *6.pt *7.pt *8.pt *9.pt

cd /private/home/fabiopetroni/DPR_KILT/__GENRE/src/fairseq-py

# remember to check `restore_file` and `total_num_udpates` in run_bart_slurm.py
python ../../scripts/run_bart_slurm.py \
    -n 1 \
    -g 8 \
    -t 1 \
    -p "nq_genre_dpr" \
    -d "/private/home/fabiopetroni/DPR_KILT/__GENRE/data/nq/bin" \
    --constraint volta32gb \
    --mem 500G \
    --resume-failed
    