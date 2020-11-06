#!/bin/bash

# rm *0.pt *1.pt *2.pt *3.pt *4.pt *5.pt *6.pt *7.pt *8.pt *9.pt

cd /private/home/ndecao/fairseq-py

# remember to check `restore_file` and `total_num_udpates` in run_bart_slurm.py
# python ../GENRE/scripts/run_bart_slurm.py \
#     -n 1 \
#     -g 8 \
#     -t 1 \
#     -p "nq_context2answer" \
#     -d "/checkpoint/ndecao/GENRE/data/fairseq/context2answer/nq/bin" \
#     --constraint volta32gb \
#     --mem 500G \
#     --resume-failed
    
python ../GENRE/scripts/run_bart_slurm.py \
    -n 1 \
    -g 8 \
    -t 1 \
    -p "nq_answer_context2query" \
    -d "/checkpoint/ndecao/GENRE/data/fairseq/answer_context2query/nq/bin" \
    --constraint volta32gb \
    --mem 500G \
    --resume-failed


python ../GENRE/scripts/run_bart_slurm.py \
    -n 1 \
    -g 1 \
    -t 1 \
    -p "nq_answer_context2query_local" \
    -d "/checkpoint/ndecao/GENRE/data/fairseq/answer_context2query/nq/bin" \
    --resume-failed \
    --local