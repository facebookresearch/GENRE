#!/bin/bash

DATASET=$1
echo "Processing $1"

cd src/fairseq-py

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json /checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/bart.large/encoder.json\
            --vocab-bpe /checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/bart.large/vocab.bpe \
            --inputs "$DATASET/$SPLIT.$LANG" \
            --outputs "$DATASET/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

cd ..

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET/train.bpe" \
    --validpref "$DATASET/dev.bpe" \
    --destdir "$DATASET/bin" \
    --workers 60 \
    --srcdict /checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/bart.large/dict.txt \
    --tgtdict /checkpoint/fabiopetroni/GENRE/checkpoint/GeNeRe/data/bart.large/dict.txt;
