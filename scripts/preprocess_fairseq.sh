#!/bin/bash

DATASET=$1
echo "Processing $1"

cd ../fairseq

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json /home/ndecao/GENRE/models/bart.large/encoder.json\
            --vocab-bpe /home/ndecao/GENRE/models/bart.large/vocab.bpe \
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
    --srcdict /home/ndecao/GENRE/models/bart.large/dict.txt \
    --tgtdict /home/ndecao/GENRE/models/bart.large/dict.txt;
