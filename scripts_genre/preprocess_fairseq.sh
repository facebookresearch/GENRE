#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


DATASET_PATH=$1
MODEL_PATH=$2

echo "Processing ${DATASET}"

cd ../fairseq

# BPE preprocessing.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python -m examples.roberta.multiprocessing_bpe_encoder\
            --encoder-json "$MODEL_PATH/encoder.json" \
            --vocab-bpe "$MODEL_PATH//vocab.bpe" \
            --inputs "$DATASET_PATH/$SPLIT.$LANG" \
            --outputs "$DATASET_PATH/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

cd ..

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET_PATH/train.bpe" \
    --validpref "$DATASET_PATH/dev.bpe" \
    --destdir "$DATASET_PATH/bin" \
    --workers 60 \
    --srcdict "$MODEL_PATH//dict.source.txt" \
    --tgtdict "$MODEL_PATH/dict.target.txt";
