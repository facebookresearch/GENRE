#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


DATASET=$1
MODEL=$2
DICT=$3

echo "Processing ${DATASET}"

for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python scripts/preprocess_sentencepiece.py --m ${MODEL} \
        --inputs ${DATASET}/${SPLIT}.${LANG} \
        --outputs ${DATASET}/${SPLIT}.spm.${LANG} \
        --workers 40
    done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref ${DATASET}/train.spm \
  --validpref ${DATASET}/dev.spm \
  --destdir ${DATASET}/bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 40
