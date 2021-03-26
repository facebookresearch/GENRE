#!/bin/bash

DATASET=$1

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref ${DATASET}/train.spm \
  --validpref ${DATASET}/dev.spm \
  --destdir ${DATASET}/bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict /private/home/ndecao/mGENRE/models/mbart.cc100/dict.txt \
  --tgtdict /private/home/ndecao/mGENRE/models/mbart.cc100/dict.txt \
  --workers 40
