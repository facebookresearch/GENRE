#!/bin/bash

# DATASET=$1
# MODEL=/private/home/ndecao/mGENRE/models/mbart.cc100/spm_256000.model

# echo "Processing ${DATASET}"
# LANG="target"
# for SPLIT in train dev; do
#     python scripts/preprocess_sentencepiece.py --m ${MODEL} \
#     --inputs ${DATASET}/${SPLIT}.${LANG} \
#     --outputs ${DATASET}/${SPLIT}.spm.${LANG} \
#     --workers 40
# done

DATASET=$1
MODEL=/private/home/ndecao/mGENRE/models/mbart.cc100/spm_256000.model

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
  --srcdict /private/home/ndecao/mGENRE/models/mbart.cc100/dict.txt \
  --tgtdict /private/home/ndecao/mGENRE/models/mbart.cc100/dict.txt \
  --workers 40
