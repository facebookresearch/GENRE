#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

fairseq-train $DATASET/bin/ \
    --save-dir models/$NAME \
    --tensorboard-logdir tensorboard_logs/$NAME \
    --restore-file models/bart.large/model.pt \
    --arch bart_large  \
    --task translation  \
    --criterion label_smoothed_cross_entropy  \
    --source-lang source  \
    --target-lang target  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens 1024  \
    --update-freq 1  \
    --max-update 200000  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.01  \
    --optimizer adam  \
    --adam-betas "(0.9, 0.999)"  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 3e-05  \
    --total-num-update 200000  \
    --warmup-updates 500  \
    --ddp-backend no_c10d  \
    --num-workers 20  \
    --reset-meters  \
    --reset-optimizer \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 10  \
    --patience 200 \
    --encoder-normalize-before \
    --decoder-normalize-before
