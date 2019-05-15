#!/usr/bin/env bash
set -eux
export CUDA_VISIBLE_DEVICES=0,1

cd ../kflod

python -u train.py \
        --batch_size=128 \
        --val_batch_size=128 \
        --lite


python -u test.py \
       --test_batch_size=2000 \
       --lite

python -u test_avg.py




