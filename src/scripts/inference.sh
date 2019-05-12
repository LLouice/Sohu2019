#!/usr/bin/env bash
set -eux
test_batch_size=500
python -u ../test.py \
    --test_batch_size=${test_batch_size} \

python -u ../get_result.py
