#!/usr/bin/env bash
set -eux
src=$(pwd)/..
export PYTHONPATH=:${src}
gpu=$1
cv=$2
export CUDA_VISIBLE_DEVICES=${gpu}

cd ../kfold
bs=62
val_bs=90
test_bs=1400
pred="pred_5cv_new.h5"
pred_avg="pred_avg.h5"

train(){
    python -u train.py \
            --batch_size=${bs} \
            --val_batch_size=${val_bs} \
            --cv=${cv}
}


test(){
    python -u test.py \
           --test_batch_size=${test_bs} \
           --pred=${pred} \
           --cv=${cv}
}

avg(){
    python -u test_avg.py \
           --pred=${pred} \
           --pred_avg=${pred_avg}
}

train
test
#avg



