#!/usr/bin/env bash
set -eux
export CUDA_VISIBLE_DEVICES=0,1

cd ../kflod
bs=128
val_bs=128
test_bs=2000
pred="pred_5cv.h5"
pred_avg="pred_avg.h5"

train(){
    python -u train.py \
            --batch_size=${bs} \
            --val_batch_size=${val_bs} \
            --lite
}


test(){
    python -u test.py \
           --test_batch_size=${test_bs} \
           --pred=${pred} \
           --lite
}

avg(){
    python -u test_avg.py \
           --pred=${pred} \
           --pred_avg=${pred_avg}
}




