#!/usr/bin/env bash
set -eux
export CUDA_VISIBLE_DEVICE=0,1

bs=48
val_bs=48
epos=5
lr=3e-5
alpha=1
warm=0.1
dp=0.2
wd=0.01

cd ..

run_lite(){
    python -u trian.py \
        --lr=${lr} \
        --batch_size=${bs} \
        --val_batch_size=${val_bs} \
        --epochs=${epos} \
        --alpha=${alpha} \
        --warmup_proportion=${warm} \
        --dp=${dp} \
        --wd=${wd} \
        --hyper_cfg=a_${alpha}_lr_${lr}_dp_${dp}_wu_${warm}_wd_${wd}_lite \
        --lite \
        &>> ../../logs/lite.log
}
run_full(){
    python -u trian.py \
        --lr=${lr} \
        --batch_size=${bs} \
        --val_batch_size=${val_bs} \
        --epochs=${epos} \
        --alpha=${alpha} \
        --warmup_proportion=${warm} \
        --dp=${dp} \
        --wd=${wd} \
        --hyper_cfg=a_${alpha}_lr_${lr}_dp_${dp}_wu_${warm}_wc_${wd}_lite \
        &>> ../../logs/full.log
}

run_lite