#!/usr/bin/env bash
set -eux
export CUDA_VISIBLE_DEVICES=0,1
test_batch_size=500
best_model="best_model.pt.bak"
pred="pred_new.h5"


cd ../.

test(){
    python -u test.py \
        --test_batch_size=${test_batch_size} \
        --best_model=${best_model} \
        --pred=${pred}
}



res="result_new.txt"

get_res(){
    python -u get_result.py \
        --pred=${pred} \
        --res=${res}
}

test
get_res
echo "over!"

