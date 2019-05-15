#!/usr/bin/env bash
# 服务器间同步数据集
server_75="llouice@10.108.208.113"
server_lzk="lzk@10.112.101.47"
server_68="llouice@10.108.209.96"
cd ..

Sohu2019="/home/llouice/Projects/BERT/Sohu2019/datasets"
target_dir="../."

function init() {
    scp -r "${server_75}:${Sohu2019}" ${target_dir}
    echo "copy Sohu2019 over!"
}

init



