#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 15:57
# @Author  : 邵明岩
# @File    : test_avg.py
# @Software: PyCharm

import os
import h5py
import torch
from argparse import ArgumentParser

os.chdir("../.")
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pred",
                        default="pred_5cv.h5",
                        type=str, required=False)
    parser.add_argument("--pred_avg",
                        default="pred_avg.h5",
                        type=str, required=False)
    args = parser.parse_args()

    fresult = h5py.File(f"../preds/{args.pred_avg}", "w")
    f = h5py.File(f"../preds/{args.pred}", "r")
    ent_raw = fresult.create_dataset("ent_raw", shape=(0, 128, 3), maxshape=(None, 128, 3), compression="gzip")
    emo_raw = fresult.create_dataset("emo_raw", shape=(0, 128, 4), maxshape=(None, 128, 4), compression="gzip")
    ent = fresult.create_dataset("ent", shape=(0, 128), maxshape=(None, 128), compression="gzip")
    emo = fresult.create_dataset("emo", shape=(0, 128), maxshape=(None, 128), compression="gzip")

    for cv in range(1, 6):
        print('cv : {}'.format(cv))
        if cv == 1:
            pred_ent_conf = f.get(f"cv1/ent_raw")[()]
            pred_emo_conf = f.get("cv1/emo_raw")[()]
        else:
            pred_ent_cur = f.get(f"cv{cv}/ent_raw")[()]
            pred_emo_cur = f.get(f"cv{cv}/emo_raw")[()]
            pred_ent_conf += pred_ent_cur
            pred_emo_conf += pred_emo_cur

    pred_ent_conf /= 5.0
    pred_emo_conf /= 5.0

    pred_ent_t = torch.from_numpy(pred_ent_conf)
    pred_emo_t = torch.from_numpy(pred_emo_conf)
    pred_ent = torch.argmax(torch.softmax(pred_ent_t, dim=-1), dim=-1)  # [-1, 128]
    pred_emo = torch.argmax(torch.softmax(pred_emo_t, dim=-1), dim=-1)  # [-1, 128]
    size = pred_ent_t.shape[0]
    # ent.resize(size, axis=0)
    # emo.resize(size, axis=0)
    # ent[0: size] = pred_ent_t.numpy()
    # emo[0: size] = pred_emo_t.numpy()
    assert  pred_ent_conf.shape[0] == pred_emo_conf.shape[0] == pred_ent.shape[0] == pred_emo.shape[0]
    ent_raw.resize(size, axis=0)
    emo_raw.resize(size, axis=0)
    ent.resize(size, axis=0)
    emo.resize(size, axis=0)
    ent_raw[...] = pred_ent_conf
    emo_raw[...] = pred_emo_conf
    ent[...] = pred_ent
    emo[...] = pred_emo
    f.close()
    fresult.close()
    print('over!')
