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
    parser.add_argument("--pred_dir",
                        default="pred_5cv.h5",
                        type=str, required=False)
    parser.add_argument("--pred_avg",
                        default="pred_avg.h5",
                        type=str, required=False)
    args = parser.parse_args()

    all_preds = os.listdir(args.pred_dir)
    fresult = h5py.File(f"../preds/{args.pred_avg}", "w")
    ent_raw = fresult.create_dataset("ent_raw", shape=(0, 256, 5), maxshape=(None, 256, 5), compression="gzip")
    emo_raw = fresult.create_dataset("emo_raw", shape=(0, 256, 4), maxshape=(None, 256, 4), compression="gzip")
    ent = fresult.create_dataset("ent", shape=(0, 256), maxshape=(None, 256), compression="gzip")
    emo = fresult.create_dataset("emo", shape=(0, 256), maxshape=(None, 256), compression="gzip")
    for i, name in enumerate(all_preds):
        f = h5py.File(os.path.join(args.pred_dir, f"{name}"), "r")
        if i == 0:
            pred_ent_conf = f.get("ent_raw")[()]
            pred_emo_conf = f.get("emo_raw")[()]
        else:
            pred_ent_cur = f.get(f"ent_raw")[()]
            pred_emo_cur = f.get(f"emo_raw")[()]
            pred_ent_conf += pred_ent_cur
            pred_emo_conf += pred_emo_cur
        f.close()

    pred_ent_conf /= len(all_preds)
    pred_emo_conf /= len(all_preds)

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
    fresult.close()
    print('over!')
