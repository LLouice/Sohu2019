#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 15:57
# @Author  : 邵明岩
# @File    : test_avg.py
# @Software: PyCharm

import os
import h5py
import torch

os.chdir("../.")
if __name__ == '__main__':
    fresult = h5py.File("../preds/pred_avg.h5", "w")
    ent_raw = fresult.create_dataset("ent_raw", shape=(0, 128, 3), maxshape=(None, 128, 3), compression="gzip")
    emo_raw = fresult.create_dataset("emo_raw", shape=(0, 128, 4), maxshape=(None, 128, 4), compression="gzip")
    # ent = fresult.create_dataset("ent", shape=(0, 128), maxshape=(None, 128), compression="gzip")
    # emo = fresult.create_dataset("emo", shape=(0, 128), maxshape=(None, 128), compression="gzip")

    f = h5py.File("../preds/pred_5cv.h5", "r")

    print('cv : {}'.format(1))
    pred_ent = f.get(f"cv1/ent_raw")[()]
    pred_emo = f.get("cv1/emo_raw")[()]

    for cv in range(2, 6):
        print('cv : {}'.format(cv))
        pred_ent_conf = f.get(f"cv{cv}/ent_raw")[()]
        pred_emo_conf = f.get(f"cv{cv}/emo_raw")[()]
        pred_ent = pred_ent + pred_ent_conf
        pred_emo = pred_emo + pred_emo_conf

    pred_ent = pred_ent / 5.0
    pred_emo = pred_emo / 5.0

    pred_ent_t = torch.from_numpy(pred_ent)
    pred_emo_t = torch.from_numpy(pred_emo)
    # pred_ent_t = torch.argmax(torch.softmax(pred_ent_t, dim=-1), dim=-1)  # [-1, 128]
    # pred_emo_t = torch.argmax(torch.softmax(pred_emo_t, dim=-1), dim=-1)  # [-1, 128]
    size = pred_ent_t.shape[0]
    # ent.resize(size, axis=0)
    # emo.resize(size, axis=0)
    # ent[0: size] = pred_ent_t.numpy()
    # emo[0: size] = pred_emo_t.numpy()
    ent_raw.resize(size, axis=0)
    emo_raw.resize(size, axis=0)
    ent_raw[0: size] = pred_ent
    emo_raw[0: size] = pred_emo
    f.close()
    fresult.close()
    print('over!')
