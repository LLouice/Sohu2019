from sklearn.model_selection import KFold
import h5py
from utils import data_dump
kfold = KFold(n_splits=5)

full = "../../datasets/full.h5"
f = h5py.File(full, "r")
trn_size = f.get("train")["input_ids"].shape[0]
val_size = f.get("val")["input_ids"].shape[0]
all_size = trn_size + val_size

cv = 0
for train_indexs, dev_indexs in kfold.split([1]*all_size):
     cv = cv + 1
     index_t_d = (train_indexs, dev_indexs)
     index_file = '5cv_indexs_{}'.format(cv)
     data_dump(index_t_d, index_file)