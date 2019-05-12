import pickle
import json
import re
import os
import h5py
from collections import OrderedDict


################## pickle ################
def data_dump(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
    print("store {} successfully!".format(file))


def load_data(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    print("load data from {} successfully!".format(path))
    return res


################## News ################
class News2(object):
    def __init__(self, idx, newsId, title, coreEE):
        self.newsID = newsId
        self.title = title
        self.sents = []
        self.idx = idx
        self.coreEE = coreEE

    def add_sent(self, sent):
        self.sents.append(sent)


def gen_news(file="/home/lzk/llouice/BERT/souhu/BERT-NER-master/att/coreEntityEmotion_example.txt"):
    with open(file, "rt", encoding="utf-8") as f:
        for line in f:
            news = json.loads(line)
            yield news

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w.strip() for w in chars if len(w.strip()) > 0]
    new_chars = []
    for c in chars:
        if len(c) > 1:
            punctuation = '！？｡，。＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’“”„‟…‧'
            punctuation = '|'.join([p for p in punctuation])
            pattern = re.compile(' |({})'.format(punctuation))
            cs = pattern.split(c)
            for w in cs:
                if w and len(w.strip()) > 0:
                    new_chars.append(w.strip())
        else:
            new_chars.append(c)
    return new_chars

def get_sentences(content):
    sentences = re.split(r'(。|！|\!|？|\?)', content)  # 保留分割符
    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        sent = sent.strip()
        sent.replace("\n", "")
        sent.replace("\r", "")
        sent.replace("\u200B", "")
        new_sents.append(sent.strip())
    return new_sents


########################################## HDF5 ##################################################
def save2hdf5(filename):
    file_path = os.path.join("/home/lzk/llouice/BERT/souhu/datasets", filename)
    f = h5py.File(file_path, "a",  libver="latest")


########################################## my token ################################################
def covert_mytokens_to_myids(TOK2ID, mytokens):
    myids = []
    for token in mytokens:
        myids.append(TOK2ID[token])
    return myids

def covert_myids_to_mytokens(ID2TOK, myids):
    mytokens = []
    for i in myids:
        mytokens.append(ID2TOK[i])
    return mytokens
