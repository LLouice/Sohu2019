from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import pickle

import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
import h5py
import gc
from collections import OrderedDict
import re
from utils import covert_mytokens_to_myids

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def data_dump(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
    logger.info("store {} successfully!".format(file))


def load_data(path):
    with open(path, "rb") as f:
        res = pickle.load(f)
    logger.info("load data from {} successfully!".format(path))
    return res


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, text_title=None, label_sent=None, label_title=None):
        '''
        :param guid:  example id, may be is newId
        :param text:  cur sentence text
        :param text_title:  the news title text
        :parm  label: label text [B-POS, I-POS, I-POS]
        在 fetures 中 label =>
        label_ent: entity label [0,1,2...10]
        label_emo: emotion label [0,1,2...7]  can transform from the label_ent
         情感的标记到底应该时什么 ？ [pos pos pos] or [B-pos, I-pos, I-pos]
        '''
        self.guid = guid
        self.text = text
        self.text_title = text_title
        self.label_text = label_sent
        self.label_title = label_title


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, ID, input_ids, myinput_ids, input_mask, segment_ids):
        self.ID = ID
        self.input_ids = input_ids
        self.myinput_ids = myinput_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_ent_ids = label_ent_ids
        # self.label_emo_ids = label_emo_ids


def readfile(filename):
    '''
    read file
    return format :
    '''
    logger.info("read file:{}....".format(filename))
    data = []

    def _read():
        f = open(filename, encoding="utf-8")
        title = None
        sentence = []
        label_sent = []
        label_title = []
        count = 0
        ID = 0
        has_title = False
        for line in f:
            if len(line) == 0 or line[0] == "\n":
                if len(sentence) > 0:
                    if not has_title:
                        # the title
                        title = sentence
                        has_title = True
                    data.append((ID, sentence, title, label_sent, label_title))
                    # refresh
                    sentence = []
                    label_sent = []
                else:
                    # 第二个空行 sentence 已空
                    has_title = False
                    ID += 1
                    count += 1
                continue
            sentence.append(line[:-1])

            if count % 100 == 0:
                print(count)

        # 防止因最后一行非空行而没加入最后一个
        if len(sentence) > 0:
            if not has_title:
                # the title
                title = sentence
                label_title = label_sent
                has_title = True
            data.append((ID, sentence, title, label_sent, label_title))
            # refresh
            # sentence = []
            # label_sent = []
        f.close()

    _read()
    # 最后一句查看
    print("The Last Sentence....")
    print(data[-1][0])
    print(data[-1][1])
    print("sentence num: ", len(data))
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
            self._read_tsv(os.path.join(data_dir, "train_full.txt")), "train")
        # self._read_tsv(os.path.join(data_dir, "train_full.txt")), "train")
        # self._read_tsv(os.path.join(data_dir, "train_lite.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_full.txt")), "dev")
        # self._read_tsv(os.path.join(data_dir, "dev_full.txt")), "dev")
        # self._read_tsv(os.path.join(data_dir, "dev_lite.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-POS", "I-POS", "B-NEG", "I-NEG", "B-NORM", "I-NORM", "X", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (ID, sentence, title, label_sent, label_title) in enumerate(lines):
            guid = ID
            text = ' '.join(sentence)
            text_title = ' '.join(title)
            examples.append(InputExample(guid=guid, text=text, text_title=text_title, label_sent=label_sent,
                                         label_title=label_title))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, mytokens_a, mytokens_b, labels_A, labels_B, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            mytokens_a.pop()
            # labels_A.pop()
        else:
            tokens_b.pop()
            mytokens_b.pop()
            # labels_B.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, TOK2ID):
    """Loads a data file into a list of `InputBatch`s."""

    logger.info("gen features...")
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    label_map["PAD"] = 0
    label_map_reversed = {v: k for k, v in label_map.items()}

    features = []
    logger.info("prepare features.....")
    count = 0
    for (ex_index, example) in enumerate(examples):
        textlist_A = example.text.split(' ')
        labellist_A = example.label_text
        textlist_B = example.text_title.split(' ')
        labellist_B = example.label_title
        tokens_A = []
        mytokens_A = []
        labels_A = []
        tokens_B = []
        mytokens_B = []
        labels_B = []

        # if count == 529320:
        #     pdb.set_trace()
        ####################### tokenize ##########################
        def _tokenize(textlist, labellist, tokens, mytokens, labels):
            '''
            # 重新检查textlist 除去多个汉字并联
            new_textlist = []
            for word in textlist:
                if len(word) == 1:
                    new_textlist.append(word)
                else:
                    new_word = seg_char(word)
                    if len(new_word) > 1:
                        print("!wrong seg:", new_word)
                        new_textlist.extend(new_word)
                    else:
                        new_textlist.append(word)
            textlist = new_textlist
            '''
            SAMESPLIT = False
            for i, word in enumerate(textlist):
                SAMESPLIT = False
                token = tokenizer.tokenize(word)
                mytoken = token.copy()
                # debug [UNK]
                # 检查 [UNK], 编入 自定义 字典
                if "[UNK]" in token:
                    assert "##" not in word
                    # ##
                    sharp_index = [i for i, tok in enumerate(token) if tok.startswith("##")]
                    for si in sharp_index:
                        no_sharp_tok = mytoken[si][2:]
                        if no_sharp_tok not in TOK2ID:
                            TOK2ID[no_sharp_tok] = len(TOK2ID)
                        mytoken[si] = no_sharp_tok

                    unks_index = [i for i, tok in enumerate(mytoken) if tok == "[UNK]"]
                    unks_index.reverse()
                    not_unks = [tok for tok in mytoken if tok != "[UNK]"]
                    if not_unks:
                        not_unks = [re.escape(nu) for nu in not_unks]
                        if not (len(set(not_unks)) == 1 and len(not_unks) > 1):  # 全是同一个字符，会无限循环
                            pattern = "(.*)".join(not_unks)
                            pattern = re.compile("(.*)" + pattern + "(.*)")
                        else:
                            SAMESPLIT = True
                            print(word)
                            f = "([^{}]*)".format(not_unks[0])
                            pattern = f.join(not_unks)
                            pattern = re.compile(f + pattern + f)
                        for res in pattern.findall(word):
                            for r in res:
                                if len(r) > 0 and r != "\u202a":  # whitespace!!
                                    if r not in TOK2ID:
                                        TOK2ID[r] = len(TOK2ID)
                                    mytoken[unks_index[-1]] = r
                                    unks_index.pop()

                    else:
                        # 理论上应该是单个 如 4G  但是很奇怪分字有问题这里 如不了
                        # assert len(token) == 1
                        if len(token) == 1:
                            if word not in TOK2ID:
                                TOK2ID[word] = len(TOK2ID)
                            mytoken[0] = word
                        else:
                            # ?不处理
                            # '不了'
                            mytoken = list(word)
                            print("BUG!:", word, token, mytoken)
                            for mytok in mytoken:
                                if mytok not in TOK2ID:
                                    TOK2ID[mytok] = len(TOK2ID)
                if SAMESPLIT:
                    print(word, token, mytoken, sep="\t")
                tokens.extend(token)
                mytokens.extend(mytoken)
                assert len(tokens) == len(mytokens)
                '''
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                    else:
                        labels.append("X")
                '''

        _tokenize(textlist_A, labellist_A, tokens_A, mytokens_A, labels_A)
        _tokenize(textlist_B, labellist_B, tokens_B, mytokens_B, labels_B)
        ######################################################

        ################### 处理tokenize后新增的位置 以及 合并 截断 #####################
        # [CLS] A [SEP] B [SEP]
        _truncate_seq_pair(tokens_A, tokens_B, mytokens_A, mytokens_B, labels_A, labels_B, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_A + ["[SEP]"] + tokens_B + ["[SEP]"]
        mytokens = ["[CLS]"] + mytokens_A + ["[SEP]"] + mytokens_B + ["[SEP]"]
        assert len(tokens) == len(mytokens)
        # labels = ["[CLS]"] + labels_A + ["[SEP]"] + labels_B + ["[SEP]"]
        segment_ids = [0] * (len(tokens_A) + 2) + [1] * (len(tokens_B) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        myinput_ids = covert_mytokens_to_myids(TOK2ID, mytokens)
        input_mask = [1] * len(input_ids)
        # label_ids = [label_map[lbl] for lbl in labels]
        # ------------------------------- PAD -----------------------------------
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        myinput_ids += padding
        input_mask += padding
        segment_ids += padding
        # label_ids += padding
        # -----------------------------------------------------------------------
        #############################分为两个label_ids###########################
        # emo label id 还有待实验 B-POS I-POS  or POS POS
        # PAD O B-POS I-POS B-NEG I-NEG B-NORM I-NORM X [CLS] [SEP]
        #  0  1   2    3     4     5      6     7     8   9    10
        # PAD O   B    I     X   [CLS]  [SEP]

        # O   O  POS  POS   NEG   NEG    NORM   NORM  O   O    O
        # 0       1          2            3

        # 类最少的模式
        # O   B    I
        # 0   1    2     3
        # 0  POS NEG  NORM

        # label_emo_ids = label_ids
        # 这里可能有两种emo标法 只标 B 或者 BI都标, 这里使用后者,即上面所示
        '''
        L = np.array(label_ids)
        L[L == 10] = 0
        L[L == 9] = 0
        L[L == 8] = 0
        L[L == 1] = 0

        L[L == 2] = 1
        L[L == 3] = 1
        L[L == 4] = 2
        L[L == 5] = 2
        L[L == 6] = 3
        L[L == 7] = 3
        label_emo_ids = L.tolist()
        # ------------------
        L = np.array(label_ids)
        # 带 PAD X [CLS] 标法
        # L[L==4]=2
        # L[L==6]=2
        # L[L==5]=3
        # L[L==7]=3
        # L[L==8]=4
        # L[L==9]=5
        # L[L==10]=
        
        #----
        L[L == 1] = 0
        L[L == 2] = 1
        L[L == 4] = 1
        L[L == 6] = 1
        L[L == 3] = 2
        L[L == 5] = 2
        L[L == 7] = 2
        L[L == 8] = 0
        L[L == 9] = 0
        L[L == 10] = 0
        label_ent_ids = L.tolist()
        '''

        #######################################################################

        assert len(input_ids) == len(myinput_ids)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_ent_ids) == max_seq_length
        # assert len(label_emo_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info(
            #     "label: %s" % " ".join([label_map_reversed[x] for x in label_ids]))
        features.append(
            InputFeatures(ID=example.guid,
                          input_ids=input_ids,
                          myinput_ids=myinput_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
        # label_ent_ids=label_ent_ids,
        # label_emo_ids=label_emo_ids))
        count += 1
        if count % 10000 == 0:
            print("gen example {} feature".format(count))

    logger.info("finish features gen")

    return features


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_token_model",
                        default="/home/lzk/llouice/.pytorch_pretrained_bert/8a0c070123c1f794c42a29c6904beb7c1b8715741e235bee04aca2c7636fc83f.9b42061518a39ca00b8b52059fd2bede8daa613f8a8671500e518a8c29de8c00",
                        type=str, required=False)
    parser.add_argument("--task_name",
                        default="ner",
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--parent_dir",
                        default=".",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    logger.info("data_dir: {}".format(args.data_dir))
    logger.info("parent_dir: {}".format(args.parent_dir))

    processors = {"ner": NerProcessor}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_token_model, do_lower_case=args.do_lower_case)

    ID2TOK = tokenizer.ids_to_tokens
    TOK2ID = OrderedDict((tok, id) for id, tok in ID2TOK.items())


    logger.info("in do_test now")
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length, tokenizer, TOK2ID)
    logger.info("***** Running on Test *****")
    logger.info("  Num examples = %d", len(test_examples))
    # logger.info("  Batch size = %d", args.test_batch_size)
    all_IDs = torch.tensor([f.ID for f in test_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_myinput_ids = torch.tensor([f.myinput_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)

    f = h5py.File("../datasets/data_test.h5", "w", libver="latest")
    f.create_dataset("test/IDs", data=all_IDs, compression="gzip")
    f.create_dataset("test/input_ids", data=all_input_ids, compression="gzip")
    f.create_dataset("test/myinput_ids", data=all_myinput_ids, compression="gzip")
    f.create_dataset("test/input_mask", data=all_input_mask, compression="gzip")
    del all_input_ids, all_input_mask
    gc.collect()
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    f.create_dataset("test/segment_ids", data=all_segment_ids, compression="gzip")
    f.close()
    # data_dump(TOK2ID, "../datasets/TOK2ID_test.pkl")
    ID2TOK = OrderedDict((id, tok) for tok, id in TOK2ID.items())
    data_dump(ID2TOK, "../datasets/ID2TOK_test.pkl")
    print("save to h5 over!")



if __name__ == "__main__":
    main()

    '''
    --data_dir=/home/lzk/llouice/BERT/souhu/BERT-NER-master/full/data_full
    --parent_dir=/home/lzk/llouice/BERT/souhu/BERT-NER-master/full
    '''
