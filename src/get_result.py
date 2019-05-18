import argparse
import h5py
import numpy as np
import re
from collections import defaultdict, Counter
from utils import load_data, covert_myids_to_mytokens
import time

EMOS_MAP = {"0": "OTHER", "1": "POS", "2": "NEG", "3": "NORM"}
ID2TOK = load_data("../datasets/ID2TOK.pkl")
pattern = re.compile("1[2]*")  # 贪婪匹配
######################################################################################


def _get_res(cur_input_ids, cur_myinput_ids, cur_pred_ent, cur_pred_ent_conf, cur_pred_emo, cur_pred_emo_conf, S):
    '''
    :param cur_myinput_ids:
    :param cur_pred_ent:
    :param pattern_ent:
    :param cur_pred_emo:
    :param ENTS:
    :return:
    '''

    for r in pattern.finditer(cur_pred_ent):
        i, j = r.span()[0], r.span()[1]
        res = "".join(covert_myids_to_mytokens(ID2TOK, cur_myinput_ids[i:j]))
        res = res.replace("##", "")
        if "[PAD]" in res or "X" in res or "[CLS]" in res or "[SEP]" in res:
            continue
        if "[UNK]" in res:
            print("UNK: ", res)
            continue
        if len(res) == 1:
            continue
        emos = cur_pred_emo[i:j]
        # conf = cur_pred_ent_conf[i:j]
        # emo = Counter(emos).most_common(1)[0][0]
        S[res].extend(emos)


def _get_ent(S):
    R = defaultdict()
    for ent, emos in S.items():
        c = Counter(emos).most_common()
        if not c[0][0] == '0':
            R[ent] = c[0][0]
        elif len(c) > 1:
            R[ent] = c[1][0]
        else:
            # not certain
            pass
    return R


#################################### Every News ###########################################
def main():
    # ------------------------------------- args ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--res",
                        default="result_new.txt",
                        type=str,
                        required=False,
                        help="result file")

    parser.add_argument("--pred",
                        default="pred_new.h5",
                        type=str, required=False)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------
    # ------------------------------- output file ----------------------------------
    result_file = f"../results/{args.res}"
    pred_file = f"../preds/{args.pred}"
    f_result = open(result_file, "wt")
    f_pred = h5py.File(pred_file, "r")
    # ------------------------------------------------------------------------------
    # -------------------------------- test data -----------------------------------
    NEWS_MAP = load_data("../datasets/news_map.pkl")
    test_file = "../datasets/full.h5"

    f_test = h5py.File(test_file, "r")
    IDs = f_test.get("test/IDs")[()]
    # input_ids = f_test.get("test/input_ids")[()]
    myinput_ids = f_test.get("test/myinput_ids")[()]
    input_mask = f_test.get("test/input_mask")[()]
    segement_ids = f_test.get("test/segment_ids")[()]
    unique_IDs = np.unique(IDs)
    assert np.max(unique_IDs) == 79999
    # ------------------------------------------------------------------------------
    # -------------------------------- pred data -----------------------------------
    pred_ent = f_pred.get("ent")[()]
    pred_emo = f_pred.get("emo")[()]
    assert IDs.shape[0] == pred_ent.shape[0] == pred_emo.shape[0]
    # pred_ent_conf = f_pred.get("ent_raw")
    # pred_emo_conf = f_pred.get("emo_raw")
    # ------------------------------------------------------------------------------

    idx1 = 0
    time0 = time.time()
    time1 = time.time()
    for id in range(80000):
        num = np.sum(IDs == id)
        idx2 = idx1 + num
        # [bs, 128, 3]
        cur_pred_ent = pred_ent[idx1:idx2, :].reshape(1, -1).squeeze().astype(np.int)  # (6912,)
        # cur_pred_ent_conf = pred_ent_conf[idx1:idx2, :, :]
        # cur_pred_ent_conf = np.max(torch.softmax(torch.from_numpy(cur_pred_ent_conf), dim=-1).numpy(), axis=-1).reshape(1,
        #                                                                                                                 -1).squeeze()

        cur_pred_emo = pred_emo[idx1:idx2, :].reshape(1, -1).squeeze().astype(np.int)  # (6912,)
        # cur_pred_emo_conf = pred_emo_conf[idx1:idx2, :, :]
        # cur_pred_emo_conf = np.max(torch.softmax(torch.from_numpy(cur_pred_emo_conf), dim=-1).numpy(), axis=-1).reshape(1,
        #                                                                                                                 -1).squeeze()

        # mask
        # cur_pred_emo = cur_pred_emo[cur_pred_ent == 1]
        # 原文
        # cur_input_ids = input_ids[idx1:idx2, :].reshape(1, -1).squeeze()
        cur_myinput_ids = myinput_ids[idx1:idx2, :].reshape(1, -1).squeeze()
        cur_input_mask = input_mask[idx1:idx2, :].reshape(1, -1).squeeze()
        cur_segment_ids = segement_ids[idx1:idx2, :].reshape(1, -1).squeeze()
        ################################# 2 mask: input mask + segment mask #####################
        # '''
        active_mask = cur_input_mask == 1
        active_seg = cur_segment_ids[active_mask]
        active_seg = active_seg == 0
        # cur_input_ids = cur_input_ids[active_mask][active_seg]
        cur_myinput_ids = cur_myinput_ids[active_mask][active_seg]
        cur_pred_ent = cur_pred_ent[active_mask][active_seg]
        # cur_pred_ent_conf = cur_pred_ent_conf[active_mask][active_seg]
        cur_pred_emo = cur_pred_emo[active_mask][active_seg]
        # cur_pred_emo_conf = cur_pred_emo_conf[active_mask][active_seg]
        # '''
        # cur_pred_emo = cur_pred_emo[cur_pred_ent == 2]
        # cur_pred_emo_conf = cur_pred_emo_conf[cure_pred_ent == 2]
        #########################################################################################
        # 10 -> 0
        # cur_pred_emo[cur_pred_emo == 10] = 0
        idx1 = idx2
        cur_pred_ent = "".join([str(cur_pred_ent[i]) for i in range(cur_pred_ent.shape[-1])])
        cur_pred_emo = "".join([str(cur_pred_emo[i]) for i in range(cur_pred_emo.shape[-1])])

        ENTS_LIST = defaultdict(list)
        # _get_res(cur_input_ids, cur_myinput_ids, cur_pred_ent, cur_pred_ent_conf, cur_pred_emo, cur_pred_emo_conf, ENTS_LIST)
        # _get_res(cur_input_ids, cur_myinput_ids, cur_pred_ent, "", cur_pred_emo, "", ENTS_LIST)
        _get_res("", cur_myinput_ids, cur_pred_ent, "", cur_pred_emo, "", ENTS_LIST)

        #################################################################

        ################### !!综合!! ########################################
        R = _get_ent(ENTS_LIST)
        # result1: 舍弃单个汉字 取交集作为提交版本  TODO 空集策略 标点问题{青山 青山.}

        ######################### write to file ############################
        newsID = NEWS_MAP[id]
        line = "{}\t{}\t{}"
        ents = []
        emos = []
        for ent, emo in R.items():
            ent = ent.replace(",", "")
            ents.append(ent)
            emos.append(EMOS_MAP[emo])

        assert len(ents) == len(emos)
        ents = ",".join(ents)
        emos = ",".join(emos)
        answer = line.format(newsID, ents, emos)
        answer = answer.replace("\r", "")
        answer = answer.replace("\n", "")
        answer = answer.replace("\u200B", "")

        # print(answer)
        f_result.write(answer + "\n")
        # if (id + 1) %  100 == 0:
        #     print("=" * 10, id, "=" * 10)
        #     break
        if id % 1000 == 0:
            time2 = time.time()
            print("=" * 10, "time used: [{}]  {}".format(time2 - time1, id), "=" * 10)
            time1 = time2

    #################################################################################

    f_result.close()
    f_pred.close()
    f_test.close()

    print("*" * 10, "time used: [{} min]".format((time.time() - time0) / 60), "*" * 10)
    print("over!!!")


if __name__ == '__main__':
    main()
