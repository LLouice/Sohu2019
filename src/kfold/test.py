import os

os.chdir("../.")
import h5py
from argparse import ArgumentParser
import torch
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from models import NetX3
from utils import load_data


def get_all_data():
    print("get all data...........")
    # -------------------------------read from h5-------------------------
    if not args.lite:
        f = h5py.File("../datasets/full.h5")
    else:
        f = h5py.File("../datasets/lite.h5")
    input_ids_trn = torch.from_numpy(f["train/input_ids"][()])
    myinput_ids_trn = torch.from_numpy(f["train/myinput_ids"][()])
    input_mask_trn = torch.from_numpy(f["train/input_mask"][()])
    segment_ids_trn = torch.from_numpy(f["train/segment_ids"][()])
    label_ent_ids_trn = torch.from_numpy(f["train/label_ent_ids"][()])
    label_emo_ids_trn = torch.from_numpy(f["train/label_emo_ids"][()])
    assert input_ids_trn.size() == segment_ids_trn.size() == label_ent_ids_trn.size() == label_emo_ids_trn.size() == myinput_ids_trn.size()

    input_ids_val = torch.from_numpy(f["val/input_ids"][()])
    myinput_ids_val = torch.from_numpy(f["val/myinput_ids"][()])
    input_mask_val = torch.from_numpy(f["val/input_mask"][()])
    segment_ids_val = torch.from_numpy(f["val/segment_ids"][()])
    label_ent_ids_val = torch.from_numpy(f["val/label_ent_ids"][()])
    label_emo_ids_val = torch.from_numpy(f["val/label_emo_ids"][()])
    assert input_ids_val.size() == segment_ids_val.size() == label_ent_ids_val.size() == label_emo_ids_val.size() == myinput_ids_val.size()
    f.close()
    print("read h5 over!")
    input_ids = torch.cat([input_ids_trn, input_ids_val], dim=0)
    myinput_ids = torch.cat([myinput_ids_trn, myinput_ids_val], dim=0)
    input_mask = torch.cat([input_mask_trn, input_mask_val], dim=0)
    segment_ids = torch.cat([segment_ids_trn, segment_ids_val], dim=0)
    label_ent_ids = torch.cat([label_ent_ids_trn, label_ent_ids_val], dim=0)
    label_emo_ids = torch.cat([label_emo_ids_trn, label_emo_ids_val], dim=0)
    dataset = TensorDataset(input_ids, myinput_ids, input_mask, segment_ids, label_ent_ids, label_emo_ids)

    return dataset


def get_val_loader(dataset, cv):
    print(f"get dataloader {cv}")
    # 从 index 中取出 trn_dataset val_dataset
    index_file = "kfold/5cv_indexs_{}".format(cv)
    if os.path.exists(index_file):
        _, val_index = load_data(index_file)
        val_dataset = [dataset[idx] for idx in val_index]
    else:
        print("Not find index file!")
        exit(0)
    # ---------------------------------------------------------------------
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.val_batch_size,
                                num_workers=args.nw, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.val_batch_size,
    #                             pin_memory=True)
    print("get date loader over!")
    return val_dataloader, len(val_dataset)


def get_test_dataloader():
    print("get test dataloader...........")
    # -------------------------------read from h5-------------------------
    f = h5py.File("../datasets/full.h5", "r")
    input_ids = torch.from_numpy(f["test/input_ids"][()])
    input_mask = torch.from_numpy(f["test/input_mask"][()])
    segment_ids = torch.from_numpy(f["test/segment_ids"][()])
    assert input_ids.size() == segment_ids.size() == input_mask.size()
    print("test dataset num: ", input_ids.size(0))
    test_dataset = TensorDataset(input_ids, input_mask, segment_ids)
    f.close()
    print("read h5 over!")
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.test_batch_size,
                                 num_workers=4)
    print("get date loader over!")
    return test_dataloader, len(test_dataset)


def run(test_dataloader, cv):
    ################################ Model Config ###################################
    num_labels_emo = 4
    num_labels_ent = 3
    model = NetX3.from_pretrained(args.bert_model,
                                  cache_dir="",
                                  num_labels_ent=num_labels_ent,
                                  num_labels_emo=num_labels_emo,
                                  dp=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)

    # ------------------------------ load model from file -------------------------
    model_file = f"../ckps/cv/cv{cv}.pth"
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print("load checkpoint: {} successfully!".format(model_file))

    # -----------------------------------------------------------------------------

    def test(engine, batch):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch

        with torch.no_grad():
            logits_ent, logits_emo = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        return logits_ent, logits_emo

    tester = Engine(test)

    pbar_test = ProgressBar(persist=True)
    pbar_test.attach(tester)

    # ++++++++++++++++++++++++++++++++++ Test +++++++++++++++++++++++++++++++++
    if cv == 1:
        f = h5py.File(f"../preds/pred_5cv.h5", "w")
    else:
        f = h5py.File(f"../preds/pred_5cv.h5", "r+")
    ent_raw = f.create_dataset(f"cv{cv}/ent_raw", shape=(0, 128, 3), maxshape=(None, 128, 3), compression="gzip")
    emo_raw = f.create_dataset(f"cv{cv}/emo_raw", shape=(0, 128, 4), maxshape=(None, 128, 4), compression="gzip")

    # ent = f.create_dataset(f"cv{cv}/ent", shape=(0, 128), maxshape=(None, 128), compression="gzip")
    # emo = f.create_dataset(f"cv{cv}/emo", shape=(0, 128), maxshape=(None, 128), compression="gzip")
    # if cv == 1:

    @tester.on(Events.ITERATION_COMPLETED)
    def get_test_pred(engine):
        # cur_iter = engine.state.iteration
        batch_size = engine.state.batch[0].size(0)
        pred_ent_raw, pred_emo_raw = engine.state.output

        # pred_ent = torch.argmax(torch.softmax(pred_ent_raw, dim=-1), dim=-1)  # [-1, 128]
        def add_io():
            # pred_emo = torch.argmax(torch.softmax(pred_emo_raw, dim=-1), dim=-1)  # [-1, 128]
            old_size = ent_raw.shape[0]
            # ent.resize(old_size + batch_size, axis=0)
            # emo.resize(old_size + batch_size, axis=0)
            # ent[old_size: old_size + batch_size] = pred_ent.cpu()
            # emo[old_size: old_size + batch_size] = pred_emo.cpu()
            ent_raw.resize(old_size + batch_size, axis=0)
            emo_raw.resize(old_size + batch_size, axis=0)
            ent_raw[old_size: old_size + batch_size] = pred_ent_raw.cpu()
            emo_raw[old_size: old_size + batch_size] = pred_emo_raw.cpu()
            # if cv == 1:

        def add_mem():
            if engine.state.metrics.get("preds_ent") is None:
                engine.state.metrics["preds_ent"] = []
                engine.state.metrics["preds_emo"] = []
            else:
                engine.state.metrics["preds_ent"].append(pred_ent_raw.cpu())
                engine.state.metrics["preds_emo"].append(pred_emo_raw.cpu())

        add_mem()

    @tester.on(Events.EPOCH_COMPLETED)
    def save_and_close(engine):
        if engine.state.metrics.get("preds_ent") is not None:
            preds_ent = torch.cat(engine.state.metrics["preds_ent"], dim=0)
            preds_emo = torch.cat(engine.state.metrics["preds_emo"], dim=0)
            assert preds_ent.size(0) == preds_emo.size(0)
            ent_raw.resize(preds_ent.size(0), axis=0)
            emo_raw.resize(preds_emo.size(0), axis=0)
            ent_raw[...] = preds_ent
            emo_raw[...] = preds_emo
            print("pred size: ", ent_raw.shape)
        f.close()
        print("test over")

    tester.run(test_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bert_model", default="../bert_pretrained/bert-base-chinese",
                        type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for test (default: 1000)')
    parser.add_argument("--best_model",
                        default="best_model.pt",
                        type=str, required=False)
    parser.add_argument("--pred",
                        default="pred_new.h5",
                        type=str, required=False)
    parser.add_argument("--raw",
                        action="store_true",
                        help="是否存储置信度")

    args = parser.parse_args()

    # 5 fold
    # dataset = get_all_data()
    test_dataloader, test_size = get_test_dataloader()
    for cv in range(1, 6):
        run(test_dataloader, cv)
