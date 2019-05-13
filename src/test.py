import os
import h5py
from argparse import ArgumentParser
import torch
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from models import NetX3


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


def run():
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
    model_file = os.path.join("../ckps", args.best_model)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        print("load checkpoint: {} successfully!".format(model_file))
    # -----------------------------------------------------------------------------

    test_dataloader, test_size = get_test_dataloader()

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
    f = h5py.File(f"../preds/{args.pred}", "w")
    if args.raw:
        ent_raw = f.create_dataset("ent_raw", shape=(0, 128, 3), maxshape=(None, 128, 3), compression="gzip")
        emo_raw = f.create_dataset("emo_raw", shape=(0, 128, 4), maxshape=(None, 128, 4), compression="gzip")
    ent = f.create_dataset("ent", shape=(0, 128), maxshape=(None, 128), compression="gzip")
    emo = f.create_dataset("emo", shape=(0, 128), maxshape=(None, 128), compression="gzip")

    @tester.on(Events.ITERATION_COMPLETED)
    def get_test_pred(engine):
        # cur_iter = engine.state.iteration
        batch_size = engine.state.batch[0].size(0)
        pred_ent_raw, pred_emo_raw = engine.state.output
        pred_ent = torch.argmax(torch.softmax(pred_ent_raw, dim=-1), dim=-1)  # [-1, 128]
        pred_emo = torch.argmax(torch.softmax(pred_emo_raw, dim=-1), dim=-1)  # [-1, 128]
        old_size = ent.shape[0]
        ent.resize(old_size + batch_size, axis=0)
        emo.resize(old_size + batch_size, axis=0)
        ent[old_size: old_size + batch_size] = pred_ent.cpu()
        emo[old_size: old_size + batch_size] = pred_emo.cpu()
        if args.raw:
            ent_raw.resize(old_size + batch_size, axis=0)
            emo_raw.resize(old_size + batch_size, axis=0)
            ent_raw[old_size: old_size + batch_size] = pred_ent_raw.cpu()
            emo_raw[old_size: old_size + batch_size] = pred_emo_raw.cpu()

    pbar_test = ProgressBar(persist=True)
    pbar_test.attach(tester)

    @tester.on(Events.EPOCH_COMPLETED)
    def close_h5(engine):
        print("test over")
        f.close()

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
                        type=str, required=True)
    parser.add_argument("--pred",
                        default="pred_new.h5",
                        type=str, required=False)
    parser.add_argument("--raw",
                        action="store_true",
                        help="是否存储置信度")



    args = parser.parse_args()

    run()
