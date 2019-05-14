from utils import load_data
from sklearn.model_selection import train_test_split

'''
将新闻和标记写入txt文件
每一个句子空一行 没一则新闻空两行
'''
file = "../datasets/news.pkl"

data = load_data(file)
print(len(data))

# 分出lite (40000+1305) * 0.04 = 1653-> 1322:331
_, lite = train_test_split(data, test_size=0.04)

# 再各自分成 trn 和 val
full_trn, full_val = train_test_split(data, test_size=0.2)
lite_trn, lite_val = train_test_split(lite, test_size=0.2)


def data2txt(data, mode="train", size="lite"):
    if size == "lite":
        trn_txt = "../datasets/lite_trn.txt"
        val_txt = "../datasets/lite_val.txt"
    else:
        trn_txt = "../datasets/train.txt"
        val_txt = "../datasets/val.txt"
    if mode == "train":
        file = trn_txt
    else:
        file = val_txt
    f = open(file, "w")
    count = 0
    for news in data:
        title = news["title"][0]
        title_O = news["title"][1]
        contents = news["content"]
        for (t, tO) in zip(title, title_O):
            line = " ".join((t, tO))
            f.write(line)
            f.write("\n")
        f.write("\n")
        for content in contents:
            for t, tO in zip(content[0], content[1]):
                line = " ".join((t, tO))
                f.write(line)
                f.write("\n")
            f.write("\n")
        # 下一新闻
        f.write("\n")
        count += 1
        # if count > 2:
        #     break
    f.close()
    print(count)


# lite
data2txt(lite_trn)
data2txt(lite_val, "val")
# full
data2txt(full_trn, size="full")
data2txt(full_val, mode="val", size="full")
print("over")
