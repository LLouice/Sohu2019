from utils import load_data
from sklearn.model_selection import train_test_split

'''
将新闻和标记写入txt文件
每一个句子空一行 没一则新闻空两行
'''
# file = "/home/lzk/smy/sohu/data_process/new_data2/train_ner_has_emotion.pkl"
file = "../datasets/train_ner_has_emotion.pkl"

data = load_data(file)
print(len(data))

trn_data,val_data = train_test_split(data, test_size=0.2)


def data2txt(data, mode="train"):
    if mode == "train":
        file = "../datasets/train_full.txt"
    else:
        file = "../datasets/dev_full.txt"
    f = open(file, "w")
    count = 0
    for news in data:
        title = news["title"][0]
        title_O = news["title"][1]
        contents = news["content"]
        for (t, tO) in zip(title, title_O):
            # if len(t) > 1:
            #     new_t = seg_char(t)
            #     if len(new_t)>1:
            #         pass
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
        count+=1
        # if count > 2:
        #     break
    f.close()
    print(count)
# data2txt(data[:34000])
# data2txt(data[34000:], "val")
data2txt(trn_data)
data2txt(val_data, "val")
print("over")


