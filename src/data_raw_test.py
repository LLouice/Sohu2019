from utils import load_data, data_dump
from sklearn.model_selection import train_test_split

'''
将新闻和标记写入txt文件
每一个句子空一行 没一则新闻空两行
'''
file = "../datasets/news_test.pkl"

data = load_data(file)
print("all data: ", len(data))

# 先分成 full lite 两大部分
_, lite = train_test_split(data, test_size=0.1)

def gen_news_map(data, mode="all"):
    D = {}
    for i,news in enumerate(data):
        ID = news["newsId"]
        D[i] = ID
    if mode == "all":
        data_dump(D, "../datasets/news_map.pkl")
    else:
        data_dump(D, "../datasets/lite_news_map.pkl")



def data2txt(data, mode="all"):
    print("cur data: ", len(data))
    if mode == "all":
        file = "../datasets/test.txt"
    else:
        file = "../datasets/lite_test.txt"
    f = open(file, "w")
    count = 0
    for news in data:
        title = news["title"]
        contents = news["content"]
        for t in title:
            line = t
            f.write(line)
            f.write("\n")
        f.write("\n")
        for content in contents:
            for t in content:
                line = t
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
data2txt(lite, "val")
gen_news_map(lite, "val")
# all
data2txt(data)
gen_news_map(data)
print("over")
