from utils import load_data, data_dump
from sklearn.model_selection import train_test_split

'''
将新闻和标记写入txt文件
每一个句子空一行 每一则新闻空两行
'''
file = "../datasets/news_test.pkl"
data = load_data(file)
print("all data: ", len(data))


def gen_news_map(data):
    D = {}
    for i, news in enumerate(data):
        ID = news["newsId"]
        D[i] = ID
    data_dump(D, "../datasets/news_map.pkl")


def data2txt(data):
    print("cur data: ", len(data))
    file = "../datasets/test.txt"
    f = open(file, "w")
    count = 0
    for news in data:
        title = news["title"]
        contents = news["content"]
        for t in title:
            line = t.strip()
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            if len(line) > 0:
                f.write(line)
                f.write("\n")
        f.write("\n")
        for content in contents:
            # 避免空content
            if len(content)>0:
                for t in content:
                    line = t.strip()
                    line = line.replace("\n", "")
                    line = line.replace("\r", "")
                    if len(line)>0:
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


# all
data2txt(data)
gen_news_map(data)
print("over")
