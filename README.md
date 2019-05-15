# 2019 搜狐校园大赛决赛冲刺 llouice
---

+ TODO 工作推进状况 包括已经完成和 TODO List
+ IDEA 各种天马行空的想法~

---
## 文件说明
- src  
   - data_raw
   > news.pkl (所有的新闻及其分字列表和标签列表的集合)  
   >> 0.1 的 lite,  full lite 对应的
            train.txt val.txt 和 lite_trn.txt lite_val.txt
   - data_title_trnval.py data_title_test.py
   > 从 train.txt val.txt 或者 lite_trn.txt lite_val.txt  test.txt 
   >> 生成 tensor 序列化文件 train.h5 lite.h5 和 ID2TOK.pkl(trn val test 所有的 ID2TOK)
   >>> full.h5 包括3个数据集 train val test
   >>> lite.h5 包括2个数据集 trina val
   
## 数据集生成流程
+ data_raw_trnval news.pkl -> lite_trn.txt lite_val.txt train.txt val.txt  
+ data_raw_test   news_test.pkl -> test.txt news_mapl.pkl  
+ data_title_trnval -> lite.h5  full.h5 ID2TOK.pkl  
+ data_title_test   -> update full.h5 ID2TOK.pkl

