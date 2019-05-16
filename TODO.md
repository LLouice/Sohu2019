# 数据集构造
- [x] 混合 example + train -> 所有的新闻 news.pkl
- [x] 抽取部分 -> lite_trn.txt lite_val.txt lite_test.txt -> lite.h5 包括 train val test  

~~lite.h5 -> model -> trn val -> best_model.pt -> get_result -> score.py~~  
~~- [ ] score.py~~
- [ ] BIESO 等其它形式的数据集构造 label id 转换部分
    > python data_title_trnval.py --label_method="BIESO"
# 预测！
- [ ] 取前三
- [ ] 5 fold
# 训练调参
- [ ] lite 调参
- [ ] ent 和 emo 单独训练观察收敛情况
- [ ] batch 级别评估
    > 由于在中途就会过拟合 所以尝试取 n 个 batch(如 20) 评估一次

## ent loss 和 emo loss 的处理
- [x] emo loss 权重组
    > 先给 emo 小权重 让 ent loss 先收敛 再加大 emo loss 权重 joint 一起收敛 

# inference
- [x] 初赛模型 -> result.txt
- [x] cv 模型 -> result.txt

# 模型
- [ ] freeze and unfreeze
