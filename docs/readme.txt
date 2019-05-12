# inference 运行
## 配置
    需要安装 pytorch ignite h5py
    如果需要的话 pip install requirements.txt 进行安装对应版本

bash inference.sh 进行运行
结果保存在 ../datasets/result.txt中
说明：
    脚本会先运行 test.py 将模型的预测写进 ../datasets/pred.h5
    然后运行 get_result.py 读取 pred.h5 取出实体和情感写入到../datasets/result.txt当中

    也可以直接运行 python get_result.py 读取pred.h5生成结果 省去模型预测环节


# 项目依赖
    + pytorch >=0.1
    + pytorch-pretrained-BERT
    + ignite https://github.com/pytorch/ignite
    + h5py


# 引用的开源数据和代码
   使用了 pytorch-pretrained-BERT(https://github.com/huggingface/pytorch-pretrained-BERT) 框架运行 BERT
   使用了适用于该框架的源自 google bert 的 bert-base-chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
   和其 token 模型 保存在 bert_pretrained 下

# 文件结构说明
    src -- 所有代码源码
        --get_sent_and_word_list.py 从原始新闻中进行清洗 分字 标记标签 保存到 train_ner_has_emotion.pkl wenj
        --data_raw  从train_ner_has_emotion.pkl 读取分字和标签写入 train_full.txt dev_full.txt文件 一行一个字一个标签
        --data_title_trnval.py  读取 train_full.txt 和 dev_full.txt 进行 token 生成 bert 需要的输入 input_ids
                                input_mask segment_ids, label_ent_ids label_emo_ids 写入 data_train.h5
        --data_title_test.py  类似上面，输入的 test 版本 写入 ../datasets/data_test.h5
        --models.py  网络模型定义文件
        --loss.py loss函数的定义文件
        --utils.py 用到的辅助函数汇总文件
        --metric.py F1分数定义文件
        --train.py 模型训练脚本 输出为checkpoint
        --test.py 进行 inference 写入 ../datasets/pred.h5
        --get_result.py 读取perd.h5 取出实体写入到 ../datasets/result.txt

    --bert_pretrained google预训练模型存放
        --bert-base-chinese 中文权重
        --bert_token_model tokenizer文件

     --datasets 所有的数据保存之处
        inference 用到的有 data_test.h5 ID2TOK_test.pkl(token是分为 UNK的字的记录)

     --pred_best.pth  最好的预测文件记录
     --result_best.txt 最好的提交结果

     --ckp
        --best_model.pth 最好的模型权重


# 思路

    因为新闻的标题是新闻的上下文概括，所以将每则新闻的标题拼接到该新闻的每个分句当中，形成
     [CLS] A句 [SEP] B句(=title),这样每个分句就能学到与整个新闻上下文相关的信息

    经过 bert 得到每个字的 768 维 word embedding 和 [CLS] 即整个句子的 embeeding 拼接
    到每个字的 embedding 上去 形成 768*2 维， 思路是进一步给予单字更多的上下文信息，帮助分类

    然后分别使用两个一层的线性分类器做 实体的 O B I 分类 和 情感的 O POS NEG NORM 分类

    且用 mask 的方法只对 A 句进行 loss_ent loss_emo 的 反传
    由于 emo(情感) 是 基于 ent(实体) 的 ，所以
    loss = alpha * loss_ent + loss_emo， alpha 赋予大于1的值


## 待实验
    我们小范围测试过拼接一则新闻的多个分句形成长句作为一个样本，同样是batch化和给予更多上下文的考量，虽然训练变慢，
    但效果会有一些提升。
    还有诸如给予B更多的loss权重的实验，由于时间原因没能测试






