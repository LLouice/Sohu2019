# 2019搜狐校园算法大赛 llouice

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

## inference 运行
### 配置
- 需要安装 pytorch ignite h5py
- 如果需要的话 pip install requirements.txt 进行安装对应版本

bash inference.sh 进行运行
结果保存在 ../datasets/result.txt中

说明：
- 脚本会先运行 test.py 将模型的预测写进 ../datasets/pred.h5
- 然后运行 get_result.py 读取 pred.h5 取出实体和情感写入到../datasets/result.txt当中
- 也可以直接运行 python get_result.py 读取pred.h5生成结果 省去模型预测环节


## 项目依赖
* pytorch >=0.1
* pytorch-pretrained-BERT
* ignite https://github.com/pytorch/ignite
* h5py


## 引用的开源数据和代码
   使用了 pytorch-pretrained-BERT(https://github.com/huggingface/pytorch-pretrained-BERT) 框架运行 BERT
   使用了适用于该框架的源自 google bert 的 bert-base-chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters
   和其 token 模型 保存在 bert_pretrained 下

## 文件结构说明
- src	所有代码源码      
	- kfold	k折交叉验证	
		- test_avg_all.py	平均所有结果
		- train.py	模型训练
		- test.py	模型测试输出结果

	- scripts	存放一键运行脚本

	- data_raw_test.py	将新闻和标记写入txt文件
	- data_raw_trnval.py	将新闻和标记写入txt文件
	- data_title_test.py	读取 train_full.txt 和 dev_full.txt 进行 token 生成 bert 需要的输入 input_ids
							input_mask segment_ids, label_ent_ids label_emo_ids 写入 data_train.h5
	- data_title_trnval.py	类似上面，输入的 test 版本 写入 ../datasets/data_test.h5
	- get_result.py	读取perd.h5 取出实体写入到 ../datasets/result.txt
	- get_sents*.py	分句
	- loss.py	loss函数的定义文件
	- metric.py	F1分数定义文件
	- models.py	网络模型定义文件,选择模型网络选项 --net
	- requirements.txt	项目依赖
	- test.py	添加选择模型网络选项 --net	2 days ago
	- train.py	模型训练脚本 输出为checkpoint
	- trainx.py	单独训练 ent 和 emo
	- utils.py	用到的辅助函数汇总文件

- bert_pretrained google预训练模型存放
	- bert-base-chinese 中文权重
	- bert_token_model tokenizer文件

- ckp
	- best_model.pth 最好的模型权重
- datasets	所有数据保存之处

- preds	所有预测输出的h5文件

- pred_best.pth  最好的预测文件记录
- result/best 存放最好的提交结果
	- result_best.txt 最好的提交结果

## 算法思路
### 数据预处理
清洗：保留数字、英文、中文、中英文标点、空格和\n并将网页的转义字符还原。去除连续多余的空格。去除链接。去除多余连续的符号（无意义的、颜文字表情）。
分句：对新闻的内容进行分句处理，由于后续输入网络的最大长度是256.  减去BERT需要的三个字符 CLS SEP CLS 长度设为256-3-标题长；
一级：|｡|！|\!|？|\?分句，然后对分完的句子进行判断，不符合的进入二级截断
二级：；|、|,|，|﹔|､ 
如果还是不符合，就按最大长度截断。 2000左右条句子被截断。
分字：对内容和标题都分字。把中文当成截断字符来分，英文先保留整体单词。针对每个分开的字，再通过标点符号对英文分字。
打标签：采用BIE标签，先从长实体开始打起，BIIIIIIE，如果再打标签的时候遇到了BIE这种，则不进行标签处理。
遇到问题：某些实体当中含有分句的字符，就会被切开。 通过统计发现带有，的实体都在《》“”之内，于是我们采用哈希mask的方式，将实体进行等长度的哈希之后保护替换，然后进行分句还原。保护他们不被截断。 分句将\n\r删掉了

### 创新思路
因为新闻的标题是新闻的上下文概括，所以将每则新闻的标题拼接到该新闻的每个分句当中，形成[CLS] A句 [SEP] B句(=title),这样每个分句就能学到与整个新闻上下文相关的信息。经过 bert 得到每个字的 768 维 word embedding 和 [CLS] 即整个句子的 embeeding 拼接到每个字的 embedding 上去 形成 768*2 维， 思路是进一步给予单字更多的上下文信息，帮助分类。

分别使用两个一层的线性分类器做 实体的 O B I 分类 和 情感的 O POS NEG NORM 分类。且用 mask 的方法只对 A 句进行 loss_ent loss_emo 的 反传，由于 emo(情感) 是 基于 ent(实体) 的 ，所以loss = alpha * loss_ent + loss_emo， alpha 赋予大于1的值。

针对重叠实体问题，如预测“天然气 天然气田”，“个人信息 个人信息保护”等我们采用BIESO标志，重新生成输入，调整分类数目，训练模型

针对tokenize产生[UNK]造成无法还原的问题，我们在进行tokenize时，将最终token为[UNK]的实体使用正则表达式匹配到，添加到我们自己基于BERT Vocabulary的扩展字典当中。这样预测时就能还原出每一个字。

### 神经网络结构图
![神经网络结构图](http://wx1.sinaimg.cn/large/e8f43d1ely1g39w4lg4ebj20h90a3mxc.jpg)

