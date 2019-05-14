1. 先 freeze BERT 的底层 让上层部分改变 再放开全部训练  
    > 初定 freeze 一个 epoch ?
    
2. 多级重复 loss
    > 即 loss(A+B) + loss(A)  
    这样 loss(A) 重复了 相当于加了一倍权重
    
3. 交叉熵的代价矩阵
    > 对 B I O 的 交叉熵代价赋权重
    
4. 成词二分类 搜狐的训练集实体 + 百度实体  -> BERT vector -> 二分类器

5. 新闻分类
	> 采用分类模型，或者规则分类，将训练集分类（类别尽量少），然后构建五折或者验证集的时候就可以从每个类中提取相同比例的数据来进行测试和验证。  
	> 同时还可以把类别作为aspect来训练模型  
	> 自己训练新闻分类模型（全网新闻数据（SogouCA）http://www.sogou.com/labs/resource/ca.php）

6. 构建预测实体质量检测模型，采用CN-DBpedia(复旦大学GDM实验室中文知识图谱)或者训练集实体或者其他词汇库【采用规则或者深度学习】
	> 1）CN-DBpedia(复旦大学GDM实验室中文知识图谱) http://openkg.cn/dataset/cndbpedia  
	> 2）THUOCL（THU Open Chinese Lexicon）是由清华大学自然语言处理与社会人文计算实验室整理推出的一套高质量的中文词库，词表来自主流网站的社会标签、搜索热词、输入法词库等。https
	://github.com/thunlp/THUOCL  
	> 3）训练集+example的实体，分词后使用网络训练，要自己生成负样本，然后用这个分类模型来判断生成的测试实体的好坏  
	> 4）公司名语料库（Company-Names-Corpus）https://github.com/wainshine/Company-Names-Corpus  
	> 5）中文词语搭配库(SogouR) http://www.sogou.com/labs/resource/r.php  
	> 6）壹沓科技中文新词https://github.com/1data-inc/chinese_popular_new_words
	> 7）中文人名语料库（Chinese-Names-Corpus）https://github.com/wainshine/Chinese-Names-Corpus  
	> 8）各种语料https://github.com/fighting41love/funNLP   
	> 9）神策杯2018高校算法大师赛（中文关键词提取）第二名代码方案（带有字典）https://github.com/bigzhao/Keyword_Extraction

7. 训练多个不同想法，不同参数的bert，进行融合

8. 五折stacking融合，融合两次，第一次采用DNN(可以配合传统的特征，传统的模型进行融合)，第二次采用深度融合+伪标签，具体看学长的github：https://github.com/zhanzecheng/SOHU_competition

9. 规则纠正预测实体，得提取一些特征看看是否有效。比如取长词，或者取短词，或者融合能包含的词（如互联网工业，工业，互联网————要“工业、互联网”还是“互联网工业”）

10. 看看标点是否影响分数，中英文标点可以分别去掉一次，然后提交，如果不影响，除分句标点可以全部去掉（甚至分句标点分完句后也能去掉）