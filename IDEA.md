1. 先 freeze BERT 的底层 让上层部分改变 再放开全部训练  
    > 初定 freeze 一个 epoch ?
    
2. 多级重复 loss
    > 即 loss(A+B) + loss(A)  
    这样 loss(A) 重复了 相当于加了一倍权重
    
3. 交叉熵的代价矩阵
    > 对 B I O 的 交叉熵代价赋权重