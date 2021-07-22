# pytorch-
b站刘二视频

将本地仓库同步到git远程仓库中
1.git add test.py //添加到暂存区
2.git commit -m '说明' //添加到仓库
3.git push //添加到远程仓库

张量、矩阵和向量区别
1.张量维数等价于张量阶数。0维张量就是标量、1维张量就是向量、2维张量就是矩阵、大于3维统一叫张量、
2.混淆点：数学里面会使用3维向量，n维向量的说法，这其实指的是1维张量（即向量）的形状，即它所含分量的个数，比如[1,3]这个向量的维数为2，它有1和3这两个分量；
[1,2,3,······，4096]这个向量的维数为4096，它有1、2······4096这4096个分量，都是说的向量的形状。你不能说[1,3]这个“张量”的维数是2，只能说[1,3]这个“1维张量”的维数是2。
矩阵也是类似，常常说的n×m阶矩阵，这里的阶也是指的矩阵的形状。
3.怎么看张量维数：
维度要看张量的最左边有多少个左中括号，有n个，则这个张量就是n维张量
[[1,3],[3,5]]最左边有两个左中括号，它就2维张量；[[[1,2],[3,4]],[[1,2],[3,4]]]最左边有三个左中括号，它就3维张量
4.形状的第一个元素要看张量最左边的中括号中有几个元素，形状的第二个元素要看张量中最左边的第二个中括号中有几个被逗号隔开的元素，形状的第3,4…n个元素以此类推
https://blog.csdn.net/shenggedeqiang/article/details/84856051

import numpy as np
X = np.array ([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]])
# X[:,0]就是取所有行的第0个数据
print(X[:, 0])
#输出：[ 0  2  4  6  8 10 12 14 16 18]
# X[:,0]就是取所有行的第0个数据
print( X[:,1])
#输出：[ 1  3  5  7  9 11 13 15 17 19]
#X[1,:]第一维中下标为1的元素的所有值
print(X[1,:])
#输出：[2 3]
https://blog.csdn.net/doubledog1112/article/details/85095571

#计算交叉熵损失
import torch
y=torch.LongTensor([0])
z=torch.Tensor([[0.2,0.1,-0.1]])
criterion=torch.nn.CrossEntropyLoss()
loss=criterion(z,y)
print(loss)

import numpy as np
y1=np.array([1,0,0])
z1=np.array([0.2,0.1,-0.1])
y_pred=np.exp(z1)/np.exp(z1).sum()
loss1=(-y1*np.log(y_pred)).sum()
print(loss1)

import torch
criterion =torch.nn.CrossEntropyLoss()
Y=torch.LongTensor([2,0,1])
Y_pred1=torch.Tensor([[0.1,0.2,0.9],
                      [1.1,0.1,0.2],
                      [0.2,2.1,0.1]])
Y_pred2=torch.Tensor([[0.8,0.2,0.3],
                      [0.2,0.3,0.5],
                      [0.2,0.2,0.5]])
l1=criterion(Y_pred1,Y)
l2=criterion(Y_pred2,Y)
print("Batch Loss1=",l1.data,"\nBatch Loss2=",l2.data)


#交叉熵举例
import torch
loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
print(input.shape)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
print(target.shape)
output = loss(input, target)
output.backward()

tensor([[ 0.5934, -0.8533, -1.1076,  0.0284, -1.0160],
        [-0.5386,  0.0046,  0.0180, -0.5689, -1.4313],
        [ 0.8632,  0.0926, -1.2669, -0.8354, -0.7605]], requires_grad=True)
torch.Size([3, 5])
tensor([1, 2, 2])
torch.Size([3])



BCEloss:https://blog.csdn.net/qq_22210253/article/details/85222093
sigmoid函数:https://www.jianshu.com/p/506595ec4b58
SGD随机梯度下降:https://zhuanlan.zhihu.com/p/27609238
深度学习优化函数详解:https://blog.csdn.net/qq_26591517/article/details/79679192
CrossEntropyLoss():https://zhuanlan.zhihu.com/p/98785902