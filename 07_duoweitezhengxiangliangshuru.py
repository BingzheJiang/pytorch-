#https://blog.csdn.net/bit452/article/details/109682078
#1.乘的权重(w)都一样，加的偏置(b)也一样。b变成矩阵时使用广播机制。神经网络的参数w和b是网络需要学习的，其他是已知的。
#2.学习能力越强，有可能会把输入样本中噪声的规律也学到。我们要学习数据本身真实数据的规律，学习能力要有泛化能力
#3.该神经网络共3层；第一层是8维到6维的非线性空间变换，第二层是6维到4维的非线性空间变换，第三层是4维到1维的非线性空间变换。
#4.本算法中torch.nn.Sigmoid() # 将其看作是网络的一层，而不是简单的函数使用
import numpy as np
import torch
import matplotlib.pyplot as plt

#prepare dataset
xy=np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32)
# print(xy)
x_data=torch.from_numpy(xy[:,:-1])# 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
# print(x_data)
# print(x_data.type())
y_data=torch.from_numpy(xy[:,[-1]])# [-1] 最后得到的是个矩阵
print(y_data)