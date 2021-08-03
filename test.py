#w是Tensor(张量类型)，Tensor中包含data和grad，data和grad也是Tensorl。grad初始为None，调用l.backward()方法后w.grad为Tensor，故更新w.data时需使用w.grad.data。如果w需要计算梯度，那构建的计算图中，跟w相关的tensor都默认需要计算梯度
import torch
import math
# a=torch.tensor([1.0])
# a.requires_grad=True
# print(a)
# print(a.data)
# print(a.type())# a的类型是tensor
# print(a.data.type())# a.data的类型是tensor
# print(a.grad)
# print(type(a.grad))

# pred = torch.tensor([[-0.2], [0.2], [0.8]])
# target = torch.tensor([[0.0], [0.0], [1.0]])
#
# sigmoid = torch.nn.Sigmoid()
# pred_s = sigmoid(pred)
# print(pred_s)
# """
# pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
# 0*math.log(0.4502)+1*math.log(1-0.4502)
# 0*math.log(0.5498)+1*math.log(1-0.5498)
# 1*math.log(0.6900) + 0*log(1-0.6900)
# """
# result = 0
# i = 0
# for label in target:
#     if label.item() == 0:
#         result += math.log(1 - pred_s[i].item())
#     else:
#         result += math.log(pred_s[i].item())
#     i += 1
# result /= 3
# print("bce：", -result)
# loss = torch.nn.BCELoss()
# print('BCELoss:', loss(pred_s, target).item())

# #爬虫1
# import requests
# url='http://www.cntour.cn'
# strhtml=requests.get(url)
# print(strhtml.text)

# #爬虫2
# import requests
# import json
# def get_translate_date(word=None):
#     url='https://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
#     From_data={'i':word,'from':'AUTO','to':'AUTO','smartresult':'dict','client':'fanyideskweb','salt':'16267053010489','sign':'cd6de42c930991697d1fe3d8772ca751',
#            'lts':'1626705301048','bv':'2e190fb74c4addefd3f5c203e9d6e1de','doctype':'json','version':'2.1','keyfrom':'fanyi.web','action':'FY_BY_REALTlME'
#            }
#     # 请求表单数据
#     response=requests.post(url,data=From_data)
#     # 将Json格式字符串转字典
#     content =json.loads(response.text)
#     print(content)
#     #print(content['translateResult'][0][0]['tgt'])
#
# if __name__=='__main__':
#     get_translate_date('我爱中国')

# import requests        #导入requests包
# import json
# def get_translate_date(word=None):
#     url = 'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'
#     From_data={'i':word,'from':'zh-CHS','to':'en','smartresult':'dict','client':'fanyideskweb','salt':'15477056211258','sign':'b3589f32c38bc9e3876a570b8a992604','ts':'1547705621125','bv':'b33a2f3f9d09bde064c9275bcb33d94e','doctype':'json','version':'2.1','keyfrom':'fanyi.web','action':'FY_BY_REALTIME','typoResult':'false'}
#     #请求表单数据
#     response = requests.post(url,data=From_data)
#     #将Json格式字符串转字典
#     content = json.loads(response.text)
#     print(content)
#     #打印翻译后的数据
#     #print(content['translateResult'][0][0]['tgt'])
# if __name__=='__main__':
#     get_translate_date('我爱中国')

# #计算交叉熵损失
# import torch
# y=torch.LongTensor([0])
# z=torch.Tensor([[0.2,0.1,-0.1]])
# criterion=torch.nn.CrossEntropyLoss()
# loss=criterion(z,y)
# print(loss)
#
# import numpy as np
# y1=np.array([1,0,0])
# z1=np.array([0.2,0.1,-0.1])
# y_pred=np.exp(z1)/np.exp(z1).sum()
# loss1=(-y1*np.log(y_pred)).sum()
# print(loss1)

# import torch
# criterion =torch.nn.CrossEntropyLoss()
# Y=torch.LongTensor([2,0,1])
# Y_pred1=torch.Tensor([[0.1,0.2,0.9],
#                       [1.1,0.1,0.2],
#                       [0.2,2.1,0.1]])
# Y_pred2=torch.Tensor([[0.8,0.2,0.3],
#                       [0.2,0.3,0.5],
#                       [0.2,0.2,0.5]])
# l1=criterion(Y_pred1,Y)
# l2=criterion(Y_pred2,Y)
# print("Batch Loss1=",l1.data,"\nBatch Loss2=",l2.data)

#高阶函数，一个函数可以接受另一个函数作为参数
# def add(x,y,f):
#     return f(x)+f(y)
#
# print(add(-5,6,abs))

#可变参数求和
# def calc_sum(*args):
#     ax=0
#     for n in args:
#         ax=ax+n
#     return ax
#
# print(calc_sum(1,2,3,4,5))

# def lazy_sum(*args):
#     def sum():
#         ax=0
#         for n in args:
#             ax=ax+n
#         return ax
#     return sum
#
# f=lazy_sum(1,3,5,7,9)
# print(f())

#列表生成式
# a=[x*x for x in range(1,11)]
# print(a)
# charater=[[1,2,3,4],
#           [2,3,4,5],
#           [3,4,5,6]]
# input=[0,1,0,1,2]
# a=[charater[x] for x  in  input]
# print(a)
# import torch
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# print(input.shape)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# print(target.shape)
# output = loss(input, target)
# output.backward()

import torch
# word1=torch.LongTensor([0,1,2])
# word2=torch.LongTensor([3,1,2])
# embedding=torch.nn.Embedding(4,5)
#
# print(embedding.weight)
# print('word1:')
# print(word1)
# print(embedding(word1))
# b=torch.rand(3,3)
# print(b)
# b=b.long()
# print(b)
# x_data = [1, 0, 2, 2, 3]
# inputs=torch.LongTensor(x_data)
# embedding=torch.nn.Embedding(4,10)
# print(inputs)
# print(embedding(inputs))
# import gzip
# import csv
# file='./data/names_train.csv.gz'
# with gzip.open(file,'rt') as f:
#     f_csv=csv.reader(f)
#     # l=list(f_csv)
#     # print(l)
#     # for row in f_csv:
#     #     print(row[1])
#     rows=list(f_csv)
# names=[row[0] for row in rows]
# countrys=[row[1] for row in rows]
# # print(names)
# # print(countrys)
# countrys_list=sorted(set(countrys))
#
# countrys_dict=dict()
# t=0
# for i in  countrys_list:
#     countrys_dict[i]=t
#     t+=1
# print(countrys_dict)
#
# country_dict = dict()                                       #创建空字典
# for idx, country_name in enumerate(countrys_list):    #取出序号和对应国家名
#     country_dict[country_name] = idx                        #把对应的国家名和序号存入字典
# print(names[0])
# print(country_dict[countrys[0]])

# import torch
#
# temp=torch.rand(4,6,8)
# print(temp)
# print(temp.shape)
# print(temp[-1])
# print(temp[-2])
# print(temp[-1].shape)
# print(temp[-2].shape)

#测试embedding
# import torch
# embedding = torch.nn.Embedding(10, 3)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# o=embedding(input)
# print(o)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# input=input.t()
# o=embedding(input)
# print(o)
#测试torch.cat
# import torch
# x = torch.randn(2, 3)
# print(x)
# t=torch.cat([x, x, x], 1)
# print(t)

#测试ord()
# print(ord('a'))

#补零测试
# import torch
# seq=torch.tensor([1,2,8,45])
# seq_tensor = torch.zeros(10, 7).long()
# print(seq_tensor)
# seq_tensor[0,0:4]=seq_tensor[0,:4]=torch.LongTensor(seq)
# print(seq_tensor)
#
# idx=torch.tensor([2,0,1,3,4,5,6,7,8,9])
# seq_tensor=seq_tensor[idx]
# print(seq_tensor)

#测试max
import torch
a = torch.randn(3,3)
print(a)
print(a.max(1,keepdim=True)[1])