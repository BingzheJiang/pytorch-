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

def lazy_sum(*args):
    def sum():
        ax=0
        for n in args:
            ax=ax+n
        return ax
    return sum

f=lazy_sum(1,3,5,7,9)
