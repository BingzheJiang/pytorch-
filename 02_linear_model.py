# 一般步骤1.DataSet 2.Model 3.Training 4.inferring
#给一个输入值，通过这个模型得到一个预测值，这个模型通过给定的数据集进行训练
import numpy as np
import matplotlib.pyplot as plt #导入画图包

x_data=[1.0,2.0,3.0]#设置数据集
y_data=[2.0,4.0,6.0]

#定义模型，假设这个问题结果可以用一个线性模型进行预测
#权重w的初试值是随机给定的
#forward意思是前向传播，pytorch会自动进行反向求导
def forward(x):
    return x*w

#定义loss函数，
def loss(x,y):
    y_pred=forward(x)
    return (y_pred - y)**2

#穷举法
w_list=[]
mse_list=[]
for w in np.arange(0.0,4.1,0.1):
    print("w=",w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data): #zip返回值是一个列表，列表元素为元组
        y_pred_val=forward(x_val)#y帽
        loss_val=loss(x_val,y_val)
        l_sum+=loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
plt.plot(w_list,mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()
#np.arange()
# 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
# 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
# 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数