#随机梯度下降
#损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练函数的损失。
#梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度
#本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。梯度下降法中梯度总共更新100(epoch)次。
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=1.0

#定义模型，假设这个问题结果可以用一个线性模型进行预测
#权重w的初试值是随机给定的
#forward意思是前向传播，pytorch会自动进行反向求导
def forward(x):
    return x*w

#计算损失函数
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

#定义梯度计算函数
def gradient(x,y):
    return 2*x*(x*w-y)

epoch_list=[]
loss_list=[]
print('predict(before training)',4,forward(4))
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad=gradient(x,y)
        w=w-0.01*grad
        print("\tgrad:",x,y,grad)
        l=loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (after training)',4,forward(4))
plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#疑问，随机体现在代码哪里