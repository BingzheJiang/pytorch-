#1.w是Tensor， forward函数的返回值也是Tensor，loss函数的返回值也是Tensor
#2.本算法中反向传播主要体现在，l.backward()。调用该方法后w.grad由None更新为Tensor类型，且w.grad.data的值用于后续w.data的更新。
#3. l.backward()会把计算图中所有需要梯度(grad)的地方都会求出来，然后把梯度都存在对应的待求的参数中，最终计算图被释放。
#取tensor中的data是不会构建计算图的
import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.tensor([1.0])#w的初值为1.0
w.requires_grad=True#需要计算梯度

def forward(x):
    return x*w

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2

print("predict (before training)",4,forward(4).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)#l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()#  backward,compute grad for Tensor whose requires_grad set to True
        print('\tgrad:',x,y,w.grad.item())
        w.data=w.data-0.01*w.grad.data # 权重更新时，需要用到标量，注意grad也是一个tensor

        w.grad.data.zero_()#after update, remember set the grad to zero

    print('progress:',epoch,l.item())# 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict(after training)",4,forward(4).item())


