#用PyTorch实现线性回归
#1.prepare dataset
#2.design model using Class  # 目的是为了前向传播forward，即计算y hat(预测值)
#3.Construct loss and optimizer (using PyTorch API) 其中，计算loss是为了进行反向传播，optimizer是为了更新梯度。
#4.Training cycle (forward,backward,update)
#代码说明
#1.Module实现了魔法函数__call__()，call()里面有一条语句是要调用forward()。因此新写的类中需要重写forward()覆盖掉父类中的forward()
#2.call函数的另一个作用是可以直接在对象后面加()，例如实例化的model对象，和实例化的linear对象
#3.魔法函数call的实现,model(x_data)将会调用model.forward(x_data)函数，model.forward(x_data)函数中的
#4.self.linear(x)也由于魔法函数call的实现,将会调用torch.nn.Linear类中的forward
#每一次epoch的训练过程，总结就是
#①前向传播，求y hat （输入的预测值）
#②根据y_hat和y_label(y_data)计算loss
#③反向传播 backward (计算梯度)
#④根据梯度，更新参数
import torch
#prepare dataset
#x,y是矩阵，3行1列，也就是说总共有3个数据，每个数据只有1个特征
x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[2.0],[4.0],[6.0]])

#design model using class
'''
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called
'''
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=LinearModel()


# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion=torch.nn.MSELoss(size_average=False)# 这个函数要查一下
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)  # model.parameters()自动完成参数的初始化操作

#training cycle forward,backward,update
for epoch in range(100):
    y_pred=model(x_data)# forward:predic
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad() # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()# backward: autograd，自动计算梯度
    optimizer.step()# update 参数，即更新w和b的值

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test=torch.tensor([[4.0]])
y_test=model(x_test)
print('y_pred=',y_test.data)


