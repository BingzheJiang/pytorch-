#w是Tensor(张量类型)，Tensor中包含data和grad，data和grad也是Tensorl。grad初始为None，调用l.backward()方法后w.grad为Tensor，故更新w.data时需使用w.grad.data。如果w需要计算梯度，那构建的计算图中，跟w相关的tensor都默认需要计算梯度
import torch
a=torch.tensor([1.0])
a.requires_grad=True
print(a)
print(a.data)
print(a.type())# a的类型是tensor
print(a.data.type())# a.data的类型是tensor
print(a.grad)
print(type(a.grad))

