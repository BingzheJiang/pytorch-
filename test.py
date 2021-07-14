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

pred = torch.tensor([[-0.2], [0.2], [0.8]])
target = torch.tensor([[0.0], [0.0], [1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)
"""
pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
0*math.log(0.4502)+1*math.log(1-0.4502)
0*math.log(0.5498)+1*math.log(1-0.5498)
1*math.log(0.6900) + 0*log(1-0.6900)
"""
result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1 - pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1
result /= 3
print("bce：", -result)
loss = torch.nn.BCELoss()
print('BCELoss:', loss(pred_s, target).item())