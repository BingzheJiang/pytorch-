import torch

input_size = 4
batch_size = 1


class GRU(torch.nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.linearzh=torch.nn.Linear(4,4)
        self.linearzx=torch.nn.Linear(10,4)
        self.linearrh=torch.nn.Linear(4,4)
        self.linearrx=torch.nn.Linear(10,4)
        self.linearnx=torch.nn.Linear(10,4)
        self.linearnh=torch.nn.Linear(4,4)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, hidden):
        z=self.sigmoid(self.linearzh(hidden)+self.linearzx(x))
        r=self.sigmoid(self.linearrh(hidden)+self.linearrx(x))
        n=self.tanh(r*self.linearnh(hidden)+self.linearnx(x))
        hidden=(1-z)*n+z*hidden

        return hidden


net = GRU()

# ---计算损失和更新
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# ---计算损失和更新
def train():
    idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
    # x_data = [1, 0, 2, 2, 3]  # 输入向量
    # x_data = torch.LongTensor(x_data).view(-1,1)
    x_data = torch.LongTensor([[1, 0, 2, 2, 3]]).view(5, 1)
    y_data = [3, 1, 2, 3, 2]  # 标签
    labels = torch.LongTensor(y_data).view(-1, 1)  # 增加维度方便计算loss size(5,1),因为后面用zip拿出来每一对，后面少了第一个维度
    emb=torch.nn.Embedding(4,10)
    inputs=emb(x_data)

    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # ---计算损失和更新

    for epoch in range(100):#开始训练
        loss = 0
        optimizer.zero_grad()
        hidden = torch.zeros(batch_size, input_size)#提供初始化隐藏层（h0）
        print('Predicten string:', end='')
        for input, label in zip(inputs,labels):#并行遍历数据集 一个一个训练
            hidden = net(input, hidden)
            loss += criterion(hidden, label)#hidden.shape=(1,4) label.shape=1 只要第一个维度一样就可以
            _, idx = hidden.max(dim=1)#从第一个维度上取出预测概率最大的值和该值所在序号
            print(idx2char[idx.item()], end='')#按上面序号输出相应字母字符
        loss.backward(retain_graph=True)
        optimizer.step()
        print(', Epoch [%d/100] loss=%.4f' %(epoch+1, loss.item()))

train()