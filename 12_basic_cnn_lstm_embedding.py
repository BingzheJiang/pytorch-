import torch

input_size = 4
batch_size = 1
class emb_LSTM(torch.nn.Module):
    def __init__(self):
        super(emb_LSTM, self).__init__()
        self.linearix = torch.nn.Linear(10, 4)
        self.linearfx = torch.nn.Linear(10, 4)
        self.lineargx = torch.nn.Linear(10, 4)
        self.linearox = torch.nn.Linear(10, 4)
        self.linearih = torch.nn.Linear(4, 4)
        self.linearfh = torch.nn.Linear(4, 4)
        self.lineargh = torch.nn.Linear(4, 4)
        self.linearoh = torch.nn.Linear(4, 4)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, hidden, c):
        i = self.sigmoid(self.linearix(x) + self.linearih(hidden))
        f = self.sigmoid(self.linearfx(x) + self.linearfh(hidden))
        g = self.tanh(self.lineargx(x) + self.lineargh(hidden))
        o = self.sigmoid(self.linearox(x) + self.linearoh(hidden))
        c = f * c + i * g
        hidden = o * self.tanh(c)
        return hidden, c

model = emb_LSTM()

def emb_train():
    idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
    x_data = torch.LongTensor([[1, 0, 2, 2, 3]]).view(5, 1)
    y_data = [3, 1, 2, 3, 2]  # 标签
    labels = torch.LongTensor(y_data).view(-1, 1)  # 增加维度方便计算loss
    emb = torch.nn.Embedding(4, 10)
    inputs = emb(x_data)
    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # ---计算损失和更新
    for epoch in range(100):
        loss = 0
        optimizer.zero_grad()
        hidden = torch.zeros(batch_size, input_size)#提供初始化隐藏层（h0）
        c = torch.zeros(batch_size, input_size)  # 提供初始化隐藏层（c0）
        print('Predicten string:', end='')
        for input, label in zip(inputs,labels):#并行遍历数据集 一个一个训练
            hidden, c = model(input, hidden, c)
            loss += criterion(hidden, label)
            _, idx = hidden.max(dim=1)#从第一个维度上取出预测概率最大的值和该值所在序号
            print(idx2char[idx.item()], end='')#按上面序号输出相应字母字符
        loss.backward(retain_graph=True)#运行时报错，错误提示是下面这段话，根据提示修改参数就可以了。retain_graph=True
        '''Trying to backward through the graph a second time,
        but the saved intermediate results have already been freed.
         Specify retain_graph=True when calling backward the first time.'''
        optimizer.step()
        print(', Epoch [%d/100] loss=%.4f' %(epoch+1, loss.item()))
emb_train()