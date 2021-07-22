#把一个序列变成另外一个序列
'''
训练RNN模型使得  "hello" -> "ohlol"
输入为"hello"，可设置字典 e -> 0 h -> 1 l -> 2 o -> 3 hello对应为 10223 one-hot编码有下面对应关系
h   1   0100            o   3
e   0   1000            h   1
l   2   0010            l   2
l   2   0010            o   3
o   3   0001            l   2
输入有“helo”四个不同特征于是input_size = 4
hidden_size = 4 batch_size = 1

RNN模型维度的确认至关重要：
rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
outputs, hidden_outs = rnn(inputs, hiddens):
    inputs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒
    hiddens of shape 𝑛𝑢𝑚𝐿𝑎𝑦𝑒𝑟𝑠, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    outputs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    hidden_outs of shape 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
cell = torch.nn.RNNcell(input_size=input_size, hidden_size=hidden_size)
output, hidden_out = cell(input, hidden):
    input of shape 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒
    hidden of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    output of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
    hidden_out of shape 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
其中，seqSize：输入个数  batch：批量大小  input_size：特征维数 numLayers：网络层数  hidden_size：隐藏层维数
'''
import torch

idx2char = ['e', 'h', 'l', 'o'] #方便最后输出结果
x_data = [1, 0, 2, 2, 3]        #输入向量
y_data = [3, 1, 2, 3, 2]        #标签

one_hot_lookup = [ [1, 0, 0, 0], #查询ont hot编码 方便转换
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1] ]
x_one_hot = [one_hot_lookup[x] for x in x_data] #按"1 0 2 2 3"顺序取one_hot_lookup中的值赋给x_one_hot
'''运行结果为x_one_hot = [ [0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1] ]
刚好对应输入向量，也对应着字符值'hello'
'''
"""
inputs
tensor([[[0., 1., 0., 0.]],

        [[1., 0., 0., 0.]],

        [[0., 0., 1., 0.]],

        [[0., 0., 1., 0.]],

        [[0., 0., 0., 1.]]])
labels      
tensor([[3],
        [1],
        [2],
        [3],
        [2]])
"""
def cell():
    input_size=4
    hidden_size=4
    batch_size=1
    inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)#𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 改变维度，增加了一个维度
    labels=torch.LongTensor(y_data).view(-1,1)#增加维度方便计算loss
    print(labels.shape)

    class cell_Model(torch.nn.Module):
        def __init__(self,input_size,hidden_size,batch_size):
            super(cell_Model,self).__init__()
            self.input_size=input_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.rnncell=torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

        def forward(self, input, hidden):
            hidden=self.rnncell(input,hidden)#hiddenshape: 𝑏𝑎𝑡𝑐ℎ, hidden_𝑠𝑖𝑧𝑒
            return hidden

        def init_hidden(self):
            return torch.zeros(self.batch_size,self.hidden_size)#h0全0向量

    net=cell_Model(input_size, hidden_size, batch_size)

    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)#基于随机梯度下降的优化器
    # ---计算损失和更新

    for epoch in range(50):#训练50次
        loss=0
        optimizer.zero_grad()
        hidden=net.init_hidden()
        print('Predicten string:', end='')
        for input, label in zip(inputs,labels):#并行遍历数据集 一个一个训练
            hidden = net(input, hidden)#shape: 𝑏𝑎𝑡𝑐ℎ, 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒        𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
            #hidden输出维度 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
            loss += criterion(hidden, label)#hidden(1,4) label(1)
            _, idx = hidden.max(dim=1)#从第一个维度上取出预测概率最大的值和该值所在序号
            print(idx2char[idx.item()], end='')#按上面序号输出相应字母字符
        loss.backward()
        optimizer.step()
        print(', Epoch [%d/50] loss=%.4f' %(epoch+1, loss.item()))

cell()