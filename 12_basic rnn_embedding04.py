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

# one_hot_lookup = [ [1, 0, 0, 0], #查询ont hot编码 方便转换
#                    [0, 1, 0, 0],
#                    [0, 0, 1, 0],
#                    [0, 0, 0, 1] ]
# x_one_hot = [one_hot_lookup[x] for x in x_data] #按"1 0 2 2 3"顺序取one_hot_lookup中的值赋给x_one_hot
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
def embedding():
    num_class=4#类别数量
    input_size=4#输入维度,一共四个不同种类的字母，独热向量的维数
    hidden_size=8#隐藏层维度，自定义的，要和分类的数量一致，但是这个例子不一致，所以又接了一个线性层
    embedding_size = 10  # 嵌入到10维空间 自定义
    num_layers = 2  # RNN层数
    batch_size = 1
    seq_len = 5  # 数据量 hello 5个字母
    idx2char = ['e', 'h', 'l', 'o']  # 方便最后输出结果
    x_data = [[1, 0, 2, 2, 3]]  # (batch, seq_len)
    y_data = [3, 1, 2, 3, 2]  # (batch * seq_len)

    inputs=torch.LongTensor(x_data)#生成64位整型 tensor([1, 0, 2, 2, 3])
    labels=torch.LongTensor(y_data)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model,self).__init__()
            self.emb=torch.nn.Embedding(input_size,embedding_size)#初始的时候是随机的矩阵
            self.rnn=torch.nn.RNN(input_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
            self.fc=torch.nn.Linear(hidden_size,num_class)#把最后一个维度inputfeature转变成outputfeature

        def forward(self,x):
            hidden=torch.zeros(num_layers,x.size(0),hidden_size)
            x=self.emb(x)#input (batch,seqLen)  output (batch, seqLen, embeddingSize)
            x,_=self.rnn(x,hidden)#x为输出
            # print(x.shape) torch.Size([1, 5, 8])
            x=self.fc(x)#
            # print(x.shape) torch.Size([1, 5, 4])
            return x.view(-1,num_class)#修改维度好用交叉熵计算损失

    net = Model()
    # ---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # ---计算损失和更新

    for epoch in range(15):#训练15次
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.shape,labels.shape)
        loss=criterion(outputs, labels)#torch.Size([5, 4]) torch.Size([5])
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)  ##从第一个维度上取出预测概率最大的值和该值所在序号
        idx = idx.data.numpy()#torch.Tensor转numpy.ndarray
        # print(type(idx))
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))

embedding()