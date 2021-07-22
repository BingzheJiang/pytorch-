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
def RNN_module():
    input_size = 4
    hidden_size = 4
    num_layers = 1
    batch_size = 1
    seq_len = 5
    inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
    labels = torch.LongTensor(y_data)
    print(labels.shape)

    class RNNModel(torch.nn.Module):
        def __init__(self,input_size, hidden_size, batch_size, num_layers=1):
            super(RNNModel, self).__init__()
            self.num_layers = num_layers
            self.input_size = input_size
            self.batch_size = batch_size
            self.hidden_size = hidden_size
            self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

        def forward(self, input):
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)  # 提供初始化隐藏层（h0）
            out, _ = self.rnn(input, hidden)#out=[ h0, h1, h2, h3, h4]  _ = [[[h4]]]
            #out torch.Size([5, 1, 4]) 𝑠𝑒𝑞𝑆𝑖𝑧𝑒, 𝑏𝑎𝑡𝑐ℎ, ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒
            return out.view(-1, self.hidden_size) #(5,4)

    net = RNNModel(input_size, hidden_size,batch_size, num_layers)
    #---计算损失和更新
    criterion = torch.nn.CrossEntropyLoss()#交叉熵
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    #---计算损失和更新

    for epoch in range(100):#训练100次
        optimizer.zero_grad()
        outputs = net(inputs)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)#output 5,4  labeL 5
        loss.backward()
        optimizer.step()

        _, idx = outputs.max(dim=1)##从第一个维度上取出预测概率最大的值和该值所在序号
        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
        print(', Epoch [%d/100] loss = %.3f' % (epoch + 1, loss.item()))

if __name__=="__main__":
    RNN_module()
