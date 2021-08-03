#加载数据集

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE=200
USE_GPU=False
N_EPOCHS=100
N_LAYER=1
N_CHARS=128
HIDDEN_SIZE=100

class MyDataset(Dataset):
    def __init__(self,is_train_set=True):
        # 使用’rb’按照二进制位进行读取的，不会将读取的字节转换成字符，二进制文件用二进制读取用’rb’
        filepath='data/names_train.csv.gz' if is_train_set else 'data/names_train.csv.gz'
        with gzip.open(filepath,'rt') as f:
            f_csv=csv.reader(f)
            rows=list(f_csv)
        self.names=[row[0] for row in rows]
        self.countrys=[row[1] for row in rows]
        self.len=len(self.names)
        self.countrys_list=sorted(set(self.countrys))
        self.countrys_num=len(self.countrys_list)
        self.countrys_dict=self.getCountrysDict()

    def __getitem__(self, item):
        return self.names[item],self.countrys_dict[self.countrys[item]]#人名，国家字典序号

    def __len__(self):
        return self.len
    def getCountrysDict(self):
        countrys_dict=dict()
        for idx,country in enumerate(self.countrys_list):
            countrys_dict[country]=idx
        return countrys_dict

    def getCountrysName_id2CountrysName(self,index):
        return self.countrys_list[index]

    def getCountrysNum(self):
        return self.countrys_num

train_set=MyDataset(is_train_set=True)
train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_set=MyDataset(is_train_set=False)
test_loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)

N_COUNTRY=train_set.getCountrysNum()

def create_tensor(tensor):
    if USE_GPU:
        device=torch.device("cuda:0")
        tensor=tensor.to(device)
    return tensor

class RNNClassfication(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layer=1,shuangxiangrnn=True):
        super(RNNClassfication,self).__init__()
        self.hidden_size=hidden_size
        self.n_layers = n_layer
        self.n_direction=2 if shuangxiangrnn else 1
        self.embedding=torch.nn.Embedding(input_size,hidden_size)#input.shape=(seqlen,batch) output.shape=(seqlen,batch,hiddensize)
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layer,bidirectional=shuangxiangrnn)
        #输入维度，输出维度，层数，说明单向还是双向
        # (seqlen,batch,hiddensize)  (seqlen,batch,hiddensize*n_directions)
        self.fc=torch.nn.Linear(hidden_size*self.n_direction,output_size)
        # (seqlen,batch,hiddensize*n_directions) (seqlen,batch,hiddensize)

    def forward(self,input,seq_lengths):
        input=input.t()#input shape :  Batch x Seq -> S x B 用于embedding
        batch_size=input.size(1)
        hidden=self.__init_hidden(batch_size)
        embedding=self.embedding(input)

        seq_lengths = seq_lengths.cpu()  # 改成cpu张量
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)  # 让0值不参与运算加快运算速度的方式
        # 需要提前把输入按有效值长度降序排列 再对输入做嵌入，然后按每个输入len（seq——lengths）取值做为GRU输入

        outputs,hidden=self.gru(gru_input,hidden)#双向传播的话hidden有两个
        if self.n_direction==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)
        else:
            hidden_cat=hidden[-1]
        fc_output=self.fc(hidden_cat)
        return fc_output

    def __init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_size)
        return create_tensor(hidden)


def name2list(name):
    arr=[ord(c) for c in name]
    return arr,len(arr)

def make_tensors(names,countrys):
    sequences_and_lengths=[name2list(name) for name in names]
    name_sequences=[sl[0] for sl in sequences_and_lengths]
    seq_lengths=torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countrys=countrys.long()

    seq_tensor=torch.zeros(len(name_sequences),seq_lengths.max()).long()
    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths),0):
        seq_tensor[idx,:seq_len]=torch.LongTensor(seq)
    seq_lengths,perm_idx=seq_lengths.sort(dim=0,descending=True)
    seq_tensor=seq_tensor[perm_idx]
    countrys=countrys[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countrys)

def trainModel():
    total_loss=0

    for i,(names,countrys) in enumerate(train_loader,1):#可以改为0，是对i的给值(循环次数从0开始计数还是从1开始计数的问题)
        optimizer.zero_grad()
        inputs,seq_lengths,target=make_tensors(names,countrys)
        output=classfier(inputs,seq_lengths)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

        if i==len(train_set)//BATCH_SIZE:
            print(f'loss={total_loss/(i*len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(test_set)

    with torch.no_grad():
        for i, (names, countrys) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = make_tensors(names, countrys)
            output = classfier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set:Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    print("Train for %d epochs..."% N_EPOCHS)
    classfier=RNNClassfication(N_CHARS,HIDDEN_SIZE,N_COUNTRY,N_LAYER)
    if USE_GPU:
        device=torch.device('cuda:0')
        classfier.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # 计算损失，分类问题，计算交叉熵就可以了
    optimizer = torch.optim.Adam(classfier.parameters(), lr=0.001)  # 更新
    acc_list=[]
    for epoch in range(1,N_EPOCHS+1):
        print('%d/%d:'%(epoch,N_EPOCHS))
        trainModel()
        acc=testModel()
        acc_list.append(acc)
    epoch=np.arange(1,len(acc_list)+1,1)
    acc_list=np.array(acc_list)
    plt.plot(epoch,acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()







