#加载数据集

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv

BATCH_SIZE=200
USE_GPU=False

class MyDataset(Dataset):
    def __init__(self,is_train_set=True):
        # 使用’rb’按照二进制位进行读取的，不会将读取的字节转换成字符，二进制文件用二进制读取用’rb’
        filepath='./data/names_train.csv.gz' if is_train_set else './data/names_test.csv.gz'
        with gzip.open(filepath,'rt') as f:
            f_csv=csv.reader(f)
            rows=list(f)
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

N_COUNTRY=train_set.countrys_num

def create_tensor(tensor):
    if USE_GPU:
        device=torch.device("cuda:0")
        tensor=tensor.to(device)
    return tensor

class RNNClassfication(torch.nn.Module):
    def __init__(self,hidden_size,input_size,output_size,n_layer=1,shuangxiangrnn=True):
        super(RNNClassfication,self).__init__()
        self.hidden_size=hidden_size
        self.n_direction=2 if shuangxiangrnn else 1
        self.embedding=torch.nn.Embedding(input_size,hidden_size)#input.shape=(seqlen,batch) output.shape=(seqlen,batch,hiddensize)
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layer,bidirectionnal=shuangxiangrnn)
        #输入维度，输出维度，层数，说明单向还是双向
        # (seqlen,batch,hiddensize)  (seqlen,batch,hiddensize*n_directions)
        self.fc=torch.nn.Linear(hidden_size*self.n_direction,output_size)
        # (seqlen,batch,hiddensize*n_directions) (seqlen,batch,hiddensize)

    def forward(self,input,seq_lengths):
        input=input.t()#input shape :  Batch x Seq -> S x B 用于embedding
        batch_size=input.size(1)
        hidden=self.__init_hidden(batch_size)
        embedding=self.embedding(input)
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)  # 让0值不参与运算加快运算速度的方式
        # 需要提前把输入按有效值长度降序排列 再对输入做嵌入，然后按每个输入len（seq——lengths）取值做为GRU输入

        outputs,hidden=self.gru(gru_input,hidden)#双向传播的话hidden有两个


    def __init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)




