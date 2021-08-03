import torch
import spacy
from torchtext.legacy.data import Field,TabularDataset,BucketIterator
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import torch.nn.utils as utils
import math

INPUT_SIZE = 100
EMBED_SIZE = 200
HIDDEN_SIZE = 10
NUM_LAYERS = 1
LINERA_SIZE = 10
OUTPUT_SIZE = 100
DROPOUT = 0.5
EPOCH = 1
BATCH_SIZE = 16
LR = 0.1
CLIP = 1
BEST_EVAL_LOSS = 0.1

zh = spacy.load("zh_core_web_md")   #载入词语模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #定义处理器

def tokenize(texts):  #分词
    tokens = [token.text for token in zh.tokenizer(texts)][::-1]
    return tokens

def iter_data(batch_size):
    PRO = Field(tokenize=tokenize, init_token="<eos>", eos_token="<eos>")  # 定义数据处理
    fields = {"做法":("pro",PRO)}  #batch名称与json名称匹配
    train_data, eval_data, test_data = TabularDataset.splits(
        path="data",
        train="train.json",
        evaluate="eval.json",
        test="test.json",
        format="json",
        # skip_header=True,
        fields=fields
    )  # 导入数据；处理数据
    PRO.use_vocab(train_data,vectors="glove.6B.200d",unk_init=torch.Tensor.uniform_)    #创建词典
    train_iter = BucketIterator.splits(
        train_data,
        batch_size=batch_size,
        sort=False,
        sort_within_batch=False,
        shuffled=True,
        device=device
    )   #迭代数据
    return train_iter,PRO,eval_data,test_data

class VAE(nn.Module):
    def __init__(self,dict_size):
        super(VAE,self).__init__()
        self.embed = nn.Embedding(dict_size,EMBED_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.lstm = nn.LSTM(
            input_size=EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.linear_mean = nn.Linear(HIDDEN_SIZE, LINERA_SIZE)
        self.linear_std = nn.Linear(HIDDEN_SIZE, LINERA_SIZE)
        self.linear_out = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)

    def encoder(self,x):
        x_embed = self.embed(x)
        x_dropout = self.dropout(x_embed)
        outputs, (h_n,c_n) = self.lstm(x_dropout)
        output = outputs[:-1:]
        mu = self.linear_mean(output)
        logvar = fc.softplus(self.linear_std(output))
        return mu,logvar

    def reparameter(self,mu,logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decoder(self,x,z):
        x = torch.unsqueeze(x, 1)
        x_embed = self.embed(x)
        z = torch.unsqueeze(z, 1)
        x_conc = torch.cat((x_embed,z),1)
        x_dropout = self.dropout(x_conc)
        output, (h_n, c_n) = self.lstm(x_dropout)
        output = torch.squeeze(output,1)
        predict = self.linear_out(output)
        return predict

    def forward(self,x,is_test=False):
        mu,logvar = self.encoder(x)
        z = self.reparameter(mu,logvar)
        max_length = x.shape[1]
        input = x[0]
        predicts = torch.zeros(BATCH_SIZE,max_length,OUTPUT_SIZE).to(device)
        results = []
        for idx in range(1,max_length):
            predict = self.decoder(input,z)
            predicts[idx] = predict
            input = torch.max(predict,1)[1]   #获得结果索引
            if is_test == True:
                result = PRO.vocab.itos[input]  #通过词典将索引映射成词语
                results.append(result)
        if is_test == True:
            return predicts,mu,logvar,results
        else:
            return predicts,mu,logvar

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("LSTM") != -1:
        nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.0)
    if classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.0)

train_iterator,PRO,eval_data,test_data = iter_data(BATCH_SIZE)
dict_size = len(PRO.vocab)
vae =VAE(dict_size)
vae.embedding.weight.data.copy_(PRO.vocab.vectors) #输入embedding
vae.apply(weight_init)
vae.to(device)

optim = optim.Adam(vae.parameters(),lr=LR)

def loss_func(outputs,x,mu,logvar):
    mse_loss = nn.MSELoss()
    recon_loss = mse_loss(outputs,x)
    kl_diver = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    loss = recon_loss + kl_diver
    return loss

def train(iterator):
    vae.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(iterator):
        outputs, mu, logvar = vae(batch.pro)
        outputs = outputs[:][1:]
        batch.pro = batch.pro[:][1:]
        loss = loss_func(outputs, batch.pro, mu, logvar)
        optim.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(vae.parameters(),CLIP)    #梯度剪枝
        optim.step()
        epoch_loss += loss.item()
    aver_loss = epoch_loss/len(iterator)
    return aver_loss

def eval(eval_data):
    vae.eval()
    with torch.no_grad():
        outputs, mu, logvar = vae(eval_data)
        loss = loss_func(outputs, eval_data, mu, logvar)
    return loss

def test(test_data):
    vae.eval()
    with torch.no_grad():
        outputs, mu, logvar, results = vae(test_data, True)
        loss = loss_func(outputs, test_data, mu, logvar)
    return loss, results

for epoch in range(EPOCH):
    train_loss = train(train_iterator)
    train_ppl = math.exp(train_loss)
    eval_loss = eval(eval_data)
    eval_ppl = math.exp(eval_loss)
    test_loss,test_results = eval(test_data)
    test_ppl = math.exp(test_loss)
    print("Epoch:{:d},Train Loss:{:.3f},Train PPL:{:.3f}.".format(epoch+1,train_loss,train_ppl))
    print("Epoch:{:d},Eval Loss:{:.3f},Eval PPL:{:.3f}.".format(epoch + 1, eval_loss, eval_ppl))
    print("Epoch:{:d},Test Loss:{:.3f},Test PPL:{:.3f}.".format(epoch + 1, test_loss, test_ppl))
    if eval_loss < BEST_EVAL_LOSS:
        torch.save(vae.state_dict(), "vae_model.pt")
        with open("test_result.text","w") as f:
            f.writelines(test_results)
            f.close()

