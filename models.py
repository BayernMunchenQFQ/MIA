# -- coding: utf-8 --
# @Time : 2021/3/27 下午3:22
# @Author : ymm
# @File : models.py
from Config import config
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

device=torch.device('cuda',config.gpu_id)
def create_model(model_name,num_class,max_words=80000):
    if model_name=='wordcnn':
        print('wordcnn')
        return wordCNN(classes=num_class,maxword=max_words)
    elif model_name=='textcnn':
        print('textcnn')
        return textCNN(classes=num_class,maxword=max_words)
    elif model_name=='rnn':
        return SeqRNN(classes=num_class)
    # elif model_name=='bilstm':
    #     return smallRNN(classes=num_class,bidirection=True,length=max_words)
    elif model_name=='bilstm':
        print('TextBiLSTM')
        return TextBILSTM(num_class=num_class,max_words=max_words)
    elif model_name=='fc':
        print('full connect')
        return fc(num_class=2,input_size=num_class)
    else:
        print('the name of model is error!!')

class fc(nn.Module):
    def __init__(self,num_class=2,input_size=4):
        super().__init__()
        self.linear1=nn.Linear(input_size,16)
        self.linear2=nn.Linear(16,8)
        self.linear3 = nn.Linear(8, num_class)
    def forward(self,x):
        pred1=torch.sigmoid(self.linear1(x))
        pred2=torch.sigmoid(self.linear2(pred1))
        pred3 = torch.sigmoid(self.linear3(pred2))
        return pred3
class textCNN(nn.Module):
    def __init__(self,classes=5,kernel_num=16,kernel_size=[3,4,5],embed_dim=300,dropout=0.3,maxword=80000):
        super(textCNN,self).__init__()
        ci=1
        self.embed=nn.Embedding(maxword,embed_dim,padding_idx=1)
        self.conv11=nn.Conv2d(ci,kernel_num,(kernel_size[0],embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout=nn.Dropout(dropout)
        self.fc1=nn.Linear(len(kernel_size)*kernel_num,classes)

    @staticmethod
    def conv_and_pool(x,conv):
        x=conv(x)
        x=F.relu(x.squeeze(3))
        x=F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        x=self.embed(x)
        x=x.unsqueeze(1)
        x1=self.conv_and_pool(x,self.conv11)
        x2=self.conv_and_pool(x,self.conv12)
        x3=self.conv_and_pool(x,self.conv13)
        x=torch.cat((x1,x2,x3),1)
        logit=F.log_softmax(self.fc1(x),dim=1)
        return logit

class wordCNN(nn.Module):
    def __init__(self,classes=5,num_features=200,dropout=0.3,maxword=80000):
        super(wordCNN,self).__init__()
        self.embed=nn.Embedding(maxword,num_features,padding_idx=1)
        self.conv1=nn.Sequential(
            nn.Conv1d(num_features,256,kernel_size=7,stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=3)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),

        )
        self.conv4= nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1=nn.Sequential(
            nn.Linear(768,512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc3=nn.Linear(256,classes)
        self.log_softmax=nn.LogSoftmax()

    def forward(self, x):
        x=self.embed(x)
        x=x.transpose(1,2)
        x=self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x=self.log_softmax(x)
        return x


class SeqRNN(nn.Module):
    def __init__(self,embed_dim=300,hidden_size=128,classes=5,dropout=0.3,maxwords=80000):
        super(SeqRNN,self).__init__()
        self.embed_dim=embed_dim
        self.hidden_size=hidden_size
        self.num_class=classes
        self.embed=nn.Embedding(maxwords,embed_dim,padding_idx=1)
        self.rnn=nn.RNN(self.embed_dim,self.hidden_size,batch_first=True,dropout=dropout)
        self.linear=nn.Linear(self.hidden_size,self.num_class)

    def forward(self,x):
        h0 = torch.zeros(1, x.shape[0],self.hidden_size )
        h0=Variable(h0).to(device)
        x=self.embed(x)
        x,hidden=self.rnn(x,h0)
        x=x[:,-1,:]
        x=self.linear(x)
        x=F.softmax(x,dim=1)
        return x


class smallRNN(nn.Module):
    def __init__(self, classes=2, bidirection=False, layernum=3, length=80000, embedding_size=300, hiddensize=300):
        super(smallRNN, self).__init__()
        self.embd = nn.Embedding(length, embedding_size,padding_idx=1)
        # self.lstm = nn.LSTMCell(hiddensize, hiddensize)
        self.lstm = nn.LSTM(embedding_size, hiddensize, layernum, dropout=0.5, bidirectional=bidirection)
        self.hiddensize = hiddensize
        numdirections = 1 + bidirection
        self.hsize = numdirections * layernum
        self.linear = nn.Linear(hiddensize * numdirections, classes)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x, returnembd=False):
        embd = self.embd(x)
        if returnembd:
            embd = Variable(embd.data, requires_grad=True).cuda()
            embd.retain_grad()
            # print embd.size()
        h0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        c0 = Variable(torch.zeros(self.hsize, embd.size(0), self.hiddensize)).to(device)
        # for inputs in x:
        x = embd.transpose(0, 1)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = x[-1]
        # x = x[-1].transpose(0,1)
        # x = x.view(x.size(0),-1)
        x = self.log_softmax(self.linear(x))
        if returnembd:
            return embd, x
        else:
            return x


class TextBILSTM(nn.Module):

    def __init__(self,num_class=4,max_words=80000,dropout=0.3,embedding_size=50,hidden_size=50,rnn_layers=3):
        super(TextBILSTM, self).__init__()
        self.num_classes = num_class
        self.keep_dropout = dropout
        self.embedding_size = embedding_size
        # self.l2_reg_lambda = config.l2_reg_lambda
        self.hidden_dims = hidden_size
        self.word_size=max_words
        self.rnn_layers = rnn_layers

        self.build_model()

    def build_model(self):
        # 初始化字向量
        self.embeddings = nn.Embedding(self.word_size, self.embedding_size)
        # 字向量参与更新
        self.embeddings.weight.requires_grad = True

        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embedding_size, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.num_classes)
        )

    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, x):
        # char_id = torch.from_numpy(np.array(input[0])).long()
        # pinyin_id = torch.from_numpy(np.array(input[1])).long()

        sen_input = self.embeddings(x)

        # input : [len_seq, batch_size, embedding_dim]
        sen_input = sen_input.permute(1, 0, 2)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_input)
        # output : [batch_size, len_seq, n_hidden * 2]
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)
        # atten_out = self.attention_net(output, final_hidden_state)
        atten_out = self.attention_net_with_w(output, final_hidden_state)
        return self.fc_out(atten_out)


