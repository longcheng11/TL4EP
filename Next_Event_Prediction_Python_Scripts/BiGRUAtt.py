###############################################################################
# File name: LSTMAtt.py                                                       #
# Author: N. Weijian, S. Yujian, L. Tong, Z. Qingtian and L. Cong.            #
# Submission: DCU MCM Practicum                                               #
# Instructor: Long Cheng                                                      #
# Description: This code implements a customisable GRU in Pytorch with a      #
#                  birectional and attention mechanism module.                #
# Disclaimer: The code in this file is based on the works "Business Process   # 
#    Instance Remaining Time Prediction Using Deep Transfer Learning"         #
#    by N. Weijian, S. Yujian, L. Tong, Z. Qingtian and L. Cong.              #
###############################################################################

import numpy as np
import torch
import torch.nn as nn
import gensim
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
class BiGRUAtt(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None):
        super(BiGRUAtt, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_shape = out_size
        self.embedding = embedding
        self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim * 2, hidden_dim * 2).cuda()).cuda()
        self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim * 2, n_layer).cuda()).cuda()
        print('Initialization BiGRU Model')
        self.att_hidden_dim = 10
        self.att_b_dim = 1
        self.att_w = nn.Parameter(torch.Tensor(hidden_dim * 2, self.att_hidden_dim).cuda()).cuda()
        self.att_g = nn.Parameter(torch.Tensor(self.att_hidden_dim, 1).cuda()).cuda()
        self.att_b = nn.Parameter(torch.Tensor(self.att_hidden_dim, 1).cuda()).cuda()
        self.rnn = nn.GRU(input_size = embedding_dim, hidden_size = hidden_dim, dropout = self.dropout,
                               num_layers = self.n_layer, bidirectional=True).cuda()
        self.out = nn.Linear(hidden_dim * 2, out_size).cuda()
    def traditional_attention_net(self, rnn_output):

        attn_weights = torch.matmul(rnn_output,self.weight_Mu).cuda()
        #print(attn_weights.size())
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, hidden_dim * num_directions(=2)]
    def attention_net(self, rnn_output):
        attn_weights = torch.matmul(rnn_output, self.att_w).cuda()
        #print(attn_weights.size(), self.att_b.size())
        attn_weights = attn_weights + self.att_b
        attn_weights = torch.tanh(attn_weights)
        attn_weights = torch.matmul(attn_weights,self.att_g)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(rnn_output.transpose(1, 2), soft_attn_weights).squeeze(2).cuda()
        return context, soft_attn_weights.data.cpu().numpy()  # context : [batch_size, hidden_dim * num_directions(=2)]
    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)
        time_length = input.size()[0]
        if time_length != self.att_b.size()[0]:
            self.att_b = nn.Parameter(torch.Tensor(time_length, self.att_b_dim).cuda()).cuda()
        hidden_state = Variable(
            torch.randn(self.n_layer * 2, self.batch_size, self.hidden_dim)).cuda()
        output,final_hidden_state = self.rnn(input, hidden_state)
        output = output.permute(1, 0, 2)  # output : [batch_size, len_seq, hidden_dim]
        output, attention = self.attention_net(output)
        output = self.out(output)
        return  output # model : [batch_size, num_classes], attention : [batch_size, n_step]
