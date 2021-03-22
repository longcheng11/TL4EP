###############################################################################
# File name: LSTMAtt.py                                                       #
# Author: N. Weijian, S. Yujian, L. Tong, Z. Qingtian and L. Cong.            #
# Submission: DCU MCM Practicum                                               #
# Instructor: Long Cheng                                                      #
# Description: This code implements a customisable LSTM in Pytorch with a     #
#                  bidirectional module.                                      #
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
class BiLSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None):
        super(BiLSTM, self).__init__()
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
        self.rnn = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, dropout = self.dropout,
                               num_layers = self.n_layer, bidirectional=True).cuda() ######################################################
        self.hidden_state = Variable(
            torch.randn(self.n_layer * 2, self.batch_size, self.hidden_dim)).cuda()
        self.cell_state = Variable(
            torch.randn(self.n_layer * 2, self.batch_size, self.hidden_dim)).cuda()
        self.out = nn.Linear(hidden_dim * 2, out_size).cuda()
    def forward(self, X):
        input = self.embedding(X)
        input = input.permute(1, 0, 2)

        output, (final_hidden_state, final_cell_state) = self.rnn(input, (self.hidden_state, self.cell_state))
        hn = output[-1]
        output = self.out(hn)
        return  output # model : [batch_size, num_classes], attention : [batch_size, n_step]
