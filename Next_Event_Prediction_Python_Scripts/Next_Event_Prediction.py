###############################################################################
# File name: Next_Event_Prediction.py                                         #
# Author: Mary Murphy                                                         #
# Submission: DCU MCM Practicum                                               #
# Instructor: Long Cheng                                                      #
# Description: This code trains a varity of RNN models to predict the next    #
#     event of a running trace, using transfer learning and trace clustering  # 
#     to fully utilise the trace variation in the data to improve prediction  # 
#     accuarcy.                                                               #
# Disclaimer: The code in this file is based on the works "Business Process   # 
#    Instance Remaining Time Prediction Using Deep Transfer Learning"         #
#    by N. Weijian, S. Yujian, L. Tong, Z. Qingtian and L. Cong.              #
###############################################################################

##################################################################################################################################################################

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True) 

##################################################################################################################################################################

pip install import_ipynb #Package requirement for code

##################################################################################################################################################################

pip install pm4pyclustering #Package requirement for code

##################################################################################################################################################################

# Code to read csv file into Colaboratory:
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# NB - Google drive shareable link for each python file required. This is different for every drive

GRU = drive.CreateFile({'id':'1XheD3ckzdeUrukYzj0jeINsKhaFIrKTG'}) # https://colab.research.google.com/drive/1XheD3ckzdeUrukYzj0jeINsKhaFIrKTG?usp=sharing
BiGRU = drive.CreateFile({'id':'14KOduMX_vPFOpTrytTqorr_g_lelHQM3'}) # https://colab.research.google.com/drive/14KOduMX_vPFOpTrytTqorr_g_lelHQM3?usp=sharing
BiGRUAtt = drive.CreateFile({'id':'1rOeK2LIb0KadAYRz1MsDbMAliQORI7F9'}) # https://colab.research.google.com/drive/1rOeK2LIb0KadAYRz1MsDbMAliQORI7F9?usp=sharing
BiLSTM = drive.CreateFile({'id':'1b7OkJFVdpdArm6tkHJ5baQb8QHzM3kgz'}) # https://colab.research.google.com/drive/1b7OkJFVdpdArm6tkHJ5baQb8QHzM3kgz?usp=sharing
BiLSTMAtt = drive.CreateFile({'id':'10lkacFL-pjUrZN4xrAOekVcvS8ijiHh3'}) # https://colab.research.google.com/drive/10lkacFL-pjUrZN4xrAOekVcvS8ijiHh3?usp=sharing
GRUAtt = drive.CreateFile({'id':'172uBGbtBXPbGQeAYZ2u7kA7bgHhXLDDj'}) # https://colab.research.google.com/drive/172uBGbtBXPbGQeAYZ2u7kA7bgHhXLDDj?usp=sharing
LSTM = drive.CreateFile({'id':'1znHsM5fzJ9GiRjtrwlpMyCFrvZ_me6a7'}) # https://colab.research.google.com/drive/1znHsM5fzJ9GiRjtrwlpMyCFrvZ_me6a7?usp=sharing
LSTMAtt = drive.CreateFile({'id':'12jFEIOT5LrMe-8lX7WEyWRIs9dxksPg_'}) #https://colab.research.google.com/drive/12jFEIOT5LrMe-8lX7WEyWRIs9dxksPg_?usp=sharing
input_data = drive.CreateFile({'id':'1O3KLuOPf-Yrryb7TbuX8rva3hjhe343w'}) # https://colab.research.google.com/drive/1O3KLuOPf-Yrryb7TbuX8rva3hjhe343w?usp=sharing

GRU.GetContentFile('GRU.ipynb')
BiGRU.GetContentFile('BiGRU.ipynb')
BiGRUAtt.GetContentFile('BiGRUAtt.ipynb')
BiLSTM.GetContentFile('BiLSTM.ipynb')
BiLSTMAtt.GetContentFile('BiLSTMAtt.ipynb')
GRUAtt.GetContentFile('GRUAtt.ipynb')
LSTM.GetContentFile('LSTM.ipynb')
LSTMAtt.GetContentFile('LSTMAtt.ipynb')
input_data.GetContentFile('input_data.ipynb')

##################################################################################################################################################################

#File import test

import import_ipynb
from GRU import GRU

##################################################################################################################################################################

###############################
#  Model evaulation functions #
###############################

def evaluate(model, test_batchs):
    target_list = list()
    predict_list = list()
    for (input, target) in test_batchs:
        input = np.array(input)
        input = Variable(torch.LongTensor(input).cuda()).cuda()
        prediction = model(input)
        predict_list += [pdic.item() for pdic in prediction]
        target_list += target
    MSE = computeMSE(target_list,predict_list)
    MAE = computeMAE(target_list,predict_list)
    RMSE = sqrt(MSE)
    TOTAL = computeTOTAL(target_list,predict_list)
    MEAN = computeMEAN(target_list,predict_list)
    return MSE,MAE,RMSE,TOTAL,MEAN
def computeMAE(list_a,list_b):
    MAE_temp = []
    for num in range(len(list_a)):
        MAE_temp.append(abs(list_a[num]-list_b[num]))
    MAE = sum(MAE_temp)/len(list_a)
    return MAE
def computeMSE(list_a,list_b):
    MSE_temp = []
    for num in range(len(list_a)):
        MSE_temp.append((list_a[num] - list_b[num]) * (list_a[num] - list_b[num]))
    MSE = sum(MSE_temp) / len(list_a)
    return MSE
def computeTOTAL(list_a,list_b):
    TOTAL_temp = []
    for num in range(len(list_a)):
        TOTAL_temp.append(abs(list_a[num] - list_b[num]))
    TOTAL = sum(TOTAL_temp)
    return TOTAL
def computeMEAN(list_a,list_b):
    MEAN_temp = []
    for num in range(len(list_a)):
        MEAN_temp.append(abs(list_a[num] - list_b[num]))
    MEAN = sum(MEAN_temp)/len(list_a)
    return MEAN

##################################################################################################################################################################

# -*- coding: utf-8 -*-

##########################################################################################
#  This code implements traditional model training and testing with no data partitioning #
##########################################################################################

from GRU import GRU
from GRUAtt import GRUAtt
from BiGRU import BiGRU
from BiGRUAtt import BiGRUAtt
from LSTM import LSTM
from LSTMAtt import LSTMAtt
from BiLSTM import BiLSTM
from BiLSTMAtt import BiLSTMAtt
from input_data import InputData
from collections import deque
import numpy as np
import os
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


print(torch.cuda.is_available()) # Test if GPU available

os.chdir("/content/gdrive/My Drive/Practicum/Code/ActivityPrediction") #Load code local directory
os.listdir()


##############################
#  Set model hyperparameters #
##############################

embd_dimension = 3 #The number of expected features in the input x
hidden_dim=5 #The number of features in the hidden state h of the model
learn_rate = 0.01 # The amount that the weights are updated during training
n_layer=1 #Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
dropout=0 # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
out_size = 1 # size of each output 

learn_rate_down = 0.001 #learn deacy rate
learn_rate_min = 0.0001 # min learn rate
loss_deque = deque(maxlen=20) #double ended list for recording loss with each epoch
loss_change_deque = deque(maxlen=30) #double ended list for recording loss changes with each epoch
loss_change = 0 #intialise loss change variable

#############################
#  Set descriptor variables #
#############################


optim_type= 'Adam' # Optiizer algorithm used
model_type='BiGRU' # REMEBER TO CHANGE THIS 
train_type = 'mix' #iteration single. Independently train multiple models (referred to as Sep) on each length trajectory prefix data set;
loss_type= 'L1Loss' # criterion used

###########################
#  Select data for import #
###########################

data_name='Helpdesk'
#data_name='Receipts'
#data_name='Hospital Billing'


######################################
#  Setting training phase parameters #
######################################

start_pos=3 #smallest trace length/cluster to train or test model on.
stop_pos=10 #largest/last trace length/cluster to train or test model on.
epoch = 0 #training epoch counter
max_epoch_num=150 #number of training epoch
train_splitThreshold=0.7
batch_size=256 


######################
# Process Input data #
######################

#data = InputData('./data/Hospital Billing.csv', embd_dimension) 
data = InputData('./data/Helpdesk.csv', embd_dimension) 
data.encodeEvent()# encodes activity events using index based encoding

print("data.event2id",data.event2id)

data.encodeTrace() #joins up all the traces based on case. creates a list of lists. Each list is a trace. Each unique activity name has an index number.
print("data.encode_trace",data.encode_trace)
data.splitData(train_splitThreshold) #split training and test data
print("data.train_dataset",data.train_dataset)
data.initPrefixBucketing(start_pos)# passes the trace data for prefix bucketing. Pass shortest prefix.
#data.initDataClustering('./data/Hospital Billing.csv')#  passes the trace data for cluster bucketing. Pass raw event data ######################################################################################

#print("data.train_singleLengthData", data.train_partitionedData)
#print("data.test_singleLengthData", data.test_singleLengthData[start_pos])  #returns buckets of trace prefixes of the same length. For example, a complete trace consisting of three events would correspond to three traces in the prefix log – the partial trace after executing the first, the second and the third event. so it can be in three different buckets
#print(data.train_singleCluster) #returns buckets of trace prefixes of the same length. For example, a complete trace consisting of three events would correspond to three traces in the prefix log – the partial trace after executing the first, the second and the third event. so it can be in three different buckets

if train_type == 'mix':
    data.generateNoPartitionBatch(batch_size) #generate batch of mixed traces
else:
    #data.generatePartitionedBatch(batch_size, start_pos) #returns the traces from cluster/prefix bucket at start position
    data.generatePartitionedBatch(batch_size,start_pos) #returns the bucket with the certain prefix length 

print("data.train_batch_mix",data.train_batch_mix)


#######################################################
# Initialise prediction model, optimzer and criterion #
#######################################################

#model=GRU(vocab_size=data.vocab_size, embedding_dim = embd_dimension, hidden_dim=hidden_dim, out_size=1,
   #                 batch_size=256, n_layer=1 ,dropout=0, embedding=data.embedding)

#model=LSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size, batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=BiGRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
   #                      batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)

#model=BiLSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
  #                        batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=BiLSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                  #     batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=GRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
    #                   batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=LSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                  #      batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

model=BiGRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                      batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)
    
optimizer = optim.Adam(model.parameters(), lr= 0.01) #Initialise Adam Optimizer
criterion = nn.L1Loss().cuda() # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.

model_detail = 'embdDim' + str(embd_dimension) + '_loss' + loss_type + '_optim' + optim_type + '_hiddenDim' \
                  + str(hidden_dim) + '_startPos' + str(start_pos) + '_trainType' + train_type + '_nLayer' + str(n_layer) \
                  + '_dropout' + str(dropout) #string of model information

start_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)') #current start time


###############################################
#  Set up training and results file directory #
###############################################
model_save_folder='./model/'
save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detail + '/'
train_record_folder='./train_record/'
result_save_folder='./result/'
save_record_all = train_record_folder + data_name +'_sum.csv'
save_record_single = train_record_folder + data_name + '/' + model_type + '/' + model_detail + '/' 

save_result_folder = result_save_folder + data_name + '/' + model_type + '/'

for folder in [save_record_single]: 
  if not os.path.exists(folder):
      os.makedirs(folder)#creates results directory 
  save_record_single = save_record_single + start_time + '.csv' 
if not os.path.exists(save_record_all): #creat train directory
  save_record_all_open = open(save_record_all, 'a', encoding='utf-8') #creates a file with train directory (train_record)
  save_record_all_write = 'modelType,embdDim,lossType,optimType,hiddenDim,startPos,trainType,layerNum,' \
                          'dropout,epoch,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss,modelFile,recordFile,resultFile\n' #the headers, same as result file excluding "prefixLength" column
  save_record_all_open.writelines(save_record_all_write)#put the headers in file
  save_record_all_open.close()#close file
save_record_single_open = open(save_record_single,'w',encoding='utf-8')
save_record_single_write = 'epoch,startPos,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
save_record_single_open.writelines(save_record_single_write)



##############################
# Model training and testing #
##############################

if train_type != 'iteration':
  while epoch < max_epoch_num and learn_rate >= learn_rate_min:
      #print(epoch)
      total_loss = torch.FloatTensor([0])
      for (input, target) in data.train_batch: #input is the trace prefix, target is the last event of trace. will be a mixed batch if training type mixed
          optimizer.zero_grad()
          input = np.array(input)
          target = np.array([[t] for t in target])
          input = Variable(torch.LongTensor(input).cuda()).cuda() #turn input to a tensor
          target = Variable(torch.LongTensor(target).cuda()).cuda()
          target = target.float() #turns target times to float
          output = model(input) #predicted times. Model retains params initialised from previous epoch
          loss = criterion(output, target)# compare to actual
          loss.backward(retain_graph=True) #gradients are computed. update the parameters based on the computed gradients.
          optimizer.step() #updates the model parameters
          #total_loss += loss.data
      loss_deque.append(total_loss.item())
      loss_change_deque.append(total_loss.item())
      loss_change = total_loss.item() - sum(loss_deque) / len(loss_deque)
      loss_change = abs(loss_change)
      MSE, MAE, RMSE, TOTAL, MEAN = evaluate(model,data.test_batch) #evaluates model and runs on test batch
      if loss_change < 10 and len(loss_deque) < 20: #Once loss stabalises
          now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
          model_save = save_model_folder + now_time + '.pth'
          result_save_file = result_save_folder + 'epoch' + str(epoch) + now_time + '.csv'
          if not os.path.exists(result_save_folder):
                        os.makedirs(result_save_folder)
          result_save_open = open(result_save_file,'w',encoding='utf-8') 
          result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
          #This paper uses the mean absolute error (Mean Absolute Error, MAE) is used as an evaluation index for each method. The absolute value of the difference between the real value and the predicted 
          #value of the prefix remaining execution time is used to measure the accuracy of the remaining time prediction. The lower the MAE value, the more remaining The more accurate the remaining time 
          #prediction is. 
          result_save_open.writelines(result_save_write) # save evualation metrics to file
          result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
          + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
              total_loss.item()) + '\n'
          result_save_open.writelines(result_save_write)
          for prefix_length in range(start_pos,stop_pos + 1): #here the model is tested on each prefix length
              data.generateNoPartitionBatch(batch_size) # test model on a mix batch of traces
              if len(data.test_batch) != 0:
                  MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch) # evaluate model test
                  result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                      learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                      + ',' + str(RMSE1) + ',' + str(
                      total_loss.item() / len(data.train_batch)) + ',' + str(
                      total_loss.item()) + '\n' #save evaluation details to file
              else:
                  result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                      learn_rate) + ',' + 'No test data' + ',' + 'No test data' \
                                      + ',' + 'No test data' + ',' + str(
                      total_loss.item() / len(data.train_batch)) + ',' + str(
                      total_loss.item()) + '\n'
              result_save_open.writelines(result_save_write)
          result_save_open.close()
          if train_type == 'mix': #generate another batch of traces to train on next epoch
              data.generateNoPartitionBatch(batch_size) #mixes up the training data for next epoch
          else:
              #data.generatePartitionedBatch(batch_size, start_pos) #resets data back to prefix 3 batches for next epoch ###################################################### change
              data.generatePartitionedBatch(batch_size, start_pos)
          save_record_all_open = open(save_record_all,'a',encoding='utf-8') #saves details on final training params and evaluation
          save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                  +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos) \
                                  +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                  +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                  +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                  +','+ model_save +',' + save_record_single +',' + result_save_file + '\n'

          save_record_all_open.writelines(save_record_all_write)
          save_record_all_open.close()
          if learn_rate > learn_rate_down: #learning rate decayed
              learn_rate = learn_rate - learn_rate_down
          else:
              learn_rate_down = learn_rate_down * 0.1
              learn_rate = learn_rate - learn_rate_down
          optimizer = optim.Adam(model.parameters(), lr=learn_rate) #update the model hyperparameters
          loss_deque = deque(maxlen=20) #Losses stored
          loss_deque.append(total_loss.item())
      if len(loss_change_deque) == 30 and (max(loss_change_deque) - min(loss_change_deque) < 20): #Once loss stabalises to this level. Same code descriptions as above section.
          now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
          model_save = save_model_folder + now_time + '.pth'
          result_save_file = result_save_folder + now_time + '.pth'
          result_save_open = open(result_save_file,'w',encoding='utf-8')
          result_save_write = 'epoch,startPos,prefixLength,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
          result_save_open.writelines(result_save_write)
          result_save_write = str(epoch) + ',' + str(start_pos) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
          + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
              total_loss.item()) + '\n'  # save evualation metrics to file
          result_save_open.writelines(result_save_write)
          for prefix_length in range(start_pos,stop_pos + 1):
              #data.generatePartitionedBatch(batch_size, prefix_length) ###################################################### change
              data.generateNoPartitionBatch(batch_size) # testing on mixed batches
              if len(data.test_batch) != 0:
                  MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                  result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                      learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                      + ',' + str(RMSE1) + ',' + str(
                      total_loss.item() / len(data.train_batch)) + ',' + str(
                      total_loss.item()) + '\n'
              else:
                  result_save_write = str(epoch) + ',' + str(start_pos) + ',' + str(prefix_length) + ',' + str(
                      learn_rate) + ',' + 'No test data' + ',' + 'No test data' \
                                      + ',' + 'No test data' + ',' + str(
                      total_loss.item() / len(data.train_batch)) + ',' + str(
                      total_loss.item()) + '\n'
              result_save_open.writelines(result_save_write)
          result_save_open.close()
          if train_type == 'mix':
              data.generateNoPartitionBatch(batch_size) #generates new batches of mixed prefix length ###################################################### change
           
          else:
              data.generatePartitionedBatch(batch_size, start_pos) #generates new batches of a certain prefix "start_pos" ##################################################### change
             # data.generatePartitionedBatch(batch_size, start_pos)
          save_record_all_open = open(save_record_all,'a',encoding='utf-8')
          save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                  +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos) \
                                  +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                  +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                  +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                  +','+ model_save +',' + save_record_single + ',' + result_save_file + '\n'
          save_record_all_open.writelines(save_record_all_write)
          save_record_all_open.close()
          if learn_rate > learn_rate_down:
              learn_rate = learn_rate - learn_rate_down
          else:
              learn_rate_down = learn_rate_down * 0.1
              learn_rate = learn_rate - learn_rate_down
          optimizer = optim.Adam(model.parameters(), lr=learn_rate)
          loss_change_deque = deque(maxlen=30)
          loss_change_deque.append(total_loss.item())
      save_record_single_write = str(epoch) + ','+ str(start_pos) +','+ str(learn_rate) + ','+ str(MSE) +','+ str(MAE)\
                              +','+ str(RMSE) + ','+ str(total_loss.item()/len(data.train_batch)) + ','+ str(total_loss.item()) + '\n'
      save_record_single_open.writelines(save_record_single_write)
      epoch = epoch + 1 #increase epoch count
  save_record_single_open.close() 

##################################################################################################################################################################

#############################################################################################################################
# This code implements a transfer learning framework with data partitioning to train and test next event prediction models  #
#############################################################################################################################

#coding: utf-8
from GRU import GRU
from GRUAtt import GRUAtt
from BiGRU import BiGRU
from BiGRUAtt import BiGRUAtt
from LSTM import LSTM
from LSTMAtt import LSTMAtt
from BiLSTM import BiLSTM
from BiLSTMAtt import BiLSTMAtt
from input_data import InputData
from collections import deque
import numpy as np
import os
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
#train_type = single iteration mix

print(torch.cuda.is_available())

os.chdir("/content/gdrive/My Drive/Practicum/Code/EndTimePrediction")
os.listdir()

#############################
# Set Model hyperparameters #
#############################

embd_dimension = 2 # The number of expected features in the input x
hidden_dim=5 # The number of features in the hidden state h
learn_rate = 0.01 
n_layer=1  #Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the 
            #final results. Default: 1
dropout=0
out_size = 1

learn_rate_backup = learn_rate
learn_rate_down = 0.001 # amount by which learning rate is decayed by with each epoch
learn_rate_min = 0.0001 #minimum allowed learning rate
loss_deque = deque(maxlen=20)
loss_change_deque = deque(maxlen=30)
loss_change = 0

#learn_rate = 0.01 Initialised in next cell!

############################
# Set descriptor variables #
############################

loss_type= 'L1Loss'
optim_type= 'Adam'
train_type = 'iteration' #iteration mix
model_type='BiGRU' # REMEBER TO CHANGE THIS 

####################################
# Select event log data for import #
####################################
  
data_name='Helpdesk'
#data_name='Receipts'
#data_name='Hospital Billing'

#################################
# Set training phase parameters #
#################################

start_pos=3 # define starting cluster/prefix bucket
stop_pos=20 #largest/last trace length/cluster to train or test model on.
epoch = 0
max_epoch_num=150
train_splitThreshold=0.7
batch_size=256

######################
# Process input data #
######################

data = InputData('./data/Helpdesk.csv', embd_dimension = 2) ###################################################################################### cleaned_receipt process  event log
#data = InputData('./data/Receipts.csv', embd_dimension = 2) 
#data = InputData('./data/Hospital Billing.csv', embd_dimension = 2)

data.encodeEvent()
print("data.event2id",data.event2id)
data.encodeTrace()
print("data.encode_trace",data.encode_trace)
data.splitData(train_splitThreshold)
print("data.train_dataset",data.train_dataset)
#data.initDataClustering('./data/Receipts.csv') ############################################### 

###############################################
# Choose Trace Clustering or Prefix Bucketing #
###############################################

data.initPrefixBucketing(start_pos) # for prefix bucketing with event prediction

# set pca_components and dbscan_eps for dataset
#data.initDataClustering('./data/Helpdesk.csv') # for trace clustering with event prediction

print("data.train_partitionedData", data.train_partitionedData)
print("data.test_partitionedData", data.test_partitionedData)
if train_type == 'mix':
    data.generateNoPartitionBatch(batch_size)
else:
    data.generatePartitionedBatch(batch_size,start_pos) # initialise first trace cluster/prefix bucket


##############################################
# Set up training and results file directory #
##############################################
save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detail + '/'
train_record_folder='./train_record/'
result_save_folder='./result/'
save_record_all = train_record_folder + data_name +'_sum.csv'
save_record_all_open = open(save_record_all, 'a', encoding='utf-8')

result_save_folder='./result/'

##################################################################################################################################################################


##############################
# Cluster Densities Analysis #
##############################

# If low amount of dense clusters, run above cell again

i=0
while i<20:
  try:
    print (i, len(data.train_partitionedData[i]), data.test_partitionedData[i])             
  except BaseException as e:
    print(i)

  i=i+1
print("   ")
i=0
while i<20:
  try:
    print (i, len(data.test_partitionedData[i]), data.test_partitionedData[i])             
  except BaseException as e:
    print(i)

  i=i+1

##################################################################################################################################################################

###################################################
# Set training phase parameters undergoing tuning #
###################################################

start_pos=1 # define starting cluster/prefix bucket
stop_pos=10 # largest/last trace length/cluster to train or test model on.
learn_rate = 0.01
model_type='BiLSTMAtt' # REMEBER TO CHANGE THIS - current model type being train in next event prediction

#######################################################
# Initialise prediction model, optimzer and criterion #
####################################################### 


#model=GRU(vocab_size=data.vocab_size, embedding_dim = 2, hidden_dim=5, out_size=1,
#                  batch_size=256, n_layer=1 ,dropout=0, embedding=data.embedding)

#model=LSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size, batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=BiGRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
#                   batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)

model=BiLSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                      batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=BiLSTM(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
#                 batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=GRUAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size, batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=LSTMAtt(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
#                  batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)

#model=BiGRU(vocab_size=data.vocab_size, embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
#                 batch_size=batch_size,n_layer=n_layer,dropout=dropout, embedding=data.embedding)   

optimizer = optim.Adam(model.parameters(), lr= 0.01) #Initialise Adam Optimizer

criterion = nn.L1Loss().cuda()  # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.

start_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)') #current start time
model_detail = 'embdDim' + str(embd_dimension) + '_loss' + loss_type + '_optim' + optim_type + '_hiddenDim' \
                  + str(hidden_dim) + '_startPos' + str(start_pos) + '_trainType' + train_type + '_nLayer' + str(n_layer) \
                  + '_dropout' + str(dropout) #string of model information

###############################################
#  Set up training and results file directory #
###############################################
save_model_folder = model_save_folder + data_name + '/' + model_type + '/' + model_detail + '/'
save_record_all = train_record_folder + data_name +'_sum.csv' # saves all records?
save_record_single = train_record_folder + data_name + '/' + model_type + '/' + model_detail + '/'
save_result_folder = result_save_folder + data_name + '/' + model_type + '/' + model_detail + '/'
for folder in [save_record_single]: #make a csv for each record single?
  if not os.path.exists(folder):
      os.makedirs(folder)
  save_record_single = save_record_single + start_time + '.csv'
if not os.path.exists(save_record_all):
  save_record_all_open = open(save_record_all, 'a', encoding='utf-8')
  save_record_all_write = 'modelType,embdDim,lossType,optimType,hiddenDim,startPos,trainType,layerNum,' \
                          'dropout,epoch,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss,modelFile,recordFile,resultFile\n'
  save_record_all_open.writelines(save_record_all_write)
  save_record_all_open.close()
save_record_single_open = open(save_record_single,'w',encoding='utf-8')
save_record_single_write = 'epoch,Train_start_pos,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
save_record_single_open.writelines(save_record_single_write)


##################
# Training Phase #
##################

if train_type == 'iteration':
    for start_pos_temp in range(start_pos,stop_pos+1): # Start transfer learning method for model training. 
      if start_pos_temp !=4 and start_pos_temp !=9 and start_pos_temp !=10: # filter for low density clusters
        epoch = 1
        learn_rate = learn_rate_backup 
        while epoch < max_epoch_num and learn_rate >= learn_rate_min:
            total_loss = torch.FloatTensor([0])
            for (input, target) in data.train_batch: # train model. Input is the trace prefix, target is the last event of trace. will be a mixed batch if training type mixed
                optimizer.zero_grad()
                input = np.array(input)                   
                target = np.array([[t] for t in target])            
                #target = np.array(target)
                input = Variable(torch.LongTensor(input).cuda()).cuda()
                target = Variable(torch.LongTensor(target).cuda()).cuda()
                output = model(input)
                loss = criterion(output, target).cuda()
                loss.data = loss.data.cuda()
                # print(loss.data.device)
                loss.backward(retain_graph=True)
                optimizer.step()
                #total_loss += loss.data
            loss_deque.append(total_loss.item())
            loss_change_deque.append(total_loss.item())
            loss_change = total_loss.item() - sum(loss_deque) / len(loss_deque)
            loss_change = abs(loss_change)
            MSE, MAE, RMSE, TOTAL, MEAN = evaluate(model,data.test_batch) # test and evaluate model
            if loss_change < 10 and len(loss_deque) == 20: #Once loss stabalises
                now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                model_save = save_model_folder + now_time + " cluster " + str(start_pos_temp) + " model" + '.pth'
                result_save_file = result_save_folder + 'length' + str(start_pos_temp) + 'epoch' + str(epoch) + now_time + '.csv'
                if not os.path.exists(result_save_folder):
                    os.makedirs(result_save_folder)
                print(result_save_file)
                result_save_open = open(result_save_file,'w',encoding='utf-8')
                result_save_write = 'epoch,start_pos,cluster,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                result_save_open.writelines(result_save_write)
                result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                    total_loss.item()) + '\n'
                result_save_open.writelines(result_save_write) # write model evaluation to results file
                for prefix_length in range(start_pos_temp,stop_pos + 1): #Test model on each prefix bucket/trace cluster
                  if prefix_length !=9 and prefix_length !=4 and prefix_length !=10: # low density clusters filter if needed
                #  if prefix_length !=8 and prefix_length !=9:
                    data.generatePartitionedBatch(batch_size,prefix_length) 
                    if len(data.test_batch) != 0:
                        MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                        result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                            + ',' + str(RMSE1) + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    else:
                        result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + 'No test data' + ',' + 'No test data' \
                                            + ',' + 'No test data' + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write) # write test evaluation to results file
                result_save_open.close()
                if train_type == 'mix':
                    data.generateNoPartitionBatch(batch_size) 
                else:
                    data.generatePartitionedBatch(batch_size, start_pos_temp) # generate batch of traces from next prefix length/ next cluster
                save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                        +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos_temp) \
                                        +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                        +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                        +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                        +','+ model_save +',' + save_record_single +',' + result_save_file + '\n'

                save_record_all_open.writelines(save_record_all_write) #save model information and evaluation
                save_record_all_open.close()
                if learn_rate > learn_rate_down:
                    learn_rate = learn_rate - learn_rate_down # decay learning rate
                else:
                    learn_rate_down = learn_rate_down * 0.1
                    learn_rate = learn_rate - learn_rate_down
                optimizer = optim.Adam(model.parameters(), lr=learn_rate) # set model weights
                loss_deque = deque(maxlen=20)
                loss_deque.append(total_loss.item())
            if len(loss_change_deque) == 30 and (max(loss_change_deque) - min(loss_change_deque) < 20): #Once loss stabalises
                now_time = datetime.now().strftime('%Y-%m-%d(%H-%M-%S)')
                model_save = save_model_folder + now_time + " cluster " + str(start_pos_temp) + " model" + '.pth'
                result_save_file = result_save_folder + 'length' + str(start_pos_temp) + 'epoch' + str(epoch) + now_time + '.csv'
                print(result_save_file)
                result_save_open = open(result_save_file,'w',encoding='utf-8')
                result_save_write = 'epoch,start_pos,cluster,learnRate,MSE,MAE,RMSE,meanLoss,totalLoss\n'
                result_save_open.writelines(result_save_write)
                result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + 'mix' + ',' + str(learn_rate) + ',' + str(MSE) + ',' + str(MAE) \
                + ',' + str(RMSE) + ',' + str(total_loss.item() / len(data.train_batch)) + ',' + str(
                    total_loss.item()) + '\n'
                result_save_open.writelines(result_save_write) # write initial evaluation to file.
                for prefix_length in range(start_pos_temp,stop_pos + 1):  #Test model on each prefix bucket/trace cluster
                  if prefix_length !=4 and prefix_length !=9 and prefix_length !=10: #low density cluster filters
              #    if prefix_length !=8 and prefix_length !=9:
                    data.generatePartitionedBatch(batch_size,prefix_length) 
                    if len(data.test_batch) != 0:
                        MSE1, MAE1, RMSE1, TOTAL1, MEAN1 = evaluate(model, data.test_batch)
                        result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + str(MSE1) + ',' + str(MAE1) \
                                            + ',' + str(RMSE1) + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    else:
                        result_save_write = str(epoch) + ',' + str(start_pos_temp) + ',' + str(prefix_length) + ',' + str(
                            learn_rate) + ',' + 'No test data' + ',' + 'No test data' \
                                            + ',' + 'No test data' + ',' + str(
                            total_loss.item() / len(data.train_batch)) + ',' + str(
                            total_loss.item()) + '\n'
                    result_save_open.writelines(result_save_write) #write test evaulation to file
                result_save_open.close()
                if train_type == 'mix':
                    data.generateNoPartitionBatch(batch_size)
                else:
                    data.generatePartitionedBatch(batch_size, start_pos_temp) # generate batch of traces from next prefix length/ next cluster
                save_record_all_open = open(save_record_all,'a',encoding='utf-8')
                save_record_all_write = model_type +','+ str(embd_dimension) +','+ loss_type \
                                        +','+ optim_type +','+ str(hidden_dim) +','+ str(start_pos_temp) \
                                        +','+ train_type +','+ str(n_layer) +','+ str(dropout) \
                                        +','+ str(epoch) +','+ str(learn_rate) +','+ str(MSE) +','+ str(MAE)\
                                        +','+ str(RMSE) +','+ str(total_loss.item()/len(data.train_batch)) +','+ str(total_loss.item())\
                                        +','+ model_save +',' + save_record_single + ',' + result_save_file + '\n'
                save_record_all_open.writelines(save_record_all_write) #save model information and evaluation
                save_record_all_open.close()
                if learn_rate > learn_rate_down:
                    learn_rate = learn_rate - learn_rate_down # decay learning rate
                else:
                    learn_rate_down = learn_rate_down * 0.1
                    learn_rate = learn_rate - learn_rate_down
                optimizer = optim.Adam(model.parameters(), lr=learn_rate) # set model weights
                loss_change_deque = deque(maxlen=30)
                loss_change_deque.append(total_loss.item())
            save_record_single_write = str(epoch) + ','+ str(start_pos_temp) +','+ str(learn_rate) + ','+ str(MSE) +','+ str(MAE)\
                                    +','+ str(RMSE) + ','+ str(total_loss.item()/len(data.train_batch)) + ','+ str(total_loss.item()) + '\n'
            save_record_single_open.writelines(save_record_single_write)
            epoch = epoch + 1
save_record_single_open.close()