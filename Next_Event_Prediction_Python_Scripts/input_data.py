###############################################################################
# File name: input_data.py                                                    #
# Author: Mary Murphy                                                         #
# Submission: DCU MCM Practicum                                               #
# Instructor: Long Cheng                                                      #
# Description: This code preprocesses the event log data before the model     #
#          training phase. It encodes activity names, joins traces in the     #
#          event log and carries out cluster or prefix bucketing.             #
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
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
import random
random.seed=13
class InputData():
    def __init__(self,data_address, embd_dimension = 2):
        self.embedding = None
        self.original_data = list()
        self.original_trace = list()
        self.encode_trace = list()
        self.train_dataset = list()
        self.test_dataset = list()
        self.train_noPartitionhData = list() #train_mixLengthData
        self.test_noPartitionData = list() #test_mixLengthData
        self.event2id = dict()
        self.id2event = dict()
        self.train_batch_mix = list()
        self.test_batch_mix = list()
        self.train_partitionedData = dict() #train_singleLengthData
        self.test_partitionedData = dict() #test_singleLengthData
        self.train_batch = dict()
        self.test_batch = dict()
        self.train_batch_single = dict()
        self.test_batch_single = dict()
        self.clusterDict = dict()

        self.vocab_size = 0
        self.train_maxLength = 0
        self.test_maxLength = 0
        self.embd_dimension = embd_dimension

        self.initData(data_address)
    def initData(self,data_address): # reads and delimits raw event log into a trace dictionary
        original_trace = list()
        record = list()
        trace_temp = list()
        with open(data_address, 'r', encoding='utf-8') as f:
            next(f)
            lines = f.readlines()
            for line in lines:
                record.append(line)  #all of event log stored in record (case, event, time)
        flag = record[0].split(',')[0] #stores the the first caseID value in flag
        for line in record:
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',') #split case, event, time line on commas
            if line[0] == flag:
                trace_temp.append([line[0], line[1], line[2]]) 
            else:
                flag = line[0]
                if len(trace_temp) > 0:
                    original_trace.append(trace_temp.copy()) #new completed trace added. original trace list does not contain caseID
                trace_temp = list()
                trace_temp.append([line[0], line[1], line[2]])
        self.original_data = record
        self.original_trace = original_trace
    def encodeEvent(self): # carries out index based encoding of activity names
        event2id = dict()
        id2event = dict()
        for line in self.original_data:
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',')
            try:
                event2id[line[1]] = event2id[line[1]]
                id2event[event2id[line[1]]] = id2event[event2id[line[1]]]
            except KeyError as ke:
                event2id[line[1]] = len(event2id)
                id2event[len(id2event)] = line[1]
        self.vocab_size = len(event2id)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embd_dimension, padding_idx= self.vocab_size).cuda()
        
        self.event2id = event2id
        self.id2event = id2event
  
    def encodeTrace(self): # Joins trace sequences in event log
        encode_trace = list()
        max = 0
        for line in self.original_trace:
            trace_temp = list()
            for line2 in line:
                trace_temp.append([line2[0], self.event2id[line2[1]], line2[2]]) #######
            if len(trace_temp) > max:
                max = len(trace_temp)
            encode_trace.append(trace_temp.copy())
        self.max = max
        self.encode_trace = encode_trace
    def splitData(self,train_splitThreshold = 1): # Splits data into training and test subsets
        self.train_dataset, self.test_dataset = train_test_split(self.encode_trace, train_size=train_splitThreshold, test_size=1-train_splitThreshold)

    def initDataClustering(self, path): #Carries out clustering on event log and creates traces prefix with end activity target pairs 
        
        train_partitionedData = dict() # to store traces in a partitioned manner (index for each group)
        test_partitionedData = dict() 
        train_noPartitionhData = list() # store traces such that no partition maintained
        test_noPartitionData = list()
        train_maxLength = 0
        test_maxLength = 0
        clusterDict = dict() # stores cluster dictionary

        #libraries for processing event logs
        from pm4pyclustering.algo.other.clustering import factory as clusterer
        from pm4py.objects.log.importer.xes import factory as xes_importer
        from pm4py.objects.log.importer.csv import factory as csv_importer
        from pm4py.objects.conversion.log import factory as conversion_factory
        import pandas as pd
        from pm4py.util import constants
        
        # Set PCA and DBSACN parameters
        parameters = {}
        parameters["pca_components"] = 3 
        parameters["dbscan_eps"] = 0.01 


        df = pd.read_csv(path) # Read in raw event log data
        #Rename Case, Event and timestamp attributes
        df.rename(columns={'CaseID': 'case:concept:name'}, inplace=True)
        df.rename(columns={'ActivityID': 'concept:name'}, inplace=True)
        df.rename(columns={'CompleteTimestamp': 'time:timestamp'}, inplace=True)
        log = conversion_factory.apply(df) # convert data to log format.
        clusters = clusterer.apply(log, parameters=parameters)
        for iteration, cluster in enumerate(clusters): #create cluster dictionary            
                for trace in cluster:
                    clusterDict.setdefault(iteration,[]).append(trace.attributes['concept:name']) 
        outercount = 0
        count = 0

        #########################
        #  Create training sets #
        #########################

        #Create (trace, final event) input, target pairs. Label each pair with cluster ID
        for line in self.train_dataset: # for each trace in batch     
            train_input_temp = list() #reset for each trace
            clusterId = 0
            count = 0
            for line2 in line: #for each event in trace         
                target_activity  = line2[1]
                if (len(train_input_temp)+1) == len(self.train_dataset[outercount]): #if on second last event in trace
                  # int(line2[0]) for int case IDs, str(line2[0]) for string case IDs
                  key = int(line2[0])####################################################################int(line2[0]) ##########################################################################str(line2[0])
                  for k in clusterDict.keys(): # find trace cluster ID in cluster dictionary
                      for v in clusterDict[k]:
                          if v == key:   
                            clusterId = k
                            break
                      else:
                             
                              continue
                           
                      break

                  try: # store trace, next event pair at cluster ID address
                    if len(train_input_temp) == 0:
                      train_partitionedData[clusterId].append((target_activity, target_activity)) 
                    else:
                      train_partitionedData[clusterId].append((train_input_temp.copy(), target_activity)) 

                  except BaseException as e:
                     if len(train_input_temp) == 0:
                       train_partitionedData[clusterId] = list()
                       train_partitionedData[clusterId].append((target_activity, target_activity))
                     else:
                       train_partitionedData[clusterId] = list()
                       train_partitionedData[clusterId].append((train_input_temp.copy(), target_activity))
                  
                  if len(train_input_temp) == 0: # append trace, next event pair to no partition dictionary
                    train_noPartitionhData.append((target_activity, target_activity))
                  else:                  
                    train_noPartitionhData.append((train_input_temp.copy(), target_activity))
                 
                
                else:
                  train_input_temp.append(line2[1]) #add event ID to output list
                  if len(train_input_temp) > train_maxLength:
                      train_maxLength = len(train_input_temp) #set trace length to current trace length if bigger than last. records max length of trace.
            
                  count = count+1
        outercount=outercount+1  

        #########################
        #  Create testing sets #
        #########################

        count = 0
        outercount = 0
        #Create (trace, final event) input, target pairs. Label each pair with cluster ID
        for line in self.test_dataset: # for each trace in batch
            
            test_input_temp = list() #reset for each trace
            clusterId = 0
            count = 0
            for line2 in line: #for each event in trace
           
                target_activity  = line2[1]
                if (len(test_input_temp)+1) == len(self.test_dataset[outercount]): #if on second last event in trace
                  # int(line2[0]) for int case IDs, str(line2[0]) for string case IDs
                  key = int(line2[0]) ####################################################################int(line2[0])############################################################### str(line2[0])

                  for k in clusterDict.keys(): # find trace cluster ID in cluster dictionary
                      for v in clusterDict[k]:
                          if v == key:
                            clusterId = k
                            break
                      else:
 
                              continue
    
                      break
                  
                  try:           # store trace, next event pair at cluster ID address                       
                    if len(test_input_temp) == 0:
                      test_partitionedData[clusterId].append((target_activity, target_activity))
                    else:
                      test_partitionedData[clusterId].append((test_input_temp.copy(), target_activity)) 
                  except BaseException as e:
                    if len(test_input_temp) == 0:
                      test_partitionedData[clusterId] = list()
                      test_partitionedData[clusterId].append((target_activity, target_activity))
                    else: # append trace, next event pair to no partition dictionary
                      test_partitionedData[clusterId] = list()
                      test_partitionedData[clusterId].append((test_input_temp.copy(), target_activity))           
                 
                  if len(test_input_temp) == 0:
                    test_noPartitionData.append((target_activity, target_activity))
                  else:
                    test_noPartitionData.append((test_input_temp.copy(), target_activity))

                else:
                  test_input_temp.append(line2[1]) #add event ID to output list
                  if len(test_input_temp) > test_maxLength:
                      test_maxLength = len(test_input_temp) #set trace length to current trace length if bigger than last. records max length of trace.
                  
                  count = count+1
        outercount=outercount+1 

        self.train_partitionedData = train_partitionedData 
        self.test_partitionedData = test_partitionedData
        self.train_noPartitionhData = train_noPartitionhData
        self.test_noPartitionData = test_noPartitionData
        self.train_maxLength = train_maxLength
        self.test_maxLength = test_maxLength
        self.clusterDict = clusterDict

    def initPrefixBucketing(self, start_pos): #creates traces with end activity targets # was initBatch2
        
        train_partitionedData = dict()
        test_partitionedData = dict()
        train_noPartitionhData = list()
        test_noPartitionData = list()
        train_maxLength = 0
        test_maxLength = 0

        #########################
        #  Create training sets #
        #########################

        for line in self.train_dataset: # for each trace in batch
          
            count = 0
            train_input_temp = list() #reset for each trace
            for line2 in line: #for each event in trace
                
                  
                target_activity  = line2[1]

                if count != 0:
                  try:  # store trace, next event pair at prefix length address
                      train_partitionedData[len(train_input_temp)].append((train_input_temp.copy(), target_activity)) # appends event id and time to output at max length? index is trace identifier in each event list. So the first cell of this output array is a list of all the starting activities with a max length 3. The second position is a list of all the second activities etc 
                  except BaseException as e:
                      train_partitionedData[len(train_input_temp)] = list()
                      train_partitionedData[len(train_input_temp)].append((train_input_temp.copy(), target_activity))
                  if len(train_input_temp) >= start_pos: #if prefix is greater than or equal to starting position
                      train_noPartitionhData.append((train_input_temp.copy(), target_activity)) # append trace, next event pair to no partition dictionary
                
                train_input_temp.append(line2[1]) #add event ID to output list
                if len(train_input_temp) > train_maxLength:
                    train_maxLength = len(train_input_temp) #set trace length to current trace length if bigger than last. records max length of trace
                
                count = count+1

        #########################
        #  Create testing sets #
        #########################


        for line in self.test_dataset:
            count = 0
            test_input_temp = list()
            for line2 in line:

                target_activity  = line2[1]

                if count != 0:
                  try:
                      test_partitionedData[len(test_input_temp)].append((test_input_temp.copy(), target_activity))
                  except BaseException as e:
                      test_partitionedData[len(test_input_temp)] = list()
                      test_partitionedData[len(test_input_temp)].append((test_input_temp.copy(), target_activity))
                  if len(test_input_temp) >= start_pos:
                      test_noPartitionData.append((test_input_temp.copy(), target_activity))


                test_input_temp.append(line2[1])
                if len(test_input_temp) > test_maxLength:
                    test_maxLength = len(test_input_temp)
                
                count = count + 1
                
        self.train_partitionedData = train_partitionedData
        self.test_partitionedData = test_partitionedData
        self.train_noPartitionhData = train_noPartitionhData
        self.test_noPartitionData = test_noPartitionData
        self.train_maxLength = train_maxLength
        self.test_maxLength = test_maxLength

    
    def generatePartitionedBatch(self,batch_size,length_size): #generate batches of traces from a specified cluster/prefix bucket
        train_batch_single = list()
        test_batch_single = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if length_size in self.train_batch_single: #checks to see if there is a bucket with cluster/length "length_size"
            self.train_batch = self.train_batch_single[length_size] #copies the trace prefixes of "length_size"
            self.test_batch = self.test_batch_single[length_size]
            return 0
        for line in self.train_partitionedData[length_size]: #loads the prefixes in batches of a certain size
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch_single.append((input_temp.copy(),target_temp.copy())) #append to training batch as batch
                max_length = 0 #reset params for next batch
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size: # randomise traces in each batch
            if len(train_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(train_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(train_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(train_batch_single)-1)
                (ran_input,ran_target) = train_batch_single[ran1]
                ran2 = random.randint(0, len(train_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        train_batch_single.append((input_temp.copy(), target_temp.copy())) 
        max_length = 0
        input_temp = list()
        target_temp = list()

        #repeat for test set.

        for line in self.test_partitionedData[length_size]:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch_single.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
                input_temp.append(line[0])
                target_temp.append(line[1])
            input_temp.append(line[0])
            target_temp.append(line[1])
        while len(input_temp) < batch_size:
            #print(len(test_batch_single),test_batch_single)
            if len(test_batch_single) ==0 and len(input_temp) == 0:
                break
            elif len(test_batch_single) ==0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(input_temp) - 1)
                input_temp.append(input_temp[ran1])
                target_temp.append(target_temp[ran1])
            elif len(test_batch_single) !=0 and len(input_temp) != 0:
                ran1 = random.randint(0, len(test_batch_single)-1)
                (ran_input,ran_target) = test_batch_single[ran1]
                ran2 = random.randint(0, len(test_batch_single[ran1])-1)
                input_temp.append(ran_input[ran2])
                target_temp.append(ran_target[ran2])
        test_batch_single.append((input_temp.copy(), target_temp.copy()))
        #print(test_batch_single)
        self.train_batch_single[length_size] = train_batch_single
        self.test_batch_single[length_size] = test_batch_single
        self.train_batch = self.train_batch_single[length_size]
        self.test_batch = self.test_batch_single[length_size]

    def generateNoPartitionBatch(self, batch_size): #generate mixed batches of traces
        train_batch = list()
        test_batch = list()
        input_temp = list()
        target_temp = list()
        max_length = 0
        if len(self.test_batch_mix) > 0: # load mixed batches
            self.train_batch = self.train_batch_mix
            self.test_batch = self.test_batch_mix
            return 0
        for line in self.train_noPartitionhData: # add mix of traces to batch
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                train_batch.append((input_temp.copy(),target_temp.copy())) # add batch to training batch
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])
        while len(input_temp) < batch_size: # randomise traces in batches
            ran1 = random.randint(0, len(train_batch)-1)
            (ran_input,ran_target) = train_batch[ran1]
            ran2 = random.randint(0, len(train_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)
        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        train_batch.append((input_temp.copy(), target_temp.copy()))
        max_length = 0
        input_temp = list()
        target_temp = list()
        
        # Repeat for test batch

        for line in self.test_noPartitionData:
            if len(input_temp) == batch_size:
                for num in range(len(input_temp)):
                    while len(input_temp[num]) < max_length:
                        input_temp[num].append(self.vocab_size)
                test_batch.append((input_temp.copy(),target_temp.copy()))
                max_length = 0
                input_temp = list()
                target_temp = list()
            input_temp.append(line[0])
            target_temp.append(line[1])
            if max_length < len(line[0]):
                max_length = len(line[0])

        while len(input_temp) < batch_size:
            ran1 = random.randint(0, len(test_batch)-1)
            (ran_input,ran_target) = test_batch[ran1]
            ran2 = random.randint(0, len(test_batch[ran1])-1)
            input_temp.append(ran_input[ran2].copy())
            target_temp.append(ran_target[ran2])
        max_length = 0
        for line in input_temp:
            if max_length < len(line):
                max_length = len(line)

        for num in range(len(input_temp)):
            while len(input_temp[num]) < max_length:
                input_temp[num].append(self.vocab_size)
        test_batch.append((input_temp.copy(), target_temp.copy()))

        self.train_batch = train_batch
        self.test_batch = test_batch
        self.train_batch_mix = train_batch
        self.test_batch_mix = test_batch






