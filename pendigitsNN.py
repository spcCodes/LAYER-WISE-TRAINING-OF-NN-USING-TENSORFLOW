#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 01:03:18 2017

@author: suman
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
np.set_printoptions(threshold=100)
from sklearn.metrics import confusion_matrix
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#%matplotlib inline
np.random.seed(1)

sys.path.append('/home/suman/phdWorks')

import NNfunctions as NN
import seperabiltyCodes as sc
from tf_utils import random_mini_batches , predict , convert_to_one_hot 

Xinput = np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/xBalanced.npy')
yEncoded = np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yBalancedEncoded.npy')
Yinput=np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yCategorical.npy')

NoExp=10
MeanTrAccuracy=[]
MeanTeAccuracy=[]
MeanTrLoss=[]
MeanTeLoss=[]
VarTrAccuracy=[]
VarTeAccuracy=[]
VarTrLoss=[]
VarTeLoss=[]

for i in range(10,100,10):
    TrainingAccuracy=[]
    TestingAccuracy=[]
    TrainingLoss=[]
    TestingLoss=[]
    for j in range(NoExp):
        X_train,Y_train,X_test,Y_test=NN.GenerateDataset(Xinput,Yinput,i)
        parameters , TrAccu , TeAccu , TrLoss , TeLoss = NN.model(X_train, Y_train, X_test, Y_test)
        TrainingAccuracy.append(TrAccu)
        TestingAccuracy.append(TeAccu)
        TrainingLoss.append(TrLoss)
        TestingLoss.append(TeLoss)
        print('i:' , i,'j' , j)
    MeanTrAccuracy.append(np.mean(TrainingAccuracy))
    MeanTeAccuracy.append(np.mean(TestingAccuracy))
    MeanTrLoss.append(np.mean(TrainingLoss))
    MeanTeLoss.append(np.mean(TestingLoss))
    VarTrAccuracy.append(np.var(TrainingAccuracy))
    VarTeAccuracy.append(np.var(TestingAccuracy))
    VarTrLoss.append(np.var(TrainingLoss))
    VarTeLoss.append(np.var(TestingLoss))
        
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/MeanTrainLoss.npy' , MeanTrLoss)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/MeanTestLoss.npy' , MeanTeLoss)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/MeanTrainAccuracy.npy' , MeanTrAccuracy)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/MeanTestAccuracy.npy' , MeanTeAccuracy)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/VarTrainLoss.npy' ,VarTrLoss)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/VarTestLoss.npy' ,VarTeLoss)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/VarTrainAcc.npy' ,VarTrAccuracy)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/VarTestAcc.npy' ,VarTeAccuracy)




def Relu(X):
    return X*(X>0)

Xinput = np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/xBalanced.npy')
yEncoded = np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yBalancedEncoded.npy')
Yinput=np.load('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yCategorical.npy')

X_train,Y_train,X_test,Y_test=NN.GenerateDataset(Xinput,Yinput,70)
parameters = NN.model(X_train, Y_train, X_test, Y_test)[0]

W1 = parameters['W1']
b1 = parameters['b1']

W1 = np.float64(W1)
b1 = np.float64(b1)

Z1 = np.dot(W1 , Xinput.T) + b1
A1= Relu(Z1)

J = sc.ScatterMatrices(A1.T , Yinput)
