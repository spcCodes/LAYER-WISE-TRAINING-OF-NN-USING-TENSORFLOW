#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 22:01:22 2017

@author: suman
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
np.set_printoptions(threshold=100)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import sys
sys.path.append('/home/suman/phdWorks')

import normalisationClassWise as NC

'''dataset preparation'''
dataset = pd.read_csv('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/pendigitsConverted.csv')
dataset=dataset.dropna()
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 16].values

'''frequency of the number of labels ocuuring CHECKING IF THE DATA IS BALANCED OR UNBALANCED'''
unique, counts = np.unique(Y, return_counts=True)

'''SINCE THE DATA IS BALANCED SO NO NEED OF DATA BALANCING'''

'''encoding the y balanced data'''
labelencoderYBalanced = LabelEncoder()
yBalanced= labelencoderYBalanced.fit_transform(Y)
yBalanced=yBalanced.reshape(yBalanced.shape[0],-1)

'''perform the one hot encoding for y data'''
onehotencoder=OneHotEncoder(categorical_features=[0])
YBalancedEncoded = onehotencoder.fit_transform(yBalanced).toarray()

##normalising X
XNorm = NC.L2normalization(X)

xBalanced = XNorm
yCategorical=Y

'''saving the data'''
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/xBalanced.npy' , xBalanced)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yBalancedEncoded.npy' , YBalancedEncoded)
np.save('/home/suman/phdWorks/dataset_15dec/ConvertedCsvDataSet/pendigits/yCategorical.npy' ,yCategorical)


