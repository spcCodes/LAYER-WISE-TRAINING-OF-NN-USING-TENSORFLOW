#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:20:29 2017

@author: suman
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
np.set_printoptions(threshold=100)
from sklearn.metrics import confusion_matrix
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy.linalg import inv , det
from sklearn.metrics import silhouette_samples, silhouette_score
import math

sys.path.append('/home/suman/FINALDATASET')

import DataSeperation as ds

def AccumulatingSilhoute(Xinput ,Yinput):
    sample_silhouette_values = silhouette_samples(Xinput, Yinput)
    classes = np.unique(Yinput)
    IndiMeanSilScore=[]
    IndiVarSilScore=[]
    IndiSilThreshold=[]
    
    for i in range(len(classes)):
        indices = np.where(Yinput==classes[i])
        indices=np.asarray(indices)
        silhouteScore=[]
        for j in range(indices.shape[1]):
            temp = indices[0,j]
            silhoute =  sample_silhouette_values[temp]   
            silhouteScore.append(silhoute)
        ss=np.asarray(silhouteScore)
        SilhouteMean=ss.mean()
        SilhouteVar=ss.var()
        SilhouteThres= SilhouteMean - 3*math.sqrt(SilhouteVar)
        ##putting them into 
        IndiMeanSilScore.append(SilhouteMean)
        IndiVarSilScore.append(SilhouteVar)
        IndiSilThreshold.append(SilhouteThres)
        
    return IndiMeanSilScore , IndiVarSilScore , IndiSilThreshold

def ScatterMatrices(Xinput,Yinput):
    GlobalMean = np.mean(Xinput,axis=0)
    classes = np.unique(Yinput)
    
    '''frequency of the number of labels ocuuring CHECKING IF THE DATA IS BALANCED OR UNBALANCED'''
    unique, counts = np.unique(Yinput, return_counts=True)
    
    TotalPoints=Xinput.shape[0]
    
    WithinClassVar=np.zeros(Xinput.shape[1])
    ClassMean=[]
    for i in range(len(classes)):
        indices = np.where(Yinput==classes[i])
        indices=np.asarray(indices)
        Xvalues=[]
        for j in range(indices.shape[1]):
            temp = indices[0,j]
            xvalue =  Xinput[temp]
            Xvalues.append(xvalue)
        XvaluesClass=np.asarray(Xvalues)
        XvaluesMean=np.mean(XvaluesClass,axis=0)
        XvalueCovClass = np.cov(XvaluesClass.T)
        WithinClassVar=WithinClassVar+(counts[i]/TotalPoints)*XvalueCovClass
        ClassMean.append(XvaluesMean)
    ClassMean=np.asarray(ClassMean)
    WithinClassVar=np.matrix(WithinClassVar)
    
    '''between class covariance'''
    BetweenClassVar= np.zeros(Xinput.shape[1])
    for k in range(len(classes)):
        temp1 = np.matrix(ClassMean[i] - GlobalMean)
        value = np.dot(temp1.T , temp1)
        BetweenClassVar=BetweenClassVar++(counts[i]/TotalPoints)*value
        
        
    MixClassVar = BetweenClassVar+WithinClassVar
    
    InvWithinClassVar=mysolve(WithinClassVar)
    
    temp2= np.dot(InvWithinClassVar , MixClassVar)
    
    J = np.trace(temp2)

    return J


def mysolve(A):
    u, s, v = np.linalg.svd(A)
    Ainv = np.dot (v.transpose(), np.dot(np.diag(s**-1),u.transpose()))
    return(Ainv)