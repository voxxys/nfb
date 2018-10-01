# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:19:53 2018

@author: Александр
"""
import numpy as np



b0=b1=b2=a1=a2=0
cutoff=50/512
x0=x1=x2=y1=y2=0
numch = 64

def CommonAverage(chunk):
    avg=np.mean(chunk,1)[:,None]
    chunk-=avg
    return chunk

  
def NeorecFiltering(data):
    global x0,x1,x2,y1,y2
    for i in range(data.shape[0]):
        x2 = x1
        x1 = x0
        x0 = data[i,:]
        data[i,:] = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        y2 = y1; 
        y1 = data[i,:];
    return data
        
def initParams(coff):
    global x0,x1,x2,y1,y2,b0,b1,b2,a1,a2
    mu = 0.005;
    pi2 = 6.28318530717958647693;
    b0 = b2 = 1 - mu;
    b1 = a1 = np.cos(pi2*cutoff)*(2 * mu - 2);
    a2 = 1 - 2 * mu;
    x0=np.zeros((1,numch))
    x1=np.zeros((1,numch))
    x2=np.zeros((1,numch))
    y1=np.zeros((1,numch))
    y2=np.zeros((1,numch))
