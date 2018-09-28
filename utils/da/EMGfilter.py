# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:08:05 2018

@author: Александр
"""

#%% Filter init
from scipy.signal import butter, lfilter
import numpy as np
numch=64

#%% Subfunctions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#%% Additional vars for filtering
bband,aband=butter_bandpass(lowcut=20, highcut=200, fs=500, order=5)
blow, alow=butter_lowpass(cutoff=2, fs=500, order=5)
filtorder=5
Zband=np.zeros((2*filtorder,numch))
Zlow=np.zeros((filtorder,numch))

#%% Filtering band, then high
def filterEMG(MyoChunk): 
    global Zband, Zlow
    for j in range(MyoChunk.shape[1]):
        MyoChunk[:,j],Zband[:,j] = lfilter(bband,aband, MyoChunk[:,j],-1,Zband[:,j])
        

    np.abs(MyoChunk, out=MyoChunk)
    
    for j in range(MyoChunk.shape[1]):
        MyoChunk[:,j],Zlow[:,j] = lfilter(blow,alow, MyoChunk[:,j],-1,Zlow[:,j])

    return MyoChunk