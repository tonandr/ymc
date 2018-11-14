'''
Created on Nov 12, 2018

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import os
import glob
import argparse
import time
import sys

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.utils import multi_gpu_model
import keras

import ipyparallel as ipp

# Constant.
MODEL_FILE_NAME = 'yaw_misalignment_calibrator.h5'

IS_MULTI_GPU = False
NUM_GPUS = 4

IS_DEBUG = False

def applyKalmanFilter(data, q=1e-2):
    '''
        Apply Kalman filter.
        @param data: Data.
    '''
    
    # Apply Kalman filter.    
    # Check exception.
    if data.shape[0] == 1:
        r = 1.0
    else:
        r = data.std()**2
    
    vals = []
    x_pre = data.mean()
    p_pre = r
    
    for i in range(data.shape[0]):
        xhat = x_pre
        phat = p_pre + q
        
        k = phat/(phat + r)
        x = xhat + k * (data[i] - xhat)
        p = (1 - k) * phat
        
        vals.append(x)
        
        x_pre = x
        p_pre = p
    
    vals = np.asarray(vals)        

    return vals

def createTrValData(rawDataPath):
    '''
        Create training and validation data.
        @param rawDataPath: Raw data path.
    '''
    
    trValDataDF = pd.DataFrame(columns=['Turbine_no', 'avg_a_power', 'c_avg_ws1', 'avg_rwd1'])
    valid_columns = ['avg_a_power'
                     , 'avg_rwd1'
                     , 'avg_ws1'
                     , 'corr_factor_anem1'
                     , 'corr_offset_anem1'
                     , 'offset_anem1'
                     , 'slope_anem1'
                     , 'g_status'
                     , 'Turbine_no']
    
    # B site.
    # Make data for each wind turbine.
    # Get raw data.
    files = glob.glob(os.path.join(rawDataPath, 'SCADA Site B', '*.csv'))
    
    for i, file in enumerate(files): #?
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df.Timestamp)
        df = df[valid_columns]
        df = df.sort_values(by='avg_a_power', ascending=False)
        df = df.iloc[0:int(df.shape[0]*0.1),:]
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')     
    
        if i == 0:
            bDF = df
        else:
            bDF = bDF.append(df)
    
    bDF = bDF.sort_values(by='Timestamp')
    
    # Apply Kalman filtering to avg_rwd1 for each wind turbine
    # and calibrate avg_ws1 with coefficients.
    bDFG = bDF.groupby('Turbin_no')
    bIds = list(bDFG.groups.keys())
    
    for i, bId in enumerate(bIds):
        df = bDFG.get_group(bId)
        
        # Apply Kalman filtering to avg_rwd1.
        avg_rwd1s = applyKalmanFilter(np.asarray(df.avg_rwd1))
        
        # Calibrate avg_ws1 with coefficients.
        c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
        
        trValData = {'Turbine_no': list(df.Turbine_no)
                     , 'avg_a_power': np.asarray(df.avg_a_power)
                     , 'c_avg_ws1': c_avg_ws1s
                     , 'avg_rwd1': avg_rwd1s}
        
        trValDataDF = trValDataDF.append(pd.DataFrame(trValData))
    
    # TG site.
    # Make data for each wind turbine.
    # Get raw data.
    files = glob.glob(os.path.join(rawDataPath, 'SCADA Site TG', '*.csv'))
    
    for i, file in enumerate(files): #?
        df = pd.read_csv(file)
        df.index = pd.to_datetime(df.Timestamp)
        df = df[valid_columns]
        df = df.sort_values(by='avg_a_power', ascending=False)
        df = df.iloc[0:int(df.shape[0]*0.1),:]
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')     
    
        if i == 0:
            tgDF = df
        else:
            tgDF = tgDF.append(df)
    
    tgDF = tgDF.sort_values(by='Timestamp')
    
    # Apply Kalman filtering to avg_rwd1 for each wind turbine
    # and calibrate avg_ws1 with coefficients.
    tgDFG = tgDF.groupby('Turbin_no')
    tgIds = list(tgDFG.groups.keys())
    
    for i, tgId in enumerate(tgIds):
        df = tgDFG.get_group(tgId)
        
        # Apply Kalman filtering to avg_rwd1.
        avg_rwd1s = applyKalmanFilter(np.asarray(df.avg_rwd1))
        
        # Calibrate avg_ws1 with coefficients.
        c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
        
        trValData = {'Turbine_no': list(df.Turbine_no)
                     , 'avg_a_power': np.asarray(df.avg_a_power)
                     , 'c_avg_ws1': c_avg_ws1s
                     , 'avg_rwd1': avg_rwd1s}
        
        trValDataDF = trValDataDF.append(pd.DataFrame(trValData))   
       
    # Save data.
    trValDataDF.to_csv('train.csv')

class YawMisalignmentCalibrator(object):
    '''
        Yaw misalignment calibrator.
    '''
    
    def __init__(self, rawDataPath):
        '''
            Constructor.
        '''
        
        # Initialize.
        self.rawDataPath = rawDataPath
        
    def train(self, hps, modelLoading = False):
        '''
            Train.
            @param hps: Hyper-parameters.
            @param modelLoading: Model loading flag.
        '''
        
        self.hps = hps
        
        if modelLoading == True:
            print('Load the pre-trained model...')
            
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(keras.models.load_model(MODEL_FILE_NAME), gpus = NUM_GPUS) 
            else:

                self.model = keras.models.load_model(MODEL_FILE_NAME)
        else:
            
            # Design the model.
            
        
        
        
    
    
        

if __name__ == '__main__':
    pass