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
from keras.layers import LSTM, Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.utils import multi_gpu_model
import keras

import ipyparallel as ipp

# Constant.
MODEL_FILE_NAME = 'yaw_misalignment_calibrator.h5'

testTimeRanges = [(pd.Timestamp('19/05/2018'), pd.Timestamp('25/05/2018'))
                                , (pd.Timestamp('26/05/2018'), pd.Timestamp('01/06/2018'))
                                , (pd.Timestamp('02/06/2018'), pd.Timestamp('08/06/2018'))
                                , (pd.Timestamp('09/06/2018'), pd.Timestamp('15/06/2018'))
                                , (pd.Timestamp('24/08/2018'), pd.Timestamp('30/08/2018'))
                                , (pd.Timestamp('28/08/2018'), pd.Timestamp('03/09/2018'))]


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
    bDF = bDF[:testTimeRanges[0][0]]
    
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
            print('Design the model.')
            
            # Input1: n (n sequence) x 2 (calibrated avg_rwd1, avg_a_power)   
            input1 = Input(shape=(self.hps['num_seq1'], 2))
            _, c = LSTM(self.hps['lstm1_dim'], return_state = True, name='lstm1')(input1)
            
            # Input2: ywe value sequence.
            input2 = Input(shape=(self.hps['num_seq2', 1]))
            x, _ = LSTM(self.hps['lstm2_dim']
                     , return_sequences = True
                     , return_state = True
                     , name='lstm2')(input2, initial_state = c)
            
            for i in range(1, hps['num_layers']):
                x = Dense(self.hps['dense1_dim'], activation='relu', name='dense1_' + str(i))(x)
                        
            output = Dense(1, activation='linear', name='dense1_last')(x)
                                            
            # Create the model.
            if IS_MULTI_GPU == True:
                self.model = multi_gpu_model(Model(inputs=[input1, input2]
                                                   , outputs=[output]), gpus = NUM_GPUS)
            else:
                self.model = Model(inputs=[input1, input2], outputs=[output])  
        
        # Compile the model.
        optimizer = optimizers.Adam(lr=self.hps['lr']
                                    , beta_1=self.hps['beta_1']
                                    , beta_2=self.hps['beta_2']
                                    , decay=self.hps['decay'])
        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.summary()        
        
        # Create training and validation data.
        tr, val = self.__createTrValData__(hps)
        trInputM, trOutputM = tr
        
        # Train the model.
        hists = []
        
        hist = self.model.fit([trInputM], [trOutputM]
                      , epochs=self.hps['epochs']
                      , batch_size=self.hps['batch_size']
                      , verbose=1)
            
        hists.append(hist)
            
        # Print loss.
        print(hist.history['loss'][-1])
        
        print('Save the model.')            
        self.model.save(MODEL_FILE_NAME)
        
        # Make the prediction model.
        self.__makePredictionModel__();
        
        # Calculate loss.
        lossList = list()
        
        for h in hists:
            lossList.append(h.history['loss'][-1])
            
        lossArray = np.asarray(lossList)        
        lossMean = lossArray.mean()
        
        print('Each mean loss: {0:f} \n'.format(lossMean))
        
        with open('losses.csv', 'a') as f:
            f.write('{0:f} \n'.format(lossMean))
                
        with open('loss.csv', 'w') as f:
            f.write(str(lossMean) + '\n') #?
            
        return lossMean 

    def __makePredictionModel__(self):
        '''
            Make the prediction model.
        '''
        
        # Affecting factor sequence model.
        input1 = Input(shape=(self.hps['num_seq1'], 2))
        _, c = self.model.get_layer('lstm1')(input1)
        
        self.afModel = Model([input1], [c])
        
        # Target factor prediction model.
        input2 = Input(shape=(self.hps['num_seq2', 1]))
        recurState = Input(shape=(self.hps['lstm2_dim'],)) #?
        
        x, c2 = self.model.get_layer('lstm2')(input2, initial_state = recurState) #?
        
        for i in range(1, self.hps['num_layers']):
            x = self.model.get_layer('dense1_' + str(i))(x)
        
        output = self.model.get_layer('dense1_last')(x)
                      
        self.predModel = Model([input2, recurState], [output, c2])    

    def __createTrValData__(self, hps):
        '''
            Create training and validation data.
            @param hps: Hyper-parameters.
        '''
        
        # Load raw data.
        rawDatasDF = pd.read_csv('train.csv')
        
        # Training data.
        trRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        
        trInputM = np.concatenate([np.asarray(trRawDatasDF.loc[:-1, ['avg_a_power', 'c_avg_ws1']])
                                  , np.asarray(trRawDatasDF.loc[1:, ['avg_a_power', 'c_avg_ws1']])], axis = 1)\
                                  .reshape((trRawDatasDF.shape[0] - 1, 2, 2))
        trOutputM = np.asarray(trRawDatasDF.loc[1:, ['avg_rwd1']])
        
        tr = (trInputM, trOutputM)
        
        # Validation data.
        valRawDatasDF = rawDatasDF.iloc[int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])):, :]
        
        valInputM = np.concatenate([np.asarray(valRawDatasDF.loc[:-1, ['avg_a_power', 'c_avg_ws1']])
                                  , np.asarray(valRawDatasDF.loc[1:, ['avg_a_power', 'c_avg_ws1']])], axis = 1)\
                                  .reshape((valRawDatasDF.shape[0] - 1, 2, 2))
        valOutputM = np.asarray(valRawDatasDF.loc[1:, ['avg_rwd1']])
        
        val = (valInputM, valOutputM)        
        
        return tr, val

    def test(self, hps, modelLoading = True):
        '''
            Test.
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

        # Make the prediction model.
        self.__makePredictionModel__();
        
        # Load testing data.
        teDataDF = self.__loadTestingData__()
        
        for i, r in enumerate(testTimeRanges):
            
            # Get testing data belonging to time range.
            teDataPartDF = teDataDF[:r[0]]
                        
            if i == 0:
                predResultDF = self.predict(teDataPartDF)
            else:
                predResultDF = predResultDF.append(self.predict(teDataPartDF))
        
        # Save.

 
    def predict(self, teDataPartDF):
        '''
            Predict yaw misalignment error.
            @param teDataPartDF: Testing data frame belonging to time range.
        '''
        
        # Predict yme by each wind turbine.
        resDF = pd.DataFrame(columns=['Turbine_no', 'time_range', 'aggregate_yme_val', 'yme_vals'])
        
        teDataPartDFG = teDataPartDF.groupby('Turbine_no')
        wtNames = list(teDataPartDFG.groups.keys())
        
        for i, wtName in enumerate(wtNames):
            
        
        
 
    def __loadTestingData__(self):
        '''
            Load testing data.
        '''
        
        teDataDF = pd.DataFrame(columns=['Turbine_no', 'avg_a_power', 'c_avg_ws1', 'avg_rwd1'])
        valid_columns = ['avg_a_power'
                         , 'avg_rwd1'
                         , 'avg_ws1'
                         , 'corr_factor_anem1'
                         , 'corr_offset_anem1'
                         , 'offset_anem1'
                         , 'slope_anem1'
                         , 'g_status'
                         , 'Turbine_no']
        
        # Determine time range.
        st = testTimeRanges[0][0] - pd.Timedelta(self.hps['num_seq'] * 10.0, 'm')
        ft = testTimeRanges[0][0]
        
        # B site.
        # Make data for each wind turbine.
        # Get raw data.
        files = glob.glob(os.path.join(self.rawDataPath, 'SCADA Site B', '*.csv'))
        
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
        bDF = bDF[st:ft]
        
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
            
            teData = {'Timestamp': list(df.Timestamp)
                        , 'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            teDataDF = teDataDF.append(pd.DataFrame(teData))
        
        # TG site.
        # Make data for each wind turbine.
        # Get raw data.
        files = glob.glob(os.path.join(self.rawDataPath, 'SCADA Site TG', '*.csv'))
        
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
        tgDF = tgDF[st:ft]
        
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
            
            teData = {'Timestamp': list(df.Timestamp)
                        , 'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            teDataDF = teDataDF.append(pd.DataFrame(teData))   
           
        teDataDF.index = teDataDF.Timestamp #?
        teDataDF = teDataDF.loc[:, ['Turbine_no', 'avg_a_power', 'c_avg_ws1', 'avg_rwd1']]
        teDataDF.sort_values(by='Timestamp')
        
        return teDataDF
               
if __name__ == '__main__':
    pass