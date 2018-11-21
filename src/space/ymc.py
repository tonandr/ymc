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
RESULT_FILE_NAME = 'ymc_result.csv'

testTimeRanges = [(pd.Timestamp('19/05/2018'), pd.Timestamp('25/05/2018'))
                                , (pd.Timestamp('26/05/2018'), pd.Timestamp('01/06/2018'))
                                , (pd.Timestamp('02/06/2018'), pd.Timestamp('08/06/2018'))
                                , (pd.Timestamp('09/06/2018'), pd.Timestamp('15/06/2018'))
                                , (pd.Timestamp('24/08/2018'), pd.Timestamp('30/08/2018'))
                                , (pd.Timestamp('28/08/2018'), pd.Timestamp('03/09/2018'))]


WIND_BIN_SIZE = 1
WIND_BIN_MAX = 16

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
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')     
    
        if i == 0:
            bDF = df
        else:
            bDF = bDF.append(df)
    
    bDF = bDF.sort_values(by='Timestamp')
    bDF = bDF[:testTimeRanges[0][0]]
    
    # Get valid samples with top 10% power for each wind speed bin.
    vbDF = pd.DataFrame(columns = bDF.columns)
    
    for v in np.arange(int(WIND_BIN_MAX/WIND_BIN_SIZE)):
        df = bDF.query('({0:f} <= avg_ws1) & (avg_ws1 < {1:f})'.format(v * WIND_BIN_SIZE
                                                                       , (v + 1.) * WIND_BIN_SIZE))
        df = df.iloc[0:int(df.shape[0]*0.1),:] #?
        vbDF = vbDF.append(df)
    
    # Apply Kalman filtering to avg_rwd1 for each wind turbine
    # and calibrate avg_ws1 with coefficients.
    bDFG = vbDF.groupby('Turbin_no')
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
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')     
    
        if i == 0:
            tgDF = df
        else:
            tgDF = tgDF.append(df)
    
    tgDF = tgDF.sort_values(by='Timestamp')
    tgDF = tgDF[:testTimeRanges[0][0]]

    # Get valid samples with top 10% power for each wind speed bin.
    vtgDF = pd.DataFrame(columns = tgDF.columns)
    
    for v in np.arange(int(WIND_BIN_MAX/WIND_BIN_SIZE)):
        df = tgDF.query('({0:f} <= avg_ws1) & (avg_ws1 < {1:f})'.format(v * WIND_BIN_SIZE
                                                                       , (v + 1.) * WIND_BIN_SIZE))
        df = df.iloc[0:int(df.shape[0]*0.1),:] #?
        vtgDF = vtgDF.append(df)
    
    # Apply Kalman filtering to avg_rwd1 for each wind turbine
    # and calibrate avg_ws1 with coefficients.
    tgDFG = vtgDF.groupby('Turbin_no')
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
        trInput1M, trInput2M, trOutputM = tr
        
        # Train the model.
        hists = []
        
        hist = self.model.fit([trInput1M, trInput2M], [trOutputM]
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
        
        num_seq1 = hps['num_seq1']
        num_seq2 = hps['num_seq2']
        
        # Training data.
        trRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        numSample = trRawDatasDF.shape[0]
        
        trInput1M = np.concatenate([list(trRawDatasDF.loc[i:(numSample - num_seq1 + i + 1)
                                                                , ['avg_a_power', 'c_avg_ws1']]) \
                                   for i in range(num_seq1)], axis = 1)\
                                  .reshape((numSample - num_seq1 + 1, num_seq1, 2))
        trOutputM = np.asarray([list(trRawDatasDF.loc[(num_seq1*(i + 1)):(num_seq1*(i + 1) + num_seq2 + 1)
                                                            , ['avg_rwd1']]) for i in range(numSample - num_seq1 + 1)]) # end?
        trInput2M = np.asarray([list(np.concatenate([np.zeros(shape=(1)), trOutputM[i, :-1]]) \
                          for i in range(numSample - num_seq1 + 1))])
        
        tr = (trInput1M, trInput2M, trOutputM)
        
        # Validation data.
        valRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        numSample = valRawDatasDF.shape[0]
        
        valInput1M = np.concatenate([list(valRawDatasDF.loc[i:(numSample - num_seq1 + i + 1)
                                                                , ['avg_a_power', 'c_avg_ws1']]) \
                                   for i in range(num_seq1)], axis = 1)\
                                  .reshape((numSample - num_seq1 + 1, num_seq1, 2))
        valOutputM = np.asarray([list(valRawDatasDF.loc[(num_seq1*(i + 1)):(num_seq1*(i + 1) + num_seq2 + 1)
                                                            , ['avg_rwd1']]) for i in range(numSample - num_seq1 + 1)]) # end?
        valInput2M = np.asarray([list(np.concatenate([np.zeros(shape=(1)), valOutputM[i, :-1]]) \
                          for i in range(numSample - num_seq1 + 1))])
        
        val = (valInput1M, valInput2M, valOutputM)       
        
        return tr, val

    def evaluate(self, hps, modelLoading = True):
        '''
            Evaluate.
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
        
        # TODO   

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
            
            # Create time range string.
            timeRangeStr = '{0:d}/{1:d}/{2:d} to {3:d}/{4:d}/{5:d}'.format(r[0].year
                                                                           , r[0].month
                                                                           , r[0].day
                                                                           , r[1].year
                                                                           , r[2].month
                                                                           , r[3].day)
                        
            if i == 0:
                predResultDF = self.predict(teDataPartDF, timeRangeStr)
            else:
                predResultDF = predResultDF.append(self.predict(teDataPartDF, timeRangeStr))
        
        # Save.
        predResultDFG = predResultDF.groupby('Turbine_no')
        wtNames = list(predResultDFG.groups.keys())
        
        resDF = pd.DataFrame(columns=['Turbine', 'Date Range', 'Weekly Estimation YAW Error'])
        
        for i, wtName in enumerate(wtNames):
            df = predResultDFG.get_group(wtName)
            
            # Sort. ?
            
            res = {'Turbine': list(df.Turbine_no)
                   , 'Date Range': list(df.time_range)
                   , 'Weekly Estimation YAW Error': list(df.aggregate_yme_val)}
        
            resDF = resDF.append(pd.DataFrame(res))
        
        resDF.to_csv(RESULT_FILE_NAME, index=False) #?
        
    def predict(self, teDataPartDF, timeRangeStr):
        '''
            Predict yaw misalignment error.
            @param teDataPartDF: Testing data frame belonging to time range.
            @param timeRangeStr: Time range string.
        '''
        
        # Predict yme by each wind turbine.
        resDF = pd.DataFrame(columns=['Turbine_no', 'time_range', 'aggregate_yme_val', 'yme_vals'])
        
        teDataPartDFG = teDataPartDF.groupby('Turbine_no')
        wtNames = list(teDataPartDFG.groups.keys())
        
        num_seq1 = self.hps['num_seq1']
        num_seq2 = self.hps['num_seq2']
        
        for i, wtName in enumerate(wtNames):
            df = teDataPartDFG.get_group(wtName)
            
            # Get the first sequence's internal state.
            # Check the df size is within the number of sequence 1.
            if df.shape[0] < num_seq1:
                input = np.concatenate([list(df.loc[:, ['avg_a_power', 'c_avg_ws1']])
                                        , [list(df.loc[-1, ['avg_a_power', 'c_avg_ws1']]) for _ in range(num_seq1 - df.shape[0])]]) #?
            else:
                input = df.loc[int(-1 * num_seq1):, ['avg_a_power', 'c_avg_ws1']] #?
            
            _, c = self.afModel.predict(input)
            
            # Predict ywe values for 7 days with 10 minute interval.
            vals = []
            initVal = np.zeros(shape=(1,2)) #?
            
            val, c = self.predModel.predict([initVal, c]) # Value dimension?
            
            for j in range(num_seq2):
                val, c = self.predModel.predict([val, c])
                vals.append(val)
            
            vals = np.asarray(vals)
            
            res = {'Turbine_no': [wtName]
                   , 'time_range': [timeRangeStr]
                   , 'aggregate_yme_val': [vals.mean()]
                   , 'yme_vals': [vals]}
            
            resDF = resDF.append(pd.DataFrame(res))
        
        return resDF
        
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
        st = testTimeRanges[0][0] - pd.Timedelta(self.hps['num_seq1'] * 10.0, 'm')
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
               
def main(args):
    '''
        Main.
        @param args: Arguments.
    '''
    
    hps = {}
    
    if args.mode == 'data':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # Create training and validation data.
        createTrValData(rawDataPath)
    elif args.mode == 'train':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['lstm1_dim'] = int(args.lstm1_dim)
        hps['lstm2_dim'] = int(args.lstm2_dim)
        hps['num_layers'] = int(args.num_layers)
        hps['dense1_dim'] = int(args.dense1_dim)
        hps['dropout1_rate'] = float(args.dropout1_rate)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay) 
        hps['epochs'] = int(args.epochs) 
        hps['batch_size'] = int(args.batch_size) 
        hps['val_ratio'] = float(args.val_ratio)
        
        modelLoading = False if int(args.model_load) == 0 else True        
        
        # Train.
        ymc = YawMisalignmentCalibrator(rawDataPath)
        
        ymc.train(hps, modelLoading)
    elif args.mode == 'evaluate':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['lstm1_dim'] = int(args.lstm1_dim)
        hps['lstm2_dim'] = int(args.lstm2_dim)
        hps['num_layers'] = int(args.num_layers)
        hps['dense1_dim'] = int(args.dense1_dim)
        hps['dropout1_rate'] = float(args.dropout1_rate)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay) 
        hps['epochs'] = int(args.epochs) 
        hps['batch_size'] = int(args.batch_size) 
        hps['val_ratio'] = float(args.val_ratio)
        
        modelLoading = False if int(args.model_load) == 0 else True         
        
        # Evaluate.
        ymc = YawMisalignmentCalibrator(rawDataPath)
        
        ymc.evaluate(hps, modelLoading) #?
    elif args.mode == 'test':
        
         # Get arguments.
        rawDataPath = args.raw_data_path
        
        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['lstm1_dim'] = int(args.lstm1_dim)
        hps['lstm2_dim'] = int(args.lstm2_dim)
        hps['num_layers'] = int(args.num_layers)
        hps['dense1_dim'] = int(args.dense1_dim)
        hps['dropout1_rate'] = float(args.dropout1_rate)
        hps['lr'] = float(args.lr)
        hps['beta_1'] = float(args.beta_1)
        hps['beta_2'] = float(args.beta_2)
        hps['decay'] = float(args.decay) 
        hps['epochs'] = int(args.epochs) 
        hps['batch_size'] = int(args.batch_size) 
        hps['val_ratio'] = float(args.val_ratio)
        
        modelLoading = False if int(args.model_load) == 0 else True          
        
        # Test.
        ymc = YawMisalignmentCalibrator(rawDataPath)
        
        ymc.test(hps, modelLoading) #?           
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--num_seq1')
    parser.add_argument('--num_seq2')
    parser.add_argument('--lstm1_dim')
    parser.add_argument('--lstm2_dim')    
    parser.add_argument('--num_layers')
    parser.add_argument('--dense1_dim')
    parser.add_argument('--dropout1_rate')
    parser.add_argument('--lr')
    parser.add_argument('--beta_1')
    parser.add_argument('--beta_2')
    parser.add_argument('--decay')
    parser.add_argument('--epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--val_ratio')
    parser.add_argument('--model_load')
    args = parser.parse_args()
    
    main(args)