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
import scipy.io as io

from keras.models import Model
from keras.layers import LSTM, GRU, Dense, Dropout, Flatten, Input
from keras import optimizers
from keras.utils import multi_gpu_model
import keras

import ipyparallel as ipp

# Constant.
MODEL_FILE_NAME = 'yaw_misalignment_calibrator.h5'
RESULT_FILE_NAME = 'ymc_result.csv'

dt = pd.Timedelta(10.0, 'm')

testTimeRanges = [(pd.Timestamp('2018-05-19'), pd.Timestamp('2018-05-26') - dt)
                                , (pd.Timestamp('2018-05-26'), pd.Timestamp('2018-06-02') - dt)
                                , (pd.Timestamp('2018-06-02'), pd.Timestamp('2018-06-09') - dt)
                                , (pd.Timestamp('2018-06-09'), pd.Timestamp('2018-06-16') - dt)
                                , (pd.Timestamp('2018-08-24'), pd.Timestamp('2018-08-31') - dt)
                                , (pd.Timestamp('2018-08-28'), pd.Timestamp('2018-09-04') - dt)]

testTimeRangeStrings = ['19/05/2018 to 25/05/2018'
, '26/05/2018 to 01/06/2018'
, '02/06/2018 to 08/06/2018'
, '09/06/2018 to 15/06/2018'
, '24/08/2018 to 30/08/2018'
, '28/08/2018 to 03/09/2018']

WIND_BIN_SIZE = 1
WIND_BIN_MAX = 20

DELTA_TIME = 1

IS_MULTI_GPU = False
NUM_GPUS = 4

IS_DEBUG = False

def applyKalmanFilter(data, q=1e-5):
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
            _, c = GRU(self.hps['gru1_dim'], return_state = True, name='gru1')(input1)
            
            # Input2: ywe value sequence.
            input2 = Input(shape=(self.hps['num_seq2'], 1))
            x, _ = GRU(self.hps['gru2_dim']
                     , return_sequences = True
                     , return_state = True
                     , name='gru2')(input2, initial_state = c)
            
            for i in range(1, hps['num_layers'] + 1):
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
        tr, val = self.__createTrValData__(hps, dataLoading=True)
        trInput1M, trInput2M, trOutputM = tr
        valInput1M, valInput2M, valOutputM = val
        
        # Train the model.
        hists = []
        
        hist = self.model.fit([trInput1M, trInput2M], [trOutputM]
                      , epochs=self.hps['epochs']
                      , batch_size=self.hps['batch_size']
                      , validation_data = ([valInput1M, valInput2M], [valOutputM])
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
        _, c = self.model.get_layer('gru1')(input1)
        
        self.afModel = Model([input1], [c])
        
        # Target factor prediction model.
        input2 = Input(shape=(1,1))
        recurState = Input(shape=(self.hps['gru1_dim'],)) #?
        
        x, c2 = self.model.get_layer('gru2')(input2, initial_state = recurState) #?
        
        for i in range(1, self.hps['num_layers'] + 1):
            x = self.model.get_layer('dense1_' + str(i))(x)
        
        output = self.model.get_layer('dense1_last')(x)
                      
        self.predModel = Model([input2, recurState], [output, c2])    

    def __createTrValData__(self, hps, dataLoading): # Parallel computing is necessary.
        '''
            Create training and validation data.
            @param hps: Hyper-parameters.
            @param dataLoading: Data loading flag.
        '''
        
        if dataLoading:
            trValMs_mat = io.loadmat('trValMs.mat')
            trInput1M = trValMs_mat['trInput1M']
            trInput2M = trValMs_mat['trInput2M']
            trOutputM = trValMs_mat['trOutputM']
            valInput1M = trValMs_mat['valInput1M']
            valInput2M = trValMs_mat['valInput2M']
            valOutputM = trValMs_mat['valOutputM']
            
            tr = (trInput1M, trInput2M, trOutputM)
            val = (valInput1M, valInput2M, valOutputM)
            
            return tr, val
            
        pClient = ipp.Client()
        pView = pClient[:]
        
        # Load raw data.
        rawDatasDF = pd.read_csv('train.csv')
        
        num_seq1 = hps['num_seq1']
        num_seq2 = hps['num_seq2']
        
        # Training data.
        trRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        #trRawDatasDF = trRawDatasDF.iloc[:3000,:]
        numSample = trRawDatasDF.shape[0]
        t = 1 # One based time index.
        
        # Input 1.
        trInput1 = []
        trOutput = []
        trInput2 = []
        
        pView.push({'num_seq1': num_seq1, 'num_seq2': num_seq2, 'trRawDatasDF': trRawDatasDF})
        
        ts = []
        while ((t + num_seq1 + num_seq2  - 1) <= numSample):
            ts.append(t - 1)
            t += 1 + DELTA_TIME # One based time index.
        
        res = pView.map(getInputOutput, ts, block=True)
        
        for i in range(len(res)):
            trInput1.append(res[i][0])
            trOutput.append(res[i][1])
            trInput2.append(res[i][2])
                
        trInput1M = np.asarray(trInput1)
        trOutputM = np.expand_dims(np.asarray(trOutput), 2)
        trInput2M = np.expand_dims(np.asarray(trInput2), 2)
        
        tr = (trInput1M, trInput2M, trOutputM)
        
        # Validation data.
        valRawDatasDF = rawDatasDF.iloc[:int(rawDatasDF.shape[0]*(1.0 - hps['val_ratio'])), :]
        #valRawDatasDF = valRawDatasDF.iloc[:3000,:]
        numSample = valRawDatasDF.shape[0]
        t = 1 # One based time index.
        
        # Input 1.
        valInput1 = []
        valOutput = []
        valInput2 = []

        pView.push({'num_seq1': num_seq1, 'num_seq2': num_seq2, 'trRawDatasDF': valRawDatasDF})
        
        ts = []
        while ((t + num_seq1 + num_seq2  - 1) <= numSample):
            ts.append(t - 1)
            t += 1 + DELTA_TIME # One based time index.
        
        res = pView.map(getInputOutput, ts, block=True)
        
        for i in range(len(res)):
            valInput1.append(res[i][0])
            valOutput.append(res[i][1])
            valInput2.append(res[i][2])
                
        valInput1M = np.asarray(valInput1)
        valOutputM = np.expand_dims(np.asarray(valOutput), 2)
        valInput2M = np.expand_dims(np.asarray(valInput2), 2)
        
        val = (valInput1M, valInput2M, valOutputM)      
        
        # Save data.
        io.savemat('trValMs.mat', mdict={'trInput1M': trInput1M
                                     , 'trInput2M': trInput2M
                                     , 'trOutputM': trOutputM
                                     , 'valInput1M': valInput1M
                                     , 'valInput2M': valInput2M
                                     , 'valOutputM': valOutputM}
        , oned_as='row') #?
        
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
        teDataDF = pd.read_csv('teDataDF.csv')
        teDataDF.index = pd.to_datetime(teDataDF.Timestamp)
        teDataDF = teDataDF.iloc[:, 1:]
        
        for i, r in enumerate(testTimeRanges):
                        
            # Create time range string.
            timeRangeStr = testTimeRangeStrings[i]
                        
            if i == 0:
                predResultDF = self.predict(teDataDF, r, timeRangeStr)
            else:
                predResultDF = predResultDF.append(self.predict(teDataDF, r, timeRangeStr))
        
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
        
    def predict(self, teDataDF, r, timeRangeStr):
        '''
            Predict yaw misalignment error.
            @param teDataDF: Testing data frame belonging to time range.
            @param r; Time range.
            @param timeRangeStr: Time range string.
        '''
        
        # Predict yme by each wind turbine.
        resDF = pd.DataFrame(columns=['Turbine_no', 'time_range', 'aggregate_yme_val', 'yme_vals'])
        
        teDataDFG = teDataDF.groupby('Turbine_no')
        wtNames = list(teDataDFG.groups.keys())
        
        num_seq1 = self.hps['num_seq1']
        num_seq2 = self.hps['num_seq2']
        
        for i, wtName in enumerate(wtNames):
            
            # Filter the 28/08/2018 to 03/09/2018 TG cases.
            if (r[0] == testTimeRanges[5][0]) and (wtName[0:2] == 'TG'): #?
                continue
            
            df = teDataDFG.get_group(wtName)
            
            # Check exception.
            if df.shape[0] == 0: #?
                continue
            
            # Get testing data belonging to time range.
            df = df[(r[0] - (num_seq1 - 1) * dt):r[1]] # Because of non-unique index.
            
            # Get the first sequence's internal state.
            # Get input data.
            inputs = []
            real_rwds = []
            t = r[0]
            
            while t <= r[1]:
                input1 = np.asarray(list(zip(df[(t - (num_seq1 - 1) * dt ):t].avg_a_power, df[(t - (num_seq1 - 1) * dt ):t].c_avg_ws1)))
                
                # Check exception.
                if input1.shape[0] == 0: #?
                    if len(inputs) == 0:
                        input1 = np.concatenate([np.zeros(shape=(1,2)) for _ in range(num_seq1)]) #?
                        real_rwds.append(0.)
                    else:
                        input1 = inputs[-1]
                        inputs.append(input1)
                        real_rwds.append(real_rwds[-1])    
                        t = t + dt
                        continue
                elif input1.shape[0] < num_seq1:
                    input1 = np.concatenate([input1] + [np.expand_dims(input1[-1],0) for _ in range(num_seq1 - input1.shape[0])])
                    
                    try:
                        real_rwd = df.avg_rwd1.loc[t]
                        real_rwds.append(real_rwd)
                    except KeyError:
                        if len(inputs) == 0:
                            real_rwds.append(0.)
                        else:
                            real_rwds.append(real_rwds[-1])
                    
                inputs.append(np.expand_dims(input1, 0))    
                t = t + dt
                                
            inputs = np.concatenate(inputs) #?
            real_rwds = np.asarray(real_rwds)
            
            cs = self.afModel.predict(inputs)
            
            # Predict ywe values for 7 days with 10 minute interval.
            # Evaluate total yaw offset values.
            initVals = np.zeros(shape=(inputs.shape[0],1,1)) #?
            
            eval_rwds, _ = self.predModel.predict([initVals, cs]) # Value dimension?
            eval_rwds = np.squeeze(eval_rwds)
                        
            yme_vals = eval_rwds - real_rwds
            
            res = {'Turbine_no': [wtName]
                   , 'time_range': [timeRangeStr]
                   , 'aggregate_yme_val': [yme_vals.mean()]
                   , 'yme_vals': [yme_vals]}
            
            resDF = resDF.append(pd.DataFrame(res))
        
        return resDF

    def createTrValTeData(self, hps):
        '''
            Create training and validation data.
            @param hps: Hyper-parameters.
        '''

        self.hps = hps

        pClient = ipp.Client()
        pView = pClient[:]
                
        trValDataDF = pd.DataFrame(columns=['Turbine_no', 'avg_a_power', 'c_avg_ws1', 'avg_rwd1'])
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
        st = testTimeRanges[0][0] - pd.Timedelta(self.hps['num_seq1'] * 10.0, 'm') #?
        ft = testTimeRanges[5][1] #?

        # B site.
        # Make data for each wind turbine.
        # Get raw data.
        files = glob.glob(os.path.join(self.rawDataPath, 'SCADA Site B', '*.csv'))
        #files = files[:1]
        
        #with pView.sync_imports(): #?
        #    import pandas as pdd #
        
        pView.push({'valid_columns': valid_columns})
        
        bDFs = pView.map(loadDF, files, block=True)
        bDF = bDFs[0]
        
        for i in range(1, len(bDFs)):
            bDF = bDF.append(bDFs[i])
        
        # Load extra data.
        df = pd.read_excel(os.path.join(self.rawDataPath, 'SCADA_B6_24Aug_31_Aug.xlsx'))
        df.index = pd.to_datetime(df.Timestamp)
        df = df[valid_columns]
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')
        bDF = bDF.append(df)
        
        df = pd.read_excel(os.path.join(self.rawDataPath, 'TG_31Aug_3Sep', 'Site B 31st Aug to 3rd Sep.xlsx'))
        df.index = pd.to_datetime(df.Timestamp)
        df = df[valid_columns]
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')
        bDF = bDF.append(df)        
        
        # Sort according to time sequence.
        bTotalDF = bDF.sort_values(by='Timestamp')
        
        # Training and validation.
        bDF = bTotalDF[:(testTimeRanges[5][1] - dt)]
        
        # Get valid samples with top 10% power for each wind speed bin.
        vbDF = pd.DataFrame(columns = bDF.columns)
        
        for v in np.arange(int(WIND_BIN_MAX/WIND_BIN_SIZE)):
            df = bDF.query('({0:f} <= avg_ws1) & (avg_ws1 < {1:f})'.format(v * WIND_BIN_SIZE
                                                                           , (v + 1.) * WIND_BIN_SIZE))
            df = df.sort_values(by='avg_a_power', ascending=False)
            df = df.iloc[0:int(df.shape[0]*0.1),:]
            vbDF = vbDF.append(df)
       
        vbDF.index.name = 'Timestamp' 
        vbDF = vbDF.sort_values(by='Timestamp')
        
        # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment
        # and calibrate avg_ws1 with coefficients.
        bDFG = vbDF.groupby('Turbine_no')
        bIds = list(bDFG.groups.keys())
        
        for i, bId in enumerate(bIds):
            df = bDFG.get_group(bId)
            
            # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment.
            avg_rwd1s = np.asarray(df.avg_rwd1) - applyKalmanFilter(np.asarray(df.avg_rwd1))
            
            # Calibrate avg_ws1 with coefficients.
            c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                    + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
            
            trValData = {'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            trValDataDF = trValDataDF.append(pd.DataFrame(trValData))

        # Testing.
        bDF = bTotalDF[st:ft]
        
        # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment
        # and calibrate avg_ws1 with coefficients.
        bDFG = bDF.groupby('Turbine_no')
        bIds = list(bDFG.groups.keys())
        
        for i, bId in enumerate(bIds):
            df = bDFG.get_group(bId)
            
            # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment.
            avg_rwd1s = np.asarray(df.avg_rwd1) - applyKalmanFilter(np.asarray(df.avg_rwd1))
            
            # Calibrate avg_ws1 with coefficients.
            c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                    + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
            
            teData = {'Timestamp': list(df.index)
                        , 'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            teDataDF = teDataDF.append(pd.DataFrame(teData))
        
        # TG site.
        # Make data for each wind turbine.
        # Get raw data.
        files = glob.glob(os.path.join(self.rawDataPath, 'SCADA Site TG', '*.csv'))
        #files = files[:1]
        
        tgDFs = pView.map(loadDF, files, block=True)
        tgDF = tgDFs[0]
        
        for i in range(1, len(tgDFs)):
            tgDF = tgDF.append(tgDFs[i])

        # Load extra data.
        df = pd.read_csv(os.path.join(self.rawDataPath, 'TG_31Aug_3Sep', 'TG_31Aug_3Sep.csv'))
        df.index = pd.to_datetime(df.Timestamp)
        df = df[valid_columns]
        df = df.groupby('g_status').get_group(1.0)
        df = df.dropna(how='any')
        tgDF = tgDF.append(df)  

        # Sort according to time sequence.
        tgTotalDF = tgDF.sort_values(by='Timestamp')
        
        # Training and validation.
        tgDF = tgTotalDF[:(testTimeRanges[5][1] - dt)]
    
        # Get valid samples with top 10% power for each wind speed bin.
        vtgDF = pd.DataFrame(columns = tgDF.columns)
        
        for v in np.arange(int(WIND_BIN_MAX/WIND_BIN_SIZE)):
            df = tgDF.query('({0:f} <= avg_ws1) & (avg_ws1 < {1:f})'.format(v * WIND_BIN_SIZE
                                                                           , (v + 1.) * WIND_BIN_SIZE))
            df = df.sort_values(by='avg_a_power', ascending=False)
            df = df.iloc[0:int(df.shape[0]*0.1),:]
            vtgDF = vtgDF.append(df)
    
        vtgDF.index.name = 'Timestamp' 
        vtgDF = vtgDF.sort_values(by='Timestamp')
    
        # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment
        # and calibrate avg_ws1 with coefficients.
        tgDFG = vtgDF.groupby('Turbine_no')
        tgIds = list(tgDFG.groups.keys())
        
        for i, tgId in enumerate(tgIds):
            df = tgDFG.get_group(tgId)
            
            # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment.
            avg_rwd1s = np.asarray(df.avg_rwd1) - applyKalmanFilter(np.asarray(df.avg_rwd1))
            
            # Calibrate avg_ws1 with coefficients.
            c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                    + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
            
            trValData = {'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            trValDataDF = trValDataDF.append(pd.DataFrame(trValData))   

        # Testing.
        tgDF = tgTotalDF[st:ft]
        
        # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment
        # and calibrate avg_ws1 with coefficients.
        tgDFG = tgDF.groupby('Turbine_no')
        tgIds = list(tgDFG.groups.keys())
        
        for i, tgId in enumerate(tgIds):
            df = tgDFG.get_group(tgId)
            
            # Apply Kalman filtering to avg_rwd1 for each wind turbine and reduce yaw misalignment.
            avg_rwd1s = np.asarray(df.avg_rwd1) - applyKalmanFilter(np.asarray(df.avg_rwd1))
            
            # Calibrate avg_ws1 with coefficients.
            c_avg_ws1s = np.asarray(df.corr_offset_anem1 + df.corr_factor_anem1 * df.avg_ws1 \
                                    + df.slope_anem1 * df.avg_rwd1 + df.offset_anem1) #?
            
            teData = {'Timestamp': list(df.index)
                        , 'Turbine_no': list(df.Turbine_no)
                         , 'avg_a_power': np.asarray(df.avg_a_power)
                         , 'c_avg_ws1': c_avg_ws1s
                         , 'avg_rwd1': avg_rwd1s}
            
            teDataDF = teDataDF.append(pd.DataFrame(teData))   
           
        teDataDF.index = teDataDF.Timestamp #?
        teDataDF = teDataDF.loc[:, ['Turbine_no', 'avg_a_power', 'c_avg_ws1', 'avg_rwd1']]
        teDataDF.sort_values(by='Timestamp')
 
        # Save data.
        trValDataDF.to_csv('train.csv')
        teDataDF.to_csv('test.csv')
        
        return trValDataDF, teDataDF

def loadDF(file):
    '''
        Load data frame.
        @param file: Data file name.
    '''
 
    global valid_columns
    import pandas as pdd
 
    df = pdd.read_csv(file)
    df.index = pdd.to_datetime(df.Timestamp)
    df = df[valid_columns]
    df = df.groupby('g_status').get_group(1.0)
    df = df.dropna(how='any')
    
    return df 

def getInputOutput(t):
    '''
        Get input and output data.
        @param t: Time index.
    '''

    global num_seq1, num_seq2, trRawDatasDF
    
    trInput1 = list(zip(trRawDatasDF.loc[t:(t + num_seq1 - 1), ['avg_a_power']].avg_a_power
                                     , trRawDatasDF.loc[t:(t + num_seq1 - 1), ['c_avg_ws1']].c_avg_ws1))
    output = list(trRawDatasDF.loc[(t + num_seq1 - 1):(t + num_seq1 + num_seq2 - 1 - 1)
                                                    , ['avg_rwd1']].avg_rwd1)
    trOutput = output
    trInput2 = [0.] + output[1:]
    
    return (trInput1, trOutput, trInput2)
                       
def main(args):
    '''
        Main.
        @param args: Arguments.
    '''
    
    hps = {}
    
    if args.mode == 'data':
        
        # Get arguments.
        rawDataPath = args.raw_data_path

        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)
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
        
        # Create training and validation data.
        ts = time.time()
        ymc = YawMisalignmentCalibrator(rawDataPath)
        ymc.createTrValTeData(hps)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
    elif args.mode == 'train':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)
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
        
        ts = time.time()
        ymc.train(hps, modelLoading)
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
    elif args.mode == 'evaluate':
        
        # Get arguments.
        rawDataPath = args.raw_data_path
        
        # hps.
        hps['num_seq1'] = int(args.num_seq1)
        hps['num_seq2'] = int(args.num_seq2)
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)
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
        hps['gru1_dim'] = int(args.gru1_dim)
        hps['gru2_dim'] = int(args.gru2_dim)
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
        
        ts = time.time()
        ymc.test(hps, modelLoading) #? 
        te = time.time()
        
        print('Elasped time: {0:f}s'.format(te-ts))
        
if __name__ == '__main__':
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode')
    parser.add_argument('--raw_data_path')
    parser.add_argument('--num_seq1')
    parser.add_argument('--num_seq2')
    parser.add_argument('--gru1_dim')
    parser.add_argument('--gru2_dim')    
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