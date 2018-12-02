valid_columns = ['avg_a_power'
                 , 'avg_rwd1'
                 , 'max_rwd1'
                 , 'min_rwd1'
                 , 'avg_rwd2'
                 , 'max_rwd2'
                 , 'min_rwd2'
                 , 'avg_ws1'
                 , 'avg_ws2'
                 , 'corr_factor_anem1'
                 , 'corr_factor_anem2'
                 , 'corr_offset_anem1'
                 , 'corr_offset_anem2'
                 , 'offset_anem1'
                 , 'offset_anem2'
                 , 'slope_anem1'
                 , 'slope_anem2'
                 , 'avg_nacelle_pos'
                 , 'avg_r_speed'
                 , 'alarm_code'
                 , 'g_status'
                 , 'Turbine_no']

bDF_13 = pd.read_csv('2013_F.csv')
bDF_13.index = pd.to_datetime(bDF_13.Timestamp)
bDF_13 = bDF_13[valid_columns]
bDF_13 = bDF_13.sort_values(by='avg_a_power', ascending=False)
bDF_13 = bDF_13.iloc[0:int(bDF_13.shape[0]*0.1),:]
bDF_13 = bDF_13.sort_values(by='Timestamp')
bDF_13_G = bDF_13.groupby('Turbine_no')

bDF_13_b18 = bDF_13_G.get_group('B18')
bDF_13_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

bDF_14 = pd.read_csv('2014_F.csv')
bDF_14.index = pd.to_datetime(bDF_14.Timestamp)
bDF_14 = bDF_14[valid_columns]
bDF_14 = bDF_14.sort_values(by='avg_a_power', ascending=False)
bDF_14 = bDF_14.iloc[0:int(bDF_14.shape[0]*0.1),:]
bDF_14 = bDF_14.sort_values(by='Timestamp')
bDF_14_G = bDF_14.groupby('Turbine_no')

bDF_14_b18 = bDF_14_G.get_group('B18')
bDF_14_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

bDF_15 = pd.read_csv('2015_F.csv')
bDF_15.index = pd.to_datetime(bDF_15.Timestamp)
bDF_15 = bDF_15[valid_columns]
bDF_15 = bDF_15.sort_values(by='avg_a_power', ascending=False)
bDF_15 = bDF_15.iloc[0:int(bDF_15.shape[0]*0.1),:]
bDF_15 = bDF_15.sort_values(by='Timestamp')
bDF_15_G = bDF_15.groupby('Turbine_no')

bDF_15_b18 = bDF_15_G.get_group('B18')
bDF_15_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

bDF_16 = pd.read_csv('2016_F.csv')
bDF_16.index = pd.to_datetime(bDF_16.Timestamp)
bDF_16 = bDF_16[valid_columns]
bDF_16 = bDF_16.sort_values(by='avg_a_power', ascending=False)
bDF_16 = bDF_16.iloc[0:int(bDF_16.shape[0]*0.1),:]
bDF_16 = bDF_16.sort_values(by='Timestamp')
bDF_16_G = bDF_16.groupby('Turbine_no')

bDF_16_b18 = bDF_16_G.get_group('B18')
bDF_16_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

bDF_17 = pd.read_csv('2017_F.csv')
bDF_17.index = pd.to_datetime(bDF_17.Timestamp)
bDF_17 = bDF_17[valid_columns]
bDF_17 = bDF_17.sort_values(by='avg_a_power', ascending=False)
bDF_17 = bDF_17.iloc[0:int(bDF_17.shape[0]*0.1),:]
bDF_17 = bDF_17.sort_values(by='Timestamp')
bDF_17_G = bDF_17.groupby('Turbine_no')

bDF_17_b18 = bDF_17_G.get_group('B18')
bDF_17_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

bDF_18 = pd.read_csv('2018_F.csv')
bDF_18.index = pd.to_datetime(bDF_18.Timestamp)
bDF_18 = bDF_18[valid_columns]
bDF_18 = bDF_18.sort_values(by='avg_a_power', ascending=False)
bDF_18 = bDF_18.iloc[0:int(bDF_18.shape[0]*0.1),:]
bDF_18 = bDF_18.sort_values(by='Timestamp')
bDF_18_G = bDF_18.groupby('Turbine_no')

bDF_18_b18 = bDF_18_G.get_group('B18')
bDF_18_b18.avg_rwd1.plot(figsize=(20,10))
plt.grid()

for a, b in zip(eval_rwds, real_rwds):
    print('{0:15f}  {1:15f}  {2:15f}'.format(a, b, a - b))