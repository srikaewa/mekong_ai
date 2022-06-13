
# data st_num = all Station
# Optimizer = Adam
# loss = mae
# feature = lat - long 


############################################################################################################################################
# setup
import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from math import sqrt
from numpy import concatenate
import numpy as np
import math

import pandas as pd
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import tensorflow 
import tensorflow as tf
from tensorflow import keras

import platform
# print(platform.python_version())
# print(tf.version.VERSION)
# print(np.__version__)

############################################################################################################################################
# setup parameter

# specify the number of lag hours
n_day = 14
n_out = 7

st = 0
mode_st = 1    # 0 = 1 station  , 1  = all station

Epochs = 30

y_train_1 = '2015'
y_train_2 = '2017' 
y_val_1 = '2018'
y_val_2 = '2018'
y_pre_1 = '2019'
y_pre_2 = '2019'

model_test = 2

############################################################################################################################################
## data Preparation 

# File name and Path
path_adress1 = ".\\template-Data Parameters Required for Brown planthopper\\"
path_adress2 = "\\station\\"
path_adress3 = ".\\Import_Dataset\\"

# ข้อมูล พิกัดสถานที่เเละรายชื้ออ้างอิงสถานที่ตรวจวัด
file_name_st = 'Data_lat_long_Rice research Center'
csv_file =path_adress1 + path_adress2 + file_name_st + '.csv'

df_st = pd.read_csv(csv_file)
print("All low RiceCenter {} station" .format(df_st.shape[0]))

# st_num = [0,1,2,3,4,6,8,11,12,13,14,15,17,18,20,21,23,27,29,30]
st_num = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]   # เลือกสถานีวัดแมลง


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

############################################################################################################################################
## load data

# Preparing Pandas Dataframes for Machine Learning

def creat_dataset(st_BPH = 0, y_1 = '2015',y_2 = '2019',mode = 1):
    for i in range(len(st_num)):
        st = i
        file_name = df_st['nameEng'][st_num[st]]
        name_input = file_name
        name_locals = 'stN_' + file_name   
        
        ## File name and Path
        csv_file  = path_adress3 + name_input + '.csv'
        dataset = pd.read_csv(csv_file,header=0, index_col=0,encoding="TIS-620" )  #index_col=0 , index_col=None
        dataset = dataset.drop(['address'], axis=1)
        # dataset = dataset.drop(['latitude'], axis=1)
        # dataset = dataset.drop(['longitude'], axis=1)
        
        dataset = dataset.rename(columns = {'พันธุ์พื้นเมือง':'Varieties-1',
        'กข-6':'Varieties-2',
        'กข-15':'Varieties-3',
        'ขาวดอกมะลิ-105':'Varieties-4',
        'สุพรรณบุรี-60,90':'Varieties-5',
        'ราชการไวต่อแสง':'Varieties-6',
        'ราชการไม่ไวต่อแสง':'Varieties-7',
        'ชัยนาท-1':'Varieties-8',
        'คลองหลวง-1':'Varieties-9',
        'หอมสุพรรณบุรี':'Varieties-10',
        'ปทุมธานี-1':'Varieties-11',
        'สุพรรณบุรี-1':'Varieties-12',
        'กข 10':'Varieties-13',
        'กขไม่ไวแสง':'Varieties-14',
        'สุพรรณบุรี 60-90':'Varieties-15',
        'ราชการไม่ไวแสง':'Varieties-16',
        'พิษณุโลก2 60-2':'Varieties-17',
        'ชัยนาท 1-2':'Varieties-18',
        'ปทุมธานี 1':'Varieties-19',
        'สุพรรณบุรี 1':'Varieties-20'}, inplace = False)

        locals()[name_locals] = dataset
        # print(f'Dataframe name_station: {st+1 , name_input}')
        print('wait......')
        del dataset
        clear_output(wait=True)

    date_start = y_1 + '-01' + '-01'
    date_stop = y_2 + '-12' + '-31'  

    new_colum_1 = ['mirid bug','mint','maxt','temp','dew','humidity','wspd','wdir','precip','Varieties-1','Varieties-2','Varieties-3','Varieties-4'
                  ,'Varieties-5','Varieties-6','Varieties-7','Varieties-8','Varieties-9','Varieties-10','Varieties-11','Varieties-12','Varieties-13'
                  ,'Varieties-14','Varieties-15','Varieties-16','Varieties-17','Varieties-18','Varieties-19','Varieties-20','bph']
    new_colum_2 = ['latitude','longitude','mirid bug','mint','maxt','temp','dew','humidity','wspd','wdir','precip','Varieties-1','Varieties-2','Varieties-3','Varieties-4'
                  ,'Varieties-5','Varieties-6','Varieties-7','Varieties-8','Varieties-9','Varieties-10','Varieties-11','Varieties-12','Varieties-13'
                  ,'Varieties-14','Varieties-15','Varieties-16','Varieties-17','Varieties-18','Varieties-19','Varieties-20','bph']
#---------------------------------------------------------------------------------------------------------------------------------------------------#
    if mode == 0:
        file_name = df_st['nameEng'][st_BPH]
        locals_input = 'stN_' + file_name
        print(locals_input)
        dataset_st=locals()[locals_input].loc[date_start:date_stop]   
        frames_st = dataset_st
    else:
        m = 0
        for j in range(len(st_num)):
        # for j in range(df_st.shape[0]):
            # file_name = df_st['nameEng'][j]
            file_name = df_st['nameEng'][st_num[j]]
            locals_input = 'stN_' + file_name
            print(locals_input)
            dataset_st=locals()[locals_input].loc[date_start:date_stop]
            clear_output(wait=True)
            if m == 0:
                frames_st = dataset_st
                m=m+1
                print(m)
            else:    
                frames_st = [frames_st,dataset_st]
                frames_st  = pd.concat(frames_st)
#---------------------------------------------------------------------------------------------------------------------------------------------------#
    frames=frames_st[new_colum_2]
    return frames
#---------------------------------------------------------------------------------------------------------------------------------------------------#

# plot ตรวจสอบข้อมูล dataset 
def plot_data(frames_train,df_name):
    plt.figure()
    df_plot = frames_train
    df_plot.plot(lw=1,grid=True,figsize=(13,30),subplots=True)
    plt.xlabel('Date time-'+ df_name)
    plt.legend()
    # plt.show()   
#------------------------------------------------------------------------------------------------------------------------------------------#

############################################################################################################################################
## load data for AI  station  for train and validation

frames_train = creat_dataset(st,y_train_1,y_train_2,mode_st)
values_train = frames_train.values    #ตัด header กับ idx ออก เป็น array matrix

frames_validation = creat_dataset(st,y_val_1,y_val_2,mode_st)
values_validation = frames_validation.values    #ตัด header กับ idx ออก เป็น array matrix

frames_predict = creat_dataset(st,y_pre_1,y_pre_2,mode_st)
values_predict = frames_predict.values    #ตัด header กับ idx ออก เป็น array matrix

n_features = frames_train.shape[1]

if mode_st == 0:
    file_name = df_st['nameEng'][st]
else:
    file_name = 'station_All'

# from matplotlib import pyplot as plt

# plot_data(frames_train,file_name)
# plot_data(frames_validation,file_name)
# plot_data(frames_predict,file_name)

# /////////////////////////////////////--------/////////////////////////////////////////////////////#

#train data
# ensure all data is float
values = values_train.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, n_day, n_out)
# print(reframed.shape)
# print(reframed.head())

# predict datasets
values = reframed.values
train = values

#input 
n_obs = n_day * n_features
# train_X, train_y = train[:, :n_obs], train[:, -n_features]
train_X, train_y = train[:, :n_obs], train[:, -1]
print(train_X.shape, len(train_X), train_y.shape)  #for train


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_day, n_features))
print(train_X.shape, train_y.shape)

#validation data
# ensure all data is float
values = values_validation.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, n_day, n_out)

# predict datasets
values = reframed.values
test = values

#output 
n_obs = n_day * n_features
# test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(test_X.shape, len(test_X), test_y.shape)  #for train

# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], n_day, n_features))
print(test_X.shape, test_y.shape)

############################################################################################################################################
## LSTM model

# Define a LSTM sequential model
def create_model(model_funt=0):
      if model_funt == 0:
            model = tf.keras.models.Sequential([
                  keras.layers.LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, return_sequences=True),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, return_sequences=True),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, return_sequences=True),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, return_sequences=False),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.Dense(1)
            ])
      elif model_funt == 1:
            model = tf.keras.models.Sequential([
                  # Shape [batch, time, features] => [batch, time, lstm_units]
                  keras.layers.LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True,activation='relu'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, return_sequences=True, activation='relu'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.LSTM(512, activation='relu'),
                  keras.layers.BatchNormalization(),
                  keras.layers.Dropout(0.2),
                  keras.layers.Dense(units=1)
            ])
      elif model_funt == 2:                                            
            model = tf.keras.models.Sequential([
                  # Shape [batch, time, features] => [batch, time, lstm_units]
                  keras.layers.LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2]),activation='relu'),
                  keras.layers.Dense(units=1)
            ])
            
      Optimizer = tf.keras.optimizers.Adam(0.0001)
      model.compile(Optimizer, loss='mae', metrics=['accuracy'])
      # model.summary()
      return model  


#------------------------------------------------------------------------------------------------------------------------------------------#
# Set Day and Time
current_time = datetime.datetime.now() 
as_string = str(current_time)
print(as_string[0:19])
_date = as_string[0:10]
_time = as_string [11:13] + '-' + as_string [14:16] + '-' + as_string [17:19]

Export_folder_name = "./Export_lstm_BPH/"

# Make folder
newfolder_name = str(n_day)+"_lag_"+str(n_out)+"-forecast" 
newfolder_name = "d"+str(_date)+"_t"+str(_time)+"_"+newfolder_name
path_newfolder = Export_folder_name
path_newfolder_save = os.path.join(path_newfolder, newfolder_name)
try: 
    os.mkdir(path_newfolder_save) 
except OSError as error: 
    print(error)  
print("Directory '% s' created" % path_newfolder_save)
Export_folder_name = path_newfolder_save + '/'

#------------------------------------------------------------------------------------------------------------------------------------------#
dataset_path = Export_folder_name+'lstm_ckpt'+'/'
# Save checkpoints during training

checkpoint_path = "lstm_train_1/cp-{epoch:04d}.ckpt"
checkpoint_path = dataset_path + checkpoint_path
checkpoint_dir = os.path.dirname(checkpoint_path)

#------------------------------------------------------------------------------------------------------------------------------------------#
trainning_mode = "newtrain"

if trainning_mode == "newtrain":

    # Create a basic model instance
    model = create_model(model_test)
    # Display the model's architecture
    model.summary()

############################################################################################################################################
## Fit model

batch_size = 128

# es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    # monitor="var_loss", 
    verbose=1, 
    save_weights_only=True,
    save_freq="epoch",
    period=100
    )

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# fit network
# Epochs = 5000
history = model.fit(train_X, train_y, 
                    epochs=Epochs, 
                    batch_size=batch_size, 
                    validation_data=(test_X, test_y), 
                    verbose=2, 
                    callbacks=[cp_callback], 
                    shuffle=False)

# save history
history_name = "lstm_tr1_hist1.npy"
history_file = dataset_path + history_name
np.save(history_file,history.history)

#------------------------------------------------------------------------------------------------------------------------------------------#
# Load the previously saved weights
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# load history
history=np.load(history_file,allow_pickle='TRUE').item()

# evaluate the model
loss, acc = model.evaluate(test_X, test_y, verbose=2)
var_loss = round(loss,5)
print('Accuracy : ', acc)
print('var_loss is : ', var_loss)

############################################################################################################################################
## Save model

# performance loss
perf_loss  = round(history['loss'][-1], 5)
print('loss is : ', perf_loss)

## Save model
pre_name = "_d"+str(_date)+"_t"+str(_time)+"_loss_"+str(perf_loss)

# Export_folder_name = Export_folder_name
file_name = "model_lstm"+pre_name
# import time
# tic_lm = time.perf_counter()

export_folder = Export_folder_name
model_name = export_folder + file_name
model.save(model_name)

# toc_lm = time.perf_counter()
# print("\n",f"Time to save Model is {toc_lm - tic_lm:0.4f} seconds")

#------------------------------------------------------------------------------------------------------------------------------------------#
## save model.summary() to .txt
from contextlib import redirect_stdout
export_folder = path_newfolder_save + '/'
save_txt = export_folder+'00_Model_Summary_'+pre_name+'.txt'
with open(save_txt, 'w') as f:
    with redirect_stdout(f):
        model.summary()

#------------------------------------------------------------------------------------------------------------------------------------------#
export_folder = Export_folder_name
plot_name = 'plot_histloss'+pre_name+'.png'
# plot history
from matplotlib import pyplot as plt
def plot_loss(history):
    plt.figure()
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
#     plt.show()
plot_loss(history)
plt.savefig(export_folder + plot_name)

export_folder = Export_folder_name
plot_name = 'plot_histacc'+pre_name+'.png'
# plot history
from matplotlib import pyplot as plt
def plot_loss(history):
    plt.figure()
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
#     plt.show()
plot_loss(history)
plt.savefig(export_folder + plot_name)

#------------------------------------------------------------------------------------------------------------------------------------------#
## save discription training data
txt_name = '00_Training_model_data_discription.txt'
filepath_save_txt = path_newfolder_save + "\\" + txt_name
f = open(filepath_save_txt, "a")
f.write(file_name)
f.write('\n\n')
lines_2 = ['loss :',str(perf_loss)]
f.write('\t'.join(lines_2))
f.write('\n')
lines_3 = ['var_loss :',str(var_loss)]
f.write('\t'.join(lines_3))
f.write('\n')
lines_4 = ['accuracy :',str(acc)]
f.write('\t'.join(lines_4))
f.write('\n')
lines_5 = ['num_features :',str(n_features)]
f.write('\t'.join(lines_5))
f.write('\n')
lines_6 = ['train_year :',y_train_1,'-',y_train_2]
f.write('\t'.join(lines_6))
f.write('\n')
lines_7 = ['time_lag :',str(n_day)]
f.write('\t'.join(lines_7))
f.write('\n')
lines_8 = ['time_forecast :',str(n_out)]
f.write('\t'.join(lines_8))
f.write('\n')
lines_9 = ['Program :','Time Series Forecasting for BPH with LSTMs - train mae-adam-data_Allst ']  
f.write('\t'.join(lines_9))
f.write('\n')
lines_10 = ['model_funt :',str(model_test)]  
f.write('\t'.join(lines_10))
f.write('\n')
lines_11 = ['Epochs :',str(Epochs)]  
f.write('\t'.join(lines_11))    
f.write('\n')
lines_12 = ['Activation :','relu']  
f.write('\t'.join(lines_12))    
f.write('\n')
f.close()

############################################################################################################################################
## Performance

df = frames_predict.reset_index()
date_time_predict = pd.to_datetime(df.pop('date'))

# ensure all data is float
values = values_predict.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
    
# frame as supervised learning
reframed = series_to_supervised(scaled, n_day, n_out)
# print(reframed.shape)
# print(reframed.head())
    
# predict datasets
values = reframed.values
# n_train_day = int(values.shape[0]*0.9)
# train = values[:n_train_day, :]
test = values
    
# predict into input and outputs
n_obs = n_day * n_features
# test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(test_X.shape, len(test_X), test_y.shape)
    
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], n_day, n_features))
print(test_X.shape, test_y.shape)

#------------------------------------------------------------------------------------------------------------------------------------------#
# ensure all data is float
values = values_predict.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
    
# frame as supervised learning
reframed = series_to_supervised(scaled, n_day, n_out)
# print(reframed.shape)
# print(reframed.head())
    
# predict datasets
values = reframed.values
# n_train_day = int(values.shape[0]*0.9)
# train = values[:n_train_day, :]
test = values
    
# predict into input and outputs
n_obs = n_day * n_features
# test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -1]
print(test_X.shape, len(test_X), test_y.shape)
    
# reshape input to be 3D [samples, timesteps, features]
test_X = test_X.reshape((test_X.shape[0], n_day, n_features))
print(test_X.shape, test_y.shape)
    
# # make a prediction
yhat = model.predict(test_X)
test_X_reshape = test_X.reshape((test_X.shape[0], n_day*n_features))
    
# invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, -29:]), axis=1)
inv_yhat = concatenate((test_X_reshape[:, :(n_features-1)], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

# invert scaling for actual
test_y_reshape = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, -29:]), axis=1)
inv_y = concatenate((test_X_reshape[:, :(n_features-1)], test_y_reshape), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

#------------------------------------------------------------------------------------------------------------------------------------------#
export_folder = Export_folder_name
plot_name = 'Performance'+pre_name+'.png'
# plot history
from matplotlib import pyplot as plt
def plot_Perfor(history):
    plt.figure()
    plt.plot(date_time_predict[n_day+n_out-1:],inv_y[:],label='data test')
    plt.plot(date_time_predict[n_day+n_out-1:],inv_yhat[:],label='prediction')
 
    plt.ylabel('BPH volume')
    plt.xlabel('Datetime')
    plt.title(file_name +'  Test RMSE: %.3f' % rmse)
    plt.legend()
    plt.grid(True)
#     plt.show()
plot_Perfor(history)
plt.savefig(export_folder + plot_name)