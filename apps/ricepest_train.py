
# data st_num = all Station
# Optimizer = Adam
# loss = mae
# feature = lat - long 


############################################################################################################################################
# setup
from cProfile import label
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
from tensorflow.keras import layers

import platform
# print(platform.python_version())
# print(tf.version.VERSION)
# print(np.__version__)

from apps.data_operation import save_ann_model_to_postgis, load_all_ann_model_from_postgis, update_ann_model_to_postgis, update_dataset_title_ann_model_to_postgis, load_dataset_list

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

def bph_train_model(frames_train, frames_validation, Epochs):
    print(tf.version.VERSION)

    print(tf.config.list_physical_devices('GPU'))
    # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

    print(tf.test.is_built_with_cuda)
    # <function is_built_with_cuda at 0x7f4f5730fbf8>

    print(tf.test.gpu_device_name())
    # /device:GPU:0

    print(tf.config.get_visible_devices())
    # [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

    frames_train = frames_train.fillna(0)
    frames_validation = frames_validation.fillna(0)
    print(frames_train.head())
    #print(frames_train.tail())
    print(frames_validation.head())
    #print(frames_validation.tail())

    # specify the number of lag hours
    n_day = 14
    n_out = 3

    #frames_train = df
    values_train = frames_train.values    #ตัด header กับ idx ออก เป็น array matrix

    #frames_validation = df
    values_validation = frames_validation.values    #ตัด header กับ idx ออก เป็น array matrix

    

    n_features = frames_train.shape[1]

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
                    keras.layers.LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),activation='relu'),
                    keras.layers.Dense(units=1)
                ])
                
        Optimizer = tf.keras.optimizers.Adam(0.0001)
        model.compile(Optimizer, loss='mae', metrics=['accuracy'], run_eagerly=True)
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
        model = create_model(2)
        # Display the model's architecture
        model.summary()

    ############################################################################################################################################
    ## Fit model

    batch_size = 128

    # es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        #monitor="var_loss", 
        verbose=1, 
        save_weights_only=True,
        save_freq="epoch",
        period=100
        )

    """
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                min_delta=0.001, 
                                patience=0, 
                                verbose=0, 
                                mode='auto', 
                                baseline=None, 
                                restore_best_weights=False)
    """
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
    print(model_name)
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
    """
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
    """
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
    #lines_6 = ['train_year :',y_train_1,'-',y_train_2]
    #f.write('\t'.join(lines_6))
    #f.write('\n')
    lines_7 = ['time_lag :',str(n_day)]
    f.write('\t'.join(lines_7))
    f.write('\n')
    lines_8 = ['time_forecast :',str(n_out)]
    f.write('\t'.join(lines_8))
    f.write('\n')
    lines_9 = ['Program :','Time Series Forecasting for BPH with LSTMs - train mae-adam-data_Allst ']  
    f.write('\t'.join(lines_9))
    f.write('\n')
    lines_10 = ['model_funt :',str(2)]  
    f.write('\t'.join(lines_10))
    f.write('\n')
    lines_11 = ['Epochs :',str(Epochs)]  
    f.write('\t'.join(lines_11))    
    f.write('\n')
    lines_12 = ['Activation :','relu']  
    f.write('\t'.join(lines_12))    
    f.write('\n')
    f.close()

    return history, model_name


def bph_predict_model(frames_predict, model_file):
    frames_predict = frames_predict.fillna(0)

    #print(frames_predict.head())
    #print(frames_predict.tail())

    lat_lon_list = frames_predict.groupby(['latitude','longitude']).size().reset_index(name='count')
    #aa = [lat_lon_list.iloc[0][['latitude','longitude']].tolist()]
    #print(aa[0])

    #point_of_predict = pd.unique(frames_predict[['latitude','longitude']]).values

    #print(point_of_predict)
    n_features = frames_predict.shape[1]

    list_mse = []   
    list_st = []

    #for j in range(lat_lon_list.shape[0]):
    if True:
        
        # for j in range(1):
        

        #print(ll_df)
        print(frames_predict.columns)
        values_predict = frames_predict.values    #ตัด header กับ idx ออก เป็น array matrix
        #print(values_predict)
        n_features = frames_predict.shape[1]
        #print(n_features)
            # from matplotlib import pyplot as plt
            # plot_data(frames_train,file_name)
            # plot_data(frames_validation,file_name)
            # plot_data(frames_predict,file_name)

            # /////////////////////////////////////--------/////////////////////////////////////////////////////#
        df = frames_predict.reset_index()
        #print(df.head())
        #date_time_predict = pd.to_datetime(df.pop('date'))
        
            # ////////////////////////////////////---------////////////////////////////////////////////////////#
            # ensure all data is float
        values = values_predict.astype(float)
            # normalize features
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled = scaler.fit_transform(values)
                
        # specify the number of lag hours
        n_day = 14
        n_out = 3
        
        # frame as supervised learning
        reframed = series_to_supervised(scaled, n_day, n_out)
        # print(reframed.shape)
        # print(reframed.head())
            
        # predict datasets
        values = reframed.values
        test = values
        

        # predict into input and outputs
        n_obs = n_day * n_features
        test_X, test_y = test[:, :n_obs], test[:, -1]
        #print(test_X.shape, len(test_X), test_y.shape)
            
        # reshape input to be 3D [samples, timesteps, features]
        test_X = test_X.reshape((test_X.shape[0], n_day, n_features))
        #print(test_X.shape, test_y.shape)
            
        # # make a prediction
        ## Load model
        model_loaded = tf.keras.models.load_model(model_file)
        model_loaded.summary()

        yhat = model_loaded.predict(test_X)
        test_X_reshape = test_X.reshape((test_X.shape[0], n_day*n_features))
            
        # invert scaling for forecast
        inv_yhat = concatenate((test_X_reshape[:, :(n_features-1)], yhat),axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,-1]

        # invert scaling for actual
        test_y_reshape = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_X_reshape[:, :(n_features-1)],test_y_reshape), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,-1]

        forecast_data = (abs(inv_yhat[:])) 
        label_data =  inv_y[:]

        # calculate RMSE
        rmse = sqrt(mean_squared_error(label_data,forecast_data))

        print('Test RMSE: %.3f' % rmse)
        
        df_out = pd.DataFrame(frames_predict.index) 
        
        df_out['latitude'] = df['latitude']
        df_out['longitude'] = df['longitude']
        df_out['bph label'] = df['bph']
        df_out['bph forecast'] = 0
        #print(df_out.head())
        #print(df_out.tail())
        #print(df_out.iloc[:]['bph forecast'])
        #print(forecast_data)

        # df_out.iloc[14:,5]  = forecast_data.astype(int)    #for lag 1
        df_out.iloc[16:,4]  = forecast_data.astype(int)    #for lag 3
        # df_out.iloc[18:,5]  = forecast_data.astype(int)    #for lag 5
        # df_out.iloc[20:,5]  = forecast_data.astype(int)    #for lag 7
        print(df_out.head())
        print(df_out.tail())
    data_rmse = {'st_name': list_st,'rmse': list_mse}
    df_rmse = pd.DataFrame(data_rmse)
    
    return df_out

def blast_train_model(df, Epochs):
        ###########################################################################
    print("TensorFlow version :", tf.version.VERSION)
    ###########################################################################
    """
    Province_77 = ['Amnat Charoen','Ang Thong','Bangkok','Bueng Kan','Buri Ram',
    'Chachoengsao','Chai Nat','Chaiyaphum','Chanthaburi','Chiang Mai',
    'Chiang Rai','Chon Buri','Chumphon','Kalasin','Kamphaeng Phet',
    'Kanchanaburi','Khon Kaen','Krabi','Lampang','Lamphun',
    'Loei','Lop Buri','Mae Hong Son','Maha Sarakham','Mukdahan',
    'Nakhon Nayok','Nakhon Pathom','Nakhon Phanom','Nakhon Ratchasima','Nakhon Sawan',
    'Nakhon Si Thammarat','Nan','Narathiwat','Nong Bua Lam Phu','Nong Khai',
    'Nonthaburi','Pathum Thani','Pattani','Phang-nga','Phatthalung',
    'Phayao','Phetchabun','Phetchaburi','Phichit','Phitsanulok',
    'Phra Nakhon Si Ayutthaya','Phrae','Phuket','Prachin Buri','Prachuap Khiri Khan',
    'Ranong','Ratchaburi','Rayong','Roi Et','Sa kaeo',
    'Sakon Nakhon','Samut Prakarn','Samut Sakhon','Samut Songkhram','Saraburi',
    'Satun','Si Sa Ket','Sing Buri','Songkhla','Sukhothai',
    'Suphan Buri','Surat Thani','Surin','Tak','Trang',
    'Trat','Ubon Ratchathani','Udon Thani','Uthai Thani','Uttaradit',
    'Yala','Yasothon']
    ###########################################################################
    """
    trainning_mode = "newtrain"
    new_datetime_name = datetime.datetime.now().strftime("d%Y%m%d-t%H%M%S")

    # csv_file = 'Blast (2015-2021) level data.csv'
    """
    dataset_path = csv_file
    df = pd.read_csv(dataset_path, encoding="TIS-620")
    trainng_data_name = csv_file
    """
    ###########################################################################
    #df = df.drop(['date'], axis=1)
    #df = df.drop(['address'], axis=1)
    #df = df.drop(['year'], axis=1)
    blast_data = df.copy()
    print(blast_data.columns)
    ###########################################################################
    # get value
    values = blast_data.values
    # print(values.shape)
    # ensure all data is float
    
    values = values.astype('float32')
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
        ##########################################################################
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    #print(scaled.shape)
    ##########################################################################
    """
    var_in_col = ['พันธุ์พื้นเมือง', 'กข-6', 'กข-15', 'ขาวดอกมะลิ-105', 'สุพรรณบุรี-60,90',
                        'ราชการไวต่อแสง', 'ราชการไม่ไวต่อแสง', 'ชัยนาท-1', 'คลองหลวง-1',
                        'หอมสุพรรณบุรี', 'ปทุมธานี-1', 'สุพรรณบุรี-1']
    var_off_col = ['กข 10', 'กขไม่ไวแสง', 'สุพรรณบุรี 60-90', 'ราชการไม่ไวแสง', 'พิษณุโลก2 60-2',
                    'ชัยนาท 1-2', 'หอมสุพรรณบุรี', 'ปทุมธานี 1', 'สุพรรณบุรี 1']
    columns_Result = ['latitude','longitude','month','day',
                    'mint','maxt','temp','dew','humidity','precip',
                    var_in_col[0],var_in_col[1],var_in_col[2],var_in_col[3],
                    var_in_col[4],var_in_col[5],var_in_col[6],var_in_col[7],
                    var_in_col[8],var_in_col[9],var_in_col[10],var_in_col[11],
                    var_off_col[0],var_off_col[1],var_off_col[2],var_off_col[3],
                    var_off_col[4],var_off_col[5],var_off_col[6],var_off_col[7],
                    var_off_col[8],'bus','blast'
                    ]
    """
    #df_blast = pd.DataFrame(scaled,columns=columns_Result)
    df_blast = pd.DataFrame(scaled)
    ##########################################################################
    # max_province = 77
    lat_lon_list = df_blast.groupby([0,1]).size().reset_index(name='count')
    print(type(lat_lon_list.iloc[0]['count'].astype(int)))
    for num_province in range(len(lat_lon_list.iloc[:])):
        num_data = lat_lon_list.iloc[0]['count'].astype(int)
         # >>> 1 province year 2015-2021
        blast_prov_data =  df_blast[(num_province*num_data):(num_province*num_data)+num_data]
        # blast_wk_data[:5]
        ##########################################################################
        values = blast_prov_data.values
        # display(values.shape)
        # values[0]
        ##########################################################################
        # specify the number of lag week
        n_time_lag = 30
        n_time_forecast = 7
        n_features = df_blast.shape[1]
        # frame as supervised learning
        reframed = series_to_supervised(values, n_time_lag, n_time_forecast)
        # print(reframed.shape)
        # reframed[:5]
        ##########################################################################
        # split into train and test sets
        values = reframed.values
        n_train_day = num_data - 180
        # n_train_day = num_data - (n_time_lag+(n_time_forecast*15))
        train = values[:n_train_day, :]
        test = values[n_train_day:, :]
        # display(train.shape)
        # display(test.shape)
        ##########################################################################
        if num_province == 0:
            train_all = train
            test_all = test
        else:
            train_all = np.append(train_all,train, axis=0)
            test_all = np.append(test_all,test, axis=0)
        #     train_all = train_all.append(train, ignore_index = True)
        #     test_all = test_all.append(test, ignore_index = True)
        ##########################################################################
    print("n_features :", n_features)
    print(train_all.shape)
    print(test_all.shape)
    train = train_all
    test = test_all
    ##########################################################################
    # split into input and outputs
    n_obs = n_time_lag * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
    # print(train_X.shape, len(train_X), train_y.shape)
    # print(test_X.shape, len(test_X), test_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_time_lag, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_time_lag, n_features))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    ##########################################################################
    ##########################################################################
    # LSTM model
    # Define a LSTM sequential model
    optimi_zer = 'adam'
    los_s = 'mae'
    def create_model():      
        model = keras.Sequential()
        model.add(keras.Input(shape=(train_X.shape[1], train_X.shape[2])))
        model.add(layers.LSTM(8, return_sequences=False))
        model.add(layers.Dense(1))
        model.compile(optimizer=optimi_zer, loss=los_s, metrics=['accuracy'])      
        return model
    ##########################################################################
    Export_folder_name = "./Export_forecast_blast/"
    ## Make folder
    newfolder_name = str(n_time_lag)+"lag-"+str(n_time_forecast)+"forecast" # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # newfolder_name = "d"+str(_date)+"_t"+str(_time)+"_"+newfolder_name
    newfolder_name = new_datetime_name+"_"+newfolder_name
    path_newfolder = Export_folder_name
    path_newfolder_save = os.path.join(path_newfolder, newfolder_name)
    try: 
        os.mkdir(path_newfolder_save) 
    except OSError as error: 
        print(error)  
    print("Directory '% s' created" % path_newfolder_save)
    Export_folder_name = path_newfolder_save + '/'
    
    dataset_path = Export_folder_name+'lstm_ckpt'+'/'
    # Save checkpoints during training
    checkpoint_path = "lstm_train_1/cp-{epoch:04d}.ckpt"
    checkpoint_path = dataset_path + checkpoint_path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    ##########################################################################
    if trainning_mode == "newtrain":
        # Create a basic model instance
        model = create_model()
        # Display the model's architecture
        model.summary()

    ## save model.summary() to .txt
    from contextlib import redirect_stdout
    export_folder = path_newfolder_save + '/'
    save_txt = export_folder+'00_Model_Summary_'+new_datetime_name+'.txt'
    with open(save_txt, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    ##########################################################################
    ##########################################################################
    # Fit model
    Epochs = Epochs
    patience_values = int(Epochs*0.1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=patience_values)

    # log_dir = Export_folder_name+"logs/" + new_datetime_name
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    batch_size = 128
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        # monitor="loss", 
        verbose=1, 
        save_weights_only=True,
        save_freq="epoch",
        period=5
        )

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # fit network
    history = model.fit(train_X, train_y, 
                        epochs=Epochs, 
                        batch_size=batch_size, 
                        validation_data=(test_X, test_y), 
                        verbose=2,
                        # callbacks=[cp_callback],  
                        callbacks=[cp_callback, es_callback], 
                        shuffle=False)

    # save history
    history_name = "lstm_tr1_hist1.npy"
    history_file = dataset_path + history_name
    np.save(history_file,history.history)
    ##########################################################################
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
    # performance loss
    perf_loss  = round(history['loss'][-1], 5)
    # perf_loss = 0.00036
    # print('loss is : ', perf_loss)

    ## Save model
    pre_name = '_'+new_datetime_name+'_loss'+str(perf_loss)
    # Export_folder_name = Export_folder_name
    file_name = "model_lstm"+pre_name
    export_folder = Export_folder_name
    model_name = export_folder + file_name
    model.save(model_name)
    ##########################################################################
    export_folder = Export_folder_name
    plot_name = 'plot_histloss'+pre_name+'.png'
   
    ## save discription training data
    txt_name = '00_Training_model_data_discription.txt'
    filepath_save_txt = path_newfolder_save + "\\" + txt_name
    f = open(filepath_save_txt, "a")
    f.write(file_name)
    f.write('\n\n')
    #lines_1 = ['traing_data_name :',traing_data_name]
    #f.write('\t'.join(lines_1))
    #f.write('\n')
    lines_2 = ['loss :',str(perf_loss)]
    f.write('\t'.join(lines_2))
    f.write('\n')
    lines_3 = ['var_loss :',str(var_loss)]
    f.write('\t'.join(lines_3))
    f.write('\n')
    lines_4 = ['accuracy :',str(acc)]
    f.write('\t'.join(lines_4))
    f.write('\n')
    lines_5 = ['num_data :',str(num_data)]
    f.write('\t'.join(lines_5))
    f.write('\n')
    lines_6 = ['train_day :',str(n_train_day)]
    f.write('\t'.join(lines_6))
    f.write('\n')
    lines_7 = ['time_lag :',str(n_time_lag)]
    f.write('\t'.join(lines_7))
    f.write('\n')
    lines_8 = ['time_forecast :',str(n_time_forecast)]
    f.write('\t'.join(lines_8))
    f.write('\n')
    lines_9 = ['optimizer name :',str(optimi_zer)]
    f.write('\t'.join(lines_9))
    f.write('\n')
    lines_10 = ['loss name :',str(los_s)]
    f.write('\t'.join(lines_10))
    f.write('\n')
    f.close()

    return history, model_name

    ##########################################################################
    # make a prediction
    yhat = model.predict(test_X)
    test_X_reshape = test_X.reshape((test_X.shape[0], n_time_lag*n_features))
    # make a prediction
    yhat = model.predict(test_X)
    test_X_reshape = test_X.reshape((test_X.shape[0], n_time_lag*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((test_X_reshape[:, :(n_features-1)], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-1]
    # invert scaling for actual
    test_y_reshape = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X_reshape[:, :(n_features-1)], test_y_reshape), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

    ##########################################################################
    round_rmse = round(rmse,3)
    export_folder = Export_folder_name
    plot_name = 'plot_rmse'+str(round_rmse)+pre_name+'.png'

    # plot history
    from matplotlib import pyplot as plt
    def plot_rmse(inv_y,inv_yhat):
        plt.figure(figsize=(20, 8))
        plt.plot(inv_y, label='label',color='green')
        plt.plot(inv_yhat, label='forecast',color='orange')
        plt.legend()
        plt.grid(True)
        plt.ylabel(f'Rice blast area')
        plt.title(f'Rice blast Forecast') 
    #     plt.show()

    plot_rmse(inv_y,inv_yhat)
    plt.savefig(export_folder + "/" +plot_name)
##########################################################################
##########################################################################

def blast_predict_model(df, model_name):
        ###########################################################################
    print("TensorFlow version :", tf.version.VERSION)
    ###########################################################################
    """
    Province_77 = ['Amnat Charoen','Ang Thong','Bangkok','Bueng Kan','Buri Ram',
    'Chachoengsao','Chai Nat','Chaiyaphum','Chanthaburi','Chiang Mai',
    'Chiang Rai','Chon Buri','Chumphon','Kalasin','Kamphaeng Phet',
    'Kanchanaburi','Khon Kaen','Krabi','Lampang','Lamphun',
    'Loei','Lop Buri','Mae Hong Son','Maha Sarakham','Mukdahan',
    'Nakhon Nayok','Nakhon Pathom','Nakhon Phanom','Nakhon Ratchasima','Nakhon Sawan',
    'Nakhon Si Thammarat','Nan','Narathiwat','Nong Bua Lam Phu','Nong Khai',
    'Nonthaburi','Pathum Thani','Pattani','Phang-nga','Phatthalung',
    'Phayao','Phetchabun','Phetchaburi','Phichit','Phitsanulok',
    'Phra Nakhon Si Ayutthaya','Phrae','Phuket','Prachin Buri','Prachuap Khiri Khan',
    'Ranong','Ratchaburi','Rayong','Roi Et','Sa kaeo',
    'Sakon Nakhon','Samut Prakarn','Samut Sakhon','Samut Songkhram','Saraburi',
    'Satun','Si Sa Ket','Sing Buri','Songkhla','Sukhothai',
    'Suphan Buri','Surat Thani','Surin','Tak','Trang',
    'Trat','Ubon Ratchathani','Udon Thani','Uthai Thani','Uttaradit',
    'Yala','Yasothon']
    ###########################################################################
    """

    # csv_file = 'Blast (2015-2021) level data.csv'
    """
    dataset_path = csv_file
    df = pd.read_csv(dataset_path, encoding="TIS-620")
    trainng_data_name = csv_file
    """
    ###########################################################################
    #df = df.drop(['date'], axis=1)
    #df = df.drop(['address'], axis=1)
    #df = df.drop(['year'], axis=1)
    if 'created_at' in df.columns:
        df = df.drop(['created_at'], axis=1)
    df_cp = df.copy()
    #print(df_cp.head())
    #print(df_cp.columns)
    df_date = pd.to_datetime(df_cp.pop('date'),utc=False)
    df_lat = df_cp.pop('latitude')
    df_long = df_cp.pop('longitude') 
    blast_data = df.copy()
    if 'date' in blast_data.columns:
        blast_data = blast_data.drop(['date'], axis=1)
    print(blast_data.columns)

    ###########################################################################
    # get value
    values = blast_data.values

    values = values.astype(float)
    # print(values.shape)
    # ensure all data is float
    #values = values.astype('float32')
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
        ##########################################################################
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print("Yo")

    #print(scaled.shape)
    ##########################################################################
    """
    var_in_col = ['พันธุ์พื้นเมือง', 'กข-6', 'กข-15', 'ขาวดอกมะลิ-105', 'สุพรรณบุรี-60,90',
                        'ราชการไวต่อแสง', 'ราชการไม่ไวต่อแสง', 'ชัยนาท-1', 'คลองหลวง-1',
                        'หอมสุพรรณบุรี', 'ปทุมธานี-1', 'สุพรรณบุรี-1']
    var_off_col = ['กข 10', 'กขไม่ไวแสง', 'สุพรรณบุรี 60-90', 'ราชการไม่ไวแสง', 'พิษณุโลก2 60-2',
                    'ชัยนาท 1-2', 'หอมสุพรรณบุรี', 'ปทุมธานี 1', 'สุพรรณบุรี 1']
    columns_Result = ['latitude','longitude','month','day',
                    'mint','maxt','temp','dew','humidity','precip',
                    var_in_col[0],var_in_col[1],var_in_col[2],var_in_col[3],
                    var_in_col[4],var_in_col[5],var_in_col[6],var_in_col[7],
                    var_in_col[8],var_in_col[9],var_in_col[10],var_in_col[11],
                    var_off_col[0],var_off_col[1],var_off_col[2],var_off_col[3],
                    var_off_col[4],var_off_col[5],var_off_col[6],var_off_col[7],
                    var_off_col[8],'bus','blast'
                    ]
    """
    #df_blast = pd.DataFrame(scaled,columns=columns_Result)
    df_blast = pd.DataFrame(scaled)
    ##########################################################################
    # max_province = 77
    lat_lon_list = df_blast.groupby([0,1]).size().reset_index(name='count')
    
    print(type(lat_lon_list.iloc[0]['count'].astype(int)))
    for num_province in range(0,1):
        num_data = lat_lon_list.iloc[0]['count'].astype(int) # >>> 1 province year 2015-2021
        blast_prov_data =  df_blast[(num_province*num_data):(num_province*num_data)+num_data]
        # blast_wk_data[:5]
        ##########################################################################
        values = blast_prov_data.values
        # display(values.shape)
        # values[0]
        ##########################################################################
        # specify the number of lag week
        n_time_lag = 30
        n_time_forecast = 7
        n_features = df_blast.shape[1]
        # frame as supervised learning
        reframed = series_to_supervised(values, n_time_lag, n_time_forecast)
        # print(reframed.shape)
        # reframed[:5]
        ##########################################################################
        # split into train and test sets
        # display(train.shape)
        # display(test.shape)
        values = reframed.values
        train = values
        test = values
        ##########################################################################
        if num_province == 0:
            train_all = train
            test_all = test
        else:
            train_all = np.append(train_all,train, axis=0)
            test_all = np.append(test_all,test, axis=0)
        #     train_all = train_all.append(train, ignore_index = True)
        #     test_all = test_all.append(test, ignore_index = True)
        ##########################################################################
    print("n_features :", n_features)
    print(train_all.shape)
    print(test_all.shape)
    train = train_all
    test = test_all
    ##########################################################################
    # split into input and outputs
    n_obs = n_time_lag * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
    # print(train_X.shape, len(train_X), train_y.shape)
    # print(test_X.shape, len(test_X), test_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_time_lag, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_time_lag, n_features))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    ##########################################################################
    ##########################################################################
    model_loaded = tf.keras.models.load_model(model_name)
    print("model :", model_name)

    # make a prediction
    yhat = model_loaded.predict(test_X)
    test_X_reshape = test_X.reshape((test_X.shape[0], n_time_lag*n_features))
    # invert scaling for forecast
    inv_yhat = concatenate((test_X_reshape[:, :(n_features-1)], yhat), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,-1]
    # invert scaling for actual
    test_y_reshape = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_X_reshape[:, :(n_features-1)], test_y_reshape), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,-1]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)
    ##########################################################################


    round_rmse = round(rmse,3)

    # replace <0 by 0
    inv_yhat_fill = list(inv_yhat)
    inv_yhat_fill = [0 if j < 0 else j for j in inv_yhat_fill]      # filter <<<<<<<<<<<<<<
    inv_yhat_fill = [0 if j<0.05 and j>0 else j for j in inv_yhat_fill]  # filter <<<<<<<<<<<<<<
    inv_yhat_fill = np.array(inv_yhat_fill)

    ########################################################################
    # Export 77 Province
    num_data = lat_lon_list.iloc[0]['count'].astype(int) #(2015-2021)
    n_train_day = 0
    # n_time_lag = 21
    # n_time_forecast = 7
    ########################################################################
    # Date
    df_pov_date = df_date[:num_data].copy()
    # max_province = 77
    for n_p in range(0,1):
    # n_p = 0
        # n_d = 2192-21-7+1
        n_d = num_data-n_train_day-n_time_lag-n_time_forecast+1
        ########################################################################
        forecast_data =  inv_yhat[(n_p*n_d):((n_p*n_d)+n_d)]
        label_data =  inv_y[(n_p*n_d):(n_p*n_d)+n_d]
        date_column = df_pov_date[-(len(forecast_data)):]
        #df_lat_now = df_lat[(n_p*num_data)]
        #df_long_now = df_long[(n_p*num_data)]
        ########################################################################
        # replace <0 by 0
        forecast_data_list = list(forecast_data)
        list_predict = [0 if j < 0 else j for j in forecast_data_list]      # filter <<<<<<<<<<<<<<
        list_predict = [0 if j<0.05 and j>0 else j for j in list_predict]  # filter <<<<<<<<<<<<<<
        filted_forecast_data = np.array(list_predict)
        forecast_data = filted_forecast_data
        ########################################################################
        df_forecast_data = pd.DataFrame(date_column.values,columns=['date'])
        #df_forecast_data['latitude'] = df_lat_now
        #df_forecast_data['longitude'] = df_long_now
        print(label_data)
        df_forecast_data['blast label'] = label_data
        df_forecast_data['blast forecast'] = forecast_data


    return df_forecast_data