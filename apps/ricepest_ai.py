# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from faulthandler import disable
from lib2to3.pgen2.token import GREATER
from apps import ricepest_train
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from hydralit import HydraHeadApp
from apps.data_operation import save_ann_model_to_postgis, load_all_ann_model_from_postgis, update_ann_model_to_postgis, update_dataset_title_ann_model_to_postgis, load_dataset_list, load_bph_model_data, update_ann_model_name_to_postgis, update_training_ann_model_to_postgis

from apps.ricepest_train import bph_train_model, bph_predict_model, bph_retrain_model



class RicePestAIDashboard(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        #st.experimental_set_query_params(selected=self.title)
        print(self.title)
        st.subheader("Rice Pest Deep Neural Network with LSTM Builder")

        model_df = load_all_ann_model_from_postgis()

        row1_1, row1_2 = st.columns((2,1))
        with row1_1:
            st.markdown("**Deep Neural Network with LSTM**")
            st.markdown("This is neural network builder for modeling rice pest outbreak. Create new builder and start import input and target data to build a model. The neural network with LSTM can be configured to achieve different performance. Once data is all ready, train the model and use the trained model to make prediction. This process is computationally expensive. When the model is ready, head to Rice Pest Dashboard section to view the result of prediction")
        
        with row1_2:
            with st.expander("Create Neural Network Model with LSTM"):
                st.markdown("This default configuration has been tested with optimum performance. Normally, after neural network model is created, some of its parameters **_cannot_** be changed, such as number of layers, number of neuron for each layer, etc. The newly created model will appear in the model select box below.")
                with st.form("Deep Learning with LSTM Configuration", clear_on_submit = True):
                    text_input_ann_name = st.text_input("Enter Model Title", placeholder="Rice Pest Outbreak Model #1")
                    number_of_layers = st.select_slider(
                        'Select a number of neural network layers',
                        options=[1, 3, 5])
                    number_of_neurons = st.select_slider(
                        'Select a number of neuron for each layer',
                        options=[128, 256, 512])
                    optimizer = st.selectbox("Select optimizer for training", ("Adam", "SGD"))
                    epoch = st.slider("Select number of epoch", 10, 5000, 50, 10)
                    number_of_days_in = st.slider('Select a number of previous days required for prediction', min_value=7, max_value=30, value=14)
                    number_of_days_out = st.slider('Select a day-th for prediction', min_value=1, max_value=10, value=3)
                    submit_configuration_button = st.form_submit_button("Create")
                    if submit_configuration_button:
                        if text_input_ann_name:
                            save_ann_model_to_postgis(text_input_ann_name, number_of_layers, number_of_neurons, optimizer, epoch,number_of_days_in, number_of_days_out)
                        else:
                            st.error("Model title cannot be empty!")
            with st.form("Select Model", clear_on_submit = True):
                ml = model_df.loc[:]['title'].values.tolist()
                ml = ['-'] + ml
                select_model = st.selectbox("Select from available models", ml)
                print(ml)
                button_select_model = st.form_submit_button("Load")
                if button_select_model:
                    print("index:", ml.index(select_model))
                    print(select_model)

        if select_model != '-':
            dataset_list = load_dataset_list("data_model_input")
            st.markdown("---")            
            model = model_df.loc[ml.index(select_model)-1]
            if 'model_id' not in st.session_state:
                st.session_state['model_id'] = model['idname']
            #print("HERE :: " + st.session_state.model_id)
            st.info("### Selected Model :: " + model['title'])
            #st.write(model['id'])
            row2_1, row2_2 = st.columns((1, 4))
            
            with row2_1:
                with st.form(select_model, clear_on_submit = False):
                    text_update_ann_name = st.text_input("Model title", placeholder="Rice Pest Outbreak Model #1", value=model['title'])
                    update_number_of_layers = st.select_slider(
                        'Number of neural network layers', value=model['n_layer'],
                        options=[1, 3, 5], disabled=True)
                    update_number_of_neurons = st.select_slider(
                        'Number of neuron for each layer',
                        options=[128, 256, 512], disabled=True)
                    sv = ["Adam", "SGD"]
                    update_optimizer = st.selectbox("Optimizer for training", options=sv, index=sv.index(model['optimizer']))
                    epoch = st.slider("Select number of epoch", min_value=10, max_value=5000, value=int(model['n_epoch']), step=10)
                    ndays_in = st.slider('Select a number of previous days required for prediction', min_value=2, max_value=10, value=int(model['n_day']), disabled=True)
                    ndays_out = st.slider('Select a day-th for prediction', min_value=1, max_value=10, value=int(model['n_out']), disabled=True)
                    update_configuration_button = st.form_submit_button("Update")
                    if update_configuration_button:
                        if text_update_ann_name:
                            update_ann_model_to_postgis("ann_model", model['id'],text_update_ann_name,  epoch)
                        else:
                            st.error("Model title cannot be empty!")

            with row2_2:
                with st.expander("Input Data Frame", expanded=False):
                    if model['dataset_title'] is not None:
                        selected_dataset = st.selectbox("Select dataset for this model", dataset_list['dataset_title'])    
                    else:
                        selected_dataset = st.selectbox("Select dataset for this model", dataset_list['dataset_title'])
                    submit_input_button = st.button("Update Input Dataset")
                    if submit_input_button:
                        update_dataset_title_ann_model_to_postgis("ann_model", model['id'], selected_dataset)
                        st.success("Input added into model.")

                    if model['dataset_title'] is not None:
                        model_df = load_bph_model_data(model['dataset_title'])
                        model_data_years_list = pd.DatetimeIndex(model_df['date']).year.unique()

                        data_training_year_options = st.multiselect("Select year of data for training", model_data_years_list.values, model_data_years_list.values[0:len(model_data_years_list)-2])   
                        
                        if not data_training_year_options:
                            st.error("Training data connot be empty!")
                        
                        data_validation_year_options = st.multiselect("Select year of data for validation", model_data_years_list.values, model_data_years_list.values[len(model_data_years_list)-2])

                        if not data_validation_year_options:
                            st.error("Validation data cannot be empty")
                        
                        if data_training_year_options:
                            if data_validation_year_options:
                                check = any(item in data_validation_year_options for item in data_training_year_options)
                                if check:
                                    st.warning("It is recommended to use different set of data for training and validation")
                        else:
                            st.error("Validation data connot be empty!")

                        data_prediction_year_options = st.multiselect("Select year of data for prediction", model_data_years_list.values, model_data_years_list.values[len(model_data_years_list)-1])

                        if data_prediction_year_options:
                            if model_data_years_list[len(model_data_years_list)-1] not in  data_prediction_year_options:
                                st.warning("Data for prediction should be the latest set of all data.")
                        else:
                            st.warning("Prediction data is needed to test the model!")
                
            

            
                with st.expander("Train the model"):
                    st.write("Train the model using selected dataset. Typically, Data will be yearly split into 3 parts. First part is the training data. This is the biggest proportion of all data, e.g. 3 years of daily data (2015 - 2017). Second part is for validation, e.g. 1 year of daily data (2018). The last part is for prediction/testing, e.g. 1 year of daily data (2019). By clicking Train button, the model will be train from zero while Re-Train will have the model continue training from last train.")

                    if model['dataset_title'] is not None:
                        #print("train number :: " + model['n_train'])   
                        if int(model['n_train']) > 0:                     
                            st.metric("Accuracy", value="{:.2f}%".format(model['accuracy']*100), delta="{:.2f}%".format(100*(model['accuracy']-model['last_accuracy'])))
                            print(model)
                            st.metric("Loss", value="{:.2f}".format(model['loss']), delta="{:.2f}".format(model['loss']-model['last_loss']))
                        if int(model['n_train']) == 0:                        
                            st.info("This is the first time training for this model...")
                        button_train_model = st.button("Train")
                        if int(model['n_train']) > 0:
                            button_retrain_model = st.button("Re-train")
                            if button_retrain_model:
                                model_df = load_bph_model_data(model['dataset_title'])
                                model_df = model_df.drop(['address','geometry','dataset_title'], axis=1)
                                bph_column = model_df.pop('bph')
                                model_df.insert(len(model_df.columns), 'bph', bph_column)
                                training_set = pd.DataFrame()
                                for yr in data_training_year_options:
                                    sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                    training_set = pd.concat([training_set, sdf])
                                
                                training_set = training_set.drop(['date'], axis=1)

                                validation_set = pd.DataFrame()
                                for yr in data_validation_year_options:
                                    sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                    validation_set = pd.concat([validation_set, sdf])
                                
                                validation_set = validation_set.drop(['date'], axis=1)
                                #print(validation_set.head())
                                
                                #print(prediction_set.head())
                                with st.spinner("Re-training model... this can take very long to finish."):
                                    history, acc, var_loss = bph_retrain_model(training_set, validation_set, model['n_epoch'], model['model_name'], model['n_day'], model['n_out'], model['n_layer'], model['n_neuron'])
                                    print(history)
                                    print(str(acc) + " :: " + str(var_loss))
                                    update_ann_model_name_to_postgis("ann_model", model['id'], model['model_name'])

                                    print(len(history['loss']))
                                    source = pd.DataFrame({
                                        'epoch': np.arange(len(history['loss'])),
                                        'loss': history['loss'],
                                        'validate_loss': history['val_loss'],
                                    })
                                    dmelt = source.melt('epoch', var_name='loss', value_name='value')
                                    c = alt.Chart(dmelt, title='Model Training Loss').mark_line().encode(x='epoch', y='value', color='loss')

                                    st.altair_chart(c, use_container_width=True)
                            
                                    update_training_ann_model_to_postgis("ann_model", model['id'], int(model['n_train'] + 1), int(model['total_epoch'] + model['n_epoch']), model['accuracy'] , model['loss'], acc, var_loss)
                                    st.write("Training saved")      

                        
                        st.markdown("---")
                        #print(model)
                        st.markdown("_Last trained at_ : " + model['last_trained_at'].strftime("%d/%m/%Y, %H:%M:%S"))
                        if button_train_model:
                            model_df = load_bph_model_data(model['dataset_title'])
                            model_df = model_df.drop(['address','geometry','dataset_title'], axis=1)
                            bph_column = model_df.pop('bph')
                            model_df.insert(len(model_df.columns), 'bph', bph_column)
                            training_set = pd.DataFrame()
                            for yr in data_training_year_options:
                                sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                training_set = pd.concat([training_set, sdf])
                            
                            training_set = training_set.drop(['date'], axis=1)

                            validation_set = pd.DataFrame()
                            for yr in data_validation_year_options:
                                sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                validation_set = pd.concat([validation_set, sdf])
                            
                            validation_set = validation_set.drop(['date'], axis=1)
                            #print(validation_set.head())
                            
                            #print(prediction_set.head())
                            with st.spinner("Training model... this can take very long to finish."):
                                history, model_name, acc, var_loss = bph_train_model(training_set, validation_set, model['n_epoch'], model['n_day'], model['n_out'], model['n_layer'], model['n_neuron'])
                                print(history)
                                print(str(acc) + " :: " + str(var_loss))
                                update_ann_model_name_to_postgis("ann_model", model['id'], model_name)

                                print(len(history['loss']))
                                source = pd.DataFrame({
                                    'epoch': np.arange(len(history['loss'])),
                                    'loss': history['loss'],
                                    'validate_loss': history['val_loss'],
                                })
                                dmelt = source.melt('epoch', var_name='loss', value_name='value')
                                c = alt.Chart(dmelt, title='Model Training Loss').mark_line().encode(x='epoch', y='value', color='loss')

                                st.altair_chart(c, use_container_width=True)
                        
                                update_training_ann_model_to_postgis("ann_model", model['id'], 1, model['n_epoch'], 0 , 0, acc, var_loss)
                                st.write("Training saved")

                        
                with st.expander("Predict BPH"):                    
                    st.write("Predict the model using selected dataset. With the nature of time-series modeling, all the input need to be fed into the model. The current LSTM model requires 14 previous days of data to predict the 3rd day of future.  process the input and predict the output. In most case for predict future value, the previusly predicted value is used for prediction. The predicting result corresponding to the prediction data will be saved into database for utilization.")
                    if model['dataset_title'] is not None:
                        with st.form("Select Input Date Range for Prediction"):
                            #model_df = load_bph_model_data(model['dataset_title'])

                            #model_df = model_df.drop(['address','geometry', 'dataset_title'], axis=1)
                            
                            bph_column = model_df.pop('bph')
                            model_df.insert(len(model_df.columns), 'bph',bph_column)
                            
                            prediction_set = pd.DataFrame()
                            
                            for yr in data_prediction_year_options:
                                sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                prediction_set = pd.concat([prediction_set, sdf])
                            
                            #prediction_date = prediction_set.pop('date') 
                            #print(prediction_set.columns)
                            #prediction_set = prediction_set.drop(['address', 'geometry','dataset_title'],axis=1)
                            lat_lon_list = prediction_set.groupby(['latitude','longitude','address']).size().reset_index(name='count')

                            if 'address' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['address'],axis=1)
                            if 'geometry' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['geometry'],axis=1)
                            if 'dataset_title' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['dataset_title'],axis=1)
                            prediction_set = prediction_set.fillna(0)
                            #print(selected_rows.head())
                            
                            #print(lat_lon_list.shape[0])
                            #tt = prediction_set[(prediction_set['latitude'] == 6.752) & (prediction_set['longitude'] == 101.13)]
                    
                            #start_color, end_color = st.select_slider(
                            #'Select a date range of input data',
                            #options=[drange],
                            #value=(drange[0],drange[len(drange)-1]))

                            
                            lat_lon_list_2 = []
                            for i in range(lat_lon_list.shape[0]):
                                lat_lon_list_2.append((lat_lon_list.iloc[i][0], lat_lon_list.iloc[i][1], lat_lon_list.iloc[i][2]))
                            #print(lat_lon_list_2)
                            selected_predicted_point = st.selectbox("Select (lat,lon) to see the predicted BPH", options=lat_lon_list_2)
                            #print(selected_predicted_point[1])
                            predict_button = st.form_submit_button("Predict")
                            if predict_button:                        
                                with st.spinner("Predicting... this might not take long. It depends on size of data to be predicted."):
                                    point_df = prediction_set[(prediction_set['latitude'] == selected_predicted_point[0]) & (prediction_set['longitude'] == selected_predicted_point[1])]
                                    prediction_date = point_df.pop('date')
                                    #print(prediction_date.head())
                                    predict_bph = bph_predict_model(point_df, model['model_name'], model['n_day'], model['n_out'])     
                                    #print(prediction_date.tolist()) 
                                    predict_bph['date'] = prediction_date.tolist()
                                    #print(predict_bph.head())                          
                                    
                                    chart_data = predict_bph[pd.DatetimeIndex(predict_bph['date']).year == yr][['date','bph_label','bph_forecast']]  
                                    print(chart_data)
                                    selection = alt.selection_multi(fields=['forecast_data'], bind='legend')                  
                                    lc = alt.Chart(chart_data.melt('date', var_name='forecast_data', value_name='value')).mark_area().encode(x='date', y='value', color='forecast_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                                    st.altair_chart(lc, use_container_width=True)
                                    
                                    #print(all_predict_bph.head())
                                    #print(all_predict_bph.tail())
                                
                        with st.form("Predict All"):
                            #model_df = load_bph_model_data(model['dataset_title'])

                            #model_df = model_df.drop(['address','geometry', 'dataset_title'], axis=1)
                            
                            bph_column = model_df.pop('bph')
                            model_df.insert(len(model_df.columns), 'bph',bph_column)
                            
                            prediction_set = pd.DataFrame()
                            
                            for yr in data_prediction_year_options:
                                sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                prediction_set = pd.concat([prediction_set, sdf])
                            
                            #prediction_date = prediction_set.pop('date') 
                            #print(prediction_set.columns)
                            #prediction_set = prediction_set.drop(['address', 'geometry','dataset_title'],axis=1)
                            lat_lon_list = prediction_set.groupby(['latitude','longitude','address']).size().reset_index(name='count')

                            if 'address' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['address'],axis=1)
                            if 'geometry' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['geometry'],axis=1)
                            if 'dataset_title' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['dataset_title'],axis=1)
                            prediction_set = prediction_set.fillna(0)
                            #print(selected_rows.head())
                            
                            #print(lat_lon_list.shape[0])
                            #tt = prediction_set[(prediction_set['latitude'] == 6.752) & (prediction_set['longitude'] == 101.13)]
                    
                            #start_color, end_color = st.select_slider(
                            #'Select a date range of input data',
                            #options=[drange],
                            #value=(drange[0],drange[len(drange)-1]))

                            
                            lat_lon_list_2 = []
                            for i in range(lat_lon_list.shape[0]):
                                lat_lon_list_2.append((lat_lon_list.iloc[i][0], lat_lon_list.iloc[i][1], lat_lon_list.iloc[i][2]))

                            predict_button = st.form_submit_button("Predict All")
                            if predict_button:                        
                                with st.spinner("Predicting all input location... this might not take long. It depends on size of data to be predicted."):
                                    all_predict_bph = pd.DataFrame(columns=['0', 'latitude', 'longitude', 'bph_label', 'bph_forecast', 'date'])
                                    #print(all_predict_bph)
                                    for i in range(lat_lon_list.shape[0]):
                                        point_df = prediction_set[(prediction_set['latitude'] == lat_lon_list.iloc[i][0]) & (prediction_set['longitude'] == lat_lon_list.iloc[i][1])]
                                        prediction_date = point_df.pop('date')
                                        #print(prediction_date.head())
                                        predict_bph = bph_predict_model(point_df, model['model_name'], model['n_day'], model['n_out'])      
                                        #print(prediction_date.tolist()) 
                                        predict_bph['date'] = prediction_date.tolist()
                                        print(predict_bph.head())
                                        all_predict_bph = pd.concat([all_predict_bph, predict_bph])

                                    print("All predicted :: ")       
                                    all_predict_bph = all_predict_bph[['latitude',  'longitude', 'bph_label', 'bph_forecast', 'date']]
                                    print(all_predict_bph.head())                          
                                    st.pydeck_chart(pdk.Deck(
                                        map_style='mapbox://styles/mapbox/light-v9',
                                        initial_view_state=pdk.ViewState(
                                            latitude=all_predict_bph['latitude'].mean(),
                                            longitude=all_predict_bph['longitude'].mean(),
                                            zoom=5,
                                            pitch=50,
                                        ),
                                        layers=[
                                            pdk.Layer(
                                                'HexagonLayer',
                                                data=all_predict_bph,
                                                get_position='[longitude, latitude]',
                                                radius=200,
                                                elevation_scale=4,
                                                elevation_range=[0, 1000],
                                                pickable=True,
                                                extruded=True,
                                            ),
                                            pdk.Layer(
                                                "ColumnLayer",
                                                data=all_predict_bph,
                                                get_position=["longitude", "latitude"],
                                                get_elevation="bph_forecast",
                                                elevation_scale=100,
                                                radius=5000,
                                                get_fill_color=[255, 0, 0, 140],
                                                pickable=True,
                                                auto_highlight=True,
                                            ),
                                        ],
                                    ))
                                    
                                    #print(all_predict_bph.head())
                                    #print(all_predict_bph.tail())
