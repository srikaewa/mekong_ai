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
from apps.data_operation import save_drought_ann_model_to_postgis, load_all_drought_ann_model_from_postgis, update_ann_model_to_postgis, update_dataset_title_ann_model_to_postgis, load_dataset_list, load_drought_model_data, update_ann_model_name_to_postgis

from apps.ricepest_train import drought_train_model, drought_predict_model



class DroughtAIDashboard(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        #st.experimental_set_query_params(selected=self.title)
        print(self.title)
        st.subheader("Rice Drought Deep Neural Network with LSTM Builder")

        model_df = load_all_drought_ann_model_from_postgis()


        row1_1, row1_2 = st.columns((2,1))
        with row1_1:
            st.markdown("**Deep Neural Network with LSTM**")
            st.markdown("This is neural network builder for modeling drought natural disaster. Create new builder and start import input and target data to build a model. The neural network with LSTM can be configured to achieve different performance. Once data is all ready, train the model and use the trained model to make prediction. This process is computationally expensive. When the model is ready, head to Drought Dashboard section to view the result of prediction")
        
        with row1_2:
            with st.expander("Create Neural Network Model with LSTM"):
                st.markdown("This default configuration has been tested with optimum performance. Normally, after neural network model is created, some of its parameters **_cannot_** be changed, such as number of layers, number of neuron for each layer, etc. The newly created model will appear in the model select box below.")
                with st.form("Deep Learning with LSTM Configuration", clear_on_submit = True):
                    text_input_ann_name = st.text_input("Enter Model Title", placeholder="Drought Model #1")
                    number_of_layers = st.select_slider(
                        'Select a number of neural network layers',
                        options=[1, 3, 5])
                    number_of_neurons = st.select_slider(
                        'Select a number of neuron for each layer',
                        options=[16, 32, 64])
                    optimizer = st.selectbox("Select optimizer for training", ("Adam", "SGD"))                    
                    epoch = st.slider("Select number of epoch", 10, 500, 50, 10)
                    number_of_days_in = st.slider('Select a number of previous days required for prediction', min_value=2, max_value=10, value=5)
                    number_of_days_out = st.slider('Select a day-th for prediction', min_value=1, max_value=10, value=2)
                    submit_configuration_button = st.form_submit_button("Create")
                    if submit_configuration_button:
                        if text_input_ann_name:
                            save_drought_ann_model_to_postgis(text_input_ann_name, number_of_layers, number_of_neurons, optimizer, epoch, number_of_days_in, number_of_days_out)
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
            dataset_list = load_dataset_list("data_drought_model_input")
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
                        options=[8, 16, 32], disabled=True)
                    sv = ["Adam", "SGD"]
                    update_optimizer = st.selectbox("Optimizer for training", options=sv, index=sv.index(model['optimizer']))
                    epoch = st.slider("Select number of epoch", min_value=10, max_value=5000, value=int(model['n_epoch']), step=10)
                    
                    ndays_in = st.slider('Select a number of previous days required for prediction', min_value=2, max_value=10, value=int(model['n_day']), disabled=True)
                    ndays_out = st.slider('Select a day-th for prediction', min_value=1, max_value=10, value=int(model['n_out']), disabled=True)

                    update_configuration_button = st.form_submit_button("Update")
                    if update_configuration_button:
                        if text_update_ann_name:
                            update_ann_model_to_postgis("drought_ann_model", model['id'],text_update_ann_name,  epoch)
                        else:
                            st.error("Model title cannot be empty!")

            with row2_2:
                with st.expander("Input Data Frame", expanded=False):
                    selected_dataset = st.selectbox("Select dataset for this model", dataset_list['dataset_title'])
                    if selected_dataset:
                        submit_input_button = st.button("Update Input Dataset")
                        if submit_input_button:
                            print(selected_dataset)
                            update_dataset_title_ann_model_to_postgis("drought_ann_model", model['id'], selected_dataset)
                            st.success("Input added into model.")

                    if model['dataset_title'] is not None:
                        model_df = load_drought_model_data(model['dataset_title'])
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
                    st.write("Train the model using selected dataset. Typically, Data will be yearly split into 3 parts. First part is the training data. This is the biggest proportion of all data, e.g. 3 years of daily data (2015 - 2017). Second part is for validation, e.g. 1 year of daily data (2018). The last part is for prediction/testing, e.g. 1 year of daily data (2019).")

                    if model['dataset_title'] is not None:
                        button_train_model = st.button("Train")
                        st.markdown("---")
                        #print(model)
                        st.markdown("_Last trained at_ : " + model['last_trained_at'].strftime("%d/%m/%Y, %H:%M:%S"))
                        if button_train_model:
                            model_df = load_drought_model_data(model['dataset_title'])
                            if 'address' in model_df.columns:
                                model_df = model_df.drop(['address'], axis=1)    
                            if 'geometry' in model_df.columns:
                                model_df = model_df.drop(['geometry'], axis=1)    
                            if 'dataset_title' in model_df.columns:
                                model_df = model_df.drop(['dataset_title'], axis=1)    
                            if 'created_at' in model_df.columns:
                                model_df = model_df.drop(['created_at'], axis=1)    
                            #drought_column = model_df.pop('drought_lv')
                            #model_df.insert(len(model_df.columns), 'bph', bph_column)
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
                            with st.spinner("Training drought ANN model... this can take very long to finish."):
                                history, model_name = drought_train_model(training_set, validation_set, model['n_epoch'], model['n_day'], model['n_out'])
                                #print(history)
                                update_ann_model_name_to_postgis("drought_ann_model", model['id'], model_name)

                                print(len(history['loss']))
                                source = pd.DataFrame({
                                    'epoch': np.arange(len(history['loss'])),
                                    'loss': history['loss'],
                                    'validate_loss': history['val_loss'],
                                })
                                dmelt = source.melt('epoch', var_name='loss', value_name='value')
                                c = alt.Chart(dmelt, title='Model Training Loss').mark_line().encode(x='epoch', y='value', color='loss')

                                st.altair_chart(c, use_container_width=True)

                with st.expander("Predict Drought"):
                    st.write("Predict the model using selected dataset. With the nature of time-series modeling, all the input need to be fed into the model. The current LSTM model requires 14 previous days of data to predict the 3rd day of future.  process the input and predict the output. In most case for predict future value, the previusly predicted value is used for prediction. The predicting result corresponding to the prediction data will be saved into database for utilization.")
                    if model['dataset_title'] is not None:
                        with st.form("Select Input Date Range for Prediction"):
                            #model_df = load_bph_model_data(model['dataset_title'])

                            #model_df = model_df.drop(['address','geometry', 'dataset_title'], axis=1)
                            
                            drought_column = model_df.pop('area_drought')
                            model_df.insert(len(model_df.columns), 'area_drought', drought_column)
                            
                            prediction_set = pd.DataFrame()
                            
                            for yr in data_prediction_year_options:
                                sdf = model_df[model_df['date'].dt.strftime('%Y') == str(yr)]
                                prediction_set = pd.concat([prediction_set, sdf])
                            
                            #prediction_date = prediction_set.pop('date') 
                            if 'address' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['address'],axis=1)
                            if 'geometry' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['geometry'],axis=1)
                            if 'dataset_title' in prediction_set.columns:
                                prediction_set = prediction_set.drop(['dataset_title'],axis=1)
                            prediction_set = prediction_set.fillna(0)
                            #print(selected_rows.head())
                            lat_lon_list = prediction_set.groupby(['latitude','longitude']).size().reset_index(name='count')
                            #print(lat_lon_list.shape[0])
                            #tt = prediction_set[(prediction_set['latitude'] == 6.752) & (prediction_set['longitude'] == 101.13)]
                    
                            #start_color, end_color = st.select_slider(
                            #'Select a date range of input data',
                            #options=[drange],
                            #value=(drange[0],drange[len(drange)-1]))

                            
                            lat_lon_list_2 = []
                            for i in range(lat_lon_list.shape[0]):
                                lat_lon_list_2.append((lat_lon_list.iloc[i][0], lat_lon_list.iloc[i][1]))
                            #print(lat_lon_list_2)
                            selected_predicted_point = st.selectbox("Select (lat,lon) to see the predicted BPH", options=lat_lon_list_2)
                            #print(selected_predicted_point[1])
                            predict_button = st.form_submit_button("Predict")
                            if predict_button:                        
                                with st.spinner("Predicting... this might not takelong. It depends on size of data to be predicted."):
                                    point_df = prediction_set[(prediction_set['latitude'] == selected_predicted_point[0]) & (prediction_set['longitude'] == selected_predicted_point[1])]
                                    """
                                    prediction_date = point_df[['date']]
                                    if 'date' in point_df.columns:
                                        point_df = point_df.drop(['date'], axis=1)
                                    print("Yo")
                                    print(prediction_date)
                                    predict_drought = drought_predict_model(point_df, model['model_name'], model['n_day'], model['n_out'])     
                                    #print(prediction_date.tolist()) 
                                    #predict_drought = predict_drought.iloc[:,1:]
                                    #print(prediction_date['date'].strftime('%Y.%m.%d'))
                                    #print(prediction_date)
                                    #print(prediction_date['date'].dt.strftime('%Y-%m-%d').tolist())
                                    prediction_date = prediction_date['date'].dt.strftime('%Y-%m-%d').tolist()
                                    #print(len(prediction_date) - int(model['n_day'])+int(model['n_out'])-1)
                                    prediction_date = prediction_date[0:int(model['n_day'])+int(model['n_out'])-1]
                                    #print(prediction_date)
                                    #prediction_date = prediction_date.apply(lambda x: x.dt.strftime('%Y-%m-%d'))
                                    #print(prediction_date.tolist())
                                    #predict_drought['date'] = predict_drought.insert(loc=0, column='date', value=prediction_date)
                                    #predict_drought['date'] = predict_drought['date'].replace([1,2,3,4,5])
                                                          

                                    predict_drought.drop(columns = predict_drought.columns[0], axis = 1, inplace= True)
                                    print(predict_drought)
                                    print(predict_drought.shape)
                                    predict_drought['date'] = ['2000-01','2000-02','2000-03','2000-04','2000-05','2000-06','2000-07','2000-08','2000-09','2000-10','2000-11','2000-12']
                                    print(predict_drought)    

                                    chart_data = predict_drought[['date','drought_label','drought_forecast']]
                                    print(chart_data) 
                                    selection = alt.selection_multi(fields=['forecast_data'], bind='legend')    
             
                                    lc = alt.Chart(chart_data.melt('date', var_name='forecast_data', value_name='value')).mark_bar().encode(x='date', y='value', color='forecast_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                                    #lc = alt.Chart(chart_data).mark_bar().encode(x='date', y='drought_forecast', color='drought_forecast')

                                    st.altair_chart(lc, use_container_width=True)
                                    """
                                    prediction_date = point_df.pop('date')
                                    #print(prediction_date.head())
                                    predict_drought = drought_predict_model(point_df, model['model_name'], model['n_day'], model['n_out'])    
                                    #print(prediction_date.tolist()) 
                                    print(prediction_date)
                                    predict_drought['date'] = prediction_date.tolist()
                                    print(predict_drought.head())                          
                                    
                                    chart_data = predict_drought[pd.DatetimeIndex(predict_drought['date']).year == yr][['date','drought_label','drought_forecast']]  
                                    print(chart_data)
                                    selection = alt.selection_multi(fields=['forecast_data'], bind='legend')                  
                                    lc = alt.Chart(chart_data.melt('date', var_name='forecast_data', value_name='value')).mark_bar().encode(x='date', y='value', color='forecast_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                                    st.altair_chart(lc, use_container_width=True)
                                    
                                    #print(all_predict_bph.head())
                                    #print(all_predict_bph.tail())
                                
                        