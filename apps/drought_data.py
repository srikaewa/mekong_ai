# Copyright 2018-2019 Streamlit Inc.'
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

from pkgutil import get_data

from sqlalchemy import all_
import streamlit as st
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
import altair as alt
import pydeck as pdk
from hydralit import HydraHeadApp
from datetime import date, datetime, timedelta
from io import StringIO
import altair as alt
from vega_datasets import data
import plotly.graph_objects as go
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import time
from vega_datasets import data

from apps.data_operation import save_to_postgis, load_from_postgis, load_ann_model_from_postgis, load_dataset_list, load_all_ann_model_from_postgis, load_drought_model_data


class DroughtDataLayers(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        #st.experimental_set_query_params(selected=self.title)
        print("--------------" + self.title + "---------------")
        
        #st.markdown(
        #    """
        #<style>
        #span[data-baseweb="tag"] {
        #background-color: #229954 !important;
        #}
        #</style>
        #""",
        #    unsafe_allow_html=True,
        #)
        
        #thailand_boundary = gpd.read_file('Thailand Boundary.shp')

        @st.cache
        def load_ann_model(idname):
            model = load_ann_model_from_postgis(idname, ['title','n_layer','n_neuron','optimizer','n_epoch','idname','created_at'])
            return model

        @st.cache
        def get_data_distinct(sql):
            data = load_from_postgis(sql)
            years_list = pd.DatetimeIndex(data['date']).year.unique()
            return [str(x) for x in years_list.values]

        @st.cache
        def select_dataset_from_database(selected_dataset):
            return load_drought_model_data(selected_dataset)

        #all_model = load_all_ann_model_from_postgis()
        #print(all_model)

        with st.sidebar:
            st.header('Data Management')
            st.subheader('Drought - Natural Disaster')
            
            dataset_list = load_dataset_list("data_drought_model_input")
            selected_dataset = st.selectbox("Select dataset to view data", ['-'] + dataset_list['dataset_title'].tolist())

            if selected_dataset != '-':
                drought_df = select_dataset_from_database(selected_dataset)
                #drought_df = drought_df.dropna(axis=1)
                selected_data_layer = st.multiselect("Select data layer to view",['Drought Area','Weather'])
                drought_years_list = pd.DatetimeIndex(drought_df['date']).year.unique()
                drought_year_options = st.multiselect("Select year of data", drought_years_list.values, drought_years_list.values[0])  
 

        st.subheader("Drought Monitoring")
        row1_1, row1_2 = st.columns((2,5))
        st.write("View time-series drought GIS data from database.")

        if selected_dataset != '-':
            if selected_data_layer:
                selected_columns = ['date','address', 'latitude','longitude']
                chart_columns = ['date']
                for x in selected_data_layer:
                    if x == 'Drought Area':
                        selected_columns.append('area_drought')
                        chart_columns.append('area_drought')
                    if x == 'Weather':
                        selected_columns = selected_columns + ['temp', 'humidity', 'avg_ssm']
                        wchart_columns = ['date','temp', 'humidity', 'avg_ssm']
                    """
                    if x == 'Rice Cultivated Area':
                        #selected_columns = selected_columns + ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19','r20','r21','r22','r23','r24','r25','r26','r27','r28','r29','r30','r31','r32']
                        selected_columns = selected_columns + ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16']
                        chart_columns = chart_columns + ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15','r16']
                    """
                #print(selected_columns)
                if selected_data_layer != '-' and len(drought_year_options) > 0:
                    selected_layer_df = drought_df[selected_columns]
                buff_set = pd.DataFrame()
                for yr in drought_year_options:
                    sdf = selected_layer_df[selected_layer_df['date'].dt.strftime('%Y') == str(yr)]
                    buff_set = pd.concat([buff_set, sdf])
                    #selected_layer_df = bph_df[selected_columns]
                gb = GridOptionsBuilder.from_dataframe(buff_set)
                gb.configure_pagination(paginationAutoPageSize=True)#Add pagination
                gb.configure_side_bar() #Add a sidebar
                gb.configure_selection('multiple', use_checkbox=True,groupSelectsChildren="Group checkbox select children")#Enable multi-row selection
                gb.configure_column('date', headerCheckboxSelection =True)
                gridOptions = gb.build()
                with st.spinner("Loading data..."):
                    #AgGrid(selected_layer_df, fit_columns_on_grid_load=True)
                    grid_response = AgGrid(
                        buff_set,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=True,
                        theme='blue', #Add theme color to the table
                        enable_enterprise_modules=True,
                        height=500, 
                        width='100%',
                        reload_data=False,
                        editable=True
                    )
                    bph_all_df = grid_response['data']
                    selected = grid_response['selected_rows'] 
                    selected_df = pd.DataFrame(selected)
                with st.expander("View all drought & weather data chart", expanded=False):
                    st.write("all chart")
                    for yr in drought_year_options:
                        for x in selected_data_layer:
                            if x == 'Drought Area':
                                chart_data = selected_layer_df[pd.DatetimeIndex(selected_layer_df['date']).year == yr][chart_columns]
                                #print(chart_data)
                                dmelt = chart_data.melt('date', var_name='drought', value_name='value')
                                #print(dmelt)
                                #print(dmelt)
                                lc = alt.Chart(dmelt).mark_bar().encode(
                                        x='date', y='value', color='drought', tooltip=['date','value'])
                                st.altair_chart(lc, use_container_width=True)
                            if x == 'Weather':
                                wchart_data = selected_layer_df[pd.DatetimeIndex(selected_layer_df['date']).year == yr][wchart_columns]
                                wmelt = wchart_data.melt('date', var_name='weather', value_name='value')
                                #print(dmelt)
                                wc = alt.Chart(wmelt).mark_bar().encode(
                                        x='date', y='value', color='weather', tooltip=['date','value'])
                                st.altair_chart(wc, use_container_width=True)
                            
                        
                st.markdown("Selected **" + str(selected_df.shape[0]) + "** rows of data")
                #print(selected_df.shape[0])
                
                COLOR_BREWER_BLUE_SCALE = [
                    [240, 249, 232],
                    [204, 235, 197],
                    [168, 221, 181],
                    [123, 204, 196],
                    [67, 162, 202],
                    [8, 104, 172],
]
                if not selected_df.empty:
                    print(selected_df)
                    st.pydeck_chart(pdk.Deck(
                            map_style='mapbox://styles/mapbox/dark-v9',
                            initial_view_state=pdk.ViewState(
                            latitude=selected_df['latitude'].mean(),
                            longitude=selected_df['longitude'].mean(),
                            zoom=5,
                            pitch=50,
                        ),
                        layers=[
                            pdk.Layer(
                                "HeatmapLayer",
                                data=selected_df,
                                opacity=0.9,
                                get_position=["longitude", "latitude"],
                                threshold=0.75,
                                aggregation=pdk.types.String("MEAN"),
                                get_weight="area_drought",
                                pickable=True,
                            ),
                        ],
                    ))
            """
            page_names = ['Brown Plant Hopper','Cyrtorhinus Lividipennis Reuter','Weather', 'Rice Cultivated Area']
            
            page = st.selectbox("Choose data layer", page_names)
                                                      
            if page == 'Brown Plant Hopper':                                
                dataset_list = load_dataset_list("data_bph")
                print(dataset_list['dataset_title'])
                selected_dataset = st.multiselect("Select dataset to view data", dataset_list['dataset_title'], dataset_list['dataset_title'][0])
                #selected_dataset2 = st.selectbox("Select dataset to view data", ['-'] + dataset_list['dataset_title'].tolist())
                print(selected_dataset)
                if selected_dataset:
                    #print(selected_dataset)
                    sql = " OR ".join({"dataset_title = '{}'".format(x) for x in selected_dataset})
                    sql = "SELECT *, geometry AS geom FROM data_bph WHERE (" + sql + ") ORDER BY date DESC"
                    bph_data_distinct_year_list = get_data_distinct(sql)
                    selected_bph_data_distinct_year_list = st.multiselect("Select year to view data", bph_data_distinct_year_list, bph_data_distinct_year_list[0])         
                

            elif page == 'Cyrtorhinus Lividipennis Reuter':
                clr_data_distinct_year_list = get_data_distinct("SELECT *, geometry AS geom FROM data_clr ORDER BY date DESC")
                selected_clr_data_distinct_year_list = st.multiselect("Select year to view data", clr_data_distinct_year_list, clr_data_distinct_year_list[0]) 
                        
            elif page == 'Weather':
                weather_data_distinct_year_list = get_data_distinct("SELECT *, geometry AS geom FROM data_weather ORDER BY date DESC")
                selected_weather_data_distinct_year_list = st.multiselect("Select year to view data", weather_data_distinct_year_list, weather_data_distinct_year_list[0]) 
                            
            elif page == 'Rice Cultivated Area':
                rice_cultivated_area_data_distinct_year_list = get_data_distinct("SELECT *, geometry AS geom FROM data_rice_cultivated_area ORDER BY date DESC")
                selected_rice_cultivated_area_data_distinct_year_list = st.multiselect("Select year to view data", rice_cultivated_area_data_distinct_year_list, rice_cultivated_area_data_distinct_year_list[0])                 

            st.markdown("---")
            st.markdown("<p align=right> Powered by <b>MeKong Institute</b></p>", unsafe_allow_html=True)                               
        """

        

        """
        if page == 'Brown Plant Hopper':
            # LAYING OUT THE TOP SECTION OF THE APP
            st.subheader("Brown Plant Hopper Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("View time-series brown plant hopper GIS data from database.")
            if selected_dataset:
                if selected_bph_data_distinct_year_list is not None:
                    sql = " OR ".join({"date_part('year', date) = {}".format(x) for x in selected_bph_data_distinct_year_list})       
                    sql2 = " OR ".join({"dataset_title = '{}'".format(x) for x in selected_dataset})         
                    sql = "SELECT *, geometry AS geom FROM data_bph WHERE (" + sql + ") AND (" + sql2 + ") ORDER BY date DESC"
                    #print(sql)
                    bph_df = read_data_from_database(sql)
                    #AgGrid(bph_df)
                    gb = GridOptionsBuilder.from_dataframe(bph_df)
                    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                    gb.configure_side_bar() #Add a sidebar
                    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                    gb.configure_column('date', headerCheckboxSelection = True)
                    gridOptions = gb.build()
                    grid_response = AgGrid(
                        bph_df,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT', 
                        update_mode='MODEL_CHANGED', 
                        fit_columns_on_grid_load=True,
                        theme='blue', #Add theme color to the table
                        enable_enterprise_modules=True,
                        height=500, 
                        width='100%',
                        reload_data=True,
                        editable=True
                    )
                    bph_df = grid_response['data']
                    selected = grid_response['selected_rows'] 
                    selected_df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
                    selected_bph_data_distinct_year_list.sort()
                    with st.expander("View all BPH data chart", expanded=False):
                        for yr in selected_bph_data_distinct_year_list:
                            chart_data = bph_df[pd.DatetimeIndex(bph_df['date']).year == int(yr)][['date','bph']]
                            #print(chart_data)
                            lc = alt.Chart(chart_data).mark_bar().encode(
                                x='date', y='bph', tooltip=['date','bph'])
                            st.altair_chart(lc, use_container_width=True)
                    st.markdown("Selected **" + str(selected_df.shape[0]) + "** rows of data")
                    #print(selected_df.shape[0])
                    
                    if not selected_df.empty:
                        st.pydeck_chart(pdk.Deck(
                                map_style='mapbox://styles/mapbox/dark-v9',
                                initial_view_state=pdk.ViewState(
                                latitude=selected_df['latitude'].mean(),
                                longitude=selected_df['longitude'].mean(),
                                zoom=5,
                                pitch=50,
                            ),
                            layers=[
                                pdk.Layer(
                                    "HeatmapLayer",
                                    data=selected_df,
                                    opacity=0.9,
                                    get_position=["longitude", "latitude"],
                                    threshold=0.75,
                                    aggregation=pdk.types.String("MEAN"),
                                    get_weight="bph*100",
                                    pickable=True,
                                )
                            ],
                        ))
                    
                    if selected_df.shape[0] > 0:
                        selected_model_to_add_dataset = st.selectbox("Select ANN Model to add this dataset for target", all_model['title'])
                        button_add_bph_to_model = st.button("Add to Model")
                        #st.write(st.session_state.model_id)
                        #if model != '-':
                        #    button_add_bph_to_model = st.button("Add to Model - " + model.loc[0]['title'])
                        #selected_df = selected_df.sort_values(by='date', ascending=False)[['date', 'bph', 'latitude', 'longitude']]
                        #print(selected_df)
                        #AgGrid(selected_df, fit_columns_on_grid_load=True)
                    
                else:
                    st.warning("To view data, select at least one year from the left panel")

        elif page == 'Cyrtorhinus Lividipennis Reuter':
            # LAYING OUT THE TOP SECTION OF THE APP
            st.subheader("Cyrtorhinus Lividipennis Reuter Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("View time-series Cyrtorhinus Lividipennis Reuter GIS data from database.")
            if selected_clr_data_distinct_year_list:
                sql = " OR ".join({"date_part('year', date) = {}".format(x) for x in selected_clr_data_distinct_year_list})
                clr_df = read_data_from_database("SELECT *, geometry AS geom FROM data_clr WHERE " + sql + " ORDER BY date DESC")
                AgGrid(clr_df)
                selected_clr_data_distinct_year_list.sort()
                for yr in selected_clr_data_distinct_year_list:
                    chart_data = clr_df[pd.DatetimeIndex(clr_df['date']).year == int(yr)][['date','clr']]
                    #print(chart_data)
                    lc = alt.Chart(chart_data).mark_bar().encode(
                        x='date', y='clr', tooltip=['date','clr'])
                    st.altair_chart(lc, use_container_width=True)
            else:
                st.warning("To view data, select at least one year from the left panel")

        
        elif page == 'Weather':
            # LAYING OUT THE TOP SECTION OF THE APP
            st.subheader("Weather Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("View time-series weather GIS data from database.")
            if selected_weather_data_distinct_year_list is not None:                
                sql = " OR ".join({"date_part('year', date) = {}".format(x) for x in selected_weather_data_distinct_year_list})
                weather_df = read_data_from_database("SELECT *, geometry AS geom FROM data_weather WHERE " + sql + " ORDER BY date DESC")
                AgGrid(weather_df)
                for yr in selected_weather_data_distinct_year_list:
                    chart_data = weather_df[pd.DatetimeIndex(weather_df['date']).year == int(yr)][['date','mint','maxt','temp','dew','humidity','wspd','wdir','windchill','precip','precipcover','cloudcover']] 
                    selection = alt.selection_multi(fields=['weather_data'], bind='legend')                  
                    lc = alt.Chart(chart_data.melt('date', var_name='weather_data', value_name='value')).mark_area().encode(x='date', y='value', color='weather_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                    st.altair_chart(lc, use_container_width=True)

        elif page == 'Rice Cultivated Area':
            # LAYING OUT THE TOP SECTION OF THE APP
            st.subheader("Rice Cultivated Area Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("View time-series rice cultivated area GIS data from database.")
            if selected_rice_cultivated_area_data_distinct_year_list is not None:                
                sql = " OR ".join({"date_part('year', date) = {}".format(x) for x in selected_rice_cultivated_area_data_distinct_year_list})
                rice_cultivated_area_df = read_data_from_database("SELECT *, geometry AS geom FROM data_rice_cultivated_area WHERE " + sql + " ORDER BY date DESC")
                AgGrid(rice_cultivated_area_df)
                selected_columns = ['date', 'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12']
                for yr in selected_rice_cultivated_area_data_distinct_year_list:
                    chart_data = rice_cultivated_area_df[pd.DatetimeIndex(rice_cultivated_area_df['date']).year == int(yr)][selected_columns]  
                    selection = alt.selection_multi(fields=['rice_cultivated_area_data'], bind='legend')                  
                    lc = alt.Chart(chart_data.melt('date', var_name='rice_cultivated_area_data', value_name='value')).mark_area().encode(x='date', y='value', color='rice_cultivated_area_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                    st.altair_chart(lc, use_container_width=True)

        
        elif page == 'Rice-Variety based Cultivated Area':
            # LAYING OUT THE TOP SECTION OF THE APP
            st.subheader("Rice-Variety based Cultivated Area Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("View time-series rice-variety based cultivated area GIS data from database.")
            if selected_rice_variety_cultivated_area_data_distinct_year_list is not None:                
                sql = " OR ".join({"date_part('year', date) = {}".format(x) for x in selected_rice_variety_cultivated_area_data_distinct_year_list})
                rice_variety_cultivated_area_df = read_data_from_database("SELECT *, geometry AS geom FROM data_rice_variety_cultivated_area WHERE " + sql + " ORDER BY date DESC")
                AgGrid(rice_variety_cultivated_area_df)
                rr = rice_variety_cultivated_area_df.drop(['address', 'latitude', 'longitude'], axis=1)
                for yr in selected_rice_variety_cultivated_area_data_distinct_year_list:
                    chart_data = rr[pd.DatetimeIndex(rr['date']).year == int(yr)][[column for column in rr]]  
                    print(chart_data.head())
                    selection = alt.selection_multi(fields=['rice_variety_cultivated_area_data'], bind='legend')                  
                    lc = alt.Chart(chart_data.melt('date', var_name='rice_variety_cultivated_area_data', value_name='value')).mark_area().encode(x='date', y='value', color='rice_variety_cultivated_area_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                    st.altair_chart(lc, use_container_width=True)
        """