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

from pkgutil import get_data
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

from apps.data_operation import save_to_postgis, load_from_postgis


class DroughtImportData(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        #st.experimental_set_query_params(selected=self.title)
        print(self.title)

        st.markdown(
            """
        <style>
        span[data-baseweb="tag"] {
        background-color: #229954 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        
        #thailand_boundary = gpd.read_file('Thailand Boundary.shp')

        @st.cache(persist=True, allow_output_mutation=True)
        def read_data_from_uploaded_file(ulf):
            #print(ulf)
            print("Cached uploaded file!!!")
            df = pd.read_csv(ulf, parse_dates=["date"], encoding = "ISO-8859-1", low_memory=False)
            lowercase = lambda x: str(x).lower()
            df.rename(lowercase, axis="columns", inplace=True)
            #gdf = gpd.read_file(uploaded_file)
            #gdf = GeoDataFrame(
            #    df,
            #    crs={'init': 'epsg:4326'},
            #    geometry=[Point(xy) for xy in zip(df.latitude, df.longitude)])
            gdf = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
            gdf.set_crs(epsg=4326, inplace=True)
            return df, gdf

        with st.sidebar:
            st.header('Drought - Natural Disaster')
            st.subheader('Import Data')

            uploaded_drought_file = st.file_uploader("Choose a CSV file to import all drought-related data", type=['csv'])

            selected_data_layer = []

            if uploaded_drought_file is not None:   
                #if uploaded_file is not None:
                # Can be used wherever a "file-like" object isaccepted:
                with st.spinner("Loading data from file..."):
                    drought_df, drought_gdf = read_data_from_uploaded_file(uploaded_drought_file)
                    #print(drought_gdf.head())
                    
                    selected_data_layer = st.multiselect("Select data layer to view",['Drought Area', 'Weather'])
                    drought_years_list = pd.DatetimeIndex(drought_df['date']).year.unique()
                    drought_year_options = st.multiselect("Select year of data to import", drought_years_list.values, drought_years_list.values[0])   
                if drought_year_options:
                    if set(['date', 'address', 'latitude', 'longitude','temp','humidity','avg_ssm','drought','area_drought']).issubset(drought_df.columns):      
                        fn_size = len(uploaded_drought_file.name)         
                        drought_dataset_title = st.text_input("Dataset title", value=uploaded_drought_file.name[:fn_size-4])                  
                        process_drought_data = st.button("Process & Upload Data")
                        #print(process_drought_data)
                        if process_drought_data:
                            if drought_dataset_title:
                                drought_gdf['dataset_title'] = drought_dataset_title
                                with st.spinner("Importing data to database..."):
                                    save_to_postgis(drought_gdf, "data_drought_model_input")
                                st.success("Upload success!")
                            else:
                                st.error("Dataset title cannot be empty!")
                    else:
                        st.markdown('<span style="color:red">CSV file must contain *all required* columns of data!</span>', unsafe_allow_html=True)                                     
        # LAYING OUT THE TOP SECTION OF THE APP
        st.subheader("Import Drought Data")
        row1_1, row1_2 = st.columns((2,5))
        st.write("Import of drought raw data. Data can be easily uploaded with pre-defined format CSV file. Latitude & Longitude columns will be converted to geometry type of data and uploaded to database backend. Currently, data related to drought modeling mainly includes weather data (temperature, humidity, average soil surface moisture) and the information on drought (area and official annoncement of drought). Ascending time of data rows is also crucial to the time-series neural network.  To view specific data layer and year range, use multi select box from sidebar.")
        print(selected_data_layer)

        if selected_data_layer:
            selected_columns = ['date','address','latitude','longitude']
            for x in selected_data_layer:
                if x == 'Drought Area':
                    selected_columns.append('area_drought')
                if x == 'Weather':
                    selected_columns = selected_columns + ['temp', 'humidity', 'avg_ssm']
            #print(selected_columns)
            if uploaded_drought_file is not None and len(drought_year_options) > 0:
                selected_layer_df = drought_df[selected_columns]
                buff_set = pd.DataFrame()
                for yr in drought_year_options:
                    sdf = selected_layer_df[selected_layer_df['date'].dt.strftime('%Y') == str(yr)]
                    buff_set = pd.concat([buff_set, sdf])
                #selected_layer_df = drought_df[selected_columns]
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
                drought_all_df = grid_response['data']
                selected = grid_response['selected_rows'] 
                selected_df = pd.DataFrame(selected)
                for yr in drought_year_options:
                    chart_data = selected_layer_df[pd.DatetimeIndex(selected_layer_df['date']).year == yr][['date','area_drought']]
                    print(chart_data)
                    lc = alt.Chart(chart_data).mark_bar().encode(
                            x='date', y='area_drought', tooltip=['date','area_drought'])
                    st.altair_chart(lc, use_container_width=True)
        """
        elif selected_data_layer == 'Mirid Bug':
            st.subheader("Import Mirid Bug Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("Import of Cyrtorhinus Lividipennis Reuter raw data. Data can be easily uploaded with pre-defined format CSV file. Latitude & Longitude columns will be converted to geometry type of data for and uploaded to database backend. To view data in specific range, select start date and end date.")
            if uploaded_clr_file is not None and len(clr_year_options) > 0:                
                AgGrid(clr_df, fit_columns_on_grid_load=True)
                for yr in clr_year_options:
                    chart_data = clr_df[pd.DatetimeIndex(clr_df['date']).year == yr][['date','mirid_bug']]
                    #print(chart_data)
                    lc = alt.Chart(chart_data).mark_bar().encode(
                        x='date', y='mirid_bug', tooltip=['date','mirid_bug'])
                    st.altair_chart(lc, use_container_width=True)

        
        elif selected_data_layer == 'Weather':
            st.subheader("Import Weather Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("Import of weather raw data. Data can be easily uploaded with pre-defined format CSV file. Latitude & Longitude columns will be converted to geometry type of data for and uploaded to database backend. The corresponding chart has an interactive legend which one legend or multiple legends can be selected to get highlight on the chart.")
            if uploaded_weather_file is not None:                
                AgGrid(weather_df, fit_columns_on_grid_load=True)
                for yr in year_options:
                    chart_data = weather_df[pd.DatetimeIndex(weather_df['date']).year == yr][['date','mint','maxt','temp','dew','humidity','wspd','wdir','precip']]  
                    selection = alt.selection_multi(fields=['weather_data'], bind='legend')                  
                    lc = alt.Chart(chart_data.melt('date', var_name='weather_data', value_name='value')).mark_area().encode(x='date', y='value', color='weather_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                    st.altair_chart(lc, use_container_width=True)

        elif selected_data_layer == 'Rice Cultivated Area':
            st.subheader("Import Rice Cultivated Area Data")
            row1_1, row1_2 = st.columns((2,5))
            st.write("Import of rice cultivated area raw data. Data can be easily uploaded with pre-defined format CSV file. Latitude & Longitude columns will be converted to geometry type of data for and uploaded to database backend. The mapping between label (r1,r2,r3,...) and original name of rice variety is in Rice Pese Configuration Page. This is crucial for AI engine to recognize label of rice variety correctly. The corresponding chart has an interactive legend which one legend or multiple legends can be selected to get highlight on the chart.")
            if uploaded_rice_cultivated_area_file is not None:                
                AgGrid(rice_cultivated_area_df, fit_columns_on_grid_load=True)
                rr = rice_cultivated_area_df.drop(['latitude', 'longitude','geometry'], axis=1)
                for yr in year_options:
                    chart_data = rr[pd.DatetimeIndex(rr['date']).year == int(yr)][[column for column in rr]]  
                    #print(chart_data.head())
                    selection = alt.selection_multi(fields=['rice_cultivated_area_data'], bind='legend')                  
                    lc = alt.Chart(chart_data.melt('date', var_name='rice_cultivated_area_data', value_name='value')).mark_area().encode(x='date', y='value', color='rice_cultivated_area_data',opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).add_selection(selection)
                    st.altair_chart(lc, use_container_width=True)
        """