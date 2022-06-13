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

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import altair as alt
import pydeck as pdk
from pydeck.types import String
from hydralit import HydraHeadApp

from datetime import datetime, date

from apps.data_operation import load_from_postgis


class RiceBlastDashboard(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
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
        st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] >    div > div > div[role="slider"]{
    background-color: rgb(14, 125, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)
        st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 00%); } </style>''', unsafe_allow_html = True)

        #st.experimental_set_query_params(selected=self.title)
        print(self.title)

        

        # LOADING DATA
        DATE_TIME = "date/time"
        DATA_URL = (
            "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz"
        )

        @st.cache(persist=True)
        def load_data(nrows):
            data = pd.read_csv(DATA_URL, nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis="columns", inplace=True)
            data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
            return data

        @st.cache(persist=True)
        def load_bph_year(sql):
            data = load_from_postgis(sql)     
            data['date'] = pd.to_datetime(data['date'])
            #for i in range(int(data['BPH'])):
            #    data = data.append(data.loc[0], ignore_index=True)
                #print(i)
            return data

        def get_latest_bph_date():
            data = load_from_postgis("SELECT date, geometry AS geom FROM data_bph ORDER BY date DESC")
            return data.loc[0]
        
        def get_oldest_bph_date():
            data = load_from_postgis("SELECT date, geometry AS geom FROM data_bph ORDER BY date ASC")
            return data.loc[0]        


        data_distinct_year = load_from_postgis("SELECT DISTINCT EXTRACT(year from date), geometry AS geom FROM data_bph ORDER BY date_part DESC")
        year_list = []
        for i in range(len(data_distinct_year.loc[:])):
            year_list.append(str(data_distinct_year.loc[i].at["date_part"])[:4])

        latest_bph_date = get_latest_bph_date()
        oldest_bph_date = get_oldest_bph_date()
        
        #data = load_data(100000)

        # CREATING FUNCTION FOR MAPS
        def map(data, lat, lon, zoom):
            #print(data)
            st.write(pdk.Deck(
                #map_style="mapbox://styles/mapbox/satellite",
                map_style="mapbox://styles/mapbox/satellite-v9",
                #map_style=pdk.map_styles.SATELLITE,
                initial_view_state={
                    "latitude": lat,
                    "longitude": lon,
                    "zoom": zoom,
                    "pitch": 50,
                },
                layers=[
                    pdk.Layer(
                        "HeatmapLayer",
                        data=data,
                        opacity=0.9,
                        get_position=["longitude", "latitude"],
                        aggregation=String('MEAN'),
                        threshold=0.4,
                        intensity=5,
                        radiusPixels=50,
                        get_weight="BPH*200",
                        pickable=True),        
                    pdk.Layer(
                        "PolygonLayer",
                        load_thailand_boundary(),
                        stroked=False,
                        # processes the data as a flat longitude-latitude pair
                        get_polygon="-",
                        get_fill_color=[0, 0, 0, 200])         
                ]
            ))

        st.subheader('Rice Pest Outbreak Data Dashboard')
        
        # LAYING OUT THE TOP SECTION OF THE APP
        row1_1, row1_2 = st.columns((2,3))

        with row1_1:
            selected_date = st.slider(
            "Pick a date to view data",
            min_value=date(oldest_bph_date['date'].year, oldest_bph_date['date'].month, oldest_bph_date['date'].day),
            max_value=date(latest_bph_date['date'].year, latest_bph_date['date'].month, latest_bph_date['date'].day),
            value=date(latest_bph_date['date'].year, latest_bph_date['date'].month, latest_bph_date['date'].day),
            format="YYYY-MM-DD")
            if selected_date:
                data = load_bph_year("SELECT *, geometry AS geom FROM data_bph WHERE date = '" + str(selected_date) + "'")
                #print(data)
            #data = data[data['date'] == str(start_time)]
            #st.write("Selected Date:", selected_date)
            #print(data)

        with row1_2:
            st.write(
            """
            ##
            Monitoring all related data to rice pest outbreak. This includes raw time-series data of brown plant hopper, cyrtorhinus lividipennis reuter, weather information, rice cultivated area and rice productions. These set of time-series data are mainly used in training with our deep learning engine for forecasting brown plant hopper outbreak.
            """)


        # LAYING OUT THE MIDDLE SECTION OF THE APP WITH THE MAPS
        row2_1, row2_2, row2_3, row2_4, row2_5 = st.columns((1,1,1,1,1))

        # SETTING THE ZOOM LOCATIONS FOR THE AIRPORTS
        myanmar = [22.323073706723235, 96.51642099805225]
        cambodia= [12.715764717835954, 104.9560129825612]
        vietnam = [13.731776393071405, 108.33980200123735]
        laos = [19.24930665440802, 103.08833724983039]
        zoom_level = 5
        thailand = [14.423305048029997, 100.44137829299272]

        with row2_1:
            st.write("**Myanmar**")
            map(data, myanmar[0], myanmar[1], zoom_level)

        with row2_2:
            st.write("**Thailand**")
            map(data, thailand[0], thailand[1], zoom_level)

        with row2_3:
            st.write("**Cambodia**")
            map(data, cambodia[0],cambodia[1], zoom_level)

        with row2_4:
            st.write("**Laos**")
            map(data, laos[0],laos[1], zoom_level)

        with row2_5:
            st.write("**Vietnam**")
            map(data, vietnam[0],vietnam[1], zoom_level)

        
        