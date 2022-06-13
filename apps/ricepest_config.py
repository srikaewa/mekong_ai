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

#from parted import Geometry
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

from apps.data_operation import save_to_postgis, load_from_postgis, delete_label_rice_variety, update_label_rice_variety


class RicePestConfig(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):

        #st.experimental_set_query_params(selected=self.title)
        print(self.title)

        #thailand_boundary = gpd.read_file('Thailand Boundary.shp')

        st.subheader("Rice Pest Data Configuration")
        st.write("This page is for configuration of all related rice pest data such as mapping table for rice varieties and labels used in AI model.")

        # coordinate = [101.04255737682536, 15.802395636003219]
        # point_coord = Point(coordinate)
        # ddf = {'label': 'r1', 'rice_variety': 'พันธุ์พื้นเมือง', 'geometry': [point_coord]}
        # gdf = gpd.GeoDataFrame(ddf, geometry='geometry', crs ="EPSG:4326")
        # print(gdf)
        # save_to_postgis(gdf, 'label_rice_variety')

        label_variety_gdf = load_from_postgis("SELECT *, geometry AS geom FROM label_rice_variety")      
        label_variety_df = pd.DataFrame(label_variety_gdf.drop(['geometry','geom'], axis=1))  
        print(label_variety_df)
        #print(label_variety_df.loc[i]['label'], label_variety_df.loc[i]['rice_variety'])
        row1_1, row1_2, row1_3, row1_4 = st.columns((2,1,1,4))
        with row1_1:
            with st.form("Label - Rice Variety", clear_on_submit = True):
                st.write("Label - Rice Variety")
                input_label = st.text_input("Label", value="r" + str(label_variety_df.shape[0]+1), disabled=True) 
                input_rice_variety = st.text_input("Rice Variety")
                submitted = st.form_submit_button("Add")
                if submitted:
                    #print("SIZE :: " + str(label_variety_df.shape[0]))
                    if input_label and input_rice_variety:
                        coordinate = [101.04255737682536, 15.802395636003219]
                        point_coord = Point(coordinate)
                        ddf = {'label': input_label, 'rice_variety': input_rice_variety, 'geometry': [point_coord]}
                        gdf = gpd.GeoDataFrame(ddf, geometry='geometry', crs ="EPSG:4326")
                        #print(gdf)
                        save_to_postgis(gdf, 'label_rice_variety')
                        st.experimental_rerun()
                    else:
                        st.error("Label and Rice Variety must be entered!")

            with st.expander("View All Label - Rice Variety", expanded=True):
                gb = GridOptionsBuilder.from_dataframe(label_variety_df)
                # enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
                gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
                gb.configure_selection(selection_mode="single", use_checkbox=True)
                gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
                gridOptions = gb.build()
                response = AgGrid(
                    label_variety_df,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True,
                    theme="blue",
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=False,
                )

                df = pd.DataFrame(response["selected_rows"])
                if not df.empty:
                    input_update_rice_variety = st.text_input("Update label : " + df.loc[0]['label'], value=df.loc[0]['rice_variety'])
                    button_update = st.button("Update")
                    if button_update:
                        if input_update_rice_variety:
                            update_label_rice_variety(df.loc[0]['label'], input_update_rice_variety)
                            st.experimental_rerun()
                        else:
                            st.error("Can't update empty label!")
                    st.markdown("---")
                    st.warning("Click Delete button will immediately delete label " + df.loc[0]['label'] + ", proceed with caution?")
                    button_delete = st.button("Delete")
                    if button_delete:
                        delete_label_rice_variety(df.loc[0]['label'], df.loc[0]['rice_variety'])
                        st.experimental_rerun()

        st.markdown("---")
        st.markdown("<p align=right> Powered by <a href='https://www.mekonginstitute.org/'><b>MeKong Institute</b></a></p>", unsafe_allow_html=True)