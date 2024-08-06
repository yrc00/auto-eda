""""

페이지에 대한 설명을 작성하세요.

"""

###################################### import ######################################

import streamlit as st
import pandas as pd
from menu import show_sidebar

###################################### set  ######################################

# set page config
st.set_page_config(
    page_title="Correlation",
    page_icon="🔗",
    layout="centered")

# set the current page context
st.session_state.current_page = "Visualization"

###################################### Sidebar ######################################

# Call the sidebar function
show_sidebar()

###################################### Data ######################################

# Access the data from session_state
if 'df' in st.session_state:
    df = st.session_state.df
    dtype_str = df.dtypes.apply(lambda x: x.name).to_dict()
    st.session_state.dtype_str = dtype_str 
else:
    df = None
    dtype_str = None

###################################### Correlation ######################################

# Title
st.title("Correlation")

# Data preview
st.write("Data preview:")
if df is not None:
    st.dataframe(df)
else:
    st.markdown("**Please upload your file on the sidebar of Home page**")

# Data types of columns
if dtype_str is not None:
    st.write("Data types of columns:")
    st.dataframe(dtype_str)
else:
    st.write("No data available.")

###################################### Pairplot ######################################

###################################### Scatter plot ######################################

###################################### Correlation heatmap ######################################