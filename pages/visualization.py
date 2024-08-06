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
    page_title="Visualization",
    page_icon=":bar_chart:",
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
else:
    df = None

###################################### Visualization ######################################

# Title
st.title("Visualization")

st.write("Data preview:")
if df is not None:
    st.dataframe(df)
else:
    st.markdown("**Please upload your file on the sidebar of Home page**")