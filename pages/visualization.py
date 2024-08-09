""""

페이지에 대한 설명을 작성하세요.

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# functions
from menu import file_upload, menu

###################################### set  ######################################

# set page config
st.set_page_config(
    page_title="Visualization",
    page_icon=":material/monitoring:",
    layout="centered")

# set the current page context
st.session_state.current_page = "Visualization"

###################################### Categorical ######################################

def categorical_page():
    st.markdown("### Categorical")

###################################### Numerical ######################################

def numerical_page():
    st.markdown("### Numerical")

###################################### Time Series ######################################

def timeseries_page():
    st.markdown("### Time Series")

###################################### String ######################################

def string_page():
    st.markdown("### String")

###################################### Visualization ######################################

def visual_page():
    st.title('Visualization')
    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        tab1, tab2, tab3, tab4 = st.tabs(['Categorical', 'Numerical', 'Time Series', 'String'])

        with tab1:
            categorical_page()
        
        with tab2:
            numerical_page()
        
        with tab3:
            timeseries_page()
        
        with tab4:
            string_page()

    else:
        st.warning("**Please upload your file on the sidebar of Home page**")


###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# visualization_page
visual_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()