""""

페이지에 대한 설명을 작성하세요.

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd

# functions
from menu import file_upload, menu

###################################### set  ######################################

# set page config
st.set_page_config(
    page_title="Correlation",
    page_icon=":material/link:",
    layout="centered")

# set the current page context
st.session_state.current_page = "Correlation"

###################################### Pairplot ######################################

def pairplot_page():
    st.markdown("### Pairplot")

###################################### Scatter plot ######################################

def scatter_page():
    st.markdown("### Scatter plot")

###################################### Correlation heatmap ######################################

def heatmap_page():
    st.markdown("### Correlation Heatmap")

###################################### Correlation ######################################

def correlation_page():
    # Title
    st.title("Corrrelation")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        tab1, tab2, tab3 = st.tabs(['Pairplot', 'Scatter plot', 'Correlation Heatmap'])

        with tab1:
            pairplot_page()
        
        with tab2:
            scatter_page()
        
        with tab3:
            heatmap_page()

    else:
        st.warning("**Please upload your file on the sidebar of Home page**")

###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# correlation
correlation_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()
