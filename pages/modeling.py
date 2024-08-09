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
    page_title="Modeling",
    page_icon=":material/smart_toy:",
    layout="centered")

# set the current page context
st.session_state.current_page = "Modeling"

###################################### Supervised ######################################

def supervised_page():
    st.markdown("### Supervised")

###################################### UnSupervised ######################################

def unsupervised_page():
    st.markdown("### Unsupervised")

###################################### Modeling ######################################

def modeling_page():
    st.title("Modeling")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        tab1, tab2 = st.tabs(['Supervised', 'Unsupervised'])

        with tab1:
            supervised_page()
        
        with tab2:
            unsupervised_page()
        
    else:
        st.warning("**Please upload your file on the sidebar of Home page**")

###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# visualization_page
modeling_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()