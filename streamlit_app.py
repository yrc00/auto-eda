"""
This is the main page, that you have to run with "streamlit run" to launch the app locally.
The sidebar of this page is used to upload a CSV file and store the DataFrame in session_state.
The name of the main page is Home, and it provides the overview of the Auto EDA website.
It also provides the preview of the uploaded dataset.
"""

###################################### import ######################################

import streamlit as st
import pandas as pd
from menu import show_sidebar

###################################### set  ######################################

# set page config
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="centered"
)

# set the current page context
st.session_state.current_page = "Home"

###################################### Sidebar ######################################

# Call the sidebar function to handle file upload
uploaded_file = show_sidebar()

###################################### Data ######################################

# Handle the uploaded file
if uploaded_file is not None:
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df  # Store DataFrame in session_state
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or not a valid CSV.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if 'df' in st.session_state:
        df = st.session_state.df
    else:
        df = None

###################################### Home ######################################

# Title
st.title("Welcome to Auto EDA!")
st.markdown(
    """
    This is an Auto EDA website developed by Yerim Choi and Yujin Min.  
    You can find the source code [here]()
    """
)

st.markdown(
    """
    ### **Instructions:**
    - Upload a CSV file using the sidebar on the left.
    - Navigate to different pages using the sidebar.
    - View the data, visualization, correlation, and modeling pages.
    - Use the chatbot to ask questions about the data.

    **Note:** This is a demo website and may not work as expected.
    """
)

st.markdown(
    """
    ### **Data preview:**
    This is the preview of the dataset you uploaded.
    """
)

if df is not None:
    st.dataframe(df)
else:
    st.markdown("**Please upload your file on the sidebar**")
