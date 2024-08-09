"""

This is the main page, that you have to run with "streamlit run" to launch the app locally.
The sidebar of this page is used to upload a CSV file and store the DataFrame in session_state.
The name of the main page is Home, and it provides the overview of the Auto EDA website.
It also provides the preview of the uploaded dataset.

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
    page_title="Home",
    page_icon=":material/home:",
    layout="centered"
)

# set the current page context
st.session_state.current_page = "Home"

###################################### Home ######################################

def home_page():
    # Title
    st.title("Welcome to Auto EDA!")
    st.markdown(
        """
        This is an Auto EDA website developed by Yerim Choi and Yujin Min.  
        You can find the source code [here](https://github.com/yrc00/auto-eda/tree/main)
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

    # Data preview
    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        with st.container(border=True):
            tab1, tab2 = st.tabs(["First rows", "Last rows"])
        
            # show df.head(10)
            with tab1:
                st.dataframe(df.head(10))
            
            # show df.tail(10)
            with tab2:
                st.dataframe(df.tail(10))

    else:
        st.warning("**Please upload your file on the sidebar of Home page**")

###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# home page
home_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()
