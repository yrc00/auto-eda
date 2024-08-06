"""

This is the menu page, that you can use to navigate through the different pages of the Auto EDA website.
All the pages are linked in the sidebar, and you can upload a CSV file on the Home page.
The page files are in the pages folder, and the main streamlit app is in the streamlit_app.py file.

"""

###################################### import ######################################

import streamlit as st
import pandas as pd

###################################### Sidebar ######################################

def show_sidebar():
    st.sidebar.title("Navigation")

    # Page navigation
    st.sidebar.page_link("streamlit_app.py", label="Home")
    st.sidebar.page_link("pages/data.py", label="Data")
    st.sidebar.page_link("pages/visualization.py", label="Visualization")
    st.sidebar.page_link("pages/correlation.py", label="Correlation")
    st.sidebar.page_link("pages/modeling.py", label="Modeling")
    st.sidebar.page_link("pages/chatbot.py", label="Chatbot")

    # File uploader (only on Home page)
    if st.session_state.get('current_page') == "Home":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df  # Store DataFrame in session_state

                # Create a DataFrame with column data types
                dtype_dict = df.dtypes.apply(lambda x: x.name).to_dict()
                dtype_df = pd.DataFrame(list(dtype_dict.items()), columns=['Column', 'Dtype'])
                st.session_state.dtype_df = dtype_df  # Store dtype DataFrame in session_state

            except pd.errors.EmptyDataError:
                st.error("The uploaded file is empty or not a valid CSV.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    return st.session_state.get('uploaded_file', None)

###################################### Data sidebar ######################################

def data_sidebar():
    st.sidebar.text("Data")


###################################### DType sidebar ######################################


###################################### Chatbot sidebar ######################################
