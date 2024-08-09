"""

This is the menu page, that you can use to navigate through the different pages of the Auto EDA website.
All the pages are linked in the sidebar, and you can upload a CSV file on the Home page.
The page files are in the pages folder, and the main streamlit app is in the streamlit_app.py file.

"""

###################################### import ######################################

# library
import streamlit as st
import streamlit_option_menu as option_menu
import pandas as pd

###################################### file upload ######################################

# divide data types of columns
def get_dtype(column):
    categorical = ['object', 'category']
    numerical = ['int64', 'float64']
    datetime = ['datetime64']
    bool_ = ['bool']
    string = ['str']

    dtype_str = str(column.dtype)
    
    if dtype_str in categorical:
        return 'Categorical'
    elif dtype_str in numerical:
        return 'Numerical'
    elif dtype_str in datetime:
        return 'Datetime'
    elif dtype_str in bool_:
        return 'Boolean'
    elif dtype_str in string:
        return 'String'
    else:
        return 'Other'

# file uploader
def file_upload():
    # file uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # load uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df

            # get data types and create dtype_df
            dtype_dict = df.apply(get_dtype).to_dict()
            dtype_df = pd.DataFrame(list(dtype_dict.items()), columns=['Column', 'DType'])
            dtype_df.set_index('Column', inplace=True)
            st.session_state.dtype_df = dtype_df
        
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or not a valid CSV.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

###################################### menu ######################################

# print menu
def menu():
    with st.sidebar:
        st.title("Auto EDA")

        # pages
        st.page_link("streamlit_app.py", label="Home", icon=":material/home:")
        st.page_link("pages/data.py", label="Data", icon=":material/database:")
        st.page_link("pages/visualization.py", label="Visualization", icon=":material/monitoring:")
        st.page_link("pages/correlation.py", label="Correlation", icon=":material/link:")
        st.page_link("pages/modeling.py", label="Modeling", icon=":material/smart_toy:")
        st.page_link("pages/chatbot.py", label="Chatbot", icon=":material/chat:")

###################################### Data sidebar ######################################

# data sidedbar
def data_sidebar():
    st.sidebar.text("Data")

###################################### DType sidebar ######################################


###################################### Chatbot sidebar ######################################
