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
    page_title="Chatbot",
    page_icon=":material/chat:",
    layout="centered")

# set the current page context
st.session_state.current_page = "chatbot"

###################################### Chatbot ######################################

def chatbot_page():
    st.title("Chatbot")

###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# chatbot page
chatbot_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()