"""

This is the sidebar page

"""

###################################### import ######################################

# library
import streamlit as st

###################################### menu ######################################

# print menu
def menu():
    pages = {
        "Auto EDA": [
            st.Page("app.py", title="Home", icon="🏠"),
        ],
        "Report": [
            st.Page("pages/data.py", title="Data", icon="💾"),
            st.Page("pages/overview.py", title="Overview", icon="🔍"),
            st.Page("pages/visualization.py", title="Visualization", icon="📊"),
            st.Page("pages/correlation.py", title="Correlation", icon="🔗"),
            st.Page("pages/modeling.py", title="Modeling", icon="🤖"),
        ],
    }
    pg = st.navigation(pages)
    return pg.run()

###################################### language selection ######################################

# set language
def language_selector():
    # language list
    languages = ["en", "ko"]

    # set language
    selected_language = st.sidebar.selectbox("Select Language", options=languages)
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

###################################### llm response ######################################

# llm response
def llm_response():
    view = st.sidebar.checkbox("View LLM Analysis ", value=True)
    if view != st.session_state.llm_response:
        st.session_state.llm_response = view
        st.rerun()