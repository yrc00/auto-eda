"""
This is the Overview page

"""

###################################### import ######################################

# library
from dotenv import load_dotenv
import gettext
import os

import streamlit as st
import pandas as pd
import numpy as np

from st_mui_table import st_mui_table
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

import re
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from deep_translator import GoogleTranslator

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Overview"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

# load the .env file
load_dotenv()

# get the API token from the environment
huggingface_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if huggingface_token:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_token
else:
    st.error("HUGGINGFACEHUB_API_TOKEN is missing. Please check your .env file.")

##################################### google translate ######################################

# google translate
def google_translate(text, language):
    # Define chunk size to be safe (less than 5000, e.g., 4000 characters)
    max_chunk_length = 4000
    translator = GoogleTranslator(source='auto', target=language)
    
    # Split the text into chunks of 4000 characters
    split_text = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Translate each chunk and concatenate the results
    translated_text = ""
    for text_chunk in split_text:
        translated_text += translator.translate(text_chunk) + " "
    
    return translated_text.strip()

##################################### Overview ######################################

# find zero values
def is_zero(series):
    return series == 0

# show table
def show_table(df, key, n=None):
    st_mui_table(
        df,
        enablePagination=False,
        customCss="",  
        paginationSizes=[n] if n else [],
        size="small",
        padding="normal",
        showHeaders=False,
        key=key,
        stickyHeader=False,
        paperStyle={ 
            "width": '100%',  
            "overflow": 'auto',
            "paddingBottom": '1px', 
            "border": '1px solid rgba(224, 224, 224, 1)'
        },
        detailColumns=[],
        detailColNum=1,
        detailsHeader="Details",
        showIndex=False
    )

# overview table
def gen_overview_table(df, dtype_df):
    # rows, columns
    rows, columns = df.shape

    # missing cells
    missing = df.isna().sum().sum()
    missing_per = missing / (rows * columns) * 100

    # duplicate rows
    duplicate = df.duplicated().sum()
    duplicate_per = df.duplicated().mean() * 100

    # memory usage
    memory = df.memory_usage().sum() / 1024
    memory_per = df.memory_usage().mean()

    st.session_state.overview_table = pd.DataFrame({
        'Metric': [
            'Number of Columns', 'Number of Rows', 'Missing Cells', 'Missing Cells (%)', 
            'Duplicate Rows', 'Duplicate Rows (%)', 'Total Size in Memory (KB)', 
            'Average Size in Memory (B)'
        ],
        'Value': [
            f'{columns}',  
            f'{rows}',  
            f'{missing}',  
            f'{missing_per:.2f}%',  
            f'{duplicate}',  
            f'{duplicate_per:.2f}%',  
            f'{memory:.2f} KB',  
            f'{memory_per:.2f} B',
        ]
    })

    # data types
    dtype_table = pd.DataFrame(dtype_df['Data Type'].value_counts())
    dtype_table = dtype_table.reset_index()
    dtype_table.columns = ['Data Type', 'Count']
    st.session_state.dtype_table = dtype_table

def show_overview_table(df, dtype_df):
    st.markdown("### Overview")

    gen_overview_table(df, dtype_df)

    # overview table
    overview_table = st.session_state.overview_table
    overview_add = pd.DataFrame({'Metric': [''], 'Value': ['']})
    overview_table= pd.concat([overview_table, overview_add], ignore_index=True)

    # data type table
    dtype_table = st.session_state.dtype_table
    dtype_add = pd.DataFrame({'Data Type': ['Additional'], 'Count': ['']})
    dtype_table = pd.concat([dtype_table, dtype_add], ignore_index=True)

    # show tables
    with st.container(border=True):
        col1, col2 = st.columns(2)

        # show overview table
        with col1:
            st.markdown("**Dataset Overview**")
            show_table(overview_table, key="overview_table", n=8)
        
        # show data type table
        with col2:
            st.markdown("**Data Types**")
            show_table(dtype_table, key="dtype_table", n=dtype_table.shape[0])

# print alerts
def show_alerts(df):
    with st.expander("**Alerts**"):
        # duplicate rows
        duplicate = df.duplicated().sum()
        duplicate_per = df.duplicated().mean() * 100

        col1, col2 = st.columns([3, 1])

        if duplicate > 0:
            with col1:
                st.write(f"Dataset has {duplicate} ({duplicate_per:.2f}%) duplicated rows")
            with col2:
                st.markdown(":gray-background[Duplicated]")

        # missing values, zeros
        for col in df.columns:
            # missing values
            col_missing = df[col].isna().sum()
            col_missing_per = df[col].isna().mean() * 100
            if col_missing > 0:
                with col1:
                    st.write(f"```{col}``` has {col_missing} ({col_missing_per:.2f}%) missing values")
                with col2:
                    st.markdown(":blue-background[Missing]")
                
            # zero values
            col_zeros = is_zero(df[col]).sum()
            col_zeros_per = is_zero(df[col]).mean() * 100
            if col_zeros > 0:
                with col1:
                    st.write(f"```{col}``` has {col_zeros} ({col_zeros_per:.2f}%) zero values")
                with col2:
                    st.markdown(":green-background[Zeros]")

def show_dtype_df(dtype_df):
    # show data types
    with st.expander("**Datatype of Columns**"):
        st.table(dtype_df)

# overview explanation
def overview_slm():
    # select model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template 
    template = """
    Here is an overview of the dataset:
    {overview_table}

    the dataset has the following data types:
    {dtype_table}

    Please provide a detailed summary of the overview and data types and provide the summary.

    Answer:
    """

    # prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["overview_table", "dtype_table"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 512}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({
        "overview_table": st.session_state.overview_table.to_markdown(),
        "dtype_table": st.session_state.dtype_table.to_markdown()
    })

    response_text = response.get("text", "").strip()

    response_start = response_text.find("Answer:")
    if response_start != -1:
        response_text = response_text[response_start + len("Answer:"):].strip()
    else:
        response_text = response_text

    if st.session_state.language != "en":
        response_text = google_translate(response_text, "ko")
    
    return response_text

def overview(df, dtype_df):
    show_overview_table(df, dtype_df)
    show_alerts(df)
    show_dtype_df(dtype_df)

    # duplicate rows
    if df.duplicated().sum() > 0:
        with st.expander("**Duplicate Rows**"):
            st.dataframe(df[df.duplicated()])
    
    # show explanation
    response = overview_slm()
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Missing Values ######################################

# missing values msno plot
@st.cache_data
def missing_values_plot(df):
    # Missing Values
    st.markdown("### Missing Values")

    # missing values plots
    with st.container(border=True):
        tab1, tab2, tab3, tab4 = st.tabs(['Matrix', 'Bar', 'Heatmap', 'Dendrogram'])

        # barplot
        with tab1:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.bar(df, ax=ax)
            st.pyplot(fig)
        
        # matrix
        with tab2:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.matrix(df, ax=ax)
            st.pyplot(fig)
        
        # heatmap
        with tab3:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.heatmap(df, ax=ax)
            st.pyplot(fig)

        # dendrogram
        with tab4:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.dendrogram(df, ax=ax)
            st.pyplot(fig)

# missing values explanation
def missing_values_slm():
    # select model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    Here is a summary of the Missing Values in the dataset: {missing_table}

    Definition of Missing Values:
    Missing values refer to the absence of data in one or more columns of a dataset. This can occur due to various reasons, such as data entry errors, incomplete data collection, or limitations in data integration from multiple sources.

    Status of Missing Values in the Dataset (Columns with missing values only):
    The following columns contain missing values, and those with zero missing values have been excluded from this summary.

    Impact of Missing Values on the Dataset:
    The columns with a high proportion of missing values are likely to affect the quality and accuracy of the analysis. Specifically, the following columns have a significant number of missing values, which could lead to potential issues:

    Column Name 1: [Reason why the missing values could be problematic, e.g., "This column is crucial for prediction and has over 50% missing data."]
    Column Name 2: [Reason why the missing values could be problematic, e.g., "Contains key categorical information, making data imputation challenging."]
    Answer:
    """

    # generate prompt template 
    prompt = PromptTemplate(
        template=template,
        input_variables=["missing_table"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 1024}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({
        "missing_table": st.session_state.missing_table.to_markdown()
    })

    response_text = response.get("text", "").strip()

    response_start = response_text.find("Answer:")
    if response_start != -1:
        response_text = response_text[response_start + len("Answer:"):].strip()
    else:
        response_text = response_text
    
    if st.session_state.language != "en":
        response_text = google_translate(response_text, "ko")
    
    return response_text

# missing values page
def missing_values(df):
    # missing values plot
    missing_values_plot(df)

    # missing values table
    with st.expander("**Rows with Missing Values**"):
        missing_values_df = df[df.isna().any(axis=1)]
        st.dataframe(missing_values_df)
    
    missing_table = pd.DataFrame(df.isna().sum())
    missing_table.columns = ['Missing Values']
    st.session_state.missing_table = missing_table

    # missing values explanation
    response = missing_values_slm()
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Outlier ######################################

# detect outliers by z-score
def detect_outliers_zscore(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    zscore = (df[column] - mean) / std
    outliers = df[np.abs(zscore) > threshold]
    return outliers, outliers.shape[0]

# show zscore
def show_zscore(df, numeric_col, detail):
    z_outliers = {}
    z_outlier_df = {}

    for col in numeric_col:
        z_outlier_df[col], z_outliers[col] = detect_outliers_zscore(df, col)
    z_score_df = pd.DataFrame(z_outliers, index=['Outliers']).T
    st.dataframe(z_score_df)

    if detail:
        for col in numeric_col:
            if not z_outlier_df[col].empty:
                with st.expander(f"**Outliers in {col} (Count: {z_outliers[col]})**"):
                    st.dataframe(z_outlier_df[col])
    
    return z_score_df

# detect outliers by IQR
def detect_outliers_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q2 = df[column].quantile(0.50)
    Q3 = df[column].quantile(0.75)
    min_val = df[column].min()
    max_val = df[column].max()

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    non_outlier = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    result = {
        "Outliers": f'{outliers.shape[0]}',
        "Non-Outliers": f'{non_outlier.shape[0]}',
        "min": f'{min_val:.2f}',
        "max": f'{max_val:.2f}',
        "Q1": f'{Q1:.2f}',
        "Q2": f'{Q2:.2f}', 
        "Q3": f'{Q3:.2f}',
        "IQR": f'{IQR:.2f}',
        "Lower Bound": f'{lower_bound:.2f}',
        "Upper Bound": f'{upper_bound:.2f}',
    }

    return outliers, result

# show iqr
def show_IQR(df, numeric_col, detail):
    iqr_outliers = {}
    iqr_outlier_df = {}

    for col in numeric_col:
        iqr_outlier_df[col], iqr_outliers[col] = detect_outliers_IQR(df, col)
    iqr_df = pd.DataFrame(iqr_outliers).T
    st.dataframe(iqr_df)

    if detail:
        for col in numeric_col:
            if not iqr_outlier_df[col].empty:
                with st.expander(f"**Outliers in {col} (Count: {iqr_outliers[col]['Outliers']})**"):
                    st.dataframe(iqr_outlier_df[col])
    
    return iqr_df

# show outlier
def show_outliers(df, numeric_col):
    st.markdown("### Outlier")

    # detail info checkbox
    detail = st.checkbox("Show Details", value=False)

    # z-score
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Z-Score**")
            z_score_df = show_zscore(df, numeric_col, detail)
        
        with col2:
            st.markdown("**IQR**")
            iqr_df = show_IQR(df, numeric_col, detail)
            
        outlier_df = pd.concat([z_score_df, iqr_df['Outliers']], axis=1)
        outlier_df.columns = ['Z-Score', 'IQR']
        st.session_state.outlier_df = outlier_df

# boxplot
@st.cache_data
def draw_boxplot(df, column):
    fig = plt.figure(figsize = (18, 8))
    sns.boxplot(x=df[column])
    return fig

# outlier explanation
def outlier_slm():
    # model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    Here is a summary of the Outliers in the dataset: {outlier_table}

    Definition of Outliers:
    Outliers refer to data points that deviate significantly from other observations in a dataset. These can be caused by data entry errors, measurement errors, or they may represent actual variability in the data.

    Status of Outliers in the Dataset (Columns with outliers only):
    The following columns contain outliers, and columns with zero outliers have been excluded from this summary.

    Impact of Outliers on the Dataset:
    The columns with a high proportion of outliers are likely to affect the quality and accuracy of the analysis. Specifically, the following columns have a significant number of outliers, which could lead to potential issues:

    Column Name 1: [Reason why the outliers could be problematic, e.g., "This column is crucial for prediction and has many extreme values, which may skew the model."]
    Column Name 2: [Reason why the outliers could be problematic, e.g., "Contains values that deviate significantly from the norm, leading to potential bias in analysis."]
    Answer:
    """

    # generate prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["outlier_table"]
    )

    # HuggingFaceHub object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, 
                    "max_new_tokens" : 1024}
    )

    # LLm Chain object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # generate response
    response = llm_chain.invoke({
        "outlier_table": st.session_state.outlier_df.to_markdown()
    })

    response_text = response.get("text", "").strip()

    response_start = response_text.find("Answer:")
    if response_start != -1:
        response_text = response_text[response_start + len("Answer:"):].strip()
    else:
        response_text = response_text
    
    if st.session_state.language != "en":
        response_text = google_translate(response_text, "ko")
    
    return response_text

# outlier
def outlier(df, dtype_df):
    numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)', 'Numeric (Continuous)'])].index.to_list()

    # show outliers
    show_outliers(df, numeric_col)

    # boxplot
    with st.container(border=True):
        st.markdown("**Boxplot**")
        option = st.selectbox(
            "Select the column that you want to draw boxplot",
            numeric_col
        )
        boxplot_fig = draw_boxplot(df, option)
        st.pyplot(boxplot_fig)
    
    # outlier explanation
    response = outlier_slm()
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Overview Page ######################################

# overview page
def overview_page():
    # Title
    st.title("Overview")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        tab1, tab2, tab3 = st.tabs(['Overview', 'Missing Values', 'Outlier'])
        with tab1:
            overview(df, dtype_df)

        with tab2:
            missing_values(df)

        with tab3:
            outlier(df, dtype_df)

    else:
        st.warning(_("Please upload a CSV file to view this page."))

###################################### main ######################################

# main page
overview_page()