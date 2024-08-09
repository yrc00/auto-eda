"""
page contents

"""

###################################### import ######################################

# library
import streamlit as st
from st_mui_table import st_mui_table
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# functions
from menu import file_upload, menu, data_sidebar

###################################### set  ######################################

# set page config
st.set_page_config(
    page_title="Data",
    page_icon=":material/database:",
    layout="centered")

# set the current page context
st.session_state.current_page = "Data"

##################################### Overview ######################################

# find zero values
def is_zero(series):
    return series == 0

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

# overview tab
def overview_page(df, dtype_df):

    st.markdown("### Overview")

    # row, columns
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

    # data tyupes
    pd.DataFrame(dtype_df['DType'].value_counts())

    overview = pd.DataFrame({
        'Metric': [
            'Number of Columns', 'Number of Rows', 'Missing Cells', 'Missing Cells (%)', 
            'Duplicate Rows', 'Duplicate Rows (%)', 'Total Size in Memory (KB)', 
            'Average Size in Memory (B)', 'Data Types'
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
            'Data Types' # 테이블에 출력되지 않음
        ]
    })


    # show overview
    with st.container(border=True):
        col1, col2 = st.columns(2)

        # rows, columns, missing cells, duplicate rows, memory usage
        with col1:
            st.markdown("**Dataset Overview**")
            show_table(overview, key="overview_table", n=8)
        
        with col2:
            st.markdown("**Data Types**")
            count_dtype = pd.DataFrame(dtype_df['DType'].value_counts()).reset_index()
            count_dtype.columns = ['Data Type', 'Count']
            additional_row = pd.DataFrame({ # 테이블에 출력되지 않음
                'Data Type': ['Additional Type'],
                'Count': [0]
            })
            count_dtype = pd.concat([count_dtype, additional_row], ignore_index=True)
            show_table(count_dtype, key="dtype_table", n=count_dtype.shape[0])
        
    # alert
    with st.expander("**Alerts**"):
        col1, col2 = st.columns([3, 1])
        
        # duplicate
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
            
            col_zeros = is_zero(df[col]).sum()
            col_zeros_per = is_zero(df[col]).mean() * 100
            if col_zeros > 0:
                with col1:
                    st.write(f"```{col}``` has {col_zeros} ({col_zeros_per:.2f}%) zeros values")
                with col2:
                    st.markdown(":green-background[Zeros]")

    if duplicate > 0:
        with st.expander("**Duplicated Rows**"):
            st.dataframe(df[df.duplicated()])

    # edit data types
    with st.expander("**Datatype of Columns**"):
        st.dataframe(dtype_df)


##################################### Missing Values ######################################

def missing_values_page(df):
    st.markdown("### Missing Values")

    with st.container(border=True):
        tab1, tab2, tab3, tab4 = st.tabs(['Bar Chart', 'Matrix', 'Heat Map', 'Dendrogram'])

        with tab1:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.bar(df, ax=ax)
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.matrix(df, ax=ax)
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.heatmap(df, ax=ax)
            st.pyplot(fig)

        with tab4:
            fig, ax = plt.subplots(figsize=(18, 8))
            msno.dendrogram(df, ax=ax)
            st.pyplot(fig)
    
    # missing values table
    with st.expander("**Rows with Missing Values**"):
        missing_values_df = df[df.isna().any(axis=1)]
        st.dataframe(missing_values_df)

##################################### Outlier ######################################

# z-score
def detect_outliers_zsocre(df, column, threshold=3):
    mean = df[column].mean()
    std = df[column].std()
    zscore = (df[column] - mean) / std
    outliers = df[np.abs(zscore) > threshold]
    return outliers.shape[0]

def outlier_IQR(df, column):
    Q1 = df[column].quantile(0.25)
    Q2 = df[column].quantile(0.50)
    Q3 = df[column].quantile(0.75)
    min = df[column].min()
    max = df[column].max()

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
    non_outlier = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].shape[0]
    
    return {
        "min": f'{min:.2f}',
        "max": f'{max:.2f}',
        "Q1": f'{Q1:.2f}',
        "Q2": f'{Q2:.2f}', 
        "Q3": f'{Q3:.2f}',
        "IQR": f'{IQR:.2f}',
        "Lower Bound": f'{lower_bound:.2f}',
        "Upper Bound": f'{upper_bound:.2f}',
        "Outliers": f'{outliers:.2f}',
        "Non-Outliers": f'{non_outlier:.2f}'
    }

def outlier_page(df, dtype_df):
    st.markdown("### Outlier")

    container1 = st.container(border=True)
    container2 = st.container(border=True)
    container3 = st.container(border=True)
    numeric_col = dtype_df[dtype_df['DType'] == 'Numerical'].index.to_list()

    # z-score
    with container1:
        st.markdown("***z-score***")
        outliers = {}
        for col in numeric_col:
            outliers[col] = detect_outliers_zsocre(df, col)
        st.dataframe(outliers)
    
    # IQR
    with container2:
        st.markdown("**IQR**")
        iqr_dict = {}
        for col in numeric_col:
            iqr_dict[col] = outlier_IQR(df, col)
        iqr_df = pd.DataFrame(iqr_dict).T
        st.dataframe(iqr_df)

    # boxplot
    with container3:
        st.markdown("**Boxplot**")
        option = st.selectbox(
            "Select the column that you want to draw boxplot",
            numeric_col
        )

        boxplot_fig = plt.figure(figsize = (18, 8))
        sns.boxplot(x=df[option])
        st.pyplot(boxplot_fig)

##################################### Data ######################################
    

def data_page():
    # Title
    st.title("Data")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        tab1, tab2, tab3 = st.tabs(['Overview', 'Missing Values', 'Outlier'])
        with tab1:
            overview_page(df, dtype_df)

        with tab2:
            missing_values_page(df)

        with tab3:
            outlier_page(df, dtype_df)

    else:
        st.warning("**Please upload your file on the sidebar of Home page**")

###################################### Contents ######################################

# menu
menu()

# file uploader
file_uploaded = file_upload()

# data sidebar
data_sidebar()

# home page
data_page()

# Check if the file was uploaded and trigger the re-render
if file_uploaded:
    st.experimental_rerun()