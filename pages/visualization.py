"""
This is the Visualization page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import numpy as np
import gettext
import os

from st_mui_table import st_mui_table
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import plotly.express as px

from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from konlpy.tag import Okt
import re

###################################### set ######################################

# set the current page context
st.session_state.current_page = "Visualization"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

###################################### Categorical ######################################

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
        showIndex=False,
        key=key
    )

# categorical info
@st.cache_data
def categorical_info(df, col):
    distinct = df[col].nunique()
    distinct_per = df[col].nunique() / len(df[col]) * 100
    missing = df[col].isna().sum()
    missing_per = df[col].isna().sum() / len(df[col]) * 100
    memory_size = df[col].memory_usage(deep=True) / 1024 ** 2
    value_count = df[col].value_counts()

    # result
    result = pd.DataFrame({
        'Metrics': ['Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Memory Size (MB)'],
        'Values': [
            f'{distinct}',
            f'{distinct_per:.2f}%',
            f'{missing}',
            f'{missing_per:.2f}%',
            f'{memory_size:.2f} MB',
        ]
    })

    # value count
    missing_df = pd.DataFrame({'Values': ['Missing'], 'Counts': [missing]})
    value_count_df = pd.DataFrame({
        'Values': value_count.index,
        'Counts': value_count.values
    })    
    value_count_df = pd.concat([value_count_df, missing_df], ignore_index=True)

    return result, value_count_df

# Set the desired width and height as a percentage of the default size
width = 800 * 0.8  # 80% of the default width
height = 400 * 0.8  # 80% of the default height

# categorical barplot
@st.cache_data
def categorical_barplot(df, col):
    value_count = df[col].value_counts()
    
    # Create interactive bar plot using plotly
    fig = px.bar(
        value_count,
        x=value_count.index,
        y=value_count.values,
        labels={'x': 'Category', 'y': 'Count'},
        color=value_count.index,
        color_discrete_sequence=px.colors.qualitative.Pastel 
    )

    # Customize hover information
    fig.update_traces(hovertemplate='%{x}: %{y}')

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# categorical pie chart
@st.cache_data
def categorical_pie(df, col):
    value_count = df[col].value_counts(normalize=True) * 100

    # Create interactive pie chart using plotly
    fig = px.pie(
        values=value_count, 
        names=value_count.index, 
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Customize hover information
    fig.update_traces(hovertemplate='%{label}: %{value:.1f}%')

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

def categorical_format(df, categorical):
    for i, col in enumerate(df.columns):
        with st.expander(f"**{col}**", expanded=True):
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
                            
            # Use cached functions for the plots and data
            info, value_count = categorical_info(df, col)
            info_add = pd.DataFrame({'Metrics': [''], 'Values': ['']})
            info_table = pd.concat([info, info_add], ignore_index=True)

            # barplot
            with col1:
                fig1 = categorical_barplot(df, col)
                st.plotly_chart(fig1, key=f'barplot_{col}_{i}')
                            
            # pie chart
            with col2:
                fig2 = categorical_pie(df, col)
                st.plotly_chart(fig2, key=f'piechart_{col}_{i}')

            # info table
            with col3:
                if categorical:
                    show_table(info_table, key=f"Info_categorical_{i}", n=6)
                else:
                    show_table(info_table, key=f"Info_bool_{i}", n=5)
                            
            # value count table
            with col4:
                if categorical:
                    show_table(value_count, key=f"Value_count_categorical_{i}", n=value_count.shape[0]+1)
                else:
                    show_table(value_count, key=f"Value_count_bool_{i}", n=value_count.shape[0])

# Categorical page
def categorical(df_cat, df_bool):
    # title
    st.markdown("### Categorical")

    with st.container(border=True):
        tab1, tab2 = st.tabs(['Categorical', 'Boolean'])

        # categorical
        with tab1:
            if df_cat.shape[1] > 0:        
                categorical_format(df_cat, True)
            else:
                st.warning(_("No Categorical Columns"))

        # boolean
        with tab2:
            if df_bool.shape[1] > 0:
                categorical_format(df_bool, False)
            else:
                st.warning(_("No Boolean Columns"))

###################################### Numerical ######################################

# numerical info
def numerical_info(df, col):
    distinct = df[col].nunique()
    distinct_per = distinct / len(df) * 100
    missing = df[col].isnull().sum()
    missing_per = missing / len(df) * 100
    infinit = df[col].isin([np.inf, -np.inf]).sum()
    infinit_per = infinit / len(df) * 100
    memory_size = df[col].memory_usage(deep=True) / 1024 ** 2

    result = pd.DataFrame({
        'Metrics': ['Distinct', 'Distinct (%)', 'Missing', 'Missing (%)', 'Infinit', 'Infinit (%)', 'Memory Size (MB)'],
        'Values': [
            f'{distinct}',
            f'{distinct_per:.2f}%',
            f'{missing}',
            f'{missing_per:.2f}%',
            f'{infinit}',
            f'{infinit_per:.2f}%',
            f'{memory_size:.2f} MB',
        ]
    })

    description = df[col].describe().to_list()
    description_df = pd.DataFrame({
        'Metrics': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Values': description
    })
    nan_df = pd.DataFrame({'Metrics': ['NaN'], 'Values': [missing]})
    description_df = pd.concat([description_df, nan_df], ignore_index=True)
    
    return result, description_df

# Set the desired width and height as a percentage of the default size
width = 800 * 0.8  # 80% of the default width
height = 400 * 0.8  # 80% of the default height

# discrete plot
@st.cache_data
def discrete_plot(df, col):
    value_counts = df[col].value_counts().sort_values(ascending=False)
    
    # Create a bar plot using plotly
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        color=value_counts.index,
        color_discrete_sequence=px.colors.qualitative.Pastel 
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Customize the color bar (note that it may still be auto-generated based on color)
    fig.update_traces(marker=dict(color='skyblue'))

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig


# continuous plot
@st.cache_data
def continuous_plot(df, col):
    # Create a histogram with KDE using plotly
    fig = px.histogram(
        df,
        x=col,
        color_discrete_sequence=['skyblue'],
        marginal='rug',
        histnorm='probability density'
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

# boxplot
@st.cache_data
def boxplot(df, col):
    # Create a box plot using plotly
    fig = px.box(
        df,
        x=col,
        color_discrete_sequence=['skyblue']
    )

    # Remove title and legend
    fig.update_layout(title='', showlegend=False)

    # Set the size of the figure
    fig.update_layout(width=width, height=height)

    return fig

def numerical_format(df, discrete):
    for col in df.columns:
        with st.expander(f"**{col}**", expanded=True):
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            info, description = numerical_info(df, col)
            info_add = pd.DataFrame({'Metrics': [''], 'Values': ['']})
            info_table = pd.concat([info, info_add], ignore_index=True)

            with col1:
                if discrete:
                    fig1 = discrete_plot(df, col)
                else:
                    fig1 = continuous_plot(df, col)
                st.plotly_chart(fig1, key=f'discrete_continuous_{col}')
                        
            with col2:
                fig2 = boxplot(df, col)
                st.plotly_chart(fig2, key=f'boxplot_{col}') 
                        
            with col3:
                if discrete:
                    show_table(info_table, key=f"Info_Discrete_{col}", n=8)
                else:
                    show_table(info_table, key=f"Info_Continuous_{col}", n=8)
                        
            with col4:
                if discrete:
                    show_table(description, key=f"Description_Discrete_{col}", n=9)
                else:
                    show_table(description, key=f"Description_Continuous_{col}", n=9)
 
# Numerical page
def numerical(discrete, continuous):
    # title
    st.markdown("### Numerical")

    with st.container(border=True):
        tab1, tab2 = st.tabs(['Discrete', 'Continuous'])

        with tab1:
            if discrete.shape[1] > 0:
                numerical_format(discrete, True)

            else:
                st.warning(_("No Discrete Columns"))

        with tab2:
            if continuous.shape[1] > 0:
                numerical_format(continuous, False)

            else:
                st.warning(_("No Continuous Columns"))

###################################### TimeSeries ######################################

# timeseries plot
@st.cache_data
def timeseries_plot(df, col):
    decomposition = seasonal_decompose(df[col], model='additive', period=1)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 8))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    return fig

# arima plot
@st.cache_data
def arima_plot(df, col):
    train, test = train_test_split(df[col], test_size=0.2, shuffle=False)
    model = auto_arima(train, seasonal=True, m=12)
    model.fit(train)

    forecast = model.predict(n_periods=len(test))
    test = test.to_frame()
    test['forecast'] = forecast

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test[col], label='Test')
    ax.plot(test.index, test['forecast'], label='Forecast')
    ax.legend()
    return fig, model.summary()

# timeseries page
def timeseries(df, time_cols, cols):
    # title
    st.markdown("### Time Series")

    # Check if time_cols is not empty
    if len(time_cols) > 0:
        time = time_cols[0]
        if time in df.columns:
            df[time] = pd.to_datetime(df[time], errors='coerce')
            df.set_index(time, inplace=True)
        
        if isinstance(cols, list):
            for col in cols:
                if isinstance(col, str):
                    with st.expander(f"**{col}**"):
                        fig1 = timeseries_plot(df, col)
                        st.pyplot(fig1)
                        fig2, summary = arima_plot(df, col)
                        st.pyplot(fig2)
                        st.write(summary)

    else:
        # Initialize timeseries_index
        timeseries_index = False  # 기본값을 False로 설정

        # Check if 'timeseries_index' is already in the session state
        if 'timeseries_index' in st.session_state:
            timeseries_index = st.session_state.timeseries_index

        # Add checkbox to allow the user to indicate if the index is a timeseries
        timeseries_index = st.checkbox("The index of the dataset is timeseries", value=timeseries_index)
        st.session_state.timeseries_index = timeseries_index

        if timeseries_index:
            if isinstance(cols, list):
                for col in cols:
                    if isinstance(col, str):
                        with st.expander(f"**{col}**"):
                            fig1 = timeseries_plot(df, col)
                            st.pyplot(fig1)
                            fig2, summary = arima_plot(df, col)
                            st.pyplot(fig2)
                            st.write(summary)
        else:
            st.warning(_("No Time Series Columns"))

###################################### String ######################################

english_stopwords = set(["the", "and", "is", "in", "to", "of", "it", "that", "for", "on", "with", "as", "this", "by", "at", "from"])

def draw_word_barchart(series):
    text = ' '.join(series.dropna().astype(str))
    english = ' '.join(re.findall(r'[a-zA-Z]+', text))
    korean = ' '.join(re.findall(r'[가-힣]+', text))

    # english word extraction without nltk
    if english:
        english = english.lower()
        tokens = re.findall(r'\b[a-zA-Z]+\b', english)
        tokens = [word for word in tokens if word not in english_stopwords]
    else:
        tokens = []

    # korean word extraction using Okt
    if korean:
        okt = Okt()
        korean_nouns = okt.nouns(korean)
    else:
        korean_nouns = []

    # get top 10 words
    total_nouns = tokens + korean_nouns
    freq_total_nouns = Counter(total_nouns)
    most_common_words = freq_total_nouns.most_common(10)

    # prepare data for bar chart
    words, counts = zip(*most_common_words)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, counts, color='skyblue')
    ax.set_title('Top 10 Most Common Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig

# string page
def string(df, str_col):
    st.markdown("### String Columns")
    if len(str_col) > 0:
        for col in str_col:
            with st.expander(f"**{col}**"):
                fig = draw_word_barchart(df[col])
                st.pyplot(fig)
    else:
        st.warning(_("No String Columns"))

###################################### visualization page ######################################

def visualization_page():
    # title
    st.title("Visualization")
    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df

        cat_col = dtype_df[dtype_df['Data Type'].isin(['Categorical'])].index.to_list()
        bool_col = dtype_df[dtype_df['Data Type'].isin(['Boolean'])].index.to_list()
        num_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)', 'Numeric (Continuous)'])].index.to_list()
        discrete = dtype_df[dtype_df['Data Type'].isin(['Numeric (Discrete)'])].index.to_list()
        continuous = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)'])].index.to_list()
        time_col = dtype_df[dtype_df['Data Type'].isin(['Datetime'])].index.to_list()
        str_col = dtype_df[dtype_df['Data Type'].isin(['String'])].index.to_list()
    
        tab1, tab2, tab3, tab4 = st.tabs(['Categorical', 'Numerical', 'Time Series', 'String'])

        with tab1:
            categorical(df[cat_col], df[bool_col])

        with tab2:
            numerical(df[discrete], df[continuous])

        with tab3:
            timeseries(df, time_col, num_col)

        with tab4:
            string(df, str_col)

    else:
        st.warning(_("Please upload a CSV file to view this page."))


###################################### main ######################################
# main
visualization_page()