"""
This is the Correlation page

"""

###################################### import ######################################

# library
from dotenv import load_dotenv
import gettext
import os

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import re
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from deep_translator import GoogleTranslator

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Correlation"

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


##################################### Pairplot ######################################

# draw pairplot
@st.cache_data
def get_pairplot(df, target):
    if target is None:
        return sns.pairplot(df, height=2)
    else:
        return sns.pairplot(df, hue=target, height=2)

# pairplot preprocessing
def pairplot_preprocessing(df):
    alpha = 0.05
    results = []

    for column in df.columns:
        data = df[column].dropna()

        if len(data) < 3:
            results.append([column, None, None, 'Not Enough Data'])
            continue
        try:
            stat, p = stats.shapiro(data)
            normal = 'Normal' if p > alpha else 'Not Normal'
            results.append([column, stat, p, normal])
        except Exception as e:
            results.append([column, None, None, f'Error: {str(e)}'])
    
    result_df = pd.DataFrame(results, columns=['Column', 'Shapiro-Wilk Statistics', 'p-value', 'Normality'])
    
    pearson = df.corr(method='pearson')
    spearman = df.corr(method='spearman')
    kendall = df.corr(method='kendall')

    normal_columns = result_df[result_df['Normality'] == 'Normal']['Column'].to_list()

    if len(normal_columns) == len(df.columns):
        method='pearson'
    else:
        method='spearman'
    
    corr_matrix = df.corr(method=method)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_matrix = corr_matrix.where(mask)
    max_corr = corr_matrix.unstack().idxmax()

    return max_corr, method

# pairplot explain
def pairplot_slm(df):
    # model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template = """
    The dataset used contains the following columns:
    {column_names}

    A pairplot was generated to visualize the relationships between these columns. Please provide a general explanation of the pairplot, focusing on the relationships between variables, including notable patterns and trends.

    Among the column pairs, the most linear relationship was found between:
    {linear_columns}

    Please explain the significance of this linear relationship and any potential implications it may have on the dataset.

    Answer:
    """

    # generate LLM object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 1024},
    )

    # generate prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "linear_columns"]
    )

    # generate prompt object
    llm_chain_pairplot = LLMChain(prompt=prompt, llm=llm)

    # preprocessing
    column_names = ', '.join(df.columns)
    linear_columns, method_used = pairplot_preprocessing(df)

    # generate response
    response = llm_chain_pairplot.invoke({
        "column_names": column_names,
        "linear_columns": f'{linear_columns[0]} and {linear_columns[1]} (using {method_used} method)'
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

# pairplot page
def pairplot(df):
    # title
    st.markdown("### Pairplot")
    target = st.session_state.target

    with st.container(border=True):
        if target not in df.columns:
            pairplot_fig = get_pairplot(df, None)
        else:
            pairplot_fig = get_pairplot(df, target)
        st.pyplot(pairplot_fig)

    # show explanation
    response = pairplot_slm(df)
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Scatter Plot ######################################

# draw scatter plot
@st.cache_data
def get_scatter(df, x, y):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    return fig

# scatter plot explanation
def scatter_slm(df, x, y):
    # model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    template = """
    The dataset used contains the following columns:
    {column_names}

    A scatter plot was generated using the X-axis: {x_column} and Y-axis: {y_column}. Please provide a detailed explanation of this scatter plot. 

    Describe the general appearance of the plot, including any noticeable patterns, clusters, or outliers. 

    Additionally, explain the potential relationships between the variables on the X and Y axes and any insights that can be inferred from the scatter plot.

    Answer:
    """

    # generate LLM object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 2024},
    )

    # generate LLM object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_new_tokens": 2024},
    )

    # generate prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["column_names", "x_column", "y_column"]
    )

    # generate prompt object
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    if x and y:
        column_names = ', '.join(df.columns)

        # generate response (fixing the key name to 'column_names')
        response = llm_chain.invoke({
            "column_names": column_names,  # Correct key name here
            "x_column": x,
            "y_column": y
        })

        response_text = response.get("text", "").strip()

        response_start = response_text.find("Answer:")
        if response_start != -1:
            response_text = response_text[response_start + len("Answer:"):].strip()

        # Translate to Korean if necessary
        if st.session_state.language != "en":
            response_text = google_translate(response_text, "ko")
        
        return response_text
    else:
        return "Please select X and Y columns."

# scatter plot page
def scatter(df):
    # title
    st.markdown("### Scatter Plot")

    # input x, y
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X", df.columns)
    with col2:
        y = st.selectbox("Y", df.columns)
    
    # plot
    with st.container(border=True):
        scatter_fig = get_scatter(df, x, y)
        st.pyplot(scatter_fig)
    
    # show explanation
    response = scatter_slm(df, x, y)
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Correlation Heatmap ######################################

# normality test
def normality_test(df):
    alpha = 0.05
    results = []

    # shapiro-wilk test
    for column in df.columns:
        # drop na
        data = df[column].dropna()

        if len(data) < 3:
            results.append([column, None, None, 'Not Enough Data'])
            continue
    
        try:
            stat, p = stats.shapiro(data)
            normal = 'Normal' if p > alpha else 'Not Normal'
            results.append([column, stat, p, normal])
        except Exception as e:
            results.append([column, None, None, f'Error: {str(e)}'])
    
    result_df = pd.DataFrame(results, columns=['Column', 'Shapiro-Wilk Statistics', 'p-value', 'Normality'])
    st.session_state.normality_df = result_df

    # highlight p-value
    def highlight_p_value(val):
        if val is None:  # None 값 처리 추가
            return ''
        color = 'lightyellow' if val < alpha else ''
        return f'background-color: {color}'

    # show result
    st.dataframe(result_df.style.applymap(highlight_p_value, subset=['p-value']))

# heatmap explanation
def heatmap_slm(df):
    # model
    repo_id = "microsoft/Phi-3-mini-4k-instruct"

    # template
    template_1 = """
    Here is the result of the Shapiro-Wilk normality test for each variable:
    {normality_table}

    Based on the normality test results, which correlation method should be used? (Pearson, Spearman, or Kendall)

    Answer:
    """

    template_2 = """
    Here is the result of the {method} correlation matrix:
    {correlation_table}

    Please explain the insight from the selected correlation matrix.
    - include the correlation coefficient between {target} and other variables
    - include the strong correlations (more than 0.3 or less than -0.3) and explain the relationship
    - do not include the whole correlation matrix

    Answer:
    """

    # generate LLM object
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.2, "max_new_tokens": 256},
    )

    # generate prompt templates
    prompt_1 = PromptTemplate(
        template=template_1,
        input_variables=["normality_table"]
    )

    prompt_2 = PromptTemplate(
        template=template_2,
        input_variables=["method", "correlation_table", "target"]
    )

    # generate prompt object
    llm_chain_1 = LLMChain(prompt=prompt_1, llm=llm)
    llm_chain_2 = LLMChain(prompt=prompt_2, llm=llm)
    response_text = ""

    # response 1
    response_1 = llm_chain_1.invoke({
        "normality_table": st.session_state.normality_df.to_markdown()
    })

    response_text_1 = response_1.get("text", "").strip()

    response_start = response_text_1.find("Answer:")
    if response_start != -1:
        method_choice = response_text_1[response_start + len("Answer:"):].strip().lower()
        response_text = f"Selected method: {method_choice}\n"
    else:
        response_text = "No 'Answer 1:' found in the response.\n"
        method_choice = None  # method_choice 설정 실패 처리

    # response 2
    if method_choice == "pearson":
        corr = df.corr(method='pearson')
    elif method_choice == "spearman":
        corr = df.corr(method='spearman')
    elif method_choice == "kendall":
        corr = df.corr(method='kendall')
    else:
        response_text += "Invalid correlation method. Please check the method choice.\n"
        return response_text

    response_2 = llm_chain_2.invoke({
        "method": method_choice,
        "correlation_table": corr.to_markdown(),
        "target": "target"
    })

    response_text_2 = response_2.get("text", "").strip()

    response_start_2 = response_text_2.find("Answer:")
    if response_start_2 != -1:
        response_text_2 = response_text_2[response_start_2 + len("Answer:"):].strip()
        response_text += response_text_2
    else:
        response_text += "No 'Answer 2:' found in the response.\n"

    return response_text

# heatmap
@st.cache_data
def plot_heatmap(df, correlation_method):
    cor_matrix = df.corr(method=correlation_method)
    mask = np.triu(np.ones_like(cor_matrix, dtype=bool), k=1)
    cor_figure = plt.figure(figsize=(10, 10))
    sns.heatmap(cor_matrix, annot=True, mask=mask, fmt=".2f", cmap='coolwarm', square=True)
    st.pyplot(cor_figure)

# correlation heatmap page
def heatmap(df):
    # title
    st.markdown("### Correlation Heatmap")
    with st.container(border=True):
        st.markdown("**Normality Test**")
        normality_test(df)
    
    with st.container(border=True):
        st.markdown("**Correlation Heatmap**")
        tab1, tab2, tab3 = st.tabs(['Pearson', 'Spearman', 'Kendall'])

        # pearson
        with tab1:
            plot_heatmap(df, 'pearson')
        
        # spearman
        with tab2:
            plot_heatmap(df, 'spearman')

        # kendall
        with tab3:
            plot_heatmap(df, 'kendall')
    
    # show explanation
    response = heatmap_slm(df)
    txt = st.text_area("LLM response", response, height=500)
    st.write(f"Response: {len(txt)} characters.")

##################################### Correlation Page ######################################

def correlation_page():
    # title
    st.title("Correlation")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df
        numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)', 'Numeric (Discrete)'])].index.to_list()

        tab1, tab2, tab3 = st.tabs(['Pairplot', 'Scatter Plot', 'Correlation Heatmap'])

        # pariplot
        with tab1:
            pairplot(df[numeric_col])
        
        # scatter plot
        with tab2:
            scatter(df[numeric_col])
        
        # heatmap
        with tab3:
            heatmap(df[numeric_col])
    
    else:
        st.warning(_("Please upload a CSV file to view this page."))

##################################### main ######################################
# main page
correlation_page()
