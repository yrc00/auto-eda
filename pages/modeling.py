"""
This is the Overview page

"""

###################################### import ######################################

# library
import streamlit as st
import pandas as pd
import numpy as np
import gettext
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

###################################### set  ######################################

# set the current page context
st.session_state.current_page = "Modeling"

# language setting
locale_path = os.path.join(os.path.dirname(__file__), 'locales')
translator = gettext.translation('base', localedir=locale_path, languages=[st.session_state.language], fallback=True)
translator.install()
_ = translator.gettext

##################################### Supervised ######################################
# learn classifier model
def classifier_model(df, target, model, test_size, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)

    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = model(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    top15 = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)

    return accuracy, precision, recall, f1, top15

# learn regressor model
def regressor_model(df, target, model, test_size, random_state):
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # split data
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    # train test split and fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = model(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    top15 = pd.Series(model.feature_importances_, index=X.columns).nlargest(15)

    return mse, rmse, mae, r2, top15

# plot feature importance
def plot_feature_importances(top15, model_name):
    if not top15.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top15.values, y=top15.index)
        plt.title(f"Top 15 Feature Importance for {model_name}")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        st.pyplot(plt.gcf())
    else:
        st.write(f"No feature importance to display for {model_name}")

# supervised learning
def supervised(df, target):
    # title
    st.markdown("### Supervised Learning")

    if 'supervised_results' not in st.session_state:
        st.session_state.supervised_results = []

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        col3, col4, col5 = st.columns(3)

        # model selection
        with col1:
            model_type = st.selectbox("Model Type", ["Classification", "Regression"])

        # select models
        with col2:
            models = st.multiselect(
                "Models",
                ["Random Forest", "Decision Tree", "XGBoost"],
                ["Random Forest", "Decision Tree", "XGBoost"]
            )
        
        # test size
        with col3:
            test_size = st.number_input(
                "Test Size",
                value=0.3,
                min_value=0.1,
                max_value=0.5,
                step=0.1,
                placeholder="default is 0.3, select test size from 0.1 to 0.5"
            )
        
        # random state
        with col4:
            random_state = st.number_input(
                "Random State",
                value=42,
                min_value=0,
                max_value=100,
                step=1,
                placeholder="default is 42, select random state from 0 to 100"
            )

        # start learn button
        with col5:
            st.markdown(
                """
            <style>
            button {
                height: auto;
                padding-top: 10px !important;
                padding-bottom: 10px !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            start = st.button("Start Learn")

        # learn model
        if start:
            with st.container():
                results = []

                # Classification case
                if model_type == "Classification":
                    for model_name in models:
                        if model_name == "Random Forest":
                            model = RandomForestClassifier
                        elif model_name == "Decision Tree":
                            model = DecisionTreeClassifier
                        elif model_name == "XGBoost":
                            model = XGBClassifier

                        accuracy, precision, recall, f1, top15 = classifier_model(df, target, model, test_size, random_state)
                        results.append([model_name, accuracy, precision, recall, f1])

                        # store the result with metadata
                        st.session_state.supervised_results.append(
                            {
                                "metrics": {"Model": model_name, "Test Size": test_size, "Random State": random_state, "Type": "Classification", "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1},
                                "top15": top15
                            }
                        )

                        with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                            plot_feature_importances(top15, model_name)

                    # convert results to dataframe and display
                    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
                    st.dataframe(results_df.style.highlight_max(["Accuracy", "Precision", "Recall", "F1"], axis=0))
                
                # Regression case
                else:
                    for model_name in models:
                        if model_name == "Random Forest":
                            model = RandomForestRegressor
                        elif model_name == "Decision Tree":
                            model = DecisionTreeRegressor
                        elif model_name == "XGBoost":
                            model = XGBRegressor

                        mse, rmse, mae, r2, top15 = regressor_model(df, target, model, test_size, random_state)
                        
                        # store the result with metadata
                        st.session_state.supervised_results.append(
                            {
                                "metrics": {"Model": model_name, "Test Size": test_size, "Random State": random_state, "Type": "Regression", "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2},
                                "top15": top15
                            }
                        )

                        with st.expander(f"**Top 15 Feature Importance for {model_name}**"):
                            plot_feature_importances(top15, model_name)

                    # convert results to dataframe and display
                    results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "MAE", "R2"])
                    st.dataframe(results_df.style.highlight_min(["MSE", "RMSE", "MAE"], axis=0))

    # Display all stored results in a separate container
    if st.session_state.supervised_results:
        st.markdown("### All Stored Results")
        with st.container(border=True):
            if st.button("Reset Results"):
                st.session_state.supervised_results = []

            # Add a selectbox to choose the sorting metric
            metrics_options = []
            if model_type == "Classification":
                metrics_options = ["Accuracy", "Precision", "Recall", "F1"]
            else:  # Regression
                metrics_options = ["MSE", "RMSE", "MAE", "R2"]
            
            sort_by = st.selectbox("Sort by", metrics_options, index=0)

            # Convert stored results to a DataFrame for sorting
            metrics_df_list = []
            for result in st.session_state.supervised_results:
                metrics = result["metrics"]
                metrics_df_list.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_df_list)

            # Handle sorting based on model type
            if model_type == "Classification":
                if sort_by in metrics_df.columns:
                    metrics_df.sort_values(by=sort_by, ascending=False, inplace=True)
                else:
                    st.error(f"Metric '{sort_by}' is not available in the results.")
            else:  # Regression
                if sort_by in metrics_df.columns:
                    metrics_df.sort_values(by=sort_by, ascending=True, inplace=True)
                else:
                    st.error(f"Metric '{sort_by}' is not available in the results.")
            
            st.dataframe(metrics_df)

            for idx, result in enumerate(st.session_state.supervised_results):
                with st.expander(f"**Result {idx+1} - {result['metrics']['Model']} ({result['metrics']['Type']})**"):
                    metrics_df = pd.DataFrame(result["metrics"], index=[0])
                    st.dataframe(metrics_df)

                    # Plot top 15 feature importance (if available)
                    if not result["top15"].empty:
                        st.markdown("**Top 15 Feature Importance**")
                        plot_feature_importances(result["top15"], result["metrics"]["Model"])

##################################### Clustering ######################################

# clustering pairplot
@st.cache_data
def clustering_pairplot(df):
    return sns.pairplot(df, hue="Cluster", palette="viridis", markers='o')

# Clustinerg - DBSCAN
def DBSCAN_model(df, eps, min_samples):
    # standardize the data
    df_scaled = StandardScaler().fit_transform(df)
    
    # clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(df_scaled)
    fig = clustering_pairplot(df)
    st.pyplot(fig)    

# Clustering - KMeans
def KMeans_model(df, target, n_clusters, random_state):
    # split data
    X = df.drop(target, axis=1)
    y = df[target]

    # standardize the data
    X_scaled = StandardScaler().fit_transform(X)

    # clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    fig = clustering_pairplot(df)
    st.pyplot(fig)

# clustering
def clustering(df, target):
    # clustering
    st.markdown("### Clustering")
    
    col1, col2 = st.columns([3, 1])
        
    # select model
    with col1:
        model = st.multiselect("Model", ["DBSCAN", "KMeans"], ["DBSCAN", "KMeans"])
        
    # start button
    with col2:
        start = st.button("Start Clustering")

    # DBSCAN
    if "DBSCA in model":
        with st.container(border=True):
            st.markdown("**DBSCAN**")

            col1, col2= st.columns(2)

            # epsilon
            with col1:
                eps = st.number_input(
                    "Epsilon", 
                    value=0.5, 
                    step=0.1, 
                    format="%.1f", 
                    placeholder="default is 0.5"
                )

            # min samples            
            with col2:
                min_samples = st.number_input(
                    "Min Samples", 
                    value=df.shape[1]+1, 
                    step=1, 
                    placeholder="default is number of columns + 1"
                )

            # start dbscan
            if start:
                with st.expander("**DBSCAN Pairplot**"):
                    DBSCAN_model(df.dropna(), eps, min_samples)            
    
    # KMeans
    if "KMeans" in model:
        with st.container(border=True):
            st.markdown("**KMeans**")

            col1, col2 = st.columns(2)

            # number of clusters
            with col1:
                n_clusters = st.number_input(
                    "Number of Clusters", 
                    value=3, 
                    step=1, 
                    placeholder="default is 3"
                )
            
            # random state
            with col2:
                random_state = st.number_input(
                    "Random State", 
                    value=42, 
                    step=1, 
                    placeholder="default is 42"
                )

            # start kmeans
            if start:
                with st.expander("**KMeans Pairplot**"):
                    KMeans_model(df.dropna(), target, n_clusters, random_state)

##################################### PCA ######################################

# PCA plot
@st.cache_data
def pca_plot(df, target, n_components):
    # split data
    X = df.drop(target, axis=1)
    y = df[target]

    # standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # pca result
    pca_df = pd.DataFrame(data=pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df[target] = y.values

    # plot
    unique_target = pca_df[target].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_target)))

    fig = plt.figure()
    for color, target_value in zip(colors, unique_target):
        idx = pca_df[target] == target_value
        plt.scatter(pca_df.loc[idx, 'PC1'], pca_df.loc[idx, 'PC2'], c=[color], label=target_value)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    
    return fig, explained_variance_ratio


# PCA
def pca(df, target):
    # PCA
    st.markdown("### PCA")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        if 'n_components' in st.session_state:
            n_components = st.session_state.n_components
        else:
            n_components = 2
        
        # set number of components
        with col1:
            n_components = st.number_input("Number of Components", value=n_components, step=1, placeholder="default is 2")
        
        # start button
        with col2:
            start = st.button("Start PCA")
        
        # start PCA
        if start:
            fig, explained_variance_ratio = pca_plot(df.dropna(), target, n_components)
            st.pyplot(fig)
            
            st.write("Explained Variance Ratio")
            st.write(explained_variance_ratio)

            st.session_state.n_components = n_components

##################################### Modeling Page ######################################

# modeling page
def modeling_page():
    # title
    st.title("Modeling")

    if 'df' in st.session_state:
        df = st.session_state.df
        dtype_df = st.session_state.dtype_df
        target = st.session_state.target
        numeric_col = dtype_df[dtype_df['Data Type'].isin(['Numeric (Continuous)', 'Numeric (Discrete)'])].index.to_list()

        tab1, tab2, tab3 = st.tabs(['Supervised', 'Clustering', 'PCA'])

        # supervised
        with tab1:
            supervised(df.dropna(), target)
        
        with tab2:
            clustering(df[numeric_col], target)
        
        with tab3:
            pca(df[numeric_col], target)
    
    else:
        st.warning(_("Please upload a CSV file to view this page."))

##################################### main ######################################
# main page
modeling_page()