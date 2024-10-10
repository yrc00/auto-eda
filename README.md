## Auto-EDA Project

This repository is dedicated to the Auto-EDA Project.

By uploading a CSV file, you can easily obtain the results of Exploratory Data Analysis (EDA).

The project consists of data, reviews, visualization, correlation, and modeling pages.

- data: uploading csv files and editing data

- review: overall analysis of datasets, missing values, outliers

- visualization: visualization (categorical, unstructured, discrete, continuous, time series, string)

- correlation: visualizing correlations between features (pairplot, scatterplot graph, correlation analysis)

- modeling: supervised learning (RandomForest, Decision Tree, XGBoost), Clustering, PCA

### How to Use

You can check the demo application at this [link](https://auto-eda.streamlit.app/).

If you would like to test in your local environment, please follow the steps below:

**1. Clone the repository**

```
git clone https://github.com/yrc00/auto-eda.git
```

**2. Create a virtual environment using Anaconda**

```
conda create -n [environment_name] python=3.9
conda activate [environment_name]
```

**3. Install the required libraries from requirements.txt**

```
pip install -r requirements.txt
```

**4. huggingface api 작성**

```
# .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN="your_api_key_here"
```

- For more information on how to issue the hugging face api, please refer to [here](https://different-rat-a10.notion.site/huggingface-api-key-11be5936298380efa7eae60df00ad10c)

**5. Run the Streamlit app**

```
streamlit run app.py
```

---

## Auto-EDA 프로젝트

Auto-EDA 프로젝트를 위한 레포지토리입니다.

CSV 파일을 업로드해서 탐색적 데이터 분석 (EDA)의 결과를 쉽게 얻을 수 있습니다.

해당 프로젝트는 data, overview, visualization, correlation, modeling 페이지로 구성됩니다.

- data: csv 파일 업로드 및 데이터 편집

- overview: 데이터셋의 전반적인 분석, 결측치, 이상치

- visualization: 시각화 (범주형, 불형, 이산형, 연속형, 시계열, 문자열)

- correlation: 피처 간 상관관계 시각화 (pairplot, 산점도 그래프, 상관관계 분석)

- modeling: 지도학습 (RandomForest, Decision Tree, XGBoost), Clustering, PCA


### 사용법

이 [링크](https://auto-eda.streamlit.app/) 에서 데모 어플리케이션을 확인할 수 있습니다.

만약 로컬 환경에서 테스트하고 싶다면, 아래 단계를 따라주세요:

**1. 레포지토리 복사**

```
git clone https://github.com/yrc00/auto-eda.git
```

**2. Anaconda로 가상환경 생성**

```
conda create -n [environment_name] python=3.9
conda activate [environment_name]
```

**3. requirements.txt에 있는 라이브러리 목록 설치**

```
pip install -r requirements.txt
```

**4. huggingface api 작성**

```
# .streamlit/secrets.toml
HUGGINGFACEHUB_API_TOKEN="your_api_key_here"
```

- huggingface api 발급 방법은 [여기](https://different-rat-a10.notion.site/Huggingface-api-key-11be5936298380efa7eae60df00ad10c)를 참고해주세요

**5. streamlit 어플리케이션 실행**

```
streamlit run app.py
```