import streamlit as st
import pandas as pd

st.set_page_config(
    page_title = "Overview",
    page_icon  = "📋",
    layout     = "wide"
)

st.title("📋 Project Overview")

Dataset = pd.read_csv("Data/Main_Dataset.csv")

st.markdown("---")
st.subheader("About This Study")
st.markdown("""
This study develops a forecasting model for seasonal paddy 
yields in the **North Central Province (NCP) of Sri Lanka** 
based on rainfall and temperature patterns recorded at the 
**Maha Illuppallama meteorological station** from **1995 to 2025**.

The NCP — comprising Anuradhapura and Polonnaruwa districts — 
is Sri Lanka's primary rice producing region. Understanding 
the climate-yield relationship is critical for agricultural 
planning and food security.
""")

st.markdown("---")
st.subheader("Dataset Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records",  len(Dataset))
col2.metric("Years Covered",  "1995–2025")
col3.metric("Districts",      "2")
col4.metric("Seasons",        "2")

st.markdown("---")
st.subheader("Key Statistics by District & Season")

summary = Dataset.groupby(['District', 'Season']).agg(
    Mean_Yield       = ('Average_Yield',   'mean'),
    Max_Yield        = ('Average_Yield',   'max'),
    Min_Yield        = ('Average_Yield',   'min'),
    Mean_Rainfall    = ('Rainfall',        'mean'),
    Mean_Temp_Max    = ('Temperature_Max', 'mean'),
    Mean_Temp_Min    = ('Temperature_Min', 'mean')
).round(2).reset_index()

summary.columns = [
    'District', 'Season',
    'Mean Yield (kg/ha)', 'Max Yield (kg/ha)',
    'Min Yield (kg/ha)',  'Mean Rainfall (mm)',
    'Mean Temp Max (°C)', 'Mean Temp Min (°C)'
]

st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.subheader("Methodology Summary")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Data Sources**
    - Paddy data: Department of Census & Statistics
    - Weather: Maha Illuppallama Met Station

    **Analysis Methods**
    - Exploratory Data Analysis
    - Pearson Correlation Analysis
    - ADF Stationarity Testing
    - STL Decomposition
    - ACF & PACF Analysis
    """)

with col2:
    st.markdown("""
    **Forecasting Models**
    - ARIMA & SARIMA
    - ARIMAX & SARIMAX
    - Random Forest
    - XGBoost
    - Support Vector Regression ← Best
    - Bayesian Ridge
    - K-Nearest Neighbours

    **Train/Test Split**
    - Train: 1996–2017 (75%)
    - Test:  2018–2025 (25%)
    """)

st.markdown("---")
st.subheader("Raw Dataset")
if st.checkbox("Show raw data"):
    st.dataframe(Dataset, use_container_width=True)