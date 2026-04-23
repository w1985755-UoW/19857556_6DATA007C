import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title = "Yield Forecaster",
    page_icon  = "🌾",
    layout     = "wide"
)

st.title("🌾 Paddy Yield Forecaster")
st.markdown(
    "Enter last season's climate conditions to forecast "
    "this season's paddy yield."
)

# Load data
Dataset = pd.read_csv("Data/Main_Dataset.csv")

FEATURES = ['Rainfall_Lag1', 'TempMax_Lag1', 'TempMin_Lag1']
TARGET   = 'Average_Yield'

# Build lagged dataset and train SVR
@st.cache_resource
def train_svr_models():
    models  = {}
    scalers = {}

    for season in ['Maha', 'Yala']:
        frames = []
        for district in ['ANURADHAPURA', 'POLONNARUWA']:
            sub = Dataset[
                (Dataset['District'] == district) &
                (Dataset['Season']   == season)
            ].sort_values('Year').copy()

            sub['Rainfall_Lag1'] = sub['Rainfall'].shift(1)
            sub['TempMax_Lag1']  = sub['Temperature_Max'].shift(1)
            sub['TempMin_Lag1']  = sub['Temperature_Min'].shift(1)
            sub = sub.dropna()
            frames.append(sub)

        season_df = pd.concat(frames)
        X         = season_df[FEATURES]
        y         = season_df[TARGET]

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = SVR(kernel='rbf', C=100, epsilon=0.1)
        model.fit(X_scaled, y)

        models[season]  = model
        scalers[season] = scaler

    return models, scalers

models, scalers = train_svr_models()

st.markdown("---")
st.subheader("Input Last Season's Climate Conditions")
st.caption(
    "The model uses last season's climate data to "
    "forecast this season's yield"
)

col1, col2 = st.columns(2)

with col1:
    season = st.selectbox(
        "Select Season to Forecast",
        options = ['Maha', 'Yala'],
        help    = "Maha = Oct–Mar | Yala = Apr–Sep"
    )

    district = st.selectbox(
        "Select District",
        options = ['ANURADHAPURA', 'POLONNARUWA']
    )

with col2:
    if season == 'Maha':
        rain_default = 900
        tmax_default = 33.0
        tmin_default = 22.5
        rain_min, rain_max = 400, 1500
        tmax_min, tmax_max = 29.0, 36.0
        tmin_min, tmin_max = 20.0, 25.0
    else:
        rain_default = 450
        tmax_default = 33.5
        tmin_default = 24.5
        rain_min, rain_max = 150, 850
        tmax_min, tmax_max = 30.0, 36.0
        tmin_min, tmin_max = 22.0, 26.0

    rainfall = st.slider(
        "Last Season Rainfall (mm)",
        min_value = rain_min,
        max_value = rain_max,
        value     = rain_default,
        step      = 10
    )

    temp_max = st.slider(
        "Last Season Mean Max Temperature (°C)",
        min_value = float(tmax_min),
        max_value = float(tmax_max),
        value     = float(tmax_default),
        step      = 0.1
    )

    temp_min = st.slider(
        "Last Season Mean Min Temperature (°C)",
        min_value = float(tmin_min),
        max_value = float(tmax_max),
        value     = float(tmin_default),
        step      = 0.1
    )

st.markdown("---")

if st.button("🌾 Forecast Yield", type="primary"):

    input_data   = np.array([[rainfall, temp_max, temp_min]])
    scaler       = scalers[season]
    input_scaled = scaler.transform(input_data)
    model        = models[season]
    prediction   = model.predict(input_scaled)[0]

    # Historical context
    hist = Dataset[
        (Dataset['Season']   == season) &
        (Dataset['District'] == district)
    ]
    hist_mean = hist[TARGET].mean()
    hist_min  = hist[TARGET].min()
    hist_max  = hist[TARGET].max()

    st.markdown("---")
    st.subheader("Forecast Result")

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecasted Yield",  f"{prediction:,.0f} kg/ha")
    col2.metric("Historical Mean",   f"{hist_mean:,.0f} kg/ha")
    col3.metric("vs Historical Mean",
                f"{prediction - hist_mean:+,.0f} kg/ha")

    # Category
    if prediction < hist_mean * 0.90:
        st.error(
            "⚠️ Below Average Yield — "
            "Poor season forecast"
        )
    elif prediction > hist_mean * 1.10:
        st.success(
            "✅ Above Average Yield — "
            "Good season forecast"
        )
    else:
        st.warning(
            "📊 Average Yield — "
            "Typical season forecast"
        )

    st.markdown("---")
    st.subheader("Historical Context")

    col1, col2, col3 = st.columns(3)
    col1.metric("Historical Min",  f"{hist_min:,.0f} kg/ha")
    col2.metric("Historical Mean", f"{hist_mean:,.0f} kg/ha")
    col3.metric("Historical Max",  f"{hist_max:,.0f} kg/ha")

    # Historical yield chart with forecast point
    st.markdown("---")
    st.subheader("Forecast in Historical Context")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x    = hist['Year'],
        y    = hist[TARGET],
        mode = 'lines+markers',
        name = 'Historical Yield',
        line = dict(color='steelblue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x    = [hist['Year'].max() + 1],
        y    = [prediction],
        mode = 'markers',
        name = 'Forecast',
        marker = dict(
            color  = 'red',
            size   = 15,
            symbol = 'star'
        )
    ))

    fig.add_hline(
        y          = hist_mean,
        line_dash  = 'dash',
        line_color = 'gray',
        annotation_text = f'Historical Mean: {hist_mean:.0f}'
    )

    fig.update_layout(
        title  = f'{district} — {season} Yield Forecast',
        xaxis_title = 'Year',
        yaxis_title = 'Yield (kg/ha)',
        hovermode   = 'x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Input Summary")
    st.info(f"""
    **Season:** {season} | **District:** {district}  
    **Last Season Rainfall:** {rainfall} mm  
    **Last Season Max Temperature:** {temp_max} °C  
    **Last Season Min Temperature:** {temp_min} °C  
    **Forecasted Yield:** {prediction:,.0f} kg/ha  
    **Model Used:** Support Vector Regression (SVR)
    """)