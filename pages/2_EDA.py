import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title = "EDA",
    page_icon  = "📊",
    layout     = "wide"
)

st.title("📊 Exploratory Data Analysis")

Dataset = pd.read_csv("Data/Main_Dataset.csv")

st.markdown("---")
st.subheader("Filters")

col1, col2 = st.columns(2)
with col1:
    selected_season = st.multiselect(
        "Select Season",
        options = ['Maha', 'Yala'],
        default = ['Maha', 'Yala']
    )
with col2:
    selected_district = st.multiselect(
        "Select District",
        options = ['ANURADHAPURA', 'POLONNARUWA'],
        default = ['ANURADHAPURA', 'POLONNARUWA']
    )

filtered = Dataset[
    (Dataset['Season'].isin(selected_season)) &
    (Dataset['District'].isin(selected_district))
]

st.markdown("---")

# Yield trend — split by season
st.subheader("Yield Over Time")

col1, col2 = st.columns(2)

# Maha
with col1:
    maha_df = filtered[filtered['Season'] == 'Maha']
    fig_maha = px.line(
        maha_df,
        x         = 'Year',
        y         = 'Average_Yield',
        color     = 'District',
        markers   = True,
        labels    = {
            'Year'         : 'Year',
            'Average_Yield': 'Yield (kg/ha)',
            'District'     : 'District'
        },
        title = 'Maha Season — Yield Trend by District'
    )
    fig_maha.update_layout(hovermode='x unified')
    st.plotly_chart(fig_maha, use_container_width=True)

# Yala
with col2:
    yala_df = filtered[filtered['Season'] == 'Yala']
    fig_yala = px.line(
        yala_df,
        x         = 'Year',
        y         = 'Average_Yield',
        color     = 'District',
        markers   = True,
        labels    = {
            'Year'         : 'Year',
            'Average_Yield': 'Yield (kg/ha)',
            'District'     : 'District'
        },
        title = 'Yala Season — Yield Trend by District'
    )
    fig_yala.update_layout(hovermode='x unified')
    st.plotly_chart(fig_yala, use_container_width=True)

st.markdown("---")

# Rainfall trend
st.subheader("Seasonal Rainfall Trend")
rain_df = Dataset[
    Dataset['Season'].isin(selected_season)
].drop_duplicates(subset=['Year', 'Season'])

fig2 = px.line(
    rain_df,
    x       = 'Year',
    y       = 'Rainfall',
    color   = 'Season',
    markers = True,
    labels  = {
        'Year'    : 'Year',
        'Rainfall': 'Rainfall (mm)'
    },
    title = 'Seasonal Total Rainfall — NCP'
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Temperature trends
st.subheader("Seasonal Temperature Trend")
col1, col2 = st.columns(2)

with col1:
    fig3 = px.line(
        rain_df,
        x       = 'Year',
        y       = 'Temperature_Max',
        color   = 'Season',
        markers = True,
        labels  = {
            'Year'           : 'Year',
            'Temperature_Max': 'Max Temperature (°C)'
        },
        title = 'Seasonal Mean Max Temperature'
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = px.line(
        rain_df,
        x       = 'Year',
        y       = 'Temperature_Min',
        color   = 'Season',
        markers = True,
        labels  = {
            'Year'           : 'Year',
            'Temperature_Min': 'Min Temperature (°C)'
        },
        title = 'Seasonal Mean Min Temperature'
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Box plots
st.subheader("Yield Distribution")
col1, col2 = st.columns(2)

with col1:
    fig5 = px.box(
        filtered,
        x      = 'Season',
        y      = 'Average_Yield',
        color  = 'Season',
        labels = {
            'Average_Yield': 'Yield (kg/ha)',
            'Season'       : 'Season'
        },
        title = 'Yield Distribution by Season'
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    fig6 = px.box(
        filtered,
        x      = 'District',
        y      = 'Average_Yield',
        color  = 'District',
        labels = {
            'Average_Yield': 'Yield (kg/ha)',
            'District'     : 'District'
        },
        title = 'Yield Distribution by District'
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# Scatter plots
st.subheader("Climate vs Yield Relationships")
col1, col2 = st.columns(2)

with col1:
    fig7 = px.scatter(
        filtered,
        x          = 'Rainfall',
        y          = 'Average_Yield',
        color      = 'District',
        symbol     = 'Season',
        labels     = {
            'Rainfall'     : 'Seasonal Rainfall (mm)',
            'Average_Yield': 'Yield (kg/ha)'
        },
        title      = 'Rainfall vs Yield',
        hover_data = ['Year']
    )
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    fig8 = px.scatter(
        filtered,
        x          = 'Temperature_Max',
        y          = 'Average_Yield',
        color      = 'District',
        symbol     = 'Season',
        labels     = {
            'Temperature_Max': 'Max Temperature (°C)',
            'Average_Yield'  : 'Yield (kg/ha)'
        },
        title      = 'Max Temperature vs Yield',
        hover_data = ['Year']
    )
    st.plotly_chart(fig8, use_container_width=True)