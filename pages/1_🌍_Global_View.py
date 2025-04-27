import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils import create_plotly_choropleth # Import only needed utils

st.set_page_config(layout="wide") # Pages can have their own config

st.markdown("<div class='header-container'><h1>🌍 Global COVID-19 View</h1></div>", unsafe_allow_html=True)

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# --- Sidebar Filters --- 
st.sidebar.header("Global Filters")

# Select Metric for Map
map_metrics = {
    'total_cases_per_million': 'Total Cases per Million',
    'total_deaths_per_million': 'Total Deaths per Million',
    'people_fully_vaccinated_per_hundred': '% Population Fully Vaccinated',
    'positive_rate': 'Test Positivity Rate (%)',
    'new_cases_per_million': 'New Cases per Million (on selected date)'
}
available_map_metrics = {k:v for k,v in map_metrics.items() if k in data.columns}

if not available_map_metrics:
    st.warning("Required metrics for global map view are not available.")
    st.stop() # Stop this page if metrics missing
    
selected_map_metric = st.sidebar.selectbox(
    "Select Metric for Map",
    options=list(available_map_metrics.keys()),
    format_func=lambda k: available_map_metrics[k],
    key='global_map_metric'
)

# Select Date for Map (using slider)
min_date = data['date'].min().date()
max_date = data['date'].max().date()

# Handle potential NaT in min/max date
if pd.isna(min_date) or pd.isna(max_date):
    st.error("Date range could not be determined from the data.")
    st.stop()

date_options_list = pd.date_range(min_date, max_date).date

# Set desired default date and check if it's valid
target_default_date = datetime(2021, 6, 15).date()
default_slider_value = max_date # Default to latest date first
if target_default_date >= min_date and target_default_date <= max_date:
    default_slider_value = target_default_date
    
selected_date = st.sidebar.select_slider(
    "Select Date for Map",
    options=date_options_list,
    value=default_slider_value, # Use calculated default
    key='global_map_date'
)

# --- Main Page Content ---

# Calculate Overall Global Totals (using latest date for each country)
try:
    latest_all_countries = data.loc[data.groupby('country')['date'].idxmax()]
    overall_cases = pd.to_numeric(latest_all_countries['total_cases'], errors='coerce').sum()
    overall_deaths = pd.to_numeric(latest_all_countries['total_deaths'], errors='coerce').sum()
    overall_vax = pd.to_numeric(latest_all_countries['people_fully_vaccinated'], errors='coerce').sum()
    overall_pop = pd.to_numeric(latest_all_countries['population'], errors='coerce').sum()
    
    st.subheader("Global Cumulative Totals (All Time)")
    cols_overall = st.columns(3)
    with cols_overall[0]:
        st.metric("Total Cases Reported", f"{overall_cases:,.0f}")
    with cols_overall[1]:
        st.metric("Total Deaths Reported", f"{overall_deaths:,.0f}")
    with cols_overall[2]:
        vax_help = "Based on latest data for each country."
        if overall_pop and overall_pop > 0:
             vax_help += f" Approx. {overall_vax/overall_pop*100:.1f}% of combined population."
        st.metric("People Fully Vaccinated", f"{overall_vax:,.0f}", help=vax_help)
except Exception as e:
    st.error(f"Could not calculate global totals: {e}")
    
st.divider()

# Calculate Global Summary Stats for selected date
data_on_date = data[data['date'] == pd.to_datetime(selected_date)]

# Prefer new_cases/new_deaths, but fallback to total_cases/total_deaths if missing or all zero
if not data_on_date.empty:
    global_total_cases = pd.to_numeric(data_on_date['new_cases'], errors='coerce').sum()
    global_total_deaths = pd.to_numeric(data_on_date['new_deaths'], errors='coerce').sum()
    if (global_total_cases == 0 or np.isnan(global_total_cases)) and 'total_cases' in data_on_date.columns:
        global_total_cases = pd.to_numeric(data_on_date['total_cases'], errors='coerce').sum()
    if (global_total_deaths == 0 or np.isnan(global_total_deaths)) and 'total_deaths' in data_on_date.columns:
        global_total_deaths = pd.to_numeric(data_on_date['total_deaths'], errors='coerce').sum()
else:
    global_total_cases = 0
    global_total_deaths = 0

st.subheader(f"Global Snapshot on {selected_date}")
cols = st.columns(2)
with cols[0]:
    st.metric("Reported Cases Worldwide (on date)", f"{global_total_cases:,.0f}")
with cols[1]:
    st.metric("Reported Deaths Worldwide (on date)", f"{global_total_deaths:,.0f}")

# Display Map
st.subheader(f"Global Map: {available_map_metrics[selected_map_metric]}")
with st.spinner("Generating global map..."):
    choropleth_map = create_plotly_choropleth(
        data, # Pass the full dataset for map generation
        selected_map_metric,
        selected_date,
        title=available_map_metrics[selected_map_metric]
    )

if choropleth_map:
    st.plotly_chart(choropleth_map, use_container_width=True)
else:
    st.info("Could not generate map for the selected metric and date.")

# Top N Countries Bar Chart
st.subheader(f"Top Countries by {available_map_metrics[selected_map_metric]} on {selected_date}")
data_on_date_map = data[data['date'] == pd.to_datetime(selected_date)].copy()
data_on_date_map = data_on_date_map.dropna(subset=[selected_map_metric, 'country'])

if not data_on_date_map.empty:
    top_n = st.slider("Number of countries to show", 5, 30, 15, key='global_top_n')
    top_countries_data = data_on_date_map.nlargest(top_n, selected_map_metric)
    
    try:
        fig_top_bar = px.bar(top_countries_data,
                            x='country',
                            y=selected_map_metric,
                            title=f"Top {top_n} Countries",
                            labels={'country':'Country/Region', selected_map_metric: available_map_metrics[selected_map_metric]},
                            color='country',
                            text_auto='.2s'
                           )
        fig_top_bar.update_layout(xaxis_title="Country/Region", showlegend=False, height=400 + top_n * 5)
        fig_top_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_top_bar, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate top countries bar chart: {e}")
else:
    st.info(f"No data available for {available_map_metrics[selected_map_metric]} on {selected_date} to show top countries.") 