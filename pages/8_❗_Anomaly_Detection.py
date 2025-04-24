import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>❗ Anomaly Detection</h1></div>", unsafe_allow_html=True)
st.caption("Identify unusual data points in time series based on rolling statistics.")

# --- Data Loading ---
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# --- Sidebar Options ---
st.sidebar.header("Detection Options")
countries = sorted(data['country'].unique())

# Define metrics suitable for anomaly detection (similar to forecasting)
detectable_metrics_raw = [
    'new_cases', 'new_deaths', 
    'new_cases_smoothed', 'new_deaths_smoothed',
    'new_cases_per_million', 'new_deaths_per_million',
    'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
    'hosp_patients', 'icu_patients',
    'weekly_hosp_admissions', 'weekly_icu_admissions',
    'new_tests', 'positive_rate',
    'reproduction_rate' # Added Rt as it can show anomalies
]

# Get available countries
countries = sorted(data['country'].unique())

# Select Country
country = st.sidebar.selectbox(
    "Select Country/Region",
    countries, 
    index=countries.index('United States') if 'United States' in countries else 0,
    key='anomaly_country'
)

# Filter data for the selected country to find available metrics
country_data_full = data[data['country'] == country].copy()

# Find which detectable metrics are actually available for this country
detectable_metrics = {
    k: k.replace('_', ' ').title() 
    for k in detectable_metrics_raw 
    if k in country_data_full.columns and country_data_full[k].notna().any()
}

if not detectable_metrics:
    st.error(f"No suitable metrics found for anomaly detection for {country}.")
    st.stop()

# Select Metric
selected_metric = st.sidebar.selectbox(
    "Select Metric for Anomaly Detection",
    options=list(detectable_metrics.keys()),
    format_func=lambda k: detectable_metrics[k],
    key='anomaly_metric'
)

# Select Rolling Window
window_size = st.sidebar.slider(
    "Select Rolling Window Size (Days)", 
    min_value=7, 
    max_value=60, 
    value=21, 
    step=7,
    key='anomaly_window'
)

# Select Standard Deviation Threshold
std_threshold = st.sidebar.slider(
    "Select Standard Deviation Threshold", 
    min_value=1.0, 
    max_value=4.0, 
    value=2.5, 
    step=0.5,
    key='anomaly_std_threshold'
)

# --- Data Preparation and Detection ---
st.subheader(f"Anomaly Detection: {detectable_metrics[selected_metric]} in {country}")

# Prepare the specific time series
ts_data = country_data_full[['date', selected_metric]].copy()
ts_data = ts_data.set_index('date')
ts_data = ts_data.sort_index()

# Handle missing values - Interpolate might be better here than ffill
ts_data[selected_metric] = ts_data[selected_metric].interpolate(method='time')
ts_data = ts_data.dropna() # Drop any remaining NaNs

# Ensure data is numeric
ts_data[selected_metric] = pd.to_numeric(ts_data[selected_metric], errors='coerce')
ts_data = ts_data.dropna()

if len(ts_data) < window_size:
    st.warning(f"Not enough historical data points ({len(ts_data)}) for the selected window size ({window_size}).")
    st.stop()

# Calculate Rolling Mean and Standard Deviation
ts_data['rolling_mean'] = ts_data[selected_metric].rolling(window=window_size).mean()
ts_data['rolling_std'] = ts_data[selected_metric].rolling(window=window_size).std()

# Define Anomaly Boundaries
ts_data['upper_bound'] = ts_data['rolling_mean'] + (ts_data['rolling_std'] * std_threshold)
ts_data['lower_bound'] = ts_data['rolling_mean'] - (ts_data['rolling_std'] * std_threshold)

# Identify Anomalies
ts_data['anomaly'] = ((ts_data[selected_metric] > ts_data['upper_bound']) | 
                      (ts_data[selected_metric] < ts_data['lower_bound']))

anomalies_detected = ts_data[ts_data['anomaly']]

st.write(f"Detected {len(anomalies_detected)} anomalies using a {window_size}-day window and {std_threshold} std dev threshold.")

# --- Visualization ---
st.subheader("Time Series with Detected Anomalies")
fig_anomaly = go.Figure()

# Plot the metric time series
fig_anomaly.add_trace(go.Scatter(
    x=ts_data.index,
    y=ts_data[selected_metric],
    mode='lines',
    name=detectable_metrics[selected_metric],
    line=dict(color='blue')
))

# Add Rolling Mean (Optional, can make plot busy)
# fig_anomaly.add_trace(go.Scatter(
#     x=ts_data.index,
#     y=ts_data['rolling_mean'],
#     mode='lines',
#     name=f'{window_size}-Day Rolling Mean',
#     line=dict(color='grey', dash='dash')
# ))

# Add Anomaly Boundaries (Upper and Lower)
fig_anomaly.add_trace(go.Scatter(
    x=ts_data.index,
    y=ts_data['upper_bound'],
    mode='lines',
    name=f'Upper Bound ({std_threshold}σ)',
    line=dict(color='rgba(255,0,0,0.3)', width=0.5),
    showlegend=False
))
fig_anomaly.add_trace(go.Scatter(
    x=ts_data.index,
    y=ts_data['lower_bound'],
    mode='lines',
    name=f'Lower Bound ({std_threshold}σ)',
    line=dict(color='rgba(255,0,0,0.3)', width=0.5),
    fill='tonexty',
    fillcolor='rgba(255,0,0,0.05)',
    showlegend=False
))

# Add Markers for Detected Anomalies
if not anomalies_detected.empty:
    fig_anomaly.add_trace(go.Scatter(
        x=anomalies_detected.index,
        y=anomalies_detected[selected_metric],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8, symbol='x')
    ))

# Update layout
fig_anomaly.update_layout(
    title=f"Anomaly Detection for {detectable_metrics[selected_metric]} in {country}",
    xaxis_title="Date",
    yaxis_title=detectable_metrics[selected_metric],
    hovermode="x unified",
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_anomaly, use_container_width=True)

# Display detected anomalies table (optional)
if not anomalies_detected.empty:
    with st.expander("View Detected Anomalies Data"):
        st.dataframe(anomalies_detected[[selected_metric, 'rolling_mean', 'upper_bound', 'lower_bound']].style.format("{:.2f}")) 