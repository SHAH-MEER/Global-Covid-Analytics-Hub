import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>📈 Time Series Forecasting</h1></div>", unsafe_allow_html=True)
st.caption("Generate future projections for selected metrics using ARIMA models.")

# Suppress ConvergenceWarning from statsmodels
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Data Loading ---
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# --- Sidebar Options ---
st.sidebar.header("Forecasting Options")
countries = sorted(data['country'].unique())

# Define metrics suitable for forecasting (generally non-cumulative, numeric)
forecastable_metrics_raw = [
    'new_cases', 'new_deaths', 
    'new_cases_smoothed', 'new_deaths_smoothed',
    'new_cases_per_million', 'new_deaths_per_million',
    'new_cases_smoothed_per_million', 'new_deaths_smoothed_per_million',
    'hosp_patients', 'icu_patients',
    'weekly_hosp_admissions', 'weekly_icu_admissions',
    'new_tests', 'positive_rate'
]

# Get available countries (simple list for now)
countries = sorted(data['country'].unique())

# Select Country
country = st.sidebar.selectbox(
    "Select Country/Region",
    countries, 
    index=countries.index('United States') if 'United States' in countries else 0,
    key='forecast_country'
)

# Filter data for the selected country to find available metrics
country_data_full = data[data['country'] == country].copy()

# Find which forecastable metrics are actually available for this country
forecastable_metrics = {
    k: k.replace('_', ' ').title() 
    for k in forecastable_metrics_raw 
    if k in country_data_full.columns and country_data_full[k].notna().any()
}

if not forecastable_metrics:
    st.error(f"No suitable metrics found for forecasting for {country}.")
    st.stop()

# Select Metric
selected_metric = st.sidebar.selectbox(
    "Select Metric to Forecast",
    options=list(forecastable_metrics.keys()),
    format_func=lambda k: forecastable_metrics[k],
    key='forecast_metric'
)

# Select Forecast Horizon
forecast_days = st.sidebar.slider(
    "Select Forecast Horizon (Days)", 
    min_value=7, 
    max_value=90, 
    value=30, 
    step=7,
    key='forecast_horizon'
)

# --- Data Preparation ---
st.subheader(f"Forecasting: {forecastable_metrics[selected_metric]} in {country}")

# Prepare the specific time series
# Use the full date range for the selected country
ts_data = country_data_full[['date', selected_metric]].copy()
ts_data = ts_data.set_index('date')
ts_data = ts_data.sort_index()

# Handle missing values - Simple forward fill for ARIMA
# More sophisticated methods (interpolation, seasonal decomposition) could be used.
ts_data[selected_metric] = ts_data[selected_metric].ffill()
ts_data = ts_data.dropna() # Drop any leading NaNs if ffill couldn't fill them

# Ensure data is numeric
ts_data[selected_metric] = pd.to_numeric(ts_data[selected_metric], errors='coerce')
ts_data = ts_data.dropna() # Drop any rows that became NaN after coercion

if len(ts_data) < 30: # Need sufficient data for ARIMA
    st.warning(f"Not enough historical data points ({len(ts_data)}) for {forecastable_metrics[selected_metric]} in {country} to generate a reliable forecast.")
    st.stop()

# --- ARIMA Forecasting --- 
# ARIMA Order (p,d,q) - Using a simple default; could be optimized (e.g., auto_arima)
# (7,1,1) often works reasonably for weekly patterns with differencing
# Using (5,1,0) as a simpler starting point
ARIMA_ORDER = (5, 1, 0)

st.write(f"Generating {forecast_days}-day forecast using ARIMA{ARIMA_ORDER}...")

forecast_results = None
model_fit = None

with st.spinner("Fitting ARIMA model and forecasting..."):
    try:
        # Ensure frequency is set (important for forecasting dates)
        # Use .asfreq('D') to fill any explicitly missing dates, then ffill again
        ts_data = ts_data.asfreq('D').ffill()
        
        model = ARIMA(ts_data[selected_metric], order=ARIMA_ORDER)
        model_fit = model.fit()
        
        # Get forecast object
        forecast_obj = model_fit.get_forecast(steps=forecast_days)
        
        # Extract forecast mean and confidence intervals
        forecast_mean = forecast_obj.predicted_mean
        confidence_intervals = forecast_obj.conf_int(alpha=0.05) # 95% confidence interval
        lower_ci = confidence_intervals.iloc[:, 0]
        upper_ci = confidence_intervals.iloc[:, 1]
        
        # Combine into a DataFrame
        forecast_results = pd.DataFrame({
            'forecast': forecast_mean,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })

    except ValueError as ve:
        st.error(f"ARIMA Model Error: {ve}. This might be due to data characteristics (e.g., constant values after differencing).")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during forecasting: {e}")
        # Optional: Log the full error for debugging
        # print(traceback.format_exc())
        st.stop()

# --- Visualization ---
if forecast_results is not None:
    st.subheader("Forecast Results")
    fig_forecast = go.Figure()

    # Historical Data (plot last N days for clarity, e.g., 180 days)
    history_days_to_plot = 180
    plot_historical = ts_data.iloc[-history_days_to_plot:]
    fig_forecast.add_trace(go.Scatter(
        x=plot_historical.index,
        y=plot_historical[selected_metric],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))

    # Forecasted Values
    fig_forecast.add_trace(go.Scatter(
        x=forecast_results.index,
        y=forecast_results['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Confidence Intervals (Upper Bound)
    fig_forecast.add_trace(go.Scatter(
        x=forecast_results.index,
        y=forecast_results['upper_ci'],
        mode='lines',
        name='Upper 95% CI',
        line=dict(color='rgba(255,0,0,0.3)', width=0.5),
        showlegend=False # Often looks cleaner without CI in legend
    ))

    # Confidence Intervals (Lower Bound) - Fill area
    fig_forecast.add_trace(go.Scatter(
        x=forecast_results.index,
        y=forecast_results['lower_ci'],
        mode='lines',
        name='Lower 95% CI',
        line=dict(color='rgba(255,0,0,0.3)', width=0.5),
        fill='tonexty', # Fill the area between lower and upper CI
        fillcolor='rgba(255,0,0,0.1)',
        showlegend=False
    ))

    # Update layout
    fig_forecast.update_layout(
        title=f"{forecastable_metrics[selected_metric]} Forecast for {country} ({forecast_days} Days)",
        xaxis_title="Date",
        yaxis_title=forecastable_metrics[selected_metric],
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Optional: Ensure forecast y-axis starts at 0 if metric is non-negative
    if (forecast_results['lower_ci'] >= 0).all() and (ts_data[selected_metric] >= 0).all():
        fig_forecast.update_yaxes(rangemode='tozero')
        
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Display raw forecast values (optional)
    with st.expander("View Forecast Data"):
        st.dataframe(forecast_results.style.format("{:.2f}"))
else:
     st.error("Forecast could not be generated.") 