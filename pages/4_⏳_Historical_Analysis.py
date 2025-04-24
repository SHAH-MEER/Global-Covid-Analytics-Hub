import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from utils import create_plotly_heatmap # Import necessary utils

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>⏳ Historical Trends Analysis</h1></div>", unsafe_allow_html=True)

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# Sidebar filters
st.sidebar.header("Filters")

# Country selection
countries = sorted(data['country'].unique())
default_country_index = 0
preferred_default = 'United States'
if preferred_default in countries:
    try:
        default_country_index = countries.index(preferred_default)
    except ValueError:
        pass # Fallback
country = st.sidebar.selectbox(
    "Select Country/Region",
    options=countries,
    index=default_country_index,
    key='historical_country'
)

# Add year filter
available_years = sorted(data['date'].dt.year.unique())
selected_years = st.sidebar.multiselect(
    "Filter by years",
    options=available_years,
    default=available_years
)

# Metric selection
available_metrics = sorted([m for m in data.columns if data[m].dtype in [np.int64, np.float64] and ('new_' in m or 'total_' in m or 'rate' in m)])
selected_metric = st.sidebar.selectbox(
    "Select metric to analyze",
    options=available_metrics,
    index=available_metrics.index('new_cases') if 'new_cases' in available_metrics else 0,
    format_func=lambda k: k.replace('_', ' ').title()
)

# Filter data by country and years
country_data = data[data['country'] == country].copy()
if selected_years:
    country_data = country_data[country_data['date'].dt.year.isin(selected_years)]

if country_data.empty or selected_metric not in country_data.columns:
    st.warning(f"No data available for {country} with metric '{selected_metric.replace('_', ' ').title()}' in the selected years.")
    st.stop()

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Seasonal Patterns", "Year-over-Year", "Moving Averages", "Autocorrelation"])

with tab1:
    st.subheader("Seasonal Patterns")
    with st.spinner('Generating heatmap...'):
        heatmap = create_plotly_heatmap(
            country_data, 
            country, # Pass country name for filtering inside heatmap function if needed
            selected_metric, 
            f"Monthly Average {selected_metric.replace('_', ' ').title()} Heatmap for {country}"
        )
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
        
        # Add monthly average bar chart
        try:
            country_data['month'] = country_data['date'].dt.month
            monthly_avg = country_data.groupby('month')[selected_metric].mean().reset_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            # Ensure month column exists before mapping
            if 'month' in monthly_avg.columns:
                monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else 'Invalid')
                monthly_avg = monthly_avg.sort_values('month') # Sort by month number
                with st.spinner('Generating monthly average chart...'):
                    monthly_chart = px.bar(monthly_avg, x='month_name', y=selected_metric, 
                                           title=f"Overall Monthly Average for {selected_metric.replace('_', ' ').title()}",
                                           labels={'month_name': 'Month', selected_metric: f'Average {selected_metric.replace("_", " ").title()}'})
                    monthly_chart.update_layout(xaxis={'categoryorder':'array', 'categoryarray':month_names})
                st.plotly_chart(monthly_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate monthly average chart: {e}")
    else:
        st.info("Not enough data available for seasonal heatmap analysis.")

with tab2:
    st.subheader("Year-over-Year Comparison")
    country_data['day_of_year'] = country_data['date'].dt.dayofyear
    country_data['year'] = country_data['date'].dt.year
    
    if len(country_data['year'].unique()) > 1:
        with st.spinner('Generating year-over-year chart...'):
            try:
                yoy_chart = px.line(country_data, x='day_of_year', y=selected_metric, color='year',
                                    title=f"Year-over-Year Comparison: {selected_metric.replace('_', ' ').title()}", 
                                    labels={'day_of_year': 'Day of Year', selected_metric: selected_metric.replace('_', ' ').title()})
                yoy_chart.update_layout(yaxis_type='log')
                st.plotly_chart(yoy_chart, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate Year-over-Year chart: {e}")
        
        st.subheader("Yearly Statistics")
        try:
            yearly_stats = country_data.groupby('year')[selected_metric].agg(['mean', 'median', 'max', 'sum']).reset_index()
            yearly_stats.columns = ['Year', 'Average', 'Median', 'Maximum', 'Total']
            st.dataframe(yearly_stats, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not calculate yearly stats: {e}")
    else:
        st.info("Multiple years of data are required for year-over-year comparison.")

with tab3:
    st.subheader("Moving Averages Analysis")
    window = st.slider("Select moving average window (days)", 3, 30, 7, key='ma_window')
    
    # Calculate moving average (ensure column is numeric)
    numeric_metric_col = pd.to_numeric(country_data[selected_metric], errors='coerce')
    ma_col_name = f'{selected_metric}_ma_{window}d'
    country_data[ma_col_name] = numeric_metric_col.rolling(window=window, center=True).mean() # Use centered average
    
    with st.spinner('Generating moving average chart...'):
        try:
            ma_fig = px.line(country_data.dropna(subset=[ma_col_name]), x='date', y=ma_col_name, 
                             title=f"{window}-Day Moving Average",
                             labels={'date': 'Date', ma_col_name: f'{window}-Day Moving Average'})
            ma_fig.update_traces(line=dict(color='red', width=2))
            
            raw_fig = px.line(country_data, x='date', y=selected_metric, title="Raw Data")
            raw_fig.update_traces(line=dict(color='grey', width=1, dash='dash'), opacity=0.7)
            
            combined_fig = go.Figure(data=raw_fig.data + ma_fig.data)
            combined_fig.update_layout(
                title=f"{window}-Day Moving Average vs Daily {selected_metric.replace('_', ' ').title()} for {country}",
                height=500,
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title=selected_metric.replace('_', ' ').title(),
                yaxis_type='log',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(combined_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate moving average chart: {e}")
            
    # Trend detection based on MA
    st.subheader("Trend Detection")
    if not country_data[ma_col_name].dropna().empty and len(country_data[ma_col_name].dropna()) > window:
        try:
            analysis_period = min(90, len(country_data) // 2)
            recent_ma = country_data[ma_col_name].dropna().tail(analysis_period)
            if len(recent_ma) > 1:
                 first_value = recent_ma.iloc[0]
                 last_value = recent_ma.iloc[-1]
                 percent_change = 0
                 if first_value and first_value != 0:
                     percent_change = ((last_value - first_value) / abs(first_value)) * 100
                 
                 delta_val = f"{percent_change:.1f}%"
                 if percent_change > 10:
                     trend_label = "Rising"
                     delta_color = "inverse"
                 elif percent_change < -10:
                     trend_label = "Falling"
                     delta_color = "normal"
                 else:
                     trend_label = "Stable"
                     delta_color = "off"
                     delta_val = None
                 
                 st.metric(label=f"Trend over last {len(recent_ma)} points ({analysis_period} days max)", value=trend_label, delta=delta_val, delta_color=delta_color)
                 st.caption(f"Based on the {window}-day centered moving average of {selected_metric.replace('_', ' ').title()}. Change calculated comparing the first and last points of the MA within the analysis period.")
            else:
                 st.info("Not enough moving average points for trend detection.")
        except Exception as e:
            st.error(f"Error during trend detection: {e}")
    else:
        st.info("Not enough data for trend detection.")

with tab4:
    st.subheader("Autocorrelation Analysis (ACF/PACF)")
    st.caption("Helps identify potential seasonality and persistence in the daily data.")
    series_for_acf = country_data[selected_metric].dropna()
    
    if len(series_for_acf) > 20:
        lags_to_plot = min(40, len(series_for_acf)//2 - 1)
        if lags_to_plot > 0:
             col_acf, col_pacf = st.columns(2)
             try:
                 with col_acf:
                     with st.spinner("Generating ACF plot..."):
                         fig_acf, ax_acf = plt.subplots(figsize=(8, 3))
                         plot_acf(series_for_acf, lags=lags_to_plot, ax=ax_acf, title="Autocorrelation (ACF)", zero=False)
                         st.pyplot(fig_acf)
                         plt.close(fig_acf)
                 with col_pacf:
                     with st.spinner("Generating PACF plot..."):
                         fig_pacf, ax_pacf = plt.subplots(figsize=(8, 3))
                         # Use OLS method for PACF - more robust for some time series
                         plot_pacf(series_for_acf, lags=lags_to_plot, method='ols', ax=ax_pacf, title="Partial Autocorrelation (PACF)", zero=False)
                         st.pyplot(fig_pacf)
                         plt.close(fig_pacf)
             except Exception as e:
                 st.error(f"Could not generate ACF/PACF plots: {e}")
        else:
            st.info("Not enough data points after differencing/dropping NA for meaningful lag calculation.")
    else:
        st.info("Not enough data points (<20) for meaningful autocorrelation analysis.") 