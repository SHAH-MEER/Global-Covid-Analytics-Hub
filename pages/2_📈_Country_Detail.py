import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import (
    calculate_metrics, 
    display_metrics_row, 
    create_plotly_timeseries, 
    create_plotly_area_chart
)

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>📈 Country Detail Analysis</h1></div>", unsafe_allow_html=True)

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
        pass # Fallback to 0
country = st.sidebar.selectbox(
    "Select Country/Region",
    countries, 
    index=default_country_index,
    key='detail_country'
)

# Get data date bounds
min_data_date = data['date'].min().date()
max_data_date = data['date'].max().date()

# Date range selection with presets
today = datetime.now().date()
def cap_date(date_val):
    return min(date_val, max_data_date)

date_options = {
    "All Time": [min_data_date, max_data_date],
    "Last 30 Days": [cap_date(today - timedelta(days=30)), max_data_date],
    "Last 90 Days": [cap_date(today - timedelta(days=90)), max_data_date],
    "Last 6 Months": [cap_date(today - timedelta(days=180)), max_data_date],
    "Last Year": [cap_date(today - timedelta(days=365)), max_data_date],
    "Custom": None
}
date_selection = st.sidebar.selectbox(
    "Time period",
    list(date_options.keys()),
    key='detail_date_preset'
)

if date_selection == "Custom":
    date_range_tuple = st.sidebar.date_input(
        "Select custom date range",
        value=[min_data_date, max_data_date],
        min_value=min_data_date,
        max_value=max_data_date,
        key='detail_custom_date'
    )
    if len(date_range_tuple) == 2:
        date_range = list(date_range_tuple)
    else:
        date_range = [date_range_tuple[0], max_data_date]
else:
    date_range = date_options[date_selection]

# Convert date_range to datetime64
start_date = max(pd.to_datetime(date_range[0]).tz_localize(None), pd.to_datetime(min_data_date).tz_localize(None))
end_date = min(pd.to_datetime(date_range[1]).tz_localize(None), pd.to_datetime(max_data_date).tz_localize(None))
if start_date > end_date:
    start_date = end_date
final_date_range = [start_date, end_date]

# Filter the data
filtered_data = data[(data['country'] == country) & 
                     (data['date'].between(final_date_range[0], final_date_range[1]))].copy()

if filtered_data.empty:
    st.warning(f"No data available for {country} in the selected period.")
    st.stop()

# Calculate metrics
metrics = calculate_metrics(filtered_data)

# Display metrics
display_metrics_row(metrics)

# Create columns for Gauges
col_gauge1, col_gauge2 = st.columns(2)

with col_gauge1:
    # Add Gauge Chart for Case Fatality Rate
    if metrics["case_fatality"] is not None and metrics["case_fatality"] > 0:
        try:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics["case_fatality"],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Case Fatality Rate (%)", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 2], 'color': 'lightgreen'},
                        {'range': [2, 5], 'color': 'yellow'},
                        {'range': [5, 10], 'color': 'red'}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics["case_fatality"]}
                    }
                ))
            fig_gauge.update_layout(height=250, margin = {'t':40, 'b':0, 'l':10, 'r':10})
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create CFR gauge: {e}")

with col_gauge2:
    # Add Gauge Chart for Vaccination Rate
    if 'people_fully_vaccinated_per_hundred' in filtered_data.columns:
        # Find the last non-NA value in the filtered period
        latest_vax_rate_series = pd.to_numeric(filtered_data['people_fully_vaccinated_per_hundred'], errors='coerce').dropna()
        if not latest_vax_rate_series.empty:
            latest_vax_rate = latest_vax_rate_series.iloc[-1]
            if pd.notna(latest_vax_rate):
                try:
                    fig_vax_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = latest_vax_rate,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "% Pop. Fully Vaccinated", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "green"},
                            'steps': [
                                {'range': [0, 50], 'color': 'lightgray'},
                                {'range': [50, 75], 'color': 'darkgrey'},
                                {'range': [75, 100], 'color': 'dimgray'}]}
                        ))
                    fig_vax_gauge.update_layout(height=250, margin = {'t':40, 'b':0, 'l':10, 'r':10})
                    st.plotly_chart(fig_vax_gauge, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create Vaccination Rate gauge: {e}")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Daily Statistics", "Cumulative Data", "Summary Statistics"])

with tab1:
    st.subheader("Daily New Cases and Deaths")
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner('Generating cases chart...'):
            cases_chart = create_plotly_timeseries(
                filtered_data, 
                ['new_cases'], 
                ['blue'], 
                "Daily New Cases (with 7-Day Avg)",
                show_avg=True
            )
        if cases_chart:
            st.plotly_chart(cases_chart, use_container_width=True)
    with col2:
        with st.spinner('Generating deaths chart...'):
            deaths_chart = create_plotly_timeseries(
                filtered_data, 
                ['new_deaths'], 
                ['red'], 
                "Daily New Deaths (with 7-Day Avg)",
                show_avg=True
            )
        if deaths_chart:
            st.plotly_chart(deaths_chart, use_container_width=True)

with tab2:
    st.subheader("Cumulative Data")
    if 'total_cases' in filtered_data.columns and 'total_deaths' in filtered_data.columns:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner('Generating cumulative cases chart...'):
                total_cases_chart = create_plotly_area_chart(
                    filtered_data, 'total_cases', 'blue', "Total Cases Over Time"
                )
            if total_cases_chart:
                st.plotly_chart(total_cases_chart, use_container_width=True)
        with col2:
            with st.spinner('Generating cumulative deaths chart...'):
                total_deaths_chart = create_plotly_area_chart(
                    filtered_data, 'total_deaths', 'red', "Total Deaths Over Time"
                )
            if total_deaths_chart:
                st.plotly_chart(total_deaths_chart, use_container_width=True)
    else:
        st.info("Cumulative data (total_cases, total_deaths) not available for this selection.")

with tab3:
    st.subheader("Summary Statistics")
    if not filtered_data.empty:
        # Select only numeric columns for describe(), handle potential errors
        numeric_cols_summary = filtered_data.select_dtypes(include=np.number).columns
        cols_to_describe = [col for col in ['new_cases', 'new_deaths'] if col in numeric_cols_summary]
        if cols_to_describe:
             summary_stats = filtered_data[cols_to_describe].describe()
             st.dataframe(summary_stats, use_container_width=True)
        else:
             st.info("No numeric columns (new_cases, new_deaths) available for summary.")
        
        # Add a download button for the filtered data
        try:
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"covid_data_{country}_{final_date_range[0].date()}_{final_date_range[1].date()}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not prepare data for download: {e}")
    else:
        st.info("No data available for summary or download.") 