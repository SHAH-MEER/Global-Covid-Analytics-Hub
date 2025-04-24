import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import create_plotly_timeseries # Import necessary utils

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>🏥🧪 Hospitalization & Testing</h1></div>", unsafe_allow_html=True)
st.caption("Data availability for hospitalization and testing varies significantly between countries and over time.")

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# --- Sidebar Filters ---
st.sidebar.header("Filters")
countries = sorted(data['country'].unique())

# Country Selection (default to USA if it has data, else first with data)
default_country_index = 0
# Define the *potential* relevant columns for checking default
potential_rel_cols = ['hosp_patients', 'icu_patients', 'new_tests', 'positive_rate'] 

for i, c in enumerate(countries):
    country_subset = data[data['country']==c]
    if not country_subset.empty:
        # Check which potential columns ACTUALLY exist in THIS subset
        cols_in_subset = [col for col in potential_rel_cols if col in country_subset.columns]
        # Check if any data exists in the columns that are present
        if cols_in_subset and not country_subset[cols_in_subset].isnull().all().all():
            # Prefer US if it has data
            if c == 'United States':
                default_country_index = i
                break
            # Otherwise, take the first one found
            if default_country_index == 0:
                default_country_index = i

country = st.sidebar.selectbox(
    "Select Country/Region",
    countries, 
    index=default_country_index,
    key='hosp_test_country'
)

# Date range selection
min_data_date = data['date'].min().date()
max_data_date = data['date'].max().date()
today = datetime.now().date()
def cap_date(date_val):
    return min(date_val, max_data_date)

date_options = {
    "All Time": [min_data_date, max_data_date],
    "Last 90 Days": [cap_date(today - timedelta(days=90)), max_data_date],
    "Last 6 Months": [cap_date(today - timedelta(days=180)), max_data_date],
    "Last Year": [cap_date(today - timedelta(days=365)), max_data_date],
    "Custom": None
}
date_selection = st.sidebar.selectbox(
    "Time period", 
    list(date_options.keys()), 
    key='hosp_test_date_preset'
)
if date_selection == "Custom":
    date_range_tuple = st.sidebar.date_input(
        "Select custom date range", 
        value=[min_data_date, max_data_date],
        min_value=min_data_date,
        max_value=max_data_date, 
        key='hosp_test_custom_date'
    )
    if len(date_range_tuple) == 2: date_range = list(date_range_tuple)
    else: date_range = [date_range_tuple[0], max_data_date]
else: date_range = date_options[date_selection]

start_date = max(pd.to_datetime(date_range[0]).tz_localize(None), pd.to_datetime(min_data_date).tz_localize(None))
end_date = min(pd.to_datetime(date_range[1]).tz_localize(None), pd.to_datetime(max_data_date).tz_localize(None))
if start_date > end_date: start_date = end_date
final_date_range = [start_date, end_date]

# Filter data for the selected country and date range
country_data = data[(data['country'] == country) & 
                     (data['date'].between(final_date_range[0], final_date_range[1]))].copy()
country_data = country_data.sort_values('date')

if country_data.empty:
    st.warning(f"No data available for {country} in the selected period.")
    st.stop()

# Identify available hospitalization and testing metrics
hosp_test_metrics = {
    'hosp_patients': 'Current Hospital Patients',
    'icu_patients': 'Current ICU Patients',
    'hosp_patients_per_million': 'Hospital Patients per Million',
    'icu_patients_per_million': 'ICU Patients per Million',
    'weekly_hosp_admissions': 'Weekly Hospital Admissions',
    'weekly_icu_admissions': 'Weekly ICU Admissions',
    'weekly_hosp_admissions_per_million': 'Weekly Hospital Admissions per Million',
    'weekly_icu_admissions_per_million': 'Weekly ICU Admissions per Million',
    'new_tests': 'New Tests Conducted',
    'total_tests': 'Total Tests Conducted',
    'total_tests_per_thousand': 'Total Tests per Thousand People',
    'new_tests_per_thousand': 'New Tests per Thousand People',
    'positive_rate': 'Test Positivity Rate (%)',
    'tests_per_case': 'Tests per Confirmed Case'
}
# Separate into different categories for display
hosp_metrics = {k: v for k, v in hosp_test_metrics.items() if 'hosp' in k or 'icu' in k}
test_metrics = {k: v for k, v in hosp_test_metrics.items() if 'test' in k or 'positive' in k}

available_hosp = {k: v for k, v in hosp_metrics.items() if k in country_data.columns and country_data[k].notna().any()}
available_test = {k: v for k, v in test_metrics.items() if k in country_data.columns and country_data[k].notna().any()}

if not available_hosp and not available_test:
    st.warning(f"No hospitalization or testing data found for {country}.")
    st.stop()

# --- Display Key Metrics --- #
st.subheader(f"Latest Hospitalization & Testing Figures for {country}")
key_metric_cols = list(available_hosp.keys()) + list(available_test.keys())
latest_ht_data_row = country_data.dropna(subset=key_metric_cols, how='all').iloc[-1] if not country_data.dropna(subset=key_metric_cols, how='all').empty else None

if latest_ht_data_row is not None:
    # Display metrics in columns
    metrics_to_display_hosp = {k: v for k, v in available_hosp.items() if pd.notna(latest_ht_data_row.get(k))}
    metrics_to_display_test = {k: v for k, v in available_test.items() if pd.notna(latest_ht_data_row.get(k))}
    num_metrics_total = len(metrics_to_display_hosp) + len(metrics_to_display_test)
    
    if num_metrics_total > 0:
        cols_ht = st.columns(num_metrics_total)
        col_idx = 0
        # Display Hospitalization Metrics
        for k, v in metrics_to_display_hosp.items():
            value = latest_ht_data_row.get(k)
            with cols_ht[col_idx]:
                st.metric(label=v, value=f"{value:,.1f}" if "per_million" in k else f"{int(value):,}", 
                          help=f"Latest available data on {latest_ht_data_row['date'].date()}")
            col_idx += 1
        # Display Testing Metrics
        for k, v in metrics_to_display_test.items():
            value = latest_ht_data_row.get(k)
            with cols_ht[col_idx]:
                 if k == 'positive_rate':
                     formatted_val = f"{value * 100:.2f}%"
                 elif 'per_thousand' in k or k == 'tests_per_case':
                     formatted_val = f"{value:.1f}"
                 else:
                     formatted_val = f"{int(value):,}"
                 st.metric(label=v, value=formatted_val, 
                           help=f"Latest available data on {latest_ht_data_row['date'].date()}")
            col_idx += 1
    else:
        st.info(f"No specific metric values available for the latest date ({latest_ht_data_row['date'].date()}) with data.")
else:
    st.info(f"No recent hospitalization or testing data available for {country}.")


# --- Create Tabs for Time Series ---
tabs_needed = []
if available_hosp: tabs_needed.append("Hospitalization Trends")
if available_test: tabs_needed.append("Testing Trends")

if not tabs_needed:
    st.info("No hospitalization or testing time series data available to display.")
    st.stop()

tabs = st.tabs(tabs_needed)
tab_idx = 0

if available_hosp:
    with tabs[tab_idx]:
        st.subheader("Hospitalization Trends Over Time")
        hosp_cols_to_plot = list(available_hosp.keys())
        plot_data = country_data.dropna(subset=hosp_cols_to_plot, how='all')
        if not plot_data.empty:
            hosp_colors = px.colors.qualitative.Plotly[:len(hosp_cols_to_plot)]
            with st.spinner("Generating hospitalization chart..."):
                 hosp_chart = create_plotly_timeseries(
                     plot_data,
                     hosp_cols_to_plot,
                     hosp_colors,
                     title="Hospitalization Metrics",
                     y_title="Count / Rate",
                     show_avg=False # Averages may not make sense for all these metrics together
                 )
            if hosp_chart:
                 st.plotly_chart(hosp_chart, use_container_width=True)
            else:
                 st.info("Could not generate hospitalization chart.")
        else:
            st.info("No data available for hospitalization trends.")
    tab_idx += 1

if available_test:
    with tabs[tab_idx]:
        st.subheader("Testing Trends Over Time")
        # Separate metrics by scale (counts vs rates/ratios)
        test_count_cols = [k for k in available_test if k in ['new_tests', 'total_tests']]
        test_rate_cols = [k for k in available_test if k not in test_count_cols]
        
        plot_data_counts = country_data.dropna(subset=test_count_cols, how='all')
        plot_data_rates = country_data.dropna(subset=test_rate_cols, how='all')
        
        if not plot_data_counts.empty:
            count_colors = px.colors.qualitative.Pastel[:len(test_count_cols)]
            with st.spinner("Generating testing counts chart..."):
                 test_count_chart = create_plotly_timeseries(
                     plot_data_counts,
                     test_count_cols,
                     count_colors,
                     title="Testing Volume",
                     y_title="Count"
                 )
            if test_count_chart:
                st.plotly_chart(test_count_chart, use_container_width=True)

        if not plot_data_rates.empty:
            rate_colors = px.colors.qualitative.Bold[:len(test_rate_cols)]
            with st.spinner("Generating testing rates chart..."):
                 test_rate_chart = create_plotly_timeseries(
                     plot_data_rates,
                     test_rate_cols,
                     rate_colors,
                     title="Testing Rates/Ratios",
                     y_title="Rate / Ratio",
                     use_log=False # Log scale likely inappropriate for rates
                 )
            if test_rate_chart:
                # Modify hovertemplate for percentage
                for trace in test_rate_chart.data:
                    if 'Positive Rate' in trace.name:
                        trace.hovertemplate = '%{y:.2%}<extra></extra>' # Format as percentage
                    elif 'per Thousand' in trace.name or 'Tests Per Case' in trace.name:
                        trace.hovertemplate = '%{y:.1f}<extra></extra>' # Format as float
                st.plotly_chart(test_rate_chart, use_container_width=True)

        if plot_data_counts.empty and plot_data_rates.empty:
             st.info("Could not generate testing charts (no data). ")
        tab_idx += 1

# --- Hospitalization vs Cases Chart ---
st.divider()
st.subheader(f"Hospital Patients vs. New Cases for {country}")

hosp_vs_cases_cols = ['hosp_patients', 'new_cases_avg']

# Check if necessary columns exist
if all(col in country_data.columns for col in hosp_vs_cases_cols):
    plot_data_hvc = country_data.dropna(subset=hosp_vs_cases_cols, how='all')
    if not plot_data_hvc.empty:
        fig_hvc = go.Figure()

        # Add New Cases Avg trace (primary y-axis)
        fig_hvc.add_trace(go.Scatter(
            x=plot_data_hvc['date'], 
            y=plot_data_hvc['new_cases_avg'], 
            mode='lines', 
            name='New Cases (7-Day Avg)',
            line=dict(color='blue', width=2, dash='dash'),
            yaxis='y1' 
        ))

        # Add Hospital Patients trace (secondary y-axis)
        fig_hvc.add_trace(go.Scatter(
            x=plot_data_hvc['date'], 
            y=plot_data_hvc['hosp_patients'], 
            mode='lines', 
            name='Hospital Patients',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))

        # Update layout for dual axes
        fig_hvc.update_layout(
            title=f"Hospital Patients vs. Daily Cases for {country}",
            xaxis_title="Date",
            yaxis=dict(
                 title=dict(
                    text="New Cases (7-Day Avg)",
                    font=dict(color='blue')
                 ),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title=dict(
                    text="Hospital Patients",
                    font=dict(color='orange')
                ),
                tickfont=dict(color='orange'),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hvc, use_container_width=True)
    else:
        st.info("Not enough overlapping data for Hospital Patients and New Cases Avg to plot.")
else:
    st.info("Required data for Hospital Patients vs. Cases plot ('hosp_patients' and 'new_cases_avg') not available.") 