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

# --- Inject CSS for Key Metrics ---
st.markdown("""
<style>
    /* Style the container for each metric card */
    .metric-card {
        background-color: #f0f2f6; /* Light background */
        border: 1px solid #e0e0e0; /* Subtle border */
        border-radius: 8px;        /* Rounded corners */
        padding: 15px;             /* Internal spacing */
        margin-bottom: 10px;       /* Space below each card */
        text-align: center;        /* Center align text */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05); /* Subtle shadow */
        transition: transform 0.2s ease-in-out; /* Smooth hover effect */
    }

    /* Add a slight lift effect on hover */
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 3px 3px 8px rgba(0,0,0,0.1);
    }

    /* Style the label (title) of the metric */
    .metric-label {
        font-size: 0.9em;          /* Slightly smaller font */
        color: #555;              /* Darker grey color */
        margin-bottom: 8px;       /* Space below label */
        font-weight: 500;         /* Medium weight */
    }
    
    /* Style the main value of the metric */
    .metric-value {
        font-size: 1.7em;          /* Larger font size */
        font-weight: 600;         /* Bolder */
        color: #1e1e1e;           /* Very dark grey/black */
        margin-bottom: 8px;       /* Space below value */
    }

    /* Style the trend indicator lines */
    .trend-up, .trend-down, .trend-neutral {
        font-size: 0.85em;         /* Smaller font size */
        font-weight: 500;         /* Medium weight */
    }

    /* Specific color for upward trend (assuming higher is generally 'worse' for cases/deaths) */
    .trend-up {
        color: #e60000;           /* Red for increase */
    }

    /* Specific color for downward trend */
    .trend-down {
        color: #008000;           /* Green for decrease */
    }

    /* Specific color for neutral trend */
    .trend-neutral {
        color: #888;              /* Grey for neutral */
    }
    
    /* Style for the help icon (?) */
    .metric-label span[title] {
        font-size: 0.8em;
        color: #777;
        cursor: help;
        border-bottom: 1px dotted #777; /* Add dotted underline */
    }

</style>
""", unsafe_allow_html=True)
# --- End CSS --- 

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

# --- Remove Gauges from Sidebar --- 
# st.sidebar.markdown("--- Quick Look ---") 
# ... (Removed Gauge code that was previously here) ...
# --- End Removed Sidebar Gauges ---

st.sidebar.markdown("--- Filters (Continued) ---") # Separator

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

# Calculate metrics for display row
metrics = calculate_metrics(filtered_data)

# Display metrics row
st.subheader("Key Metrics for Selected Period")
display_metrics_row(metrics)

# --- Add Gauges Back to Main Content ---
# Use full country data to get latest values for gauges
country_full_data = data[data['country'] == country]
metrics_for_gauge = calculate_metrics(country_full_data)

col_gauge1, col_gauge2 = st.columns(2)

with col_gauge1:
    # Add Gauge Chart for Case Fatality Rate
    if metrics_for_gauge["case_fatality"] is not None and metrics_for_gauge["case_fatality"] > 0:
        try:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics_for_gauge["case_fatality"],
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
                        'value': metrics_for_gauge["case_fatality"]}
                    }
                ))
            # Reset height/margins if they were changed for sidebar
            fig_gauge.update_layout(height=250, margin = {'t':40, 'b':0, 'l':10, 'r':10})
            st.plotly_chart(fig_gauge, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create CFR gauge: {e}")

with col_gauge2:
    # Add Gauge Chart for Vaccination Rate
    if 'people_fully_vaccinated_per_hundred' in country_full_data.columns:
        country_vax_data = country_full_data[country_full_data['people_fully_vaccinated_per_hundred'].notna()]
        if not country_vax_data.empty:
            latest_vax_rate = country_vax_data.sort_values('date').iloc[-1]['people_fully_vaccinated_per_hundred']
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
                    # Reset height/margins if they were changed for sidebar
                    fig_vax_gauge.update_layout(height=250, margin = {'t':40, 'b':0, 'l':10, 'r':10})
                    st.plotly_chart(fig_vax_gauge, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create Vaccination Rate gauge: {e}")
# --- End Gauges in Main Content ---

st.divider() # Add a divider after gauges

# --- Add Log/Linear Scale Toggle ---
use_log_scale = st.checkbox("Use Logarithmic Scale for Y-axis", value=True, key='detail_log_scale')

# --- Create tabs for different visualizations ---
# Add new 'Epidemic Dynamics' tab
tabs_list = ["Daily Statistics", "Cumulative Data", "Epidemic Dynamics", "Summary Statistics"]
tab1, tab2, tab3, tab4 = st.tabs(tabs_list)

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
                show_avg=True,
                use_log=use_log_scale # Apply toggle
            )
        if cases_chart:
            st.plotly_chart(cases_chart, use_container_width=True)
        else:
            st.info("No data for daily cases.")
    with col2:
        with st.spinner('Generating deaths chart...'):
            deaths_chart = create_plotly_timeseries(
                filtered_data, 
                ['new_deaths'], 
                ['red'], 
                "Daily New Deaths (with 7-Day Avg)",
                show_avg=True,
                use_log=use_log_scale # Apply toggle
            )
        if deaths_chart:
            st.plotly_chart(deaths_chart, use_container_width=True)
        else:
            st.info("No data for daily deaths.")


with tab2:
    st.subheader("Cumulative Data")
    if 'total_cases' in filtered_data.columns and 'total_deaths' in filtered_data.columns:
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner('Generating cumulative cases chart...'):
                total_cases_chart = create_plotly_area_chart(
                    filtered_data, 'total_cases', 'blue', "Total Cases Over Time",
                    use_log=use_log_scale # Apply toggle
                )
            if total_cases_chart:
                st.plotly_chart(total_cases_chart, use_container_width=True)
            else:
                st.info("Could not generate cumulative cases chart.")
        with col2:
            with st.spinner('Generating cumulative deaths chart...'):
                total_deaths_chart = create_plotly_area_chart(
                    filtered_data, 'total_deaths', 'red', "Total Deaths Over Time",
                    use_log=use_log_scale # Apply toggle
                )
            if total_deaths_chart:
                st.plotly_chart(total_deaths_chart, use_container_width=True)
            else:
                 st.info("Could not generate cumulative deaths chart.")
    else:
        st.info("Cumulative data (total_cases, total_deaths) not available for this selection.")

# --- New Epidemic Dynamics Tab ---
with tab3:
    st.subheader("Epidemic Dynamics")
    col_rt, col_cfr = st.columns(2)
    
    # Reproduction Rate (Rt) Plot
    with col_rt:
        st.markdown("**Reproduction Rate (Rt)**")
        if 'reproduction_rate' in filtered_data.columns and filtered_data['reproduction_rate'].notna().any():
            with st.spinner('Generating Reproduction Rate chart...'):
                rt_chart = create_plotly_timeseries(
                    filtered_data, 
                    ['reproduction_rate'], 
                    ['purple'], 
                    "Estimated Reproduction Rate (Rt)",
                    y_title="Rt",
                    show_avg=False, # Usually Rt is already smoothed or estimated
                    use_log=False # Log scale not typical for Rt around 1
                )
            if rt_chart:
                # Add a horizontal line at Rt=1 for reference
                rt_chart.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="Rt = 1 Threshold")
                st.plotly_chart(rt_chart, use_container_width=True)
            else:
                st.info("Could not generate Rt chart.")
        else:
            st.info("Reproduction Rate (Rt) data not available for this country/period.")

    # Case Fatality Rate (CFR) Trend Plot
    with col_cfr:
        st.markdown("**Case Fatality Rate (CFR) Trend**")
        if 'total_cases' in filtered_data.columns and 'total_deaths' in filtered_data.columns:
            # Calculate CFR, handle potential division by zero
            cases = pd.to_numeric(filtered_data['total_cases'], errors='coerce')
            deaths = pd.to_numeric(filtered_data['total_deaths'], errors='coerce')
            # Avoid division by zero or near-zero cases
            cfr_data = filtered_data[['date']].copy()
            cfr_data['cfr'] = np.where(cases > 100, (deaths * 100) / cases, np.nan) # Calculate only if cases > 100
            cfr_data = cfr_data.dropna(subset=['cfr'])
            
            if not cfr_data.empty:
                 with st.spinner('Generating Case Fatality Rate trend chart...'):
                     cfr_chart = create_plotly_timeseries(
                         cfr_data, 
                         ['cfr'], 
                         ['black'], 
                         "Case Fatality Rate (%) Trend",
                         y_title="CFR (%)",
                         show_avg=True, # Show rolling average of CFR
                         use_log=False # Log scale not useful for CFR percentage
                     )
                 if cfr_chart:
                     # Format hover tooltip to show percentage
                     cfr_chart.update_traces(hovertemplate='%{y:.2f}%<extra></extra>')
                     st.plotly_chart(cfr_chart, use_container_width=True)
                 else:
                     st.info("Could not generate CFR trend chart.")
            else:
                 st.info("Not enough data to calculate CFR trend (requires total_cases > 100).")
        else:
            st.info("Total cases and/or total deaths data not available to calculate CFR.")

# --- Summary Statistics Tab ---
with tab4:
    st.subheader("Summary Statistics")
    if not filtered_data.empty:
        # Select only numeric columns for describe(), handle potential errors
        numeric_cols_summary = filtered_data.select_dtypes(include=np.number).columns
        # Focus on key daily metrics for summary
        cols_to_describe = [col for col in ['new_cases', 'new_deaths', 'reproduction_rate', 'hosp_patients', 'icu_patients'] if col in numeric_cols_summary]
        if cols_to_describe:
             summary_stats = filtered_data[cols_to_describe].describe()
             st.dataframe(summary_stats.style.format("{:,.2f}"), use_container_width=True)
        else:
             st.info("No key numeric columns available for summary statistics in the selected period.")
        
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