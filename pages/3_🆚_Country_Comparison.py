import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import (
    create_plotly_comparison_chart, 
    create_plotly_bar_chart
)

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>🆚 Country Comparison</h1></div>", unsafe_allow_html=True)

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# Sidebar filters
st.sidebar.header("Filters")

# Multiple country selection
countries = sorted(data['country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select countries to compare",
    options=countries,
    default=['United States', 'Italy', 'Brazil', 'India'][:min(4, len(countries))]
)

# Metric selection
available_metrics = ['new_cases', 'new_deaths']
per_million_metrics = []
if 'total_cases' in data.columns:
    available_metrics.extend(['total_cases', 'total_deaths'])
# Add vaccination/testing/hosp metrics if they exist
optional_metrics = [
    'people_fully_vaccinated_per_hundred', 'positive_rate',
    'hosp_patients_per_million', 'icu_patients_per_million' 
]
available_metrics.extend([m for m in optional_metrics if m in data.columns])
    
# Check if population data exists to add per-million metrics
if 'population' in data.columns:
    base_metrics_for_pop = ['new_cases', 'new_deaths', 'total_cases', 'total_deaths', 'hosp_patients', 'icu_patients']
    for metric in base_metrics_for_pop:
        if metric in data.columns:
             per_million_metric_name = f'{metric}_per_million'
             if per_million_metric_name not in available_metrics: # Avoid duplicates
                 per_million_metrics.append(per_million_metric_name)
    available_metrics.extend(per_million_metrics)

# Format metric names for display
def format_metric_name(metric):
    return metric.replace('_', ' ').title()

selected_metric = st.sidebar.selectbox(
    "Select metric to compare",
    options=sorted(list(set(available_metrics))), # Ensure unique & sorted
    format_func=format_metric_name,
    index=0, # Default to new_cases if available
    help="Select the metric to plot. Per-million metrics require population data."
)

# Date range selection
min_data_date = data['date'].min().date()
max_data_date = data['date'].max().date()
date_range_tuple = st.sidebar.date_input(
    "Select date range",
    value=[min_data_date, max_data_date],
    min_value=min_data_date,
    max_value=max_data_date,
    key='comparison_date'
)
if len(date_range_tuple) == 2:
    final_date_range = list(date_range_tuple)
else:
    final_date_range = [date_range_tuple[0], max_data_date]

# Convert date_range to datetime64
final_date_range = [pd.to_datetime(d).tz_localize(None) for d in final_date_range]

# Filter the data by date first
filtered_data_by_date = data[data['date'].between(final_date_range[0], final_date_range[1])].copy()

# Calculate per-million metrics if needed
metric_to_plot = selected_metric
if selected_metric.endswith('_per_million'):
    if 'population' in filtered_data_by_date.columns:
        base_metric = selected_metric.replace('_per_million', '')
        if base_metric in filtered_data_by_date.columns:
            pop = pd.to_numeric(filtered_data_by_date['population'], errors='coerce')
            # Ensure base_metric is numeric
            base_values = pd.to_numeric(filtered_data_by_date[base_metric], errors='coerce')
            # Calculate per million, handling potential NaNs/zeros in pop or base_values
            filtered_data_by_date[selected_metric] = base_values.fillna(0) * 1_000_000 / pop.replace(0, np.nan)
            filtered_data_by_date[selected_metric] = filtered_data_by_date[selected_metric].fillna(0)
        else:
            st.warning(f"Base metric '{base_metric}' needed for '{selected_metric}' not found. Cannot calculate.")
            metric_to_plot = 'new_cases' # Sensible fallback
            st.warning(f"Falling back to display: {format_metric_name(metric_to_plot)}")
    else:
        st.warning(f"Selected per-million metric ('{format_metric_name(selected_metric)}'), but 'population' column is missing. Cannot calculate.")
        metric_to_plot = selected_metric.replace('_per_million', '')
        st.warning(f"Falling back to display raw numbers: {format_metric_name(metric_to_plot)}")

# Display empty state if no countries selected
if not selected_countries:
    st.info("Please select at least one country to compare.")
    st.stop()

# Further filter by selected countries for charts/tables
filtered_data_final = filtered_data_by_date[filtered_data_by_date['country'].isin(selected_countries)].copy()

if filtered_data_final.empty:
    st.warning("No data available for the selected countries and date range.")
    st.stop()

if metric_to_plot not in filtered_data_final.columns:
     st.warning(f"Metric '{format_metric_name(metric_to_plot)}' could not be calculated or found for the selected countries/period.")
     st.stop()

# Create tabs for different comparisons
tab1, tab2, tab3, tab4 = st.tabs(["Time Series Comparison", "Current Snapshot", "Distribution Comparison", "Metric Correlation"])

with tab1:
    st.subheader(f"Time Series Comparison: {format_metric_name(metric_to_plot)}")
    # Add log scale toggle
    use_log_scale_comp = st.checkbox("Use Logarithmic Scale for Y-axis", value=True, key='comp_log_scale')
    
    with st.spinner('Generating time series comparison chart...'):
        comparison_chart = create_plotly_comparison_chart(
            filtered_data_final, 
            selected_countries, 
            metric_to_plot, 
            f"Comparison of {format_metric_name(metric_to_plot)} Across Countries",
            use_log=use_log_scale_comp # Pass toggle state
        )
    if comparison_chart:
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        st.info("No data available for time series comparison.")

with tab2:
    st.subheader(f"Current Snapshot: {format_metric_name(metric_to_plot)}")
    # Get latest data point *within the filtered date range* for each selected country
    latest_snapshot_data = filtered_data_final.loc[filtered_data_final.groupby('country')['date'].idxmax()]
    
    with st.spinner('Generating current snapshot bar chart...'):
        bar_chart = create_plotly_bar_chart(
            latest_snapshot_data, # Pass only the latest data points for selected countries
            selected_countries,
            metric_to_plot, 
            f"Current {format_metric_name(metric_to_plot)} by Country (as of latest date in range)"
        )
    if bar_chart:
        st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.info("No data available for the snapshot bar chart.")
    
    # Display data table of the snapshot
    if not latest_snapshot_data.empty:
        st.subheader("Snapshot Data Table")
        display_df = latest_snapshot_data[['country', 'date', metric_to_plot]].copy()
        display_df.rename(columns={'date': 'Latest Date in Range', metric_to_plot: format_metric_name(metric_to_plot)}, inplace=True)
        st.dataframe(display_df.sort_values(by=format_metric_name(metric_to_plot), ascending=False), 
                     use_container_width=True, 
                     hide_index=True)

with tab3:
    st.subheader(f"Distribution Comparison")
    # Allow selecting a daily metric for distribution comparison
    daily_metrics_dist = sorted([m for m in filtered_data_final.columns if ('new_' in m or 'rate' in m or 'hosp_patients' in m or 'icu_patients' in m) and filtered_data_final[m].dtype in [np.int64, np.float64]])
    
    if not daily_metrics_dist:
        st.info("No suitable daily metrics available for distribution comparison.")
    else:
        selected_dist_metric = st.selectbox("Select Daily Metric for Box Plot", 
                                            options=daily_metrics_dist, 
                                            key='comp_dist_metric',
                                            format_func=format_metric_name)
        
        box_plot_data = filtered_data_final[['country', selected_dist_metric]].copy()
        box_plot_data = box_plot_data.dropna()
        
        if not box_plot_data.empty and len(box_plot_data['country'].unique()) > 0:
             with st.spinner("Generating distribution box plot..."):
                 fig_box = px.box(box_plot_data, 
                                x="country", 
                                y=selected_dist_metric, 
                                color="country",
                                title=f"Distribution of Daily {format_metric_name(selected_dist_metric)} ({final_date_range[0].date()} to {final_date_range[1].date()})",
                                labels={'country':'Country/Region', selected_dist_metric: format_metric_name(selected_dist_metric)}
                                )
                 fig_box.update_layout(showlegend=False)
             st.plotly_chart(fig_box, use_container_width=True)
        else:
             st.info(f"Not enough data to compare distribution of {format_metric_name(selected_dist_metric)}.")

with tab4:
    st.subheader("Metric Correlation Snapshot")
    # Use latest snapshot data calculated for tab 2
    if latest_snapshot_data.empty:
        st.info("No snapshot data available for correlation.")
    else:
        # Select numeric metrics available in the snapshot data
        numeric_metrics_scatter = sorted(latest_snapshot_data.select_dtypes(include=np.number).columns.tolist())
        # Remove potentially uninteresting cols like year, month if they exist
        numeric_metrics_scatter = [m for m in numeric_metrics_scatter if m not in ['year', 'month', 'day_of_year']]

        if len(numeric_metrics_scatter) < 2:
            st.info("At least two numeric metrics required for correlation plot in the snapshot data.")
        else:
            col1, col2 = st.columns(2)
            default_x = numeric_metrics_scatter[0] if numeric_metrics_scatter else None
            default_y = numeric_metrics_scatter[1] if len(numeric_metrics_scatter)>1 else None
            with col1:
                 x_metric_scatter = st.selectbox("Select X-axis Metric", options=numeric_metrics_scatter, index=0, key='scatter_x', format_func=format_metric_name)
            with col2:
                 y_metric_scatter = st.selectbox("Select Y-axis Metric", options=numeric_metrics_scatter, index=min(1, len(numeric_metrics_scatter)-1), key='scatter_y', format_func=format_metric_name)

            # Drop rows where selected metrics are NA for plotting
            scatter_plot_data = latest_snapshot_data.dropna(subset=[x_metric_scatter, y_metric_scatter])
            
            if not scatter_plot_data.empty and x_metric_scatter != y_metric_scatter:
                with st.spinner("Generating scatter plot..."):
                     fig_scatter = px.scatter(
                         scatter_plot_data,
                         x=x_metric_scatter,
                         y=y_metric_scatter,
                         color="country",
                         hover_name="country",
                         title=f"{format_metric_name(x_metric_scatter)} vs. {format_metric_name(y_metric_scatter)} (Latest Data in Range)",
                         labels={'country':'Country/Region', x_metric_scatter: format_metric_name(x_metric_scatter), y_metric_scatter: format_metric_name(y_metric_scatter)}
                     )
                     fig_scatter.update_layout(height=600)
                st.plotly_chart(fig_scatter, use_container_width=True)
            elif x_metric_scatter == y_metric_scatter:
                 st.warning("Please select different metrics for X and Y axes.")
            else:
                 st.info("Not enough data points to generate correlation plot after dropping NAs.") 