import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import (
    create_plotly_bar_chart,
    create_plotly_timeseries,
    create_plotly_area_chart
)

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>💉 Vaccination Tracker</h1></div>", unsafe_allow_html=True)

# Check if data is loaded
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data

# --- Sidebar Filters ---
st.sidebar.header("Filters")
countries = sorted(data['country'].unique())

# Find first country with any vaccination data for a better default
default_country_index = 0
vax_cols_exist = ['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']
for i, c in enumerate(countries):
    country_subset = data[data['country']==c]
    if not country_subset.empty and not country_subset[vax_cols_exist].isnull().all().all():
        # Prefer US if it has data
        if c == 'United States':
            default_country_index = i
            break
        # Otherwise, take the first one found
        if default_country_index == 0:
             default_country_index = i
             # Don't break yet, keep searching for US
        
country = st.sidebar.selectbox(
    "Select Primary Country/Region",
    countries, 
    index=default_country_index,
    key='vax_country'
)

# --- Country Comparison Section (Sidebar) ---
st.sidebar.markdown("--- Comparing Countries ---")
selected_countries_comp = st.sidebar.multiselect(
     "Select Countries to Compare Rates",
     options=countries,
     default=[c for c in [country, 'United States', 'United Kingdom', 'Germany'] if c in countries][:3], # Ensure primary is included, max 3 default
     key='vax_comp_countries'
)

# Metric selection for comparison
vax_metrics_comp_options = {
    'total_vaccinations_per_hundred': 'Total Doses per 100 People',
    'people_vaccinated_per_hundred': '% Pop. Vaccinated (>= 1 dose)',
    'people_fully_vaccinated_per_hundred': '% Pop. Fully Vaccinated',
    'total_boosters_per_hundred': 'Boosters per 100 People'
}
available_comp_metrics = {k:v for k,v in vax_metrics_comp_options.items() if k in data.columns}

if not available_comp_metrics:
    st.sidebar.warning("No comparable rate metrics available.")
    selected_comp_metric = None
else:
    selected_comp_metric = st.sidebar.selectbox(
        "Select Rate Metric to Compare",
        options=list(available_comp_metrics.keys()),
        format_func=lambda k: available_comp_metrics[k],
        index=min(2, len(available_comp_metrics)-1), # Default to % Fully Vaxed if possible
        key='vax_comp_metric'
    )

# Filter data for the selected primary country
country_data = data[data['country'] == country].copy()

# Identify available vaccination metrics for the primary country
vax_metrics_base = {
    'total_vaccinations': 'Total Doses Administered',
    'people_vaccinated': 'People Vaccinated (at least 1 dose)',
    'people_fully_vaccinated': 'People Fully Vaccinated',
    'total_boosters': 'Total Boosters Administered'
}
vax_metrics_per_hundred = {
    'total_vaccinations_per_hundred': 'Total Doses per 100 People',
    'people_vaccinated_per_hundred': '% Pop. Vaccinated (>= 1 dose)',
    'people_fully_vaccinated_per_hundred': '% Pop. Fully Vaccinated',
    'total_boosters_per_hundred': 'Boosters per 100 People'
}
new_vax_metrics = {
     'new_vaccinations': 'Daily New Vaccinations',
     'new_vaccinations_smoothed': 'Daily New Vaccinations (7-day Smoothed)',
     # 'new_people_vaccinated_smoothed': 'Daily New People Vaccinated (7-day Smoothed)' # Often less useful
}

available_base = {k: v for k, v in vax_metrics_base.items() if k in country_data.columns and country_data[k].notna().any()}
available_per_hundred = {k: v for k, v in vax_metrics_per_hundred.items() if k in country_data.columns and country_data[k].notna().any()}
available_new = {k: v for k, v in new_vax_metrics.items() if k in country_data.columns and country_data[k].notna().any()}

if not available_base and not available_per_hundred and not available_new:
    st.warning(f"No vaccination data found for {country}.")
    st.stop()

# --- Display Key Metrics for Primary Country ---
st.subheader(f"Latest Vaccination Figures for {country}")
# Find the last row with *any* non-NaN value in the available columns used for display
display_cols = list(available_per_hundred.keys()) + list(available_base.keys())
latest_vax_data_row = country_data.dropna(subset=display_cols, how='all').iloc[-1] if not country_data.dropna(subset=display_cols, how='all').empty else None

if latest_vax_data_row is not None:
    # Use per-hundred if available, else base
    metrics_to_show = available_per_hundred if available_per_hundred else available_base
    # Determine number of columns needed dynamically
    num_metrics = sum(1 for k in metrics_to_show if pd.notna(latest_vax_data_row.get(k)))
    cols_vax = st.columns(max(1, num_metrics))
    idx = 0
    for k, v in metrics_to_show.items():
         value = latest_vax_data_row.get(k)
         if pd.notna(value):
             if idx < len(cols_vax):
                 formatted_val = f"{value:.1f}%" if 'per_hundred' in k else f"{int(value):,}"
                 with cols_vax[idx]:
                     st.metric(label=v, value=formatted_val, help=f"Latest available data on {latest_vax_data_row['date'].date()}")
                 idx += 1
else:
    st.info(f"No recent vaccination data available to display metrics for {country}.")

# Add Gauge for Primary Country Vax Rate
if 'people_fully_vaccinated_per_hundred' in available_per_hundred and latest_vax_data_row is not None:
    latest_vax_rate_gauge = pd.to_numeric(latest_vax_data_row.get('people_fully_vaccinated_per_hundred'), errors='coerce')
    if pd.notna(latest_vax_rate_gauge):
        fig_vax_gauge_detail = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_vax_rate_gauge,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "% Population Fully Vaccinated", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 75], 'color': 'darkgrey'},
                    {'range': [75, 100], 'color': 'dimgray'}]}
            ))
        fig_vax_gauge_detail.update_layout(height=200, margin = {'t':40, 'b':10, 'l':10, 'r':10})
        st.plotly_chart(fig_vax_gauge_detail, use_container_width=True)

# --- Time Series Charts for Primary Country ---
st.subheader(f"Vaccination Trends Over Time for {country}")
tabs_needed = []
if available_new: tabs_needed.append("Daily Vaccinations")
if available_per_hundred: tabs_needed.append("Population Rates")
if available_base: tabs_needed.append("Cumulative Totals")

if not tabs_needed:
    st.info("No time series vaccination data available.")
else:
    tabs = st.tabs(tabs_needed)
    tab_idx = 0

    # Daily Vaccinations Tab
    if available_new:
        with tabs[tab_idx]:
            cols_to_plot = list(available_new.keys())
            plot_data = country_data.dropna(subset=cols_to_plot, how='all')
            if not plot_data.empty:
                 colors = px.colors.qualitative.Plotly[:len(cols_to_plot)]
                 with st.spinner("Generating daily vaccinations chart..."):
                      daily_vax_chart = create_plotly_timeseries(
                          plot_data, 
                          cols_to_plot,
                          colors,
                          title="Daily Vaccination Doses Administered",
                          y_title="Doses",
                          show_avg=False # Smoothed is usually preferred if available
                      )
                 if daily_vax_chart:
                      st.plotly_chart(daily_vax_chart, use_container_width=True)
                 else:
                      st.info("Could not generate daily vaccinations chart.")
            else: st.info("No data for daily vaccinations.")
        tab_idx += 1

    # Population Rates Tab
    if available_per_hundred:
        with tabs[tab_idx]:
            cols_to_plot = list(available_per_hundred.keys())
            plot_data = country_data.dropna(subset=cols_to_plot, how='all')
            if not plot_data.empty:
                colors = px.colors.qualitative.Vivid[:len(cols_to_plot)]
                with st.spinner("Generating population rates chart..."):
                     rate_chart = create_plotly_timeseries(
                         plot_data,
                         cols_to_plot,
                         colors,
                         title="Vaccination Rates (% of Population)",
                         y_title="Percent (%)",
                         use_log=False, # Log scale not appropriate for percentages
                         show_avg=False
                     )
                if rate_chart:
                     # Format hover to show percentage
                     for trace in rate_chart.data:
                         trace.hovertemplate = '%{y:.1f}%<extra></extra>'
                     st.plotly_chart(rate_chart, use_container_width=True)
                else:
                     st.info("Could not generate population rates chart.")
            else: st.info("No data for population rates.")
        tab_idx += 1

    # Cumulative Totals Tab
    if available_base:
        with tabs[tab_idx]:
            cols_to_plot = list(available_base.keys())
            plot_data = country_data.dropna(subset=cols_to_plot, how='all')
            if not plot_data.empty:
                colors = px.colors.qualitative.Pastel[:len(cols_to_plot)]
                # Use area chart for cumulative totals
                if len(cols_to_plot) == 1:
                    with st.spinner(f"Generating cumulative {format_metric_name(cols_to_plot[0])} chart..."):
                        cumulative_chart = create_plotly_area_chart(
                            plot_data, cols_to_plot[0], colors[0], 
                            title=f"Cumulative {format_metric_name(cols_to_plot[0])}"
                        )
                    if cumulative_chart:
                         st.plotly_chart(cumulative_chart, use_container_width=True)
                    else: st.info("Could not generate cumulative chart.")
                else: # Use line chart if multiple cumulative metrics selected
                    with st.spinner("Generating cumulative totals chart..."):
                         cumulative_chart = create_plotly_timeseries(
                             plot_data,
                             cols_to_plot,
                             colors,
                             title="Cumulative Vaccination Totals",
                             y_title="Total Count",
                             show_avg=False
                         )
                    if cumulative_chart:
                         st.plotly_chart(cumulative_chart, use_container_width=True)
                    else: st.info("Could not generate cumulative totals chart.")
            else: st.info("No data for cumulative totals.")
        tab_idx += 1


# --- Comparison Bar Chart --- 
st.divider()
st.subheader(f"Comparison: {available_comp_metrics.get(selected_comp_metric, 'N/A')}")
if not selected_countries_comp:
    st.info("Select countries in the sidebar to compare vaccination rates.")
elif not selected_comp_metric:
     st.warning("No comparison metric selected or available.")
else:
     # Get latest data *for the full dataset* for comparison chart
     latest_comp_data = data.loc[data.groupby('country')['date'].idxmax()]
     # Filter *only* by selected comparison countries and drop NA for the metric
     comp_bar_data = latest_comp_data[
         latest_comp_data['country'].isin(selected_countries_comp) & 
         latest_comp_data[selected_comp_metric].notna()
     ].copy()
     
     if not comp_bar_data.empty:
         with st.spinner("Generating comparison bar chart..."):
             comp_bar_chart = create_plotly_bar_chart(
                 comp_bar_data, # Pass data with latest point for each country
                 selected_countries_comp, # Countries already filtered in comp_bar_data
                 selected_comp_metric,
                 f"Current {available_comp_metrics.get(selected_comp_metric, selected_comp_metric)} Comparison (Latest Available Data)"
             )
         if comp_bar_chart:
             st.plotly_chart(comp_bar_chart, use_container_width=True)
         else:
             st.info("Could not generate comparison bar chart for selected countries/metric.")
     else:
         st.info(f"No data available for metric '{available_comp_metrics.get(selected_comp_metric)}' for the selected comparison countries.")

# --- Vaccination vs Cases Chart ---
st.divider()
st.subheader(f"Daily Vaccinations vs. New Cases for {country}")

vax_vs_cases_cols = ['new_vaccinations_smoothed', 'new_cases_avg']

# Check if necessary columns exist in the primary country data
if all(col in country_data.columns for col in vax_vs_cases_cols):
    plot_data_vvc = country_data.dropna(subset=vax_vs_cases_cols, how='all')
    if not plot_data_vvc.empty:
        fig_vvc = go.Figure()

        # Add New Cases Avg trace (primary y-axis)
        fig_vvc.add_trace(go.Scatter(
            x=plot_data_vvc['date'], 
            y=plot_data_vvc['new_cases_avg'], 
            mode='lines', 
            name='New Cases (7-Day Avg)',
            line=dict(color='blue', width=2),
            yaxis='y1' # Assign to primary y-axis
        ))

        # Add New Vaccinations Smoothed trace (secondary y-axis)
        fig_vvc.add_trace(go.Scatter(
            x=plot_data_vvc['date'], 
            y=plot_data_vvc['new_vaccinations_smoothed'], 
            mode='lines', 
            name='New Vaccinations (7-Day Smoothed)',
            line=dict(color='green', width=2, dash='dash'),
            yaxis='y2' # Assign to secondary y-axis
        ))

        # Update layout for dual axes
        fig_vvc.update_layout(
            title=f"Daily Cases vs. Vaccinations for {country}",
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
                    text="New Vaccinations (7-Day Smoothed)",
                    font=dict(color='green')
                ),
                tickfont=dict(color='green'),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vvc, use_container_width=True)
    else:
        st.info("Not enough overlapping data for New Cases Avg and New Vaccinations Smoothed to plot.")
else:
    st.info("Required data for Cases vs. Vaccinations plot ('new_vaccinations_smoothed' and 'new_cases_avg') not available.") 