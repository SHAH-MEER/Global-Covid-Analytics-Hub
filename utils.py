import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import io # Needed for reading response content
import os # Needed to check for file existence

# OWID Data URL
DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
LOCAL_FILE_PATH = "compact.csv"

# Fetch COVID-19 data using caching
@st.cache_data(ttl=3600)  # Cache for 1 hour (applies mainly to download case)
def load_data(url=DATA_URL, local_path=LOCAL_FILE_PATH):
    """Loads OWID data, prioritizing local file, then downloading. Processes dates, and filters to end of 2023."""
    data = pd.DataFrame() # Initialize empty DataFrame
    
    try:
        if os.path.exists(local_path):
            st.info(f"Loading data from local file: {local_path}")
            data = pd.read_csv(local_path)
            st.toast("Loaded data from local file.", icon="📁")
        else:
            st.info(f"Local file '{local_path}' not found. Attempting to download from {url}...")
            # Download data
            response = requests.get(url, timeout=60) # Increased timeout for potentially large file
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Read data from response content
            csv_data = io.StringIO(response.text)
            data = pd.read_csv(csv_data)
            st.toast("Downloaded data from URL.", icon="☁️")
            
            # Optional: Save the downloaded data locally for next time?
            # try:
            #     data.to_csv(local_path, index=False)
            #     st.info(f"Downloaded data saved to {local_path}")
            # except Exception as save_e:
            #     st.warning(f"Could not save downloaded data locally: {save_e}")

        # --- Common Data Processing Steps ---
        if data.empty:
             st.error("Failed to load data from local file or URL.")
             return pd.DataFrame()
             
        # Ensure date is in datetime format
        data['date'] = pd.to_datetime(data['date'])
        
        # --- Data Cleaning/Preprocessing --- 
        core_cols = ['location', 'date', 'population']
        case_death_cols = [col for col in data.columns if 'case' in col or 'death' in col]
        vax_cols = [col for col in data.columns if 'vaccin' in col or 'booster' in col]
        hosp_cols = [col for col in data.columns if 'hosp' in col or 'icu' in col]
        test_cols = [col for col in data.columns if 'test' in col]
        other_cols = ['gdp_per_capita', 'population_density', 'median_age', 'iso_code', 'reproduction_rate'] # Added Rt
        
        relevant_cols = list(set(core_cols + case_death_cols + vax_cols + hosp_cols + test_cols + other_cols))
        
        essential_cols = ['location', 'date', 'iso_code']
        missing_essential = [col for col in essential_cols if col not in data.columns]
        if missing_essential:
             st.error(f"Essential columns missing from loaded data: {missing_essential}")
             return pd.DataFrame()
                 
        cols_to_keep = [col for col in relevant_cols if col in data.columns]
        data = data[cols_to_keep]
        
        data = data.rename(columns={"location": "country", "iso_code": "code"})

        data = data[data['code'].notna() & (~data['code'].str.startswith('OWID_'))]
        
        end_date = pd.to_datetime('2023-12-31')
        data = data[data['date'] <= end_date].copy()
        
        # Calculate rolling averages here after loading and cleaning
        data = calculate_rolling_averages(data, ['new_cases', 'new_deaths'])
        
        return data
        
    except FileNotFoundError:
        st.error(f"Error: The local file {local_path} was specified but not found during read attempt.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading data from {url}: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading or processing: {e}")
        # Consider logging the full traceback for debugging
        # import traceback
        # st.error(traceback.format_exc())
        return pd.DataFrame()

# Calculate metrics for dashboard
def calculate_metrics(df, days=30):
    if df.empty:
        return {"total_cases": 0, "total_deaths": 0, "active_cases": 0, "case_fatality": 0, 
                "cases_trend": 0, "deaths_trend": 0, "recovery_rate": 0}
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.sort_values('date').copy()
    
    # Get the latest date in the data
    latest_date = df['date'].max()
    # Calculate metrics for the most recent days
    recent_period = df[df['date'] >= (latest_date - pd.Timedelta(days=days))]
    
    # If we have enough data, calculate trends
    cases_trend = 0
    deaths_trend = 0
    if len(recent_period) > 1:
        # Split into two halves to compare
        half_period = days // 2
        # Ensure half_period is at least 1 day for comparison
        half_period = max(1, half_period)
        split_date = latest_date - pd.Timedelta(days=half_period)
        
        first_half = recent_period[recent_period['date'] < split_date]
        second_half = recent_period[recent_period['date'] >= split_date]
        
        if not first_half.empty and not second_half.empty:
            # Calculate trends (percentage change)
            avg_cases_first = first_half['new_cases'].mean()
            avg_cases_second = second_half['new_cases'].mean()
            
            avg_deaths_first = first_half['new_deaths'].mean()
            avg_deaths_second = second_half['new_deaths'].mean()
            
            if avg_cases_first and not pd.isna(avg_cases_first) and avg_cases_first != 0:
                cases_trend = ((avg_cases_second - avg_cases_first) / avg_cases_first) * 100
            elif avg_cases_second > 0:
                 cases_trend = 100.0 # Assign a large positive trend if starting from zero/NaN
            
            if avg_deaths_first and not pd.isna(avg_deaths_first) and avg_deaths_first != 0:
                deaths_trend = ((avg_deaths_second - avg_deaths_first) / avg_deaths_first) * 100
            elif avg_deaths_second > 0:
                deaths_trend = 100.0 # Assign a large positive trend if starting from zero/NaN

    # Calculate cumulative and latest metrics
    total_cases = pd.to_numeric(df['total_cases'], errors='coerce').max() if 'total_cases' in df.columns else pd.to_numeric(df['new_cases'], errors='coerce').sum()
    total_deaths = pd.to_numeric(df['total_deaths'], errors='coerce').max() if 'total_deaths' in df.columns else pd.to_numeric(df['new_deaths'], errors='coerce').sum()
    total_cases = total_cases if pd.notna(total_cases) else 0
    total_deaths = total_deaths if pd.notna(total_deaths) else 0

    # Calculate active cases and recovery rate
    active_cases = 0
    recovery_rate = 0
    if 'total_recovered' in df.columns:
        total_recovered = pd.to_numeric(df['total_recovered'], errors='coerce').max()
        total_recovered = total_recovered if pd.notna(total_recovered) else 0
        active_cases = total_cases - total_deaths - total_recovered
        if total_cases > 0:
            recovery_rate = (total_recovered / total_cases * 100)
    else:
        # Estimate active cases as cases from the last 14 days (if no recovery data)
        fourteen_days_ago = latest_date - pd.Timedelta(days=14)
        active_cases = pd.to_numeric(df[df['date'] >= fourteen_days_ago]['new_cases'], errors='coerce').sum()
        active_cases = active_cases if pd.notna(active_cases) else 0
        
    # Ensure active cases are not negative
    active_cases = max(0, active_cases)

    # Case fatality rate
    case_fatality = (total_deaths / total_cases * 100) if total_cases > 0 else 0
    
    return {
        "total_cases": total_cases,
        "total_deaths": total_deaths,
        "active_cases": active_cases,
        "case_fatality": case_fatality,
        "cases_trend": cases_trend if pd.notna(cases_trend) else 0,
        "deaths_trend": deaths_trend if pd.notna(deaths_trend) else 0,
        "recovery_rate": recovery_rate
    }

# Function to create a metric card
def metric_card(title, value, trend=None, format="number", help_text=None):
    # Format the value based on the type
    if format == "number":
        formatted_value = f"{int(value):,}"
    elif format == "percent":
        formatted_value = f"{value:.2f}%"
    else:
        formatted_value = str(value)
    
    # Determine trend class and symbol
    trend_class = "trend-neutral"
    trend_symbol = "→"
    
    if trend is not None:
        if trend > 1:  # 1% threshold for significance
            trend_class = "trend-up"
            trend_symbol = "↑"
        elif trend < -1:  # -1% threshold for significance
            trend_class = "trend-down"
            trend_symbol = "↓"
    
    # Add help text tooltip if provided
    help_span = f' <span title="{help_text}" style="cursor: help;">(?)</span>' if help_text else ''

    # Generate the HTML for the card
    # Note: This relies on CSS defined in the main app. Ensure CSS is still applied globally.
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{title}{help_span}</div>
        <div class="metric-value">{formatted_value}</div>
        {f'<div class="{trend_class}">{trend_symbol} {abs(trend):.1f}%</div>' if trend is not None else ''}
    </div>
    """
    return card_html

# Function to display metrics in multiple columns
def display_metrics_row(metrics):
    # Use 5 columns if recovery rate > 0, otherwise 4
    num_columns = 5 if metrics["recovery_rate"] > 0 else 4
    cols = st.columns(num_columns)
    
    # Display total cases
    with cols[0]:
        st.markdown(metric_card("Total Cases Reported", metrics["total_cases"], metrics["cases_trend"], help_text="Cumulative recorded cases in the selected period."), unsafe_allow_html=True)
    
    # Display total deaths
    with cols[1]:
        st.markdown(metric_card("Total Deaths Reported", metrics["total_deaths"], metrics["deaths_trend"], help_text="Cumulative recorded deaths in the selected period."), unsafe_allow_html=True)
    
    # Display active cases
    active_help = "Estimated active cases (Total Cases - Total Deaths - Total Recovered). If recovery data unavailable, estimated as new cases in the last 14 days."
    with cols[2]:
        st.markdown(metric_card("Active Cases", metrics["active_cases"], help_text=active_help), unsafe_allow_html=True)
    
    # Display case fatality rate
    cfr_help = "Case Fatality Rate (Total Deaths / Total Cases). Represents the proportion of diagnosed cases that resulted in death."
    with cols[3]:
        st.markdown(metric_card("Case Fatality Rate", metrics["case_fatality"], format="percent", help_text=cfr_help), unsafe_allow_html=True)
    
    # Display Recovery Rate if available (in the 5th column)
    if num_columns == 5:
        rr_help = "Recovery Rate (Total Recovered / Total Cases). Requires recovery data."
        with cols[4]:
           st.markdown(metric_card("Recovery Rate", metrics["recovery_rate"], format="percent", help_text=rr_help), unsafe_allow_html=True)

# Create interactive time series chart with Plotly
def create_plotly_timeseries(data, y_columns, colors, title="", y_title="Value", height=400, show_avg=True, use_log=True):
    if data.empty:
        return None
    
    fig = go.Figure()
    
    for i, col in enumerate(y_columns):
        if col in data.columns:
            # Add main trace (thicker line)
            fig.add_trace(go.Scatter(
                x=data['date'], 
                y=data[col], 
                mode='lines', 
                name=col.replace('_', ' ').title(),
                line=dict(color=colors[i], width=2),
                hovertemplate='%{y:,.0f}<extra></extra>' # Show only value on hover
            ))
            
            # Add rolling average if requested and available
            avg_col = f'{col}_avg'
            if show_avg and avg_col in data.columns:
                fig.add_trace(go.Scatter(
                    x=data['date'], 
                    y=data[avg_col], 
                    mode='lines',
                    name=f"{col.replace('_', ' ').title()} (7-Day Avg)",
                    line=dict(color=colors[i], width=1, dash='dash'),
                    hovertemplate='%{y:,.1f}<extra></extra>' # Show avg value on hover
                ))

    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=y_title,
        yaxis_type='log' if use_log else 'linear',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20) # Adjust margins
    )
    
    # Add range selector buttons but disable the slider
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

# Create an area chart for cumulative data (using Plotly)
def create_plotly_area_chart(data, column, color, title="", height=400, use_log=True):
    if data.empty or column not in data.columns:
        return None
        
    fig = px.area(data, x='date', y=column, title=title)
    fig.update_traces(line_color=color, fillcolor=color, opacity=0.6, hovertemplate='%{y:,.0f}<extra></extra>')
    fig.update_layout(
        height=height,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=column.replace('_', ' ').title(),
        yaxis_type='log' if use_log else 'linear',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

# Create a multi-country comparison chart (using Plotly)
def create_plotly_comparison_chart(data, countries, metric, title="", use_log=True):
    if data.empty or len(countries) == 0:
        return None
    
    # Filter for selected countries
    filtered_data = data[data['country'].isin(countries)].copy()
    
    if filtered_data.empty or metric not in filtered_data.columns:
         st.warning(f"Metric '{metric}' not found for selected countries.")
         return None
    
    fig = px.line(filtered_data, x="date", y=metric, color="country", 
                  title=title, labels={'country':'Country/Region'})
    fig.update_traces(hovertemplate='%{y:,.0f}<extra></extra>')
    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=metric.replace('_', ' ').title(),
        yaxis_type='log' if use_log else 'linear',
        legend_title_text='Country/Region'
    )
    return fig

# Create a bar chart for comparing countries (using Plotly)
def create_plotly_bar_chart(data, countries, metric, title=""):
    if data.empty or len(countries) == 0:
        return None
    
    # Filter for selected countries and get latest date for each
    # Important: This function expects the *latest* data for each country if comparing snapshots.
    # If the input `data` is already filtered by date range, this needs adjustment.
    # Assuming `data` might contain multiple dates for the function's reusability.
    latest_data_points = []
    for country in countries:
        country_data = data[data['country'] == country].copy()
        if not country_data.empty:
            latest = country_data.sort_values('date').iloc[-1]
            latest_data_points.append({
                'country': country,
                'value': latest[metric] if metric in latest.index else 0
            })
    
    if not latest_data_points:
        return None
        
    # Convert to DataFrame
    chart_data = pd.DataFrame(latest_data_points)
    chart_data = chart_data.sort_values('value', ascending=False)
    
    # Create the chart
    fig = px.bar(chart_data, x='country', y='value', title=title, 
                 labels={'country':'Country/Region', 'value': metric.replace('_', ' ').title()},
                 color='country', text_auto='.2s') # Format text on bars
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, xaxis_title="Country/Region", showlegend=False)
    return fig

# Create a heatmap for historical data analysis (using Plotly)
def create_plotly_heatmap(data, country, metric, title=""):
    if data.empty:
        return None
    
    # Filter for the selected country (this might be redundant if data is pre-filtered)
    country_data = data[data['country'] == country].copy()
    
    if country_data.empty or metric not in country_data.columns:
        return None
    
    # Add year and month columns
    country_data['year'] = country_data['date'].dt.year
    country_data['month'] = country_data['date'].dt.month
    
    # Aggregate by year and month (use mean for daily metrics)
    agg_data = country_data.groupby(['year', 'month'])[metric].mean().reset_index()
    
    # Pivot for heatmap format
    try:
        heatmap_pivot = agg_data.pivot(index="year", columns="month", values=metric)
    except Exception as e:
        st.error(f"Could not create pivot table for heatmap: {e}")
        return None
        
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = px.imshow(heatmap_pivot, 
                  labels=dict(x="Month", y="Year", color=metric.replace('_', ' ').title()),
                  x=month_names,
                  y=heatmap_pivot.index,
                  aspect="auto", # Adjust aspect ratio
                  text_auto='.1f', # Show values on heatmap cells
                  color_continuous_scale='RdBu_r') # Use a diverging color scale
                  
    fig.update_layout(
        title=title,
        height=300 + len(heatmap_pivot.index) * 20 # Dynamic height
    )
    fig.update_xaxes(side="top")
    
    return fig

# Calculate rolling averages (ensure this function is present and handles grouping)
def calculate_rolling_averages(df, columns, window=7):
    df_copy = df.sort_values(by=["country", "date"]).copy()
    for col in columns:
        if col in df_copy.columns:
            # Ensure column is numeric before rolling
            numeric_col = pd.to_numeric(df_copy[col], errors='coerce')
            # Group by country before applying rolling window
            df_copy[f'{col}_avg'] = numeric_col.groupby(df_copy['country'], group_keys=False).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    return df_copy

# Create a choropleth map for global visualization
def create_plotly_choropleth(data, metric, date, title=""):
    if data.empty or metric not in data.columns or 'code' not in data.columns:
        st.warning("Cannot generate map. Data is missing, or metric/country code column not found.")
        return None

    # Filter data for the specific date
    map_data = data[data['date'] == pd.to_datetime(date)].copy()
    map_data = map_data.dropna(subset=[metric, 'code']) # Drop rows where metric or code is null

    if map_data.empty:
        st.warning(f"No data available for metric '{metric}' on {date}.")
        return None

    # Use log scale for color if the range is large
    min_val = map_data[metric].min()
    max_val = map_data[metric].max()
    use_log_color = False
    # Check min_val > 0 before division
    if pd.notna(min_val) and pd.notna(max_val) and min_val > 0 and max_val / min_val > 100:
        use_log_color = True
        map_data['log_metric'] = np.log10(map_data[metric] + 1) # Add 1 to handle zeros before log
        color_metric = 'log_metric'
        color_axis_label = f"Log10({metric.replace('_', ' ').title()})"
    else:
        color_metric = metric
        color_axis_label = metric.replace('_', ' ').title()
        
    fig = px.choropleth(
        map_data,
        locations="code", # Use ISO alpha-3 codes
        color=color_metric,
        hover_name="country", # Show country name on hover
        hover_data={metric: ':,.0f', 'code': False, 'country': False}, # Show formatted metric value, hide code and country in hover data
        color_continuous_scale=px.colors.sequential.YlOrRd, # Yellow-Orange-Red scale often used for density
        title=f"{title} ({date})",
        labels={color_metric: color_axis_label}
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        ),
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    return fig 