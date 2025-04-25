import streamlit as st
import pandas as pd
from utils import load_data

# --- Page Config and Styling ---
st.set_page_config(
    page_title="COVID-19 Dashboard Docs", # Updated title for clarity
    page_icon="📄", # Documentation icon
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Data into Session State (Required for other pages) ---
def initialize_data():
    """Loads and preprocesses data, storing it in session state."""
    if not st.session_state.get('data_initialized', False):
        with st.spinner('Loading initial data for dashboard pages...'):
            # Use the updated load_data function from utils
            data = load_data() 
            if data.empty:
                st.error("Fatal Error: Failed to load base data. Dashboard pages cannot function.")
                st.stop()
            st.session_state.data = data
            st.session_state.data_initialized = True
            st.toast("Data loaded for pages.", icon="✅")
    elif 'data' not in st.session_state:
        st.error("Session state error: Data should be initialized but is missing. Please refresh.")
        st.stop()

initialize_data()

# --- Documentation Content (Copied from pages/0_📄_Documentation.py) ---

st.markdown("<div class='header-container'><h1>📄 Dashboard Documentation</h1></div>", unsafe_allow_html=True)

st.markdown("""
Welcome to the COVID-19 Data Analysis Dashboard! This application provides tools to explore, visualize, and analyze global COVID-19 trends using data compiled by Our World in Data (OWID).
""")

st.divider()

# --- Data Source ---
st.subheader("💾 Data Source")
st.markdown("""
*   **Source:** Our World in Data (OWID) COVID-19 Dataset.
*   **Update Frequency:** The dashboard automatically fetches the latest data directly from the [OWID GitHub repository](https://github.com/owid/covid-19-data/tree/master/public/data). Downloaded data is cached for 1 hour.
*   **Data Cutoff:** Analysis and visualizations are based on data available up to **December 31, 2023**.
*   **Processing:** The raw data undergoes cleaning, including date parsing, selection of relevant columns, filtering out aggregate regions, and calculation of 7-day rolling averages for new cases/deaths.
""")

st.divider()

# --- Features & Pages ---
st.subheader("✨ Features & Pages")
st.markdown("""
This dashboard is organized into several pages, each focusing on a different aspect of COVID-19 data analysis:

1.  **🌎 Global Overview:** 
    *   Provides a high-level snapshot of the global situation.
    *   Displays key cumulative metrics (Total Cases, Deaths, Active Cases, Fatality Rate).
    *   Features an interactive world map visualizing selected metrics (e.g., Cases per Million, Deaths per Million) for a chosen date.

2.  **📊 Country Detail:**
    *   Allows in-depth exploration of data for a single selected country.
    *   Shows key metrics and trends (cases, deaths) specific to the country.
    *   Includes interactive time series charts for new cases, deaths, and cumulative figures.
    *   Provides options to toggle logarithmic scale and view rolling averages.

3.  **🆚 Country Comparison:**
    *   Enables comparison of multiple countries side-by-side.
    *   Users can select several countries and a specific metric.
    *   Displays time series line charts comparing the chosen metric across the selected countries.
    *   Includes options to toggle logarithmic scale.

4.  **🌐 Regional Analysis (Clustering):**
    *   Groups countries based on similarities in selected COVID-19 metrics (e.g., cases per million, deaths per million, vaccination rates) using different clustering algorithms.
    *   **Algorithms:** K-Means, DBSCAN, Hierarchical Clustering.
    *   **Features:** Users can select features for clustering and adjust algorithm parameters (e.g., k for K-Means/Hierarchical, eps/min_samples for DBSCAN, linkage for Hierarchical).
    *   **Evaluation:** Includes Elbow Plot (K-Means), Silhouette Score plot (K-Means/Hierarchical), and Dendrogram (Hierarchical) to help assess cluster quality and choose parameters.
    *   **Visualization:** Displays results on an interactive world map and lists countries within each identified cluster/region.

5.  **💉 Vaccination Tracker:**
    *   Focuses on vaccination progress for a selected country.
    *   Visualizes key vaccination metrics (e.g., total vaccinations, people vaccinated/fully vaccinated per hundred, daily vaccinations) over time.
    *   Utilizes dual-axis charts where appropriate (e.g., comparing daily vaccinations to total vaccinations per hundred).

6.  **🏥🧪 Hospitalization & Testing:**
    *   Visualizes trends in hospital and ICU patient counts, as well as testing metrics (new tests, positive rate) for a selected country.
    *   *Note:* Data availability varies significantly by country.
    *   Includes dual-axis charts comparing related metrics (e.g., hospital patients vs. new cases).

7.  **📈 Time Series Forecasting:**
    *   Generates future projections for selected metrics (e.g., new cases, hospital patients) for a chosen country.
    *   Uses the ARIMA (AutoRegressive Integrated Moving Average) model.
    *   Users can select the metric and the forecast horizon (number of days ahead).
    *   Visualizes the historical data, the forecasted trend, and 95% confidence intervals.

8.  **❗ Anomaly Detection:**
    *   Identifies unusual spikes or dips in a selected time series metric for a chosen country.
    *   Uses a rolling statistics approach: points falling outside a specified number of standard deviations from the rolling mean are flagged.
    *   Users can adjust the rolling window size and the standard deviation threshold.
    *   Visualizes the time series with detected anomalies marked.

""")

st.divider()

# --- Usage ---
st.subheader("🖱️ Usage")
st.markdown("""
*   Use the sidebar on the left to navigate between pages and adjust filters (country selection, date ranges, metric choices, model parameters).
*   Charts are interactive: hover over points for details, zoom, pan, and use legend entries to toggle traces.
*   Data loading might take a moment on the first run or after the cache expires (1 hour).
""")

st.divider()
st.info("Developed using Streamlit, Pandas, Plotly, Statsmodels, and Scikit-learn.")

# Ensure data is loaded before other pages try to access it
# (This check might be redundant now as initialize_data handles stop/error)
# if 'data' not in st.session_state:
#     st.warning("Data initialization failed. Dashboard pages may not work correctly.")
#     st.stop() 