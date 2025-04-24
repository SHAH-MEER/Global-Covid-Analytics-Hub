# COVID-19 Data Analysis Dashboard 📊

This project presents an interactive Streamlit web application designed for exploring, visualizing, and analyzing global COVID-19 data trends.

## Overview

The dashboard allows users to investigate various aspects of the pandemic through dedicated pages, offering insights into global patterns, country-specific details, comparisons, regional clustering, vaccination progress, healthcare system impacts, time series forecasting, and anomaly detection.

## Features ✨

The application is structured into several focused pages:

*   **📄 Documentation:** Provides an overview of the dashboard, data sources, features, and usage instructions (this page!).
*   **🌎 Global Overview:** Visualizes worldwide trends using an interactive choropleth map and key global metrics.
*   **📊 Country Detail:** Offers a deep dive into the statistics for a single selected country, including time series charts and key indicators.
*   **🆚 Country Comparison:** Enables side-by-side comparison of multiple selected countries across various metrics using line charts, bar charts, and more.
*   **🌐 Regional Analysis (Clustering):** Groups countries based on selected metrics using **K-Means, DBSCAN, or Hierarchical Clustering**. Includes interactive selection of features/parameters and visualizations like maps, Silhouette plots, and dendrograms.
*   **💉 Vaccination Tracker:** Focuses on vaccination progress for a selected country with time series and dual-axis charts.
*   **🏥🧪 Hospitalization & Testing:** Presents data on healthcare system load (hospital/ICU patients, admissions, testing metrics). *Note: Data availability varies significantly.* Includes dual-axis charts.
*   **📈 Time Series Forecasting:** Generates future projections for selected metrics (e.g., new cases) for a chosen country using an **ARIMA model**, displaying the forecast and confidence intervals.
*   **❗ Anomaly Detection:** Identifies unusual data points in a selected time series using a **rolling mean/standard deviation approach**, highlighting anomalies on the chart.

## Data Source 💾

The primary data source is the COVID-19 dataset maintained by **[Our World in Data (OWID)](https://github.com/owid/covid-19-data/tree/master/public/data)**.

*   **Data Fetching:** The dashboard **automatically fetches the latest data** directly from the OWID GitHub repository upon startup.
*   **Caching:** Downloaded data is cached for 1 hour to improve performance on subsequent loads.
*   **Data Cutoff:** Analysis and visualizations currently use data available up to **December 31, 2023** (defined in `utils.py`).

## Project Structure 📁

```
Covid/
├── .gitignore
├── app.py                  # Main application script (Documentation & Landing Page)
├── pages/                  # Directory for individual Streamlit pages
│   ├── 1_🌍_Global_Overview.py # Renamed from Global_View
│   ├── 2_📊_Country_Detail.py
│   ├── 3_🆚_Country_Comparison.py
│   ├── 4_🌐_Regional_Analysis.py # Renamed from Historical_Analysis & updated
│   ├── 5_💉_Vaccination_Tracker.py
│   ├── 6_🏥🧪_Hospitalization_Testing.py
│   ├── 7_📈_Time_Series_Forecasting.py # New
│   └── 8_❗_Anomaly_Detection.py      # New
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── utils.py                # Helper functions (data loading, plotting, etc.)
└── .venv/                  # Virtual environment directory (if created)
```
*(Note: Page filenames might differ slightly based on previous steps)*

## Setup and Installation ⚙️

1.  **Clone the repository (Optional):**
    ```bash
    git clone <repository-url>
    cd Covid
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv .venv 
    ```

3.  **Activate the Virtual Environment:**
    *   **macOS/Linux:** `source .venv/bin/activate`
    *   **Windows:** `.venv\Scripts\activate`

4.  **Install Dependencies:**
    Ensure your `requirements.txt` file includes at least the following:
    ```txt
    streamlit
    pandas
    numpy
    plotly
    statsmodels
    matplotlib
    requests
    scikit-learn
    ```
    Then, run the installation command:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application ▶️

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project's root directory (`Covid/`) in your terminal.**
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  **Open your web browser:** Streamlit will typically provide a local URL (like `http://localhost:8501`).

## Usage 🖱️

*   Navigate between the different analysis pages using the sidebar.
*   Each page may have specific filters available in the sidebar (e.g., country selection, date ranges, metrics, model parameters) to customize the displayed data and visualizations.
*   Interact with the charts (hover for details, zoom, pan, select date ranges).
*   Refer to the **📄 Documentation** page within the app for detailed feature descriptions. 