import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt  # For Elbow plot initially

st.set_page_config(layout="wide")
st.markdown("<div class='header-container'><h1>🌐 Regional Analysis (Clustering)</h1></div>", unsafe_allow_html=True)
st.caption("Grouping countries based on similarity in key COVID-19 metrics using K-Means clustering.")

# --- Helper Functions ---
def get_latest_data(df):
    """Gets the most recent row for each country."""
    return df.loc[df.groupby('country')['date'].idxmax()]

def assign_cluster_name(cluster_means, global_means):
    """Assigns a descriptive name based on cluster characteristics relative to global."""
    name_parts = []
    
    # Example Naming Logic (Customize based on features and desired granularity)
    cases_level = "Avg Cases"
    if cluster_means.get('total_cases_per_million', global_means.get('total_cases_per_million', 0)) > global_means.get('total_cases_per_million', 0) * 1.2:
        cases_level = "High Cases"
    elif cluster_means.get('total_cases_per_million', global_means.get('total_cases_per_million', 0)) < global_means.get('total_cases_per_million', 0) * 0.8:
        cases_level = "Low Cases"
    name_parts.append(cases_level)

    deaths_level = "Avg Deaths"
    if cluster_means.get('total_deaths_per_million', global_means.get('total_deaths_per_million', 0)) > global_means.get('total_deaths_per_million', 0) * 1.2:
        deaths_level = "High Deaths"
    elif cluster_means.get('total_deaths_per_million', global_means.get('total_deaths_per_million', 0)) < global_means.get('total_deaths_per_million', 0) * 0.8:
        deaths_level = "Low Deaths"
    name_parts.append(deaths_level)

    vax_level = "Avg Vax"
    if cluster_means.get('people_fully_vaccinated_per_hundred', global_means.get('people_fully_vaccinated_per_hundred', 0)) > global_means.get('people_fully_vaccinated_per_hundred', 0) * 1.1:
        vax_level = "High Vax"
    elif cluster_means.get('people_fully_vaccinated_per_hundred', global_means.get('people_fully_vaccinated_per_hundred', 0)) < global_means.get('people_fully_vaccinated_per_hundred', 0) * 0.9:
         vax_level = "Low Vax"
    # Only add vax level if data was available
    if 'people_fully_vaccinated_per_hundred' in cluster_means:
         name_parts.append(vax_level)

    return ", ".join(name_parts)

# --- Data Loading and Preparation ---
if 'data' not in st.session_state:
    st.warning("Data not loaded. Please return to the main page.")
    st.stop()

data = st.session_state.data
latest_data = get_latest_data(data)

# Define features for clustering
features_for_clustering = [
    'total_cases_per_million', 
    'total_deaths_per_million', 
    'people_fully_vaccinated_per_hundred',
    'positive_rate',
]

# Filter features that actually exist in the data and have *some* non-NaN values
available_features = [f for f in features_for_clustering 
                    if f in latest_data.columns and latest_data[f].notna().any()]

if not available_features:
    st.error("No suitable features found in the data for clustering.")
    st.stop()

st.sidebar.header("Clustering Options")

# --- Algorithm Selection ---
algorithm = st.sidebar.radio(
    "Select Clustering Algorithm:",
    ('K-Means', 'DBSCAN', 'Hierarchical'),
    key='cluster_algo'
)

# --- Feature Selection --- 
selected_features = st.sidebar.multiselect(
    "Select features for clustering:", 
    options=available_features, 
    default=[f for f in ['total_cases_per_million', 'total_deaths_per_million'] if f in available_features]
)

# --- Algorithm-Specific Parameters ---
k_selected = None
eps_selected = None
min_samples_selected = None
linkage_selected = None

if algorithm == 'K-Means' or algorithm == 'Hierarchical':
    st.sidebar.subheader("Number of Clusters (k)")
    k_selected = st.sidebar.slider("Select number of clusters (k):", min_value=2, max_value=15, value=6, key='k_clusters_kmeans_hier')
    if algorithm == 'Hierarchical':
        linkage_selected = st.sidebar.selectbox("Linkage Method:", ('ward', 'complete', 'average', 'single'), key='hier_linkage')
elif algorithm == 'DBSCAN':
    st.sidebar.subheader("DBSCAN Parameters")
    # These parameters are highly data-dependent. Good defaults are hard.
    # Providing sliders allows exploration, but might require tuning based on data.
    eps_selected = st.sidebar.slider("Epsilon (eps - max distance for neighborhood):", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key='dbscan_eps')
    min_samples_selected = st.sidebar.slider("Minimum Samples (min_samples in neighborhood):", min_value=2, max_value=10, value=5, key='dbscan_minsamp')

# --- Prepare Data for Clustering --- 
if not selected_features:
    st.warning("Please select at least one feature for clustering.")
    st.stop()

# Prepare data for clustering - Select only relevant columns first
cluster_data_prep = latest_data[selected_features + ['country', 'code']].copy()

# Store initial count of countries with codes
initial_country_count = latest_data['code'].notna().sum()

# Drop rows where *any* selected feature is NaN
cluster_data_prep.dropna(subset=selected_features, inplace=True)

# Calculate how many countries were dropped
final_country_count = len(cluster_data_prep)
excluded_count = initial_country_count - final_country_count

# Display info about excluded countries
if excluded_count > 0:
    st.info(f"ℹ️ {excluded_count} countries were excluded from clustering due to missing data in the selected features: {', '.join(selected_features)}.")

# Check if enough data remains *after* knowing k (if k is relevant)
if cluster_data_prep.empty:
    st.error("Not enough countries with complete data for all selected features after filtering.")
    st.stop()
# Only check k if it's relevant (i.e., K-Means or Hierarchical)
elif k_selected is not None and len(cluster_data_prep) < k_selected:
    st.error(f"Not enough countries ({len(cluster_data_prep)}) with complete data for all selected features to form {k_selected} clusters. Try selecting fewer features or reducing k.")
    st.stop()

# Check for zero variance after dropping NaNs (can happen if all remaining countries have the same value)
features_to_scale = []
for feature in selected_features:
    if cluster_data_prep[feature].nunique() > 1:
        features_to_scale.append(feature)
    else:
        st.warning(f"Feature '{feature}' has zero variance after filtering and will be excluded from scaling/clustering.")

if not features_to_scale:
    st.error("No features with variance remaining after filtering. Cannot perform clustering.")
    st.stop()

# Scale the valid features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_data_prep[features_to_scale])

# --- Elbow Plot & Silhouette Score (Only relevant for K-Means/Hierarchical) ---
if algorithm == 'K-Means' or algorithm == 'Hierarchical':
    st.sidebar.subheader("Optimize Number of Clusters (k)")
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    with st.spinner("Calculating Elbow curve & Silhouette scores..."):
        for k in k_range:
            if algorithm == 'K-Means':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else: # Hierarchical
                # Cannot calculate inertia directly for AgglomerativeClustering
                # Silhouette score is more appropriate here
                 model = AgglomerativeClustering(n_clusters=k, linkage=linkage_selected)
            
            labels = model.fit_predict(scaled_features)
            if algorithm == 'K-Means':
                 inertia.append(model.inertia_)
            # Calculate silhouette score only if more than 1 cluster label exists
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(scaled_features, labels))
            else:
                 silhouette_scores.append(-1) # Indicate invalid score

    # Plot Elbow curve in sidebar expander (only if K-Means)
    if algorithm == 'K-Means':
        with st.sidebar.expander("View Elbow Plot"):
            fig_elbow, ax_elbow = plt.subplots(figsize=(5, 2.5))
            ax_elbow.plot(k_range, inertia, marker='o')
            ax_elbow.set_xlabel("Number of Clusters (k)")
            ax_elbow.set_ylabel("Inertia (WCSS)")
            ax_elbow.set_title("Elbow Method for Optimal k")
            ax_elbow.grid(True)
            st.pyplot(fig_elbow)
            st.caption("Look for the 'elbow' point where the rate of decrease sharply slows down.")

    # Plot Silhouette scores in sidebar expander
    with st.sidebar.expander("View Silhouette Scores"):
        fig_sil, ax_sil = plt.subplots(figsize=(5, 2.5))
        valid_k = [k for k, score in zip(k_range, silhouette_scores) if score > -1]
        valid_scores = [score for score in silhouette_scores if score > -1]
        if valid_scores:
            ax_sil.plot(valid_k, valid_scores, marker='o')
            ax_sil.set_xlabel("Number of Clusters (k)")
            ax_sil.set_ylabel("Silhouette Score")
            ax_sil.set_title("Silhouette Score per k")
            ax_sil.grid(True)
            st.pyplot(fig_sil)
            st.caption("Higher scores (closer to 1) indicate better-defined clusters. Peaks suggest optimal k.")
        else:
             st.write("Could not calculate Silhouette scores.")
             
    st.sidebar.markdown(f"**Selected k = {k_selected}**")

# --- Apply Clustering ---
cluster_labels = None
model = None
cluster_data = cluster_data_prep.copy() # Use the filtered data

with st.spinner(f"Applying {algorithm}..."):
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(scaled_features)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=eps_selected, min_samples=min_samples_selected)
        cluster_labels = model.fit_predict(scaled_features)
        # DBSCAN uses -1 for noise points
        n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        st.subheader(f"DBSCAN Results (eps={eps_selected}, min_samples={min_samples_selected})")
        st.write(f"Found {n_clusters_found} clusters and {list(cluster_labels).count(-1)} noise points.")
    elif algorithm == 'Hierarchical':
         model = AgglomerativeClustering(n_clusters=k_selected, linkage=linkage_selected)
         cluster_labels = model.fit_predict(scaled_features)
         # Plot Dendrogram
         st.subheader("Hierarchical Clustering Dendrogram")
         try:
             linked_matrix = linkage(scaled_features, method=linkage_selected)
             fig_dendro, ax_dendro = plt.subplots(figsize=(12, 5))
             dendrogram(linked_matrix, 
                        labels=cluster_data['country'].tolist(),
                        leaf_rotation=90.,
                        leaf_font_size=7.,
                        ax=ax_dendro,
                        color_threshold=0 # Optional: Color by distance
                       )
             plt.tight_layout()
             st.pyplot(fig_dendro)
         except Exception as e:
             st.error(f"Could not generate dendrogram: {e}")

# Add cluster labels back to the prepared (filtered) data
cluster_data['Cluster'] = cluster_labels

# --- Analyze and Name Clusters ---
st.subheader("Cluster Characteristics")

# Exclude noise points (-1) from mean calculation and naming
valid_clusters = cluster_data[cluster_data['Cluster'] != -1]
if valid_clusters.empty:
    st.warning("No valid clusters found (only noise points with DBSCAN). Cannot analyze characteristics.")
    # Skip subsequent steps that rely on valid clusters
    # ... (consider adding st.stop() or conditional rendering)
else:
    cluster_means = valid_clusters.groupby('Cluster')[features_to_scale].mean()
    global_mean_features = [f for f in selected_features if f in latest_data.columns]
    global_means = latest_data[global_mean_features].median()
    
    cluster_ids = sorted(valid_clusters['Cluster'].unique())
    cluster_names = {i: f"Region {i+1}: {assign_cluster_name(cluster_means.loc[i], global_means)}" for i in cluster_ids}
    
    # Add a name for noise points if DBSCAN was used
    if algorithm == 'DBSCAN':
        cluster_names[-1] = "Noise / Outliers"
        # Map names, filling NaN for noise points which will be handled by mapping
        cluster_data['Cluster Name'] = cluster_data['Cluster'].map(cluster_names)
    else:
        cluster_data['Cluster Name'] = cluster_data['Cluster'].map(cluster_names)

    # Display cluster means and names (only for non-noise clusters)
    cluster_summary = cluster_means.copy()
    cluster_summary['Assigned Name'] = cluster_summary.index.map(cluster_names)
    cluster_summary.index.name = "Cluster ID"
    # Apply formatting only to the numeric feature columns
    st.dataframe(cluster_summary.style.format("{:.2f}", subset=features_to_scale))
    st.caption("Showing the average values of the features used for clustering.")

    # --- Visualize Clusters --- (Ensure this uses cluster_data which has Cluster Name)
    st.subheader("Cluster Visualization")
    # World Map colored by Cluster Name
    try:
        plot_df_map = cluster_data.copy()
        map_color_map = {name: color for name, color in zip(sorted(cluster_names.values()), px.colors.qualitative.Vivid)} 
        if algorithm == 'DBSCAN':
             map_color_map["Noise / Outliers"] = '#808080' # Grey for noise
             
        map_fig = px.choropleth(
            plot_df_map.dropna(subset=['code', 'Cluster Name']),
            locations="code",
            color="Cluster Name",
            hover_name="country",
            hover_data={'Cluster Name': True, 'code': False},
            title=f"World Map Colored by {algorithm} Regions",
            color_discrete_map=map_color_map,
            category_orders={"Cluster Name": sorted(cluster_names.values())}
        )
        map_fig.update_layout(
            legend_title_text='Cluster Region Name',
            geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth')
        )
        st.plotly_chart(map_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate cluster map: {e}")

    # Scatter plot
    # ... (Scatter plot code needs similar color map handling)

    # --- List Countries per Cluster ---
    st.subheader("Countries per Cluster")
    all_cluster_ids = sorted(cluster_data['Cluster'].unique())
    for i in all_cluster_ids:
        cluster_name = cluster_names[i]
        countries_in_cluster = cluster_data[cluster_data['Cluster'] == i]['country'].sort_values().tolist()
        if countries_in_cluster:
            with st.expander(f"{cluster_name} ({len(countries_in_cluster)} countries)"):
                st.write(", ".join(countries_in_cluster)) 