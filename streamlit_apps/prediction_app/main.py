import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import hopsworks
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from branca.colormap import LinearColormap
import requests
import zipfile

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(parent_dir)

from src.inference import get_feature_store
import src.config as config

# Load environment variables
load_dotenv()

# Initialize session state
if "map_created" not in st.session_state:
    st.session_state.map_created = False

def load_shape_data_file(data_dir):
    """Downloads and loads NYC taxi zone shapefile"""
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    data_dir = Path(data_dir)
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"
    
    if not shapefile_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            with st.spinner("Downloading taxi zone data..."):
                response = requests.get(url)
                with open(zip_path, "wb") as f:
                    f.write(response.content)
        
        with st.spinner("Extracting files..."):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
    
    return gpd.read_file(shapefile_path).to_crs("epsg:4326")

def create_taxi_map(shapefile_path, prediction_data):
    """Creates interactive choropleth map"""
    nyc_zones = gpd.read_file(shapefile_path)
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left"
    )
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)
    nyc_zones = nyc_zones.to_crs(epsg=4326)
    
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    
    # Create color scale
    colormap = LinearColormap(
        colors=['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026'],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )
    colormap.add_to(m)
    
    # Add choropleth layer
    folium.GeoJson(
        nyc_zones.to_json(),
        style_function=lambda x: {
            'fillColor': colormap(x['properties']['predicted_demand']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['zone', 'predicted_demand'],
            aliases=['Zone:', 'Predicted Demand:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(m)
    
    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m

def main():
    # Page config
    st.set_page_config(page_title="NYC Taxi Demand", layout="wide", page_icon="🚕")
    
    # Sidebar progress
    progress_bar = st.sidebar.header("Progress")
    progress = st.sidebar.progress(0)
    N_STEPS = 4
    
    try:
        # Step 1: Load map data
        with st.spinner("Loading map data..."):
            geo_df = load_shape_data_file(config.DATA_DIR)
            progress.progress(1/N_STEPS)
        
        # Step 2: Get predictions
        with st.spinner("Fetching predictions..."):
            project = hopsworks.login(api_key_value=os.getenv('HOPSWORKS_API_KEY'))
            fs = project.get_feature_store()
            fg = fs.get_feature_group(
                name='taxi_hourly_model_prediction',
                version=1
            )
            predictions = fg.read()
            latest_ts = predictions["pickup_hour"].max()
            current_predictions = predictions[predictions["pickup_hour"] == latest_ts]
            progress.progress(2/N_STEPS)
        
        # Step 3: Create visualizations
        st.title("🚕 NYC Taxi Demand Prediction")
        st.write(f"Predictions for: {latest_ts}")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.subheader("📊 Demand Statistics")
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric("Average", f"{current_predictions['predicted_demand'].mean():.0f}")
            with metric2:
                st.metric("Maximum", f"{current_predictions['predicted_demand'].max():.0f}")
            with metric3:
                st.metric("Locations", len(current_predictions))
            
            st.subheader("🏆 Top Demand Zones")
            top_locations = current_predictions.nlargest(10, "predicted_demand")[
                ["pickup_location_id", "predicted_demand"]
            ].reset_index(drop=True)
            st.dataframe(top_locations, use_container_width=True)
        
        with col2:
            st.subheader("🗺️ Demand Heatmap")
            shapefile_path = config.DATA_DIR / "taxi_zones" / "taxi_zones.shp"
            map_obj = create_taxi_map(shapefile_path, current_predictions)
            if st.session_state.map_created:
                st_folium(st.session_state.map_obj, width=800, height=600)
        
        progress.progress(3/N_STEPS)
        
        # Step 4: Time series plots
        st.subheader("📈 Demand Trends")
        for location_id in top_locations['pickup_location_id'][:5]:
            location_data = predictions[predictions['pickup_location_id'] == location_id]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=location_data['pickup_hour'],
                y=location_data['predicted_demand'],
                mode='lines+markers',
                name=f'Location {location_id}'
            ))
            fig.update_layout(
                title=f"Demand Trend for Location {location_id}",
                xaxis_title="Time",
                yaxis_title="Predicted Demand"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        progress.progress(4/N_STEPS)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Data updates hourly • Last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()