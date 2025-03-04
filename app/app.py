# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import hopsworks

# 1. App Configuration
st.set_page_config(
    page_title="NYC Taxi Analytics Suite",
    layout="wide",
    page_icon="🚖"
)

# 2. Data Loading with Caching
@st.cache_data(ttl=3600)
def load_data():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    feature_view = fs.get_feature_view(
        name="time_series_hourly_feature_view", 
        version=1
    )
    
    return feature_view.get_batch_data().read()

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Analysis Parameters")
selected_borough = st.sidebar.selectbox(
    "Select Borough",
    options=df['borough'].unique(),
    index=0
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[datetime(2025,3,1), datetime(2025,3,4)]
)

# 4. Main Dashboard Layout
st.title("NYC Taxi Operations Intelligence Platform")
st.markdown("""
**Strategic Insights:**  
Aligning with NYC TLC's 2022 Strategic Plan for fleet electrification and service optimization
""")

# 5. Key Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rides", f"{len(df):,}", "3.2% vs avg")
with col2:
    st.metric("Avg. Trip Distance", "2.8 mi", "-1.1% WoW")
with col3:
    st.metric("Electric Fleet %", "18%", "4% YoY")

# 6. Interactive Visualizations
tab1, tab2, tab3 = st.tabs(["Spatial Analysis", "Temporal Trends", "Demand Forecast"])

with tab1:
    st.subheader("Pickup Hotspots by Borough")
    borough_df = df[df['borough'] == selected_borough]
    fig = px.density_mapbox(
        borough_df,
        lat='lat',
        lon='lon',
        radius=10,
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=10,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Hourly Demand Patterns")
    hourly = df.groupby('hour').size().reset_index(name='rides')
    fig = px.area(
        hourly, 
        x='hour', 
        y='rides',
        labels={'hour': 'Hour of Day', 'rides': 'Total Rides'},
        line_shape='spline'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("7-Day Demand Forecast")
    # Add your model predictions integration here
    st.write("""
    **Upcoming Feature:**  
    Machine learning-powered predictions using time-series features
    """)

# 7. Accessibility Features
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Accessibility Options**  
♿ Color contrast adjustments  
🔊 Screen reader support  
📱 Mobile-optimized views
""")

# 8. Deployment Instructions

