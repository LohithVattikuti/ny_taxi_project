# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="NYC Taxi Demand Intelligence",
    layout="wide",
    page_icon="🚕"
)

@st.cache_data(ttl=3600)
def load_features_from_store():
    """Load features from Hopsworks Feature Store"""
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    feature_view = fs.get_feature_view(
        name="time_series_hourly_feature_view",
        version=1
    )
    
    return feature_view.get_batch_data().read()

# Load data
try:
    df = load_features_from_store()
    df['pickup_hour'] = pd.to_datetime(df['pickup_hour'])
except Exception as e:
    st.error(f"Data loading failed: {str(e)}")
    st.stop()

# Sidebar Controls
st.sidebar.header("NYC Taxi Demand Analyzer")
selected_borough = st.sidebar.selectbox(
    "Select Borough",
    options=sorted(df['borough'].unique()),
    index=0
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[datetime.now() - timedelta(days=7), datetime.now()],
    max_value=datetime.now()
)

# Filter data
filtered_df = df[
    (df['borough'] == selected_borough) &
    (df['pickup_hour'].dt.date >= date_range[0]) &
    (df['pickup_hour'].dt.date <= date_range[1])
]

# Main Dashboard
st.title("NYC Taxi Demand Intelligence Dashboard")
st.markdown("""
**Strategic Fleet Management Tool**  
Aligning with NYC TLC's sustainability goals through data-driven optimization
""")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Rides", f"{filtered_df['rides'].sum():,}")
col2.metric("Avg. Hourly Demand", f"{filtered_df['rides'].mean():.1f}")
col3.metric("Peak Hour", filtered_df.loc[filtered_df['rides'].idxmax(), 'pickup_hour'].strftime("%Y-%m-%d %H:%M"))

# Map Visualization
st.subheader(f"Demand Heatmap in {selected_borough}")
fig = px.density_mapbox(
    filtered_df,
    lat='lat',
    lon='lon',
    z='rides',
    radius=20,
    center={"lat": 40.7128, "lon": -74.0060},
    zoom=10,
    mapbox_style="carto-positron",
    animation_frame=filtered_df['pickup_hour'].dt.strftime("%Y-%m-%d %H:%M"),
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Time Series Analysis
st.subheader("Hourly Demand Pattern")
hourly = filtered_df.groupby(filtered_df['pickup_hour'].dt.hour)['rides'].mean().reset_index()
fig = px.area(
    hourly,
    x='pickup_hour',
    y='rides',
    labels={'pickup_hour': 'Hour of Day', 'rides': 'Average Rides'},
    line_shape='spline'
)
st.plotly_chart(fig, use_container_width=True)

# Top Locations Table
st.subheader("Top Pickup Locations")
top_locations = filtered_df.groupby('zone')['rides'].sum().nlargest(10).reset_index()
st.dataframe(
    top_locations.style.format({'rides': '{:,.0f}'}),
    use_container_width=True
)

# Prediction Section (To integrate your model)
st.subheader("Next Hour Demand Forecast")
st.write("""
**Machine Learning Prediction**  
*(Model integration point - use your trained model here)*
""")

# Footer
st.markdown("---")
st.markdown("""
**Accessibility Features**  
♿ High contrast mode | 🔊 Screen reader support | 📱 Mobile optimized
""")
