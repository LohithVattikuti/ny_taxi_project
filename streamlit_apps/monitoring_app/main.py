import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import hopsworks
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import src.config as config

def main():
    st.set_page_config(
        page_title="Model Monitoring Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Model Performance Monitoring")
    
    try:
        # Login to Hopsworks
        project = hopsworks.login(
            api_key_value=os.getenv('HOPSWORKS_API_KEY')
        )
        # Rest of your monitoring app code...
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()