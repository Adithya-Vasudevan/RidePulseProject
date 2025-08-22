from __future__ import annotations

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RidePulseProject", 
    page_icon="🚲", 
    layout="wide"
)

# Main Home page
st.title("🚲 RidePulseProject")
st.markdown("""
Welcome to **RidePulseProject** — a comprehensive bikeshare analytics platform built with Streamlit.

### About This Project

This application provides interactive tools for analyzing bikeshare data, featuring real-time GBFS (General Bikeshare Feed Specification) integration and purpose-built models for operational insights.

### Features

Navigate to the **Models Lab** page in the sidebar to explore our advanced analytics tools, or explore other available pages.
""")

# Data readiness check
st.subheader("📊 Data Readiness Status")

try:
    from utils.gbfs import merged_station_frame
    
    # Test data availability
    with st.spinner("Checking data connection..."):
        df = merged_station_frame()
        if df is not None and len(df) > 0:
            st.success(f"✅ Live GBFS data available ({len(df)} stations)")
            st.info("The Models Lab and other pages can access real-time bikeshare station data.")
        else:
            st.warning("⚠️ Live GBFS data not currently available")
            st.info("The Models Lab page may handle data loading on its own or use cached data.")
except Exception as e:
    st.warning("⚠️ Data connection could not be established")
    st.info("The Models Lab page may handle data loading on its own or use alternative data sources.")

# Analytics capabilities overview
st.subheader("🧪 Analytics Capabilities")

st.markdown("""
This platform includes several purpose-built analytics components:

**🎯 Rebalancing Classifier**  
Identifies stations that need bike redistribution based on capacity, fill levels, and usage patterns.

**🗺️ Station Clustering**  
Groups stations by similar characteristics using KMeans clustering for planning and deployment insights.

**🔍 Anomaly Detection**  
Detects unusual patterns in station usage and availability to identify operational issues.

**🎮 What‑If Simulator**  
Interactive simulation tools for exploring different operational scenarios and their impacts.

All models run on live GBFS data when available, with real-time analysis capabilities.
""")

# Getting started
st.subheader("🚀 Getting Started")

st.markdown("""
1. **Explore the Models Lab**: Use the sidebar to navigate to the Models Lab page for interactive analytics
2. **Check other pages**: Browse additional analysis tools and visualizations in the sidebar
3. **Real-time data**: All analysis uses live GBFS data when available

The sidebar navigation will show all available pages automatically.
""")

# Footer info
st.markdown("---")
st.caption("""
RidePulseProject • Built with Streamlit • Uses live GBFS data from Citi Bike NYC
""")