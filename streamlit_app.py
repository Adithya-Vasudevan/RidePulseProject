"""
RidePulseProject Streamlit Application
Main entry point for the multipage Streamlit app.
"""

import streamlit as st

# Configure the page
st.set_page_config(
    page_title="RidePulseProject", 
    page_icon="🚲", 
    layout="wide"
)

# Title and project overview
st.title("🚲 RidePulseProject")
st.write("**Bikeshare data analytics and modeling platform**")

st.markdown("""
Welcome to RidePulseProject! This application provides comprehensive analytics and machine learning models 
for bikeshare operations, built on live GBFS (General Bikeshare Feed Specification) data.

### Features

Explore the following capabilities using the pages in the sidebar:

- **🧪 Models Lab** - Interactive machine learning models for bikeshare operations
- **📈 Trends** - Historical usage patterns and trend analysis  
- **📍 Stations** - Station-level analytics and insights
- **🗺️ Live Map** - Real-time station status visualization
- **🔍 Fun Facts** - Discover interesting bikeshare insights
- **🧠 Quiz** - Test your bikeshare knowledge
- **📖 Story Builder** - Create data-driven narratives

""")

# Data readiness check
st.subheader("📊 Data Status")

try:
    from utils.gbfs import merged_station_frame
    
    with st.spinner("Checking data availability..."):
        df = merged_station_frame()
        if df is not None and len(df) > 0:
            st.success(f"✅ Live GBFS data available ({len(df)} stations)")
            st.info("All features are fully operational!")
        else:
            st.warning("⚠️ Live GBFS data not available")
            st.info("Some features may work with cached data or alternative data sources.")
            
except ImportError as e:
    st.warning("⚠️ Data utilities not fully available")
    st.info("The Models Lab page may handle data loading independently.")
except Exception as e:
    st.warning("⚠️ Unable to connect to live data sources")
    st.info("The Models Lab page may handle data loading on its own.")

# System status overview
st.subheader("🚀 Available Models & Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🤖 Rebalancing Classifier**
    - Identifies stations needing bike redistribution
    - Uses machine learning to predict rebalancing needs
    - Helps optimize fleet management
    
    **🎯 Station Clustering**
    - Groups stations by similar characteristics
    - Uses KMeans clustering on capacity and usage patterns
    - Useful for planning and deployment strategies
    """)

with col2:
    st.markdown("""
    **⚠️ Anomaly Detection**
    - Identifies unusual station behavior
    - Detects outliers in capacity vs fill patterns
    - Helps spot operational issues
    
    **🎛️ What‑If Simulator**
    - Interactive scenario modeling
    - Simulate changes to station parameters
    - Explore impact of operational decisions
    """)

# Navigation hint
st.markdown("---")
st.markdown("👈 **Use the sidebar to navigate between different pages and explore the full capabilities of RidePulseProject!**")

# Footer
st.markdown("---")
st.caption("RidePulseProject - Real-time bikeshare analytics and modeling platform")