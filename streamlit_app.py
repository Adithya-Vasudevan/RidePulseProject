"""
RidePulse - Bike Sharing Analytics and Prediction
Production-ready Streamlit app for NYC bike sharing analytics
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="RidePulse",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import project modules
try:
    from utils.gbfs import merged_station_frame
except ImportError as e:
    st.error(f"Failed to import project modules: {e}")
    st.stop()

# Theme helper (simplified version from app.py)
def _is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True


@st.cache_resource
def load_model() -> Dict[str, Any]:
    """
    Load and initialize the bike sharing prediction models.
    Returns a dictionary containing trained models and metadata.
    """
    try:
        # Load live GBFS data for training
        df_raw = merged_station_frame()
        if df_raw is None or len(df_raw) == 0:
            # Use fallback with rule-based model when live data unavailable
            return {
                "status": "fallback",
                "classifier": None,
                "classifier_name": "Rule-based",
                "kmeans": None,
                "cluster_scaler": None,
                "features": [
                    "capacity", "bikes", "docks", "percent_full", "bikes_per_cap", 
                    "docks_per_cap", "hour", "is_peak", "nn_percent_full_mean", "nn_capacity_mean"
                ],
                "thresholds": {"high": 90, "low": 10},
                "error": "Live GBFS data unavailable - using rule-based fallback"
            }
        
        # Prepare features using the same logic as Models Lab
        df = _ensure_metrics(df_raw)
        feat = _feature_frame(df)
        
        # Prepare classification features
        clf_X_cols = [
            "capacity", "bikes", "docks", "percent_full", "bikes_per_cap", 
            "docks_per_cap", "hour", "is_peak", "nn_percent_full_mean", "nn_capacity_mean"
        ]
        
        # Check if we have required columns
        missing_cols = [col for col in clf_X_cols if col not in feat.columns]
        if missing_cols:
            return {
                "status": "error",
                "error": f"Missing required columns: {missing_cols}",
                "classifier": None,
                "kmeans": None,
                "features": []
            }
        
        X = feat[clf_X_cols].fillna(0.0).to_numpy()
        
        # Create binary labels based on thresholds (same as Models Lab)
        high_thr = 90
        low_thr = 10
        y = ((feat["percent_full"] >= high_thr) | (feat["percent_full"] <= low_thr)).astype(int)
        
        # Train classifier if scikit-learn is available
        classifier = None
        classifier_name = "Rule-based"
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            if len(np.unique(y)) > 1:  # Need both classes for training
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    class_weight="balanced"
                )
                classifier.fit(X_train, y_train)
                classifier_name = "RandomForest"
            
        except ImportError:
            pass  # Fall back to rule-based
        except Exception as e:
            st.warning(f"Model training failed: {e}. Using rule-based fallback.")
        
        # Train clustering model
        kmeans = None
        cluster_scaler = None
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            cluster_cols = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
            cluster_data = feat[cluster_cols].fillna(0.0).to_numpy()
            
            cluster_scaler = StandardScaler()
            cluster_data_scaled = cluster_scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            kmeans.fit(cluster_data_scaled)
            
        except ImportError:
            pass
        except Exception as e:
            st.warning(f"Clustering model failed: {e}")
        
        return {
            "status": "success",
            "classifier": classifier,
            "classifier_name": classifier_name,
            "kmeans": kmeans,
            "cluster_scaler": cluster_scaler,
            "features": clf_X_cols,
            "feature_data": feat,
            "thresholds": {"high": high_thr, "low": low_thr}
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "classifier": None,
            "kmeans": None,
            "features": []
        }


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metrics to the station dataframe."""
    df = df.copy()
    
    # Ensure numeric types
    for col in ["num_bikes_available", "num_docks_available", "capacity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Add derived columns with safe division
    df["bikes"] = df.get("num_bikes_available", 0)
    df["docks"] = df.get("num_docks_available", 0)
    df["capacity"] = df.get("capacity", df["bikes"] + df["docks"])
    
    # Avoid division by zero
    capacity_safe = df["capacity"].replace(0, 1)
    df["percent_full"] = (df["bikes"] / capacity_safe * 100).round(1)
    df["bikes_per_cap"] = (df["bikes"] / capacity_safe).round(3)
    df["docks_per_cap"] = (df["docks"] / capacity_safe).round(3)
    
    return df


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Generate feature frame with time and neighbor features."""
    f = df.copy()
    
    # Time features
    try:
        f["_ts"] = pd.to_datetime(f.get("last_reported", pd.Timestamp.utcnow()), unit="s", errors="coerce")
    except Exception:
        f["_ts"] = pd.Timestamp.utcnow()
    
    f["hour"] = f["_ts"].dt.hour.fillna(12).astype(int)
    f["is_peak"] = f["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    f.drop(columns=["_ts"], inplace=True, errors="ignore")
    
    # Neighbor features (simplified)
    if {"lat", "lon"}.issubset(f.columns):
        try:
            from sklearn.neighbors import NearestNeighbors
            
            coords = f[["lat", "lon"]].to_numpy()
            k = min(6, len(f))
            if k > 1:
                nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(coords)
                distances, indices = nbrs.kneighbors(coords)
                
                pf = f["percent_full"].to_numpy()
                cap = f["capacity"].to_numpy()
                
                nn_pf_mean = []
                nn_cap_mean = []
                
                for row_idx in range(indices.shape[0]):
                    neighbor_idxs = indices[row_idx][1:] if indices.shape[1] > 1 else []
                    if len(neighbor_idxs) > 0:
                        nn_pf_mean.append(np.nanmean(pf[neighbor_idxs]))
                        nn_cap_mean.append(np.nanmean(cap[neighbor_idxs]))
                    else:
                        nn_pf_mean.append(f["percent_full"].mean())
                        nn_cap_mean.append(f["capacity"].mean())
                
                f["nn_percent_full_mean"] = pd.Series(nn_pf_mean, index=f.index).fillna(f["percent_full"].mean())
                f["nn_capacity_mean"] = pd.Series(nn_cap_mean, index=f.index).fillna(f["capacity"].mean())
            else:
                f["nn_percent_full_mean"] = f["percent_full"].mean()
                f["nn_capacity_mean"] = f["capacity"].mean()
        except Exception:
            f["nn_percent_full_mean"] = f["percent_full"].mean()
            f["nn_capacity_mean"] = f["capacity"].mean()
    else:
        f["nn_percent_full_mean"] = f["percent_full"].mean()
        f["nn_capacity_mean"] = f["capacity"].mean()
    
    return f


@st.cache_data
def predict_with_model(model_dict: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
    """
    Make predictions using the loaded model.
    
    Args:
        model_dict: Dictionary containing model and metadata
        features: Dictionary of feature values
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        if model_dict["status"] == "error":
            return {
                "prediction": "Error",
                "confidence": 0.0,
                "explanation": f"Model error: {model_dict.get('error', 'Unknown error')}",
                "cluster": None
            }
        
        # Build feature vector
        clf_X_cols = model_dict["features"]
        feature_vector = np.array([features.get(col, 0.0) for col in clf_X_cols]).reshape(1, -1)
        
        # Make classification prediction
        classifier = model_dict["classifier"]
        if classifier is not None:
            try:
                prediction_proba = classifier.predict_proba(feature_vector)[0, 1]
                prediction = "Needs Rebalancing" if prediction_proba >= 0.5 else "OK"
                confidence = prediction_proba * 100
            except Exception:
                prediction_val = classifier.predict(feature_vector)[0]
                prediction = "Needs Rebalancing" if prediction_val == 1 else "OK"
                confidence = float(prediction_val) * 100
        else:
            # Rule-based fallback
            percent_full = features.get("percent_full", 50)
            thresholds = model_dict.get("thresholds", {"high": 90, "low": 10})
            
            if percent_full >= thresholds["high"] or percent_full <= thresholds["low"]:
                prediction = "Needs Rebalancing"
                confidence = 90.0
            else:
                prediction = "OK"
                confidence = 60.0
        
        # Make cluster prediction
        cluster_assignment = None
        kmeans = model_dict.get("kmeans")
        cluster_scaler = model_dict.get("cluster_scaler")
        
        if kmeans is not None and cluster_scaler is not None:
            try:
                cluster_cols = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
                cluster_features = np.array([features.get(col, 0.0) for col in cluster_cols]).reshape(1, -1)
                cluster_features_scaled = cluster_scaler.transform(cluster_features)
                cluster_assignment = int(kmeans.predict(cluster_features_scaled)[0])
            except Exception as e:
                cluster_assignment = f"Error: {str(e)}"
        
        explanation = f"Based on {len(clf_X_cols)} features including station capacity, current fill level, time of day, and neighborhood context."
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "cluster": cluster_assignment,
            "model_type": model_dict.get("classifier_name", "Rule-based")
        }
        
    except Exception as e:
        return {
            "prediction": "Error",
            "confidence": 0.0,
            "explanation": f"Prediction failed: {str(e)}",
            "cluster": None
        }


def main():
    """Main application interface."""
    st.title("🚕 RidePulse - Bike Sharing Analytics")
    st.caption("Real-time bike sharing station analysis and rebalancing predictions for NYC Citi Bike")
    
    # Load model
    try:
        model_dict = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    # Show model status
    with st.expander("Model Status", expanded=False):
        if model_dict["status"] == "success":
            st.success(f"✅ Models loaded successfully")
            st.write(f"**Classifier**: {model_dict.get('classifier_name', 'Unknown')}")
            st.write(f"**Features**: {len(model_dict.get('features', []))} features")
            st.write(f"**Clustering**: {'Available' if model_dict.get('kmeans') else 'Not available'}")
        elif model_dict["status"] == "fallback":
            st.warning("⚠️ Using fallback model - live data unavailable")
        else:
            st.error(f"❌ Model loading failed: {model_dict.get('error', 'Unknown error')}")
    
    # Sidebar input interface
    with st.sidebar:
        st.header("Station Parameters")
        st.caption("Enter station characteristics for rebalancing prediction")
        
        # Basic station info
        st.subheader("Station Basics")
        capacity = st.number_input(
            "Station Capacity", 
            min_value=1, max_value=100, value=20,
            help="Total number of docking points at this station"
        )
        
        bikes_available = st.number_input(
            "Bikes Available", 
            min_value=0, max_value=capacity, value=10,
            help="Current number of bikes at the station"
        )
        
        # Auto-calculate docks
        docks_available = max(0, capacity - bikes_available)
        st.write(f"**Docks Available**: {docks_available} (auto-calculated)")
        
        # Time context
        st.subheader("Time Context")
        current_hour = st.slider(
            "Current Hour", 
            min_value=0, max_value=23, value=12,
            help="Hour of day (0-23, 24-hour format)"
        )
        
        is_peak = st.checkbox(
            "Peak Hour", 
            value=current_hour in [7, 8, 9, 16, 17, 18],
            help="Check if this is a peak commuting hour"
        )
        
        # Neighborhood context
        st.subheader("Neighborhood Context")
        neighbor_fill = st.slider(
            "Neighbor Fill Level (%)",
            min_value=0, max_value=100, value=50,
            help="Average fill percentage of nearby stations"
        )
        
        neighbor_capacity = st.number_input(
            "Neighbor Capacity (avg)",
            min_value=1, max_value=100, value=25,
            help="Average capacity of nearby stations"
        )
        
        # TODO: Add more specific inputs based on actual model requirements
        # This is a placeholder section - replace with actual model inputs
        with st.expander("Advanced Parameters (TODO)", expanded=False):
            st.info("🚧 TODO: Add specific model inputs based on feature analysis")
            st.write("Future inputs might include:")
            st.write("- Weather conditions")
            st.write("- Event indicators") 
            st.write("- Historical patterns")
            st.write("- Station type classification")
    
    # Main prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction Input")
        
        # Calculate derived features
        percent_full = (bikes_available / capacity * 100) if capacity > 0 else 0
        bikes_per_cap = bikes_available / capacity if capacity > 0 else 0
        docks_per_cap = docks_available / capacity if capacity > 0 else 0
        
        # Show current input as JSON
        features = {
            "capacity": float(capacity),
            "bikes": float(bikes_available),
            "docks": float(docks_available),
            "percent_full": percent_full,
            "bikes_per_cap": bikes_per_cap,
            "docks_per_cap": docks_per_cap,
            "hour": float(current_hour),
            "is_peak": float(is_peak),
            "nn_percent_full_mean": float(neighbor_fill),
            "nn_capacity_mean": float(neighbor_capacity)
        }
        
        st.write("**Current Input Features:**")
        st.json(features)
        
        # Prediction button
        if st.button("🔮 Run Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    result = predict_with_model(model_dict, features)
                    
                    # Show results
                    st.subheader("Prediction Results")
                    
                    # Main prediction
                    if result["prediction"] == "Needs Rebalancing":
                        st.error(f"🚨 **{result['prediction']}**")
                    elif result["prediction"] == "OK":
                        st.success(f"✅ **{result['prediction']}**")
                    else:
                        st.warning(f"⚠️ **{result['prediction']}**")
                    
                    # Confidence and details
                    st.write(f"**Confidence**: {result['confidence']:.1f}%")
                    st.write(f"**Model Type**: {result.get('model_type', 'Unknown')}")
                    
                    if result.get('cluster') is not None:
                        st.write(f"**Station Cluster**: {result['cluster']}")
                    
                    st.write(f"**Explanation**: {result['explanation']}")
                    
                except Exception as e:
                    st.error("Prediction failed!")
                    with st.expander("Diagnostics", expanded=False):
                        st.code(traceback.format_exc())
    
    with col2:
        st.subheader("Station Overview")
        
        # Visual indicators
        st.metric("Fill Percentage", f"{percent_full:.1f}%")
        st.metric("Bikes/Capacity Ratio", f"{bikes_per_cap:.2f}")
        st.metric("Docks/Capacity Ratio", f"{docks_per_cap:.2f}")
        
        # Status indicators
        if percent_full >= 90:
            st.error("🔴 Station Nearly Full")
        elif percent_full <= 10:
            st.error("🔴 Station Nearly Empty")
        elif percent_full >= 70:
            st.warning("🟡 Station Getting Full")
        elif percent_full <= 30:
            st.warning("🟡 Station Getting Empty")
        else:
            st.success("🟢 Station Balanced")
        
        if is_peak:
            st.info("⏰ Peak Hours")
        else:
            st.info("🕐 Off-Peak Hours")
    
    # Footer information
    st.markdown("---")
    with st.expander("About This App", expanded=False):
        st.write("""
        **RidePulse** analyzes NYC Citi Bike sharing data to predict when stations need rebalancing.
        
        **Features:**
        - Real-time station status analysis
        - Machine learning-based predictions
        - Neighborhood context awareness
        - Time-based pattern recognition
        
        **Models Used:**
        - Random Forest Classifier for rebalancing predictions
        - K-Means clustering for station grouping
        - Rule-based fallbacks when ML models unavailable
        
        **Data Source**: Live NYC Citi Bike GBFS (General Bikeshare Feed Specification) data
        """)


if __name__ == "__main__":
    main()