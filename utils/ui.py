from __future__ import annotations
import requests
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None  # Optional dependency; handle gracefully


@st.cache_data(ttl=3600, show_spinner=False)
def _load_lottie(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None


def show_lottie(url: str, height: int = 120):
    data = _load_lottie(url)
    if data and st_lottie:
        st_lottie(data, height=height)
    else:
        # Silent no-op if the animation cannot be loaded or package not installed
        return


def set_mapbox_key_from_secrets():
    """Set pydeck Mapbox API key from st.secrets if available."""
    try:
        if "MAPBOX_API_KEY" in st.secrets:
            pdk.settings.MAPBOX_API_KEY = st.secrets["MAPBOX_API_KEY"]
    except Exception:
        # Secrets may not be available in all environments
        pass


def apply_theme(fig: go.Figure, a11y: bool = False) -> go.Figure:
    """
    Unified Plotly theming with optional accessibility boosts.
    
    Args:
        fig: Plotly figure to theme
        a11y: If True, apply accessibility enhancements (larger text, thicker lines)
    """
    try:
        dark = st.get_option("theme.base") == "dark"
    except Exception:
        dark = True
    
    font_color = "#C9D1D9" if dark else "#111827"
    grid_color = "#30363d" if dark else "#e5e7eb"
    
    # Base font size with accessibility boost
    base_font_size = 16 if a11y else 12
    title_font_size = 20 if a11y else 16
    
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=base_font_size),
        title_font=dict(size=title_font_size),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
    )
    
    # Accessibility enhancements
    if a11y:
        # Thicker lines for traces
        for trace in fig.data:
            if hasattr(trace, 'line') and trace.line:
                trace.line.width = max(trace.line.width or 2, 3)
            if hasattr(trace, 'marker') and trace.marker:
                # Larger markers
                if hasattr(trace.marker, 'size'):
                    trace.marker.size = max(trace.marker.size or 6, 8)
    
    return fig