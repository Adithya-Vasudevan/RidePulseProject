from __future__ import annotations
import requests
import streamlit as st

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