from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="RidePulse • Overview",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Theme helpers ----------------
def _is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True

def _apply_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    dark = _is_dark()
    font_color = "#C9D1D9" if dark else "#111827"
    grid_color = "#30363d" if dark else "#e5e7eb"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        margin=dict(l=8, r=8, t=24, b=8),
        height=height or fig.layout.height,
    )
    return fig

# ---------------- Lightweight Lottie embed (no extra deps) ----------------
def lottie(url: str, height: int = 260):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{url}" background="transparent" speed="1"
        style="width:100%;height:{height}px" loop autoplay></lottie-player>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Global styles ----------------
dark = _is_dark()
primary = "#10b981"  # emerald
accent = "#60a5fa"   # blue
hero_grad = "linear-gradient(90deg, rgba(16,185,129,0.10) 0%, rgba(96,165,250,0.10) 100%)" if not dark \
            else "linear-gradient(90deg, rgba(16,185,129,0.18) 0%, rgba(96,165,250,0.18) 100%)"

st.markdown(
    f"""
    <style>
    .hero {{
        background: {hero_grad};
        border: 1px solid rgba(128,128,128,0.15);
        border-radius: 14px;
        padding: 22px 22px;
        margin-bottom: 14px;
        animation: fadeIn 0.6s ease-out both;
    }}
    .fadein {{
        animation: fadeIn 0.6s ease-out both;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(4px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .cta-btn {{
        display:inline-block;
        padding:10px 14px;
        border-radius:10px;
        border:1px solid rgba(128,128,128,0.25);
        text-decoration:none;
        margin-right:8px;
        margin-top:8px;
        background: rgba(255,255,255,0.04);
    }}
    .card {{
        border:1px solid rgba(128,128,128,0.2);
        border-radius:12px;
        padding:16px;
        height:100%;
    }}
    .muted {{
        opacity: .85;
        font-size: 0.95rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Data: UCI Bike Sharing (HTTPS, no S3) ----------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
UCI_HOURLY_CSV = "hour.csv"
UA = {"User-Agent": "RidePulse/1.0 (+https://github.com/Adithya-Vasudevan)"}

@st.cache_data(show_spinner=True, ttl=24*3600)
def fetch_uci_bikeshare() -> Tuple[pd.DataFrame, str]:
    r = requests.get(UCI_ZIP_URL, headers=UA, timeout=60)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    with zipfile.ZipFile(bio) as zf:
        with zf.open(UCI_HOURLY_CSV) as fh:
            df = pd.read_csv(fh)
    return df, "UCI Bike Sharing (DC, 2011–2012, hour.csv)"

def normalize_uci_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"], errors="coerce")
    out["hr"] = pd.to_numeric(out["hr"], errors="coerce").astype("Int64")
    out["ts"] = out["dteday"] + pd.to_timedelta(out["hr"].fillna(0).astype(int), unit="h")
    out["date"] = out["ts"].dt.date
    out["hour"] = out["ts"].dt.hour
    out["weekday"] = out["ts"].dt.day_name()
    out["cnt"] = pd.to_numeric(out["cnt"], errors="coerce")
    out["temp"] = pd.to_numeric(out["temp"], errors="coerce")
    return out

# Try fetching dataset (non-blocking UX: if it fails, we still render hero + cards)
trips: pd.DataFrame | None = None
data_source: str = ""
try:
    raw_df, data_source = fetch_uci_bikeshare()
    trips = normalize_uci_hour(raw_df)
except Exception:
    trips = None

# ---------------- Hero ----------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
col_h1, col_h2 = st.columns([1.2, 1])

with col_h1:
    st.markdown("### 🚲 RidePulse")
    st.markdown(
        f"<div class='fadein' style='font-size:1.6rem; line-height:1.5;'>"
        f"<b>See bikeshare usage at a glance</b> — trends, peaks, stations, and routes, "
        f"with clear visuals and lightweight motion."
        f"</div>",
        unsafe_allow_html=True,
    )
    # CTA row
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("📈 Explore Trends", key="cta_trends"):
            # Prefer built-in navigation if available
            try:
                st.switch_page("pages/03_Trends.py")
            except Exception:
                st.session_state["_nav_to"] = "Trends"
    with c2:
        if st.button("🧭 Trip Explorer", key="cta_explorer"):
            try:
                st.switch_page("pages/02_Explorer.py")
            except Exception:
                st.session_state["_nav_to"] = "Explorer"
    with c3:
        if st.button("📍 Stations & Routes", key="cta_stations"):
            try:
                st.switch_page("pages/04_Stations.py")
            except Exception:
                st.session_state["_nav_to"] = "Stations"

with col_h2:
    # Lottie: lightweight animation
    lottie("https://assets5.lottiefiles.com/packages/lf20_rhnmhzwq.json", height=220)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Quick KPIs + Sparklines ----------------
st.markdown("#### At a glance")

kcols = st.columns(4)
# Defaults if data missing
rides_30d = "—"
daily_med = "—"
peak_hour = "—"
temp_corr = "—"

spark1 = go.Figure()
spark2 = go.Figure()
spark1.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0))
spark2.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0))

if trips is not None and not trips.empty:
    by_day = trips.groupby("date", as_index=False)["cnt"].sum().sort_values("date")
    last_30 = by_day.tail(30)
    rides_30d = f"{int(last_30['cnt'].sum()):,}" if len(last_30) else "—"
    daily_med = f"{float(by_day['cnt'].tail(60).median()):,.0f}" if len(by_day) else "—"
    # Peak hour in last 60 days
    recent = trips[pd.to_datetime(trips["date"]) >= pd.to_datetime(max(trips["date"])) - pd.Timedelta(days=60)]
    if not recent.empty:
        peak_hour_val = int(recent.groupby("hour")["cnt"].median().idxmax())
        peak_hour = f"{peak_hour_val:02d}:00"
    # Simple temperature correlation (Spearman to handle non-linear-ish)
    try:
        if len(recent) > 200:
            corr = float(recent["temp"].corr(recent["cnt"], method="spearman"))
            temp_corr = f"{corr:+.2f}"
    except Exception:
        pass

    # Sparklines
    spark1 = go.Figure(go.Scatter(
        x=last_30["date"], y=last_30["cnt"],
        mode="lines",
        line=dict(color=accent, width=2),
        fill="tozeroy",
        hoverinfo="x+y",
    ))
    spark1.update_layout(yaxis=dict(visible=False), xaxis=dict(visible=False))

    by_hour = recent.groupby("hour", as_index=False)["cnt"].median() if not recent.empty else trips.groupby("hour", as_index=False)["cnt"].median()
    spark2 = go.Figure(go.Scatter(
        x=by_hour["hour"], y=by_hour["cnt"],
        mode="lines",
        line=dict(color=primary, width=2),
        hoverinfo="x+y",
    ))
    spark2.update_layout(yaxis=dict(visible=False), xaxis=dict(visible=False))

with kcols[0]:
    st.metric("Rides (last 30 days)", rides_30d)
    st.plotly_chart(_apply_theme(spark1, 60), use_container_width=True, config={"displayModeBar": False})
with kcols[1]:
    st.metric("Median rides/day (60d)", daily_med)
    st.plotly_chart(_apply_theme(spark2, 60), use_container_width=True, config={"displayModeBar": False})
with kcols[2]:
    st.metric("Peak hour (recent)", peak_hour)
    st.caption("Typical commuting peaks drive this.")
with kcols[3]:
    st.metric("Temp vs rides (ρ)", temp_corr)
    st.caption("Spearman correlation over recent period.")

# ---------------- Recent Trend (Area) ----------------
st.markdown("#### Recent trend")
if trips is not None and not trips.empty:
    by_day = trips.groupby("date", as_index=False)["cnt"].sum().sort_values("date")
    # Focus last ~6 months
    if len(by_day) > 180:
        by_day = by_day.tail(180)
    by_day["trend_7d"] = by_day["cnt"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=by_day["date"], y=by_day["cnt"],
        mode="lines",
        line=dict(color=accent, width=1.5),
        fill="tozeroy",
        name="Daily rides",
    ))
    fig.add_trace(go.Scatter(
        x=by_day["date"], y=by_day["trend_7d"],
        mode="lines",
        line=dict(color=primary, width=3),
        name="7-day trend",
    ))
    fig.update_layout(
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        xaxis_title="Date",
        yaxis_title="Rides",
    )
    st.plotly_chart(_apply_theme(fig), use_container_width=True, theme="streamlit")
else:
    st.info("Live trend will appear when the dataset is reachable.")

# ---------------- Explore cards ----------------
st.markdown("#### Explore")
cA, cB, cC = st.columns(3)
with cA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Trends")
    st.write("Daily volumes, intraday/weekday profiles, and smoothing to reveal patterns.")
    if hasattr(st, "page_link"):
        st.page_link("pages/03_Trends.py", label="Open Trends", icon="📈")
    else:
        if st.button("Open Trends"):
            try:
                st.switch_page("pages/03_Trends.py")
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧭 Explorer")
    st.write("Filter and slice trips to answer specific questions quickly.")
    if hasattr(st, "page_link"):
        st.page_link("pages/02_Explorer.py", label="Open Explorer", icon="🧭")
    else:
        if st.button("Open Explorer"):
            try:
                st.switch_page("pages/02_Explorer.py")
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

with cC:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📍 Stations & Routes")
    st.write("Top stations, station trends, and popular OD pairs.")
    if hasattr(st, "page_link"):
        st.page_link("pages/04_Stations.py", label="Open Stations & Routes", icon="📍")
    else:
        if st.button("Open Stations & Routes"):
            try:
                st.switch_page("pages/04_Stations.py")
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Source note ----------------
st.markdown("---")
with st.container():
    st.caption(
        "Data source for overview: UCI Bike Sharing Dataset (Washington, DC, 2011–2012). "
        "This page avoids S3 and requires no uploads. Visuals are theme-aware and include subtle motion."
    )