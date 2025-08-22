from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.gbfs import merged_station_frame

# Optional analytics/badges; degrade gracefully if missing
try:
    from utils.badges import award_badge, track_page_visit  # type: ignore
except Exception:
    def award_badge(_: str) -> None:
        return
    def track_page_visit(_: str) -> None:
        return

# Optional Lottie helper (graceful if missing)
try:
    from utils.ui import show_lottie  # type: ignore
except Exception:
    def show_lottie(_: str, height: int = 120) -> None:
        return


# ---------- Theming ----------
def _is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True


def _apply_theme(fig: go.Figure) -> go.Figure:
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
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------- Helpers / data wrangling ----------
def _capacity_from(df: pd.DataFrame) -> pd.Series:
    if "capacity" in df.columns and df["capacity"].fillna(0).max() > 0:
        return pd.to_numeric(df["capacity"], errors="coerce")
    bikes = pd.to_numeric(df.get("num_bikes_available", np.nan), errors="coerce")
    docks = pd.to_numeric(df.get("num_docks_available", np.nan), errors="coerce")
    cap = bikes.add(docks, fill_value=np.nan)
    # If both bikes/docks are NaN for a row, result is NaN. Good.
    return cap


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Standardize key fields
    out["num_bikes_available"] = pd.to_numeric(out.get("num_bikes_available"), errors="coerce")
    out["num_docks_available"] = pd.to_numeric(out.get("num_docks_available"), errors="coerce")
    out["capacity"] = _capacity_from(out)

    # Compute percent full safely
    with np.errstate(divide="ignore", invalid="ignore"):
        out["percent_full"] = (out["num_bikes_available"] / out["capacity"]) * 100.0
    out.loc[~np.isfinite(out["percent_full"]), "percent_full"] = np.nan

    # Clip to 0..100 for display sanity
    out["percent_full"] = out["percent_full"].clip(lower=0, upper=100)

    # Clean names
    if "name" in out.columns:
        out["name"] = out["name"].fillna("Unnamed station")
    else:
        out["name"] = "Station"

    return out


@dataclass
class SystemFacts:
    stations: int
    total_capacity: int
    total_bikes: int
    total_docks: int
    avg_capacity: float
    median_capacity: float
    mean_fill_pct: float
    median_fill_pct: float
    full_stations: int
    empty_stations: int


def _summarize(df: pd.DataFrame) -> SystemFacts:
    cap = pd.to_numeric(df["capacity"], errors="coerce")
    bikes = pd.to_numeric(df["num_bikes_available"], errors="coerce")
    docks = pd.to_numeric(df["num_docks_available"], errors="coerce")

    stations = len(df)
    total_capacity = int(np.nansum(cap))
    total_bikes = int(np.nansum(bikes))
    total_docks = int(np.nansum(docks))
    avg_capacity = float(np.nanmean(cap)) if stations else 0.0
    median_capacity = float(np.nanmedian(cap)) if stations else 0.0

    fill = pd.to_numeric(df["percent_full"], errors="coerce")
    mean_fill_pct = float(np.nanmean(fill)) if stations else 0.0
    median_fill_pct = float(np.nanmedian(fill)) if stations else 0.0

    full_stations = int(np.nansum((fill >= 95).astype(int)))
    empty_stations = int(np.nansum((fill <= 5).astype(int)))

    return SystemFacts(
        stations=stations,
        total_capacity=total_capacity,
        total_bikes=total_bikes,
        total_docks=total_docks,
        avg_capacity=avg_capacity,
        median_capacity=median_capacity,
        mean_fill_pct=mean_fill_pct,
        median_fill_pct=median_fill_pct,
        full_stations=full_stations,
        empty_stations=empty_stations,
    )


# ---------- Charts ----------
def system_gauge(pct: float) -> go.Figure:
    # Use RGBA strings (no 8-digit hex) for steps
    step_bg = "#2A2F36" if _is_dark() else "#f3f4f6"
    step_0_30 = "rgba(239, 68, 68, 0.25)"     # red-ish
    step_30_70 = "rgba(59, 130, 246, 0.25)"   # blue-ish
    step_70_100 = "rgba(16, 185, 129, 0.25)"  # green-ish
    bar_color = "#10b981" if pct >= 70 else "#3b82f6" if pct >= 30 else "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 30], "color": step_0_30},
                    {"range": [30, 70], "color": step_30_70},
                    {"range": [70, 100], "color": step_70_100},
                ],
            },
            title={"text": "System fill (avg %)"},
        )
    )
    fig.update_layout(height=260)
    return _apply_theme(fig)


def fill_bucket_chart(df: pd.DataFrame) -> go.Figure:
    # Bucket stations by percent_full
    bins = [0, 10, 25, 40, 60, 75, 90, 100]
    labels = ["0–10%", "10–25%", "25–40%", "40–60%", "60–75%", "75–90%", "90–100%"]
    fill = pd.to_numeric(df["percent_full"], errors="coerce")
    cats = pd.cut(fill, bins=bins, labels=labels, include_lowest=True)
    s = cats.value_counts().reindex(labels, fill_value=0)

    fig = px.bar(
        x=s.index,
        y=s.values,
        labels={"x": "Fill level bucket", "y": "Stations"},
        title="Where are stations right now?",
        color=s.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_xaxes(tickangle=0)
    return _apply_theme(fig)


def capacity_top_chart(df: pd.DataFrame, n: int = 12) -> go.Figure:
    top = df.sort_values("capacity", ascending=False).head(n)
    fig = px.bar(
        top,
        x="name",
        y="capacity",
        title=f"Top {len(top)} stations by capacity",
        labels={"name": "Station", "capacity": "Capacity"},
    )
    fig.update_layout(xaxis_tickangle=-35)
    return _apply_theme(fig)


def bikes_vs_docks(df: pd.DataFrame) -> go.Figure:
    # Simple balance scatter as a fun visual
    if "num_docks_available" not in df.columns:
        # synthesize docks if not present
        docks = df["capacity"].fillna(0) - df["num_bikes_available"].fillna(0)
        df = df.assign(num_docks_available=docks)

    fig = px.scatter(
        df,
        x="num_bikes_available",
        y="num_docks_available",
        hover_name="name",
        title="Balance snapshot (bikes vs docks)",
        labels={"num_bikes_available": "Bikes available", "num_docks_available": "Docks available"},
        opacity=0.8,
    )
    # Add diagonal y=x
    max_axis = float(
        np.nanmax(
            [
                df["num_bikes_available"].max(skipna=True),
                df["num_docks_available"].max(skipna=True),
            ]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            line=dict(dash="dash", color="#9ca3af"),
            name="Perfect balance",
            showlegend=False,
        )
    )
    return _apply_theme(fig)


def availability_map(df: pd.DataFrame) -> Optional[go.Figure]:
    if not {"lat", "lon"}.issubset(df.columns):
        return None
    # Marker color by percent_full and size by capacity
    cap = df["capacity"].fillna(df["capacity"].median()).clip(lower=1)
    size = np.sqrt(cap) * 3  # moderate scaling
    color = df["percent_full"].fillna(0)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        hover_name="name",
        hover_data={"percent_full": ":.1f", "capacity": True, "lat": False, "lon": False},
        color=color,
        color_continuous_scale="Turbo",
        size=size,
        size_max=30,
        zoom=11,
        title="Map: station fill",
    )
    fig.update_layout(
        mapbox_style="carto-positron" if not _is_dark() else "carto-darkmatter",
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="% full"),
    )
    return fig


# ---------- Page ----------
st.set_page_config(page_title="Fun Facts • RidePulse NYC", page_icon="🎉", layout="wide")
track_page_visit("Fun Facts")
award_badge("fun_facts")

st.title("🎉 Fun Facts")

with st.sidebar:
    st.header("Options")
    auto_refresh = st.toggle("Auto-refresh data every 60s", value=False)
    if st.button("Refresh now", use_container_width=True):
        # Simple rerun to pull latest merged frame
        st.rerun()

# Load data
df_raw = merged_station_frame()
if df_raw is None or len(df_raw) == 0:
    st.warning("Live data not available right now. Please try again shortly.")
    st.stop()

df = _ensure_metrics(df_raw)

# Auto refresh note
if auto_refresh:
    st.caption("Auto-refreshing every 60 seconds.")
    # lightweight auto-refresh: bump a query param each minute
    st.query_params.update({"ts": str(int(time.time() // 60))})

# ---------- Highlights ----------
facts = _summarize(df)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Stations", f"{facts.stations:,}")
with col2:
    st.metric("System capacity", f"{facts.total_capacity:,}")
with col3:
    st.metric("Bikes available", f"{facts.total_bikes:,}")
with col4:
    st.metric("Docks available", f"{facts.total_docks:,}")

cga, cgb = st.columns([1.3, 1])
with cga:
    st.plotly_chart(system_gauge(round(facts.mean_fill_pct, 1)), use_container_width=True, theme="streamlit")
with cgb:
    st.metric("Avg capacity", f"{facts.avg_capacity:.1f}")
    st.metric("Median capacity", f"{facts.median_capacity:.1f}")
    st.metric("Very full (≥95%)", f"{facts.full_stations}")
    st.metric("Very empty (≤5%)", f"{facts.empty_stations}")

# ---------- Tabs for structure ----------
tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Top & Extremes", "Balance", "Map"])

with tab1:
    st.subheader("How are stations distributed right now?")
    st.plotly_chart(fill_bucket_chart(df), use_container_width=True, theme="streamlit")

    st.markdown("—")
    st.caption(
        "Tip: Stations at the extremes (very full or very empty) are prime candidates for rebalancing. "
        "Mid-range fill helps accommodate trips in both directions."
    )

with tab2:
    st.subheader("Biggest stations")
    st.plotly_chart(capacity_top_chart(df), use_container_width=True, theme="streamlit")

    st.markdown("—")
    st.subheader("Extremes right now")
    # Fullest / emptiest by percent_full (filter to stations with known capacity)
    valid = df.dropna(subset=["percent_full", "capacity"]).copy()
    # If two have same %, prefer higher capacity for fullness and lower for emptiness
    fullest = valid.sort_values(["percent_full", "capacity"], ascending=[False, False]).head(5)
    emptiest = valid.sort_values(["percent_full", "capacity"], ascending=[True, True]).head(5)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Most full")
        for _, r in fullest.iterrows():
            st.write(
                f"• {r['name']} — {r['percent_full']:.0f}% full "
                f"(bikes: {int(r['num_bikes_available'] or 0)}, cap: {int(r['capacity'] or 0)})"
            )
    with c2:
        st.markdown("##### Most empty")
        for _, r in emptiest.iterrows():
            st.write(
                f"• {r['name']} — {r['percent_full']:.0f}% full "
                f"(bikes: {int(r['num_bikes_available'] or 0)}, cap: {int(r['capacity'] or 0)})"
            )

    st.markdown("—")
    st.caption(
        "Context: Smaller stations can swing from empty to full quickly during peaks. Larger hubs tend to be more stable."
    )

with tab3:
    st.subheader("Bikes vs Docks balance")
    st.plotly_chart(bikes_vs_docks(df), use_container_width=True, theme="streamlit")
    st.caption(
        "Points near the dashed line indicate a good balance at this moment. "
        "Points far from it suggest constrained options for either picking up or returning a bike."
    )

with tab4:
    st.subheader("Live availability map")
    fig_map = availability_map(df)
    if fig_map is None:
        st.info("Location data unavailable for this feed.")
    else:
        st.plotly_chart(fig_map, use_container_width=True, theme="streamlit")

# ---------- Did you know? ----------
st.markdown("---")
st.markdown("### Did you know?")
tips = [
    "System-wide average fill can mask local extremes—check the distribution to spot imbalances.",
    "Stations right outside transit hubs often hit extremes first during rush hour.",
    "If two neighboring stations differ wildly in fill, move the bike to the emptier one to help balance supply.",
    "Larger capacity reduces volatility: each check-in/out has a smaller effect on percent full.",
]
random.shuffle(tips)
for tip in tips[:3]:
    st.markdown(f"- {tip}")

# Optional celebration if the network looks healthy
if facts.mean_fill_pct >= 60 and facts.empty_stations <= max(1, facts.stations // 50):
    show_lottie("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json", height=140)
    st.success("Looking good! The system is broadly ready for both pickups and returns.")