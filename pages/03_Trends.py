from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
UCI_HOURLY_CSV = "hour.csv"   # inside the ZIP

UA = {"User-Agent": "RidePulse/1.0 (+https://github.com/Adithya-Vasudevan)"}


# ---------- Theme ----------
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
        margin=dict(l=20, r=20, t=50, b=30),
    )
    return fig


# ---------- Data fetch + normalize (UCI) ----------
@st.cache_data(show_spinner=True, ttl=24*3600)
def fetch_uci_bikeshare() -> Tuple[pd.DataFrame, str]:
    """
    Download the UCI Bike Sharing Dataset ZIP and read hour.csv.
    Returns (df, source_label). No S3 needed.
    """
    r = requests.get(UCI_ZIP_URL, headers=UA, timeout=60)
    r.raise_for_status()
    bio = io.BytesIO(r.content)
    with zipfile.ZipFile(bio) as zf:
        with zf.open(UCI_HOURLY_CSV) as fh:
            df = pd.read_csv(fh)
    return df, f"UCI Machine Learning Repository — {UCI_HOURLY_CSV}"

def normalize_uci_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    UCI hour.csv columns (selected):
      dteday (date), hr (0-23), season (1..4), weathersit (1..4),
      temp (scaled 0..1), atemp, hum, windspeed,
      weekday (0=Sun), workingday (0/1), holiday (0/1),
      cnt (total rides per hour), casual, registered
    """
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"], errors="coerce")
    out["hr"] = pd.to_numeric(out["hr"], errors="coerce").astype("Int64")
    # Compose hourly timestamp
    out["ts"] = pd.to_datetime(out["dteday"]) + pd.to_timedelta(out["hr"].fillna(0).astype(int), unit="h")
    # Derivations for charting
    out["date"] = out["ts"].dt.date                # Python date objects
    out["hour"] = out["ts"].dt.hour
    out["weekday_name"] = out["ts"].dt.day_name()
    out["month"] = out["ts"].dt.to_period("M").astype(str)
    # Weather label
    weather_map = {1: "Clear", 2: "Mist/Cloudy", 3: "Light Snow/Rain", 4: "Severe Weather"}
    out["weathersit_label"] = out["weathersit"].map(weather_map).fillna("Unknown")
    # Numeric fields
    out["cnt"] = pd.to_numeric(out["cnt"], errors="coerce")
    out["casual"] = pd.to_numeric(out.get("casual", np.nan), errors="coerce")
    out["registered"] = pd.to_numeric(out.get("registered", np.nan), errors="coerce")
    return out


# ---------- Simple smoother (no extra deps) ----------
def smooth_by_rolling(x: np.ndarray, y: np.ndarray, frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple x-sorted rolling mean smoother.
    - Sort by x
    - Rolling mean over a window = max(20, frac * n)
    Returns (x_sorted, y_smooth)
    """
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    n = len(xs)
    win = max(20, int(max(1, frac) if frac >= 1 else frac * n))
    s = pd.Series(ys).rolling(win, center=True, min_periods=max(5, win // 4)).mean().to_numpy()
    return xs, s


# ---------- Page ----------
st.set_page_config(page_title="Trends • RidePulse NYC", page_icon="📈", layout="wide")
st.title("📈 Trip Trends (auto-fetched, UCI dataset)")

with st.sidebar:
    st.header("Data source")
    st.caption("This tab uses the UCI Bike Sharing Dataset (Washington, DC Capital Bikeshare, 2011–2012). No uploads or paths needed.")
    if st.button("Reload data"):
        st.query_params.update({"reload": datetime.utcnow().strftime("%Y%m%d%H%M%S")})

# Fetch + normalize
try:
    raw_df, source_label = fetch_uci_bikeshare()
except Exception:
    st.error("Could not fetch the UCI Bike Sharing dataset right now. Please try again in a minute.")
    st.stop()

trips = normalize_uci_hour(raw_df)

with st.container(border=True):
    st.write("Data source")
    st.write(f"- {source_label}")
    st.caption("Fields include hourly ride counts (cnt), temperature (normalized), humidity, windspeed, working day/holiday flags, and a weather category.")

# Date range filter (use Python date objects, not pandas Timestamps)
min_date = min(trips["date"]) if len(trips) else None
max_date = max(trips["date"]) if len(trips) else None

if not min_date or not max_date:
    st.error("Dataset has no date values to filter.")
    st.stop()

default_start = max(min_date, max_date - timedelta(days=120))
c1, c2 = st.columns([3, 1])
with c1:
    date_range = st.slider(
        "Filter date range",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, max_date),
        format="YYYY-MM-DD",
    )
with c2:
    st.caption("Use the slider to focus on a time window.")

# Apply filter using date objects
mask = (trips["date"] >= date_range[0]) & (trips["date"] <= date_range[1])
trips = trips.loc[mask].copy()

# KPIs
by_day = trips.groupby("date", as_index=False)["cnt"].sum().sort_values("date")
rides_total = int(by_day["cnt"].sum())
rides_daily_med = float(by_day["cnt"].median()) if len(by_day) else np.nan
wkday = trips["weekday_name"].value_counts().idxmax() if not trips.empty else "—"

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Rides (sum in range)", f"{rides_total:,}")
with k2:
    st.metric("Median rides/day", f"{rides_daily_med:,.0f}" if np.isfinite(rides_daily_med) else "—")
with k3:
    st.metric("Most common weekday in range", wkday)

# Tabs
tab_sys, tab_profiles, tab_weather = st.tabs(
    ["System trends", "Usage profiles", "Weather effects"]
)

# -------- System trends --------
with tab_sys:
    st.subheader("System trends")
    if not by_day.empty:
        by_day["trend_7d"] = by_day["cnt"].rolling(7, min_periods=1).mean()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(x=by_day["date"], y=by_day["cnt"], name="Rides per day",
                                   marker_color="#60a5fa", opacity=0.6))
        fig_daily.add_trace(go.Scatter(x=by_day["date"], y=by_day["trend_7d"], name="7-day trend",
                                       line=dict(color="#10b981", width=3)))
        fig_daily.update_layout(
            yaxis_title="Rides",
            xaxis_title="Date",
            barmode="overlay",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
        )
        st.plotly_chart(_apply_theme(fig_daily), use_container_width=True, theme="streamlit")
        with st.container(border=True):
            st.write("What this shows")
            st.caption("Daily ride volume and a 7-day moving average to reveal the underlying trend and seasonality.")

    # Month seasonality (boxplot)
    trips["month_lbl"] = pd.to_datetime(trips["date"]).dt.to_period("M").astype(str)
    fig_m = px.box(trips, x="month_lbl", y="cnt", points=False,
                   labels={"month_lbl": "Month", "cnt": "Rides per hour"},
                   title="Monthly seasonality (distribution of hourly rides)")
    fig_m.update_layout(height=360)
    st.plotly_chart(_apply_theme(fig_m), use_container_width=True, theme="streamlit")
    with st.container(border=True):
        st.write("What this shows")
        st.caption("Seasonal differences across months. Taller boxes/whiskers indicate more variability in hourly rides.")

# -------- Usage profiles --------
with tab_profiles:
    st.subheader("Usage profiles")

    cA, cB = st.columns(2)
    with cA:
        hour_prof = trips.groupby("hour", as_index=False)["cnt"].median()
        fig_hr = px.line(hour_prof, x="hour", y="cnt", markers=True, title="Intraday profile (median rides by hour)")
        fig_hr.update_layout(height=320, xaxis_title="Hour", yaxis_title="Rides (median)")
        st.plotly_chart(_apply_theme(fig_hr), use_container_width=True, theme="streamlit")
        with st.container(border=True):
            st.write("What this shows")
            st.caption("Typical intraday shape: commuting peaks vs mid-day/evening usage.")

    with cB:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_prof = trips.groupby("weekday_name", as_index=False)["cnt"].median()
        dow_prof["weekday_name"] = pd.Categorical(dow_prof["weekday_name"], categories=order, ordered=True)
        dow_prof = dow_prof.sort_values("weekday_name")
        fig_dow = px.bar(dow_prof, x="weekday_name", y="cnt", title="Day-of-week profile (median rides per hour)")
        fig_dow.update_traces(marker_color="#60a5fa")
        fig_dow.update_layout(height=320, xaxis_title="Weekday", yaxis_title="Rides (median)")
        st.plotly_chart(_apply_theme(fig_dow), use_container_width=True, theme="streamlit")
        with st.container(border=True):
            st.write("What this shows")
            st.caption("Differences between weekdays and weekends.")

    # Heatmap: hour x weekday (median)
    heat = (trips.groupby(["weekday_name", "hour"])["cnt"]
            .median()
            .reset_index())
    heat["weekday_name"] = pd.Categorical(heat["weekday_name"],
                                          categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                                          ordered=True)
    heat_wide = heat.pivot(index="weekday_name", columns="hour", values="cnt")
    fig_heat = px.imshow(heat_wide, color_continuous_scale="Turbo", aspect="auto", origin="lower",
                         title="Heatmap: rides by hour and weekday (median)")
    fig_heat.update_layout(height=420, coloraxis_colorbar=dict(title="Rides"))
    st.plotly_chart(_apply_theme(fig_heat), use_container_width=True, theme="streamlit")
    with st.container(border=True):
        st.write("What this shows")
        st.caption("When during the week riding is most intense. Bright cells = busier periods.")

# -------- Weather effects --------
with tab_weather:
    st.subheader("Weather effects")

    c1, c2 = st.columns(2)
    with c1:
        # Scatter with built-in smoothing (no statsmodels)
        df_sc = trips[["temp", "cnt"]].dropna()
        if not df_sc.empty:
            df_sc = df_sc.sample(min(20_000, len(df_sc)), random_state=42)
            x = df_sc["temp"].to_numpy()
            y = df_sc["cnt"].to_numpy()
            xs, ys = smooth_by_rolling(x, y, frac=0.05)

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=df_sc["temp"], y=df_sc["cnt"],
                mode="markers",
                name="Hour",
                marker=dict(color="#60a5fa", opacity=0.5, size=6),
            ))
            if ys is not None and np.isfinite(ys).any():
                fig_sc.add_trace(go.Scatter(
                    x=xs[np.isfinite(ys)], y=ys[np.isfinite(ys)],
                    mode="lines",
                    name="Smoothed",
                    line=dict(color="#10b981", width=3),
                ))
            fig_sc.update_layout(
                title="Rides vs temperature",
                xaxis_title="Temperature (normalized 0–1)",
                yaxis_title="Rides per hour",
                height=360,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(_apply_theme(fig_sc), use_container_width=True, theme="streamlit")
        else:
            st.info("No temperature/ride data available for scatter.")

        with st.container(border=True):
            st.write("What this shows")
            st.caption("Warmer hours generally correlate with more rides in this dataset.")

    with c2:
        by_w = trips.groupby("weathersit_label", as_index=False)["cnt"].median()
        fig_w = px.bar(by_w, x="weathersit_label", y="cnt",
                       labels={"weathersit_label": "Weather", "cnt": "Rides per hour (median)"},
                       title="Rides by weather condition (median)")
        st.plotly_chart(_apply_theme(fig_w), use_container_width=True, theme="streamlit")
        with st.container(border=True):
            st.write("What this shows")
            st.caption("Clear conditions see more riding; severe weather depresses ridership.")

st.markdown("---")
st.subheader("Notes")
st.write(
    "- Data: UCI Bike Sharing Dataset (Washington, DC Capital Bikeshare, 2011–2012), hourly records of ride counts and weather features.\n"
    "- Smoothing uses an internal rolling mean, so no extra packages (like statsmodels) are required."
)