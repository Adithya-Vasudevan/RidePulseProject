from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------
# Constants
# -----------------------------------------------
UA = {"User-Agent": "RidePulse/1.0 (+https://github.com/Adithya-Vasudevan)"}
REQUEST_TIMEOUT = 45

# Citi Bike monthly tripdata
CITIBIKE_S3_BASE = "https://s3.amazonaws.com/tripdata/"
CITIBIKE_S3_INDEX = CITIBIKE_S3_BASE + "?list-type=2"
CITIBIKE_SYSTEM_DATA = "https://ride.citibikenyc.com/system-data"

# -----------------------------------------------
# Theme helpers
# -----------------------------------------------
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

# -----------------------------------------------
# Citi Bike discovery + loading
# -----------------------------------------------
def _extract_ym_from_key(key: str) -> Optional[Tuple[int, int]]:
    """
    Extract (year, month) from a Citi Bike filename/key.
    Examples:
      2025-07-citibike-tripdata.csv.zip
      202507-citibike-tripdata.csv.zip
      JC-2025-07-citibike-tripdata.csv.zip
    """
    m = re.search(r"(20\d{2})[-_]?(\d{2})", key)
    if not m:
        m2 = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec).*(20\d{2})", key, flags=re.I)
        if m2:
            months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
            return int(m2.group(2)), months.index(m2.group(1).lower()) + 1
        return None
    return int(m.group(1)), int(m.group(2))

def _parse_s3_list(xml_text: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse S3 ListV2 XML and return (keys, next_continuation_token).
    """
    keys: List[str] = []
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], None

    for contents in root.findall(".//s3:Contents", ns):
        key_elem = contents.find("s3:Key", ns)
        if key_elem is not None and key_elem.text:
            k = key_elem.text
            if re.search(r"\.(csv|csv\.zip|parquet)$", k, flags=re.I):
                keys.append(k)

    next_token_elem = root.find(".//s3:NextContinuationToken", ns)
    next_token = next_token_elem.text if next_token_elem is not None else None
    return keys, next_token

def _discover_from_s3(max_pages: int = 15) -> List[str]:
    """
    List keys via S3 ListV2 with pagination.
    """
    all_keys: List[str] = []
    token: Optional[str] = None
    pages = 0
    while pages < max_pages:
        url = CITIBIKE_S3_INDEX + (f"&continuation-token={token}" if token else "")
        r = requests.get(url, headers=UA, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        keys, token = _parse_s3_list(r.text)
        all_keys.extend(keys)
        pages += 1
        if not token:
            break
    return all_keys

def _discover_from_system_data_page() -> List[str]:
    """
    Scrape the official system-data page for monthly trip links.
    """
    r = requests.get(CITIBIKE_SYSTEM_DATA, headers=UA, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    html = r.text
    keys: List[str] = []
    # Links like .../tripdata/2025-07-citibike-tripdata.csv.zip
    for m in re.finditer(r'href="[^"]*?/tripdata/([^"]+?\.(?:csv(?:\.zip)?|parquet))"', html, flags=re.I):
        keys.append(m.group(1))
    # Direct S3 links if present
    for m in re.finditer(r'href="https://s3\.amazonaws\.com/tripdata/([^"]+)"', html, flags=re.I):
        keys.append(m.group(1))
    return list(dict.fromkeys(keys))  # dedupe, preserve order

def _head_exists(url: str, timeout: int = REQUEST_TIMEOUT) -> bool:
    try:
        r = requests.head(url, headers=UA, timeout=timeout, allow_redirects=True)
        if r.status_code == 405:
            r = requests.get(url, headers=UA, timeout=timeout, stream=True)
        return 200 <= r.status_code < 400
    except requests.RequestException:
        return False

def _generate_recent_candidates(n_months: int) -> List[str]:
    """
    Guess recent filenames by pattern to bypass index if needed.
    """
    today = datetime.utcnow().replace(day=1)
    names: List[str] = []
    y, m = today.year, today.month
    for _ in range(n_months + 2):
        ym_no_dash = f"{y}{m:02d}"
        ym_dash = f"{y}-{m:02d}"
        names.extend([
            f"{ym_no_dash}-citibike-tripdata.csv.zip",
            f"{ym_dash}-citibike-tripdata.csv.zip",
            f"{ym_no_dash}-citibike-tripdata.csv",
            f"{ym_dash}-citibike-tripdata.csv",
        ])
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return names

def _urls_from_keys(keys: List[str], n_months: int) -> List[str]:
    keyed: List[Tuple[Tuple[int, int], str]] = []
    for k in keys:
        ym = _extract_ym_from_key(k)
        if ym:
            keyed.append((ym, k))
    if not keyed:
        return []
    keyed.sort(key=lambda t: (t[0][0], t[0][1]), reverse=True)
    seen: set[Tuple[int, int]] = set()
    urls: List[str] = []
    for _, k in keyed:
        ym = _extract_ym_from_key(k)
        if ym and ym not in seen:
            urls.append(CITIBIKE_S3_BASE + k)
            seen.add(ym)
        if len(urls) >= n_months:
            break
    return urls

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def list_citibike_recent_urls(max_months: int = 3) -> Tuple[List[str], str]:
    """
    Discover URLs for the most recent N monthly trip files.
    Returns (urls, discovery_method).
    Tries S3 index, then system-data page, then guessed filenames with HEAD validation.
    """
    # Strategy A: S3 listing
    try:
        keys = _discover_from_s3(max_pages=15)
        urls = _urls_from_keys(keys, max_months)
        if urls:
            return urls, "S3 index"
    except Exception:
        pass

    # Strategy B: system-data page
    try:
        keys = _discover_from_system_data_page()
        urls = _urls_from_keys(keys, max_months)
        if urls:
            return urls, "System-data page"
    except Exception:
        pass

    # Strategy C: guess + HEAD
    candidates = _generate_recent_candidates(max_months)
    valid_keys: List[str] = []
    for name in candidates:
        u = CITIBIKE_S3_BASE + name
        if _head_exists(u, timeout=15):
            valid_keys.append(name)
        if len(_urls_from_keys(valid_keys, max_months)) >= max_months:
            break
    urls = _urls_from_keys(valid_keys, max_months)
    if urls:
        return urls, "Guessed filenames (HEAD verified)"

    raise RuntimeError("Could not discover Citi Bike trip files (S3 listing blocked or site unavailable).")

def _read_month_from_zip_bytes(b: bytes, usecols: Iterable[str]) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(b)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Zip has no CSV")
        names.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, low_memory=False, usecols=lambda c: c.lower() in {u.lower() for u in usecols})

def _read_month_from_url(url: str, usecols: Iterable[str]) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    if url.lower().endswith(".zip"):
        return _read_month_from_zip_bytes(r.content, usecols=usecols)
    return pd.read_csv(io.BytesIO(r.content), low_memory=False, usecols=lambda c: c.lower() in {u.lower() for u in usecols})

def _normalize_trip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize trip start timestamp into 'started_at' (datetime64[ns]),
    handling schema variations.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    started_col = None
    for candidate in ["started_at", "starttime", "start_time", "start_time_local", "start_time_utc", "start time", "started at"]:
        if candidate in cols_lower:
            started_col = cols_lower[candidate]
            break
    if started_col is None:
        raise ValueError("Could not find a start time column.")
    df = df[[started_col]].copy()
    df.rename(columns={started_col: "started_at"}, inplace=True)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce", utc=False)
    df = df.dropna(subset=["started_at"])
    return df

@st.cache_data(show_spinner=True, ttl=6 * 3600)
def fetch_citibike_recent_hourly(n_months: int = 3) -> Tuple[pd.DataFrame, str]:
    """
    Download the most recent N monthly Citi Bike trip files, normalize, and aggregate to hourly counts.
    Returns (hourly_df, source_label).
    """
    urls, method = list_citibike_recent_urls(n_months)
    if not urls:
        raise RuntimeError("No recent Citi Bike monthly files found.")
    progress = st.progress(0.0, text="Downloading Citi Bike monthly files…")
    frames: List[pd.DataFrame] = []
    for i, url in enumerate(urls, start=1):
        try:
            df = _read_month_from_url(
                url,
                usecols=["started_at", "starttime", "start_time", "start_time_local", "start_time_utc", "start time", "started at"],
            )
            df = _normalize_trip_columns(df)
            df["ts"] = df["started_at"].dt.floor("H")
            counts = df.groupby("ts", as_index=False).size().rename(columns={"size": "cnt"})
            frames.append(counts)
        except Exception as e:
            st.warning(f"Skipping {url.split('/')[-1]}: {e}")
        finally:
            progress.progress(i / len(urls))
    progress.empty()

    if not frames:
        raise RuntimeError("No monthly data could be loaded. The files may have changed schema or were unreachable.")
    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.groupby("ts", as_index=False)["cnt"].sum().sort_values("ts")
    # Derivations
    hourly["date"] = hourly["ts"].dt.date
    hourly["hour"] = hourly["ts"].dt.hour
    hourly["weekday_name"] = hourly["ts"].dt.day_name()
    hourly["month"] = hourly["ts"].dt.to_period("M").astype(str)
    source = f"Citi Bike NYC — last {len(urls)} month(s), discovery: {method}"
    return hourly, source

# -----------------------------------------------
# Page
# -----------------------------------------------
st.set_page_config(page_title="Trends • RidePulse", page_icon="📈", layout="wide")
st.title("📈 Trip Trends (recent data)")

with st.sidebar:
    st.header("Data source")
    st.caption("This view uses NYC Citi Bike monthly trip files aggregated to hourly counts.")
    months = st.slider("Months to load", min_value=1, max_value=6, value=3, help="More months = more data to download.")

# Fetch + normalize
try:
    with st.spinner("Loading latest Citi Bike trips…"):
        trips, source_label = fetch_citibike_recent_hourly(n_months=months)
except Exception as e:
    st.error(f"Could not fetch data: {e}")
    st.info("Please try again later.")
    st.stop()

with st.container(border=True):
    st.write("Data source")
    st.write(f"- {source_label}")

# Date range filter (use Python date objects)
min_date = min(trips["date"]) if len(trips) else None
max_date = max(trips["date"]) if len(trips) else None

if not min_date or not max_date:
    st.error("Dataset has no date values to filter.")
    st.stop()

# Default to last ~120 days, but clamp by data availability
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
tab_sys, tab_profiles = st.tabs(["System trends", "Usage profiles"])

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

    heat = (trips.groupby(["weekday_name", "hour"])["cnt"].median().reset_index())
    heat["weekday_name"] = pd.Categorical(
        heat["weekday_name"],
        categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
        ordered=True,
    )
    heat_wide = heat.pivot(index="weekday_name", columns="hour", values="cnt")
    fig_heat = px.imshow(heat_wide, color_continuous_scale="Turbo", aspect="auto", origin="lower",
                         title="Heatmap: rides by hour and weekday (median)")
    fig_heat.update_layout(height=420, coloraxis_colorbar=dict(title="Rides"))
    st.plotly_chart(_apply_theme(fig_heat), use_container_width=True, theme="streamlit")
    with st.container(border=True):
        st.write("What this shows")
        st.caption("When during the week riding is most intense. Bright cells = busier periods.")

st.markdown("---")
st.subheader("Notes")
st.write(
    "- Source: Citi Bike monthly trip history (public S3). We aggregate to hourly counts for recent months.\n"
    "- If data discovery fails (e.g., S3 listing blocked), please try again later."
)