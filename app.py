# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from streamlit.components.v1 import html

# Project utilities
from utils.gbfs import merged_station_frame, record_snapshot_if_due, get_snapshot_history
from utils.plots import kpi_cards, short_term_trend_chart
from utils.helpers import human_time
from utils.badges import init_badges, render_badges
from utils.theme import inject_css

# ----------------------------- Page config -----------------------------
st.set_page_config(
    page_title="RidePulse NYC — Live Bike Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styles (fix cropped badge + tighter grid) -----------------------------
inject_css()
st.markdown(
    """
    <style>
      /* Increase top padding to avoid cropping the top-most line/badge */
      .main .block-container { padding-top: 1.25rem; max-width: 1400px; }
      .rp-gap { margin-top: 0.75rem; }

      /* HERO */
      .hero {
        border-radius: 14px;
        padding: 16px 18px 14px 18px; /* extra top/bottom to avoid clipping */
        background: linear-gradient(120deg, #0ea5e9 0%, #10b981 50%, #a78bfa 100%);
        background-size: 180% 180%;
        animation: rpGradientShift 12s ease infinite;
        color: #0b1020;
      }
      @keyframes rpGradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
      }
      .hero-row {
        display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      }
      .hero-badge {
        display: inline-flex; gap: 8px; align-items: center;
        border-radius: 999px; padding: 6px 10px;
        background: rgba(255,255,255,0.16); color: #fff; font-weight: 700;
        line-height: 1.1;
      }
      .pulse-dot {
        width: 10px; height: 10px; background: #22c55e; border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(34,197,94,0.7); animation: rpPulse 2s infinite;
      }
      @keyframes rpPulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.7); }
        70% { box-shadow: 0 0 0 16px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
      }
      .hero-title {
        font-weight: 800; font-size: 1.5rem; color: #ffffff;
        line-height: 1.35; margin: 0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.25);
      }
      .hero-meta {
        color: #eef6ff; margin-top: 2px; line-height: 1.35;
      }

      /* Compact status banner */
      .rp-status {
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 0.9rem;
      }

      /* Chip row (overview-only insights) */
      .chip {
        border: 1px solid rgba(148,163,184,0.25);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 0.9rem;
        background: rgba(255,255,255,0.04);
      }

      /* Marquee ticker */
      @keyframes rpMarquee {
        0%   { transform: translateX(0%); }
        100% { transform: translateX(-100%); }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Helpers -----------------------------
def pick_col(dframe: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in dframe.columns:
            return c
    return None

@dataclass
class ColRefs:
    name: Optional[str]
    lat: Optional[str]
    lon: Optional[str]
    bikes: Optional[str]
    docks: Optional[str]
    capacity: Optional[str]
    is_installed: Optional[str]
    is_renting: Optional[str]
    is_returning: Optional[str]
    last_reported: Optional[str]

def get_colrefs(dframe: pd.DataFrame) -> ColRefs:
    return ColRefs(
        name=pick_col(dframe, ("station_name", "name", "station", "title")),
        lat=pick_col(dframe, ("lat", "latitude", "station_latitude")),
        lon=pick_col(dframe, ("lon", "lng", "longitude", "station_longitude")),
        bikes=pick_col(dframe, ("num_bikes_available", "bikes_available", "bikes")),
        docks=pick_col(dframe, ("num_docks_available", "docks_available", "docks")),
        capacity=pick_col(dframe, ("capacity", "total_docks", "num_docks_total")),
        is_installed=pick_col(dframe, ("is_installed",)),
        is_renting=pick_col(dframe, ("is_renting",)),
        is_returning=pick_col(dframe, ("is_returning",)),
        last_reported=pick_col(dframe, ("last_reported", "last_updated", "updated_at")),
    )

def derive(df_in: pd.DataFrame, cols: ColRefs) -> pd.DataFrame:
    df2 = df_in.copy()
    # Capacity fallback
    if cols.capacity is None:
        if cols.bikes and cols.docks:
            df2["_capacity"] = pd.to_numeric(df2[cols.bikes], errors="coerce") + pd.to_numeric(df2[cols.docks], errors="coerce")
            cap_col = "_capacity"
        else:
            df2["_capacity"] = np.nan
            cap_col = "_capacity"
    else:
        cap_col = cols.capacity

    # Numeric coercions
    if cols.bikes:
        df2[cols.bikes] = pd.to_numeric(df2[cols.bikes], errors="coerce")
    if cols.docks:
        df2[cols.docks] = pd.to_numeric(df2[cols.docks], errors="coerce")
    df2[cap_col] = pd.to_numeric(df2[cap_col], errors="coerce")

    # Utilization
    if cols.bikes:
        denom = df2[cap_col].replace(0, np.nan)
        df2["_util"] = df2[cols.bikes] / denom
    else:
        df2["_util"] = np.nan

    # Online/offline (best-effort)
    def to_bool(series_name: Optional[str]) -> Optional[pd.Series]:
        if not series_name or series_name not in df2.columns:
            return None
        s = df2[series_name]
        if s.dtype == bool:
            return s
        try:
            return s.astype(float).fillna(1.0) > 0.0
        except Exception:
            return None

    b_inst = to_bool(cols.is_installed)
    b_rent = to_bool(cols.is_renting)
    b_ret  = to_bool(cols.is_returning)
    if b_rent is not None and b_ret is not None:
        df2["_online"] = b_rent & b_ret
    elif b_inst is not None:
        df2["_online"] = b_inst
    else:
        df2["_online"] = True

    # Timestamp normalization (tz-aware/naive/numeric)
    from pandas.api.types import is_numeric_dtype
    if cols.last_reported and cols.last_reported in df2.columns:
        s = df2[cols.last_reported]
        try:
            if is_numeric_dtype(s):
                dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            else:
                dt = pd.to_datetime(s, utc=True, errors="coerce")
        except Exception:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
        df2["_last_reported_dt"] = dt
    else:
        df2["_last_reported_dt"] = pd.NaT

    df2["_capacity_col"] = cap_col
    return df2

def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def last_two_totals_from_hist(hist: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract (bikes_now, bikes_prev, docks_now, docks_prev) from history if possible.
    """
    if hist is None:
        return None
    try:
        if isinstance(hist, pd.DataFrame):
            dfh = hist.copy()
            bikes_cands = [c for c in dfh.columns if ("bike" in c.lower() and "avail" in c.lower()) or ("total_bikes" in c.lower())]
            docks_cands = [c for c in dfh.columns if ("dock" in c.lower() and "avail" in c.lower()) or ("total_docks" in c.lower())]
            t_cands = [c for c in dfh.columns if c.lower() in ("ts", "time", "timestamp", "datetime")]
            bcol = bikes_cands[0] if bikes_cands else None
            dcol = docks_cands[0] if docks_cands else None
            if bcol is None and dcol is None:
                return None
            if t_cands:
                dfh = dfh.sort_values(t_cands[0])
            if len(dfh) < 2:
                return None
            b_now = float(dfh[bcol].iloc[-1]) if bcol else np.nan
            b_prev = float(dfh[bcol].iloc[-2]) if bcol else np.nan
            d_now = float(dfh[dcol].iloc[-1]) if dcol else np.nan
            d_prev = float(dfh[dcol].iloc[-2]) if dcol else np.nan
            return (b_now, b_prev, d_now, d_prev)
        if isinstance(hist, (list, tuple)) and len(hist) >= 2:
            def pick(d: dict, keys: Tuple[str, ...]) -> Optional[float]:
                for k in keys:
                    if k in d and d[k] is not None:
                        try:
                            return float(d[k])
                        except Exception:
                            pass
                return None
            now = hist[-1]
            prev = hist[-2]
            b_now = pick(now, ("total_bikes", "bikes", "bikes_available", "num_bikes_available"))
            b_prev = pick(prev, ("total_bikes", "bikes", "bikes_available", "num_bikes_available"))
            d_now = pick(now, ("total_docks", "docks", "docks_available", "num_docks_available"))
            d_prev = pick(prev, ("total_docks", "docks", "docks_available", "num_docks_available"))
            if b_now is None and d_now is None:
                return None
            return (b_now or np.nan, b_prev or np.nan, d_now or np.nan, d_prev or np.nan)
    except Exception:
        return None
    return None

# Initialize achievements/badges storage (if needed by your badges system)
init_badges()

# ----------------------------- Sidebar (Explore removed) -----------------------------
with st.sidebar:
    st.title("🚲 RidePulse NYC")
    st.caption("Live Bike Intelligence • Citi Bike GBFS")
    render_badges()
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        refresh = st.button("🔄 Refresh", use_container_width=True)
    with c2:
        auto = st.toggle("Auto", value=False, help="Refresh every 60 seconds")
    st.caption("Use the Pages section above to navigate.")

# Lightweight auto-refresh
if auto:
    html(
        """
        <script>
          if (!window.rpAutoRefreshSet) {
            window.rpAutoRefreshSet = true;
            setTimeout(() => window.parent.location.reload(), 60000);
          }
        </script>
        """,
        height=0,
    )

# ----------------------------- Hero (badge inside to avoid cropping) -----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-row">
        <div class="hero-badge"><span class="pulse-dot"></span> Live GBFS</div>
        <div class="hero-title">RidePulse NYC — Live Bike Intelligence</div>
      </div>
      <div class="hero-meta">Fast overview only. Deep charts and maps live in the tabs/pages.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Data fetch + snapshot -----------------------------
t0 = time.perf_counter()
with st.spinner("Fetching live availability…"):
    df = merged_station_frame(force=refresh)
if df is None or len(df) == 0:
    st.error("No station data received. The GBFS feed may be temporarily unavailable.")
    st.stop()
try:
    record_snapshot_if_due(df)
except Exception:
    pass
ms = (time.perf_counter() - t0) * 1000.0
st.markdown(f'<div class="rp-status rp-gap">✅ Ready • fetched in {ms:.0f} ms</div>', unsafe_allow_html=True)

# ----------------------------- Derive + columns -----------------------------
cols = get_colrefs(df)
dfd = derive(df, cols)

# KPI row (compact)
k1, k2, k3, k4 = st.columns(4)
try:
    kpi_cards(dfd, k1, k2, k3, k4)
except Exception:
    with k1: st.metric("Stations", f"{len(dfd):,}")
    with k2: st.metric("Bikes available", "—")
    with k3: st.metric("Docks available", "—")
    with k4:
        pct_online = 100.0 * (dfd["_online"].mean() if "_online" in dfd.columns else 1.0)
        st.metric("Stations online", f"{pct_online:.1f}%")

# ----------------------------- Overview-only insights (not in other tabs) -----------------------------
chips = st.columns(3)
# Feed freshness (age of newest last_reported)
with chips[0]:
    try:
        if "_last_reported_dt" in dfd.columns:
            latest_dt = pd.to_datetime(dfd["_last_reported_dt"]).max()
            if pd.isna(latest_dt):
                st.markdown('<div class="chip">🕒 Feed freshness: unknown</div>', unsafe_allow_html=True)
            else:
                now = datetime.now(timezone.utc)
                age_s = (now - latest_dt.to_pydatetime()).total_seconds()
                def human_duration(seconds: float) -> str:
                    seconds = max(0, int(round(seconds)))
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    if h: return f"{h}h {m}m"
                    if m: return f"{m}m {s}s"
                    return f"{s}s"
                st.markdown(f'<div class="chip">🕒 Feed freshness: {human_duration(age_s)} ago</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chip">🕒 Feed freshness: unavailable</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="chip">🕒 Feed freshness: unavailable</div>', unsafe_allow_html=True)

# Delta since previous snapshot
with chips[1]:
    delta_html = '<div class="chip">Δ since last snapshot: unavailable</div>'
    try:
        hist = get_snapshot_history()
        last2 = last_two_totals_from_hist(hist)
        if last2:
            b_now, b_prev, d_now, d_prev = last2
            db = b_now - b_prev if np.isfinite(b_now) and np.isfinite(b_prev) else np.nan
            dd = d_now - d_prev if np.isfinite(d_now) and np.isfinite(d_prev) else np.nan
            arrow_b = "▲" if db > 0 else ("▼" if db < 0 else "➖")
            arrow_d = "▲" if dd > 0 else ("▼" if dd < 0 else "➖")
            delta_html = f'<div class="chip">Δ since last snapshot: {arrow_b} Bikes {int(db) if np.isfinite(db) else "—"} • {arrow_d} Docks {int(dd) if np.isfinite(dd) else "—"}</div>'
    except Exception:
        pass
    st.markdown(delta_html, unsafe_allow_html=True)

# Quick data integrity: stations with stale timestamps (>30 min)
with chips[2]:
    try:
        stale_ct = 0
        if "_last_reported_dt" in dfd.columns:
            now = datetime.now(timezone.utc)
            ages = (now - pd.to_datetime(dfd["_last_reported_dt"]).dt.tz_convert("UTC")).dt.total_seconds()
            stale_ct = int((ages > 1800).sum())
        st.markdown(f'<div class="chip">🧪 Data integrity: {stale_ct} station(s) > 30m stale</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="chip">🧪 Data integrity: n/a</div>', unsafe_allow_html=True)

# ----------------------------- Content row: Micro‑trend | Mini map -----------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("📈 Micro‑trend (last few snapshots)")
    hist = None
    try:
        hist = get_snapshot_history()
    except Exception:
        hist = None

    ok_hist = False
    try:
        ok_hist = hist is not None and len(hist) >= 2
    except Exception:
        ok_hist = False

    if not ok_hist:
        st.info("Collecting snapshots… trends appear once two or more points are available.")
    else:
        try:
            t = short_term_trend_chart(hist)
            fig = t["fig"]
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Trend temporarily unavailable: {e}")

with right:
    st.subheader("🗺️ Mini live map")
    lat = pick_col(dfd, ("lat", "latitude", "station_latitude"))
    lon = pick_col(dfd, ("lon", "lng", "longitude", "station_longitude"))
    bikes_col = pick_col(dfd, ("num_bikes_available", "bikes_available", "bikes"))
    name_col = pick_col(dfd, ("station_name", "name", "station", "title"))

    if lat and lon and lat in dfd.columns and lon in dfd.columns:
        dfm = dfd[[c for c in [lat, lon, bikes_col, name_col] if c]].dropna().copy()
        dfm = dfm.sample(min(len(dfm), 600), random_state=42) if len(dfm) > 600 else dfm

        v = dfm[bikes_col].fillna(0).astype(float).to_numpy() if bikes_col else np.zeros(len(dfm))
        vmax = float(np.nanmax(v)) if v.size else 0.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        R = np.clip(255.0 - (v / vmax) * 155.0, 100.0, 255.0)
        G = np.clip(100.0 + (v / vmax) * 155.0, 100.0, 255.0)
        B = np.full_like(R, 140.0)
        color = np.column_stack([R, G, B])
        color = np.clip(np.nan_to_num(color, nan=128.0), 0, 255).astype(np.uint8)

        dfm["_r"], dfm["_g"], dfm["_b"] = color[:, 0], color[:, 1], color[:, 2]
        dfm["_radius"] = 25.0 + 2.0 * (v if v.size else 0.0)

        try:
            center_lat = float(dfm[lat].astype(float).mean())
            center_lon = float(dfm[lon].astype(float).mean())
        except Exception:
            center_lat, center_lon = 40.73, -73.99

        tooltip_lines = []
        if name_col: tooltip_lines.append(f"<b>{{{{{name_col}}}}}</b>")
        if bikes_col: tooltip_lines.append(f"Bikes: {{{{{bikes_col}}}}}")
        tooltip_html = f"<div style='font-size:12px;'>{'<br/>'.join(tooltip_lines)}</div>" if tooltip_lines else "Station"

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=dfm,
            get_position=[lon, lat],
            get_radius="_radius",
            get_fill_color=["_r", "_g", "_b", 185],
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"html": tooltip_html, "style": {"color": "white"}},
            height=360,  # set height here (st.pydeck_chart has no height arg)
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption("Full live map and filters are available in the Map/Stations pages.")
    else:
        st.info("Map preview unavailable (lat/lon not found). See Map/Stations pages for the full view.")

# ----------------------------- Running notification (ticker) -----------------------------
try:
    top5 = None
    s_col = cols.name
    b_col = cols.bikes
    if s_col and b_col and s_col in dfd.columns and b_col in dfd.columns:
        top5 = (
            dfd[[s_col, b_col]]
            .dropna()
            .sort_values(b_col, ascending=False)
            .head(5)
            .values.tolist()
        )
    if top5:
        items = " • ".join([f"🚲 {s} — {int(b)} bikes" for s, b in top5])
        st.markdown(
            f"""
            <div style="margin-top:8px; margin-bottom:6px; white-space: nowrap; overflow: hidden; position: relative; border: 1px solid rgba(148,163,184,0.25); border-radius: 10px; padding: 6px 0 4px 0;">
              <div style="display:inline-block; padding-left:100%; animation: rpMarquee 22s linear infinite;">
                {items} • {items}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ----------------------------- Footer -----------------------------
st.markdown("#### 🗓 Last Updated")
st.caption(f"{human_time(datetime.utcnow())} UTC • Snapshots are recorded about once per minute when refreshed.")
st.success("Use the sidebar to jump into Stations (deep dive), Trends (full analytics), and more.")

if 'refresh' in locals() and refresh:
    st.toast("Live data refreshed", icon="✅")
    try:
        st.balloons()
    except Exception:
        pass