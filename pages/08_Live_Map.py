from __future__ import annotations

import pandas as pd
import streamlit as st
import pydeck as pdk

from utils.gbfs import merged_station_frame

# Optional badges/analytics; degrade gracefully if missing
try:
    from utils.badges import award_badge, track_page_visit  # type: ignore
except Exception:
    def award_badge(_: str) -> None:
        return

    def track_page_visit(_: str) -> None:
        return

st.set_page_config(page_title="Live Map • RidePulse NYC", page_icon="🗺️", layout="wide")
track_page_visit("Live Map")
award_badge("cartographer")

st.title("🗺️ Live Map")

# Load data
df = merged_station_frame()
if df is None or df.empty:
    st.warning("Mapping unavailable: no station data yet. Try refreshing.")
    st.stop()

# Keep only rows with coordinates
df = df.dropna(subset=["lat", "lon"]).copy()
if df.empty:
    st.warning("No stations have coordinates available.")
    st.stop()

# Utilization helper
if "percent_full" not in df.columns:
    bikes = pd.to_numeric(df.get("num_bikes_available", 0), errors="coerce").fillna(0)
    docks = pd.to_numeric(df.get("num_docks_available", 0), errors="coerce").fillna(0)
    capacity = (bikes + docks).replace(0, 1)
    df["percent_full"] = (bikes / capacity).clip(0, 1)

# Fallback view: Times Square; otherwise center on medians
lat_med = df["lat"].median()
lon_med = df["lon"].median()
center_lat = float(lat_med) if pd.notna(lat_med) else 40.7580
center_lon = float(lon_med) if pd.notna(lon_med) else -73.9855

INITIAL_VIEW_STATE = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=12,
    pitch=30,
)

# Tabs: show both the utilization scatter and the Hex+Arcs map
tab1, tab2 = st.tabs(["Utilization Scatter", "Hex + Arcs (Rebalancing view)"])

with tab1:
    st.caption("Color encodes utilization (green = fuller, red = emptier). Size encodes station capacity.")
    # Precompute visuals to avoid JS expressions in accessors
    bikes = pd.to_numeric(df.get("num_bikes_available", 0), errors="coerce").fillna(0)
    docks = pd.to_numeric(df.get("num_docks_available", 0), errors="coerce").fillna(0)
    capacity = (bikes + docks).clip(lower=1)
    util = df["percent_full"].clip(0, 1)

    df_map = df.copy()
    df_map["radius"] = ((capacity).clip(lower=10) * 1.3).astype(float)
    # Green up when full, red when empty
    df_map["fill_r"] = ((1 - util) * 255).round().astype(int)
    df_map["fill_g"] = (util * 180).round().astype(int)
    df_map["fill_b"] = 60
    df_map["fill_a"] = 200

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[lon, lat]",
        get_fill_color="[fill_r, fill_g, fill_b, fill_a]",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>Bikes: {num_bikes_available}<br/>Docks: {num_docks_available}<br/>% Full: {percent_full}",
        "style": {"backgroundColor": "rgba(13,17,23,0.9)", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(layers=[scatter_layer], initial_view_state=INITIAL_VIEW_STATE, tooltip=tooltip),
        use_container_width=True,
    )

    with st.expander("Legend"):
        st.markdown("- Color: greener = higher utilization, redder = lower")
        st.markdown("- Size: proportional to station capacity")

with tab2:
    st.caption("Composite view: capacity heat (hex), station dots, and suggested rebalancing arcs.")
    # Prepare columns Deck.gl expects for the provided recipe
    df_deck = df.copy().rename(columns={"lat": "latitude", "lon": "longitude"})

    # Precompute scatter radius and color channels for stability
    df_deck["scatter_radius"] = (pd.to_numeric(df_deck.get("num_bikes_available", 0), errors="coerce").fillna(0) + 1) * 3
    util = df_deck["percent_full"].clip(0, 1)
    df_deck["s_r"] = (util * 255).round().astype(int)
    df_deck["s_g"] = 120
    df_deck["s_b"] = 200
    df_deck["s_a"] = 160

    # Controls
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        hex_radius = st.slider("Hex radius (meters)", min_value=80, max_value=300, value=150, step=10)
    with c2:
        pairs = st.slider("Rebalancing pairs", min_value=0, max_value=10, value=6, step=1)

    # Layers
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df_deck,
        get_position=["longitude", "latitude"],
        get_radius="scatter_radius",
        get_fill_color=["s_r", "s_g", "s_b", "s_a"],
        pickable=True,
        auto_highlight=True,
    )

    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=df_deck,
        get_position=["longitude", "latitude"],
        radius=hex_radius,
        elevation_scale=4,
        elevation_range=[0, 1200],
        extruded=True,
        coverage=0.8,
        get_weight="num_bikes_available",
    )

    # Build arcs: donors (most full) -> receivers (emptiest)
    donors = df_deck.sort_values("percent_full", ascending=False).head(pairs).reset_index(drop=True)
    receivers = df_deck.sort_values("percent_full", ascending=True).head(pairs).reset_index(drop=True)
    pair_n = min(len(donors), len(receivers))
    arcs_data = []
    for i in range(pair_n):
        d = donors.iloc[i]
        r = receivers.iloc[i]
        arcs_data.append(
            {
                "from_name": d.get("name", ""),
                "to_name": r.get("name", ""),
                "from_lon": float(d["longitude"]),
                "from_lat": float(d["latitude"]),
                "to_lon": float(r["longitude"]),
                "to_lat": float(r["latitude"]),
                # Width scaled by donor utilization (2..10)
                "width": int(max(2, min(10, float(d["percent_full"]) * 10))),
            }
        )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arcs_data,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_width="width",
        get_tilt=15,
        get_source_color=[255, 130, 0],
        get_target_color=[0, 160, 255],
        pickable=True,
    )

    tooltip2 = {
        "text": "{name}\n🚲 {num_bikes_available}  🅿️ {num_docks_available}",
        "style": {"backgroundColor": "rgba(13,17,23,0.9)", "color": "white"},
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[hex_layer, scatter, arc_layer],
            initial_view_state=pdk.ViewState(latitude=40.7580, longitude=-73.9855, zoom=11, pitch=35),
            tooltip=tooltip2,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
<div class="rp-fact" style="margin-top:10px;">
<b>Legend:</b> Hex elevation ~ local bike availability. Dots use fill level for color and bikes for size.
Arcs suggest rebalancing (near-full → near-empty).
</div>
""",
        unsafe_allow_html=True,
    )