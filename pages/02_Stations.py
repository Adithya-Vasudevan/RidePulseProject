import math
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk

from utils.gbfs import merged_station_frame
from utils.badges import award_badge, track_page_visit

st.set_page_config(page_title="Stations • RidePulse NYC", page_icon="🏙️", layout="wide")
track_page_visit("Stations")
award_badge("station_sage")

st.title("🏙️ Stations Explorer")

df = merged_station_frame()
if df is None or df.empty:
    st.warning("No station data available yet. Try refreshing.")
    st.stop()

# Basic filters/search
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    q = st.text_input("Search by name", placeholder="Type part of a station name…").strip().lower()
with c2:
    min_bikes = st.number_input("Min bikes", 0, 200, 0, step=1)
with c3:
    min_docks = st.number_input("Min docks", 0, 200, 0, step=1)

flt = df.copy()
if q:
    flt = flt[flt["name"].str.lower().str.contains(q, na=False)]
flt = flt[
    (flt["num_bikes_available"].fillna(0) >= min_bikes) &
    (flt["num_docks_available"].fillna(0) >= min_docks)
].reset_index(drop=True)

st.caption(f"{len(flt)} stations match filters")

# Station selector + details
left, right = st.columns([1, 1])

with left:
    names = sorted(flt["name"].dropna().unique().tolist())
    sel = st.selectbox("Select a station", names, index=0 if names else None)
    if sel:
        row = flt[flt["name"] == sel].head(1).iloc[0]
        bikes = int(row.get("num_bikes_available", 0))
        docks = int(row.get("num_docks_available", 0))
        capacity = max(bikes + docks, 1)
        k1, k2, k3 = st.columns(3)
        k1.metric("Bikes", f"{bikes}")
        k2.metric("Docks", f"{docks}")
        k3.metric("Utilization", f"{(bikes/capacity)*100:.1f}%")
        st.caption(f"Last reported (provider): {row.get('last_reported', '')}")

        comp = pd.DataFrame({"Type": ["Bikes", "Docks"], "Count": [bikes, docks]})
        fig = px.pie(comp, values="Count", names="Type", title="Composition", hole=0.45,
                     color_discrete_sequence=["#2563eb", "#10b981"])
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Map")
    if sel:
        row = flt[flt["name"] == sel].head(1).iloc[0]
        if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": row["lat"], "lon": row["lon"], "name": row["name"]}],
                get_position="[lon, lat]",
                get_fill_color=[63, 185, 80, 200],
                get_radius=80,
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=row["lat"], longitude=row["lon"], zoom=14, pitch=30)
            tooltip = {"html": "<b>{name}</b>", "style": {"backgroundColor": "rgba(13,17,23,0.9)", "color": "white"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
        else:
            st.info("Selected station has no coordinates.")
    else:
        st.info("Select a station to view its location and details.")

st.markdown("### Table")
show_cols = ["name", "num_bikes_available", "num_docks_available", "capacity", "percent_full", "last_reported", "lat", "lon"]
for c in show_cols:
    if c not in flt.columns:
        flt[c] = pd.NA
tbl = flt[show_cols].copy()
tbl["percent_full"] = (tbl["percent_full"].fillna(0) * 100).round(1)
st.dataframe(tbl.rename(columns={"percent_full": "% Full"}), use_container_width=True, height=420)