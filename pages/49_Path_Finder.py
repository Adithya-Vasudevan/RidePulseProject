# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import streamlit as st
import pandas as pd
import pydeck as pdk

from utils.gbfs import merged_station_frame
from utils.routing import (
    build_station_graph,
    path_metrics,
    bfs_steps,
    dijkstra_steps,
    astar_steps,
    greedy_best_first_steps,
)

st.set_page_config(page_title="Path Finder", page_icon="🧭", layout="wide")
st.title("🧭 Path Finder (GBFS Stations)")

# -------------------------- Input Guide (clarity) --------------------------
with st.expander("Input guide (what you need to choose)", expanded=True):
    st.markdown(
        """
        - From station: The station where you start.
        - To station: The station where you want to end.
        - Graph options (left sidebar):
          - Use only online stations: Use only stations currently renting and returning bikes.
          - k-nearest neighbors per station: How many nearby station links to keep per station (higher = denser graph).
          - Max edge length (km): Maximum distance allowed for a direct link between stations.
          - Assumed cycling speed (km/h): Used to estimate time in minutes.
        - Optimize for:
          - time: fastest by estimated minutes.
          - distance: shortest by kilometers.
          - stops: fewest station-to-station hops.
        - Note: It may take up to a couple of minutes while we prepare the best path for you.
        """
    )

st.caption("Uses the live GBFS station data.")

# Sidebar controls
with st.sidebar:
    st.subheader("Graph options")
    use_only_online = st.toggle(
        "Use only online stations",
        value=False,
        help="Include stations that are currently renting and returning.",
    )
    k_neighbors = st.slider(
        "k-nearest neighbors per station",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
        help="Each station connects to this many nearest neighbors (within max edge length).",
    )
    max_edge_km = st.slider(
        "Max edge length (km)",
        min_value=0.3,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Maximum allowed distance for a direct connection between stations.",
    )
    avg_speed_kmh = st.slider(
        "Assumed cycling speed (km/h)",
        min_value=8.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="Used to translate distance to time for the route metrics.",
    )

# Load stations and build graph
with st.spinner("Loading live stations…"):
    df = merged_station_frame()

if df is None or len(df) == 0:
    st.error("No station data available from GBFS.")
    st.stop()

try:
    nodes, adj = build_station_graph(
        df,
        use_only_online=use_only_online,
        k_neighbors=int(k_neighbors),
        max_edge_km=float(max_edge_km),
        avg_speed_kmh=float(avg_speed_kmh),
    )
except Exception as e:
    st.error(f"Could not build graph: {e}")
    st.stop()

if not nodes or not adj:
    st.warning("Graph is empty. Try increasing k-neighbors or the max edge length.")
    st.stop()

# Station selectors
id_to_label = {n.id: f"{n.name} ({n.id})" if n.name != n.id else n.id for n in nodes.values()}
sorted_ids = sorted(id_to_label.keys(), key=lambda i: id_to_label[i].lower())

c1, c2 = st.columns([1, 1])
with c1:
    src = st.selectbox("From station (required)", options=sorted_ids, format_func=lambda i: id_to_label[i], key="src_global")
with c2:
    dst = st.selectbox("To station (required)", options=sorted_ids, index=min(1, len(sorted_ids)-1), format_func=lambda i: id_to_label[i], key="dst_global")

if not src or not dst:
    st.info("Please select both a start and a destination station to proceed.")

if src and dst and src == dst:
    st.error("Start and destination must be different.")

# ------------------------------ Helpers ------------------------------
def path_to_lonlat(path_ids):
    return [[nodes[sid].lon, nodes[sid].lat] for sid in path_ids]

def segment_midpoint(a_lon, a_lat, b_lon, b_lat):
    return (a_lon + b_lon) / 2.0, (a_lat + b_lat) / 2.0

def path_midpoint_lonlat(path_ids):
    if not path_ids:
        return None, None
    idx = len(path_ids) // 2
    sid = path_ids[idx]
    return nodes[sid].lon, nodes[sid].lat

def default_view_state():
    df_nodes = pd.DataFrame([{"lat": n.lat, "lon": n.lon} for n in nodes.values()])
    return pdk.ViewState(
        latitude=float(df_nodes["lat"].mean()),
        longitude=float(df_nodes["lon"].mean()),
        zoom=12,
        pitch=0,
    )

def _mercator_y(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return (1 - math.log(math.tan(lat_rad / 2 + math.pi / 4)) / math.pi) / 2

def fit_view_state_for_points(points_lonlat, width_px=1100, height_px=650, padding=0.15):
    # points_lonlat: list of [lon, lat]
    if not points_lonlat:
        return default_view_state()
    lons = [p[0] for p in points_lonlat]
    lats = [p[1] for p in points_lonlat]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    lon_range = max(max_lon - min_lon, 1e-6)
    x0_range = (lon_range / 360.0) * 256.0
    y0_min = _mercator_y(max_lat) * 256.0
    y0_max = _mercator_y(min_lat) * 256.0
    y0_range = abs(y0_max - y0_min)
    usable_w = max(width_px * (1 - 2 * padding), 1)
    usable_h = max(height_px * (1 - 2 * padding), 1)
    z_lon = math.log2(max(usable_w / x0_range, 1e-6))
    z_lat = math.log2(max(usable_h / max(y0_range, 1e-9), 1e-6))
    zoom = max(min(z_lon, z_lat, 20.0), 1.0)

    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=float(zoom), pitch=0)

# Session state for map and results
if "pf_view_state" not in st.session_state:
    st.session_state["pf_view_state"] = default_view_state()
if "pf_best_path" not in st.session_state:
    st.session_state["pf_best_path"] = []

# ------------------------------- Tabs --------------------------------
tab1, tab2 = st.tabs(["Route Planner", "Algorithms Explorer"])

with tab1:
    st.subheader("🗺️ Route Planner")
    st.caption("Find the best route. The map will auto-center on the chosen path after you submit.")

    colA, colB = st.columns([1, 1])
    with colA:
        opt_for = st.radio(
            "Optimize for",
            options=["time", "distance", "stops"],
            index=0,
            horizontal=True,
            key="planner_opt_for",
            help="Choose the objective for the best route.",
        )
    with colB:
        st.empty()  # placeholder to keep layout balanced

    # Label toggles
    c6, c7, c8 = st.columns([1, 1, 1])
    with c6:
        show_segment_times = st.toggle(
            "Show segment times on best route",
            value=True,
            help="Label each leg (station to station) with minutes.",
            key="show_segment_times",
        )
    with c7:
        show_stop_cumulative = st.toggle(
            "Show cumulative time at each stop",
            value=True,
            help="Label each stop with total minutes from start.",
            key="show_stop_cum",
        )
    with c8:
        show_total_time_badge = st.toggle(
            "Show total time badge",
            value=True,
            help="Display total minutes for the best route.",
            key="show_total_badge",
        )

    run_plan = st.button("Find best route", type="primary", key="planner_run")

    best_path = st.session_state.get("pf_best_path", [])
    steps = []

    if run_plan and src and dst and src != dst:
        # Best route
        steps, best_path = astar_steps(
            adj=adj,
            nodes=nodes,
            source=src,
            target=dst,
            optimize_for=opt_for,
            avg_speed_kmh=float(avg_speed_kmh),
        )
        st.session_state["pf_best_path"] = best_path

        # Auto-center on best route
        if best_path:
            st.session_state["pf_view_state"] = fit_view_state_for_points(path_to_lonlat(best_path))

        # Summary
        st.subheader("Best route")
        if not best_path:
            st.error("No route found between the selected stations.")
        else:
            m = path_metrics(adj, best_path)
            st.success(
                f"Path: {' → '.join([nodes[i].name for i in best_path])} | "
                f"Stops: {m['stops']} • Distance: {m['distance_km']} km • Time: {m['time_min']} min"
            )

    st.divider()
    st.subheader("Map")

    # Legend
    st.markdown(
        """
        <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
          <div><span style="display:inline-block;width:14px;height:14px;background:#a0a0a0;opacity:0.5;margin-right:6px;border-radius:2px;"></span>Graph edges</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#c61e1e;margin-right:6px;border-radius:2px;"></span>Best route</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#228b22;margin-right:6px;border-radius:50%;"></span>Start</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#b03060;margin-right:6px;border-radius:50%;"></span>Stop (intermediate)</div>
          <div><span style="display:inline-block;width:14px;height:14px;background:#b22222;margin-right:6px;border-radius:50%;"></span>End</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        # Node DF (for base points if needed)
        nt = pd.DataFrame([{"id": n.id, "name": n.name, "lat": n.lat, "lon": n.lon} for n in nodes.values()])

        # Base graph edges
        edges = []
        for u, nbrs in adj.items():
            for v in nbrs.keys():
                if u < v:
                    edges.append(
                        {"from_lat": nodes[u].lat, "from_lon": nodes[u].lon, "to_lat": nodes[v].lat, "to_lon": nodes[v].lon}
                    )
        et = pd.DataFrame(edges)

        # Best path data
        best_path_ids = st.session_state.get("pf_best_path", [])
        best_path_df = pd.DataFrame()
        best_segments_text = pd.DataFrame()
        best_nodes_df = pd.DataFrame()
        total_badge_df = pd.DataFrame()

        if best_path_ids:
            # Path layer row
            best_metrics = path_metrics(adj, best_path_ids)
            best_path_df = pd.DataFrame(
                [
                    {
                        "path": path_to_lonlat(best_path_ids),
                        "label": "Best route",
                        "stops": best_metrics["stops"],
                        "distance_km": best_metrics["distance_km"],
                        "time_min": best_metrics["time_min"],
                    }
                ]
            )

            # Segment times text at segment midpoints
            if show_segment_times:
                seg_rows = []
                for a, b in zip(best_path_ids[:-1], best_path_ids[1:]):
                    u = nodes[a]
                    v = nodes[b]
                    tmin = float(adj[a][b]["time_min"])
                    mid_lon, mid_lat = segment_midpoint(u.lon, u.lat, v.lon, v.lat)
                    seg_rows.append({"lon": mid_lon, "lat": mid_lat, "text": f"{tmin:.1f} min"})
                if seg_rows:
                    best_segments_text = pd.DataFrame(seg_rows)

            # Stop circles with cumulative time
            if show_stop_cumulative:
                cum = 0.0
                stop_rows = []
                for idx, sid in enumerate(best_path_ids):
                    if idx > 0:
                        prev = best_path_ids[idx - 1]
                        cum += float(adj[prev][sid]["time_min"])
                    role = "start" if idx == 0 else ("end" if idx == len(best_path_ids) - 1 else "mid")
                    color = [34, 139, 34, 230] if role == "start" else ([178, 34, 34, 230] if role == "end" else [176, 48, 96, 220])
                    stop_rows.append(
                        {
                            "lon": nodes[sid].lon,
                            "lat": nodes[sid].lat,
                            "role": role,
                            "color": color,
                            "text": f"{nodes[sid].name}\n{cum:.1f} min",
                        }
                    )
                if stop_rows:
                    best_nodes_df = pd.DataFrame(stop_rows)

            # Total time badge
            if show_total_time_badge:
                mid_lon, mid_lat = path_midpoint_lonlat(best_path_ids)
                total_badge_df = pd.DataFrame(
                    [{"lon": mid_lon, "lat": mid_lat, "text": f"Total: {best_metrics['time_min']} min"}]
                )

        # Build deck layers (order matters)
        layers = []

        # Base graph edges
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=et,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_color=[160, 160, 160, 60],
                get_width=1,
                pickable=False,
            )
        )

        # Best route (red)
        if not best_path_df.empty:
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    data=best_path_df,
                    get_path="path",
                    get_color=[200, 30, 30, 230],
                    get_width=7,
                    width_min_pixels=6,
                    width_max_pixels=12,
                    pickable=True,
                )
            )
            # Segment time labels
            if not best_segments_text.empty:
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=best_segments_text,
                        get_position=["lon", "lat"],
                        get_text="text",
                        get_color=[200, 30, 30, 230],
                        get_size=13,
                        get_text_anchor="middle",
                        get_alignment_baseline="bottom",
                    )
                )
            # Stop circles with cumulative labels
            if not best_nodes_df.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=best_nodes_df,
                        get_position=["lon", "lat"],
                        get_fill_color="color",
                        get_radius=55,
                        radius_min_pixels=5,
                        radius_max_pixels=12,
                        pickable=True,
                    )
                )
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=best_nodes_df,
                        get_position=["lon", "lat"],
                        get_text="text",
                        get_color=[40, 40, 40, 230],
                        get_size=12,
                        get_text_anchor="start",
                        get_alignment_baseline="center",
                    )
                )
            # Total time badge
            if not total_badge_df.empty:
                layers.append(
                    pdk.Layer(
                        "TextLayer",
                        data=total_badge_df,
                        get_position=["lon", "lat"],
                        get_text="text",
                        get_color=[0, 0, 0, 255],
                        get_size=16,
                        get_text_anchor="middle",
                        get_alignment_baseline="top",
                    )
                )

        tooltip = {"text": "Route: {label}\nTime: {time_min} min\nDistance: {distance_km} km\nStops: {stops}"}

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=st.session_state["pf_view_state"],
                layers=layers,
                tooltip=tooltip,
            )
        )
    except Exception as e:
        st.warning(f"Map rendering issue: {e}")

with tab2:
    st.subheader("🧭 Algorithms Explorer")
    st.caption("Visualize BFS, Dijkstra, A*, and Greedy Best-First step-by-step with clear status and colors.")

    algo = st.selectbox(
        "Algorithm (choose one)",
        options=["BFS (fewest stops)", "Dijkstra", "A*", "Greedy Best-First"],
        index=2,
        key="algo_select",
    )
    opt_for_alg = st.radio(
        "Objective",
        options=["time", "distance", "stops"],
        index=0,
        horizontal=True,
        key="alg_opt_for",
        help="Affects Dijkstra/A*/Greedy; BFS always optimizes for 'stops'.",
    )
    max_steps = st.slider(
        "Max steps (safety cap)",
        min_value=1000,
        max_value=500000,
        value=100000,
        step=1000,
        key="alg_max_steps",
        help="Upper bound on the number of algorithm steps to record.",
    )

    run_search = st.button("Run search", type="primary", key="alg_run")

    steps = []
    path = []
    if run_search and src and dst and src != dst:
        with st.spinner("Running search…"):
            if algo.startswith("BFS"):
                steps, path = bfs_steps(adj, src, dst, max_steps=int(max_steps))
            elif algo.startswith("Dijkstra"):
                weight_key = {"time": "time_min", "distance": "distance_km", "stops": None}[opt_for_alg]
                if opt_for_alg == "stops":
                    adj_unit = {u: {v: {"weight": 1.0} for v in nbrs} for u, nbrs in adj.items()}
                    steps, path = dijkstra_steps(adj_unit, src, dst, weight_key="weight", max_steps=int(max_steps))
                else:
                    steps, path = dijkstra_steps(adj, src, dst, weight_key=weight_key or "weight", max_steps=int(max_steps))
            elif algo.startswith("A*"):
                steps, path = astar_steps(
                    adj=adj,
                    nodes=nodes,
                    source=src,
                    target=dst,
                    optimize_for=opt_for_alg,
                    avg_speed_kmh=float(avg_speed_kmh),
                    max_steps=int(max_steps),
                )
            else:
                steps, path = greedy_best_first_steps(
                    adj=adj,
                    nodes=nodes,
                    source=src,
                    target=dst,
                    optimize_for=opt_for_alg,
                    avg_speed_kmh=float(avg_speed_kmh),
                    max_steps=int(max_steps),
                )

    if steps:
        st.success(f"Steps: {len(steps)} | Path found: {'Yes' if path else 'No'}")
        step_idx = st.slider("Step", 0, max(0, len(steps) - 1), value=0, key="alg_step_idx")
        frame = steps[step_idx]

        # Map visualization for the current step (auto-center on final path if found)
        nt = pd.DataFrame([{"id": n.id, "name": n.name, "lat": n.lat, "lon": n.lon} for n in nodes.values()])

        relax_edges = [{"from_lat": nodes[u].lat, "from_lon": nodes[u].lon, "to_lat": nodes[v].lat, "to_lon": nodes[v].lon} for u, v in frame.relaxations]
        relax_df = pd.DataFrame(relax_edges)

        pe = [{"from_lat": nodes[a].lat, "from_lon": nodes[a].lon, "to_lat": nodes[b].lat, "to_lon": nodes[b].lon} for a, b in zip(path[:-1], path[1:])] if path else []
        pe_df = pd.DataFrame(pe)

        visited_ids = list(frame.visited)
        frontier_ids = [n for _, n in frame.frontier]
        current_id = frame.current

        def color_for(node_id: str):
            if node_id == src:
                return [34, 139, 34, 220]  # green
            if node_id == dst:
                return [200, 30, 30, 220]  # red
            if node_id == current_id:
                return [155, 89, 182, 230]  # purple
            if node_id in frontier_ids:
                return [241, 196, 15, 220]  # yellow
            if node_id in visited_ids:
                return [52, 152, 219, 200]  # blue
            return [150, 150, 150, 120]  # grey

        nt["color"] = nt["id"].apply(color_for)

        view_state_alg = default_view_state()
        if path:
            view_state_alg = fit_view_state_for_points([[nodes[i].lon, nodes[i].lat] for i in path])

        layers = [
            pdk.Layer(
                "LineLayer",
                data=relax_df,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_color=[255, 140, 0, 200],
                get_width=4,
                pickable=False,
            ),
            pdk.Layer(
                "LineLayer",
                data=pe_df,
                get_source_position=["from_lon", "from_lat"],
                get_target_position=["to_lon", "to_lat"],
                get_color=[200, 30, 30, 230],
                get_width=6,
                pickable=False,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=nt,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=50,
                pickable=True,
                radius_min_pixels=4,
                radius_max_pixels=10,
            ),
        ]
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=view_state_alg, layers=layers))
    elif run_search and (not src or not dst or src == dst):
        st.info("Pick two different stations first.")