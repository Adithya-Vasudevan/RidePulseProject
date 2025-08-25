from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

UA = {"User-Agent": "RidePulse/1.0 (+https://github.com/Adithya-Vasudevan)"}
REQUEST_TIMEOUT = 30

GBFS_INFO = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
GBFS_STATUS = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

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
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig

# ---------- Helpers ----------
def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def haversine_km_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _load_gbfs() -> Tuple[pd.DataFrame, int]:
    """
    Return stations dataframe and last_updated (status feed).
    Columns: station_id(str), name, lat, lon, capacity, is_renting, is_installed, bikes_available, docks_available
    """
    r1 = requests.get(GBFS_INFO, headers=UA, timeout=REQUEST_TIMEOUT)
    r1.raise_for_status()
    info = r1.json()

    r2 = requests.get(GBFS_STATUS, headers=UA, timeout=REQUEST_TIMEOUT)
    r2.raise_for_status()
    status = r2.json()

    info_df = pd.DataFrame(info["data"]["stations"])
    status_df = pd.DataFrame(status["data"]["stations"])
    last_updated = int(status.get("last_updated", 0))

    status_df = status_df.rename(columns={
        "num_bikes_available": "bikes_available",
        "num_docks_available": "docks_available",
        "is_renting": "is_renting",
        "is_installed": "is_installed",
    })

    df = info_df.merge(status_df[["station_id", "is_renting", "is_installed", "bikes_available", "docks_available"]],
                       on="station_id", how="left")
    df["is_renting"] = df["is_renting"].fillna(0).astype(int)
    df["is_installed"] = df["is_installed"].fillna(0).astype(int)
    df["bikes_available"] = df["bikes_available"].fillna(0).astype(int)
    df["docks_available"] = df["docks_available"].fillna(0).astype(int)
    df["station_id"] = df["station_id"].astype(str)
    return df, last_updated

@dataclass(frozen=True)
class GraphConfig:
    use_only_online: bool
    k_neighbors: int
    max_edge_km: float
    avg_speed_kmh: float

@dataclass
class GraphData:
    stations: pd.DataFrame
    edges: List[Tuple[str, str, float]]  # (u, v, base_km)
    adjacency: Dict[str, List[Tuple[str, float]]]

@st.cache_resource(show_spinner=True)
def build_station_graph(stations_in: pd.DataFrame, data_sig: Tuple, cfg: GraphConfig) -> GraphData:
    """
    Vectorized k-NN graph builder with distance-based edges.
    Cache keyed by a lightweight data_sig and GraphConfig.
    """
    # Filter stations
    if cfg.use_only_online:
        s = stations_in[(stations_in["is_installed"] == 1) & (stations_in["is_renting"] == 1)
                        & (stations_in["bikes_available"] > 0)].copy()
    else:
        s = stations_in.copy()

    s = s.dropna(subset=["lat", "lon", "station_id"])
    s["station_id"] = s["station_id"].astype(str)
    s = s.reset_index(drop=True)

    lat = s["lat"].to_numpy()
    lon = s["lon"].to_numpy()
    ids = s["station_id"].to_numpy()

    n = len(s)
    edges: List[Tuple[str, str, float]] = []
    for i in range(n):
        di = haversine_km(lat[i], lon[i], lat, lon)  # km to all j
        order = np.argsort(di)
        added = 0
        for j in order:
            if i == j:
                continue
            d = float(di[j])
            if d <= 0:
                continue
            if d > cfg.max_edge_km:
                # Skip overly long straight-line edges
                continue
            edges.append((ids[i], ids[j], d))
            added += 1
            if added >= cfg.k_neighbors:
                break

    adj: Dict[str, List[Tuple[str, float]]] = {}
    for u, v, d in edges:
        adj.setdefault(u, []).append((v, d))
        adj.setdefault(v, []).append((u, d))
    return GraphData(stations=s, edges=edges, adjacency=adj)

# ---------- Search (A* + Dijkstra with custom weights) ----------
def make_weight_func(
    mode: str,
    alpha_avail: float,
    hop_penalty_min: float,
    avg_speed_kmh: float,
    station_index: pd.DataFrame,
) -> Callable[[str, str, float], float]:
    """
    Return a function w(u, v, base_km) -> effective_cost
    Modes: dist | avail | mix | hops
    """
    hop_penalty_km = (hop_penalty_min * avg_speed_kmh) / 60.0 if hop_penalty_min > 0 else 0.0

    def w(u: str, v: str, base_km: float) -> float:
        if mode == "hops":
            # Prefer fewer edges; tiny distance tiebreak to avoid degenerate equal-cost paths
            return 1.0 + 1e-6 * base_km

        avail_mult = 1.0
        if mode in ("avail", "mix"):
            # Availability penalty (encourage edges from stocked → to dock-rich)
            su = station_index.loc[u]
            sv = station_index.loc[v]
            p_bikes = 0.0 if su["bikes_available"] > 2 else (2 - su["bikes_available"]) * 0.1
            p_docks = 0.0 if sv["docks_available"] > 2 else (2 - sv["docks_available"]) * 0.1
            avail_mult = 1.0 + alpha_avail * (p_bikes + p_docks)

        extra = hop_penalty_km if mode == "mix" else 0.0
        return base_km * avail_mult + extra

    return w

def make_heuristic(mode: str, coords: Dict[str, Tuple[float, float]], dst: str) -> Callable[[str], float]:
    """
    Admissible heuristic for A*. For non-negative edge penalties we can still use straight-line km.
    For hops mode, use zero heuristic.
    """
    if mode == "hops":
        return lambda u: 0.0

    lat2, lon2 = coords[dst]
    def h(u: str) -> float:
        lat1, lon1 = coords[u]
        return haversine_km_scalar(lat1, lon1, lat2, lon2)
    return h

def astar_generic(
    adjacency: Dict[str, List[Tuple[str, float]]],
    src: str,
    dst: str,
    weight: Callable[[str, str, float], float],
    heuristic: Callable[[str], float],
) -> Tuple[List[str], float]:
    import heapq
    g = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    openq = [(heuristic(src), 0.0, src)]  # (f, g, node)
    visited = set()

    while openq:
        f, curg, u = heapq.heappop(openq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, base_w in adjacency.get(u, []):
            w = weight(u, v, base_w)
            ng = curg + w
            if ng < g.get(v, float("inf")):
                g[v] = ng
                prev[v] = u
                heapq.heappush(openq, (ng + heuristic(v), ng, v))

    if dst not in g:
        return [], float("inf")

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, g[dst]

def dijkstra_generic(
    adjacency: Dict[str, List[Tuple[str, float]]],
    src: str,
    dst: str,
    weight: Callable[[str, str, float], float],
) -> Tuple[List[str], float]:
    import heapq
    dist = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    pq = [(0.0, src)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            break
        for v, base_w in adjacency.get(u, []):
            w = weight(u, v, base_w)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dst not in dist:
        return [], float("inf")

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path, dist[dst]

# ---------- Page ----------
st.set_page_config(page_title="Path Finder • RidePulse", page_icon="🧭", layout="wide")
st.title("🧭 Path Finder (NYC Citi Bike)")

# Read deep-linking params
qp = st.query_params
qp_src = qp.get("src")
qp_dst = qp.get("dst")
qp_opt = qp.get("opt", "dist")
def _to_float(s, default):
    try:
        return float(s)
    except Exception:
        return default
qp_alpha = _to_float(qp.get("alpha", 0.3), 0.3)
qp_hop = _to_float(qp.get("hop", 0.5), 0.5)
qp_astar = str(qp.get("astar", "1")).lower() in ("1", "true", "yes")

# Sidebar options
with st.sidebar:
    st.header("Options")
    c1, c2 = st.columns(2)
    with c1:
        use_only_online = st.toggle("Only active", value=True, help="Only include installed, renting stations with bike availability.")
    with c2:
        show_edges = st.toggle("Show edges", value=False, help="Render base graph edges (can be heavy).")

    k_neighbors = st.slider("Neighbors per node (k)", min_value=3, max_value=12, value=6, step=1)
    max_edge_km = st.slider("Max edge length (km)", min_value=0.3, max_value=3.0, value=2.0, step=0.1)
    avg_speed_kmh = st.slider("Avg speed (km/h)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)

    # Optimization modes
    opt_labels = ["Shortest distance", "Availability priority", "Balanced (distance + availability)", "Fewest hops"]
    opt_keys = ["dist", "avail", "mix", "hops"]
    try:
        opt_index_default = opt_keys.index(qp_opt)  # resume from deep link if valid
    except Exception:
        opt_index_default = 0
    optimize_for = st.radio("Optimize for", opt_labels, index=opt_index_default, horizontal=False)

    # Resolve mode key
    mode = opt_keys[opt_labels.index(optimize_for)]

    # A* toggle
    use_astar = st.toggle("Use A* (faster)", value=qp_astar, help="Speed up search while preserving optimality.")

    # Extra controls depending on mode
    alpha_avail = qp_alpha
    hop_penalty_min = qp_hop
    if mode in ("avail", "mix"):
        alpha_avail = st.slider("Availability priority (α)", min_value=0.0, max_value=1.0, value=float(qp_alpha), step=0.05,
                                help="How strongly to prefer routes starting with more bikes and ending with more docks.")
    if mode in ("mix", "hops"):
        hop_penalty_min = st.slider("Per-hop penalty (min)", min_value=0.0, max_value=5.0, value=float(qp_hop), step=0.1,
                                    help="Adds a fixed time per station-to-station hop to encourage fewer hops.")

# Load GBFS
try:
    with st.spinner("Fetching stations…"):
        stations_raw, last_updated = _load_gbfs()
except Exception as e:
    st.error(f"Could not load Citi Bike GBFS feeds: {e}")
    st.stop()

# Build graph (cached)
cfg = GraphConfig(
    use_only_online=use_only_online,
    k_neighbors=int(k_neighbors),
    max_edge_km=float(max_edge_km),
    avg_speed_kmh=float(avg_speed_kmh),
)
data_sig = (last_updated, len(stations_raw))
graph = build_station_graph(stations_raw, data_sig, cfg)

# Station selectors (respect deep-link IDs if valid)
station_ids = graph.stations["station_id"].astype(str).tolist()
station_map_name_to_id = dict(zip(graph.stations["name"], graph.stations["station_id"].astype(str)))
station_map_id_to_name = dict(zip(graph.stations["station_id"].astype(str), graph.stations["name"]))

def _default_station_id(idx: int) -> str:
    if 0 <= idx < len(station_ids):
        return station_ids[idx]
    return station_ids[0] if station_ids else ""

src_id_default = qp_src if (isinstance(qp_src, str) and qp_src in station_ids) else _default_station_id(0)
dst_id_default = qp_dst if (isinstance(qp_dst, str) and qp_dst in station_ids) else _default_station_id(min(1, len(station_ids)-1))

colA, colB, colC = st.columns([3, 3, 1])
with colA:
    src_name = station_map_id_to_name.get(src_id_default, graph.stations.iloc[0]["name"])
    src_sel = st.selectbox(
        "Start station",
        options=graph.stations["name"],
        index=int(graph.stations.index[graph.stations["name"] == src_name][0]) if src_name in graph.stations["name"].values else 0,
    )
with colB:
    dst_name = station_map_id_to_name.get(dst_id_default, graph.stations.iloc[min(1, len(graph.stations)-1)]["name"])
    dst_sel = st.selectbox(
        "End station",
        options=graph.stations["name"],
        index=int(graph.stations.index[graph.stations["name"] == dst_name][0]) if dst_name in graph.stations["name"].values else min(1, len(graph.stations)-1),
    )
with colC:
    if st.button("Swap"):
        src_sel, dst_sel = dst_sel, src_sel

src_id = station_map_name_to_id[src_sel]
dst_id = station_map_name_to_id[dst_sel]

# Prepare weighting and heuristic
station_index = graph.stations.set_index("station_id")
coords = {row.station_id: (float(row.lat), float(row.lon)) for _, row in graph.stations.iterrows()}
weight = make_weight_func(mode, float(alpha_avail), float(hop_penalty_min), float(cfg.avg_speed_kmh), station_index)
heuristic = make_heuristic(mode, coords, dst_id)

# Route search
route_btn = st.button("Find best route", type="primary", use_container_width=True)
path_ids: List[str] = []
total_cost = float("inf")
if route_btn:
    with st.spinner("Computing route…"):
        if use_astar:
            path_ids, total_cost = astar_generic(graph.adjacency, src_id, dst_id, weight, heuristic)
        else:
            path_ids, total_cost = dijkstra_generic(graph.adjacency, src_id, dst_id, weight)
    # Update deep link params
    st.query_params.update({
        "src": src_id,
        "dst": dst_id,
        "opt": mode,
        "alpha": f"{alpha_avail:.2f}",
        "hop": f"{hop_penalty_min:.2f}",
        "astar": "1" if use_astar else "0",
    })

# Map rendering (Plotly ScatterMapbox using open-street-map style; no token needed)
center_lat = float(graph.stations["lat"].mean())
center_lon = float(graph.stations["lon"].mean())

fig = go.Figure()

# Base edges (optional)
if show_edges and len(graph.edges) > 0:
    edges_to_plot = graph.edges
    if len(edges_to_plot) > 8000:
        edges_to_plot = edges_to_plot[::max(1, len(edges_to_plot)//8000)]
    xs: List[float] = []
    ys: List[float] = []
    idx_by_id = graph.stations.set_index("station_id")[["lon", "lat"]]
    for u, v, _ in edges_to_plot:
        su = idx_by_id.loc[u]
        sv = idx_by_id.loc[v]
        xs += [float(su["lon"]), float(sv["lon"]), None]
        ys += [float(su["lat"]), float(sv["lat"]), None]
    fig.add_trace(go.Scattermapbox(
        lon=xs, lat=ys, mode="lines",
        line=dict(width=1, color="#94a3b8"),
        name="Edges", showlegend=False, hoverinfo="skip"
    ))

# Stations
fig.add_trace(go.Scattermapbox(
    lon=graph.stations["lon"], lat=graph.stations["lat"],
    mode="markers",
    marker=dict(size=6, color="#3b82f6"),
    text=graph.stations["name"],
    hoverinfo="text",
    name="Stations",
))

# Path overlay
if path_ids:
    pts = graph.stations.set_index("station_id").loc[path_ids][["lon", "lat"]]
    fig.add_trace(go.Scattermapbox(
        lon=pts["lon"], lat=pts["lat"],
        mode="lines+markers",
        line=dict(width=4, color="#10b981"),
        marker=dict(size=8, color="#10b981"),
        name="Best path",
    ))

fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=11,
    ),
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01),
)
st.plotly_chart(_apply_theme(fig), use_container_width=True, theme="streamlit")

# Metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Stations in graph", f"{len(graph.stations):,}")
with m2:
    st.metric("Edges", f"{len(graph.edges):,}")
with m3:
    if path_ids:
        # For distance-like modes, estimate ETA from straight-line km sum proxy
        # For hops mode, we can’t derive km from cost; show hop count instead.
        if mode == "hops":
            st.metric("Path (hops)", f"{len(path_ids)-1} hops")
        else:
            # Approximate total km by summing base distances along path
            base_km_sum = 0.0
            adj_map = {u: {v: d for v, d in nbrs} for u, nbrs in graph.adjacency.items()}
            for u, v in zip(path_ids[:-1], path_ids[1:]):
                base_km_sum += adj_map[u][v]
            eta_min = base_km_sum / max(1e-3, cfg.avg_speed_kmh) * 60.0
            st.metric("Path length / ETA", f"{base_km_sum:.2f} km • {eta_min:.0f} min")
    else:
        st.metric("Path", "—")

st.caption("Tip: After computing a route, the URL updates with src/dst/opt so you can share the exact selection and options.")