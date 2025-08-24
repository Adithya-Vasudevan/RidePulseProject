# -*- coding: utf-8 -*-
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -------- Column picking (robust to different GBFS station schemas) --------
def pick_col(dframe: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in dframe.columns:
            return c
    # try case-insensitive
    low = {c.lower(): c for c in dframe.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


@dataclass
class StationCols:
    sid: Optional[str]
    name: Optional[str]
    lat: Optional[str]
    lon: Optional[str]
    is_installed: Optional[str]
    is_renting: Optional[str]
    is_returning: Optional[str]


def get_station_cols(df: pd.DataFrame) -> StationCols:
    return StationCols(
        sid=pick_col(df, ("station_id", "short_name", "id", "stationCode")),
        name=pick_col(df, ("station_name", "name", "station", "title")),
        lat=pick_col(df, ("lat", "latitude", "station_latitude")),
        lon=pick_col(df, ("lon", "lng", "longitude", "station_longitude")),
        is_installed=pick_col(df, ("is_installed",)),
        is_renting=pick_col(df, ("is_renting",)),
        is_returning=pick_col(df, ("is_returning",)),
    )


# ------------------------ Geo helpers and graph build ------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@dataclass
class Node:
    id: str
    name: str
    lat: float
    lon: float


# Adjacency: node_id -> neighbor_id -> edge attrs
Adj = Dict[str, Dict[str, Dict[str, float]]]


def build_station_graph(
    df: pd.DataFrame,
    use_only_online: bool = False,
    k_neighbors: int = 6,
    max_edge_km: float = 2.0,
    avg_speed_kmh: float = 15.0,
) -> Tuple[Dict[str, Node], Adj]:
    cols = get_station_cols(df)
    if not cols.lat or not cols.lon:
        raise ValueError("Latitude/Longitude columns not found in station data.")
    # Derive online if requested
    df2 = df.copy()
    if use_only_online and cols.is_renting and cols.is_returning:
        try:
            online = df2[cols.is_renting].astype(float).fillna(1.0) > 0.0
            online &= df2[cols.is_returning].astype(float).fillna(1.0) > 0.0
            df2 = df2.loc[online].copy()
        except Exception:
            # Fall back to full set if booleans not parseable
            pass

    # Coerce numeric lat/lon
    df2[cols.lat] = pd.to_numeric(df2[cols.lat], errors="coerce")
    df2[cols.lon] = pd.to_numeric(df2[cols.lon], errors="coerce")
    df2 = df2.dropna(subset=[cols.lat, cols.lon])

    # Node id + name
    sid_col = cols.sid or cols.name
    if sid_col is None:
        raise ValueError("No station id/name column found for node identity.")
    name_col = cols.name or cols.sid or sid_col

    # Build node list
    nodes: Dict[str, Node] = {}
    for _, r in df2.iterrows():
        sid = str(r[sid_col])
        nm = str(r[name_col]) if name_col in df2.columns else sid
        lat = float(r[cols.lat])
        lon = float(r[cols.lon])
        if sid not in nodes:
            nodes[sid] = Node(id=sid, name=nm, lat=lat, lon=lon)

    if not nodes:
        return nodes, {}

    ids = list(nodes.keys())
    lat_arr = np.array([nodes[i].lat for i in ids], dtype=float)
    lon_arr = np.array([nodes[i].lon for i in ids], dtype=float)

    # Build adjacency using k-NN by haversine distance (undirected)
    adj: Adj = {i: {} for i in ids}
    N = len(ids)
    # For each station, compute distances vectorized
    for idx in range(N):
        lat1 = lat_arr[idx]
        lon1 = lon_arr[idx]
        # Vector haversine: approximate with per-row loop for clarity/perf balance
        dists = np.empty(N, dtype=float)
        for j in range(N):
            if j == idx:
                dists[j] = np.inf
            else:
                dists[j] = haversine_km(lat1, lon1, lat_arr[j], lon_arr[j])
        order = np.argsort(dists)
        added = 0
        for j in order:
            if added >= k_neighbors:
                break
            dist_km = float(dists[j])
            if not np.isfinite(dist_km) or dist_km > max_edge_km:
                continue
            u = ids[idx]
            v = ids[j]
            time_min = (dist_km / max(1e-6, avg_speed_kmh)) * 60.0
            # Add both directions (undirected)
            adj[u][v] = {"distance_km": dist_km, "time_min": time_min, "weight": dist_km}
            adj[v][u] = {"distance_km": dist_km, "time_min": time_min, "weight": dist_km}
            added += 1

    return nodes, adj


# ----------------------------- Search algorithms -----------------------------
@dataclass
class StepFrame:
    algo: str
    current: Optional[str]
    visited: Set[str]
    frontier: List[Tuple[float, str]]
    relaxations: List[Tuple[str, str]]
    found: bool
    step_idx: int


def bfs_steps(adj: Adj, source: str, target: str, max_steps: int = 100000) -> Tuple[List[StepFrame], List[str]]:
    visited: Set[str] = set()
    prev: Dict[str, Optional[str]] = {source: None}
    q: List[str] = [source]
    steps: List[StepFrame] = []
    sidx = 0

    while q and sidx < max_steps:
        u = q.pop(0)
        if u in visited:
            continue
        visited.add(u)
        relax: List[Tuple[str, str]] = []

        if u == target:
            steps.append(StepFrame("BFS", u, set(visited), [(0.0, n) for n in q], relax, True, sidx))
            break

        for v in adj.get(u, {}):
            if v not in prev:
                prev[v] = u
                q.append(v)
                relax.append((u, v))

        steps.append(StepFrame("BFS", u, set(visited), [(0.0, n) for n in q], relax, False, sidx))
        sidx += 1

    return steps, reconstruct_path(prev, target)


def dijkstra_steps(
    adj: Adj, source: str, target: str, weight_key: str = "weight", max_steps: int = 500000
) -> Tuple[List[StepFrame], List[str]]:
    dist: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    pq: List[Tuple[float, str]] = [(0.0, source)]
    visited: Set[str] = set()
    steps: List[StepFrame] = []
    sidx = 0

    while pq and sidx < max_steps:
        du, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        relax: List[Tuple[str, str]] = []

        if u == target:
            steps.append(StepFrame("Dijkstra", u, set(visited), list(pq), relax, True, sidx))
            break

        for v, data in adj.get(u, {}).items():
            w = float(data.get(weight_key, 1.0))
            alt = du + w
            if v not in dist or alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))
                relax.append((u, v))

        steps.append(StepFrame("Dijkstra", u, set(visited), list(pq), relax, False, sidx))
        sidx += 1

    return steps, reconstruct_path(prev, target)


def astar_steps(
    adj: Adj,
    nodes: Dict[str, Node],
    source: str,
    target: str,
    optimize_for: str = "time",  # "time" | "distance" | "stops"
    avg_speed_kmh: float = 15.0,
    max_steps: int = 500000,
) -> Tuple[List[StepFrame], List[str]]:
    def heuristic(a: str, b: str) -> float:
        na = nodes[a]
        nb = nodes[b]
        dist_km = haversine_km(na.lat, na.lon, nb.lat, nb.lon)
        if optimize_for == "distance":
            return dist_km
        if optimize_for == "time":
            return (dist_km / max(1e-6, avg_speed_kmh)) * 60.0
        return 0.0  # stops

    weight_key = {"time": "time_min", "distance": "distance_km", "stops": None}[optimize_for]
    g: Dict[str, float] = {source: 0.0}
    prev: Dict[str, Optional[str]] = {source: None}
    pq: List[Tuple[float, str]] = [(heuristic(source, target), source)]
    visited: Set[str] = set()
    steps: List[StepFrame] = []
    sidx = 0

    while pq and sidx < max_steps:
        f_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        relax: List[Tuple[str, str]] = []

        if u == target:
            steps.append(StepFrame("A*", u, set(visited), list(pq), relax, True, sidx))
            break

        for v, data in adj.get(u, {}).items():
            w = 1.0 if optimize_for == "stops" else float(data.get(weight_key or "weight", 1.0))
            tentative = g[u] + w
            if v not in g or tentative < g[v]:
                g[v] = tentative
                prev[v] = u
                heapq.heappush(pq, (tentative + heuristic(v, target), v))
                relax.append((u, v))

        steps.append(StepFrame("A*", u, set(visited), list(pq), relax, False, sidx))
        sidx += 1

    return steps, reconstruct_path(prev, target)


def greedy_best_first_steps(
    adj: Adj,
    nodes: Dict[str, Node],
    source: str,
    target: str,
    optimize_for: str = "time",
    avg_speed_kmh: float = 15.0,
    max_steps: int = 500000,
) -> Tuple[List[StepFrame], List[str]]:
    def heuristic(a: str, b: str) -> float:
        na = nodes[a]
        nb = nodes[b]
        dist_km = haversine_km(na.lat, na.lon, nb.lat, nb.lon)
        if optimize_for == "distance":
            return dist_km
        if optimize_for == "time":
            return (dist_km / max(1e-6, avg_speed_kmh)) * 60.0
        return 0.0

    prev: Dict[str, Optional[str]] = {source: None}
    pq: List[Tuple[float, str]] = [(heuristic(source, target), source)]
    visited: Set[str] = set()
    steps: List[StepFrame] = []
    sidx = 0

    while pq and sidx < max_steps:
        h_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        relax: List[Tuple[str, str]] = []

        if u == target:
            steps.append(StepFrame("Greedy Best-First", u, set(visited), list(pq), relax, True, sidx))
            break

        for v in adj.get(u, {}):
            if v not in visited and v not in prev:
                prev[v] = u
                heapq.heappush(pq, (heuristic(v, target), v))
                relax.append((u, v))

        steps.append(StepFrame("Greedy Best-First", u, set(visited), list(pq), relax, False, sidx))
        sidx += 1

    return steps, reconstruct_path(prev, target)


def reconstruct_path(prev: Dict[str, Optional[str]], target: str) -> List[str]:
    if target not in prev:
        return []
    path = [target]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# ----------------------------- Paths enumeration -----------------------------
def all_simple_paths_limited(
    adj: Adj,
    source: str,
    target: str,
    cutoff_hops: int = 10,
    max_paths: int = 200,
) -> List[List[str]]:
    paths: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(source, [source])]
    visited: Set[str] = set()

    while stack and len(paths) < max_paths:
        node, path = stack.pop()
        if len(path) - 1 > cutoff_hops:
            continue
        if node == target:
            paths.append(path)
            continue
        for nbr in adj.get(node, {}):
            if nbr in path:
                continue
            stack.append((nbr, path + [nbr]))
    return paths


# ----------------------------- Metrics -----------------------------
def path_metrics(adj: Adj, path: List[str]) -> Dict[str, float]:
    total_time = 0.0
    total_dist = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = adj[u][v]
        total_time += float(data.get("time_min", 1.0))
        total_dist += float(data.get("distance_km", 1.0))
    return {
        "stops": max(len(path) - 1, 0),
        "time_min": round(total_time, 2),
        "distance_km": round(total_dist, 2),
    }