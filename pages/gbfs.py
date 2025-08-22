from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Any, Dict, Union, List

import pandas as pd
import requests
from datetime import datetime, timezone

# Data directory for local persistence (snapshots, etc.)
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Citi Bike GBFS endpoints
GBFS_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
GBFS_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

# Simple in-memory cache to avoid excessive network calls
_CACHE: Dict[str, Any] = {"t": 0.0, "df": None}
_CACHE_TTL_SEC = 30


def _fetch_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    headers = {"User-Agent": "RidePulse/1.0 (+https://github.com/)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def load_station_information() -> pd.DataFrame:
    payload = _fetch_json(GBFS_INFO_URL)
    stations = payload.get("data", {}).get("stations", [])
    df = pd.DataFrame(stations)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    return df


def load_station_status() -> pd.DataFrame:
    payload = _fetch_json(GBFS_STATUS_URL)
    stations = payload.get("data", {}).get("stations", [])
    df = pd.DataFrame(stations)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    # Normalize core numeric fields
    for col in ["num_bikes_available", "num_docks_available"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0
    # Convert last_reported to UTC timestamp if present
    if "last_reported" in df.columns:
        df["last_reported"] = pd.to_datetime(df["last_reported"], unit="s", utc=True, errors="coerce")
    return df


def merged_station_frame(force: bool = False) -> pd.DataFrame:
    """
    Returns a merged live DataFrame of station information and status:
    - Columns: station_id, name, lat, lon, num_bikes_available, num_docks_available,
               capacity, percent_full, last_reported (UTC)
    Caches for a short time to reduce network calls.
    """
    now = time.time()
    if not force and _CACHE["df"] is not None and now - float(_CACHE["t"]) < _CACHE_TTL_SEC:
        return _CACHE["df"].copy()

    info = load_station_information()
    status = load_station_status()

    if info.empty and status.empty:
        return pd.DataFrame()

    df = pd.merge(status, info, on="station_id", how="left", suffixes=("_status", "_info"))

    # Standardize column names we rely on
    if "name_info" in df.columns and "name" not in df.columns:
        df.rename(columns={"name_info": "name"}, inplace=True)
    if "lat_info" in df.columns and "lat" not in df.columns:
        df.rename(columns={"lat_info": "lat"}, inplace=True)
    if "lon_info" in df.columns and "lon" not in df.columns:
        df.rename(columns={"lon_info": "lon"}, inplace=True)
    if "capacity_info" in df.columns and "capacity" not in df.columns:
        df.rename(columns={"capacity_info": "capacity"}, inplace=True)

    # Compute capacity and percent_full
    bikes = pd.to_numeric(df.get("num_bikes_available", 0), errors="coerce").fillna(0)
    docks = pd.to_numeric(df.get("num_docks_available", 0), errors="coerce").fillna(0)
    capacity = pd.to_numeric(df.get("capacity", bikes + docks), errors="coerce").fillna(bikes + docks)
    capacity = capacity.where(capacity > 0, other=bikes + docks).replace(0, 1)  # avoid zero
    df["capacity"] = capacity.astype(int)
    df["percent_full"] = (bikes / df["capacity"]).clip(0, 1)

    # Ensure last_reported exists
    if "last_reported" not in df.columns:
        df["last_reported"] = pd.Timestamp.now(tz=timezone.utc)

    # Order columns and sort
    wanted = [
        "station_id", "name", "lat", "lon",
        "num_bikes_available", "num_docks_available",
        "capacity", "percent_full", "last_reported",
    ]
    for col in wanted:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[wanted].sort_values("name", na_position="last").reset_index(drop=True)

    _CACHE["df"] = df.copy()
    _CACHE["t"] = now
    return df


def _snapshots_path_parquet() -> Path:
    return DATA_DIR / "snapshots.parquet"


def _snapshots_path_csv() -> Path:
    return DATA_DIR / "snapshots.csv"


def record_snapshot_if_due(df: pd.DataFrame, min_interval_sec: int = 60) -> bool:
    """
    Append an aggregate snapshot row if the last snapshot is older than min_interval_sec.
    Returns True if a new snapshot was recorded.
    """
    if df is None or df.empty:
        return False

    ts_utc = pd.Timestamp.now(tz="UTC")
    row = pd.DataFrame(
        [{
            "ts": ts_utc,
            "total_bikes": int(pd.to_numeric(df["num_bikes_available"], errors="coerce").fillna(0).sum()),
            "total_docks": int(pd.to_numeric(df["num_docks_available"], errors="coerce").fillna(0).sum()),
            "avg_percent_full": float(pd.to_numeric(df["percent_full"], errors="coerce").fillna(0).mean()),
        }]
    )

    pq_path = _snapshots_path_parquet()
    csv_path = _snapshots_path_csv()

    try:
        # Try Parquet first
        if pq_path.exists():
            existing = pd.read_parquet(pq_path)
            # Ensure tz-aware
            if existing["ts"].dtype.tz is None:
                existing["ts"] = existing["ts"].dt.tz_localize("UTC")
            last_ts = pd.to_datetime(existing["ts"].iloc[-1], utc=True)
            if (ts_utc - last_ts).total_seconds() < min_interval_sec:
                return False
            out = pd.concat([existing, row], ignore_index=True)
        else:
            out = row

        out.to_parquet(pq_path, index=False)
        return True
    except Exception:
        # Fallback to CSV
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True, errors="coerce")
            last_ts = existing["ts"].iloc[-1]
            if pd.notna(last_ts) and (ts_utc - last_ts).total_seconds() < min_interval_sec:
                return False
            out = pd.concat([existing, row], ignore_index=True)
        else:
            out = row

        out.to_csv(csv_path, index=False)
        return True


def get_snapshot_history(max_rows: int = 500) -> Union[pd.DataFrame, List]:
    """
    Returns a DataFrame of recent snapshots with tz-aware 'ts'.
    If no history exists yet, returns [] to align with callers that check for list.
    """
    pq_path = _snapshots_path_parquet()
    csv_path = _snapshots_path_csv()

    try:
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            return []
    except Exception:
        return []

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        # Drop any NaT rows
        df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    return df