from __future__ import annotations
from datetime import datetime, timezone

def human_time(dt_utc: datetime) -> str:
    """
    Format a UTC datetime to a concise string.
    Accepts naive datetimes (assumed UTC) or tz-aware UTC datetimes.
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")