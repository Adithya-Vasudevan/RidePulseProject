import os
import csv
import json
import getpass
from pathlib import Path
from datetime import datetime, timezone
import streamlit as st

BADGE_CATALOG = {
    "explorer": ("Explorer", "🧭", "Visit multiple pages to unlock."),
    "station_sage": ("Station Sage", "🏙️", "Dive into station details."),
    "trend_hunter": ("Trend Hunter", "📈", "Explore snapshot trends."),
    "forecaster": ("Forecaster", "🔮", "Experiment with the Models Lab."),
    "fact_finder": ("Fact Finder", "🔎", "Discover live facts."),
    "quiz_whiz": ("Quiz Whiz", "🧠", "Ace the quiz."),
    "storyteller": ("Storyteller", "📖", "Build a narrative with Story Builder."),
    "cartographer": ("Cartographer", "🗺️", "Explore the Live Map."),
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _current_username() -> str:
    env_name = os.getenv("RIDEPULSE_USER")
    if env_name:
        return env_name
    try:
        return getpass.getuser()
    except Exception:
        return "anonymous"

def _awards_log_path() -> Path:
    return DATA_DIR / "badge_awards.csv"

def _persist_path_for_user(user: str) -> Path:
    safe_user = "".join(c for c in user if c.isalnum() or c in ("-", "_")).strip() or "anonymous"
    return DATA_DIR / f"badges_{safe_user}.json"

def init_badges():
    if "badges" not in st.session_state:
        st.session_state.badges = set()
    if "visited_pages" not in st.session_state:
        st.session_state.visited_pages = set()
    if "badges_loaded" not in st.session_state:
        user = _current_username()
        path = _persist_path_for_user(user)
        if path.exists():
            try:
                stored = json.loads(path.read_text())
                if isinstance(stored, list):
                    st.session_state.badges |= set(k for k in stored if k in BADGE_CATALOG)
            except Exception:
                pass
        st.session_state.badges_loaded = True

def _save_user_badges():
    user = _current_username()
    path = _persist_path_for_user(user)
    try:
        path.write_text(json.dumps(sorted(list(st.session_state.badges))))
    except Exception:
        pass

def _append_award_log(badge_key: str):
    log_path = _awards_log_path()
    write_header = not log_path.exists()
    try:
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_utc", "username", "badge_key"])
            w.writerow([datetime.now(timezone.utc).isoformat(), _current_username(), badge_key])
    except Exception:
        pass

def award_badge(key: str):
    init_badges()
    if key not in BADGE_CATALOG:
        return
    if key in st.session_state.badges:
        return
    st.session_state.badges.add(key)
    _save_user_badges()
    _append_award_log(key)
    try:
        st.toast(f"You earned {BADGE_CATALOG[key][1]} {BADGE_CATALOG[key][0]}!")
    except Exception:
        pass

def track_page_visit(page_name: str):
    init_badges()
    st.session_state.visited_pages.add(page_name)
    if "explorer" not in st.session_state.badges and len(st.session_state.visited_pages) >= 3:
        award_badge("explorer")
    lname = page_name.lower()
    if "station" in lname:
        award_badge("station_sage")
    if "map" in lname:
        award_badge("cartographer")

def render_badges():
    init_badges()
    if not st.session_state.badges:
        st.caption("No badges yet — explore pages to earn!")
        return
    st.markdown("#### 🏅 Badges")
    cols = st.columns(4)
    i = 0
    for key in sorted(st.session_state.badges):
        label, emoji, *_ = BADGE_CATALOG.get(key, (key, "🏅", ""))
        with cols[i % 4]:
            st.button(f"{emoji} {label}", disabled=True)
        i += 1