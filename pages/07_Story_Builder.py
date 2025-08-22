from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.gbfs import merged_station_frame

# Optional badges/analytics; degrade gracefully if missing
try:
    from utils.badges import award_badge, track_page_visit  # type: ignore
except Exception:
    def award_badge(_: str) -> None:
        return

    def track_page_visit(_: str) -> None:
        return


# ------------------------------
# Helpers (theme + accessibility)
# ------------------------------
def _is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True


def _apply_theme(fig: go.Figure, large_text: bool = False) -> go.Figure:
    dark = _is_dark()
    font_color = "#C9D1D9" if dark else "#111827"
    grid_color = "#30363d" if dark else "#e5e7eb"

    base_font_size = 16 if large_text else 13
    legend_size = 15 if large_text else 12
    tick_size = 14 if large_text else 11

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color, size=base_font_size),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(size=tick_size)),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, tickfont=dict(size=tick_size)),
        legend=dict(title_font_size=legend_size, font_size=legend_size, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=70, b=40),
    )
    return fig


# Colorblind-friendly (Okabe–Ito) palette
CBLUE = "#0072B2"
CORANGE = "#E69F00"
CGREEN = "#009E73"
CSKY = "#56B4E9"
CVERM = "#D55E00"
CPURP = "#CC79A7"
CGRAY = "#949494"


def _ensure_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("num_bikes_available", "num_docks_available"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "percent_full" not in df.columns:
        bikes = pd.to_numeric(df["num_bikes_available"], errors="coerce").fillna(0)
        docks = pd.to_numeric(df["num_docks_available"], errors="coerce").fillna(0)
        cap = (bikes + docks).replace(0, 1)
        df["percent_full"] = (bikes / cap).clip(0, 1)
    return df


# ------------------------------
# Story builders
# ------------------------------
def story_1_top_composition(
    df: pd.DataFrame, top_n: int = 12, show_labels: bool = False, large_text: bool = False
) -> go.Figure:
    """
    Story 1: Top stations by total capacity, stacked composition (Available Bikes vs Open Docks).
    Legend title: "Resource" (Available Bikes, Open Docks)
    """
    bikes = pd.to_numeric(df["num_bikes_available"], errors="coerce").fillna(0)
    docks = pd.to_numeric(df["num_docks_available"], errors="coerce").fillna(0)
    df = df.assign(capacity=(bikes + docks)).sort_values("capacity", ascending=False).head(top_n).copy()

    # Short labels but keep full name in hover
    df["label"] = df["name"].astype(str).str.slice(0, 22) + df["name"].astype(str).apply(lambda s: "…" if len(s) > 22 else "")

    base = go.Figure()

    base.add_bar(
        x=df["label"],
        y=df["num_bikes_available"],
        name="Available Bikes",
        marker_color=CBLUE,
        hovertemplate="<b>%{customdata[0]}</b><br>Available Bikes: %{y}<extra></extra>",
        customdata=df[["name"]],
    )
    base.add_bar(
        x=df["label"],
        y=df["num_docks_available"],
        name="Open Docks",
        marker_color=CORANGE,
        hovertemplate="<b>%{customdata[0]}</b><br>Open Docks: %{y}<extra></extra>",
        customdata=df[["name"]],
    )

    base.update_layout(
        barmode="stack",
        title=f"Top {top_n} Stations — Resource Composition",
        xaxis_title="Station",
        yaxis_title="Count",
        legend_title_text="Resource",
    )
    base.update_xaxes(tickangle=40, automargin=True)

    if show_labels:
        base.update_traces(
            texttemplate="%{y}",
            textposition="inside",
            insidetextanchor="end",
            textfont_size=14 if large_text else 11,
        )

    return _apply_theme(base, large_text=large_text)


def _balance_category(row, threshold: float) -> str:
    bikes = float(row.get("num_bikes_available", 0))
    docks = float(row.get("num_docks_available", 0))
    cap = max(bikes + docks, 1.0)
    diff_ratio = (bikes - docks) / cap
    if diff_ratio >= threshold:
        return "Bike-heavy (needs docks)"
    if diff_ratio <= -threshold:
        return "Dock-heavy (needs bikes)"
    return "Balanced"


def story_2_balance_scatter(
    df: pd.DataFrame, threshold: float = 0.2, large_text: bool = False, high_contrast: bool = False
) -> go.Figure:
    """
    Story 2: Bikes vs Docks per station with a clear categorical legend explaining balance.
    Legend title: "Balance category"
    """
    df = df.copy()
    df["capacity"] = (pd.to_numeric(df["num_bikes_available"], errors="coerce").fillna(0) +
                      pd.to_numeric(df["num_docks_available"], errors="coerce").fillna(0)).clip(lower=1)

    df["Category"] = df.apply(lambda r: _balance_category(r, threshold), axis=1)

    color_map = {
        "Balanced": CGREEN,
        "Dock-heavy (needs bikes)": CSKY,
        "Bike-heavy (needs docks)": CVERM,
    }
    symbol_map = {
        "Balanced": "circle",
        "Dock-heavy (needs bikes)": "triangle-up",
        "Bike-heavy (needs docks)": "triangle-down",
    }

    fig = px.scatter(
        df,
        x="num_bikes_available",
        y="num_docks_available",
        color="Category",
        color_discrete_map=color_map,
        symbol="Category",
        symbol_map=symbol_map,
        size="capacity",
        size_max=18 if large_text else 14,
        hover_data={"name": True, "capacity": True, "percent_full": ":.0%"},
        labels={
            "num_bikes_available": "Available Bikes",
            "num_docks_available": "Open Docks",
        },
        title="Network Balance — Bikes vs Docks by Station",
    )
    fig.update_layout(legend_title_text="Balance category")

    # Reference line (y=x) to indicate perfect balance
    max_axis = max(df["num_bikes_available"].max(), df["num_docks_available"].max()) + 5
    fig.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            line=dict(color=CGRAY, width=2, dash="dash"),
            name="Perfect balance (Bikes = Docks)",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # Improve marker contrast if requested
    if high_contrast:
        fig.update_traces(marker_line_color="white", marker_line_width=1.5, selector=dict(mode="markers"))

    return _apply_theme(fig, large_text=large_text)


def story_3_util_by_size(df: pd.DataFrame, large_text: bool = False) -> go.Figure:
    """
    Story 3: Utilization by Station Size — violin with jittered points.
    Legend title: "Station size group"
    """
    df = df.copy()
    # Build capacity and groups by quantiles (Small/Medium/Large)
    bikes = pd.to_numeric(df["num_bikes_available"], errors="coerce").fillna(0)
    docks = pd.to_numeric(df["num_docks_available"], errors="coerce").fillna(0)
    df["capacity"] = (bikes + docks).astype(int).clip(lower=1)

    # Guard against identical capacities (quantile bins would fail)
    if df["capacity"].nunique() < 3:
        # Fallback simple bins
        bins = [-1, 20, 35, df["capacity"].max() + 1]
    else:
        q33, q66 = df["capacity"].quantile([0.33, 0.66]).astype(int)
        # Ensure strictly increasing bins
        bins = sorted({-1, max(0, q33), max(q33 + 1, q66), int(df["capacity"].max()) + 1})

    labels = ["Small", "Medium", "Large"]
    # Align labels length with bins-1
    while len(labels) < len(bins) - 1:
        labels.append(f"Group {len(labels)+1}")
    labels = labels[: len(bins) - 1]

    df["size_group"] = pd.cut(df["capacity"], bins=bins, labels=labels, include_lowest=True, right=True)

    # Percent full in percent form for an intuitive scale
    df["pct_full"] = (df["percent_full"].clip(0, 1) * 100.0).round(1)

    # Construct a color map that covers however many labels we ended up with
    palette = [CBLUE, CGREEN, CPURP, CSKY, CVERM]
    group_colors = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    fig = px.violin(
        df,
        x="size_group",
        y="pct_full",
        color="size_group",
        color_discrete_map=group_colors,
        box=True,
        points="all",  # show all points
        hover_data={"name": True, "capacity": True, "pct_full": True},
        labels={"size_group": "Station size group", "pct_full": "Station fill level (%)"},
        title="Utilization by Station Size",
    )
    fig.update_layout(legend_title_text="Station size group")
    fig.update_yaxes(range=[0, 100])

    # Improve readability of overlay points: control via marker and jitter
    fig.update_traces(
        marker=dict(size=6 if large_text else 4, opacity=0.6),
        jitter=0.35,
        pointpos=0.0,  # center
        selector=dict(type="violin"),
    )

    return _apply_theme(fig, large_text=large_text)


# ------------------------------
# Page
# ------------------------------
st.set_page_config(page_title="Story Builder • RidePulse NYC", page_icon="📖", layout="wide")
track_page_visit("Story Builder")
award_badge("storyteller")

st.title("📖 Story Builder")

df = merged_station_frame()
if df is None or df.empty:
    st.warning("No station data available yet. Try refreshing.")
    st.stop()

df = _ensure_fields(df)

# Sidebar: global display options (no story selection required)
with st.sidebar:
    st.header("Display options")
    large_text = st.checkbox("Larger text (accessibility)", value=False, help="Increase font sizes in charts.")
    high_contrast = st.checkbox("High contrast markers (Story 2)", value=False, help="Add white outlines for better contrast.")
    show_labels = st.checkbox("Show numeric value labels (Story 1)", value=False)

    st.markdown("---")
    st.subheader("Story controls")
    top_n = st.slider("Story 1 — Top stations (by capacity)", min_value=5, max_value=30, value=12, step=1)
    threshold = st.slider(
        "Story 2 — Balance threshold",
        min_value=0.05,
        max_value=0.40,
        value=0.20,
        step=0.05,
        help="How different Bikes vs Docks must be (as share of capacity) to be considered unbalanced.",
    )

st.caption("Legends and labels are optimized for clarity and accessibility.")

# ------------------------------
# Story 1
# ------------------------------
st.subheader("Story 1 — Resource Composition (Top Stations)")
fig1 = story_1_top_composition(
    df.sort_values("num_bikes_available", ascending=False),
    top_n=top_n,
    show_labels=show_labels,
    large_text=large_text,
)
st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
st.info(
    "Did you know? Stations with high capacity aren’t always the ones with the most bikes right now. "
    "Capacity (total docks) shows potential, but live availability reflects current demand and rebalancing."
)

st.markdown("---")

# ------------------------------
# Story 2
# ------------------------------
st.subheader("Story 2 — Balance View (Bikes vs Docks)")
fig2 = story_2_balance_scatter(df, threshold=threshold, large_text=large_text, high_contrast=high_contrast)
st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
st.info(
    "Did you know? Points near the dashed line are well-balanced. "
    "Bike‑heavy stations (down‑triangles) often appear near popular morning destinations, "
    "while dock‑heavy stations (up‑triangles) cluster near residential areas after commute hours."
)

st.markdown("---")

# ------------------------------
# Story 3
# ------------------------------
st.subheader("Story 3 — Utilization by Station Size")
fig3 = story_3_util_by_size(df, large_text=large_text)
st.plotly_chart(fig3, use_container_width=True, theme="streamlit")
st.info(
    "Did you know? Smaller stations tend to reach high fill levels more quickly during peaks, "
    "while larger stations spread demand over more docks, leading to steadier utilization profiles."
)

st.markdown("---")
st.caption(
    "Tip: Adjust controls in the sidebar to tailor the stories. Colors are colorblind‑friendly, text sizes are adjustable, and legend labels are explicit."
)