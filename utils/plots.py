import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def _is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return True

def _apply_theme_layout(fig):
    dark = _is_dark()
    paper_bg = "rgba(0,0,0,0)"
    plot_bg = "rgba(0,0,0,0)"
    font_color = "#C9D1D9" if dark else "#111827"
    grid_color = "#30363d" if dark else "#e5e7eb"

    # Use Streamlit theme-friendly templates
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        margin=dict(l=10, r=10, t=80, b=40),
    )
    # Colorbar ticks in theme color (if present)
    if "coloraxis" in fig.layout and hasattr(fig.layout.coloraxis, "colorbar"):
        fig.layout.coloraxis.colorbar.tickfont.color = font_color
        fig.layout.coloraxis.colorbar.titlefont.color = font_color
    return fig

def kpi_cards(df, c1, c2, c3, c4):
    total_bikes = int(df["num_bikes_available"].sum())
    total_docks = int(df["num_docks_available"].sum())
    stations = len(df)
    avg_full = float(df["percent_full"].mean() * 100)

    c1.metric("🚲 Available Bikes", f"{total_bikes:,}")
    c2.metric("🅿️ Open Docks", f"{total_docks:,}")
    c3.metric("📍 Active Stations", f"{stations:,}")
    c4.metric("⚙️ Avg Station Fill", f"{avg_full:.1f}%")

def top_stations_bar(df, n=10):
    top = df.sort_values("num_bikes_available", ascending=False).head(n).copy()
    top["label"] = top["name"].str.slice(0, 26) + top["name"].apply(lambda s: "…" if len(s) > 26 else "")
    fig = px.bar(
        top,
        x="label",
        y="num_bikes_available",
        color="percent_full",
        color_continuous_scale="Viridis" if _is_dark() else "Blues",
        title=f"Top {n} Stations — Available Bikes",
        labels={"label": "Station", "num_bikes_available": "Bikes"},
    )
    fig.update_traces(
        customdata=top[["name", "percent_full", "num_bikes_available", "num_docks_available"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>Bikes: %{customdata[2]}<br>Docks: %{customdata[3]}<br>% Full: %{customdata[1]:.0%}<extra></extra>",
    )
    fig.update_xaxes(tickangle=40, automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(title={"y": 0.98, "yanchor": "top"}, height=380)
    return _apply_theme_layout(fig)

def short_term_trend_chart(hist_df: pd.DataFrame):
    plot_df = hist_df.copy()
    plot_df["ts_local"] = plot_df["ts"].dt.tz_convert(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["ts_local"], y=plot_df["total_bikes"],
                             mode="lines+markers", name="Total Bikes", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=plot_df["ts_local"], y=plot_df["total_docks"],
                             mode="lines+markers", name="Total Docks", line=dict(color="#10b981")))
    fig.update_layout(title="Last Snapshots", height=360, legend=dict(orientation="h"))
    return {"fig": _apply_theme_layout(fig), "data": plot_df}

def utilization_hist(df):
    series = (df["percent_full"]*100).round(1)
    fig = px.histogram(series, nbins=30, title="Station Utilization (%)",
                       color_discrete_sequence=["#6366f1"], labels={"value": "% Full"})
    fig.update_layout(height=320, bargap=0.02)
    return _apply_theme_layout(fig)

def station_utilization_gauge(name: str, available: int, capacity: int):
    capacity = max(int(capacity), 1)
    available = max(int(available), 0)
    pct = (available / capacity) * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": f"Utilization — {name}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#3FB950"},
                "steps": [
                    {"range": [0, 40], "color": "#2A2F36" if _is_dark() else "#eef2ff"},
                    {"range": [40, 70], "color": "#1F6FEB"},
                    {"range": [70, 100], "color": "#DA3633"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=300)
    return _apply_theme_layout(fig)