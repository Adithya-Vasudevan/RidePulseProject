from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.gbfs import merged_station_frame

# Optional analytics/badges; degrade gracefully if missing
try:
    from utils.badges import award_badge, track_page_visit  # type: ignore
except Exception:
    def award_badge(_: str) -> None:
        return
    def track_page_visit(_: str) -> None:
        return


# ---------------- Theme helpers ----------------
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
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ---------------- Data wrangling ----------------
def _capacity_from(df: pd.DataFrame) -> pd.Series:
    if "capacity" in df.columns and pd.to_numeric(df["capacity"], errors="coerce").fillna(0).max() > 0:
        return pd.to_numeric(df["capacity"], errors="coerce")
    bikes = pd.to_numeric(df.get("num_bikes_available", np.nan), errors="coerce")
    docks = pd.to_numeric(df.get("num_docks_available", np.nan), errors="coerce")
    return bikes.add(docks, fill_value=np.nan)


def _ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["num_bikes_available"] = pd.to_numeric(out.get("num_bikes_available"), errors="coerce")
    out["num_docks_available"] = pd.to_numeric(out.get("num_docks_available"), errors="coerce")
    out["capacity"] = _capacity_from(out)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["percent_full"] = (out["num_bikes_available"] / out["capacity"]) * 100.0
    out.loc[~np.isfinite(out["percent_full"]), "percent_full"] = np.nan
    out["percent_full"] = out["percent_full"].clip(lower=0, upper=100)
    out["name"] = out.get("name", "Station")
    out["name"] = out["name"].fillna("Unnamed station")
    return out


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["capacity"] = pd.to_numeric(f["capacity"], errors="coerce").fillna(0)
    f["bikes"] = pd.to_numeric(f["num_bikes_available"], errors="coerce").fillna(0)
    f["docks"] = pd.to_numeric(f["num_docks_available"], errors="coerce").fillna(0)
    f["percent_full"] = pd.to_numeric(f["percent_full"], errors="coerce").fillna(0).clip(0, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["bikes_per_cap"] = np.where(f["capacity"] > 0, f["bikes"] / f["capacity"], 0.0)
        f["docks_per_cap"] = np.where(f["capacity"] > 0, f["docks"] / f["capacity"], 0.0)
    # Time context
    try:
        f["_ts"] = pd.to_datetime(f.get("last_reported", pd.Timestamp.utcnow()), unit="s", errors="coerce")
    except Exception:
        f["_ts"] = pd.Timestamp.utcnow()
    f["hour"] = f["_ts"].dt.hour.fillna(0).astype(int)
    f["is_peak"] = f["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    f.drop(columns=["_ts"], inplace=True, errors="ignore")
    # Neighbor features (if coords available)
    if {"lat", "lon"}.issubset(f.columns):
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore
            coords = f[["lat", "lon"]].to_numpy()
            k = min(6, len(f))
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            pf = f["percent_full"].to_numpy()
            cap = f["capacity"].to_numpy()
            nn_pf_mean = []
            nn_cap_mean = []
            for row_idx in range(indices.shape[0]):
                idxs = indices[row_idx][1:] if indices.shape[1] > 1 else indices[row_idx]
                vals_pf = pf[idxs]
                vals_cap = cap[idxs]
                nn_pf_mean.append(np.nanmean(vals_pf) if len(vals_pf) else np.nan)
                nn_cap_mean.append(np.nanmean(vals_cap) if len(vals_cap) else np.nan)
            f["nn_percent_full_mean"] = pd.Series(nn_pf_mean, index=f.index).fillna(f["percent_full"].mean())
            f["nn_capacity_mean"] = pd.Series(nn_cap_mean, index=f.index).fillna(f["capacity"].mean())
        except Exception:
            f["nn_percent_full_mean"] = f["percent_full"].mean()
            f["nn_capacity_mean"] = f["capacity"].mean()
    else:
        f["nn_percent_full_mean"] = f["percent_full"].mean()
        f["nn_capacity_mean"] = f["capacity"].mean()
    return f


# ---------------- Model notes renderer ----------------
def model_note_block(model_name: str, why: str, strengths: List[str], tradeoffs: List[str]) -> None:
    with st.container(border=True):
        st.write(f"Model: {model_name}")
        st.write(f"Why: {why}")
        if strengths:
            st.write("Strengths: " + ", ".join(strengths))
        if tradeoffs:
            st.write("Tradeoffs: " + ", ".join(tradeoffs))


# ---------------- Streamlit page ----------------
st.set_page_config(page_title="Model Labs • RidePulse NYC", page_icon="🧪", layout="wide")
track_page_visit("Model Labs")
award_badge("model_labs")

st.title("🧪 Model Labs — GBFS Data Models")
st.caption("Purpose-built models for bikeshare operations on the live GBFS snapshot. Includes an interactive What‑If simulator.")

# Load data
df_raw = merged_station_frame()
if df_raw is None or len(df_raw) == 0:
    st.warning("Live data not available right now. Please try again shortly.")
    st.stop()

df = _ensure_metrics(df_raw)
feat = _feature_frame(df)

# Sidebar controls
with st.sidebar:
    st.header("Options")
    st.caption("These models run on the current snapshot. Add history later for forecasting.")
    show_names = st.toggle("Show station names in charts", value=True)

# Expose shared model objects for use in What-If tab
clf = None
clf_model_name = None
clf_X_cols: List[str] = []
kmeans = None
cluster_scaler = None
kmeans_used_scaler = False
iso = None
Xa_cols: List[str] = []

tab_cls, tab_cluster, tab_anom, tab_sim = st.tabs(
    ["Rebalancing Classifier", "Station Clustering", "Anomaly Detection", "What‑If Simulator"]
)

# ---------------- Rebalancing Classifier ----------------
with tab_cls:
    st.subheader("Rebalancing need — binary classifier")
    st.caption(
        "Goal: Flag stations likely needing attention now (very empty or very full). "
        "We train a small model on 'silver labels' derived from the snapshot itself."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        low_thr = st.slider("Empty ≤ %", 0, 40, 10, 1, help="Stations at or below this fill % are labeled as 'needs rebalancing'.")
    with c2:
        high_thr = st.slider("Full ≥ %", 60, 100, 90, 1, help="Stations at or above this fill % are labeled as 'needs rebalancing'.")
    with c3:
        model_choice = st.selectbox("Model", ["RandomForestClassifier", "LogisticRegression"], index=0)

    clf_X_cols = [
        "capacity",
        "bikes",
        "docks",
        "percent_full",
        "bikes_per_cap",
        "docks_per_cap",
        "hour",
        "is_peak",
        "nn_percent_full_mean",
        "nn_capacity_mean",
    ]
    X = feat[clf_X_cols].fillna(0.0).to_numpy()
    y = ((feat["percent_full"] <= low_thr) | (feat["percent_full"] >= high_thr)).astype(int).to_numpy()

    metrics_text = ""
    probas = None
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # type: ignore

        if model_choice == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier  # type: ignore
            _clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                class_weight="balanced",
            )
            clf_model_name = "RandomForestClassifier"
            why = "Handles non-linear interactions and heterogeneous features well with minimal tuning."
            strengths = ["Robust to feature scaling", "Good baseline", "Feature importance"]
            tradeoffs = ["Less interpretable than linear models", "Bigger models can be slower"]
        else:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            _clf = LogisticRegression(
                max_iter=200,
                class_weight="balanced",
            )
            clf_model_name = "LogisticRegression"
            why = "Fast, simple, and interpretable baseline for binary decisions."
            strengths = ["Speed", "Interpretability", "Low variance"]
            tradeoffs = ["Linear decision boundary", "Needs scaled features for best performance"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        _clf.fit(X_train, y_train)
        y_pred = _clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        metrics_text = f"Accuracy: {acc:.3f} • Precision: {prec:.3f} • Recall: {rec:.3f} • F1: {f1:.3f}"
        try:
            probas = _clf.predict_proba(X)[:, 1]
        except Exception:
            probas = None

        clf = _clf  # expose to What-If
        model_note_block(clf_model_name, why, strengths, tradeoffs)

    except ImportError as e:
        st.warning("scikit-learn not found. Install with: pip install scikit-learn")
        model_note_block("Rule-based threshold", "No ML lib; using thresholds as baseline.", ["Transparent", "No deps"], ["No learning", "Rigid"])
    except Exception as e:
        st.error(f"Classifier failed: {e}")

    # Score snapshot
    if clf is not None:
        pred_now = clf.predict(X)
        score_now = (probas if probas is not None else pred_now.astype(float))
    else:
        # Rule-based fallback
        pred_now = y
        score_now = (feat["percent_full"] / 100.0).where(feat["percent_full"] >= high_thr, 1 - (feat["percent_full"] / 100.0))

    # Display metrics
    if metrics_text:
        st.info(f"Validation metrics on hold-out: {metrics_text}")

    # Table of top stations needing attention
    out_tbl = feat[["name", "percent_full", "capacity", "bikes", "docks"]].copy()
    out_tbl["need_rebalance"] = pred_now.astype(int)
    out_tbl["score"] = np.round(score_now, 3)
    need_now = out_tbl.sort_values(["need_rebalance", "score"], ascending=[False, False])

    st.markdown("##### Stations ranked by rebalancing need")
    st.dataframe(
        need_now if show_names else need_now.drop(columns=["name"]),
        use_container_width=True,
        hide_index=True,
    )

    # Visual: percent full vs capacity, colored by need
    vis = feat.copy()
    vis["need"] = pred_now.astype(int)
    fig_cls = px.scatter(
        vis,
        x="capacity",
        y="percent_full",
        color=vis["need"].map({1: "Needs attention", 0: "OK"}),
        hover_name="name" if show_names else None,
        labels={"capacity": "Capacity", "percent_full": "% full"},
        title="Capacity vs Fill — Rebalancing decision",
        opacity=0.85,
    )
    st.plotly_chart(_apply_theme(fig_cls), use_container_width=True, theme="streamlit")

# ---------------- Station Clustering ----------------
with tab_cluster:
    st.subheader("Station clustering — KMeans")
    st.caption(
        "Goal: Group stations by similar characteristics (capacity, fill, neighbor context). "
        "Useful for planning, deployment, and storytelling."
    )

    k = st.slider("Number of clusters (K)", 2, 10, 4, 1)
    scale_xy = st.toggle("Scale features (StandardScaler) — recommended", value=True)

    Xc_cols = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
    Xc = feat[Xc_cols].fillna(0.0).to_numpy()

    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore

        Xc_proc = Xc
        if scale_xy:
            cluster_scaler = StandardScaler().fit(Xc)
            Xc_proc = cluster_scaler.transform(Xc)
            kmeans_used_scaler = True

        _kmeans = KMeans(n_clusters=k, n_init="auto" if hasattr(KMeans(), "n_init") else 10, random_state=42)
        labels = _kmeans.fit_predict(Xc_proc)
        feat["cluster"] = labels
        kmeans = _kmeans  # expose to What-If

        model_note_block(
            "KMeans",
            "Simple, fast clustering to group similar stations for operational insights.",
            ["Fast", "Scales well", "Easy to explain"],
            ["Assumes spherical clusters", "Requires choosing K"],
        )

        # 2D projection for visualization
        fig_cluster = px.scatter(
            feat,
            x="capacity",
            y="percent_full",
            color=feat["cluster"].astype(str),
            hover_name="name" if show_names else None,
            title="Clusters projected on Capacity vs % Full",
            labels={"capacity": "Capacity", "percent_full": "% full", "color": "Cluster"},
            opacity=0.9,
        )
        st.plotly_chart(_apply_theme(fig_cluster), use_container_width=True, theme="streamlit")

        # Cluster summaries (flatten multi-index columns)
        st.markdown("##### Cluster summaries")
        cols_for_summary = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
        summary = feat.groupby("cluster")[cols_for_summary].agg(["count", "mean", "median"])
        summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]  # flatten
        summary = summary.reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

    except ImportError:
        st.warning("scikit-learn not found. Install with: pip install scikit-learn")
        model_note_block("N/A", "scikit-learn not installed; unable to run clustering.", [], [])
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# ---------------- Anomaly Detection ----------------
with tab_anom:
    st.subheader("Anomaly detection — IsolationForest")
    st.caption(
        "Goal: Surface stations that look unusual given the current network (possible data issues or operational outliers)."
    )

    contam = st.slider("Expected anomaly rate", 0.01, 0.20, 0.05, 0.01)
    Xa_cols = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
    Xa = feat[Xa_cols].fillna(0.0).to_numpy()

    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        iso = IsolationForest(
            contamination=contam,
            n_estimators=300,
            random_state=42,
        )
        iso.fit(Xa)

        # score_samples: lower = more anomalous; invert and normalize for size
        raw = -iso.score_samples(Xa)  # higher = more anomalous
        feat["anomaly_score"] = raw

        # Normalize to a positive marker size [6..28]
        rmin, rmax = float(np.nanmin(raw)), float(np.nanmax(raw))
        scale = (raw - rmin) / (rmax - rmin + 1e-9)
        feat["anomaly_size"] = 6.0 + 22.0 * scale

        preds = iso.predict(Xa)  # -1 = anomaly, 1 = normal
        feat["is_anomaly"] = (preds == -1).astype(int)

        model_note_block(
            "IsolationForest",
            "Tree-based model that isolates outliers efficiently without labels.",
            ["Unsupervised", "Fast", "Handles non-linearities"],
            ["Scores are relative to current data", "Sensitive to feature scaling"],
        )

        top_anom = feat.sort_values("anomaly_score", ascending=False).head(15)
        st.markdown("##### Top anomalies")
        cols_show = ["name", "percent_full", "capacity", "bikes", "docks", "nn_percent_full_mean", "anomaly_score"]
        st.dataframe(top_anom[cols_show] if show_names else top_anom[[c for c in cols_show if c != "name"]],
                     use_container_width=True, hide_index=True)

        fig_anom = px.scatter(
            feat,
            x="capacity",
            y="percent_full",
            color=feat["is_anomaly"].map({1: "Anomaly", 0: "Normal"}),
            size="anomaly_size",  # guaranteed positive
            size_max=28,
            hover_name="name" if show_names else None,
            title="Anomalies on Capacity vs % Full",
            labels={"capacity": "Capacity", "percent_full": "% full"},
            opacity=0.9,
        )
        st.plotly_chart(_apply_theme(fig_anom), use_container_width=True, theme="streamlit")

    except ImportError:
        st.warning("scikit-learn not found. Install with: pip install scikit-learn")
        model_note_block("N/A", "scikit-learn not installed; unable to run anomaly model.", [], [])
    except Exception as e:
        st.error(f"Anomaly detection failed: {e}")

# ---------------- What‑If Simulator ----------------
with tab_sim:
    st.subheader("What‑If Simulator")
    st.caption(
        "Pick a real station to prefill, tweak values, and see predictions from the models above. "
        "Great for explaining how each model works."
    )

    # Prefill selector
    names = df["name"].tolist()
    default_idx = 0
    sel_name = st.selectbox("Prefill from station", options=names, index=default_idx)
    sel_row = feat.loc[feat["name"] == sel_name].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        cap_in = st.number_input("Capacity", min_value=1, max_value=1000, value=int(sel_row["capacity"]))
        hour_in = st.slider("Hour", 0, 23, int(sel_row["hour"]))
    with c2:
        bikes_in = st.number_input("Bikes available", min_value=0, max_value=int(cap_in), value=int(min(sel_row["bikes"], cap_in)))
        nn_pf_in = st.slider("Neighbor % full (mean)", 0, 100, int(round(float(sel_row["nn_percent_full_mean"]))), 1)
    with c3:
        docks_in = max(0, int(cap_in) - int(bikes_in))
        st.write(f"Docks (auto): {docks_in}")
        is_peak_in = st.toggle("Peak hour", value=bool(sel_row["is_peak"]))

    pf_in = (bikes_in / cap_in) * 100.0 if cap_in > 0 else 0.0
    bikes_per_cap_in = bikes_in / cap_in if cap_in > 0 else 0.0
    docks_per_cap_in = docks_in / cap_in if cap_in > 0 else 0.0

    # Show derived
    st.caption(f"Derived: % full = {pf_in:.1f} • bikes/cap = {bikes_per_cap_in:.2f} • docks/cap = {docks_per_cap_in:.2f}")

    # Build feature vector in the same order as training
    x_clf = np.array(
        [
            cap_in,
            bikes_in,
            docks_in,
            pf_in,
            bikes_per_cap_in,
            docks_per_cap_in,
            hour_in,
            1 if is_peak_in else 0,
            nn_pf_in,
            cap_in,  # proxy for nn_capacity_mean when simulating
        ],
        dtype=float,
    ).reshape(1, -1)

    # Classifier prediction
    st.markdown("##### Rebalancing classifier")
    if clf is not None:
        try:
            prob = float(clf.predict_proba(x_clf)[0, 1])
        except Exception:
            prob = float(clf.predict(x_clf)[0])
        lbl = "Needs attention" if prob >= 0.5 else "OK"
        st.metric("Prediction", lbl, delta=f"{prob*100:.1f}%")
        st.caption("Explanation: The model considers fill level, capacity, neighbor context, and peak hours to decide if a station likely needs rebalancing.")
    else:
        st.info("Classifier not trained (scikit‑learn missing). Using simple thresholds.")
        lbl = "Needs attention" if (pf_in <= 10 or pf_in >= 90) else "OK"
        st.metric("Prediction (rule‑based)", lbl)

    # Cluster assignment
    st.markdown("##### Cluster assignment")
    Xc_cols = ["capacity", "percent_full", "bikes_per_cap", "docks_per_cap", "nn_percent_full_mean"]
    x_cluster = np.array([cap_in, pf_in, bikes_per_cap_in, docks_per_cap_in, nn_pf_in], dtype=float).reshape(1, -1)
    if kmeans is not None:
        try:
            x_proc = cluster_scaler.transform(x_cluster) if (kmeans_used_scaler and cluster_scaler is not None) else x_cluster
            label = int(kmeans.predict(x_proc)[0])
            st.metric("Cluster", f"{label}")
            st.caption("Explanation: KMeans groups stations by similar capacity/fill patterns; nearby values land in the same group.")
        except Exception as e:
            st.error(f"Cluster prediction failed: {e}")
    else:
        st.info("Clustering not available (scikit‑learn missing).")

    # Anomaly score
    st.markdown("##### Anomaly score")
    x_anom = x_cluster  # same feature set used for anomalies
    if iso is not None:
        try:
            raw = float(-iso.score_samples(x_anom)[0])  # higher = more anomalous
            # Normalize with current dataset for context
            rmin, rmax = float(np.nanmin(feat["anomaly_score"])), float(np.nanmax(feat["anomaly_score"]))
            norm = (raw - rmin) / (rmax - rmin + 1e-9)
            pct = norm * 100.0
            st.metric("Relative anomaly", f"{pct:.1f}%")
            st.caption("Explanation: IsolationForest ranks how unusual the point is vs the current network; higher means more unusual.")
        except Exception as e:
            st.error(f"Anomaly scoring failed: {e}")
    else:
        st.info("Anomaly model not available (scikit‑learn missing).")