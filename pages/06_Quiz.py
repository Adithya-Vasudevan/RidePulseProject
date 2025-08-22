from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional

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

# Optional Lottie helper (graceful if missing)
try:
    from utils.ui import show_lottie  # type: ignore
except Exception:
    def show_lottie(_: str, height: int = 120) -> None:
        return


# ------------- Theming helpers -------------
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


# ------------- Data & question bank -------------
@dataclass
class QA:
    q: str
    options: List[str]
    answer: str
    explanation: str
    difficulty: str  # "easy" | "medium" | "hard"


def _static_question_bank() -> List[QA]:
    return [
        QA(
            q="What does GBFS stand for in the context of bike share data?",
            options=[
                "General Bikeshare Feed Specification",
                "Global Bike Fleet Standard",
                "Geo Bike Feature Set",
                "General Bike Framework Schema",
            ],
            answer="General Bikeshare Feed Specification",
            explanation="GBFS is the General Bikeshare Feed Specification, an open standard for real‑time bikeshare data.",
            difficulty="easy",
        ),
        QA(
            q="Which metric best indicates current rider demand at a station?",
            options=[
                "Station capacity (total docks)",
                "Available bikes",
                "Installed docks (historical)",
                "Street address",
            ],
            answer="Available bikes",
            explanation="Capacity shows potential, but available bikes reflects live conditions and demand right now.",
            difficulty="easy",
        ),
        QA(
            q="If a station’s percent_full is 0.85, what does it mean?",
            options=[
                "85% of stations are full",
                "85% of docks at this station have a bike",
                "It has 85 bikes",
                "The station updates every 85 seconds",
            ],
            answer="85% of docks at this station have a bike",
            explanation="percent_full ≈ available_bikes / capacity. 0.85 means most docks currently hold bikes.",
            difficulty="medium",
        ),
        QA(
            q="Which situation most likely needs rebalancing?",
            options=[
                "Stations near 50% full",
                "Stations at 0–10% or 90–100% full",
                "Stations with high capacity",
                "Stations with frequent updates",
            ],
            answer="Stations at 0–10% or 90–100% full",
            explanation="Extremes (very empty or very full) limit rider options and indicate rebalancing need.",
            difficulty="medium",
        ),
        QA(
            q="In our Live Map, what do larger dots generally represent?",
            options=[
                "Higher station capacity",
                "Faster update rates",
                "Closer distance to Times Square",
                "More reported incidents",
            ],
            answer="Higher station capacity",
            explanation="Marker size scales with capacity so you can quickly spot large hubs.",
            difficulty="easy",
        ),
        QA(
            q="On the balance scatter (Bikes vs Docks), points near the dashed diagonal indicate:",
            options=[
                "High demand areas only in the evening",
                "Perfect or near‑perfect balance of bikes and docks",
                "Stations that never need rebalancing",
                "Stations with poor data quality",
            ],
            answer="Perfect or near‑perfect balance of bikes and docks",
            explanation="The y=x line means Available Bikes ≈ Open Docks at that moment.",
            difficulty="medium",
        ),
        QA(
            q="Why might smaller stations reach high fill levels faster during peaks?",
            options=[
                "They update less frequently",
                "They have fewer docks, so swings in usage move their fill faster",
                "They are always downtown",
                "Because GBFS prioritizes small stations",
            ],
            answer="They have fewer docks, so swings in usage move their fill faster",
            explanation="With fewer docks, each check‑in/out changes percent_full more dramatically.",
            difficulty="hard",
        ),
    ]


def _dynamic_question_from_live() -> Optional[QA]:
    try:
        df = merged_station_frame()
        if df is None or df.empty:
            return None
        row = df.sort_values("num_bikes_available", ascending=False).head(1).iloc[0]
        correct = str(row["name"])
        names = df["name"].dropna().unique().tolist()
        random.shuffle(names)
        distractors = [n for n in names if n != correct][:3]
        options = distractors + [correct]
        random.shuffle(options)
        return QA(
            q="Which station currently has the most available bikes?",
            options=options,
            answer=correct,
            explanation="Live data can change quickly—recheck the Overview if the situation looks different.",
            difficulty="medium",
        )
    except Exception:
        return None


def _assemble_quiz(num_questions: int, difficulty: str) -> List[QA]:
    bank = _static_question_bank()
    dyn = _dynamic_question_from_live()
    if dyn:
        bank.append(dyn)

    if difficulty in {"easy", "medium", "hard"}:
        bank = [q for q in bank if q.difficulty == difficulty]
        if len(bank) < num_questions:
            bank = _static_question_bank() + ([dyn] if dyn else [])
    random.shuffle(bank)
    return bank[:num_questions]


# ------------- Score visuals -------------
def _score_gauge(pct: float) -> go.Figure:
    if pct >= 90:
        bar_color = "#f59e0b"  # gold-ish
    elif pct >= 75:
        bar_color = "#9ca3af"  # silver-ish
    elif pct >= 50:
        bar_color = "#cd7f32"  # bronze-ish
    else:
        bar_color = "#ef4444"  # red

    # Use RGBA strings (no 8-digit hex) for steps
    step_bg = "#2A2F36" if _is_dark() else "#f3f4f6"
    step_50_75 = "rgba(31, 111, 235, 0.20)"
    step_75_90 = "rgba(156, 163, 175, 0.25)"
    step_90_100 = "rgba(245, 158, 11, 0.25)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 50], "color": step_bg},
                    {"range": [50, 75], "color": step_50_75},
                    {"range": [75, 90], "color": step_75_90},
                    {"range": [90, 100], "color": step_90_100},
                ],
            },
            title={"text": "Your Score"},
        )
    )
    fig.update_layout(height=260)
    return _apply_theme(fig)


def _medal_for(pct: float) -> tuple[str, str]:
    if pct >= 90:
        return "🥇 Gold", "quiz_gold"
    if pct >= 75:
        return "🥈 Silver", "quiz_silver"
    if pct >= 50:
        return "🥉 Bronze", "quiz_bronze"
    return "🎖️ Participant", "quiz_participant"


# ------------- State -------------
def _reset_quiz(num_q: int, difficulty: str, timer_on: bool, seconds_per_q: int) -> None:
    st.session_state.quiz = {
        "questions": _assemble_quiz(num_q, difficulty),
        "index": 0,
        "correct": 0,
        "answers": [],
        "completed": False,
        "timer_on": timer_on,
        "seconds_per_q": seconds_per_q,
        "q_start": time.monotonic(),
        "best_pct": st.session_state.get("quiz", {}).get("best_pct", 0.0),
        "flash": None,  # brief feedback for previous question
    }


def _remaining_time() -> Optional[int]:
    q = st.session_state.quiz
    if not q.get("timer_on"):
        return None
    elapsed = int(time.monotonic() - q.get("q_start", time.monotonic()))
    return max(0, int(q.get("seconds_per_q", 20)) - elapsed)


# ------------- Page -------------
st.set_page_config(page_title="Quiz • RidePulse NYC", page_icon="🧠", layout="wide")
track_page_visit("Quiz")
award_badge("quizzer")

st.title("🧠 RidePulse Quiz")

with st.sidebar:
    st.header("Quiz Settings")
    num_questions = st.slider("Number of questions", 5, 10, 7, step=1)
    difficulty = st.selectbox("Difficulty", ["mixed", "easy", "medium", "hard"], index=0)
    timer_on = st.toggle("Enable timer per question", value=True, help="Your answer is counted wrong if time runs out.")
    seconds_per_q = st.slider("Seconds per question", 10, 45, 20, step=5, disabled=not timer_on)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start / Restart", use_container_width=True):
            _reset_quiz(num_questions, difficulty, timer_on, seconds_per_q)
            st.rerun()
    with c2:
        if st.button("Give up 😅", use_container_width=True):
            if "quiz" in st.session_state:
                st.session_state.quiz["completed"] = True
            st.rerun()

# Initialize quiz state if first load
if "quiz" not in st.session_state or not st.session_state.quiz.get("questions"):
    _reset_quiz(num_questions, difficulty, timer_on, seconds_per_q)

qstate = st.session_state.quiz
questions: List[QA] = qstate["questions"]
idx: int = qstate["index"]
completed: bool = qstate["completed"]

# Progress header
progress = int((idx / len(questions)) * 100) if not completed else 100
try:
    st.progress(progress, text=f"Progress: {idx}/{len(questions)}")
except TypeError:
    st.progress(progress)

# Flash summary for previous question
if qstate.get("flash"):
    f = qstate["flash"]
    if f.get("ok"):
        st.success(f"Previous: Correct — {f.get('answer')}")
    else:
        prefix = "Time's up! Marked incorrect." if f.get("timed_out") else "Incorrect."
        st.error(f"Previous: {prefix} Answer: {f.get('answer')}")
    if f.get("explanation"):
        with st.expander("Why?"):
            st.caption(f.get("explanation"))
    qstate["flash"] = None  # clear after display

# Active quiz or results
if not completed and idx < len(questions):
    qa: QA = questions[idx]

    # Stable option order per question
    options = qa.options[:]
    random.seed(42 + idx)
    random.shuffle(options)

    # Header row with timer and difficulty chip
    colA, colB = st.columns([4, 1])
    with colA:
        st.subheader(f"Question {idx + 1} of {len(questions)}")
    with colB:
        st.caption(f"Difficulty: {qa.difficulty.title()}")

    remaining = _remaining_time()
    if remaining is not None:
        st.caption(f"⏱️ Time left: {remaining}s")

    # Form: use the return value from radio (no session_state key)
    with st.form(key=f"qform_{idx}", clear_on_submit=False):
        choice = st.radio(qa.q, options=options, index=None)
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    if submitted:
        if choice is None:
            st.warning("Please select an option before submitting.")
        else:
            timed_out = remaining is not None and remaining <= 0
            ok_now = (choice == qa.answer) and not timed_out

            if ok_now:
                qstate["correct"] += 1

            # Record and prepare flash feedback
            qstate["answers"].append({"q": qa.q, "chosen": choice, "correct": qa.answer, "ok": ok_now})
            qstate["flash"] = {
                "chosen": choice,
                "answer": qa.answer,
                "ok": ok_now,
                "timed_out": timed_out,
                "explanation": qa.explanation,
            }

            # Advance to next question or finish
            qstate["index"] += 1
            if qstate["index"] >= len(questions):
                qstate["completed"] = True

            # Reset timer
            qstate["q_start"] = time.monotonic()

            # Quick celebratory note
            if ok_now:
                st.balloons()

            # Immediately display the next question (Streamlit 1.33 supports st.rerun)
            st.rerun()

else:
    # Results
    total = len(questions)
    correct = qstate["correct"]
    pct = round((correct / total) * 100.0, 1) if total else 0.0
    qstate["best_pct"] = max(qstate.get("best_pct", 0.0), pct)

    st.subheader("Results")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.plotly_chart(_score_gauge(pct), use_container_width=True, theme="streamlit")
    with c2:
        medal_label, badge_slug = _medal_for(pct)
        st.metric("Score", f"{correct}/{total} ({pct}%)")
        st.metric("Medal", medal_label)
        award_badge(badge_slug)
    with c3:
        st.metric("Personal best (this session)", f"{qstate.get('best_pct', pct)}%")

    # Celebrate with graphics
    if pct >= 90:
        show_lottie("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json", height=180)
        st.success("Outstanding! You’ve earned the Gold medal.")
    elif pct >= 75:
        st.success("Great job! Silver medal secured.")
    elif pct >= 50:
        st.info("Nice effort! You got a Bronze medal.")
    else:
        st.warning("Keep going! You’re close—try again for a medal.")

    # Storytelling snippets (Did you know?)
    st.markdown("---")
    st.markdown("### Did you know?")
    tips = [
        "Stations that are either very full or very empty are prime candidates for rebalancing—watch for those extremes on the Live Map.",
        "Percent full is more stable at larger stations because each dock change moves the needle less.",
        "Balance (bikes ≈ docks) is ideal right before rush hour to absorb demand in both directions.",
        "Capacity isn’t destiny—two stations with the same capacity can have very different live availability.",
    ]
    random.shuffle(tips)
    for tip in tips[:3]:
        st.markdown(f"- {tip}")

    # Review table
    st.markdown("---")
    st.markdown("#### Review your answers")
    review_rows = []
    for row in qstate["answers"]:
        review_rows.append(
            {
                "Question": row["q"],
                "Your answer": row["chosen"] if row["chosen"] is not None else "—",
                "Correct answer": row["correct"],
                "Result": "✅" if row["ok"] else "❌",
            }
        )
    st.dataframe(review_rows, use_container_width=True, hide_index=True)

    # Play again
    st.markdown("---")
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Play again", type="primary", use_container_width=True):
            _reset_quiz(num_questions, difficulty, timer_on, seconds_per_q)
            st.rerun()
    with colB:
        st.caption("Adjust settings in the sidebar and hit Start / Restart to try a different challenge.")