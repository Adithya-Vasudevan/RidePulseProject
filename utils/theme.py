import streamlit as st

def inject_css():
    st.markdown(
        """
        <style>
        /* Adapt surfaces/colors to the active OS theme. Keep it subtle to work with Streamlit's theme. */

        @media (prefers-color-scheme: dark) {
          body {
            background: linear-gradient(140deg, #0d1117, #0f172a, #111827);
            background-size: 400% 400%;
            animation: gradientShift 28s ease infinite;
          }
          div[data-testid="stMetric"] {
            background: rgba(22,27,34,0.8);
            border: 1px solid #30363d;
            color: #c9d1d9;
          }
          .rp-fact {
            background: linear-gradient(180deg, #0f172a, #111827);
            border: 1px solid #30363d;
            color: #c9d1d9;
          }
          .rp-badge {
            background: #161b22;
            border: 1px solid #30363d;
            color: #c9d1d9;
          }
        }

        @media (prefers-color-scheme: light) {
          body {
            background: linear-gradient(120deg, #eef2ff, #f0f9ff, #fdf2f8);
            background-size: 400% 400%;
            animation: gradientShift 28s ease infinite;
          }
          div[data-testid="stMetric"] {
            background: #ffffffaa;
            border: 1px solid #e5e7eb;
            color: #111827;
          }
          .rp-fact {
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            border: 1px solid #e5e7eb;
            color: #0f172a;
          }
          .rp-badge {
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            color: #0f172a;
          }
        }

        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        /* Container card feel */
        .main .block-container {
          backdrop-filter: saturate(120%) blur(3px);
        }

        /* Headings accent underline */
        h2, h3 {
          position: relative;
        }
        h2::after, h3::after {
          content: "";
          position: absolute;
          left: 0;
          bottom: -6px;
          width: 64px;
          height: 4px;
          background: linear-gradient(90deg, #2563eb, #22c55e, #f59e0b);
          border-radius: 4px;
          opacity: 0.85;
        }

        /* Subtle hover lift for charts */
        .stPlotlyChart, .stPydeckGlJsonChart {
          transition: transform 200ms ease, box-shadow 200ms ease;
        }
        .stPlotlyChart:hover, .stPydeckGlJsonChart:hover {
          transform: translateY(-3px);
          box-shadow: 0 12px 28px rgba(0,0,0,0.08);
        }
        </style>
        """
        , unsafe_allow_html=True
    )