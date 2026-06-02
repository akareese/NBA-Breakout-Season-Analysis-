import streamlit as st
import pandas as pd

st.set_page_config(page_title="NBA Breakout Analysis", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🏀 NBA Breakout Player Detection</h1>
    <p style='text-align: center; color: gray; font-size: 1.1rem;'>2024 → 2025 Season Comparison</p>
    <hr>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("breakouts_2024_to_2025.csv")

df = load_data()

st.sidebar.header("Filters")

positions = ["All"] + sorted(df["Pos"].dropna().unique().tolist())
selected_pos = st.sidebar.selectbox("Position", positions)

teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", teams)

min_score, max_score = float(df["Breakout Score"].min()), float(df["Breakout Score"].max())
score_range = st.sidebar.slider(
    "Breakout Score Range",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
    step=0.1
)

top_n = st.sidebar.slider("Show Top N Players", min_value=5, max_value=len(df), value=50, step=5)

filtered = df.copy()
if selected_pos != "All":
    filtered = filtered[filtered["Pos"] == selected_pos]
if selected_team != "All":
    filtered = filtered[filtered["Team"] == selected_team]
filtered = filtered[
    (filtered["Breakout Score"] >= score_range[0]) &
    (filtered["Breakout Score"] <= score_range[1])
]
filtered = filtered.head(top_n).reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Players Shown", len(filtered))
col2.metric("Top Breakout Score", f"{filtered['Breakout Score'].max():.1f}" if not filtered.empty else "—")
col3.metric("Avg Breakout Score", f"{filtered['Breakout Score'].mean():.2f}" if not filtered.empty else "—")

st.markdown("### 📊 Breakout Leaderboard")

delta_cols = ["MPG Δ", "PPG Δ", "APG Δ", "RPG Δ", "SPG Δ", "FT% Δ", "FG% Δ", "3P% Δ"]
present_delta_cols = [c for c in delta_cols if c in filtered.columns]

def style_table(row):
    styles = [""] * len(row)
    idx = row.index.tolist()
    if "Breakout Score" in idx:
        val = row["Breakout Score"]
        i = idx.index("Breakout Score")
        if val > 3:
            styles[i] = "background-color: #1a7a3a; color: white; font-weight: bold;"
        elif val > 1:
            styles[i] = "background-color: #a07800; color: white; font-weight: bold;"
        elif val < 0:
            styles[i] = "background-color: #8b1a1a; color: white; font-weight: bold;"
        else:
            styles[i] = "background-color: #2a5a2a; color: white; font-weight: bold;"
    for col in present_delta_cols:
        if col in idx:
            val = row[col]
            i = idx.index(col)
            try:
                if val > 0:
                    styles[i] = "color: #4caf50;"
                elif val < 0:
                    styles[i] = "color: #f44336;"
            except:
                pass
    return styles

fmt = {col: "{:+.1f}" for col in present_delta_cols}
fmt["Breakout Score"] = "{:.1f}"

styled = filtered.style.apply(style_table, axis=1).format(fmt)
st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("### 📈 Breakout Score Chart")
st.markdown("""
> **How to read this chart:** Each bar represents a player's **Breakout Score** — a composite metric that measures 
> how much a player improved from the 2024 season to the 2025 season. The score is calculated using a weighted 
> formula that factors in Points per 36 minutes (55%), True Shooting % (30%), Assists per 36 minutes (10%), 
> and Turnovers per 36 minutes (5%). A **higher score means a stronger breakout season**. Players with a score 
> above 3 are considered standout breakouts, while negative scores indicate a statistical decline.
""")

chart_data = filtered.set_index("Player")[["Breakout Score"]].sort_values("Breakout Score", ascending=True)
st.bar_chart(chart_data)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Data sourced from Basketball-Reference · Built with Python, Pandas & Streamlit</p>",
    unsafe_allow_html=True
)
