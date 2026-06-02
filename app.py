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
    df = pd.read_csv("breakouts_2024_to_2025.csv")
    return df

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

top_n = st.sidebar.slider("Show Top N Players", min_value=5, max_value=len(df), value=15, step=5)

filtered = df.copy()
if selected_pos != "All":
    filtered = filtered[filtered["Pos"] == selected_pos]
if selected_team != "All":
    filtered = filtered[filtered["Team"] == selected_team]
filtered = filtered[
    (filtered["Breakout Score"] >= score_range[0]) &
    (filtered["Breakout Score"] <= score_range[1])
]
filtered = filtered.head(top_n)

col1, col2, col3 = st.columns(3)
col1.metric("Players Shown", len(filtered))
col2.metric("Top Breakout Score", f"{filtered['Breakout Score'].max():.1f}" if not filtered.empty else "—")
col3.metric("Avg Breakout Score", f"{filtered['Breakout Score'].mean():.2f}" if not filtered.empty else "—")

st.markdown("### 📊 Breakout Leaderboard")

def color_score(val):
    if val > 3:
        return "background-color: #d4edda; color: #155724;"
    elif val > 1:
        return "background-color: #fff3cd; color: #856404;"
    elif val < 0:
        return "background-color: #f8d7da; color: #721c24;"
    return ""

def color_delta(val):
    try:
        if val > 0:
            return "color: green;"
        elif val < 0:
            return "color: red;"
    except:
        pass
    return ""

delta_cols = ["MPG Δ", "PPG Δ", "APG Δ", "RPG Δ", "SPG Δ", "FT% Δ", "FG% Δ", "3P% Δ"]

styled = filtered.style \
    .applymap(color_score, subset=["Breakout Score"]) \
    .applymap(color_delta, subset=[c for c in delta_cols if c in filtered.columns]) \
    .format({col: "{:+.1f}" for col in delta_cols if col in filtered.columns}) \
    .format({"Breakout Score": "{:.1f}"})

st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("### 📈 Breakout Score Chart")

chart_data = filtered.set_index("Player")[["Breakout Score"]].sort_values("Breakout Score", ascending=True)
st.bar_chart(chart_data)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Data sourced from Basketball-Reference · Built with Python, Pandas & Streamlit</p>",
    unsafe_allow_html=True
)
