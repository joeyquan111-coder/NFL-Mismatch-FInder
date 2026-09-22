"""
app.py
------
Streamlit dashboard: pick a season/week, pull live data from nflverse,
and browse the ranked list of offense-vs-defense mismatches.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data_loader import load_all
from mismatch_engine import build_all_mismatches, STAT_POSITION_MAP

st.set_page_config(page_title="NFL Matchup Mismatch Finder", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_cached(season: int, week: int):
    """Cached wrapper so repeated visits/filters don't re-pull data from
    nflverse every time -- important on free hosting where each fetch
    costs real time and bandwidth. Cache expires after 1 hour."""
    data = load_all(season, week)
    mismatches = build_all_mismatches(
        data["weekly"], data["defense_allowed"], data["matchups"], week
    )
    return mismatches

st.title("🏈 NFL Offense vs. Defense Mismatch Finder")
st.caption(
    "Cross-references player form against opponent defensive weaknesses. "
    "This is descriptive research tooling, not a prediction of outcomes — "
    "always do your own diligence, and bet responsibly."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    season = st.number_input("Season", min_value=2015, max_value=2100, value=2025, step=1)
    week = st.number_input("Upcoming week to analyze", min_value=2, max_value=22, value=4, step=1)
    position_filter = st.multiselect(
        "Position filter", options=list(set(STAT_POSITION_MAP.values())),
        default=list(set(STAT_POSITION_MAP.values()))
    )
    stat_filter = st.multiselect(
        "Stat categories", options=list(STAT_POSITION_MAP.keys()),
        default=list(STAT_POSITION_MAP.keys())
    )
    min_score = st.slider("Minimum mismatch score", -2.0, 6.0, 1.0, 0.1)
    load_btn = st.button("Load / Refresh Data", type="primary")

if "data" not in st.session_state:
    st.session_state["data"] = None

if load_btn or st.session_state["data"] is None:
    with st.spinner("Pulling live data from nflverse (requires internet)..."):
        try:
            mismatches = load_all_cached(season, week)
            st.session_state["data"] = mismatches
            st.success(f"Loaded {len(mismatches)} player/stat rows for week {week}, {season}.")
        except Exception as e:
            st.error(
                f"Couldn't load data: {e}\n\n"
                "Common causes: no internet access from this machine, or nflverse "
                "hasn't published data for that season/week yet."
            )
            st.session_state["data"] = None

mismatches = st.session_state["data"]

if mismatches is not None and not mismatches.empty:
    filtered = mismatches[
        mismatches["position"].isin(position_filter)
        & mismatches["stat"].isin(stat_filter)
        & (mismatches["mismatch_score"] >= min_score)
    ]

    st.subheader(f"Top Mismatches — Week {week}, {season}")
    st.dataframe(
        filtered.style.format({
            "player_form": "{:.2f}", "player_form_z": "{:.2f}",
            "defense_allowed": "{:.2f}", "defense_allowed_z": "{:.2f}",
            "mismatch_score": "{:.2f}",
        }),
        use_container_width=True,
        height=500,
    )

    st.download_button(
        "Download filtered results as CSV",
        filtered.to_csv(index=False),
        file_name=f"nfl_mismatches_week{week}_{season}.csv",
        mime="text/csv",
    )

    st.subheader("Top 15 mismatches, visualized")
    top15 = filtered.head(15)
    if not top15.empty:
        fig = px.bar(
            top15.sort_values("mismatch_score"),
            x="mismatch_score", y="player_display_name",
            color="stat", orientation="h",
            hover_data=["recent_team", "opponent", "position"],
            labels={"mismatch_score": "Mismatch Score", "player_display_name": "Player"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rows meet the current filters — try lowering the minimum score.")

    with st.expander("How the mismatch score is calculated"):
        st.markdown(
            """
            For each stat (e.g. receiving yards for WRs):

            - **player_form_z**: the player's recency-weighted per-game average,
              z-scored against all players at that position (recent games count more).
            - **defense_allowed_z**: the upcoming opponent's recency-weighted
              per-game average *allowed* in that stat to that position,
              z-scored against the rest of the league. Higher = weaker defense.
            - **mismatch_score = player_form_z + defense_allowed_z**

            A high score means a player who's been performing well is about to
            face a defense that has struggled to contain that exact stat.

            **target_share** is a fraction (0–1): the share of the team's total
            targets a player has been getting, and the share of an opponent's
            pass attempts a defense has been conceding to that position. 0.28
            means 28% of targets/attempts.
            """
        )
else:
    st.info("Set your season/week in the sidebar and click **Load / Refresh Data** to begin.")
