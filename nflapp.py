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

from data_loader import load_all, infer_current_season, infer_current_week
from mismatch_engine import build_all_mismatches, STAT_POSITION_MAP

st.set_page_config(page_title="NFL Matchup Mismatch Finder", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_default_season_week():
    """Auto-detects the current NFL season and week from real schedule
    dates, so the app opens pointed at 'right now' instead of a stale
    hardcoded default. Cached for 6 hours since this doesn't need to be
    recomputed on every rerun."""
    season = infer_current_season()
    try:
        week = infer_current_week(season)
    except Exception:
        week = 4
    # Week 1 has no prior week of data to build form/defense stats from,
    # so never default below week 2.
    week = max(week, 2)
    return season, week


DEFAULT_SEASON, DEFAULT_WEEK = get_default_season_week()


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
    season = st.number_input(
        "Season", min_value=2015, max_value=2100, value=DEFAULT_SEASON, step=1,
        help="Auto-detected from today's date. Change it to look at a different season."
    )
    week = st.number_input(
        "Upcoming week to analyze", min_value=2, max_value=22, value=DEFAULT_WEEK, step=1,
        help="Auto-detected as the current/next NFL week based on real game dates."
    )
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
            - **defense_allowed_z**: how weak the upcoming opponent's defense is
              against that stat, z-scored against the rest of the league. Higher =
              weaker defense. As of the latest update, this is **not** just a raw
              per-game total — see "Why defense_allowed_z isn't a raw total" below.
            - **mismatch_score = player_form_z + defense_allowed_z**

            A high score means a player who's been performing well is about to
            face a defense that has struggled to contain that exact stat.

            **opportunity** is context, not part of the score: it shows each
            player's recent usage/volume, using whichever measure fits their
            position — target share for WR/TE, rush attempts per game for RB,
            pass attempts per game for QB. It's there to help you judge whether
            a mismatch is backed by real opportunity or just a couple of big plays.

            ---

            **Why defense_allowed_z isn't a raw total**

            A defense's raw yards/TDs-allowed total mixes together two different
            things: how many plays it actually faced (which depends on the pace
            and pass/run tendencies of the offenses it happened to play) and how
            well it defended each one. A defense that faces a lot of pass attempts
            will rack up passing yards allowed even if it's efficient on a
            per-play basis — and vice versa, a defense that's only faced a few
            attempts can hide a real weakness behind a low total.

            To correct for this, `defense_allowed_z` is built from **rate stats**
            instead — the same recency weighting is applied, but to per-attempt
            or per-target efficiency rather than raw totals:

            | Stat | What actually drives the weakness score |
            |---|---|
            | Passing yards | Yards per attempt allowed, completion % allowed, completions per game allowed |
            | Passing TDs | TD rate per pass attempt allowed |
            | Rushing yards | Yards per carry allowed, rush attempts per game allowed |
            | Rushing TDs | TD rate per carry allowed |
            | Receiving yards | Yards per target allowed, catch rate allowed, targets per game allowed |
            | Receiving TDs | TD rate per target allowed |
            | Receptions | Catch rate allowed, targets per game allowed |

            When a stat has more than one rate listed, each is z-scored
            separately and then averaged into one composite — so, for example,
            a defense isn't rated as pass-weak *just* because it faces a lot of
            attempts, but it also isn't let off the hook for being genuinely
            bad on a per-attempt basis. The raw per-game total (shown in the
            `defense_allowed` column) is still displayed for reference — it's
            just no longer what drives the score.

            **A caution on small samples**: rate stats can swing hard early in a
            season when a defense has only faced a handful of attempts or
            carries — one long touchdown run can make a TD rate look extreme.
            Treat scores built on just 1-2 games of opponent data with some
            skepticism until more weeks are in.
            """
        )
else:
    st.info("Set your season/week in the sidebar and click **Load / Refresh Data** to begin.")
