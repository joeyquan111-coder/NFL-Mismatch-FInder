"""
data_loader.py
---------------
Pulls free, public NFL data from nflverse (via the nfl_data_py package) and
shapes it into three tables used by the rest of the app:

1. weekly_player_stats   -> one row per player per week (offense box score)
2. team_defense_allowed  -> one row per team per week per position group,
                             showing what that team's DEFENSE allowed
3. upcoming_matchups     -> this week's schedule, so we know who plays whom

nfl_data_py wraps the nflverse-data GitHub releases, which are updated
weekly during the season and are free to use. No API key required.

NOTE: This machine building the code has no internet access, so these
functions are written against nfl_data_py's documented/known schema but
have NOT been executed against live data. When you run this on your own
machine (with internet), if a column name has changed upstream, the
functions will raise a clear error telling you what was expected -- just
run `df.columns.tolist()` on the offending frame and adjust the constant
lists at the top of this file.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import nfl_data_py as nfl

# ---------------------------------------------------------------------------
# Columns we expect from nfl_data_py.import_weekly_data(). If nflverse
# renames something, tweak this list -- everything downstream reads from it.
# ---------------------------------------------------------------------------
ID_COLS = [
    "player_id", "player_display_name", "position", "position_group",
    "recent_team", "opponent_team", "season", "week", "season_type",
]

OFFENSE_STAT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "fantasy_points_ppr",
]

# Position groups we build defense-allowed splits for.
DEFENSE_TRACKED_POSITIONS = ["QB", "RB", "WR", "TE"]


def _require_columns(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] Missing expected columns: {missing}\n"
            f"Available columns were: {sorted(df.columns.tolist())}\n"
            f"nflverse likely renamed a field -- update the constant lists "
            f"at the top of data_loader.py to match."
        )


def load_weekly_player_stats(season: int) -> pd.DataFrame:
    """One row per player per week: the offensive box score."""
    df = nfl.import_weekly_data([season])
    df = df[df["season_type"] == "REG"].copy()
    keep = [c for c in ID_COLS + OFFENSE_STAT_COLS if c in df.columns]
    _require_columns(df, ["player_id", "position", "recent_team", "week"], "import_weekly_data")
    df = df[keep].copy()
    df = df[df["position"].isin(DEFENSE_TRACKED_POSITIONS)]
    return df


def load_team_defense_allowed(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the offensive box scores by OPPONENT (i.e. by the defense
    that gave them up) and by position group, producing a week-by-week
    "what did this defense allow" table.
    """
    if "opponent_team" not in weekly.columns:
        raise ValueError(
            "weekly data has no 'opponent_team' column -- cannot compute "
            "defense-allowed stats. Merge in schedules to derive opponent "
            "instead (see load_schedules())."
        )

    group_cols = ["opponent_team", "position", "week"]
    agg_cols = [c for c in OFFENSE_STAT_COLS if c in weekly.columns]

    allowed = (
        weekly.groupby(group_cols)[agg_cols]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team"})
    )
    return allowed


def load_schedules(season: int) -> pd.DataFrame:
    """Full-season schedule, used to find each team's upcoming opponent."""
    sched = nfl.import_schedules([season])
    keep = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
    keep = [c for c in keep if c in sched.columns]
    return sched[keep].copy()


def get_upcoming_week_matchups(schedules: pd.DataFrame, week: int) -> pd.DataFrame:
    """Returns home/away pairs for a given week, expanded to one row per team
    with an 'opponent' column, so it can be merged onto player stats."""
    wk = schedules[schedules["week"] == week].copy()
    home = wk.rename(columns={"home_team": "team", "away_team": "opponent"})
    away = wk.rename(columns={"away_team": "team", "home_team": "opponent"})
    home["is_home"] = True
    away["is_home"] = False
    cols = ["game_id", "week", "team", "opponent", "is_home"]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def load_all(season: int, upcoming_week: int) -> dict[str, pd.DataFrame]:
    """Convenience loader used by the Streamlit app."""
    weekly = load_weekly_player_stats(season)
    defense_allowed = load_team_defense_allowed(weekly)
    schedules = load_schedules(season)
    matchups = get_upcoming_week_matchups(schedules, upcoming_week)
    return {
        "weekly": weekly,
        "defense_allowed": defense_allowed,
        "matchups": matchups,
    }
