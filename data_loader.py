"""
data_loader.py
---------------
Pulls free, public NFL data from nflverse (via the nflreadpy package) and
shapes it into three tables used by the rest of the app:

1. weekly_player_stats   -> one row per player per week (offense box score)
2. team_defense_allowed  -> one row per team per week per position group,
                             showing what that team's DEFENSE allowed
3. upcoming_matchups     -> this week's schedule, so we know who plays whom

nflreadpy wraps the nflverse-data GitHub releases, which are updated
weekly during the season and are free to use. No API key required.

NOTE: This was switched from the older `nfl_data_py` package, which was
deprecated and archived in September 2025 and stopped reliably serving
current-season data (hence 404 errors on recent seasons). `nflreadpy` is
the actively maintained replacement, but it returns Polars DataFrames
(not pandas) and uses slightly different column names in places, so this
file converts to pandas immediately and normalizes column names so the
rest of the app (mismatch_engine.py, app.py) doesn't need to change at all.

NOTE: This machine building the code has no internet access, so these
functions are written against nflreadpy's documented/known schema but
have NOT been executed against live data. When you run this on your own
machine (with internet), if a column name has changed upstream, the
functions will raise a clear error telling you what was expected -- just
run `df.columns.tolist()` on the offending frame and adjust the constant
lists or the _first_present() lookups below to match.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import nflreadpy as nfl

# ---------------------------------------------------------------------------
# Columns we expect from nflreadpy.load_player_stats(). If nflverse renames
# something, tweak these lists -- everything downstream reads from them.
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


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Returns the first of `candidates` that's actually a column in df.
    Used because nflreadpy has renamed a field or two vs. the old package
    (e.g. team column has appeared as either 'team' or 'recent_team')."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    df = nfl.load_player_stats([season]).to_pandas()

    # Normalize the "current team" column name across nflreadpy versions.
    team_col = _first_present(df, ["recent_team", "team"])
    if team_col is None:
        raise ValueError(
            "Couldn't find a team column in load_player_stats() output. "
            f"Available columns were: {sorted(df.columns.tolist())}"
        )
    if team_col != "recent_team":
        df = df.rename(columns={team_col: "recent_team"})

    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    keep = [c for c in ID_COLS + OFFENSE_STAT_COLS if c in df.columns]
    _require_columns(df, ["player_id", "position", "recent_team", "week"], "load_player_stats")
    df = df[keep].copy()
    df = df[df["position"].isin(DEFENSE_TRACKED_POSITIONS)]

    # --- target share -------------------------------------------------
    # What fraction of the TEAM's total targets that game went to this
    # player. This captures usage/opportunity independent of whether the
    # catches actually turned into yards or scores that particular week,
    # which makes it a steadier signal for prop research than raw yards.
    # Denominator = sum of targets across all tracked pass-catching
    # positions (WR/RB/TE) on that team in that game. QBs have 0 targets
    # so they don't distort the sum.
    if "targets" in df.columns:
        team_week_targets = df.groupby(["recent_team", "week"])["targets"].transform("sum")
        df["target_share"] = np.where(
            team_week_targets > 0, df["targets"] / team_week_targets, np.nan
        )

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

    # --- target share allowed -----------------------------------------
    # For each defense-week-position row, what fraction of the OPPOSING
    # OFFENSE's total pass attempts that game went to this position group.
    # We derive the offense's total attempts from the QB rows already in
    # `weekly` (summed, in case of a QB change mid-game), then match each
    # defense-week to the one offense that played them that week.
    if "targets" in allowed.columns and "attempts" in weekly.columns:
        qb_attempts = (
            weekly[weekly["position"] == "QB"]
            .groupby(["recent_team", "opponent_team", "week"])["attempts"]
            .sum()
            .reset_index()
            .rename(columns={
                "recent_team": "offense_team",
                "opponent_team": "defense_team",
                "attempts": "opponent_pass_attempts",
            })
        )
        allowed = allowed.merge(qb_attempts, on=["defense_team", "week"], how="left")
        allowed["target_share"] = np.where(
            allowed["opponent_pass_attempts"] > 0,
            allowed["targets"] / allowed["opponent_pass_attempts"],
            np.nan,
        )
        allowed = allowed.drop(columns=["offense_team", "opponent_pass_attempts"])

    return allowed


def load_schedules(season: int) -> pd.DataFrame:
    """Full-season schedule, used to find each team's upcoming opponent."""
    sched = nfl.load_schedules([season]).to_pandas()
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
