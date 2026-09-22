"""
mismatch_engine.py
-------------------
Turns raw weekly stats + defense-allowed stats into a ranked list of
"best mismatches": players whose own recent form is strong AND whose
upcoming opponent is weak against that position/stat.

The core idea, per stat category (e.g. receiving_yards for WRs):

  player_form_z   = how far above/below the position average this
                     player's per-game rate has been (recency-weighted)
  defense_weak_z   = how far above the league average the upcoming
                     opponent's defense allows that same stat, to that
                     same position group
  mismatch_score   = player_form_z + defense_weak_z

Both halves are z-scores (mean 0, std 1) so they're comparable and
addable regardless of raw units (yards vs touchdowns vs receptions).
A high positive score means "hot player vs. defense that struggles to
stop this exact thing" -- the kind of spot bettors look for on player
props. This is descriptive analytics, not a prediction of outcomes --
treat it as a research aid, not a guarantee.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

# stat -> which position group it's meaningful for
STAT_POSITION_MAP = {
    "passing_yards": "QB",
    "passing_tds": "QB",
    "rushing_yards": "RB",
    "rushing_tds": "RB",
    "receiving_yards": "WR",
    "receiving_tds": "WR",
    "receptions": "WR",
}

RECENCY_HALF_LIFE_WEEKS = 4  # more recent games weigh more


def _recency_weights(weeks: pd.Series, current_week: int) -> np.ndarray:
    age = (current_week - weeks).clip(lower=0)
    return 0.5 ** (age / RECENCY_HALF_LIFE_WEEKS)


def player_form_scores(weekly: pd.DataFrame, stat: str, current_week: int) -> pd.DataFrame:
    """Recency-weighted per-game average for `stat`, z-scored within position."""
    position = STAT_POSITION_MAP[stat]
    df = weekly[weekly["position"] == position].copy()
    df = df[df["week"] < current_week]
    if df.empty:
        return pd.DataFrame(columns=["player_id", "player_display_name", "recent_team",
                                      "position", f"{stat}_form", f"{stat}_form_z"])

    df["_w"] = _recency_weights(df["week"], current_week)
    df["_weighted_stat"] = df[stat] * df["_w"]

    grouped = df.groupby(["player_id", "player_display_name", "recent_team", "position"]).apply(
        lambda g: g["_weighted_stat"].sum() / g["_w"].sum()
    ).reset_index(name=f"{stat}_form")

    mean, std = grouped[f"{stat}_form"].mean(), grouped[f"{stat}_form"].std(ddof=0)
    std = std if std > 0 else 1.0
    grouped[f"{stat}_form_z"] = (grouped[f"{stat}_form"] - mean) / std
    return grouped


def defense_weakness_scores(defense_allowed: pd.DataFrame, stat: str, current_week: int) -> pd.DataFrame:
    """Recency-weighted per-game average of `stat` ALLOWED by each defense,
    z-scored across the league. Positive z = allows more than average = weak."""
    position = STAT_POSITION_MAP[stat]
    df = defense_allowed[defense_allowed["position"] == position].copy()
    df = df[df["week"] < current_week]
    if df.empty:
        return pd.DataFrame(columns=["defense_team", f"{stat}_allowed", f"{stat}_allowed_z"])

    df["_w"] = _recency_weights(df["week"], current_week)
    df["_weighted_stat"] = df[stat] * df["_w"]

    grouped = df.groupby("defense_team").apply(
        lambda g: g["_weighted_stat"].sum() / g["_w"].sum()
    ).reset_index(name=f"{stat}_allowed")

    mean, std = grouped[f"{stat}_allowed"].mean(), grouped[f"{stat}_allowed"].std(ddof=0)
    std = std if std > 0 else 1.0
    grouped[f"{stat}_allowed_z"] = (grouped[f"{stat}_allowed"] - mean) / std
    return grouped


def build_mismatch_table(
    weekly: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    matchups: pd.DataFrame,
    stat: str,
    current_week: int,
) -> pd.DataFrame:
    """
    Full pipeline for one stat category: player form + opponent weakness,
    joined through this week's schedule, sorted by mismatch_score desc.
    """
    form = player_form_scores(weekly, stat, current_week)
    weakness = defense_weakness_scores(defense_allowed, stat, current_week)

    # Attach each player's upcoming opponent via their current team.
    team_opp = matchups[["team", "opponent", "is_home"]].rename(
        columns={"team": "recent_team"}
    )
    merged = form.merge(team_opp, on="recent_team", how="inner")
    merged = merged.merge(
        weakness, left_on="opponent", right_on="defense_team", how="inner"
    )

    merged["mismatch_score"] = merged[f"{stat}_form_z"] + merged[f"{stat}_allowed_z"]
    merged["stat"] = stat
    merged = merged.sort_values("mismatch_score", ascending=False).reset_index(drop=True)

    display_cols = [
        "player_display_name", "position", "recent_team", "opponent", "is_home",
        f"{stat}_form", f"{stat}_form_z",
        f"{stat}_allowed", f"{stat}_allowed_z",
        "mismatch_score", "stat",
    ]
    return merged[display_cols]


def build_all_mismatches(
    weekly: pd.DataFrame,
    defense_allowed: pd.DataFrame,
    matchups: pd.DataFrame,
    current_week: int,
) -> pd.DataFrame:
    """Runs build_mismatch_table for every tracked stat and stacks the results
    into one long table with a common column set for easy filtering/sorting."""
    frames = []
    for stat in STAT_POSITION_MAP:
        t = build_mismatch_table(weekly, defense_allowed, matchups, stat, current_week)
        t = t.rename(columns={
            f"{stat}_form": "player_form",
            f"{stat}_form_z": "player_form_z",
            f"{stat}_allowed": "defense_allowed",
            f"{stat}_allowed_z": "defense_allowed_z",
        })
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("mismatch_score", ascending=False).reset_index(drop=True)
