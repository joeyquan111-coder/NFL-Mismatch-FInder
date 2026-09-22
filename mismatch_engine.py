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

# ---------------------------------------------------------------------------
# Opportunity context: NOT a scored stat, doesn't affect mismatch_score.
# Shown alongside every row as a plain-English usage signal so you can see
# whether a player's volume backs up the mismatch, or if their form number
# is being carried by a couple of big plays. Which raw stat is used, and how
# it's formatted, depends on the player's position:
#   WR/TE -> target share (their share of the team's targets)
#   RB    -> rush attempts per game (target share isn't meaningful for a runner)
#   QB    -> pass attempts per game
# ---------------------------------------------------------------------------
OPPORTUNITY_CONTEXT = {
    "QB": ("attempts", lambda v: f"{v:.1f} pass att/gm"),
    "RB": ("carries", lambda v: f"{v:.1f} rush att/gm"),
    "WR": ("target_share", lambda v: f"{v:.0%} target share"),
    "TE": ("target_share", lambda v: f"{v:.0%} target share"),
}

RECENCY_HALF_LIFE_WEEKS = 4  # more recent games weigh more

# ---------------------------------------------------------------------------
# Rate-based defense weakness components. A defense's raw yards/TDs-allowed
# total conflates two different things: how many plays/opportunities it
# faced (pace/volume, driven by the offenses it happened to play) and how
# well it actually defended each one (efficiency). Every scored stat below
# gets a composite weakness score built from rate components instead of a
# single raw total, so a defense that "allows a lot" only because it faced
# high-volume offenses isn't mistaken for one that's actually bad at
# defending the position.
#
# Each component is either:
#   ("ratio", numerator_col, denominator_col) -- a true rate, e.g. yards per
#       attempt or TDs per opportunity, computed as (recency-weighted total
#       numerator) / (recency-weighted total denominator) across recent
#       games -- NOT an average of weekly ratios, which would let
#       low-attempt weeks skew things.
#   ("avg", col, None)                        -- a plain recency-weighted
#       per-game average, used for volume context (e.g. completions/game,
#       targets/game) alongside the efficiency rates.
# ---------------------------------------------------------------------------
RATE_COMPONENTS = {
    "passing_yards": [
        ("ratio", "passing_yards", "attempts"),   # yards per attempt allowed
        ("ratio", "completions", "attempts"),     # completion % allowed
        ("avg", "completions", None),             # completions per game allowed
    ],
    "passing_tds": [
        ("ratio", "passing_tds", "attempts"),     # TD rate per pass attempt allowed
    ],
    "rushing_yards": [
        ("ratio", "rushing_yards", "carries"),    # yards per carry allowed
        ("avg", "carries", None),                 # rush attempts per game allowed
    ],
    "rushing_tds": [
        ("ratio", "rushing_tds", "carries"),      # TD rate per carry allowed
    ],
    "receiving_yards": [
        ("ratio", "receiving_yards", "targets"),  # yards per target allowed
        ("ratio", "receptions", "targets"),       # catch rate allowed
        ("avg", "targets", None),                 # targets per game allowed
    ],
    "receiving_tds": [
        ("ratio", "receiving_tds", "targets"),    # TD rate per target allowed
    ],
    "receptions": [
        ("ratio", "receptions", "targets"),       # catch rate allowed
        ("avg", "targets", None),                 # targets per game allowed
    ],
}


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


def defense_weakness_rate_z(defense_allowed: pd.DataFrame, stat: str, current_week: int) -> pd.DataFrame | None:
    """
    Pace/volume-adjusted alternative to the raw-total z-score, for stats
    listed in RATE_COMPONENTS. Builds each component (a true rate for
    "ratio" components, a recency-weighted per-game average for "avg"
    components), z-scores each across the league, then averages those
    z-scores into one composite. Returns None if `stat` isn't in
    RATE_COMPONENTS (caller should fall back to the raw-total z-score).
    """
    components = RATE_COMPONENTS.get(stat)
    if not components:
        return None

    position = STAT_POSITION_MAP[stat]
    df = defense_allowed[(defense_allowed["position"] == position) & (defense_allowed["week"] < current_week)].copy()
    if df.empty:
        return None
    df["_w"] = _recency_weights(df["week"], current_week)

    z_series_list = []
    for kind, num_col, den_col in components:
        if num_col not in df.columns or (den_col and den_col not in df.columns):
            continue
        tmp = df.copy()
        tmp["_wn"] = tmp[num_col] * tmp["_w"]
        if kind == "ratio":
            tmp["_wd"] = tmp[den_col] * tmp["_w"]
            grouped = tmp.groupby("defense_team").apply(
                lambda g: g["_wn"].sum() / g["_wd"].sum() if g["_wd"].sum() > 0 else np.nan
            )
        else:  # "avg"
            grouped = tmp.groupby("defense_team").apply(
                lambda g: g["_wn"].sum() / g["_w"].sum()
            )
        mean_, std_ = grouped.mean(), grouped.std(ddof=0)
        std_ = std_ if std_ > 0 else 1.0
        z_series_list.append((grouped - mean_) / std_)

    if not z_series_list:
        return None

    composite = pd.concat(z_series_list, axis=1).mean(axis=1)
    return composite.reset_index(name=f"{stat}_allowed_z_composite")


def compute_opportunity_context(weekly: pd.DataFrame, current_week: int) -> pd.DataFrame:
    """
    Recency-weighted per-game usage metric for each player, using whichever
    raw stat is meaningful for their position (see OPPORTUNITY_CONTEXT).
    This is purely informational context -- it is never z-scored and never
    added into mismatch_score.
    """
    frames = []
    for position, (stat, formatter) in OPPORTUNITY_CONTEXT.items():
        df = weekly[(weekly["position"] == position) & (weekly["week"] < current_week)].copy()
        if df.empty or stat not in df.columns:
            continue

        df["_w"] = _recency_weights(df["week"], current_week)
        df["_weighted_stat"] = df[stat] * df["_w"]

        grouped = df.groupby(["player_id", "recent_team"]).apply(
            lambda g: g["_weighted_stat"].sum() / g["_w"].sum()
        ).reset_index(name="_raw_value")
        grouped["opportunity"] = grouped["_raw_value"].apply(
            lambda v: formatter(v) if pd.notna(v) else None
        )
        frames.append(grouped[["player_id", "recent_team", "opportunity"]])

    if not frames:
        return pd.DataFrame(columns=["player_id", "recent_team", "opportunity"])
    return pd.concat(frames, ignore_index=True)


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

    # For stats with rate-based components defined (see RATE_COMPONENTS),
    # replace the raw-total z-score with the pace/volume-adjusted composite
    # before scoring. The raw f"{stat}_allowed" total is left untouched so
    # it still displays as familiar "yards allowed per game" context.
    composite = defense_weakness_rate_z(defense_allowed, stat, current_week)
    if composite is not None:
        composite = composite.rename(columns={"defense_team": "opponent"})
        merged = merged.merge(composite, on="opponent", how="left")
        merged[f"{stat}_allowed_z"] = merged[f"{stat}_allowed_z_composite"].combine_first(
            merged[f"{stat}_allowed_z"]
        )
        merged = merged.drop(columns=[f"{stat}_allowed_z_composite"])

    merged["mismatch_score"] = merged[f"{stat}_form_z"] + merged[f"{stat}_allowed_z"]
    merged["stat"] = stat
    merged = merged.sort_values("mismatch_score", ascending=False).reset_index(drop=True)

    display_cols = [
        "player_id", "player_display_name", "position", "recent_team", "opponent", "is_home",
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
    into one long table with a common column set for easy filtering/sorting.
    Also attaches the (unscored) opportunity context column."""
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

    opportunity = compute_opportunity_context(weekly, current_week)
    out = out.merge(opportunity, on=["player_id", "recent_team"], how="left")

    # Put opportunity right after is_home, drop the now-unneeded id column.
    ordered_cols = [
        "player_display_name", "position", "recent_team", "opponent", "is_home",
        "opportunity",
        "player_form", "player_form_z", "defense_allowed", "defense_allowed_z",
        "mismatch_score", "stat",
    ]
    out = out[ordered_cols]

    return out.sort_values("mismatch_score", ascending=False).reset_index(drop=True)
