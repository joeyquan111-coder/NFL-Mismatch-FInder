# NFL Matchup Mismatch Finder

Cross-references NFL player performance ("form") with opponent defensive
weaknesses to surface the best statistical mismatches for the upcoming week —
useful for researching player props. Data comes free from
[nflverse](https://github.com/nflverse) via the `nfl_data_py` package.

**This is a research tool, not a prediction engine.** It surfaces
statistical patterns; it doesn't account for injuries, weather, game script,
coaching tendencies, or line movement. Use it as one input among many, and
gamble responsibly (see resources at the bottom).

## What it does

1. Pulls every player's weekly box score for the season (`data_loader.py`).
2. Aggregates those box scores by opponent to see what each **defense**
   allows, by position (e.g. "yards allowed to WRs" per team).
3. For a chosen upcoming week, computes two z-scores per player/stat:
   - How hot the player has been recently, vs. their position peers
   - How weak their upcoming opponent's defense is at stopping that
     exact stat, vs. the rest of the league
4. Adds them into a single **mismatch score** and ranks everyone
   (`mismatch_engine.py`).
5. Displays it all in an interactive, filterable, sortable dashboard
   with a chart and CSV export (`app.py`).

Stat categories covered out of the box: passing yards/TDs (QB), rushing
yards/TDs (RB), receiving yards/TDs and receptions (WR). Easy to extend —
see "Extending it" below.

## Setup

Requires Python 3.10+ and **internet access** (this was built in a sandboxed
environment with no internet, so run it on your own machine).

```bash
cd nfl_matchups
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the dashboard in your browser (usually
`http://localhost:8501`).

## Using it

1. In the sidebar, set the **season** and the **upcoming week** you want
   mismatches for (data through the prior week is used to build form/defense
   stats — you can't analyze week 1 since there's no prior data yet).
2. Click **Load / Refresh Data**. First load takes a bit since it's pulling
   a full season of play-by-play-derived stats.
3. Filter by position, stat category, and minimum mismatch score.
4. Sort the table, download as CSV, or eyeball the top-15 chart.

## Extending it

- **More stats**: add entries to `STAT_POSITION_MAP` in `mismatch_engine.py`
  (e.g. `"rushing_yards": "RB"` is already there — you could add
  `"targets": "WR"` for target-share-driven props).
- **Home/away splits**: `is_home` is already carried through if you want to
  weight road defenses differently.
- **Recency sensitivity**: tweak `RECENCY_HALF_LIFE_WEEKS` in
  `mismatch_engine.py` — lower = more weight on the last couple games.
- **Vegas lines**: if you have access to an odds API, you could merge in
  prop lines and compute your own "edge" (mismatch score vs. how the market
  has priced it) rather than raw z-scores.
- **Injuries**: `nfl_data_py` also exposes injury reports
  (`nfl.import_injuries()`) — worth merging in and flagging players who are
  questionable/out, since form scores don't know about that.

## A note on responsible use

Sports betting carries real financial risk. This tool is meant to support
your own research and judgment, not replace it — treat any single stat
mismatch skeptically and consider matchup context (injuries, weather,
game script) before acting on it. If betting stops being fun or feels hard
to control, the National Council on Problem Gambling helpline (US) is
1-800-522-4700, available 24/7.
