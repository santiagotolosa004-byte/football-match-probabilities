"""Pre-match features for the machine-learning model.

Every feature for a match is computed **only from matches played before it**
(no look-ahead / data leakage), exactly as it would be available on match day.

* Elo ratings (updated after every match, goal-difference aware)
* Recent form per team, last 5 and last 10 matches: points, goals, shots,
  shots on target, corners (for and against)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FORM_STATS = ["points", "goals_for", "goals_against", "shots_for", "shots_against",
              "sot_for", "sot_against", "corners_for", "corners_against"]
WINDOWS = (5, 10)


def elo_ratings(matches: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0,
                start: float = 1500.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return pre-match Elo for each match and the final ratings."""
    ratings: dict[str, float] = {}
    pre_h, pre_a = np.empty(len(matches)), np.empty(len(matches))
    for n, m in enumerate(matches.itertuples(index=False)):
        rh, ra = ratings.get(m.home, start), ratings.get(m.away, start)
        pre_h[n], pre_a[n] = rh, ra
        expected_h = 1 / (1 + 10 ** ((ra - rh - home_adv) / 400))
        result_h = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        margin = abs(m.home_goals - m.away_goals)
        mult = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        delta = k * mult * (result_h - expected_h)
        ratings[m.home], ratings[m.away] = rh + delta, ra - delta
    out = pd.DataFrame({"elo_home": pre_h, "elo_away": pre_a}, index=matches.index)
    return out, ratings


def team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match."""
    home = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"], "team": matches["home"],
        "goals_for": matches["home_goals"], "goals_against": matches["away_goals"],
        "shots_for": matches["home_shots"], "shots_against": matches["away_shots"],
        "sot_for": matches["home_sot"], "sot_against": matches["away_sot"],
        "corners_for": matches["home_corners"], "corners_against": matches["away_corners"],
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"], "team": matches["away"],
        "goals_for": matches["away_goals"], "goals_against": matches["home_goals"],
        "shots_for": matches["away_shots"], "shots_against": matches["home_shots"],
        "sot_for": matches["away_sot"], "sot_against": matches["home_sot"],
        "corners_for": matches["away_corners"], "corners_against": matches["home_corners"],
    })
    long = pd.concat([home, away], ignore_index=True)
    gd = long["goals_for"] - long["goals_against"]
    long["points"] = np.select([gd > 0, gd == 0], [3, 1], 0)
    return long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling means of the previous N matches (the current match is excluded via shift)."""
    out = long[["match_id", "team"]].copy()
    grouped = long.groupby("team", sort=False)
    for w in WINDOWS:
        for s in FORM_STATS:
            out[f"{s}_l{w}"] = grouped[s].transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=3).mean())
    return out


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Feature table indexed like `matches` (requires a `match_id` column)."""
    elo, _ = elo_ratings(matches)
    form = rolling_form(team_long(matches))
    cols = [c for c in form.columns if c not in ("match_id", "team")]

    feats = matches[["match_id", "home", "away"]].copy()
    feats = feats.merge(form.rename(columns={c: f"home_{c}" for c in cols}),
                        left_on=["match_id", "home"], right_on=["match_id", "team"], how="left").drop(columns="team")
    feats = feats.merge(form.rename(columns={c: f"away_{c}" for c in cols}),
                        left_on=["match_id", "away"], right_on=["match_id", "team"], how="left").drop(columns="team")
    feats.index = matches.index
    feats["elo_home"], feats["elo_away"] = elo["elo_home"], elo["elo_away"]
    feats["elo_diff"] = feats["elo_home"] - feats["elo_away"]
    for c in cols:
        feats[f"diff_{c}"] = feats[f"home_{c}"] - feats[f"away_{c}"]
    return feats.drop(columns=["match_id", "home", "away"])


def current_team_state(matches: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Latest Elo and form for every team, to build features for a future fixture."""
    _, ratings = elo_ratings(matches)
    long = team_long(matches)
    rows = {}
    for team, g in long.groupby("team"):
        rows[team] = {f"{s}_l{w}": g[s].tail(w).mean() for w in WINDOWS for s in FORM_STATS}
    return ratings, pd.DataFrame.from_dict(rows, orient="index")


def fixture_features(home: str, away: str, ratings: dict[str, float], form: pd.DataFrame,
                     columns: list[str]) -> pd.DataFrame:
    row: dict[str, float] = {}
    for side, team in (("home", home), ("away", away)):
        f = form.loc[team] if team in form.index else pd.Series(dtype=float)
        for c in form.columns:
            row[f"{side}_{c}"] = f.get(c, np.nan)
    row["elo_home"], row["elo_away"] = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
    row["elo_diff"] = row["elo_home"] - row["elo_away"]
    for c in form.columns:
        row[f"diff_{c}"] = row[f"home_{c}"] - row[f"away_{c}"]
    return pd.DataFrame([row])[[c for c in columns if c in row]]
