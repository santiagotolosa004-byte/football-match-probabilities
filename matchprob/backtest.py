"""Walk-forward backtest of the statistical models.

The season is cut into weekly blocks. Before each block the models are refitted
using only matches played *before* the block starts (last `train_years` years),
then they predict every match in the block. This simulates using the model in
real time, one matchweek at a time, and is the only honest way to measure it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .markets import count_markets, goal_markets
from .models import CountModel, GoalModel

log = logging.getLogger(__name__)

SOT_TEAM_LINES, SOT_TOTAL_LINES = (3.5, 4.5, 5.5), (7.5, 8.5, 9.5)
CORNER_TEAM_LINES, CORNER_TOTAL_LINES = (4.5, 5.5), (8.5, 9.5, 10.5)


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """What actually happened, in the same shape as the predictions."""
    hg, ag = df["home_goals"], df["away_goals"]
    out = pd.DataFrame(index=df.index)
    out["result"] = np.select([hg > ag, hg == ag], [0, 1], 2)  # 0 home, 1 draw, 2 away
    for line in (1.5, 2.5, 3.5):
        out[f"over_{str(line).replace('.', '')}"] = ((hg + ag) > line).astype(int)
    out["btts"] = ((hg > 0) & (ag > 0)).astype(int)
    for line in SOT_TEAM_LINES:
        k = str(line).replace(".", "")
        out[f"sot_home_over_{k}"] = (df["home_sot"] > line).astype(float).where(df["home_sot"].notna())
        out[f"sot_away_over_{k}"] = (df["away_sot"] > line).astype(float).where(df["away_sot"].notna())
    tot_sot = df["home_sot"] + df["away_sot"]
    for line in SOT_TOTAL_LINES:
        out[f"sot_total_over_{str(line).replace('.', '')}"] = (tot_sot > line).astype(float).where(tot_sot.notna())
    tot_c = df["home_corners"] + df["away_corners"]
    for line in CORNER_TOTAL_LINES:
        out[f"corners_total_over_{str(line).replace('.', '')}"] = (tot_c > line).astype(float).where(tot_c.notna())
    return out


def fit_models(train: pd.DataFrame, ref_date: pd.Timestamp, xi: float = 0.0019, l2: float = 0.5):
    goals = GoalModel(xi, l2).fit(train, ref_date)
    sot = CountModel("home_sot", "away_sot", xi, l2).fit(train, ref_date)
    corners = CountModel("home_corners", "away_corners", xi, l2).fit(train, ref_date)
    return goals, sot, corners


def predict_match(home: str, away: str, goals: GoalModel, sot: CountModel, corners: CountModel) -> dict[str, float]:
    gm = goal_markets(goals.score_matrix(home, away))
    row = {f"dc_{k}": v for k, v in gm.items()}
    row.update(count_markets(*sot.distributions(home, away), SOT_TEAM_LINES, SOT_TOTAL_LINES, "sot"))
    row.update(count_markets(*corners.distributions(home, away), CORNER_TEAM_LINES, CORNER_TOTAL_LINES, "corners"))
    return row


def walk_forward(df: pd.DataFrame, first_test_season: int, train_years: float = 3.0,
                 refit_days: int = 7, xi: float = 0.0019, l2: float = 0.5) -> pd.DataFrame:
    test = df[df["season"] >= first_test_season]
    if test.empty:
        raise ValueError("no matches in the test period")
    start = test["date"].min().normalize()
    block = ((test["date"] - start).dt.days // refit_days).to_numpy()
    window = pd.Timedelta(days=int(365 * train_years))

    rows = []
    for b in np.unique(block):
        t = start + pd.Timedelta(days=int(b) * refit_days)
        train = df[(df["date"] < t) & (df["date"] >= t - window)]
        if len(train) < 100:
            continue
        assert train["date"].max() < t  # no information from the future
        goals, sot, corners = fit_models(train, t, xi, l2)
        for idx, m in test[block == b].iterrows():
            row = predict_match(m["home"], m["away"], goals, sot, corners)
            row["match_id"] = m["match_id"]
            row["fit_date"] = t
            rows.append((idx, row))
    preds = pd.DataFrame([r for _, r in rows], index=[i for i, _ in rows])
    log.info("Walk-forward: %d matches predicted with %d refits", len(preds), len(np.unique(block)))
    return preds
