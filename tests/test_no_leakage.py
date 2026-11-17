"""The most important property of a forecasting backtest: no information from
the future may influence a prediction. We check it by changing future results
and asserting that earlier predictions/features do not move."""

import numpy as np
import pandas as pd
import pytest

from matchprob.backtest import walk_forward
from matchprob.features import build_features, elo_ratings

from .conftest import simulate_league


def _scramble_after(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    future = out["date"] >= cutoff
    for col in ["home_goals", "away_goals", "home_sot", "away_sot", "home_corners", "away_corners"]:
        out.loc[future, col] = out.loc[future, col].to_numpy()[::-1] + 3
    return out


def test_features_do_not_use_future_matches(league):
    df, _ = league
    cutoff = df["date"].quantile(0.6)
    a = build_features(df)
    b = build_features(_scramble_after(df, cutoff))
    past = df["date"] <= cutoff  # matches on the cutoff day are pre-match features too
    pd.testing.assert_frame_equal(a[past], b[past])
    assert not a[~past].equals(b[~past])  # sanity: future features do change


def test_walk_forward_predictions_do_not_use_future_matches():
    df, _ = simulate_league(n_teams=10, seasons=4, seed=1)
    cutoff = df.loc[df["season"] == 2013, "date"].min() + pd.Timedelta(days=60)
    a = walk_forward(df, 2012)
    b = walk_forward(_scramble_after(df, cutoff), 2012)
    before = df.loc[a.index, "date"] < cutoff
    assert before.sum() > 30 and (~before).sum() > 30
    cols = [c for c in a.columns if c not in ("match_id", "fit_date")]
    np.testing.assert_allclose(a.loc[before, cols].to_numpy(), b.loc[before, cols].to_numpy())
    assert not np.allclose(a.loc[~before, cols].to_numpy(), b.loc[~before, cols].to_numpy())
    assert (a["fit_date"] <= df.loc[a.index, "date"]).all()


def test_elo_is_zero_sum(league):
    df, _ = league
    _, ratings = elo_ratings(df)
    assert np.mean(list(ratings.values())) == pytest.approx(1500.0)
