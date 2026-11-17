from itertools import pairwise

import numpy as np
import pytest

from matchprob.markets import count_markets, goal_markets, implied_from_odds, top_scores
from matchprob.metrics import brier, calibration_table, log_loss, rps
from matchprob.models import GoalModel

M = GoalModel.matrix_from_rates(1.6, 1.0, rho=-0.08)


def test_result_probabilities_sum_to_one():
    g = goal_markets(M)
    assert g["home_win"] + g["draw"] + g["away_win"] == pytest.approx(1.0)


def test_btts_matches_brute_force():
    brute = sum(M[i, j] for i in range(1, M.shape[0]) for j in range(1, M.shape[1]))
    assert goal_markets(M)["btts_yes"] == pytest.approx(brute)


def test_over_lines_are_monotonic():
    g = goal_markets(M)
    overs = [g[f"over_{k}"] for k in ("05", "15", "25", "35", "45")]
    assert all(a > b for a, b in pairwise(overs))


def test_top_score_is_the_matrix_maximum():
    score, p = top_scores(M, 1)[0]
    i, j = np.unravel_index(M.argmax(), M.shape)
    assert score == f"{i}-{j}" and p == pytest.approx(M.max())


def test_count_total_is_convolution_of_teams():
    ph = np.array([0.5, 0.5, 0, 0])
    pa = np.array([0.5, 0.5, 0, 0])
    out = count_markets(ph, pa, team_lines=(0.5,), total_lines=(1.5,), prefix="x")
    assert out["x_home_over_05"] == pytest.approx(0.5)
    assert out["x_total_over_15"] == pytest.approx(0.25)  # only 1+1


def test_bookmaker_margin_is_removed():
    p = implied_from_odds(2.0, 3.4, 4.0)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_rps_known_values():
    y = np.array([0])
    assert rps(y, np.array([[1.0, 0, 0]])) == pytest.approx(0.0)
    # predicting a draw when home wins is better than predicting an away win
    assert rps(y, np.array([[0, 1.0, 0]])) < rps(y, np.array([[0, 0, 1.0]]))
    assert rps(y, np.array([[0, 0, 1.0]])) == pytest.approx(1.0)


def test_log_loss_and_brier_binary():
    y = np.array([1, 0])
    p = np.array([0.8, 0.3])
    assert log_loss(y, p) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert brier(y, p) == pytest.approx((0.04 + 0.09) / 2)


def test_calibration_of_a_perfect_forecaster():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = rng.uniform(0, 1, 50_000) < p
    t = calibration_table(y, p)
    assert np.allclose(t["predicted"], t["observed"], atol=0.02)
