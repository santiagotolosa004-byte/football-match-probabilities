import numpy as np
import pandas as pd
import pytest
from scipy import stats

from matchprob.models import CountModel, GoalModel

from .conftest import simulate_league


def test_goal_model_recovers_true_strengths(league):
    df, truth = league
    m = GoalModel(xi=0.0, l2=0.1).fit(df)
    r = m.ratings().loc[truth.index]
    assert np.corrcoef(r["attack"], truth["attack"])[0, 1] > 0.9
    assert np.corrcoef(r["defence"], truth["defence"])[0, 1] > 0.9
    assert m.home_adv == pytest.approx(0.25, abs=0.07)


def test_score_matrix_is_a_distribution(league):
    df, _ = league
    m = GoalModel().fit(df)
    mat = m.score_matrix("Team A", "Team B")
    assert mat.sum() == pytest.approx(1.0)
    assert (mat >= 0).all()


def test_dixon_coles_with_zero_rho_is_independent_poisson():
    mat = GoalModel.matrix_from_rates(1.5, 1.1, rho=0.0, max_goals=15)
    expected = np.outer(stats.poisson.pmf(range(16), 1.5), stats.poisson.pmf(range(16), 1.1))
    assert np.allclose(mat, expected / expected.sum())


def test_negative_rho_inflates_draws_at_0_0_and_1_1():
    base = GoalModel.matrix_from_rates(1.3, 1.1, rho=0.0)
    dc = GoalModel.matrix_from_rates(1.3, 1.1, rho=-0.1)
    assert dc[0, 0] > base[0, 0] and dc[1, 1] > base[1, 1]
    assert dc[1, 0] < base[1, 0] and dc[0, 1] < base[0, 1]


def test_unknown_team_gets_league_average(league):
    df, _ = league
    m = GoalModel().fit(df)
    lh, la = m.expected("Promoted FC", "Another New FC")
    assert lh == pytest.approx(np.exp(m.mu + m.home_adv))
    assert la == pytest.approx(np.exp(m.mu))


def test_recent_matches_weigh_more():
    # a team that was bad long ago but is good recently should be rated by the recent form
    rows = []
    d0 = pd.Timestamp("2020-01-01")
    for k in range(200):
        good = k >= 100
        rows.append({"date": d0 + pd.Timedelta(days=7 * k), "home": "X", "away": f"Opp{k % 5}",
                     "home_goals": 3 if good else 0, "away_goals": 0 if good else 3})
        rows.append({"date": d0 + pd.Timedelta(days=7 * k + 1), "home": f"Opp{k % 5}", "away": f"Opp{(k + 1) % 5}",
                     "home_goals": 1, "away_goals": 1})
    df = pd.DataFrame(rows)
    decayed = GoalModel(xi=0.01).fit(df).attack["X"]
    flat = GoalModel(xi=0.0).fit(df).attack["X"]
    assert decayed > flat


def test_negative_binomial_dispersion_is_recovered():
    df, _ = simulate_league(n_teams=12, seasons=8, seed=3, corner_alpha=0.25)
    m = CountModel("home_corners", "away_corners", xi=0.0).fit(df)
    assert m.alpha == pytest.approx(0.25, abs=0.07)
    p = m.pmf(5.0)
    assert p.sum() == pytest.approx(1.0)
    assert (np.arange(len(p)) @ p) == pytest.approx(5.0, rel=0.01)
