"""Simulated leagues with *known* team strengths, so tests run offline and can
check that the models recover the truth."""

import numpy as np
import pandas as pd
import pytest


def simulate_league(n_teams=16, seasons=6, first_season=2010, seed=0, home_adv=0.25,
                    sot_alpha=0.05, corner_alpha=0.15):
    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    attack = rng.normal(0, 0.3, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0, 0.3, n_teams)
    defence -= defence.mean()
    rows = []
    for s in range(seasons):
        start = pd.Timestamp(f"{first_season + s}-08-10")
        fixtures = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(fixtures)
        per_week = n_teams // 2
        for k, (h, a) in enumerate(fixtures):
            date = start + pd.Timedelta(days=7 * (k // per_week) + (k % 2))
            lh = np.exp(0.2 + home_adv + attack[h] + defence[a])
            la = np.exp(0.2 + attack[a] + defence[h])
            mh_sot, ma_sot = 3.2 * lh, 3.2 * la

            def nb(mean, alpha):
                r = 1 / alpha
                return rng.negative_binomial(r, r / (r + mean))

            hsot, asot = nb(mh_sot, sot_alpha), nb(ma_sot, sot_alpha)
            rows.append({
                "date": date, "home": teams[h], "away": teams[a],
                "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                "home_sot": hsot, "away_sot": asot,
                "home_shots": hsot + rng.poisson(7), "away_shots": asot + rng.poisson(7),
                "home_corners": nb(5.5 * np.exp(0.5 * attack[h]), corner_alpha),
                "away_corners": nb(4.5 * np.exp(0.5 * attack[a]), corner_alpha),
                "season": first_season + s,
            })
    df = pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True)
    df["match_id"] = range(len(df))
    truth = pd.DataFrame({"attack": attack, "defence": defence}, index=teams)
    return df, truth


@pytest.fixture(scope="session")
def league():
    return simulate_league()
