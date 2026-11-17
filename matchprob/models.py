"""Statistical team-strength models.

Both models share the same log-linear structure, fitted by weighted maximum
likelihood:

    log E[home stat] = mu + home_adv + attack[home] + defence[away]
    log E[away stat] = mu            + attack[away] + defence[home]

* ``attack`` = how much of the stat a team produces (goals, shots on target...)
* ``defence`` = how much it concedes (higher = worse defence)
* Recent matches weigh more: weight = exp(-xi * days_ago) (xi=0.0019 -> half-life ~1 year)
* A small L2 penalty keeps ratings of teams with few matches (promoted sides)
  close to league average and makes the model identifiable.

GoalModel  : Poisson goals + Dixon-Coles correction for low scores (0-0, 1-0, 0-1, 1-1),
             which plain Poisson gets wrong because goals are not fully independent.
CountModel : Negative binomial for over-dispersed counts (shots on target, corners):
             variance = mean + alpha * mean^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


class TeamLogLinearModel:
    def __init__(self, home_col: str, away_col: str, xi: float = 0.0019, l2: float = 0.5):
        self.home_col, self.away_col = home_col, away_col
        self.xi, self.l2 = xi, l2
        self.teams: list[str] = []
        self.mu = self.home_adv = 0.0
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}

    # -- fitting -------------------------------------------------------------
    def _weights(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        days = (ref_date - dates).dt.days.to_numpy(float)
        return np.exp(-self.xi * np.clip(days, 0, None))

    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> TeamLogLinearModel:
        df = matches.dropna(subset=[self.home_col, self.away_col])
        if df.empty:
            raise ValueError("no matches to fit")
        ref_date = ref_date or df["date"].max() + pd.Timedelta(days=1)
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h = df["home"].map(idx).to_numpy()
        a = df["away"].map(idx).to_numpy()
        yh = df[self.home_col].to_numpy(float)
        ya = df[self.away_col].to_numpy(float)
        w = self._weights(df["date"], ref_date)
        self._w, self._yh, self._ya = w, yh, ya

        def nll_and_grad(theta):
            mu, home = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            lh = np.exp(mu + home + att[h] + dfn[a])
            la = np.exp(mu + att[a] + dfn[h])
            nll = np.sum(w * (lh - yh * np.log(lh) + la - ya * np.log(la)))
            nll += 0.5 * self.l2 * (att @ att + dfn @ dfn)
            rh, ra = w * (lh - yh), w * (la - ya)          # d nll / d log(lambda)
            g = np.empty_like(theta)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + n] = np.bincount(h, rh, n) + np.bincount(a, ra, n) + self.l2 * att
            g[2 + n:] = np.bincount(a, rh, n) + np.bincount(h, ra, n) + self.l2 * dfn
            return nll, g

        mean = max((np.average(yh, weights=w) + np.average(ya, weights=w)) / 2, 1e-6)
        theta0 = np.zeros(2 + 2 * n)
        theta0[0] = np.log(mean)
        res = optimize.minimize(nll_and_grad, theta0, jac=True, method="L-BFGS-B")
        theta = res.x
        self.mu, self.home_adv = float(theta[0]), float(theta[1])
        self.attack = dict(zip(self.teams, theta[2:2 + n]))
        self.defence = dict(zip(self.teams, theta[2 + n:]))
        lh = np.exp(self.mu + self.home_adv + theta[2:2 + n][h] + theta[2 + n:][a])
        la = np.exp(self.mu + theta[2:2 + n][a] + theta[2 + n:][h])
        self._post_fit(lh, la)
        return self

    def _post_fit(self, lh: np.ndarray, la: np.ndarray) -> None:
        """Hook for model-specific extra parameters."""

    # -- prediction ----------------------------------------------------------
    def expected(self, home: str, away: str) -> tuple[float, float]:
        """Expected stat for each side. Unknown teams get league-average ratings."""
        ah, dh = self.attack.get(home, 0.0), self.defence.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defence.get(away, 0.0)
        lh = np.exp(self.mu + self.home_adv + ah + da)
        la = np.exp(self.mu + aa + dh)
        return float(lh), float(la)

    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({"attack": self.attack, "defence": self.defence}).sort_values("attack", ascending=False)


class GoalModel(TeamLogLinearModel):
    """Dixon & Coles (1997): Poisson goals with a low-score dependence parameter rho."""

    def __init__(self, xi: float = 0.0019, l2: float = 0.5):
        super().__init__("home_goals", "away_goals", xi, l2)
        self.rho = 0.0

    @staticmethod
    def tau(x, y, lh, la, rho):
        x, y = np.asarray(x), np.asarray(y)
        t = np.ones(np.broadcast(x, y, lh, la).shape)
        t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
        t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
        t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
        t = np.where((x == 1) & (y == 1), 1 - rho, t)
        return t

    def _post_fit(self, lh, la):
        x, y, w = self._yh, self._ya, self._w
        low = (x <= 1) & (y <= 1)
        # rho must keep every tau positive
        hi = min(1.0, 1 / max(lh[low].max() * la[low].max(), 1e-9)) * 0.99
        lo = max(-1 / max(lh[low].max(), la[low].max()), -1.0) * 0.99

        def neg_ll(rho):
            return -np.sum(w[low] * np.log(self.tau(x[low], y[low], lh[low], la[low], rho)))

        self.rho = float(optimize.minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded").x)

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals."""
        lh, la = self.expected(home, away)
        return self.matrix_from_rates(lh, la, self.rho, max_goals)

    @classmethod
    def matrix_from_rates(cls, lh: float, la: float, rho: float, max_goals: int = 10) -> np.ndarray:
        g = np.arange(max_goals + 1)
        m = np.outer(stats.poisson.pmf(g, lh), stats.poisson.pmf(g, la))
        i, j = np.meshgrid(g, g, indexing="ij")
        m *= cls.tau(i, j, lh, la, rho)
        return m / m.sum()


class CountModel(TeamLogLinearModel):
    """Negative-binomial team model for shots on target / corners."""

    def __init__(self, home_col: str, away_col: str, xi: float = 0.0019, l2: float = 0.5):
        super().__init__(home_col, away_col, xi, l2)
        self.alpha = 0.0

    def _post_fit(self, lh, la):
        # method of moments for NB2: Var = mu + alpha * mu^2
        mu = np.concatenate([lh, la])
        y = np.concatenate([self._yh, self._ya])
        w = np.concatenate([self._w, self._w])
        self.alpha = float(max(0.0, np.sum(w * ((y - mu) ** 2 - y)) / np.sum(w * mu ** 2)))

    def pmf(self, mean: float, max_count: int = 40) -> np.ndarray:
        k = np.arange(max_count + 1)
        if self.alpha < 1e-6:
            p = stats.poisson.pmf(k, mean)
        else:
            r = 1 / self.alpha
            p = stats.nbinom.pmf(k, r, r / (r + mean))
        return p / p.sum()

    def distributions(self, home: str, away: str, max_count: int = 40) -> tuple[np.ndarray, np.ndarray]:
        lh, la = self.expected(home, away)
        return self.pmf(lh, max_count), self.pmf(la, max_count)
