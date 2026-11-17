"""Turn probability distributions into the questions people actually ask:
who wins, how many goals, do both teams score, how many shots on target..."""

from __future__ import annotations

import numpy as np

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


def goal_markets(m: np.ndarray, lines=GOAL_LINES) -> dict[str, float]:
    """`m[i, j]` = P(home i goals, away j goals)."""
    n = m.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total = i + j
    out = {
        "home_win": float(m[i > j].sum()),
        "draw": float(np.trace(m)),
        "away_win": float(m[i < j].sum()),
        "btts_yes": float(m[(i > 0) & (j > 0)].sum()),
        "home_clean_sheet": float(m[:, 0].sum()),
        "away_clean_sheet": float(m[0, :].sum()),
        "expected_home_goals": float((i * m).sum()),
        "expected_away_goals": float((j * m).sum()),
    }
    out["btts_no"] = 1 - out["btts_yes"]
    out["home_or_draw"] = out["home_win"] + out["draw"]
    out["away_or_draw"] = out["away_win"] + out["draw"]
    for line in lines:
        key = str(line).replace(".", "")
        out[f"over_{key}"] = float(m[total > line].sum())
        out[f"under_{key}"] = 1 - out[f"over_{key}"]
        out[f"home_over_{key}"] = float(m[i > line].sum())
        out[f"away_over_{key}"] = float(m[j > line].sum())
    return out


def top_scores(m: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(f"{a}-{b}", float(m[a, b])) for a, b in zip(*np.unravel_index(flat, m.shape))]


def count_markets(p_home: np.ndarray, p_away: np.ndarray, team_lines, total_lines, prefix: str) -> dict[str, float]:
    """Over/under lines for each team and for the match total of a count stat."""
    total = np.convolve(p_home, p_away)
    k_team, k_total = np.arange(len(p_home)), np.arange(len(total))
    out = {
        f"{prefix}_expected_home": float(k_team @ p_home),
        f"{prefix}_expected_away": float(k_team @ p_away),
    }
    for line in team_lines:
        key = str(line).replace(".", "")
        out[f"{prefix}_home_over_{key}"] = float(p_home[k_team > line].sum())
        out[f"{prefix}_away_over_{key}"] = float(p_away[k_team > line].sum())
    for line in total_lines:
        key = str(line).replace(".", "")
        out[f"{prefix}_total_over_{key}"] = float(total[k_total > line].sum())
    return out


def fair_odds(p: float) -> float:
    """Decimal odds implied by a probability (no bookmaker margin)."""
    return float("inf") if p <= 0 else 1 / p


def implied_from_odds(*odds: float) -> np.ndarray:
    """Bookmaker odds -> probabilities, removing the margin (overround) proportionally."""
    raw = 1 / np.asarray(odds, dtype=float)
    return raw / raw.sum()
