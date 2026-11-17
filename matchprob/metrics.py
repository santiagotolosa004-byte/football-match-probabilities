"""Scoring rules for probability forecasts. Lower is better for all losses.

* Log loss  : punishes confident wrong predictions heavily
* Brier     : mean squared error of the probabilities
* RPS       : ranked probability score, the standard for home/draw/away because
              it knows a draw is "closer" to a home win than an away win is
* ECE       : expected calibration error - when the model says 30%, does it happen ~30% of the time?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if p.ndim == 1:
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p, float)
    if p.ndim == 1:
        return float(np.mean((p - y) ** 2))
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(y: np.ndarray, p: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes (home, draw, away)."""
    y, p = np.asarray(y), np.asarray(p, float)
    onehot = np.eye(p.shape[1])[y]
    cum_diff = np.cumsum(p, axis=1) - np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum(cum_diff[:, :-1] ** 2, axis=1) / (p.shape[1] - 1)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p, float)
    pred = p.argmax(axis=1) if p.ndim == 2 else (p >= 0.5).astype(int)
    return float(np.mean(pred == y))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": which, "p": p, "y": y})
    t = df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float(np.sum(t["n"] * (t["predicted"] - t["observed"]).abs()) / t["n"].sum())


def score_multiclass(y, p) -> dict[str, float]:
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "rps": rps(y, p), "accuracy": accuracy(y, p)}


def score_binary(y, p) -> dict[str, float]:
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "accuracy": accuracy(y, p), "ece": ece(y, p)}
