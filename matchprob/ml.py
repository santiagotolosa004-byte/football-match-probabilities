"""Machine-learning layer.

Inputs : pre-match form features (features.py) + the statistical model's own
         probabilities and expected goals / shots on target (stacking).
Models : logistic regression (regularised, scaled) and gradient boosting.
Training is expanding-window by season: to predict season S the models only
see seasons before S, mirroring how they would be used in practice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import score_binary, score_multiclass

log = logging.getLogger(__name__)

# target column (in outcomes) -> statistical-model probability column(s)
TARGETS: dict[str, list[str]] = {
    "result": ["dc_home_win", "dc_draw", "dc_away_win"],
    "over_25": ["dc_over_25"],
    "btts": ["dc_btts_yes"],
    "sot_home_over_45": ["sot_home_over_45"],
    "sot_total_over_85": ["sot_total_over_85"],
    "corners_total_over_95": ["corners_total_over_95"],
}
TARGET_LABELS = {
    "result": "Match result (1X2)",
    "over_25": "Over 2.5 goals",
    "btts": "Both teams to score",
    "sot_home_over_45": "Home team 5+ shots on target",
    "sot_total_over_85": "9+ total shots on target",
    "corners_total_over_95": "10+ total corners",
}
STACK_COLUMNS = ["dc_home_win", "dc_draw", "dc_away_win", "dc_over_25", "dc_btts_yes",
                 "dc_expected_home_goals", "dc_expected_away_goals",
                 "sot_expected_home", "sot_expected_away", "corners_expected_home", "corners_expected_away"]


def build_design(features: pd.DataFrame, stat_preds: pd.DataFrame, seasons: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Join form features + statistical predictions; return the table and the ML input columns."""
    idx = stat_preds.index
    stat_cols = [c for c in stat_preds.columns if c not in ("match_id", "fit_date")]
    design = pd.concat([features.loc[idx], stat_preds[stat_cols], seasons.loc[idx].rename("season")], axis=1)
    return design, list(features.columns) + STACK_COLUMNS


def make_model(kind: str):
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=250,
                                              min_samples_leaf=60, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def _proba(model, X, multiclass: bool) -> np.ndarray:
    p = model.predict_proba(X)
    return p if multiclass else p[:, 1]


@dataclass
class Evaluation:
    scores: pd.DataFrame        # target x model x metric
    predictions: pd.DataFrame   # out-of-sample probabilities for every model


def evaluate(design: pd.DataFrame, outcomes: pd.DataFrame, feature_cols: list[str],
             test_seasons: list[int], odds: pd.DataFrame | None = None) -> Evaluation:
    """Expanding-window evaluation. `design` holds features + statistical predictions + season."""
    score_rows, pred_frames = [], []
    for target, stat_cols in TARGETS.items():
        multiclass = target == "result"
        y_all = outcomes[target]
        ok = y_all.notna()
        preds = {}
        for season in test_seasons:
            tr = ok & (design["season"] < season)
            te = ok & (design["season"] == season)
            if te.sum() == 0 or tr.sum() < 500:
                continue
            ytr = y_all[tr].astype(int)
            idx = design.index[te]
            p_stat = design.loc[te, stat_cols].to_numpy()
            p_stat = p_stat / p_stat.sum(axis=1, keepdims=True) if multiclass else p_stat[:, 0]
            if multiclass:
                base = np.bincount(ytr, minlength=3) / len(ytr)
                p_base = np.tile(base, (te.sum(), 1))
            else:
                p_base = np.full(te.sum(), ytr.mean())
            out = {"baseline": p_base, "statistical": p_stat}
            for kind in ("logit", "gbm"):
                m = make_model(kind).fit(design.loc[tr, feature_cols], ytr)
                out[f"ml_{kind}"] = _proba(m, design.loc[te, feature_cols], multiclass)
            out["ensemble"] = (out["statistical"] + out["ml_logit"]) / 2
            if odds is not None and multiclass and {"odds_home", "odds_draw", "odds_away"} <= set(odds.columns):
                o = odds.loc[idx, ["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
                raw = 1 / o
                out["bookmaker"] = raw / raw.sum(axis=1, keepdims=True)
            if odds is not None and target == "over_25" and {"odds_over25", "odds_under25"} <= set(odds.columns):
                o = odds.loc[idx, ["odds_over25", "odds_under25"]].to_numpy(float)
                raw = 1 / o
                out["bookmaker"] = raw[:, 0] / raw.sum(axis=1)
            for name, p in out.items():
                preds.setdefault(name, []).append(pd.DataFrame(
                    p if multiclass else p.reshape(-1, 1), index=idx,
                    columns=[f"{name}__{target}__{k}" for k in ("home", "draw", "away")] if multiclass
                    else [f"{name}__{target}"]))

        y_test = None
        for name, parts in preds.items():
            frame = pd.concat(parts)
            pred_frames.append(frame)
            valid = frame.notna().all(axis=1)
            y_test = y_all.loc[frame.index[valid]].astype(int).to_numpy()
            p = frame[valid].to_numpy() if multiclass else frame[valid].iloc[:, 0].to_numpy()
            s = score_multiclass(y_test, p) if multiclass else score_binary(y_test, p)
            s.update({"target": target, "model": name, "n": int(valid.sum())})
            score_rows.append(s)
        log.info("Evaluated %s", target)

    scores = pd.DataFrame(score_rows).set_index(["target", "model"])
    return Evaluation(scores=scores, predictions=pd.concat(pred_frames, axis=1))


def fit_final(design: pd.DataFrame, outcomes: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Fit one logistic model per target on all available history (used by `predict`)."""
    models = {}
    for target in TARGETS:
        y = outcomes[target]
        ok = y.notna()
        models[target] = make_model("logit").fit(design.loc[ok, feature_cols], y[ok].astype(int))
    return models
