"""Orchestration: backtest a league, or predict a single fixture."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtest import fit_models, outcomes, predict_match, walk_forward
from .data import LEAGUES, load_league
from .features import build_features, current_team_state, fixture_features
from .markets import top_scores
from .ml import Evaluation, build_design, evaluate, fit_final

log = logging.getLogger(__name__)


@dataclass
class Config:
    league: str = "premier-league"
    first_season: int = 2010
    last_season: int = 2025
    first_wf_season: int = 2013        # first season predicted by the statistical walk-forward
    first_test_season: int = 2019      # first season used to score the ML models
    source: str = "github"
    cache_dir: str = "data"
    out_dir: str = "output"
    train_years: float = 3.0


@dataclass
class Prepared:
    matches: pd.DataFrame
    stat_preds: pd.DataFrame
    design: pd.DataFrame
    feature_cols: list[str]
    outcomes: pd.DataFrame


def prepare(cfg: Config) -> Prepared:
    matches = load_league(cfg.league, cfg.first_season, cfg.last_season, cfg.source, cfg.cache_dir)
    cache = Path(cfg.cache_dir) / "walk_forward" / (
        f"{cfg.league}_{cfg.source}_{cfg.first_wf_season}_{len(matches)}_{matches['date'].max():%Y%m%d}.pkl")
    if cache.exists():
        stat_preds = pd.read_pickle(cache)
    else:
        stat_preds = walk_forward(matches, cfg.first_wf_season, cfg.train_years)
        cache.parent.mkdir(parents=True, exist_ok=True)
        stat_preds.to_pickle(cache)
    design, cols = build_design(build_features(matches), stat_preds, matches["season"])
    return Prepared(matches, stat_preds, design, cols, outcomes(matches.loc[stat_preds.index]))


def run_backtest(cfg: Config) -> tuple[Prepared, Evaluation]:
    prep = prepare(cfg)
    seasons = sorted(s for s in prep.matches["season"].unique() if s >= cfg.first_test_season)
    odds_cols = [c for c in prep.matches.columns if c.startswith("odds_")]
    odds = prep.matches.loc[prep.stat_preds.index, odds_cols] if odds_cols else None
    ev = evaluate(prep.design, prep.outcomes, prep.feature_cols, seasons, odds)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev.scores.round(5).to_csv(out / "scores.csv")
    preds = prep.matches.loc[ev.predictions.index, ["date", "season", "home", "away", "home_goals", "away_goals"]]
    pd.concat([preds, ev.predictions.round(4)], axis=1).to_csv(out / "predictions.csv", index=False)

    from .report import write_backtest_report
    write_backtest_report(cfg, prep, ev)
    return prep, ev


def resolve_team(name: str, teams: list[str]) -> str:
    if name in teams:
        return name
    lower = {t.lower(): t for t in teams}
    if name.lower() in lower:
        return lower[name.lower()]
    contains = [t for t in teams if name.lower() in t.lower() or t.lower() in name.lower()]
    if len(contains) == 1:
        return contains[0]
    close = difflib.get_close_matches(name, teams, n=1, cutoff=0.5)
    if close:
        return close[0]
    raise ValueError(f"team {name!r} not found. Available: {', '.join(sorted(teams))}")


@dataclass
class Prediction:
    home: str
    away: str
    probabilities: dict[str, float]     # blended (statistical + ML) where available
    statistical: dict[str, float]       # every market from the statistical models
    ml: dict[str, float]
    score_matrix: object
    top_scores: list[tuple[str, float]]
    as_of: pd.Timestamp


def predict_fixture(cfg: Config, home: str, away: str, prep: Prepared | None = None) -> Prediction:
    prep = prep or prepare(cfg)
    m = prep.matches
    recent_teams = sorted(set(m.loc[m["season"] >= m["season"].max() - 1, "home"]) |
                          set(m.loc[m["season"] >= m["season"].max() - 1, "away"]))
    home, away = resolve_team(home, recent_teams), resolve_team(away, recent_teams)
    as_of = m["date"].max() + pd.Timedelta(days=1)

    train = m[m["date"] >= as_of - pd.Timedelta(days=int(365 * cfg.train_years))]
    goals, sot, corners = fit_models(train, as_of)
    stat = predict_match(home, away, goals, sot, corners)
    matrix = goals.score_matrix(home, away)

    ratings, form = current_team_state(m)
    x = fixture_features(home, away, ratings, form, [c for c in prep.feature_cols if c not in stat])
    for c in prep.feature_cols:
        if c in stat:
            x[c] = stat[c]
    x = x[prep.feature_cols]

    models = fit_final(prep.design, prep.outcomes, prep.feature_cols)
    ml = {}
    p_res = models["result"].predict_proba(x)[0]
    ml.update({"home_win": p_res[0], "draw": p_res[1], "away_win": p_res[2]})
    for target in ("over_25", "btts", "sot_home_over_45", "sot_total_over_85", "corners_total_over_95"):
        ml[target] = float(models[target].predict_proba(x)[0, 1])

    blend = {
        "home_win": (stat["dc_home_win"] + ml["home_win"]) / 2,
        "draw": (stat["dc_draw"] + ml["draw"]) / 2,
        "away_win": (stat["dc_away_win"] + ml["away_win"]) / 2,
        "over_25": (stat["dc_over_25"] + ml["over_25"]) / 2,
        "btts": (stat["dc_btts_yes"] + ml["btts"]) / 2,
        "sot_home_over_45": (stat["sot_home_over_45"] + ml["sot_home_over_45"]) / 2,
        "sot_total_over_85": (stat["sot_total_over_85"] + ml["sot_total_over_85"]) / 2,
        "corners_total_over_95": (stat["corners_total_over_95"] + ml["corners_total_over_95"]) / 2,
    }
    s = sum(blend[k] for k in ("home_win", "draw", "away_win"))
    for k in ("home_win", "draw", "away_win"):
        blend[k] /= s
    return Prediction(home, away, blend, stat, ml, matrix, top_scores(matrix, 5), as_of)


def format_prediction(p: Prediction, league: str) -> str:
    s, b = p.statistical, p.probabilities

    def row(label: str, value: str) -> str:
        return f"  {label:<30}{value}"

    def pct(v: float) -> str:
        return f"{v:6.1%}"

    goal_lines = "  ".join(f"o{k[0]}.{k[1]} {s[f'dc_over_{k}']:.0%}" for k in ("05", "15", "25", "35", "45"))
    lines = [
        f"\n{p.home} vs {p.away}  ({LEAGUES[league][1]}, data up to {p.as_of - pd.Timedelta(days=1):%Y-%m-%d})",
        "=" * 66,
        "Blended probability (statistical model + machine learning)",
        row(f"{p.home} win", pct(b["home_win"])),
        row("Draw", pct(b["draw"])),
        row(f"{p.away} win", pct(b["away_win"])),
        row("Over 2.5 goals", pct(b["over_25"])),
        row("Both teams score", pct(b["btts"])),
        row(f"{p.home} 5+ shots on target", pct(b["sot_home_over_45"])),
        row("9+ total shots on target", pct(b["sot_total_over_85"])),
        row("10+ total corners", pct(b["corners_total_over_95"])),
        "",
        "Statistical model details",
        row("Expected goals", f"{s['dc_expected_home_goals']:.2f} - {s['dc_expected_away_goals']:.2f}"),
        row("Expected shots on target", f"{s['sot_expected_home']:.1f} - {s['sot_expected_away']:.1f}"),
        row("Expected corners", f"{s['corners_expected_home']:.1f} - {s['corners_expected_away']:.1f}"),
        row("Total goals", goal_lines),
        row(f"{p.home} clean sheet", pct(s["dc_home_clean_sheet"])),
        row(f"{p.away} clean sheet", pct(s["dc_away_clean_sheet"])),
        row("Most likely scores", ", ".join(f"{sc} ({pr:.0%})" for sc, pr in p.top_scores)),
    ]
    return "\n".join(lines)
