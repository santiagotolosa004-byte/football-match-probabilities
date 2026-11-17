"""Charts and markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import LEAGUES
from .metrics import calibration_table
from .ml import TARGET_LABELS

if TYPE_CHECKING:
    from .ml import Evaluation
    from .pipeline import Config, Prediction, Prepared

# validated categorical palette (fixed order), recessive ink and grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODEL_LABELS = {"statistical": "Statistical (Dixon-Coles / NB)", "ml_logit": "ML - logistic regression",
                "ml_gbm": "ML - gradient boosting", "ensemble": "Ensemble (statistical + logistic)"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def plot_model_comparison(scores: pd.DataFrame, path: Path) -> None:
    """Log-loss improvement over the baseline, per target and model."""
    targets = list(TARGET_LABELS)
    models = list(MODEL_LABELS)
    imp = {m: [] for m in models}
    for t in targets:
        base = scores.loc[(t, "baseline"), "log_loss"]
        for m in models:
            imp[m].append((base - scores.loc[(t, m), "log_loss"]) / base * 100)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(targets))
    h = 0.18
    for k, m in enumerate(models):
        pos = y + (k - 1.5) * (h + 0.02)
        ax.barh(pos, imp[m], height=h, color=SERIES[k], label=MODEL_LABELS[m])
    best = [max(models, key=lambda m: imp[m][i]) for i in range(len(targets))]
    for i, m in enumerate(best):
        k = models.index(m)
        v = imp[m][i]
        ax.text(v + 0.15 if v >= 0 else 0.15, y[i] + (k - 1.5) * (h + 0.02), f"{v:+.1f}%",
                va="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [TARGET_LABELS[t] for t in targets], color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("Log-loss improvement vs. baseline (%), higher is better")
    _style(ax, "Out-of-sample accuracy by market",
           "Walk-forward backtest; baseline = historical frequency. Best model per market labelled.")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_calibration(prep: Prepared, ev: Evaluation, path: Path) -> None:
    panels = [("result", "ensemble__result__home", "Home win", lambda y: (y == 0).astype(int)),
              ("over_25", "ensemble__over_25", "Over 2.5 goals", None),
              ("btts", "ensemble__btts", "Both teams score", None),
              ("sot_home_over_45", "ensemble__sot_home_over_45", "Home 5+ shots on target", None)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.3), sharey=True)
    for ax, (target, col, title, transform) in zip(axes, panels):
        p = ev.predictions[col].dropna()
        y = prep.outcomes.loc[p.index, target].astype(int)
        y = transform(y) if transform else y
        t = calibration_table(y.to_numpy(), p.to_numpy(), bins=10)
        t = t[t["n"] >= 20]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.plot(t["predicted"], t["observed"], color=SERIES[0], lw=2, marker="o", ms=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlabel("Predicted probability")
        _style(ax, title)
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle("Calibration of the ensemble: points on the diagonal = probabilities mean what they say",
                 x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ratings(prep: Prepared, cfg: Config, path: Path) -> None:
    from .models import GoalModel
    m = prep.matches
    as_of = m["date"].max() + pd.Timedelta(days=1)
    g = GoalModel().fit(m[m["date"] >= as_of - pd.Timedelta(days=int(365 * cfg.train_years))], as_of)
    last = m[m["season"] == m["season"].max()]
    teams = sorted(set(last["home"]) | set(last["away"]))
    r = g.ratings().loc[teams]
    att = np.exp(r["attack"])           # x league-average goals scored
    dfn = np.exp(-r["defence"])         # >1 = concedes fewer than average

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.axvline(1, color=MUTED, lw=0.8)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.scatter(att, dfn, s=60, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.grid(color=GRID, lw=0.8)
    texts = [ax.text(att[t], dfn[t], t, fontsize=8.5, color=INK) for t in teams]
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=att.to_numpy(), y=dfn.to_numpy(), ax=ax, expand=(1.15, 1.4),
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    except ImportError:  # labels may overlap without the optional dependency
        pass
    ax.set_xlabel("Attack: goals scored vs. league average (x)")
    ax.set_ylabel("Defence: goals prevented vs. league average (x)")
    _style(ax, f"Team strength - {LEAGUES[cfg.league][1]}, end of {m['season'].max()}-{(m['season'].max() + 1) % 100:02d}",
           "Dixon-Coles ratings, recent matches weighted more. Top-right = strong at both ends.")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_match_card(p: Prediction, league: str, path: Path) -> None:
    rows = [
        (f"{p.home} win", p.probabilities["home_win"]),
        ("Draw", p.probabilities["draw"]),
        (f"{p.away} win", p.probabilities["away_win"]),
        ("Over 2.5 goals", p.probabilities["over_25"]),
        ("Both teams score", p.probabilities["btts"]),
        (f"{p.home} 5+ shots on target", p.probabilities["sot_home_over_45"]),
        ("9+ total shots on target", p.probabilities["sot_total_over_85"]),
        ("10+ total corners", p.probabilities["corners_total_over_95"]),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    y = np.arange(len(rows))
    ax.barh(y, [1] * len(rows), height=0.56, color=GRID)
    ax.barh(y, [v for _, v in rows], height=0.56, color=SERIES[0])
    for i, (_, v) in enumerate(rows):
        ax.text(1.02, i, f"{v:.0%}", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_yticks(y, [label for label, _ in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    s = p.statistical
    sub = (f"{LEAGUES[league][1]} · expected goals {s['dc_expected_home_goals']:.2f}-{s['dc_expected_away_goals']:.2f} · "
           f"most likely score {p.top_scores[0][0]} ({p.top_scores[0][1]:.0%})")
    ax.set_title(f"{p.home} vs {p.away}", loc="left", fontsize=14, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_backtest_report(cfg: Config, prep: Prepared, ev: Evaluation) -> None:
    out = Path(cfg.out_dir)
    plot_model_comparison(ev.scores, out / "model_comparison.png")
    plot_calibration(prep, ev, out / "calibration.png")
    plot_ratings(prep, cfg, out / "team_ratings.png")

    sc = ev.scores.reset_index()
    seasons = sorted(prep.matches.loc[ev.predictions.index, "season"].unique())
    n_matches = int(sc["n"].max())
    rows = []
    for t in TARGET_LABELS:
        s = sc[sc["target"] == t].set_index("model")
        best = s.drop(index=["baseline", "bookmaker"], errors="ignore")["log_loss"].idxmin()
        row = {"Market": TARGET_LABELS[t]}
        for m in ["baseline", "statistical", "ml_logit", "ml_gbm", "ensemble", "bookmaker"]:
            if m in s.index:
                v = f"{s.loc[m, 'log_loss']:.4f}"
                row[m] = f"**{v}**" if m == best else v
        rows.append(row)
    table = pd.DataFrame(rows).rename(columns={"baseline": "Baseline", "statistical": "Statistical",
                                               "ml_logit": "ML logistic", "ml_gbm": "ML boosting",
                                               "ensemble": "Ensemble", "bookmaker": "Bookmaker"})

    r = sc[sc["target"] == "result"].set_index("model")
    lines = [
        f"# Backtest report - {LEAGUES[cfg.league][1]}",
        "",
        f"- Data: {len(prep.matches):,} matches, seasons {cfg.first_season}-{cfg.last_season + 1} (football-data.co.uk)",
        (f"- Scored out-of-sample on **{n_matches:,} matches**, seasons "
         f"{seasons[0]}-{(seasons[0] + 1) % 100:02d} to {seasons[-1]}-{(seasons[-1] + 1) % 100:02d}"),
        ("- Statistical models refitted every week on the previous 3 years; "
         "ML models retrained every season on previous seasons only"),
        "",
        "## Log loss by market (lower is better; best model in bold)",
        "",
        _md_table(table),
        "",
        "## Match result (1X2)",
        "",
        _md_table(pd.DataFrame({
            "Model": [MODEL_LABELS.get(m, m.title()) for m in r.index],
            "RPS": [f"{v:.4f}" for v in r["rps"]],
            "Brier": [f"{v:.4f}" for v in r["brier"]],
            "Accuracy": [f"{v:.1%}" for v in r["accuracy"]],
        })),
        "",
        "![model comparison](model_comparison.png)",
        "",
        "![calibration](calibration.png)",
        "",
        "![team ratings](team_ratings.png)",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
