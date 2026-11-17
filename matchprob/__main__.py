"""Command line.

    python -m matchprob backtest --league premier-league
    python -m matchprob predict  --league premier-league --home Arsenal --away "Man City"
    python -m matchprob teams    --league la-liga
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .data import LEAGUES
from .pipeline import Config, format_prediction, predict_fixture, prepare, run_backtest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m matchprob",
                                description="Football match probabilities from public match statistics.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("backtest", "predict", "teams"):
        s = sub.add_parser(name)
        s.add_argument("--league", default="premier-league", choices=sorted(LEAGUES))
        s.add_argument("--first-season", type=int, default=2010)
        s.add_argument("--last-season", type=int, default=2025,
                       help="start year of the last season (2025 = 2025-26)")
        s.add_argument("--source", default="github", choices=["github", "football-data"],
                       help="football-data = original site, includes current season and betting odds")
        s.add_argument("--data-dir", default="data")
        s.add_argument("--out", default="output")
        if name == "predict":
            s.add_argument("--home", required=True)
            s.add_argument("--away", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = Config(league=a.league, first_season=a.first_season, last_season=a.last_season,
                 source=a.source, cache_dir=a.data_dir, out_dir=a.out)

    if a.command == "teams":
        m = prepare(cfg).matches
        last = m[m["season"] == m["season"].max()]
        print("\n".join(sorted(set(last["home"]) | set(last["away"]))))
    elif a.command == "backtest":
        _, ev = run_backtest(cfg)
        print(ev.scores[["log_loss", "brier", "accuracy"]].round(4).to_string())
        print(f"\nReport: {Path(cfg.out_dir) / 'report.md'}")
    else:
        pred = predict_fixture(cfg, a.home, a.away)
        print(format_prediction(pred, cfg.league))
        from .report import plot_match_card
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cfg.out_dir) / f"{pred.home}_vs_{pred.away}.png".replace(" ", "_")
        plot_match_card(pred, cfg.league, path)
        print(f"\nMatch card: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
