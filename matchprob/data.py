"""Download, cache and normalise match statistics.

Source: football-data.co.uk, the most widely used free source of historical
match statistics (results, shots, shots on target, corners, cards, betting odds).

Two ways to get it:
* ``github``       : a public mirror of the football-data.co.uk files (datasets/football-datasets).
                     Complete seasons only, no betting odds.
* ``football-data``: the original CSVs. Includes the season in progress and the
                     closing betting odds, which are used as an extra benchmark.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# league slug -> (football-data.co.uk division code, display name)
LEAGUES = {
    "premier-league": ("E0", "Premier League"),
    "la-liga": ("SP1", "LaLiga"),
    "serie-a": ("I1", "Serie A"),
    "bundesliga": ("D1", "Bundesliga"),
    "ligue-1": ("F1", "Ligue 1"),
}

URLS = {
    "github": "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/{league}/season-{code}.csv",
    "football-data": "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv",
}

COLUMN_MAP = {
    "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "home_goals", "FTAG": "away_goals",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
}
STAT_COLUMNS = ["home_goals", "away_goals", "home_shots", "away_shots",
                "home_sot", "away_sot", "home_corners", "away_corners"]

# Betting odds (market average preferred, then Bet365); used only as a benchmark.
ODDS_CANDIDATES = {
    "odds_home": ["AvgH", "BbAvH", "B365H"],
    "odds_draw": ["AvgD", "BbAvD", "B365D"],
    "odds_away": ["AvgA", "BbAvA", "B365A"],
    "odds_over25": ["Avg>2.5", "BbAv>2.5", "B365>2.5"],
    "odds_under25": ["Avg<2.5", "BbAv<2.5", "B365<2.5"],
}


def season_code(start_year: int) -> str:
    """2024 -> '2425' (the 2024-25 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def normalize(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Map a football-data.co.uk style CSV to the project's column names."""
    df = raw.rename(columns=COLUMN_MAP).copy()
    missing = {"home", "away", "home_goals", "away_goals"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    date_str = df["Date"].astype(str)
    iso = date_str.str.match(r"^\d{4}-\d{2}-\d{2}")
    df["date"] = pd.NaT
    df.loc[iso, "date"] = pd.to_datetime(date_str[iso], format="%Y-%m-%d")
    df.loc[~iso, "date"] = pd.to_datetime(date_str[~iso], dayfirst=True, format="mixed")
    df["date"] = pd.to_datetime(df["date"])

    for col in STAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else float("nan")

    keep = ["date", "home", "away", *STAT_COLUMNS]
    for target, candidates in ODDS_CANDIDATES.items():
        src = next((c for c in candidates if c in df.columns), None)
        if src is not None:
            df[target] = pd.to_numeric(df[src], errors="coerce")
            keep.append(target)

    df = df[keep].dropna(subset=["home", "away", "home_goals", "away_goals"])
    df["home"] = df["home"].astype(str).str.strip()
    df["away"] = df["away"].astype(str).str.strip()
    df["season"] = season
    return df


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "matchprob/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def load_season(league: str, season: int, source: str = "github",
                cache_dir: str | Path = "data") -> pd.DataFrame | None:
    """Load one season, using a local CSV cache. Returns None if not available."""
    if league not in LEAGUES:
        raise ValueError(f"unknown league {league!r}; choose from {sorted(LEAGUES)}")
    division = LEAGUES[league][0]
    code = season_code(season)
    path = Path(cache_dir) / source / league / f"{code}.csv"
    if not path.exists():
        url = URLS[source].format(league=league, code=code, division=division)
        try:
            content = _download(url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Season %s not available from %s (%s)", code, source, exc)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    raw = pd.read_csv(path, encoding_errors="replace")
    return normalize(raw, season)


def load_league(league: str, first_season: int, last_season: int, source: str = "github",
                cache_dir: str | Path = "data") -> pd.DataFrame:
    frames = [load_season(league, s, source, cache_dir) for s in range(first_season, last_season + 1)]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError(f"no data could be loaded for {league}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "home"]).reset_index(drop=True)
    df["match_id"] = range(len(df))
    log.info("Loaded %d matches of %s (%d-%d)", len(df), LEAGUES[league][1], first_season, last_season)
    return df


def read_csv_text(text: str, season: int) -> pd.DataFrame:
    """Helper for tests / custom files."""
    return normalize(pd.read_csv(io.StringIO(text)), season)
