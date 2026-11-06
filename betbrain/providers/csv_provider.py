"""Load football-data.co.uk style CSVs (free historical results + bookmaker odds)."""
import hashlib

import numpy as np
import pandas as pd

DIV_NAMES = {"E0": "Premier League", "E1": "Championship", "SP1": "La Liga",
             "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
EMPTY_I = ["match_id", "team", "player", "importance"]


def _mid(date, home, away):
    return "C" + hashlib.md5(f"{date}|{home}|{away}".encode()).hexdigest()[:12]


def _clean_columns(df):
    """Drop a UTF-8 BOM (football-data.co.uk fixtures.csv has one) and stray spaces."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def load_csv_frame(raw, league=None):
    raw = _clean_columns(raw)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    for c in ("FTHG", "FTAG"):          # fixtures.csv has no result columns at all
        if c not in raw.columns:
            raw[c] = np.nan
    raw["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    raw = raw.dropna(subset=["date"])
    raw["match_id"] = [_mid(d, h, a) for d, h, a in zip(raw.date, raw.HomeTeam, raw.AwayTeam)]
    fin = raw["FTHG"].notna() & raw["FTAG"].notna()
    lg = league if league else raw["Div"].map(DIV_NAMES).fillna(raw["Div"]) if "Div" in raw.columns else "league"
    matches = pd.DataFrame({
        "match_id": raw.match_id, "date": raw.date, "league": lg,
        "home": raw.HomeTeam.str.strip(), "away": raw.AwayTeam.str.strip(),
        "home_goals": raw.FTHG.where(fin), "away_goals": raw.FTAG.where(fin),
        "status": np.where(fin, "finished", "scheduled"),
        "formation_home": None, "formation_away": None, "source": "football-data.co.uk",
    })
    colmap = {"B365H": "HOME_WIN", "B365D": "DRAW", "B365A": "AWAY_WIN",
              "B365>2.5": "OVER_2_5", "B365<2.5": "UNDER_2_5"}
    parts = []
    for col, market in colmap.items():
        if col in raw.columns:
            d = raw[["match_id", col]].dropna().rename(columns={col: "odds"})
            d["market"], d["source"] = market, "bet365"
            parts.append(d)
    odds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["match_id", "market", "odds", "source"])
    return matches, odds, pd.DataFrame(columns=EMPTY_I)


def load_csv(path, league=None):
    return load_csv_frame(pd.read_csv(path, encoding="latin-1"), league)
