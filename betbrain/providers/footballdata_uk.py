"""football-data.co.uk downloader: free historical results + bookmaker odds, and upcoming fixtures.

URL pattern (public, as I know the site): https://www.football-data.co.uk/mmz4281/<SSYY>/<DIV>.csv for a
season, and https://www.football-data.co.uk/fixtures.csv for upcoming matches with odds. NOT reachable
from the build sandbox, so downloads are untested here; parsing is tested on files in the same format.
Check the site's terms of use for how you may use the data.
"""
import io
from datetime import date

import pandas as pd
import requests

from .csv_provider import load_csv_frame

BASE = "https://www.football-data.co.uk"
DIVS = {"E0": "Premier League", "E1": "Championship", "SP1": "La Liga",
        "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}


def season_codes(today=None, back=5):
    """['2122', ..., '2627'] ending at the season containing `today` (seasons start in July)."""
    t = pd.Timestamp(today or date.today())
    start = t.year if t.month >= 7 else t.year - 1
    return [f"{(y % 100):02d}{((y + 1) % 100):02d}" for y in range(start - back, start + 1)]


def _get(session, url):
    try:
        r = session.get(url, headers={"User-Agent": "betbrain/0.1 (private research)"}, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200 or len(r.content) < 200:
        return None
    df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
    # fixtures.csv starts with a UTF-8 BOM, so pandas names the first column "\ufeffDiv".
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def download(seasons=None, divs=None, include_fixtures=True, session=None, today=None):
    """Returns (matches, odds, injuries, log). Missing files are logged, not fatal."""
    s = session or requests.Session()
    divs = divs or list(DIVS)
    log, parts = [], []
    for sc in seasons or season_codes(today):
        for dv in divs:
            df = _get(s, f"{BASE}/mmz4281/{sc}/{dv}.csv")
            if df is None:
                log.append(f"missing {sc}/{dv}")
                continue
            parts.append(load_csv_frame(df, DIVS[dv]))
            log.append(f"ok {sc}/{dv}: {len(df)} rows")
    if include_fixtures:
        df = _get(s, f"{BASE}/fixtures.csv")
        if df is None:
            log.append("missing fixtures.csv")
        else:
            if "Div" in df.columns:
                df["Div"] = df["Div"].astype(str).str.strip()
                df = df[df["Div"].isin(divs)]
            if len(df):
                parts.append(load_csv_frame(df, None))
                log.append(f"ok fixtures.csv: {len(df)} upcoming")
    if not parts:
        return None, None, None, log
    return (pd.concat([p[0] for p in parts], ignore_index=True).drop_duplicates("match_id", keep="last"),
            pd.concat([p[1] for p in parts], ignore_index=True).drop_duplicates(["match_id", "market"], keep="last"),
            pd.concat([p[2] for p in parts], ignore_index=True), log)
