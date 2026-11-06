"""football-data.org v4 adapter (fixtures + results; free tier needs a token, no odds).

Set FOOTBALL_DATA_TOKEN. NOTE: written against the public v4 docs; not exercised against the
live API in the build environment, so check field names on first run.
"""
import os

import pandas as pd
import requests

BASE = "https://api.football-data.org/v4"


def fetch(date_from, date_to, token=None):
    token = token or os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("Set FOOTBALL_DATA_TOKEN (free key from football-data.org)")
    r = requests.get(f"{BASE}/matches", headers={"X-Auth-Token": token},
                     params={"dateFrom": date_from, "dateTo": date_to}, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("matches", []):
        ft = (m.get("score") or {}).get("fullTime") or {}
        done = m.get("status") == "FINISHED"
        rows.append({
            "match_id": f"FD{m['id']}", "date": m["utcDate"][:10],
            "league": m["competition"]["name"], "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "home_goals": ft.get("home") if done else None,
            "away_goals": ft.get("away") if done else None,
            "status": "finished" if done else "scheduled",
            "formation_home": None, "formation_away": None, "source": "football-data.org",
        })
    empty_o = pd.DataFrame(columns=["match_id", "market", "odds", "source"])
    empty_i = pd.DataFrame(columns=["match_id", "team", "player", "importance"])
    return pd.DataFrame(rows), empty_o, empty_i
