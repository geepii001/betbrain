"""API-Football (api-sports.io) v3 adapter: the licensed route to WhoScored/Flashscore-type data.

Gives fixtures + results, bookmaker odds, injuries, lineups/formations and head-to-head.
Auth: set API_FOOTBALL_KEY (header x-apisports-key, host v3.football.api-sports.io).

STATUS: confirmed from public docs: base URL, auth header, endpoints /fixtures /odds /predictions
/standings /players. Response parsing below follows the v3 schema as I know it and has only been
tested against MOCK responses (tests/test_apifootball.py), not the live API. Free plans are limited
in requests per day and in which seasons they can read, so check your plan before a big backfill.
"""
import os
import time
from datetime import date

import pandas as pd
import requests

BASE = "https://v3.football.api-sports.io"
LEAGUES = {"Premier League": 39, "Championship": 40, "La Liga": 140, "Serie A": 135,
           "Bundesliga": 78, "Ligue 1": 61}
FINISHED = {"FT", "AET", "PEN"}
EMPTY_O = ["match_id", "market", "odds", "source"]
EMPTY_I = ["match_id", "team", "player", "importance"]


def season_for(d):
    d = pd.Timestamp(d)
    return d.year if d.month >= 7 else d.year - 1


class ApiFootball:
    def __init__(self, key=None, daily_budget=90, session=None):
        self.key = key or os.environ.get("API_FOOTBALL_KEY")
        if not self.key:
            raise RuntimeError("Set API_FOOTBALL_KEY (from dashboard.api-football.com)")
        self.budget, self.used = daily_budget, 0
        self.http = session or requests.Session()

    def _get(self, path, **params):
        if self.used >= self.budget:
            raise RuntimeError(f"Request budget of {self.budget} reached; raise daily_budget only "
                               "if your plan allows it")
        self.used += 1
        r = self.http.get(f"{BASE}{path}", headers={"x-apisports-key": self.key},
                          params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        errs = j.get("errors")
        if errs:
            raise RuntimeError(f"API-Football error on {path}: {errs}")
        time.sleep(0.2)
        return j

    def _paged(self, path, **params):
        page, out = 1, []
        while True:
            j = self._get(path, page=page, **params)
            out += j.get("response", [])
            if page >= (j.get("paging") or {}).get("total", 1):
                return out
            page += 1

    # ---- fixtures and results ------------------------------------------------------------
    def fixtures(self, league_id, season, date_from, date_to):
        rows = []
        for f in self._get("/fixtures", league=league_id, season=season,
                           **{"from": date_from, "to": date_to}).get("response", []):
            fx, tm, gl = f["fixture"], f["teams"], f.get("goals") or {}
            done = (fx.get("status") or {}).get("short") in FINISHED
            rows.append({
                "match_id": f"AF{fx['id']}", "date": fx["date"][:10], "league": f["league"]["name"],
                "home": tm["home"]["name"], "away": tm["away"]["name"],
                "home_goals": gl.get("home") if done else None,
                "away_goals": gl.get("away") if done else None,
                "status": "finished" if done else "scheduled",
                "formation_home": None, "formation_away": None, "source": "api-football"})
        return pd.DataFrame(rows)

    # ---- odds ----------------------------------------------------------------------------
    @staticmethod
    def parse_odds(resp, prefer="Bet365"):
        """resp: list of /odds response items. Match Winner, Goals Over/Under, Both Teams Score."""
        rows = []
        for item in resp:
            mid = f"AF{item['fixture']['id']}"
            books = item.get("bookmakers") or []
            if not books:
                continue
            book = next((b for b in books if prefer.lower() in b["name"].lower()), books[0])
            for bet in book.get("bets", []):
                vals = {v["value"]: float(v["odd"]) for v in bet.get("values", [])}
                name = bet.get("name", "")
                if name == "Match Winner":
                    for k, mk in (("Home", "HOME_WIN"), ("Draw", "DRAW"), ("Away", "AWAY_WIN")):
                        if k in vals:
                            rows.append((mid, mk, vals[k], book["name"]))
                elif name == "Goals Over/Under":
                    for k, mk in (("Over 2.5", "OVER_2_5"), ("Under 2.5", "UNDER_2_5"),
                                  ("Over 1.5", "OVER_1_5")):
                        if k in vals:
                            rows.append((mid, mk, vals[k], book["name"]))
                elif name == "Both Teams Score" and "Yes" in vals:
                    rows.append((mid, "BTTS_YES", vals["Yes"], book["name"]))
        return pd.DataFrame(rows, columns=EMPTY_O)

    def odds(self, league_id, season, day):
        return self.parse_odds(self._paged("/odds", league=league_id, season=season, date=day))

    # ---- injuries ------------------------------------------------------------------------
    @staticmethod
    def parse_injuries(resp):
        """Importance is a rough proxy: 'Missing Fixture' = 0.6, doubtful = 0.3. Replace with a
        player-rating-based value once /players data is wired in."""
        rows = []
        for it in resp:
            typ = (it.get("player") or {}).get("type", "")
            rows.append((f"AF{it['fixture']['id']}", it["team"]["name"], it["player"]["name"],
                         0.6 if "Missing" in typ else 0.3))
        return pd.DataFrame(rows, columns=EMPTY_I).drop_duplicates(["match_id", "team", "player"])

    def injuries(self, league_id, season, day):
        return self.parse_injuries(self._paged("/injuries", league=league_id, season=season, date=day))

    # ---- lineups / formations (only published roughly an hour before kick-off) -------------
    @staticmethod
    def parse_lineups(resp):
        out = {}
        for t in resp:
            out[t["team"]["name"]] = t.get("formation")
        return out

    def lineups(self, fixture_id):
        return self.parse_lineups(self._get("/fixtures/lineups", fixture=fixture_id).get("response", []))

    # ---- API's own prediction, usable as another opinion to compare against --------------------
    def api_prediction(self, fixture_id):
        r = self._get("/predictions", fixture=fixture_id).get("response", [])
        if not r:
            return None
        pc = (r[0].get("predictions") or {}).get("percent") or {}
        try:
            return {k: float(str(pc[k]).strip("%")) for k in ("home", "draw", "away")}
        except Exception:
            return None


def fetch(date_from, date_to, leagues=None, key=None, with_odds=True, with_injuries=True):
    """Returns (matches, odds, injuries) for the given leagues and date range."""
    api = ApiFootball(key)
    names = leagues or list(LEAGUES)
    ms, os_, is_ = [], [], []
    season = season_for(date_from)
    for n in names:
        lid = LEAGUES[n]
        ms.append(api.fixtures(lid, season, date_from, date_to))
    m = pd.concat(ms, ignore_index=True) if ms else pd.DataFrame()
    upcoming_days = sorted(m[m.status == "scheduled"].date.unique()) if len(m) else []
    for day in upcoming_days:
        for n in names:
            if with_odds:
                os_.append(api.odds(LEAGUES[n], season, day))
            if with_injuries:
                is_.append(api.injuries(LEAGUES[n], season, day))
    o = pd.concat(os_, ignore_index=True) if os_ else pd.DataFrame(columns=EMPTY_O)
    i = pd.concat(is_, ignore_index=True) if is_ else pd.DataFrame(columns=EMPTY_I)
    print(f"API-Football requests used: {api.used}")
    return m, o, i
