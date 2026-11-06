"""Parser tests against MOCK API-Football responses (live API not reachable from the build sandbox)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from betbrain.providers.apifootball import ApiFootball, season_for

class Resp:
    def __init__(s, j): s.j = j
    def raise_for_status(s): pass
    def json(s): return s.j
class Sess:
    def __init__(s, j): s.j, s.calls = j, []
    def get(s, url, headers=None, params=None, timeout=None):
        s.calls.append((url, headers, params)); return Resp(s.j)

fx = {"errors": [], "paging": {"current": 1, "total": 1}, "response": [
  {"fixture": {"id": 11, "date": "2026-11-07T15:00:00+00:00", "status": {"short": "NS"}},
   "league": {"name": "Premier League"}, "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
   "goals": {"home": None, "away": None}},
  {"fixture": {"id": 12, "date": "2026-11-01T15:00:00+00:00", "status": {"short": "FT"}},
   "league": {"name": "Premier League"}, "teams": {"home": {"name": "Spurs"}, "away": {"name": "Everton"}},
   "goals": {"home": 2, "away": 1}}]}
s = Sess(fx); api = ApiFootball("k", session=s)
m = api.fixtures(39, 2026, "2026-11-01", "2026-11-08")
assert list(m.status) == ["scheduled", "finished"] and m.home_goals.iloc[1] == 2 and m.home_goals.iloc[0] != m.home_goals.iloc[0]
assert s.calls[0][1] == {"x-apisports-key": "k"} and s.calls[0][2]["league"] == 39

odds = [{"fixture": {"id": 11}, "bookmakers": [
  {"name": "Bwin", "bets": []},
  {"name": "Bet365", "bets": [
    {"name": "Match Winner", "values": [{"value": "Home", "odd": "2.10"}, {"value": "Draw", "odd": "3.40"}, {"value": "Away", "odd": "3.60"}]},
    {"name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.85"}, {"value": "Under 2.5", "odd": "1.95"}, {"value": "Over 1.5", "odd": "1.25"}]},
    {"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.70"}, {"value": "No", "odd": "2.10"}]}]}]}]
o = ApiFootball.parse_odds(odds)
d = {r.market: r.odds for r in o.itertuples()}
assert d == {"HOME_WIN": 2.10, "DRAW": 3.40, "AWAY_WIN": 3.60, "OVER_2_5": 1.85, "UNDER_2_5": 1.95, "OVER_1_5": 1.25, "BTTS_YES": 1.70}, d
assert set(o.source) == {"Bet365"}

inj = [{"player": {"name": "A", "type": "Missing Fixture"}, "team": {"name": "Arsenal"}, "fixture": {"id": 11}},
       {"player": {"name": "B", "type": "Questionable"}, "team": {"name": "Chelsea"}, "fixture": {"id": 11}}]
i = ApiFootball.parse_injuries(inj)
assert list(i.importance) == [0.6, 0.3]
assert ApiFootball.parse_lineups([{"team": {"name": "Arsenal"}, "formation": "4-3-3"}]) == {"Arsenal": "4-3-3"}
assert season_for("2026-11-04") == 2026 and season_for("2026-03-04") == 2025
b = ApiFootball("k", daily_budget=1, session=Sess(fx)); b._get("/fixtures")
try: b._get("/fixtures"); raise SystemExit("budget guard failed")
except RuntimeError: pass
print("api-football mock tests passed")
