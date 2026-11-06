import argparse
from datetime import date

import pandas as pd

from . import pipeline, report
from .db import connect, upsert_injuries, upsert_matches, upsert_odds
from .engine import Engine
from .providers import csv_provider, footballdata_org, synthetic
from .store import settle


def _d(s):
    return date.fromisoformat(s) if s else date.today()


def _ingest(con, triple):
    m, o, i = triple
    upsert_matches(con, m)
    upsert_odds(con, o)
    upsert_injuries(con, i)


def _no_upcoming(con):
    """Explain *why* daily has nothing to do, including a stale fixture feed."""
    last = con.execute("SELECT MAX(date) FROM matches WHERE status='scheduled'").fetchone()[0]
    if last:
        return (f"No fixtures in the next 7 days. The latest scheduled match in the database is "
                f"{last}; football-data.co.uk's fixtures.csv can lag, so re-download closer to kick-off.")
    return "No upcoming fixtures in the database for the next 7 days. Ingest fixtures first."


def cmd_demo(a):
    import os
    from .config import DB_PATH
    if DB_PATH.exists():
        DB_PATH.unlink()
    today = _d(a.today)
    con = connect()
    m, o, i, hidden = synthetic.generate(today, years=a.years)
    _ingest(con, (m, o, i))
    print(f"Synthetic world: {len(m)} matches ({(m.status == 'finished').sum()} finished)")
    n = pipeline.backtest(con, today, days=a.backtest_days)
    print(f"Backtest complete over {n} match days; engine has history to learn from")
    from . import crowd as C
    from .db import odds_lookup
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    stub = C.StubCrowd(odds_lookup(o))
    recent = m[(m.status == "finished") & (m.date >= (today - pd.Timedelta(days=150)).isoformat())]
    C.collect(con, stub, recent, ts, max_matches=10**6)
    C.collect(con, stub, m[(m.status == "scheduled") & (m.date == today.isoformat())], ts)
    res = pipeline.daily(con, today)
    print(report.render(res, res["engine"].insights()))
    report.write(res, res["engine"].insights())
    # advance one day: reveal today's real results, settle, let the engine learn
    day = hidden[hidden.date == today.isoformat()].copy()
    day["status"] = "finished"
    cols = m.columns
    upd = m[m.match_id.isin(day.match_id)].drop(columns=["home_goals", "away_goals", "status"]).merge(
        day[["match_id", "home_goals", "away_goals", "status"]], on="match_id")
    upsert_matches(con, upd[cols])
    k = settle(con)
    eng = Engine(con)
    print(f"\n--- Next day: {len(day)} results revealed, {k} predictions settled ---")
    sl = pd.read_sql_query("SELECT section, COUNT(*) legs, ROUND(AVG(outcome),2) hit_rate FROM predictions "
                           "WHERE run_date=? AND section!='POOL' AND outcome IS NOT NULL GROUP BY section",
                           con, params=(today.isoformat(),))
    print(sl.to_string(index=False))
    print(f"Engine now holds {eng.n} settled pool predictions")


def cmd_fetch_api(a):
    from .providers import apifootball
    con = connect()
    names = a.leagues.split(",") if a.leagues else None
    _ingest(con, apifootball.fetch(a.date_from, a.date_to, names))
    print("fetched from API-Football")


def cmd_crowd(a):
    from datetime import datetime, timezone
    from . import crowd as C
    from .db import load_all, odds_lookup
    con = connect()
    t0 = _d(a.today).isoformat()
    m, o, _ = load_all(con)
    up = m[(m.status == "scheduled") & (m.date == t0)]
    prov = C.GrokCrowd() if a.source == "grok" else C.StubCrowd(odds_lookup(o))
    n = C.collect(con, prov, up, datetime.now(timezone.utc).isoformat(), a.max)
    print(f"stored {n} crowd takes from {prov.source}")


def cmd_real_data(a):
    """Download free football-data.co.uk history + upcoming fixtures, then backtest, edge-test, predict."""
    from .providers import footballdata_uk as FU
    from .edge import print_edge
    from .config import DB_PATH
    if a.reset and DB_PATH.exists():
        DB_PATH.unlink()
    con = connect()
    synth = con.execute("SELECT COUNT(*) FROM matches WHERE source='synthetic'").fetchone()[0]
    if synth:
        print("This database holds synthetic demo data; mixing it with real matches would corrupt the "
              "results. Re-run with --reset to start a clean database.")
        return
    seasons = a.seasons.split(",") if a.seasons else None
    m, o, i, log = FU.download(seasons=seasons)
    print("\n".join(log))
    if m is None:
        print("Nothing downloaded. Check your internet connection, or download the CSVs by hand and use "
              "`ingest-csv`.")
        return
    _ingest(con, (m, o, i))
    fin, up = (m.status == "finished").sum(), (m.status == "scheduled").sum()
    print(f"Loaded {len(m)} matches: {fin} finished, {up} upcoming")
    today = _d(a.today)
    print(f"Walk-forward backtest over the last {a.days} days...")
    print(pipeline.backtest(con, today, days=a.days), "match days backtested")
    print_edge(con)
    if up:
        res = pipeline.daily(con, today)
        if res is not None:
            print(report.render(res, res["engine"].insights()))
            print("saved:", report.write(res, res["engine"].insights()))
        else:
            print(_no_upcoming(con))


def cmd_team_check(a):
    """Teams with very few games usually mean the same club appears under two names
    (e.g. 'Man United' vs 'Manchester United') after mixing data sources."""
    con = connect()
    d = pd.read_sql_query("SELECT team, COUNT(*) n FROM (SELECT home team FROM matches UNION ALL "
                          "SELECT away FROM matches) GROUP BY team ORDER BY n", con)
    print(d.head(a.top).to_string(index=False))


def cmd_ingest_csv(a):
    con = connect()
    _ingest(con, csv_provider.load_csv(a.path, a.league))
    print("ingested", a.path)


def cmd_fetch(a):
    con = connect()
    _ingest(con, footballdata_org.fetch(a.date_from, a.date_to))
    print("fetched fixtures/results")


def cmd_backtest(a):
    con = connect()
    print(pipeline.backtest(con, _d(a.today), days=a.days), "match days backtested")


def cmd_daily(a):
    con = connect()
    res = pipeline.daily(con, _d(a.today))
    if res is None:
        print(_no_upcoming(con))
        return
    ins = res["engine"].insights()
    print(report.render(res, ins))
    print("saved:", report.write(res, ins))


def cmd_edge(a):
    from .edge import print_edge
    print_edge(connect(), primary=a.threshold)


def cmd_settle(a):
    con = connect()
    print(settle(con), "predictions settled")
    print(Engine(con).insights())


def main():
    p = argparse.ArgumentParser(prog="betbrain")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("demo", help="end-to-end run on synthetic data")
    s.add_argument("--today", default="2026-11-03")
    s.add_argument("--years", type=int, default=3)
    s.add_argument("--backtest-days", type=int, default=150)
    s.set_defaults(f=cmd_demo)
    s = sub.add_parser("ingest-csv", help="load football-data.co.uk CSV")
    s.add_argument("path")
    s.add_argument("--league")
    s.set_defaults(f=cmd_ingest_csv)
    s = sub.add_parser("fetch", help="pull fixtures/results from football-data.org")
    s.add_argument("date_from")
    s.add_argument("date_to")
    s.set_defaults(f=cmd_fetch)
    s = sub.add_parser("backtest", help="walk-forward history for the engine")
    s.add_argument("--today")
    s.add_argument("--days", type=int, default=150)
    s.set_defaults(f=cmd_backtest)
    s = sub.add_parser("daily", help="settle, learn, retrain, predict, build slips")
    s.add_argument("--today")
    s.set_defaults(f=cmd_daily)
    s = sub.add_parser("settle", help="settle finished matches and show engine insights")
    s.set_defaults(f=cmd_settle)
    s = sub.add_parser("fetch-api", help="fixtures, odds, injuries from API-Football (API_FOOTBALL_KEY)")
    s.add_argument("date_from")
    s.add_argument("date_to")
    s.add_argument("--leagues", help="comma list, default all six: " + "Premier League,Championship,La Liga,Serie A,Bundesliga,Ligue 1")
    s.set_defaults(f=cmd_fetch_api)
    s = sub.add_parser("crowd", help="collect crowd takes for today's fixtures (grok needs XAI_API_KEY)")
    s.add_argument("--source", choices=["grok", "stub"], default="grok")
    s.add_argument("--today")
    s.add_argument("--max", type=int, default=15)
    s.set_defaults(f=cmd_crowd)
    s = sub.add_parser("real-data", help="download free football-data.co.uk data, backtest, edge-test, predict")
    s.add_argument("--seasons", help="comma list like 2324,2425,2526,2627 (default: last 5 + current)")
    s.add_argument("--days", type=int, default=365)
    s.add_argument("--today")
    s.add_argument("--reset", action="store_true", help="delete the existing database first (use after a demo)")
    s.set_defaults(f=cmd_real_data)
    s = sub.add_parser("team-check", help="spot duplicate team names after mixing sources")
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(f=cmd_team_check)
    s = sub.add_parser("edge-test", help="does the model beat the bookmaker? (run after backtest)")
    s.add_argument("--threshold", type=float, default=0.05,
                   help="primary minimum-edge gate (default 0.05 = 5 percent)")
    s.set_defaults(f=cmd_edge)
    a = p.parse_args()
    a.f(a)
