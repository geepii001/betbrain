import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .config import DATA, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches(
  match_id TEXT PRIMARY KEY, date TEXT, league TEXT, home TEXT, away TEXT,
  home_goals INTEGER, away_goals INTEGER, status TEXT,
  formation_home TEXT, formation_away TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS odds(
  match_id TEXT, market TEXT, odds REAL, source TEXT, PRIMARY KEY(match_id, market));
CREATE TABLE IF NOT EXISTS injuries(
  match_id TEXT, team TEXT, player TEXT, importance REAL, PRIMARY KEY(match_id, team, player));
CREATE TABLE IF NOT EXISTS predictions(
  pred_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, match_id TEXT, market TEXT,
  model_prob REAL, confidence REAL, odds REAL, odds_source TEXT,
  section TEXT, slip_id TEXT, outcome INTEGER, backtest INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS slips(
  slip_id TEXT PRIMARY KEY, run_date TEXT, section TEXT, kind TEXT,
  total_odds REAL, win_prob REAL, outcome INTEGER, booking_code TEXT,
  backtest INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE IF NOT EXISTS crowd(
  match_id TEXT, source TEXT, home_pct REAL, draw_pct REAL, away_pct REAL, over25_pct REAL,
  n_posts INTEGER, summary TEXT, fetched_at TEXT, PRIMARY KEY(match_id, source));
CREATE TABLE IF NOT EXISTS engine_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_pred_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS ix_pred_section ON predictions(section);
"""


def connect(path=None):
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path or DB_PATH)
    con.executescript(SCHEMA)
    return con


def q(con, sql, params=()):
    return pd.read_sql_query(sql, con, params=params)


def _rows(df, cols):
    d = df.reindex(columns=cols).astype(object)
    d = d.where(pd.notna(d), None)
    return d.values.tolist()


def _upsert(con, table, df, cols):
    if df is None or len(df) == 0:
        return
    sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
    con.executemany(sql, _rows(df, cols))
    con.commit()


def upsert_matches(con, df):
    _upsert(con, "matches", df, ["match_id", "date", "league", "home", "away", "home_goals",
                                 "away_goals", "status", "formation_home", "formation_away", "source"])


def upsert_odds(con, df):
    _upsert(con, "odds", df, ["match_id", "market", "odds", "source"])


def upsert_injuries(con, df):
    _upsert(con, "injuries", df, ["match_id", "team", "player", "importance"])


def load_all(con):
    return (q(con, "SELECT * FROM matches"), q(con, "SELECT * FROM odds"),
            q(con, "SELECT * FROM injuries"))


def odds_lookup(odds_df):
    if odds_df is None or len(odds_df) == 0:
        return {}
    return odds_df.pivot_table(index="match_id", columns="market", values="odds",
                               aggfunc="first").to_dict("index")


def set_state(con, key, value):
    con.execute("INSERT OR REPLACE INTO engine_state VALUES(?,?,?)",
                (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))
    con.commit()


def get_state(con, key, default=None):
    r = con.execute("SELECT value FROM engine_state WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default
