"""Persist predictions/slips and settle them against results."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .db import q


def outcome_for(market, hg, ag):
    """Single market or a same-game stack like 'DC_1X+OVER_1_5' (all parts must win)."""
    t = hg + ag
    ev = {"HOME_WIN": hg > ag, "AWAY_WIN": ag > hg, "DC_1X": hg >= ag, "DC_X2": ag >= hg,
          "OVER_1_5": t >= 2, "OVER_2_5": t >= 3, "BTTS_YES": hg > 0 and ag > 0}
    return int(all(ev[m] for m in market.split("+")))


def _now():
    return datetime.now(timezone.utc).isoformat()


def clear_unsettled(con, run_date):
    con.execute("DELETE FROM predictions WHERE run_date=? AND backtest=0 AND outcome IS NULL", (run_date,))
    con.execute("DELETE FROM slips WHERE run_date=? AND backtest=0 AND outcome IS NULL", (run_date,))
    con.commit()


def save_pool(con, run_date, pool, backtest=0, results=None):
    rows = []
    for r in pool.to_dict("records"):
        out = None
        if results is not None and r["match_id"] in results:
            out = outcome_for(r["market"], *results[r["match_id"]])
        rows.append((run_date, r["match_id"], r["market"], r["p_raw"], r["conf"], r["odds"],
                     r["odds_source"], "POOL", None, out, backtest))
    con.executemany("INSERT INTO predictions(run_date,match_id,market,model_prob,confidence,odds,"
                    "odds_source,section,slip_id,outcome,backtest) VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


def save_slate(con, run_date, slate, backtest=0, results=None):
    for s in slate:
        if not s["legs"]:
            continue
        sid = f"{run_date}-{s['section']}" + ("-bt" if backtest else "")
        outs = []
        for l in s["legs"]:
            out = None
            if results is not None and l["match_id"] in results:
                out = outcome_for(l["market"], *results[l["match_id"]])
            outs.append(out)
            con.execute("INSERT INTO predictions(run_date,match_id,market,model_prob,confidence,odds,"
                        "odds_source,section,slip_id,outcome,backtest) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (run_date, l["match_id"], l["market"], l["p_raw"], l["conf"], l["odds"],
                         l["odds_source"], s["section"], sid, out, backtest))
        slip_out = None
        if s["kind"] == "acca" and all(o is not None for o in outs):
            slip_out = int(all(outs))
        con.execute("INSERT OR REPLACE INTO slips VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (sid, run_date, s["section"], s["kind"], s["total_odds"], s["win_prob"],
                     slip_out, None, backtest, _now()))
    con.commit()


def settle(con):
    d = q(con, "SELECT p.pred_id, p.market, m.home_goals hg, m.away_goals ag FROM predictions p "
               "JOIN matches m USING(match_id) WHERE p.outcome IS NULL AND m.status='finished' "
               "AND m.home_goals IS NOT NULL")
    if len(d):
        con.executemany("UPDATE predictions SET outcome=? WHERE pred_id=?",
                        [(outcome_for(r.market, r.hg, r.ag), int(r.pred_id)) for r in d.itertuples()])
    sl = q(con, "SELECT s.slip_id, MIN(p.outcome) lo, COUNT(*) n, COUNT(p.outcome) k FROM slips s "
                "JOIN predictions p USING(slip_id) WHERE s.outcome IS NULL AND s.kind='acca' "
                "GROUP BY s.slip_id")
    sl = sl[sl.n == sl.k]
    if len(sl):
        con.executemany("UPDATE slips SET outcome=? WHERE slip_id=?",
                        [(int(r.lo), r.slip_id) for r in sl.itertuples()])
    con.commit()
    return len(d)
