from datetime import date, timedelta

import pandas as pd

from . import models as M
from .db import load_all, odds_lookup
from .engine import Engine
from .features import build_features
from .picks import build_acca, build_day_slate, make_candidates, top_board
from .store import clear_unsettled, save_pool, save_slate, settle


def _iso(d):
    return d.isoformat()


def _finished_results(feats):
    f = feats[feats.status == "finished"]
    return {r.match_id: (int(r.hg), int(r.ag)) for r in f.itertuples()}


def backtest(con, today, days=150, step=30):
    """Walk-forward: train only on the past, predict the next block, store outcomes.
    Gives the intelligence engine real history to learn from on day one."""
    matches, odds, inj = load_all(con)
    feats = build_features(matches, odds, inj)
    ol, res = odds_lookup(odds), _finished_results(feats)
    con.execute("DELETE FROM predictions WHERE backtest=1")
    con.execute("DELETE FROM slips WHERE backtest=1")
    con.commit()
    cs, n_days = today - timedelta(days=days), 0
    while cs < today:
        ce = min(cs + timedelta(days=step), today)
        mdl = M.train(feats, before=_iso(cs))
        test = feats[(feats.status == "finished") & (feats.date >= _iso(cs)) & (feats.date < _iso(ce))]
        if len(test):
            cand = make_candidates(M.predict(mdl, test), ol, None)
            for d, pool in cand.groupby("date"):
                save_pool(con, d, pool, backtest=1, results=res)
                save_slate(con, d, build_day_slate(pool, None), backtest=1, results=res)
                n_days += 1
        cs = ce
    return n_days


def daily(con, today):
    settle(con)
    eng = Engine(con)
    matches, odds, inj = load_all(con)
    feats = build_features(matches, odds, inj)
    mdl = M.train(feats)
    M.save(mdl)
    ol = odds_lookup(odds)
    t0, t7 = _iso(today), _iso(today + timedelta(days=7))
    up = feats[(feats.status == "scheduled") & (feats.date >= t0) & (feats.date <= t7)]
    if not len(up):
        return None
    preds = M.predict(mdl, up)
    cand = make_candidates(preds, ol, eng)
    pool = cand[cand.date == t0]
    clear_unsettled(con, t0)
    slate, board = [], pd.DataFrame()
    if len(pool):
        save_pool(con, t0, pool)
        slate = build_day_slate(pool, eng)
        board = top_board(pool)
    wk = [_iso(today + timedelta(days=i)) for i in range(1, 8)
          if (today + timedelta(days=i)).weekday() in (5, 6)]
    # Defensive: today's slate and the long slips normally use different dates, but
    # never let a match appear in two sections of the same report.
    slate_used = {l["match_id"] for s in slate for l in s["legs"]}
    weekend = build_acca(cand[cand.date.isin(wk)], slate_used, 10.0, 8.5, 13.0, 8,
                         "WEEKEND_10", min_legs=4)
    wk_used = slate_used | {l["match_id"] for l in weekend["legs"]}
    week_pool = cand[(cand.date > t0) & (cand.date <= t7)]
    weekly = build_acca(week_pool, wk_used, 10.0, 8.5, 13.0, 8, "WEEKLY_10", min_legs=4)
    long_s = [weekend, weekly]
    for sl in slate + long_s:
        if sl["kind"] == "acca" and sl["win_prob"] is not None:
            sl["win_prob_raw"] = sl["win_prob"]
            sl["win_prob"] = sl["win_prob"] * eng.slip_ratio(sl["section"])
    save_slate(con, t0, slate + long_s)
    from . import crowd as C
    return {"run_date": t0, "board": board, "slate": slate, "long": long_s, "engine": eng,
            "crowd": C.compare(con, preds[preds.date == t0]), "crowd_learn": C.learn(con),
            "con": con, "preds": preds}
