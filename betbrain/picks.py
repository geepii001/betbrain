"""Candidate scoring and slip building.

Tipster logic encoded here:
- A slip should hit its target odds with the FEWEST games, because every extra leg multiplies risk
  and adds another bookmaker margin. The optimiser tries every leg count and keeps the combination
  with the highest joint win probability.
- Legs are ranked by "safety per unit of odds": -ln(confidence)/ln(odds). Low = you get paid well
  for little risk.
- Two markets in one match can be stacked (e.g. Home-or-draw & Over 1.5). Their joint probability
  comes from the scoreline model, so the correlation between them is respected.
- A match is used in only one section of a slate.
"""
import math
from itertools import chain, combinations

import numpy as np
import pandas as pd

from .config import BANKER_FLOOR, EST_MARGIN, MIN_ODDS

LABELS = {
    "HOME_WIN": "{h} to win", "AWAY_WIN": "{a} to win",
    "DC_1X": "{h} or draw", "DC_X2": "{a} or draw",
    "OVER_1_5": "Over 1.5 goals", "OVER_2_5": "Over 2.5 goals", "BTTS_YES": "Both teams to score",
}
G = np.arange(10)
FACT = np.array([math.factorial(int(k)) for k in G], float)
I, J = np.meshgrid(G, G, indexing="ij")
EVENTS = {"HOME_WIN": I > J, "AWAY_WIN": J > I, "DC_1X": I >= J, "DC_X2": J >= I,
          "OVER_1_5": I + J >= 2, "OVER_2_5": I + J >= 3, "BTTS_YES": (I > 0) & (J > 0)}
COMBOS = [("DC_1X", "OVER_1_5"), ("DC_X2", "OVER_1_5"), ("HOME_WIN", "OVER_1_5"),
          ("AWAY_WIN", "OVER_1_5"), ("DC_1X", "OVER_2_5"), ("DC_X2", "OVER_2_5"),
          ("DC_1X", "BTTS_YES"), ("DC_X2", "BTTS_YES")]


def _grid(lh, la):
    ph = np.exp(-lh) * lh ** G / FACT
    pa = np.exp(-la) * la ** G / FACT
    g = np.outer(ph, pa)
    return g / g.sum()


def _odds_for(mk, p, o):
    real = o.get(mk)
    if mk in ("HOME_WIN", "AWAY_WIN", "OVER_1_5", "OVER_2_5", "BTTS_YES") and real and real == real:
        return float(real), "book"
    if mk == "DC_1X" and o.get("HOME_WIN") and o.get("DRAW"):
        return round(1 / (1 / o["HOME_WIN"] + 1 / o["DRAW"]), 2), "derived"
    if mk == "DC_X2" and o.get("AWAY_WIN") and o.get("DRAW"):
        return round(1 / (1 / o["AWAY_WIN"] + 1 / o["DRAW"]), 2), "derived"
    return round(EST_MARGIN / max(p, 0.05), 2), "est"


def make_candidates(preds, odds_by_match, engine=None):
    rows = []
    for r in preds.itertuples(index=False):
        probs = {"HOME_WIN": r.p_home, "AWAY_WIN": r.p_away, "DC_1X": r.p_home + r.p_draw,
                 "DC_X2": r.p_draw + r.p_away, "OVER_1_5": r.p_o15, "OVER_2_5": r.p_o25,
                 "BTTS_YES": r.p_btts}
        o = odds_by_match.get(r.match_id, {})
        base = {"match_id": r.match_id, "date": r.date, "league": r.league, "home": r.home,
                "away": r.away}
        for mk, p in probs.items():
            p = float(p)
            odds, src = _odds_for(mk, p, o)
            conf = engine.adjust(mk, p) if engine else p
            rows.append({**base, "market": mk, "label": LABELS[mk].format(h=r.home, a=r.away),
                         "p_raw": p, "conf": conf, "odds": odds, "odds_source": src,
                         "edge": conf * odds - 1 if src != "est" else np.nan})
        g = _grid(float(r.lam_h), float(r.lam_a))
        for a, b in COMBOS:
            ga, gb = g[EVENTS[a]].sum(), g[EVENTS[b]].sum()
            lift = g[EVENTS[a] & EVENTS[b]].sum() / max(ga * gb, 1e-9)
            pj = float(np.clip(probs[a] * probs[b] * lift, 0.0, min(probs[a], probs[b])))
            mk = f"{a}+{b}"
            conf = engine.adjust(mk, pj) if engine else pj
            lab = (LABELS[a].format(h=r.home, a=r.away) + " & " + LABELS[b].format(h=r.home, a=r.away))
            rows.append({**base, "market": mk, "label": lab, "p_raw": pj, "conf": conf,
                         "odds": round(EST_MARGIN / max(pj, 0.05), 2), "odds_source": "est",
                         "edge": np.nan})
    return pd.DataFrame(rows)


def _banker_ok(engine, mk, conf):
    return engine.banker_ok(mk, conf) if engine else conf >= BANKER_FLOOR


def _singles(pool, used, n, name):
    legs, seen = [], set(used)
    for r in pool.sort_values("conf", ascending=False).to_dict("records"):
        if r["match_id"] in seen or r["odds"] < MIN_ODDS:
            continue
        legs.append(r)
        seen.add(r["match_id"])
        if len(legs) == n:
            break
    return {"section": name, "kind": "singles", "legs": legs, "total_odds": None,
            "win_prob": float(np.mean([l["conf"] for l in legs])) if legs else None,
            "alts": [], "notes": None}


def _empty(name):
    return {"section": name, "kind": "acca", "legs": [], "total_odds": None, "win_prob": None,
            "alts": [], "notes": None}


def build_acca(pool, used, target, lo, hi, max_legs, name, min_legs=2):
    """Exact search: for each leg count, the combination of distinct matches whose odds land in
    [lo, hi] with the highest joint win probability. Keep the best; ties go to fewer legs."""
    pool = pool[(~pool.match_id.isin(used)) & (pool.odds >= MIN_ODDS) & (pool.odds <= hi)].copy()
    if len(pool) < min_legs:
        return _empty(name)
    pool["_s"] = -np.log(pool.conf.clip(0.01, 0.99)) / np.log(pool.odds)
    pool = pool.sort_values("_s").groupby("match_id", sort=False).head(3).sort_values("_s")
    recs = pool.to_dict("records")
    found = []
    for n in range(min_legs, max_legs + 1):
        K = len(recs)
        while K > n and math.comb(K, n) > 120000:
            K -= 1
        if K < n:
            continue
        sub = recs[:K]
        odds = np.array([r["odds"] for r in sub])
        conf = np.array([r["conf"] for r in sub])
        mid = pd.factorize(np.array([r["match_id"] for r in sub]))[0]
        idx = np.fromiter(chain.from_iterable(combinations(range(K), n)), dtype=np.int32).reshape(-1, n)
        s = np.sort(mid[idx], axis=1)
        ok = (np.diff(s, axis=1) != 0).all(axis=1)
        prod = odds[idx].prod(axis=1)
        ok &= (prod >= lo) & (prod <= hi)
        if not ok.any():
            continue
        joint = np.where(ok, conf[idx].prod(axis=1), -1.0)
        k = int(joint.argmax())
        found.append({"n": n, "joint": float(joint[k]), "odds": float(prod[k]),
                      "legs": [sub[i] for i in idx[k]]})
    if not found:
        return _empty(name)
    top = max(found, key=lambda e: (round(e["joint"], 4), -e["n"]))
    alts = [{"n": e["n"], "joint": e["joint"], "odds": e["odds"]} for e in found]
    stacked = sum("+" in l["market"] for l in top["legs"])
    note = (f"Chose {top['n']} leg(s): fewest games for the target gave the best joint probability."
            + (f" {stacked} leg(s) are same-game stacks priced from the scoreline model." if stacked else ""))
    return {"section": name, "kind": "acca", "legs": top["legs"], "total_odds": round(top["odds"], 2),
            "win_prob": top["joint"], "alts": alts, "notes": note}


def _used(sections):
    return {l["match_id"] for s in sections for l in s["legs"]}


def build_day_slate(pool, engine=None):
    """Bankers, 2-odds, 3-odds, Over 1.5, Straight wins, Multi: all on different matches."""
    out = []
    bankers_pool = pool[[_banker_ok(engine, m, c) for m, c in zip(pool.market, pool.conf)]]
    out.append(_singles(bankers_pool[~bankers_pool.market.str.contains(r"\+")], set(), 3, "BANKERS"))
    out.append(build_acca(pool, _used(out), 2.0, 1.9, 2.4, 3, "TWO_ODDS"))
    out.append(build_acca(pool, _used(out), 3.0, 2.85, 3.6, 4, "THREE_ODDS"))
    out.append(_singles(pool[pool.market == "OVER_1_5"], _used(out), 5, "OVER_1_5"))
    out.append(_singles(pool[pool.market.isin(["HOME_WIN", "AWAY_WIN"])], _used(out), 5,
                        "STRAIGHT_WINS"))
    out.append(build_acca(pool, _used(out), 6.0, 5.4, 7.5, 5, "MULTI_ACC", min_legs=3))
    return out


def top_board(pool, n=10):
    """Ranked overview: strongest single (non-stacked) pick per match, not subject to exclusivity."""
    b = pool[(pool.odds >= 1.15) & (~pool.market.str.contains(r"\+"))]
    return b.sort_values("conf", ascending=False).drop_duplicates("match_id").head(n)
