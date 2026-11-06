"""Part A: development walk-forward (2021/22-2025/26) + frozen 2026/27 holdout.

No model behaviour is changed. A2 removes the mkt_* features at runtime only.
"""
import json
import time

import numpy as np
import pandas as pd
import requests

from betbrain import models as M
from betbrain.db import connect, load_all, odds_lookup
from betbrain.features import FEATURES, build_features
from betbrain.providers import footballdata_uk as FU
from betbrain.providers.csv_provider import _mid

DEV_CUTOFF = "2026-07-01"          # 2026/27 starts after this; frozen holdout
WF_START = "2022-08-01"            # 2021/22 is the initial training season
STEP = 30
PRIMARY = 0.05
MARKETS = ("HOME_WIN", "AWAY_WIN", "OVER_2_5")


def brier(probs, y):
    P = np.asarray(probs, dtype=float)
    Y = np.zeros((len(y), 3))
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def picks_from(preds, ol):
    """The three markets the production edge test prices with real bookmaker odds."""
    rows = []
    for r in preds.itertuples(index=False):
        o = ol.get(r.match_id, {})
        hg, ag = float(r.hg), float(r.ag)
        for mk, p in (("HOME_WIN", r.p_home), ("AWAY_WIN", r.p_away), ("OVER_2_5", r.p_o25)):
            od = o.get(mk)
            if od is None or not np.isfinite(od) or od <= 0:
                continue
            y = int(hg > ag) if mk == "HOME_WIN" else int(ag > hg) if mk == "AWAY_WIN" else int(hg + ag >= 3)
            rows.append({"match_id": r.match_id, "date": r.date, "league": r.league,
                         "market": mk, "model_prob": float(p), "odds": float(od), "outcome": y})
    d = pd.DataFrame(rows)
    if len(d):
        d["edge"] = d.model_prob * d.odds - 1.0
        d["ret"] = d.outcome * d.odds - 1.0
    return d


def run_wf(feats, ol, start, cutoff, use_mkt=True, step=STEP):
    old = M.FEATURES
    if not use_mkt:
        M.FEATURES = [f for f in old if not f.startswith("mkt_")]
    match_rows, pick_frames = [], []
    try:
        cs = pd.Timestamp(start)
        end = pd.Timestamp(cutoff)
        i = 0
        while cs < end:
            ce = min(cs + pd.Timedelta(days=step), end)
            a, b = cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")
            mdl = M.train(feats, before=a)
            test = feats[(feats.status == "finished") & (feats.date >= a) & (feats.date < b)]
            if len(test):
                pr = M.predict(mdl, test).merge(
                    test[["match_id", "hg", "ag", "y_result"]], on="match_id")
                for r in pr.itertuples(index=False):
                    match_rows.append({"match_id": r.match_id, "date": r.date, "league": r.league,
                                       "p_home": float(r.p_home), "p_draw": float(r.p_draw),
                                       "p_away": float(r.p_away), "p_o25": float(r.p_o25),
                                       "hg": float(r.hg), "ag": float(r.ag),
                                       "y_result": int(r.y_result)})
                pick_frames.append(picks_from(pr, ol))
            i += 1
            if i % 8 == 0:
                print(f"  block {i}: {a} -> {b}, {len(test)} matches")
            cs = ce
    finally:
        M.FEATURES = old
    return pd.DataFrame(match_rows), (pd.concat(pick_frames, ignore_index=True) if pick_frames else pd.DataFrame())


def edge_row(picks, t=PRIMARY, seed=0):
    s = picks[picks.edge > t]
    n = len(s)
    if n < 30:
        return {"threshold": t, "bets": n, "roi": None, "lo": None, "hi": None}
    r = s.ret.to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(r, len(r)).mean() for _ in range(2000)])
    return {"threshold": t, "bets": n, "roi": float(r.mean()),
            "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def edge_table(picks, thresholds=(0.0, 0.03, 0.05, 0.10)):
    return [edge_row(picks, t) for t in thresholds]


def attach_market_baseline(match_df, odds):
    mkt = odds[odds.market.isin(["HOME_WIN", "DRAW", "AWAY_WIN"])].pivot_table(
        index="match_id", columns="market", values="odds", aggfunc="first")
    mkt = mkt.dropna(subset=["HOME_WIN", "DRAW", "AWAY_WIN"])
    s = 1 / mkt.HOME_WIN + 1 / mkt.DRAW + 1 / mkt.AWAY_WIN
    mkt["mkt_h"] = (1 / mkt.HOME_WIN) / s
    mkt["mkt_d"] = (1 / mkt.DRAW) / s
    mkt["mkt_a"] = (1 / mkt.AWAY_WIN) / s
    mkt = mkt.reset_index()[["match_id", "mkt_h", "mkt_d", "mkt_a"]]
    return match_df.merge(mkt, on="match_id", how="inner").dropna(
        subset=["p_home", "p_draw", "p_away", "mkt_h", "mkt_d", "mkt_a"])


def fetch_extra():
    need = ["B365H", "B365D", "B365A", "MaxH", "MaxD", "MaxA", "PSH", "PSD", "PSA",
            "B365>2.5", "B365<2.5", "Max>2.5", "Max<2.5"]
    s, frames, missing = requests.Session(), [], []
    for sc in ("2122", "2223", "2324", "2425", "2526"):
        for dv in FU.DIVS:
            df = FU._get(s, f"{FU.BASE}/mmz4281/{sc}/{dv}.csv")
            if df is None:
                missing.append(f"{sc}/{dv}")
                continue
            df = df.copy()
            for c in need:
                if c not in df.columns:
                    df[c] = np.nan
            df = df.dropna(subset=["HomeTeam", "AwayTeam"])
            df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date"])
            df["match_id"] = [_mid(d, str(h).strip(), str(a).strip())
                              for d, h, a in zip(df.date, df.HomeTeam, df.AwayTeam)]
            frames.append(df[["match_id"] + need])
    return pd.concat(frames, ignore_index=True).drop_duplicates("match_id"), missing


def overrounds(extra):
    out = {}
    for label, (h, d, a) in {"Bet365 1X2": ("B365H", "B365D", "B365A"),
                             "Best 1X2": ("MaxH", "MaxD", "MaxA"),
                             "Pinnacle 1X2": ("PSH", "PSD", "PSA")}.items():
        x = extra[[h, d, a]].dropna()
        out[label] = float((1 / x[h] + 1 / x[d] + 1 / x[a] - 1).mean())
    for label, (o, u) in {"Bet365 O/U 2.5": ("B365>2.5", "B365<2.5"),
                          "Best O/U 2.5": ("Max>2.5", "Max<2.5")}.items():
        x = extra[[o, u]].dropna()
        out[label] = float((1 / x[o] + 1 / x[u] - 1).mean())
    return out


def main():
    t0 = time.time()
    con = connect()
    matches, odds, inj = load_all(con)
    feats = build_features(matches, odds, inj)
    ol = odds_lookup(odds)
    con.close()

    results = {"dev_cutoff": DEV_CUTOFF, "wf_start": WF_START, "step": STEP,
               "train_cols_full": len(FEATURES), "train_cols_no_mkt": len([f for f in FEATURES if not f.startswith("mkt_")]),
               "markets": list(MARKETS)}

    print("=== A1: full model development walk-forward ===")
    dev_matches, dev_picks = run_wf(feats, ol, WF_START, DEV_CUTOFF, use_mkt=True)
    print(f"walk-forward: {len(dev_matches)} matches, {len(dev_picks)} priced picks, {time.time()-t0:.0f}s")
    both = attach_market_baseline(dev_matches, odds)
    results["A1"] = {
        "matches": int(len(dev_matches)), "matches_with_market": int(len(both)),
        "brier_model": brier(both[["p_home", "p_draw", "p_away"]], both.y_result),
        "brier_market": brier(both[["mkt_h", "mkt_d", "mkt_a"]], both.y_result),
        "edge_full": edge_table(dev_picks),
        "priced_picks": int(len(dev_picks)),
    }
    print(f"A1 Brier full={results['A1']['brier_model']:.4f} "
          f"market={results['A1']['brier_market']:.4f} n={len(both)}")
    for r in results["A1"]["edge_full"]:
        print(f"  >{r['threshold']:.0%}: {r['bets']} bets, roi={r['roi']}, ci=[{r['lo']},{r['hi']}]")

    print("\n=== A2: no mkt_* ablation walk-forward ===")
    a2_matches, a2_picks = run_wf(feats, ol, WF_START, DEV_CUTOFF, use_mkt=False)
    both2 = attach_market_baseline(a2_matches, odds)
    results["A2"] = {
        "matches": int(len(a2_matches)), "matches_with_market": int(len(both2)),
        "brier_model": brier(both2[["p_home", "p_draw", "p_away"]], both2.y_result),
        "brier_market": brier(both2[["mkt_h", "mkt_d", "mkt_a"]], both2.y_result),
        "edge_no_mkt": edge_table(a2_picks), "priced_picks": int(len(a2_picks)),
    }
    print(f"A2 Brier no-mkt={results['A2']['brier_model']:.4f} "
          f"market={results['A2']['brier_market']:.4f} n={len(both2)}")
    for r in results["A2"]["edge_no_mkt"]:
        print(f"  >{r['threshold']:.0%}: {r['bets']} bets, roi={r['roi']}, ci=[{r['lo']},{r['hi']}]")

    print("\n=== A3: best-of-market / Pinnacle evaluation of the same picks ===")
    extra, missing = fetch_extra()
    results["A3"] = {"csv_files_missing": missing, "overround": overrounds(extra)}
    base = dev_picks[dev_picks.edge > PRIMARY].merge(extra, on="match_id", how="inner")
    cols = {"HOME_WIN": ("B365H", "MaxH", "PSH"), "AWAY_WIN": ("B365A", "MaxA", "PSA"),
            "OVER_2_5": ("B365>2.5", "Max>2.5", None)}
    per, all_ret = [], {"Bet365": [], "Best": [], "Pinnacle": []}
    for mk, (bc, xc, pc) in cols.items():
        sub = base[base.market == mk]
        row = {"market": mk, "picks": int(len(sub))}
        for name, col in (("Bet365", bc), ("Best", xc), ("Pinnacle", pc)):
            if col is None:
                row[name] = None
                continue
            s = sub.dropna(subset=[col])
            row[name] = float((s.outcome * s[col] - 1).mean()) if len(s) else None
            row[name + "_n"] = int(len(s))
            all_ret[name] += list((s.outcome * s[col] - 1).to_numpy())
        per.append(row)
    results["A3"]["per_market"] = per
    results["A3"]["all_roi"] = {k: (float(np.mean(v)) if v else None) for k, v in all_ret.items()}
    results["A3"]["all_n"] = {k: len(v) for k, v in all_ret.items()}
    results["A3"]["primary_bets"] = int(len(dev_picks[dev_picks.edge > PRIMARY]))
    results["A3"]["matched_bets"] = int(len(base))
    print(json.dumps(results["A3"], indent=2, default=str))

    print("\n=== A4: frozen 2026/27 holdout (one fit, no retuning) ===")
    mdl = M.train(feats, before=DEV_CUTOFF)
    hold = feats[(feats.status == "finished") & (feats.date >= DEV_CUTOFF)]
    hp = M.predict(mdl, hold).merge(hold[["match_id", "hg", "ag", "y_result"]], on="match_id")
    hpicks = picks_from(hp, ol)
    hboth = attach_market_baseline(
        pd.DataFrame([{"match_id": r.match_id, "p_home": r.p_home, "p_draw": r.p_draw,
                       "p_away": r.p_away, "y_result": int(r.y_result)} for r in hp.itertuples()]),
        odds)
    results["A4"] = {
        "holdout_matches": int(len(hold)), "priced_picks": int(len(hpicks)),
        "brier_model": brier(hboth[["p_home", "p_draw", "p_away"]], hboth.y_result) if len(hboth) else None,
        "brier_market": brier(hboth[["mkt_h", "mkt_d", "mkt_a"]], hboth.y_result) if len(hboth) else None,
        "brier_n": int(len(hboth)),
        "edge_holdout": edge_table(hpicks),
    }
    for r in results["A4"]["edge_holdout"]:
        print(f"  >{r['threshold']:.0%}: {r['bets']} bets, roi={r['roi']}, ci=[{r['lo']},{r['hi']}]")
    print(f"A4 holdout: {len(hold)} matches, {len(hpicks)} priced picks, Brier={results['A4']['brier_model']}")

    results["elapsed_s"] = round(time.time() - t0, 1)
    with open("reports/_part_a.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDONE in {results['elapsed_s']}s -> reports/_part_a.json")


if __name__ == "__main__":
    main()
