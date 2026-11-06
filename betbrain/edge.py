"""Does the model actually beat the bookmaker? Flat-stake ROI on walk-forward predictions at real odds,
with a bootstrap confidence interval. This is the gate before trusting any pick with money.

Several minimum-edge thresholds are reported for context. The primary threshold (5% by default) is the
pre-registered gate; declaring a win on whichever of several thresholds happens to clear zero would
overstate the evidence."""
import numpy as np

from .db import q


def edge_test(con, thresholds=(0.0, 0.03, 0.05, 0.10), seed=0):
    d = q(con, "SELECT market, model_prob, odds, outcome FROM predictions WHERE section='POOL' "
               "AND backtest=1 AND outcome IS NOT NULL AND odds_source='book'")
    rng = np.random.default_rng(seed)
    rows = []
    for t in thresholds:
        s = d[d.model_prob * d.odds - 1 > t]
        if len(s) < 30:
            rows.append((t, len(s), np.nan, np.nan, np.nan))
            continue
        r = (s.outcome * s.odds - 1).to_numpy()
        boots = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        rows.append((t, len(s), r.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
    return d, rows


def print_edge(con, primary=0.05):
    d, rows = edge_test(con)
    print(f"Walk-forward bets at real bookmaker odds: {len(d)} priced predictions")
    print("min edge | bets | ROI per unit | 95% CI")
    for t, n, roi, lo, hi in rows:
        mark = "  <- primary gate" if abs(t - primary) < 1e-9 else ""
        if n < 30:
            print(f"  >{t:.0%}   | {n:5d} | too few bets{mark}")
        else:
            print(f"  >{t:.0%}   | {n:5d} | {roi:+.1%}       | [{lo:+.1%}, {hi:+.1%}]{mark}")
    gate = next((r for r in rows if abs(r[0] - primary) < 1e-9), None)
    if gate is None or gate[1] < 30:
        verdict = f"not enough bets at the primary {primary:.0%} threshold to judge"
    elif gate[3] > 0:
        verdict = (f"primary {primary:.0%} threshold has a 95% CI above zero; "
                   "re-test on fresh data before staking")
    else:
        verdict = (f"no evidence of an edge at the primary {primary:.0%} threshold "
                   "(95% CI includes or is below zero)")
    print("VERDICT:", verdict)
    print("Note: the other thresholds are exploratory; the primary threshold is the gate.")
