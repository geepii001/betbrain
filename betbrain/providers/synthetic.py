"""Synthetic league simulator so the whole pipeline can be exercised without real data.

Teams have drifting attack/defence strength; injuries weaken teams; the 'bookmaker' prices every
market from the true probabilities with ~5-6% margin plus noise. Future results are held back
in `hidden` so the demo can reveal them day by day and let the engine learn.
"""
import numpy as np
import pandas as pd
from math import exp, factorial

CITIES = ["Arden", "Brook", "Carlow", "Dunmore", "Eastvale", "Falcon", "Garnet", "Halden",
          "Ironwood", "Juniper", "Kestrel", "Linden", "Marlow", "Norwick", "Oakhurst", "Pennant"]
LEAGUES = {"Premier Division": "FC", "Championship": "United", "Liga Norte": "City", "Serie Sul": "Town"}
FORMATIONS = ["4-4-2", "4-3-3", "3-5-2", "4-2-3-1", "5-3-2", "4-1-4-1"]
DAY_P = {0: 0.15, 1: 0.45, 2: 0.45, 3: 0.30, 4: 0.30, 5: 0.95, 6: 0.80}


def _pmf(lam, n=10):
    return np.array([exp(-lam) * lam ** k / factorial(k) for k in range(n)])


def _odds(p, rng, margin, sigma=0.03):
    p = np.asarray(p, float) * rng.lognormal(0, sigma, len(p))
    p = p / p.sum()
    return np.round(1.0 / (p * (1 + margin)), 2)


def generate(today, years=3, seed=7):
    rng = np.random.default_rng(seed)
    today = pd.Timestamp(today)
    teams = {lg: [f"{c} {suf}" for c in CITIES] for lg, suf in LEAGUES.items()}
    atk = {t: rng.normal(0, 0.25) for ts in teams.values() for t in ts}
    dfn = {t: rng.normal(0, 0.25) for ts in teams.values() for t in ts}
    base = np.log(1.3)
    matches, odds, injuries, hidden = [], [], [], []
    n = 0
    start = today - pd.Timedelta(days=365 * years)
    for d in pd.date_range(start, today + pd.Timedelta(days=7)):
        if d.day == 1:
            for t in atk:
                atk[t] = 0.98 * atk[t] + rng.normal(0, 0.04)
                dfn[t] = 0.98 * dfn[t] + rng.normal(0, 0.04)
        for lg, ts in teams.items():
            if rng.random() > DAY_P[d.weekday()]:
                continue
            k = int(rng.integers(4, 8))
            pick = list(rng.permutation(ts))[: 2 * k]
            for i in range(k):
                h, a = pick[2 * i], pick[2 * i + 1]
                n += 1
                mid = f"S{n:06d}"
                imp = {}
                for t in (h, a):
                    cnt = int(rng.poisson(1.1))
                    imp[t] = rng.uniform(0.2, 1.0, cnt)
                    for j, v in enumerate(imp[t]):
                        injuries.append({"match_id": mid, "team": t, "player": f"Player {j + 1}",
                                         "importance": round(float(v), 2)})
                ih, ia = imp[h].sum(), imp[a].sum()
                lh = np.exp(base + 0.12 + atk[h] - 0.06 * ih - dfn[a] + 0.04 * ia)
                la = np.exp(base + atk[a] - 0.06 * ia - dfn[h] + 0.04 * ih)
                grid = np.outer(_pmf(lh), _pmf(la))
                p1, px, p2 = np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()
                lam = lh + la
                po15 = 1 - exp(-lam) * (1 + lam)
                po25 = 1 - exp(-lam) * (1 + lam + lam ** 2 / 2)
                pbt = (1 - exp(-lh)) * (1 - exp(-la))
                o1x2 = _odds([p1, px, p2], rng, 0.06)
                o25 = _odds([po25, 1 - po25], rng, 0.05)
                o15 = _odds([po15, 1 - po15], rng, 0.05)
                obt = _odds([pbt, 1 - pbt], rng, 0.05)
                for mk, v in (("HOME_WIN", o1x2[0]), ("DRAW", o1x2[1]), ("AWAY_WIN", o1x2[2]),
                              ("OVER_2_5", o25[0]), ("UNDER_2_5", o25[1]),
                              ("OVER_1_5", o15[0]), ("BTTS_YES", obt[0])):
                    odds.append({"match_id": mid, "market": mk, "odds": float(v), "source": "synthetic"})
                hg, ag = int(rng.poisson(lh)), int(rng.poisson(la))
                done = d < today
                matches.append({
                    "match_id": mid, "date": d.strftime("%Y-%m-%d"), "league": lg, "home": h, "away": a,
                    "home_goals": hg if done else None, "away_goals": ag if done else None,
                    "status": "finished" if done else "scheduled",
                    "formation_home": str(rng.choice(FORMATIONS)),
                    "formation_away": str(rng.choice(FORMATIONS)), "source": "synthetic"})
                if not done:
                    hidden.append({"match_id": mid, "date": d.strftime("%Y-%m-%d"),
                                   "home_goals": hg, "away_goals": ag})
    return (pd.DataFrame(matches), pd.DataFrame(odds), pd.DataFrame(injuries), pd.DataFrame(hidden))
