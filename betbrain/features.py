"""Leak-free feature builder: one chronological pass, features for a match use only prior matches."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURES = [
    "elo_h", "elo_a", "elo_diff",
    "h_gf5", "h_ga5", "h_pts5", "a_gf5", "a_ga5", "a_pts5",
    "h_rest", "a_rest", "h_games", "a_games",
    "h_inj", "a_inj",
    "h_def", "h_mid", "h_fwd", "a_def", "a_mid", "a_fwd",
    "mkt_h", "mkt_d", "mkt_a", "mkt_o25", "mkt_o15",
    "h2h_n", "h2h_pts_h", "h2h_goals",
]
NAN = float("nan")


def _formation(f):
    try:
        p = [int(x) for x in str(f).split("-")]
        return p[0], sum(p[1:-1]), p[-1]
    except Exception:
        return NAN, NAN, NAN


def _form5(d):
    if not d:
        return NAN, NAN, NAN
    return (np.mean([x[0] for x in d]), np.mean([x[1] for x in d]), np.mean([x[2] for x in d]))


def build_features(matches, odds, injuries):
    ow = {}
    if len(odds):
        ow = odds.pivot_table(index="match_id", columns="market", values="odds",
                              aggfunc="first").to_dict("index")
    inj = {}
    if len(injuries):
        inj = injuries.groupby(["match_id", "team"])["importance"].sum().to_dict()

    m = matches.copy()
    m["dt"] = pd.to_datetime(m["date"])
    m = m.sort_values(["dt", "match_id"]).reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    form = defaultdict(lambda: deque(maxlen=5))
    last, games = {}, defaultdict(int)
    h2h = defaultdict(lambda: deque(maxlen=6))   # last 6 meetings per pair, either venue
    out = []

    for r in m.itertuples(index=False):
        h, a = r.home, r.away
        o = ow.get(r.match_id, {})

        def imp(k):
            v = o.get(k)
            return 1.0 / v if v is not None and v == v and v > 0 else NAN

        ih, idr, ia = imp("HOME_WIN"), imp("DRAW"), imp("AWAY_WIN")
        s = ih + idr + ia
        io, iu = imp("OVER_2_5"), imp("UNDER_2_5")
        hf, af = _form5(form[h]), _form5(form[a])
        meet = h2h[tuple(sorted((h, a)))]
        pts_h = []
        for (mh, mhg, mag) in meet:
            gd_h = (mhg - mag) if mh == h else (mag - mhg)
            pts_h.append(3 if gd_h > 0 else 1 if gd_h == 0 else 0)
        hd, hm, hfw = _formation(r.formation_home)
        ad, am, afw = _formation(r.formation_away)
        out.append({
            "match_id": r.match_id, "date": r.date, "league": r.league, "home": h, "away": a,
            "status": r.status,
            "elo_h": elo[h], "elo_a": elo[a], "elo_diff": elo[h] - elo[a],
            "h_gf5": hf[0], "h_ga5": hf[1], "h_pts5": hf[2],
            "a_gf5": af[0], "a_ga5": af[1], "a_pts5": af[2],
            "h_rest": (r.dt - last[h]).days if h in last else NAN,
            "a_rest": (r.dt - last[a]).days if a in last else NAN,
            "h_games": games[h], "a_games": games[a],
            "h_inj": inj.get((r.match_id, h), 0.0), "a_inj": inj.get((r.match_id, a), 0.0),
            "h_def": hd, "h_mid": hm, "h_fwd": hfw, "a_def": ad, "a_mid": am, "a_fwd": afw,
            "mkt_h": ih / s, "mkt_d": idr / s, "mkt_a": ia / s,
            "mkt_o25": io / (io + iu) if io == io and iu == iu else NAN,
            "mkt_o15": imp("OVER_1_5"),
            "h2h_n": len(meet),
            "h2h_pts_h": float(np.mean(pts_h)) if pts_h else NAN,
            "h2h_goals": float(np.mean([x[1] + x[2] for x in meet])) if meet else NAN,
            "hg": r.home_goals, "ag": r.away_goals,
        })
        if r.status == "finished" and r.home_goals == r.home_goals and r.home_goals is not None:
            hg, ag = float(r.home_goals), float(r.away_goals)
            gd = hg - ag
            exp = 1 / (1 + 10 ** (-(elo[h] + 60 - elo[a]) / 400))
            act = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            k = 20 * (1 + 0.5 * np.log1p(abs(gd)))
            elo[h] += k * (act - exp)
            elo[a] -= k * (act - exp)
            form[h].append((hg, ag, 3 if gd > 0 else 1 if gd == 0 else 0))
            form[a].append((ag, hg, 3 if gd < 0 else 1 if gd == 0 else 0))
            h2h[tuple(sorted((h, a)))].append((h, hg, ag))
            last[h] = last[a] = r.dt
            games[h] += 1
            games[a] += 1

    df = pd.DataFrame(out)
    fin = df["status"] == "finished"
    df["y_result"] = np.where(fin, np.where(df.hg > df.ag, 0, np.where(df.hg == df.ag, 1, 2)), -1)
    df["y_o15"] = np.where(fin, ((df.hg + df.ag) >= 2).astype(int), -1)
    df["y_o25"] = np.where(fin, ((df.hg + df.ag) >= 3).astype(int), -1)
    df["y_btts"] = np.where(fin, ((df.hg > 0) & (df.ag > 0)).astype(int), -1)
    return df
