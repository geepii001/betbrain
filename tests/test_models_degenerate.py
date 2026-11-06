"""Regression test for the real-data crash: a column empty in the first time-ordered fold only."""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from betbrain import models as M
from betbrain.features import FEATURES

rng = np.random.default_rng(0)
for n in (441, 927, 1176, 2500):          # the exact sizes that mattered in the real run
    d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    d["h2h_pts_h"] = np.where(np.arange(n) < 0.3 * n, np.nan, d["h2h_pts_h"])   # empty early, filled later
    d["h2h_goals"] = np.where(np.arange(n) < 0.3 * n, np.nan, d["h2h_goals"])
    for c in ("h_def", "a_def"): d[c] = np.nan                                  # empty everywhere
    d["h_inj"] = 0.0                                                            # constant
    hg = rng.poisson(1.4, n); ag = rng.poisson(1.1, n)
    d["status"], d["date"] = "finished", pd.date_range("2021-08-01", periods=n).strftime("%Y-%m-%d")
    d["hg"], d["ag"] = hg, ag
    d["y_result"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    d["y_o15"], d["y_o25"] = ((hg + ag) >= 2).astype(int), ((hg + ag) >= 3).astype(int)
    d["y_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    for c in ("match_id", "league", "home", "away"): d[c] = "x"
    mdl = M.train(d, min_rows=400)
    p = M.predict(mdl, d.head(5))
    assert np.allclose(p[["p_home", "p_draw", "p_away"]].sum(axis=1), 1) and p.p_o15.between(0, 1).all()
    print(f"n={n}: train+predict OK")
print("degenerate-column regression test passed")
