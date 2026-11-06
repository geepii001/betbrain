"""Intelligence engine: learns from settled predictions what actually works.

It does three things:
1. Calibration - compares predicted probability with realised hit rate per market and confidence
   band (shrunk toward zero when data is thin) and corrects future confidence.
2. Banker gating - per market, finds the lowest confidence at which history really hit >= target;
   markets that never reach the target are barred from the Bankers list.
3. Insights - hit rate / ROI by market, odds band, league and section so you can see what works.
"""
import numpy as np
import pandas as pd

from .config import BANKER_FLOOR, BANKER_TARGET_HIT
from .db import q, set_state

BAND = 0.1
PRIOR = 40       # shrinkage strength for calibration corrections
MIN_BAND_N = 30
MIN_THR_N = 40


class Engine:
    def __init__(self, con):
        self.con = con
        self.refresh()

    def _pool(self):
        return q(self.con, "SELECT p.run_date, p.market, p.model_prob, p.confidence, p.outcome, p.odds, "
                           "p.odds_source, m.league FROM predictions p JOIN matches m USING(match_id) "
                           "WHERE p.section='POOL' AND p.outcome IS NOT NULL")

    def refresh(self):
        d = self._pool()
        self.n = len(d)
        self.mcorr, self.bcorr, self.thr, self.sratio = {}, {}, {}, {}
        sl = q(self.con, "SELECT section, outcome, win_prob FROM slips WHERE kind='acca' AND "
                         "outcome IS NOT NULL AND win_prob IS NOT NULL")
        if len(sl):
            k = 8.0  # prior weight toward ratio 1
            self.sratio["_all"] = float(np.clip((sl.outcome.sum() + k) / (sl.win_prob.sum() + k), 0.4, 1.1))
            for sec, g in sl.groupby("section"):
                self.sratio[sec] = float(np.clip((g.outcome.sum() + k) / (g.win_prob.sum() + k), 0.4, 1.1))
        if not len(d):
            return self
        d["band"] = np.floor(d.model_prob / BAND).astype(int)
        for mk, g in d.groupby("market"):
            self.mcorr[mk] = ((g.outcome.sum() - g.model_prob.sum()) / (len(g) + PRIOR), len(g))
        for (mk, b), g in d.groupby(["market", "band"]):
            self.bcorr[(mk, int(b))] = ((g.outcome.sum() - g.model_prob.sum()) / (len(g) + PRIOR), len(g))
        for mk, g in d.groupby("market"):
            if len(g) < 2 * MIN_THR_N:
                continue
            self.thr[mk] = None
            for t in np.arange(0.50, 0.96, 0.01):
                sel = g[g.model_prob >= t]
                if len(sel) >= MIN_THR_N and sel.outcome.mean() >= BANKER_TARGET_HIT:
                    self.thr[mk] = round(float(t), 2)
                    break
        set_state(self.con, "banker_thresholds", {k: v for k, v in self.thr.items()})
        return self

    def slip_ratio(self, section):
        """Learned correction for slips: searching many combinations picks the ones whose model
        errors look best (winner's curse), so realised win rate runs below predicted. Smoothed."""
        r = self.sratio.get(section)
        return r if r is not None else self.sratio.get("_all", 1.0)

    def adjust(self, market, p):
        bc = self.bcorr.get((market, int(np.floor(p / BAND))))
        mc = self.mcorr.get(market)
        corr = bc[0] if bc and bc[1] >= MIN_BAND_N else (mc[0] if mc else 0.0)
        return float(np.clip(p + corr, 0.01, 0.99))

    def banker_ok(self, market, conf):
        if market in self.thr:
            t = self.thr[market]
            return t is not None and conf >= max(t, BANKER_FLOOR - 0.05)
        return conf >= BANKER_FLOOR

    # ---- tipster-style learned rules ------------------------------------------------------
    def rules(self):
        """Plain-English lessons from settled history."""
        d = self._pool()
        if not len(d):
            return ["No settled history yet; engine is using raw model probabilities."]
        d["band"] = (np.floor(d.model_prob / BAND) * BAND).round(1)
        g = d.groupby(["market", "band"]).agg(n=("outcome", "size"), hit=("outcome", "mean"),
                                              pred=("model_prob", "mean")).reset_index()
        g = g[g.n >= MIN_BAND_N]
        out = []
        for r in g[(g.hit >= 0.75)].sort_values("hit", ascending=False).head(6).itertuples():
            out.append(f"TRUST: {r.market} rated {r.band:.0%}-{r.band + 0.1:.0%} has landed "
                       f"{r.hit:.0%} of the time ({r.n} picks).")
        for r in g[(g.pred - g.hit >= 0.05)].sort_values("n", ascending=False).head(4).itertuples():
            out.append(f"CAUTION: {r.market} rated {r.band:.0%}-{r.band + 0.1:.0%} is overrated by "
                       f"{r.pred - r.hit:.0%} (predicted {r.pred:.0%}, landed {r.hit:.0%}); confidence "
                       "is being cut.")
        for mk, t in sorted(self.thr.items()):
            out.append(f"BANKER RULE: {mk} " + (f"only qualifies at {t:.0%}+ confidence."
                                               if t else "never reached the hit-rate target; barred."))
        return out or ["History shows no strong pattern yet."]

    # ---- insights -------------------------------------------------------------------------
    def insights(self):
        d = self._pool()
        res = {"n_settled": len(d)}
        if not len(d):
            return res

        def roi(g):
            r = g[g.odds_source != "est"]
            return float((r.outcome * r.odds - 1).mean()) if len(r) else np.nan

        def agg(g):
            return pd.Series({"n": len(g), "hit_rate": g.outcome.mean(), "avg_pred": g.model_prob.mean(),
                              "brier": ((g.model_prob - g.outcome) ** 2).mean(), "roi": roi(g)})

        res["by_market"] = d.groupby("market").apply(agg, include_groups=False).round(3)
        d["conf_band"] = (np.floor(d.model_prob / BAND) * BAND).round(1)
        res["calibration"] = d.groupby("conf_band").apply(agg, include_groups=False).round(3)
        real = d[d.odds_source != "est"].copy()
        real["odds_band"] = pd.cut(real.odds, [1, 1.3, 1.6, 2.0, 3.0, 100],
                                   labels=["1.0-1.3", "1.3-1.6", "1.6-2.0", "2.0-3.0", "3.0+"])
        res["by_odds_band"] = real.groupby("odds_band", observed=True).apply(
            agg, include_groups=False).round(3)
        lg = d.groupby("league").apply(agg, include_groups=False).round(3)
        res["by_league"] = lg.sort_values("hit_rate", ascending=False)
        res["banker_thresholds"] = dict(self.thr)

        sec = q(self.con, "SELECT section, outcome, odds, odds_source, confidence FROM predictions "
                          "WHERE section!='POOL' AND outcome IS NOT NULL")
        if len(sec):
            def sagg(g):
                return pd.Series({"legs": len(g), "hit_rate": g.outcome.mean(),
                                  "avg_conf": g.confidence.mean(), "single_roi": roi(g)})
            res["by_section"] = sec.groupby("section").apply(sagg, include_groups=False).round(3)
        lg = q(self.con, "SELECT s.slip_id, s.outcome, s.win_prob, s.total_odds, COUNT(p.pred_id) legs "
                         "FROM slips s JOIN predictions p USING(slip_id) WHERE s.kind='acca' AND "
                         "s.outcome IS NOT NULL GROUP BY s.slip_id")
        if len(lg):
            res["by_legs"] = lg.groupby("legs").agg(slips=("outcome", "size"), win_rate=("outcome", "mean"),
                                                    avg_pred=("win_prob", "mean"),
                                                    avg_odds=("total_odds", "mean")).round(3)
        acc = q(self.con, "SELECT section, COUNT(*) slips, AVG(outcome) win_rate, AVG(total_odds) avg_odds, "
                          "AVG(win_prob) avg_pred, SUM(CASE WHEN outcome=1 THEN total_odds-1 ELSE -1 END)"
                          "*1.0/COUNT(*) roi FROM slips WHERE kind='acca' AND outcome IS NOT NULL "
                          "GROUP BY section")
        if len(acc):
            res["accas"] = acc.set_index("section").round(3)
        return res
