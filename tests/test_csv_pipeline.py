"""Round-trip the synthetic world through football-data.co.uk CSV format, then run the real-data path
(download is faked with a stub session). Proves ingest -> backtest -> edge -> daily works on CSV input."""
import io, sys, pathlib, os, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd, numpy as np
from datetime import date
from betbrain.providers import synthetic, footballdata_uk as FU
from betbrain import config
tmp = pathlib.Path(tempfile.mkdtemp())
config.DB_PATH = tmp / "t.db"
import betbrain.db as D; D.DB_PATH = config.DB_PATH
from betbrain.db import connect
from betbrain import pipeline, report
from betbrain.edge import edge_test

today = date(2026, 11, 3)
m, o, i, hidden = synthetic.generate(today, years=2)
W = o.pivot_table(index="match_id", columns="market", values="odds", aggfunc="first")
m = m.merge(W, left_on="match_id", right_index=True, how="left")
m["Div"] = m.league.map({"Premier Division": "E0", "Championship": "E1", "Liga Norte": "SP1", "Serie Sul": "I1"})
def fdfmt(df, results=True):
    out = pd.DataFrame({"Div": df.Div, "Date": pd.to_datetime(df.date).dt.strftime("%d/%m/%Y"),
                        "HomeTeam": df.home, "AwayTeam": df.away,
                        "B365H": df.HOME_WIN, "B365D": df.DRAW, "B365A": df.AWAY_WIN,
                        "B365>2.5": df.OVER_2_5, "B365<2.5": df.UNDER_2_5})
    if results:
        out["FTHG"], out["FTAG"] = df.home_goals, df.away_goals
    return out
hist, fut = m[m.status == "finished"], m[m.status == "scheduled"]
files = {}
for dv, g in hist.groupby("Div"):
    files[f"/mmz4281/2526/{dv}.csv"] = fdfmt(g).to_csv(index=False).encode()
files["/fixtures.csv"] = fdfmt(fut, results=False).to_csv(index=False).encode()   # no FTHG/FTAG columns, like the real file
class R:
    def __init__(s, b): s.content, s.status_code = b, 200 if b else 404
class S:
    def get(s, url, headers=None, timeout=None):
        return R(files.get(url.replace(FU.BASE, ""), b""))
mm, oo, ii, log = FU.download(seasons=["2526", "2425"], session=S())
print("\n".join(log))
assert (mm.status == "finished").sum() == len(hist) and (mm.status == "scheduled").sum() == len(fut)
assert mm.league.isin(["Premier League", "Championship", "La Liga", "Serie A"]).all()
assert {"HOME_WIN", "OVER_2_5"} <= set(oo.market)
con = connect()
from betbrain.db import upsert_matches, upsert_odds, upsert_injuries
upsert_matches(con, mm); upsert_odds(con, oo); upsert_injuries(con, ii)
n = pipeline.backtest(con, pd.Timestamp(today).date(), days=90)
d, rows = edge_test(con)
res = pipeline.daily(con, pd.Timestamp(today).date())
assert n > 40 and len(d) > 1000 and res is not None
txt = report.render(res, res["engine"].insights())
assert "Daily Bankers" in txt and "Crowd" not in txt
print(f"CSV pipeline OK: {n} backtest days, {len(d)} priced predictions, report {len(txt)} chars")
