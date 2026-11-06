# BetBrain

An end-to-end machine-learning football prediction engine, built to answer one question honestly:

> **Can a calibrated model beat the bookmaker's closing price?**

On 12,277 real matches across six leagues, the answer is **no**. I stopped betting-related
development on a pre-registered holdout instead of tuning a losing strategy. This repository contains
the engine, its tests, and the full reproducible analysis.

## Key result

| Test | Result |
|---|---|
| Development walk-forward, 2022/23–2025/26 (9,279 matches) | 5% gate: **−12.2% ROI** (95% CI [−15.9%, −8.4%]) |
| Brier score, 1X2: market vs model | **0.5875 vs 0.6029** (model worse) |
| Ablation without market-implied features | −12.7% ROI; Brier 0.6174 |
| Same picks at best available price | −7.3% ROI (still negative) |
| Frozen 2026/27 holdout, one fit (609 matches) | −2.2% ROI (95% CI [−17.0%, +12.9%]) |

The model is well calibrated, but it learns to reproduce the market and then pays the 5–8% margin.
The full write-up, including every table and the stop-rule decision, is in
[`reports/analysis_2026-11-06.md`](reports/analysis_2026-11-06.md).

## What's inside

| Module | Role |
|---|---|
| `features.py` | Leak-free chronological pass: Elo, last-5 form, rest days, injuries, formation shape, head-to-head, market-implied probabilities |
| `models.py` | Calibrated gradient boosting for 1X2, Over 1.5, Over 2.5 and BTTS, plus Poisson goal-rate regressors used to price same-game stacks |
| `robust.py` | `RobustHGBClassifier`: drops all-empty columns **per fit**, so time-ordered calibration folds survive features that are empty early in history |
| `pipeline.py` | Walk-forward backtest, daily prediction, settlement and retraining |
| `edge.py` | Flat-stake ROI at real bookmaker odds, with a bootstrap confidence interval and a primary 5% gate |
| `engine.py` | Intelligence layer: calibration by market/confidence band, banker gating, learned rules |
| `picks.py` | Candidate generation and slip construction (Bankers, 2/3-odds, Over 1.5, Straight Wins, Multi, Weekend/Weekly) |
| `store.py` / `db.py` | SQLite persistence for every prediction and outcome |
| `providers/` | football-data.co.uk CSV loader, football-data.org, API-Football, synthetic demo data |

## Engineering highlights

- **No look-ahead.** Every feature for a match uses only earlier matches; calibration uses
  `TimeSeriesSplit`, not ordinary K-fold.
- **Honest evaluation.** Walk-forward blocks, real closing odds, a pre-registered 5% gate, and a
  2,000-resample bootstrap CI. The holdout is used once.
- **Regression-tested failure mode.** `tests/test_models_degenerate.py` reproduces the real crash at
  1,176 rows and proves the per-fit empty-column fix.
- **Reproducible analysis.** `reports/_part_a.py` regenerates the development walk-forward and the
  frozen holdout; `reports/_part_a.json` holds the raw numbers.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m betbrain demo                 # full synthetic loop, no network
```

Real history (football-data.co.uk, six seasons, six leagues):

```powershell
python -m betbrain real-data --reset    # download, backtest, edge-test
python -m betbrain edge-test            # 5% is the primary gate
python -m betbrain daily                # settle, learn, retrain, predict
```

## Tests

```powershell
python tests/test_models_degenerate.py
python tests/test_apifootball.py
python tests/test_crowd.py
python tests/test_csv_pipeline.py       # downloads real CSVs
```

## Reproduce the analysis

```powershell
python -m betbrain real-data --reset    # builds data/betbrain.db (not committed)
python reports/_part_a.py               # development walk-forward + frozen holdout
```

The database and trained models are intentionally not committed (`data/*.db`, `data/*.joblib`).

## API-Football

The adapter is implemented for fixtures, odds, injuries and lineups. It is mock-tested in
`tests/test_apifootball.py`. The live smoke test was **not run** in the published environment because
the API key was not exposed to the execution shell; it can be run with:

```powershell
$env:API_FOOTBALL_KEY = "<your key>"
python -m betbrain fetch-api 2026-11-07 2026-11-09 --leagues "Premier League"
python reports/_part_b_capture.py
```

The capture script never prints the key and scrubs it from saved samples.

## What I would try next

The negative result is about this feature set and these markets, not about football being
unpredictable in principle. The obvious next steps are genuinely new information:

- lineups and player-level injury impact, rather than a 0.6/0.3 proxy;
- xG/xA and shot-quality data;
- less efficient leagues or niche markets;
- a different probability model (e.g. Dixon–Coles-style goal models) on richer inputs.

Any re-opening of betting development needs a **new, separately pre-registered test** on data that has
not been used for tuning. The 2026/27 holdout has already been spent.

## Stack

Python 3.13, pandas, NumPy, scikit-learn, SQLite, requests. No paid data is required for the core
backtest.

## A note on the result

A negative result is still a result. The engine was built to be wrong in a measurable way, and it was:
the market's probabilities beat the model's, consistently, before and after the margin. Stopping was
the correct engineering decision.
