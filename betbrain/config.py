from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DB_PATH = DATA / "betbrain.db"
MODEL_PATH = DATA / "models.joblib"

# Odds used when a bookmaker price is missing: model fair odds shaved by a typical margin.
EST_MARGIN = 0.93
MIN_ODDS = 1.10          # picks below this are not worth a leg
BANKER_FLOOR = 0.80      # default minimum confidence for a banker
BANKER_TARGET_HIT = 0.80 # hit rate the engine tries to guarantee historically

MARKETS = ["HOME_WIN", "AWAY_WIN", "DC_1X", "DC_X2", "OVER_1_5", "OVER_2_5", "BTTS_YES"]
