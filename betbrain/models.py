import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from .config import MODEL_PATH
from .features import FEATURES
from .robust import RobustHGBClassifier

TARGETS = {"result": "y_result", "o15": "y_o15", "o25": "y_o25", "btts": "y_btts"}


def _clf():
    base = RobustHGBClassifier(max_iter=120, learning_rate=0.05, max_depth=4,
                               l2_regularization=1.0, random_state=0)
    # Calibration must not look into the future. Ordinary K-fold lets later matches
    # help calibrate predictions for earlier ones; TimeSeriesSplit keeps every
    # calibration fold strictly chronological.
    return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))


def train(feats, before=None, min_rows=400):
    d = feats[feats["status"] == "finished"]
    if before is not None:
        d = d[d["date"] < before]
    if len(d) < min_rows:
        raise ValueError(f"Need at least {min_rows} finished matches to train, have {len(d)}")
    d = d.sort_values("date", kind="mergesort")
    # columns with no data at all (e.g. formations when the source has none) crash sklearn's binning
    cols = [c for c in FEATURES if d[c].notna().any()]
    mdl = {"_cols": cols}
    mdl.update({name: _clf().fit(d[cols], d[col].astype(int)) for name, col in TARGETS.items()})
    # goal-rate models: give the full scoreline distribution, needed to price same-game stacks
    for name, col in (("lam_h", "hg"), ("lam_a", "ag")):
        mdl[name] = HistGradientBoostingRegressor(
            loss="poisson", max_iter=150, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=0).fit(d[cols], d[col].astype(float))
    return mdl


def _proba(clf, X, n_classes):
    """predict_proba with stable columns even if a calibration fold missed a class."""
    p = clf.predict_proba(X)
    if p.shape[1] == n_classes:
        return p
    out = np.zeros((len(X), n_classes))
    for j, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, j]
    return out


def _p_positive(clf, X):
    p = clf.predict_proba(X)
    if p.shape[1] == 2:
        return p[:, 1]
    # Only one class was seen in training; the honest probability is 0 or 1.
    return np.full(len(X), float(clf.classes_[0]))


def predict(models, df):
    X = df[models["_cols"]]
    res = _proba(models["result"], X, 3)
    out = df[["match_id", "date", "league", "home", "away"]].copy()
    out["p_home"], out["p_draw"], out["p_away"] = res[:, 0], res[:, 1], res[:, 2]
    out["p_o15"] = _p_positive(models["o15"], X)
    out["p_o25"] = _p_positive(models["o25"], X)
    out["p_btts"] = _p_positive(models["btts"], X)
    out["lam_h"] = np.clip(models["lam_h"].predict(X), 0.15, 5.0)
    out["lam_a"] = np.clip(models["lam_a"].predict(X), 0.15, 5.0)
    return out.reset_index(drop=True)


def save(models):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)


def load():
    return joblib.load(MODEL_PATH)
