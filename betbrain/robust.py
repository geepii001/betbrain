"""HistGradientBoosting wrapper that survives columns that are entirely empty inside a training fold.

Why: this scikit-learn version raises "window shape cannot be larger than input array shape" when a
column has no non-missing values. Features like head-to-head are legitimately empty at the start of
the history, and time-ordered calibration folds train on only the earliest rows, so a column that has
data in the full training set can still be empty inside the first fold. Dropping empty columns once on
the full set (the earlier fix) cannot catch that; this drops them per fit, at every level.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier


class RobustHGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_iter=120, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                 random_state=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.keep_ = np.flatnonzero(~np.isnan(X).all(axis=0))
        if len(self.keep_) == 0:                      # nothing usable: fall back to one constant column
            X, self.keep_, self.dummy_ = np.zeros((len(X), 1)), np.array([0]), True
        else:
            X, self.dummy_ = X[:, self.keep_], False
        self.model_ = HistGradientBoostingClassifier(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_depth=self.max_depth,
            l2_regularization=self.l2_regularization, random_state=self.random_state).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def _x(self, X):
        X = np.asarray(X, dtype=float)
        return np.zeros((len(X), 1)) if self.dummy_ else X[:, self.keep_]

    def predict_proba(self, X):
        return self.model_.predict_proba(self._x(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
