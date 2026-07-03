"""Calibration maps (SPEC_M2 §5): temperature, Platt, isotonic.

Discipline is structural: every fit_* function's data parameters are named
val_* — calibration maps are fit on the validation split only, never train,
never test. All fits are deterministic (bounded scalar minimization / lbfgs
with fixed setup / PAVA).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EPS = 1e-15


def _softmax(z: FloatArray) -> FloatArray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    result: FloatArray = e / e.sum(axis=1, keepdims=True)
    return result


@dataclass(frozen=True)
class TemperatureScaler:
    """p ∝ exp(log p / T). T > 1 softens, T < 1 sharpens."""

    temperature: float

    def apply(self, logprobs: FloatArray) -> FloatArray:
        return _softmax(logprobs / self.temperature)


def fit_temperature(val_logprobs: FloatArray, val_y: IntArray) -> TemperatureScaler:
    """T minimizing val NLL, bounded scalar search (deterministic)."""

    def nll_at(t: float) -> float:
        p = _softmax(val_logprobs / t)
        picked = p[np.arange(len(val_y)), val_y]
        return float(-np.mean(np.log(np.clip(picked, EPS, None))))

    res = minimize_scalar(nll_at, bounds=(0.05, 20.0), method="bounded")
    return TemperatureScaler(temperature=float(res.x))


@dataclass(frozen=True)
class PlattScaler:
    """sigmoid(a*logit(p) + b), the classic sigmoid recalibration."""

    a: float
    b: float

    def apply(self, p: FloatArray) -> FloatArray:
        z = self.a * _logit(p) + self.b
        result: FloatArray = 1.0 / (1.0 + np.exp(-z))
        return result


def _logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, EPS, 1 - EPS)
    result: FloatArray = np.log(p / (1 - p))
    return result


def fit_platt(val_p: FloatArray, val_y: IntArray) -> PlattScaler:
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(val_p).reshape(-1, 1), val_y)
    return PlattScaler(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))


@dataclass(frozen=True)
class IsotonicScaler:
    """Monotone non-parametric map (PAVA); the T2 leaderboard entry."""

    model: IsotonicRegression

    def apply(self, p: FloatArray) -> FloatArray:
        out = self.model.predict(p)
        result: FloatArray = np.clip(np.asarray(out, dtype=np.float64), EPS, 1 - EPS)
        return result


def fit_isotonic(val_p: FloatArray, val_y: IntArray) -> IsotonicScaler:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(val_p, val_y.astype(np.float64))
    return IsotonicScaler(model=iso)
