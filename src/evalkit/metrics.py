"""Proper scoring rules + calibration metrics (SPEC_M2 §3).

Binning is equal-mass throughout (quantile edges; ties share a bin). The Murphy
decomposition is exact for the binned Brier score: BS_binned = REL - RES + UNC,
asserted to 1e-9 in the known-answer suite.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EPS = 1e-15


def nll_multiclass(probs: FloatArray, y: IntArray) -> float:
    """-(1/N) Σ log p_i[y_i], nats. probs: [n, K]; y: class indices."""
    picked = probs[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, EPS, None))))


def nll_binary(p: FloatArray, y: IntArray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_multiclass(probs: FloatArray, y: IntArray) -> float:
    """(1/N) Σ_i Σ_k (p_ik - 1[y_i = k])², range [0, 2]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def brier_binary(p: FloatArray, y: IntArray) -> float:
    return float(np.mean((p - y) ** 2))


def _equal_mass_bins(p: FloatArray, n_bins: int) -> list[IntArray]:
    """Indices per equal-mass bin, via quantile edges.

    Quantile edges (not index-splitting) so tied p values share a bin —
    splitting ties across bins would fabricate calibration error for
    constant or heavily-tied predictors.
    """
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    ids = np.searchsorted(edges, p, side="right")
    return [np.where(ids == b)[0] for b in range(n_bins) if np.any(ids == b)]


@dataclass(frozen=True)
class MurphyDecomposition:
    bs: float  # raw Brier score
    bs_binned: float  # Brier with p replaced by its bin mean
    rel: float  # reliability (miscalibration)
    res: float  # resolution
    unc: float  # uncertainty ȳ(1 - ȳ)
    n_bins: int


def murphy_decomposition(p: FloatArray, y: IntArray, n_bins: int = 20) -> MurphyDecomposition:
    """Binned Murphy decomposition over equal-mass bins (SPEC_M2 §3)."""
    n = len(p)
    y_f = y.astype(np.float64)
    ybar = float(np.mean(y_f))
    rel = res = bs_binned = 0.0
    for idx in _equal_mass_bins(p, n_bins):
        w = len(idx) / n
        pbar = float(np.mean(p[idx]))
        ybar_b = float(np.mean(y_f[idx]))
        rel += w * (pbar - ybar_b) ** 2
        res += w * (ybar_b - ybar) ** 2
        bs_binned += w * ((pbar - ybar_b) ** 2 + ybar_b * (1 - ybar_b))  # E[(p̄_b - y)²] within bin
    unc = ybar * (1 - ybar)
    return MurphyDecomposition(
        bs=brier_binary(p, y),
        bs_binned=bs_binned,
        rel=rel,
        res=res,
        unc=unc,
        n_bins=n_bins,
    )


@dataclass(frozen=True)
class ReliabilityData:
    """Per-bin means for reliability diagrams and ECE."""

    p_mean: FloatArray
    y_mean: FloatArray
    weight: FloatArray  # n_b / N


def reliability_data(p: FloatArray, y: IntArray, n_bins: int = 20) -> ReliabilityData:
    y_f = y.astype(np.float64)
    n = len(p)
    bins = _equal_mass_bins(p, n_bins)
    return ReliabilityData(
        p_mean=np.array([np.mean(p[b]) for b in bins]),
        y_mean=np.array([np.mean(y_f[b]) for b in bins]),
        weight=np.array([len(b) / n for b in bins]),
    )


def ece(p: FloatArray, y: IntArray, n_bins: int = 20) -> float:
    """Equal-mass ECE, B = 20 (SPEC_M2 §3)."""
    r = reliability_data(p, y, n_bins)
    return float(np.sum(r.weight * np.abs(r.p_mean - r.y_mean)))


def max_ce(p: FloatArray, y: IntArray, n_bins: int = 20) -> float:
    r = reliability_data(p, y, n_bins)
    return float(np.max(np.abs(r.p_mean - r.y_mean)))


def ece_per_class(probs: FloatArray, y: IntArray, n_bins: int = 20) -> FloatArray:
    """One-vs-rest ECE per class for T1 (SPEC_M2 §3)."""
    k = probs.shape[1]
    return np.array([ece(probs[:, c], (y == c).astype(np.int64), n_bins) for c in range(k)])


def skill_score(metric: float, metric_b0: float) -> float:
    """1 - metric/metric_B0 (positive = better than the marginal baseline)."""
    return 1.0 - metric / metric_b0
