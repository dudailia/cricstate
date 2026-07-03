"""Weighted PAVA + alternating 2D monotone smoothing (SPEC_M2 amendment #3).

B1 (T2) applies this post-shrinkage over each (innings, phase) lattice of
wickets-band x rate-band cells, weighted by train leaf counts (+1 so unseen
lattice cells conform to their neighbours with minimal influence). Fit on
TRAIN quantities only.
"""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MAX_SWEEPS = 200
TOL = 1e-12


def pava_1d(values: FloatArray, weights: FloatArray, increasing: bool = True) -> FloatArray:
    """Weighted isotonic regression (pool adjacent violators), L2."""
    if not increasing:
        return pava_1d(values[::-1], weights[::-1], increasing=True)[::-1]
    n = len(values)
    level_val = values.astype(np.float64).copy()
    level_w = weights.astype(np.float64).copy()
    # blocks as (start, value, weight); classic stack-based PAVA
    starts = list(range(n))
    vals = list(level_val)
    ws = list(level_w)
    out_starts: list[int] = []
    out_vals: list[float] = []
    out_ws: list[float] = []
    for s, v, w in zip(starts, vals, ws, strict=True):
        out_starts.append(s)
        out_vals.append(v)
        out_ws.append(w)
        while len(out_vals) > 1 and out_vals[-2] > out_vals[-1] + TOL:
            v2, w2 = out_vals.pop(), out_ws.pop()
            out_starts.pop()
            v1, w1 = out_vals[-1], out_ws[-1]
            merged = (v1 * w1 + v2 * w2) / (w1 + w2)
            out_vals[-1] = merged
            out_ws[-1] = w1 + w2
    result = np.empty(n)
    bounds = [*out_starts[1:], n]
    for (s, v), e in zip(zip(out_starts, out_vals, strict=True), bounds, strict=True):
        result[s:e] = v
    return result


def _violates(grid: FloatArray, rows_increasing: bool, cols_increasing: bool) -> bool:
    row_diffs = np.diff(grid, axis=1)  # along rate axis
    col_diffs = np.diff(grid, axis=0)  # along wickets axis
    row_bad = np.any(row_diffs < -TOL) if rows_increasing else np.any(row_diffs > TOL)
    col_bad = np.any(col_diffs < -TOL) if cols_increasing else np.any(col_diffs > TOL)
    return bool(row_bad or col_bad)


def monotone_smooth_2d(
    grid: FloatArray,
    weights: FloatArray,
    rate_increasing: bool = True,
    wickets_increasing: bool = False,
) -> FloatArray:
    """Alternate row/column weighted PAVA sweeps until both constraints hold.

    grid: [n_wickets_bands, n_rate_bands]. Deterministic; raises if the sweep
    cap is hit without convergence (never ships silently non-monotone).
    """
    out = grid.astype(np.float64).copy()
    w = np.maximum(weights.astype(np.float64), 1e-9)
    for _ in range(MAX_SWEEPS):
        for i in range(out.shape[0]):  # along rate bands
            out[i] = pava_1d(out[i], w[i], increasing=rate_increasing)
        for j in range(out.shape[1]):  # along wickets bands
            out[:, j] = pava_1d(out[:, j], w[:, j], increasing=wickets_increasing)
        if not _violates(out, rate_increasing, wickets_increasing):
            return out
    raise RuntimeError("monotone_smooth_2d failed to converge — refusing to ship")
