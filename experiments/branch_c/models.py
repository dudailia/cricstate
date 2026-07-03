"""Branch C predictors: frozen B3, and B3 tilted by the causal latent (C1).

    M_state          = frozen B3 (loaded from the M2 artifact cache, never
                       refit) + val temperature — reproduces the M2 t1/t20 val
                       NLL 1.62604 exactly.
    M_latent         = B3 logits + a logit tilt driven by theta_hat(m,t), the
                       full-match causal latent (carry-over across innings).
    M_latent_innings = same, but the latent resets at the innings boundary (no
                       carry-over) — the confound-isolation variant.

SIGNAL = B3 RESIDUAL (per protocol): s_i = valence(y_i) - E_B3[valence | state_i].
The latent therefore explains only the scoring B3 does NOT already capture via
run-rate / wickets / state; it cannot relabel signal B3 already has.

LINK (gain-free, per "kappa is the only tunable"): B3's per-ball distribution
is exponentially tilted along the centred-valence direction,
    logit_shift_k = theta_hat * (valence_k - vbar) / tau,
with vbar, tau = train-population valence mean and variance (fixed constants).
kappa alone controls both the latent's magnitude (via shrinkage) and hence the
shift size: kappa -> inf collapses M_latent to M_state.

LOGGED ASSUMPTIONS: temperature calibration per model, fit on val (M2
convention); the val kappa curve is scored post-temperature. Same kappa is
used for both latent variants so their ONLY difference is the carry-over.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit import cache
from evalkit.calibrate import TemperatureScaler, fit_temperature
from evalkit.datasets import assemble_t1
from evalkit.metrics import nll_multiclass
from evalkit.models.base import CLASSES, Predictor
from experiments.branch_c.latent import ORDER_COLS, VALENCE, causal_theta, population_valence

FloatArray = NDArray[np.float64]

EPS = 1e-12
K = len(CLASSES)
FULL_MATCH: tuple[str, ...] = ("match_id",)
PER_INNINGS: tuple[str, ...] = ("match_id", "innings_idx")
KAPPA_GRID = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 600.0)


def load_frozen_b3(task: str = "t1", fmt: str = "t20") -> Predictor:
    model = cache.load(task, fmt, "B3_gbm")
    if model is None:
        raise RuntimeError(
            "frozen B3 not in the artifact cache — run `uv run evalkit run-all` first; "
            "Branch C never refits it"
        )
    return model


def assemble_with_order(deliveries: pl.DataFrame, fmt: str, split: str) -> pl.DataFrame:
    """Frozen-harness T1 assembly + match_id (already present) + ORDER_COLS,
    via the identical filter the frozen assembler applies (row-aligned)."""
    rows = deliveries.filter(
        (pl.col("fmt") == fmt)
        & (pl.col("temporal_split") == split)
        & ~pl.col("excluded_from_tuples")
    )
    base = assemble_t1(deliveries, fmt, split)
    assert base.height == rows.height
    return base.with_columns(*(rows[c].alias(c) for c in ORDER_COLS))


def residual_signal(b3_probs: FloatArray, y: NDArray[np.int64]) -> FloatArray:
    """s_i = valence(outcome_i) - B3's expected valence for ball i (residual)."""
    expected = b3_probs @ VALENCE
    result: FloatArray = VALENCE[y] - expected
    return result


def latent_tilt(b3_probs: FloatArray, theta: FloatArray, vbar: float, tau: float) -> FloatArray:
    """Exponential tilt of B3 along centred valence: gain-free, kappa-scaled."""
    direction = (VALENCE - vbar) / tau  # [K]
    logits = np.log(np.clip(b3_probs, EPS, None)) + theta[:, None] * direction[None, :]
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    result: FloatArray = e / e.sum(axis=1, keepdims=True)
    return result


def latent_probs(
    b3_probs: FloatArray,
    df: pl.DataFrame,
    kappa: float,
    group_cols: tuple[str, ...],
    vbar: float,
    tau: float,
) -> FloatArray:
    """Pre-calibration tilted probabilities for a latent model."""
    y = df["y"].to_numpy().astype(np.int64)
    sig = residual_signal(b3_probs, y)
    order = df.select("match_id", *ORDER_COLS).with_columns(pl.Series("_sig", sig))
    theta = causal_theta(order, signal_col="_sig", kappa=kappa, group_cols=group_cols)
    return latent_tilt(b3_probs, theta, vbar, tau)


@dataclass(frozen=True)
class LatentModel:
    name: str
    kappa: float | None  # None = M_state (no latent)
    group_cols: tuple[str, ...]
    scaler: TemperatureScaler
    vbar: float
    tau: float

    def raw_probs(self, b3_probs: FloatArray, df: pl.DataFrame) -> FloatArray:
        if self.kappa is None:
            return b3_probs
        return latent_probs(b3_probs, df, self.kappa, self.group_cols, self.vbar, self.tau)

    def predict(self, b3_probs: FloatArray, df: pl.DataFrame) -> FloatArray:
        return self.scaler.apply(np.log(np.clip(self.raw_probs(b3_probs, df), EPS, None)))


def _calibrate(raw_val: FloatArray, y_val: NDArray[np.int64]) -> TemperatureScaler:
    return fit_temperature(np.log(np.clip(raw_val, EPS, None)), y_val)


def build_state(b3_val: FloatArray, val: pl.DataFrame) -> LatentModel:
    y_val = val["y"].to_numpy().astype(np.int64)
    return LatentModel("M_state", None, FULL_MATCH, _calibrate(b3_val, y_val), 0.0, 1.0)


def build_latent(
    name: str,
    kappa: float,
    group_cols: tuple[str, ...],
    b3_val: FloatArray,
    val: pl.DataFrame,
    vbar: float,
    tau: float,
) -> LatentModel:
    y_val = val["y"].to_numpy().astype(np.int64)
    raw = latent_probs(b3_val, val, kappa, group_cols, vbar, tau)
    return LatentModel(name, kappa, group_cols, _calibrate(raw, y_val), vbar, tau)


@dataclass(frozen=True)
class KappaCurve:
    kappa_grid: tuple[float, ...]
    val_nll: tuple[float, ...]
    best_kappa: float

    def is_unimodal(self) -> bool:
        v = np.array(self.val_nll)
        i = int(np.argmin(v))
        return bool(np.all(np.diff(v[: i + 1]) <= 1e-12) and np.all(np.diff(v[i:]) >= -1e-12))


def tune_kappa(
    b3_val: FloatArray,
    val: pl.DataFrame,
    vbar: float,
    tau: float,
    group_cols: tuple[str, ...] = FULL_MATCH,
    grid: tuple[float, ...] = KAPPA_GRID,
) -> KappaCurve:
    """Val-NLL-only tuning of the single tunable kappa. Test never enters."""
    y_val = val["y"].to_numpy().astype(np.int64)
    scores: list[float] = []
    for kappa in grid:
        model = build_latent("M_latent", kappa, group_cols, b3_val, val, vbar, tau)
        scores.append(nll_multiclass(model.predict(b3_val, val), y_val))
    best = grid[int(np.argmin(scores))]
    return KappaCurve(kappa_grid=tuple(grid), val_nll=tuple(scores), best_kappa=best)


def train_population_valence(train: pl.DataFrame) -> tuple[float, float]:
    """(vbar, tau) from the TRAIN class marginal — fixed population constants."""
    y = train["y"].to_numpy().astype(np.int64)
    freq = np.bincount(y, minlength=K).astype(np.float64)
    freq /= freq.sum()
    return population_valence(freq)
