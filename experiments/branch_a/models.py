"""Branch A predictors as logit augmentations of frozen B3 (phase A1).

    M_state  = B3 unchanged (loaded from the M2 artifact cache; never refit).
    M_flat   = B3 logits + UNSHRUNK striker/bowler offsets (lambda -> 0).
    M_shrunk = B3 logits + shrunk striker/bowler offsets (lambda tuned on val).

LOGGED ASSUMPTIONS (minimal, per protocol):
- "M_flat = state + one-hot striker + one-hot bowler (multinomial logistic)"
  is implemented as the lambda->0 limit of the same augmentation (per-player
  empirical log-odds tables, epsilon-floored at LAMBDA_FLAT pseudo-balls for
  finite logs). Under the spec's umbrella — "three predictors as logit
  augmentations of frozen B3" — this is the unshrunk MLE of the identical
  model family, which is exactly what makes the M_flat-vs-M_shrunk gap a
  clean measurement of shrinkage value.
- One shared lambda for striker and bowler effects ("lambda is the only
  tunable").
- Every model (including M_state) gets M2's calibration convention:
  temperature scaling fit on val — numbers stay directly comparable to the
  M2 leaderboard, and the val lambda curve is scored post-temperature, i.e.
  under the exact final-evaluation protocol.

The test split is not referenced anywhere in this module.
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
from experiments.branch_a.effects import PlayerEffects, fit_effects

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EPS = 1e-12
K = len(CLASSES)
LAMBDA_FLAT = 0.01  # pseudo-balls: numerically-floored "unshrunk" M_flat
LAMBDA_GRID = (25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)


def load_frozen_b3(task: str = "t1", fmt: str = "t20") -> Predictor:
    model = cache.load(task, fmt, "B3_gbm")
    if model is None:
        raise RuntimeError(
            "frozen B3 not in the artifact cache — run `uv run evalkit run-all` first; "
            "Branch A never refits it"
        )
    return model


def assemble_with_ids(deliveries: pl.DataFrame, fmt: str, split: str) -> pl.DataFrame:
    """Frozen-harness T1 assembly + the two Branch A identity columns.

    Row alignment is guaranteed: the id columns come from the identical
    filter expression the frozen assembler applies.
    """
    rows = deliveries.filter(
        (pl.col("fmt") == fmt)
        & (pl.col("temporal_split") == split)
        & ~pl.col("excluded_from_tuples")
    )
    base = assemble_t1(deliveries, fmt, split)
    assert base.height == rows.height  # same frozen filter, same order
    return base.with_columns(
        rows["striker_id"].alias("striker_id"), rows["bowler_id"].alias("bowler_id")
    )


@dataclass(frozen=True)
class IdentityEffects:
    striker: PlayerEffects
    bowler: PlayerEffects

    def augment(self, base_probs: FloatArray, df: pl.DataFrame) -> FloatArray:
        """softmax(log base + striker offset + bowler offset)."""
        logits = (
            np.log(np.clip(base_probs, EPS, None))
            + self.striker.offset_matrix(df["striker_id"])
            + self.bowler.offset_matrix(df["bowler_id"])
        )
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        result: FloatArray = e / e.sum(axis=1, keepdims=True)
        return result


def fit_identity_effects(train: pl.DataFrame, lam: float) -> IdentityEffects:
    """TRAIN fold only; frozen thereafter."""
    y = train["y"].to_numpy().astype(np.int64)
    return IdentityEffects(
        striker=fit_effects(train["striker_id"], y, lam=lam, n_classes=K),
        bowler=fit_effects(train["bowler_id"], y, lam=lam, n_classes=K),
    )


@dataclass(frozen=True)
class CalibratedAugmented:
    """A Branch A predictor: frozen B3 + (optional) effects + temperature."""

    name: str
    effects: IdentityEffects | None
    scaler: TemperatureScaler

    def predict(self, base_probs: FloatArray, df: pl.DataFrame) -> FloatArray:
        p = base_probs if self.effects is None else self.effects.augment(base_probs, df)
        return self.scaler.apply(np.log(np.clip(p, EPS, None)))


def calibrated(
    name: str,
    effects: IdentityEffects | None,
    base_val_probs: FloatArray,
    val: pl.DataFrame,
) -> CalibratedAugmented:
    """Temperature fit on val (M2 convention), applied to the augmented probs."""
    p_val = base_val_probs if effects is None else effects.augment(base_val_probs, val)
    y_val = val["y"].to_numpy().astype(np.int64)
    scaler = fit_temperature(np.log(np.clip(p_val, EPS, None)), y_val)
    return CalibratedAugmented(name=name, effects=effects, scaler=scaler)


@dataclass(frozen=True)
class LambdaCurve:
    lam_grid: tuple[float, ...]
    val_nll: tuple[float, ...]  # post-temperature, the final-eval protocol
    best_lam: float

    def is_unimodal(self) -> bool:
        """Decreases to the minimum, then increases (non-strict)."""
        v = np.array(self.val_nll)
        i = int(np.argmin(v))
        return bool(np.all(np.diff(v[: i + 1]) <= 1e-12) and np.all(np.diff(v[i:]) >= -1e-12))


def tune_lambda(
    train: pl.DataFrame,
    val: pl.DataFrame,
    base_val_probs: FloatArray,
    grid: tuple[float, ...] = LAMBDA_GRID,
) -> tuple[IdentityEffects, LambdaCurve]:
    """Val-NLL-only tuning of the single lambda. Test never enters."""
    y_val = val["y"].to_numpy().astype(np.int64)
    scores: list[float] = []
    fitted: list[IdentityEffects] = []
    for lam in grid:
        effects = fit_identity_effects(train, lam)
        model = calibrated(f"M_shrunk[{lam:g}]", effects, base_val_probs, val)
        scores.append(nll_multiclass(model.predict(base_val_probs, val), y_val))
        fitted.append(effects)
    best_idx = int(np.argmin(scores))
    curve = LambdaCurve(lam_grid=tuple(grid), val_nll=tuple(scores), best_lam=grid[best_idx])
    return fitted[best_idx], curve
