"""Predictor protocol + registry (SPEC_M2 §8) and shared model plumbing.

Frames passed to Predictor.fit/predict_proba are assembly frames: exactly
FEATURE_COLUMNS (+ `fmt`) from the whitelisted builder, plus `y` (the label)
and `match_id` (bookkeeping for match-level bootstrap — models must never
read it; `to_xy` structurally prevents that by selecting FEATURE_COLUMNS).
"""

from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit.features import FEATURE_COLUMNS

SEED = 1337

# Frozen T1 class order (the K=11 alphabet from the M1 freeze; enumeration
# order is part of the freeze — runs ascending, then extras, then wicket).
CLASSES: tuple[str, ...] = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "6",
    "other_runs",
    "bye_legbye",
    "no_ball",
    "wide",
    "wicket",
)
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


@runtime_checkable
class Predictor(Protocol):
    name: str
    version: str

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None: ...

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]: ...


# --- reserved seams (SPEC_M2 §13): named, empty, no implementation ----------


class OddsProvider(Protocol):
    """Reserved for Module 4+. No implementation in M2."""


class AsOfFeatureStore(Protocol):
    """Reserved for Module 3+. No implementation in M2."""


# --- registry -----------------------------------------------------------------

_REGISTRY: dict[tuple[str, str, str], Predictor] = {}


def register(task: str, fmt: str, model: Predictor) -> None:
    key = (task, fmt, model.name)
    if key in _REGISTRY:
        raise ValueError(f"duplicate registration: {key}")
    _REGISTRY[key] = model


def get(task: str, fmt: str, name: str) -> Predictor:
    return _REGISTRY[(task, fmt, name)]


def registered(task: str | None = None, fmt: str | None = None) -> list[tuple[str, str, str]]:
    return sorted(
        k for k in _REGISTRY if (task is None or k[0] == task) and (fmt is None or k[1] == fmt)
    )


def clear_registry() -> None:
    _REGISTRY.clear()


# --- shared plumbing ------------------------------------------------------------


def to_x(df: pl.DataFrame) -> NDArray[np.float64]:
    """Feature matrix — structurally restricted to FEATURE_COLUMNS."""
    return df.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)


def to_y(df: pl.DataFrame) -> NDArray[np.int64]:
    return df["y"].to_numpy().astype(np.int64)
