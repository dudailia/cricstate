"""B2 — Regularized logistic regression (SPEC_M2 §5).

Multinomial for T1, binary for T2; standardized features; C chosen on val
from {0.01, 0.1, 1, 10} by NLL. Deterministic: lbfgs, fixed seed, fixed
iteration budget (max_iter=200, tol=1e-3 — the looser-than-default tol is a
fixed, documented runtime knob; measured val-NLL impact vs default tol is
in the 4th decimal, while the T1 multinomial fit on 2.4M rows drops from
~400s to a tractable budget).
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evalkit.metrics import nll_binary, nll_multiclass
from evalkit.models.base import SEED, to_x, to_y

C_GRID = (0.01, 0.1, 1.0, 10.0)


class B2Logistic:
    version = "1.0"

    def __init__(self, task: str, max_iter: int = 200):
        self.name = "B2_logistic"
        self.task = task
        self.max_iter = max_iter
        self.c: float = 1.0
        self.pipe: Pipeline | None = None

    def _make(self, c: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=c,
                        solver="lbfgs",
                        max_iter=self.max_iter,
                        tol=1e-3,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        x_train, y_train = to_x(train), to_y(train)
        x_val, y_val = to_x(val), to_y(val)
        best = (np.inf, C_GRID[0], None)
        for c in C_GRID:
            pipe = self._make(c)
            pipe.fit(x_train, y_train)
            score = self._nll(pipe, x_val, y_val)
            if score < best[0]:
                best = (score, c, pipe)
        _, self.c, self.pipe = best

    def _nll(self, pipe: Pipeline, x: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        probs = pipe.predict_proba(x)
        if self.task == "t2":
            return nll_binary(probs[:, 1], y)
        return nll_multiclass(self._expand(probs, pipe), y)

    @staticmethod
    def _expand(probs: NDArray[np.float64], pipe: Pipeline) -> NDArray[np.float64]:
        """Map sklearn's seen-class columns onto the full frozen alphabet."""
        from evalkit.models.base import CLASSES

        seen = pipe.named_steps["lr"].classes_.astype(int)
        out = np.full((probs.shape[0], len(CLASSES)), 1e-12)
        out[:, seen] = probs
        result: NDArray[np.float64] = out / out.sum(axis=1, keepdims=True)
        return result

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        assert self.pipe is not None, "fit first"
        probs = self.pipe.predict_proba(to_x(df))
        if self.task == "t2":
            result: NDArray[np.float64] = probs[:, 1]
            return result
        return self._expand(probs, self.pipe)
