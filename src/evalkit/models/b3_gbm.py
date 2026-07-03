"""B3 — Gradient-boosted trees via sklearn HistGradientBoostingClassifier
(SPEC_M2 §5). No lightgbm/xgboost/torch.

Early stopping is on OUR val split (not sklearn's internal random fraction):
warm-start growth in fixed steps, stop when val NLL hasn't improved for
`patience` rounds, then keep the best iteration count. Fixed seed, fixed
sensible defaults.
"""

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

from evalkit.metrics import nll_binary, nll_multiclass
from evalkit.models.base import SEED, to_x, to_y

STEP = 25
MAX_ROUNDS = 500
PATIENCE = 3


class B3Gbm:
    version = "1.0"

    def __init__(self, task: str):
        self.name = "B3_gbm"
        self.task = task
        self.n_iter: int = STEP
        self.model: HistGradientBoostingClassifier | None = None

    def _make(self, max_iter: int, warm: bool) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=max_iter,
            warm_start=warm,
            early_stopping=False,  # our early stopping runs on OUR val split
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=SEED,
        )

    def fit(self, train: pl.DataFrame, val: pl.DataFrame) -> None:
        x_train, y_train = to_x(train), to_y(train)
        x_val, y_val = to_x(val), to_y(val)
        model = self._make(STEP, warm=True)
        best_score, best_iter, since_best = np.inf, STEP, 0
        for rounds in range(STEP, MAX_ROUNDS + 1, STEP):
            model.set_params(max_iter=rounds)
            model.fit(x_train, y_train)
            score = self._nll(model, x_val, y_val)
            if score < best_score - 1e-6:
                best_score, best_iter, since_best = score, rounds, 0
            else:
                since_best += 1
                if since_best >= PATIENCE:
                    break
        self.n_iter = best_iter
        final = self._make(best_iter, warm=False)
        final.fit(x_train, y_train)
        self.model = final

    def _nll(
        self,
        model: HistGradientBoostingClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> float:
        probs = model.predict_proba(x)
        if self.task == "t2":
            return nll_binary(probs[:, 1], y)
        return nll_multiclass(self._expand(probs, model), y)

    @staticmethod
    def _expand(
        probs: NDArray[np.float64], model: HistGradientBoostingClassifier
    ) -> NDArray[np.float64]:
        from evalkit.models.base import CLASSES

        seen = model.classes_.astype(int)
        out = np.full((probs.shape[0], len(CLASSES)), 1e-12)
        out[:, seen] = probs
        result: NDArray[np.float64] = out / out.sum(axis=1, keepdims=True)
        return result

    def predict_proba(self, df: pl.DataFrame) -> NDArray[np.float64]:
        assert self.model is not None, "fit first"
        probs = self.model.predict_proba(to_x(df))
        if self.task == "t2":
            result: NDArray[np.float64] = probs[:, 1]
            return result
        return self._expand(probs, self.model)
