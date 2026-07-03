"""Leakage canaries (SPEC_M2 §7). All run in CI; failures are stop-the-line.

1. shuffled-target: B2 retrained on train with y shuffled must sit within
   EPSILON_SHUFFLE nats of B0 on val — anything better means the pipeline leaks.
2. poison-column: the feature builder ignores injected outcome columns
   (structural; also unit-tested in tests/test_features.py).
3. ladder-inversion (P3 form, on val): NLL(B2) > NLL(B0) + 0.001 or
   NLL(B3) > NLL(B2) + 0.005 → STOP and report; never tune to fix.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from evalkit.metrics import nll_binary, nll_multiclass
from evalkit.models.b2_logistic import B2Logistic
from evalkit.models.base import SEED, to_y

EPSILON_SHUFFLE = 0.01
EPSILON_B2_VS_B0 = 0.001
EPSILON_B3_VS_B2 = 0.005


@dataclass(frozen=True)
class CanaryResult:
    name: str
    passed: bool
    detail: str


def shuffled_target_canary(
    task: str,
    train: pl.DataFrame,
    val: pl.DataFrame,
    nll_b0_val: float,
    chosen_c: float,
) -> CanaryResult:
    """Retrain B2 on shuffled labels; val NLL must sit at B0 (no signal left)."""
    rng = np.random.default_rng(SEED)
    shuffled = train.with_columns(pl.Series("y", rng.permutation(train["y"].to_numpy())))
    model = B2Logistic(task)
    model.c = chosen_c
    pipe = model._make(chosen_c)
    from evalkit.models.base import to_x

    pipe.fit(to_x(shuffled), to_y(shuffled))
    model.pipe = pipe
    probs = model.predict_proba(val)
    y_val = to_y(val)
    nll = nll_binary(probs, y_val) if task == "t2" else nll_multiclass(probs, y_val)
    delta = nll - nll_b0_val
    return CanaryResult(
        name=f"shuffled_target[{task}]",
        passed=abs(delta) <= EPSILON_SHUFFLE,
        detail=(
            f"shuffled-B2 val NLL {nll:.6f} vs B0 {nll_b0_val:.6f} "
            f"(delta {delta:+.6f}, eps {EPSILON_SHUFFLE})"
        ),
    )


def ladder_inversion_canary(
    nll_b0: float, nll_b2: float, nll_b3: float, where: str
) -> CanaryResult:
    b2_ok = nll_b2 <= nll_b0 + EPSILON_B2_VS_B0
    b3_ok = nll_b3 <= nll_b2 + EPSILON_B3_VS_B2
    return CanaryResult(
        name=f"ladder_inversion[{where}]",
        passed=b2_ok and b3_ok,
        detail=(f"B0 {nll_b0:.6f}, B2 {nll_b2:.6f} (ok={b2_ok}), B3 {nll_b3:.6f} (ok={b3_ok})"),
    )
