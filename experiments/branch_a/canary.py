"""Shuffled-identity canary (Branch A phase A2).

Replace player IDs with random IDs — implemented as independent fixed-seed
permutations of the striker and bowler columns across TRAIN rows, which
destroys the id-outcome linkage while preserving every player's ball count —
then refit M_shrunk at the operative lambda. Its NLL must land within
EPS_CANARY = 0.01 nats of M_state. If shuffled identity "helps", the pipeline
leaks: STOP, declare the result VOID (not KILL).

LOGGED ASSUMPTIONS (minimal):
- "random IDs" = within-train permutation (multiset of IDs preserved).
- The refit uses the lambda already frozen by A1's val tuning; the canary
  tests the pipeline, not the tuning loop.
- The spec pins the canary to TEST NLL but also mandates a single test touch
  with all models together: A2 therefore gates on a VAL preliminary, and the
  binding test-NLL canary is computed inside A3's single test pass, before
  the verdict is trusted.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from evalkit.models.base import SEED
from experiments.branch_a.models import IdentityEffects, fit_identity_effects

EPS_CANARY = 0.01


def shuffled_identity_effects(train: pl.DataFrame, lam: float) -> IdentityEffects:
    """Fit effects on identity-destroyed train (fixed-seed permutations)."""
    rng = np.random.default_rng(SEED)
    shuffled = train.with_columns(
        pl.Series("striker_id", train["striker_id"].gather(rng.permutation(train.height))),
        pl.Series("bowler_id", train["bowler_id"].gather(rng.permutation(train.height))),
    )
    return fit_identity_effects(shuffled, lam)


@dataclass(frozen=True)
class CanaryResult:
    nll_shuffled: float
    nll_state: float
    fold: str  # "val" (A2 preliminary) or "test" (binding, inside A3)

    @property
    def delta(self) -> float:
        return self.nll_shuffled - self.nll_state

    @property
    def passed(self) -> bool:
        return abs(self.delta) <= EPS_CANARY

    @property
    def leaks(self) -> bool:
        """Shuffled identity 'helping' beyond eps ⇒ pipeline leak ⇒ VOID."""
        return self.delta < -EPS_CANARY

    def line(self) -> str:
        status = "PASS" if self.passed else ("VOID (leak)" if self.leaks else "FAIL")
        return (
            f"shuffled-identity canary [{self.fold}]: shuffled {self.nll_shuffled:.5f} "
            f"vs M_state {self.nll_state:.5f} (delta {self.delta:+.5f}, "
            f"eps {EPS_CANARY}) -> {status}"
        )
