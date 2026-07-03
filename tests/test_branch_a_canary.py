"""Branch A phase A2: shuffled-identity canary."""

import numpy as np
from experiments.branch_a.canary import EPS_CANARY, CanaryResult, shuffled_identity_effects
from experiments.branch_a.models import fit_identity_effects

from evalkit.metrics import nll_multiclass
from tests.test_branch_a_models import synthetic_split


def test_shuffle_destroys_a_real_signal() -> None:
    train, val, base_val = synthetic_split(n=16_000)
    y_val = val["y"].to_numpy()
    lam = 100.0
    real = fit_identity_effects(train, lam)
    shuffled = shuffled_identity_effects(train, lam)
    nll_base = nll_multiclass(base_val, y_val)
    nll_real = nll_multiclass(real.augment(base_val, val), y_val)
    nll_shuf = nll_multiclass(shuffled.augment(base_val, val), y_val)
    assert nll_real < nll_base - 0.02  # true identities carry signal here
    assert abs(nll_shuf - nll_base) <= EPS_CANARY  # shuffled ones must not


def test_shuffle_preserves_id_multiset_and_is_deterministic() -> None:
    train, _, _ = synthetic_split(n=4_000)
    a = shuffled_identity_effects(train, 100.0)
    b = shuffled_identity_effects(train, 100.0)
    assert set(a.striker.offsets) == set(
        pid for pid in train["striker_id"].unique().to_list() if pid is not None
    )
    for pid in a.striker.offsets:
        assert np.array_equal(a.striker.offsets[pid], b.striker.offsets[pid])


def test_canary_classification() -> None:
    ok = CanaryResult(nll_shuffled=1.6251, nll_state=1.6260, fold="val")
    assert ok.passed and not ok.leaks and "PASS" in ok.line()
    leak = CanaryResult(nll_shuffled=1.6100, nll_state=1.6260, fold="test")
    assert not leak.passed and leak.leaks and "VOID" in leak.line()
    worse = CanaryResult(nll_shuffled=1.6400, nll_state=1.6260, fold="val")
    assert not worse.passed and not worse.leaks and worse.line().endswith("FAIL")
