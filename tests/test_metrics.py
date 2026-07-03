"""Known-answer metric tests (SPEC_M2 §12.1) + supporting properties."""

import math

import numpy as np
import pytest

from evalkit.metrics import (
    brier_binary,
    brier_multiclass,
    ece,
    ece_per_class,
    max_ce,
    murphy_decomposition,
    nll_binary,
    nll_multiclass,
    reliability_data,
    skill_score,
)


def test_known_answer_brier() -> None:
    """Brier(p=0.7, y=1) = 0.09."""
    assert brier_binary(np.array([0.7]), np.array([1])) == pytest.approx(0.09, abs=1e-12)


def test_known_answer_nll_half() -> None:
    """NLL(p=0.5) = ln 2 = 0.693147…"""
    assert nll_binary(np.array([0.5]), np.array([1])) == pytest.approx(math.log(2), abs=1e-12)
    probs = np.array([[0.5, 0.5]])
    assert nll_multiclass(probs, np.array([0])) == pytest.approx(math.log(2), abs=1e-12)


def test_known_answer_calibrated_synthetic_ece() -> None:
    """ECE of a synthetically calibrated set ≤ 0.005."""
    rng = np.random.default_rng(7)
    n = 400_000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(np.int64)
    assert ece(p, y, n_bins=20) <= 0.005


def test_known_answer_murphy_identity() -> None:
    """BS_binned = REL - RES + UNC to 1e-9, on rough (miscalibrated) predictions."""
    rng = np.random.default_rng(11)
    n = 50_000
    p = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
    y = (rng.uniform(0, 1, n) < np.clip(p * 1.2 - 0.05, 0, 1)).astype(np.int64)
    d = murphy_decomposition(p, y, n_bins=20)
    assert d.bs_binned == pytest.approx(d.rel - d.res + d.unc, abs=1e-9)
    assert d.unc == pytest.approx(float(np.mean(y)) * (1 - float(np.mean(y))), abs=1e-12)
    assert d.rel >= 0 and d.res >= 0


def test_multiclass_brier_range_and_known_case() -> None:
    # perfect prediction → 0; uniform over 2 → 0.5; total miss → 2
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    y = np.array([0, 0, 0])
    per_row = [brier_multiclass(probs[i : i + 1], y[i : i + 1]) for i in range(3)]
    assert per_row == [pytest.approx(0.0), pytest.approx(0.5), pytest.approx(2.0)]


def test_ece_detects_gross_miscalibration() -> None:
    n = 10_000
    p = np.full(n, 0.9)
    y = np.zeros(n, dtype=np.int64)
    y[: n // 2] = 1  # true rate 0.5, predicted 0.9
    assert ece(p, y) == pytest.approx(0.4, abs=1e-9)
    assert max_ce(p, y) == pytest.approx(0.4, abs=1e-9)


def test_ece_per_class_shape() -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(4), size=1000)
    y = rng.integers(0, 4, size=1000)
    per_class = ece_per_class(probs, y)
    assert per_class.shape == (4,)
    assert (per_class >= 0).all() and (per_class <= 1).all()


def test_reliability_data_masses_sum_to_one() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(0, 1, 1234)
    y = (rng.uniform(0, 1, 1234) < p).astype(np.int64)
    r = reliability_data(p, y)
    assert float(np.sum(r.weight)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(r.p_mean) >= 0)  # equal-mass bins are sorted by p


def test_skill_score() -> None:
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert skill_score(1.2, 1.0) == pytest.approx(-0.2)
