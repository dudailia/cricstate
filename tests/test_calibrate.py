import inspect

import numpy as np
import pytest

from evalkit.calibrate import (
    _softmax,
    fit_isotonic,
    fit_platt,
    fit_temperature,
)
from evalkit.metrics import FloatArray, IntArray, ece, nll_multiclass


def overconfident_logprobs(
    n: int = 20_000, k: int = 4, sharpen: float = 3.0
) -> tuple[FloatArray, IntArray]:
    """True probs q; model reports q sharpened by 1/T — temperature must recover T."""
    rng = np.random.default_rng(42)
    q = rng.dirichlet(np.ones(k), size=n)
    y = np.array([rng.choice(k, p=qi) for qi in q])
    logprobs = np.log(np.clip(q, 1e-12, None)) * sharpen  # overconfident by 3x
    return logprobs, y


def test_temperature_recovers_sharpening_and_improves_nll() -> None:
    logprobs, y = overconfident_logprobs()
    scaler = fit_temperature(logprobs, y)
    assert scaler.temperature == pytest.approx(3.0, rel=0.1)
    before = nll_multiclass(_softmax(logprobs), y)
    after = nll_multiclass(scaler.apply(logprobs), y)
    assert after < before


def test_temperature_identity_when_calibrated() -> None:
    logprobs, y = overconfident_logprobs(sharpen=1.0)
    scaler = fit_temperature(logprobs, y)
    assert scaler.temperature == pytest.approx(1.0, abs=0.05)


def binary_overconfident(n: int = 30_000) -> tuple[FloatArray, IntArray]:
    rng = np.random.default_rng(17)
    q = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < q).astype(np.int64)
    z = np.log(q / (1 - q)) * 2.5  # overconfident scores
    p = 1 / (1 + np.exp(-z))
    return p, y


def test_platt_improves_ece() -> None:
    p, y = binary_overconfident()
    scaler = fit_platt(p, y)
    assert ece(scaler.apply(p), y) < ece(p, y)
    assert scaler.a == pytest.approx(1 / 2.5, rel=0.15)


def test_isotonic_improves_ece_and_is_monotone() -> None:
    p, y = binary_overconfident()
    scaler = fit_isotonic(p, y)
    out = scaler.apply(p)
    assert ece(out, y) < ece(p, y)
    order = np.argsort(p)
    assert np.all(np.diff(out[order]) >= -1e-12)
    assert out.min() > 0.0 and out.max() < 1.0  # clipped away from {0, 1}


def test_fitting_signatures_are_val_only() -> None:
    """SPEC discipline is structural: every data parameter is named val_*."""
    for fn in (fit_temperature, fit_platt, fit_isotonic):
        params = list(inspect.signature(fn).parameters)
        assert params and all(p.startswith("val_") for p in params), fn.__name__


def test_fits_are_deterministic() -> None:
    logprobs, y = overconfident_logprobs(n=5_000)
    assert fit_temperature(logprobs, y) == fit_temperature(logprobs, y)
    p, yb = binary_overconfident(n=5_000)
    s1, s2 = fit_platt(p, yb), fit_platt(p, yb)
    assert (s1.a, s1.b) == (s2.a, s2.b)
    i1, i2 = fit_isotonic(p, yb), fit_isotonic(p, yb)
    assert np.array_equal(i1.apply(p), i2.apply(p))
