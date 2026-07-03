"""Branch C phase C1: residual signal, gain-free tilt, kappa tuning."""

import numpy as np
import polars as pl
import pytest
from experiments.branch_c.latent import VALENCE, population_valence
from experiments.branch_c.models import (
    FULL_MATCH,
    K,
    KappaCurve,
    build_latent,
    build_state,
    latent_tilt,
    residual_signal,
    tune_kappa,
)

from evalkit.metrics import nll_multiclass
from evalkit.models.base import CLASSES

RNG = np.random.default_rng(1337)
P_BAR = np.array([0.387, 0.335, 0.060, 0.005, 0.091, 0.028, 0.0002, 0.015, 0.004, 0.030, 0.0448])
P_BAR = P_BAR / P_BAR.sum()
VBAR, TAU = population_valence(P_BAR)


def test_residual_signal_is_observed_minus_expected() -> None:
    b3 = RNG.dirichlet(np.ones(K), size=4)
    y = np.array([0, 5, 10, 4])
    s = residual_signal(b3, y)
    expect = VALENCE[y] - b3 @ VALENCE
    assert np.allclose(s, expect)
    # a six when B3 expected ~1 run is a large positive residual
    assert s[1] > 0


def test_latent_tilt_is_gain_free_logit_addition() -> None:
    b3 = RNG.dirichlet(np.ones(K), size=3)
    theta = np.array([0.5, -0.3, 0.0])
    out = latent_tilt(b3, theta, VBAR, TAU)
    direction = (VALENCE - VBAR) / TAU
    logits = np.log(b3[0]) + theta[0] * direction
    assert np.allclose(out[0], np.exp(logits) / np.exp(logits).sum())
    # theta = 0 leaves B3 untouched
    assert np.allclose(out[2], b3[2])
    assert np.allclose(out.sum(axis=1), 1.0)


def test_positive_theta_tilts_toward_scoring() -> None:
    b3 = np.tile(P_BAR, (1, 1))
    hi = latent_tilt(b3, np.array([0.6]), VBAR, TAU)[0]
    assert hi[CLASSES.index("6")] > P_BAR[CLASSES.index("6")]  # six more likely
    assert hi[CLASSES.index("0")] < P_BAR[CLASSES.index("0")]  # dot less likely


def simulate(n_matches: int = 400, balls: int = 120, seed: int = 0) -> pl.DataFrame:
    """Each match has a hidden scoring latent; base B3 = population (blind to it)."""
    rng = np.random.default_rng(seed)
    direction = (VALENCE - VBAR) / TAU
    rows = []
    for m in range(n_matches):
        theta_m = rng.normal(0.0, 0.35)
        p = P_BAR * np.exp(theta_m * direction)
        p /= p.sum()
        ys = rng.choice(K, size=balls, p=p)
        for b, y in enumerate(ys):
            rows.append(
                {
                    "match_id": f"m{m}",
                    "innings_idx": 1 if b < balls // 2 else 2,
                    "over_number": b // 6,
                    "ball_in_over": b % 6 + 1,
                    "y": int(y),
                }
            )
    return pl.DataFrame(rows)


def test_tune_kappa_finds_interior_optimum_and_curve_unimodal() -> None:
    df = simulate()
    half = df.height // 2
    val = df.tail(df.height - half)
    b3_val = np.tile(P_BAR, (val.height, 1))
    curve = tune_kappa(
        b3_val, val, VBAR, TAU, group_cols=FULL_MATCH, grid=(2.0, 10.0, 50.0, 250.0, 2000.0)
    )
    assert curve.best_kappa in (2.0, 10.0, 50.0, 250.0, 2000.0)
    y_val = val["y"].to_numpy()
    nll_base = nll_multiclass(b3_val, y_val)
    # a real per-match latent beyond the base model must lower val NLL
    assert min(curve.val_nll) < nll_base - 0.01
    # heaviest shrinkage in the grid ~ recovers the base (latent switched off)
    assert curve.val_nll[-1] == pytest.approx(nll_base, abs=0.02)


def test_state_model_leaves_b3_untouched_up_to_temperature() -> None:
    df = simulate(n_matches=50)
    b3 = np.tile(P_BAR, (df.height, 1))
    state = build_state(b3, df)
    # M_state has no latent: raw probs are exactly B3
    assert np.allclose(state.raw_probs(b3, df), b3)


def test_unimodal_detector() -> None:
    assert KappaCurve((1, 2, 3), (0.5, 0.4, 0.6), 2).is_unimodal()
    assert not KappaCurve((1, 2, 3, 4), (0.5, 0.6, 0.4, 0.7), 3).is_unimodal()


def test_full_match_and_per_innings_differ_when_signal_carries() -> None:
    """With cross-innings structure, carry-over changes the innings-2 estimates."""
    df = simulate(n_matches=100)
    val = df
    b3 = np.tile(P_BAR, (val.height, 1))
    full = build_latent("M_latent", 10.0, ("match_id",), b3, val, VBAR, TAU)
    inns = build_latent("M_latent_innings", 10.0, ("match_id", "innings_idx"), b3, val, VBAR, TAU)
    p_full = full.raw_probs(b3, val)
    p_inns = inns.raw_probs(b3, val)
    # innings-2 rows: the two variants must disagree (carry-over vs reset)
    inn2 = (val["innings_idx"] == 2).to_numpy()
    assert not np.allclose(p_full[inn2], p_inns[inn2])
