"""Branch C phase C0: causal latent filter — ordering proof + recovery."""

import numpy as np
import polars as pl
import pytest
from experiments.branch_c.latent import (
    VALENCE,
    FilterParams,
    causal_theta,
    population_valence,
    theta_path,
    theta_path_kalman,
)

from evalkit.models.base import CLASSES

RNG = np.random.default_rng(1337)


def test_valence_covers_frozen_alphabet() -> None:
    assert VALENCE.shape == (len(CLASSES),)
    # dot and wicket score nothing; a six scores six
    assert VALENCE[CLASSES.index("0")] == 0.0
    assert VALENCE[CLASSES.index("wicket")] == 0.0
    assert VALENCE[CLASSES.index("6")] == 6.0


def test_warmup_is_the_prior() -> None:
    s = RNG.normal(size=50)
    theta = theta_path(s, kappa=10.0)
    assert theta[0] == 0.0  # first ball sees no evidence -> prior mean
    # early estimates are heavily shrunk toward the prior
    assert abs(theta[1]) < abs(s[0])


def test_causal_ordering_permutation_proof() -> None:
    """THE causal test: permuting balls at/after t cannot change theta_hat(t)."""
    n = 200
    s = RNG.normal(loc=0.4, scale=1.0, size=n)
    base = theta_path(s, kappa=8.0)
    for t in (1, 5, 37, 199):
        perm = s.copy()
        tail = perm[t:].copy()
        RNG.shuffle(tail)
        perm[t:] = tail
        moved = theta_path(perm, kappa=8.0)
        assert moved[t] == pytest.approx(base[t])  # depends only on s[:t]
        assert np.allclose(moved[: t + 1], base[: t + 1])


def test_recovers_injected_latent() -> None:
    """With many balls and light shrinkage, theta_hat -> the true latent."""
    theta_true = 0.8
    s = RNG.normal(loc=theta_true, scale=1.0, size=20_000)
    theta = theta_path(s, kappa=5.0)
    assert theta[-1] == pytest.approx(theta_true, abs=0.05)
    # and it is still exactly the prior at the start
    assert theta[0] == 0.0


def test_kappa_extremes() -> None:
    s = RNG.normal(loc=1.0, scale=0.5, size=500)
    hard = theta_path(s, kappa=1e9)  # infinite prior strength -> stuck at prior
    assert np.max(np.abs(hard)) < 1e-3
    soft = theta_path(s, kappa=1e-6)  # no prior -> raw running mean of priors
    manual = np.concatenate([[0.0], np.cumsum(s)[:-1] / np.arange(1, len(s))])
    assert np.allclose(soft, manual, atol=1e-4)
    with pytest.raises(ValueError, match="positive"):
        theta_path(s, kappa=0.0)


def test_vectorized_matches_reference_kalman() -> None:
    s = RNG.normal(loc=0.3, scale=1.2, size=300)
    for kappa in (2.0, 10.0, 100.0):
        # process_var=0, prior_var=1 -> obs_var=kappa gives the same filter
        ref = theta_path_kalman(
            s, FilterParams(kappa=kappa, prior_var=1.0, obs_var=kappa, process_var=0.0)
        )
        assert np.allclose(theta_path(s, kappa), ref, atol=1e-9)


def test_population_valence_stats() -> None:
    freq = np.full(len(CLASSES), 1.0 / len(CLASSES))
    mean, var = population_valence(freq)
    assert mean == pytest.approx(float(np.mean(VALENCE)))
    assert var == pytest.approx(float(np.var(VALENCE)), abs=1e-9)


def _match_frame() -> pl.DataFrame:
    # two matches interleaved in a scrambled row order
    rows = []
    for mid, base in (("mA", 0.5), ("mB", -0.3)):
        for inn in (1, 2):
            for over in range(3):
                for ball in range(1, 7):
                    rows.append(
                        {
                            "match_id": mid,
                            "innings_idx": inn,
                            "over_number": over,
                            "ball_in_over": ball,
                            "s": base + RNG.normal(scale=0.1),
                        }
                    )
    return pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=7)


def test_causal_theta_resets_per_match_and_restores_order() -> None:
    df = _match_frame()
    theta = causal_theta(df, signal_col="s", kappa=4.0)
    assert len(theta) == df.height
    # recompute independently per match in delivery order, compare aligned
    got = df.with_columns(pl.Series("theta", theta))
    for mid in ("mA", "mB"):
        sub = got.filter(pl.col("match_id") == mid).sort(
            ["innings_idx", "over_number", "ball_in_over"]
        )
        expect = theta_path(sub["s"].to_numpy(), kappa=4.0)
        assert np.allclose(sub["theta"].to_numpy(), expect)
        # first ball of each match is the prior, regardless of interleaving
        assert sub["theta"].to_numpy()[0] == 0.0


def test_causal_theta_first_innings_does_feed_second() -> None:
    """Latent persists across innings within a match (per-match, not per-innings)."""
    df = _match_frame().filter(pl.col("match_id") == "mA")
    theta = causal_theta(df, signal_col="s", kappa=4.0)
    ordered = df.with_columns(pl.Series("theta", theta)).sort(
        ["innings_idx", "over_number", "ball_in_over"]
    )
    n_inn1 = ordered.filter(pl.col("innings_idx") == 1).height
    # the first innings-2 ball already carries all innings-1 evidence
    assert ordered["theta"].to_numpy()[n_inn1] != 0.0
