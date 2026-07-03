"""Causal per-match "conditions" latent filter (Branch C, phase C0).

theta_hat(m, t) is a conjugate-Bayesian estimate of a scalar per-match latent,
seeded at the population prior and updated ball-by-ball over the outcomes of
balls STRICTLY BEFORE t in the SAME match m. It is applied downstream (C1) as
a logit shift on B3's per-ball distribution. Nothing here uses ball t or later.

Conjugate model (Gaussian-Gaussian):
    prior   theta ~ N(0, prior_var)
    obs     s_i   ~ N(theta, obs_var)      (s_i = a scalar signal from ball i)
    theta_hat(m,t) = posterior mean after balls 1..t-1   (predict step)

LOGGED ASSUMPTIONS (minimal, per protocol):
- The latent is a per-MATCH property (pitch/conditions) that persists across
  both innings: it accumulates over the full match in delivery order
  (innings_idx, over_number, ball_in_over), not reset per innings.
- Process noise Q = 0 (conditions ~constant within a match). With Q = 0 the
  Kalman filter reduces EXACTLY to a shrunk running mean with a single tunable
  kappa = obs_var / prior_var (prior strength in pseudo-balls); kappa is the
  primary val-tuned knob. The general Kalman form is kept as the reference the
  vectorized path is tested against.
- Valence vector (runs typically scored on a ball of each class) gives the
  scalar signal; it is fixed, listed in VALENCE, and reported.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit.models.base import CLASSES

FloatArray = NDArray[np.float64]

# Runs typically scored on a ball of each class, in the frozen CLASSES order
# ("0","1","2","3","4","6","other_runs","bye_legbye","no_ball","wide","wicket").
# other_runs ~ overthrow 5s; extras ~ 1 run; a wicket ball typically scores 0.
VALENCE: FloatArray = np.array(
    [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64
)
assert VALENCE.shape == (len(CLASSES),)

ORDER_COLS = ("innings_idx", "over_number", "ball_in_over")


def population_valence(class_freq: FloatArray) -> tuple[float, float]:
    """(mean, variance) of valence under a class-probability vector.

    class_freq: length-K probabilities (e.g. train marginal). The variance is
    the fixed scale tau used to convert the runs-unit latent into a logit tilt.
    """
    p = np.asarray(class_freq, dtype=np.float64)
    mean = float(np.sum(p * VALENCE))
    var = float(np.sum(p * VALENCE**2) - mean**2)
    return mean, var


@dataclass(frozen=True)
class FilterParams:
    """kappa is the operative tunable; prior_var/obs_var/process_var back the
    reference Kalman. With process_var == 0, kappa == obs_var / prior_var."""

    kappa: float
    prior_var: float = 1.0
    obs_var: float = 1.0
    process_var: float = 0.0


def theta_path(signals: FloatArray, kappa: float) -> FloatArray:
    """Vectorized causal shrunk running mean (Q = 0 conjugate reduction).

    theta_hat[i] = sum(signals[:i]) / (i + kappa)  — uses only balls before i.
    theta_hat[0] = 0 (the prior mean): the warm-up value is the prior.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive (prior strength in pseudo-balls)")
    s = np.asarray(signals, dtype=np.float64)
    n = len(s)
    prior_sum = np.concatenate([[0.0], np.cumsum(s)[:-1]])  # sum strictly before i
    n_prior = np.arange(n, dtype=np.float64)  # count strictly before i
    return prior_sum / (n_prior + kappa)


def theta_path_kalman(signals: FloatArray, params: FilterParams) -> FloatArray:
    """Reference 1-D Kalman local-level filter (predict-then-update).

    theta_hat[i] is the prior mean for ball i (posterior after balls < i), so
    the estimate for ball i never sees signal i. Equals theta_path() when
    process_var == 0 and prior_var/obs_var give the same kappa.
    """
    s = np.asarray(signals, dtype=np.float64)
    m = 0.0
    p = params.prior_var
    out = np.empty(len(s))
    for i in range(len(s)):
        out[i] = m  # predict: estimate for ball i uses only balls < i
        gain = p / (p + params.obs_var)
        m = m + gain * (s[i] - m)
        p = (1.0 - gain) * p
        p = p + params.process_var  # process/decay for the next ball
    return out


def causal_theta(
    df: pl.DataFrame,
    *,
    signal_col: str,
    kappa: float,
    group_cols: tuple[str, ...] = ("match_id",),
) -> FloatArray:
    """theta_hat per row, accumulating within each group in delivery order.

    Requires columns: match_id, ORDER_COLS, signal_col, and group_cols. The
    accumulation resets at each group boundary; group_cols=("match_id",) is the
    full-match latent (carry-over across innings), ("match_id","innings_idx")
    is the per-innings variant (no carry-over). Returns an array aligned to
    df's current row order (the frame is sorted internally, then restored).
    """
    indexed = df.with_row_index("_row")
    ordered = indexed.sort(["match_id", *ORDER_COLS])
    prior_sum = pl.col(signal_col).cum_sum().shift(1, fill_value=0.0).over(list(group_cols))
    n_prior = pl.int_range(0, pl.len()).over(list(group_cols)).cast(pl.Float64)
    theta = ordered.with_columns((prior_sum / (n_prior + kappa)).alias("_theta")).sort("_row")
    return theta["_theta"].to_numpy().astype(np.float64)
