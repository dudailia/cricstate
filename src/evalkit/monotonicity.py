"""B1 (T2) monotonicity checks (SPEC_M2 §5, as amended #2). Failures fail
the build — and are never 'fixed' by tuning.

Directions (p̂ = P(first-batting side wins)):
- innings 1, holding (phase, wickets) fixed: non-decreasing in CRR band;
  holding (phase, CRR band) fixed: non-increasing in wickets band.
- innings 2, holding (phase, wickets) fixed: non-decreasing in RRR band;
  holding (phase, RRR band) fixed: non-decreasing in chaser-wickets band.
"""

import numpy as np

from evalkit.models.b1_table import B1TableT2

N_PHASE, N_WKTS, N_RATE = 4, 4, 6
TOL = 1e-9


def _grid(model: B1TableT2) -> np.ndarray:
    """p̂ over the full (innings, phase, wickets, rate) lattice — exactly the
    grid predictions index into (amendment #3: smoothed by construction)."""
    return model.lattice()


def check_b1_t2_monotonicity(model: B1TableT2) -> list[str]:
    """Return a list of violation descriptions (empty = pass)."""
    g = _grid(model)
    violations: list[str] = []
    for innings in (0, 1):
        for p in range(N_PHASE):
            # rate direction: non-decreasing in CRR (inn 1) / RRR (inn 2)
            for w in range(N_WKTS):
                series = g[innings, p, w, :]
                if np.any(np.diff(series) < -TOL):
                    violations.append(
                        f"innings{innings + 1} phase{p} wkts{w}: not non-decreasing in rate band "
                        f"({np.round(series, 4).tolist()})"
                    )
            # wickets direction: innings 1 non-increasing, innings 2 non-decreasing
            for r in range(N_RATE):
                series = g[innings, p, :, r]
                diffs = np.diff(series)
                bad = np.any(diffs > TOL) if innings == 0 else np.any(diffs < -TOL)
                if bad:
                    direction = "non-increasing" if innings == 0 else "non-decreasing"
                    violations.append(
                        f"innings{innings + 1} phase{p} rate{r}: not {direction} in wickets band "
                        f"({np.round(series, 4).tolist()})"
                    )
    return violations
