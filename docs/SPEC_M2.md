
---

## Changelog (gate-documented amendments)

- **Amendment #1 (P3 gate, 2026-07-03) — thin-cell calibration:** for any
  (task, fmt) cell with < 300 labeled val matches (currently odi), the
  leaderboard calibration map is **Platt**; isotonic is still fitted and
  reported alongside. t20 stays isotonic per §5.
- **Amendment #2 (P3 gate, 2026-07-03) — §5 B1 buckets, spec bug fix:** the
  monotonicity clause referenced an innings-1 runs dimension the bucket list
  never defined. Innings 1 gains a **current-run-rate (CRR) band** with the
  same edges as the innings-2 RRR bands: (−inf,4], (4,6], (6,8], (8,10],
  (10,12], (12,inf). Leaf depth 4 in both innings. Edge case, defined not
  inferred: legal_balls == 0 → lowest CRR band. Monotonicity, innings 1,
  corrected: holding phase and wickets fixed, p̂ non-decreasing in CRR band;
  non-increasing in wickets band as already specified. Innings 2 unchanged.
- **Amendment #3 (P4 gate, 2026-07-03) — B1 monotone by construction:**
  post-shrinkage, B1 (T2) applies weighted alternating PAVA over each
  (innings, phase) lattice: non-decreasing along the rate band (CRR/RRR
  ascending) and along the wickets axis non-increasing for innings 1 /
  non-decreasing for innings 2. Fit on TRAIN quantities only. The
  monotonicity property tests must pass by construction; if PAVA measurably
  worsens B1 val NLL beyond noise, STOP and report.
- **Amendment #4 (P4 gate, 2026-07-03) — artifact cache & runtime budget:**
  P4 caches fitted model + calibration artifacts; run-all re-evaluates from
  cache rather than re-fitting. Budget: cold fit < 60 min (documented),
  cached eval-only run-all < 5 min. Fits are deterministic and the cache is
  fingerprinted (corpus hash, labels hash, model version, seed) so validity
  is checkable.
