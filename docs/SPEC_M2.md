
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
