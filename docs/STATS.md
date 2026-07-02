# STATS — v1 corpus build

Snapshot: `snapshot_2026-07-02` (see `data/MANIFEST`). Scope: T20 + ODI.

## Corpus

- File disposition: **22211 files = 16754 parsed + 5457 quarantined-with-reason**
- In-scope matches (T20/ODI): **16905**
- In-scope parse-clean rate: **16754/16905 = 99.107%**
- Parse-or-quarantine-with-reason: **100.000%** (DoD ≥ 99.5%)
- Delivery rows: **4748382**
- Miscounted-over warnings (non-fatal): 291

## Quarantine histogram

| reason | count |
|--------|-------|
| E_BALL_ACCOUNTING | 149 |
| E_DEAD_STATE | 2 |
| E_FORMAT_OOS | 5306 |

## Temporal split (80/10/10 by start date, baked into the tables)

- train < 2024-11-03 ≤ val < 2025-08-30 ≤ test

## Determinism

- Run 1 corpus hash: `54c2a2ef435bb0c8ca9a5595536982a5f50efe29c86c64802dde79a2e650956f`
- Run 2 corpus hash: `54c2a2ef435bb0c8ca9a5595536982a5f50efe29c86c64802dde79a2e650956f`
- Identical: **True**

## Runtime (Apple Silicon laptop, single process)

- Run 1 (with parquet write): 592.3s
- Run 2 (hash-only re-run): 456.9s

## Notes

- `fow` is flattened to `fow_last_runs`/`fow_last_ball`; the full tuple
  is reconstructible via `cricstate.replay.replay(match_id)`.
- Free hits are not encoded in Cricsheet; `last_ball_was_noball` is
  carried instead (SPEC §11 bias note).
- Field placements are not in this data at all (SPEC §11).
- Leakage rule: nothing downstream may condition on `outcome_*` columns.
