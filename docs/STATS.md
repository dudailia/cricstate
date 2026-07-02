# STATS — v1 corpus build

Snapshot: `snapshot_2026-07-02` (see `data/MANIFEST`). Scope: T20 + ODI.

## Corpus

- Files in snapshot: **22211**
- In-scope matches (T20/ODI): **16905**
- Parsed + replayed clean: **16737** (99.006%)
- Parse-or-quarantine-with-reason: **100.000%** (DoD ≥ 99.5%)
- Delivery rows: **4742942**
- Miscounted-over warnings (non-fatal): 290

## Quarantine histogram

| reason | count |
|--------|-------|
| E_BALL_ACCOUNTING | 149 |
| E_DEAD_STATE | 2 |
| E_FORMAT_OOS | 5306 |
| E_UNKNOWN_WICKET_KIND | 17 |

## Temporal split (80/10/10 by start date, baked into the tables)

- train < 2024-11-03 ≤ val < 2025-08-30 ≤ test

## Determinism

- Run 1 corpus hash: `e6de4df917e4d2c5240b021a68cdee1491bedc3bf26c0d7bb4a240cdc86576b6`
- Run 2 corpus hash: `e6de4df917e4d2c5240b021a68cdee1491bedc3bf26c0d7bb4a240cdc86576b6`
- Identical: **True**

## Runtime (Apple Silicon laptop, single process)

- Run 1 (with parquet write): 342.3s
- Run 2 (hash-only re-run): 236.9s

## Notes

- `fow` is flattened to `fow_last_runs`/`fow_last_ball`; the full tuple
  is reconstructible via `cricstate.replay.replay(match_id)`.
- Free hits are not encoded in Cricsheet; `last_ball_was_noball` is
  carried instead (SPEC §11 bias note).
- Field placements are not in this data at all (SPEC §11).
- Leakage rule: nothing downstream may condition on `outcome_*` columns.
