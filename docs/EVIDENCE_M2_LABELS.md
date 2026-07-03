# EVIDENCE — T2 label census (M1.2 corpus, schema 1.1.0)

Every distinct outcome shape in `matches.parquet`, mapped through
`evalkit.labels.resolve_one` (the real code path, not a re-description).

| result | winner | eliminator | bowl_out | method | matches | rule | disposition |
|---|---|---|---|---|---|---|---|
| (won) | set | null | null | Awarded | 3 | (2) winner set | LABELED |
| (won) | set | null | null | D/L | 713 | (2) winner set | LABELED |
| (won) | set | null | null | Lost fewer wickets | 1 | (2) winner set | LABELED |
| (won) | set | null | null | VJD | 5 | (2) winner set | LABELED |
| (won) | set | null | null | — | 15459 | (2) winner set | LABELED |
| no result | null | null | null | — | 393 | (1) no-result | EXCLUDE_NO_RESULT |
| tie | null | null | null | D/L | 13 | (4) true tie | EXCLUDE_TRUE_TIE |
| tie | null | null | null | — | 58 | (4) true tie | EXCLUDE_TRUE_TIE |
| tie | null | null | set | — | 2 | (3) tie + one tie-breaker | LABELED |
| tie | null | set | null | D/L | 1 | (3) tie + one tie-breaker | LABELED |
| tie | null | set | null | — | 106 | (3) tie + one tie-breaker | LABELED |

Total shapes: 11; total matches: 16754; all mapped.

## Disposition counts per split

| split | disposition | matches |
|---|---|---|
| test | EXCLUDE_NO_RESULT | 44 |
| test | EXCLUDE_TRUE_TIE | 4 |
| test | LABELED | 1625 |
| train | EXCLUDE_NO_RESULT | 316 |
| train | EXCLUDE_TRUE_TIE | 62 |
| train | LABELED | 13038 |
| val | EXCLUDE_NO_RESULT | 33 |
| val | EXCLUDE_TRUE_TIE | 5 |
| val | LABELED | 1627 |

## First-batting-team win base rates (LABELED only)

| fmt | split | n | base rate |
|---|---|---|---|
| odi | test | 136 | 0.4706 |
| odi | train | 2654 | 0.4736 |
| odi | val | 150 | 0.5467 |
| t20 | test | 1489 | 0.4916 |
| t20 | train | 10384 | 0.4791 |
| t20 | val | 1477 | 0.4868 |

## Determinism

- Label-build hash, run 1: `e11bf9c83276d10871b401263c351d7e4d93da85183b3c1d141fefbdad20324f`
- Label-build hash, run 2: `e11bf9c83276d10871b401263c351d7e4d93da85183b3c1d141fefbdad20324f`
- Identical: **True**
