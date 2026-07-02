# Golden pathological matches (SPEC_M1 §11)

One real Cricsheet match per failure mode, selected by a full-corpus scan of
`snapshot_2026-07-02` (see `data/MANIFEST`). Each file is byte-identical to the
snapshot copy. IDs are Cricsheet match IDs (ESPNcricinfo).

| # | Failure mode | Match ID | Format | Evidence in file |
|---|--------------|----------|--------|------------------|
| 1 | Tie resolved by super over | 1187669 | T20 | `outcome = {result: tie, eliminator: England}`; innings 3+ marked `super_over` |
| 2 | D/L result | 1499666 | T20 | `outcome.method = "D/L"`; revised `target` in chasing innings |
| 3 | Penalty runs | 1298152 | T20 | delivery with `extras.penalty` |
| 4 | Retired hurt | 804685 | T20 | wicket `kind = "retired hurt"` (GG Wagg) — not a dismissal |
| 5 | Miscounted over | 65273 | ODI | `innings.miscounted_overs = {"2": {balls: "5"}}` (5-ball over allowed) |
| 6 | Stumping off a wide | 1197049 | T20 | delivery with `extras.wides` ≥ 1 and wicket `kind = "stumped"` |
| 7 | Uncontested toss | 1322004 | T20 | `toss.uncontested = true` (only such T20/ODI match in the snapshot) |
| 8 | No-result match | 1409478 | T20 | `outcome.result = "no result"` — parse, flag, exclude from tuples |
| 9 | Concussion replacement | 1334913 | T20 | `delivery.replacements.role` with `reason = "concussion"` (L Tywaku → KI Simmonds) |
| 10 | Wide with runs | 1534732 | T20 | delivery with `extras.wides = 2` (crossing parity uses wides − 1) |
