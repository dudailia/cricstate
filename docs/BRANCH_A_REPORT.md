# BRANCH A REPORT — player identity information, T1 per-ball, t20

Frozen M2 harness · corpus v1.2 · seed 1337 · bootstrap: match-level,
paired, B=10,000, seed 90210 (frozen module) · calibration: temperature
per model, fit on val (M2 convention) · test split touched ONCE, all
models evaluated together in this run.

## Logged assumptions (minimal, per protocol)

1. No separate Branch A spec file exists; the milestone prompt is the spec.
2. lambda parameterized as prior strength in pseudo-balls (conjugate
   Dirichlet form); the only tunable. Grid: 25, 50, 100, 200, 400, 800, 1600, 3200.
3. M_flat = lambda->0 unshrunk limit of the same augmentation family
   (epsilon floor 0.01 pseudo-balls) — the unshrunk MLE, making
   the shrinkage gap like-for-like.
4. One shared lambda for striker and bowler effects.
5. Null striker IDs (M1 observation gap) count as unseen -> zero offset.
6. Canary 'random IDs' = fixed-seed within-train permutations; binding
   canary NLL computed on test inside this single touch (A2 gated on a
   val preliminary: delta +0.00094, PASS).
7. Val lambda curve scored post-temperature — the final-eval protocol.
8. Bowler-type is not in the frozen corpus: no matchup terms, no
   scraping — logged and proceeded, per spec.

## Failure checks (before the verdict is trusted)

- (a) Task: T1 per-ball outcome, K=11 frozen alphabet
  (labels = outcome_class indices; NOT win probability). CONFIRMED.
- (b) Val lambda curve unimodal: True (best lambda = 1600, interior grid point).
- (c) Dilution stated: null-striker 5.25% · unseen striker 14.45% · unseen bowler 19.15% of test deliveries.
- (d) shuffled-identity canary [test]: shuffled 1.61533 vs M_state 1.61438 (delta +0.00095, eps 0.01) -> PASS
- Laplace-floor perturbation (carry-item): max |dlog p| = 2.66e-03
  (rarest class); NLL impact bound 4.64e-06 nats — below
  4th-decimal resolution. CONFIRMED negligible.

## Val lambda curve (post-temperature NLL)

| lambda | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|---|
| val NLL | 1.65400 | 1.64359 | 1.63423 | 1.62695 | 1.62230 | 1.62014 | 1.61979 | 1.62057 |

## Test results (single touch; n = 1493 matches / 344278 deliveries)

| model | test NLL [95% CI] | multiclass Brier | mean per-class ECE (B=20) |
|---|---|---|---|
| M_state | 1.61438 [1.60858, 1.61995] | 0.72782 | 0.00354 |
| M_flat | 1.76775 [1.75993, 1.77592] | 0.75520 | 0.02836 |
| M_shrunk | 1.60934 [1.60360, 1.61489] | 0.72686 | 0.00402 |
| M_shuffled | 1.61533 [1.60952, 1.62090] | 0.72801 | 0.00359 |

Brier with CI — M_state: 0.72782 [0.72595, 0.72964]; M_shrunk: 0.72686 [0.72500, 0.72869].

### Paired deltas (match-level bootstrap)

- **dNLL (M_shrunk - M_state), PRIMARY: -0.00504 [-0.00561, -0.00449]**
- dNLL (M_flat - M_shrunk), shrinkage gap: 0.15841 [0.15233, 0.16461]
- dNLL (M_shuffled - M_state), canary: 0.00095 [0.00082, 0.00108]

### Effect size

- Relative NLL improvement: **0.31%**
- Information gain: **0.00727 bits/ball**

### Per-class ECE (test, B=20 equal-mass bins)

| class | M_state | M_shrunk |
|---|---|---|
| 0 | 0.00969 | 0.01312 |
| 1 | 0.00734 | 0.00740 |
| 2 | 0.00318 | 0.00259 |
| 3 | 0.00133 | 0.00138 |
| 4 | 0.00333 | 0.00549 |
| 6 | 0.00272 | 0.00375 |
| other_runs | 0.00010 | 0.00013 |
| bye_legbye | 0.00260 | 0.00210 |
| no_ball | 0.00074 | 0.00054 |
| wide | 0.00628 | 0.00572 |
| wicket | 0.00165 | 0.00196 |

### Dilution by season (test)

| season | null-striker % | unseen striker % | unseen bowler % |
|---|---|---|---|
| 2025 | 5.21 | 13.20 | 18.48 |
| 2026 | 5.27 | 15.12 | 19.51 |

## VERDICT (frozen bands; classified, not editorialized)

- Primary dNLL 95% CI: [-0.00561, -0.00449] — excludes 0 (improving)
- Relative NLL improvement: **0.31%** — band: [0.3%, 1.0%) (AMBIGUOUS band)
- Dilution context: null-striker 5.25%, unseen striker 14.45%, unseen bowler 19.15%
- Canary: PASS

# VERDICT: AMBIGUOUS
