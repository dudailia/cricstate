# P3 VAL REPORT — baseline ladder (val only; test untouched)

Corpus tag v1.2 · seed 1337 · all constants chosen on val:
B1 τ ∈ {50,200,800}; B2 C ∈ {0.01,0.1,1,10}; B3 early-stop iteration
count (warm-start, patience 3, step 25); calibration maps (temperature
T for T1; Platt a,b and isotonic for T2). Calibration maps are fit on
val and, in this P3 report, also evaluated on val — post-calibration
numbers are in-sample for the calibrator until P4's test run.

## t1/t20

train rows: 2409881 · val rows: 340375 · cell runtime 2282.9s

| model | val NLL | val Brier | val ECE* | post-cal NLL | post-cal ECE* | T | tuned |
|---|---|---|---|---|---|---|---|
| B0_marginal | 1.69597 | 0.75749 | 0.00497 | 1.69532 | 0.00472 | 1.0396 | — |
| B1_table | 1.66039 | 0.74185 | 0.00494 | 1.65950 | 0.00428 | 1.0453 | τ=200.0 |
| B2_logistic | 1.64136 | 0.73596 | 0.00508 | 1.64069 | 0.00453 | 1.0387 | C=10.0 |
| B3_gbm | 1.62672 | 0.73115 | 0.00417 | 1.62604 | 0.00340 | 1.0392 | iters=125 |

\* mean one-vs-rest ECE over the 11 classes, B=20 equal-mass bins

- ladder-inversion canary: **PASS** — B0 1.695970, B2 1.641359 (ok=True), B3 1.626721 (ok=True)
- shuffled-target canary: **PASS** — shuffled-B2 val NLL 1.695989 vs B0 1.695970 (delta +0.000019, eps 0.01)
- poison-column canary: structural (whitelist select-first); asserted in tests/test_features.py — PASS

## t1/odi

train rows: 1451638 · val rows: 79758 · cell runtime 761.0s

| model | val NLL | val Brier | val ECE* | post-cal NLL | post-cal ECE* | T | tuned |
|---|---|---|---|---|---|---|---|
| B0_marginal | 1.39519 | 0.65884 | 0.00226 | 1.39517 | 0.00196 | 1.0068 | — |
| B1_table | 1.36003 | 0.64062 | 0.00364 | 1.36000 | 0.00352 | 1.0067 | τ=800.0 |
| B2_logistic | 1.35237 | 0.63640 | 0.00466 | 1.35237 | 0.00467 | 0.9999 | C=0.01 |
| B3_gbm | 1.33925 | 0.63148 | 0.00321 | 1.33921 | 0.00301 | 1.0087 | iters=50 |

\* mean one-vs-rest ECE over the 11 classes, B=20 equal-mass bins

- ladder-inversion canary: **PASS** — B0 1.395193, B2 1.352370 (ok=True), B3 1.339251 (ok=True)
- shuffled-target canary: **PASS** — shuffled-B2 val NLL 1.394732 vs B0 1.395193 (delta -0.000461, eps 0.01)
- poison-column canary: structural (whitelist select-first); asserted in tests/test_features.py — PASS

## t2/t20

train rows: 2400926 · val rows: 339411 · cell runtime 205.4s

| model | val NLL | val Brier | val ECE | Platt NLL/ECE | isotonic NLL/ECE | leaderboard map | tuned |
|---|---|---|---|---|---|---|---|
| B0_marginal | 0.69468 | 0.25077 | 0.02878 | 0.69302 / 0.00000 | 0.69302 / 0.00000 | isotonic (1477 val matches) | — |
| B1_table | 0.54825 | 0.18781 | 0.01625 | 0.54776 / 0.01223 | 0.54558 / 0.00000 | isotonic (1477 val matches) | τ=200.0 |
| B2_logistic | 0.52235 | 0.17808 | 0.00710 | 0.52225 / 0.00698 | 0.52054 / 0.00000 | isotonic (1477 val matches) | C=10.0 |
| B3_gbm | 0.51021 | 0.17427 | 0.01751 | 0.50917 / 0.01091 | 0.50765 / 0.00000 | isotonic (1477 val matches) | iters=150 |

B1 monotonicity: **FAIL** — 28 violation(s): ['innings1 phase0 wkts2: not non-decreasing in rate band ([0.1795, 0.2194, 0.2121, 0.3015, 0.2388, 0.2508])', 'innings1 phase0 wkts3: not non-decreasing in rate band ([0.2114, 0.2972, 0.2993, 0.293, 0.3048, 0.3048])', 'innings1 phase0 rate0: not non-increasing in wickets band ([0.4116, 0.2179, 0.1795, 0.2114])', 'innings1 phase0 rate1: not non-increasing in wickets band ([0.4701, 0.3276, 0.2194, 0.2972])', 'innings1 phase0 rate2: not non-increasing in wickets band ([0.5194, 0.4136, 0.2121, 0.2993])']

- ladder-inversion canary: **PASS** — B0 0.694680, B2 0.522354 (ok=True), B3 0.510210 (ok=True)
- shuffled-target canary: **PASS** — shuffled-B2 val NLL 0.693200 vs B0 0.694680 (delta -0.001481, eps 0.01)
- poison-column canary: structural (whitelist select-first); asserted in tests/test_features.py — PASS

## t2/odi

train rows: 1437925 · val rows: 79758 · cell runtime 93.7s

| model | val NLL | val Brier | val ECE | Platt NLL/ECE | isotonic NLL/ECE | leaderboard map | tuned |
|---|---|---|---|---|---|---|---|
| B0_marginal | 0.70149 | 0.25417 | 0.09221 | 0.68445 / 0.00001 | 0.68445 / 0.00000 | platt (150 val matches) | — |
| B1_table | 0.55440 | 0.18879 | 0.04628 | 0.54853 / 0.03758 | 0.54046 / 0.00000 | platt (150 val matches) | τ=50.0 |
| B2_logistic | 0.53590 | 0.18158 | 0.03400 | 0.52999 / 0.02369 | 0.52548 / 0.00000 | platt (150 val matches) | C=1.0 |
| B3_gbm | 0.52219 | 0.17814 | 0.03782 | 0.51940 / 0.02844 | 0.51207 / 0.00000 | platt (150 val matches) | iters=50 |

B1 monotonicity: **FAIL** — 37 violation(s): ['innings1 phase0 wkts0: not non-decreasing in rate band ([0.4364, 0.5654, 0.6189, 0.6404, 0.6116, 0.6024])', 'innings1 phase0 wkts1: not non-decreasing in rate band ([0.2516, 0.3774, 0.4675, 0.4008, 0.6515, 0.3425])', 'innings1 phase0 wkts2: not non-decreasing in rate band ([0.0709, 0.1978, 0.2206, 0.2821, 0.1267, 0.1267])', 'innings1 phase0 rate0: not non-increasing in wickets band ([0.4364, 0.2516, 0.0709, 0.0726])', 'innings1 phase0 rate4: not non-increasing in wickets band ([0.6116, 0.6515, 0.1267, 0.1729])']

- ladder-inversion canary: **PASS** — B0 0.701493, B2 0.535905 (ok=True), B3 0.522191 (ok=True)
- shuffled-target canary: **PASS** — shuffled-B2 val NLL 0.694716 vs B0 0.701493 (delta -0.006777, eps 0.01)
- poison-column canary: structural (whitelist select-first); asserted in tests/test_features.py — PASS

