# Evaluating cricstate — a reviewer's guide

How to read this project critically and check its claims yourself. Every number
below is quoted from the frozen evidence set (`results/summary.json`,
`docs/LEADERBOARD.md`); nothing is rounded up.

## The project in three beats

1. **The instrument, not the model.** A deterministic parser with
   quarantine-not-crash semantics (22,211 files → 100% parse-or-quarantined);
   temporal 80/10/10 splits baked into the corpus with split integrity as a
   red-build test; a whitelisted feature surface; leakage canaries (a poisoned
   outcome column is *structurally unreachable* by the feature builder, and
   shuffled identities must collapse to the state baseline — the structural and
   synthetic canaries run in CI, the data-driven ones over the pinned corpus in
   the validation report); match-level paired bootstrap because ball-level
   resampling fakes precision; calibration fit on validation only.

2. **The pre-commitment.** The measurement apparatus and the single-touch
   discipline were frozen before the test split was read (P3, `c584d62`); the
   test split was then evaluated once. `docs/SPEC_M2.md` §6 gives the exact,
   git-checkable timeline — including the honest detail that the verdict
   thresholds (challenger ≥ 0.5%; enrichment 1% / 0.3%) were applied *in* the
   single test touches, not committed strictly before them.

3. **The refusal.** Shrunk empirical-Bayes player effects improve test NLL by
   −0.00504 nats [−0.00561, −0.00449] — real (CI excludes zero) but 0.31%,
   verdict AMBIGUOUS, below the 1% enrichment bar. A strictly causal per-match
   conditions latent added 0.024% on validation, so that arm was frozen without
   spending its one-time test evaluation. The fixed rule said the hierarchical
   player-modelling tower wasn't justified, so it wasn't built.

## Questions worth asking

**"Isn't a negative result just a failed project?"**
Only if the question was bad or the measurement untrustworthy. The question —
where does per-ball signal live? — is the first question any cricket-modelling
effort must answer, and the measurement is the strongest part of the repo:
canaries, temporal splits, a fixed rule, a single test touch. Knowing that
state saturates free data is *actionable*: it says the next dollar goes to
richer data (ball-tracking, venue covariates), not richer models on the same
data.

**"How do I know the goalposts weren't moved after seeing results?"**
Check the commit order — and check what it actually shows. What was frozen
before any test read (`c584d62`) is the *apparatus and discipline*: the model
ladder, the metrics, the calibration policy, the leakage canaries (including the
ladder-inversion STOP rule), the corpus/label hash pins, and the single-touch
protocol (the P3 report is validation-only, test untouched). The verdict
thresholds themselves were applied *in* the one-time test evaluations
(`08ed836` for the challenger ladder; `0dc5425` for the identity bands), not
committed strictly before them — the code says so outright ("no separate Branch
A spec file exists; the milestone prompt is the spec"). So the guarantee is a
single, documented, canary-gated, byte-identical test touch — and the identity
result (0.31%, the band floor) was reported as AMBIGUOUS, not talked up into a
win or down into a kill. `docs/SPEC_M2.md` §6 lays out the full timeline.

**"Couldn't a better identity model have cleared the bar?"**
Possibly — that's why the verdict is AMBIGUOUS rather than KILL, and the report
says so. But the gate was designed to measure the *cheap* increment honestly:
shrunk striker+bowler offsets with one tunable. The unshrunk version is 0.153
nats *worse* than no identity at all, and 14–19% of test deliveries involve
unseen players — free data dilutes any identity signal. The decision was: below
1%, don't build the tower. A hierarchical model would have to argue against that
measurement, not against zero.

**"Why NLL and not accuracy / AUC?"**
The deliverable is a probability distribution over 11 outcomes per ball, not a
classification. NLL is the proper scoring rule that prices the whole
distribution; Brier and per-class ECE are reported alongside. Accuracy on an
11-class problem dominated by dot balls rewards predicting the mode.

**"Why match-level bootstrap?"**
Balls within a match are strongly dependent (same pitch, same batters, same
phase structure). Resampling balls treats 344k deliveries as 344k independent
draws and shrinks CIs by roughly the square root of the within-match cluster
size — fake precision. Resampling the 1,493 test matches respects the
dependence structure. The repo forbids ball-level resampling outright.

**"What would change the conclusion?"**
Richer inputs, not richer models: ball-tracking data, venue/weather covariates,
or a wicket-focused conditions latent (the published null only covers scoring
valence — that scope limit is stated in the paper). Or a challenger under the
Module 3 protocol that clears the frozen bar: CI excluding zero on val *and*
test, ≥ 0.5% relative, no calibration regression.

**"What's the strongest engineering here?"**
The canary suite. A poisoned outcome column must be *structurally unreachable*
by the feature builder (not just unused — unreachable); shuffled identity must
collapse to the state baseline within 0.01 nats; and the data-driven
shuffled-target and ladder-inversion canaries run over the pinned corpus in the
validation report. Leakage is the failure mode that silently invalidates sports
ML, so the harness hunts its own bugs.

## Common misreadings

- **"Players don't matter."** The measured claim is: shrunk identity effects add
  0.31% NLL over a strong state model on free ball-by-ball data, which is below
  the fixed materiality bar. Identity is real (CI excludes zero); it is
  immaterial *here*.
- **"Conditions don't exist."** The conditions null is validation-only,
  scoring-valence-only, and the arm is marked partial — the paper's scope
  section says exactly what remains open.
- **Corpus size.** 16,754 matches / 4,748,382 deliveries in the corpus; the t20
  test cell is 1,493 matches / 344,278 deliveries (T1) and 1,489 / 343,287 (T2).
