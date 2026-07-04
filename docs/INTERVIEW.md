# Talking about cricstate

How to present this project honestly and well — in 30 seconds, 2 minutes, or
10. Every number below is quoted from the frozen evidence set
(`results/summary.json`, `docs/LEADERBOARD.md`); nothing is rounded up.

## 30 seconds

> I built a leakage-audited benchmark on 4.7M balls of cricket data to answer
> one question: once you model match state well, is there anything left in the
> features everyone reaches for next — player identity and pitch conditions?
> I pre-registered a decision rule, evaluated the test split once, and the
> answer was essentially no: identity is worth 0.31%, conditions an order of
> magnitude less. I published the negative result instead of building the
> player-modelling tower. The interesting part isn't the model — it's that the
> measurement was constructed so I couldn't fool myself.

## 2 minutes

The three beats, in order:

1. **The instrument, not the model.** Deterministic parser with
   quarantine-not-crash semantics (22,211 files → 100% parse-or-quarantined);
   temporal 80/10/10 splits baked into the corpus with split integrity as a
   red-build test; a whitelisted feature surface; leakage canaries in CI
   (shuffled targets must score at the base rate, a poisoned outcome column
   must be structurally unreachable); match-level paired bootstrap because
   ball-level resampling fakes precision; calibration fit on validation only.

2. **The pre-commitment.** SPEC_M2 §6 fixes the verdict bands — JUSTIFIES
   needs the 95% CI of ΔNLL to exclude zero *and* ≥ 1% relative improvement —
   and the commit that froze the rule precedes every commit that contains a
   result. The test split was evaluated once.

3. **The refusal.** Shrunk empirical-Bayes player effects improve test NLL by
   −0.00504 nats [−0.00561, −0.00449] — real (CI excludes zero) but 0.31%,
   verdict AMBIGUOUS, below the bar. A strictly causal per-match conditions
   latent added 0.024% on validation, so I froze that arm without spending its
   one-time test evaluation. The pre-committed rule said the hierarchical
   player-modelling tower wasn't justified, so I didn't build it.

## The questions a skeptical interviewer should ask

**"Isn't a negative result just a failed project?"**
Only if the question was bad or the measurement untrustworthy. The question —
where does per-ball signal live? — is the first question any cricket-modelling
effort must answer, and the measurement is the strongest part of the repo:
canaries, temporal splits, frozen rule, single test touch. Knowing that state
saturates free data is *actionable*: it says the next dollar goes to richer
data (ball-tracking, venue covariates), not richer models on the same data.

**"How do I know you didn't just move the goalposts after seeing results?"**
You don't have to trust me — check the commit order. The commit introducing
the frozen §6 decision rule (`c584d62`) precedes the single test evaluation
(`08ed836`) and both experiment branches. The bands (0.3%/1%) were fixed
there; the identity result landed at 0.31% — the band floor — and the verdict
was reported as AMBIGUOUS, not talked up into a win or down into a kill.

**"Couldn't a better identity model have cleared the bar?"**
Possibly — that's why the verdict is AMBIGUOUS rather than KILL, and the
report says so. But the gate was designed to measure the *cheap* increment
honestly: shrunk striker+bowler offsets with one tunable. The unshrunk
version is 0.153 nats *worse* than no identity at all, and 14–19% of test
deliveries involve unseen players — free data dilutes any identity signal.
The pre-committed decision was: below 1%, don't build the tower. A
hierarchical model would have to argue against that measurement, not against
zero.

**"Why NLL and not accuracy / AUC?"**
The deliverable is a probability distribution over 11 outcomes per ball, not
a classification. NLL is the proper scoring rule that prices the whole
distribution; Brier and per-class ECE are reported alongside. Accuracy on an
11-class problem dominated by dot balls rewards predicting the mode.

**"Why match-level bootstrap?"**
Balls within a match are strongly dependent (same pitch, same batters, same
phase structure). Resampling balls treats 344k deliveries as 344k independent
draws and shrinks CIs by roughly the square root of the within-match cluster
size — fake precision. Resampling the 1,493 test matches respects the
dependence structure. The repo forbids ball-level resampling outright.

**"What would change your conclusion?"**
Richer inputs, not richer models: ball-tracking data, venue/weather
covariates, or a wicket-focused conditions latent (the published null only
covers scoring valence — that scope limit is stated in the paper). Or a
challenger under the Module 3 protocol that clears the frozen bar: CI
excluding zero on val *and* test, ≥ 0.5% relative, no calibration regression.

**"What's the engineering you're proudest of?"**
The canary suite. A shuffled-target model must score exactly at the base
rate; a poisoned outcome column must be *structurally unreachable* by the
feature builder (not just unused — unreachable); shuffled identity must
collapse to the state baseline within 0.01 nats. Leakage is the failure mode
that silently invalidates sports ML, so the harness hunts its own bugs in CI.

## What not to say

- Don't say "players don't matter." The measured claim is: *shrunk identity
  effects add 0.31% NLL over a strong state model on free ball-by-ball data,
  which is below the pre-registered materiality bar.* Identity is real
  (CI excludes zero); it is immaterial *here*.
- Don't say "conditions don't exist." The conditions null is validation-only,
  scoring-valence-only, and the arm is marked partial — the paper's scope
  section says exactly what remains open.
- Don't inflate the corpus: 16,754 matches / 4,748,382 deliveries in the
  corpus; the t20 test cell is 1,493 matches / 344,278 deliveries (T1) and
  1,489 / 343,287 (T2).
