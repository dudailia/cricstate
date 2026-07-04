# Show HN draft

> Post when Pages is live and the repo is public. Title under 80 chars.
> HN culture rewards understatement and punishes hype — this draft leans on
> the negative result and the methodology, which is the genuinely HN-shaped
> part of the project.

**Title:**

Show HN: I measured whether player identity matters in cricket. It doesn't (much)

*(alternate: "Show HN: A pre-registered negative result on 4.7M balls of cricket data")*

**URL:** https://dudailia.github.io/cricstate/

**First comment (post immediately after submitting):**

I built a leakage-audited benchmark on 16,754 matches (4.7M deliveries) of
free ball-by-ball cricket data to answer one question: once you condition on
match state (score, wickets, balls, run rate, phase), how much extra
predictive signal is in the features everyone reaches for next — player
identity and pitch/conditions?

The setup, because sports ML is usually untrustworthy by construction:

- Temporal 80/10/10 splits, baked into the corpus; split integrity is a
  red-build test.
- Leakage canaries in CI: a shuffled-target model must score exactly at the
  base rate, a poisoned outcome column must be structurally unreachable by
  the feature builder, shuffled player identities must collapse to the state
  baseline.
- Match-level paired bootstrap (10k resamples) — resampling balls fakes
  precision because balls within a match are dependent.
- A decision rule frozen in git before any result existed (you can check the
  commit order), and a test split evaluated exactly once.

Results: a gradient-boosted state model captures ~93% of the recoverable
signal. Shrunk empirical-Bayes player effects add 0.31% (real — the CI
excludes zero — but below the pre-registered 1% bar; the unshrunk version is
0.153 nats *worse* than no identity at all). A strictly causal per-match
conditions latent added 0.024% on validation, so that arm was frozen without
spending its one-time test evaluation.

So I published the negative result instead of building the player-modelling
tower. Everything is deterministic and hash-pinned; two consecutive runs
produce a byte-identical leaderboard. The repo doubles as a benchmark: if you
think a better model clears the frozen bar, the challenger protocol is open.

Happy to answer questions about the leakage auditing, the bootstrap choice,
or why the conditions arm was stopped at validation.
