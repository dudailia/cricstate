# FAQ — the skeptic's tour

Questions a careful reader should ask, answered from the frozen evidence.
Numbers are quoted verbatim; nothing here modifies a result.

### Why publish a negative result at all?

Because it's the honest output of a pre-registered measurement, and because
negative results are exactly what sports ML is missing. Most published models
are tuned on the data they report and scored without uncertainty; nobody
publishes "the obvious feature wasn't worth it." The value of this repository
is that the measurement was built to be trusted: whatever it returned was
going to be published, and what it returned was "state saturates."

### How do I verify the pre-registration claim without trusting the authors?

Git history is the receipt. The decision rule (SPEC_M2 §6: verdict bands,
1% materiality, single test touch) entered the repository in commit
`c584d62`, which precedes the leaderboard's one-time test evaluation
(`08ed836`) and every experiment result (Branch A: `8c17726`…`0dc5425`;
Branch C: `cf06cb2`…`5d8ec8e`). Run:

```
git log --oneline -- docs/SPEC_M2.md
```

and compare against the result commits. Amendments after gates are listed in
SPEC_M2's changelog as dated amendments — additive, never edits to the rule.

### Isn't +0.31% with a CI excluding zero… a positive result?

Statistically, yes — identity is real. Materially, no — and the distinction
is the point. The pre-registered bands say an effect justifies further
modelling work only at ≥ 1% relative improvement; 0.31% (0.0073 bits/ball)
lands in the AMBIGUOUS band [0.3%, 1.0%). The frozen protocol shipped the
cheap increment (M_shrunk is on the leaderboard) and declined the expensive
tower. Both facts are reported; neither is inflated.

### Maybe your identity model is just weak?

The gate measured the *strong-prior, cheap* version deliberately: empirical-
Bayes shrunk striker+bowler offsets on top of the frozen B3, λ tuned on
validation only. Three facts bound the space:

- The unshrunk version (M_flat) is **0.158 nats worse than no identity at
  all** — free data cannot support per-player tables.
- The shuffled-identity canary sits at the state baseline (+0.00095 nats) —
  the pipeline creates no phantom identity signal.
- 5.25% of test balls have an unknown incoming batter and 14–19% involve
  players unseen in train — dilution any identity model on this data faces.

A hierarchical model might do somewhat better than 0.31%. The pre-committed
judgment was that chasing the gap between 0.31% and 1% wasn't worth the
tower; that judgment is documented in `BRANCH_A_REPORT.md`, not hidden.

### Why was Branch C (conditions) stopped without a test evaluation?

Discipline, not laziness. Its validation effect (+0.024%, 0.0006 bits/ball)
is an order of magnitude below an already-sub-bar identity effect. The
protocol allows exactly one test touch per arm; spending it to confirm a
negligible validation effect buys nothing and costs the arm's only bullet.
The arm is marked **partial (frozen at C1)** and the paper makes no test-set
claim for conditions. The open scope is stated: the latent is scoring-valence
only, so a wicket-prone-but-not-low-scoring pitch signal remains unmeasured.

### Why no deep learning?

The question is *where signal lives*, not *which architecture wins*. B3
(gradient-boosted trees over ~25 state features) is a strong, calibrated,
cheap-to-audit state model; the enrichment arms then isolate identity and
conditions as *increments* with one tunable each. A transformer over ball
sequences would entangle state, identity, and conditions into one score and
answer a different, less decision-relevant question. The challenger protocol
(Module 3) is open: any model that clears the frozen bar earns its place.

### Aren't the CIs suspiciously tight?

They're match-level. 10,000 paired bootstrap resamples over 1,493 test
matches. Ball-level resampling would be far tighter — and wrong, because
balls within a match are dependent. That's why the repo forbids it.

### What does "byte-identical" reproducibility actually cover?

Two consecutive `uv run evalkit run-all` invocations produce byte-identical
`docs/LEADERBOARD.md` (seed 1337 end-to-end, bootstrap seed 90210). The
corpus is pinned by SHA-256 (`c08e4eba…`), labels too, and both hashes are
red-build tests. Figures/tables regenerate deterministically from
`results/summary.json`. Cross-platform pixel-identity of PNGs is not claimed;
the numbers are.

### The ODI cell shows B3 not beating B2. Why is that in the README?

Because it's the methodology working. ΔNLL test −0.006 with 95% CI
[−0.019, +0.008] includes zero at n = 136 test matches, so the frozen rule
says "did not beat the bar" — and that's what's reported. A framework that
only ever certifies wins should not be trusted; this one demonstrably
refuses a close call.

### Can I build on this?

Yes — three doors:

1. **Challenger models** (Module 3): implement the `Predictor` protocol,
   tune on val only, and face the frozen rule. The T2/t20 bar is 0.490 test
   NLL.
2. **The open conditions question**: a wicket-focused (non-valence) latent,
   under the same harness.
3. **Richer data**: the conclusion is scoped to free ball-by-ball data;
   ball-tracking or venue covariates change the question legitimately.

The one thing the protocol forbids is renegotiating the decision rule after
seeing results.
