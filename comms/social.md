# Social drafts

> All numbers verbatim from the frozen evidence. Post after the repo is
> public and Pages is live. The X thread and LinkedIn post tell the same
> story at different registers; don't post both the same day.

## X / Twitter thread

**1/**
I spent a project measuring something everyone assumes: that player identity
matters for predicting cricket, ball by ball.

Pre-registered rule, frozen test split, 4.7M deliveries.

Answer: it's worth 0.31%. I declined to build the player-modelling tower.
🧵

**2/**
Setup: a gradient-boosted model of match state (score, wickets, balls, run
rate, phase) as the baseline. Then the two features every practitioner
reaches for next, measured as increments:

— who's batting/bowling (identity)
— what kind of day/pitch it is (conditions)

**3/**
The measurement is the interesting part, because sports ML is usually
unfalsifiable:

— temporal splits, baked into the data
— leakage canaries in CI (shuffled targets must hit the base rate exactly)
— match-level bootstrap, because resampling balls fakes precision
— decision rule frozen in git BEFORE results

**4/**
Results:
— match state: ~93% of the recoverable signal
— player identity: +0.31% (real, CI excludes zero — but below the
pre-registered 1% bar)
— conditions: +0.024% on validation. An order of magnitude smaller. Arm
frozen without touching test.

**5/**
The part I care about: the unshrunk per-player model is 0.158 nats WORSE
than ignoring players entirely. Free ball-by-ball data cannot support player
tables without heavy shrinkage — and with it, there's almost nothing left to
win.

**6/**
So the published result is negative, on purpose. The rule said "don't build
the tower," and the repo's value is that you can verify the rule predates
the results in git history.

Site + paper + reproducible everything:
https://dudailia.github.io/cricstate/

## LinkedIn

**Headline options:**
- I published a negative result on purpose. Here's why it's the strongest
  project in my portfolio.

**Body:**

Every cricket-prediction project eventually builds the player-modelling
tower: batter form, bowler matchups, hierarchical skill effects. Before
building it, I did something unusual — I measured whether it was worth
building, under rules I couldn't bend afterwards.

The measurement: 16,754 matches, 4.7M deliveries of free ball-by-ball data.
Temporal splits baked into the corpus. Leakage canaries running in CI. A
match-level bootstrap for honest uncertainty. And a decision rule — how big
an effect must be to justify further work — frozen in git before any result
existed, with the test split evaluated exactly once.

The result: match state (score, wickets, balls, run rate, phase) captures
~93% of the recoverable per-ball signal. Player identity adds 0.31% —
statistically real, materially negligible, below the pre-registered bar.
Match conditions add an order of magnitude less.

So the tower didn't get built, and the negative result got published instead
— with the full audit trail: byte-reproducible pipeline, hash-pinned data,
frozen leaderboard, and an open challenger protocol for anyone who thinks
their model clears the bar.

What I took away: in applied ML, the expensive failure mode isn't a weak
model — it's spending a quarter building on signal that was never there. A
day of adversarial measurement is the cheapest insurance against a quarter
of wasted modelling.

Project: https://dudailia.github.io/cricstate/
