# Documentation index

A guided reading order. Reader-facing documents first; frozen specs and
evidence packs (the audit trail) after.

## Start here

| doc | what it is | read it if |
|---|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The two modules + two gate experiments, and how every number stays comparable | you want the system in one page |
| [`REPRODUCE.md`](REPRODUCE.md) | Fast path (seconds, no download) and full path (~25 min) with the exact values you should see | you want to verify anything |
| [`LEADERBOARD.md`](LEADERBOARD.md) | The frozen baseline results, all cells, post-calibration, with CIs | you want the numbers |

## The experiments

| doc | what it is |
|---|---|
| [`BRANCH_A_REPORT.md`](BRANCH_A_REPORT.md) | Player identity gate experiment — verdict **AMBIGUOUS** (+0.31%), tower declined |
| [`../report/paper.md`](../report/paper.md) | The paper: the pre-registered negative result ([typeset version](https://dudailia.github.io/cricstate/paper/)) |
| [`FAQ.md`](FAQ.md) | The skeptic's tour — every hard question, answered from the frozen evidence |
| [`INTERVIEW.md`](INTERVIEW.md) | How to present the project in 30s / 2min / 10min, and what not to claim |

## The frozen contract (auditor mode)

These are specifications and evidence packs. They are frozen; changes after
their gate dates are recorded as explicit amendments, never edits.

| doc | what it is |
|---|---|
| [`SPEC_M1.md`](SPEC_M1.md) | Module 1 spec — parser, transition function δ, corpus contract |
| [`SPEC_M2.md`](SPEC_M2.md) | Module 2 spec — **§6 is the decision rule**, committed before any result existed |
| [`EVIDENCE_M1.md`](EVIDENCE_M1.md) | Module 1 evidence pack (corpus build, quarantine accounting) |
| [`EVIDENCE_M2_LABELS.md`](EVIDENCE_M2_LABELS.md) | Frozen label-set evidence (hash-pinned) |
| [`P3_VAL_REPORT.md`](P3_VAL_REPORT.md) | Validation-stage report preceding the single test evaluation |
| [`STATS.md`](STATS.md) | Corpus accounting — dual denominators, coverage |

**The pre-registration receipt:** `git log --follow docs/SPEC_M2.md` shows the
decision rule committed before any experiment result. That ordering — not our
say-so — is the evidence that the rule was not fitted to the outcome.
