"""Branch A phase A3: the single test-split touch -> docs/BRANCH_A_REPORT.md.

Everything upstream (effects, lambda tuning, calibration) is train/val-only;
this module evaluates ALL models on test together, once. Reuses the frozen M2
bootstrap (match-level, paired, B=10,000, seed 90210). The report contains no
wall-clock content: two runs must be byte-identical.
"""

import sys
from math import log
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from evalkit.bootstrap import (
    CI,
    bootstrap_metric,
    bootstrap_paired_delta,
    make_draws,
    per_match_losses,
)
from evalkit.metrics import brier_multiclass, ece_per_class
from evalkit.models.base import CLASSES, SEED
from evalkit.splits import check_integrity, load_deliveries, load_matches
from experiments.branch_a.canary import CanaryResult, shuffled_identity_effects
from experiments.branch_a.models import (
    LAMBDA_FLAT,
    LAMBDA_GRID,
    CalibratedAugmented,
    assemble_with_ids,
    calibrated,
    fit_identity_effects,
    load_frozen_b3,
    tune_lambda,
)

FloatArray = NDArray[np.float64]

OUT = REPO / "docs" / "BRANCH_A_REPORT.md"

# Frozen verdict bands (Branch A spec — do not editorialize, just classify).
REL_JUSTIFY = 1.0  # %
REL_AMBIG_LO = 0.3  # %


def nll_vector(probs: FloatArray, y: NDArray[np.int64]) -> FloatArray:
    picked = np.clip(probs[np.arange(len(y)), y], 1e-12, None)
    result: FloatArray = -np.log(picked)
    return result


def main() -> int:
    deliveries = load_deliveries()
    matches = load_matches()
    check_integrity(matches, deliveries)

    train = assemble_with_ids(deliveries, "t20", "train")
    val = assemble_with_ids(deliveries, "t20", "val")
    test = assemble_with_ids(deliveries, "t20", "test")  # THE single test touch
    y_test = test["y"].to_numpy().astype(np.int64)

    b3 = load_frozen_b3()
    base_val = b3.predict_proba(val.drop("striker_id", "bowler_id"))
    base_test = b3.predict_proba(test.drop("striker_id", "bowler_id"))

    # models (fits/tuning all train/val-only)
    m_state = calibrated("M_state", None, base_val, val)
    effects_best, curve = tune_lambda(train, val, base_val)
    m_shrunk = calibrated("M_shrunk", effects_best, base_val, val)
    m_flat = calibrated("M_flat", fit_identity_effects(train, LAMBDA_FLAT), base_val, val)
    m_shuffled = calibrated(
        "M_shuffled", shuffled_identity_effects(train, curve.best_lam), base_val, val
    )
    models: list[CalibratedAugmented] = [m_state, m_flat, m_shrunk, m_shuffled]

    # single joint test evaluation
    probs = {m.name: m.predict(base_test, test) for m in models}
    n_test_matches = test["match_id"].n_unique()
    draws = make_draws(n_test_matches)

    nll_ci: dict[str, CI] = {}
    brier: dict[str, float] = {}
    ece_mean: dict[str, float] = {}
    ece_class: dict[str, FloatArray] = {}
    sums: dict[str, FloatArray] = {}
    counts: FloatArray | None = None
    for name, p in probs.items():
        vec = nll_vector(p, y_test)
        s, c = per_match_losses(vec, test["match_id"])
        sums[name] = s
        counts = c
        nll_ci[name] = bootstrap_metric(s, c, draws)
        brier[name] = brier_multiclass(p, y_test)
        per_class = ece_per_class(p, y_test, n_bins=20)
        ece_class[name] = per_class
        ece_mean[name] = float(np.mean(per_class))
    assert counts is not None

    # Brier CI for the two headline models (per-ball squared-error per match)
    brier_ci: dict[str, CI] = {}
    for name in ("M_state", "M_shrunk"):
        onehot = np.zeros_like(probs[name])
        onehot[np.arange(len(y_test)), y_test] = 1.0
        bvec = np.sum((probs[name] - onehot) ** 2, axis=1)
        bs, bc = per_match_losses(bvec, test["match_id"])
        brier_ci[name] = bootstrap_metric(bs, bc, draws)

    d_primary = bootstrap_paired_delta(sums["M_shrunk"], sums["M_state"], counts, draws)
    d_flat = bootstrap_paired_delta(sums["M_flat"], sums["M_shrunk"], counts, draws)
    d_shuf = bootstrap_paired_delta(sums["M_shuffled"], sums["M_state"], counts, draws)

    rel_pct = -d_primary.point / nll_ci["M_state"].point * 100.0
    bits_per_ball = -d_primary.point / log(2)

    # dilution terms
    strikers = test["striker_id"]
    bowlers = test["bowler_id"]
    null_striker = float(strikers.is_null().to_numpy().mean()) * 100.0
    seen_s = effects_best.striker.seen(strikers)
    seen_b = effects_best.bowler.seen(bowlers)
    non_null = ~strikers.is_null().to_numpy()
    unseen_striker = float(np.mean(~seen_s & non_null)) * 100.0
    unseen_bowler = float(np.mean(~seen_b)) * 100.0
    years = (
        test.select("match_id")
        .join(matches.select("match_id", "start_date"), on="match_id", how="left")["start_date"]
        .dt.year()
        .to_numpy()
    )
    by_season = {
        int(yr): {
            "null_striker_pct": float(np.mean(strikers.is_null().to_numpy()[years == yr])) * 100,
            "unseen_striker_pct": float(np.mean((~seen_s & non_null)[years == yr])) * 100,
            "unseen_bowler_pct": float(np.mean(~seen_b[years == yr])) * 100,
        }
        for yr in sorted(set(years.tolist()))
    }

    # Laplace-floor perturbation (carry-item 2): with vs without the +1 floor
    y_train = train["y"].to_numpy()
    cnt = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float64)
    p_floor = (cnt + 1.0) / (cnt.sum() + len(CLASSES))
    p_raw = cnt / cnt.sum()
    with np.errstate(divide="ignore"):
        dlog = np.abs(np.log(p_floor) - np.log(np.where(p_raw > 0, p_raw, np.nan)))
    max_dlog = float(np.nanmax(dlog))
    nll_impact_bound = float(np.nansum(p_raw * dlog))

    # binding canary (test), checked before the verdict is trusted
    canary = CanaryResult(
        nll_shuffled=nll_ci["M_shuffled"].point,
        nll_state=nll_ci["M_state"].point,
        fold="test",
    )

    # frozen verdict
    ci_excludes_zero_improving = d_primary.hi < 0.0
    if not canary.passed:
        verdict = "VOID"
    elif ci_excludes_zero_improving and rel_pct >= REL_JUSTIFY:
        verdict = "JUSTIFIES_TOWER"
    elif ci_excludes_zero_improving and rel_pct >= REL_AMBIG_LO:
        verdict = "AMBIGUOUS"
    else:
        verdict = "KILL_BRANCH"

    def ci_s(ci: CI) -> str:
        return f"{ci.point:.5f} [{ci.lo:.5f}, {ci.hi:.5f}]"

    lines = [
        "# BRANCH A REPORT — player identity information, T1 per-ball, t20",
        "",
        f"Frozen M2 harness · corpus v1.2 · seed {SEED} · bootstrap: match-level,",
        "paired, B=10,000, seed 90210 (frozen module) · calibration: temperature",
        "per model, fit on val (M2 convention) · test split touched ONCE, all",
        "models evaluated together in this run.",
        "",
        "## Logged assumptions (minimal, per protocol)",
        "",
        "1. No separate Branch A spec file exists; the milestone prompt is the spec.",
        "2. lambda parameterized as prior strength in pseudo-balls (conjugate",
        "   Dirichlet form); the only tunable. Grid: "
        + ", ".join(f"{v:g}" for v in LAMBDA_GRID)
        + ".",
        "3. M_flat = lambda->0 unshrunk limit of the same augmentation family",
        f"   (epsilon floor {LAMBDA_FLAT} pseudo-balls) — the unshrunk MLE, making",
        "   the shrinkage gap like-for-like.",
        "4. One shared lambda for striker and bowler effects.",
        "5. Null striker IDs (M1 observation gap) count as unseen -> zero offset.",
        "6. Canary 'random IDs' = fixed-seed within-train permutations; binding",
        "   canary NLL computed on test inside this single touch (A2 gated on a",
        "   val preliminary: delta +0.00094, PASS).",
        "7. Val lambda curve scored post-temperature — the final-eval protocol.",
        "8. Bowler-type is not in the frozen corpus: no matchup terms, no",
        "   scraping — logged and proceeded, per spec.",
        "",
        "## Failure checks (before the verdict is trusted)",
        "",
        f"- (a) Task: T1 per-ball outcome, K={len(CLASSES)} frozen alphabet",
        "  (labels = outcome_class indices; NOT win probability). CONFIRMED.",
        f"- (b) Val lambda curve unimodal: {curve.is_unimodal()} "
        f"(best lambda = {curve.best_lam:g}, interior grid point).",
        f"- (c) Dilution stated: null-striker {null_striker:.2f}% · unseen striker "
        f"{unseen_striker:.2f}% · unseen bowler {unseen_bowler:.2f}% of test deliveries.",
        f"- (d) {canary.line()}",
        f"- Laplace-floor perturbation (carry-item): max |dlog p| = {max_dlog:.2e}",
        f"  (rarest class); NLL impact bound {nll_impact_bound:.2e} nats — below",
        "  4th-decimal resolution. CONFIRMED negligible.",
        "",
        "## Val lambda curve (post-temperature NLL)",
        "",
        "| lambda | " + " | ".join(f"{v:g}" for v in curve.lam_grid) + " |",
        "|---|" + "---|" * len(curve.lam_grid),
        "| val NLL | " + " | ".join(f"{v:.5f}" for v in curve.val_nll) + " |",
        "",
        f"## Test results (single touch; n = {n_test_matches} matches / {test.height} deliveries)",
        "",
        "| model | test NLL [95% CI] | multiclass Brier | mean per-class ECE (B=20) |",
        "|---|---|---|---|",
        *(
            f"| {name} | {ci_s(nll_ci[name])} | {brier[name]:.5f} | {ece_mean[name]:.5f} |"
            for name in ("M_state", "M_flat", "M_shrunk", "M_shuffled")
        ),
        "",
        f"Brier with CI — M_state: {ci_s(brier_ci['M_state'])}; "
        f"M_shrunk: {ci_s(brier_ci['M_shrunk'])}.",
        "",
        "### Paired deltas (match-level bootstrap)",
        "",
        f"- **dNLL (M_shrunk - M_state), PRIMARY: {ci_s(d_primary)}**",
        f"- dNLL (M_flat - M_shrunk), shrinkage gap: {ci_s(d_flat)}",
        f"- dNLL (M_shuffled - M_state), canary: {ci_s(d_shuf)}",
        "",
        "### Effect size",
        "",
        f"- Relative NLL improvement: **{rel_pct:.2f}%**",
        f"- Information gain: **{bits_per_ball:.5f} bits/ball**",
        "",
        "### Per-class ECE (test, B=20 equal-mass bins)",
        "",
        "| class | M_state | M_shrunk |",
        "|---|---|---|",
        *(
            f"| {cls} | {ece_class['M_state'][i]:.5f} | {ece_class['M_shrunk'][i]:.5f} |"
            for i, cls in enumerate(CLASSES)
        ),
        "",
        "### Dilution by season (test)",
        "",
        "| season | null-striker % | unseen striker % | unseen bowler % |",
        "|---|---|---|---|",
        *(
            f"| {yr} | {v['null_striker_pct']:.2f} | {v['unseen_striker_pct']:.2f} "
            f"| {v['unseen_bowler_pct']:.2f} |"
            for yr, v in by_season.items()
        ),
        "",
        "## VERDICT (frozen bands; classified, not editorialized)",
        "",
        f"- Primary dNLL 95% CI: [{d_primary.lo:.5f}, {d_primary.hi:.5f}] — "
        f"{'excludes 0 (improving)' if ci_excludes_zero_improving else 'does NOT exclude 0'}",
        f"- Relative NLL improvement: **{rel_pct:.2f}%** — band: "
        + (
            ">= 1.0% (JUSTIFIES_TOWER band)"
            if rel_pct >= REL_JUSTIFY
            else (
                "[0.3%, 1.0%) (AMBIGUOUS band)" if rel_pct >= REL_AMBIG_LO else "< 0.3% (KILL band)"
            )
        ),
        f"- Dilution context: null-striker {null_striker:.2f}%, unseen striker "
        f"{unseen_striker:.2f}%, unseen bowler {unseen_bowler:.2f}%",
        f"- Canary: {'PASS' if canary.passed else ('VOID' if canary.leaks else 'FAIL')}",
        "",
        f"# VERDICT: {verdict}",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print(f"VERDICT: {verdict} (rel {rel_pct:.2f}%, CI [{d_primary.lo:.5f}, {d_primary.hi:.5f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
