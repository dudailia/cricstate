"""run-all: the P4 leaderboard release (SPEC_M2 §8, §10, amendment #4).

Fits (or loads from the fingerprinted cache) B0-B3 per (task, fmt), fits
calibration on val, and performs the SINGLE test-split evaluation. Emits
docs/LEADERBOARD.md (byte-identical across consecutive runs: no wall-clock
content), reliability PNGs, bucket tables, and the frozen §6 decision-rule
outcome per cell with match-level paired-bootstrap CIs.
"""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from evalkit import cache
from evalkit.bootstrap import (
    B_RESAMPLES,
    BOOTSTRAP_SEED,
    CI,
    bootstrap_metric,
    bootstrap_paired_delta,
    make_draws,
    per_match_losses,
)
from evalkit.calibrate import fit_isotonic, fit_platt, fit_temperature
from evalkit.datasets import DataBundle
from evalkit.freeze import PINNED_CORPUS_HASH
from evalkit.metrics import (
    ece,
    ece_per_class,
    max_ce,
    reliability_data,
)
from evalkit.models.b0_marginal import B0MarginalT1, B0MarginalT2
from evalkit.models.b1_table import B1TableT1, B1TableT2
from evalkit.models.b2_logistic import B2Logistic
from evalkit.models.b3_gbm import B3Gbm
from evalkit.models.base import CLASSES, SEED, Predictor, to_y
from evalkit.policy import t2_leaderboard_calibration
from evalkit.reports import plot_reliability
from evalkit.splits import check_integrity, metadata_lines, split_metadata

REPO = Path(__file__).resolve().parents[2]
LEADERBOARD = REPO / "docs" / "LEADERBOARD.md"
PLOTS = REPO / "docs" / "plots"

EPS = 1e-12
MODEL_NAMES = ("B0_marginal", "B1_table", "B2_logistic", "B3_gbm")


def _mk_model(task: str, fmt: str, name: str) -> Predictor:
    if name == "B0_marginal":
        return B0MarginalT1() if task == "t1" else B0MarginalT2()
    if name == "B1_table":
        return B1TableT1(fmt) if task == "t1" else B1TableT2(fmt)
    if name == "B2_logistic":
        return B2Logistic(task)
    if name == "B3_gbm":
        return B3Gbm(task)
    raise ValueError(f"unknown model {name!r}")


def _fit_or_load(
    task: str, fmt: str, name: str, train: pl.DataFrame, val: pl.DataFrame, cold: bool
) -> tuple[Predictor, bool]:
    if not cold:
        cached = cache.load(task, fmt, name)
        if cached is not None:
            return cached, True
    model = _mk_model(task, fmt, name)
    model.fit(train, val)
    cache.store(task, fmt, model)
    return model, False


def _t1_losses(probs: NDArray[np.float64], y: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
    picked = np.clip(probs[np.arange(len(y)), y], EPS, None)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return {
        "nll": -np.log(picked),
        "brier": np.sum((probs - onehot) ** 2, axis=1),
    }


def _t2_losses(p: NDArray[np.float64], y: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
    pc = np.clip(p, EPS, 1 - EPS)
    y_f = y.astype(np.float64)
    return {
        "nll": -(y_f * np.log(pc) + (1 - y_f) * np.log(1 - pc)),
        "brier": (p - y_f) ** 2,
    }


def _cis(
    losses: dict[str, NDArray[np.float64]],
    match_ids: pl.Series,
    draws: NDArray[np.float64],
) -> dict[str, CI]:
    out = {}
    for metric, vec in losses.items():
        sums, counts = per_match_losses(vec, match_ids)
        out[metric] = bootstrap_metric(sums, counts, draws)
    return out


def _bucket_table(
    df: pl.DataFrame, p: NDArray[np.float64], y: NDArray[np.int64], fmt: str
) -> list[dict[str, Any]]:
    """T2 calibration by bucket: innings x phase; wickets bands; last-30-balls."""
    from evalkit.models.b1_table import PHASE_EDGES, WICKET_EDGES

    phase = np.searchsorted(np.array(PHASE_EDGES[fmt]), df["legal_balls"].to_numpy(), "right")
    innings = (df["is_chase"].to_numpy() >= 1.0).astype(int) + 1
    wkts = np.searchsorted(np.array(WICKET_EDGES), df["wickets"].to_numpy(), "right")
    last30 = df["balls_remaining"].to_numpy() <= 30
    rows = []
    wkt_names = ("0-1", "2-3", "4-5", "6+")
    for inn in (1, 2):
        for ph in range(4):
            m = (innings == inn) & (phase == ph)
            if m.sum():
                rows.append(_bucket_row(f"innings{inn}/phase{ph + 1}", p[m], y[m]))
    for band in range(4):
        m = wkts == band
        if m.sum():
            rows.append(_bucket_row(f"wickets {wkt_names[band]}", p[m], y[m]))
    if last30.sum():
        rows.append(_bucket_row("last 30 balls", p[last30], y[last30]))
    return rows


def _bucket_row(name: str, p: NDArray[np.float64], y: NDArray[np.int64]) -> dict[str, Any]:
    return {
        "bucket": name,
        "n": len(p),
        "p_mean": float(np.mean(p)),
        "y_mean": float(np.mean(y)),
        "ece": ece(p, y, n_bins=min(20, max(2, len(p) // 500))),
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()


def run_cell(task: str, fmt: str, bundle: DataBundle, cold: bool) -> dict[str, Any]:
    train = bundle.load_split(task, fmt, "train")
    val = bundle.load_split(task, fmt, "val")
    test = bundle.load_split(task, fmt, "test", allow_test=True)  # the single touch
    y_val, y_test = to_y(val), to_y(test)
    n_val_matches = val["match_id"].n_unique()
    n_test_matches = test["match_id"].n_unique()
    draws_test = make_draws(n_test_matches)
    draws_val = make_draws(n_val_matches)

    cell: dict[str, Any] = {
        "n_train_rows": train.height,
        "n_val_rows": val.height,
        "n_test_rows": test.height,
        "n_val_matches": n_val_matches,
        "n_test_matches": n_test_matches,
        "models": {},
    }
    nll_sums: dict[str, dict[str, NDArray[np.float64]]] = {}

    for name in MODEL_NAMES:
        model, from_cache = _fit_or_load(task, fmt, name, train, val, cold)
        p_val = model.predict_proba(val)
        p_test = model.predict_proba(test)
        entry: dict[str, Any] = {"from_cache": from_cache, "seed": SEED}
        for attr in ("tau", "c", "n_iter"):
            if hasattr(model, attr):
                entry[attr] = getattr(model, attr)

        if task == "t1":
            scaler = fit_temperature(np.log(np.clip(p_val, EPS, None)), y_val)
            cal_val = scaler.apply(np.log(np.clip(p_val, EPS, None)))
            cal_test = scaler.apply(np.log(np.clip(p_test, EPS, None)))
            entry["calibration"] = {"method": "temperature", "T": round(scaler.temperature, 4)}
            losses_val, losses_test = _t1_losses(cal_val, y_val), _t1_losses(cal_test, y_test)
            entry["ece_val"] = float(np.mean(ece_per_class(cal_val, y_val)))
            entry["ece_test"] = float(np.mean(ece_per_class(cal_test, y_test)))
            entry["max_ce_test"] = float(np.max(ece_per_class(cal_test, y_test)))
            conf = cal_test.max(axis=1)
            correct = (cal_test.argmax(axis=1) == y_test).astype(np.int64)
            entry["_reliability"] = reliability_data(conf, correct)
        else:
            platt = fit_platt(p_val, y_val)
            iso = fit_isotonic(p_val, y_val)
            method = t2_leaderboard_calibration(n_val_matches)
            calibrator = platt if method == "platt" else iso
            cal_val, cal_test = calibrator.apply(p_val), calibrator.apply(p_test)
            entry["calibration"] = {
                "method": method,
                "alternate": {
                    "isotonic_test_nll": float(
                        np.mean(_t2_losses(iso.apply(p_test), y_test)["nll"])
                    ),
                    "platt_test_nll": float(
                        np.mean(_t2_losses(platt.apply(p_test), y_test)["nll"])
                    ),
                },
            }
            losses_val, losses_test = _t2_losses(cal_val, y_val), _t2_losses(cal_test, y_test)
            entry["ece_val"] = ece(cal_val, y_val)
            entry["ece_test"] = ece(cal_test, y_test)
            entry["max_ce_test"] = max_ce(cal_test, y_test)
            entry["_reliability"] = reliability_data(cal_test, y_test)
            entry["buckets"] = _bucket_table(test, cal_test, y_test, fmt)

        entry["val"] = _cis(losses_val, val["match_id"], draws_val)
        entry["test"] = _cis(losses_test, test["match_id"], draws_test)
        nll_sums[name] = {
            "val_s": per_match_losses(losses_val["nll"], val["match_id"])[0],
            "val_n": per_match_losses(losses_val["nll"], val["match_id"])[1],
            "test_s": per_match_losses(losses_test["nll"], test["match_id"])[0],
            "test_n": per_match_losses(losses_test["nll"], test["match_id"])[1],
        }
        # per-season (calendar year) test NLL for drift exposure
        years = (
            test.select("match_id")
            .join(bundle.matches.select("match_id", "start_date"), on="match_id", how="left")[
                "start_date"
            ]
            .dt.year()
            .to_numpy()
        )
        entry["by_season"] = {
            int(yr): float(np.mean(losses_test["nll"][years == yr])) for yr in np.unique(years)
        }
        cell["models"][name] = entry

    # frozen §6 decision rule, demonstrated: B3 as challenger vs best of B0-B2
    best_baseline = min(
        ("B0_marginal", "B1_table", "B2_logistic"),
        key=lambda n: cell["models"][n]["test"]["nll"].point,
    )
    ch, bl = nll_sums["B3_gbm"], nll_sums[best_baseline]
    delta_val = bootstrap_paired_delta(ch["val_s"], bl["val_s"], ch["val_n"], draws_val)
    delta_test = bootstrap_paired_delta(ch["test_s"], bl["test_s"], ch["test_n"], draws_test)
    rel_improvement = -delta_test.point / cell["models"][best_baseline]["test"]["nll"].point
    ece_worsening = cell["models"]["B3_gbm"]["ece_test"] - cell["models"][best_baseline]["ece_test"]
    cell["decision_rule"] = {
        "challenger": "B3_gbm",
        "baseline": best_baseline,
        "delta_nll_val": delta_val,
        "delta_nll_test": delta_test,
        "clause1_ci_excludes_zero_both": bool(
            delta_val.point < 0
            and delta_test.point < 0
            and delta_val.excludes_zero()
            and delta_test.excludes_zero()
        ),
        "clause2_rel_improvement": float(rel_improvement),
        "clause2_pass": bool(rel_improvement >= 0.005),
        "clause3_ece_worsening": float(ece_worsening),
        "clause3_pass": bool(ece_worsening <= 0.005),
    }
    cell["decision_rule"]["beats_bar"] = bool(
        cell["decision_rule"]["clause1_ci_excludes_zero_both"]
        and cell["decision_rule"]["clause2_pass"]
        and cell["decision_rule"]["clause3_pass"]
    )
    return cell


def run_all(cold: bool = False) -> dict[str, Any]:
    bundle = DataBundle()
    check_integrity(bundle.matches, bundle.deliveries)
    report: dict[str, Any] = {"_meta": {}}
    for task in ("t1", "t2"):
        for fmt in ("t20", "odi"):
            print(f"[run-all] {task}/{fmt}", flush=True)
            report[f"{task}/{fmt}"] = run_cell(task, fmt, bundle, cold)
    meta = split_metadata(bundle.matches, bundle.deliveries)
    train_del = bundle.deliveries.filter(
        (pl.col("temporal_split") == "train") & ~pl.col("excluded_from_tuples")
    )
    freq = train_del.group_by("outcome_class").len().sort("len", descending=True)
    report["_meta"] = {
        "split_lines": metadata_lines(meta),
        "class_freq": freq.to_dicts(),
        "git_sha": _git_sha(),
        "corpus_hash": PINNED_CORPUS_HASH,
        "seed": SEED,
        "bootstrap": {"B": B_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }
    render(report)
    return report


def _fmt_ci(ci: CI) -> str:
    return f"{ci.point:.5f} [{ci.lo:.5f}, {ci.hi:.5f}]"


# Frozen record of the Branch A gate experiment (docs/BRANCH_A_REPORT.md).
# These numbers come from Branch A's own single test touch — folding them in
# here re-renders that record; it does not re-evaluate the test split.
_BRANCH_A_INCREMENT = [
    "### Shipped increment — Branch A (M_shrunk = B3 + shrunk player effects)",
    "",
    "| model | test NLL [95% CI] | ΔNLL vs B3 [95% CI] | rel | bits/ball | tuned |",
    "|---|---|---|---|---|---|",
    "| M_shrunk | 1.60934 [1.60360, 1.61489] | -0.00504 [-0.00561, -0.00449] "
    "| +0.31% | 0.00727 | λ=1600 (val) |",
    "",
    "Player identity (striker + bowler, train-only empirical-Bayes shrinkage)",
    "measured under the frozen Branch A protocol: real signal (CI excludes 0),",
    "economically negligible — verdict **AMBIGUOUS** at the band floor.",
    "Per the frozen rule: the cheap increment ships; the hierarchical modeling",
    "tower (Branches B/C) was **declined on evidence**. Canary PASS; dilution:",
    "5.25% null-striker, 14.45% unseen striker, 19.15% unseen bowler.",
    "Full protocol and per-class breakdown: docs/BRANCH_A_REPORT.md.",
    "",
]


def render(report: dict[str, Any]) -> None:
    meta = report["_meta"]
    lines = [
        "# LEADERBOARD — cricstate M2 baselines (frozen release)",
        "",
        f"corpus `v1.2` hash `{meta['corpus_hash'][:16]}…` · git `{meta['git_sha']}` · "
        f"seed {meta['seed']} · bootstrap B={meta['bootstrap']['B']} "
        f"(match-level, paired, seed {meta['bootstrap']['seed']})",
        "",
        "All numbers are POST-calibration (the shipped predictor): T1 temperature,",
        "T2 isotonic (t20) / Platt (odi thin cell, amendment #1), maps fit on val",
        "only. Test split evaluated exactly once, in this release.",
        "",
        "## Splits",
        "",
        *meta["split_lines"],
        "",
    ]
    for cell_key in ("t1/t20", "t1/odi", "t2/t20", "t2/odi"):
        cell = report[cell_key]
        thin = " ⚠ THIN CELL" if cell["n_test_matches"] < 300 else ""
        lines += [
            f"## {cell_key} — test n = {cell['n_test_matches']} matches / "
            f"{cell['n_test_rows']} deliveries{thin}",
            "",
            "| model | test NLL [95% CI] | test Brier [95% CI] | test ECE | max-CE | "
            "skill vs B0 | val NLL [95% CI] | calibration | tuned |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        b0_nll = cell["models"]["B0_marginal"]["test"]["nll"].point
        for name in MODEL_NAMES:
            e = cell["models"][name]
            tuned = []
            if "tau" in e:
                tuned.append(f"τ={e['tau']:g}")
            if "c" in e:
                tuned.append(f"C={e['c']:g}")
            if "n_iter" in e:
                tuned.append(f"iters={e['n_iter']}")
            cal = e["calibration"]["method"]
            if cal == "temperature":
                cal += f" (T={e['calibration']['T']})"
            skill = 1.0 - e["test"]["nll"].point / b0_nll
            lines.append(
                f"| {name} | {_fmt_ci(e['test']['nll'])} | {_fmt_ci(e['test']['brier'])} "
                f"| {e['ece_test']:.5f} | {e['max_ce_test']:.5f} | {skill:+.4f} "
                f"| {_fmt_ci(e['val']['nll'])} | {cal} | {', '.join(tuned) or '—'} |"
            )
        d = cell["decision_rule"]
        lines += [
            "",
            f"**Frozen §6 decision rule** (challenger {d['challenger']} vs best baseline "
            f"{d['baseline']}): ΔNLL test {_fmt_ci(d['delta_nll_test'])}, "
            f"val {_fmt_ci(d['delta_nll_val'])}; "
            f"clause 1 (CI<0 both) {'PASS' if d['clause1_ci_excludes_zero_both'] else 'FAIL'}; "
            f"clause 2 (rel ≥ 0.5%) {d['clause2_rel_improvement']:.4%} "
            f"{'PASS' if d['clause2_pass'] else 'FAIL'}; "
            f"clause 3 (ΔECE ≤ 0.005) {d['clause3_ece_worsening']:+.5f} "
            f"{'PASS' if d['clause3_pass'] else 'FAIL'} → "
            f"**{'BEATS BAR' if d['beats_bar'] else 'DID NOT BEAT THE BAR'}**",
            "",
            "Per-season test NLL (drift):",
            "",
            "| model | "
            + " | ".join(str(y) for y in sorted(next(iter(cell["models"].values()))["by_season"]))
            + " |",
            "|---|" + "---|" * len(next(iter(cell["models"].values()))["by_season"]),
        ]
        for name in MODEL_NAMES:
            seasons = cell["models"][name]["by_season"]
            lines.append(
                f"| {name} | " + " | ".join(f"{seasons[y]:.5f}" for y in sorted(seasons)) + " |"
            )
        lines.append("")
        if cell_key == "t1/t20":
            lines += _BRANCH_A_INCREMENT

        if cell_key.startswith("t2"):
            lines += ["### Calibration by bucket (test, shipped calibration)", ""]
            buckets0 = cell["models"]["B3_gbm"]["buckets"]
            header = "| bucket | n | " + " | ".join(f"{n} p̄/ȳ (ECE)" for n in MODEL_NAMES) + " |"
            lines += [header, "|---|---|" + "---|" * len(MODEL_NAMES)]
            for i, row in enumerate(buckets0):
                cells = []
                for name in MODEL_NAMES:
                    b = cell["models"][name]["buckets"][i]
                    cells.append(f"{b['p_mean']:.3f}/{b['y_mean']:.3f} ({b['ece']:.3f})")
                lines.append(f"| {row['bucket']} | {row['n']} | " + " | ".join(cells) + " |")
            lines.append("")
    freq_total = sum(r["len"] for r in meta["class_freq"])
    lines += [
        "## T1 class frequencies (train, frozen K=11 alphabet)",
        "",
        "| class | rows | freq |",
        "|---|---|---|",
        *(
            f"| {r['outcome_class']} | {r['len']} | {r['len'] / freq_total:.6f} |"
            for r in meta["class_freq"]
        ),
        "",
        "Alphabet order (frozen): " + ", ".join(CLASSES),
        "",
        "## STATS block",
        "",
        "```",
        f"corpus_hash   {meta['corpus_hash']}",
        f"git_sha       {meta['git_sha']}",
        f"seed          {meta['seed']}",
        f"bootstrap     B={meta['bootstrap']['B']} seed={meta['bootstrap']['seed']} "
        "(match-level, paired)",
        "val-only constants: B1 tau; B2 C; B3 early-stop iters; calibration maps",
        "test split: evaluated once, in this release",
        "```",
    ]
    LEADERBOARD.write_text("\n".join(lines) + "\n")
    # reliability PNGs (not part of the byte-identity contract)
    for cell_key in ("t1/t20", "t1/odi", "t2/t20", "t2/odi"):
        cell = report[cell_key]
        for name in MODEL_NAMES:
            e = cell["models"][name]
            out = PLOTS / f"{cell_key.replace('/', '_')}_{name}.png"
            plot_reliability(
                e["_reliability"],
                out,
                title=f"{cell_key} {name} (test)",
                ece_value=e["ece_test"],
            )
    print(f"wrote {LEADERBOARD} and {PLOTS}/", flush=True)
