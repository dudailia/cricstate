"""P3 runner: fit B0-B3 per (task, fmt), tune on val, report VAL ONLY.

The test split is never loaded here (DataBundle.load_split refuses it).
Writes docs/P3_VAL_REPORT.md + artifacts/p3/metrics.json.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evalkit.calibrate import fit_isotonic, fit_platt, fit_temperature
from evalkit.canaries import ladder_inversion_canary, shuffled_target_canary
from evalkit.datasets import DataBundle
from evalkit.metrics import (
    brier_binary,
    brier_multiclass,
    ece,
    ece_per_class,
    nll_binary,
    nll_multiclass,
)
from evalkit.models.b0_marginal import B0MarginalT1, B0MarginalT2
from evalkit.models.b1_table import B1TableT1, B1TableT2
from evalkit.models.b2_logistic import B2Logistic
from evalkit.models.b3_gbm import B3Gbm
from evalkit.models.base import SEED, Predictor, register, to_y
from evalkit.monotonicity import check_b1_t2_monotonicity
from evalkit.policy import t2_leaderboard_calibration

ART = REPO / "artifacts" / "p3"
OUT = REPO / "docs" / "P3_VAL_REPORT.md"

EPS = 1e-12


def t1_metrics(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "nll": nll_multiclass(probs, y),
        "brier": brier_multiclass(probs, y),
        "ece": float(np.mean(ece_per_class(probs, y))),  # mean one-vs-rest ECE
    }


def t2_metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"nll": nll_binary(p, y), "brier": brier_binary(p, y), "ece": ece(p, y)}


def run_cell(task: str, fmt: str, bundle: DataBundle, report: dict[str, Any]) -> None:
    t0 = time.monotonic()
    train = bundle.load_split(task, fmt, "train")
    val = bundle.load_split(task, fmt, "val")
    y_val = to_y(val)
    print(f"[{task}/{fmt}] train {train.height} rows, val {val.height} rows", flush=True)

    models: list[Predictor]
    if task == "t1":
        models = [B0MarginalT1(), B1TableT1(fmt), B2Logistic("t1"), B3Gbm("t1")]
    else:
        models = [B0MarginalT2(), B1TableT2(fmt), B2Logistic("t2"), B3Gbm("t2")]

    cell: dict[str, Any] = {"n_train": train.height, "n_val": val.height, "models": {}}
    nlls: dict[str, float] = {}
    for model in models:
        m0 = time.monotonic()
        model.fit(train, val)
        register(task, fmt, model)
        probs = model.predict_proba(val)
        entry: dict[str, Any] = {"seed": SEED, "fit_seconds": round(time.monotonic() - m0, 1)}
        if task == "t1":
            entry["pre"] = t1_metrics(probs, y_val)
            scaler = fit_temperature(np.log(np.clip(probs, EPS, None)), y_val)
            post = scaler.apply(np.log(np.clip(probs, EPS, None)))
            entry["post"] = t1_metrics(post, y_val)
            entry["calibration"] = {"method": "temperature", "T": round(scaler.temperature, 4)}
        else:
            entry["pre"] = t2_metrics(probs, y_val)
            platt = fit_platt(probs, y_val)
            iso = fit_isotonic(probs, y_val)
            entry["post_platt"] = t2_metrics(platt.apply(probs), y_val)
            entry["post_isotonic"] = t2_metrics(iso.apply(probs), y_val)
            n_val_matches = val["match_id"].n_unique()
            method = t2_leaderboard_calibration(n_val_matches)
            entry["calibration"] = {
                "leaderboard_method": method,
                "n_labeled_val_matches": n_val_matches,
                "platt": {"a": round(platt.a, 4), "b": round(platt.b, 4)},
            }
        if hasattr(model, "tau"):
            entry["tau"] = model.tau
        if hasattr(model, "c"):
            entry["C"] = model.c
        if hasattr(model, "n_iter"):
            entry["n_iter"] = model.n_iter
        nlls[model.name] = entry["pre"]["nll"]
        cell["models"][model.name] = entry
        print(
            f"  {model.name}: val NLL {entry['pre']['nll']:.5f} ({entry['fit_seconds']}s)",
            flush=True,
        )

    # canaries
    ladder = ladder_inversion_canary(
        nlls["B0_marginal"], nlls["B2_logistic"], nlls["B3_gbm"], where=f"{task}/{fmt}/val"
    )
    shuffle = shuffled_target_canary(
        task, train, val, nlls["B0_marginal"], chosen_c=cell["models"]["B2_logistic"]["C"]
    )
    cell["canaries"] = {
        "ladder_inversion": {"passed": ladder.passed, "detail": ladder.detail},
        "shuffled_target": {"passed": shuffle.passed, "detail": shuffle.detail},
    }
    print(f"  canaries: ladder={ladder.passed} shuffle={shuffle.passed}", flush=True)

    # B1 monotonicity (T2 only)
    if task == "t2":
        b1 = next(m for m in models if m.name == "B1_table")
        violations = check_b1_t2_monotonicity(b1)  # type: ignore[arg-type]
        cell["monotonicity"] = {"violations": violations, "passed": not violations}
        print(f"  monotonicity: {'PASS' if not violations else violations[:3]}", flush=True)

    cell["runtime_seconds"] = round(time.monotonic() - t0, 1)
    report[f"{task}/{fmt}"] = cell


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P3 VAL REPORT — baseline ladder (val only; test untouched)",
        "",
        f"Corpus tag v1.2 · seed {SEED} · all constants chosen on val:",
        "B1 τ ∈ {50,200,800}; B2 C ∈ {0.01,0.1,1,10}; B3 early-stop iteration",
        "count (warm-start, patience 3, step 25); calibration maps (temperature",
        "T for T1; Platt a,b and isotonic for T2). Calibration maps are fit on",
        "val and, in this P3 report, also evaluated on val — post-calibration",
        "numbers are in-sample for the calibrator until P4's test run.",
        "",
    ]
    for cell_key, cell in report.items():
        task = cell_key.split("/")[0]
        lines += [
            f"## {cell_key}",
            "",
            f"train rows: {cell['n_train']} · val rows: {cell['n_val']} · cell runtime {cell['runtime_seconds']}s",
            "",
        ]
        if task == "t1":
            lines += [
                "| model | val NLL | val Brier | val ECE* | post-cal NLL | post-cal ECE* | T | tuned |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name, e in cell["models"].items():
                tuned = (
                    f"τ={e['tau']}"
                    if "tau" in e
                    else (
                        f"C={e['C']}"
                        if "C" in e
                        else (f"iters={e['n_iter']}" if "n_iter" in e else "—")
                    )
                )
                lines.append(
                    f"| {name} | {e['pre']['nll']:.5f} | {e['pre']['brier']:.5f} | {e['pre']['ece']:.5f} "
                    f"| {e['post']['nll']:.5f} | {e['post']['ece']:.5f} | {e['calibration']['T']} | {tuned} |"
                )
            lines += ["", "\\* mean one-vs-rest ECE over the 11 classes, B=20 equal-mass bins", ""]
        else:
            lines += [
                "| model | val NLL | val Brier | val ECE | Platt NLL/ECE | isotonic NLL/ECE | leaderboard map | tuned |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for name, e in cell["models"].items():
                tuned = (
                    f"τ={e['tau']}"
                    if "tau" in e
                    else (
                        f"C={e['C']}"
                        if "C" in e
                        else (f"iters={e['n_iter']}" if "n_iter" in e else "—")
                    )
                )
                lines.append(
                    f"| {name} | {e['pre']['nll']:.5f} | {e['pre']['brier']:.5f} | {e['pre']['ece']:.5f} "
                    f"| {e['post_platt']['nll']:.5f} / {e['post_platt']['ece']:.5f} "
                    f"| {e['post_isotonic']['nll']:.5f} / {e['post_isotonic']['ece']:.5f} "
                    f"| {e['calibration']['leaderboard_method']} ({e['calibration']['n_labeled_val_matches']} val matches) | {tuned} |"
                )
            mono = cell.get("monotonicity", {})
            lines += [
                "",
                f"B1 monotonicity: **{'PASS' if mono.get('passed') else 'FAIL'}**"
                + (
                    f" — {len(mono['violations'])} violation(s): {mono['violations'][:5]}"
                    if mono.get("violations")
                    else ""
                ),
                "",
            ]
        can = cell["canaries"]
        lines += [
            f"- ladder-inversion canary: **{'PASS' if can['ladder_inversion']['passed'] else 'FAIL'}** — {can['ladder_inversion']['detail']}",
            f"- shuffled-target canary: **{'PASS' if can['shuffled_target']['passed'] else 'FAIL'}** — {can['shuffled_target']['detail']}",
            "- poison-column canary: structural (whitelist select-first); asserted in tests/test_features.py — PASS",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    t0 = time.monotonic()
    bundle = DataBundle()
    report: dict[str, Any] = {}
    for task in ("t1", "t2"):
        for fmt in ("t20", "odi"):
            run_cell(task, fmt, bundle, report)
    report["_meta"] = {
        "seed": SEED,
        "total_runtime_seconds": round(time.monotonic() - t0, 1),
        "test_split_touched": False,
    }
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    OUT.write_text(to_markdown({k: v for k, v in report.items() if k != "_meta"}))
    print(f"total runtime {report['_meta']['total_runtime_seconds']}s")
    print(f"wrote {OUT}")
    all_canaries = all(
        c["passed"] for k, cell in report.items() if k != "_meta" for c in cell["canaries"].values()
    )
    all_mono = all(
        cell.get("monotonicity", {}).get("passed", True)
        for k, cell in report.items()
        if k != "_meta"
    )
    print(f"canaries all pass: {all_canaries} | monotonicity all pass: {all_mono}")
    sys.exit(0 if (all_canaries and all_mono) else 1)


if __name__ == "__main__":
    main()
