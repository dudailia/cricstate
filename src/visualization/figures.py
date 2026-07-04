"""The five required publication figures + a state-ladder context figure.

All numbers are read from results/summary.json (the frozen evidence set); no
model is run here. Conditions figures are labelled VALIDATION ONLY because
Branch C is frozen at C1 (no test evaluation).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from visualization import plots

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "results" / "summary.json"
FIGDIR = REPO / "report" / "figures"


def load_summary() -> dict[str, Any]:
    with open(SUMMARY) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _ci_err(point: float, ci: list[float]) -> list[float]:
    return [point - ci[0], ci[1] - point]


def fig_state_ladder(s: dict[str, Any]) -> None:
    """Context: the B0->B3 state ladder (test NLL, lower is better)."""
    ladder = s["state_ladder_t1_t20_test"]
    names = ["B0_marginal", "B1_table", "B2_logistic", "B3_gbm"]
    labels = ["B0 marginal", "B1 table", "B2 logistic", "B3 state"]
    vals = [ladder[n]["nll"] for n in names]
    errs = np.array([_ci_err(ladder[n]["nll"], ladder[n]["ci"]) for n in names]).T
    fig, ax = plots.new_axes(
        "Match state captures the recoverable signal",
        "t20 · per-ball outcome NLL on the test split (lower is better) · 95% CI",
    )
    y = np.arange(len(names))[::-1]
    # Dot + CI, not bars: position (not length-from-zero) encodes the value, so a
    # zoomed axis that reveals the small, real gaps does not imply a false ratio.
    ax.errorbar(
        vals, y, xerr=errs, fmt="none", ecolor=plots.INK, elinewidth=1.2, capsize=3, zorder=3
    )
    ax.scatter(
        vals, y, c=plots.GREEN_RAMP, s=120, zorder=4, edgecolors=plots.SURFACE, linewidths=1.2
    )
    for yi, n in zip(y, names, strict=True):
        hi = ladder[n]["ci"][1]
        ax.text(
            hi + 0.004,
            yi,
            f"{ladder[n]['nll']:.3f}",
            va="center",
            ha="left",
            color=plots.INK,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(1.60, 1.715)
    ax.set_xlabel("test NLL (nats)")
    plots.footnote(fig, s["meta"])
    plots.save(fig, FIGDIR / "fig1_state_ladder.png")


def fig_identity_delta(s: dict[str, Any]) -> None:
    """Identity vs state: model NLLs with state as the reference line."""
    m = s["identity_t1_t20_test"]["models"]
    order = ["M_flat", "M_shuffled", "M_shrunk", "M_state"]
    labels = ["M_flat\n(unshrunk)", "M_shuffled\n(canary)", "M_shrunk\n(identity)", "M_state\n(B3)"]
    colors = [plots.BALL, plots.GRAY, plots.PITCH, plots.MUTED]
    vals = [m[n]["nll"] for n in order]
    errs = np.array([_ci_err(m[n]["nll"], m[n]["ci"]) for n in order]).T
    fig, ax = plots.new_axes(
        "Player identity vs match state — a 0.31% effect",
        "t20 test NLL · shrunk identity barely beats state; unshrunk overfits; shuffled ≈ null",
    )
    x = np.arange(len(order))
    # Dot + CI, not bars: with M_flat far above a tight cluster, a zoomed dot plot
    # shows both the overfit gap and the near-null shrunk effect honestly.
    state = m["M_state"]["nll"]
    ax.axhline(state, color=plots.MUTED, ls="--", lw=1.0, zorder=2)
    ax.errorbar(
        x, vals, yerr=errs, fmt="none", ecolor=plots.INK, elinewidth=1.2, capsize=3, zorder=3
    )
    ax.scatter(x, vals, c=colors, s=130, zorder=4, edgecolors=plots.SURFACE, linewidths=1.2)
    for xi, v in zip(x, vals, strict=True):
        ax.text(xi + 0.14, v, f"{v:.3f}", va="center", ha="left", color=plots.INK, fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(1.58, 1.80)
    ax.set_ylabel("test NLL (nats)")
    d = s["identity_t1_t20_test"]
    ax.text(
        0.985,
        0.62,
        f"ΔNLL vs state {d['paired_deltas']['dNLL_shrunk_vs_state']['point']:+.5f}\n"
        f"[{d['paired_deltas']['dNLL_shrunk_vs_state']['ci'][0]:.5f}, "
        f"{d['paired_deltas']['dNLL_shrunk_vs_state']['ci'][1]:.5f}]\n"
        f"= {d['effect_size']['relative_nll_pct']:.2f}% · "
        f"{d['effect_size']['bits_per_ball']:.4f} bits/ball\nverdict {d['verdict']}",
        transform=ax.transAxes,
        fontsize=8.6,
        color=plots.INK,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.4", "fc": plots.SURFACE, "ec": plots.LINE},
    )
    plots.footnote(fig, s["meta"])
    plots.save(fig, FIGDIR / "fig2_identity_delta.png")


def fig_conditions_val(s: dict[str, Any]) -> None:
    """Conditions vs state: the validation kappa curve (test never evaluated)."""
    c = s["conditions_t1_t20"]["val"]
    grid = np.array(c["kappa_grid"])
    curve = np.array(c["val_nll_curve"])
    fig, ax = plots.new_axes(
        "Per-match conditions latent — a 0.024% effect on validation",
        "t20 · causal residual latent · VALIDATION ONLY (Branch C frozen at C1, no test touch)",
    )
    ax.grid(axis="y", color=plots.GRID, linewidth=0.8, zorder=0)
    ax.plot(
        grid,
        curve,
        color=plots.PITCH,
        lw=2.0,
        marker="o",
        ms=6,
        zorder=3,
        label="M_latent (val NLL)",
    )
    best_i = int(np.argmin(curve))
    ax.scatter(
        [grid[best_i]],
        [curve[best_i]],
        s=90,
        facecolor="white",
        edgecolor=plots.PITCH,
        lw=2,
        zorder=5,
    )
    ax.axhline(c["M_state_nll"], color=plots.MUTED, ls="--", lw=1.2, zorder=2)
    ax.text(
        grid[-1],
        c["M_state_nll"],
        "M_state (B3) val NLL",
        va="bottom",
        ha="right",
        color=plots.MUTED,
        fontsize=8.5,
    )
    ax.set_xscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([f"{g:.0f}" for g in grid])
    ax.set_xlabel("κ  (prior strength — the only tunable)")
    ax.set_ylabel("validation NLL (nats)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.05,
        f"best κ = {c['best_kappa']:.0f} · val ΔNLL {c['dNLL_full_vs_state']:+.5f} = "
        f"{c['relative_nll_pct']:.3f}% · {c['bits_per_ball']:.4f} bits/ball\n"
        f"carry-over vs per-innings: "
        f"{c['carryover_effect_full_minus_innings']:+.5f} nats (negligible)",
        transform=ax.transAxes,
        fontsize=8.4,
        color=plots.INK,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "fc": plots.SURFACE, "ec": plots.LINE},
    )
    plots.footnote(
        fig, s["meta"], extra="Conditions arm is partial (C1): validation evidence only."
    )
    plots.save(fig, FIGDIR / "fig3_conditions_val.png")


def fig_residual_decomposition(s: dict[str, Any]) -> None:
    """Information decomposition: nats extracted above the marginal floor."""
    d = s["conclusion"]["information_decomposition_nats"]
    segs = [
        ("state (B3)", d["captured_by_state_B3"], plots.PITCH, "test"),
        ("+ identity", d["added_by_identity_test"], plots.SAGE, "test"),
        ("+ conditions", d["added_by_conditions_val"], plots.AMBER, "val"),
    ]
    fig, ax = plots.new_axes(
        "Information decomposition — state extracts almost all of it",
        "t20 · NLL reduction below the B0 marginal floor, in nats · bars to scale",
    )
    total = sum(v for _, v, _, _ in segs)
    y = np.arange(len(segs))[::-1]
    for yi, (name, v, color, fold) in zip(y, segs, strict=True):
        ax.barh(yi, v, color=color, height=0.62, zorder=3)
        tag = "" if fold == "test" else " [val]"
        ax.text(
            v + total * 0.015,
            yi,
            f"{name}: {v:.4f} nats · {100 * v / total:.1f}%{tag}",
            va="center",
            ha="left",
            fontsize=9.6,
            color=plots.INK,
        )
    ax.set_xlim(0, total * 1.5)
    ax.set_ylim(-0.6, len(segs) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("nats extracted above the B0 marginal floor")
    ax.text(
        0.99,
        0.06,
        f"B0 marginal floor = {d['no_information_floor_B0']:.3f} nats  ·  "
        "identity & conditions are the near-invisible slivers, by design",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.3,
        color=plots.MUTED,
    )
    plots.footnote(fig, s["meta"])
    plots.save(fig, FIGDIR / "fig4_residual_decomposition.png")


def fig_effect_sizes(s: dict[str, Any]) -> None:
    """Effect-size bar chart in bits/ball, against the 1%-improvement bar."""
    ident = s["identity_t1_t20_test"]["effect_size"]["bits_per_ball"]
    cond = s["conditions_t1_t20"]["val"]["bits_per_ball"]
    state_nll = s["state_ladder_t1_t20_test"]["B3_gbm"]["nll"]
    thresh_bits = 0.01 * state_nll / np.log(2)  # 1% relative NLL, in bits
    fig, ax = plots.new_axes(
        "Effect sizes are far below the decision bar",
        "t20 · information gain over state, bits per ball · pre-registered bar: ≥ 1% relative NLL",
    )
    names = ["identity\n(test)", "conditions\n(val)"]
    vals = [ident, cond]
    colors = [plots.PITCH, plots.AMBER]
    x = np.arange(2)
    ax.bar(x, vals, color=colors, width=0.5, zorder=3)
    for xi, v in zip(x, vals, strict=True):
        ax.text(
            xi,
            v + thresh_bits * 0.02,
            f"{v:.4f}",
            va="bottom",
            ha="center",
            color=plots.INK,
            fontsize=10,
        )
    ax.axhline(thresh_bits, color=plots.BALL, ls="--", lw=1.4, zorder=2)
    ax.text(
        1.45,
        thresh_bits,
        f"  1% bar ≈ {thresh_bits:.4f} bits/ball",
        va="center",
        ha="left",
        color=plots.BALL,
        fontsize=8.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, thresh_bits * 1.25)
    ax.set_ylabel("bits / ball")
    plots.footnote(fig, s["meta"], extra="Conditions bar is validation-only (Branch C partial).")
    plots.save(fig, FIGDIR / "fig5_effect_sizes.png")


PER_CLASS_GROUPS = [
    ("dot", ["0"]),
    ("single", ["1"]),
    ("boundary", ["4", "6"]),
    ("wicket", ["wicket"]),
    ("wide", ["wide"]),
]


def fig_calibration(s: dict[str, Any]) -> None:
    """Per-class calibration (ECE): state vs +identity on test."""
    ece = s["identity_t1_t20_test"]["per_class_ece"]
    labels, state_v, shrunk_v = [], [], []
    for name, members in PER_CLASS_GROUPS:
        labels.append(name)
        state_v.append(float(np.mean([ece[m]["state"] for m in members])))
        shrunk_v.append(float(np.mean([ece[m]["shrunk"] for m in members])))
    fig, ax = plots.new_axes(
        "Per-class calibration error (ECE) — state vs + identity",
        "t20 test · 20 equal-mass bins · boundary = mean(4,6) · conditions not evaluated on test",
    )
    ax.grid(axis="y", color=plots.GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(labels))
    w = 0.36  # slot 0.40 with a surface gap between paired bars
    ax.bar(x - 0.20, state_v, w, color=plots.MUTED, label="B3 state", zorder=3)
    ax.bar(x + 0.20, shrunk_v, w, color=plots.PITCH, label="+ identity", zorder=3)
    for xi, v in zip(x - 0.20, state_v, strict=True):
        ax.text(
            xi, v + 0.0002, f"{v:.3f}", va="bottom", ha="center", fontsize=7.6, color=plots.MUTED
        )
    for xi, v in zip(x + 0.20, shrunk_v, strict=True):
        ax.text(
            xi, v + 0.0002, f"{v:.3f}", va="bottom", ha="center", fontsize=7.6, color=plots.PITCH
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ECE (lower is better)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    plots.footnote(fig, s["meta"])
    plots.save(fig, FIGDIR / "fig6_calibration_perclass.png")


ALL_FIGURES = [
    fig_state_ladder,
    fig_identity_delta,
    fig_conditions_val,
    fig_residual_decomposition,
    fig_effect_sizes,
    fig_calibration,
]


def generate_all() -> list[str]:
    plots.apply_style()
    s = load_summary()
    for fn in ALL_FIGURES:
        fn(s)
    return sorted(p.name for p in FIGDIR.glob("*.png"))
