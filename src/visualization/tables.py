"""Markdown tables for the paper, rendered from results/summary.json."""

from pathlib import Path
from typing import Any

from visualization.figures import PER_CLASS_GROUPS, load_summary

REPO = Path(__file__).resolve().parents[2]
TABLEDIR = REPO / "report" / "tables"


def _ci(point: float, ci: list[float]) -> str:
    return f"{point:.5f} [{ci[0]:.5f}, {ci[1]:.5f}]"


def table_state_ladder(s: dict[str, Any]) -> str:
    ladder = s["state_ladder_t1_t20_test"]
    rows = ["| model | kind | test NLL [95% CI] |", "|---|---|---|"]
    for n in ("B0_marginal", "B1_table", "B2_logistic", "B3_gbm"):
        e = ladder[n]
        rows.append(f"| {n} | {e['kind']} | {_ci(e['nll'], e['ci'])} |")
    return "\n".join(rows)


def table_identity(s: dict[str, Any]) -> str:
    d = s["identity_t1_t20_test"]
    rows = ["| model | test NLL [95% CI] | note |", "|---|---|---|"]
    for n in ("M_state", "M_shrunk", "M_flat", "M_shuffled"):
        e = d["models"][n]
        rows.append(f"| {n} | {_ci(e['nll'], e['ci'])} | {e.get('note', '')} |")
    dd = d["paired_deltas"]["dNLL_shrunk_vs_state"]
    rows += [
        "",
        f"Primary ΔNLL (shrunk minus state): **{_ci(dd['point'], dd['ci'])}** "
        f"= {d['effect_size']['relative_nll_pct']:.2f}% relative, "
        f"{d['effect_size']['bits_per_ball']:.4f} bits/ball. Verdict: **{d['verdict']}**.",
        "",
        f"Dilution: null-striker {d['dilution_pct']['null_striker']:.2f}%, "
        f"unseen striker {d['dilution_pct']['unseen_striker']:.2f}%, "
        f"unseen bowler {d['dilution_pct']['unseen_bowler']:.2f}%.",
    ]
    return "\n".join(rows)


def table_conditions(s: dict[str, Any]) -> str:
    c = s["conditions_t1_t20"]
    v = c["val"]
    rows = [
        f"_{c['status']}_",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| M_state val NLL | {v['M_state_nll']:.5f} |",
        f"| M_latent (full match) val NLL | {v['M_latent_full_nll']:.5f} |",
        f"| M_latent (per-innings) val NLL | {v['M_latent_innings_nll']:.5f} |",
        f"| best κ (val) | {v['best_kappa']:.0f} |",
        f"| val ΔNLL vs state | {v['dNLL_full_vs_state']:+.5f} ({v['relative_nll_pct']:.3f}%) |",
        f"| bits/ball (val) | {v['bits_per_ball']:.4f} |",
        f"| carry-over (full minus per-innings) | "
        f"{v['carryover_effect_full_minus_innings']:+.5f} nats |",
        "",
        f"Scope: {c['scope_caveat']}",
    ]
    return "\n".join(rows)


def table_per_class(s: dict[str, Any]) -> str:
    ece = s["identity_t1_t20_test"]["per_class_ece"]
    rows = ["| class group | B3 state ECE | + identity ECE |", "|---|---|---|"]
    for name, members in PER_CLASS_GROUPS:
        st = sum(ece[m]["state"] for m in members) / len(members)
        sh = sum(ece[m]["shrunk"] for m in members) / len(members)
        rows.append(f"| {name} | {st:.5f} | {sh:.5f} |")
    return "\n".join(rows)


def table_leakage(s: dict[str, Any]) -> str:
    rows = ["| test | scope | result |", "|---|---|---|"]
    for k, v in s["leakage_tests"].items():
        rows.append(f"| {k} | {v['scope']} | **{v['result']}** |")
    return "\n".join(rows)


def generate_all() -> list[str]:
    s = load_summary()
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "state_ladder": table_state_ladder(s),
        "identity": table_identity(s),
        "conditions": table_conditions(s),
        "per_class_calibration": table_per_class(s),
        "leakage": table_leakage(s),
    }
    for name, md in tables.items():
        (TABLEDIR / f"{name}.md").write_text(md + "\n")
    return sorted(tables)
