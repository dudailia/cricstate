"""Census of outcome shapes → docs/EVIDENCE_M2_LABELS.md (P2 milestone, step 1).

Every distinct (result, winner-null, eliminator-null, bowl_out-null, method)
shape in the v1.2 matches table, with counts, each mapped through the REAL
resolution code path in evalkit.labels. An unmapped shape raises — the run
fails loudly rather than emitting a partial census.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalkit.labels import (
    Disposition,
    build_labels,
    disposition_counts,
    labels_hash,
    resolve_one,
)
from evalkit.splits import load_deliveries, load_matches

OUT = Path(__file__).resolve().parents[1] / "docs" / "EVIDENCE_M2_LABELS.md"

RULE_OF = {
    Disposition.EXCLUDE_NO_RESULT: "(1) no-result",
    Disposition.LABELED: "",  # refined below
    Disposition.EXCLUDE_TRUE_TIE: "(4) true tie",
}


def main() -> None:
    matches = load_matches()
    deliveries = load_deliveries()

    shapes = (
        matches.group_by(
            pl.col("outcome_result").fill_null("(won)").alias("result"),
            pl.col("outcome_winner").is_not_null().alias("winner_set"),
            pl.col("outcome_eliminator").is_not_null().alias("eliminator_set"),
            pl.col("outcome_bowl_out").is_not_null().alias("bowl_out_set"),
            pl.col("outcome_method").fill_null("—").alias("method"),
        )
        .len()
        .sort("result", "winner_set", "eliminator_set", "bowl_out_set", "method")
    )

    lines = [
        "# EVIDENCE — T2 label census (M1.2 corpus, schema 1.1.0)",
        "",
        "Every distinct outcome shape in `matches.parquet`, mapped through",
        "`evalkit.labels.resolve_one` (the real code path, not a re-description).",
        "",
        "| result | winner | eliminator | bowl_out | method | matches | rule | disposition |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in shapes.iter_rows(named=True):
        disp, _ = resolve_one(
            match_id="census",
            no_result=row["result"] == "no result",
            outcome_result=None if row["result"] == "(won)" else row["result"],
            outcome_winner="W" if row["winner_set"] else None,
            outcome_eliminator="E" if row["eliminator_set"] else None,
            outcome_bowl_out="B" if row["bowl_out_set"] else None,
            first_batting_team="W",
        )
        if disp is Disposition.EXCLUDE_NO_RESULT:
            rule = "(1) no-result"
        elif disp is Disposition.LABELED and row["winner_set"]:
            rule = "(2) winner set"
        elif disp is Disposition.LABELED:
            rule = "(3) tie + one tie-breaker"
        else:
            rule = "(4) true tie"
        mark = "set" if row["winner_set"] else "null"
        emark = "set" if row["eliminator_set"] else "null"
        bmark = "set" if row["bowl_out_set"] else "null"
        lines.append(
            f"| {row['result']} | {mark} | {emark} | {bmark} | {row['method']} "
            f"| {row['len']} | {rule} | {disp.value} |"
        )

    labels = build_labels(matches, deliveries)
    h1 = labels_hash(labels)
    h2 = labels_hash(build_labels(matches, deliveries))
    counts = disposition_counts(labels)
    base_rates = (
        labels.filter(pl.col("disposition") == Disposition.LABELED.value)
        .group_by("fmt", "temporal_split")
        .agg(pl.col("y").mean().alias("first_batting_win_rate"), pl.len().alias("n"))
        .sort("fmt", "temporal_split")
    )

    lines += [
        "",
        f"Total shapes: {shapes.height}; total matches: {matches.height}; all mapped.",
        "",
        "## Disposition counts per split",
        "",
        "| split | disposition | matches |",
        "|---|---|---|",
        *(
            f"| {r['temporal_split']} | {r['disposition']} | {r['len']} |"
            for r in counts.iter_rows(named=True)
        ),
        "",
        "## First-batting-team win base rates (LABELED only)",
        "",
        "| fmt | split | n | base rate |",
        "|---|---|---|---|",
        *(
            f"| {r['fmt']} | {r['temporal_split']} | {r['n']} | {r['first_batting_win_rate']:.4f} |"
            for r in base_rates.iter_rows(named=True)
        ),
        "",
        "## Determinism",
        "",
        f"- Label-build hash, run 1: `{h1}`",
        f"- Label-build hash, run 2: `{h2}`",
        f"- Identical: **{h1 == h2}**",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print(f"labels hash: {h1} (stable: {h1 == h2})")


if __name__ == "__main__":
    main()
