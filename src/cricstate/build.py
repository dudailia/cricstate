"""Full-corpus build: snapshot → matches/deliveries/players parquet + STATS.md.

Single process, single machine (SPEC §9). Every in-scope match either
contributes rows or lands in the quarantine log with a coded reason — the
`except Exception` at the per-match boundary is the quarantine-not-crash
seam demanded by SPEC §5 (reason E_OTHER), never a silent swallow.
"""

import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from cricstate import SCHEMA_VERSION
from cricstate.download import DATA_DIR, snapshot_dir
from cricstate.parser import ParsedMatch, parse_match
from cricstate.quarantine import QuarantineError, QuarantineRecord, ReasonCode
from cricstate.replay import match_warnings, replay_parsed
from cricstate.transition import NON_DISMISSALS

V1_DIR = DATA_DIR / "v1"
QUARANTINE_DIR = DATA_DIR / "quarantine"
STATS_PATH = Path(__file__).resolve().parents[2] / "docs" / "STATS.md"


# §10 outcome alphabet: ~12 collapsed classes; sub-typing stays in columns.
def outcome_class(
    runs_batter: int, wides: int, noballs: int, byes: int, legbyes: int, n_wickets: int
) -> str:
    if n_wickets:
        return "wicket"
    if wides:
        return "wide"
    if noballs:
        return "no_ball"
    if byes or legbyes:
        return "bye_legbye"
    if runs_batter in (0, 1, 2, 3, 4, 6):
        return str(runs_batter)
    return "other_runs"


DELIVERY_SCHEMA: dict[str, Any] = {
    "schema_version": pl.Utf8,
    "match_id": pl.Utf8,
    "fmt": pl.Utf8,
    "gender": pl.Utf8,
    "venue_id": pl.Utf8,
    "start_date": pl.Date,
    "competition": pl.Utf8,
    "temporal_split": pl.Utf8,
    "innings_idx": pl.Int32,
    "over_number": pl.Int32,
    "ball_in_over": pl.Int32,
    "batting_team": pl.Utf8,
    "bowling_team": pl.Utf8,
    "balls_per_over": pl.Int32,
    "max_balls": pl.Int32,
    "legal_balls": pl.Int32,
    "runs": pl.Int32,
    "wickets": pl.Int32,
    "striker_id": pl.Utf8,
    "striker_runs": pl.Int32,
    "striker_balls": pl.Int32,
    "non_striker_id": pl.Utf8,
    "non_striker_runs": pl.Int32,
    "non_striker_balls": pl.Int32,
    "bowler_id": pl.Utf8,
    "bowler_legal_balls": pl.Int32,
    "bowler_runs_conceded": pl.Int32,
    "bowler_wickets": pl.Int32,
    "target": pl.Int32,
    "dls_applied": pl.Boolean,
    "in_powerplay": pl.Boolean,
    "partnership_runs": pl.Int32,
    "partnership_balls": pl.Int32,
    "fow_last_runs": pl.Int32,
    "fow_last_ball": pl.Int32,
    "last_ball_was_noball": pl.Boolean,
    "is_super_over": pl.Boolean,
    "excluded_from_tuples": pl.Boolean,
    "outcome_class": pl.Utf8,
    "outcome_runs_batter": pl.Int32,
    "outcome_runs_extras": pl.Int32,
    "outcome_runs_total": pl.Int32,
    "outcome_wides": pl.Int32,
    "outcome_noballs": pl.Int32,
    "outcome_byes": pl.Int32,
    "outcome_legbyes": pl.Int32,
    "outcome_penalty": pl.Int32,
    "outcome_n_wickets": pl.Int32,
    "outcome_wicket_kinds": pl.Utf8,
    "outcome_players_out": pl.Utf8,
}


@dataclass
class PlayerAgg:
    name: str
    n_matches: int = 0
    first_seen: date = date.max
    last_seen: date = date.min


@dataclass
class BuildResult:
    n_files: int = 0
    n_in_scope: int = 0
    n_parsed: int = 0
    n_delivery_rows: int = 0
    quarantines: list[QuarantineRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    corpus_hash: str = ""
    runtime_s: float = 0.0
    split_bounds: tuple[str, str] = ("", "")


def _delivery_rows(pm: ParsedMatch, split: str) -> tuple[list[dict[str, Any]], str]:
    """Rows + per-match stream hash in one deterministic pass."""
    rows: list[dict[str, Any]] = []
    h = hashlib.sha256()
    # over/ball position tracked from the parsed structure via replay order
    positions = (
        (i + 1, ov.number, k + 1)
        for i, inn in enumerate(pm.innings)
        for ov in inn.overs
        for k in range(len(ov.deliveries))
    )
    for pre, d, post in replay_parsed(pm):
        innings_idx, over_number, ball_in_over = next(positions)
        assert innings_idx == pre.innings_idx  # positions iterate the same order
        h.update(repr(pre).encode())
        h.update(repr(d).encode())
        h.update(repr(post).encode())
        n_wkts = sum(1 for w in d.wickets if w.kind not in NON_DISMISSALS)
        rows.append(
            {
                "schema_version": pre.schema_version,
                "match_id": pre.match_id,
                "fmt": pre.fmt,
                "gender": pre.gender,
                "venue_id": pre.venue_id,
                "start_date": pre.start_date,
                "competition": pre.competition,
                "temporal_split": split,
                "innings_idx": pre.innings_idx,
                "over_number": over_number,
                "ball_in_over": ball_in_over,
                "batting_team": pre.batting_team,
                "bowling_team": pre.bowling_team,
                "balls_per_over": pre.balls_per_over,
                "max_balls": pre.max_balls,
                "legal_balls": pre.legal_balls,
                "runs": pre.runs,
                "wickets": pre.wickets,
                "striker_id": pre.striker.player_id if pre.striker else None,
                "striker_runs": pre.striker.runs if pre.striker else None,
                "striker_balls": pre.striker.balls if pre.striker else None,
                "non_striker_id": pre.non_striker.player_id if pre.non_striker else None,
                "non_striker_runs": pre.non_striker.runs if pre.non_striker else None,
                "non_striker_balls": pre.non_striker.balls if pre.non_striker else None,
                "bowler_id": pre.bowler.player_id,
                "bowler_legal_balls": pre.bowler.legal_balls,
                "bowler_runs_conceded": pre.bowler.runs_conceded,
                "bowler_wickets": pre.bowler.wickets,
                "target": pre.target,
                "dls_applied": pre.dls_applied,
                "in_powerplay": pre.in_powerplay,
                "partnership_runs": pre.partnership_runs,
                "partnership_balls": pre.partnership_balls,
                "fow_last_runs": pre.fow[-1][0] if pre.fow else None,
                "fow_last_ball": pre.fow[-1][1] if pre.fow else None,
                "last_ball_was_noball": pre.last_ball_was_noball,
                "is_super_over": pre.innings_idx > 2,
                "excluded_from_tuples": pre.innings_idx > 2 or pm.no_result,
                "outcome_class": outcome_class(
                    d.runs_batter, d.wides, d.noballs, d.byes, d.legbyes, n_wkts
                ),
                "outcome_runs_batter": d.runs_batter,
                "outcome_runs_extras": d.runs_extras,
                "outcome_runs_total": d.runs_total,
                "outcome_wides": d.wides,
                "outcome_noballs": d.noballs,
                "outcome_byes": d.byes,
                "outcome_legbyes": d.legbyes,
                "outcome_penalty": d.penalty,
                "outcome_n_wickets": n_wkts,
                "outcome_wicket_kinds": ";".join(w.kind.value for w in d.wickets) or None,
                "outcome_players_out": ";".join(w.player_out for w in d.wickets) or None,
            }
        )
    return rows, h.hexdigest()


def _scan_dates(files: list[Path]) -> dict[str, date]:
    """First pass: start_date per in-scope file, for the temporal split."""
    out: dict[str, date] = {}
    for f in files:
        try:
            with open(f) as fh:
                raw = json.load(fh)
            info = raw.get("info", {})
            if info.get("match_type") in ("T20", "ODI"):
                out[f.stem] = date.fromisoformat(info["dates"][0])
        except Exception:  # noqa: BLE001, S112 — defects surface in the main pass, coded
            continue
    return out


def _temporal_bounds(dates: list[date]) -> tuple[date, date]:
    """80/10/10 by match start date; boundaries recorded in STATS.md."""
    ordered = sorted(dates)
    return ordered[int(len(ordered) * 0.8)], ordered[int(len(ordered) * 0.9)]


def build(write_parquet: bool = True) -> BuildResult:
    t0 = time.monotonic()
    res = BuildResult()
    snap = snapshot_dir() / "json"
    files = sorted(snap.glob("*.json"))
    res.n_files = len(files)

    date_by_id = _scan_dates(files)
    train_end, val_end = _temporal_bounds(list(date_by_id.values()))
    res.split_bounds = (train_end.isoformat(), val_end.isoformat())

    def split_of(d: date) -> str:
        if d < train_end:
            return "train"
        if d < val_end:
            return "val"
        return "test"

    match_rows: list[dict[str, Any]] = []
    players: dict[str, PlayerAgg] = {}
    corpus = hashlib.sha256()
    delivery_batch: list[dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None
    if write_parquet:
        V1_DIR.mkdir(parents=True, exist_ok=True)
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        nonlocal writer, delivery_batch
        if not (write_parquet and delivery_batch):
            return
        table = pl.DataFrame(delivery_batch, schema=DELIVERY_SCHEMA).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(V1_DIR / "deliveries.parquet", table.schema)
        writer.write_table(table)
        delivery_batch = []

    for f in files:
        match_id = f.stem
        try:
            with open(f) as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                raise QuarantineError(match_id, ReasonCode.E_SCHEMA, "top-level not an object")
            pm = parse_match(raw, match_id)
        except QuarantineError as q:
            res.quarantines.append(q.record)
            if q.record.reason is not ReasonCode.E_FORMAT_OOS:
                res.n_in_scope += 1
            continue
        except Exception as exc:  # noqa: BLE001 — quarantine-not-crash boundary (E_OTHER)
            res.n_in_scope += 1
            res.quarantines.append(
                QuarantineRecord(match_id, ReasonCode.E_OTHER, f"parse: {exc!r}")
            )
            continue
        res.n_in_scope += 1
        split = split_of(pm.start_date)
        try:
            rows, shash = _delivery_rows(pm, split)
        except QuarantineError as q:
            res.quarantines.append(q.record)
            continue
        except Exception as exc:  # noqa: BLE001 — quarantine-not-crash boundary (E_OTHER)
            res.quarantines.append(
                QuarantineRecord(match_id, ReasonCode.E_OTHER, f"replay: {exc!r}")
            )
            continue

        res.n_parsed += 1
        res.n_delivery_rows += len(rows)
        res.warnings.extend(f"{match_id}: {w}" for w in match_warnings(pm))
        corpus.update(f"{match_id}:{shash}\n".encode())
        delivery_batch.extend(rows)
        if len(delivery_batch) >= 250_000:
            flush()

        match_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "match_id": pm.match_id,
                "fmt": pm.fmt,
                "gender": pm.gender,
                "venue_id": pm.venue_id,
                "start_date": pm.start_date,
                "competition": pm.competition,
                "temporal_split": split,
                "team1": pm.teams[0],
                "team2": pm.teams[1],
                "balls_per_over": pm.balls_per_over,
                "scheduled_overs": pm.scheduled_overs,
                "outcome_result": pm.outcome_result,
                "outcome_winner": pm.outcome_winner,
                "outcome_method": pm.outcome_method,
                "dls_applied": pm.dls_applied,
                "no_result": pm.no_result,
                "toss_uncontested": pm.toss_uncontested,
                "has_super_over": any(inn.super_over for inn in pm.innings),
                "n_innings": len(pm.innings),
                "n_deliveries": len(rows),
                "stream_hash": shash,
            }
        )
        # players table: everyone fielded in info.players, via the registry —
        # already resolved during parse; re-derive from delivery rows instead
        seen_ids = {
            pid
            for r in rows
            for pid in (r["striker_id"], r["non_striker_id"], r["bowler_id"])
            if pid is not None
        }
        raw_names: dict[str, str] = raw["info"]["registry"]["people"]
        id_to_name = {v: k for k, v in sorted(raw_names.items())}
        for pid in sorted(seen_ids):
            agg = players.get(pid)
            name = id_to_name.get(pid, pid)
            if agg is None:
                agg = players[pid] = PlayerAgg(name=name)
            agg.name = min(agg.name, name)
            agg.n_matches += 1
            agg.first_seen = min(agg.first_seen, pm.start_date)
            agg.last_seen = max(agg.last_seen, pm.start_date)

    flush()
    if writer is not None:
        writer.close()
    res.corpus_hash = corpus.hexdigest()

    if write_parquet:
        pl.DataFrame(match_rows).write_parquet(V1_DIR / "matches.parquet")
        pl.DataFrame(
            [
                {
                    "player_id": pid,
                    "name": agg.name,
                    "n_matches": agg.n_matches,
                    "first_seen": agg.first_seen,
                    "last_seen": agg.last_seen,
                }
                for pid, agg in sorted(players.items())
            ]
        ).write_parquet(V1_DIR / "players.parquet")
        pl.DataFrame(
            [
                {"match_id": q.match_id, "reason": q.reason.value, "detail": q.detail}
                for q in res.quarantines
            ],
            schema={"match_id": pl.Utf8, "reason": pl.Utf8, "detail": pl.Utf8},
        ).write_csv(QUARANTINE_DIR / "quarantine.csv")

    res.runtime_s = time.monotonic() - t0
    return res


def write_stats(run1: BuildResult, run2: BuildResult) -> None:
    hist = Counter(q.reason.value for q in run1.quarantines)
    in_scope_quarantined = sum(
        1 for q in run1.quarantines if q.reason is not ReasonCode.E_FORMAT_OOS
    )
    parse_rate = 100.0 * run1.n_parsed / run1.n_in_scope
    poq_rate = 100.0 * (run1.n_parsed + in_scope_quarantined) / run1.n_in_scope
    other_examples = [q for q in run1.quarantines if q.reason is ReasonCode.E_OTHER][:5]
    lines = [
        "# STATS — v1 corpus build",
        "",
        "Snapshot: `snapshot_2026-07-02` (see `data/MANIFEST`). Scope: T20 + ODI.",
        "",
        "## Corpus",
        "",
        f"- Files in snapshot: **{run1.n_files}**",
        f"- In-scope matches (T20/ODI): **{run1.n_in_scope}**",
        f"- Parsed + replayed clean: **{run1.n_parsed}** ({parse_rate:.3f}%)",
        f"- Parse-or-quarantine-with-reason: **{poq_rate:.3f}%** (DoD ≥ 99.5%)",
        f"- Delivery rows: **{run1.n_delivery_rows}**",
        f"- Miscounted-over warnings (non-fatal): {len(run1.warnings)}",
        "",
        "## Quarantine histogram",
        "",
        "| reason | count |",
        "|--------|-------|",
    ]
    for reason, count in sorted(hist.items()):
        lines.append(f"| {reason} | {count} |")
    if other_examples:
        lines += ["", "E_OTHER examples:", ""]
        lines += [f"- `{q.match_id}`: {q.detail[:140]}" for q in other_examples]
    lines += [
        "",
        "## Temporal split (80/10/10 by start date, baked into the tables)",
        "",
        f"- train < {run1.split_bounds[0]} ≤ val < {run1.split_bounds[1]} ≤ test",
        "",
        "## Determinism",
        "",
        f"- Run 1 corpus hash: `{run1.corpus_hash}`",
        f"- Run 2 corpus hash: `{run2.corpus_hash}`",
        f"- Identical: **{run1.corpus_hash == run2.corpus_hash}**",
        "",
        "## Runtime (Apple Silicon laptop, single process)",
        "",
        f"- Run 1 (with parquet write): {run1.runtime_s:.1f}s",
        f"- Run 2 (hash-only re-run): {run2.runtime_s:.1f}s",
        "",
        "## Notes",
        "",
        "- `fow` is flattened to `fow_last_runs`/`fow_last_ball`; the full tuple",
        "  is reconstructible via `cricstate.replay.replay(match_id)`.",
        "- Free hits are not encoded in Cricsheet; `last_ball_was_noball` is",
        "  carried instead (SPEC §11 bias note).",
        "- Field placements are not in this data at all (SPEC §11).",
        "- Leakage rule: nothing downstream may condition on `outcome_*` columns.",
    ]
    STATS_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("run 1/2: full build with parquet emission…", flush=True)
    r1 = build(write_parquet=True)
    print(
        f"  parsed {r1.n_parsed}/{r1.n_in_scope} in-scope, "
        f"{len(r1.quarantines)} quarantined, {r1.n_delivery_rows} rows, "
        f"{r1.runtime_s:.0f}s",
        flush=True,
    )
    print("run 2/2: determinism re-run (hash only)…", flush=True)
    r2 = build(write_parquet=False)
    print(f"  {r2.runtime_s:.0f}s", flush=True)
    write_stats(r1, r2)
    ok = r1.corpus_hash == r2.corpus_hash
    print(f"determinism: {'IDENTICAL' if ok else 'MISMATCH'}")
    print(f"stats → {STATS_PATH}")
    sys.exit(0 if ok else 1)
