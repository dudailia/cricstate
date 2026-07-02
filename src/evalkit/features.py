"""Whitelisted feature builder for both tasks (SPEC_M2 §4).

Enforcement mechanism: the builder's first act is `df.select(SOURCE_WHITELIST)`.
Anything outside that frozen list — poisoned columns included — is unreachable
by construction. The emitted frame contains exactly FEATURE_COLUMNS (plus the
`fmt` slicing column, itself allowlisted in §4); the denylist contract test
greps it in CI.

Chase-only features are 0.0-sentinel on non-chase rows, with `is_chase` as the
innings indicator. A row is a chase iff innings_idx == 2 AND a target exists
(second innings of no-result stubs can lack one). `new_batter` is derived from
the striker observation gap: identity unknown pre-ball, counters known-zero.
"""

import re

import polars as pl

# The ONLY corpus columns the builder may read (SPEC_M2 §4 allowlist sources).
# striker_id/non_striker_id appear here solely to derive the new-batter flag
# and impute the gap counters; they are never emitted.
SOURCE_WHITELIST: tuple[str, ...] = (
    "innings_idx",
    "balls_per_over",
    "max_balls",
    "legal_balls",
    "runs",
    "wickets",
    "in_powerplay",
    "partnership_runs",
    "partnership_balls",
    "striker_id",
    "striker_runs",
    "striker_balls",
    "non_striker_id",
    "non_striker_runs",
    "non_striker_balls",
    "bowler_legal_balls",
    "bowler_runs_conceded",
    "bowler_wickets",
    "fmt",
    "gender",
    "dls_applied",
    "last_ball_was_noball",
    "target",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "innings_idx",
    "is_chase",
    "legal_balls",
    "balls_remaining",
    "runs",
    "wickets",
    "wickets_in_hand",
    "crr",
    "in_powerplay",
    "partnership_runs",
    "partnership_balls",
    "striker_runs",
    "striker_balls",
    "non_striker_runs",
    "non_striker_balls",
    "new_batter",
    "bowler_legal_balls",
    "bowler_runs_conceded",
    "bowler_wickets",
    "gender_female",
    "dls_applied",
    "last_ball_was_noball",
    "target_runs",
    "required_runs",
    "rrr",
    "rr_gap",
    "wih_x_rrr",
)

# §4 denylist, enforced by a CI contract test over the emitted frame.
DENYLIST_PATTERNS: tuple[str, ...] = (
    r"_id$",  # every *_id column (match, venue, player)
    r"(^|_)name($|_)",  # every name column
    r"^outcome_",  # every outcome column
    r"winner",  # winner fields
    r"date",  # raw dates live only in split logic
    r"team",  # team identity
)


def denylist_violations(columns: list[str]) -> list[str]:
    return [
        c for c in columns if any(re.search(p, c, flags=re.IGNORECASE) for p in DENYLIST_PATTERNS)
    ]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Model matrix for both tasks. Emits FEATURE_COLUMNS + `fmt` (slicing)."""
    src = df.select(SOURCE_WHITELIST)  # whitelist enforcement: nothing else exists

    bpo = pl.col("balls_per_over").cast(pl.Float64)
    legal = pl.col("legal_balls").cast(pl.Float64)
    runs = pl.col("runs").cast(pl.Float64)
    balls_remaining = (pl.col("max_balls") - pl.col("legal_balls")).cast(pl.Float64)
    is_chase = ((pl.col("innings_idx") == 2) & pl.col("target").is_not_null()).cast(pl.Float64)
    crr = pl.when(legal > 0).then(bpo * runs / legal).otherwise(0.0).alias("crr")
    required = (pl.col("target").cast(pl.Float64) - runs).alias("required")
    rrr_expr = (
        pl.when((is_chase == 1.0) & (balls_remaining > 0))
        .then(bpo * pl.col("required") / balls_remaining)
        .otherwise(0.0)
    )

    out = (
        src.with_columns(crr, required, is_chase.alias("is_chase"))
        .with_columns(
            pl.when(pl.col("is_chase") == 1.0)
            .then(pl.col("required"))
            .otherwise(0.0)
            .alias("required_runs"),
            rrr_expr.alias("rrr"),
        )
        .with_columns(
            pl.col("innings_idx").cast(pl.Float64),
            balls_remaining.alias("balls_remaining"),
            legal.alias("legal_balls"),
            runs.alias("runs"),
            pl.col("wickets").cast(pl.Float64),
            (10.0 - pl.col("wickets").cast(pl.Float64)).alias("wickets_in_hand"),
            pl.col("in_powerplay").cast(pl.Float64),
            pl.col("partnership_runs").cast(pl.Float64),
            pl.col("partnership_balls").cast(pl.Float64),
            pl.col("striker_runs").fill_null(0).cast(pl.Float64),
            pl.col("striker_balls").fill_null(0).cast(pl.Float64),
            pl.col("non_striker_runs").fill_null(0).cast(pl.Float64),
            pl.col("non_striker_balls").fill_null(0).cast(pl.Float64),
            pl.col("striker_id").is_null().cast(pl.Float64).alias("new_batter"),
            pl.col("bowler_legal_balls").cast(pl.Float64),
            pl.col("bowler_runs_conceded").cast(pl.Float64),
            pl.col("bowler_wickets").cast(pl.Float64),
            (pl.col("gender") == "female").cast(pl.Float64).alias("gender_female"),
            pl.col("dls_applied").cast(pl.Float64),
            pl.col("last_ball_was_noball").cast(pl.Float64),
            pl.when(pl.col("is_chase") == 1.0)
            .then(pl.col("target").cast(pl.Float64))
            .otherwise(0.0)
            .alias("target_runs"),
        )
        .with_columns(
            pl.when(pl.col("is_chase") == 1.0)
            .then(pl.col("rrr") - pl.col("crr"))
            .otherwise(0.0)
            .alias("rr_gap"),
            (pl.col("wickets_in_hand") * pl.col("rrr")).alias("wih_x_rrr"),
        )
        .select(*FEATURE_COLUMNS, "fmt")
    )

    violations = denylist_violations(out.columns)
    if violations:  # defense in depth; the contract test also greps in CI
        raise ValueError(f"denylist violation in emitted features: {violations}")
    return out
