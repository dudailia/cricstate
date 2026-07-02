import polars as pl
import pytest

from evalkit.features import (
    DENYLIST_PATTERNS,
    FEATURE_COLUMNS,
    SOURCE_WHITELIST,
    build_features,
    denylist_violations,
)


def synthetic_rows() -> pl.DataFrame:
    """Two innings-1 rows (one in the observation gap) + one chase row."""
    return pl.DataFrame(
        {
            "innings_idx": [1, 1, 2],
            "balls_per_over": [6, 6, 6],
            "max_balls": [120, 120, 120],
            "legal_balls": [0, 30, 60],
            "runs": [0, 45, 80],
            "wickets": [0, 2, 3],
            "in_powerplay": [True, False, False],
            "partnership_runs": [0, 10, 12],
            "partnership_balls": [0, 8, 9],
            "striker_id": ["p1", None, "p9"],
            "striker_runs": [0, None, 33],
            "striker_balls": [0, None, 21],
            "non_striker_id": ["p2", "p3", "p8"],
            "non_striker_runs": [0, 5, 11],
            "non_striker_balls": [0, 12, 14],
            "bowler_legal_balls": [0, 12, 18],
            "bowler_runs_conceded": [0, 20, 22],
            "bowler_wickets": [0, 1, 0],
            "fmt": ["t20", "t20", "t20"],
            "gender": ["male", "female", "male"],
            "dls_applied": [False, False, False],
            "last_ball_was_noball": [False, True, False],
            "target": [None, None, 161],
        },
        schema_overrides={"striker_runs": pl.Int32, "striker_balls": pl.Int32},
    )


def test_emits_exactly_the_frozen_columns() -> None:
    out = build_features(synthetic_rows())
    assert out.columns == [*FEATURE_COLUMNS, "fmt"]


def test_derived_arithmetic() -> None:
    out = build_features(synthetic_rows())
    row1 = out.row(1, named=True)  # innings 1, 30 balls, 45 runs, 2 wkts
    assert row1["balls_remaining"] == 90.0
    assert row1["wickets_in_hand"] == 8.0
    assert row1["crr"] == pytest.approx(9.0)  # 6 * 45/30
    assert row1["is_chase"] == 0.0
    # chase-only sentinels on innings-1 rows
    assert row1["target_runs"] == row1["required_runs"] == row1["rrr"] == row1["rr_gap"] == 0.0
    row2 = out.row(2, named=True)  # chase: 80/3 off 60, target 161
    assert row2["is_chase"] == 1.0
    assert row2["required_runs"] == 81.0
    assert row2["rrr"] == pytest.approx(8.1)  # 6 * 81/60
    assert row2["crr"] == pytest.approx(8.0)
    assert row2["rr_gap"] == pytest.approx(0.1)
    assert row2["wih_x_rrr"] == pytest.approx(7 * 8.1)


def test_new_batter_gap_imputation() -> None:
    out = build_features(synthetic_rows())
    assert out["new_batter"].to_list() == [0.0, 1.0, 0.0]
    row1 = out.row(1, named=True)
    assert row1["striker_runs"] == 0.0 and row1["striker_balls"] == 0.0


def test_crr_zero_at_innings_start() -> None:
    assert build_features(synthetic_rows()).row(0, named=True)["crr"] == 0.0


def test_gender_is_a_feature() -> None:
    assert build_features(synthetic_rows())["gender_female"].to_list() == [0.0, 1.0, 0.0]


def test_denylist_contract_on_emitted_frame() -> None:
    out = build_features(synthetic_rows())
    assert denylist_violations(out.columns) == []


def test_denylist_patterns_catch_the_banned_families() -> None:
    banned = [
        "match_id",
        "venue_id",
        "striker_id",
        "name",
        "player_name",
        "outcome_class",
        "outcome_runs_total",
        "winner",
        "outcome_winner",
        "start_date",
        "batting_team",
    ]
    assert denylist_violations(banned) == banned
    assert len(DENYLIST_PATTERNS) >= 5


def test_whitelist_enforcement_poison_column_is_unreachable() -> None:
    """SPEC_M2 §7.3: output identical with/without a poisoned extra column."""
    clean = synthetic_rows()
    poisoned = clean.with_columns(
        pl.Series("outcome_class", ["wicket", "4", "0"]),
        pl.Series("outcome_runs_total", [99, 99, 99]),
        pl.Series("winner", ["A", "A", "A"]),
    )
    out_clean = build_features(clean)
    out_poisoned = build_features(poisoned)
    assert out_poisoned.columns == out_clean.columns
    assert "outcome_class" not in out_poisoned.columns
    assert out_poisoned.equals(out_clean)


def test_missing_source_column_is_an_error() -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        build_features(synthetic_rows().drop("target"))


def test_source_whitelist_is_frozen_and_minimal() -> None:
    assert isinstance(SOURCE_WHITELIST, tuple)
    # ids are readable only for gap/flag derivation; never emitted
    assert "striker_id" in SOURCE_WHITELIST
    emitted = set(build_features(synthetic_rows()).columns)
    assert not emitted & {"striker_id", "non_striker_id", "target"}
