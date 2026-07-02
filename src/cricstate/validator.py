"""Strict structural validation of raw Cricsheet 1.2.0 dicts (SPEC §5).

Hand-rolled (stack rule: no jsonschema dependency) but closed-world: unknown
keys, wrong types, or a data_version other than the pin quarantine the whole
match. Allowlists were derived from an exhaustive survey of the pinned
snapshot, not guessed.
"""

from typing import Any

from cricstate import PINNED_DATA_VERSION
from cricstate.quarantine import QuarantineError, ReasonCode

IN_SCOPE_MATCH_TYPES = frozenset({"T20", "ODI"})

_INFO_KEYS = frozenset(
    {
        "balls_per_over",
        "bowl_out",
        "city",
        "dates",
        "event",
        "gender",
        "match_type",
        "match_type_number",
        "missing",
        "officials",
        "outcome",
        "overs",
        "player_of_match",
        "players",
        "registry",
        "season",
        "supersubs",
        "team_type",
        "teams",
        "toss",
        "venue",
    }
)
_OUTCOME_KEYS = frozenset({"winner", "by", "result", "method", "eliminator", "bowl_out"})
_TOSS_KEYS = frozenset({"decision", "winner", "uncontested"})
_INNINGS_KEYS = frozenset(
    {
        "team",
        "overs",
        "powerplays",
        "target",
        "super_over",
        "absent_hurt",
        "miscounted_overs",
        "penalty_runs",
    }
)
_OVER_KEYS = frozenset({"over", "deliveries"})
_DELIVERY_KEYS = frozenset(
    {
        "actual_delivery",
        "batter",
        "bowler",
        "non_striker",
        "runs",
        "review",
        "wickets",
        "extras",
        "replacements",
        "over",
    }
)
_RUNS_KEYS = frozenset({"batter", "extras", "total", "non_boundary"})
_EXTRAS_KEYS = frozenset({"wides", "noballs", "byes", "legbyes", "penalty"})
_WICKET_KEYS = frozenset({"player_out", "kind", "fielders"})
_FIELDER_KEYS = frozenset({"name", "substitute"})
_TARGET_KEYS = frozenset({"overs", "runs"})
_POWERPLAY_KEYS = frozenset({"from", "to", "type"})


def _fail(match_id: str, detail: str) -> QuarantineError:
    return QuarantineError(match_id, ReasonCode.E_SCHEMA, detail)


def _require_dict(obj: Any, where: str, match_id: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise _fail(match_id, f"{where}: expected object, got {type(obj).__name__}")
    return obj


def _require_list(obj: Any, where: str, match_id: str) -> list[Any]:
    if not isinstance(obj, list):
        raise _fail(match_id, f"{where}: expected array, got {type(obj).__name__}")
    return obj


def _require_keys(
    d: dict[str, Any], allowed: frozenset[str], required: frozenset[str], where: str, match_id: str
) -> None:
    keys = set(d)
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise _fail(match_id, f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise _fail(match_id, f"{where}: missing keys {sorted(missing)}")


def _require_int(obj: Any, where: str, match_id: str) -> int:
    # bool is an int subclass; a bare bool here is still malformed data.
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise _fail(match_id, f"{where}: expected integer, got {obj!r}")
    return obj


def _require_str(obj: Any, where: str, match_id: str) -> str:
    if not isinstance(obj, str):
        raise _fail(match_id, f"{where}: expected string, got {obj!r}")
    return obj


def _validate_delivery(dl: Any, where: str, match_id: str) -> None:
    d = _require_dict(dl, where, match_id)
    _require_keys(
        d, _DELIVERY_KEYS, frozenset({"batter", "bowler", "non_striker", "runs"}), where, match_id
    )
    _require_str(d["batter"], f"{where}.batter", match_id)
    _require_str(d["bowler"], f"{where}.bowler", match_id)
    _require_str(d["non_striker"], f"{where}.non_striker", match_id)
    runs = _require_dict(d["runs"], f"{where}.runs", match_id)
    _require_keys(
        runs, _RUNS_KEYS, frozenset({"batter", "extras", "total"}), f"{where}.runs", match_id
    )
    for k in ("batter", "extras", "total"):
        _require_int(runs[k], f"{where}.runs.{k}", match_id)
    if "extras" in d:
        extras = _require_dict(d["extras"], f"{where}.extras", match_id)
        _require_keys(extras, _EXTRAS_KEYS, frozenset(), f"{where}.extras", match_id)
        for k, v in extras.items():
            _require_int(v, f"{where}.extras.{k}", match_id)
    for i, w in enumerate(_require_list(d.get("wickets", []), f"{where}.wickets", match_id)):
        wd = _require_dict(w, f"{where}.wickets[{i}]", match_id)
        _require_keys(
            wd, _WICKET_KEYS, frozenset({"player_out", "kind"}), f"{where}.wickets[{i}]", match_id
        )
        _require_str(wd["player_out"], f"{where}.wickets[{i}].player_out", match_id)
        _require_str(wd["kind"], f"{where}.wickets[{i}].kind", match_id)
        for j, fl in enumerate(
            _require_list(wd.get("fielders", []), f"{where}.wickets[{i}].fielders", match_id)
        ):
            fld = _require_dict(fl, f"{where}.wickets[{i}].fielders[{j}]", match_id)
            _require_keys(
                fld, _FIELDER_KEYS, frozenset(), f"{where}.wickets[{i}].fielders[{j}]", match_id
            )


def validate_raw(raw: dict[str, Any], match_id: str) -> None:
    """Shape-check a raw Cricsheet match. Raises QuarantineError on any defect.

    Order matters: version pin first, then format scope, then deep structure —
    so an out-of-scope Test never reports E_SCHEMA for a Test-only field.
    """
    _require_keys(
        _require_dict(raw, "match", match_id),
        frozenset({"meta", "info", "innings"}),
        frozenset({"meta", "info", "innings"}),
        "match",
        match_id,
    )
    meta = _require_dict(raw["meta"], "meta", match_id)
    version = meta.get("data_version")
    if version != PINNED_DATA_VERSION:
        raise QuarantineError(
            match_id,
            ReasonCode.E_VERSION,
            f"data_version {version!r} != pinned {PINNED_DATA_VERSION!r}",
        )

    info = _require_dict(raw["info"], "info", match_id)
    match_type = info.get("match_type")
    if match_type not in IN_SCOPE_MATCH_TYPES:
        raise QuarantineError(
            match_id, ReasonCode.E_FORMAT_OOS, f"match_type {match_type!r} out of v1 scope"
        )
    _require_keys(
        info,
        _INFO_KEYS,
        frozenset(
            {
                "balls_per_over",
                "dates",
                "gender",
                "match_type",
                "outcome",
                "overs",
                "registry",
                "teams",
                "toss",
                "venue",
            }
        ),
        "info",
        match_id,
    )
    _require_int(info["balls_per_over"], "info.balls_per_over", match_id)
    _require_int(info["overs"], "info.overs", match_id)
    dates = _require_list(info["dates"], "info.dates", match_id)
    if not dates:
        raise _fail(match_id, "info.dates is empty")
    _require_str(dates[0], "info.dates[0]", match_id)
    teams = _require_list(info["teams"], "info.teams", match_id)
    if len(teams) != 2:
        raise _fail(match_id, f"info.teams has {len(teams)} entries, expected 2")
    outcome = _require_dict(info["outcome"], "info.outcome", match_id)
    _require_keys(outcome, _OUTCOME_KEYS, frozenset(), "info.outcome", match_id)
    toss = _require_dict(info["toss"], "info.toss", match_id)
    _require_keys(toss, _TOSS_KEYS, frozenset(), "info.toss", match_id)
    registry = _require_dict(info["registry"], "info.registry", match_id)
    people = _require_dict(registry.get("people", None), "info.registry.people", match_id)
    for name, pid in people.items():
        _require_str(pid, f"info.registry.people[{name!r}]", match_id)

    for i, inn in enumerate(_require_list(raw["innings"], "innings", match_id)):
        where = f"innings[{i}]"
        inn_d = _require_dict(inn, where, match_id)
        _require_keys(inn_d, _INNINGS_KEYS, frozenset({"team", "overs"}), where, match_id)
        _require_str(inn_d["team"], f"{where}.team", match_id)
        if "target" in inn_d:
            tgt = _require_dict(inn_d["target"], f"{where}.target", match_id)
            _require_keys(tgt, _TARGET_KEYS, frozenset({"runs"}), f"{where}.target", match_id)
            _require_int(tgt["runs"], f"{where}.target.runs", match_id)
        if "miscounted_overs" in inn_d:
            for over_no, spec in _require_dict(
                inn_d["miscounted_overs"], f"{where}.miscounted_overs", match_id
            ).items():
                if not over_no.isdigit():
                    raise _fail(match_id, f"{where}.miscounted_overs key {over_no!r} not an int")
                balls = _require_dict(spec, f"{where}.miscounted_overs[{over_no}]", match_id).get(
                    "balls"
                )
                if not (isinstance(balls, int) or (isinstance(balls, str) and balls.isdigit())):
                    raise _fail(match_id, f"{where}.miscounted_overs[{over_no}].balls = {balls!r}")
        for j, pp in enumerate(
            _require_list(inn_d.get("powerplays", []), f"{where}.powerplays", match_id)
        ):
            pp_d = _require_dict(pp, f"{where}.powerplays[{j}]", match_id)
            _require_keys(
                pp_d, _POWERPLAY_KEYS, _POWERPLAY_KEYS, f"{where}.powerplays[{j}]", match_id
            )
        for j, ov in enumerate(_require_list(inn_d["overs"], f"{where}.overs", match_id)):
            ov_d = _require_dict(ov, f"{where}.overs[{j}]", match_id)
            _require_keys(ov_d, _OVER_KEYS, _OVER_KEYS, f"{where}.overs[{j}]", match_id)
            _require_int(ov_d["over"], f"{where}.overs[{j}].over", match_id)
            for k, dl in enumerate(
                _require_list(ov_d["deliveries"], f"{where}.overs[{j}].deliveries", match_id)
            ):
                _validate_delivery(dl, f"{where}.overs[{j}].deliveries[{k}]", match_id)
