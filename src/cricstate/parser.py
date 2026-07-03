"""Cricsheet JSON → validated intermediate representation (SPEC §5).

The parser is strict: every field it reads has already been shape-checked by
`validator.validate_raw`. Entity resolution goes through the per-file
`registry.people` map — a name absent from the registry is a quarantine
(E_REGISTRY_MISS), never a guess.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from cricstate.quarantine import QuarantineError, ReasonCode
from cricstate.schemas import Delivery, Wicket, WicketKind
from cricstate.validator import validate_raw

BallRef = tuple[int, int]  # (over index 0-based, ball number 1-based incl. extras)


@dataclass(frozen=True, slots=True)
class OverData:
    number: int
    deliveries: tuple[Delivery, ...]


@dataclass(frozen=True, slots=True)
class InningsData:
    batting_team: str
    bowling_team: str
    super_over: bool
    target_runs: int | None
    target_balls: int | None  # revised allotment in balls, from target.overs
    miscounted: dict[int, int]  # over number → balls the umpire allowed
    powerplays: tuple[tuple[BallRef, BallRef], ...]  # inclusive (from, to)
    penalty_pre: int  # penalty runs awarded to this innings before it starts
    penalty_post: int
    absent_hurt: tuple[str, ...]
    overs: tuple[OverData, ...]


@dataclass(frozen=True, slots=True)
class ParsedMatch:
    match_id: str
    fmt: Literal["t20", "odi"]
    gender: str
    venue_id: str
    start_date: date
    competition: str | None
    balls_per_over: int
    scheduled_overs: int
    teams: tuple[str, str]
    outcome_result: str | None  # "tie" | "no result" | "draw" | None (= won)
    outcome_winner: str | None
    outcome_method: str | None
    outcome_eliminator: str | None  # super-over winner on a tie
    outcome_bowl_out: str | None  # pre-2008 tie resolution
    dls_applied: bool
    no_result: bool
    toss_uncontested: bool
    innings: tuple[InningsData, ...]


def _resolve(name: str, registry: dict[str, str], match_id: str) -> str:
    try:
        return registry[name]
    except KeyError:
        raise QuarantineError(
            match_id, ReasonCode.E_REGISTRY_MISS, f"name not in registry.people: {name!r}"
        ) from None


def _ball_ref(marker: float, match_id: str) -> BallRef:
    """Parse Cricsheet over.ball notation (e.g. 5.6 → over 5, ball 6).

    Rendered via str() so JSON decimals like 0.11 (over 0, ball 11) survive —
    a plain float multiply would alias ball 11 onto ball 1.
    """
    text = str(marker)
    over_part, _, ball_part = text.partition(".")
    if not ball_part:
        ball_part = "0"
    if not (over_part.isdigit() and ball_part.isdigit()):
        raise QuarantineError(
            match_id, ReasonCode.E_SCHEMA, f"unparseable over.ball marker: {marker!r}"
        )
    return int(over_part), int(ball_part)


def _overs_to_balls(overs: float, balls_per_over: int, match_id: str) -> int:
    """Cricket decimal notation → ball count (17.2 → 17*bpo + 2)."""
    whole, part = _ball_ref(overs, match_id)
    if part >= balls_per_over:
        raise QuarantineError(
            match_id,
            ReasonCode.E_SCHEMA,
            f"target overs {overs} has ball part >= balls_per_over {balls_per_over}",
        )
    return whole * balls_per_over + part


def _parse_delivery(raw: dict[str, Any], registry: dict[str, str], match_id: str) -> Delivery:
    extras = raw.get("extras", {})
    wides = int(extras.get("wides", 0))
    noballs = int(extras.get("noballs", 0))
    byes = int(extras.get("byes", 0))
    legbyes = int(extras.get("legbyes", 0))
    penalty = int(extras.get("penalty", 0))
    runs = raw["runs"]
    if runs["batter"] + runs["extras"] != runs["total"]:
        raise QuarantineError(
            match_id, ReasonCode.E_BALL_ACCOUNTING, f"runs.total mismatch: {runs}"
        )
    if wides + noballs + byes + legbyes + penalty != runs["extras"]:
        raise QuarantineError(
            match_id,
            ReasonCode.E_BALL_ACCOUNTING,
            f"extras breakdown {extras} != runs.extras {runs['extras']}",
        )
    wickets: list[Wicket] = []
    for w in raw.get("wickets", []):
        try:
            kind = WicketKind(w["kind"])
        except ValueError:
            raise QuarantineError(
                match_id, ReasonCode.E_UNKNOWN_WICKET_KIND, f"wicket kind {w['kind']!r}"
            ) from None
        # A few hundred fielder entries in the corpus are anonymous substitutes
        # with no "name" key; there is no entity to resolve, so they are
        # omitted from the tuple rather than quarantining the match.
        fielders = tuple(
            _resolve(fl["name"], registry, match_id) for fl in w.get("fielders", []) if "name" in fl
        )
        wickets.append(
            Wicket(
                player_out=_resolve(w["player_out"], registry, match_id),
                kind=kind,
                fielders=fielders,
            )
        )
    return Delivery(
        batter_id=_resolve(raw["batter"], registry, match_id),
        bowler_id=_resolve(raw["bowler"], registry, match_id),
        non_striker_id=_resolve(raw["non_striker"], registry, match_id),
        runs_batter=int(runs["batter"]),
        runs_extras=int(runs["extras"]),
        runs_total=int(runs["total"]),
        wides=wides,
        noballs=noballs,
        byes=byes,
        legbyes=legbyes,
        penalty=penalty,
        wickets=tuple(wickets),
    )


def parse_match(raw: dict[str, Any], match_id: str) -> ParsedMatch:
    """Validate + parse one raw Cricsheet match dict. Raises QuarantineError."""
    validate_raw(raw, match_id)
    info = raw["info"]
    registry: dict[str, str] = info["registry"]["people"]
    fmt: Literal["t20", "odi"] = "t20" if info["match_type"] == "T20" else "odi"
    balls_per_over = int(info["balls_per_over"])
    outcome = info["outcome"]
    result = outcome.get("result")
    method = outcome.get("method")
    teams = (str(info["teams"][0]), str(info["teams"][1]))

    innings_out: list[InningsData] = []
    for inn in raw["innings"]:
        batting = str(inn["team"])
        if batting not in teams:
            raise QuarantineError(
                match_id, ReasonCode.E_SCHEMA, f"innings team {batting!r} not in {teams}"
            )
        bowling = teams[1] if batting == teams[0] else teams[0]
        target = inn.get("target")
        target_runs = target_balls = None
        if target is not None:
            target_runs = int(target["runs"])
            if target.get("overs") is not None:
                target_balls = _overs_to_balls(target["overs"], balls_per_over, match_id)
        miscounted: dict[int, int] = {}
        for over_no, spec in inn.get("miscounted_overs", {}).items():
            miscounted[int(over_no)] = int(spec["balls"])
        powerplays = tuple(
            (_ball_ref(p["from"], match_id), _ball_ref(p["to"], match_id))
            for p in inn.get("powerplays", [])
        )
        pen = inn.get("penalty_runs", {})
        overs = tuple(
            OverData(
                number=int(ov["over"]),
                deliveries=tuple(
                    _parse_delivery(dl, registry, match_id) for dl in ov["deliveries"]
                ),
            )
            for ov in inn.get("overs", [])
        )
        innings_out.append(
            InningsData(
                batting_team=batting,
                bowling_team=bowling,
                super_over=bool(inn.get("super_over", False)),
                target_runs=target_runs,
                target_balls=target_balls,
                miscounted=miscounted,
                powerplays=powerplays,
                penalty_pre=int(pen.get("pre", 0)),
                penalty_post=int(pen.get("post", 0)),
                absent_hurt=tuple(
                    _resolve(n, registry, match_id) for n in inn.get("absent_hurt", [])
                ),
                overs=overs,
            )
        )

    return ParsedMatch(
        match_id=match_id,
        fmt=fmt,
        gender=str(info["gender"]),
        venue_id=str(info.get("venue", "unknown")),
        start_date=date.fromisoformat(info["dates"][0]),
        competition=(info.get("event", {}) or {}).get("name"),
        balls_per_over=balls_per_over,
        scheduled_overs=int(info["overs"]),
        teams=teams,
        outcome_result=result,
        outcome_winner=outcome.get("winner"),
        outcome_method=method,
        outcome_eliminator=outcome.get("eliminator"),
        outcome_bowl_out=outcome.get("bowl_out"),
        dls_applied=bool(method and "D/L" in method),
        no_result=result == "no result",
        toss_uncontested=bool(info["toss"].get("uncontested", False)),
        innings=tuple(innings_out),
    )
