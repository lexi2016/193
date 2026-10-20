"""Pure frozen environment-tilt mathematics; no IO, producer, model or runner.

Adapters MUST bind the original full common-finite M0 pool, original S1 prior
(kappa=100, complete 2006 <= year < T.year, T+1 observed-session month), original
all-observed-panel breadth, calendar, OOT model metadata and unfilled h2 labels.
This module does not authenticate those upstream facts. Calendar SHA/manifest
references describe caller-certified coverage; passing them is not certification.
Select/freeze before supplying outcomes. All returns are gross overlapping h2;
there is no NAV, net return, production, order or sealed-confirmation interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math
import statistics
from typing import Mapping, Sequence


COMPLETE_FOLDS = ("2024H1", "2024H2", "2025H1", "2025H2", "2026H1")
PARTIAL_FOLD = "2026H2_PARTIAL_TO_2026_10_15"
NEUTRAL_PRIORS = frozenset(("missing", "unknown", "unmapped", "zero_support"))
POLICY_JSON = json.dumps({
    "schema": "ENVIRONMENT_TILT_PURE_CORE_V1", "top_k": 50, "top_n": 10,
    "lambda_min": 0, "lambda_max": .25, "scale_cap": .01, "rho": 1,
    "prior": "original_S1_kappa100_complete_2006_le_year_lt_Tyear_Tplus1_month",
    "z": "tanh(max(prior,0)); original_missing_neutral",
    "g": "max(0,2*original_all_observed_panel_breadth-1)",
    "r": "center_within_TopK(y-p)/0.01", "a": "center_within_TopK(s*g*z)/0.01",
    "fit": "date_equal_closed_form_no_search", "hac": "calendar_Bartlett20",
    "screen": {"mean_delta_bp_min": 8, "positive_full_folds_min": 4,
               "fractional_mass_CVaR5_candidate_ge_base": True},
    "confirmation_gate": None, "research_only": True,
    "production_allowed": False, "buy_gate_updated": False, "orders_sent": False,
}, sort_keys=True, separators=(",", ":"))
CONFIG_SHA256 = hashlib.sha256(POLICY_JSON.encode()).hexdigest()


class ContractError(ValueError):
    """Stop: conflicting/invalid binding or arithmetic; never silently repair."""


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"nonfinite or invalid {name}")
    return float(value)


def _mean(values: Sequence[float]) -> float:
    # Exact-ratio accumulation avoids overflow and preserves constant vectors.
    if not values:
        raise ContractError("empty mean")
    return _finite(statistics.mean(values), "mean")


def _date(value: date, name: str) -> date:
    if type(value) is not date:
        raise ContractError(f"{name} must be a bound datetime.date")
    return value


def _code(value: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ContractError("invalid canonical code; adapter must not normalize here")


@dataclass(frozen=True)
class ScoreRow:
    code: str
    score: float
    prior: float | None = None
    prior_status: str = "missing"


@dataclass(frozen=True)
class ScoreDay:
    signal_date: date
    next_session: date
    h2_session: date
    rows: tuple[ScoreRow, ...]
    breadth: float | None = None  # None alone is an explicitly bound missing value.


@dataclass(frozen=True)
class RankedRow:
    code: str
    base_score: float
    base_rank: int
    score: float
    unit_addition: float  # s*g*z, before multiplication by lambda.


@dataclass(frozen=True)
class Selection:
    signal_date: date
    h2_session: date
    lambda_value: float
    top_k: tuple[RankedRow, ...]  # Original M0 order, independent of labels.
    selected: tuple[RankedRow, ...]  # Candidate score desc/code asc.
    scale: float | None
    gate: float | None
    activation: str

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                             default=lambda x: x.isoformat(), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def _top_k(day: ScoreDay) -> tuple[ScoreRow, ...]:
    _date(day.signal_date, "signal_date")
    if not (day.signal_date < _date(day.next_session, "next_session")
            < _date(day.h2_session, "h2_session")):
        raise ContractError("missing or invalid T+1/T+2 calendar anchor")
    seen = set()
    for row in day.rows:
        _code(row.code)
        if row.code in seen:
            raise ContractError("duplicate score key")
        seen.add(row.code)
        _finite(row.score, "M0 score")
    return tuple(sorted(day.rows, key=lambda x: (-x.score, x.code))[:50])


def _unit_additions(day: ScoreDay, rows: tuple[ScoreRow, ...]):
    scale = min(statistics.pstdev([r.score for r in rows]), .01) if rows else 0.
    if scale == 0:
        return scale, None, "zero_scale", (0.,) * len(rows)
    if day.breadth is None:
        gate, activation = 0., "deactivated_missing_environment"
    else:
        breadth = _finite(day.breadth, "breadth")
        if not 0 <= breadth <= 1:
            raise ContractError("breadth outside [0,1]")
        gate, activation = max(0., 2 * breadth - 1), "observed_environment"
    units = []
    for row in rows:
        if row.prior_status in NEUTRAL_PRIORS:
            z = 0.
        elif row.prior_status == "observed":
            z = math.tanh(max(_finite(row.prior, "unexplained prior"), 0.))
        else:
            raise ContractError("unbound prior missing/classification status")
        units.append(_finite(scale * gate * z, "unit addition"))
    return scale, gate, activation, tuple(units)


def select(day: ScoreDay, lambda_value: float) -> Selection:
    """No outcome argument: full-pool Top50 followed by immutable Top10, no refill."""
    lam = _finite(lambda_value, "lambda")
    if not 0 <= lam <= .25:
        raise ContractError("lambda outside frozen [0,.25]")
    rows = _top_k(day)
    # Exact identity path BEFORE any prior/breadth/std arithmetic, including NaNs.
    if lam == 0:
        scale, gate, activation, units = None, None, "lambda_zero", (0.,) * len(rows)
    else:
        scale, gate, activation, units = _unit_additions(day, rows)
    ranked = tuple(RankedRow(r.code, r.score, i + 1,
                            r.score if lam == 0 or u == 0 else
                            _finite(r.score + lam * u, "candidate score"), u)
                   for i, (r, u) in enumerate(zip(rows, units)))
    selected = tuple(sorted(ranked, key=lambda r: (-r.score, r.code))[:10])
    return Selection(day.signal_date, day.h2_session, lam, ranked, selected,
                     scale, gate, activation)


@dataclass(frozen=True)
class CalendarBinding:
    """Original observed calendar + external identity, not a generated calendar."""
    start: date
    end: date  # Exclusive declared coverage bound; not first/last observed day.
    sessions: tuple[date, ...]
    calendar_sha256: str
    score_manifest_sha256: str

    def validate(self, start: date, end: date) -> None:
        if self.start != start or self.end != end:
            raise ContractError("incomplete calendar/manifest coverage bounds")
        for ref in (self.calendar_sha256, self.score_manifest_sha256):
            if not isinstance(ref, str) or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
                raise ContractError("missing calendar/score manifest identity")
        for d in self.sessions:
            _date(d, "calendar session")
        if (not self.sessions or tuple(sorted(set(self.sessions))) != self.sessions
                or any(not start <= d < end for d in self.sessions)):
            raise ContractError("invalid bound calendar sessions")


@dataclass(frozen=True)
class Outcome:
    signal_date: date
    code: str
    value: float | None
    label_end: date | None


def _outcome_index(outcomes: Sequence[Outcome]):
    output = {}
    for row in outcomes:
        _date(row.signal_date, "outcome signal_date")
        _code(row.code)
        key = (row.signal_date, row.code)
        if key in output:
            raise ContractError("duplicate outcome key")
        output[key] = row
    return output


def _known(row: Outcome | None) -> bool:
    return (row is not None and row.value is not None and not isinstance(row.value, bool)
            and isinstance(row.value, (int, float)) and math.isfinite(row.value))


@dataclass(frozen=True)
class OOTDay:
    scores: ScoreDay
    max_model_train_label_end: date
    score_model_sha256: str


@dataclass(frozen=True)
class FitResult:
    lambda_value: float
    status: str
    numerator: float
    denominator_without_rho: float
    scored_dates: int
    complete_label_dates: int
    activated_dates: int
    nonzero_unit_dates: int
    train_mask: tuple[tuple[date, bool, str], ...]
    rho: float = 1.


def fit_lambda(outer_start: date, coverage: CalendarBinding,
               days: Sequence[OOTDay], outcomes: Sequence[Outcome]) -> FitResult:
    """Exactly one date-equal ridge scalar, rho=1, no search/fallback/base fit.

    Coverage requires EVERY externally bound score date in the complete nominal
    two-year window, even when that date's TopK outcomes are missing or immature.
    Model timestamps are the actual maximum train label_end, never purge proxies.
    """
    outer_start = _date(outer_start, "outer_start")
    if (outer_start.month, outer_start.day) not in ((1, 1), (7, 1)):
        raise ContractError("outer fold must start on January/July 1")
    start = outer_start.replace(year=outer_start.year - 2)
    coverage.validate(start, outer_start)
    by_date = {}
    for day in days:
        t = day.scores.signal_date
        if t in by_date:
            raise ContractError("duplicate OOT date")
        if not _date(day.max_model_train_label_end, "model_train_label_end") < t:
            raise ContractError("model_train_label_end must precede score date")
        ref = day.score_model_sha256
        if not isinstance(ref, str) or len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            raise ContractError("missing OOT score model identity")
        by_date[t] = day
    if set(by_date) != set(coverage.sessions):
        raise ContractError("incomplete two-year score coverage; no shortened window")
    labels = _outcome_index(outcomes)
    cross, squares, mask = [], [], []
    activated = nonzero = 0
    for t in coverage.sessions:
        day = by_date[t].scores
        rows = _top_k(day)  # ALWAYS before labels; never remove/refill bad labels.
        _, gate, _, units = _unit_additions(day, rows)
        activated += int(gate is not None and gate > 0)
        nonzero += int(any(u != 0 for u in units))
        selected_labels = [labels.get((t, row.code)) for row in rows]
        if not rows or not all(_known(label) for label in selected_labels):
            mask.append((t, False, "incomplete_TopK_labels"))
            continue
        for label in selected_labels:
            if _date(label.label_end, "label_end") != day.h2_session:
                raise ContractError("label_end differs from bound h2 session")
        if day.h2_session >= outer_start:
            mask.append((t, False, "label_not_mature_before_outer_start"))
            continue
        y = [label.value for label in selected_labels]
        p = [row.score for row in rows]
        u_mean, y_mean, p_mean = _mean(units), _mean(y), _mean(p)
        a = [_finite((u - u_mean) / .01, "centered a") for u in units]
        r = [_finite(((yi - y_mean) - (pi - p_mean)) / .01, "centered r")
             for yi, pi in zip(y, p)]
        cross.append(_mean([_finite(ai * ri, "a*r") for ai, ri in zip(a, r)]))
        squares.append(_mean([ai * ai for ai in a]))
        mask.append((t, True, "complete"))
    numerator = _mean(cross) if cross else 0.
    denominator = _mean(squares) if squares else 0.
    lam = min(.25, max(0., numerator / (denominator + 1.)))
    status = "FITTED" if cross and denominator > 0 else "NO_IDENTIFIABLE_TILT"
    return FitResult(lam, status, numerator, denominator, len(days), len(cross),
                     activated, nonzero, tuple(mask))


@dataclass(frozen=True)
class FrozenPair:
    baseline: Selection
    candidate: Selection


def _validate_selection(selection: Selection) -> None:
    """Check read-back list structure before an isolated outcome join."""
    _date(selection.signal_date, "selection date")
    if not selection.signal_date < _date(selection.h2_session, "selection h2 session"):
        raise ContractError("invalid frozen selection horizon")
    lam = _finite(selection.lambda_value, "frozen lambda")
    if not 0 <= lam <= .25 or len(selection.top_k) > 50:
        raise ContractError("invalid frozen selection bounds")
    codes = set()
    for rank, row in enumerate(selection.top_k, 1):
        _code(row.code)
        if row.code in codes or row.base_rank != rank:
            raise ContractError("duplicate or invalid frozen selection ranks")
        codes.add(row.code)
        _finite(row.base_score, "frozen M0 score")
        score = _finite(row.score, "frozen score")
        u = _finite(row.unit_addition, "frozen unit addition")
        if not 0 <= u <= .01 or score != row.base_score + lam * u:
            raise ContractError("invalid frozen score formula")
        if lam == 0 and u != 0:
            raise ContractError("invalid frozen lambda-zero unit")
    if (selection.top_k != tuple(sorted(selection.top_k, key=lambda r: (-r.base_score, r.code)))
            or selection.selected != tuple(sorted(selection.top_k, key=lambda r: (-r.score, r.code))[:10])):
        raise ContractError("frozen lists must preserve label-independent TopK/Top10 order")


def freeze_pair(day: ScoreDay, lambda_value: float) -> FrozenPair:
    """Persist/hash these two selections before the isolated outcome join."""
    return FrozenPair(select(day, 0.), select(day, lambda_value))


def fold_id(t: date) -> str:
    fold = f"{t.year}H{1 if t.month <= 6 else 2}"
    if fold in COMPLETE_FOLDS:
        return fold
    if date(2026, 7, 1) <= t <= date(2026, 10, 15):
        return PARTIAL_FOLD
    raise ContractError("date outside frozen outer folds")


def fractional_cvar5(values: Sequence[float]) -> float:
    """Mean of exactly .05*N lower-tail mass, including a fractional last item."""
    ordered = sorted(_finite(v, "CVaR value") for v in values)
    if not ordered:
        raise ContractError("empty CVaR")
    mass = .05 * len(ordered)
    whole = math.floor(mass)
    fraction = mass - whole
    return _finite((math.fsum(ordered[:whole]) + fraction * ordered[whole]) / mass, "CVaR")


@dataclass(frozen=True)
class HACResult:
    mean: float | None
    variance: float | None
    standard_error: float | None
    paired_dates: int
    calendar_sessions: int
    status: str
    normal_reference_uncalibrated: tuple[float, float] | None


def calendar_hac20(calendar: Sequence[date], deltas: Mapping[date, float | None]) -> HACResult:
    """Bartlett20, n^2 denominator, no calendar compression or fold reset.

    None/absent delta is an excluded day, NOT a zero-return observation.
    Optional 1.96 interval is an uncalibrated normal reference, never a CI gate.
    """
    dates = tuple(calendar)
    for d in dates:
        _date(d, "HAC calendar session")
    if tuple(sorted(set(dates))) != dates or set(deltas) - set(dates):
        raise ContractError("ambiguous HAC calendar or delta outside calendar")
    observed = {t: _finite(v, "HAC delta") for t, v in deltas.items() if v is not None}
    n = len(observed)
    mu = _mean(list(observed.values())) if n else None
    if n == 0 or len(dates) < 21:
        return HACResult(mu, None, None, n, len(dates), "CANNOT_CONFIRM", None)
    e = [observed[t] - mu if t in observed else 0. for t in dates]
    gamma = [math.fsum(_finite(e[j] * e[j - lag], "HAC product")
                       for j in range(lag, len(e))) for lag in range(21)]
    variance = _finite((gamma[0] + 2 * math.fsum((1 - lag / 21) * gamma[lag]
                                               for lag in range(1, 21))) / n ** 2, "HAC variance")
    if variance <= 0:
        return HACResult(mu, variance, None, n, len(dates), "CANNOT_CONFIRM", None)
    se = math.sqrt(variance)
    return HACResult(mu, variance, se, n, len(dates), "WORKING_SE_ONLY_UNCALIBRATED",
                     (mu - 1.96 * se, mu + 1.96 * se))


@dataclass(frozen=True)
class PairedDate:
    signal_date: date
    fold: str
    paired_complete: bool
    base_selected: int
    candidate_selected: int
    base_known: int
    candidate_known: int
    base_unknown: int
    candidate_unknown: int
    reason: str
    baseline_return: float | None
    candidate_return: float | None
    delta: float | None
    entrants: tuple[str, ...]
    exits: tuple[str, ...]


@dataclass(frozen=True)
class Evaluation:
    dates: tuple[PairedDate, ...]
    full_fold_means: tuple[tuple[str, float | None], ...]
    partial_fold_mean: float | None
    mean_delta: float | None
    median_delta: float | None
    selected_event_p10_base: float | None
    selected_event_p10_candidate: float | None
    cvar5_base: float | None
    cvar5_candidate: float | None
    positive_full_folds: int
    historical_screen: bool | None
    decision: str
    hac: HACResult


def _quantile10(values):
    if not values:
        return None
    ordered = sorted(values)
    pos = .1 * (len(ordered) - 1)
    i = math.floor(pos)
    return ordered[i] if i == len(ordered) - 1 else ordered[i] + (pos - i) * (ordered[i + 1] - ordered[i])


def evaluate(full_calendar: CalendarBinding, pairs: Sequence[FrozenPair],
             outcomes: Sequence[Outcome], partial_calendar: CalendarBinding | None = None) -> Evaluation:
    """New paired mask from frozen lists; historical screen cannot confirm alpha.

    Full/partial calendars are independently bound. Require a frozen pair on every
    declared score date; missing selected labels exclude whole dates. Partial
    dates never enter pooled primary, fold vote, tails, selected P10 or HAC.
    """
    full_calendar.validate(date(2024, 1, 1), date(2026, 7, 1))
    if partial_calendar:
        partial_calendar.validate(date(2026, 7, 1), date(2026, 10, 16))
    expected = full_calendar.sessions + (partial_calendar.sessions if partial_calendar else ())
    by_date = {}
    for pair in pairs:
        base, cand = pair.baseline, pair.candidate
        _validate_selection(base)
        _validate_selection(cand)
        t = base.signal_date
        identity = lambda s: tuple((r.code, r.base_score, r.base_rank) for r in s.top_k)
        if (t in by_date or t != cand.signal_date or base.h2_session != cand.h2_session
                or base.lambda_value != 0 or identity(base) != identity(cand)
                or base.selected != base.top_k[:10]):
            raise ContractError("ambiguous or mismatched frozen selection pair")
        by_date[t] = pair
    if set(by_date) != set(expected):
        raise ContractError("incomplete outer score/list coverage")
    labels = _outcome_index(outcomes)
    dates, base_events, cand_events = [], [], []
    for t in expected:
        base, cand = by_date[t].baseline, by_date[t].candidate
        arms = [[labels.get((t, r.code)) for r in arm.selected] for arm in (base, cand)]
        known = [sum(_known(label) for label in arm) for arm in arms]
        for arm in arms:
            for label in arm:
                if _known(label) and _date(label.label_end, "label_end") != base.h2_session:
                    raise ContractError("evaluation label_end differs from bound h2 session")
        complete = len(base.selected) == len(cand.selected) == known[0] == known[1] == 10
        br, cr = (_mean([label.value for label in arm]) for arm in arms) if complete else (None, None)
        bkeys, ckeys = {r.code for r in base.selected}, {r.code for r in cand.selected}
        entrants, exits = tuple(sorted(ckeys - bkeys)), tuple(sorted(bkeys - ckeys))
        delta = _finite(cr - br, "paired delta") if complete else None
        dates.append(PairedDate(t, fold_id(t), complete, len(base.selected), len(cand.selected),
                                *known, len(base.selected) - known[0], len(cand.selected) - known[1],
                                "complete" if complete else "incomplete_selected_Top10_labels_or_slots",
                                br, cr, delta, entrants, exits))
        if complete and fold_id(t) in COMPLETE_FOLDS:
            base_events.extend(label.value for label in arms[0])
            cand_events.extend(label.value for label in arms[1])
    full = [row for row in dates if row.fold in COMPLETE_FOLDS and row.paired_complete]
    means = tuple((f, _mean([r.delta for r in full if r.fold == f])
                   if any(r.fold == f for r in full) else None) for f in COMPLETE_FOLDS)
    partial = [r.delta for r in dates if r.fold == PARTIAL_FOLD and r.paired_complete]
    mu = _mean([r.delta for r in full]) if full else None
    cb = fractional_cvar5([r.baseline_return for r in full]) if full else None
    cc = fractional_cvar5([r.candidate_return for r in full]) if full else None
    positive = sum(m is not None and m > 0 for _, m in means)
    screen = (mu >= .0008 and positive >= 4 and cc >= cb) if all(m is not None for _, m in means) else None
    decision = "CANNOT_CONFIRM" if screen is None else ("DIRECTION_ONLY" if screen else "NO_BREAKTHROUGH")
    hac = calendar_hac20(full_calendar.sessions, {r.signal_date: r.delta for r in dates if r.fold in COMPLETE_FOLDS})
    return Evaluation(tuple(dates), means, _mean(partial) if partial else None, mu,
                      statistics.median([r.delta for r in full]) if full else None,
                      _quantile10(base_events), _quantile10(cand_events), cb, cc,
                      positive, screen, decision, hac)
