"""Synthetic-only frozen mathematics; no real payload/model/registry/network IO."""
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
import statistics
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_model as m


SHA = "a" * 64


def day(t=date(2023, 1, 3), n=50, breadth=1., slope=.0001):
    return m.ScoreDay(t, t + timedelta(days=1), t + timedelta(days=2),
                      tuple(m.ScoreRow(f"{i:06}.SZ", .02 - i * slope,
                                       10. if i % 2 else -10., "observed")
                            for i in range(n)), breadth)


def binding(start, end, sessions):
    return m.CalendarBinding(start, end, tuple(sessions), SHA, SHA)


def oot(d, max_end=None):
    return m.OOTDay(d, max_end or d.signal_date - timedelta(days=1), SHA)


def outcomes(d, values=None):
    return tuple(m.Outcome(d.signal_date, r.code,
                           values[i] if values is not None else r.score,
                           d.h2_session) for i, r in enumerate(d.rows))


def fit(days, labels=None, outer=date(2024, 1, 1)):
    return m.fit_lambda(outer, binding(outer.replace(year=outer.year - 2), outer,
                                      [d.signal_date for d in days]),
                        [oot(d) for d in days],
                        labels if labels is not None else sum((outcomes(d) for d in days), ()))


def test_config_frozen_identity_safety_and_no_tunable_objective():
    import inspect
    import json
    config = json.loads(m.POLICY_JSON)
    assert m.CONFIG_SHA256 == hashlib.sha256(m.POLICY_JSON.encode()).hexdigest()
    assert (config['rho'], config['lambda_max'], config['scale_cap']) == (1, .25, .01)
    assert config['confirmation_gate'] is None
    assert (config['research_only'], config['production_allowed'],
            config['buy_gate_updated'], config['orders_sent']) == (True, False, False, False)
    assert list(inspect.signature(m.fit_lambda).parameters) == ['outer_start', 'coverage', 'days', 'outcomes']
    assert 'outcomes' not in inspect.signature(m.select).parameters


def test_lambda_zero_is_exact_before_auxiliary_arithmetic():
    d = day(n=12)
    # Invalid auxiliary payloads must be untouched on the exact M0 identity path.
    d = replace(d, breadth=float('nan'), rows=tuple(replace(r, prior=float('inf'), prior_status='bogus') for r in d.rows))
    result = m.select(d, 0.)
    assert tuple(r.code for r in result.selected) == tuple(r.code for r in d.rows[:10])
    assert tuple(r.score for r in result.selected) == tuple(r.score for r in d.rows[:10])
    assert result.scale is None and result.gate is None
    zeros = replace(day(n=2), rows=(m.ScoreRow('000001.SZ', -0.), m.ScoreRow('000002.SZ', 0.)))
    assert math.copysign(1, m.select(zeros, 0.).selected[0].score) == -1


def test_stable_ties_cut_exactly_50_no_tie_expansion_and_no_refill():
    d = day(n=60, slope=0)
    d = replace(d, rows=tuple(reversed(d.rows)))
    result = m.select(d, .25)
    assert [r.code for r in result.top_k] == [f'{i:06}.SZ' for i in range(50)]
    assert [r.code for r in result.selected] == [f'{i:06}.SZ' for i in range(10)]
    assert result.activation == 'zero_scale'
    short = m.select(day(n=7), .25)
    assert len(short.selected) == len(short.top_k) == 7


def test_positive_tilt_cap_gate_and_population_scale():
    d = day(slope=.1)
    selected = m.select(d, .25)
    assert selected.scale == .01 and selected.gate == 1
    for original, scored in zip(sorted(d.rows, key=lambda r: (-r.score, r.code)), selected.top_k):
        expected_u = .01 * math.tanh(max(original.prior, 0))
        assert scored.unit_addition == pytest.approx(expected_u)
        assert scored.score == original.score + .25 * expected_u
        assert 0 <= .25 * scored.unit_addition <= .0025
    d = day(n=2)
    result = m.select(d, .25)
    assert result.scale == statistics.pstdev([r.score for r in d.rows])
    assert m.select(replace(d, breadth=.5), .25).gate == 0
    assert m.select(replace(d, breadth=.75), .25).gate == .5


@pytest.mark.parametrize('status', ['missing', 'unknown', 'unmapped', 'zero_support'])
def test_original_missing_prior_status_is_neutral_and_preserves_pool(status):
    d = day()
    d = replace(d, rows=tuple(replace(r, prior=float('nan'), prior_status=status) for r in d.rows))
    result = m.select(d, .25)
    assert all(r.unit_addition == 0 for r in result.top_k)
    assert [r.code for r in result.selected] == [r.code for r in d.rows[:10]]
    assert len(result.top_k) == 50


def test_missing_environment_retains_date_and_stocks():
    result = m.select(replace(day(), breadth=None), .25)
    assert result.gate == 0
    assert result.activation == 'deactivated_missing_environment'
    assert len(result.selected) == 10


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), -.1, 1.1, True])
def test_invalid_breadth_stops(bad):
    with pytest.raises(m.ContractError, match='breadth'):
        m.select(replace(day(), breadth=bad), .25)


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf')])
def test_unexplained_observed_nonfinite_prior_stops_even_when_gate_off(bad):
    d = day(breadth=.2)
    d = replace(d, rows=(replace(d.rows[0], prior=bad), *d.rows[1:]))
    with pytest.raises(m.ContractError, match='prior'):
        m.select(d, .25)


@pytest.mark.parametrize('lam', [-.1, .25001, float('nan'), True])
def test_lambda_outside_fixed_bounds_stops(lam):
    with pytest.raises(m.ContractError):
        m.select(day(), lam)


def test_nonfinite_scores_duplicate_keys_and_bad_calendar_stop_without_filtering():
    d = day(n=60)
    # A bad row outside Top50 still invalidates the ORIGINAL full finite pool.
    for rows in ((*d.rows[:-1], replace(d.rows[-1], score=float('nan'))), (*d.rows, d.rows[0])):
        with pytest.raises(m.ContractError):
            m.select(replace(d, rows=rows), 0.)
    with pytest.raises(m.ContractError, match='calendar anchor'):
        m.select(replace(d, next_session=d.signal_date), 0.)


def test_closed_form_residual_centering_and_exact_date_equal_not_row_equal():
    days = [day(date(2022, 1, 4), n=2), day(date(2023, 5, 4), n=50)]
    ys, cross, squares = [], [], []
    for d, coefficient in zip(days, [.2, 2.]):
        selection = m.select(d, .25)
        units = [r.unit_addition for r in selection.top_k]
        # Add an arbitrary daily intercept: it must vanish in the centered target.
        y = [r.score + coefficient * u + 17 for r, u in zip(d.rows, units)]
        ys.extend(outcomes(d, y))
        a = [(u - statistics.mean(units)) / .01 for u in units]
        r = [((yi - statistics.mean(y)) - (p.score - statistics.mean([p.score for p in d.rows]))) / .01
             for yi, p in zip(y, d.rows)]
        cross.append(statistics.mean(ai * ri for ai, ri in zip(a, r)))
        squares.append(statistics.mean(ai * ai for ai in a))
    expected = min(.25, max(0., statistics.mean(cross) / (statistics.mean(squares) + 1)))
    row_equal = (2 * cross[0] + 50 * cross[1]) / (2 * squares[0] + 50 * squares[1] + 52)
    result = fit(days, ys)
    assert result.lambda_value == pytest.approx(expected, rel=1e-10)
    assert abs(result.lambda_value - row_equal) > 1e-4
    assert result.rho == 1 and result.status == 'FITTED'
    assert result.complete_label_dates == result.scored_dates == 2


def test_fit_fixed_clip_nonnegative_and_no_identifiable_tilt():
    d = day(slope=.01)
    u = [r.unit_addition for r in m.select(d, .25).top_k]
    for multiplier, expected in [(10000, .25), (-10000, 0)]:
        labels = outcomes(d, [r.score + multiplier * ui for r, ui in zip(d.rows, u)])
        assert fit([d], labels).lambda_value == expected
    neutral = replace(d, breadth=.1)
    result = fit([neutral])
    assert result.lambda_value == 0 and result.status == 'NO_IDENTIFIABLE_TILT'
    assert result.complete_label_dates == 1


def test_training_topk_before_label_join_whole_date_exclusion_and_no_refill():
    d = day(n=60)
    labels = outcomes(d)
    missing_selected = tuple(replace(row, value=None) if i == 49 else row for i, row in enumerate(labels))
    result = fit([d], missing_selected)
    assert result.complete_label_dates == 0 and result.lambda_value == 0
    assert result.train_mask == ((d.signal_date, False, 'incomplete_TopK_labels'),)
    missing_outside_topk = tuple(replace(row, value=float('nan')) if i == 59 else row for i, row in enumerate(labels))
    assert fit([d], missing_outside_topk).complete_label_dates == 1


def test_training_time_boundaries_and_missing_coverage_stop():
    d = day()
    coverage = binding(date(2022, 1, 1), date(2024, 1, 1), [d.signal_date])
    with pytest.raises(m.ContractError, match='precede'):
        m.fit_lambda(date(2024, 1, 1), coverage, [oot(d, d.signal_date)], outcomes(d))
    with pytest.raises(m.ContractError, match='two-year score coverage'):
        m.fit_lambda(date(2024, 1, 1), coverage, [], ())
    with pytest.raises(m.ContractError, match='coverage bounds'):
        m.fit_lambda(date(2024, 1, 1), replace(coverage, start=date(2023, 1, 1)), [oot(d)], outcomes(d))
    with pytest.raises(m.ContractError, match='identity'):
        m.fit_lambda(date(2024, 1, 1), coverage, [replace(oot(d), score_model_sha256='')], outcomes(d))
    with pytest.raises(m.ContractError, match='label_end'):
        fit([d], tuple(replace(r, label_end=d.next_session) for r in outcomes(d)))
    boundary = day(date(2023, 12, 30))
    result = fit([boundary])
    assert result.complete_label_dates == 0
    assert result.train_mask[0][2] == 'label_not_mature_before_outer_start'


def test_duplicate_training_labels_or_dates_stop():
    d = day()
    with pytest.raises(m.ContractError, match='duplicate outcome'):
        fit([d], (*outcomes(d), outcomes(d)[0]))
    with pytest.raises(m.ContractError, match='duplicate OOT'):
        m.fit_lambda(date(2024, 1, 1), binding(date(2022, 1, 1), date(2024, 1, 1),
                                              [d.signal_date]), [oot(d), oot(d)], outcomes(d))


def test_identical_nonzero_units_are_no_identifiable_tilt():
    d = day()
    d = replace(d, rows=tuple(replace(r, prior=1.) for r in d.rows))
    result = fit([d], outcomes(d, [r.score + i * .1 for i, r in enumerate(d.rows)]))
    assert result.nonzero_unit_dates == 1
    assert result.denominator_without_rho == 0
    assert result.lambda_value == 0 and result.status == 'NO_IDENTIFIABLE_TILT'


@pytest.mark.parametrize('n', [1, 19, 20, 21, 30, 60])
def test_fractional_mass_cvar_is_tail_mean_not_quantile(n):
    values = list(range(n))
    mass = n / 20
    # Independent integration of the empirical inverse CDF over [0,.05].
    expected = sum(max(0, min(1, mass - i)) * value for i, value in enumerate(values)) / mass
    assert m.fractional_cvar5(values) == pytest.approx(expected)
    assert m.fractional_cvar5(list(reversed(values))) == pytest.approx(expected)
    if n == 30:
        assert expected == pytest.approx(1 / 3)


def evaluation_fixture():
    # Calendar sessions are explicitly SYNTHETIC, not claimed observed history.
    dates = (date(2024, 1, 3), date(2024, 7, 3), date(2025, 1, 3),
             date(2025, 7, 3), date(2026, 1, 3))
    full = binding(date(2024, 1, 1), date(2026, 7, 1), dates)
    days = tuple(day(t) for t in dates)
    pairs = tuple(m.freeze_pair(d, .25) for d in days)
    labels = []
    for d, pair in zip(days, pairs):
        entrants = {r.code for r in pair.candidate.selected} - {r.code for r in pair.baseline.selected}
        assert entrants
        labels.extend(outcomes(d, [.02 if row.code in entrants else 0. for row in d.rows]))
    return full, days, pairs, tuple(labels)


def test_paired_mask_new_selected_keys_frozen_before_labels_and_swap_identity():
    full, days, pairs, labels = evaluation_fixture()
    hashes = tuple((p.baseline.sha256, p.candidate.sha256) for p in pairs)
    result = m.evaluate(full, pairs, labels)
    assert result.decision == 'DIRECTION_ONLY' and result.historical_screen is True
    assert result.positive_full_folds == 5
    assert result.mean_delta >= .0008
    assert all(r.delta == pytest.approx(len(r.entrants) * .02 / 10) for r in result.dates)
    assert all(len(r.entrants) == len(r.exits) for r in result.dates)
    assert result.cvar5_candidate >= result.cvar5_base
    # Remove a candidate-only selected label: baseline remains complete, pair lost.
    entrant = result.dates[0].entrants[0]
    changed = tuple(replace(r, value=None) if (r.signal_date, r.code) == (days[0].signal_date, entrant) else r for r in labels)
    bad = m.evaluate(full, pairs, changed)
    first = bad.dates[0]
    assert not first.paired_complete and (first.base_known, first.candidate_known) == (10, 9)
    assert (first.base_unknown, first.candidate_unknown) == (0, 1)
    assert first.delta is None and bad.historical_screen is None and bad.decision == 'CANNOT_CONFIRM'
    assert hashes == tuple((p.baseline.sha256, p.candidate.sha256) for p in pairs)
    # Unknown labels never alter the pre-frozen list or trigger a refill.
    assert pairs[0].candidate.selected == m.select(days[0], .25).selected


def test_partial_fold_excluded_from_primary_votes_tails_and_hac():
    full, _, pairs, labels = evaluation_fixture()
    pday = day(date(2026, 8, 3))
    partial = binding(date(2026, 7, 1), date(2026, 10, 16), [pday.signal_date])
    ppair = m.freeze_pair(pday, .25)
    entrant = {r.code for r in ppair.candidate.selected} - {r.code for r in ppair.baseline.selected}
    plabels = outcomes(pday, [-1000 if r.code in entrant else 0 for r in pday.rows])
    a = m.evaluate(full, pairs, labels)
    b = m.evaluate(full, (*pairs, ppair), (*labels, *plabels), partial)
    assert b.partial_fold_mean < 0
    for name in ('mean_delta', 'median_delta', 'cvar5_base', 'cvar5_candidate',
                 'positive_full_folds', 'historical_screen', 'decision', 'hac',
                 'selected_event_p10_base', 'selected_event_p10_candidate'):
        assert getattr(a, name) == getattr(b, name)


def test_primary_is_equal_dates_not_equal_folds():
    full, _, pairs, labels = evaluation_fixture()
    extra = day(date(2024, 2, 3))
    pair = m.freeze_pair(extra, .25)
    entrants = {r.code for r in pair.candidate.selected} - {r.code for r in pair.baseline.selected}
    extra_labels = outcomes(extra, [.2 if r.code in entrants else 0 for r in extra.rows])
    sessions = tuple(sorted((*full.sessions, extra.signal_date)))
    result = m.evaluate(replace(full, sessions=sessions), (*pairs, pair), (*labels, *extra_labels))
    assert result.mean_delta == pytest.approx(statistics.mean(r.delta for r in result.dates))
    assert abs(result.mean_delta - statistics.mean(v for _, v in result.full_fold_means)) > 1e-4


def test_evaluation_exact_route_negative_result_and_no_sealed_upgrade():
    full, _, pairs, labels = evaluation_fixture()
    negative = tuple(replace(r, value=-r.value) for r in labels)
    result = m.evaluate(full, pairs, negative)
    assert result.decision == 'NO_BREAKTHROUGH' and result.historical_screen is False
    assert result.mean_delta < 0 and result.positive_full_folds == 0
    same = tuple(m.FrozenPair(p.baseline, p.baseline) for p in pairs)
    zero = m.evaluate(full, same, labels)
    assert zero.mean_delta == 0 and zero.decision == 'NO_BREAKTHROUGH'


def test_evaluation_short_lists_no_refill_and_missing_score_dates_stop():
    full, days, pairs, labels = evaluation_fixture()
    short = day(days[0].signal_date, n=7)
    result = m.evaluate(full, (m.freeze_pair(short, .25), *pairs[1:]), labels)
    assert result.dates[0].base_selected == result.dates[0].candidate_selected == 7
    assert not result.dates[0].paired_complete
    with pytest.raises(m.ContractError, match='coverage'):
        m.evaluate(full, pairs[1:], labels)
    with pytest.raises(m.ContractError, match='mismatched'):
        m.evaluate(full, (m.FrozenPair(pairs[0].baseline, pairs[1].candidate), *pairs[1:]), labels)
    with pytest.raises(m.ContractError, match='label_end'):
        m.evaluate(full, pairs, tuple(replace(r, label_end=r.signal_date) for r in labels))


def test_readback_rejects_forged_label_selected_refill_or_duplicate_list():
    full, _, pairs, labels = evaluation_fixture()
    pair = pairs[0]
    # It is not sufficient for an arbitrary label-chosen list to have ten rows.
    for chosen in (pair.candidate.top_k[20:30], (pair.candidate.selected[0],) * 10):
        forged = m.FrozenPair(pair.baseline, replace(pair.candidate, selected=chosen))
        with pytest.raises(m.ContractError, match='label-independent'):
            m.evaluate(full, (forged, *pairs[1:]), labels)
    corrupt = replace(pair.candidate.top_k[0], score=float('inf'))
    forged = m.FrozenPair(pair.baseline, replace(pair.candidate, top_k=(corrupt, *pair.candidate.top_k[1:])))
    with pytest.raises(m.ContractError, match='frozen score'):
        m.evaluate(full, (forged, *pairs[1:]), labels)


def direct_hac(calendar, deltas):
    observed = [v for v in deltas.values() if v is not None]
    mu = statistics.mean(observed)
    # Independent double-sum kernel, including pairs across any fold boundary.
    e = [deltas[t] - mu if deltas.get(t) is not None else 0 for t in calendar]
    variance = sum(e[i] * e[j] * (1 - abs(i-j) / 21)
                   for i in range(len(e)) for j in range(len(e)) if abs(i-j) <= 20) / len(observed) ** 2
    return variance


def test_hac_preserves_calendar_gaps_and_cross_fold_lag_pairs():
    calendar = tuple(date(2024, 6, 15) + timedelta(days=i) for i in range(42))
    deltas = {t: ((i % 5) - 2) * .01 if i not in {4, 7, 20, 29} else None for i, t in enumerate(calendar)}
    result = m.calendar_hac20(calendar, deltas)
    assert result.variance == pytest.approx(direct_hac(calendar, deltas), abs=1e-18)
    compressed = tuple(t for t in calendar if deltas[t] is not None)
    assert result.variance != pytest.approx(direct_hac(compressed, deltas), rel=1e-3)
    assert result.paired_dates == 38 and result.calendar_sessions == 42
    # A reset at July 1 deletes real cross-boundary lag products.
    mu = statistics.mean([v for v in deltas.values() if v is not None])
    e = [deltas[t] - mu if deltas[t] is not None else 0 for t in calendar]
    reset = sum(e[i] * e[j] * (1 - abs(i-j) / 21)
                for i in range(42) for j in range(42)
                if abs(i-j) <= 20 and calendar[i].month == calendar[j].month) / 38 ** 2
    assert abs(result.variance - reset) > 1e-8
    assert result.status == 'WORKING_SE_ONLY_UNCALIBRATED'
    assert result.normal_reference_uncalibrated[0] == pytest.approx(result.mean - 1.96 * result.standard_error)


def test_hac_nonpositive_insufficient_empty_and_nonfinite_do_not_fake_zero_se():
    calendar = tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(30))
    for dates, values in [(calendar, {t: 1. for t in calendar}), (calendar[:20], {calendar[0]: .1}), (calendar, {})]:
        result = m.calendar_hac20(dates, values)
        assert result.status == 'CANNOT_CONFIRM' and result.standard_error is None
        assert result.normal_reference_uncalibrated is None
    with pytest.raises(m.ContractError, match='delta'):
        m.calendar_hac20(calendar, {calendar[0]: float('nan')})
    with pytest.raises(m.ContractError, match='calendar'):
        m.calendar_hac20((*calendar, calendar[0]), {})


def test_calendar_binding_identity_order_and_bounds_fail_closed():
    c = binding(date(2022, 1, 1), date(2024, 1, 1), [date(2023, 1, 1)])
    for broken in (replace(c, calendar_sha256=''), replace(c, sessions=(date(2023, 1, 1),) * 2),
                   replace(c, sessions=(date(2024, 1, 1),)), replace(c, sessions=())):
        with pytest.raises(m.ContractError):
            broken.validate(date(2022, 1, 1), date(2024, 1, 1))


def test_partial_unobserved_h2_retains_exact_lists_and_invalid_pair_without_changing_full_screen():
    full,_,pairs,labels=evaluation_fixture()
    observed=day(date(2026,10,15))
    unobserved=replace(observed,h2_session=None)
    p=m.freeze_pair(unobserved,.25)
    original=m.freeze_pair(observed,.25)
    assert p.baseline.selected==original.baseline.selected
    assert p.candidate.selected==original.candidate.selected
    assert p.baseline.h2_session is p.candidate.h2_session is None
    assert p.baseline.sha256==m.select(unobserved,0).sha256
    # The identity path still skips invalid auxiliary arithmetic.
    untouched=replace(unobserved,breadth=float('nan'),rows=tuple(replace(r,prior=float('inf'),prior_status='bad') for r in unobserved.rows))
    assert m.select(untouched,0).selected==p.baseline.selected
    unknown=tuple(m.Outcome(unobserved.signal_date,r.code,None,None) for r in unobserved.rows)
    partial=binding(date(2026,7,1),date(2026,10,16),[unobserved.signal_date])
    before=m.evaluate(full,pairs,labels)
    after=m.evaluate(full,(*pairs,p),(*labels,*unknown),partial)
    last=after.dates[-1]
    assert last.signal_date==date(2026,10,15) and not last.paired_complete
    assert last.base_selected==last.candidate_selected==10
    assert last.base_known==last.candidate_known==0 and last.base_unknown==last.candidate_unknown==10
    assert last.reason=='h2_calendar_anchor_unavailable'
    assert last.delta is last.baseline_return is last.candidate_return is after.partial_fold_mean is None
    for name in ('mean_delta','median_delta','cvar5_base','cvar5_candidate','positive_full_folds',
                 'historical_screen','decision','hac','selected_event_p10_base','selected_event_p10_candidate'):
        assert getattr(after,name)==getattr(before,name)


@pytest.mark.parametrize('labels',['none','unknown','known'])
def test_unobserved_h2_is_never_admitted_to_lambda_training(labels):
    d=replace(day(),h2_session=None)
    supplied=() if labels=='none' else outcomes(d,[None]*len(d.rows) if labels=='unknown' else None)
    with pytest.raises(m.ContractError,match='training h2_session'):
        fit([d],supplied)


@pytest.mark.parametrize('end',[None,date(2026,10,19)])
def test_unobserved_h2_with_any_known_selected_outcome_is_a_conflict(end):
    full,_,pairs,labels=evaluation_fixture()
    d=replace(day(date(2026,10,15)),h2_session=None)
    pair=m.freeze_pair(d,.25)
    partial=binding(date(2026,7,1),date(2026,10,16),[d.signal_date])
    known=m.Outcome(d.signal_date,pair.baseline.selected[0].code,.01,end)
    with pytest.raises(m.ContractError,match='label_end'):
        m.evaluate(full,(*pairs,pair),(*labels,known),partial)


def test_missing_h2_does_not_relax_required_next_session_or_known_training_label_consistency():
    d=replace(day(),h2_session=None,next_session=None)
    with pytest.raises(m.ContractError,match='next_session'):
        m.select(d,0)
    d=day();mixed=list(outcomes(d));mixed[0]=replace(mixed[0],value=None)
    mixed[1]=replace(mixed[1],label_end=None)
    with pytest.raises(m.ContractError,match='label_end'):
        fit([d],tuple(mixed))
