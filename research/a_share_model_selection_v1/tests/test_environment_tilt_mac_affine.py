"""Synthetic-only affine tests; no actual payload/registry/network access."""
from __future__ import annotations

import copy
import json
import sys
from decimal import Decimal, localcontext, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_mac_affine as m
import environment_tilt_mac_recovery as base
from test_environment_tilt_mac_recovery import fixture_run, namespace, frame, PROJECT


def event(date, dividend=0, rights_price=0, bonus=0, rights=0):
    return dict(_date_int=date, category=1, xdxr_fenhong=dividend,
        xdxr_peigujia=rights_price, xdxr_songzhuangu=bonus, xdxr_peigu=rights)


def chosen_fixture():
    raw = frame()
    oracle = raw.copy()
    oracle[list(m.OHLC)] = oracle[list(m.OHLC)] * 2 + 3
    return raw, oracle, base.anchors(raw, oracle)


def test_affine_cash_dividend_is_additive_not_ratio():
    a, b = m.event_affine(event(20210101, dividend=20))
    assert (a, b) == (Decimal(1), Decimal(-2))
    assert a * 8 + b == 6 and a * 12 + b == 10


def test_rights_bonus_formula_and_ordered_composition():
    rights = event(20210102, rights_price=4, bonus=5, rights=5)
    assert m.event_affine(rights) == (Decimal('.5'), Decimal(1))
    a, b = m.compose_events([event(20210101, dividend=20), rights])
    assert (a, b) == (Decimal('.5'), Decimal(0))
    assert a * 10 + b == 5  # rights(cash(10)), not cash(rights(10)) ==4


def test_composition_rejects_duplicate_or_reversed_source_dates():
    for events in ([event(1), event(1)], [event(2), event(1)]):
        with pytest.raises(m.CannotConfirm, match='AMBIGUOUS'):
            m.compose_events(events)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), None, 'bad'])
def test_nonfinite_missing_scalars_fail_closed(bad):
    with pytest.raises(m.CannotConfirm):
        m.decimal_scalar(bad)


@pytest.mark.parametrize('change', ['missing', 'category', 'denominator'])
def test_invalid_action_fields_are_unknown_not_neutral(change):
    e = event(1)
    if change == 'missing': del e['xdxr_peigu']
    if change == 'category': e['category'] = 11
    if change == 'denominator': e['xdxr_songzhuangu'] = -10
    with pytest.raises(m.CannotConfirm): m.event_affine(e)


def test_negative_and_zero_predictions_are_permitted():
    raw, oracle, chosen = chosen_fixture()
    oracle[list(m.OHLC)] = raw[list(m.OHLC)] - 10
    empty = pd.DataFrame(columns=['_date_int', *base.ACTION_FIELDS])
    rebuilt, calibration = m.reconstruct_holdout(raw, empty, chosen,
        oracle.set_index('trade_date').loc[chosen['B']])
    assert rebuilt[0]['low']['cent'] == '-2.00'
    assert rebuilt[0]['close']['cent'] == '0.00'
    assert calibration['v'] == '-10.0'


def test_anchor_holdout_and_no_action_unknown_are_unchanged():
    raw, oracle, chosen = chosen_fixture()
    assert chosen['B'] == int(raw.trade_date.iloc[20])
    assert chosen['H'] == raw.trade_date.iloc[10:20].tolist()
    empty = pd.DataFrame(columns=['_date_int', *base.ACTION_FIELDS])
    rebuilt, calibration = m.reconstruct_holdout(raw, empty, chosen,
        oracle.set_index('trade_date').loc[chosen['B']])
    assert calibration['observed_cache_events'] == 0
    assert calibration['action_completeness'] == 'CANNOT_CONFIRM'
    assert 'ZERO_EVENTS' in calibration['operator_evidence']
    report, mismatches = m.compare_prices(rebuilt, oracle.loc[oracle.trade_date.isin(chosen['H'])])
    assert mismatches == 0 and len(report) == 10


def test_event_on_t_excluded_event_on_B_included_and_nontrading_gap_order():
    raw, oracle, chosen = chosen_fixture()
    dates = chosen['H']
    events = pd.DataFrame([event(dates[1], dividend=10), event(chosen['B'], dividend=20)])
    rebuilt, info = m.reconstruct_holdout(raw, events, chosen,
        oracle.set_index('trade_date').loc[chosen['B']])
    assert Decimal(rebuilt[0]['close']['unrounded']) == 17  #2*(10-1-2)+3
    assert Decimal(rebuilt[1]['close']['unrounded']) == 19  # event at t excluded
    assert info['observed_cache_events'] == 2
    a, b = m.compose_events([event(20210508, dividend=10), event(20210509, dividend=20)])
    assert a * 10 + b == 7  # multiple events between observed stock dates


@pytest.mark.parametrize('failure', ['missing', 'flat_raw', 'flat_oracle', 'nonfinite'])
def test_missing_or_degenerate_fixed_B_never_substitutes(failure):
    raw, oracle, chosen = chosen_fixture()
    if failure == 'missing': raw = raw.loc[raw.trade_date.ne(chosen['B'])]
    if failure == 'flat_raw': raw.loc[raw.trade_date.eq(chosen['B']), 'high'] = 8
    if failure == 'flat_oracle': oracle.loc[oracle.trade_date.eq(chosen['B']), 'high'] = 19
    if failure == 'nonfinite': oracle.loc[oracle.trade_date.eq(chosen['B']), 'high'] = float('nan')
    with pytest.raises(m.CannotConfirm):
        m.reconstruct_holdout(raw, pd.DataFrame(columns=['_date_int', *base.ACTION_FIELDS]),
            chosen, oracle.set_index('trade_date').loc[chosen['B']])


def test_reference_never_rounded_and_decimal_half_up_signed_ties():
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 50, ROUND_HALF_UP
        assert Decimal('1.005').quantize(m.CENT) == Decimal('1.01')
        assert Decimal('-1.005').quantize(m.CENT) == Decimal('-1.01')
        assert m.cent_reference('1.00') == 1
        with pytest.raises(m.CannotConfirm, match='UNSUPPORTED_REFERENCE_SERIALIZATION'):
            m.cent_reference('1.005')


def test_decimal_precision50_independent_of_ambient_context():
    raw, oracle, chosen = chosen_fixture()
    raw['low'], raw['high'] = 1., 4.
    oracle['low'], oracle['high'] = 1., 2.
    with localcontext() as ctx:
        ctx.prec = 6
        _, info = m.reconstruct_holdout(raw, pd.DataFrame(columns=['_date_int', *base.ACTION_FIELDS]),
            chosen, oracle.set_index('trade_date').loc[chosen['B']])
    assert info['u'] == '0.' + '3' * 50


def test_residuals_retained_and_no_epsilon_fitting():
    raw, oracle, chosen = chosen_fixture()
    rebuilt, _ = m.reconstruct_holdout(raw, pd.DataFrame(columns=['_date_int', *base.ACTION_FIELDS]),
        chosen, oracle.set_index('trade_date').loc[chosen['B']])
    expected = oracle.loc[oracle.trade_date.isin(chosen['H'])].copy()
    expected.loc[expected.index[0], 'close'] += .01
    report, mismatches = m.compare_prices(rebuilt, expected)
    assert mismatches == 1
    assert report[0]['fields']['close']['unrounded_residual'] == '-0.01'
    assert report[0]['fields']['close']['exact_cent_match'] is False


def test_six_fixed_symbols_and_ancillary_fields_not_normalized():
    raws, oracles = {}, []
    for code in base.CODES:
        raw = frame(code=code)
        qfq = raw.copy()
        qfq[list(m.OHLC)] = raw[list(m.OHLC)] * 2 + 3
        qfq['amount'] += 1
        raws[code] = raw
        oracles.append(qfq)
    result = m.diagnose(raws, pd.concat(oracles), {})
    assert len(result['symbols']) == 6 and result['training_ready'] is False
    assert result['decision'] == 'SAMPLED_AFFINE_CENT_MATCH_UNCONFIRMED_HYPOTHESIS_ONLY'
    for item in result['symbols'].values():
        assert item['raw_ancillary_comparison']['amount']['mismatches'] == 10
        assert item['calibration']['action_completeness'] == 'CANNOT_CONFIRM'


def test_missing_H_reports_unknown_no_alternative():
    raw = frame(15)
    raws = {code: raw.assign(ts_code=code) for code in base.CODES}
    result = m.diagnose(raws, pd.concat(raws.values()), {})
    assert result['decision'] == 'CANNOT_CONFIRM'
    assert all(x['reason'] == 'MISSING_FIXED_ANCHOR_OR_HOLDOUT' for x in result['symbols'].values())


def test_new_route_preserves_old_lifecycle_and_openbook_identity():
    c = dict(execution_id='SYNTHETIC_AFFINE_01', contract_path='c', contract_sha256='a',
        config_path='i', config_sha256='b', code_composite_sha256='d',
        diagnostic_dates={}, caps=base.CAPS)
    e = m.AffineLifecycle(None, c).event('RUN_STARTED', 'RUNNING', 'r', {})
    assert e['route_id'].endswith('affine-diagnostic-20261021')
    assert e['truth_access'] == 'NOT_READ'
    assert e['sealed_status'] == 'OPENBOOK_M7_REVEALED_INPUT_SEMANTICS_ONLY'
    assert e['deterministic_metrics']['real_fit_budget_debit'] == 0


@pytest.mark.parametrize('kind', ['RUN_STARTED', 'TERMINAL'])
def test_baseline_id_matches_frozen_M8_contract_for_start_and_terminal(kind):
    contract = PROJECT / 'research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020/M8_CONTRACT.json'
    expected = json.loads(contract.read_text())['BASELINE_ID']
    c = dict(execution_id='SYNTHETIC_AFFINE_01', contract_path=str(contract), contract_sha256='a',
        config_path='i', config_sha256='b', code_composite_sha256='d',
        diagnostic_dates={}, caps=base.CAPS)
    e = m.AffineLifecycle(None, c).event(kind, 'RUNNING' if kind == 'RUN_STARTED' else
        'SAMPLED_AFFINE_CENT_MATCH_UNCONFIRMED_HYPOTHESIS_ONLY', 'r', {})
    assert e['baseline_id'] == expected


def test_registered_synthetic_run_guards_pools_and_retains_terminal(fixture_run, monkeypatch):
    c, adapter = fixture_run
    c['run_dir'] = str(Path(c['fixture_root']) / 'M8_SYNTHETIC_01')
    c['execution_id'] = 'MAC_AFFINE_SYNTHETIC_01'
    import pyarrow as pa
    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)
    original_read = pd.read_parquet
    calls = []
    def checked_read(*args, **kwargs):
        events = [json.loads(x) for x in Path(c['registry_path']).read_text().splitlines()]
        assert any(x['execution_id'] == c['execution_id'] and x['event_type'] == 'RUN_STARTED' for x in events)
        assert (pa.cpu_count(), pa.io_thread_count()) == (4, 4)
        calls.append(1)
        return original_read(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', checked_read)
    result = m.run(c, adapter, fixture=True)
    assert len(calls) == 2
    assert result['arrow_pools'] == dict(cpu=4, io=4)
    assert result['status'] == 'COMPLETED' and result['training_ready'] is False
    events = [json.loads(x) for x in Path(c['registry_path']).read_text().splitlines()]
    terminal = [e for e in events if e['execution_id'] == c['execution_id']][-1]
    assert terminal['event_type'] == 'TERMINAL'
    assert terminal['runtime']['arrow_pools'] == dict(cpu=4, io=4)
    with pytest.raises(FileExistsError): m.run(c, adapter, fixture=True)


@pytest.mark.parametrize('failure', ['hash', 'resource', 'interrupt'])
def test_synthetic_failure_terminal_without_unregistered_read(fixture_run, monkeypatch, failure):
    c, adapter = fixture_run
    c['run_dir'] = str(Path(c['fixture_root']) / 'M8_SYNTHETIC_01')
    c['execution_id'] = 'MAC_AFFINE_SYNTHETIC_01'
    if failure == 'hash': c['qfq']['sha256'] = '0' * 64
    if failure == 'resource': monkeypatch.setattr(base, 'available_memory', lambda: 0)
    if failure == 'interrupt': monkeypatch.setattr(base, 'pure_functions', lambda c: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(BaseException): m.run(c, adapter, fixture=True)
    events = [json.loads(x) for x in Path(c['registry_path']).read_text().splitlines()]
    current = [e for e in events if e['execution_id'] == c['execution_id']]
    assert [x['event_type'] for x in current] == ['RUN_STARTED', 'TERMINAL']
    assert current[-1]['deterministic_metrics']['real_fit_budget_debit'] == 0
