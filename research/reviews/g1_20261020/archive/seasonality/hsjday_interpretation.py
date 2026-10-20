"""Chinese descriptive interpretation of accepted aggregates; never reads prices."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from research.a_share_top100_environment_v1 import hsjday_industry_coverage as util

ROOT = Path('/LOCAL_ONLY/factor_factory')
WT = ROOT / 'workspace/.task_worktrees/a-hsjday-20y-20261019'
BATCH = ROOT / 'research/a_share_top100_environment_v1/batches/hsjday_20y_20261019'
TASK = BATCH / 'interpretation_20261020_01'
STUDY = TASK / 'freeze_01'
FULL = BATCH / 'historical_descriptive_04/FULL_01'
PRIOR_BINDING = FULL.parent / 'INPUT_BINDING.json'
SOURCE_MANIFEST_SHA = '02256b4d5f154268f7d167e02841b2fbb08491f2931ecf3f48eb0eb930053fe8'
CODE = Path(__file__).resolve()
TEST = CODE.parent / 'tests/test_hsjday_interpretation.py'
EXPERIMENT = 'A_SHARE_HSJDAY_INTERPRETATION_20261020'
ROUTE = 'a-share-hsjday-aggregate-chinese-seasonality-interpretation-20261020'
TRACK = 'A_SHARE_HSJDAY_AGGREGATE_INTERPRETATION'
SAFE = util.SAFETY_FLAGS
YEARS = np.arange(2006, 2026)
GRIDS = {'industry': [], 'month': ['month'], 'third': ['third'], 'month_third': ['month', 'third']}
IDS = ['grid', 'industry_key', 'month', 'third']


def sha(path):
    return util.sha256_file(Path(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    util.write_new(Path(path), util.canonical_bytes(value))


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ratio(numerator, denominator):
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else None


def present(value):
    return value is not None and np.isfinite(value)


def gt(value, limit=1.):
    return present(value) and value > limit


def ge(value, limit):
    return present(value) and value >= limit


def median(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else None


def period_metrics(hits, expected, industry_hits, industry_expected, mask):
    """Both numerator ratios are pooled from counts, never averaged lifts."""
    h, e = float(hits[mask].sum()), float(expected[mask].sum())
    ih, ie = float(industry_hits[mask].sum()), float(industry_expected[mask].sum())
    lift, baseline = ratio(h, e), ratio(ih, ie)
    seasonal = ratio(lift, baseline) if present(lift) else None
    return {'actual_hits': h, 'expected_hits': e, 'lift': lift,
            'industry_actual_hits': ih, 'industry_expected_hits': ie,
            'industry_lift': baseline, 'seasonal_ratio': seasonal}


def remove_three(hits, expected, industry_hits, industry_expected, score):
    """Rank excess descending, year ascending for deterministic tied scores."""
    finite = np.isfinite(score)
    order = sorted(np.flatnonzero(finite), key=lambda i: (-score[i], int(YEARS[i])))
    removed = order[:3]
    mask = np.ones(20, dtype=bool)
    mask[removed] = False
    remaining = period_metrics(hits, expected, industry_hits, industry_expected, mask)
    positive = np.maximum(score[finite], 0.)
    denominator = float(positive.sum())
    numerator = float(np.maximum(score[removed], 0.).sum())
    share = numerator / denominator if denominator > 0 else None
    if len(removed) < 3:
        remaining['lift'] = remaining['seasonal_ratio'] = None
    return {'removed_years': [int(YEARS[i]) for i in removed],
            'rankable_years': int(finite.sum()), 'positive_excess_total': denominator,
            'top3_positive_excess_share': share, **remaining}


def stability_flags(m):
    eligible = m['eligible'] and m['explainable_taxonomy']
    absolute = (eligible and ge(m['lift'], 1.2) and m['years_lift_above_1'] >= 15
                and gt(m['median_annual_lift']) and gt(m['half1_lift'])
                and gt(m['half2_lift']) and gt(m['absolute_drop3_lift']))
    seasonal_only = (eligible and m['seasonal_years_supported'] == 20
                     and ge(m['seasonal_ratio'], 1.2) and m['years_seasonal_above_1'] >= 15
                     and gt(m['median_annual_seasonal']) and gt(m['half1_seasonal_ratio'])
                     and gt(m['half2_seasonal_ratio']) and gt(m['seasonal_drop3_seasonal_ratio']))
    abs_few = (eligible and ge(m['lift'], 1.2)
               and (m['years_lift_above_1'] <= 10 or not gt(m['median_annual_lift']))
               and present(m['absolute_drop3_lift']) and m['absolute_drop3_lift'] <= 1)
    seasonal_few = (eligible and m['seasonal_years_supported'] == 20 and ge(m['seasonal_ratio'], 1.2)
                    and (m['years_seasonal_above_1'] <= 10 or not gt(m['median_annual_seasonal']))
                    and present(m['seasonal_drop3_seasonal_ratio'])
                    and m['seasonal_drop3_seasonal_ratio'] <= 1)
    return {'stable_absolute': bool(absolute), 'stable_seasonal_only': bool(seasonal_only),
            'stable_seasonal': bool(absolute and seasonal_only),
            'few_year_inflated_absolute': bool(abs_few), 'few_year_inflated_seasonal': bool(seasonal_few)}


def cell_metrics(hits, expected, industry_hits, industry_expected, *, explainable=True):
    arrays = [np.asarray(x, dtype=float) for x in (hits, expected, industry_hits, industry_expected)]
    if any(x.shape != (20,) for x in arrays):
        raise ValueError('exactly twenty aligned years required')
    if any((~np.isfinite(x)).any() or (x < 0).any() for x in arrays):
        raise ValueError('nonfinite or negative input counts')
    hits, expected, industry_hits, industry_expected = arrays
    if ((expected == 0) & (hits > 0)).any() or ((industry_expected == 0) & (industry_hits > 0)).any():
        raise ValueError('positive hits with zero expected')
    if (hits > industry_hits + 1e-8).any() or (expected > industry_expected + 1e-8).any():
        raise ValueError('cell exceeds its annual industry baseline')
    lifts = np.divide(hits, expected, out=np.full(20, np.nan), where=expected > 0)
    base = np.divide(industry_hits, industry_expected, out=np.full(20, np.nan), where=industry_expected > 0)
    seasonal = np.divide(lifts, base, out=np.full(20, np.nan), where=np.isfinite(base) & (base > 0))
    pooled = period_metrics(*arrays, np.ones(20, dtype=bool))
    half1 = period_metrics(*arrays, YEARS <= 2015)
    half2 = period_metrics(*arrays, YEARS >= 2016)
    absolute_excess = hits - expected
    seasonal_excess = hits - expected * base
    absolute_drop = remove_three(*arrays, absolute_excess)
    seasonal_drop = remove_three(*arrays, seasonal_excess)
    supported = int((expected > 0).sum())
    m = {**pooled, 'years_supported': supported,
         'seasonal_years_supported': int(np.isfinite(seasonal).sum()),
         'years_lift_above_1': int((lifts > 1).sum()),
         'years_seasonal_above_1': int((seasonal > 1).sum()),
         'median_annual_lift': median(lifts), 'median_annual_seasonal': median(seasonal),
         'eligible': bool(supported == 20 and pooled['expected_hits'] >= 100),
         'explainable_taxonomy': bool(explainable),
         'annual_lifts_json': json.dumps({str(y): float(v) if np.isfinite(v) else None for y, v in zip(YEARS, lifts)}, separators=(',', ':')),
         'annual_seasonal_json': json.dumps({str(y): float(v) if np.isfinite(v) else None for y, v in zip(YEARS, seasonal)}, separators=(',', ':'))}
    for prefix, values in [('half1', half1), ('half2', half2), ('absolute_drop3', absolute_drop), ('seasonal_drop3', seasonal_drop)]:
        for k, v in values.items():
            m[prefix + '_' + k] = json.dumps(v, separators=(',', ':')) if isinstance(v, list) else v
    m.update(stability_flags(m))
    m['support_label'] = ('分类未知，不作稳定结论' if not explainable else
                          '支持不足：非20年完整支持或期望命中少于100' if not m['eligible'] else '支持满足描述门槛')
    annual = {'year': YEARS, 'actual_hits': hits, 'expected_hits': expected, 'lift': lifts,
              'industry_actual_hits': industry_hits, 'industry_expected_hits': industry_expected,
              'industry_lift': base, 'seasonal_ratio': seasonal,
              'absolute_excess': absolute_excess, 'seasonal_excess': seasonal_excess}
    return m, pd.DataFrame(annual)


def normalize_mapping(dictionary, mapping_rows):
    required = {'industry_key', 'classification_system', 'industry_code', 'chinese_name', 'mapping_status'}
    if not mapping_rows:
        raise ValueError('empty Chinese mapping')
    mapping = pd.DataFrame(mapping_rows).fillna('')
    if not required.issubset(mapping):
        raise ValueError('unrecognized Chinese mapping schema')
    if mapping.industry_key.duplicated().any():
        raise ValueError('duplicate Chinese mapping keys')
    lookup = mapping.set_index('industry_key').to_dict('index')
    rows = []
    for row in dictionary.to_dict('records'):
        key = row['industry_key']; supplied = lookup.get(key)
        if supplied is None:
            supplied = {'classification_system': row['classification_system'], 'industry_code': row['industry_code'],
                        'chinese_name': '中文名称未知', 'mapping_status': 'CANNOT_CONFIRM', 'source_url': ''}
        if supplied['classification_system'] != row['classification_system'] or supplied['industry_code'] != row['industry_code']:
            raise ValueError('Chinese mapping taxonomy mismatch ' + key)
        name = str(supplied['chinese_name']).strip()
        if not name or not any('\u4e00' <= c <= '\u9fff' for c in name):
            name = '中文名称未知'; supplied = {**supplied, 'mapping_status': 'CANNOT_CONFIRM'}
        explainable = (row['classification_system'] != 'UNKNOWN' and supplied['mapping_status'] not in ['CANNOT_CONFIRM', 'UNKNOWN', 'UNMAPPED'])
        rows.append({**row, **supplied, 'industry_key': key, 'chinese_name': name, 'explainable': explainable})
    return pd.DataFrame(rows)


def source_files():
    names = ['INDUSTRY_DICTIONARY.csv']
    for population in ['MAIN', 'EX_RESUMED']:
        for grid in GRIDS:
            names += [f'{population}_{grid.upper()}.csv', f'{population}_{grid.upper()}_YEAR.csv']
    return [FULL / name for name in names]


def freeze(mapping_path, reuse_derived=None):
    if (STUDY / 'CONTRACT.json').exists():
        raise FileExistsError('do not overwrite an existing freeze')
    manifest_path = FULL / 'OUTPUT_MANIFEST.json'
    if sha(manifest_path) != SOURCE_MANIFEST_SHA:
        raise RuntimeError('accepted source manifest drift')
    manifest = read(manifest_path)
    STUDY.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(mapping_path).resolve()
    inputs = {str(path): manifest['files'][str(path)] for path in source_files()}
    inputs.update({str(manifest_path): SOURCE_MANIFEST_SHA, str(mapping_path): sha(mapping_path)})
    if mapping_path.suffix.lower() == '.json':
        mapping_meta = read(mapping_path)
        if isinstance(mapping_meta, dict) and mapping_meta.get('source_mapping_path'):
            inputs[mapping_meta['source_mapping_path']] = mapping_meta['source_mapping_sha256']
    shared = read(PRIOR_BINDING)['shared_dependencies']
    binding = {'schema': 'HSJDAY_AGGREGATE_INTERPRETATION_INPUT_V1', 'input_paths': inputs,
               'mapping_path': str(mapping_path), 'source_full': str(FULL), 'shared_dependencies': shared,
               'baseline_record_path': str(STUDY / 'BASELINE.json'), 'registry_path': str(ROOT / 'contracts/EXPERIMENT_REGISTRY.jsonl'),
               'source_manifest_path': str(manifest_path), 'source_manifest_sha256': SOURCE_MANIFEST_SHA}
    if reuse_derived is not None:
        reuse_derived = Path(reuse_derived).resolve()
        prior_receipt = read(reuse_derived / 'RUN_RECEIPT.json')
        if prior_receipt['status'] != 'COMPLETED': raise RuntimeError('cannot reuse incomplete derivation')
        snapshot = reuse_derived.parent / 'NUMERICAL_SOURCE_SNAPSHOT.py'
        if sha(snapshot) != prior_receipt['code_sha256']: raise RuntimeError('numerical source snapshot drift')
        reused = [reuse_derived / n for n in ['OUTPUT_MANIFEST.json', 'RUN_RECEIPT.json', '主表与排除复牌比较.csv', 'SUMMARY.json']] + [snapshot]
        inputs.update({str(p): sha(p) for p in reused})
        binding['reuse_derived'] = {'path': str(reuse_derived), 'manifest_sha256': sha(reuse_derived / 'OUTPUT_MANIFEST.json'),
                                   'source_snapshot_path': str(snapshot), 'source_code_sha256': prior_receipt['code_sha256']}
    write(STUDY / 'INPUT_BINDING.json', binding)
    config = {'years': [2006, 2025], 'half_periods': [[2006, 2015], [2016, 2025]],
              'L': 'sum(cell hits)/sum(cell expected)', 'industry_baseline': 'sum(industry all-year hits)/sum(industry all-year expected)',
              'annual_S': 'annual cell L / annual industry L; undefined if industry L<=0 or expected<=0',
              'pooled_S': 'pooled cell L / pooled industry L; descriptive ratio affected by annual composition',
              'half_and_remaining_baselines': 'recompute ratios from sums using identical retained years; never average lifts',
              'absolute_remove3': 'H-E descending, year ascending ties; remove3; recompute remaining L and S',
              'seasonal_remove3': 'H-E*annual industry L descending, year ascending ties; remove3 finite years; undefined if fewer than3 finite',
              'positive_excess_share': 'sum positive excess of removed years / sum positive excess of all finite years, only if denominator>0',
              'eligible': '20 supported cell years and total expected>=100; UNKNOWN/unmapped excluded from conclusions',
              'stable_absolute': 'eligible and L>=1.2, >=15/20 annual L>1, median L>1, both half L>1, absolute remove3 L>1',
              'stable_seasonal_only': 'eligible and all20 S finite, S>=1.2, >=15/20 annual S>1, median S>1, both half S>1, seasonal remove3 S>1',
              'stable_seasonal': 'stable_absolute AND stable_seasonal_only',
              'robust': 'corresponding stable criteria pass separately in MAIN and EX_RESUMED',
              'few_year_inflated': 'eligible and corresponding pooled L/S>=1.2, (positive years<=10 OR median<=1), corresponding remove3 L/S<=1; S needs20 finite years',
              'comparison': 'strict >1 and >=1.2; no p-values or significance claims',
              'report_selection': 'top12 robust entries by minimum(MAIN S,EX S), then minimum L, then expected descending and key/month/third ascending; absolute-only month and month-third sections rank minimum L first; top20 common industries by MAIN expected; mandatory C40 November and I65 January/February with actual gate results; at most6 inflated/half-period-crossing examples; all rows retained',
              'stage_example': 'eligible both populations, not robust seasonal, at least one half-period L or S crosses1; descriptive stage contrast only',
              'reuse_derived': binding.get('reuse_derived'),
              'limits': 'current-industry retrospective; same-day observed-package Top100; no all-market/PIT/official complete-adjustment or future-return prediction claim', **SAFE}
    write(STUDY / 'CONFIG.json', config)
    baseline = {'source_manifest_sha256': SOURCE_MANIFEST_SHA, 'retained_prior_pass': str(FULL.parent / 'REVIEW_DATA_CHECK_01/INDEPENDENT_REVIEW_TRANSCRIPTION.json'),
                'git_baseline': '20610d06fc1767bdd8dd9c6c54d48776b4b874d8', 'branch': 'codex/a-hsjday-20y-20261019',
                'remote_sync': 'NOT_SYNCED_NO_REMOTE', 'requested_model_effort': 'gpt-6.1-sol/high', 'actual_runtime_identity': 'CANNOT_CONFIRM'}
    write(STUDY / 'BASELINE.json', baseline)
    contract = {'GOAL': '中文说明行业月份/上中下旬的历史Top100频率及20年稳健性',
                'HYPOTHESIS': 'pooled industry-calendar lift may reflect stable seasonality, sector baseline or a few years',
                'PREDICTION': 'descriptive only; no future-return claim', 'REJECTION_CONDITION': 'SHA drift, unexplained schema, invalid metric fixtures or independent review failure',
                'BASELINE_ID': 'ACCEPTED_HSJDAY_DESCRIPTIVE_04_AGGREGATES',
                'SIGNAL_DATE_RANGE': {'start': '2006-01-01', 'end': '2025-12-31'}, 'TARGET_DATE_RANGE': None,
                'INPUT_PATHS': list(inputs), 'INPUT_SHA256': inputs, 'CODE_SHA': sha(CODE), 'CONFIG_SHA': sha(STUDY / 'CONFIG.json'),
                'ONE_VARIABLE': 'aggregate-relative-to-annual-industry-baseline interpretation; no source reselection',
                'METRICS': config, 'PASS_THRESHOLD': 'fixtures and independent aggregate-output verification pass; no threshold retuning',
                'SEALED_DATA': 'none; exposed historical aggregate tables', 'WRITER': '/root/top100_stats_writer',
                'REVIEWER': '/root/stats_independent_review',
                'ALLOWED_PATHS': [str(CODE), str(TEST), str(TASK), str(ROOT / 'contracts/EXPERIMENT_REGISTRY.jsonl'),
                                  str(ROOT / 'contracts/A_SHARE_TDX_20Y_DATA_INDEX_20261019.json'), str(ROOT / 'contracts/A_SHARE_TOP100_ENVIRONMENT_INDEX_20261019.json'),
                                  str(ROOT / 'CURRENT_STATE.md'), str(ROOT / 'HK_OHLCV_RESEARCH_STATE.md'), str(ROOT / 'PROGRESS.md'), str(BATCH / 'INTERPRETATION_FINAL_HANDOFF.json')],
                'FORBIDDEN_PATHS': ['old scripts and FULL/study artifacts', 'raw prices/new data/network', 'training/backtest/V4/services/schedules/production/trading/push'],
                'STOP_RULE': 'on source SHA drift or unexplained schema stop numerical work; preserve failed attempt', **SAFE}
    write(STUDY / 'CONTRACT.json', contract)
    code_paths = [CODE, TEST, Path(util.__file__).resolve()] + [Path(x['path']) for x in shared.values()]
    write(STUDY / 'FROZEN_IDENTITY.json', {'code_files': {str(p): sha(p) for p in code_paths},
          'frozen_files': {str(STUDY / n): sha(STUDY / n) for n in ['INPUT_BINDING.json', 'CONFIG.json', 'BASELINE.json', 'CONTRACT.json']}})


def adapter_for(stage, out, claim_new=True):
    identity = read(STUDY / 'FROZEN_IDENTITY.json')
    for path, expected in {**identity['code_files'], **identity['frozen_files']}.items():
        if sha(path) != expected:
            raise RuntimeError('frozen identity drift ' + path)
    binding = read(STUDY / 'INPUT_BINDING.json')
    env = load_module(binding['shared_dependencies']['official_registry_helper']['path'], 'hsjday_interpretation_registry')
    env.EXPERIMENT_ID, env.ROUTE_ID, env.TRACK_ID = EXPERIMENT, ROUTE, TRACK
    a, core = env.load_official_registry(binding, contract_path=STUDY / 'CONTRACT.json', input_binding_path=STUDY / 'INPUT_BINDING.json', config_path=STUDY / 'CONFIG.json', code_path=CODE, test_path=TEST)
    a.event_profile.update(baseline_id='ACCEPTED_HSJDAY_DESCRIPTIVE_04_AGGREGATES', one_variable='FROZEN_AGGREGATE_INTERPRETATION', signal_scope='EXPOSED_HSJDAY_AGGREGATES_ONLY', evidence_level='DIRECTION_ONLY')
    ctx = {'contract_sha256': sha(STUDY / 'CONTRACT.json'), 'config_sha256': sha(STUDY / 'CONFIG.json'),
           'input_binding_sha256': sha(STUDY / 'INPUT_BINDING.json'), 'code_sha256': sha(CODE),
           'input_paths': list(binding['input_paths']), 'input_component_sha256': binding['input_paths'],
           'code_sha256_by_file': identity['code_files'], 'artifact_paths': [str(STUDY / n) for n in ['CONTRACT.json', 'CONFIG.json', 'INPUT_BINDING.json', 'FROZEN_IDENTITY.json', 'BASELINE.json']] + [str(CODE), str(TEST)],
           'signal_date_range': {'start': '2006-01-01', 'end': '2025-12-31'},
           'truth_access': 'NOT_READ' if stage == 'TEST' else 'OPENED_EXPOSED_AGGREGATES', 'sealed_status': 'EXPOSED',
           'runtime': util.runtime_metadata(), 'evidence_level': 'DIRECTION_ONLY',
           'deterministic_metrics': {'execution_kind': 'NO_FIT_ENGINEERING_CANARY', 'planned_real_fit_count': 0, 'real_fit_budget_debit': 0, 'real_fit_returned_count': 0, 'fit_call_entered': False, 'truth_positive_denominator': None, 'selection_slots': 0},
           'null_reasons': {'checkpoint_ref': 'NO_MODEL', 'target_date_range': 'NO_FORWARD_LABEL', 'deterministic_metrics.truth_positive_denominator': 'DESCRIPTIVE_NOT_RECALL'}}
    eid = EXPERIMENT + '_' + STUDY.name.upper() + '_' + stage + '_' + out.name
    if claim_new:
        claim = env._claim_stage(a, core, stage=stage, execution_id=eid, contract_sha256=ctx['contract_sha256'], input_binding_sha256=ctx['input_binding_sha256'], context=ctx)
    else:
        with a._locked():
            _, events = a._read_validated()
        starts = [e for e in events if e.get('execution_id') == eid and e.get('event_type') == 'RUN_STARTED']
        if len(starts) != 1 or any(e.get('execution_id') == eid and e.get('event_type') == 'TERMINAL' for e in events):
            raise RuntimeError('claim missing/already terminal')
        if starts[0]['contract_sha256'] != ctx['contract_sha256'] or starts[0]['input_binding_sha256'] != ctx['input_binding_sha256']:
            raise RuntimeError('claim identity drift')
        a.set_route_context(eid, ctx)
        claim = core.ExecutionClaim(experiment_id=EXPERIMENT, execution_id=eid, arm=stage, contract_sha256=ctx['contract_sha256'], input_binding_sha256=ctx['input_binding_sha256'], execution_kind='NO_FIT_ENGINEERING_CANARY', real_fit_budget_debit=0)
    return env, a, claim


def progress(a, claim, out, phase, metrics):
    path = out / 'PROGRESS.jsonl'
    util.append_jsonl(path, {'at': now(), 'phase': phase, 'metrics': metrics})
    ctx = dict(a._route_context[claim.execution_id]); ctx['artifact_paths'] = list(set(ctx['artifact_paths'] + [str(path)])); a.set_route_context(claim.execution_id, ctx)
    spec = a._event_spec({'experiment_id': EXPERIMENT, 'execution_id': claim.execution_id, 'arm': claim.arm, 'contract_sha256': claim.contract_sha256, 'input_binding_sha256': claim.input_binding_sha256, 'execution_kind': 'NO_FIT_ENGINEERING_CANARY', 'planned_real_fit_count': 0, 'real_fit_budget_debit': 0}, event_type='RUN_STARTED')
    with a._locked():
        raw, events = a._read_validated()
        n = sum(e.get('execution_id') == claim.execution_id and e.get('event_type') == 'PROGRESS' for e in events) + 1
        spec.update(registry_event_id=f'{claim.execution_id}_PROGRESS_{n:04}', event_type='PROGRESS', stage=phase, occurred_at=now(), decision='RUNNING', checkpoint_ref=str(path), deterministic_metrics={**spec['deterministic_metrics'], **metrics})
        spec.get('null_reasons', {}).pop('checkpoint_ref', None)
        a._append_unlocked(raw, events, spec)


def registry_snapshot(a):
    with a._locked():
        raw, events = a._read_validated()
    route = [e for e in events if e.get('route_id') == ROUTE]
    def counts(es):
        return {'routes': len({e.get('route_id') for e in es}), 'events': len(es), 'executions': len({e.get('execution_id') for e in es}), 'corrections': sum(e.get('event_type') == 'CORRECTION' for e in es)}
    starts = {e['execution_id'] for e in route if e.get('event_type') == 'RUN_STARTED'}
    terminal = {e['execution_id'] for e in route if e.get('event_type') == 'TERMINAL'}
    return {'registry_path': str(a.registry_path), 'registry_sha256': hashlib.sha256(raw).hexdigest(), 'head_event_sha256': events[-1]['event_sha256'], 'global': counts(events), 'route': counts(route), 'route_open_executions': sorted(starts - terminal), 'scoped_unregistered_route_count': 0, 'scoped_contract_only_count': 0, 'global_unregistered_route_count': None, 'global_contract_only_count': None, 'null_reason': 'GLOBAL_INVENTORY_NOT_AUDITED', 'external_or_git_persistence': 'CANNOT_CONFIRM'}


def verify_inputs():
    for path, expected in read(STUDY / 'INPUT_BINDING.json')['input_paths'].items():
        if sha(path) != expected:
            raise RuntimeError('source SHA drift ' + path)


def mapping_rows(path):
    if Path(path).suffix.lower() == '.csv':
        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records')
    payload = read(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get('rows'), list):
        return payload['rows']
    raise ValueError('unexplained mapping schema; expected row list or rows member')


def derive(out, a, claim):
    dictionary = pd.read_csv(FULL / 'INDUSTRY_DICTIONARY.csv', dtype=str, keep_default_na=False)
    names = normalize_mapping(dictionary, mapping_rows(read(STUDY / 'INPUT_BINDING.json')['mapping_path']))
    names.to_csv(out / '中文行业字典.csv', index=False)
    explainable = names.set_index('industry_key').explainable.to_dict()
    records, annual_records, parity = [], [], []
    for population in ['MAIN', 'EX_RESUMED']:
        baseline = pd.read_csv(FULL / f'{population}_INDUSTRY_YEAR.csv').set_index(['industry_key', 'year'])
        for grid, dimensions in GRIDS.items():
            annual = pd.read_csv(FULL / f'{population}_{grid.upper()}_YEAR.csv')
            summary = pd.read_csv(FULL / f'{population}_{grid.upper()}.csv')
            groupcols = ['industry_key'] + dimensions
            if annual.duplicated(groupcols + ['year']).any() or summary.duplicated(groupcols).any():
                raise ValueError('duplicate aggregate key')
            lookup = summary.set_index(groupcols)
            grid_rows = 0
            for key, values in annual.groupby(groupcols, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                industry_key = key[0]; values = values.sort_values('year')
                if not np.array_equal(values.year.to_numpy(), YEARS):
                    raise ValueError('twenty-year grid misalignment')
                base = baseline.loc[[(industry_key, int(y)) for y in YEARS]]
                m, ay = cell_metrics(values.actual_hits.to_numpy(), values.expected_hits.to_numpy(), base.actual_hits.to_numpy(), base.expected_hits.to_numpy(), explainable=explainable[industry_key])
                source = lookup.loc[key if dimensions else key[0]]
                if not math.isclose(m['actual_hits'], source.actual_hits, abs_tol=0.) or not math.isclose(m['expected_hits'], source.expected_hits, rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('source annual/pooled aggregate parity failure')
                ident = {'population': population, 'grid': grid, 'industry_key': industry_key, 'month': 0, 'third': 0, **dict(zip(dimensions, key[1:]))}
                m.update(valid_stock_days=int(values.valid_stock_days.sum()), support_dates=int(values.support_dates.sum()))
                records.append({**ident, **m})
                for name, value in ident.items():
                    ay[name] = value
                annual_records.append(ay); grid_rows += 1
            expected_rows = len(names) * {'industry': 1, 'month': 12, 'third': 3, 'month_third': 36}[grid]
            if grid_rows != expected_rows or len(annual) != expected_rows * 20:
                raise ValueError('incomplete input grid')
            parity.append({'population': population, 'grid': grid, 'cells': grid_rows, 'annual_rows': len(annual), 'source_parity': True})
            progress(a, claim, out, 'AGGREGATE_GRID_DERIVED', parity[-1])
    diagnostics = pd.DataFrame(records).merge(names, on='industry_key', validate='many_to_one')
    diagnostics.to_csv(out / '完整诊断.csv', index=False)
    pd.concat(annual_records, ignore_index=True).merge(names, on='industry_key', validate='many_to_one').to_csv(out / '20年逐年明细.csv', index=False)
    pure = pd.DataFrame(records)
    main = pure[pure.population == 'MAIN'].drop(columns='population')
    ex = pure[pure.population == 'EX_RESUMED'].drop(columns='population')
    main = main.rename(columns={c: 'main_' + c for c in main if c not in IDS})
    ex = ex.rename(columns={c: 'ex_' + c for c in ex if c not in IDS})
    comparison = main.merge(ex, on=IDS, validate='one_to_one').merge(names, on='industry_key', validate='many_to_one')
    comparison['robust_absolute'] = comparison.main_stable_absolute & comparison.ex_stable_absolute
    comparison['robust_seasonal_only'] = comparison.main_stable_seasonal_only & comparison.ex_stable_seasonal_only
    comparison['robust_seasonal'] = comparison.main_stable_seasonal & comparison.ex_stable_seasonal
    comparison['few_year_inflated_both_absolute'] = comparison.main_few_year_inflated_absolute & comparison.ex_few_year_inflated_absolute
    comparison['few_year_inflated_both_seasonal'] = comparison.main_few_year_inflated_seasonal & comparison.ex_few_year_inflated_seasonal
    comparison['min_L'] = comparison[['main_lift', 'ex_lift']].min(axis=1, skipna=False)
    comparison['min_S'] = comparison[['main_seasonal_ratio', 'ex_seasonal_ratio']].min(axis=1, skipna=False)
    comparison.to_csv(out / '主表与排除复牌比较.csv', index=False)
    write(out / 'SOURCE_PARITY.json', {'grids': parity, 'raw_price_reads': 0, 'old_execution_repeats': 0})
    metrics = {'industries': len(names), 'diagnostic_rows': len(diagnostics), 'comparison_rows': len(comparison),
               'annual_rows': sum(len(x) for x in annual_records), 'mapping_unknown': int((~names.explainable).sum()),
               'robust_absolute_cells': int(comparison.robust_absolute.sum()), 'robust_seasonal_cells': int(comparison.robust_seasonal.sum()),
               'robust_seasonal_only_cells': int(comparison.robust_seasonal_only.sum()),
               'few_year_inflated_both_absolute': int(comparison.few_year_inflated_both_absolute.sum()),
               'few_year_inflated_both_seasonal': int(comparison.few_year_inflated_both_seasonal.sum()),
               'grid_counts': {grid: {'robust_absolute': int(g.robust_absolute.sum()), 'robust_seasonal': int(g.robust_seasonal.sum()), 'robust_seasonal_only': int(g.robust_seasonal_only.sum())} for grid, g in comparison.groupby('grid')},
               'raw_price_reads': 0, 'old_execution_repeats': 0}
    write(out / 'SUMMARY.json', metrics)
    make_report(out, comparison, metrics)
    return metrics


def period_name(row):
    third = {1: '上旬', 2: '中旬', 3: '下旬'}
    if row['grid'] == 'industry': return '全年'
    if row['grid'] == 'month': return f"{int(row['month'])}月"
    if row['grid'] == 'third': return '所有月份合并·' + third[int(row['third'])]
    return f"{int(row['month'])}月" + third[int(row['third'])]


def fmt(value):
    return f'{value:.2f}' if present(value) else '无支持'


def ranked(frame, seasonal=True):
    return frame.sort_values((['min_S', 'min_L'] if seasonal else ['min_L', 'min_S']) + ['main_expected_hits', 'industry_key', 'month', 'third'], ascending=[False, False, False, True, True, True], na_position='last')


def display_table(frame):
    lines = ['|中文行业及分类|月份/旬|L 主/排复牌|S 主/排复牌|L>1年 主/排|S>1年 主/排|中位L/S（主）|两半期L（主）|两半期S（主）|去3年L/S（主）|期望命中（主）|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in frame.iterrows():
        lines.append(f"|{r.chinese_name}（{r.classification_system}:{r.industry_code}）|{period_name(r)}|{fmt(r.main_lift)}/{fmt(r.ex_lift)}|{fmt(r.main_seasonal_ratio)}/{fmt(r.ex_seasonal_ratio)}|{int(r.main_years_lift_above_1)}/{int(r.ex_years_lift_above_1)}|{int(r.main_years_seasonal_above_1)}/{int(r.ex_years_seasonal_above_1)}|{fmt(r.main_median_annual_lift)}/{fmt(r.main_median_annual_seasonal)}|{fmt(r.main_half1_lift)}/{fmt(r.main_half2_lift)}|{fmt(r.main_half1_seasonal_ratio)}/{fmt(r.main_half2_seasonal_ratio)}|{fmt(r.main_absolute_drop3_lift)}/{fmt(r.main_seasonal_drop3_seasonal_ratio)}|{r.main_expected_hits:.1f}|")
    return lines


def failed_conditions(row, prefix):
    get = lambda name: row[prefix + '_' + name]
    failures = []
    if not get('eligible'): failures.append('支持不足（需20年且期望≥100）')
    if not get('explainable_taxonomy'): failures.append('分类未知')
    if not ge(get('lift'), 1.2): failures.append('L未达1.20')
    if get('years_lift_above_1') < 15: failures.append('L>1少于15年')
    if not gt(get('median_annual_lift')): failures.append('中位L未大于1')
    if not gt(get('half1_lift')) or not gt(get('half2_lift')): failures.append('两个十年L未同时>1')
    if not gt(get('absolute_drop3_lift')): failures.append('绝对去3年后L未>1')
    if get('seasonal_years_supported') != 20: failures.append('S非20年完整支持')
    if not ge(get('seasonal_ratio'), 1.2): failures.append('S未达1.20')
    if get('years_seasonal_above_1') < 15: failures.append('S>1少于15年')
    if not gt(get('median_annual_seasonal')): failures.append('中位S未大于1')
    if not gt(get('half1_seasonal_ratio')) or not gt(get('half2_seasonal_ratio')): failures.append('两个十年S未同时>1')
    if not gt(get('seasonal_drop3_seasonal_ratio')): failures.append('季节去3年后S未>1')
    return '；'.join(failures) if failures else '全部通过'


def make_report(out, c, metrics):
    lines = ['# 哪些行业、月份和上中下旬更容易进入同日涨幅前100', '',
             '本报告只解释已验收的2006—2025年聚合表：观察包内4,859个有效日、每日100只。当前行业回溯归类；未知分类单列。它不能证明历史全市场、历史PIT或官方完整复权，也不是未来收益预测。', '',
             'L=实际入选次数/按行业每日有效股票占比计算的期望次数。L=1.30表示历史入选频率为该市场基准的1.30倍，不是上涨概率或未来收益。S=该月/旬的L÷本行业全年L；S=1.20表示相对本行业自身全年基准的描述比值高20%。行业全年本来就强，不等于某个月有额外季节性。合并S仍受各年规模和构成变化影响，因此同时检查逐年S。', '',
             '年度L和S均逐年计算；2006—2015、2016—2025两半期与去3年后的行业基准，都以保留年份的实际/期望总和重算，不平均年度lift。绝对诊断按年度“实际−期望”删除最大的3年；季节诊断按“实际−期望×该年行业全年L”删除最大的3年。占比只在正超额合计大于0时计算。', '',
             '稳健门槛事前冻结：20年有支持、总期望≥100；L≥1.20，至少15/20年L>1、中位L>1、两半期L>1、绝对去3年L>1。稳定季节增量还要S≥1.20，至少15/20年S>1、中位S>1、两半期S>1、季节去3年S>1。主表与排除复牌后独立排名的伴随表都过门，才称稳健。门槛是描述筛选，不是统计显著性。', '',
             '表中“主/排”依次为MAIN与EX_RESUMED；上旬1—10日，中旬11—20日，下旬21日至月末。未达门槛归为中间/不稳定，不自动称为少数年份拉高。UNKNOWN及零支持分类不作稳定结论。', '']
    selections = []
    sections = [('行业总体的稳健绝对优势', 'industry', 'robust_absolute', False),
                ('稳健月份：同时超越市场基准与本行业全年基准', 'month', 'robust_seasonal', True),
                ('稳健月旬：具体月份的上中下旬', 'month_third', 'robust_seasonal', True),
                ('上中下旬整体：合并所有月份', 'third', 'robust_seasonal', True),
                ('月份的稳健市场优势：可能主要来自行业长期强势', 'month', 'robust_absolute', False),
                ('月旬的稳健市场优势：不等同于额外季节增量', 'month_third', 'robust_absolute', False)]
    for title, grid, flag, seasonal in sections:
        all_rows = ranked(c[(c.grid == grid) & c[flag]], seasonal)
        shown = all_rows.head(12)
        lines += ['## ' + title, '', f'符合冻结条件共{len(all_rows)}项，展示前{len(shown)}项；不足8项时不凑数。', '']
        lines += display_table(shown) if len(shown) else ['没有符合项，保持原门槛。']
        lines += ['']
        selections += [{**r, 'report_group': title} for r in shown.to_dict('records')]
    relative = ranked(c[(c.grid != 'industry') & c.robust_seasonal_only & ~c.robust_seasonal])
    lines += ['## 相对本行业更强，但未达到稳定市场优势', '', f'仅S门通过、但L门未同时通过的格子共{len(relative)}项。它们有相对季节性描述价值，不能称作稳定市场优势。', '']
    lines += display_table(relative.head(12)) if len(relative) else ['没有符合项。']
    lines += ['']
    named = c[(c.classification_system == 'CAPCO_CURRENT_RETROSPECTIVE') & (c.grid == 'month') &
              (((c.industry_code == 'C40') & (c.month == 11)) | ((c.industry_code == 'I65') & c.month.isin([1, 2])))].sort_values(['industry_code', 'month'])
    lines += ['## 关注行业逐项：仪器仪表11月、软件和信息技术服务1—2月', '',
              '以下按指定月份展示真实数值，无论是否达到严格稳健门都保留；门未通过不等于没有历史表现，也不能把描述优势升级为未来可预测性。', '']
    lines += display_table(named)
    for _, r in named.iterrows():
        lines += ['', f"- {r.chinese_name}·{period_name(r)}：稳健市场优势={'通过' if r.robust_absolute else '未过'}；稳健季节增量={'通过' if r.robust_seasonal else '未过'}。主表未过条件：{failed_conditions(r, 'main')}。排复牌未过条件：{failed_conditions(r, 'ex')}。"]
    lines += ['']
    inflated = c[(c.grid != 'industry') & (c.main_few_year_inflated_absolute | c.main_few_year_inflated_seasonal | c.ex_few_year_inflated_absolute | c.ex_few_year_inflated_seasonal)].copy()
    lines += ['## 是否由少数年份拉高', '',
              '严格定义：合并L或S≥1.20，同时对应优势年份≤10或年度中位数≤1，并且按对应超额删去3年后≤1；需满足支持门槛。以下显示删掉的年份与剩余结果，不能把所有未达稳定门槛者都放进这一类。', '', f'任一人口符合严格定义的格子共{len(inflated)}项。', '']
    selected = ranked(inflated).head(6)
    if not len(selected):
        lines += ['没有符合“少数年份拉高”严格定义的格子，不放宽条件。', '']
    for _, r in selected.iterrows():
        triggers = '、'.join(label for flag, label in [('main_few_year_inflated_absolute', '主表绝对'), ('main_few_year_inflated_seasonal', '主表季节'), ('ex_few_year_inflated_absolute', '排复牌绝对'), ('ex_few_year_inflated_seasonal', '排复牌季节')] if r[flag])
        lines += [f"- {r.chinese_name}·{period_name(r)}（符合定义：{triggers}）：L主/排={fmt(r.main_lift)}/{fmt(r.ex_lift)}，S主/排={fmt(r.main_seasonal_ratio)}/{fmt(r.ex_seasonal_ratio)}；主表L/S优势年={int(r.main_years_lift_above_1)}/{int(r.main_years_seasonal_above_1)}，中位L/S={fmt(r.main_median_annual_lift)}/{fmt(r.main_median_annual_seasonal)}。绝对删年{r.main_absolute_drop3_removed_years}后L={fmt(r.main_absolute_drop3_lift)}；季节删年{r.main_seasonal_drop3_removed_years}后S={fmt(r.main_seasonal_drop3_seasonal_ratio)}。对应前三年正超额占比={fmt(r.main_absolute_drop3_top3_positive_excess_share)}/{fmt(r.main_seasonal_drop3_top3_positive_excess_share)}（无正超额时为无支持）。排复牌后的去3年L/S={fmt(r.ex_absolute_drop3_lift)}/{fmt(r.ex_seasonal_drop3_seasonal_ratio)}。"]
    eligible = c.main_eligible & c.ex_eligible & c.explainable
    crossing = (((c.main_half1_lift > 1) & (c.main_half2_lift <= 1)) | ((c.main_half2_lift > 1) & (c.main_half1_lift <= 1)) |
                ((c.main_half1_seasonal_ratio > 1) & (c.main_half2_seasonal_ratio <= 1)) | ((c.main_half2_seasonal_ratio > 1) & (c.main_half1_seasonal_ratio <= 1)))
    stages = c[(c.grid != 'industry') & eligible & ~c.robust_seasonal & crossing].copy()
    stages['half_gap'] = (stages.main_half2_seasonal_ratio - stages.main_half1_seasonal_ratio).abs()
    stages = stages.sort_values(['half_gap', 'industry_key', 'month', 'third'], ascending=[False, True, True, True]).head(6)
    lines += ['', '## 阶段差异典型：不能等同于20年持续有效', '', '这些格子的两个十年L或S跨过1；仅描述阶段差异，不臆测行业轮动原因。', '']
    lines += display_table(stages) if len(stages) else ['没有符合预设跨期对比条件的例子。']
    lines += ['', '## 常见行业全景：即使未过门槛也完整展示', '',
              '按全年主表期望命中规模取前20个已知行业。每个行业展示相对季节比最强的月份和月旬，明确是否通过稳健门；“最高”只是本表描述排序。', '',
              '|中文行业|全年L主/排|全年L>1年主/排|相对较强月份 L/S（主）|相对较强月旬 L/S（主）|月/月旬稳健季节门|', '|---|---:|---:|---|---|---|']
    common = c[(c.grid == 'industry') & c.explainable].sort_values(['main_expected_hits', 'industry_key'], ascending=[False, True]).head(20)
    for _, r in common.iterrows():
        pool = c[(c.industry_key == r.industry_key) & eligible]
        months = ranked(pool[pool.grid == 'month']).head(1)
        thirds = ranked(pool[pool.grid == 'month_third']).head(1)
        def best(g):
            if not len(g): return '支持不足', '未过'
            row = g.iloc[0]
            return f'{period_name(row)} {fmt(row.main_lift)}/{fmt(row.main_seasonal_ratio)}', '通过' if row.robust_seasonal else '未过'
        m, mp = best(months); t, tp = best(thirds)
        lines.append(f'|{r.chinese_name}|{fmt(r.main_lift)}/{fmt(r.ex_lift)}|{int(r.main_years_lift_above_1)}/{int(r.ex_years_lift_above_1)}|{m}|{t}|{mp}/{tp}|')
    lines += ['', '## 全部行业与分类边界', '', '|中文名称|原taxonomy及代码|支持年（主）|全年L主/排|稳健绝对优势|说明|', '|---|---|---:|---:|---|---|']
    for _, r in c[c.grid == 'industry'].sort_values('industry_key').iterrows():
        lines.append(f'|{r.chinese_name}|{r.classification_system}:{r.industry_code}|{int(r.main_years_supported)}|{fmt(r.main_lift)}/{fmt(r.ex_lift)}|{"是" if r.robust_absolute else "否"}|{r.main_support_label}|')
    lines += ['', '## 文件与使用边界', '',
              '完整诊断.csv 保留两个人口全部格子、年度L/S数组、两半期、删年列表与超额贡献占比；20年逐年明细.csv 保留年度分子分母、行业基准和超额；主表与排除复牌比较.csv 保留完整交叉判定。中文行业字典.csv 保留原taxonomy、代码、中文展示名和来源，不将不同分类体系合并。', '',
              '中文名称仅作展示映射；原行业归属和命中/期望均不改变。当前行业回溯归类存在存续与分类偏差。无显著性、因果、突破或未来收益结论；未接V4、未训练、未回测、未生产或交易。', '',
              'research_only=true；production_allowed=false；buy_gate_updated=false；orders_sent=false。独立验收另行登记。']
    (out / '中文季节性与稳健性报告.md').write_text('\n'.join(lines) + '\n')
    pd.DataFrame(selections).to_csv(out / '报告展示条目.csv', index=False)


def execute(stage, attempt):
    out = STUDY / f'{stage}_{attempt}'; out.mkdir(parents=True, exist_ok=False)
    _, a, claim = adapter_for(stage, out)
    started = now(); metrics = {}; error = None
    try:
        verify_inputs()
        if stage == 'TEST':
            result = subprocess.run([sys.executable, '-m', 'unittest', 'research.a_share_top100_environment_v1.tests.test_hsjday_interpretation', '-v'], cwd=WT, capture_output=True, text=True)
            (out / 'TEST_OUTPUT.txt').write_text(result.stdout + result.stderr)
            metrics = {'returncode': result.returncode}
            if result.returncode: raise RuntimeError('focused fixture failures')
        elif stage == 'DERIVE':
            test = read(STUDY / 'TEST_01/RUN_RECEIPT.json')
            if test['status'] != 'COMPLETED' or test['code_sha256'] != sha(CODE): raise RuntimeError('matching TEST_01 required')
            metrics = derive(out, a, claim)
        elif stage == 'REPORT_ONLY':
            reuse = read(STUDY / 'INPUT_BINDING.json')['reuse_derived']
            old = Path(reuse['path'])
            original = ast.parse(Path(reuse['source_snapshot_path']).read_text())
            current = ast.parse(CODE.read_text())
            core_names = ['ratio', 'present', 'gt', 'ge', 'median', 'period_metrics', 'remove_three', 'stability_flags', 'cell_metrics', 'normalize_mapping', 'derive']
            def identities(tree):
                return {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in core_names}
            if identities(original) != identities(current): raise RuntimeError('report-only patch changed numerical mechanism')
            c = pd.read_csv(old / '主表与排除复牌比较.csv')
            metrics = read(old / 'SUMMARY.json')
            make_report(out, c, metrics)
            write(out / 'NUMERICAL_RESULT_LINEAGE.json', {'source_derived_manifest_path': str(old / 'OUTPUT_MANIFEST.json'), 'source_derived_manifest_sha256': reuse['manifest_sha256'], 'numerical_source_code_sha256': reuse['source_code_sha256'], 'numerical_function_sha256': identities(current), 'statistical_recomputation_count': 0, 'numerical_csv_bytes_unchanged': True})
            metrics = {**metrics, 'report_only': True, 'statistical_recomputation_count': 0}
        else: raise ValueError(stage)
    except BaseException as exc:
        error = exc; (out / 'TRACEBACK.txt').write_text(traceback.format_exc())
    receipt = {'status': 'COMPLETED' if error is None else 'CANNOT_CONFIRM', 'stage': stage,
               'execution_id': claim.execution_id, 'started': started, 'finished': now(), 'metrics': metrics,
               'failure': None if error is None else str(error), 'code_sha256': sha(CODE), 'contract_sha256': sha(STUDY / 'CONTRACT.json'),
               'input_binding_sha256': sha(STUDY / 'INPUT_BINDING.json'), 'config_sha256': sha(STUDY / 'CONFIG.json'), **SAFE}
    write(out / 'RUN_RECEIPT.json', receipt)
    ctx = dict(a._route_context[claim.execution_id]); ctx['artifact_paths'] += [str(p) for p in out.iterdir() if p.is_file()]
    ctx['deterministic_metrics'] = {**ctx['deterministic_metrics'], **metrics}; a.set_route_context(claim.execution_id, ctx)
    a.terminalize(claim, {'experiment_id': EXPERIMENT, 'execution_id': claim.execution_id, 'arm': stage, 'contract_sha256': claim.contract_sha256, 'input_binding_sha256': claim.input_binding_sha256, 'execution_kind': 'NO_FIT_ENGINEERING_CANARY', 'terminal_status': receipt['status'], 'real_fit_returned_count': 0, 'fit_call_entered': False, 'finished_at': now(), 'error_type': str(error) if error else None})
    write(out / 'REGISTRY_SNAPSHOT.json', registry_snapshot(a))
    write(out / 'OUTPUT_MANIFEST.json', {'status': receipt['status'], 'files': {str(p): sha(p) for p in sorted(out.iterdir()) if p.is_file()}, **SAFE})
    print(json.dumps({'out': str(out), **receipt}, ensure_ascii=False), flush=True)
    return 0 if error is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'TEST', 'DERIVE', 'REPORT_ONLY'])
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--reuse-derived', type=Path)
    parser.add_argument('--attempt', default='01')
    args = parser.parse_args(); STUDY = args.study.resolve()
    if not STUDY.is_relative_to(TASK.resolve()): parser.error('study must remain inside interpretation_20261020_01')
    if args.stage == 'freeze':
        if args.mapping is None: parser.error('freeze requires --mapping')
        freeze(args.mapping, args.reuse_derived)
    else:
        raise SystemExit(execute(args.stage, args.attempt))
