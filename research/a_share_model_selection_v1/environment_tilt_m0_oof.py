"""Isolated M0/h2/2023H2 wrapper over the unchanged original recipe.

No import-time IO/model/registry calls. Only the explicit CLI can admit real
inputs, after a phase-specific, identity-bound independent activation. Synthetic
tests inject recipe/guard objects; no alternate model recipe is implemented.
Preflight is a narrow new-time-use proof, NOT first PIT/source certification.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time


BATCH = Path('/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/environment_tilt_execution_20261020')
FLAGS = dict(research_only=True, production_allowed=False, buy_gate_updated=False, orders_sent=False)
CELL = dict(arm='M0', horizon=2, fold='2023H2', train_start='2021-07-01',
            score_start='2023-07-01', score_end='2023-12-31', partial=False)
EXPERIMENT = 'A_SHARE_ENV_TILT_OOF_2023H2_20261021'
ROUTE = 'a-share-environment-tilt-oof-2023h2-20261021'
ORIGINAL_ZERO_POOL_DATES = ('2021-08-02', '2021-08-03', '2021-08-04', '2022-04-27', '2022-04-28')
FEATURES = ('DISTW','DISTB','C','YEL0','LCOUNT','HIST0','AMTR','VOLR','VMA20R','VSPHHV',
            'FCOUNT','SFCOUNT','RET20P','GAIN60','RANGE60','RANGE120','JVAL','BKRSI','CPOS',
            'UPSH','WASHSCR','B1PRE','BKEXIST','FRESH','CONT0','B1SIG','B2SIG','B3SIG',
            'BREAKOUT','BKRES','RTETYPE','RISK0','HEALTH','LOWEXT','KQUAL','LIQSTD')
PARAMS = dict(loss='squared_error', max_iter=150, learning_rate=.05, max_depth=3,
              max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=10.,
              early_stopping=False, random_state=193)
POLICY = dict(cell=CELL, features=list(FEATURES), params=PARAMS, threads=4,
              weighting='original_equal_day_weights_normalized_mean1',
              pool='EXACT_EXISTING_FIVE_ARM_COMMON_FINITE_CACHE',
              input_version='S1_PREPARATION_EXISTING_CACHE_NO_PREFIX',
              label='original_close_Tplus2_over_close_T_minus1',
              truth_scope='TRAINING_SIGNALS_ONLY_LABEL_END_BEFORE_2023_07_01',
              max_real_fit_debits=1, **FLAGS)


class Blocked(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


POLICY_SHA256 = digest(POLICY)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stat_identity(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def write_json(path, value):
    """Caller holds the output-owner lock; durable owned-file replacement only."""
    path = Path(path)
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('x') as handle:
        handle.write(canonical(value) + '\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_json(ref):
    if sha(ref['path']) != ref['sha256']:
        raise Blocked('METADATA_SHA_DRIFT: ' + ref['path'])
    return json.loads(Path(ref['path']).read_text())


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_recipe(config):
    """Hash only declared original source components; never mutate its globals."""
    for ref in config['recipe_sources'].values():
        if sha(ref['path']) != ref['sha256']:
            raise Blocked('ORIGINAL_RECIPE_SOURCE_DRIFT')
    ref = config['recipe_sources']['model_selection']
    recipe = _load(ref['path'], 'm13_original_recipe')
    if tuple(recipe.CORE_FEATURES) != FEATURES or recipe.MODEL_PARAMS != PARAMS:
        raise Blocked('ORIGINAL_RECIPE_OR_FEATURE_ORDER_CONFLICT')
    return recipe


def validate_config(config):
    if config.get('policy') != POLICY or config.get('policy_sha256') != POLICY_SHA256:
        raise Blocked('EXACT_SINGLE_CELL_POLICY_REQUIRED')
    evidence = config.get('original_empty_pool_proof')
    if (not isinstance(evidence, dict) or set(evidence) != {'ruling', 'diagnostic', 's1_runner'}
            or any(not isinstance(ref, dict) or not ref.get('path') or not ref.get('sha256')
                   for ref in evidence.values())):
        raise Blocked('BOUND_ORIGINAL_EMPTY_POOL_PROOF_REQUIRED')
    probe = config.get('resource_probe_source')
    if (not isinstance(probe, dict) or not probe.get('path') or not probe.get('sha256')
            or Path(probe['path']).resolve() != Path(__file__).with_name('environment_tilt_mac_recovery.py').resolve()):
        raise Blocked('EXISTING_MAC_RESOURCE_PROBE_BINDING_REQUIRED')
    if any(config.get(k) != v for k, v in FLAGS.items()):
        raise Blocked('RESEARCH_BOUNDARY_REQUIRED')
    output = Path(config['output_root'])
    if (not output.is_absolute() or output.parent.resolve() != BATCH.resolve()
            or not output.name.startswith('M13_') or output.is_symlink()):
        raise Blocked('OUTPUT_MUST_BE_NEW_M13_BATCH_CHILD')
    inputs = [Path(ref['path']).resolve() for ref in config['cache_files'].values()]
    if set(config['cache_files']) != {'m0', 'calendar', 'panel'} or any(output.resolve() in p.parents for p in inputs):
        raise Blocked('ONLY_BOUND_EXISTING_M0_CALENDAR_PANEL_CACHES')
    if config['execution_ids'].keys() != {'preflight', 'fit'} or len(set(config['execution_ids'].values())) != 2:
        raise Blocked('TWO_DISTINCT_PHASE_EXECUTION_IDS_REQUIRED')
    if config.get('state') != 'FROZEN_READY_FOR_ACTIVATION':
        raise Blocked('DRAFT_IS_NOT_EXECUTABLE')


def require_activation(config, config_path, activation, phase):
    validate_config(config)
    wanted = dict(phase=phase, decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',
                  config_sha256=sha(config_path), policy_sha256=POLICY_SHA256,
                  wrapper_sha256=sha(__file__), contract_sha256=config['contract']['sha256'],
                  execution_id=config['execution_ids'][phase])
    if any(activation.get(k) != v for k, v in wanted.items()):
        raise Blocked('PHASE_REVIEW_ACTIVATION_IDENTITY_MISMATCH')
    if not activation.get('reviewer') or not activation.get('review_receipt_sha256'):
        raise Blocked('INDEPENDENT_REVIEW_RECEIPT_REQUIRED')
    checked_json(config['contract'])
    if phase == 'fit':
        pre = config.get('accepted_preflight')
        if not pre or activation.get('preflight_sha256') != pre['sha256']:
            raise Blocked('REVIEWED_PREFLIGHT_BINDING_REQUIRED')
        receipt = checked_json(pre)
        if (receipt.get('status') != 'PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW'
                or receipt.get('policy_sha256') != POLICY_SHA256
                or receipt.get('contract_sha256') != config['contract']['sha256']
                or receipt.get('wrapper_sha256') != sha(__file__)
                or receipt.get('input_binding_identity') != input_binding_identity(config)):
            raise Blocked('PREFLIGHT_SCOPE_OR_INPUT_DRIFT')


def input_binding_identity(config):
    # Mutable phase activation/accepted-receipt references are not scientific inputs.
    return digest({k: config[k] for k in ('policy', 'recipe_sources', 'cache_files',
                                        'preparation_receipt', 'original_binding', 'new_use_passport',
                                        'runtime_versions', 'resource_probe_source', 'original_empty_pool_proof')})


def load_original_empty_pool_proof(config):
    """Authenticate only the existing M22 metadata/source ruling, not raw data.

    Cache bytes are admitted separately by read_bound_caches (or accepted stat
    FAST_BIND). This evidence applies solely to this existing 2023H2 cell.
    """
    refs = config['original_empty_pool_proof']
    ruling, diagnostic = checked_json(refs['ruling']), checked_json(refs['diagnostic'])
    dates = list(ORIGINAL_ZERO_POOL_DATES)
    wanted = dict(schema='ENV_TILT_M22_ORIGINAL_EMPTY_COMMON_POOL_RULING_V1',
        verdict='ORIGINAL_FIVE_ARM_COMMON_POOL_ZERO_KEY_DATES_PROVEN',
        train_start=CELL['train_start'], train_end_exclusive=CELL['score_start'],
        nominal_train_calendar_days=485, nonempty_train_pool_days=480,
        score_calendar_days=124, nonempty_score_pool_days=124, original_zero_pool_dates=dates,
        preparation_sha256=config['preparation_receipt']['sha256'],
        m0_sha256=config['cache_files']['m0']['sha256'],
        diagnostic_receipt_path=refs['diagnostic']['path'],
        diagnostic_receipt_sha256=refs['diagnostic']['sha256'],
        producer_source_sha256=dict(model_selection=config['recipe_sources']['model_selection']['sha256'],
                                    s1_runner=refs['s1_runner']['sha256']))
    if any(ruling.get(k) != v for k, v in wanted.items()):
        raise Blocked('ORIGINAL_EMPTY_POOL_RULING_BINDING_CONFLICT')
    if sha(refs['s1_runner']['path']) != refs['s1_runner']['sha256']:
        raise Blocked('ORIGINAL_EMPTY_POOL_PRODUCER_SOURCE_DRIFT')
    if (diagnostic.get('status') != 'DATE_DIAGNOSTIC_COMPLETE_NO_REPAIR'
            or diagnostic.get('fit_calls') != 0):
        raise Blocked('ORIGINAL_EMPTY_POOL_DIAGNOSTIC_NOT_ACCEPTED')
    for phase, start, end, inclusive, nominal, nonempty, missing in (
            ('train', CELL['train_start'], CELL['score_start'], 'left', 485, 480, dates),
            ('score', CELL['score_start'], CELL['score_end'], 'both', 124, 124, [])):
        record = diagnostic.get('phases', {}).get(phase, {})
        expected = dict(start=start, end=end, inclusive=inclusive, missing_dates=missing,
                        missing_dates_empty_in_whole_m0_cache=missing, extra_dates=[])
        if (any(record.get(k) != v for k, v in expected.items())
                or record.get('expected', {}).get('count') != nominal
                or record.get('actual', {}).get('unique_date_count') != nonempty):
            raise Blocked('ORIGINAL_EMPTY_POOL_DIAGNOSTIC_COVERAGE_CONFLICT')
    authenticated = diagnostic.get('authenticated_inputs', [])
    for name in ('m0', 'calendar'):
        matches = [row for row in authenticated if row.get('name') == name]
        ref = config['cache_files'][name]
        if (len(matches) != 1 or matches[0].get('path') != ref['path']
                or matches[0].get('actual_sha256') != ref['sha256']
                or matches[0].get('expected_sha256') != ref['sha256']):
            raise Blocked('ORIGINAL_EMPTY_POOL_DIAGNOSTIC_CACHE_BINDING_CONFLICT')
    return dict(original_zero_pool_dates=dates, nominal_train_calendar_days=485,
                nonempty_train_pool_days=480, score_calendar_days=124,
                nonempty_score_pool_days=124, evidence_refs=refs)


def _frame_keys(frame):
    return frame[['date', 'code']].sort_values(['date', 'code'], kind='mergesort').reset_index(drop=True)


def prepare_single_cell(recipe, features, panel, calendar, *, original_empty_pool_proof=None):
    """Original split/matrices/labels; fail before finite filtering can change pool."""
    import numpy as np
    import pandas as pd
    if tuple(features.columns) != ('date', 'code', *FEATURES):
        raise Blocked('EXACT_36_FEATURE_SCHEMA_REQUIRED')
    for name, frame in [('features', features), ('panel', panel)]:
        if frame[['date', 'code']].isna().any().any() or frame.duplicated(['date', 'code']).any():
            raise Blocked('AMBIGUOUS_' + name.upper() + '_KEYS')
        dates = pd.to_datetime(frame.date)
        if not dates.equals(dates.dt.normalize()) or dates.dt.tz is not None:
            raise Blocked('NORMALIZED_NAIVE_DATE_KEYS_REQUIRED')
        if not frame.code.map(lambda c: isinstance(c, str) and bool(c) and c.strip() == c).all():
            raise Blocked('EXACT_CANONICAL_CODE_KEYS_REQUIRED')
    if not np.isfinite(features[list(FEATURES)].to_numpy(dtype=float)).all():
        raise Blocked('NONFINITE_EXISTING_COMMON_POOL_NO_REFILTER')
    cal = calendar[['date', 'calendar_index']].copy()
    cal['date'] = pd.to_datetime(cal.date)
    if (cal.date.isna().any() or cal.date.duplicated().any()
            or not pd.api.types.is_integer_dtype(cal.calendar_index.dtype)
            or pd.api.types.is_bool_dtype(cal.calendar_index.dtype)
            or not np.array_equal(cal.calendar_index.to_numpy(), np.arange(len(cal)))
            or not cal.date.is_monotonic_increasing or not cal.date.equals(cal.date.dt.normalize())):
        raise Blocked('ORIGINAL_COMPLETE_OBSERVED_CALENDAR_REQUIRED')
    if not set(features.date).issubset(set(cal.date)) or not set(panel.date).issubset(set(cal.date)):
        raise Blocked('CACHE_DATES_OUTSIDE_ORIGINAL_CALENDAR')
    train_start, score_start, score_end = (pd.Timestamp(CELL[k]) for k in ('train_start', 'score_start', 'score_end'))
    if cal.date.min() > train_start or cal.date.max() < score_end:
        # The last score boundary may be a non-session: require calendar past it.
        raise Blocked('NOMINAL_TWO_YEAR_OR_SCORE_CALENDAR_SPAN_INCOMPLETE')
    expected_train = tuple(cal.loc[cal.date.between(train_start, score_start, inclusive='left'), 'date'])
    expected_test = tuple(cal.loc[cal.date.between(score_start, score_end), 'date'])
    feature_train = features.loc[features.date.between(train_start, score_start, inclusive='left')]
    feature_test = features.loc[features.date.between(score_start, score_end)]
    actual_train = tuple(sorted(feature_train.date.unique()))
    actual_test = tuple(sorted(feature_test.date.unique()))
    missing_train = tuple(t.date().isoformat() for t in expected_train if t not in set(actual_train))
    if (not expected_train or not expected_test or actual_test != expected_test
            or not set(actual_train).issubset(expected_train)):
        raise Blocked('MISSING_OBSERVED_SCORE_OR_TRAIN_FEATURE_DATES_NO_SHORT_WINDOW')
    if original_empty_pool_proof is None:
        if missing_train:
            raise Blocked('MISSING_OBSERVED_SCORE_OR_TRAIN_FEATURE_DATES_NO_SHORT_WINDOW')
    elif (original_empty_pool_proof.get('original_zero_pool_dates') != list(ORIGINAL_ZERO_POOL_DATES)
            or missing_train != ORIGINAL_ZERO_POOL_DATES
            or (len(expected_train), len(actual_train), len(expected_test), len(actual_test)) != (485, 480, 124, 124)
            or any(original_empty_pool_proof.get(k) != v for k, v in dict(
                nominal_train_calendar_days=485, nonempty_train_pool_days=480,
                score_calendar_days=124, nonempty_score_pool_days=124).items())):
        raise Blocked('BOUND_ORIGINAL_EMPTY_POOL_PARTITION_MISMATCH')
    fold = recipe.Fold(CELL['fold'], score_start, score_end, False)
    labels = recipe.build_close_return_labels(panel, horizons=(2,), calendar=cal,
        signal_start=train_start, signal_end_exclusive=score_start)[2]
    if not labels.date.between(train_start, score_start, inclusive='left').all():
        raise Blocked('TEST_TRUTH_IN_TRAIN_LABEL_PREPARATION')
    prepared = recipe.prepare_cell_data(features=features, labels=labels, fold=fold, canary=False)
    if tuple(prepared['feature_columns']) != FEATURES or not _frame_keys(prepared['test']).equals(_frame_keys(feature_test)):
        raise Blocked('ORIGINAL_POOL_OR_FEATURE_ORDER_CHANGED')
    train, test = prepared['train'], prepared['test']
    if (train.empty or test.empty or train.label_end_date.isna().any()
            or not (pd.to_datetime(train.label_end_date) < score_start).all()
            or not pd.Timestamp(train.label_end_date.max()) < test.date.min()):
        raise Blocked('SAME_MODEL_OOT_OR_PURGE_VIOLATION')
    if {'label_return', 'target_date', 'label_end_date'} & set(test.columns):
        raise Blocked('UNLABELED_TEST_FEATURES_REQUIRED')
    proof = dict(cell=CELL, feature_columns=list(FEATURES), params=PARAMS,
                 data_identity=prepared['identity'], calendar_identity=recipe.key_frame_identity(
                     cal.assign(code='__OBSERVED_SESSION__')),
                 expected_train_feature_dates=len(actual_train), expected_score_dates=len(expected_test),
                 calendar_pool_coverage=dict(nominal_train_calendar_days=len(expected_train),
                     nonempty_train_pool_days=len(actual_train), empty_train_pool_days=len(missing_train),
                     original_zero_pool_dates=list(missing_train), score_calendar_days=len(expected_test),
                     nonempty_score_pool_days=len(actual_test), empty_score_pool_dates=[],
                     complete_nominal_partition=True, calendar_dates_removed=0, synthetic_rows_added=0,
                     evidence=original_empty_pool_proof),
                 train_rows=len(train), train_signal_dates=int(train.date.nunique()), score_rows=len(test),
                 training_label_preparation_slots=len(labels),
                 excluded_or_purged_train_feature_rows=len(feature_train) - len(train),
                 max_model_train_label_end=pd.Timestamp(train.label_end_date.max()).date().isoformat(),
                 earliest_score_date=pd.Timestamp(test.date.min()).date().isoformat(),
                 train_labels_only=True, test_labels_read=False, source_pit_certification='CANNOT_CONFIRM')
    return prepared, proof


def new_time_use_status(config, proof):
    ref = config.get('new_use_passport')
    if ref is None:
        return dict(mode='NARROW_NEW_TIME_USE_PROOF_REQUIRES_REVIEW',
                    first_source_pit_certification='CANNOT_CONFIRM', full_data_reaudit=False)
    passport = checked_json(ref)
    if (passport.get('status') != 'CERTIFIED_FOR_THIS_TIME_USE'
            or passport.get('policy_sha256') != POLICY_SHA256
            or passport.get('recipe_sources') != config['recipe_sources']
            or passport.get('cache_files') != config['cache_files']
            or passport.get('data_identity') != proof['data_identity']):
        raise Blocked('PASSPORT_DOES_NOT_COVER_NEW_TIME_USE')
    return dict(mode='FAST_BIND_EXISTING_VALID_TIME_USE_PASSPORT', passport_sha256=ref['sha256'],
                first_source_pit_certification=passport.get('source_pit_status', 'CANNOT_CONFIRM'), full_data_reaudit=False)


class DeferredFitTerminal:
    """Use official claim/terminal; finish only AFTER durable model/scores/receipt."""
    def __init__(self, guard):
        self.guard, self.claim = guard, None
        self.entered = self.returned = False
        self.terminal_attempted = False

    def execute_once(self, *, fit_callable, **identity):
        if self.claim is not None:
            raise Blocked('ONE_FIT_ONLY_NO_RETRY')
        self.claim = self.guard.claim(**identity)
        self.entered = True
        result = fit_callable()
        self.returned = True
        return result

    def finish(self, success, error=None):
        if self.claim is not None and not self.terminal_attempted:
            self.terminal_attempted = True
            self.guard.append_terminal(self.claim, terminal_status='COMPLETED' if success else 'FAILED_DURING_FIT_OR_FINALIZATION',
                fit_call_entered=self.entered, real_fit_returned_count=int(self.returned), error_type=error)


def fit_single_cell(recipe, prepared, guard, model_path, execution_id, contract_sha, binding_sha):
    """Delegate every fit parameter/weight/fit/predict/roundtrip to frozen code."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        return recipe.predict_with_roundtrip(recipe.model_for_arm(), prepared['train'], prepared['test'],
            FEATURES, Path(model_path), guard=guard, execution_id=execution_id, arm='M0:H2:2023H2',
            contract_sha256=contract_sha, input_binding_sha256=binding_sha)


def annotate_scores(predictions, prepared, model_sha, proof, config_sha):
    import numpy as np
    if (tuple(predictions.columns) != ('date', 'code', 'prediction')
            or predictions.duplicated(['date', 'code']).any()
            or not _frame_keys(predictions).equals(_frame_keys(prepared['test']))
            or not np.isfinite(predictions.prediction.to_numpy(dtype=float)).all()):
        raise Blocked('SCORE_KEYS_OR_FINITE_SCHEMA_MISMATCH')
    result = predictions.copy()
    result['model_sha256'] = model_sha
    result['model_train_label_end'] = proof['max_model_train_label_end']
    result['fold'] = '2023H2'; result['horizon'] = 2
    result['feature_schema_sha256'] = digest(list(FEATURES))
    result['config_sha256'] = config_sha
    result['input_version'] = POLICY['input_version']
    result['schema'] = 'M13_ORIGINAL_M0_OOT_SCORES_V1'
    return result


def registry_truth_access(event_type, execution_id, report):
    if event_type == 'RUN_STARTED' and not execution_id.endswith('_MODEL'):
        return 'NOT_READ'
    if execution_id.endswith('_MODEL') or report.get('proof', {}).get('train_labels_only'):
        return 'PAST_TRAIN_H2_ONLY_NO_TEST_OUTCOMES'
    return 'CANNOT_CONFIRM_TRAIN_LABEL_PREPARATION_NO_TEST_OUTCOMES'


def make_guard(config, config_path, phase, receipt_path):
    deps = config['registry_dependencies']
    for ref in deps.values():
        if sha(ref['path']) != ref['sha256']:
            raise Blocked('OFFICIAL_REGISTRY_DEPENDENCY_DRIFT')
    core_ref = deps['core']; cached = sys.modules.get('execution_guard_core')
    if cached and Path(cached.__file__).resolve() != Path(core_ref['path']).resolve():
        raise Blocked('REGISTRY_CORE_IMPORT_COLLISION')
    core = cached or _load(core_ref['path'], 'execution_guard_core')
    adapter_module = _load(deps['adapter']['path'], 'm13_official_adapter')
    class ContextAdapter(adapter_module.OfficialRegistryAdapter):
        def _event_spec(self, event, *, event_type):
            spec = super()._event_spec(event, event_type=event_type)
            report = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            spec.update(truth_access=registry_truth_access(event_type, str(event['execution_id']), report),
                sealed_status='EXPOSED_HISTORY_NO_CONFIRMATION', input_sha256=input_binding_identity(config),
                code_sha256=sha(__file__), config_sha256=sha(config_path),
                signal_date_range=dict(start=CELL['train_start'], end=CELL['score_end']),
                target_date_range=dict(train_label_end_exclusive=CELL['score_start']),
                runtime=dict(python=sys.executable, version=platform.python_version(), threads=4))
            if receipt_path.is_file():
                spec['deterministic_metrics'].update(report.get('metrics', {}))
                spec['input_identity']['phase_receipt_sha256'] = sha(receipt_path)
                spec['checkpoint_ref'] = report.get('checkpoint_ref')
            return spec
    profile = dict(track_id='A_SHARE_PANEL_MODEL_SELECTION_RESEARCH', route_id=ROUTE,
        registration_mode='LIVE_EXECUTION', start_stage='M13_' + phase.upper() + '_RUN_STARTED',
        terminal_stage='M13_' + phase.upper() + '_TERMINAL', occurred_at_reason='LIVE_EXECUTION_UTC',
        baseline_id='PV_ENVIRONMENT_R1_M0_FORMAL_EXACT_4THREAD_H2',
        one_variable='MISSING_CHRONOLOGICAL_2023H2_SCORE_PERIOD_ORIGINAL_RECIPE_AND_CACHE',
        signal_scope='SINGLE_2023H2_M0_H2_EXISTING_FIVE_ARM_COMMON_CACHE',
        target_scope='PAST_TRAIN_H2_ONLY_NO_TEST_OUTCOMES', device='CPU_RESEARCH_ONLY',
        evidence_level='ENGINEERING_CANARY_ONLY')
    adapter = ContextAdapter(registry_path=Path(config['registry_path']),
        contract_path=Path(config['contract']['path']), input_binding_path=Path(config_path),
        artifact_paths=(receipt_path,), builder_path=Path(deps['builder']['path']),
        validator_path=Path(deps['validator']['path']), event_profile=profile)
    return core.ExecutionGuard(adapter, experiment_id=EXPERIMENT, max_real_fit_debits=1)


def read_bound_caches(config, *, accepted_stats=None):
    """Preflight authenticates once; accepted fit FAST_BIND checks unchanged stat."""
    import pandas as pd
    import pyarrow as pa
    pa.set_cpu_count(4); pa.set_io_thread_count(4)
    prep = checked_json(config['preparation_receipt'])
    original_binding = checked_json(config['original_binding'])
    if prep.get('status') != 'COMPLETED' or prep.get('feature_columns', {}).get('M0') != list(FEATURES):
        raise Blocked('ORIGINAL_PREPARATION_OR_SCHEMA_NOT_ACCEPTED')
    stats = {}
    frames = {}
    for name in ('m0', 'calendar', 'panel'):
        ref = config['cache_files'][name]; path = Path(ref['path'])
        if prep['cache_files'].get(str(path)) != ref['sha256']:
            raise Blocked('CACHE_NOT_ORIGINAL_PREPARATION_OUTPUT')
        before = stat_identity(path)
        if accepted_stats is None:
            if sha(path) != ref['sha256']:
                raise Blocked('CACHE_SHA_DRIFT')
        elif accepted_stats.get(name) != before:
            raise Blocked('FAST_BIND_STAT_DRIFT_REQUIRES_NEW_PREFLIGHT')
        filters = None
        if name == 'm0':
            filters = [('date','>=',pd.Timestamp(CELL['train_start'])),
                       ('date','<=',pd.Timestamp(CELL['score_end']))]
        elif name == 'panel':
            # No score-period prices/outcomes are needed for this cell's fit.
            filters = [('date','>=',pd.Timestamp(CELL['train_start'])),
                       ('date','<',pd.Timestamp(CELL['score_start']))]
        frames[name] = pd.read_parquet(path, columns=['date','code','close'] if name == 'panel' else None,
                                      filters=filters)
        if before != stat_identity(path):
            raise Blocked('CACHE_CHANGED_DURING_READ')
        stats[name] = before
    if original_binding.get('schema_version') is None:
        raise Blocked('ORIGINAL_BINDING_SCHEMA_MISSING')
    return frames, stats


def resource_snapshot(config):
    probe_ref = config.get('resource_probe_source')
    expected_path = Path(__file__).with_name('environment_tilt_mac_recovery.py').resolve()
    if (not probe_ref or Path(probe_ref['path']).resolve() != expected_path
            or sha(expected_path) != probe_ref['sha256']):
        raise Blocked('EXISTING_MAC_RESOURCE_PROBE_BINDING_OR_SHA_DRIFT')
    probe = _load(expected_path, 'm13_bound_mac_resource_probe')
    # The accepted helper returns getrusage peak RSS, not current resident size.
    # Using that peak for the unchanged RSS ceiling is conservative.
    peak = probe.rss_bytes()
    available = probe.available_memory()
    disk = shutil.disk_usage(config['output_root']).free
    limits = config['resource_caps']
    if peak > limits['rss_bytes'] or available < limits['min_available_bytes'] or disk < limits['min_disk_bytes']:
        raise Blocked('RESOURCE_GUARD_STOP')
    return dict(rss_bytes=peak, peak_rss_bytes=peak, rss_is_current=False,
                rss_measurement='GETRUSAGE_PEAK_RSS_CONSERVATIVE_CAP',
                available_bytes=available, free_disk_bytes=disk, platform=platform.system(),
                resource_probe_source=dict(probe_ref))


@contextlib.contextmanager
def deadline(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise Blocked('FINITE_POSITIVE_WALL_BUDGET_REQUIRED')
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('M13_FROZEN_PHASE_WALL_BUDGET')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def run_phase(config, config_path, activation, phase):
    require_activation(config, config_path, activation, phase)
    if platform.system() != 'Darwin' or sys.executable != config['python']:
        raise Blocked('FROZEN_MAC_PYTHON_ONLY')
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy','pandas','scikit-learn','joblib','pyarrow','threadpoolctl')}
    versions['python'] = platform.python_version()
    if config.get('runtime_versions') != versions:
        raise Blocked('FROZEN_RUNTIME_VERSION_DRIFT')
    out = Path(config['output_root']); out.mkdir(exist_ok=True)
    with (out / '.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = out / (phase.upper() + '_RECEIPT.json')
        if receipt_path.exists():
            raise Blocked('EXISTING_PHASE_RECEIPT_NO_AUTOMATIC_REENTRY')
        report = dict(status='INTENT_BEFORE_REGISTERED_START', phase=phase, policy_sha256=POLICY_SHA256,
            contract_sha256=config['contract']['sha256'], input_binding_identity=input_binding_identity(config),
            wrapper_sha256=sha(__file__), runtime_versions=versions,
            metrics=dict(truth_positive_denominator=None, selection_slots=None), **FLAGS)
        write_json(receipt_path, report)
        guard = make_guard(config, config_path, phase, receipt_path)
        execution_id = config['execution_ids'][phase]
        claim = guard.claim_no_fit(execution_id=execution_id, arm='M13_' + phase.upper() + '_INPUT_BINDING',
            contract_sha256=config['contract']['sha256'], input_binding_sha256=sha(config_path))
        transaction = DeferredFitTerminal(guard)
        nofit_terminal_attempted = False
        started = time.monotonic()
        try:
            with deadline(config['resource_caps'][phase + '_wall_seconds']):
                report['resource_before'] = resource_snapshot(config)
                empty_pool_proof = load_original_empty_pool_proof(config)
                accepted = checked_json(config['accepted_preflight']) if phase == 'fit' else None
                frames, stats = read_bound_caches(config, accepted_stats=accepted['cache_stats'] if accepted else None)
                recipe = load_recipe(config)
                prepared, proof = prepare_single_cell(recipe, frames['m0'], frames['panel'], frames['calendar'],
                                                       original_empty_pool_proof=empty_pool_proof)
                report.update(proof=proof, cache_stats=stats, new_time_use=new_time_use_status(config, proof))
                if accepted and proof != accepted['proof']:
                    raise Blocked('ACCEPTED_PREFLIGHT_PREPARED_IDENTITY_DRIFT')
                if phase == 'preflight':
                    report['status'] = 'PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW'
                else:
                    report['resource_pre_fit'] = resource_snapshot(config)
                    model_path = out / 'M0_H2_2023H2.joblib'
                    if model_path.exists() or (out / 'SCORES.csv').exists():
                        raise Blocked('EXISTING_CHECKPOINT_OR_SCORES_NO_OVERWRITE')
                    predictions = fit_single_cell(recipe, prepared, transaction, model_path, execution_id + '_MODEL',
                        config['contract']['sha256'], sha(config_path))
                    model_sha = sha(model_path)
                    scores = annotate_scores(predictions, prepared, model_sha, proof, sha(config_path))
                    with (out / 'SCORES.csv').open('x') as handle:
                        scores.to_csv(handle, index=False); handle.flush(); os.fsync(handle.fileno())
                    # Persist actual estimator bytes before declaring the fit complete.
                    with model_path.open('rb') as handle:
                        os.fsync(handle.fileno())
                    report.update(status='SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION',
                        checkpoint_ref=dict(path=str(model_path), sha256=model_sha),
                        scores=dict(path=str(out / 'SCORES.csv'), sha256=sha(out / 'SCORES.csv'), rows=len(scores)),
                        accepted_preflight_sha256=config['accepted_preflight']['sha256'])
                    report['metrics'].update(fit_calls=1, model_prediction_rows=len(scores), roundtrip_exact=True)
                report.update(elapsed_seconds=time.monotonic()-started, resource_after=resource_snapshot(config))
                write_json(receipt_path, report)
                if phase == 'fit':
                    transaction.finish(True)
                nofit_terminal_attempted = True
                guard.append_no_fit_terminal(claim, succeeded=True)
        except BaseException as exc:
            report.update(status='FAILED_OR_INTERRUPTED_NO_RETRY', error_type=type(exc).__name__,
                          error=str(exc)[:500], elapsed_seconds=time.monotonic()-started)
            finalizers = [lambda: write_json(receipt_path, report),
                          lambda: transaction.finish(False, type(exc).__name__)]
            if not nofit_terminal_attempted:
                finalizers.append(lambda: guard.append_no_fit_terminal(claim, succeeded=False,
                                                                       error_type=type(exc).__name__))
            for finalize in finalizers:
                try:
                    finalize()
                except BaseException as terminal_error:
                    exc.add_note('Finalization unconfirmed: ' + repr(terminal_error))
            raise
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight','fit'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--activation', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    activation = json.loads(args.activation.read_text())
    print(canonical(run_phase(config, args.config, activation, args.phase)))


if __name__ == '__main__':
    main()
