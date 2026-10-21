"""One-symbol original-formula warmup diagnostic; no labels, models or pool.

Pure APIs accept synthetic frames. The sole actual entry consumes an official
no-fit guard supplied by the executor, after exact activation. There is no query
or unregistered data entry CLI and no alternate formula implementation.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import environment_tilt_m0_oof as support

FLAGS = support.FLAGS
FEATURES = support.FEATURES
BATCH = support.BATCH
SYMBOL = '600000.SH'
PREFIX_EXECUTION = 'M19_PARENT8G_CANARY_20261021_01_600000_SH'
UNIT_RULING_SHA256 = 'f4927febb4345c37cebd539715a893606763ac21c69bdc0ec43245978b0d8e24'
UNIT_DATES = ('2021-08-02', '2022-01-04', '2023-01-03', '2024-01-02', '2026-10-15')
UNIT_POLICY = dict(status='OPERATIONAL_CHECK_REQUIRED', old_amount_multiplier=10000.0,
    prefix_amount_input='BOUND_ORIGINAL_RAW_CNY', volume_unit='SHARES',
    common_amount_scale='UNCHANGED_ORIGINAL_RECIPE_NUMERIC_SCALE',
    candidate_old_numeric_over_raw_CNY=[1.0, 0.0001], dates=list(UNIT_DATES),
    predicate='FLOAT32_OLD_NUMERIC_DIV_RATIO_EQUALS_ORIGINAL_RAW_FLOAT32_ALL_FIVE_UNIQUE',
    physical_or_economic_unit_certification=False)
POLICY = dict(symbol=SYMBOL, prefix_rows=5147,
    formula='ORIGINAL_CORE36_NO_TRIGGERS_NO_INDEX_NO_AUDITM',
    history='A_ORIGINAL_OLD_ROWS_B_ALL_BOUND_PREFIX_PLUS_IDENTICAL_OLD_ROWS',
    taint='ORIGINAL_FEATURE_VIEW_AND_M0_RECURSIVE_SUFFIX',
    oracle='A_REPRODUCES_CACHE_KEYS_FIRST_THEN_B_OLD_OVERLAP_AND_CACHE_KEYS',
    equality='EXACT_KEYS_ORDER_AND_LITTLE_ENDIAN_FLOAT64_BYTES',
    scope='SINGLE_SYMBOL_FORMULA_ONLY_NOT_UNIVERSE_OR_CELL_PARITY', **FLAGS)
POLICY['unit_admission'] = UNIT_POLICY
POLICY_SHA256 = support.digest(POLICY)
# New explicit admission only; POLICY remains the historical strict identity.
M40_SOURCE_RECEIPT_SHA256 = 'ca421b51979cbb232301d329536532ed338283b70e09264d92b0add007f32020'
M40_REVIEW_SHA256 = '6c1de79e4a15b891aad4cdd3f9222e49297a555198ced8354ddd42e093d50e9b'
M40_CONTROLLER_SHA256 = '599a4f78f9cad4074781cf7ab543206e4c4926fcdac1fa3fa58d36ac41656a97'
PRECISION_UNIT_POLICY = {**UNIT_POLICY, 'status': 'M40_REVIEWED_FIXED_PAIR_RATIO1',
    'candidate_old_numeric_over_raw_CNY': [1.0],
    'predicate': 'M40_INDEPENDENT_PRECISION_RECEIPT_AND_UNCHANGED_FIXED_INPUT_PAIRS',
    'precision_source_receipt_sha256': M40_SOURCE_RECEIPT_SHA256}
PRECISION_POLICY = {**POLICY, 'unit_admission': PRECISION_UNIT_POLICY}

CAPS = dict(workers=1, threads=4, max_rows=10000, batch_rows=512,
    parent_rss_bytes=2**31, worker_rss_bytes=2**31, rss_bytes=2**31,
    min_available_bytes=8*2**30, min_disk_bytes=4*2**30,
    wall_seconds=600, output_bytes=64*2**20, metadata_bytes=2**20,
    hash_chunk_bytes=2**20, hash_total_bytes=136285944)
AUTHENTICATION_MODES = dict(prefix='HASH_BIND_AFTER_START', old_qfq='HASH_BIND_AFTER_START',
    raw_day='HASH_BIND_AFTER_START', old_m0='FAST_BIND_ACCEPTED', old_calendar='FAST_BIND_ACCEPTED')
EXPERIMENT = 'A_SHARE_ENV_TILT_600000_WARMUP_CANARY_20261021'
REGISTRY_PROFILE = dict(track_id='A_SHARE_PANEL_MODEL_SELECTION_RESEARCH',
    route_id='a-share-environment-tilt-600000-warmup-canary-20261021', registration_mode='LIVE_EXECUTION',
    start_stage='M30_FORMULA_CANARY_RUN_STARTED', terminal_stage='M30_FORMULA_CANARY_TERMINAL',
    occurred_at_reason='LIVE_EXECUTION_UTC', baseline_id='ORIGINAL_S1_M0_CORE36_CACHE_SINGLE_SYMBOL',
    one_variable='PREPEND_BOUND_FULL_HISTORY_PRESERVING_ORIGINAL_EFFECTIVE_AMOUNT_SCALE',
    signal_scope='FIXED_600000_ONLY_ORIGINAL_AVAILABLE_HISTORY', target_scope='NO_LABELS_NO_MODEL_NO_RETURNS',
    device='CPU_RESEARCH_ONLY', evidence_level='ENGINEERING_CANARY_ONLY')


class Blocked(ValueError):
    pass


def reference(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get('path'), str)
            or not Path(ref['path']).is_absolute() or not isinstance(ref.get('sha256'), str)
            or len(ref['sha256']) != 64 or any(c not in '0123456789abcdef' for c in ref['sha256'])):
        raise Blocked('FINAL_ABSOLUTE_PATH_AND_SHA_REQUIRED')


def scientific_identity(config):
    keys = ('policy', 'caps', 'sources', 'inputs', 'columns', 'units', 'fast_bind_origins',
            'prefix_terminal_event_sha256', 'python', 'executor_owner', 'authentication')
    if 'precision_admission' in config:
        keys += ('precision_admission',)
    return support.digest({k: config[k] for k in keys})


def validate_config(config):
    if config.get('state') != 'FROZEN_READY_FOR_ACTIVATION':
        raise Blocked('DRAFT_NOT_EXECUTABLE')
    precision = 'precision_admission' in config
    policy = PRECISION_POLICY if precision else POLICY
    if config.get('policy') != policy or config.get('caps') != CAPS:
        raise Blocked('EXACT_SINGLE_SYMBOL_POLICY_AND_CAPS_REQUIRED')
    if any(config.get(k) != v for k, v in FLAGS.items()):
        raise Blocked('RESEARCH_BOUNDARY_REQUIRED')
    if set(config.get('sources', {})) != {'recipe', 'formula', 'engine', 'fast_builder', 'c_source',
            'c_binary', 'support', 'resource_probe', 'guard_adapter', 'registry_adapter',
            'registry_builder', 'registry_validator', 'raw_reader', 'v192', 'v193'}:
        raise Blocked('EXACT_ORIGINAL_FORMULA_C_AND_OFFICIAL_GUARD_SOURCES_REQUIRED')
    if set(config.get('inputs', {})) != {'prefix', 'old_qfq', 'old_m0', 'old_calendar', 'raw_day',
            'fast_bind', 'unit_ruling', 'm27_checkpoint', 'm27_terminal'}:
        raise Blocked('COMPLETE_INPUT_AND_ORIGIN_BINDINGS_REQUIRED')
    authentication = config.get('authentication', {})
    if set(authentication) != set(AUTHENTICATION_MODES):
        raise Blocked('EXPLICIT_FROZEN_PER_FILE_AUTHENTICATION_REQUIRED')
    total = 0
    for name, mode in AUTHENTICATION_MODES.items():
        item = authentication[name]
        if not isinstance(item, dict) or item.get('mode') != mode:
            raise Blocked('AUTHENTICATION_MODE_CHANGE_NO_FALLBACK')
        if mode == 'HASH_BIND_AFTER_START':
            if set(item) != {'mode', 'expected_bytes'} or type(item.get('expected_bytes')) is not int or item['expected_bytes'] <= 0:
                raise Blocked('FROZEN_POSITIVE_EXPECTED_HASH_BYTES_REQUIRED')
            total += item['expected_bytes']
        elif set(item) != {'mode'}:
            raise Blocked('FAST_BIND_ACCEPTED_ONLY_NO_HASH_SUBSTITUTION')
    if total > CAPS['hash_total_bytes']:
        raise Blocked('TOTAL_HASH_INPUT_BYTE_BOUND')
    for ref in (*config['sources'].values(), *config['inputs'].values(), config['contract'],
                *config.get('fast_bind_origins', {}).values()):
        reference(ref)
    if not config.get('fast_bind_origins') or not config.get('prefix_terminal_event_sha256'):
        raise Blocked('ORIGINAL_PASSPORT_AND_M27_TERMINAL_REQUIRED')
    if config['inputs']['unit_ruling']['sha256'] != UNIT_RULING_SHA256:
        raise Blocked('FROZEN_CONTROLLER_UNIT_RULING_REQUIRED')
    if precision:
        refs = config['precision_admission']
        if (not isinstance(refs, dict) or set(refs) != {'source_receipt', 'independent_review', 'controller_ruling'}
                or M40_REVIEW_SHA256 is None):
            raise Blocked('EXACT_INDEPENDENT_M40_RECEIPTS_REQUIRED')
        for ref in refs.values(): reference(ref)
        if (refs['source_receipt']['sha256'] != M40_SOURCE_RECEIPT_SHA256
                or refs['independent_review']['sha256'] != M40_REVIEW_SHA256
                or refs['controller_ruling']['sha256'] != M40_CONTROLLER_SHA256):
            raise Blocked('FROZEN_M40_RECEIPT_IDENTITY_REQUIRED')
    if config.get('units') != (PRECISION_UNIT_POLICY if precision else UNIT_POLICY):
        raise Blocked('EXACT_OPERATIONAL_UNIT_ADMISSION_NO_INFERRED_MULTIPLIER_REQUIRED')
    if set(config.get('columns', {})) != {'prefix', 'old_qfq'}:
        raise Blocked('EXACT_RAW_COLUMN_MAPS_REQUIRED')
    for mapping in config['columns'].values():
        if set(mapping) != {'date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount'}:
            raise Blocked('EXACT_RAW_COLUMN_MAPS_REQUIRED')
    out = Path(config['output_root'])
    if (not out.is_absolute() or out.parent.resolve() != BATCH.resolve()
            or not out.name.startswith('M30_') or out.is_symlink()):
        raise Blocked('NEW_OWNED_M30_OUTPUT_REQUIRED')
    if not config.get('execution_id') or config.get('executor_owner') != '/root/environment_tilt_design_writer':
        raise Blocked('UNIQUE_EXECUTOR_IDENTITY_REQUIRED')
    if not isinstance(config.get('python'), str) or not Path(config['python']).is_absolute():
        raise Blocked('EXACT_RUNTIME_REQUIRED')
    if config.get('registry_path') != str(BATCH.parents[3] / 'contracts/EXPERIMENT_REGISTRY.jsonl'):
        raise Blocked('ORIGINAL_CANONICAL_REGISTRY_REQUIRED')


def require_activation(config, activation):
    validate_config(config)
    expected = dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE', execution_id=config['execution_id'],
        scientific_identity=scientific_identity(config), wrapper_sha256=support.sha(__file__),
        config_sha256=support.digest(config), contract_sha256=config['contract']['sha256'])
    if any(activation.get(k) != v for k, v in expected.items()) or not activation.get('review_receipt_sha256'):
        raise Blocked('EXACT_REVIEWED_ACTIVATION_REQUIRED')


def symbol_frame(frame, *, columns=None, max_rows=10000, preserve_order=False):
    import pandas as pd
    if len(frame) > max_rows or frame.empty:
        raise Blocked('EMPTY_OR_OVERSIZE_SYMBOL_FRAME')
    if columns is not None and tuple(frame.columns) != tuple(columns):
        raise Blocked('EXACT_FEATURE_SCHEMA_REQUIRED')
    if not {'date', 'code'}.issubset(frame.columns) or not frame.code.eq(SYMBOL).all():
        raise Blocked('ONLY_FIXED_SYMBOL_ALLOWED')
    dates = pd.to_datetime(frame.date)
    if dates.isna().any() or getattr(dates.dt, 'tz', None) is not None or not dates.equals(dates.dt.normalize()):
        raise Blocked('EXACT_NORMALIZED_DATES_REQUIRED')
    result = frame.copy(); result['date'] = dates
    if result.duplicated(['date', 'code']).any():
        raise Blocked('DUPLICATE_SYMBOL_KEYS')
    return (result if preserve_order else result.sort_values(['date', 'code'], kind='mergesort')).reset_index(drop=True)


def compose(recipe, prefix_raw, old_raw, maps, units, *, expected_prefix_rows=5147, max_rows=10000):
    """Original normalizer only; no inferred currency or adjustment conversion."""
    import pandas as pd
    if len(prefix_raw) != expected_prefix_rows or len(prefix_raw) + len(old_raw) > max_rows:
        raise Blocked('EXACT_PREFIX_COUNT_OR_TOTAL_ROW_BOUND')
    ratio = units['accepted_old_numeric_over_raw_CNY']
    if ratio not in UNIT_POLICY['candidate_old_numeric_over_raw_CNY'] or units['old_amount_multiplier'] != 10000.0:
        raise Blocked('ONLY_ADMITTED_RATIO_AND_ORIGINAL_MULTIPLIER')
    old = symbol_frame(recipe.normalize_price_panel(old_raw, maps['old_qfq'],
        amount_multiplier=units['old_amount_multiplier']), max_rows=max_rows)
    # Preserve the locked order of floating-point operations: first map raw CNY
    # into original input numeric scale, THEN call the original x10000 recipe.
    # Combining those multipliers algebraically can change float64 bytes.
    numeric_prefix = prefix_raw.copy(deep=True)
    numeric_prefix[maps['prefix']['amount']] = pd.to_numeric(numeric_prefix[maps['prefix']['amount']], errors='raise').astype(float) * ratio
    prefix = symbol_frame(recipe.normalize_price_panel(numeric_prefix, maps['prefix'],
        amount_multiplier=10000.0), max_rows=max_rows)
    if prefix.date.max() >= old.date.min():
        raise Blocked('PREFIX_OVERLAPS_OLD_NO_OVERWRITE')
    full = pd.concat([prefix, old], ignore_index=True).sort_values(['date', 'code'], kind='mergesort').reset_index(drop=True)
    retained = full.loc[full.date >= old.date.min()].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, old, check_exact=True)
    return old, full


def unit_admission(old_numeric, raw_CNY, *, precision_evidence=None):
    """Frozen operational relation, based on original raw float32 storage.

    This is not an economic-unit/source-chain certification. No relative-error
    fitting, alternative dates, dropped zeros or replacement samples allowed.
    """
    if precision_evidence is not None:
        return receipt_unit_admission(old_numeric, raw_CNY, precision_evidence)
    import numpy as np
    import pandas as pd
    old_numeric = symbol_frame(old_numeric)
    raw_CNY = symbol_frame(raw_CNY)
    pairs = []
    for date in UNIT_DATES:
        day = pd.Timestamp(date)
        a = old_numeric.loc[old_numeric.date.eq(day), 'amount']
        b = raw_CNY.loc[raw_CNY.date.eq(day), 'amount']
        if len(a) != 1 or len(b) != 1:
            return dict(decision='UNIT_ADMISSION_BLOCKED', reason='FIXED_UNIT_DATE_MISSING', pairs=pairs, B_allowed=False)
        x, y = float(a.iloc[0]), float(b.iloc[0])
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
            return dict(decision='UNIT_ADMISSION_BLOCKED', reason='NONPOSITIVE_OR_NONFINITE_UNIT_PAIR', pairs=pairs, B_allowed=False)
        with np.errstate(over='ignore', invalid='ignore'):
            if float(np.float32(y)) != y:
                return dict(decision='UNIT_ADMISSION_BLOCKED', reason='RAW_AMOUNT_NOT_ORIGINAL_FLOAT32', pairs=pairs, B_allowed=False)
            matches = [bool(np.isfinite(np.float32(x/r))) and float(np.float32(x/r)) == y
                       for r in UNIT_POLICY['candidate_old_numeric_over_raw_CNY']]
        pairs.append(dict(date=date, code=SYMBOL, old_numeric=x, raw_CNY=y,
                          candidate_matches=dict(zip(('1', '0.0001'), matches))))
    accepted = [r for r, key in zip(UNIT_POLICY['candidate_old_numeric_over_raw_CNY'], ('1', '0.0001'))
                if all(pair['candidate_matches'][key] for pair in pairs)]
    if len(accepted) != 1:
        return dict(decision='UNIT_ADMISSION_BLOCKED', reason='NO_UNIQUE_FIXED_RATIO_ALL_FIVE', pairs=pairs, B_allowed=False)
    return dict(decision='OPERATIONAL_UNIT_RELATION_ACCEPTED_SINGLE_SYMBOL', pairs=pairs,
        accepted_old_numeric_over_raw_CNY=accepted[0], prefix_amount_multiplier=10000.0*accepted[0],
        B_allowed=True, physical_or_economic_unit_certification=False,
        rule=UNIT_POLICY['predicate'], tolerance='ORIGINAL_RAW_IEEE754_FLOAT32_QUANTIZATION_ONLY')


def m40_precision_evidence(config):
    """Authenticate the fixed independent ruling, not infer a new tolerance.

    Called after durable START, before market bytes. Receipt SHA binds the
    independent PASS_WITH_LIMITS; internal native formatter attribution remains
    unverified and is not a reuse blocker. Only the same five bound input files
    and the exact same recorded pairs may consume this one-symbol ruling.
    """
    refs = config['precision_admission']
    for name, wanted in (('source_receipt', M40_SOURCE_RECEIPT_SHA256),
                         ('independent_review', M40_REVIEW_SHA256),
                         ('controller_ruling', M40_CONTROLLER_SHA256)):
        ref = refs[name]; reference(ref)
        if wanted is None or ref['sha256'] != wanted:
            raise Blocked('FROZEN_M40_RECEIPT_IDENTITY_REQUIRED')
        if Path(ref['path']).stat().st_size > CAPS['metadata_bytes']:
            raise Blocked('M40_METADATA_BYTE_BOUND')
    source = support.checked_json(refs['source_receipt'])
    review = support.checked_json(refs['independent_review'])
    ruling = support.checked_json(refs['controller_ruling'])
    if (review.get('status') != 'PASS_WITH_LIMITS'
            or ruling.get('schema') != 'M40_CONTROLLER_REUSE_RULING_V1'
            or ruling.get('decision') != 'REUSE_ACCEPTED_ORIGINAL_AMOUNT_VALUES'
            or ruling.get('review_ref') != refs['independent_review']
            or ruling.get('writer_ref') != refs['source_receipt']
            or ruling.get('raw_CNY_to_existing_input_numeric_ratio') != 1
            or any(ruling.get(k) is not True for k in ('old_values_unchanged',
                'old_m30_failure_unchanged', 'feature_matrix_changes_from_real_history_repairs_not_waived',
                'original_model_and_evaluation_unchanged'))
            or any(document.get(k) != v for document in (source, review, ruling) for k, v in FLAGS.items())):
        raise Blocked('M40_INDEPENDENT_PASS_AND_CONTROLLER_RATIO1_REQUIRED')
    if (source.get('schema') != 'M40_AMOUNT_PRECISION_SOURCE_REVIEW_V1'
            or source.get('source_rule_gap_blocks_original_data_reuse') is not False
            or source.get('candidate_rounding', {}).get('all_five_match') is not True
            or source.get('candidate_rounding', {}).get('universal_tolerance_admitted') is not False):
        raise Blocked('M40_FIXED_PRECISION_SCOPE_REQUIRED')
    old_ref = source['evidence']['m30_receipt']; reference(old_ref)
    if Path(old_ref['path']).stat().st_size > CAPS['metadata_bytes']:
        raise Blocked('M40_METADATA_BYTE_BOUND')
    old = support.checked_json(old_ref)
    if (old.get('status') != 'BLOCKED_NO_RETRY'
            or old.get('unit_admission', {}).get('reason') != 'NO_UNIQUE_FIXED_RATIO_ALL_FIVE'
            or any(old.get('input_refs', {}).get(k) != config['inputs'][k]
                   for k in AUTHENTICATION_MODES)):
        raise Blocked('M40_ORIGINAL_FIXED_INPUT_IDENTITIES_REQUIRED')
    pairs = old['unit_admission']['pairs']
    if (not isinstance(pairs, list) or len(pairs) != len(UNIT_DATES)
            or any(row.get('date') != date or row.get('code') != SYMBOL
                   or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
                          for k in ('old_numeric', 'raw_CNY'))
                   for date, row in zip(UNIT_DATES, pairs))):
        raise Blocked('M40_EXACT_FIVE_RECORDED_PAIRS_REQUIRED')
    return dict(schema='M40_AUTHENTICATED_FIXED_PAIR_RATIO1', accepted_old_numeric_over_raw_CNY=1.0,
                pairs=pairs, receipt_refs=refs, original_m30_receipt=old_ref)


def receipt_unit_admission(old_numeric, raw_CNY, evidence):
    """Reuse a reviewed fixed receipt; no arithmetic tolerance or scale search."""
    import pandas as pd
    if (evidence.get('schema') != 'M40_AUTHENTICATED_FIXED_PAIR_RATIO1'
            or evidence.get('accepted_old_numeric_over_raw_CNY') != 1.0
            or len(evidence.get('pairs', [])) != len(UNIT_DATES)):
        raise Blocked('M40_AUTHENTICATED_FIXED_RATIO1_REQUIRED')
    old_numeric, raw_CNY = symbol_frame(old_numeric), symbol_frame(raw_CNY)
    pairs = []
    for date, bound in zip(UNIT_DATES, evidence['pairs']):
        day = pd.Timestamp(date)
        a = old_numeric.loc[old_numeric.date.eq(day), 'amount']
        b = raw_CNY.loc[raw_CNY.date.eq(day), 'amount']
        if (bound.get('date') != date or bound.get('code') != SYMBOL or len(a) != 1 or len(b) != 1):
            raise Blocked('M40_FIXED_CODE_DATE_PAIR_REQUIRED')
        x, y = float(a.iloc[0]), float(b.iloc[0])
        if (not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0
                or x != bound['old_numeric'] or y != bound['raw_CNY']):
            raise Blocked('M40_FIXED_PAIR_VALUES_CHANGED_NO_TOLERANCE')
        pairs.append(dict(date=date, code=SYMBOL, old_numeric=x, raw_CNY=y))
    return dict(decision='M40_REVIEWED_OPERATIONAL_RATIO1_ACCEPTED_SINGLE_SYMBOL',
                accepted_old_numeric_over_raw_CNY=1.0, prefix_amount_multiplier=10000.0,
                B_allowed=True, pairs=pairs, receipt_refs=evidence['receipt_refs'],
                original_m30_receipt=evidence['original_m30_receipt'],
                rule=PRECISION_UNIT_POLICY['predicate'], tolerance='NONE_FIXED_REVIEWED_PAIR_REUSE_ONLY',
                old_numeric_values_preserved=True, physical_or_economic_unit_certification=False)


def compare(left, right):
    """All 36 byte identities including nonfinite payloads, with bounded details."""
    import numpy as np
    left = symbol_frame(left, columns=('date', 'code', *FEATURES), preserve_order=True)
    right = symbol_frame(right, columns=('date', 'code', *FEATURES), preserve_order=True)
    keys = tuple(zip(left.date, left.code)) == tuple(zip(right.date, right.code))
    if not keys:
        return dict(exact=False, keys_order_exact=False, left_rows=len(left), right_rows=len(right), columns={})
    report = {}
    for name in FEATURES:
        a = np.asarray(left[name], dtype='<f8'); b = np.asarray(right[name], dtype='<f8')
        finite_diff = np.isfinite(a) != np.isfinite(b)
        bits = a.view('<u8') != b.view('<u8')
        positions = np.flatnonzero(bits)
        report[name] = dict(byte_difference_count=int(bits.sum()), finite_mask_difference_count=int(finite_diff.sum()),
            left_finite_count=int(np.isfinite(a).sum()), right_finite_count=int(np.isfinite(b).sum()),
            left_nan_count=int(np.isnan(a).sum()), right_nan_count=int(np.isnan(b).sum()),
            left_inf_count=int(np.isinf(a).sum()), right_inf_count=int(np.isinf(b).sum()),
            first_difference_date=left.date.iloc[positions[0]].date().isoformat() if len(positions) else None,
            last_difference_date=left.date.iloc[positions[-1]].date().isoformat() if len(positions) else None,
            left_float64_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            right_float64_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    return dict(exact=not any(v['byte_difference_count'] for v in report.values()), keys_order_exact=True,
        rows=len(left), columns=report, dtype='FLOAT64_LITTLE_ENDIAN_EXACT_BYTES_NO_TOLERANCE')


def on_keys(frame, keys):
    result = keys[['date', 'code']].merge(frame, on=['date', 'code'], how='left', validate='one_to_one',
        sort=False, indicator=True)
    if not result['_merge'].eq('both').all():
        raise Blocked('REFERENCE_KEYS_MISSING_NO_REFILL')
    return result.drop(columns='_merge')[['date', 'code', *FEATURES]]


def original_features(recipe, formula, panel, calendar, texts):
    """No build_nested_arm_frames/common-pool path; original M0 taint only."""
    if tuple(recipe.CORE_FEATURES) != FEATURES or tuple(formula.CORE_OUTPUTS) != FEATURES:
        raise Blocked('ORIGINAL_CORE36_ORDER_CHANGED')
    feature_view, quarantine = recipe.derive_price_feature_view(panel)
    ff = formula.formula_feature_frame(feature_view, v192_text=texts['v192'], v193_text=texts['v193'],
        index_close_by_date=None, include_v192_trigger=False, include_v193_trigger=False, include_auditm=False)
    ff = symbol_frame(ff, columns=('date', 'code', *FEATURES))
    if tuple(ff.date) != tuple(panel.date):
        raise Blocked('ORIGINAL_FORMULA_CHANGED_KEYS')
    # Original helper accepts M0 alone; no market, industry or other arms built.
    ff, arms, taint = recipe.apply_ohlc_taint_masks(ff, {'M0': ff.copy()}, calendar, quarantine)
    return arms['M0'][['date', 'code', *FEATURES]], dict(quarantine=quarantine, taint=taint)


def run_frames(recipe, formula, old, full, reference_cache, old_calendar, texts, *, check=lambda stage: None):
    import numpy as np
    import pandas as pd
    old = symbol_frame(old); full = symbol_frame(full)
    reference_cache = symbol_frame(reference_cache, columns=('date', 'code', *FEATURES), preserve_order=True)
    if not np.isfinite(reference_cache[list(FEATURES)].to_numpy(dtype=float)).all():
        raise Blocked('ORIGINAL_COMMON_CACHE_MUST_BE_FINITE')
    retained = full.loc[full.date >= old.date.min()].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, old, check_exact=True)
    base_calendar = recipe.normalize_calendar(old_calendar.rename(columns={'date': 'trade_date'}), old)
    dates = sorted(set(base_calendar.date) | set(full.date))
    extended = recipe.normalize_calendar(pd.DataFrame(dict(trade_date=dates, calendar_index=range(len(dates)))), full)
    overlap = extended.loc[extended.date.between(base_calendar.date.min(), base_calendar.date.max())]
    if tuple(overlap.date) != tuple(base_calendar.date):
        raise Blocked('OLD_CALENDAR_SEQUENCE_DRIFT')
    shifts = overlap.calendar_index.to_numpy() - base_calendar.calendar_index.to_numpy()
    if len(set(shifts)) != 1:
        raise Blocked('OLD_CALENDAR_INDEX_SHIFT_NOT_CONSTANT')
    calendar_proof = dict(old_sequence_exact=True, index_constant_shift=int(shifts[0]),
        old_dates=len(base_calendar), extended_dates=len(extended), official_calendar='CANNOT_CONFIRM')
    check('before_A')
    a, a_proof = original_features(recipe, formula, old, base_calendar, texts)
    a_cache = compare(on_keys(a, reference_cache), reference_cache)
    if not a_cache['exact']:
        return dict(decision='A_REPRODUCTION_FAILED_STOP', A_cache=a_cache, A_proof=a_proof,
            B_computed=False, calendar_proof=calendar_proof, universe_parity='CANNOT_CONFIRM', **FLAGS)
    check('before_B')
    b, b_proof = original_features(recipe, formula, full, extended, texts)
    ab_overlap = compare(a, on_keys(b, a))
    b_cache = compare(reference_cache, on_keys(b, reference_cache))
    check('after_B')
    return dict(decision='SINGLE_SYMBOL_EXACT' if ab_overlap['exact'] and b_cache['exact'] else 'SINGLE_SYMBOL_DIFFERENT',
        A_cache=a_cache, B_old_overlap=ab_overlap, B_cache=b_cache, A_proof=a_proof, B_proof=b_proof,
        calendar_proof=calendar_proof,
        B_computed=True, old_rows=len(old), full_rows=len(full), cache_rows=len(reference_cache),
        old_first=old.date.min().date().isoformat(), full_first=full.date.min().date().isoformat(),
        universe_parity='CANNOT_CONFIRM', affected_model_cells='CANNOT_CONFIRM_NOT_CHECKED',
        whole_market_arms=0, labels=0, fit_calls=0, query_calls=0, **FLAGS)


def terminal_identity(document, expected_sha256):
    """Deduplicate equal metadata projections; any conflicting terminal stops."""
    def entries(value):
        if isinstance(value, dict):
            yield value
            for child in value.values(): yield from entries(child)
        elif isinstance(value, list):
            for child in value: yield from entries(child)
    projections = [v for v in entries(document) if v.get('execution_id') == PREFIX_EXECUTION
                   and ('terminal_event_sha256' in v or 'closed' in v)]
    identities = {(v.get('execution_id'), v.get('terminal_event_sha256'), v.get('closed'), v.get('decision'))
                  for v in projections}
    wanted = (PREFIX_EXECUTION, expected_sha256, True, 'HISTORICAL_PREFIX_ONLY')
    statuses = {v['status'] for v in projections if 'status' in v}
    if (identities != {wanted} or any(v.get('closed') is not True for v in projections)
            or len(statuses) > 1 or (statuses and not statuses.issubset({'COMPLETED', 'CLOSED', 'HISTORICAL_PREFIX_ONLY'}))):
        raise Blocked('M27_DURABLE_TERMINAL_NOT_BOUND_OR_CONFLICTING')
    return dict(execution_id=PREFIX_EXECUTION, terminal_event_sha256=expected_sha256, closed=True,
        decision='HISTORICAL_PREFIX_ONLY', projection_count=len(projections), distinct_terminal_identity_count=1)


def historical_cache_bindings(origins, origin_refs, files):
    """Recover the already accepted M23 SHA+stat association, never fresh-stat it."""
    names = ('m23_config', 'm23_preflight', 'm23_review')
    if any(name not in origins for name in names):
        raise Blocked('M23_ACCEPTED_CACHE_BINDING_CHAIN_REQUIRED')
    config, preflight, review = (origins[name] for name in names)
    if (review.get('verdict') != 'PASS_NARROW_NEW_TIME_USE'
            or review.get('preflight_receipt_sha256') != origin_refs['m23_preflight']['sha256']
            or preflight.get('status') != 'PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW'
            or preflight.get('input_binding_identity') != support.input_binding_identity(config)):
        raise Blocked('M23_CACHE_PROOF_CHAIN_IDENTITY_DRIFT')
    result = {}
    for name, old_name in (('old_m0', 'm0'), ('old_calendar', 'calendar')):
        ref = config.get('cache_files', {}).get(old_name, {})
        stat = preflight.get('cache_stats', {}).get(old_name, {})
        if ref != files[name] or set(stat) != {'bytes', 'device', 'inode', 'mtime_ns'}:
            raise Blocked('M23_CACHE_SHA_STAT_BINDING_INCOMPLETE_OR_DIFFERENT')
        result[name] = dict(sha256=ref['sha256'], stat=stat)
    return result


class BoundInputs:
    """Explicit post-START hash/accepted FAST_BIND, then bounded numerical IO."""
    def __init__(self, config, session, check):
        self.config, self.session, self.check = config, session, check
        self.accepted = None
        self.hash_bytes = 0
        self.authentication_receipt = None
        self.precision_evidence = None
        self.archive = lambda receipt: None

    def metadata(self, name):
        self.session.require_active()
        ref = self.config['inputs'][name]
        if Path(ref['path']).stat().st_size > CAPS['metadata_bytes']:
            raise Blocked('METADATA_BYTE_BOUND')
        return support.checked_json(ref)

    def admit(self):
        self.session.require_active()
        proof = self.metadata('fast_bind')
        files = {k: self.config['inputs'][k] for k in ('prefix', 'old_qfq', 'old_m0', 'old_calendar', 'raw_day')}
        if (proof.get('schema') != 'M30_SYMBOL_FORMULA_FAST_BIND_V1' or proof.get('decision') != 'PASS'
                or proof.get('symbol') != SYMBOL or proof.get('files') != files
                or proof.get('origins') != self.config['fast_bind_origins']
                or proof.get('sources') != self.config['sources']
                or proof.get('units') != self.config['units']
                or proof.get('authentication') != self.config['authentication']
                or proof.get('prefix_terminal_event_sha256') != self.config['prefix_terminal_event_sha256']):
            raise Blocked('EXACT_REVIEWED_NEW_USE_FAST_BIND_REQUIRED')
        ruling = self.metadata('unit_ruling')
        if (ruling.get('symbol') != SYMBOL
                or ruling.get('unit_check', {}).get('dates') != [int(v.replace('-', '')) for v in UNIT_DATES]
                or ruling.get('unit_check', {}).get('candidate_old_numeric_per_raw_CNY') != [1, .0001]
                or any(ruling.get(k) != v for k, v in FLAGS.items())):
            raise Blocked('UNRESOLVED_OR_CHANGED_UNIT_RULING')
        if 'precision_admission' in self.config:
            self.precision_evidence = m40_precision_evidence(self.config)
        checkpoint = self.metadata('m27_checkpoint')
        prefix = self.config['inputs']['prefix']
        refs = [r for r in checkpoint.get('artifacts', []) if r.get('path') == prefix['path'] and r.get('sha256') == prefix['sha256']]
        if (checkpoint.get('symbol') != SYMBOL or checkpoint.get('status') != 'HISTORICAL_PREFIX_ONLY'
                or checkpoint.get('block_reasons') or checkpoint.get('prefix_rows') != 5147
                or checkpoint.get('coverage', {}).get('history_scope') != 'ALL_ORIGINAL_AVAILABLE_BEFORE_OLD_FIRST'
                or len(refs) != 1 or refs[0].get('amount') != 'CNY' or refs[0].get('vol') != 'shares'):
            raise Blocked('EXACT_M27_PREFIX_CHECKPOINT_REQUIRED')
        member = checkpoint.get('raw_member', {})
        if (member.get('sha256') != self.config['inputs']['raw_day']['sha256']
                or member.get('bytes') != 205056 or member.get('name') != 'sh\\lday\\sh600000.day'):
            raise Blocked('EXACT_M27_ORIGINAL_RAW_MEMBER_REQUIRED')
        terminal = self.metadata('m27_terminal')
        terminal_binding = terminal_identity(terminal, self.config['prefix_terminal_event_sha256'])
        existing = proof.get('bound_files', {})
        if set(existing) != {'old_m0', 'old_calendar'}:
            raise Blocked('ONLY_PREVIOUSLY_ACCEPTED_M0_CALENDAR_FAST_BIND_PROOFS')
        origins = {}
        for name, ref in self.config['fast_bind_origins'].items():
            if Path(ref['path']).stat().st_size > CAPS['metadata_bytes']:
                raise Blocked('ORIGIN_METADATA_BYTE_BOUND')
            origins[name] = support.checked_json(ref)
        historical = historical_cache_bindings(origins, self.config['fast_bind_origins'], files)
        if existing != historical:
            raise Blocked('FAST_BIND_PROOF_DIFFERS_FROM_ACCEPTED_M23_SHA_STAT')
        self.authentication_receipt = dict(schema='M30_PER_FILE_AUTHENTICATION_V1', status='RUNNING',
            execution_id=self.config['execution_id'], terminal_binding=terminal_binding, files={},
            hash_bytes=0, no_numeric_payload_read_yet=True)
        self.archive(self.authentication_receipt)
        accepted = {}
        try:
            # A stale accepted binding stops before any hash; it never falls back.
            for name in ('old_m0', 'old_calendar', 'prefix', 'old_qfq', 'raw_day'):
                ref = files[name]; mode = self.config['authentication'][name]['mode']
                self.session.require_active()
                if mode == 'FAST_BIND_ACCEPTED':
                    bound = existing[name]
                    if bound.get('sha256') != ref['sha256'] or bound.get('stat') != support.stat_identity(ref['path']):
                        raise Blocked('FAST_BIND_PAYLOAD_IDENTITY_OR_STAT_DRIFT')
                    admitted = dict(mode=mode, sha256=ref['sha256'], stat=bound['stat'], rehashed=False)
                elif mode == 'HASH_BIND_AFTER_START':
                    admitted = self.hash_bind(name)
                else:
                    raise Blocked('AUTHENTICATION_MODE_CHANGE_NO_FALLBACK')
                accepted[name] = admitted
                self.authentication_receipt['files'][name] = admitted
                self.authentication_receipt['hash_bytes'] = self.hash_bytes
                self.archive(self.authentication_receipt)
            self.accepted = accepted
            self.authentication_receipt['status'] = 'AUTHENTICATED_BEFORE_NUMERIC_READ'
            self.archive(self.authentication_receipt)
        except BaseException as exc:
            self.authentication_receipt.update(status='FAILED_NO_NUMERIC_READ', error_type=type(exc).__name__,
                error=str(exc)[:1000], hash_bytes=self.hash_bytes)
            self.archive(self.authentication_receipt)
            raise

    def hash_bind(self, name):
        self.session.require_active()
        auth = self.config['authentication'][name]
        if auth.get('mode') != 'HASH_BIND_AFTER_START':
            raise Blocked('HASH_BIND_MODE_NOT_EXPLICIT')
        ref = self.config['inputs'][name]; reference(ref)
        expected = auth['expected_bytes']; before = support.stat_identity(ref['path'])
        if type(expected) is not int or expected <= 0 or before['bytes'] != expected:
            raise Blocked('EXPECTED_HASH_FILE_BYTES_MISMATCH')
        if self.hash_bytes + expected > CAPS['hash_total_bytes']:
            raise Blocked('TOTAL_HASH_INPUT_BYTE_BOUND')
        digest = hashlib.sha256(); count = 0; chunks = 0
        self.check('before_hash_' + name)
        with Path(ref['path']).open('rb') as handle:
            while count < expected:
                self.session.require_active(); self.check('hash_chunk_' + name)
                chunk = handle.read(min(CAPS['hash_chunk_bytes'], expected - count))
                if not chunk: break
                count += len(chunk); self.hash_bytes += len(chunk); chunks += 1
                if count > expected or self.hash_bytes > CAPS['hash_total_bytes']:
                    raise Blocked('HASH_STREAM_BYTE_BOUND')
                digest.update(chunk); self.check('after_hash_chunk_' + name)
        after = support.stat_identity(ref['path'])
        if before != after or count != expected:
            raise Blocked('HASH_FILE_STAT_CHANGED_DURING_AUTHENTICATION')
        observed = digest.hexdigest()
        if observed != ref['sha256']:
            raise Blocked('EXPECTED_PAYLOAD_SHA_DRIFT')
        return dict(mode='HASH_BIND_AFTER_START', expected_sha256=ref['sha256'], sha256=observed,
            stat=after, stat_before=before, stat_stable=True, expected_bytes=expected, bytes_hashed=count, chunks=chunks)

    def raw_amounts(self):
        """Compile only original pure reader definitions; no collector imports."""
        import ast
        import __future__
        import struct
        import numpy as np
        import pandas as pd
        self.session.require_active()
        if self.accepted is None: raise Blocked('FAST_BIND_BEFORE_PAYLOAD_REQUIRED')
        ref = self.config['inputs']['raw_day']; before = support.stat_identity(ref['path'])
        if (before != self.accepted['raw_day']['stat'] or before['bytes'] != 205056
                or before['bytes'] % 32 or before['bytes']//32 > CAPS['max_rows']):
            raise Blocked('RAW_MEMBER_IDENTITY_OR_RECORD_BOUND')
        source = self.config['sources']['raw_reader']
        if support.sha(source['path']) != source['sha256']: raise Blocked('ORIGINAL_RAW_READER_SOURCE_DRIFT')
        nodes = ast.parse(Path(source['path']).read_text()).body
        names = {'read_day_tail', 'is_ext_market_ts'}
        selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
        selected += [n for n in nodes if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id in {'DAY_STRUCT', 'EXT_MARKET_DAY_STRUCT'} for t in n.targets)]
        if {n.name for n in selected if isinstance(n, ast.FunctionDef)} != names:
            raise Blocked('ORIGINAL_RAW_READER_DEFINITIONS_MISSING')
        scope = dict(np=np, pd=pd, struct=struct, Path=Path)
        exec(compile(ast.Module(body=selected, type_ignores=[]), source['path'], 'exec',
            flags=__future__.annotations.compiler_flag), scope)
        self.check('before_original_raw_reader')
        raw = scope['read_day_tail'](SYMBOL, Path(ref['path']), tail_records=before['bytes']//32)
        if support.stat_identity(ref['path']) != before: raise Blocked('RAW_MEMBER_CHANGED_DURING_READ')
        raw = raw.rename(columns={'ts_code':'code'})
        raw['date'] = pd.to_datetime(raw.trade_date.astype(str), format='%Y%m%d', errors='raise')
        return symbol_frame(raw[['date', 'code', 'amount']])

    def m0_symbol_frame(self, path, columns, code_column):
        """Read the bound cache by physical row positions, one value column.

        A filtered wide scanner can buffer global column chunks before emitting
        the few symbol rows. Keep keys only in the first pass; every later pass
        projects one original value column and takes exactly those positions.
        No sorting, conversion, pool filtering, refill or feature arithmetic.
        """
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        if tuple(columns) != ('date', 'code', *FEATURES) or code_column != 'code':
            raise Blocked('EXACT_M0_SINGLE_SYMBOL_PROJECTION_REQUIRED')
        with pq.ParquetFile(path, pre_buffer=False, buffer_size=64 * 1024) as parquet:
            self.check('before_old_m0_keys')
            keys, positions, scanned = [], [], 0
            for batch in parquet.iter_batches(batch_size=CAPS['batch_rows'], columns=['date', 'code'],
                                              use_threads=False):
                self.session.require_active()
                selected = pc.indices_nonzero(pc.equal(batch.column('code'), SYMBOL))
                if len(selected):
                    if len(positions) + len(selected) > CAPS['max_rows']:
                        raise Blocked('SYMBOL_OR_STREAM_ROW_BOUND_NO_TRUNCATION')
                    positions.extend((selected.to_numpy() + scanned).tolist())
                    keys.append(batch.take(selected).to_pandas()[['date', 'code']])
                scanned += len(batch)
            if scanned != parquet.metadata.num_rows:
                raise Blocked('M0_PROJECTED_ROW_PARTITION_CHANGED')
            self.check('after_old_m0_keys')
            if not positions:
                raise Blocked('EMPTY_BOUND_INPUT')
            result = pd.concat(keys, ignore_index=True)
            indices = np.asarray(positions, dtype=np.int64)
            for column in FEATURES:
                self.check('before_old_m0_column_' + column)
                pieces, offset, matched = [], 0, 0
                for batch in parquet.iter_batches(batch_size=CAPS['batch_rows'], columns=[column],
                                                  use_threads=False):
                    self.session.require_active()
                    end = offset + len(batch)
                    left, right = np.searchsorted(indices, [offset, end])
                    if right > left:
                        local = pa.array(indices[left:right] - offset, type=pa.int64())
                        pieces.append(batch.take(local).to_pandas()[column])
                        matched += right - left
                    offset = end
                if offset != scanned or matched != len(indices):
                    raise Blocked('M0_PROJECTED_ROW_PARTITION_CHANGED')
                result[column] = pd.concat(pieces, ignore_index=True)
                # The unchanged wall alarm also covers the bounded batch loops;
                # resource guards surround every single-column projection.
                self.check('after_old_m0_column_' + column)
            return result[list(columns)]

    def frame(self, name, columns, *, code_column=None):
        import pandas as pd
        self.session.require_active()
        if self.accepted is None:
            raise Blocked('FAST_BIND_BEFORE_PAYLOAD_REQUIRED')
        ref = self.config['inputs'][name]; path = Path(ref['path'])
        before = support.stat_identity(path)
        if before != self.accepted[name]['stat']:
            raise Blocked('PAYLOAD_CHANGED_AFTER_ADMISSION')
        parts = []; count = 0
        if path.suffix == '.parquet':
            import pyarrow as pa
            import pyarrow.dataset as ds
            pa.set_cpu_count(4); pa.set_io_thread_count(4)
            if name == 'old_m0':
                batches = (self.m0_symbol_frame(path, columns, code_column),)
            else:
                predicate = ds.field(code_column) == SYMBOL if code_column else None
                scanner = ds.dataset(path, format='parquet').scanner(columns=columns, filter=predicate,
                    batch_size=CAPS['batch_rows'], batch_readahead=1, fragment_readahead=1, use_threads=False)
                batches = (batch.to_pandas() for batch in scanner.to_batches())
        elif path.suffix == '.csv' and name == 'prefix':
            batches = pd.read_csv(path, usecols=columns, dtype={code_column: str}, float_precision='round_trip',
                chunksize=CAPS['batch_rows'])
        else:
            raise Blocked('ONLY_FILTERED_PARQUET_OR_BOUND_PREFIX_CSV')
        for part in batches:
            self.check('read_' + name); count += len(part)
            if count > CAPS['max_rows'] or (code_column and not part[code_column].eq(SYMBOL).all()):
                raise Blocked('SYMBOL_OR_STREAM_ROW_BOUND_NO_TRUNCATION')
            parts.append(part)
        if support.stat_identity(path) != before:
            raise Blocked('PAYLOAD_CHANGED_DURING_READ')
        if not parts:
            raise Blocked('EMPTY_BOUND_INPUT')
        return pd.concat(parts, ignore_index=True)


def original_modules(config):
    refs = config['sources']
    for ref in refs.values():
        if support.sha(ref['path']) != ref['sha256']:
            raise Blocked('ORIGINAL_SOURCE_OR_C_BINARY_DRIFT')
    if Path(support.__file__).resolve() != Path(refs['support']['path']).resolve():
        raise Blocked('SUPPORT_IMPORT_COLLISION')
    for name in ('engine', 'formula'):
        module_name = {'engine': 'tdx_ast_engine_injected', 'formula': 'formula_adapter'}[name]
        existing = sys.modules.get(module_name)
        if existing is not None and Path(existing.__file__).resolve() != Path(refs[name]['path']).resolve():
            raise Blocked('ORIGINAL_MODULE_IMPORT_COLLISION')
    parent = str(Path(refs['formula']['path']).parent)
    if parent not in sys.path: sys.path.insert(0, parent)
    engine = importlib.import_module('tdx_ast_engine_injected')
    formula = importlib.import_module('formula_adapter')
    if (Path(engine.__file__).resolve() != Path(refs['engine']['path']).resolve()
            or Path(formula.__file__).resolve() != Path(refs['formula']['path']).resolve()
            or Path(engine.fbc.__file__).resolve() != Path(refs['fast_builder']['path']).resolve()
            or Path(engine.fbc._lib._name).resolve() != Path(refs['c_binary']['path']).resolve()
            or formula.evaluate_tdx_formula_on_df is not engine.evaluate_tdx_formula_on_df):
        raise Blocked('ACTUAL_C_ENGINE_OR_FORMULA_LINKAGE_DRIFT')
    recipe = support._load(refs['recipe']['path'], 'm30_original_recipe')
    texts = {name: formula.exact_formula_source(Path(refs[name]['path']), refs[name]['sha256'])[0]
             for name in ('v192', 'v193')}
    return recipe, formula, texts


class Session:
    """The existing official guard claim/terminal protocol, one no-fit execution."""
    def __init__(self, guard, config):
        self.guard, self.config, self.claim = guard, config, None
        self.active = False; self.terminal_attempted = False

    def start(self):
        self.claim = self.guard.claim_no_fit(execution_id=self.config['execution_id'], arm='M30_FORMULA_ONLY',
            contract_sha256=self.config['contract']['sha256'], input_binding_sha256=support.digest(self.config))
        if self.claim is None: raise Blocked('DURABLE_OFFICIAL_CLAIM_NOT_RETURNED')
        self.active = True

    def require_active(self):
        if not self.active: raise Blocked('DURABLE_OFFICIAL_START_REQUIRED_BEFORE_PAYLOAD')

    def finish(self, succeeded, error=None):
        if self.claim is not None and not self.terminal_attempted:
            self.terminal_attempted = True; self.active = False
            self.guard.append_no_fit_terminal(self.claim, succeeded=succeeded, error_type=error)


def run_registered(config, activation, guard):
    """Sole actual entry; caller constructs the original official registry guard.

    Source/config inspection is allowed before START; payload reads occur only
    after claim_no_fit returns. Failed/completed output is immutable on reentry.
    """
    import fcntl
    require_activation(config, activation)
    if platform.system() != 'Darwin' or sys.executable != config['python']:
        raise Blocked('EXACT_MAC_RUNTIME_REQUIRED')
    if any(os.environ.get(key) != '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')):
        raise Blocked('FOUR_THREAD_RUNTIME_ENVIRONMENT_REQUIRED')
    guard_path = Path(inspect.getfile(type(guard))).resolve()
    if guard_path != Path(config['sources']['guard_adapter']['path']).resolve() or support.sha(guard_path) != config['sources']['guard_adapter']['sha256']:
        raise Blocked('OFFICIAL_EXECUTOR_GUARD_SOURCE_DRIFT')
    out = Path(config['output_root']); out.mkdir(exist_ok=False)
    with (out / '.owner.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / 'CANARY_RECEIPT.json'; started = time.monotonic()
        report = dict(status='INTENT_BEFORE_DURABLE_START', actual_execution_id=config['execution_id'],
            config_sha256=support.digest(config), wrapper_sha256=support.sha(__file__),
            contract_ref=config['contract'], policy_sha256=support.digest(config['policy']),
            source_refs=config['sources'], input_refs=config['inputs'],
            activation_review_receipt_sha256=activation['review_receipt_sha256'],
            scientific_identity=scientific_identity(config), planned_executor_owner=config['executor_owner'],
            actual_agent_identity=None, actual_agent_identity_reason='NOT_EXTERNALLY_VERIFIED',
            runtime=dict(python=sys.executable, pid=os.getpid(), workers=1, threads=4,
                parent_and_worker_same_process=True, parent_rss_cap=2**31, worker_rss_cap=2**31), **FLAGS)
        support.write_json(path, report)
        session = Session(guard, config)
        def check(stage):
            if time.monotonic() - started > CAPS['wall_seconds']: raise TimeoutError('M30_WALL_BOUND')
            snap = support.resource_snapshot(dict(output_root=str(out), resource_probe_source=config['sources']['resource_probe'], resource_caps=CAPS))
            if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > CAPS['output_bytes']:
                raise Blocked('OUTPUT_BYTE_BOUND')
            return snap
        try:
            session.start()
            with support.deadline(CAPS['wall_seconds']):
                support.checked_json(config['contract'])
                report['resource_before'] = check('before_admission')
                recipe, formula, texts = original_modules(config)
                adapter = BoundInputs(config, session, check)
                adapter.archive = lambda receipt: support.write_json(out/'AUTHENTICATION_RECEIPT.json', receipt)
                adapter.admit()
                report['actual_fast_bind_files'] = adapter.accepted
                report['authentication_receipt_ref'] = dict(path=str(out/'AUTHENTICATION_RECEIPT.json'),
                    sha256=support.sha(out/'AUTHENTICATION_RECEIPT.json')) if (out/'AUTHENTICATION_RECEIPT.json').is_file() else None
                maps = config['columns']
                frames = {name: adapter.frame(name, list(mapping.values()), code_column=mapping['code']) for name, mapping in maps.items()}
                old_numeric = recipe.normalize_price_panel(frames['old_qfq'], maps['old_qfq'], amount_multiplier=1.0)
                precision = dict(precision_evidence=adapter.precision_evidence) if 'precision_admission' in config else {}
                report['unit_admission'] = unit_admission(old_numeric, adapter.raw_amounts(), **precision)
                if report['unit_admission']['B_allowed']:
                    multipliers = dict(old_amount_multiplier=10000.0,
                        accepted_old_numeric_over_raw_CNY=report['unit_admission']['accepted_old_numeric_over_raw_CNY'])
                    old, full = compose(recipe, frames['prefix'], frames['old_qfq'], maps, multipliers)
                    cache = adapter.frame('old_m0', ['date', 'code', *FEATURES], code_column='code')
                    calendar = adapter.frame('old_calendar', ['date', 'calendar_index'])
                    report['result'] = run_frames(recipe, formula, old, full, cache, calendar, texts, check=check)
                else:
                    report['result'] = dict(decision='UNIT_ADMISSION_BLOCKED_B_NOT_COMPUTED', B_computed=False,
                        universe_parity='CANNOT_CONFIRM', **FLAGS)
                report['resource_after'] = check('after_comparison')
                succeeded = report['result']['decision'] in ('SINGLE_SYMBOL_EXACT', 'SINGLE_SYMBOL_DIFFERENT')
                report.update(status='COMPLETED' if succeeded else 'BLOCKED_NO_RETRY', elapsed_seconds=time.monotonic()-started)
                if len(support.canonical(report).encode()) > CAPS['output_bytes']:
                    raise Blocked('RECEIPT_BYTE_BOUND')
                support.write_json(path, report)
                session.finish(succeeded, None if succeeded else report['result']['decision'])
        except BaseException as exc:
            report.update(status='FAILED_OR_INTERRUPTED_NO_RETRY', error_type=type(exc).__name__, error=str(exc)[:1000], elapsed_seconds=time.monotonic()-started)
            authentication_path = out/'AUTHENTICATION_RECEIPT.json'
            if authentication_path.is_file():
                report['authentication_receipt_ref'] = dict(path=str(authentication_path), sha256=support.sha(authentication_path))
            try: support.write_json(path, report)
            except BaseException as error: exc.add_note('Failure archive unconfirmed: ' + repr(error))
            try: session.finish(False, type(exc).__name__)
            except BaseException as error: exc.add_note('Official terminal unconfirmed: ' + repr(error))
            raise
    return report


def make_guard(config, config_path):
    """Reuse official lock/append/fsync implementation; no registry fork."""
    refs = config['sources']
    for key in ('guard_adapter', 'registry_adapter', 'registry_builder', 'registry_validator'):
        if support.sha(refs[key]['path']) != refs[key]['sha256']:
            raise Blocked('OFFICIAL_REGISTRY_SOURCE_DRIFT')
    cached = sys.modules.get('execution_guard_core')
    if cached is not None and Path(cached.__file__).resolve() != Path(refs['guard_adapter']['path']).resolve():
        raise Blocked('OFFICIAL_GUARD_IMPORT_COLLISION')
    core = cached or support._load(refs['guard_adapter']['path'], 'execution_guard_core')
    adapter_module = support._load(refs['registry_adapter']['path'], 'm30_official_adapter')
    adapter = adapter_module.OfficialRegistryAdapter(registry_path=Path(config['registry_path']),
        contract_path=Path(config['contract']['path']), input_binding_path=Path(config_path),
        artifact_paths=(Path(config['output_root'])/'CANARY_RECEIPT.json',),
        builder_path=Path(refs['registry_builder']['path']), validator_path=Path(refs['registry_validator']['path']),
        event_profile=REGISTRY_PROFILE)
    return core.ExecutionGuard(adapter, experiment_id=EXPERIMENT, max_real_fit_debits=0)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--activation', required=True)
    args = parser.parse_args(argv)
    for path in (args.config, args.activation):
        if Path(path).stat().st_size > CAPS['metadata_bytes']: raise Blocked('CONTROL_METADATA_BYTE_BOUND')
    config = json.loads(Path(args.config).read_text())
    activation = json.loads(Path(args.activation).read_text())
    require_activation(config, activation)
    result = run_registered(config, activation, make_guard(config, args.config))
    print(support.canonical(dict(status=result['status'], execution_id=result['actual_execution_id'],
        decision=result['result']['decision'], universe_parity='CANNOT_CONFIRM')))
    return 0 if result['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
