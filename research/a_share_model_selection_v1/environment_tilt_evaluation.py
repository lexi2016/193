"""Three-stage environment-tilt orchestration; frozen math/producers are reused.

Import is inert. Real work needs independently activated, registered phases and
complete SHA-bound references. No base model fit/predict, source collection,
policy search, NAV, net-return, production or trading interface exists here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import environment_tilt_model as core
import environment_tilt_m0_oof as support


BATCH = support.BATCH
FLAGS = support.FLAGS
EARLY = ('2022H1','2022H2','2023H1','2023H2')
OUTER = (*core.COMPLETE_FOLDS, core.PARTIAL_FOLD)
SEGMENTS = (*EARLY, *OUTER)
PHASES = ('prepare','select','evaluate')
EXPERIMENT = 'A_SHARE_ENV_TILT_FROZEN_EVALUATION_20261021'
ROUTE = 'a-share-environment-tilt-frozen-evaluation-20261021'
POLICY = dict(core_policy_sha256=core.CONFIG_SHA256, segments=list(SEGMENTS),
              outer_folds=list(OUTER), label_horizon=2, base_fits=0,
              topk_provenance='FULL_BOUND_COMMON_POOL_BEFORE_LABELS',
              historical_maximum_state='DIRECTION_ONLY', confirmation_gate=None,
              stages=list(PHASES), **FLAGS)
POLICY_SHA256 = support.digest(POLICY)


class Blocked(ValueError):
    pass


def bounds(fold):
    if fold not in SEGMENTS:
        raise Blocked('UNFROZEN_FOLD')
    year = int(fold[:4]); half = int(fold[5])
    start = date(year, 1 if half == 1 else 7, 1)
    end = (date(year, 7, 1) if half == 1 else date(year + 1, 1, 1))
    return start, date(2026,10,16) if fold == core.PARTIAL_FOLD else end


def _day(value):
    import pandas as pd
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tz is not None or t != t.normalize():
        raise Blocked('INVALID_NORMALIZED_DATE')
    return t.date()


def jsonable(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [jsonable(v) for v in value]
    return value


def calendar_dates(calendar):
    import numpy as np
    import pandas as pd
    if tuple(calendar.columns) != ('date','calendar_index'):
        raise Blocked('EXACT_BOUND_CALENDAR_SCHEMA_REQUIRED')
    dates = tuple(_day(t) for t in calendar.date)
    if (not dates or tuple(sorted(set(dates))) != dates
            or not pd.api.types.is_integer_dtype(calendar.calendar_index.dtype)
            or pd.api.types.is_bool_dtype(calendar.calendar_index.dtype)
            or not np.array_equal(calendar.calendar_index.to_numpy(), np.arange(len(calendar)))):
        raise Blocked('COMPLETE_ORIGINAL_OBSERVED_CALENDAR_REQUIRED')
    return dates


def sessions(dates, start, end):
    result = tuple(t for t in dates if start <= t < end)
    if not result or dates[0] > start or dates[-1] < end:
        raise Blocked('NOMINAL_CALENDAR_COVERAGE_UNPROVEN')
    return result


def restore_model_time(sidecar, dates, *, model_source_sha256, calendar_sha256):
    """Original actual-used training keys + original fixed-h2 calendar mapping.

    An explicit UPPER_BOUND may be admitted only by its separately bound proof;
    it is never relabeled as the actual maximum. Purge cutoffs are unsupported.
    """
    identity = sidecar.get('identity', {})
    if (sidecar.get('status') != 'COMPLETED' or identity.get('arm') != 'M0'
            or identity.get('horizon') != 2
            or identity.get('code_sha256', {}).get('model_selection') != model_source_sha256):
        raise Blocked('ORIGINAL_MODEL_RECIPE_IDENTITY_REQUIRED')
    data = identity.get('data_identity', {})
    keys = data.get('training_keys', {})
    if (data.get('schema_version') != 'A_SHARE_CANARY_CELL_DATA_IDENTITY_V1'
            or tuple(data.get('feature_columns',())) != support.FEATURES
            or keys.get('schema_version') != 'A_SHARE_DATE_CODE_SORTED_KEYS_V1'
            or not keys.get('ordered_keys_sha256') or not keys.get('sorted_keyset_sha256')
            or keys.get('row_count') != keys.get('unique_key_count')
            or not keys.get('row_count') or not data.get('training_label_vector', {}).get('sha256')):
        raise Blocked('ACTUAL_USED_TRAIN_KEY_OR_LABEL_IDENTITY_UNPROVEN')
    last = date.fromisoformat(keys['date_max'])
    try:
        position = dates.index(last)
        value = dates[position + 2]
    except (ValueError, IndexError):
        raise Blocked('TRAIN_LABEL_END_CALENDAR_ANCHOR_MISSING') from None
    return dict(value=value.isoformat(), kind='EXACT_MAX',
                method='ACTUAL_USED_TRAIN_KEY_MAX_PLUS_ORIGINAL_H2_OBSERVED_SESSIONS',
                training_key_semantics='ORIGINAL_PREPARED_POST_PURGE_ACTUAL_TRAIN_ROWS',
                training_keyset_sha256=keys['sorted_keyset_sha256'],
                training_label_vector_sha256=data['training_label_vector']['sha256'],
                model_source_sha256=model_source_sha256, calendar_sha256=calendar_sha256)


def validate_time_proof(proof, first_score_date):
    if (proof.get('kind') not in ('EXACT_MAX','UPPER_BOUND')
            or not proof.get('evidence_sha256') or not proof.get('method')
            or 'PURGE_CUTOFF' in proof['method']):
        raise Blocked('MODEL_TRAIN_TIME_PROOF_REQUIRED_NO_PURGE_PROXY')
    value = date.fromisoformat(proof['value'])
    if not value < first_score_date:
        raise Blocked('SAME_MODEL_OOT_TIME_GUARD_FAILED')
    return value


def prepare_segment(recipe, prior, calendar, breadth, roster, lookup, predictions, provenance, limits):
    """Bounded full-pool validation, label-free Top50, original prior projection."""
    import numpy as np
    import pandas as pd
    dates = calendar_dates(calendar)
    fold = provenance['fold']; start,end = bounds(fold)
    expected = sessions(dates,start,end)
    if (len(predictions) > limits['max_segment_rows'] or len(expected) > limits['max_segment_days']
            or not {'date','code','prediction'}.issubset(predictions.columns)):
        raise Blocked('SCORE_SEGMENT_SCHEMA_OR_RESOURCE_BOUND')
    frame = predictions[['date','code','prediction']].copy()
    frame['date'] = pd.to_datetime(frame.date)
    actual_dates = tuple(sorted({_day(t) for t in frame.date}))
    missing_dates = tuple(t for t in expected if t not in set(actual_dates))
    if (frame[['date','code']].isna().any().any() or frame.duplicated(['date','code']).any()
            or not frame.code.map(lambda c:isinstance(c,str) and bool(c) and c.strip()==c).all()
            or not np.isfinite(frame.prediction.to_numpy(dtype=float)).all()
            or not set(actual_dates).issubset(expected)):
        raise Blocked('FULL_SCORE_KEY_FINITE_OR_DATE_COVERAGE_CONFLICT')
    identity = recipe.key_frame_identity(frame)
    if identity != provenance['full_pool_test_keys']:
        raise Blocked('FULL_ORIGINAL_COMMON_POOL_KEY_IDENTITY_MISMATCH')
    zero = provenance.get('zero_pool_proof')
    if missing_dates:
        if (not zero or zero.get('zero_pool_dates') != [t.isoformat() for t in missing_dates]
                or zero.get('nominal_calendar_days') != len(expected)
                or zero.get('nonempty_pool_days') != len(actual_dates)):
            raise Blocked('EXACT_VERSION_BOUND_ORIGINAL_ZERO_POOL_PROOF_REQUIRED')
    elif zero and zero.get('zero_pool_dates'):
        raise Blocked('PROVEN_EMPTY_POOL_HAS_UNEXPECTED_SCORE_ROWS')
    train_end = validate_time_proof(provenance['model_train_time'], expected[0])
    if len(frame.groupby('date').size()) and frame.groupby('date').size().max() > limits['max_rows_per_date']:
        raise Blocked('PER_DATE_POOL_RESOURCE_BOUND')
    # Only this narrow derived Top50 projection is converted to Python rows.
    ordered = frame.sort_values(['date','prediction','code'], ascending=[True,False,True],kind='mergesort')
    top = ordered.groupby('date',sort=False).head(50).reset_index(drop=True)
    projection = (prior.project_prior(top[['date','code']], roster, calendar, lookup) if len(top)
                  else pd.DataFrame(columns=['date','code',*prior.PRIOR_COLUMNS]).astype({'date':'datetime64[ns]'}))
    if not projection[['date','code']].equals(top[['date','code']]):
        raise Blocked('ORIGINAL_PRIOR_KEY_ORDER_MISMATCH')
    top = top.merge(projection,on=['date','code'],how='left',validate='one_to_one',sort=False)
    if roster.code.duplicated().any():
        raise Blocked('AMBIGUOUS_INDUSTRY_ROSTER')
    industry = dict(zip(roster.code,roster.industry_key))
    if breadth.date.duplicated().any():
        raise Blocked('AMBIGUOUS_ORIGINAL_BREADTH_DATE')
    breadth_map = dict(zip((_day(t) for t in breadth.date),breadth.market_advancer_share))
    full_counts = frame.groupby('date').size()
    output = []
    groups = { _day(t):group for t,group in top.groupby('date',sort=True) }
    for day in expected:
        group = groups.get(day,top.iloc[:0]);pos = dates.index(day)
        if pos + 1 >= len(dates):
            raise Blocked('SCORE_TPLUS1_CALENDAR_ANCHOR_MISSING')
        h2 = dates[pos+2] if pos+2<len(dates) else None
        if h2 is None and fold != core.PARTIAL_FOLD:
            raise Blocked('TRAINABLE_SCORE_H2_CALENDAR_ANCHOR_MISSING')
        raw_b = breadth_map.get(day, float('nan'))
        b = None if pd.isna(raw_b) else float(raw_b)
        if b is not None and (not math.isfinite(b) or not 0 <= b <= 1):
            raise Blocked('INVALID_ORIGINAL_BREADTH')
        rows = []
        industries = {}
        for row in group.itertuples(index=False):
            p = float(getattr(row,prior.PRIOR_COLUMNS[0])); missing = float(getattr(row,prior.PRIOR_COLUMNS[-1]))
            if missing not in (0.,1.) or not math.isfinite(p):
                raise Blocked('INVALID_ORIGINAL_PRIOR_PROJECTION')
            rows.append(core.ScoreRow(row.code,float(row.prediction),p,'missing' if missing else 'observed'))
            key = industry.get(row.code)
            industries[row.code] = 'UNKNOWN' if pd.isna(key) else str(key)
        scored = core.ScoreDay(day,dates[pos+1],h2,tuple(rows),b)
        # Validate the immutable core boundary, without reading any outcomes.
        core.select(scored,.25)
        output.append(dict(day=jsonable(asdict(scored)), segment=fold,
            model_train_time=provenance['model_train_time'], model_sha256=provenance['model_sha256'],
            input_version=provenance['input_version'], industries=industries,
            full_pool_rows=int(full_counts.get(pd.Timestamp(day),0)),
            original_pool_state='PROVEN_EMPTY' if not len(group) else 'NONEMPTY',
            zero_pool_proof=zero, h2_anchor_state='UNOBSERVED_TARGET_INVALID_PAIR' if h2 is None else 'OBSERVED',
            full_pool_key_identity_sha256=support.digest(identity),
            topk_provenance='STABLE_DESC_SCORE_ASC_CODE_FROM_AUTHENTICATED_FULL_COMMON_POOL'))
    return output


def restore_day(record):
    d = record['day']
    return core.ScoreDay(date.fromisoformat(d['signal_date']),date.fromisoformat(d['next_session']),
                        date.fromisoformat(d['h2_session']) if d['h2_session'] is not None else None,
                        tuple(core.ScoreRow(**r) for r in d['rows']),d['breadth'])


def restore_pair(record):
    def arm(d):
        return core.Selection(date.fromisoformat(d['signal_date']),
            date.fromisoformat(d['h2_session']) if d['h2_session'] is not None else None,
            d['lambda_value'],tuple(core.RankedRow(**r) for r in d['top_k']),
            tuple(core.RankedRow(**r) for r in d['selected']),d['scale'],d['gate'],d['activation'])
    pair = core.FrozenPair(arm(record['baseline']),arm(record['candidate']))
    if pair.baseline.sha256 != record['baseline_sha256'] or pair.candidate.sha256 != record['candidate_sha256']:
        raise Blocked('FROZEN_SELECTION_HASH_MISMATCH')
    return pair


def fit_and_freeze(records, dates, calendar_sha, score_manifest_sha, outcome_provider, sink, checkpoint):
    """Chronological folds: persist each list BEFORE later folds consume its labels."""
    by_date = {}
    for record in records:
        day = restore_day(record)
        if day.signal_date in by_date:
            raise Blocked('OVERLAPPING_SCORE_SEGMENTS')
        by_date[day.signal_date] = record
    all_pairs, coefficients = [], {}
    for fold in OUTER:
        start,end = bounds(fold); train_start = start.replace(year=start.year-2)
        training_dates = sessions(dates,train_start,start)
        outer_dates = sessions(dates,start,end)
        if any(t not in by_date for t in (*training_dates,*outer_dates)):
            raise Blocked('COMPLETE_TWO_YEAR_AND_SIX_OUTER_COVERAGE_REQUIRED')
        oot, requested = [], []
        for t in training_dates:
            record = by_date[t]; day = restore_day(record)
            bound = validate_time_proof(record['model_train_time'],t)
            oot.append(core.OOTDay(day,bound,record['model_sha256']))
            requested.extend((t,r.code) for r in day.rows)
        request = dict(role='LAMBDA_TRAINING_PAST_ONLY',fold=fold,cutoff=start.isoformat(),
                       keys_sha256=support.digest(jsonable(requested)),slots=len(requested))
        sink.json('TRAIN_REQUEST_'+fold+'.json',request)
        with checkpoint(fold):
            outcomes = outcome_provider(requested,role='training',cutoff=start)
            result = core.fit_lambda(start,core.CalendarBinding(train_start,start,training_dates,
                calendar_sha,score_manifest_sha),oot,outcomes)
            pairs = []
            for t in outer_dates:
                record = by_date[t]; day = restore_day(record)
                pair = core.freeze_pair(day,result.lambda_value)
                frozen = dict(baseline=jsonable(asdict(pair.baseline)),candidate=jsonable(asdict(pair.candidate)),
                    baseline_sha256=pair.baseline.sha256,candidate_sha256=pair.candidate.sha256,
                    industries=record['industries'], input_version=record['input_version'],
                    model_sha256=record['model_sha256'], source_segment=record['segment'],
                    prior_by_code={r.code:r.prior for r in day.rows}, breadth=day.breadth)
                pairs.append(frozen)
            # Durable fold lists and coefficient precede next fold's training truth.
            sink.jsonlines('SELECTIONS_'+fold+'.jsonl',pairs)
            coefficients[fold] = jsonable(asdict(result))
            sink.json('LAMBDA_'+fold+'.json',dict(fit=coefficients[fold],request=request))
            all_pairs.extend(pairs)
    selection_ref = sink.jsonlines('SELECTIONS.jsonl',all_pairs)
    coefficients_ref = sink.json('LAMBDAS.json',coefficients)
    return dict(selection_ref=selection_ref,coefficients_ref=coefficients_ref,
                outer_dates=len(all_pairs),scalar_fits=len(coefficients),base_fits=0)


def selected_keys(pairs):
    return sorted({(pair.baseline.signal_date,r.code) for pair in pairs
                   for arm in (pair.baseline,pair.candidate) for r in arm.selected})


def diagnostics(records, pairs, evaluation, outcomes):
    labels = {(o.signal_date,o.code):o.value for o in outcomes}
    rows = []; turnover = []; industry_contribution = defaultdict(float)
    for record,pair,paired in zip(records,pairs,evaluation.dates):
        d = paired.signal_date; industries = record['industries']
        arms = {}
        for name,arm in [('M0',pair.baseline),('TILT',pair.candidate)]:
            counts = Counter(industries.get(r.code,'UNKNOWN') for r in arm.selected)
            weights = {k:v/len(arm.selected) for k,v in counts.items()} if arm.selected else {}
            arms[name] = dict(industry_weights=weights,hhi=sum(v*v for v in weights.values()),
                known_industry_slots=sum(v for k,v in counts.items() if 'UNKNOWN' not in k.upper()),
                slots=len(arm.selected))
        swap = None; decomposition = {}
        if paired.paired_complete:
            swap = (math.fsum(labels[(d,c)] for c in paired.entrants)-math.fsum(labels[(d,c)] for c in paired.exits))/10
            if not math.isclose(swap,paired.delta,rel_tol=1e-10,abs_tol=1e-12):
                raise Blocked('TEN_SLOT_SWAP_IDENTITY_FAILED')
            for sign,arm in [(-1,pair.baseline),(1,pair.candidate)]:
                for r in arm.selected:
                    key = industries.get(r.code,'UNKNOWN')
                    decomposition[key] = decomposition.get(key,0.)+sign*labels[(d,r.code)]/10
            if paired.fold in core.COMPLETE_FOLDS:
                for key,value in decomposition.items():industry_contribution[key]+=value
        base_ranks = {r.code:r.base_rank for r in pair.baseline.top_k}
        additions = [r.score-r.base_score for r in pair.candidate.top_k]
        rows.append(dict(date=d.isoformat(),fold=paired.fold,paired_complete=paired.paired_complete,
            retained=sorted({r.code for r in pair.baseline.selected}&{r.code for r in pair.candidate.selected}),
            entrants={c:base_ranks[c] for c in paired.entrants},exits={c:base_ranks[c] for c in paired.exits},
            swap_count=len(paired.entrants),swap_delta=swap,industry=arms,
            industry_gross_delta_contribution=decomposition,
            positive_prior_slots=sum(v is not None and v>0 for v in record['prior_by_code'].values()),
            breadth=record['breadth'],gate=pair.candidate.gate,scale=pair.candidate.scale,
            activation=pair.candidate.activation,
            score_addition=dict(min=min(additions) if additions else None,max=max(additions) if additions else None,
                                mean=statistics.mean(additions) if additions else None)))
        if paired.fold in core.COMPLETE_FOLDS:turnover.append(len(paired.entrants))
    complete = [r for r in evaluation.dates if r.paired_complete and r.fold in core.COMPLETE_FOLDS]
    total = math.fsum(r.delta for r in complete)
    fold_totals = {f:math.fsum(r.delta for r in complete if r.fold==f) for f in core.COMPLETE_FOLDS}
    less_best = sorted((r.delta for r in complete))[:-5]
    return dict(dates=rows,full_fold_total_delta=fold_totals,
        fold_fraction_of_total={f:v/total for f,v in fold_totals.items()} if total else None,
        fold_fraction_reason=None if total else 'CANNOT_CONFIRM_ZERO_TOTAL',
        industry_gross_delta_sum=dict(industry_contribution),
        average_swaps=statistics.mean(turnover) if turnover else None,
        swap_quantiles={str(q):_quantile(turnover,q) for q in (0,.1,.5,.9,1)},
        delta_cvar5=core.fractional_cvar5([r.delta for r in complete]) if complete else None,
        mean_delta_without_best5_diagnostic=statistics.mean(less_best) if less_best else None,
        worst_candidate_day=min(complete,key=lambda r:r.candidate_return).signal_date.isoformat() if complete else None,
        optional_risk=dict(beta='CANNOT_CONFIRM_NO_BOUND_ORIGINAL_FIELD',size='CANNOT_CONFIRM_NO_BOUND_ORIGINAL_FIELD',
                           liquidity='CANNOT_CONFIRM_NO_BOUND_ORIGINAL_FIELD',same_industry_excess='CANNOT_CONFIRM_NO_BOUND_SAME_DEFINITION_RETURN'),
        interpretation='GROSS_OVERLAPPING_H2_ONLY_NO_NAV_NET_MAXDD; retrospective classification; no risk diagnostic is an extra gate')


def _quantile(values,q):
    if not values:return None
    s=sorted(values);p=q*(len(s)-1);i=math.floor(p)
    return s[i] if i==len(s)-1 else s[i]+(p-i)*(s[i+1]-s[i])


def evaluate_frozen(records, dates, calendar_sha, score_manifest_sha, outcome_provider):
    pairs = tuple(restore_pair(record) for record in records)
    expected = sessions(dates,date(2024,1,1),date(2026,10,16))
    if tuple(pair.baseline.signal_date for pair in pairs) != expected:
        raise Blocked('ALL_SIX_FROZEN_OUTER_LISTS_REQUIRED_IN_CALENDAR_ORDER')
    requested = selected_keys(pairs)
    outcomes = outcome_provider(requested,role='outer',cutoff=None)
    full = core.CalendarBinding(date(2024,1,1),date(2026,7,1),
        sessions(dates,date(2024,1,1),date(2026,7,1)),calendar_sha,score_manifest_sha)
    partial = core.CalendarBinding(date(2026,7,1),date(2026,10,16),
        sessions(dates,date(2026,7,1),date(2026,10,16)),calendar_sha,score_manifest_sha)
    result = core.evaluate(full,pairs,outcomes,partial)
    return dict(evaluation=jsonable(asdict(result)),diagnostics=diagnostics(records,pairs,result,outcomes),
                coverage=dict(nominal_outer_dates=len(expected),
                    paired_complete_dates=sum(d.paired_complete for d in result.dates),
                    last_evaluable_date=max((d.signal_date.isoformat() for d in result.dates if d.paired_complete),default=None),
                    unobserved_h2_dates=[p.baseline.signal_date.isoformat() for p in pairs if p.baseline.h2_session is None],
                    invalid_dates=[dict(date=d.signal_date.isoformat(),fold=d.fold,reason=d.reason)
                                   for d in result.dates if not d.paired_complete]),
                outer_requested_slots=len(requested),state=result.decision,
                historical_only=True,sealed_confirmation='CANNOT_CONFIRM_NOT_FROZEN',**FLAGS)


def _reference(ref):
    if (not isinstance(ref,dict) or not isinstance(ref.get('path'),str)
            or not Path(ref['path']).is_absolute() or not isinstance(ref.get('sha256'),str)
            or len(ref['sha256'])!=64 or any(c not in '0123456789abcdef' for c in ref['sha256'])):
        raise Blocked('FINAL_ABSOLUTE_PATH_AND_SHA_REQUIRED_NO_LOGICAL_OR_NULL_REFERENCE')


def scientific_identity(config):
    return support.digest({k:config[k] for k in ('policy','sources','inputs','segments',
        'resource_probe_source','runtime_versions','passport')})


def validate_config(config):
    if config.get('state')!='FROZEN_READY_FOR_ACTIVATION' or config.get('policy')!=POLICY:
        raise Blocked('DRAFT_OR_CHANGED_POLICY_NOT_EXECUTABLE')
    if config.get('policy_sha256')!=POLICY_SHA256 or any(config.get(k)!=v for k,v in FLAGS.items()):
        raise Blocked('FROZEN_POLICY_AND_RESEARCH_FLAGS_REQUIRED')
    specs=config.get('segments',[])
    if tuple(spec.get('fold') for spec in specs)!=SEGMENTS:
        raise Blocked('EXACT_EARLY_FOUR_AND_OUTER_SIX_REQUIRED_NO_MISSING_FOLD')
    for spec in specs:
        version=spec.get('input_version')
        if (spec.get('format') not in ('original_checkpoint','normalized_oof')
                or not isinstance(version,str) or not version or version.strip()!=version):
            raise Blocked('SCORE_SEGMENT_FORMAT_AND_INPUT_VERSION_REQUIRED')
        _reference(spec['artifact']);_reference(spec['metadata']);_reference(spec['accepted_review'])
        if spec.get('zero_pool') is not None:
            if set(spec['zero_pool'])!={'proof','accepted_review'}:raise Blocked('EXACT_ZERO_POOL_PROOF_AND_REVIEW_REFS_REQUIRED')
            for ref in spec['zero_pool'].values():_reference(ref)
    for ref in config['sources'].values():_reference(ref)
    for ref in config['inputs'].values():_reference(ref)
    _reference(config['contract']);_reference(config['resource_probe_source']);_reference(config['accepted_core_review'])
    for ref in config['registry_dependencies'].values():_reference(ref)
    if config['passport'] is not None:_reference(config['passport'])
    if set(config['sources'])!={'recipe','prior','core','support'}:
        raise Blocked('EXACT_EXISTING_PRODUCER_DEPENDENCIES_REQUIRED')
    if set(config['inputs'])!={'panel','calendar','roster','industry_year','month_year'}:
        raise Blocked('BOUND_ORIGINAL_PRICE_CALENDAR_PRIOR_DEPENDENCIES_REQUIRED')
    out=Path(config['output_root'])
    if out.parent.resolve()!=BATCH.resolve() or not out.name.startswith('M17_') or out.is_symlink():
        raise Blocked('OWNED_M17_BATCH_OUTPUT_REQUIRED')
    caps=config['caps']
    names=('rss_bytes','min_available_bytes','min_disk_bytes','max_payload_file_bytes','max_segment_rows',
           'max_panel_rows','max_aux_rows','max_rows_per_date','max_segment_days','max_prepared_days',
           'max_artifact_bytes','max_json_record_bytes','prepare_wall_seconds','select_wall_seconds','evaluate_wall_seconds')
    if caps.get('threads')!=4 or any(type(caps.get(n)) is not int or caps[n]<=0 for n in names):
        raise Blocked('FINITE_POSITIVE_FOUR_THREAD_RESOURCE_LIMITS_REQUIRED')
    if set(config['execution_ids'])!=set(PHASES) or len(set(config['execution_ids'].values()))!=3:
        raise Blocked('DISTINCT_STAGE_EXECUTION_IDENTITIES_REQUIRED')


def checked_control(ref):
    _reference(ref)
    return support.checked_json(ref)


def require_activation(config,config_path,activation,phase):
    validate_config(config)
    required=dict(phase=phase,decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',
        execution_id=config['execution_ids'][phase],config_sha256=support.sha(config_path),
        wrapper_sha256=support.sha(__file__),policy_sha256=POLICY_SHA256,
        scientific_identity=scientific_identity(config),contract_sha256=config['contract']['sha256'])
    review_sha=activation.get('review_receipt_sha256')
    if (any(activation.get(k)!=v for k,v in required.items()) or not activation.get('reviewer')
            or not isinstance(review_sha,str) or len(review_sha)!=64
            or any(c not in '0123456789abcdef' for c in review_sha)):
        raise Blocked('PHASE_SPECIFIC_INDEPENDENT_ACTIVATION_REQUIRED')
    contract=checked_control(config['contract'])
    if (contract.get('state')!='FROZEN_READY_FOR_ACTIVATION'
            or contract.get('policy_sha256')!=POLICY_SHA256
            or contract.get('wrapper_sha256')!=support.sha(__file__)
            or contract.get('scientific_identity')!=scientific_identity(config)):
        raise Blocked('FINAL_FROZEN_CONTRACT_AND_COMPLETE_INPUT_IDENTITY_REQUIRED')
    if phase!='prepare':
        previous='prepare' if phase=='select' else 'select'
        ref=config.get('accepted_'+previous)
        if not ref or activation.get('upstream_receipt_sha256')!=ref.get('sha256'):
            raise Blocked('INDEPENDENTLY_ACCEPTED_PREVIOUS_STAGE_REQUIRED')
        receipt=checked_control(ref)
        if (receipt.get('status')!='COMPLETED' or receipt.get('phase')!=previous
                or receipt.get('scientific_identity')!=scientific_identity(config)
                or receipt.get('wrapper_sha256')!=support.sha(__file__)):
            raise Blocked('PREVIOUS_STAGE_IDENTITY_DRIFT')
        if phase=='evaluate' and receipt.get('accepted_prepare_sha256')!=config.get('accepted_prepare',{}).get('sha256'):
            raise Blocked('SELECTED_LISTS_BOUND_TO_DIFFERENT_PREPARATION')


class ArtifactSink:
    """New owned outputs only, bounded records, durable receipt before terminal."""
    def __init__(self,root,caps):self.root,self.caps=Path(root),caps

    def _path(self,name):
        if Path(name).name!=name or name.startswith('.'):
            raise Blocked('FLAT_OWNED_ARTIFACT_PATH_REQUIRED')
        path=self.root/name
        if path.exists():raise Blocked('EXISTING_ARTIFACT_NO_OVERWRITE_OR_RETRY')
        return path

    def json(self,name,value):
        path=self._path(name);payload=support.canonical(jsonable(value))
        if len(payload.encode())>self.caps['max_artifact_bytes']:raise Blocked('ARTIFACT_SIZE_BOUND')
        support.write_json(path,jsonable(value))
        return dict(path=str(path),sha256=support.sha(path))

    def jsonlines(self,name,records):
        import os
        path=self._path(name);count=0
        with path.open('x') as handle:
            for record in records:
                line=support.canonical(jsonable(record))+'\n'
                if len(line.encode())>self.caps['max_json_record_bytes']:raise Blocked('JSON_RECORD_SIZE_BOUND')
                handle.write(line);count+=1
                if handle.tell()>self.caps['max_artifact_bytes']:raise Blocked('ARTIFACT_SIZE_BOUND')
            handle.flush();os.fsync(handle.fileno())
        # JSON manifest fsyncs this directory through the shared durable writer.
        return dict(path=str(path),sha256=support.sha(path),records=count)


class BoundFiles:
    """One registered adapter. Bound data never enters via metadata-only calls."""
    def __init__(self,config,session,accepted_bound_files=None):
        self.config,self.session=config,session
        self.bound={};self.accepted=accepted_bound_files
        self.recipe=self.prior=None
        self.checkpoint_unpacks=0

    def authenticate(self,ref):
        self.session.require_active();_reference(ref)
        path=Path(ref['path']);before=support.stat_identity(path)
        if before['bytes']>self.config['caps']['max_payload_file_bytes']:raise Blocked('INPUT_FILE_BYTE_BOUND')
        old=self.bound.get(str(path))
        if old is not None:
            if old['sha256']!=ref['sha256'] or old['stat']!=before:raise Blocked('INPUT_DRIFT_DURING_STAGE')
            return before
        if self.accepted is not None:
            old=self.accepted.get(str(path))
            if not old or old['sha256']!=ref['sha256'] or old['stat']!=before:
                raise Blocked('FAST_BIND_IDENTITY_OR_STAT_DRIFT_NEW_PREPARE_REQUIRED')
        elif support.sha(path)!=ref['sha256']:
            raise Blocked('BOUND_PAYLOAD_SHA_DRIFT')
        if support.stat_identity(path)!=before:raise Blocked('INPUT_DRIFT_DURING_AUTHENTICATION')
        self.bound[str(path)]=dict(sha256=ref['sha256'],stat=before)
        return before

    def metadata(self,ref):
        before=self.authenticate(ref)
        if Path(ref['path']).stat().st_size>self.config['caps']['max_json_record_bytes']:
            raise Blocked('METADATA_SIZE_BOUND')
        result=json.loads(Path(ref['path']).read_text())
        if support.stat_identity(ref['path'])!=before:raise Blocked('METADATA_CHANGED_DURING_READ')
        return result

    def modules(self):
        if self.recipe is not None:return self.recipe,self.prior
        for name,ref in self.config['sources'].items():
            # Source text is authenticated in every phase, never a data recertification.
            if support.sha(ref['path'])!=ref['sha256']:raise Blocked('SOURCE_DEPENDENCY_SHA_DRIFT')
            loaded={'core':core,'support':support}.get(name)
            if loaded and Path(loaded.__file__).resolve()!=Path(ref['path']).resolve():raise Blocked('SOURCE_IMPORT_COLLISION')
        self.recipe=support._load(self.config['sources']['recipe']['path'],'m17_original_recipe')
        self.prior=support._load(self.config['sources']['prior']['path'],'m17_original_prior')
        if tuple(self.recipe.CORE_FEATURES)!=support.FEATURES or self.prior.KAPPA!=100:
            raise Blocked('ORIGINAL_RECIPE_OR_PRIOR_POLICY_DRIFT')
        return self.recipe,self.prior

    def frame(self,ref,*,columns=None,filters=None,max_rows=None):
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        path=Path(ref['path'])
        if path.suffix=='.csv' and filters is not None:
            raise Blocked('FILTERED_CSV_UNSUPPORTED_USE_BOUND_PARQUET')
        before=self.authenticate(ref)
        pa.set_cpu_count(4);pa.set_io_thread_count(4)
        limit=max_rows or self.config['caps']['max_aux_rows']
        if path.suffix=='.parquet':
            if pq.ParquetFile(path).metadata.num_rows>limit:raise Blocked('PARQUET_ROW_RESOURCE_BOUND')
            result=pd.read_parquet(path,columns=columns,filters=filters)
        elif path.suffix=='.csv':
            parts=[];count=0
            for chunk in pd.read_csv(path,usecols=columns,dtype={'code':str},float_precision='round_trip',
                                     chunksize=min(100000,limit)):
                count+=len(chunk)
                if count>limit:raise Blocked('CSV_ROW_RESOURCE_BOUND')
                parts.append(chunk)
            result=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=columns)
        else:raise Blocked('EXPLICIT_SUPPORTED_FRAME_FORMAT_REQUIRED')
        if len(result)>limit or support.stat_identity(path)!=before:raise Blocked('FRAME_RESOURCE_OR_READ_IDENTITY_DRIFT')
        return result

    def context(self):
        import pandas as pd
        recipe,prior=self.modules();inputs=self.config['inputs']
        calendar=self.frame(inputs['calendar'],max_rows=self.config['caps']['max_prepared_days']+1000)
        calendar['date']=pd.to_datetime(calendar.date);calendar_dates(calendar)
        panel=self.frame(inputs['panel'],columns=['date','code','close','amount'],max_rows=self.config['caps']['max_panel_rows'])
        panel['date']=pd.to_datetime(panel.date)
        if panel.duplicated(['date','code']).any():raise Blocked('AMBIGUOUS_ORIGINAL_PANEL_KEYS')
        breadth=recipe._calendar_daily_market(panel,calendar)
        del panel
        roster=self.frame(inputs['roster']);annual=self.frame(inputs['industry_year']);monthly=self.frame(inputs['month_year'])
        lookup=prior.prior_lookup(annual,monthly,range(2022,2027))
        return calendar,breadth,roster,lookup

    def score_segment(self,spec,calendar):
        import pandas as pd
        import joblib
        recipe,_=self.modules();dates=calendar_dates(calendar)
        review=self.metadata(spec['accepted_review'])
        if (review.get('decision') not in ('PASS','PASS_BEHAVIOR_VERIFIED','PASS_INDEPENDENT_RESULT_READBACK')
                or review.get('score_binding')!=dict(fold=spec['fold'],artifact=spec['artifact'],
                    metadata=spec['metadata'],input_version=spec['input_version'])):
            raise Blocked('ACCEPTED_SOURCE_SCORE_REVIEW_REQUIRED')
        metadata=self.metadata(spec['metadata'])
        if spec['format']=='original_checkpoint':
            identity=metadata.get('identity',{})
            if (metadata.get('artifact_sha256')!=spec['artifact']['sha256']
                    or identity.get('fold')!=spec['fold'] or identity.get('stage')!='M0_FORMAL'):
                raise Blocked('ORIGINAL_CHECKPOINT_SIDECAR_MISMATCH')
            proof=restore_model_time(metadata,dates,model_source_sha256=self.config['sources']['recipe']['sha256'],
                                     calendar_sha256=self.config['inputs']['calendar']['sha256'])
            proof['evidence_sha256']=spec['metadata']['sha256']
            before=self.authenticate(spec['artifact'])
            payload=joblib.load(spec['artifact']['path']);self.checkpoint_unpacks+=1
            frame=payload['predictions'].copy();del payload
            if support.stat_identity(spec['artifact']['path'])!=before:raise Blocked('CHECKPOINT_CHANGED_DURING_READ')
            full_keys=identity['data_identity']['test_keys'];model_sha=spec['artifact']['sha256']
        else:
            if (metadata.get('schema')!='M17_NORMALIZED_OOF_BINDING_V1'
                    or metadata.get('fold')!=spec['fold'] or metadata.get('scores_ref')!=spec['artifact']
                    or metadata.get('model_recipe_sha256')!=self.config['sources']['recipe']['sha256']
                    or metadata.get('pool_policy')!='ORIGINAL_FIVE_ARM_COMMON_FINITE'
                    or metadata.get('input_version')!=spec['input_version']):
                raise Blocked('NORMALIZED_OOF_PROVENANCE_INCOMPLETE')
            evidence=self.metadata(metadata['fit_receipt_ref'])
            if evidence.get('status')!='SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION':
                raise Blocked('ACTUAL_COMPLETED_ORIGINAL_CELL_RECEIPT_REQUIRED')
            actual=evidence.get('proof',{})
            fit_config=self.metadata(metadata['fit_config_ref'])
            cell=fit_config.get('policy',{}).get('cell',{})
            if (fit_config.get('recipe_sources',{}).get('model_selection',{}).get('sha256')
                    !=self.config['sources']['recipe']['sha256']
                    or cell.get('arm')!='M0' or cell.get('horizon')!=2 or cell.get('fold')!=spec['fold']
                    or evidence.get('input_binding_identity')!=metadata.get('fit_input_binding_identity')
                    or not metadata.get('fit_input_binding_identity')):
                raise Blocked('ORIGINAL_FIT_CONFIG_CELL_SOURCE_OR_INPUT_BINDING_CONFLICT')
            if tuple(actual.get('feature_columns',()))!=support.FEATURES or actual.get('params')!=support.PARAMS:
                raise Blocked('ORIGINAL_36_FEATURE_AND_MODEL_RECIPE_PROOF_REQUIRED')
            synthetic_sidecar=dict(status='COMPLETED',identity=dict(arm='M0',horizon=2,
                code_sha256={'model_selection':metadata['model_recipe_sha256']},data_identity=actual['data_identity']))
            proof=restore_model_time(synthetic_sidecar,dates,model_source_sha256=metadata['model_recipe_sha256'],
                                     calendar_sha256=self.config['inputs']['calendar']['sha256'])
            if proof['value']!=actual.get('max_model_train_label_end'):
                raise Blocked('ACTUAL_TRAIN_LABEL_MAX_CONFLICT')
            proof['evidence_sha256']=metadata['fit_receipt_ref']['sha256']
            if evidence.get('scores',{}).get('sha256')!=spec['artifact']['sha256']:
                raise Blocked('SCORE_FILE_NOT_LINKED_TO_COMPLETED_FIT')
            frame=self.frame(spec['artifact'],max_rows=self.config['caps']['max_segment_rows'])
            model_sha=evidence['checkpoint_ref']['sha256'];full_keys=actual['data_identity']['test_keys']
            if (not frame.model_sha256.eq(model_sha).all()
                    or not frame.model_train_label_end.eq(proof['value']).all()
                    or not frame.config_sha256.eq(metadata['fit_config_ref']['sha256']).all()):
                raise Blocked('ROW_MODEL_OR_TRAIN_TIME_BINDING_CONFLICT')
        frame['date']=pd.to_datetime(frame.date)
        zero=None
        if spec.get('zero_pool') is not None:
            zero=self.zero_pool(spec,calendar,full_keys)
        return frame,dict(fold=spec['fold'],model_sha256=model_sha,model_train_time=proof,
            full_pool_test_keys=full_keys,input_version=spec['input_version'],zero_pool_proof=zero)

    def zero_pool(self,spec,calendar,full_keys):
        refs=spec['zero_pool'];proof=self.metadata(refs['proof']);review=self.metadata(refs['accepted_review'])
        expected=sessions(calendar_dates(calendar),*bounds(spec['fold']))
        wanted=dict(schema='M17_ORIGINAL_ZERO_POOL_COVERAGE_V1',fold=spec['fold'],
            input_version=spec['input_version'],pool_policy='ORIGINAL_FIVE_ARM_COMMON_FINITE',
            score_artifact_sha256=spec['artifact']['sha256'],
            full_pool_test_keys_sha256=support.digest(full_keys),
            calendar_sha256=self.config['inputs']['calendar']['sha256'],
            recipe_source_sha256=self.config['sources']['recipe']['sha256'],
            nominal_sessions_sha256=support.digest(jsonable(expected)),nominal_calendar_days=len(expected),
            nonempty_pool_days=full_keys['unique_date_count'])
        if (any(proof.get(k)!=v for k,v in wanted.items())
                or review.get('decision')!='PASS' or review.get('proof_ref')!=refs['proof']):
            raise Blocked('VERSION_BOUND_ZERO_POOL_PROOF_OR_REVIEW_CONFLICT')
        empty=proof.get('zero_pool_dates')
        if (not isinstance(empty,list) or empty!=sorted(set(empty))
                or any(date.fromisoformat(t) not in expected for t in empty)
                or len(expected)!=full_keys['unique_date_count']+len(empty)):
            raise Blocked('ZERO_POOL_NOMINAL_DATE_PARTITION_CONFLICT')
        # Existing registered producer/date diagnostic, accepted independently;
        # retain/authenticate its exact evidence, never re-audit raw inputs here.
        self.metadata(proof['producer_evidence_ref'])
        return {**proof,'proof_ref':refs['proof'],'accepted_review_ref':refs['accepted_review']}

    def labels(self,requested,*,role,cutoff,calendar):
        import pandas as pd
        self.session.require_active()
        if role not in ('training','outer') or (role=='outer' and self.session.phase!='evaluate'):
            raise Blocked('OUTER_TRUTH_REQUIRES_SEPARATE_FROZEN_EVALUATION_STAGE')
        if role=='training' and (self.session.phase!='select' or cutoff not in {bounds(f)[0] for f in OUTER}):
            raise Blocked('CHRONOLOGICAL_TRAINING_STAGE_CUTOFF_REQUIRED')
        keys=pd.DataFrame(requested,columns=['date','code'])
        if keys.duplicated(['date','code']).any():raise Blocked('EXACT_UNIQUE_LABEL_REQUEST_REQUIRED')
        if not len(keys):return ()
        keys['date']=pd.to_datetime(keys.date)
        if role=='training' and not keys.date.lt(pd.Timestamp(cutoff)).all():raise Blocked('FUTURE_LABEL_REQUEST')
        dates=calendar_dates(calendar);positions={d:i for i,d in enumerate(dates)}
        if any(_day(t) not in positions for t in keys.date.unique()):
            raise Blocked('LABEL_REQUEST_CALENDAR_ANCHOR_MISSING')
        targets={pd.Timestamp(t):(dates[positions[_day(t)]+2] if positions[_day(t)]+2<len(dates) else None)
                 for t in keys.date.unique()}
        if role=='training' and any(t is None for t in targets.values()):
            raise Blocked('TRAINING_LABEL_REQUEST_H2_ANCHOR_REQUIRED')
        needed=set(keys.date)|{pd.Timestamp(t) for t in targets.values() if t is not None and (cutoff is None or t<cutoff)}
        filters=[('date','in',sorted(needed)),('code','in',sorted(set(keys.code)))]
        panel=self.frame(self.config['inputs']['panel'],columns=['date','code','close'],filters=filters,
                         max_rows=self.config['caps']['max_panel_rows'])
        panel['date']=pd.to_datetime(panel.date)
        recipe,_=self.modules()
        labels=recipe.build_close_return_labels(panel,horizons=(2,),calendar=calendar,signal_keys=keys)[2]
        values={(r.date,r.code):r.label_return for r in labels.itertuples(index=False)}
        return tuple(core.Outcome(_day(t),code,
            float(values[(t,code)]) if (t,code) in values and math.isfinite(values[(t,code)]) else None,
            targets[t]) for t,code in keys.itertuples(index=False,name=None))


def make_guard(config,config_path,phase,receipt_path):
    deps=config['registry_dependencies']
    for ref in deps.values():
        if support.sha(ref['path'])!=ref['sha256']:raise Blocked('OFFICIAL_REGISTRY_SOURCE_DRIFT')
    cached=sys.modules.get('execution_guard_core')
    if cached and Path(cached.__file__).resolve()!=Path(deps['core']['path']).resolve():
        raise Blocked('REGISTRY_CORE_IMPORT_COLLISION')
    registry_core=cached or support._load(deps['core']['path'],'execution_guard_core')
    adapter_module=support._load(deps['adapter']['path'],'m17_official_registry_adapter')
    class Adapter(adapter_module.OfficialRegistryAdapter):
        def _event_spec(self,event,*,event_type):
            spec=super()._event_spec(event,event_type=event_type)
            report=json.loads(receipt_path.read_text())
            scope={'prepare':'ORIGINAL_SOURCE_PRICES_FOR_T_CLOSE_BREADTH_NO_H2',
                   'select':'CHRONOLOGICAL_PAST_TRAIN_H2_PER_OUTER_CUTOFF_NO_CURRENT_FOLD_TRUTH',
                   'evaluate':'ISOLATED_SELECTED_OUTER_H2_AFTER_ACCEPTED_LIST_FREEZE'}[phase]
            spec.update(truth_access=('NOT_READ_FOR_THIS_STAGE_YET' if event_type=='RUN_STARTED' else scope),
                sealed_status='EXPOSED_HISTORY_NO_CONFIRMATORY_SAMPLE',
                input_sha256=scientific_identity(config),code_sha256=support.sha(__file__),
                config_sha256=support.sha(config_path),runtime=report['runtime'],
                signal_date_range=dict(start='2022-01-01',end='2026-10-15'),
                target_date_range=dict(horizon=2,definition='ORIGINAL_OBSERVED_CALENDAR_CLOSE_RATIO',
                                       stage_scope=scope))
            spec['input_identity'].update(phase_receipt_sha256=support.sha(receipt_path),
                scientific_identity=scientific_identity(config),core_policy_sha256=core.CONFIG_SHA256)
            spec['deterministic_metrics'].update(report.get('metrics',{}))
            spec['deterministic_metrics'].update(base_fit_count=0,base_fit_budget_debit=0)
            for fold in OUTER:
                if event['execution_id'].endswith('_'+fold):
                    path=receipt_path.parent/('LAMBDA_'+fold+'.json')
                    if path.is_file():
                        spec['checkpoint_ref']=dict(path=str(path),sha256=support.sha(path),
                            kind='CLOSED_FORM_SCALAR_AND_FROZEN_LISTS_NO_BASE_MODEL_FIT')
                        spec['deterministic_metrics']['scalar_fit_count']=1
            return spec
    profile=dict(track_id='A_SHARE_PANEL_MODEL_SELECTION_RESEARCH',route_id=ROUTE,
        registration_mode='LIVE_EXECUTION',start_stage='M17_'+phase.upper()+'_RUN_STARTED',
        terminal_stage='M17_'+phase.upper()+'_TERMINAL',occurred_at_reason='LIVE_EXECUTION_UTC',
        baseline_id='PV_ENVIRONMENT_R1_M0_FORMAL_EXACT_4THREAD_H2',
        one_variable='FROZEN_ENVIRONMENT_MODULATED_POSITIVE_PRIOR_SCALAR_TILT',
        signal_scope='ORIGINAL_COMMON_POOL_TOP50_THEN_TOP10_NO_REFILL',target_scope='GROSS_H2_ONLY',
        device='CPU_RESEARCH_ONLY',evidence_level='DIRECTION_ONLY')
    adapter=Adapter(registry_path=Path(config['registry_path']),contract_path=Path(config['contract']['path']),
        input_binding_path=Path(config_path),artifact_paths=(receipt_path,),
        builder_path=Path(deps['builder']['path']),validator_path=Path(deps['validator']['path']),event_profile=profile)
    # The existing guard's no-base-fit envelope; scalar math counts are explicit.
    return registry_core.ExecutionGuard(adapter,experiment_id=EXPERIMENT,max_real_fit_debits=0)


class RegisteredSession:
    def __init__(self,guard,config,config_path,phase):
        self.guard,self.config,self.config_path,self.phase=guard,config,config_path,phase
        self.active=False;self.claim=None;self.terminal_attempted=False

    def start(self):
        self.claim=self.guard.claim_no_fit(execution_id=self.config['execution_ids'][self.phase],
            arm='M17_'+self.phase.upper(),contract_sha256=self.config['contract']['sha256'],
            input_binding_sha256=support.sha(self.config_path))
        self.active=True

    def require_active(self):
        if not self.active:raise Blocked('DURABLE_REGISTERED_STAGE_REQUIRED_BEFORE_ANY_PAYLOAD')

    def finish(self,success,error=None):
        if self.claim is not None and not self.terminal_attempted:
            self.terminal_attempted=True;self.active=False
            self.guard.append_no_fit_terminal(self.claim,succeeded=success,error_type=error)

    def checkpoint(self,fold):
        import contextlib
        @contextlib.contextmanager
        def managed():
            self.require_active()
            child=self.guard.claim_no_fit(execution_id=self.config['execution_ids']['select']+'_'+fold,
                arm='M17_CLOSED_FORM_LAMBDA_'+fold,contract_sha256=self.config['contract']['sha256'],
                input_binding_sha256=support.sha(self.config_path))
            try:yield
            except BaseException as exc:
                self.guard.append_no_fit_terminal(child,succeeded=False,error_type=type(exc).__name__)
                raise
            else:self.guard.append_no_fit_terminal(child,succeeded=True)
        return managed()


def read_artifact(ref,session,caps,*,lines=False):
    session.require_active();_reference(ref)
    path=Path(ref['path'])
    if path.stat().st_size>caps['max_artifact_bytes'] or support.sha(path)!=ref['sha256']:
        raise Blocked('ACCEPTED_STAGE_ARTIFACT_SIZE_OR_SHA_DRIFT')
    if not lines:return json.loads(path.read_text())
    records=[]
    with path.open() as handle:
        for line in handle:
            if len(line.encode())>caps['max_json_record_bytes']:raise Blocked('JSON_RECORD_SIZE_BOUND')
            records.append(json.loads(line))
            if len(records)>caps['max_prepared_days']:raise Blocked('PREPARED_DAY_RESOURCE_BOUND')
    return records


def prepare_dependencies(config,adapter,sink):
    if config['passport'] is None:
        passport_mode='NARROW_NEW_USE_BINDING_REQUIRES_INDEPENDENT_REVIEW_NOT_FIRST_PIT_CERTIFICATION'
    else:
        passport=adapter.metadata(config['passport'])
        dependency_sha=support.digest({k:config[k] for k in ('sources','inputs','segments')})
        if passport.get('status')!='CERTIFIED_FOR_THIS_USE' or passport.get('dependencies_sha256')!=dependency_sha:
            raise Blocked('PASSPORT_DOES_NOT_COVER_EXACT_NEW_USE')
        adapter.accepted=passport['bound_files']
        passport_mode='FAST_BIND_EXISTING_VALID_EXACT_USE_PASSPORT'
    core_review=checked_control(config['accepted_core_review'])
    reviewed_source=core_review.get('source_sha256',core_review.get('sha256',{}).get('source'))
    reviewed_policy=core_review.get('config_sha256_unchanged',core_review.get('sha256',{}).get('config'))
    if (core_review.get('verdict')!='REVIEWED/PASS'
            or reviewed_source!=config['sources']['core']['sha256'] or reviewed_policy!=core.CONFIG_SHA256):
        raise Blocked('INDEPENDENT_CORE_PASS_DOES_NOT_COVER_BOUND_CORE')
    calendar,breadth,roster,lookup=adapter.context()
    recipe,prior=adapter.modules();records=[]
    for spec in config['segments']:
        predictions,provenance=adapter.score_segment(spec,calendar)
        records.extend(prepare_segment(recipe,prior,calendar,breadth,roster,lookup,predictions,
                                       provenance,config['caps']))
        del predictions
        if len(records)>config['caps']['max_prepared_days']:raise Blocked('PREPARED_DAY_RESOURCE_BOUND')
        resource_check(config)
    dates=calendar_dates(calendar)
    return dict(prepared_ref=sink.jsonlines('PREPARED_DAYS.jsonl',records),
        calendar_dates_ref=sink.json('CALENDAR_DATES.json',[d.isoformat() for d in dates]),
        calendar_sha256=config['inputs']['calendar']['sha256'],
        score_manifest_sha256=support.digest(config['segments']),bound_files=adapter.bound,
        passport_mode=passport_mode,days=len(records),checkpoint_unpacks=adapter.checkpoint_unpacks,
        source_pit='CANNOT_CONFIRM_RETAIN_ORIGINAL_LIMITS',core_policy_sha256=core.CONFIG_SHA256)


def resource_check(config):
    return support.resource_snapshot(dict(output_root=config['output_root'],
        resource_probe_source=config['resource_probe_source'],resource_caps=config['caps']))


def run_phase(config,config_path,activation,phase,*,guard_factory=make_guard,adapter_factory=BoundFiles):
    import importlib.metadata
    import os
    require_activation(config,config_path,activation,phase)
    if platform.system()!='Darwin' or sys.executable!=config['python']:
        raise Blocked('EXACT_ACTIVATED_MAC_RUNTIME_REQUIRED')
    actual={name:importlib.metadata.version(name) for name in
            ('numpy','pandas','scikit-learn','joblib','pyarrow','threadpoolctl')}
    actual['python']=platform.python_version()
    if actual!=config['runtime_versions']:raise Blocked('RUNTIME_VERSION_DRIFT')
    out=Path(config['output_root']);out.mkdir(exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt_path=out/(phase.upper()+'_RECEIPT.json')
        if receipt_path.exists():raise Blocked('EXISTING_STAGE_RECEIPT_NO_AUTOMATIC_REENTRY')
        report=dict(status='INTENT_BEFORE_DURABLE_START',phase=phase,execution_id=config['execution_ids'][phase],
            scientific_identity=scientific_identity(config),wrapper_sha256=support.sha(__file__),
            config_sha256=support.sha(config_path),contract_sha256=config['contract']['sha256'],
            activation_review_receipt_sha256=activation['review_receipt_sha256'],
            core_policy_sha256=core.CONFIG_SHA256,policy_sha256=POLICY_SHA256,
            planned_executor_owner=config['executor_owner'],actual_agent_identity=None,
            actual_agent_identity_reason='NOT_EXTERNALLY_VERIFIED',
            runtime=dict(python=sys.executable,versions=actual,pid=os.getpid(),host=platform.node(),threads=4),
            metrics=dict(truth_positive_denominator=None,selection_slots=None,base_fit_count=0),**FLAGS)
        if phase!='prepare':report['accepted_prepare_sha256']=config['accepted_prepare']['sha256']
        if phase=='evaluate':report['accepted_select_sha256']=config['accepted_select']['sha256']
        support.write_json(receipt_path,report)
        session=RegisteredSession(guard_factory(config,config_path,phase,receipt_path),config,config_path,phase)
        session.start();started=time.monotonic()
        try:
            with support.deadline(config['caps'][phase+'_wall_seconds']):
                report['resource_before']=resource_check(config)
                sink=ArtifactSink(out,config['caps'])
                if phase=='prepare':
                    adapter=adapter_factory(config,session)
                    result=prepare_dependencies(config,adapter,sink)
                else:
                    prepared_receipt=checked_control(config['accepted_prepare'])
                    prepared=prepared_receipt['result']
                    adapter=adapter_factory(config,session,accepted_bound_files=prepared['bound_files'])
                    dates=tuple(date.fromisoformat(d) for d in read_artifact(prepared['calendar_dates_ref'],session,config['caps']))
                    import pandas as pd
                    calendar=pd.DataFrame(dict(date=pd.to_datetime(dates),calendar_index=range(len(dates))))
                    provider=lambda requested,**kw:adapter.labels(requested,calendar=calendar,**kw)
                    if phase=='select':
                        records=read_artifact(prepared['prepared_ref'],session,config['caps'],lines=True)
                        result=fit_and_freeze(records,dates,prepared['calendar_sha256'],prepared['score_manifest_sha256'],
                                              provider,sink,session.checkpoint)
                    else:
                        selected=checked_control(config['accepted_select'])['result']
                        records=read_artifact(selected['selection_ref'],session,config['caps'],lines=True)
                        result=evaluate_frozen(records,dates,prepared['calendar_sha256'],prepared['score_manifest_sha256'],provider)
                        result_ref=sink.json('RESULT.json',result)
                        result=dict(result_ref=result_ref,state=result['state'],historical_only=True,
                                    outer_requested_slots=result['outer_requested_slots'])
                report.update(status='COMPLETED',result=jsonable(result),elapsed_seconds=time.monotonic()-started,
                              resource_after=resource_check(config))
                report['metrics'].update(scalar_fit_count=result.get('scalar_fits',0),
                                         outer_requested_slots=result.get('outer_requested_slots'))
                support.write_json(receipt_path,report)
                session.finish(True)
        except BaseException as exc:
            report.update(status='FAILED_OR_INTERRUPTED_NO_RETRY',error_type=type(exc).__name__,
                          error=str(exc)[:1000],elapsed_seconds=time.monotonic()-started)
            try:support.write_json(receipt_path,report)
            except BaseException as write_error:exc.add_note('Failure receipt unconfirmed: '+repr(write_error))
            try:session.finish(False,type(exc).__name__)
            except BaseException as terminal_error:exc.add_note('Official terminal unconfirmed: '+repr(terminal_error))
            raise
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=PHASES,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--activation',type=Path,required=True)
    args=parser.parse_args(argv)
    result=run_phase(json.loads(args.config.read_text()),args.config,
                     json.loads(args.activation.read_text()),args.phase)
    print(support.canonical(result))


if __name__=='__main__':main()
