"""Fixed month-bonus continuation: eight-key flags diagnostic, no fit or collection."""
from __future__ import annotations
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import time

import environment_tilt_evaluation as old
import environment_tilt_mac_recovery as resources
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, load_adapter, sha

KEYS = (
    ('2026-01-09','2026-01-13','001270.SZ'),
    ('2026-01-12','2026-01-14','300986.SZ'),
    ('2026-01-13','2026-01-15','002931.SZ'),
    ('2026-02-25','2026-02-27','000711.SZ'),
    ('2026-03-11','2026-03-13','000711.SZ'),
    ('2026-03-12','2026-03-16','000711.SZ'),
    ('2026-04-27','2026-04-29','600734.SH'),
    ('2026-04-28','2026-04-30','600491.SH'),
)
CAPS = dict(threads=4,rss_bytes=2*2**30,total_wall_sec=120,payload_bytes=10*2**20,
    min_memory_bytes=8*2**30,min_disk_bytes=2*2**30,max_payload_file_bytes=2*2**30,
    max_json_record_bytes=3*2**20,max_aux_rows=10000,max_panel_subset_rows=16)
EXECUTION='MONTH_BONUS_EIGHT_KEY_DIAG_20261021_01'

class Blocked(ValueError): pass

class Session:
    def __init__(self,life): self.life=life
    def require_active(self): return self.life.require()


def flags_for_keys(panel, calendar, recipe):
    """Exact keys only. Numeric values/returns never leave this function."""
    import numpy as np
    import pandas as pd
    if list(panel.columns)!=['date','code','close']:
        raise Blocked('EXACT_PANEL_PROJECTION_REQUIRED')
    panel=panel.copy();panel['date']=pd.to_datetime(panel.date)
    dates=old.calendar_dates(calendar)
    requested={(pd.Timestamp(d),c)for s,t,c in KEYS for d in (s,t)}
    actual=set(panel[['date','code']].itertuples(index=False,name=None))
    if panel.duplicated(['date','code']).any():
        raise Blocked('DUPLICATE_EXACT_PANEL_KEY')
    if not actual.issubset(requested) or len(panel)>16:
        raise Blocked('UNREQUESTED_PANEL_KEY_OR_SUBSET_BOUND')
    for signal,target,code in KEYS:
        try: observed=dates[dates.index(old._day(signal))+2].isoformat()
        except (ValueError,IndexError): raise Blocked('EXACT_TPLUS2_CALENDAR_UNAVAILABLE') from None
        if observed!=target: raise Blocked('FROZEN_TPLUS2_TARGET_CONFLICT')
    selected=pd.DataFrame([(s,c)for s,t,c in KEYS],columns=['date','code'])
    selected['date']=pd.to_datetime(selected.date)
    labels=recipe.build_close_return_labels(panel,[2],calendar=calendar,signal_keys=selected)[2]
    label_map={(old._day(d).isoformat(),c):v for d,c,v in labels[['date','code','label_return']].itertuples(index=False,name=None)}
    result=[]
    def inspect(day,code):
        values=panel.loc[(panel.date==pd.Timestamp(day))&(panel.code==code),'close']
        return dict(row_count=len(values),present=bool(len(values)),
            close_finite=bool(len(values)==1 and np.isfinite(values.iloc[0])),
            close_positive=bool(len(values)==1 and np.isfinite(values.iloc[0]) and values.iloc[0]>0))
    for signal,target,code in KEYS:
        a=inspect(signal,code);b=inspect(target,code);present=(signal,code)in label_map
        finite=bool(present and np.isfinite(label_map[(signal,code)]))
        reason=('SIGNAL_BAR_ABSENT'if not a['present'] else
            'SIGNAL_CLOSE_NONFINITE'if not a['close_finite'] else
            'SIGNAL_CLOSE_NONPOSITIVE'if not a['close_positive'] else
            'TARGET_BAR_ABSENT_STATUS_UNCLASSIFIED'if not b['present'] else
            'TARGET_CLOSE_NONFINITE'if not b['close_finite'] else
            'TARGET_CLOSE_NONPOSITIVE'if not b['close_positive'] else
            'PRODUCER_LABEL_NONFINITE'if not finite else 'LABEL_FINITE_PREVIOUS_GAP_NOT_REPRODUCED')
        result.append(dict(date=signal,target_date=target,code=code,signal=a,target=b,
            producer_label_present=present,producer_label_finite=finite,reason=reason,
            source_status='CANNOT_CONFIRM_NO_BOUND_STATUS_FIELD'))
    return result


class DiagnosticLifecycle(Lifecycle):
    def event(self,*args):
        e=super().event(*args);started=e['event_type']=='RUN_STARTED';a={}if started else self.config.get('_access',{})
        truth=('NOT_READ'if started else
            'SELECTED_H2_PRODUCER_FLAGS_COMPLETED'if a.get('flags_completed')else
            'SELECTED_H2_PANEL_RETURNED_FLAGS_NOT_COMPLETED'if a.get('panel_returned')else
            'SELECTED_H2_PANEL_READ_ATTEMPTED_NOT_RETURNED'if a.get('panel_read_attempted')else
            'EXACT_KEY_COUNT_RETURNED_NO_CLOSE_DECODE'if a.get('key_count_returned')else
            'EXACT_KEY_COUNT_ATTEMPTED_NO_CLOSE_DECODE'if a.get('key_count_attempted')else
            'PANEL_IDENTITY_AUTHENTICATION_ATTEMPTED_NO_CLOSE_DECODE'if a.get('panel_authentication_attempted')else
            'METADATA_ONLY_NO_CLOSE_SOURCE')
        e.update(route_id='a-share-month-bonus-eight-key-diagnostic-15',
            experiment_id='MONTH_BONUS_STABILITY_15',stage='EIGHT_SELECTED_H2_KEY_FLAGS',arm='A_C_FIXED_SELECTED_KEYS',
            one_variable='EXPLAIN_EIGHT_EXISTING_MISSING_LABEL_KEYS_NO_RULE_CHANGE',
            baseline_id='PV_ENVIRONMENT_R1_M0_H2_SAVED_SCORES_FIXED_A_C_PLUS5',
            truth_access=truth,
            sealed_status='EXPOSED_2026H1_DEVELOPMENT_KEYS',evidence_level='OPENBOOK_DIAGNOSTIC_ONLY')
        e['checkpoint_ref']=None
        returned=a.get('returned_target_dates',[])
        e['target_date_range']=(dict(count=0,scope='NOT_READ_BEFORE_START')if started else
            dict(count=len(returned),start=min(returned)if returned else None,end=max(returned)if returned else None,
                 requested_count=8,scope='EXACT_TARGET_CLOSE_ROWS_RETURNED_FLAGS_'+('COMPLETED'if a.get('flags_completed')else 'NOT_COMPLETED'))if a.get('panel_returned')else
            dict(count=0,requested_count=8 if a.get('key_count_attempted')else 0,scope='NO_TARGET_CLOSE_ROWS_RETURNED'))
        e['input_identity'].update(access=a,existing_selection_ref=self.config['selection_ref'])
        e['input_identity']['null_reasons'].update(checkpoint_ref='NO_CHECKPOINT_READ_IN_KEY_DIAGNOSTIC',
            selection_slots='EXISTING_SELECTION_REFERENCE_ONLY_NO_NEW_SELECTION',truth_positive_denominator='NO_POSITIVE_RETURN_COUNT_IN_FLAGS_ONLY_DIAGNOSTIC')
        e['deterministic_metrics'].update(requested_unique_signal_keys=0 if started else a.get('requested_keys',0),
            diagnosed_keys=0 if started else a.get('diagnosed_keys',0),new_fit_count=0,new_predict_count=0,network_calls=0)
        return e


def metadata_config(path):
    c=resources.read_json(path);c.update(config_path=str(Path(path).resolve()),config_sha256=sha(path))
    if c['caps']!=CAPS or c['execution_id']!=EXECUTION or c['network_allowed']is not False or c['fit_count']!=0 or c['predict_count']!=0:
        raise Blocked('FROZEN_DIAGNOSTIC_SCOPE_DRIFT')
    root=Path(c['run_dir'])
    if root!=Path(c['scope_root'])/'DIAGNOSTIC_01' or any(p.is_symlink()for p in (root,*root.parents)):
        raise Blocked('EXACT_OWNED_OUTPUT_REQUIRED')
    for r in [*c['code_refs'].values(),c['contract_ref'],c['recipe_ref']]:
        if sha(r['path'])!=r['sha256']: raise Blocked('SOURCE_OR_CONTRACT_DRIFT')
    contract=resources.read_json(c['contract_ref']['path'])
    if tuple((r['date'],r['target_date'],r['code'])for r in contract['keys'])!=KEYS:
        raise Blocked('FROZEN_EIGHT_KEYS_REQUIRED')
    if c['code_refs']['runner']['path']!=str(Path(__file__).resolve()):raise Blocked('RUNNER_PATH_DRIFT')
    c.update(code_composite_sha256=old.support.digest(c['code_refs']),contract_path=c['contract_ref']['path'],contract_sha256=c['contract_ref']['sha256'])
    return c


def read_exact_panel(adapter,c,pairs,filters):
    """Count the identical key predicate before materializing any close column."""
    import pandas as pd
    import pyarrow.dataset as ds
    a=c.setdefault('_access',{});p=c['inputs']['panel']
    a.update(panel_authentication_attempted=True,requested_keys=8)
    before=adapter.authenticate(p)
    predicate=None
    for day,code in pairs:
        term=(ds.field('date')==pd.Timestamp(day).to_pydatetime())&(ds.field('code')==code)
        predicate=term if predicate is None else predicate|term
    a['key_count_attempted']=True
    count=ds.dataset(p['path'],format='parquet').count_rows(filter=predicate)
    a.update(key_count_returned=True,matched_panel_rows=count)
    if old.support.stat_identity(p['path'])!=before:raise Blocked('PANEL_CHANGED_DURING_EXACT_KEY_COUNT')
    if count>CAPS['max_panel_subset_rows']:raise Blocked('EXACT_PANEL_PREDECODE_ROW_BOUND')
    a['panel_read_attempted']=True
    panel=adapter.frame(p,columns=['date','code','close'],filters=filters,max_rows=c['panel_row_binding']['file_rows'])
    a['panel_returned']=True
    targets={(t,code)for signal,t,code in KEYS}
    a['returned_target_dates']=sorted({old._day(day).isoformat()for day,code in panel[['date','code']].itertuples(index=False,name=None)if(old._day(day).isoformat(),code)in targets})
    if len(panel)!=count:raise Blocked('EXACT_KEY_COUNT_VS_DECODED_ROWS_DRIFT')
    return panel


def diagnose(c,life):
    import pandas as pd
    access=c.setdefault('_access',{});adapter=old.BoundFiles(c,Session(life))
    # Existing selection/review prove these are the already frozen A/C keys, no score read.
    review=adapter.metadata(c['prior_actual_review_ref'])
    if review['verdict']!='READBACK_AND_REGISTRY_CHAIN_PASS_WITH_LIMITATIONS; BUSINESS_ACCEPTANCE_CANNOT_CONFIRM':
        raise Blocked('EXISTING_ACTUAL_REVIEW_REQUIRED')
    selection=adapter.metadata(c['selection_ref']);records={r['date']:r for r in selection['records']}
    for signal,target,code in KEYS:
        r=records[signal]
        if r['h2_date']!=target or not all(code in r['selections'][a]for a in ('A_ORIGINAL','C_MONTH_BONUS')):
            raise Blocked('ORIGINAL_A_C_SELECTION_BINDING_CONFLICT')
    proof=adapter.metadata(c['panel_row_binding']['preparation_ref']);p=c['inputs']['panel']
    if proof['status']!='COMPLETED' or proof['panel_rows']!=c['panel_row_binding']['file_rows']or proof['cache_files'].get(p['path'])!=p['sha256']:
        raise Blocked('BOUND_PANEL_METADATA_DRIFT')
    calendar=adapter.frame(c['inputs']['calendar'],columns=['date','calendar_index']);calendar['date']=pd.to_datetime(calendar.date)
    old.calendar_dates(calendar)
    recipe=old.support._load(c['recipe_ref']['path'],'month_bonus_original_label_recipe')
    pairs=sorted({(d,code)for s,t,code in KEYS for d in (s,t)})
    filters=[[('date','==',pd.Timestamp(d)),('code','==',code)]for d,code in pairs]
    panel=read_exact_panel(adapter,c,pairs,filters)
    rows=flags_for_keys(panel,calendar,recipe);access.update(flags_completed=True,diagnosed_keys=len(rows))
    from collections import Counter
    return dict(status='EIGHT_KEYS_DIAGNOSED',rows=rows,reason_counts=dict(Counter(r['reason']for r in rows)),
        input_bindings=adapter.bound,source_status='CANNOT_CONFIRM_SUSPENSION_VS_SOURCE_OMISSION',
        new_fit_count=0,new_predict_count=0,network_calls=0,**FLAGS)


def run(c,activation,adapter=None,compute_fn=diagnose,*,lifecycle_type=DiagnosticLifecycle,approval='APPROVED_FOR_EIGHT_KEY_DIAGNOSTIC'):
    if activation.get('status')!=approval:raise Blocked('ACTIVATION_REQUIRED')
    for k in ('config_sha256','contract_sha256','code_composite_sha256','execution_id'):
        if activation.get(k)!=c[k]:raise Blocked('ACTIVATION_IDENTITY_DRIFT')
    if approval=='APPROVED_FOR_FIXED_AC_EXPANSION':
        required=activation.get('raw_prerequisite_ref',{})
        if required.get('path')!=c['raw_prerequisite_receipt_path']or not required.get('sha256'):raise Blocked('COMPLETED_RAW_DIAGNOSTIC_ACTIVATION_BINDING_REQUIRED')
        c['_raw_prerequisite_ref']=required
    r=activation['review_receipt']
    if sha(r['path'])!=r['sha256']:raise Blocked('REVIEW_RECEIPT_DRIFT')
    root=Path(c['run_dir']);root.mkdir(exist_ok=False)
    lock=(root/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    life=lifecycle_type(adapter or load_adapter(c),c);receipt=root/'RECEIPT.json';budget=resources.Budget(root,c['caps']);report={}
    try:
        resources.write_json(receipt,dict(status='PREPARED_NOT_STARTED',**FLAGS));life.start(receipt)
        with budget:
            if resources.available_memory()<CAPS['min_memory_bytes']or shutil.disk_usage(root).free<CAPS['min_disk_bytes']:
                raise Blocked('DEFERRED_RESOURCE')
            os.environ.update({k:'4'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
            c['arrow_pools']=resources.configure_arrow_pools(4)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):report=compute_fn(c,life)
            budget.check()
    except BaseException as e:
        report=dict(status='STOPPED',error_type=type(e).__name__,error=str(e),new_fit_count=0,new_predict_count=0,network_calls=0,**FLAGS)
    finally:
        report.update(access=c.get('_access',{}),elapsed_seconds=time.monotonic()-budget.start_time,
            sampled_peak_rss_bytes=budget.peak,arrow_pools=c.get('arrow_pools'))
        resources.write_json(receipt,report)
        if life.started and not life.finished:life.finish(receipt,report['status'],dict(diagnosed_keys=len(report.get('rows',[])),fits=0,network_calls=0))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return report


RAW_EXECUTION='MONTH_BONUS_RAW_TARGET_DIAG_20261021_01'
RAW_POLICY=dict(stage='RAW_TARGET_ONLY',keys=[dict(date=t,code=c)for s,t,c in KEYS],
    columns=['trade_date','ts_code','close'],date_type='int64_YYYYMMDD',row_cap=16,
    no_network=True,no_returns=True,no_source_repair=True)


class RawLifecycle(DiagnosticLifecycle):
    def event(self,*args):
        e=super().event(*args)
        e.update(route_id='a-share-month-bonus-raw-eight-target-diagnostic-15',
            stage='ORIGINAL_RAW_EIGHT_TARGET_FLAGS',arm='EXACT_EIGHT_EXISTING_TARGET_KEYS',
            one_variable='DISTINGUISH_RAW_ABSENCE_FROM_CANONICAL_PROJECTION_LOSS')
        e['truth_access']=e['truth_access'].replace('SELECTED_H2_PRODUCER_FLAGS_COMPLETED','RAW_TARGET_FLAGS_COMPLETED_NO_RETURN_LABELS').replace('SELECTED_H2','RAW_TARGET')
        e['target_date_range']['scope']=e['target_date_range']['scope'].replace('EXACT_TARGET_CLOSE','RAW_EXACT_TARGET_CLOSE')
        e['deterministic_metrics']['requested_unique_target_keys']=e['deterministic_metrics'].pop('requested_unique_signal_keys')
        return e


def phase_config(path,execution,subdir,caps,policy):
    c=resources.read_json(path);c.update(config_path=str(Path(path).resolve()),config_sha256=sha(path))
    if c['caps']!=caps or c['execution_id']!=execution or c['policy']!=policy or c['network_allowed']is not False or c['fit_count']!=0 or c['predict_count']!=0:
        raise Blocked('FROZEN_PHASE_POLICY_OR_CAP_DRIFT')
    root=Path(c['run_dir'])
    if root!=Path(c['scope_root'])/subdir or any(p.is_symlink()for p in(root,*root.parents)):raise Blocked('EXACT_PHASE_OUTPUT_REQUIRED')
    for r in [*c['code_refs'].values(),c['contract_ref'],c['recipe_ref']]:
        if sha(r['path'])!=r['sha256']:raise Blocked('PHASE_SOURCE_OR_CONTRACT_DRIFT')
    contract=resources.read_json(c['contract_ref']['path'])
    if contract['policy']!=policy or contract['caps']!=caps:raise Blocked('PHASE_CONTRACT_BINDING_DRIFT')
    if execution==RESUME_EXECUTION and(not c.get('resume_ref')or contract.get('resume_ref')!=c['resume_ref']):
        raise Blocked('FROZEN_RECOVERY_BINDING_REQUIRED')
    if c['code_refs']['runner']['path']!=str(Path(__file__).resolve()):raise Blocked('PHASE_RUNNER_PATH_DRIFT')
    c.update(code_composite_sha256=old.support.digest(c['code_refs']),contract_path=c['contract_ref']['path'],contract_sha256=c['contract_ref']['sha256'])
    return c


def raw_target_flags(c,life):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds
    a=c.setdefault('_access',{});adapter=old.BoundFiles(c,Session(life))
    binding=adapter.metadata(c['original_binding_ref']);projection=adapter.metadata(c['original_projection_ref'])
    bound=binding['sources']['price_panel'];p=c['inputs']['raw']
    if bound['path']!=p['path']or bound['sha256']!=p['sha256']or bound['receipt']['sha256']!=c['original_projection_ref']['sha256']or bound['column_map']['date']!='trade_date'or bound['column_map']['code']!='ts_code':raise Blocked('ORIGINAL_RAW_BINDING_CONFLICT')
    meta=projection['panels']['assembled_qfq']
    if meta['path']!=p['path']or meta['metadata_row_count']!=c['raw_rows']or meta['field_summaries']['trade_date']['type']!='int64':raise Blocked('ORIGINAL_RAW_SCHEMA_RECEIPT_CONFLICT')
    accepted=adapter.metadata(c['canonical_actual_review_ref'])
    if accepted['boundary_status']['diagnostic_integrity']!='PASS_WITH_LIMITATIONS':raise Blocked('CANONICAL_DIAGNOSTIC_REVIEW_REQUIRED')
    canonical=adapter.metadata(c['canonical_flags_ref']);rows=canonical['rows']
    if [(r['date'],r['target_date'],r['code'])for r in rows]!=list(KEYS)or any(r['target']['present']for r in rows):raise Blocked('EXACT_CANONICAL_FLAGS_BINDING_CONFLICT')
    a.update(panel_authentication_attempted=True,requested_keys=8)
    before=adapter.authenticate(p);dataset=ds.dataset(p['path'],format='parquet')
    if not pa.types.is_int64(dataset.schema.field('trade_date').type)or not pa.types.is_string(dataset.schema.field('ts_code').type):raise Blocked('ORIGINAL_RAW_DATE_CODE_SCHEMA_DRIFT')
    pairs=[(int(t.replace('-','')),code)for signal,t,code in KEYS];predicate=None
    for day,code in pairs:
        term=(ds.field('trade_date')==day)&(ds.field('ts_code')==code);predicate=term if predicate is None else predicate|term
    a['key_count_attempted']=True;count=dataset.count_rows(filter=predicate);a.update(key_count_returned=True,matched_panel_rows=count)
    if old.support.stat_identity(p['path'])!=before:raise Blocked('RAW_CHANGED_DURING_KEY_COUNT')
    if count>16:raise Blocked('RAW_TARGET_PREDECODE_ROW_BOUND')
    filters=[[('trade_date','==',d),('ts_code','==',code)]for d,code in pairs]
    a['panel_read_attempted']=True
    frame=adapter.frame(p,columns=RAW_POLICY['columns'],filters=filters,max_rows=c['raw_rows']);a['panel_returned']=True
    if len(frame)!=count or frame.duplicated(['trade_date','ts_code']).any():raise Blocked('RAW_TARGET_COUNT_OR_DUPLICATE_CONFLICT')
    if not set(frame[['trade_date','ts_code']].itertuples(index=False,name=None)).issubset(pairs):raise Blocked('UNREQUESTED_RAW_TARGET_KEY')
    a['returned_target_dates']=sorted(pd.to_datetime(frame.trade_date.astype(str),format='%Y%m%d').dt.strftime('%Y-%m-%d').unique().tolist())
    output=[]
    for signal,target,code in KEYS:
        values=frame.loc[(frame.trade_date==int(target.replace('-','')))&(frame.ts_code==code),'close'];present=len(values)==1
        output.append(dict(date=signal,target_date=target,code=code,raw_target_row_count=len(values),raw_target_present=present,
            raw_close_finite=bool(present and np.isfinite(values.iloc[0])),raw_close_positive=bool(present and np.isfinite(values.iloc[0])and values.iloc[0]>0),canonical_target_present=False,
            reason='RAW_AND_CANONICAL_TARGET_ABSENT_STATUS_UNCLASSIFIED'if not present else 'RAW_TARGET_PRESENT_CANONICAL_ABSENT_REQUIRES_EXACT_FIX',
            normalize_key_retention='NO_RAW_ROW_TO_NORMALIZE'if not present else 'SOURCE_FUNCTION_RETAINS_KEY_IF_FULL_INPUT_VALIDATION_PASSES',
            full_OHLC_normalization_validation='NOT_RUN_THREE_COLUMNS_ONLY'))
    a.update(flags_completed=True,diagnosed_keys=8)
    return dict(status='RAW_EIGHT_TARGETS_DIAGNOSED',rows=output,input_bindings=adapter.bound,
        new_fit_count=0,new_predict_count=0,network_calls=0,**FLAGS)


PRIMARY=('2024H1','2024H2','2025H1','2025H2')
EXP_FOLDS=('2023H2',*PRIMARY,'2026H2_PARTIAL_TO_2026_10_15')
EXP_EXECUTION='MONTH_BONUS_STABILITY_EXPANSION_20261021_01'
EXP_CAPS=dict(CAPS,rss_bytes=4*2**30,total_wall_sec=600,payload_bytes=64*2**20,
    max_segment_rows=700000,max_selected_panel_rows=5000,max_aux_rows=2000000)
EXP_POLICY=dict(arms=['A_ORIGINAL','C_MONTH_BONUS'],primary=list(PRIMARY),auxiliary=['2023H2'],
    seen=['2026H1'],partial=['2026H2_PARTIAL_TO_2026_10_15'],topk=50,topn=10,rank_scale=100,bonus=5,
    horizon=2,market_gate=False,prior='ORIGINAL_S1_DPLUS1_MONTH_PAST_COMPLETE_YEARS_KAPPA100',
    bootstrap_block=20,bootstrap_replicates=2000,bootstrap_seed=20261021,
    screen='COMPLETE_COVERAGE_AND_4_FULL_HALVES_2_YEARS_MEAN_GT0_LCB_GT0_CEIL_TWO_THIRDS_POSITIVE',
    historical_scope='OPENBOOK_DIRECTION_ONLY_GROSS_OVERLAPPING_H2_NOT_NAV_NET_OR_PORTFOLIO')


def select_ac(predictions,calendar,lookup,roster,prior,fold):
    import numpy as np
    import pandas as pd
    from environment_tilt_simple_rule_canary import fixed_rank
    dates=old.calendar_dates(calendar);year=int(fold[:4]);half=int(fold[5])
    start=f'{year}-{1 if half==1 else 7:02d}-01'
    end='2026-10-15'if 'PARTIAL'in fold else f'{year}-{6 if half==1 else 12:02d}-{30 if half==1 else 31}'
    nominal=[d for d in dates if start<=d.isoformat()<=end]
    if not nominal:raise Blocked('NO_BOUND_NOMINAL_SESSIONS')
    if dates[0].isoformat()>start or dates[-1].isoformat()<end:raise Blocked('BOUND_CALENDAR_WINDOW_COVERAGE_UNPROVEN')
    frame=predictions[['date','code','prediction']].copy();frame.date=pd.to_datetime(frame.date)
    if frame.duplicated(['date','code']).any()or not np.isfinite(frame.prediction).all()or not set(old._day(d)for d in frame.date).issubset(nominal):raise Blocked('EXPANSION_SCORE_KEY_OR_VALUE_DRIFT')
    selected_lookup=lookup.loc[lookup.signal_year==year]
    if selected_lookup.empty or not(selected_lookup.max_prior_year<=year-1).all():raise Blocked('PRIOR_YEAR_CUTOFF_VIOLATION')
    groups={old._day(d):g for d,g in frame.groupby('date',sort=False)};records=[]
    for day in nominal:
        pos=dates.index(day);g=groups.get(day);anchor=dates[pos+1]if pos+1<len(dates)else None;target=dates[pos+2]if pos+2<len(dates)else None
        reason='NO_SAVED_SCORE_DATE_ZERO_POOL_UNCONFIRMED'if g is None else 'LESS_THAN_TEN_ORIGINAL_POOL_ROWS'if len(g)<10 else 'DPLUS1_PRIOR_CALENDAR_UNAVAILABLE'if anchor is None else None
        record=dict(fold=fold,date=day.isoformat(),calendar_index=pos,h2_date=target.isoformat()if target else None,
            full_pool_rows=0 if g is None else len(g),reason=reason,selections=None)
        if reason is None:
            top=g.sort_values(['prediction','code'],ascending=[False,True],kind='mergesort').head(50)
            projected=prior.project_prior(top[['date','code']],roster,calendar,selected_lookup)
            pairs={r.code:(float(getattr(r,prior.PRIOR_COLUMNS[0])),float(getattr(r,prior.PRIOR_COLUMNS[-1])))for r in projected.itertuples(index=False)}
            base,candidate,ranked=fixed_rank(g,pairs)
            record.update(selections=dict(A_ORIGINAL=base,C_MONTH_BONUS=candidate),month_rank_changed=base!=candidate,prior_anchor_date=anchor.isoformat(),prior_max_year=int(selected_lookup.max_prior_year.max()),positive_bonus_candidates=sum(r['bonus']>0 for r in ranked))
            if target is None:record['reason']='H2_TARGET_CALENDAR_NOT_MATURE_PARTIAL_ONLY'
        records.append(record)
    return records


def evaluate_ac(records,outcomes):
    import statistics
    rows=[]
    for record in records:
        r=dict(fold=record['fold'],date=record['date'],calendar_index=record['calendar_index'],valid=False,reason=record['reason'],missing_codes=[],returns=None)
        if record['reason']is None:
            values={};missing=set()
            for arm in EXP_POLICY['arms']:
                codes=record['selections'][arm]
                if len(codes)!=10 or len(set(codes))!=10:raise Blocked('EXACT_TEN_SELECTED_SLOTS_REQUIRED')
                observed=[outcomes.get((record['date'],c))for c in codes]
                missing.update(c for c,v in zip(codes,observed)if v is None or not math.isfinite(v))
                values[arm]=None if any(v is None or not math.isfinite(v)for v in observed)else statistics.mean(observed)
            r.update(valid=not missing,reason='MISSING_SELECTED_H2_LABEL_ENTIRE_AC_PAIR_EXCLUDED'if missing else None,missing_codes=sorted(missing),returns=None if missing else values)
        rows.append(r)
    return rows


def ac_stats(rows):
    import statistics
    good=[r for r in rows if r['valid']];complete=bool(rows)and len(good)==len(rows)
    arm={a:dict(mean=statistics.mean(r['returns'][a]for r in good)if good else None,
        cvar5=old.core.fractional_cvar5([r['returns'][a]for r in good])if good else None)for a in EXP_POLICY['arms']}
    delta=statistics.mean(r['returns']['C_MONTH_BONUS']-r['returns']['A_ORIGINAL']for r in good)if good else None
    return dict(nominal_dates=len(rows),valid_common_dates=len(good),complete_coverage=complete,arms=arm,
        descriptive_paired_mean_delta=delta,full_nominal_mean_delta=delta if complete else None,
        cvar5_difference=arm['C_MONTH_BONUS']['cvar5']-arm['A_ORIGINAL']['cvar5']if good else None,
        missing_dates=[dict(date=r['date'],reason=r['reason'],missing_codes=r['missing_codes'])for r in rows if not r['valid']])


def contiguous_runs(rows):
    runs=[]
    for r in rows:
        if not r['valid']:continue
        if not runs or runs[-1][-1]['fold']!=r['fold']or runs[-1][-1]['calendar_index']+1!=r['calendar_index']:
            runs.append([])
        runs[-1].append(r)
    return runs


def paired_interval(rows):
    import numpy as np
    runs=contiguous_runs(rows);lengths=[len(r)for r in runs]
    if not runs or any(n<20 for n in lengths):return dict(lower=None,upper=None,status='CANNOT_CONFIRM_RUN_SHORTER_FROZEN_20',run_lengths=lengths)
    arrays=[np.array([r['returns']['C_MONTH_BONUS']-r['returns']['A_ORIGINAL']for r in run],dtype=float)for run in runs]
    rng=np.random.default_rng(20261021);replicates=[];total=sum(lengths)
    for _ in range(2000):
        sums=[]
        for values in arrays:
            n=len(values);starts=rng.integers(0,n-20+1,size=math.ceil(n/20))
            ix=np.concatenate([np.arange(start,start+20)for start in starts])[:n]
            sums.append(float(values[ix].sum()))
        replicates.append(sum(sums)/total)
    lo,hi=np.quantile(replicates,[.025,.975])
    return dict(lower=float(lo),upper=float(hi),status='APPROXIMATE_OPENBOOK_PAIRED_MOVING_BLOCK_PERCENTILE95',run_lengths=lengths,block=20,replicates=2000,seed=20261021)


def stability_summary(by_fold):
    import statistics
    primary=[r for f in PRIMARY for r in by_fold[f]];summary=ac_stats(primary);interval=paired_interval(primary)
    halves={f:ac_stats(by_fold[f])for f in PRIMARY};positive=sum(v['descriptive_paired_mean_delta']is not None and v['descriptive_paired_mean_delta']>0 for v in halves.values())
    gates=dict(complete_nominal_coverage=summary['complete_coverage'],four_full_halves=len(halves)>=4,two_full_years=all(set([f'{y}H1',f'{y}H2']).issubset(PRIMARY)for y in [2024,2025]),
        mean_positive=summary['descriptive_paired_mean_delta']is not None and summary['descriptive_paired_mean_delta']>0,
        approximate_95_lower_positive=interval['lower']is not None and interval['lower']>0,
        positive_half_count_at_least_ceil_two_thirds=positive>=math.ceil(2*len(PRIMARY)/3))
    all_rows=[r for f in EXP_FOLDS for r in by_fold[f]]
    # Annual/quarter reports are descriptive and never pool different input versions.
    years={str(y):ac_stats([r for r in primary if r['date'].startswith(str(y))])for y in [2024,2025]}
    quarters={f'{y}Q{q}':ac_stats([r for r in primary if r['date'].startswith(str(y))and(int(r['date'][5:7])-1)//3+1==q])for y in [2024,2025]for q in [1,2,3,4]}
    return dict(primary=summary,halves=halves,years=years,quarters=quarters,interval=interval,gates=gates,positive_halves=positive,
        screen_passed=all(gates.values()),status='CANNOT_CONFIRM_STABILITY_FROZEN_BLOCK_INTERVAL'if interval['lower']is None else 'CANNOT_CONFIRM_STABILITY_INCOMPLETE_COVERAGE'if not summary['complete_coverage']else 'OPENBOOK_STABILITY_SCREEN_MET_CONTINUE_RESEARCH_ONLY'if all(gates.values())else 'OPENBOOK_STABILITY_SCREEN_NOT_MET_NO_RETUNING',
        auxiliary_2023H2=ac_stats(by_fold['2023H2']),auxiliary_2023H2_quarters={f'2023Q{q}':ac_stats([r for r in by_fold['2023H2']if(int(r['date'][5:7])-1)//3+1==q])for q in [3,4]},
        partial_2026H2=ac_stats(by_fold['2026H2_PARTIAL_TO_2026_10_15']),partial_2026H2_quarters={f'2026Q{q}':ac_stats([r for r in by_fold['2026H2_PARTIAL_TO_2026_10_15']if(int(r['date'][5:7])-1)//3+1==q])for q in [3,4]},
        historical_classification='DIRECTION_ONLY',net_return=None,nav=None,portfolio_drawdown=None,new_fit_count=0,new_predict_count=0,network_calls=0,**FLAGS)


def load_saved_scores(adapter,c,spec,calendar,recipe):
    import pandas as pd
    import joblib
    fold=spec['fold'];a=c.setdefault('_access',{});a['active_fold']=fold
    if spec['format']=='normalized_oof':
        review=adapter.metadata(spec['original_QA'])
        if review['verdict']!='PASS_SINGLE_CELL_FIT_RECEIPT_AND_SCORE_IDENTITY':raise Blocked('AUXILIARY_ORIGINAL_CELL_REVIEW_REQUIRED')
        a['score_read_attempted']=True
        frame,provenance=adapter.score_segment(dict(fold=fold,format='normalized_oof',artifact=spec['score_ref'],metadata=spec['metadata'],accepted_review=spec['accepted_review'],input_version=spec['as_run_input_version']),calendar)
        a.setdefault('score_payload_reads',[]).append(spec['score_ref']);return frame,provenance
    chain=spec['acceptance_chain'];metadata=adapter.metadata(spec['metadata']);identity=metadata['identity']
    review=adapter.metadata(chain['review']);lifecycle=adapter.metadata(chain['lifecycle_review']);stage=adapter.metadata(chain['stage_manifest']);output=adapter.metadata(chain['output_manifest']);replay=adapter.metadata(chain['checkpoint_replay'])
    cells=[v for v in stage['cell_results']if v.get('checkpoint_path')==spec['artifact']['path']]
    if len(cells)!=1 or cells[0]['status']!='COMPLETED'or cells[0]['fit_called']is not True:raise Blocked('EXACT_ORIGINAL_COMPLETED_CELL_REQUIRED')
    if(review['verdict']!='METRICS_CONSISTENT'or review['research_decision']!='M4_GATE_FAILED'or lifecycle['verdict']!='PASS'or review['artifacts']['contract_sha256']!=spec['as_run_contract_sha256']or review['artifacts']['output_manifest_sha256']!=chain['output_manifest']['sha256']or lifecycle['output_manifest_sha256']!=chain['output_manifest']['sha256']or output['contract_sha256']!=spec['as_run_contract_sha256']or stage['contract_sha256']!=spec['as_run_contract_sha256']):raise Blocked('ORIGINAL_ARTIFACT_REVIEW_CHAIN_CONFLICT')
    if(metadata['status']!='COMPLETED'or metadata['artifact_sha256']!=spec['artifact']['sha256']or identity['arm']!='M0'or identity['fold']!=fold or identity['horizon']!=2 or identity['config_sha256']!=spec['as_run_config_sha256']or identity['contract_sha256']!=spec['as_run_contract_sha256']):raise Blocked('EXACT_CHECKPOINT_SIDECAR_CELL_CONFLICT')
    if(replay['status']!='VERIFIED'or replay['checkpoint_path']!=spec['artifact']['path']or replay['checkpoint_metadata_path']!=spec['metadata']['path']or replay['config_sha256']!=identity['config_sha256']or replay['checkpoint_sha256']!=spec['artifact']['sha256']or replay['checkpoint_metadata_sha256']!=spec['metadata']['sha256']or replay['data_identity']!=identity['data_identity']or replay['prediction_identity']!=replay['replay_prediction_identity']or replay['top10_parity']['status']!='EXACT'or replay['comparison']['key_order_equal']is not True or replay['comparison']['prediction_equal']is not True):raise Blocked('EXACT_REPLAY_BINDING_CONFLICT')
    # The authenticated sidecar+replay explicitly bind these exact rows to the
    # exact checkpoint SHA. This is a pre-unpack bound, not a post-load estimate.
    keys=identity['data_identity']['test_keys'];recorded_rows=keys.get('row_count')
    if(type(recorded_rows)is not int or not 1<=recorded_rows<=EXP_CAPS['max_segment_rows']
            or keys.get('unique_key_count')!=recorded_rows
            or spec['accepted_identity']['test']['row_count']!=recorded_rows
            or replay['prediction_identity'].get('row_count')!=recorded_rows):
        raise Blocked('AUTHENTICATED_CHECKPOINT_PREUNPACK_ROW_BOUND_OR_IDENTITY_CONFLICT')
    dates=old.calendar_dates(calendar);proof=old.restore_model_time(metadata,dates,model_source_sha256=c['sources']['recipe']['sha256'],calendar_sha256=c['inputs']['calendar']['sha256']);proof['evidence_sha256']=spec['metadata']['sha256']
    a['score_read_attempted']=True;before=adapter.authenticate(spec['artifact']);payload=joblib.load(spec['artifact']['path']);frame=payload['predictions'].copy();del payload
    a.setdefault('checkpoint_refs',[]).append(spec['artifact']);a.setdefault('score_payload_reads',[]).append(spec['artifact'])
    if len(frame)!=recorded_rows:raise Blocked('LOADED_CHECKPOINT_EXACT_ROW_COUNT_DRIFT')
    if old.support.stat_identity(spec['artifact']['path'])!=before:raise Blocked('CHECKPOINT_CHANGED_DURING_READ')
    frame.date=pd.to_datetime(frame.date)
    if recipe.key_frame_identity(frame)!=identity['data_identity']['test_keys']:raise Blocked('EXACT_FULLPOOL_SCORE_KEYS_CHANGED')
    old.validate_time_proof(proof,old._day(frame.date.min()))
    return frame,dict(fold=fold,input_version=spec['as_run_input_version'],model_train_time=proof,full_pool_test_keys=identity['data_identity']['test_keys'],existing_research_decision=review['research_decision'],new_predict_called=False)


class ExpansionLifecycle(Lifecycle):
    def event(self,*args):
        e=super().event(*args);started=e['event_type']=='RUN_STARTED';a={}if started else self.config.get('_access',{})
        e.update(route_id='a-share-fixed-month-bonus-stability-15',experiment_id='MONTH_BONUS_STABILITY_15',stage='SAVED_SCORE_FIXED_A_C_OPENBOOK_STABILITY',arm='A_ORIGINAL_C_MONTH_BONUS',baseline_id='ORIGINAL_M0_H2_SAVED_SCORES_BY_INPUT_VERSION',one_variable='TOP50_RANKPOINTS_PLUS5_POSITIVE_ORIGINAL_S1_MONTH_INDUSTRY_PRIOR',sealed_status='OPENBOOK_HISTORY_NO_SEALED_CLAIM',evidence_level='DIRECTION_ONLY',truth_access='NOT_READ'if started else 'SELECTED_H2_LABELS_COMPUTED_IN_RECORDED_FOLDS'if a.get('label_folds_completed')else 'SELECTED_CLOSE_READ_ATTEMPTED'if a.get('label_source_attempted')else 'EXISTING_SAVED_SCORE_READ'if a.get('score_payload_reads')else 'EXISTING_SCORE_READ_ATTEMPTED_NOT_RETURNED'if a.get('score_read_attempted')else 'METADATA_ONLY')
        e['checkpoint_ref']=None if started else a.get('checkpoint_refs')or None
        requested=a.get('requested_target_dates',[]);returned=a.get('returned_target_dates',[])
        e['target_date_range']=dict(count=len(returned),start=min(returned)if returned else None,end=max(returned)if returned else None,requested_count=len(requested),scope='EXACT_SELECTED_H2_TARGET_CLOSE_ROWS_RETURNED'if returned else 'NO_SELECTED_TARGET_ROWS_RETURNED')
        e['input_identity'].update(access=a,input_versions=self.config['input_versions'],raw_prerequisite_ref=self.config.get('_raw_prerequisite_ref'))
        reasons=e['input_identity']['null_reasons'];reasons['truth_positive_denominator']='NO_CLASSIFICATION_POSITIVE_TARGET'
        if e['checkpoint_ref']is None:reasons['checkpoint_ref']='NO_CHECKPOINT_PAYLOAD_UNPACKED_YET'
        else:reasons.pop('checkpoint_ref',None)
        if not started and a.get('selection_slots')is not None:reasons.pop('selection_slots',None)
        e['deterministic_metrics'].update(truth_positive_denominator=None,selection_slots=None if started else a.get('selection_slots'),selection_slots_by_arm=None if started else a.get('selection_slots_by_arm'),nominal_dates=None if started else a.get('nominal_dates'),valid_common_dates=None if started else a.get('valid_dates'),new_fit_count=0,new_predict_count=0,network_calls=0)
        return e


    def checkpoint(self,artifact,fold):
        """Use the same official lock/builder/validator/fsync; no adapter edits."""
        self.require();event=self.event('CHECKPOINT','FIXED_CELL_PAIRED_RESULTS_SAVED',artifact['path'],dict(fold=fold))
        event['registry_event_id']=self.config['execution_id']+'_CHECKPOINT_'+fold
        event['input_identity']['result_checkpoint']=artifact
        with self.adapter._locked():
            raw,events=self.adapter._read_validated()
            same=[e for e in events if e.get('execution_id')==self.config['execution_id']]
            starts=[e for e in same if e['event_type']=='RUN_STARTED']
            if len(starts)!=1 or any(e['event_type']=='TERMINAL'or e['registry_event_id']==event['registry_event_id']for e in same):raise Blocked('ACTIVE_UNIQUE_CHECKPOINT_REQUIRED')
            if any(starts[0][k]!=event[k]for k in ('contract_sha256','input_binding_sha256','code_sha256','config_sha256')):raise Blocked('CHECKPOINT_START_IDENTITY_DRIFT')
            self.adapter._append_unlocked(raw,events,event)


def reuse_diagnostic_bindings(adapter,c):
    """Reuse only accepted post-START panel/calendar SHA+stat, not new provenance."""
    proof=adapter.metadata(c['diagnostic_receipt_ref']);review=adapter.metadata(c['diagnostic_actual_review_ref'])
    if proof['status']!='EIGHT_KEYS_DIAGNOSED'or review['boundary_status']['diagnostic_integrity']!='PASS_WITH_LIMITATIONS':raise Blocked('ACCEPTED_DIAGNOSTIC_FAST_BIND_REQUIRED')
    reused=[]
    for name in ('panel','calendar'):
        r=c['inputs'][name];bound=proof['input_bindings'].get(r['path'])
        if not bound or bound['sha256']!=r['sha256']or bound['stat']!=old.support.stat_identity(r['path']):raise Blocked('DIAGNOSTIC_FAST_BIND_IDENTITY_OR_STAT_DRIFT')
        adapter.bound[r['path']]=dict(sha256=bound['sha256'],stat=dict(bound['stat']))
        reused.append(dict(path=r['path'],sha256=r['sha256'],authentication='FAST_BIND_ACCEPTED_DIAGNOSTIC_SHA_PLUS_CURRENT_UNCHANGED_STAT'))
    c.setdefault('_access',{})['fast_bound_inputs']=reused


EXACT_PAIR_BATCH_SIZE=64
RESUME_EXECUTION='MONTH_BONUS_STABILITY_EXPANSION_20261023_RESUME_01'


def read_selected_panel(adapter,c,pairs):
    """Disjoint bounded predicates; count the entire requested set before close decode."""
    import pandas as pd
    import pyarrow.dataset as ds
    pairs=sorted(set(pairs));ref=c['inputs']['panel'];a=c.setdefault('_access',{})
    before=adapter.authenticate(ref);dataset=ds.dataset(ref['path'],format='parquet')
    batches=[pairs[i:i+EXACT_PAIR_BATCH_SIZE]for i in range(0,len(pairs),EXACT_PAIR_BATCH_SIZE)]
    counts=[];a['selected_key_count_attempted']=True
    for batch in batches:
        predicate=None
        for day,code in batch:
            term=(ds.field('date')==pd.Timestamp(day).to_pydatetime())&(ds.field('code')==code)
            predicate=term if predicate is None else predicate|term
        counts.append(dataset.count_rows(filter=predicate))
    count=sum(counts)
    a.update(selected_key_count_returned=True,selected_matched_panel_rows=count,
        selected_predicate_batches=len(batches),selected_predicate_max_pairs=EXACT_PAIR_BATCH_SIZE)
    if old.support.stat_identity(ref['path'])!=before:raise Blocked('SELECTED_PANEL_COUNT_IDENTITY_DRIFT')
    if count>EXP_CAPS['max_selected_panel_rows']:raise Blocked('SELECTED_PANEL_PREDECODE_ROW_BOUND')
    parts=[];a['label_source_attempted']=True
    for batch,expected in zip(batches,counts):
        filters=[[('date','==',pd.Timestamp(d)),('code','==',code)]for d,code in batch]
        part=adapter.frame(ref,columns=['date','code','close'],filters=filters,max_rows=c['panel_row_binding']['file_rows'])
        if len(part)!=expected:raise Blocked('SELECTED_PANEL_COUNT_OR_KEYS_CONFLICT')
        parts.append(part)
    panel=pd.concat(parts,ignore_index=True)if parts else pd.DataFrame(columns=['date','code','close'])
    panel.date=pd.to_datetime(panel.date)
    actual={(old._day(d).isoformat(),code)for d,code in panel[['date','code']].itertuples(index=False,name=None)}
    if len(panel)!=count or panel.duplicated(['date','code']).any()or not actual.issubset(pairs):raise Blocked('SELECTED_PANEL_COUNT_OR_KEYS_CONFLICT')
    if old.support.stat_identity(ref['path'])!=before:raise Blocked('SELECTED_PANEL_READ_IDENTITY_DRIFT')
    return panel


def recovered_selections(adapter,c,fold,calendar,spec):
    """Only the explicitly frozen durable selection is reusable, never recompute it."""
    if 'resume_ref'not in c:return None
    manifest=adapter.metadata(c['resume_ref'])
    if manifest['status']!='FROZEN_SELECTION_ONLY_RECOVERY'or set(manifest['selections'])!={'2023H2'}:
        raise Blocked('RECOVERY_MANIFEST_SCOPE_CONFLICT')
    previous=adapter.metadata(manifest['previous_config_ref'])
    interrupted=adapter.metadata(manifest['interruption_handoff_ref'])
    if interrupted['status']!='HARD_INTERRUPTED'or previous['execution_id']!=EXP_EXECUTION:
        raise Blocked('RECOVERY_PREVIOUS_EXECUTION_CONFLICT')
    for key in ('inputs','input_versions','inventory_ref','recipe_ref','sources','policy','panel_row_binding'):
        if previous[key]!=c[key]:raise Blocked('RECOVERY_INPUT_OR_POLICY_IDENTITY_DRIFT')
    if fold not in manifest['selections']:return None
    ref=manifest['selections'][fold]
    if ref!=interrupted['recovery_checkpoint']:raise Blocked('RECOVERY_CHECKPOINT_IDENTITY_CONFLICT')
    saved=adapter.metadata(ref);proof=saved['provenance'];records=saved['records'];dates=old.calendar_dates(calendar)
    nominal=[d.isoformat()for d in dates if '2023-07-01'<=d.isoformat()<='2023-12-31']
    if(saved['policy']!=EXP_POLICY or saved['outcomes_not_read_for_this_fold']is not True
            or proof['fold']!=fold or proof['input_version']!=spec['as_run_input_version']
            or [r['date']for r in records]!=nominal):raise Blocked('RECOVERY_SELECTION_SEMANTIC_CONFLICT')
    for k,v in spec['accepted_identity']['test'].items():
        if proof['full_pool_test_keys'].get(k)!=v:raise Blocked('RECOVERY_SCORE_IDENTITY_CONFLICT')
    old.validate_time_proof(proof['model_train_time'],old._day(nominal[0]))
    for r in records:
        pos=dates.index(old._day(r['date']))
        if r['fold']!=fold or r['calendar_index']!=pos or r['h2_date']!=dates[pos+2].isoformat():raise Blocked('RECOVERY_CALENDAR_CONFLICT')
        if r['selections']is not None:
            if set(r['selections'])!=set(EXP_POLICY['arms'])or any(len(x)!=10 or len(set(x))!=10 for x in r['selections'].values()):raise Blocked('RECOVERY_SELECTED_KEYS_CONFLICT')
    c.setdefault('_access',{}).setdefault('reused_selection_refs',[]).append(ref)
    return records,proof,ref


def compute_expansion(c,life):
    import numpy as np
    import pandas as pd
    a=c.setdefault('_access',{});adapter=old.BoundFiles(c,Session(life));recipe,prior=adapter.modules()
    reuse_diagnostic_bindings(adapter,c)
    raw_prior=adapter.metadata(c['_raw_prerequisite_ref'])
    if raw_prior['status']!='RAW_EIGHT_TARGETS_DIAGNOSED'or len(raw_prior['rows'])!=8:raise Blocked('RAW_DIAGNOSTIC_NOT_COMPLETED')
    with life.adapter._locked():
        _,events=life.adapter._read_validated()
        prior_events=[e for e in events if e.get('execution_id')==RAW_EXECUTION]
        if len(prior_events)!=2 or sorted(e['event_type']for e in prior_events)!=['RUN_STARTED','TERMINAL']or next(e for e in prior_events if e['event_type']=='TERMINAL')['decision']!='RAW_EIGHT_TARGETS_DIAGNOSED':raise Blocked('RAW_DIAGNOSTIC_REGISTRY_NOT_CLOSED')
    a['raw_diagnostic_ref']=c['_raw_prerequisite_ref']
    inventory=adapter.metadata(c['inventory_ref']);delta=adapter.metadata(c['version_delta_ref'])
    if delta['pooled_primary']!=list(PRIMARY)or [x['fold']for x in inventory['cells']]!=['2023H2',*PRIMARY,'2026H1','2026H2_PARTIAL_TO_2026_10_15']:raise Blocked('FIXED_INVENTORY_OR_VERSION_STRATA_CONFLICT')
    cells={x['fold']:x for x in inventory['cells']}
    if any(cells[f]['as_run_input_version']!='ORIGINAL_M0_FORMAL_V1_EXACT_S1_PREPARATION_PARITY'for f in PRIMARY):raise Blocked('POOLED_INPUT_VERSION_CONFLICT')
    calendar=adapter.frame(c['inputs']['calendar'],columns=['date','calendar_index']);calendar.date=pd.to_datetime(calendar.date);dates=old.calendar_dates(calendar)
    annual=adapter.frame(c['inputs']['industry_year']);monthly=adapter.frame(c['inputs']['month_year']);roster=adapter.frame(c['inputs']['roster'])
    lookup=prior.prior_lookup(annual,monthly,[2023,2024,2025,2026]);by_fold={};provenance={};selection_refs={};trigger_counts={}
    rowproof=adapter.metadata(c['panel_row_binding']['preparation_ref']);panel_ref=c['inputs']['panel']
    if rowproof['status']!='COMPLETED'or rowproof['panel_rows']!=c['panel_row_binding']['file_rows']or rowproof['cache_files'].get(panel_ref['path'])!=panel_ref['sha256']:raise Blocked('EXISTING_PANEL_BINDING_CONFLICT')
    root=Path(c['run_dir'])
    def sink(name,value):
        p=root/name
        with p.open('x')as handle:handle.write(canonical(old.jsonable(value))+'\n');handle.flush();os.fsync(handle.fileno())
        return dict(path=str(p),sha256=sha(p))
    for fold in EXP_FOLDS:
        recovered=recovered_selections(adapter,c,fold,calendar,cells[fold])
        if recovered is None:
            frame,proof=load_saved_scores(adapter,c,cells[fold],calendar,recipe)
            if len(frame)>EXP_CAPS['max_segment_rows']:raise Blocked('SAVED_SCORE_SEGMENT_ROW_BOUND')
            if recipe.key_frame_identity(frame)!=proof['full_pool_test_keys']:raise Blocked('EXACT_SAVED_FULLPOOL_KEYS_CONFLICT')
            old.validate_time_proof(proof['model_train_time'],old._day(frame.date.min()))
            records=select_ac(frame,calendar,lookup,roster,prior,fold);del frame
            selected=sink(fold+'_SELECTIONS.json',dict(policy=EXP_POLICY,provenance=proof,records=records,outcomes_not_read_for_this_fold=True))
        else:records,proof,selected=recovered
        trigger_counts[fold]=dict(changed_dates=sum(bool(r.get('month_rank_changed'))for r in records),positive_bonus_candidates=sum(r.get('positive_bonus_candidates',0)for r in records))
        selection_refs[fold]=selected
        counts={arm:sum(len(r['selections'][arm])for r in records if r['selections'])for arm in EXP_POLICY['arms']}
        a.setdefault('selection_slots_by_arm',{arm:0 for arm in EXP_POLICY['arms']})
        for arm,v in counts.items():a['selection_slots_by_arm'][arm]+=v
        a['selection_slots']=sum(a['selection_slots_by_arm'].values());a['nominal_dates']=a.get('nominal_dates',0)+len(records)
        keys=sorted({(r['date'],code)for r in records if r['reason']is None for codes in r['selections'].values()for code in codes})
        requested=sorted({r['h2_date']for r in records if r['reason']is None});a['requested_target_dates']=sorted(set(a.get('requested_target_dates',[]))|set(requested))
        if sha(selected['path'])!=selected['sha256']:raise Blocked('SELECTION_CHANGED_BEFORE_LABEL_READ')
        outcomes={}
        if keys:
            # One trading-date/code predicate set per fold, before any close materialize.
            pairs=sorted({(day,code)for signal,code in keys for day in(signal,dates[dates.index(old._day(signal))+2].isoformat())})
            panel=read_selected_panel(adapter,c,pairs)
            target_keys={(r['h2_date'],code)for r in records if r['reason']is None for codes in r['selections'].values()for code in codes}
            a['returned_target_dates']=sorted(set(a.get('returned_target_dates',[]))|{old._day(d).isoformat()for d,code in panel[['date','code']].itertuples(index=False,name=None)if(old._day(d).isoformat(),code)in target_keys})
            selected_keys=pd.DataFrame(keys,columns=['date','code']);selected_keys.date=pd.to_datetime(selected_keys.date)
            labels=recipe.build_close_return_labels(panel,[2],calendar=calendar,signal_keys=selected_keys)[2]
            outcomes={(old._day(d).isoformat(),code):float(v)if np.isfinite(v)else None for d,code,v in labels[['date','code','label_return']].itertuples(index=False,name=None)}
        evaluated=evaluate_ac(records,outcomes);by_fold[fold]=evaluated;provenance[fold]=proof
        a.setdefault('label_folds_completed',[]).append(fold);a['valid_dates']=a.get('valid_dates',0)+sum(r['valid']for r in evaluated)
        checkpoint=sink(fold+'_PAIRED_RESULTS.json',dict(selection_ref=selected,rows=evaluated,summary=ac_stats(evaluated)))
        life.checkpoint(checkpoint,fold)
    result=stability_summary(by_fold)
    # Development H1 is reused from its already independently read-back result, no repeated score/label access.
    seen=adapter.metadata(c['seen_actual_ref']);seen_review=adapter.metadata(c['seen_actual_review_ref'])
    if seen_review['verdict']!='READBACK_AND_REGISTRY_CHAIN_PASS_WITH_LIMITATIONS; BUSINESS_ACCEPTANCE_CANNOT_CONFIRM':raise Blocked('SEEN_DEVELOPMENT_REVIEW_BINDING_CONFLICT')
    seen_rows=[dict(date=r['date'],valid=r['valid'],reason=r['reason'],missing_codes=r['missing_codes'],returns={arm:r['returns'][arm]for arm in EXP_POLICY['arms']}if r['valid']else None)for r in seen['dates']]
    result['seen_2026H1']=dict(status='REUSED_EXISTING_DEVELOPMENT_RESULT',source=c['seen_actual_ref'],summary=ac_stats(seen_rows),quarters={f'2026Q{q}':ac_stats([r for r in seen_rows if(int(r['date'][5:7])-1)//3+1==q])for q in [1,2]})
    result.update(rows_by_fold=by_fold,provenance=provenance,selection_refs=selection_refs,input_bindings=adapter.bound,selection_slots_by_arm=a['selection_slots_by_arm'],mechanism_triggers=trigger_counts)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--activation');p.add_argument('--execute',action='store_true');p.add_argument('--stage',choices=('diagnostic','raw','expansion','expansion_resume'),default='diagnostic');a=p.parse_args();c=(metadata_config(a.config)if a.stage=='diagnostic'else phase_config(a.config,RAW_EXECUTION,'RAW_TARGET_02',CAPS,RAW_POLICY)if a.stage=='raw'else phase_config(a.config,RESUME_EXECUTION,'EXPANSION_RESUME_05',EXP_CAPS,EXP_POLICY)if a.stage=='expansion_resume'else phase_config(a.config,EXP_EXECUTION,'EXPANSION_03',EXP_CAPS,EXP_POLICY))
    if not a.execute:print(canonical(dict(status='MONTH_BONUS_METADATA_ONLY_NO_SCIENCE')));return
    if not a.activation:p.error('--activation required')
    report=(run(c,resources.read_json(a.activation))if a.stage=='diagnostic'else run(c,resources.read_json(a.activation),compute_fn=raw_target_flags,lifecycle_type=RawLifecycle,approval='APPROVED_FOR_RAW_EIGHT_TARGET_DIAGNOSTIC')if a.stage=='raw'else run(c,resources.read_json(a.activation),compute_fn=compute_expansion,lifecycle_type=ExpansionLifecycle,approval='APPROVED_FOR_FIXED_AC_EXPANSION'));print(canonical({k:report[k]for k in ('status','screen_passed','elapsed_seconds','sampled_peak_rss_bytes','reason_counts','new_fit_count','new_predict_count','network_calls')if k in report}))

if __name__=='__main__':main()
