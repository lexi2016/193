"""One fixed four-arm historical opportunity canary; no fit/predict/network/NAV."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import math
import os
from pathlib import Path
import statistics
import time

import environment_tilt_evaluation as old
import environment_tilt_mac_recovery as resources
from environment_tilt_mac_registry import Lifecycle, FLAGS, canonical, load_adapter, sha

POLICY = dict(fold='2026H1',start='2026-01-01',end='2026-06-30',horizon=2,
    weak='CSI300_CLOSE_LT_MA20_T_CLOSE',rank_scale=100,bonus=5,topk=50,topn=10,
    month='ORIGINAL_S1_TPLUS1_MONTH_COMPLETE_PRIOR_YEARS_KAPPA100',
    cash='CANCEL_NEW_INDEPENDENT_H2_BASKET_ONLY',mean_lift_min=.0008,
    positive_quarters=['2026Q1','2026Q2'],cvar5_difference_min=0,
    scope='OPENBOOK_GROSS_OVERLAPPING_H2_OPPORTUNITY_NOT_PORTFOLIO',**FLAGS)
CAPS=dict(threads=4,rss_bytes=4*2**30,total_wall_sec=300,payload_bytes=32*2**20,
          min_memory_bytes=8*2**30,min_disk_bytes=2*2**30,
          max_payload_file_bytes=2*2**30,max_json_record_bytes=3*2**20,
          max_aux_rows=2000000,max_segment_rows=700000,max_panel_rows=2000000)
ARMS=('A_ORIGINAL','B_WEAK_GATE','C_MONTH_BONUS','D_COMBINED')


class Blocked(ValueError):pass


def _day(value):return old._day(value).isoformat()


def weak_states(index,calendar):
    """Same original20-session mean and BELOW definition; never forward-fill."""
    import pandas as pd
    import numpy as np
    if set(index.columns)!={'date','code','close'}or index.code.nunique()!=1 or index.code.iloc[0]!='SH.000300':raise Blocked('EXACT_CSI300_SCHEMA_CODE')
    frame=index.copy();frame['date']=pd.to_datetime(frame.date)
    if frame.date.duplicated().any()or not np.isfinite(frame.close).all()or(frame.close<=0).any():raise Blocked('INVALID_BOUND_INDEX')
    value=frame.set_index('date').close.reindex(pd.DatetimeIndex(calendar.date));ma=value.rolling(20,min_periods=20).mean()
    return {_day(t):(None if pd.isna(v)or pd.isna(m)else bool(v<m))for t,v,m in zip(value.index,value,ma)}


def fixed_rank(group,prior):
    """Stable fullpool Top50 -> fixed rank points+5 -> Top10, no outcome input."""
    ordered=group.sort_values(['prediction','code'],ascending=[False,True],kind='mergesort').head(50)
    n=len(ordered);base_codes=ordered.code.tolist()[:10];ranked=[]
    for rank,row in enumerate(ordered.itertuples(index=False),1):
        p,missing=prior[row.code]
        if missing not in (0.,1.)or not math.isfinite(p):raise Blocked('INVALID_ORIGINAL_PRIOR')
        points=100.*(n-rank)/(n-1)if n>1 else 100.
        bonus=5 if missing==0 and p>0 else 0
        ranked.append(dict(code=row.code,original_rank=rank,rank_points=points,bonus=bonus,adjusted_points=points+bonus))
    ranked.sort(key=lambda r:(-r['adjusted_points'],r['original_rank'],r['code']))
    return base_codes,[r['code']for r in ranked[:10]],ranked


def select_all(predictions,calendar,states,projection):
    """Retain every nominal date, including no-score/unknown dates. No labels."""
    import pandas as pd
    import numpy as np
    if not {'date','code','prediction'}.issubset(predictions)or predictions.duplicated(['date','code']).any()or not np.isfinite(predictions.prediction).all():raise Blocked('INVALID_FULLPOOL_SCORE_KEYS_OR_VALUES')
    days=[t for t in calendar.date if POLICY['start']<=_day(t)<=POLICY['end']]
    if set(predictions.date)-set(days):raise Blocked('SCORE_OUTSIDE_FROZEN_HALF_YEAR')
    groups={_day(t):g for t,g in predictions.groupby('date',sort=False)}
    prior={(_day(r.date),r.code):(float(getattr(r,'industry_month_log_relative_lift')),float(getattr(r,'industry_month_missing')))for r in projection.itertuples(index=False)}
    if len(prior)!=len(projection):raise Blocked('AMBIGUOUS_PRIOR_KEYS')
    dates=old.calendar_dates(calendar);out=[]
    for value in days:
        day=_day(value);group=groups.get(day);weak=states.get(day);pos=dates.index(old._day(value));h2=dates[pos+2].isoformat()if pos+2<len(dates)else None
        reason='NO_EXISTING_SCORE_DATE_NOT_CASH'if group is None else 'LESS_THAN_TEN_ORIGINAL_POOL_ROWS'if len(group)<10 else 'UNKNOWN_INDEX_STATE_NOT_CASH'if weak is None else 'H2_CALENDAR_UNAVAILABLE'if h2 is None else None
        record=dict(date=day,h2_date=h2,weak=weak,full_pool_rows=0 if group is None else len(group),reason=reason,selections=None)
        if reason is None:
            top=group.sort_values(['prediction','code'],ascending=[False,True],kind='mergesort').head(50)
            p={code:prior[(day,code)]for code in top.code}
            base,candidate,ranked=fixed_rank(group,p)
            record.update(selections=dict(zip(ARMS,(base,[]if weak else base,candidate,[]if weak else candidate))),ranked_top50=ranked,month_rank_changed=base!=candidate)
        out.append(record)
    return out


def selected_keys(records):
    return sorted({(r['date'],c)for r in records if r['selections']for codes in r['selections'].values()for c in codes})


def evaluate(records,outcomes):
    """Cash0; unknown is not0. Screen requires entire nominal calendar coverage."""
    from datetime import date
    rows=[]
    for record in records:
        values={};missing=[]
        if record['reason']is None:
            for arm,codes in record['selections'].items():
                if not codes:values[arm]=0.;continue
                if len(codes)!=10:raise Blocked('TEN_STOCK_DENOMINATOR_CHANGED')
                observed=[outcomes.get((record['date'],c))for c in codes]
                missing.extend(c for c,v in zip(codes,observed)if v is None or not math.isfinite(v))
                values[arm]=None if any(v is None or not math.isfinite(v)for v in observed)else statistics.mean(observed)
        valid=record['reason']is None and not missing
        rows.append(dict(date=record['date'],quarter=record['date'][:4]+('Q1'if int(record['date'][5:7])<=3 else 'Q2'),
            valid=valid,reason=record['reason']or('MISSING_SELECTED_H2_LABEL_ENTIRE_DATE_EXCLUDED'if missing else None),
            missing_codes=sorted(set(missing)),returns=values if valid else None))
    good=[r for r in rows if r['valid']];full=bool(rows)and len(good)==len(rows)
    arm_stats={}
    for arm in ARMS:
        vals=[r['returns'][arm]for r in good]
        arm_stats[arm]=dict(same_mask_mean=statistics.mean(vals)if vals else None,
            full_calendar_mean=statistics.mean(vals)if vals and full else None,
            cvar5=old.core.fractional_cvar5(vals)if vals else None,worst_h2=min(vals)if vals else None,
            negative_days=sum(v<0 for v in vals),valid_dates=len(vals),
            stock_slots=sum(len(r['selections'][arm])for r in records if r['selections']),
            cash_dates=sum(r['selections']is not None and not r['selections'][arm]for r in records))
    for arm in ARMS:
        arm_stats[arm]['same_mask_lift_vs_original']=statistics.mean(r['returns'][arm]-r['returns']['A_ORIGINAL']for r in good)if good else None
        arm_stats[arm]['full_calendar_lift_vs_original']=arm_stats[arm]['same_mask_lift_vs_original']if full else None
        arm_stats[arm]['quarter_lifts_vs_original']={q:statistics.mean(r['returns'][arm]-r['returns']['A_ORIGINAL']for r in good if r['quarter']==q)if any(r['quarter']==q for r in good)else None for q in POLICY['positive_quarters']}
    delta={r['date']:r['returns']['D_COMBINED']-r['returns']['A_ORIGINAL']for r in good}
    quarter={q:statistics.mean([delta[r['date']]for r in good if r['quarter']==q])if any(r['quarter']==q for r in good)else None for q in POLICY['positive_quarters']}
    mean=statistics.mean(delta.values())if delta else None
    tail=(arm_stats['D_COMBINED']['cvar5']-arm_stats['A_ORIGINAL']['cvar5'])if good else None
    weak=sum(r['weak']is True and r['selections']is not None for r in records);changed=sum(bool(r.get('month_rank_changed'))for r in records)
    passed=full and mean>=.0008 and all(v is not None and v>0 for v in quarter.values())and tail>=0 and weak>0 and changed>0
    reason='INCOMPLETE_FULL_CALENDAR_NOT_A_PASS'if not full else 'MECHANISM_NOT_EXERCISED'if not weak or not changed else 'FROZEN_SCREEN_MET'if passed else 'FROZEN_SCREEN_NOT_MET_NO_RETUNING'
    hac=old.core.calendar_hac20([date.fromisoformat(r['date'])for r in records],{date.fromisoformat(k):v for k,v in delta.items()})
    return dict(status='SMALL_SAMPLE_WORTH_EXPANDING'if passed else 'CANNOT_CONFIRM_SCREEN'if not full else 'SMALL_SAMPLE_NOT_YET_WORTH_EXPANDING',
        historical_classification='DIRECTION_ONLY',screen_passed=passed,screen_reason=reason,
        nominal_dates=len(rows),valid_common_dates=len(good),full_calendar_complete=full,
        primary_full_calendar_lift=mean if full else None,descriptive_common_mask_lift=mean,
        quarter_lifts=quarter,cvar5_difference=tail,weak_trigger_dates=weak,month_rank_change_dates=changed,
        arms=arm_stats,working_hac=old.jsonable(old.asdict(hac)),dates=rows,
        nav=None,net_return=None,portfolio_max_drawdown=None,fit_count=0,network_calls=0,**FLAGS)


class Session:
    def __init__(self,life):self.life=life
    def require_active(self):return self.life.require()


class RuleLifecycle(Lifecycle):
    def event(self,*args):
        e=super().event(*args);access=dict(self.config.get('_access',{}));started=e['event_type']=='RUN_STARTED'
        truth=('NOT_READ' if started else 'SELECTED_H2_OUTCOME_SOURCE_ACCESSED' if access.get('selected_label_source_entered')
               else 'EXISTING_SCORE_CHECKPOINT_ACCESSED_NO_OUTCOMES' if access.get('score_checkpoint_entered')
               else 'ACCEPTANCE_METADATA_ONLY_NO_SCORE_OR_OUTCOMES')
        e.update(route_id='a-share-simple-rule-canary-2026h1',experiment_id='A_SHARE_SIMPLE_RULE_CANARY_14',stage='FIXED_RULE_OPPORTUNITY_EVALUATION',arm='FOUR_FIXED_ARMS',one_variable='FIXED_BELOW_MA20_GATE_AND_MONTH_INDUSTRY_PLUS5_ABLATIONS',evidence_level='DIRECTION_ONLY',sealed_status='EXPOSED_HISTORICAL_HALF_YEAR_NO_SEALED_CLAIM',truth_access=truth)
        e['input_identity'].update(access={}if started else access,existing_score=self.config.get('score'),selection_ref=access.get('selection_ref'))
        e['checkpoint_ref']=None if started else access.get('existing_checkpoint_ref')
        e['target_date_range']=(dict(count=0,scope='NO_OUTCOME_TARGET_ACCESS_BEFORE_START')if started else access.get('selected_h2_target_range',dict(count=0,scope='NO_SELECTED_OUTCOME_SOURCE_ACCESSED')))
        reasons=e['input_identity']['null_reasons'];reasons['truth_positive_denominator']='NO_CLASSIFICATION_POSITIVE_TARGET'
        if e['checkpoint_ref']is None:reasons['checkpoint_ref']='NO_VERIFIED_EXISTING_CHECKPOINT_READ_YET'
        else:reasons.pop('checkpoint_ref',None)
        if not started and access.get('stock_slots_total')is not None:reasons.pop('selection_slots',None)
        e['deterministic_metrics'].update(truth_positive_denominator=None,selection_slots=None if started else access.get('stock_slots_total'),
            selection_slots_by_arm=None if started else access.get('stock_slots_by_arm'),
            nominal_opportunity_dates=None if started else access.get('nominal_dates'),valid_common_dates=None if started else access.get('valid_dates'),
            requested_unique_selected_label_keys=None if started else access.get('requested_label_keys'),
            finite_unique_selected_h2_labels=None if started else access.get('finite_label_keys'),
            positive_selected_h2_labels=None if started else access.get('positive_label_keys'),
            new_fit_count=0,new_predict_count=0,existing_checkpoint_unpacks=0 if started else int(access.get('score_checkpoint_unpacked',False)))
        return e


def metadata_config(path):
    c=resources.read_json(path);c.update(config_path=str(Path(path).resolve()),config_sha256=sha(path))
    if c['policy']!=POLICY or c['caps']!=CAPS or c['fit_count']!=0 or c['network_allowed']is not False:raise Blocked('FROZEN_SMALL_SCOPE_CHANGED')
    root=Path(c['run_dir'])
    output_names={'SIMPLE_RULE_2026H1_20261021_01':'REGISTERED_01','SIMPLE_RULE_2026H1_20261021_02':'REGISTERED_02','SIMPLE_RULE_2026H1_20261021_03':'REGISTERED_03'}
    if c['execution_id']not in output_names or root!=Path(c['scope_root'])/output_names[c['execution_id']] or root.is_symlink()or any(p.is_symlink()for p in root.parents):raise Blocked('OWNED_OUTPUT_PATH_REQUIRED')
    for ref in [*c['sources'].values(),*c['code_refs'].values(),c['contract_ref']]:
        if sha(ref['path'])!=ref['sha256']:raise Blocked('SOURCE_OR_CONTRACT_SHA_DRIFT')
    for name in ('recipe','prior'):
        if Path(c['sources'][name]['path']).suffix!='.py':raise Blocked('BOUND_ORIGINAL_SOURCE_REQUIRED')
    if c['code_refs']['runner']['path']!=str(Path(__file__).resolve()):raise Blocked('RUNNER_PATH_DRIFT')
    c.update(code_composite_sha256=old.support.digest(c['code_refs']),contract_path=c['contract_ref']['path'],contract_sha256=c['contract_ref']['sha256'])
    return c


def score_acceptance(adapter,c,sidecar):
    """Existing reviewed manifest/replay chain; metadata only, before score payload."""
    spec=c['score'];manifest=adapter.metadata(spec['acceptance_manifest'])
    cells=[v for v in manifest['accepted_cell_candidates']if v.get('fold')=='2026H1']
    if len(cells)!=1:raise Blocked('EXACT_ACCEPTED_CELL_MANIFEST_REQUIRED')
    cell=cells[0]
    if (cell.get('artifact')!=spec['artifact']or cell.get('metadata')!=spec['metadata']
            or cell.get('as_run_input_version')!=spec['input_version']
            or manifest['accepted_cell_common_rules']['old_outer_acceptance_ref']!=spec['original_accepted_review']):raise Blocked('ACCEPTED_CELL_EXACT_BINDING_DRIFT')
    review=adapter.metadata(spec['original_accepted_review']);output=adapter.metadata(spec['accepted_output_manifest']);replay=adapter.metadata(spec['checkpoint_replay']);stage=adapter.metadata(spec['m0_stage_manifest'])
    stage_cells=[v for v in stage['cell_results']if v.get('cell')=='M0_FORMAL_M0_H2_2026H1_attempt_01']
    if len(stage_cells)!=1 or stage_cells[0].get('checkpoint_path')!=spec['artifact']['path']or stage_cells[0].get('status')!='COMPLETED'or stage_cells[0].get('fit_called')is not True or stage['contract_sha256']!=cell['as_run_contract_sha256']:raise Blocked('ORIGINAL_COMPLETED_M0_STAGE_BINDING_CONFLICT')
    identity=sidecar['identity']
    if (review.get('verdict')!='METRICS_CONSISTENT' or review.get('research_decision')!='M4_GATE_FAILED' or review['artifacts']['contract_sha256']!=cell['as_run_contract_sha256']
            or review['artifacts']['output_manifest_sha256']!=spec['accepted_output_manifest']['sha256']
            or output['contract_sha256']!=cell['as_run_contract_sha256']
            or identity['contract_sha256']!=cell['as_run_contract_sha256']or identity['config_sha256']!=cell['as_run_config_sha256']):raise Blocked('EXISTING_ACCEPTED_REVIEW_CHAIN_CONFLICT')
    if (replay.get('status')!='VERIFIED'or replay.get('schema_version')!='A_SHARE_SAVED_CHECKPOINT_REPLAY_V1'
            or replay.get('checkpoint_path')!=spec['artifact']['path']or replay.get('checkpoint_sha256')!=spec['artifact']['sha256']
            or replay.get('checkpoint_metadata_path')!=spec['metadata']['path']or replay.get('checkpoint_metadata_sha256')!=spec['metadata']['sha256']
            or replay.get('data_identity')!=identity['data_identity']or replay.get('config_sha256')!=identity['config_sha256']
            or replay.get('prediction_identity')!=replay.get('replay_prediction_identity')
            or replay.get('top10_parity',{}).get('status')!='EXACT'
            or replay.get('comparison',{}).get('key_order_equal')is not True or replay.get('comparison',{}).get('prediction_equal')is not True):raise Blocked('EXACT_SAVED_SCORE_REPLAY_BINDING_CONFLICT')
    return dict(fold='2026H1',input_version=spec['input_version'],artifact=spec['artifact'],metadata=spec['metadata'],accepted_review=spec['original_accepted_review'],manifest=spec['acceptance_manifest'],checkpoint_replay=spec['checkpoint_replay'],new_prediction_or_replay_called=False,existing_review_verdict=review['verdict'],existing_research_decision=review['research_decision'],business_promotion_claimed=False)


def read_selected_panel(adapter,c,start,end,codes):
    """File metadata and filtered decoded-row guards are different bounds."""
    bound=c['panel_row_binding'];proof=adapter.metadata(bound['preparation_ref']);ref=c['inputs']['panel']
    if (proof.get('status')!='COMPLETED'or proof.get('panel_rows')!=bound['file_rows']
            or proof.get('cache_files',{}).get(ref['path'])!=ref['sha256']):raise Blocked('EXISTING_PANEL_FILE_ROWS_BINDING_CONFLICT')
    panel=adapter.frame(ref,columns=['date','code','close'],
        filters=[('date','>=',start),('date','<=',end),('code','in',codes)],max_rows=bound['file_rows'])
    if len(panel)>c['caps']['max_panel_rows']:raise Blocked('SELECTED_PANEL_SUBSET_ROW_BOUND')
    return panel


def compute(c,life,sink):
    """All authentication afterSTART; selections durable before panel/labels."""
    import pandas as pd
    import joblib
    import numpy as np
    access=c.setdefault('_access',{});adapter=old.BoundFiles(c,Session(life));recipe,prior=adapter.modules()
    calendar=adapter.frame(c['inputs']['calendar']);calendar['date']=pd.to_datetime(calendar.date);dates=old.calendar_dates(calendar)
    sidecar=adapter.metadata(c['score']['metadata']);ident=sidecar['identity']
    if sidecar['artifact_sha256']!=c['score']['artifact']['sha256']or ident['fold']!='2026H1'or ident['stage']!='M0_FORMAL':raise Blocked('ORIGINAL_COMPLETED_SCORE_BINDING_REQUIRED')
    time_proof=old.restore_model_time(sidecar,dates,model_source_sha256=c['sources']['recipe']['sha256'],calendar_sha256=c['inputs']['calendar']['sha256'])
    time_proof['evidence_sha256']=c['score']['metadata']['sha256']
    acceptance=score_acceptance(adapter,c,sidecar);access['existing_acceptance_chain_verified']=True
    access['score_checkpoint_entered']=True
    before=adapter.authenticate(c['score']['artifact']);payload=joblib.load(c['score']['artifact']['path']);predictions=payload['predictions'].copy();del payload
    access.update(score_checkpoint_unpacked=True,existing_checkpoint_ref=dict(c['score']['artifact']))
    if old.support.stat_identity(c['score']['artifact']['path'])!=before:raise Blocked('CHECKPOINT_CHANGED_DURING_READ')
    predictions['date']=pd.to_datetime(predictions.date)
    if recipe.key_frame_identity(predictions)!=ident['data_identity']['test_keys']:raise Blocked('ORIGINAL_FULLPOOL_KEY_IDENTITY_DRIFT')
    old.validate_time_proof(time_proof,old._day(predictions.date.min()))
    index=adapter.frame(c['inputs']['index'],columns=['date','code','close']);states=weak_states(index,calendar)
    roster=adapter.frame(c['inputs']['roster']);annual=adapter.frame(c['inputs']['industry_year']);monthly=adapter.frame(c['inputs']['month_year'])
    lookup=prior.prior_lookup(annual,monthly,[2026]);top=predictions.sort_values(['date','prediction','code'],ascending=[True,False,True],kind='mergesort').groupby('date',sort=False).head(50)
    projection=prior.project_prior(top[['date','code']],roster,calendar,lookup)
    records=select_all(predictions,calendar,states,projection);del predictions
    selection=sink('FOUR_ARM_SELECTIONS.json',dict(policy=POLICY,records=records,model_train_time=time_proof,outcomes_not_read=True))
    # No future close payload is opened until this file has been durably saved.
    if sha(selection['path'])!=selection['sha256']:raise Blocked('SELECTION_FREEZE_DRIFT')
    access.update(selection_ref=selection,nominal_dates=len(records),stock_slots_by_arm={a:sum(len(r['selections'][a])for r in records if r['selections'])for a in ARMS})
    access['stock_slots_total']=sum(access['stock_slots_by_arm'].values())
    if c.get('selection_identity_ref'):
        adapter.authenticate(c['selection_identity_ref'])
        if selection['sha256']!=c['selection_identity_ref']['sha256']:raise Blocked('SELECTION_OR_CASH_FLAGS_DRIFT_FROM_EXECUTION02')
        access['selection_identity_vs_execution02']=True
    keys=selected_keys(records);access['requested_label_keys']=len(keys)
    if not keys:raise Blocked('NO_SELECTED_KEYS')
    selected=pd.DataFrame(keys,columns=['date','code']);selected['date']=pd.to_datetime(selected.date)
    dates_lookup={d:i for i,d in enumerate(dates)}
    target_days=[dates[dates_lookup[old._day(d)]+2]for d in selected.date]
    access['selected_h2_target_range']=dict(count=len(set(target_days)),start=min(target_days).isoformat(),end=max(target_days).isoformat(),scope='SELECTED_H2_OUTCOME_TARGETS_REQUESTED_NOT_ALLPOOL_TRUTH')
    access['selected_label_source_entered']=True
    panel=read_selected_panel(adapter,c,selected.date.min(),pd.Timestamp(max(target_days)),sorted(selected.code.unique()))
    panel['date']=pd.to_datetime(panel.date)
    labels=recipe.build_close_return_labels(panel,[2],calendar=calendar,signal_keys=selected)[2]
    outcomes={(d.strftime('%Y-%m-%d'),code):(float(v)if np.isfinite(v)else None)for d,code,v in labels[['date','code','label_return']].itertuples(index=False,name=None)}
    access.update(finite_label_keys=sum(v is not None for v in outcomes.values()),positive_label_keys=sum(v is not None and v>0 for v in outcomes.values()))
    result=evaluate(records,outcomes);access['valid_dates']=result['valid_common_dates'];result.update(selection_ref=selection,input_bindings=adapter.bound,model_train_time=time_proof,checkpoint_unpacks=1,score_acceptance=acceptance)
    return result


def run(c,proof,adapter=None,compute_fn=compute):
    import shutil
    if proof.get('status')!='APPROVED_FOR_ONE_FIXED_RULE_CANARY':raise Blocked('INDEPENDENT_ACTIVATION_REQUIRED')
    for key in ('config_sha256','contract_sha256','code_composite_sha256','execution_id'):
        if proof.get(key)!=c[key]:raise Blocked('ACTIVATION_IDENTITY_DRIFT')
    ref=proof['review_receipt']
    if sha(ref['path'])!=ref['sha256']:raise Blocked('REVIEW_RECEIPT_DRIFT')
    root=Path(c['run_dir']);root.mkdir(exist_ok=False);lock=(root/'owner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    life=RuleLifecycle(adapter or load_adapter(c),c);receipt=root/'RECEIPT.json';budget=resources.Budget(root,c['caps']);report={}
    def sink(name,value):
        p=root/name
        if p.exists()or p.is_symlink()or Path(name).name!=name:raise Blocked('NO_OUTPUT_OVERWRITE_OR_ESCAPE')
        data=(canonical(value)+'\n').encode()
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
        return dict(path=str(p),sha256=sha(p),bytes=len(data))
    try:
        resources.write_json(receipt,dict(status='PREPARED_NOT_STARTED',**FLAGS));life.start(receipt)
        with budget:
            if resources.available_memory()<CAPS['min_memory_bytes']or shutil.disk_usage(root).free<CAPS['min_disk_bytes']:raise Blocked('DEFERRED_RESOURCE')
            os.environ.update({k:'4'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')});c['arrow_pools']=resources.configure_arrow_pools(4)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):report=compute_fn(c,life,sink)
            budget.check()
    except BaseException as e:report=dict(status='STOPPED',error_type=type(e).__name__,error=str(e),fit_count=0,network_calls=0,**FLAGS)
    finally:
        report.update(access=dict(c.get('_access',{})),elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak,arrow_pools=c.get('arrow_pools'))
        resources.write_json(receipt,report)
        if life.started and not life.finished:life.finish(receipt,report['status'],dict(fits=0,network_calls=0,nominal_dates=report.get('nominal_dates'),valid_dates=report.get('valid_common_dates')))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--activation');p.add_argument('--execute',action='store_true');a=p.parse_args();c=metadata_config(a.config)
    if not a.execute:print(canonical(dict(status='SIMPLE_RULE_METADATA_ONLY_NO_EVALUATION')));return
    if not a.activation:p.error('--activation required')
    result=run(c,resources.read_json(a.activation));print(canonical({k:v for k,v in result.items()if k not in ('dates','input_bindings')}))


if __name__=='__main__':main()
