"""Bounded fresh Mac TDX captures through the original M19 producer/interface.

Default CLI validates source/metadata only. No old cache, model, label or
historical SDK is consumed. Fresh canonical units do not establish the old
model's effective numeric amount scale.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import localcontext
import fcntl
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import environment_tilt_full_prefix as original

base, registry = original.base, __import__('environment_tilt_mac_registry')
FLAGS = registry.FLAGS
DATES = (20210802, 20220104, 20230103, 20240102, 20261015)
TARGETS = (('000002.SZ',20051201),('600004.SH',20051219),('000014.SZ',20070630))
REQUESTS = tuple(dict(category=9,market=1,code='600000',start=start,count=800) for start in (0,800))
CAPS = dict(threads=4,rss_bytes=2**31,total_wall_sec=240,payload_bytes=16*2**20,
            min_memory_bytes=8*2**30,min_disk_bytes=2**30)
POLICY = dict(symbol='600000.SH',cutoff=20261015,fixed_amount_dates=list(DATES),
              bar_requests=list(REQUESTS),action_targets=[list(v) for v in TARGETS],
              max_query_calls=5,retry=False,query_wall_seconds=30,
              raw_amount_unit='CNY',raw_volume_unit='lots',volume_to_shares=100,
              ancient_unresolved_does_not_fail_capture=True,old_model_bridge=False)


class Stop(base.DiagnosticError): pass


def normalize_amounts(pages):
    """Original endpoint units; no old-amount fitting or rounding."""
    rows={}
    for page in pages:
        if not isinstance(page,list):raise Stop('DAILY_RESPONSE_NOT_LIST')
        for row in page:
            if not isinstance(row,dict):raise Stop('DAILY_ROW_SCHEMA')
            values=[row[k] for k in ('year','month','day')]
            if any(type(v) is not int for v in values):raise Stop('DAILY_DATE_TYPE')
            dt=date(*values);key=dt.year*10000+dt.month*100+dt.day
            # Later prices/amounts are not inspected or used.
            if key>POLICY['cutoff']:continue
            if key in rows:raise Stop('DUPLICATE_DAILY_KEY')
            values={k:row[k] for k in ('open','high','low','close','vol','amount')}
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
                raise Stop('NONFINITE_OR_NONNUMERIC_DAILY_FIELD')
            if any(values[k]<=0 for k in ('open','high','low','close')) or values['vol']<0 or values['amount']<0:
                raise Stop('INVALID_DAILY_FIELD')
            shares=values['vol']*100
            if not math.isfinite(shares):raise Stop('NONFINITE_VOLUME_SHARES')
            rows[key]=dict(trade_date=key,ts_code=POLICY['symbol'],**{k:values[k] for k in ('open','high','low','close')},
                amount_CNY=values['amount'],raw_volume_lots=values['vol'],volume_shares=shares)
    missing=[d for d in DATES if d not in rows]
    if missing:raise Stop('MISSING_FIXED_AMOUNT_DATES:'+','.join(map(str,missing)))
    if any(rows[d]['amount_CNY']<=0 for d in DATES):raise Stop('FIXED_AMOUNT_NOT_POSITIVE')
    return dict(symbol=POLICY['symbol'],cutoff=POLICY['cutoff'],fixed_rows=[rows[d] for d in DATES],
        retained_date_min=min(rows),retained_date_max=max(rows),retained_rows=len(rows),
        units=dict(amount='CNY_UNCHANGED_FROM_ENDPOINT',volume='SHARES_RAW_LOTS_TIMES100'),
        old_model_numeric_scale='NOT_INFERRED',old_IEEE_quantization_gate_reused=False)


def summarize_actions(symbol,target,events):
    matching=[e for e in events if e['_date_int']==target]
    output=[]
    for event in matching:
        row=event['original_record'];category=event['category']
        item=dict(category=category,original_record=row,standard_factor_status='UNRESOLVED_UNSUPPORTED_TERMS',
            native_factor=None,native_factor_reason='NOT_RETURNED_BY_ORIGINAL_GET_XDXR_INFO')
        fields=('fenhong','peigujia','songzhuangu','peigu')
        missing=[key for key in fields if key not in row or row[key] is None]
        item['missing_standard_terms']=missing
        if category==1 and not missing:
            try:
                mapped=dict(category=1,**dict(zip(base.ACTION_FIELDS,(row[k] for k in fields))))
                with localcontext() as ctx:
                    ctx.prec=50
                    a,b=original.affine.event_affine(mapped)
                item.update(standard_factor_status='STANDARD_AFFINE_OPERATOR_COMPUTED',
                    a=str(a),b=str(b),formula='Pex=a*Pprev+b; Pprev not fetched for these ancient dates',
                    native_QFQ_certified=False)
            except base.DiagnosticError as exc:
                item.update(standard_factor_status='UNRESOLVED_INVALID_STANDARD_TERMS',error=str(exc))
        elif category in (13,14):
            item['minimal_missing_info']='Warrant count/strike are not share rights or an event price reference; supported economic cash/share terms or vendor factor still unavailable.'
        elif category==4:
            item['minimal_missing_info']='Capital-change record alone does not establish an applicable price adjustment or no-op.'
        output.append(item)
    return dict(symbol=symbol,date=target,same_date_records=output,record_count=len(output),
        same_date_category1_count=sum(e['category']==1 for e in matching),
        empty_is_not_identity_factor=True,semantic_resolution_required_for_future_prefix=True,
        raw_capture_acceptance_independent_of_terms=True)


@contextmanager
def original_session(config):
    """Same bound DLI/pytdx root as original m11.query_events; no alternate API."""
    for ref in config['query_sources'].values():
        if registry.sha(ref['path'])!=ref['sha256']:raise Stop('QUERY_SOURCE_DRIFT')
    root=Path(config['query_package_root']).resolve()
    for name,mod in list(sys.modules.items()):
        if name=='pytdx' or name.startswith('pytdx.'):
            if not getattr(mod,'__file__',None) or root not in Path(mod.__file__).resolve().parents:
                raise Stop('PYTDX_IMPORT_COLLISION')
    sys.path.insert(0,str(root))
    try:
        ref=config['query_sources']['interface']
        spec=importlib.util.spec_from_file_location('m33_bound_original_DLI',ref['path'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with module.tdx_hq_session(host=config['query_host'],port=config['query_port'],multithread=False) as api:
            yield api
    finally:sys.path.pop(0)


def fetch_bar_page(life,config,budget,request,index,evidence,session_factory=None):
    life.require()
    if request!=REQUESTS[index] or evidence['bar_query_calls']>=2:raise Stop('BAR_REQUEST_SCOPE')
    factory=session_factory or original_session
    with original.m11.query_wall(budget,POLICY['query_wall_seconds']):
        evidence['bar_connect_calls']+=1
        with factory(config) as api:
            if api.auto_retry or api.heartbeat:raise Stop('UNBOUNDED_API_CONFIGURATION')
            api.client.settimeout(5)
            evidence['bar_query_calls']+=1
            response=api.get_security_bars(**request)
        payload=json.dumps(dict(symbol=POLICY['symbol'],request=request,
            observed_at=datetime.now(timezone.utc).isoformat(),response=response),
            ensure_ascii=False,sort_keys=True,indent=2,allow_nan=True).encode()+b'\n'
        path=Path(config['run_dir'])/f'DAILY_PAGE_{index}_RAW_RESPONSE.json'
        binding=original.durable_new(path,payload)
        evidence['bar_raw_responses'].append(binding)
        budget.check()
    return response


class RefreshLifecycle(registry.Lifecycle):
    def event(self,kind,decision,receipt,metrics):
        event=super().event(kind,decision,receipt,metrics)
        event.update(route_id='a-share-environment-tilt-m33-original-tdx-source-refresh',
            experiment_id='A_SHARE_ENV_TILT_M33_TDX_SOURCE_REFRESH',stage='M33_FRESH_FIXED_SOURCE_CAPTURE',
            arm='FIVE_FIXED_AMOUNTS_THREE_ACTION_RESPONSES',baseline_id='M30_R1_BLOCKED_PRESERVED_SOURCE_REFRESH_ONLY',
            one_variable='FRESH_ORIGINAL_MAC_TDX_CNY_SHARES_AND_RAW_ACTIONS_NO_MODEL_BRIDGE',
            truth_access='NO_LABELS_MODELS_SCORES_OUTCOMES',sealed_status='NOT_ENTERED',
            evidence_level='OPENBOOK_DIAGNOSTIC_ONLY')
        return event


def load_config(path):
    c=base.read_json(path);c.update(config_path=str(Path(path).resolve()),config_sha256=registry.sha(path))
    if c['policy']!=POLICY or c['caps']!=CAPS or any(c.get(k)!=v for k,v in FLAGS.items()):raise Stop('FROZEN_SCOPE_DRIFT')
    if c.get('network_allowed') is not True or c.get('fit_count')!=0:raise Stop('BOUNDARY_DRIFT')
    for ref in [*c['code_by_file'].values(),*c['query_sources'].values(),c['unit_source']]:
        if registry.sha(ref['path'])!=ref['sha256']:raise Stop('SOURCE_IDENTITY_DRIFT')
    if (Path(c['code_by_file']['wrapper']['path']).resolve()!=Path(__file__).resolve()
            or Path(c['code_by_file']['original_m19']['path']).resolve()!=Path(original.__file__).resolve()
            or Path(c['code_by_file']['original_m11']['path']).resolve()!=Path(original.m11.__file__).resolve()
            or Path(c['code_by_file']['affine']['path']).resolve()!=Path(inspect.getfile(original.affine.event_affine)).resolve()):
        raise Stop('ORIGINAL_PRODUCER_IMPORT_COLLISION')
    c['code_composite_sha256']=original.digest({k:v['sha256'] for k,v in c['code_by_file'].items()})
    if registry.sha(c['contract_path'])!=c['contract_sha256']:raise Stop('CONTRACT_DRIFT')
    return c


def run(c,adapter,activation,*,bar_fetch=fetch_bar_page,action_fetch=None):
    if (c.get('state')!='FROZEN_READY_FOR_ACTIVATION'
            or activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE'
            or activation.get('execution_id')!=c['execution_id']
            or activation.get('config_sha256')!=c['config_sha256']
            or activation.get('code_composite_sha256')!=c['code_composite_sha256']):raise Stop('NOT_ACTIVATED')
    review_sha=activation.get('review_receipt_sha256')
    if not review_sha or registry.sha(c['review_receipt_path'])!=review_sha:
        raise Stop('REVIEW_IDENTITY_DRIFT')
    root=Path(c['run_dir'])
    if (root.parent.name!='M33_SOURCE_REFRESH_01' or root.parent.parent.resolve()!=Path(c['batch_root']).resolve()
            or root.name!='OUTPUT_01' or root.is_symlink()):raise Stop('OUTPUT_SCOPE')
    root.mkdir(parents=True,exist_ok=False)
    with (root/'.owner.lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=root/'RECEIPT.json';base.write_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**FLAGS))
        life,budget=RefreshLifecycle(adapter,c),base.Budget(root,CAPS)
        evidence=dict(bar_connect_calls=0,bar_query_calls=0,bar_raw_responses=[],actions={},no_retry=True)
        report=dict(status='INTENT',execution_id=c['execution_id'],evidence=evidence,**FLAGS)
        error=None
        try:
            life.start(receipt)
            with budget:
                start=life.require()
                report['resource_before']=dict(available_bytes=base.available_memory(),rss_bytes=base.rss_bytes(),disk_free_bytes=shutil.disk_usage(root).free)
                if report['resource_before']['available_bytes']<CAPS['min_memory_bytes'] or report['resource_before']['disk_free_bytes']<CAPS['min_disk_bytes']:
                    raise Stop('DEFERRED_RESOURCE')
                pages=[bar_fetch(life,c,budget,request,index,evidence) for index,request in enumerate(REQUESTS)]
                actions=[]
                for symbol,target in TARGETS:
                    life.require();folder=root/symbol;folder.mkdir(exist_ok=False)
                    try:
                        events,ev=original.actions_once(folder,symbol,c,budget,start,query_fn=action_fetch)
                        evidence['actions'][symbol]=ev
                        actions.append(summarize_actions(symbol,target,events))
                    finally:
                        # Original source persists raw/evidence even when normalization fails.
                        if (folder/'QUERY_EVIDENCE.json').exists():
                            evidence['actions'][symbol]=base.read_json(folder/'QUERY_EVIDENCE.json')
                        base.write_json(receipt,report)
                amounts=normalize_amounts(pages)
                original.new_json(root/'FRESH_FIXED_AMOUNTS.json',amounts)
                original.new_json(root/'ACTION_SUMMARY.json',dict(targets=actions))
                action_calls=sum(v.get('query_calls',0) for v in evidence['actions'].values())
                total=evidence['bar_query_calls']+action_calls
                if total!=5:raise Stop('EXACT_QUERY_COUNT_NOT_FIVE')
                report.update(status='FRESH_SOURCE_CAPTURE_COMPLETE',query_calls=total,normalized_amount_dates=list(DATES),
                    amounts_ref=dict(path=str(root/'FRESH_FIXED_AMOUNTS.json'),sha256=registry.sha(root/'FRESH_FIXED_AMOUNTS.json')),
                    actions_ref=dict(path=str(root/'ACTION_SUMMARY.json'),sha256=registry.sha(root/'ACTION_SUMMARY.json')),
                    action_terms_resolved=[dict(symbol=v['symbol'],all_standard=bool(v['same_date_records']) and all(x['standard_factor_status']=='STANDARD_AFFINE_OPERATOR_COMPUTED' for x in v['same_date_records'])) for v in actions],
                    standard_factor_reused='original.affine.event_affine; source identity and synthetic algebra verified',
                    old_model_effective_amount_bridge='NOT_ESTABLISHED',A_B_not_run=True,fits=0,
                    training_ready=False,full_prefix_activated=False)
                budget.check()
        except BaseException as exc:
            error=exc;report.update(status='FAILED_NO_RETRY',error_type=type(exc).__name__,error=str(exc)[:1000])
        finally:
            report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_peak_rss_bytes=budget.peak)
            base.write_json(receipt,report)
            if life.started and not life.finished:
                try:life.finish(receipt,report['status'],dict(fits=0,bar_query_calls=evidence['bar_query_calls'],action_query_calls=sum(v.get('query_calls',0) for v in evidence['actions'].values())))
                except BaseException as exc:
                    report.update(terminal_status='CANNOT_CONFIRM',terminal_error=repr(exc));base.write_json(receipt,report);raise
        if error is not None:raise error
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--activation');p.add_argument('--execute',action='store_true');args=p.parse_args()
    c=load_config(args.config)
    if not args.execute:
        print(registry.canonical(dict(status='M33_SOURCE_METADATA_ONLY_NOT_ACTIVATED',max_queries=5)));return
    if not args.activation:p.error('--activation required')
    result=run(c,registry.load_adapter(c),base.read_json(args.activation))
    print(registry.canonical(dict(status=result['status'],query_calls=result['query_calls'],training_ready=False)))


if __name__=='__main__':main()
