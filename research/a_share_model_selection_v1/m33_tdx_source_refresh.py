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


# Original archive field projection: no OHLC adjustment or feature calculation.
_AMOUNT_READER = None
_AMOUNT_CONTEXT = None
AMOUNT_FIELD_POLICY = dict(universe_count=5576, cutoff=20261015,
    history="ALL_ORIGINAL_OBSERVED_HISTORY", amount="CNY", volume="shares",
    input_version="TDX_ORIGINAL_DAY_AMOUNT_CNY_SHARES_V1", workers=2, worker_threads=1)


def _amount_reader(binding):
    import ast, __future__, struct
    import numpy as np
    import pandas as pd
    if registry.sha(binding['path']) != binding['sha256']:raise Stop('ORIGINAL_READER_DRIFT')
    tree=ast.parse(Path(binding['path']).read_text());names={'read_day_tail','is_ext_market_ts'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'DAY_STRUCT','EXT_MARKET_DAY_STRUCT'} for t in n.targets)]
    if {n.name for n in nodes if isinstance(n,ast.FunctionDef)}!=names:raise Stop('ORIGINAL_READER_DEFINITIONS')
    scope=dict(np=np,pd=pd,struct=struct,Path=Path)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),binding['path'],'exec',flags=__future__.annotations.compiler_flag),scope)
    return scope['read_day_tail']


class _AmountDayBytes:
    """Pass authenticated member bytes into the unchanged original reader."""
    def __init__(self,payload):self.payload=payload
    def stat(self):
        from types import SimpleNamespace
        return SimpleNamespace(st_size=len(self.payload))
    def open(self,mode):
        import io
        if mode!='rb':raise Stop('RAW_READER_WRITE_FORBIDDEN')
        return io.BytesIO(self.payload)


def amount_field_frame(payload,code,reader,cutoff=20261015):
    import numpy as np
    import pandas as pd
    if not payload or len(payload)%32:raise Stop('RAW_MEMBER_RECORD_SIZE')
    raw=reader(code,_AmountDayBytes(payload),tail_records=len(payload)//32)
    if raw.empty:raise Stop('NO_OBSERVED_RAW_ROWS')
    raw=raw.loc[raw.trade_date<=cutoff,['trade_date','ts_code','amount','vol']].copy()
    if raw.empty:raise Stop('NO_ROWS_THROUGH_CUTOFF')
    dates=pd.to_datetime(raw.trade_date.astype(str),format='%Y%m%d',errors='raise')
    if raw.trade_date.duplicated().any() or not dates.is_monotonic_increasing:raise Stop('RAW_DATE_KEYS')
    if set(raw.ts_code)!={code}:raise Stop('RAW_SYMBOL_IDENTITY')
    if not np.isfinite(raw[['amount','vol']].to_numpy(dtype='float64')).all() or (raw[['amount','vol']]<0).any().any():raise Stop('INVALID_AMOUNT_VOLUME')
    if not (raw.vol==raw.vol.astype('int64')).all():raise Stop('NONINTEGER_SHARES')
    return raw.rename(columns={'amount':'amount_CNY','vol':'volume_shares'}).astype({'trade_date':'int32','amount_CNY':'float64','volume_shares':'int64'})


def _amount_worker_initialize(config,active_checker=None):
    global _AMOUNT_READER,_AMOUNT_CONTEXT
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    if active_checker is None:
        worker_life=AmountFieldLifecycle(registry.load_adapter(config),config);worker_life.started=True
        event=worker_life.require()
        if event['event_sha256']!=config['worker_start_sha256']:raise Stop('WORKER_START_IDENTITY_DRIFT')
    else:active_checker()  # dependency injection for synthetic fixtures; never selected by JSON/CLI
    _AMOUNT_READER=_amount_reader(config['original_reader'])
    config=dict(config,arrow_pools=base.configure_arrow_pools(1));_AMOUNT_CONTEXT=config


def _amount_existing(root,semantic,member,producers=None):
    path=root/'CHECKPOINT.json'
    if not path.exists():return None
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise Stop('CHECKPOINT_SYMLINK')
    item=base.read_json(path)
    if item.get('symbol')!=root.name:raise Stop('CHECKPOINT_SYMBOL_DRIFT')
    if item.get('semantic_sha256')!=semantic or item.get('member')!=member:raise Stop('CHECKPOINT_INPUT_DRIFT')
    producer=(producers or {}).get(item.get('owner_execution_id'))
    if producers is not None and (not producer or item.get('producing_code_sha256',producer)!=producer):raise Stop('CHECKPOINT_PRODUCER_DRIFT')
    for artifact in item.get('artifacts',[]):original.check_artifact(artifact,root)
    return dict(item,reused=True,reuse_producing_code_sha256=producer,checkpoint=dict(path=str(path),sha256=registry.sha(path)))


def _amount_symbol_job(task):
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    code,member,old_dates=task
    c=_AMOUNT_CONTEXT;root=Path(c['run_dir'])/'symbols'/code
    reused=_amount_existing(root,c['semantic_sha256'],member,c.get('accepted_checkpoint_producers')) if root.exists() else None
    if reused:return reused
    if root.exists():raise Stop('PARTIAL_SYMBOL_OUTPUT_REQUIRES_SCOPED_RECOVERY:'+code)
    root.mkdir(parents=True,exist_ok=False)
    result=dict(symbol=code,semantic_sha256=c['semantic_sha256'],member=member,
        owner_execution_id=c['execution_id'],producing_code_sha256=c['code_by_file']['wrapper']['sha256'],worker_pid=os.getpid(),artifacts=[],reused=False,training_ready=False,**FLAGS)
    budget=original.RecoveryBudget(root,c['worker_caps'])
    try:
        with budget:
            if member is None:raise Stop('ORIGINAL_MEMBER_NOT_IN_INVENTORY')
            payload=original.read_member(c,member,budget)
            frame=amount_field_frame(payload,code,_AMOUNT_READER)
            actual=set(map(int,frame.trade_date));old=set(old_dates);missing=sorted(old-actual)
            first=min(old);prefix=int((frame.trade_date<first).sum())
            sink=pa.BufferOutputStream();pq.write_table(pa.Table.from_pandas(frame,preserve_index=False),sink,compression='zstd')
            output=original.durable_new(root/'AMOUNT_VOLUME.parquet',sink.getvalue().to_pybytes())
            result.update(status='PROJECTED' if not missing else 'PROJECTED_WITH_OLD_KEY_GAPS',rows=len(frame),
                date_min=int(frame.trade_date.min()),date_max=int(frame.trade_date.max()),old_first=first,
                early_prefix_rows=prefix,old_key_count=len(old),missing_old_dates=missing,coverage_complete=not missing,
                observed_history_only=True,historical_source_completeness='CANNOT_CONFIRM',units=c['unit_binding'],
                arrow_pools=c['arrow_pools'],artifacts=[output])
            budget.check()
    except Exception as exc:
        result.update(status='REFUSED',reason=str(exc),exception_class=type(exc).__name__,
            fatal_run=any(t in str(exc) for t in ('ARCHIVE_STAT_DRIFT','ARCHIVE_CHANGED','ORIGINAL_READER_DRIFT','RSS_LIMIT','WALL_LIMIT','PAYLOAD_LIMIT')))
    result.update(sampled_peak_rss_bytes=budget.peak,wall_seconds=time.monotonic()-budget.start_time)
    binding=original.new_json(root/'CHECKPOINT.json',result)
    return dict(result,checkpoint=binding)


def _amount_compact_keys(grouped,rows):
    from array import array
    for row in rows:
        code=row['ts_code'];date=int(row['trade_date'])
        if not isinstance(code,str) or not original.re.fullmatch(r'\d{6}\.(SH|SZ|BJ)',code):raise Stop('ORIGINAL_UNIVERSE_KEY')
        if date>20261015:raise Stop('OLD_KEY_AFTER_CUTOFF')
        grouped.setdefault(code,array('I')).append(date)


def _amount_validate_compact_keys(grouped):
    import numpy as np
    for code,dates in grouped.items():
        values=np.sort(np.frombuffer(dates,dtype=np.uint32)).copy()
        if len(values)>1 and (values[1:]==values[:-1]).any():raise Stop('DUPLICATE_OLD_DATE_CODE')
        grouped[code]=values
    return grouped


def load_amount_projection_config(path):
    c=base.read_json(path)
    if c['policy']!=AMOUNT_FIELD_POLICY or c.get('network_allowed') is not False or c.get('fit_count')!=0:raise Stop('FIELD_POLICY_DRIFT')
    if not 1<=c['workers']<=2 or c['worker_caps']['threads']!=1 or c['caps']['threads']!=1:raise Stop('FIELD_CONCURRENCY_DRIFT')
    for name,binding in c['code_by_file'].items():
        if registry.sha(binding['path'])!=binding['sha256']:raise Stop('SOURCE_DRIFT:'+name)
    if Path(c['code_by_file']['wrapper']['path']).resolve()!=Path(__file__).resolve():raise Stop('SOURCE_IMPORT_COLLISION')
    if registry.sha(c['contract_path'])!=c['contract_sha256']:raise Stop('CONTRACT_DRIFT')
    c.update(config_path=str(Path(path).resolve()),config_sha256=registry.sha(path),
        code_composite_sha256=original.digest({k:v['sha256'] for k,v in c['code_by_file'].items()}))
    root=Path(c['run_dir'])
    if root.parent.resolve()!=Path(c['batch_root']).resolve()/'M37_TDX_REPAIR' or root.name!='AMOUNT_FIELDS_FULL':raise Stop('FIELD_OUTPUT_SCOPE')
    return c


class AmountFieldLifecycle(RefreshLifecycle):
    def event(self,kind,decision,receipt,metrics):
        event=super().event(kind,decision,receipt,metrics)
        event.update(route_id='a-share-environment-tilt-original-archive-field-projection',
            experiment_id='A_SHARE_ENV_TILT_ORIGINAL_ARCHIVE_FIELDS',stage='ORIGINAL_5576_AMOUNT_SHARES_PROJECTION',
            arm='ORIGINAL_5576_RAW_AMOUNT_VOLUME',baseline_id='ORIGINAL_TDX_DAY_RAW_CNY_SHARES',
            one_variable='VERSIONED_RAW_FIELDS_NO_IEEE_GATE_NO_QFQ_OVERWRITE',truth_access='ONLY_RAW_AMOUNT_VOLUME_DATE_AND_ORIGINAL_UNIVERSE_KEYS')
        return event


def run_amount_projection(c,activation=None):
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    import multiprocessing
    import pandas as pd
    import pyarrow.parquet as pq
    if (not activation or activation.get('decision')!='INDEPENDENT_REVIEW_PASS_ACTIVATE'
        or activation.get('execution_id')!=c['execution_id'] or activation.get('config_sha256')!=c['config_sha256']
        or activation.get('code_composite_sha256')!=c['code_composite_sha256']
        or registry.sha(c['source_review_receipt_path'])!=activation.get('review_receipt_sha256')):raise Stop('FIELD_NOT_REVIEW_ACTIVATED')
    if registry.sha(c['canary_review']['path'])!=c['canary_review']['sha256']:raise Stop('CANARY_ACCEPTANCE_DRIFT')
    root=Path(c['run_dir']);root.mkdir(exist_ok=True)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):raise Stop('OUTPUT_SYMLINK')
    receipt=root/('RECEIPT_'+c['execution_id']+'.json');original.new_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**FLAGS))
    adapter=registry.load_adapter(c);life=AmountFieldLifecycle(adapter,c);budget=original.RecoveryBudget(root,c['caps'])
    report=dict(status='INTENT',execution_id=c['execution_id'],results=[],network_calls=0,fits=0,training_ready=False,**FLAGS)
    with original.owner(root):
        try:
            life.start(receipt)
            with budget:
                if base.available_memory()<c['caps']['min_memory_bytes'] or shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise Stop('DEFERRED_RESOURCE')
                c['arrow_pools']=base.configure_arrow_pools(1)
                proof=c['qfq_fast_bind_proof']
                if registry.sha(proof['path'])!=proof['sha256']:raise Stop('FAST_BIND_PROOF_DRIFT')
                old=base.read_json(proof['path'])['files']['old_qfq'];path=Path(c['qfq']['path'])
                if old['sha256']!=c['qfq']['sha256'] or base.file_identity(path.stat())!=old['stat']:raise Stop('OLD_QFQ_FAST_BIND_DRIFT')
                if base.file_identity(Path(c['archive']['path']).stat())!=c['archive']['frozen_stat']:raise Stop('ARCHIVE_STAT_DRIFT')
                life.require()
                # Only original universe/date keys; no OHLC, labels or model columns.
                grouped={};pf=pq.ParquetFile(path)
                for batch in pf.iter_batches(columns=['ts_code','trade_date'],batch_size=65536):
                    _amount_compact_keys(grouped,batch.to_pylist())
                    budget.check()
                _amount_validate_compact_keys(grouped)
                del batch,pf
                if base.file_identity(path.stat())!=old['stat'] or len(grouped)!=5576:raise Stop('ORIGINAL_5576_OR_STAT_DRIFT')
                with base.authenticated(life,c['inventory'],budget) as handle:members=original.inventory(handle)
                codes=sorted(grouped);universe=original.content_once(root/'UNIVERSE_AND_OLD_KEYS.json',dict(qfq=c['qfq'],codes=codes,old_first={k:int(v[0]) for k,v in grouped.items()},count=5576))
                c['worker_start_sha256']=life.require()['event_sha256']
                tasks=iter((code,members.get(code),grouped[code].tolist()) for code in codes);results=[];checkpointed=0
                with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=_amount_worker_initialize,initargs=(c,)) as executor:
                    active={};exhausted=False;last_check=0.0
                    while active or not exhausted:
                        while len(active)<c['workers'] and not exhausted:
                            try:task=next(tasks)
                            except StopIteration:exhausted=True;break
                            active[executor.submit(_amount_symbol_job,task)]=task[0]
                        if not active:break
                        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                        for future in done:
                            code=active.pop(future);result=future.result();results.append(result)
                            if result.get('fatal_run'):raise Stop('FATAL_SYMBOL:'+code+':'+result.get('reason',''))
                        if len(results)-checkpointed>=100:
                            bundle=original.new_json(root/('PROGRESS_'+c['execution_id']+'_'+str(len(results))+'.json'),dict(completed=len(results),checkpoints=[r['checkpoint'] for r in results[checkpointed:]]))
                            spec=life.event('CHECKPOINT','FIELDS_PROGRESS',bundle['path'],dict(completed_members=len(results),fits=0))
                            spec.update(registry_event_id=c['execution_id']+'_PROGRESS_'+str(len(results)),checkpoint_ref=bundle)
                            life.require()
                            with adapter._locked():
                                raw,events=adapter._read_validated();adapter._append_unlocked(raw,events,spec)
                            checkpointed=len(results)
                        if time.monotonic()-last_check>=10:
                            budget.check()
                            if base.available_memory()<c['caps']['min_memory_bytes'] or shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise Stop('DEFERRED_RESOURCE_DURING_RUN')
                            last_check=time.monotonic();report.update(completed_members=len(results));base.write_json(receipt,report)
                report.update(status='FIELD_PROJECTION_COMPLETE',results=results,universe=universe,
                    completed_members=len(results),projected=sum(r['status'].startswith('PROJECTED') for r in results),
                    refused=sum(r['status']=='REFUSED' for r in results),with_old_key_gaps=sum(r['status']=='PROJECTED_WITH_OLD_KEY_GAPS' for r in results),
                    total_rows=sum(r.get('rows',0) for r in results),early_prefix_rows=sum(r.get('early_prefix_rows',0) for r in results),
                    input_version=AMOUNT_FIELD_POLICY['input_version'],qfq_authentication='FAST_BIND_ACCEPTED_SHA_AND_SAME_STAT_NO_REHASH',
                    whole_zip='REUSED_RECORDED_NOT_REHASHED; EACH_CONSUMED_MEMBER_SHA_VERIFIED',arrow_pools=c['arrow_pools'])
                original.new_json(root/('COVERAGE_AND_REFUSALS_'+c['execution_id']+'.json'),report);budget.check()
        except BaseException as exc:report.update(status='STOPPED_WITH_CHECKPOINTS',error_class=type(exc).__name__,error=str(exc))
        finally:
            report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_parent_peak_rss_bytes=budget.peak)
            base.write_json(receipt,report)
            if life.started and not life.finished:life.finish(receipt,report['status'],dict(fits=0,completed_members=report.get('completed_members',0),network_calls=0))
    return report


def amount_projection_main():
    p=argparse.ArgumentParser(description='Original archive amount/share fields only');p.add_argument('--amount-projection-config',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--activation');args=p.parse_args();c=load_amount_projection_config(args.amount_projection_config)
    if not args.execute:print(registry.canonical(dict(status='FIELD_METADATA_ONLY_NO_SCIENCE',workers=c['workers'])));return
    if not args.activation:p.error('--activation required')
    report=run_amount_projection(c,base.read_json(args.activation));print(registry.canonical({k:v for k,v in report.items() if k!='results'}))


if __name__=='__main__':
    if '--amount-projection-config' in sys.argv:amount_projection_main()
    else:main()
