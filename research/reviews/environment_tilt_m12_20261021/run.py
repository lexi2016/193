"""One scoped continuation: reference4 accepted old checkpoints, rebuild3 blocked.

Uses original M19 functions/worker/lifecycle. No network, second producer,
formula change, old artifact rewrite, labels or fit.
"""
import argparse,copy,json,multiprocessing,sys,time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import environment_tilt_full_prefix as full

REDO=('000002.SZ','600004.SH','000014.SZ')
REUSE=('000001.SZ','600000.SH','920000.BJ','920001.BJ')
OLD_SOURCE_SHA='d2e0f8f5da5c6f76e076a34f8eed56df9530f3cd66aaae219203c935248fa512'

class Life(full.FullLifecycle):
    def event(self,kind,decision,receipt,metrics):
        e=super().event(kind,decision,receipt,metrics)
        e.update(stage='PRICE_RULE_THREE_BLOCKED_REDO_CANARY',baseline_id='M27_FIXED7_PARTIAL_PREFIX_CHECKPOINTS',one_variable='THREE_EXACT_OFFICIAL_PRICE_EVENT_EXCLUSIONS_WITH_FOUR_ORIGINAL_CHECKPOINT_REFERENCES')
        return e


def compatible_configuration(current,old,code):
    if code not in REUSE or code in REDO:raise full.Stop('REUSE_ONLY_FOUR_UNAFFECTED_SYMBOLS')
    for key in ('policy','qfq','actions','archive','inventory','pure_sources','unit_binding','bj_capability_probe'):
        if current[key]!=old[key]:raise full.Stop('OLD_CHECKPOINT_INPUT_SEMANTICS_CHANGED_'+key)
    if old['code_by_file']['environment_tilt_full_prefix.py']!=OLD_SOURCE_SHA:raise full.Stop('UNREVIEWED_OLD_PRODUCER')
    for name in full.DEPENDENCIES:
        if current['code_by_file'][name]!=old['code_by_file'][name]:raise full.Stop('ORIGINAL_PRICE_OPERATOR_CHANGED')
    if old['semantic_id']!=full.semantic_identity(old):raise full.Stop('OLD_SEMANTIC_IDENTITY_DRIFT')
    # New reviewed exceptions concern only REDO codes; the underlying M11 affine,
    # all original inputs/units and supported-action gate for REUSE are unchanged.
    return True


def reuse_one(adapter,current,old,code,expected):
    compatible_configuration(current,old,code)
    path=Path(old['run_dir'])/'symbols'/code/'CHECKPOINT.json'
    if expected['path']!=str(path):raise full.Stop('OLD_CHECKPOINT_MAPPING_ESCAPE')
    proof=full.trusted_checkpoint_binding(adapter,old,code,path)
    if proof['checkpoint']['sha256']!=expected['sha256']:raise full.Stop('OLD_CHECKPOINT_TERMINAL_MAPPING_DRIFT')
    saved=full.resume_checkpoint(dict(config=old,code=code,checkpoint_binding=proof),path.parent)
    if saved['status'] not in ('HISTORICAL_PREFIX_ONLY','NO_OBSERVED_EARLY_ROWS'):
        raise full.Stop('ONLY_VALID_OLD_CHECKPOINT_REUSE')
    return dict(symbol=code,status=saved['status'],original_checkpoint=expected,
        original_semantic_id=saved['semantic_id'],original_code_by_file=old['code_by_file'],
        original_terminal_event_sha256=proof['terminal']['event_sha256'],
        producing_execution_id=proof['terminal']['execution_id'],reference_only=True,
        recomputed=False,relabelled=False,**full.FLAGS)


def require_bound_redo_events(codes,events):
    """Reject absent/malformed saved actions before any worker can query fallback."""
    if not isinstance(events,dict):raise full.Stop('FIXED_REDO_SOURCE_EVENT_GAP_NO_QUERY')
    expected={'000002.SZ':(20051201,14),'600004.SH':(20051219,14),'000014.SZ':(20070630,4)}
    for code in REDO:
        rows=events.get(code)
        if code not in codes or not isinstance(rows,list) or not rows:
            raise full.Stop('FIXED_REDO_SOURCE_EVENT_GAP_NO_QUERY:'+code)
        for event in rows:
            if (not isinstance(event,dict) or type(event.get('_date_int')) is not int
                    or type(event.get('category')) is not int or event['category']<=0
                    or not isinstance(event.get('original_record'),dict)):
                raise full.Stop('MALFORMED_BOUND_REDO_EVENT_NO_QUERY:'+code)
            try:
                if len(str(event['_date_int']))!=8:raise ValueError('date width')
                datetime.strptime(str(event['_date_int']),'%Y%m%d')
                if event['_date_int']>full.POLICY['cutoff']:raise ValueError('after cutoff')
                if event['category']==1:
                    for key in full.base.ACTION_FIELDS:full.affine.decimal_scalar(event[key])
            except (KeyError,ValueError,full.affine.CannotConfirm) as exc:
                raise full.Stop('MALFORMED_BOUND_REDO_EVENT_NO_QUERY:'+code) from exc
        matches=[e for e in rows if (e['_date_int'],e['category'])==expected[code]]
        if len(matches)!=1 or not full.exact_nonprice_record(code,matches[0]):
            raise full.Stop('EXACT_REDO_PRICE_RECORD_NOT_BOUND_NO_QUERY:'+code)


def aggregate(reused,new):
    if set(reused)!=set(REUSE) or set(new)!=set(REDO):raise full.Stop('EXACT_SEVEN_RESULT_LINKS_REQUIRED')
    rows={**reused,**new};passed=all(v['status'] in ('HISTORICAL_PREFIX_ONLY','NO_OBSERVED_EARLY_ROWS')for v in rows.values())
    return dict(status='FIXED7_CANARY_COMPLETE' if passed else 'FIXED7_CANARY_BLOCKED',canary_pass=passed,
        fixed_sample_count=7,original_universe_count=5576,member_links=rows,
        reused_reference_count=4,newly_executed_count=3,network_calls=0,fit_count=0,
        full_universe_prefix_ready=False,training_ready=False,**full.FLAGS)


def run(c,activation):
    here=Path(__file__).resolve().parent
    if (c['redo_symbols']!=list(REDO) or c['reuse_symbols']!=list(REUSE)
            or c['network_allowed'] is not False or c['query_budget']!=0):raise full.Stop('EXACT_NO_NETWORK_CONTINUATION_REQUIRED')
    if full.sha(__file__)!=c['continuation_script_sha256']:raise full.Stop('CONTINUATION_SOURCE_DRIFT')
    if (activation.get('status')!='ACTIVATE_ONE_REVIEWED_FIXED7_CONTINUATION'
            or activation.get('execution_id')!=c['execution_id']
            or activation.get('config_sha256')!=c['config_sha256']
            or activation.get('script_sha256')!=c['continuation_script_sha256']):raise full.Stop('NOT_ACTIVATED')
    composite=full.digest(dict(m19_code_by_file=c['code_by_file'],continuation_script_sha256=c['continuation_script_sha256']))
    if composite!=c['execution_code_composite_sha256']:raise full.Stop('EXECUTION_CODE_COMPOSITE_DRIFT')
    c=dict(c,code_composite_sha256=composite)
    full.check_artifact(activation['review_receipt'])
    for ref in c['acceptance_refs'].values():full.check_artifact(ref)
    old=full.base.read_json(c['original_config']['path']);full.check_artifact(c['original_config'])
    for code in REUSE:compatible_configuration(c,old,code)
    root=Path(c['run_dir'])
    if root.exists() or root.is_symlink():raise full.Stop('NEW_OUTPUT_REQUIRED_NO_RETRY')
    with full.owner(root):
        invocation=root/'invocations'/c['execution_id'];invocation.mkdir(parents=True,exist_ok=False)
        receipt=invocation/'RECEIPT.json';full.base.write_json(receipt,dict(status='PREPARED_NOT_REGISTERED',**full.FLAGS))
        life=Life(full.load_adapter(c),c);budget=full.RecoveryBudget(root,c['caps']);children={};reused={};new={};pool=None
        try:
            life.start(receipt);token=life.require()
            with budget:
                if full.base.available_memory()<c['caps']['min_memory_bytes'] or full.shutil.disk_usage(root).free<c['caps']['min_disk_bytes']:raise full.Stop('DEFERRED_RESOURCE')
                full.os.environ.update({k:'1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')});c['arrow_pools']=full.base.configure_arrow_pools(1)
                # All actual authentication/reads occur under this durable START.
                reference,codes,members,events,cached=full.source_bindings(life,c,budget)
                require_bound_redo_events(codes,events)
                for code in REUSE:
                    reused[code]=reuse_one(life.adapter,c,old,code,c['reuse_checkpoint_refs'][code])
                    full.new_json(invocation/('REUSED_'+code+'.json'),reused[code]);budget.check()
                groups=reference.groupby('ts_code',sort=False)
                pool=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'));active={};pending=iter(REDO)
                while True:
                    while len(active)<2:
                        try:code=next(pending)
                        except StopIteration:break
                        cc=dict(c,execution_id=c['execution_id']+'_'+code.replace('.','_'),scope_symbol=code);child=Life(life.adapter,cc);cr=invocation/(code+'_TERMINAL.json');full.base.write_json(cr,dict(status='PREPARED_NOT_REGISTERED',symbol=code));child.start(cr);children[code]=(child,cr)
                        task=dict(config=c,code=code,start=child.require(),checkpoint_binding=None,members={code:members[code]} if code in members else {},unit_binding=c['unit_binding'],events=events[code],cached_events=cached.get(code,[]),reference=groups.get_group(code).sort_values('trade_date',kind='stable').to_dict('records'))
                        active[pool.submit(full.symbol_job,task)]=code
                    if not active:break
                    done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED);budget.check()
                    for future in done:
                        code=active.pop(future)
                        try:r=future.result()
                        except BaseException as exc:r=dict(symbol=code,status='BLOCKED',stop_reason='WORKER_EXIT:'+str(exc))
                        checkpoint=root/'symbols'/code/'CHECKPOINT.json'
                        if checkpoint.exists():
                            full.validate_checkpoint(r,checkpoint.parent,c,code);bound=dict(path=str(checkpoint),sha256=full.sha(checkpoint),bytes=checkpoint.stat().st_size)
                        else:bound=None
                        child,cr=children[code];full.base.write_json(cr,r);child.finish(cr,r['status'],dict(symbol=code,checkpoint_path=str(checkpoint),checkpoint_sha256=bound['sha256']if bound else None))
                        new[code]=dict(symbol=code,status=r['status'],new_checkpoint=bound,producing_execution_id=child.config['execution_id'],semantic_id=c['semantic_id'],reference_only=False)
                        full.base.write_json(receipt,dict(status='RUNNING',reused_members=4,new_completed=len(new),**full.FLAGS))
                pool.shutdown(wait=True);pool=None
                report=aggregate(reused,new);report.update(elapsed_seconds=time.monotonic()-budget.start_time,sampled_parent_peak_rss_bytes=budget.peak,max_worker_rss_bytes=max((full.base.read_json(v['new_checkpoint']['path']).get('sampled_peak_rss_bytes',0)for v in new.values()if v['new_checkpoint']),default=0),input_version=c['input_version'],semantic_id=c['semantic_id'],source_config_ref=dict(path=c['config_path'],sha256=c['config_sha256']),query_claim_count=len(list((root/'symbols').glob('*/QUERY_CLAIM.json'))));budget.check()
                if report['query_claim_count']!=0:raise full.Stop('UNEXPECTED_QUERY_CLAIM')
                full.base.write_json(receipt,report)
            life.finish(receipt,report['status'],dict(reused=4,new_completed=3,network_calls=0,fits=0));return report
        except BaseException as exc:
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
            for code,(child,cr)in children.items():
                if child.started and not child.finished:
                    cp=root/'symbols'/code/'CHECKPOINT.json';r=full.base.read_json(cp)if cp.exists()else dict(symbol=code,status='INTERRUPTED_OR_NOT_DISPATCHED',stop_reason=str(exc));full.base.write_json(cr,r);child.finish(cr,r['status'],dict(checkpoint_path=str(cp),checkpoint_sha256=full.sha(cp)if cp.exists()else None))
            failure=dict(status='FAILED_WITH_EVIDENCE',error=str(exc),error_type=type(exc).__name__,reused=reused,new=new,elapsed_seconds=time.monotonic()-budget.start_time,sampled_parent_peak_rss_bytes=budget.peak,network_calls=0,fit_count=0,**full.FLAGS);full.base.write_json(receipt,failure)
            if life.started and not life.finished:life.finish(receipt,failure['status'],dict(network_calls=0,fits=0))
            return failure

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--activation',required=True);a=p.parse_args();c=full.load_config(a.config);r=run(c,full.base.read_json(a.activation));print(full.canonical({k:r.get(k)for k in ('status','canary_pass','elapsed_seconds','sampled_parent_peak_rss_bytes','network_calls','fit_count')}))
