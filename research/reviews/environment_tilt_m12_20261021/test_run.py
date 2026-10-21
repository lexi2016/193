"""Synthetic compatibility/linkage/exception closure only; no actual data or registry."""
import copy,importlib.util,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path('/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/environment-tilt-recovery-20261020/research/a_share_model_selection_v1')))
spec=importlib.util.spec_from_file_location('price_redo12',Path(__file__).with_name('run.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def configs():
    old={k:{}for k in ('policy','qfq','actions','archive','inventory','pure_sources','query_sources','allcategory_bindings','unit_binding','accepted_overlap','bj_capability_probe')}
    old['code_by_file']={name:'a'*64 for name in m.full.DEPENDENCIES}
    old['code_by_file']['environment_tilt_full_prefix.py']=m.OLD_SOURCE_SHA
    old['semantic_id']=m.full.semantic_identity(old)
    current=copy.deepcopy(old);current['code_by_file']['environment_tilt_full_prefix.py']='b'*64
    return current,old


def test_compatibility_only_four_disjoint_codes_and_exact_original_inputs():
    current,old=configs()
    for code in m.REUSE:assert m.compatible_configuration(current,old,code)
    with pytest.raises(m.full.Stop,match='FOUR_UNAFFECTED'):m.compatible_configuration(current,old,m.REDO[0])
    changed=copy.deepcopy(current);changed['qfq']={'sha256':'changed'}
    with pytest.raises(m.full.Stop,match='INPUT_SEMANTICS_CHANGED'):m.compatible_configuration(changed,old,m.REUSE[0])
    changed=copy.deepcopy(current);changed['code_by_file'][m.full.DEPENDENCIES[0]]='changed'
    with pytest.raises(m.full.Stop,match='PRICE_OPERATOR_CHANGED'):m.compatible_configuration(changed,old,m.REUSE[0])


def test_reuse_reference_retains_original_semantic_producer_and_terminal(tmp_path,monkeypatch):
    current,old=configs();old['run_dir']=str(tmp_path/'old');code=m.REUSE[0];path=Path(old['run_dir'])/'symbols'/code/'CHECKPOINT.json';expected={'path':str(path),'sha256':'d'*64}
    proof={'checkpoint':expected,'terminal':{'event_sha256':'e'*64,'execution_id':'ORIGINAL_OWNER'}}
    monkeypatch.setattr(m.full,'trusted_checkpoint_binding',lambda *a:proof)
    monkeypatch.setattr(m.full,'resume_checkpoint',lambda *a:dict(symbol=code,status='HISTORICAL_PREFIX_ONLY',semantic_id=old['semantic_id']))
    result=m.reuse_one(object(),current,old,code,expected)
    assert result['original_checkpoint']==expected and result['original_semantic_id']==old['semantic_id']
    assert result['original_code_by_file']==old['code_by_file'] and result['producing_execution_id']=='ORIGINAL_OWNER'
    assert result['reference_only'] and not result['recomputed'] and not result['relabelled'] and not tmp_path.joinpath('old').exists()


def test_aggregate_requires_original_four_plus_new_three_no_extra_or_omission():
    reused={c:dict(status='NO_OBSERVED_EARLY_ROWS'if c=='920001.BJ'else'HISTORICAL_PREFIX_ONLY',original_semantic_id='original')for c in m.REUSE}
    new={c:dict(status='HISTORICAL_PREFIX_ONLY',semantic_id='new')for c in m.REDO}
    r=m.aggregate(reused,new);assert r['canary_pass'] and set(r['member_links'])==set(m.REDO+m.REUSE)
    assert r['original_universe_count']==5576 and r['fixed_sample_count']==7 and not r['training_ready']
    new[m.REDO[0]]['status']='BLOCKED';assert not m.aggregate(reused,new)['canary_pass']
    new.pop(m.REDO[0])
    with pytest.raises(m.full.Stop,match='EXACT_SEVEN'):m.aggregate(reused,new)


def test_failure_after_registered_start_terminalizes_without_data_or_worker(tmp_path,monkeypatch):
    current,old=configs();old_path=tmp_path/'old_config.json';old_path.write_text(json.dumps(old));root=tmp_path/'NEW_OUTPUT'
    current.update(redo_symbols=list(m.REDO),reuse_symbols=list(m.REUSE),network_allowed=False,query_budget=0,continuation_script_sha256=m.full.sha(m.__file__),execution_id='SYNTHETIC_12',config_sha256='c'*64,acceptance_refs={},original_config={'path':str(old_path),'sha256':m.full.sha(old_path)},run_dir=str(root),phase='CANARY',caps=dict(rss_bytes=8*2**30,total_wall_sec=600,payload_bytes=2**30,min_memory_bytes=0,min_disk_bytes=0))
    current['execution_code_composite_sha256']=m.full.digest(dict(m19_code_by_file=current['code_by_file'],continuation_script_sha256=current['continuation_script_sha256']))
    calls=[]
    class Life:
        def __init__(self,*args):self.started=self.finished=False;self.adapter=object()
        def start(self,receipt):self.started=True;calls.append('START')
        def require(self):return {'event_sha256':'synthetic'}
        def finish(self,*args):self.finished=True;calls.append('TERMINAL')
    monkeypatch.setattr(m,'Life',Life);monkeypatch.setattr(m.full,'load_adapter',lambda c:object());monkeypatch.setattr(m.full.base,'configure_arrow_pools',lambda n:{'cpu':1,'io':1});monkeypatch.setattr(m.full.base,'available_memory',lambda:99*2**30)
    def fail(*args):calls.append('BEFORE_PAYLOAD');raise m.full.Stop('SYNTHETIC_AUTH_FAILURE')
    monkeypatch.setattr(m.full,'source_bindings',fail)
    a=dict(status='ACTIVATE_ONE_REVIEWED_FIXED7_CONTINUATION',execution_id=current['execution_id'],config_sha256=current['config_sha256'],script_sha256=current['continuation_script_sha256'],review_receipt={'path':str(old_path),'sha256':m.full.sha(old_path)})
    r=m.run(current,a);assert r['status']=='FAILED_WITH_EVIDENCE' and r['error']=='SYNTHETIC_AUTH_FAILURE'
    assert calls==['START','BEFORE_PAYLOAD','TERMINAL']


def bound_events():
    result={}
    for code,date,category in (('000002.SZ',20051201,14),('600004.SH',20051219,14),('000014.SZ',20070630,4)):
        r=dict(fenhong=None,peigujia=None,songzhuangu=None,peigu=None)
        if category==14:r.update(xingquanjia=3.7300000190734863 if code=='000002.SZ'else 7.,fenshu=8. if code=='000002.SZ'else 6.)
        else:r.update(panqianliutong=0.,panhouliutong=0.,qianzongguben=0.,houzongguben=0.)
        result[code]=[dict(_date_int=date,category=category,original_record=r)]
    return result


def test_complete_bound_event_lists_admit_without_mutation():
    events=bound_events();before=copy.deepcopy(events)
    m.require_bound_redo_events(m.REDO,events)
    assert events==before


@pytest.mark.parametrize('invalid',['missing','null','empty','nonlist','malformed_row','bad_date','bad_category','missing_original_record','incomplete_cash','duplicate_exact_record','changed_exact_terms'])
def test_invalid_saved_events_terminalize_before_worker_or_query(tmp_path,monkeypatch,invalid):
    events=bound_events();code=m.REDO[0]
    if invalid=='missing':events.pop(code)
    elif invalid=='null':events[code]=None
    elif invalid=='empty':events[code]=[]
    elif invalid=='nonlist':events[code]=tuple(events[code])
    elif invalid=='malformed_row':events[code]=[None]
    elif invalid=='bad_date':events[code][0]['_date_int']=20051301
    elif invalid=='bad_category':events[code][0]['category']=None
    elif invalid=='missing_original_record':events[code][0].pop('original_record')
    elif invalid=='incomplete_cash':events[code].append(dict(_date_int=20070101,category=1,original_record={}))
    elif invalid=='duplicate_exact_record':events[code]*=2
    else:events[code][0]['original_record']['fenshu']=7.
    current,old=configs();old_path=tmp_path/'old_config.json';old_path.write_text(json.dumps(old));root=tmp_path/'NEW_OUTPUT'
    current.update(redo_symbols=list(m.REDO),reuse_symbols=list(m.REUSE),network_allowed=False,query_budget=0,continuation_script_sha256=m.full.sha(m.__file__),execution_id='SYNTHETIC_12_R1',config_sha256='c'*64,acceptance_refs={},original_config={'path':str(old_path),'sha256':m.full.sha(old_path)},run_dir=str(root),phase='CANARY',caps=dict(rss_bytes=8*2**30,total_wall_sec=600,payload_bytes=2**30,min_memory_bytes=0,min_disk_bytes=0))
    current['execution_code_composite_sha256']=m.full.digest(dict(m19_code_by_file=current['code_by_file'],continuation_script_sha256=current['continuation_script_sha256']))
    calls=[]
    class Life:
        def __init__(self,*args):self.started=self.finished=False;self.adapter=object()
        def start(self,receipt):self.started=True;calls.append('START')
        def require(self):return {'event_sha256':'synthetic'}
        def finish(self,*args):self.finished=True;calls.append('TERMINAL')
    monkeypatch.setattr(m,'Life',Life);monkeypatch.setattr(m.full,'load_adapter',lambda c:object());monkeypatch.setattr(m.full.base,'configure_arrow_pools',lambda n:{'cpu':1,'io':1});monkeypatch.setattr(m.full.base,'available_memory',lambda:99*2**30)
    def bindings(*args):calls.append('BOUND_SAVED_SOURCE');return None,list(m.REDO+m.REUSE),{},events,{}
    def forbidden(*args,**kwargs):calls.append('FORBIDDEN_WORKER_OR_QUERY');raise AssertionError('worker/query must not be reached')
    monkeypatch.setattr(m.full,'source_bindings',bindings);monkeypatch.setattr(m,'ProcessPoolExecutor',forbidden);monkeypatch.setattr(m.full,'symbol_job',forbidden);monkeypatch.setattr(m.full,'actions_once',forbidden);monkeypatch.setattr(m.full.m11,'query_events',forbidden)
    a=dict(status='ACTIVATE_ONE_REVIEWED_FIXED7_CONTINUATION',execution_id=current['execution_id'],config_sha256=current['config_sha256'],script_sha256=current['continuation_script_sha256'],review_receipt={'path':str(old_path),'sha256':m.full.sha(old_path)})
    r=m.run(current,a)
    assert r['status']=='FAILED_WITH_EVIDENCE' and r['network_calls']==r['fit_count']==0
    assert calls==['START','BOUND_SAVED_SOURCE','TERMINAL']
    assert not list(root.rglob('QUERY_CLAIM.json')) and not (root/'symbols').exists()
    durable=json.loads((root/'invocations'/current['execution_id']/'RECEIPT.json').read_text())
    assert durable['status']=='FAILED_WITH_EVIDENCE' and durable['error_type']=='Stop'
