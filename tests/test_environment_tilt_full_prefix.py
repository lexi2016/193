"""Synthetic M19 acceptance; all payloads are temporary."""
import io
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/a_share_model_selection_v1'))
import environment_tilt_full_prefix as full


def frame(code='000014.SZ',dates=(19990104,20191231,20200102,20210506,20261015)):
    return pd.DataFrame([dict(trade_date=d,ts_code=code,open=10.0,high=10.0,low=10.0,close=10.0,vol=100,amount=1000) for d in dates])


def binding(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=full.sha(path))
def unit():return dict(variant='ORIGINAL_STANDARD_A_BJ_DAY',vol='shares',amount='CNY',scale='identity')


def test_entire_observed_history():
    raw=frame();r,p=full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[],'000014.SZ')
    assert not r['block_reasons'] and r['OHLC_denominator']==8
    assert [x['trade_date'] for x in p]==[19990104,20191231,20200102]


def test_pre2020_unsupported_action_blocks():
    raw=frame();r,p=full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[dict(_date_int=20000104,category=4)],'000014.SZ')
    assert r['block_reasons']==['UNSUPPORTED_PARTICIPATING_ACTION'] and len(p)==3


def test_category1_exdate_boundary():
    raw=frame(dates=(20191231,20200102,20210506,20261015))
    e=dict(_date_int=20200102,category=1,xdxr_fenhong=0,xdxr_peigujia=0,xdxr_songzhuangu=10,xdxr_peigu=0)
    r,p=full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[e],'000014.SZ')
    assert not r['block_reasons'] and p[0]['close']['cent']=='5.00' and p[1]['close']['cent']=='10.00'


def test_fixed_anchor_failure():
    raw=frame();ref=raw.iloc[-2:].reset_index(drop=True);ref.loc[1,'close']=9
    r,_=full.analyze_full(raw,ref,[],'000014.SZ')
    assert r['anchor']==20261015 and 'LATEST_COMMON_ANCHOR_NOT_IDENTITY' in r['block_reasons']
    assert 'WHOLE_OVERLAP_CENT_MISMATCH' in r['block_reasons']


def test_key_hole():
    raw=frame()
    with pytest.raises(full.Stop,match='WHOLE_OVERLAP_DATE_KEYS_MISMATCH'):
        full.analyze_full(raw.drop(index=3).reset_index(drop=True),raw.iloc[-3:].reset_index(drop=True),[],'000014.SZ')


def test_reference_not_rounded():
    raw=frame();ref=raw.iloc[-2:].reset_index(drop=True);ref.loc[0,'close']=10.001
    with pytest.raises(full.affine.CannotConfirm):full.analyze_full(raw,ref,[],'000014.SZ')


def test_observed_not_ipo():
    r=full.coverage_gate('000014.SZ',[20220104,20261015],20220104,unit())
    assert not r['needs_prefix'] and r['observed_source_first']==20220104
    assert r['ipo_date']=='CANNOT_CONFIRM' and r['historical_source_completeness']=='CANNOT_CONFIRM'


@pytest.mark.parametrize('code,change',[('920000.XX',{}),('000014.SZ',{'vol':'lots'}),('000014.SZ',{'variant':'UNKNOWN'})])
def test_unknown_source(code,change):
    with pytest.raises(full.Stop,match='CANNOT_CONFIRM'):full.coverage_gate(code,[19990104,20261015],20210506,dict(unit(),**change))


def test_inventory_bj_and_duplicate():
    h='market,security_code,member_name,uncompressed_bytes,uncompressed_member_sha256\n';r='bj,920000,bj920000.day,32,abc\n'
    assert '920000.BJ' in full.inventory(io.BytesIO((h+r).encode()))
    with pytest.raises(full.Stop,match='AMBIGUOUS'):full.inventory(io.BytesIO((h+r+r).encode()))


def test_complete_coverage_table_even_unfinished():
    r=full.summarize(['000001.SZ','920000.BJ','000014.SZ'],{'000001.SZ':dict(status='HISTORICAL_PREFIX_ONLY'),'920000.BJ':dict(status='BLOCKED')})
    assert r['reported_members']==3 and not r['prefix_ready']
    assert r['counts']==dict(HISTORICAL_PREFIX_ONLY=1,BLOCKED=1,NOT_EXECUTED=1)


def test_prefix_units_and_immutable_output(tmp_path):
    raw=frame();r,p=full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[],'000014.SZ')
    output=pd.read_csv(full.emit_full(tmp_path,r,p,raw)['path'])
    assert output.vol.tolist()==[100]*3 and output.amount.tolist()==[1000]*3
    stamp=(tmp_path/'historical_prefix_v1.csv').stat().st_mtime_ns
    full.emit_full(tmp_path,r,p,raw)
    assert (tmp_path/'historical_prefix_v1.csv').stat().st_mtime_ns==stamp


def test_overlap_reuse_not_rerun(monkeypatch):
    raw=frame()
    def fail(*a,**k):raise AssertionError('accepted overlap repeated')
    monkeypatch.setattr(full.affine,'compare_prices',fail)
    proof=dict(anchor=20261015,first_old_bar=20210506,overlap_rows=2,receipt={'sha256':'accepted'})
    r,p=full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[],'000014.SZ',proof)
    assert not r['block_reasons'] and len(p)==3 and r['overlap_comparison']['reused_receipt']==proof['receipt']


def test_reuse_scope_changed():
    raw=frame()
    with pytest.raises(full.Stop,match='SCOPE_DRIFT'):
        full.analyze_full(raw,raw.iloc[-2:].reset_index(drop=True),[],'000014.SZ',dict(anchor=20210506,first_old_bar=20210506,overlap_rows=2))


class Budget:
    def check(self):pass


def test_query_claim_and_replay(tmp_path):
    calls=[];c=dict(semantic_id='same')
    def query(life,local,budget,code,evidence):
        assert (tmp_path/'QUERY_CLAIM.json').exists();calls.append(code)
        evidence['raw_response_snapshot']=full.new_json(tmp_path/'QUERY_RAW_RESPONSE.json',dict(symbol=code,response=[]));return []
    _,r=full.actions_once(tmp_path,'000014.SZ',c,Budget(),{'event_sha256':'s'},query_fn=query)
    assert r['query_debit']==1
    _,r=full.actions_once(tmp_path,'000014.SZ',c,Budget(),{'event_sha256':'s'},query_fn=query)
    assert len(calls)==1 and r['source']=='SINGLE_ORIGINAL_MAC_TDX_QUERY'


def test_query_failure_no_retry(tmp_path):
    calls=[];c=dict(semantic_id='same')
    def fail(*a):calls.append(1);raise RuntimeError('network')
    with pytest.raises(RuntimeError):full.actions_once(tmp_path,'000014.SZ',c,Budget(),{'event_sha256':'s'},query_fn=fail)
    with pytest.raises(full.Stop,match='CONSUMED_QUERY_WITHOUT_DURABLE_RESPONSE_NO_RETRY'):
        full.actions_once(tmp_path,'000014.SZ',c,Budget(),{'event_sha256':'s'},query_fn=fail)
    assert calls==[1]


def test_malformed_durable_response_retained(tmp_path):
    full.new_json(tmp_path/'QUERY_CLAIM.json',dict(symbol='000014.SZ',semantic_id='same',query_debit=1))
    full.new_json(tmp_path/'QUERY_RAW_RESPONSE.json',dict(symbol='000014.SZ',response=[dict(year=2020,month=13,day=1,category=1)]))
    full.new_json(tmp_path/'QUERY_EVIDENCE.json',dict(raw_response_snapshot=binding(tmp_path/'QUERY_RAW_RESPONSE.json')))
    before=full.sha(tmp_path/'QUERY_RAW_RESPONSE.json')
    with pytest.raises(ValueError):full.actions_once(tmp_path,'000014.SZ',dict(semantic_id='same'),Budget(),{'event_sha256':'s'})
    assert full.sha(tmp_path/'QUERY_RAW_RESPONSE.json')==before


def test_bound_events_zero_query(tmp_path):
    def fail(*a):raise AssertionError('query')
    e=[dict(category=2,_date_int=20200102)];value,r=full.actions_once(tmp_path,'000014.SZ',{},Budget(),{},existing=e,query_fn=fail)
    assert value==e and r['query_debit']==0 and not (tmp_path/'QUERY_CLAIM.json').exists()


def task_fixture(tmp_path,code='000014.SZ'):
    root=tmp_path/'run';root.mkdir();c=dict(run_dir=str(root),semantic_id='stable',worker_caps=dict(full.base.CAPS),pure_sources={})
    c['worker_caps'].update(min_memory_bytes=0,min_disk_bytes=0,total_wall_sec=20)
    return dict(config=c,code=code,start={'event_sha256':'s'},members={code:dict(name='member.day',bytes=32,sha256='member')},reference=frame(code).iloc[-2:].to_dict('records'),events=[],unit_binding=unit())


def bind_checkpoint(task):
    root=Path(task['config']['run_dir'])/'symbols'/task['code'];path=root/'CHECKPOINT.json'
    saved=full.base.read_json(path)
    event=dict(event_type='TERMINAL',arm=task['code'],decision=saved['status'],
        input_identity=dict(semantic_id=task['config']['semantic_id'],recovery_run_dir=task['config']['run_dir']),
        deterministic_metrics=dict(checkpoint_path=str(path),checkpoint_sha256=full.sha(path)))
    event['event_sha256']=full.event_digest(event)
    task['checkpoint_binding']=dict(terminal=event,checkpoint=binding(path))


def mock_symbol(task,monkeypatch):
    raw=frame(task['code']);payload=b'x'*32
    task['members'][task['code']]['sha256']=full.hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(full,'read_member',lambda *a:payload)
    monkeypatch.setattr(full.m11,'raw_dates',lambda *a:raw.trade_date.tolist())
    monkeypatch.setattr(full.base,'pure_functions',lambda *a:{'read_day_tail':lambda *a,**k:raw})
    return raw


def test_successful_checkpoint_authenticated_not_recomputed(tmp_path,monkeypatch):
    task=task_fixture(tmp_path);mock_symbol(task,monkeypatch)
    assert full.symbol_job(task)['status']=='HISTORICAL_PREFIX_ONLY'
    bind_checkpoint(task)
    def fail(*a):raise AssertionError('recomputed')
    monkeypatch.setattr(full,'read_member',fail);assert full.symbol_job(task)['resumed']
    root=Path(task['config']['run_dir'])/'symbols'/task['code']
    (root/'historical_prefix_v1.csv').write_text('changed')
    with pytest.raises(full.Stop,match='ARTIFACT_IDENTITY_DRIFT'):full.symbol_job(task)


def test_failed_checkpoint_sealed(tmp_path,monkeypatch):
    task=task_fixture(tmp_path,'920000.BJ')
    monkeypatch.setattr(full,'read_member',lambda *a:b'x'*64)
    monkeypatch.setattr(full.m11,'raw_dates',lambda *a:[19990104,20261015])
    monkeypatch.setattr(full.base,'pure_functions',lambda *a:{'read_day_tail':lambda *a,**k:frame('920000.BJ')})
    r=full.symbol_job(task)
    assert r['status']=='BLOCKED' and 'BJ' in r['stop_reason']
    def fail(*a):raise AssertionError('member')
    bind_checkpoint(task)
    monkeypatch.setattr(full,'read_member',fail);assert full.symbol_job(task)['resumed']


def test_symbol_full_history_success(tmp_path,monkeypatch):
    task=task_fixture(tmp_path);raw=frame()
    monkeypatch.setattr(full,'read_member',lambda *a:b'x'*32)
    monkeypatch.setattr(full.m11,'raw_dates',lambda *a:raw.trade_date.tolist())
    monkeypatch.setattr(full.base,'pure_functions',lambda *a:{'read_day_tail':lambda *a,**k:raw})
    r=full.symbol_job(task)
    assert r['status']=='HISTORICAL_PREFIX_ONLY' and r['prefix_rows']==3 and len(r['artifacts'])==3
    bind_checkpoint(task)
    assert full.symbol_job(task)['resumed']


def test_no_observed_early_rows_without_query(tmp_path,monkeypatch):
    task=task_fixture(tmp_path);monkeypatch.setattr(full,'read_member',lambda *a:b'x'*64)
    monkeypatch.setattr(full.m11,'raw_dates',lambda *a:[20210506,20261015])
    def fail(*a,**k):raise AssertionError('unneeded query')
    monkeypatch.setattr(full,'actions_once',fail);r=full.symbol_job(task)
    assert r['status']=='NO_OBSERVED_EARLY_ROWS' and r['coverage']['observed_source_first']==20210506


def test_owner_lock(tmp_path):
    with full.owner(tmp_path):
        with pytest.raises(full.Stop,match='OWNER_BUSY'):
            with full.owner(tmp_path):pass


def test_symlink_and_overwrite_failclosed(tmp_path):
    full.new_json(tmp_path/'v.json',dict(a=1));(tmp_path/'link').symlink_to(tmp_path/'v.json')
    with pytest.raises(full.Stop,match='PATH_ESCAPE'):full.check_artifact(binding(tmp_path/'link'))
    with pytest.raises(full.Stop,match='IDENTITY_DRIFT'):full.content_once(tmp_path/'v.json',dict(a=2))


def test_mac_gate_precedes_execution(monkeypatch):
    monkeypatch.setattr(full.sys,'platform','linux')
    with pytest.raises(full.Stop,match='MAC_ONLY'):full.activation({})


def test_two_spawn_workers_independent_symbols_and_outputs(tmp_path):
    import multiprocessing
    import struct
    import zipfile
    from concurrent.futures import ProcessPoolExecutor
    source=tmp_path/'synthetic_decoder.py'
    source.write_text('''def read_day_tail(code,path,tail_records):
    payload=Path(path).read_bytes()
    return pd.DataFrame([dict(trade_date=struct.unpack_from('<I',payload,i)[0],ts_code=code,open=10.,high=10.,low=10.,close=10.,vol=100.,amount=1000.) for i in range(0,len(payload),32)])
def is_ext_market_ts(code):return False
def apply_xdxr_pre_close(*args):return None
''')
    history=tmp_path/'synthetic_history.py';history.write_text('def action_adjust(*args):return 1\n')
    archive=tmp_path/'synthetic.zip'
    payload=b''.join(struct.pack('<I',d)+b'\0'*28 for d in (19990104,20210506,20261015))
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('000001.SZ.day',payload);z.writestr('600000.SH.day',payload)
    root=tmp_path/'recovery';root.mkdir()
    config=dict(run_dir=str(root),semantic_id='spawn-synthetic',archive=dict(path=str(archive),frozen_stat=full.base.file_identity(archive.stat())),
                member_byte_limit=1024,worker_caps=dict(full.base.CAPS),accepted_overlap={},
                pure_sources=dict(data_layer=binding(source),history=binding(history)))
    jobs=[]
    for code in ('000001.SZ','600000.SH'):
        m=dict(name=code+'.day',bytes=len(payload),sha256=full.hashlib.sha256(payload).hexdigest())
        jobs.append(dict(config=config,code=code,start={'event_sha256':'synthetic-start'},members={code:m},unit_binding=unit(),events=[],reference=frame(code,dates=(20210506,20261015)).to_dict('records')))
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(full.symbol_job,task) for task in jobs]
        results=[future.result(timeout=30) for future in futures]
    assert [r['status'] for r in results]==['HISTORICAL_PREFIX_ONLY']*2
    assert all(r['prefix_rows']==1 for r in results)
    assert len(list((root/'symbols').glob('*/CHECKPOINT.json')))==2
    assert not list(root.rglob('QUERY_CLAIM.json'))


def test_registered_parent_failure_still_terminal_before_propagation(tmp_path,monkeypatch):
    import contextlib
    from environment_tilt_mac_registry import digest as event_digest
    class Adapter:
        def __init__(self):self.events=[]
        def append_parity_event(self,event,execution_id):
            event['event_sha256']=event_digest(event);self.events.append(event)
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
    adapter=Adapter()
    config=dict(run_dir=str(tmp_path/'M19_run'),execution_id='unused',phase='CANARY',
        caps=dict(full.base.CAPS),diagnostic_dates={'scope':'synthetic'},contract_path='synthetic-contract',
        contract_sha256='c',config_sha256='config',code_composite_sha256='code',config_path='synthetic-config')
    config['caps'].update(min_memory_bytes=0,min_disk_bytes=0)
    monkeypatch.setattr(full,'activation',lambda c:dict(execution_id='M19_SYNTHETIC_FAILURE_01'))
    def fail(life,*a):
        assert [e['event_type'] for e in adapter.events]==['RUN_STARTED']
        life.require();raise full.Stop('SYNTHETIC_INPUT_FAILURE')
    monkeypatch.setattr(full,'source_bindings',fail)
    with pytest.raises(full.Stop,match='SYNTHETIC_INPUT_FAILURE'):full.run(config,adapter)
    assert [e['event_type'] for e in adapter.events]==['RUN_STARTED','TERMINAL']
    assert adapter.events[-1]['decision']=='FAILED'
    assert full.base.read_json(tmp_path/'M19_run/invocations/M19_SYNTHETIC_FAILURE_01/RECEIPT.json')['status']=='FAILED'


def test_parent_is_only_registry_owner_and_symbol_progress_uses_official_lifecycle(tmp_path,monkeypatch):
    import contextlib
    import struct
    import zipfile
    from environment_tilt_mac_registry import digest as event_digest
    class Adapter:
        def __init__(self):self.events=[]
        def append_parity_event(self,event,execution_id):
            assert event['event_type'] in ('RUN_STARTED','TERMINAL')
            event['event_sha256']=event_digest(event);self.events.append(event)
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
    adapter=Adapter();decoder=tmp_path/'decoder.py'
    decoder.write_text('''def read_day_tail(code,path,tail_records):
    payload=Path(path).read_bytes()
    return pd.DataFrame([dict(trade_date=struct.unpack_from('<I',payload,i)[0],ts_code=code,open=10.,high=10.,low=10.,close=10.,vol=100.,amount=1000.) for i in range(0,len(payload),32)])
def is_ext_market_ts(code):return False
def apply_xdxr_pre_close(*args):return None
''')
    history=tmp_path/'history.py';history.write_text('def action_adjust(*args):return 1\n')
    payload=b''.join(struct.pack('<I',d)+b'\0'*28 for d in (19990104,20210506,20261015))
    archive=tmp_path/'fixture.zip';members={}
    with zipfile.ZipFile(archive,'w') as z:
        for code in full.CANARY:
            z.writestr(code+'.day',payload)
            members[code]=dict(name=code+'.day',bytes=len(payload),sha256=full.hashlib.sha256(payload).hexdigest())
    config=dict(run_dir=str(tmp_path/'M19_run'),semantic_id='synthetic',phase='CANARY',workers=2,query_budget=0,
        unit_binding=unit(),worker_caps=dict(full.base.CAPS),caps=dict(full.base.CAPS),
        diagnostic_dates={'scope':'synthetic'},contract_path='synthetic-contract',contract_sha256='c',
        config_sha256='cfg',config_path='synthetic-config',code_composite_sha256='code',
        member_byte_limit=1024,archive=dict(path=str(archive),frozen_stat=full.base.file_identity(archive.stat())),
        pure_sources=dict(data_layer=binding(decoder),history=binding(history)),accepted_overlap={})
    config['caps'].update(min_memory_bytes=0,min_disk_bytes=0)
    reference=pd.concat([frame(code,dates=(20210506,20261015)) for code in full.CANARY],ignore_index=True)
    monkeypatch.setattr(full,'activation',lambda c:dict(execution_id='M19_SYNTHETIC_PARENT_01'))
    def sources(life,*a):life.require();return reference,list(full.CANARY),members,{code:[] for code in full.CANARY},{}
    monkeypatch.setattr(full,'source_bindings',sources)
    result=full.run(config,adapter)
    assert result['counts']==dict(HISTORICAL_PREFIX_ONLY=5,BLOCKED=2)
    assert not result['canary_pass'] and not result['prefix_ready']
    assert len(adapter.events)==16
    for execution in {e['execution_id'] for e in adapter.events}:
        assert [e['event_type'] for e in adapter.events if e['execution_id']==execution]==['RUN_STARTED','TERMINAL']
    assert result['actual_query_debits']==0
    # Same successful/failed symbol checkpoints need no new symbol executions on resume.
    monkeypatch.setattr(full,'activation',lambda c:dict(execution_id='M19_SYNTHETIC_RESUME_01'))
    resumed=full.run(config,adapter)
    assert len(adapter.events)==18 and all(r['resumed'] for r in resumed['members'])


def test_orphan_execution_requires_audit_without_starting_duplicate(tmp_path):
    import contextlib
    class Adapter:
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):
            return b'',[dict(route_id='a-share-environment-tilt-original-universe-full-prefix',event_type='RUN_STARTED',execution_id='M19_OLD',input_identity=dict(recovery_run_dir=str(tmp_path)))]
    with pytest.raises(full.Stop,match='COMPLETION_PENDING_AUDIT'):
        full.ensure_no_orphan(Adapter(),dict(run_dir=str(tmp_path)))


def test_accepted_overlap_requires_exact_source_review_member_and_events(tmp_path):
    code='000014.SZ';member=dict(name='bound.day',bytes=32,sha256='m')
    original=dict(qfq={'sha256':'q'},archive={'recorded_sha256':'a'},pure_sources={'decoder':'d'})
    original_path=tmp_path/'old_config.json';full.new_json(original_path,original)
    receipt=dict(symbol=code,decision='ONE_SYMBOL_CANDIDATE_PREFIX_ONLY',anchor=20261015,
        first_old_bar=20210506,overlap_OHLC_denominator=8,overlap_price_mismatches=0,
        anchor_identity=True,overlap_exact_keys=True,input_identity=dict(touched_candidates=[dict(symbol=code,**member)]))
    receipt_path=tmp_path/'old_receipt.json';full.new_json(receipt_path,receipt)
    review=dict(actual_run_receipt_sha256=full.sha(receipt_path),reviewer='independent',verdict='OPERATIONAL_UNIT_BINDING_SUPPORTED_FOR_BOUND_SH_SZ_VARIANT_AND_OBSERVED_MEMBERS')
    review_path=tmp_path/'review.json';full.new_json(review_path,review)
    proof=dict(config=binding(original_path),receipt=binding(receipt_path),review=binding(review_path),m11_source_sha256='source',events_sha256='e')
    config=dict(original,accepted_overlap={code:proof},code_by_file={'environment_tilt_mac_prefix_canary.py':'source'},allcategory_bindings=[dict(symbols=[code],sha256='e')])
    assert full.accepted_overlap(config,code,member)['overlap_rows']==2
    with pytest.raises(full.Stop,match='MEMBER_DRIFT'):full.accepted_overlap(config,code,dict(member,sha256='different'))
    config['allcategory_bindings'][0]['sha256']='changed'
    with pytest.raises(full.Stop,match='EVENT_DRIFT'):full.accepted_overlap(config,code,member)


def test_semantic_identity_ignores_phase_but_tracks_policy_and_inputs():
    config={key:{} for key in ('policy','code_by_file','qfq','actions','archive','inventory','pure_sources','query_sources','allcategory_bindings','unit_binding','accepted_overlap','bj_capability_probe')}
    identity=full.semantic_identity(config)
    config.update(phase='FULL',workers=4,execution_id='new')
    assert full.semantic_identity(config)==identity
    config['qfq']={'sha256':'changed'}
    assert full.semantic_identity(config)!=identity


def test_original_category1_cache_conflict_is_not_ignored():
    old=dict(_date_int=20200102,xdxr_fenhong=1,xdxr_peigujia=0,xdxr_songzhuangu=0,xdxr_peigu=0)
    event=dict(old,category=1)
    assert full.verify_cached_actions([old],[event],19990104,20261015)['participating_cached_events_checked']==1
    with pytest.raises(full.Stop,match='CACHE_VS_ALLCATEGORY_CONFLICT'):
        full.verify_cached_actions([old],[dict(event,xdxr_fenhong=2)],19990104,20261015)
    with pytest.raises(full.Stop,match='CACHE_VS_ALLCATEGORY_CONFLICT'):
        full.verify_cached_actions([old],[],19990104,20261015)


def test_cache_absence_does_not_prove_no_actions():
    assert full.verify_cached_actions([],[],19990104,20261015)['cache_absence_not_treated_as_no_action']


def test_bj_no_observed_early_rows_has_date_only_status_without_query(tmp_path,monkeypatch):
    task=task_fixture(tmp_path,'920000.BJ')
    monkeypatch.setattr(full,'read_member',lambda *a:b'x'*64)
    monkeypatch.setattr(full.m11,'raw_dates',lambda *a:[20210506,20261015])
    def fail(*a,**k):raise AssertionError('BJ action query without bound route')
    monkeypatch.setattr(full,'actions_once',fail)
    result=full.symbol_job(task)
    assert result['status']=='NO_OBSERVED_EARLY_ROWS'
    assert result['observed_source_first']==20210506
    assert result['coverage']['historical_source_completeness']=='CANNOT_CONFIRM'


def bj_session(response,calls):
    import contextlib
    class Client:
        def settimeout(self,value):assert value==5
    class API:
        auto_retry=False;heartbeat=False;client=Client()
        def get_xdxr_info(self,market,code):calls.append((market,code));return response
    @contextlib.contextmanager
    def session(**kwargs):
        assert kwargs==dict(host='218.6.170.47',port=7709,multithread=False)
        yield API()
    return session


def test_bj_capability_request_market2_raw_first_and_no_symbol_echo_claim(tmp_path):
    calls=[];response=[dict(year=2020,month=1,day=2,category=1,fenhong=1,peigujia=0,songzhuangu=0,peigu=0)]
    config=dict(run_dir=str(tmp_path),query_sources={},query_host='218.6.170.47',query_port=7709,bj_capability_probe={'market':2})
    events,evidence=[],dict(query_calls=0,connect_calls=0)
    events=full.bj_query_events(full.WorkerGate({'event_sha256':'synthetic'}),config,Budget(),'920000.BJ',evidence,bj_session(response,calls))
    assert calls==[(2,'920000')] and len(events)==1
    assert evidence['request_identity']['canonical_code']=='920000.BJ'
    assert evidence['response_symbol_echo'].startswith('NOT_EXPOSED')
    assert full.base.read_json(tmp_path/'QUERY_RAW_RESPONSE.json')['response']==response
    assert full.sha(tmp_path/'QUERY_RAW_RESPONSE.json')==evidence['raw_response_snapshot']['sha256']


@pytest.mark.parametrize('response',[[],None,[dict(year=2020,month=13,day=1,category=1)]])
def test_bj_empty_unknown_or_bad_date_keeps_raw_and_never_retries(tmp_path,response):
    calls=[];config=dict(run_dir=str(tmp_path),semantic_id='BJ',query_sources={},query_host='218.6.170.47',query_port=7709,bj_capability_probe={'market':2})
    def query(l,c,b,code,e):return full.bj_query_events(l,c,b,code,e,bj_session(response,calls))
    with pytest.raises((full.Stop,ValueError)):
        full.actions_once(tmp_path,'920000.BJ',config,Budget(),{'event_sha256':'s'},query_fn=query)
    assert calls==[(2,'920000')] and (tmp_path/'QUERY_RAW_RESPONSE.json').exists()
    with pytest.raises((full.Stop,ValueError)):
        full.actions_once(tmp_path,'920000.BJ',config,Budget(),{'event_sha256':'s'},query_fn=query)
    assert calls==[(2,'920000')]


def test_bj_known_standard_member_has_no_invented_decoder_unit_gate():
    assert full.coverage_gate('920000.BJ',[19990104,20261015],20210506,unit())['needs_prefix']


def test_bj_symbol_response_conflict_keeps_raw(tmp_path):
    response=[dict(code='920001',year=2020,month=1,day=2,category=1)]
    config=dict(run_dir=str(tmp_path),query_sources={},query_host='218.6.170.47',query_port=7709,bj_capability_probe={'market':2})
    with pytest.raises(full.Stop,match='BJ_RESPONSE_SYMBOL_CONFLICT'):
        full.bj_query_events(full.WorkerGate({'event_sha256':'s'}),config,Budget(),'920000.BJ',dict(query_calls=0,connect_calls=0),bj_session(response,[]))
    assert (tmp_path/'QUERY_RAW_RESPONSE.json').exists()


def test_r1_checkpoint_status_drift_and_missing_trusted_binding_failclosed(tmp_path,monkeypatch):
    task=task_fixture(tmp_path);task['members']={}
    assert full.symbol_job(task)['status']=='BLOCKED'
    with pytest.raises(full.Stop,match='TRUSTED_CHECKPOINT_BINDING_REQUIRED'):full.symbol_job(task)
    bind_checkpoint(task)
    path=Path(task['config']['run_dir'])/'symbols'/task['code']/'CHECKPOINT.json'
    value=full.base.read_json(path);value['status']='HISTORICAL_PREFIX_ONLY'
    path.write_text(full.canonical(value)+'\n')
    with pytest.raises(full.Stop,match='ARTIFACT_IDENTITY_DRIFT'):full.symbol_job(task)
    # Even a synthetic terminal that authenticates these bytes cannot omit success artifacts.
    bind_checkpoint(task)
    with pytest.raises(full.Stop,match='SUCCESS_CHECKPOINT_MISSING_PREFIX_ARTIFACTS'):full.symbol_job(task)


def test_r1_bj_conflicting_record_rejected_on_fresh_and_replay_without_query(tmp_path):
    calls=[];response=[dict(code='920001',market=2,year=2020,month=1,day=2,category=1)]
    config=dict(run_dir=str(tmp_path),semantic_id='BJ',query_sources={},query_host='218.6.170.47',query_port=7709,bj_capability_probe={'market':2})
    def query(l,c,b,code,e):return full.bj_query_events(l,c,b,code,e,bj_session(response,calls))
    for _ in range(2):
        with pytest.raises(full.Stop,match='BJ_RESPONSE_SYMBOL_CONFLICT'):
            full.actions_once(tmp_path,'920000.BJ',config,Budget(),{'event_sha256':'s'},query_fn=query)
    assert calls==[(2,'920000')]
    response[0]['code']='920000';response[0]['market']=0
    outer=dict(symbol='920000.BJ',market=2,response=response)
    other=tmp_path/'market_conflict.json';full.new_json(other,outer)
    with pytest.raises(full.Stop,match='BJ_RESPONSE_MARKET_CONFLICT'):full.normalized_response(other,'920000.BJ')


class PowerLoss(BaseException):pass


def partial_recovery_fixture(tmp_path,monkeypatch):
    task=task_fixture(tmp_path);mock_symbol(task,monkeypatch);task['events']=None;calls=[]
    def query(l,c,b,code,e):
        calls.append(code);e.update(query_calls=1,connect_calls=1,wall_sec=.01)
        e['raw_response_snapshot']=full.new_json(Path(c['run_dir'])/'QUERY_RAW_RESPONSE.json',dict(symbol=code,response=[]))
        return []
    monkeypatch.setattr(full.m11,'query_events',query)
    return task,calls


def test_r1_crash_after_allcategory_recovers_without_metadata_drift_or_requery(tmp_path,monkeypatch):
    task,calls=partial_recovery_fixture(tmp_path,monkeypatch)
    analyze,publish=full.analyze_full,full.publish_checkpoint
    def crash(*a,**k):raise PowerLoss('crash')
    monkeypatch.setattr(full,'analyze_full',crash);monkeypatch.setattr(full,'publish_checkpoint',crash)
    with pytest.raises(PowerLoss):full.symbol_job(task)
    root=Path(task['config']['run_dir'])/'symbols'/task['code']
    before=(root/'ALLCATEGORY.json').read_bytes();stamp=(root/'ALLCATEGORY.json').stat().st_mtime_ns
    assert not (root/'CHECKPOINT.json').exists()
    monkeypatch.setattr(full,'analyze_full',analyze);monkeypatch.setattr(full,'publish_checkpoint',publish)
    result=full.symbol_job(task)
    assert result['status']=='HISTORICAL_PREFIX_ONLY' and calls==[task['code']]
    assert (root/'ALLCATEGORY.json').read_bytes()==before and (root/'ALLCATEGORY.json').stat().st_mtime_ns==stamp
    assert result['query']['source']=='SINGLE_ORIGINAL_MAC_TDX_QUERY'


def test_r1_crash_after_prefix_recovers_identical_csv_without_overwrite_or_requery(tmp_path,monkeypatch):
    task,calls=partial_recovery_fixture(tmp_path,monkeypatch);publish=full.publish_checkpoint
    def crash(*a,**k):raise PowerLoss('crash before checkpoint')
    monkeypatch.setattr(full,'publish_checkpoint',crash)
    with pytest.raises(PowerLoss):full.symbol_job(task)
    root=Path(task['config']['run_dir'])/'symbols'/task['code'];prefix=root/'historical_prefix_v1.csv'
    before=prefix.read_bytes();identity=(prefix.stat().st_ino,prefix.stat().st_mtime_ns)
    assert not (root/'CHECKPOINT.json').exists()
    monkeypatch.setattr(full,'publish_checkpoint',publish)
    result=full.symbol_job(task)
    assert result['status']=='HISTORICAL_PREFIX_ONLY' and calls==[task['code']]
    assert prefix.read_bytes()==before and (prefix.stat().st_ino,prefix.stat().st_mtime_ns)==identity
    bind_checkpoint(task);assert full.symbol_job(task)['resumed']
