"""New synthetic exact-key diagnostic checks; no financial data or official registry."""
from pathlib import Path
import contextlib
import json
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/a_share_model_selection_v1'))
import environment_tilt_missing_label_recovery as m


def ledger_fixture():
    dates=pd.bdate_range('2024-01-01',periods=487);calendar=pd.DataFrame(dict(date=dates,calendar_index=range(487)))
    ledger=dict(rows_by_fold={});selections={};pos=0
    for fold,n in zip(m.PRIMARY,[117,125,117,126]):
        rows=[];records=[]
        for ix in range(pos,pos+n):
            invalid=ix<24;codes=['000001.SZ','000002.SZ']if ix<2 else ['000001.SZ']
            rows.append(dict(fold=fold,date=dates[ix].strftime('%Y-%m-%d'),calendar_index=ix,valid=not invalid,
                reason='MISSING_SELECTED_H2_LABEL_ENTIRE_AC_PAIR_EXCLUDED'if invalid else None,missing_codes=codes if invalid else [],returns=None if invalid else {'A_ORIGINAL':.01,'C_MONTH_BONUS':.02}))
            records.append(dict(fold=fold,date=dates[ix].strftime('%Y-%m-%d'),h2_date=dates[ix+2].strftime('%Y-%m-%d'),calendar_index=ix,reason=None,
                selections={a:[f'{j:06d}.SZ'for j in range(1,11)]for a in m.ARMS}))
        ledger['rows_by_fold'][fold]=rows;selections[fold]=dict(records=records);pos+=n
    return ledger,selections,calendar


@pytest.mark.parametrize('drift',['none','calendar','code','scope','selection'])
def test_missing_scope_uses_only_original_metadata_exact_shared_calendar(drift):
    ledger,selections,calendar=ledger_fixture()
    if drift=='calendar':selections['2024H1']['records'][0]['h2_date']='2024-01-04'
    if drift=='code':ledger['rows_by_fold']['2024H1'][0]['missing_codes']=['UNSELECTED']
    if drift=='scope':ledger['rows_by_fold']['2024H1'][0]['valid']=True
    if drift=='selection':selections['2024H1']['records'][0]['selections']['A_ORIGINAL']=['same']*10
    if drift!='none':
        with pytest.raises(m.Blocked):m.missing_metadata(ledger,selections,calendar)
    else:
        rows=m.missing_metadata(ledger,selections,calendar)
        assert len(rows)==26 and len({r['signal_date']for r in rows})==24
        assert rows[0]['target_date']=='2024-01-03'and rows[0]['arms']==m.ARMS
        assert 'returns'not in json.dumps(rows)and 'prediction'not in json.dumps(rows)


def source_fixture(tmp_path,n=70,kind='timestamp',duplicate=False):
    dc,cc=('date','code')if kind=='timestamp'else('trade_date','ts_code')
    day=pd.Timestamp('2024-01-02')if kind=='timestamp'else '20240102'if kind=='string_YYYYMMDD'else 20240102
    rows=[{dc:day,cc:f'{i:06d}.SZ','close':float(i+1)}for i in range(n)]
    rows.append({dc:day,cc:'NOT_REQUESTED','close':-1.})
    if duplicate:rows.append(dict(rows[0]))
    path=tmp_path/'source.parquet';data=pd.DataFrame(rows)
    if kind=='int32_YYYYMMDD':data[dc]=data[dc].astype('int32');data[cc]=data[cc].astype('category');data['close']=data.close.astype('float32')
    data.to_parquet(path,index=False)
    ref=dict(path=str(path),sha256=m.sha(path));spec=dict(id='canonical'if kind=='timestamp'else'original_raw',ref=ref,date_column=dc,code_column=cc,date_kind=kind,file_rows=len(rows))
    c=dict(caps=m.CAPS)
    class Active:
        def require_active(self):return True
    adapter=m.previous.old.BoundFiles(c,Active());pairs=[('2024-01-02'if kind=='timestamp'else '20240102'if kind=='string_YYYYMMDD'else 20240102,f'{i:06d}.SZ')for i in range(n)]
    return adapter,c,spec,pairs


@pytest.mark.parametrize('kind',['timestamp','int64_YYYYMMDD','int32_YYYYMMDD','string_YYYYMMDD'])
def test_exact_batched_read_key_parity_and_schema_date_mapping(tmp_path,kind):
    a,c,spec,pairs=source_fixture(tmp_path,kind=kind)
    frame=m.read_exact_source(a,c,spec,pairs+[pairs[0]])
    assert len(frame)==70 and list(frame.columns)==['date','code','close']
    assert frame.close.sum()==70*71/2 and 'NOT_REQUESTED'not in set(frame.code)
    assert set(frame.date)=={pd.Timestamp('2024-01-02')}
    assert c['_access']['sources'][spec['id']]['batch_count']==2


def test_total_cap_counts_all_batches_before_any_close_decode(tmp_path,monkeypatch):
    a,c,spec,pairs=source_fixture(tmp_path,n=129)
    def forbidden(*args,**kwargs):raise AssertionError('close decoded before combined cap')
    monkeypatch.setattr(m,'decode_exact_table',forbidden)
    with pytest.raises(m.Blocked,match='TOTAL_EXACT_PREDECODE_ROW_CAP'):m.read_exact_source(a,c,spec,pairs)
    access=c['_access']['sources'][spec['id']]
    assert access['matched_rows']==129 and access['batch_count']==3 and access['count_returned']
    assert not access['close_decode_attempted']


def test_duplicate_is_preserved_for_diagnosis_never_silently_deduplicated(tmp_path):
    a,c,spec,pairs=source_fixture(tmp_path,duplicate=True)
    frame=m.read_exact_source(a,c,spec,pairs)
    assert len(frame)==71 and m.flags(frame,'2024-01-02','000000.SZ')['issue']=='DUPLICATE_KEY'


@pytest.mark.parametrize('mode',['target_absent','signal_nan','signal_nonpositive','duplicate','raw_projection','price_conflict','now_finite'])
def test_diagnostic_causes_flags_no_return_or_suspension_invention(mode):
    key=dict(fold='2024H1',signal_date='2024-01-02',target_date='2024-01-04',code='000001.SZ',calendar_index=0,arms=m.ARMS)
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2024-01-02','2024-01-04']),code=['000001.SZ']*2,close=[10.,11.]))
    if mode=='target_absent':frame=frame.iloc[:1].copy()
    if mode=='signal_nan':frame.loc[0,'close']=float('nan')
    if mode=='signal_nonpositive':frame.loc[0,'close']=0.
    if mode=='duplicate':frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    raw=frame.copy()
    if mode=='raw_projection':frame=frame.iloc[:1].copy()
    if mode=='price_conflict':raw.loc[1,'close']=12.
    rows=m.diagnostic_rows([key],frame,raw,{('2024-01-02','000001.SZ'):mode=='now_finite'})
    result=rows[0]
    expected={'target_absent':'ORIGINAL_RAW_AND_CANONICAL_TARGET_BAR_ABSENT_STATUS_UNKNOWN','signal_nan':'ORIGINAL_RAW_AND_CANONICAL_SIGNAL_CLOSE_NONFINITE','signal_nonpositive':'ORIGINAL_RAW_AND_CANONICAL_SIGNAL_CLOSE_NONPOSITIVE','duplicate':'ORIGINAL_RAW_AND_CANONICAL_SIGNAL_DUPLICATE_KEY','raw_projection':'SEMANTIC_CONFLICT_CANONICAL_VS_ORIGINAL_RAW_PROJECTION','price_conflict':'SEMANTIC_CONFLICT_CANONICAL_VS_ORIGINAL_RAW_PROJECTION','now_finite':'SEMANTIC_CONFLICT_ORIGINAL_MISSING_LABEL_NOW_FINITE'}
    assert result['cause']==expected[mode]
    assert result['recovery_status']=='NOT_REPAIRED_DIAGNOSTIC_ONLY'
    assert result['suspension_or_collection_status']=='STATUS_UNKNOWN_NO_BOUND_STATUS_FIELD'
    assert 'label_return'not in json.dumps(result)and '11.0'not in json.dumps(result)


def test_activation_required_before_output_or_data_access(tmp_path):
    with pytest.raises(m.previous.Blocked,match='ACTIVATION_REQUIRED'):
        m.run(dict(run_dir=str(tmp_path/'run')),dict(status='NOT_APPROVED'))
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('drift',[False,True])
def test_fast_bind_only_accepted_sha_and_unchanged_stat(tmp_path,drift):
    path=tmp_path/'data';path.write_bytes(b'synthetic');ref=dict(path=str(path),sha256=m.sha(path))
    known=dict(sha256=ref['sha256'],stat=m.previous.old.support.stat_identity(path))
    if drift:known['sha256']='0'*64
    class Adapter:
        bound={}
        def metadata(self,ref):return dict(input_bindings={str(path):known})
    c=dict(source_specs=[dict(id='canonical',ref=ref),dict(id='original_raw',ref=ref)],inputs=dict(calendar=ref),accepted_binding_receipts={k:'receipt'for k in ['canonical','original_raw','calendar']})
    adapter=Adapter()
    if drift:
        with pytest.raises(m.Blocked,match='FAST_BIND'):m.reuse_accepted_inputs(adapter,c)
    else:
        m.reuse_accepted_inputs(adapter,c)
        assert adapter.bound[str(path)]==known


def test_null_and_ieee_nan_are_distinct_flags(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=tmp_path/'null_nan.parquet'
    pq.write_table(pa.table(dict(date=pa.array([pd.Timestamp('2024-01-02')]*2,type=pa.timestamp('ns')),code=['NULL','NAN'],close=pa.array([None,float('nan')],type=pa.float64()))),path)
    ref=dict(path=str(path),sha256=m.sha(path));spec=dict(id='canonical',ref=ref,date_column='date',code_column='code',date_kind='timestamp',file_rows=2)
    c=dict(caps=m.CAPS)
    class Active:
        def require_active(self):return True
    frame=m.read_exact_source(m.previous.old.BoundFiles(c,Active()),c,spec,[('2024-01-02','NULL'),('2024-01-02','NAN')])
    assert m.flags(frame,'2024-01-02','NULL')['issue']=='CLOSE_NULL'
    assert m.flags(frame,'2024-01-02','NAN')['issue']=='CLOSE_NONFINITE'


def test_full_synthetic_diagnostic_never_substitutes_retention_prices_or_recalculates_461(tmp_path,monkeypatch):
    ledger,selections,calendar=ledger_fixture();keys=m.missing_metadata(ledger,selections,calendar)
    pairs=sorted({(d,r['code'])for r in keys for d in [r['signal_date'],r['target_date']]})
    # Exactly one requested target missing from both canonical and original raw;
    # retained sources contain it, but lack the admitted QFQ bridge.
    missing=(keys[-1]['target_date'],keys[-1]['code'])
    full=pd.DataFrame([dict(date=pd.Timestamp(d),code=code,close=10.)for d,code in pairs])
    canonical=full.loc[~((full.date==pd.Timestamp(missing[0]))&(full.code==missing[1]))].copy()
    frames={n:canonical.copy()if n in ['canonical','original_raw']else full.copy()for n in m.POLICY['source_ids']}
    refs={n:dict(path=n,sha256=n)for n in m.POLICY['source_ids']}
    c=dict(run_dir=str(tmp_path),caps=m.CAPS,old_accepted_refs=dict(actual_review='review',ledger='ledger',selection_refs={f:f for f in m.PRIMARY}),
        missing_key_metadata_ref='keys',overlap_metadata_ref='overlap',original_binding_ref='binding',original_projection_ref='projection',preparation_ref='prep',
        inputs=dict(calendar=dict(path='calendar')),source_specs=[dict(id=n,ref=refs[n],date_kind='timestamp')for n in m.POLICY['source_ids']],recipe_ref=dict(path=str(m.SUPPORT_DIR/'frozen_m0/research/a_share_model_selection_v1/model_selection.py')))
    fixture_metadata={'review':dict(verdict='PASS_INDEPENDENT_ACTUAL_DERIVED_RESULTS_AND_REGISTRY'),'ledger':ledger,**selections,
        'keys':dict(keys=keys),'overlap':dict(keys=[dict(date=r['signal_date'],code=r['code'])for r in keys]),
        'binding':dict(sources=dict(price_panel=dict(**refs['original_raw'],column_map=dict(date='trade_date',code='ts_code')))),
        'projection':dict(panels=dict(assembled_qfq=dict(path='original_raw'))),'prep':dict(status='COMPLETED',cache_files={'canonical':'canonical'})}
    class Adapter:
        bound={}
        def __init__(self,*a):pass
        def metadata(self,ref):return fixture_metadata[ref]
        def frame(self,*a,**kw):return calendar.copy()
    def read(adapter,config,spec,pairs):
        config.setdefault('_access',{}).setdefault('sources',{})[spec['id']]=dict(arrow_schema={'close':'float'if spec['id']=='backtest_retention'else'double'},close_decode_attempted=True)
        return frames[spec['id']].copy()
    monkeypatch.setattr(m.previous.old,'BoundFiles',Adapter);monkeypatch.setattr(m,'read_exact_source',read);monkeypatch.setattr(m,'reuse_accepted_inputs',lambda *a:None)
    events=[]
    class Life:
        def checkpoint(self,ref,name):
            assert Path(ref['path']).is_file();events.append(name)
    original_hash=m.previous.old.support.digest([r for f in m.PRIMARY for r in ledger['rows_by_fold'][f]if r['valid']])
    report=m.diagnose(c,Life())
    assert report['status']=='MISSING24_EXACT_KEY_DIAGNOSTIC_COMPLETE'
    assert events==m.POLICY['source_ids']and len(report['rows'])==26
    assert report['old_successful_dates_recomputed']==report['restored_dates']==0
    assert report['evaluation_delta']is None and report['source_repair_performed']is False
    assert c['_access']['old_valid_primary_records_logical_sha256']==original_hash
    assert all(r['status']=='RETENTION_DIAGNOSTIC_ONLY'and r['new_label_values']==0 for r in report['retention'].values())
    assert 'label_return'not in json.dumps(report)


@pytest.mark.parametrize('failure',[False,True])
def test_registered_start_precedes_synthetic_compute_and_truthful_terminal(tmp_path,monkeypatch,failure):
    from environment_tilt_mac_registry import digest
    class Adapter:
        events=[]
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
        def append_parity_event(self,event,**kwargs):
            event['previous_event_sha256']=self.events[-1]['event_sha256']if self.events else None
            event['event_sha256']=digest(event);self.events.append(event)
    official=Adapter();review=tmp_path/'review.json';review.write_text('{}')
    c=dict(execution_id=m.EXECUTION,run_dir=str(tmp_path/'run'),caps=m.CAPS,old_accepted_refs={},diagnostic_dates={'start':'2024-01-01','end':'2025-12-31'},
        contract_path='synthetic',contract_sha256='contract',config_path='synthetic',config_sha256='config',code_composite_sha256='code')
    activation=dict(status='APPROVED_FOR_MISSING24_DIAGNOSTIC',review_receipt=dict(path=str(review),sha256=m.sha(review)),
        **{k:c[k]for k in ['execution_id','contract_sha256','config_sha256','code_composite_sha256']})
    def compute(config,life):
        assert len(official.events)==1 and official.events[0]['truth_access']=='NOT_READ';life.require()
        config['_access']=dict(sources={'canonical':dict(close_decode_attempted=True)},returned_target_dates=['2024-01-04'],requested_target_dates=['2024-01-04'])
        if failure:raise m.Blocked('SYNTHETIC_EXACT_READ_FAILURE')
        config['_access']['diagnosed_keys']=26
        return dict(status='MISSING24_EXACT_KEY_DIAGNOSTIC_COMPLETE',rows=[{}]*26)
    monkeypatch.setattr(m,'diagnose',compute);monkeypatch.setattr(m.previous.resources,'available_memory',lambda:64*2**30)
    report=m.run(c,activation,adapter=official)
    assert [e['event_type']for e in official.events]==['RUN_STARTED','TERMINAL']
    terminal=official.events[-1]
    assert terminal['truth_access']=='EXACT_SELECTED_CLOSE_FLAGS_NO_NUMERIC_RETURNS'
    assert terminal['target_date_range']['count']==1 and terminal['deterministic_metrics']['selection_slots']is None
    assert report['status']==('STOPPED'if failure else'MISSING24_EXACT_KEY_DIAGNOSTIC_COMPLETE')


def test_flags_checkpoint_never_claims_paired_returns_and_uses_official_guard(tmp_path):
    from environment_tilt_mac_registry import digest
    class Adapter:
        def __init__(self):self.events=[]
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
        def append_parity_event(self,event,**kwargs):
            event['previous_event_sha256']=self.events[-1]['event_sha256']if self.events else None
            event['event_sha256']=digest(event);self.events.append(event)
        def _append_unlocked(self,raw,events,event):self.append_parity_event(event)
    c=dict(execution_id=m.EXECUTION,caps=m.CAPS,old_accepted_refs={},diagnostic_dates={},contract_path='synthetic',contract_sha256='contract',config_path='synthetic',config_sha256='config',code_composite_sha256='code')
    adapter=Adapter();life=m.RecoveryLifecycle(adapter,c);path=tmp_path/'flags.json';path.write_text('{}');life.start(path)
    ref=dict(path=str(path),sha256=m.sha(path));life.checkpoint(ref,'canonical')
    event=adapter.events[-1]
    assert event['decision']=='EXACT_SOURCE_FLAGS_CHECKPOINT_SAVED'
    assert event['input_identity']['diagnostic_flags_checkpoint']==ref and event['deterministic_metrics']['source_id']=='canonical'
    assert 'PAIRED_RESULTS'not in json.dumps(event)
    with pytest.raises(m.Blocked,match='ACTIVE_UNIQUE_DIAGNOSTIC_CHECKPOINT_REQUIRED'):life.checkpoint(ref,'canonical')
