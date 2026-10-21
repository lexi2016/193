"""Minimal synthetic units/request/durable-receipt checks. No live calls."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import m33_tdx_source_refresh as m


def daily_rows():
    return [dict(year=int(str(d)[:4]),month=int(str(d)[4:6]),day=int(str(d)[6:]),
        open=10.,high=11.,low=9.,close=10.,vol=2.5,amount=12345.67) for d in m.DATES]


def test_amount_CNY_identity_lots_times100_and_future_prices_unused():
    rows=daily_rows()+[dict(year=2026,month=10,day=16)]
    result=m.normalize_amounts([rows,[]]);assert len(result['fixed_rows'])==5
    assert result['retained_date_max']==20261015
    for row in result['fixed_rows']:
        assert row['amount_CNY']==12345.67 and row['raw_volume_lots']==2.5 and row['volume_shares']==250
    with pytest.raises(m.Stop,match='DUPLICATE'):m.normalize_amounts([daily_rows(),daily_rows()])
    with pytest.raises(m.Stop,match='MISSING_FIXED'):m.normalize_amounts([daily_rows()[:-1],[]])


def test_existing_standard_affine_is_called_missing_or_warrant_not_identity(monkeypatch):
    event=dict(category=1,_date_int=20051201,original_record=dict(fenhong=2,peigujia=0,songzhuangu=10,peigu=0))
    called=[];fn=m.original.affine.event_affine
    def spy(v):called.append(v);return fn(v)
    monkeypatch.setattr(m.original.affine,'event_affine',spy)
    got=m.summarize_actions('000002.SZ',20051201,[event])['same_date_records'][0]
    assert got['a']=='0.5' and got['b']=='-0.1' and len(called)==1
    for category in (4,14):
        e=dict(event,category=category,original_record=dict(category=category,fenshu=8,xingquanjia=3.73))
        result=m.summarize_actions('000002.SZ',20051201,[e])['same_date_records'][0]
        assert result['standard_factor_status']=='UNRESOLVED_UNSUPPORTED_TERMS' and result['native_factor'] is None
    assert len(called)==1 and m.summarize_actions('000002.SZ',20051201,[])['empty_is_not_identity_factor']


class Life:
    def __init__(self):self.active=True
    def require(self):assert self.active;return dict(event_sha256='synthetic_start')


def test_bar_request_count_and_raw_written_before_schema_check(tmp_path):
    evidence=dict(bar_connect_calls=0,bar_query_calls=0,bar_raw_responses=[]);calls=[]
    class API:
        auto_retry=False;heartbeat=False
        client=SimpleNamespace(settimeout=lambda v:None)
        def get_security_bars(self,**kw):calls.append(kw);return [dict(malformed=True)]
    @contextmanager
    def session(c):yield API()
    budget=m.base.Budget(tmp_path,m.CAPS);c=dict(run_dir=str(tmp_path))
    response=m.fetch_bar_page(Life(),c,budget,m.REQUESTS[0],0,evidence,session)
    assert json.loads((tmp_path/'DAILY_PAGE_0_RAW_RESPONSE.json').read_text())['response']==response
    assert calls==[m.REQUESTS[0]]
    with pytest.raises(m.Stop):m.fetch_bar_page(Life(),c,budget,dict(m.REQUESTS[0],count=801),0,evidence,session)
    assert len(calls)==1
    evidence['bar_query_calls']=2
    with pytest.raises(m.Stop):m.fetch_bar_page(Life(),c,budget,m.REQUESTS[1],1,evidence,session)
    assert len(calls)==1


def fixture_run(tmp_path,monkeypatch):
    root=tmp_path/'M33_SOURCE_REFRESH_01'/'OUTPUT_01';review=tmp_path/'review.json';review.write_text('{}')
    c=dict(state='FROZEN_READY_FOR_ACTIVATION',execution_id='M33_SYNTHETIC',config_sha256='config',code_composite_sha256='code',
        batch_root=str(tmp_path),run_dir=str(root),semantic_id='synthetic',review_receipt_path=str(review))
    activation=dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',execution_id=c['execution_id'],config_sha256='config',code_composite_sha256='code',review_receipt_sha256=m.registry.sha(review))
    history=[]
    class RunLife:
        def __init__(self,*args):self.started=False;self.finished=False
        def start(self,receipt):history.append('START');self.started=True
        def require(self):assert self.started and not self.finished;return dict(event_sha256='synthetic')
        def finish(self,receipt,decision,metrics):history.append('TERMINAL');self.finished=True
    monkeypatch.setattr(m,'RefreshLifecycle',RunLife);monkeypatch.setattr(m.base,'available_memory',lambda:2**36)
    def bars(life,c,budget,request,index,ev):
        assert history==['START'];ev['bar_query_calls']+=1
        m.original.durable_new(Path(c['run_dir'])/f'DAILY_PAGE_{index}_RAW_RESPONSE.json',b'{}')
        return daily_rows() if index==0 else []
    def actions(life,c,budget,symbol,ev):
        assert history==['START'];life.require();ev.update(query_calls=1,connect_calls=1)
        target=dict(m.TARGETS)[symbol];row=dict(year=int(str(target)[:4]),month=int(str(target)[4:6]),day=int(str(target)[6:]),category=14)
        payload=json.dumps(dict(symbol=symbol,response_type='list',response=[row])).encode()
        ev['raw_response_snapshot']=m.original.durable_new(Path(c['run_dir'])/'QUERY_RAW_RESPONSE.json',payload)
        return [dict(_date_int=target,category=14,original_record=row)]
    return c,activation,history,bars,actions


def test_five_queries_raw_actions_unresolved_still_capture_pass_and_terminal(tmp_path,monkeypatch):
    c,a,events,bars,actions=fixture_run(tmp_path,monkeypatch)
    result=m.run(c,object(),a,bar_fetch=bars,action_fetch=actions)
    assert result['status']=='FRESH_SOURCE_CAPTURE_COMPLETE' and result['query_calls']==5
    assert events==['START','TERMINAL'] and not any(v['all_standard'] for v in result['action_terms_resolved'])
    for symbol,_ in m.TARGETS:assert (Path(c['run_dir'])/symbol/'QUERY_RAW_RESPONSE.json').is_file()
    with pytest.raises(FileExistsError):m.run(c,object(),a,bar_fetch=bars,action_fetch=actions)


def test_draft_and_fetch_failure_do_not_retry_and_keep_terminal(tmp_path,monkeypatch):
    c,a,events,bars,actions=fixture_run(tmp_path,monkeypatch)
    c['state']='DRAFT'
    with pytest.raises(m.Stop,match='NOT_ACTIVATED'):m.run(c,object(),a,bar_fetch=bars,action_fetch=actions)
    assert events==[] and not Path(c['run_dir']).exists()
    c['state']='FROZEN_READY_FOR_ACTIVATION';calls=[]
    def fail(*args):calls.append(1);raise ConnectionError('synthetic unavailable')
    with pytest.raises(ConnectionError):m.run(c,object(),a,bar_fetch=fail,action_fetch=actions)
    assert calls==[1] and events==['START','TERMINAL']
    assert json.loads((Path(c['run_dir'])/'RECEIPT.json').read_text())['status']=='FAILED_NO_RETRY'


@pytest.fixture
def archive_fields_reader(tmp_path):
    import hashlib
    source=tmp_path/'original_reader_fixture.py'
    source.write_text("""raise RuntimeError('module main must not execute')
DAY_STRUCT = struct.Struct('<IIIIIfII')
EXT_MARKET_DAY_STRUCT = struct.Struct('<IfffffII')
def is_ext_market_ts(ts_code):
    return ts_code.endswith('.HK')
def read_day_tail(ts_code, path, tail_records=360):
    data=path.open('rb').read(); rows=[]
    for offset in range(0,len(data),32):
        d,o,h,l,c,a,v,_=DAY_STRUCT.unpack(data[offset:offset+32])
        rows.append(dict(trade_date=d,ts_code=ts_code,amount=float(a),vol=int(v)))
    return pd.DataFrame(rows)
""")
    binding=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return binding,m._amount_reader(binding)


def test_archive_fields_original_reader_units_zero_and_cutoff(archive_fields_reader):
    import struct
    _,reader=archive_fields_reader
    pack=lambda d,a,v:struct.pack('<IIIIIfII',d,100,110,90,100,a,v,0)
    payload=pack(20200102,123.5,250)+pack(20200103,0.,0)+pack(20261016,float('nan'),3)
    frame=m.amount_field_frame(payload,'600000.SH',reader)
    assert list(frame.trade_date)==[20200102,20200103]
    assert list(frame.amount_CNY)==[123.5,0.] and list(frame.volume_shares)==[250,0]
    assert str(frame.volume_shares.dtype)=='int64' and str(frame.amount_CNY.dtype)=='float64'


def test_archive_fields_invalid_member_keys_and_nonfinite(archive_fields_reader):
    import struct
    _,reader=archive_fields_reader
    row=lambda d,a=1.:struct.pack('<IIIIIfII',d,100,100,100,100,a,1,0)
    for data,reason in [(b'x','RECORD_SIZE'),(row(20200102)*2,'DATE_KEYS'),(row(20200103)+row(20200102),'DATE_KEYS'),(row(20200102,float('inf')),'INVALID_AMOUNT_VOLUME')]:
        with pytest.raises(m.Stop,match=reason):m.amount_field_frame(data,'600000.SH',reader)


def test_archive_fields_start_guard_precedes_reader_and_arrow_one(archive_fields_reader,monkeypatch):
    import pyarrow as pa
    binding,_=archive_fields_reader
    calls=[]
    def blocked():calls.append('guard');raise PermissionError('NO_DURABLE_START')
    original=m._amount_reader
    monkeypatch.setattr(m,'_amount_reader',lambda _:calls.append('reader'))
    with pytest.raises(PermissionError):m._amount_worker_initialize(dict(original_reader=binding),active_checker=blocked)
    assert calls==['guard']
    monkeypatch.setattr(m,'_amount_reader',original);pa.set_cpu_count(4);pa.set_io_thread_count(8)
    m._amount_worker_initialize(dict(original_reader=binding),active_checker=lambda:calls.append('synthetic_active'))
    assert m._AMOUNT_CONTEXT['arrow_pools']==dict(cpu=1,io=1)


def test_archive_fields_checkpoint_preserves_values_gaps_and_reuses(archive_fields_reader,tmp_path,monkeypatch):
    import hashlib,struct,zipfile
    import pyarrow.parquet as pq
    binding,_=archive_fields_reader
    payload=b''.join(struct.pack('<IIIIIfII',d,100,100,100,100,a,v,0) for d,a,v in [(19990101,12.5,100),(20200102,99.25,700)])
    archive=tmp_path/'raw.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('sh/lday/sh600000.day',payload)
    member=dict(name='sh/lday/sh600000.day',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    run=tmp_path/'outputs';run.mkdir()
    c=dict(original_reader=binding,run_dir=str(run),execution_id='SOFTWARE_FIXTURE',semantic_sha256='fixed-source-semantic',archive=dict(path=str(archive),frozen_stat=m.base.file_identity(archive.stat())),member_byte_limit=2**20,worker_caps=dict(threads=1,rss_bytes=2**31,payload_bytes=2**20,total_wall_sec=30),unit_binding=dict(amount='CNY',vol='shares'))
    m._amount_worker_initialize(c,active_checker=lambda:None)
    result=m._amount_symbol_job(('600000.SH',member,[20200102,20200103]))
    assert result['status']=='PROJECTED_WITH_OLD_KEY_GAPS' and result['missing_old_dates']==[20200103]
    assert result['early_prefix_rows']==1 and result['rows']==2
    table=pq.read_table(result['artifacts'][0]['path']).to_pandas()
    assert list(table.amount_CNY)==[12.5,99.25] and list(table.volume_shares)==[100,700]
    monkeypatch.setattr(m.original,'read_member',lambda *args:(_ for _ in ()).throw(AssertionError('must reuse checkpoint')))
    reused=m._amount_symbol_job(('600000.SH',member,[20200102,20200103]));assert reused['reused']
    Path(result['artifacts'][0]['path']).write_bytes(b'tampered')
    with pytest.raises(m.original.Stop,match='ARTIFACT_IDENTITY_DRIFT'):m._amount_symbol_job(('600000.SH',member,[20200102,20200103]))


def test_archive_fields_no_activation_no_output_or_input_read(tmp_path):
    root=tmp_path/'not-created'
    with pytest.raises(m.Stop,match='FIELD_NOT_REVIEW_ACTIVATED'):
        m.run_amount_projection(dict(run_dir=str(root)))
    assert not root.exists()
