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
