"""M11 pure/synthetic only. No actual member, event payload or query is read."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import signal
import struct
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import environment_tilt_mac_prefix_canary as m
from test_environment_tilt_mac_recovery import frame, fixture_run, namespace


def event(date,bonus=10,rights=0,category=1):
    return dict(_date_int=date,category=category,xdxr_fenhong=0,
        xdxr_peigujia=0,xdxr_songzhuangu=bonus,xdxr_peigu=rights)


def identity_fixture():
    # Old source starts2021, prefix2020; event is between old bars.
    dates=[20200102,20200103,20210104,20210105,20210106,20210107]
    raw=frame(dates=dates,code='000001.SZ')
    chosen=dict(symbol='000001.SZ',anchor=20210107,first_old_bar=20210104,
        selected_event=event(20210106))
    oracle=raw.loc[raw.trade_date.ge(20210104)].copy()
    oracle.loc[oracle.trade_date.lt(20210106),list(m.affine.OHLC)]/=2
    return raw,oracle,[event(20210106)],chosen


def test_identity_anchor_nonunit_composition_without_fit():
    raw,oracle,events,chosen=identity_fixture()
    result,candidate=m.analyze(raw,oracle,events,chosen)
    assert result['anchor_identity'] and result['overlap_price_mismatches']==0
    assert result['overlap_OHLC_denominator']==16
    assert result['decision']=='ONE_SYMBOL_CANDIDATE_PREFIX_ONLY'
    assert [r['trade_date'] for r in candidate]==[20200102,20200103]
    assert Decimal(candidate[0]['close']['cent'])==5
    events=[event(20210105),event(20210106)]
    rebuilt=m.reconstruct(raw,events,chosen['anchor'])
    assert Decimal(rebuilt[0]['close']['cent'])==Decimal('2.5')
    assert Decimal(rebuilt[-1]['close']['cent'])==10


def test_latest_common_anchor_must_be_identity_no_affine_calibration():
    raw,oracle,events,chosen=identity_fixture()
    oracle[list(m.affine.OHLC)]-=1
    result,_=m.analyze(raw,oracle,events,chosen)
    assert not result['anchor_identity']
    assert 'LATEST_COMMON_ANCHOR_NOT_IDENTITY' in result['prefix_block_reasons']


def test_old_date_keys_string_and_integer_have_identical_canary_semantics():
    raw,oracle,events,chosen=identity_fixture()
    expected,_=m.analyze(raw,oracle,events,chosen)
    oracle['trade_date']=oracle.trade_date.astype(str)
    actual,_=m.analyze(raw,oracle,events,chosen)
    assert actual==expected


def test_whole_overlap_mismatch_not_only_local_event_window():
    raw,oracle,events,chosen=identity_fixture()
    oracle.loc[oracle.trade_date.eq(20210104),'open']+=.01
    result,_=m.analyze(raw,oracle,events,chosen)
    assert result['overlap_price_mismatches']==1
    assert result['decision']=='PREFIX_BLOCKED'


def test_whole_overlap_missing_key_blocks_output():
    raw,oracle,events,chosen=identity_fixture()
    raw=raw.loc[raw.trade_date.ne(20210105)]
    result,_=m.analyze(raw,oracle,events,chosen)
    assert not result['overlap_exact_keys'] and result['decision']=='PREFIX_BLOCKED'


@pytest.mark.parametrize('category',[4,11,12,13,14,99])
def test_unresolved_price_effective_type_keeps_overlap_diagnostic_blocks_prefix(category):
    raw,oracle,events,chosen=identity_fixture();events.append(event(20210105,category=category))
    result,_=m.analyze(raw,oracle,events,chosen)
    assert result['overlap_price_mismatches']==0
    assert result['decision']=='PREFIX_BLOCKED'
    assert result['unsupported_participating_categories']=={str(category):1}


@pytest.mark.parametrize('category',[2,3,5,6,7,8,9,10])
def test_bound_informational_labels_do_not_invent_adjustment_or_universal_block(category):
    raw,oracle,events,chosen=identity_fixture();events.append(event(20210105,category=category))
    result,_=m.analyze(raw,oracle,events,chosen)
    assert result['decision']=='ONE_SYMBOL_CANDIDATE_PREFIX_ONLY'
    assert result['informational_participating_categories']=={str(category):1}


def test_no_residual_selection_and_preferred_then_lexicographic_fallback():
    keys={code:[20210104,20210105,20210106,20210107] for code in ['600000.SH','000001.SZ','000002.SZ','000003.SZ']}
    touched=[]
    def load(code):return keys[code],dict(bytes=32,status='SYNTHETIC')
    chosen=m.select_symbol(keys,{'000001.SZ':[event(20210106)]},{'000002.SZ':[event(20210105)]},load,touched)
    assert chosen['symbol']=='000001.SZ' and len(touched)==1
    chosen=m.select_symbol(keys,{}, {'000003.SZ':[event(20210105)],'000002.SZ':[event(20210106)]},load,[])
    assert chosen['symbol']=='000002.SZ'
    assert chosen['anchor']==20210107


def test_selection_needs_nonunit_event_inside_overlap_and_strict_both_sides():
    keys={'000001.SZ':[20210104,20210105,20210106]}
    for e in [event(20210106),event(20210103),event(20210105,bonus=0)]:
        with pytest.raises(m.PrefixStop,match='NO_ELIGIBLE'):
            m.select_symbol(keys,{'000001.SZ':[e]},{},lambda c:(keys[c],dict(bytes=32)),[])


def test_dateword_only_decoder_and_key_integrity():
    payload=b''.join(struct.pack('<IIIIIfII',d,1000,1200,800,1000,10000.,1000,0) for d in [20210104,20210105])
    assert m.raw_dates(payload)==[20210104,20210105]
    with pytest.raises(m.PrefixStop):m.raw_dates(payload+b'x')


def test_source_event_cutoff_preserves_all_categories_and_raw_fields():
    rows=[dict(event_date='2026-10-15',record=dict(category=11,suogu=0.5)),
          dict(event_date='2026-10-16',record=dict(category=1,fenhong=9))]
    parsed=m.normalize_allcategory(rows,20261015)
    assert len(parsed)==1 and parsed[0]['original_record']['suogu']==.5
    assert parsed[0]['category']==11 and 'xdxr_fenhong' not in parsed[0]


def test_prefix_output_new_version_only_after_complete_gates(tmp_path):
    raw,oracle,events,chosen=identity_fixture();result,candidate=m.analyze(raw,oracle,events,chosen)
    class Life:
        def require(self):return True
    output=m.emit_prefix(tmp_path,result,candidate,raw,Life())
    assert output['rows']==2 and output['ancillary_feature_use']=='BLOCKED'
    with pytest.raises(FileExistsError):m.emit_prefix(tmp_path,result,candidate,raw,Life())
    other=tmp_path/'blocked';other.mkdir();result['decision']='PREFIX_BLOCKED'
    assert m.emit_prefix(other,result,candidate,raw,Life()) is None
    assert list(other.iterdir())==[]


def test_registry_identity_exact_baseline_start_and_terminal():
    c=dict(execution_id='PREFIX_SYNTHETIC_01',contract_path='c',contract_sha256='a',config_path='i',config_sha256='b',code_composite_sha256='d',diagnostic_dates={},caps=m.base.CAPS)
    for kind in ['RUN_STARTED','TERMINAL']:
        e=m.PrefixLifecycle(None,c).event(kind,'RUNNING','r',{})
        assert e['baseline_id']==m.BASELINE
        assert e['one_variable']=='IDENTITY_ANCHOR_CATEGORY1_AFFINE_NONUNIT_OVERLAP_AND_SEPARATE_PREFIX'
        assert e['runtime']['active_concurrency_limit']==4


def test_query_exactly_once_no_retry_and_BJ_forbidden(monkeypatch,tmp_path):
    class Life:
        def require(self):return True
    class Budget:
        def check(self):pass
    class Client:
        def settimeout(self,value):assert value==5
    class API:
        auto_retry=False;heartbeat=False;client=Client()
        def get_xdxr_info(self,market,code):
            assert (market,code)==(0,'000002')
            return [dict(year=2021,month=1,day=6,category=1,fenhong=0,peigujia=0,songzhuangu=10,peigu=0)]
    @contextlib.contextmanager
    def session(**kwargs):yield API()
    monkeypatch.setattr(m,'query_wall',lambda *args:contextlib.nullcontext())
    evidence=dict(query_calls=0,connect_calls=0)
    c=dict(query_sources={},query_host='SYNTHETIC_NO_NETWORK',query_port=1,run_dir=str(tmp_path))
    result=m.query_events(Life(),c,Budget(),'000002.SZ',evidence,session)
    assert len(result)==1 and evidence['query_calls']==evidence['connect_calls']==1
    assert evidence['wall_sec']>=0
    snapshot=json.loads(Path(evidence['raw_response_snapshot']['path']).read_text())
    assert snapshot['response'][0]['month']==1
    with pytest.raises(m.PrefixStop):m.query_events(Life(),c,Budget(),'000002.SZ',evidence,session)
    with pytest.raises(m.PrefixStop):m.query_events(Life(),c,Budget(),'920000.BJ',dict(query_calls=0,connect_calls=0),session)


def test_query_hardwall_bypasses_legacy_Exception_wrapper(monkeypatch):
    class Budget:
        def check(self):pass
    times=iter([0,31]);monkeypatch.setattr(m.time,'monotonic',lambda:next(times))
    previous=signal.getsignal(signal.SIGALRM)
    with m.query_wall(Budget(),30):
        check=signal.getsignal(signal.SIGALRM)
        with pytest.raises(m.QueryWallStop):check()
    assert signal.getsignal(signal.SIGALRM)==previous


def test_activation_explicitly_binds_one_query_network_permission(tmp_path):
    receipt=tmp_path/'review.json';receipt.write_text('{}')
    proof=dict(status='APPROVED_FOR_ONE_MAC_PREFIX_CANARY',writer='environment_tilt_design_writer',reviewer='independent-reviewer',query_count_limit=1,network_allowed=True,fit_count=0,receipt_path=str(receipt),receipt_sha256=m.sha(receipt),code_composite_sha256='a',config_sha256='b',contract_sha256='c')
    activation=tmp_path/'activation.json';activation.write_text(json.dumps(proof))
    c=dict(activation_path=str(activation),code_composite_sha256='a',config_sha256='b',contract_sha256='c')
    m.validate_activation(c)
    proof['network_allowed']=False;activation.write_text(json.dumps(proof))
    with pytest.raises(PermissionError):m.validate_activation(c)


def test_registered_fixture_whole_overlap_terminal_no_network_and_measured_pools(fixture_run,monkeypatch):
    c,adapter=fixture_run;root=Path(c['fixture_root']);c['run_dir']=str(root/'M11_SYNTHETIC_01');c['execution_id']='PREFIX_SYNTHETIC_01'
    inventory=root/'inventory.csv'
    with inventory.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['market','security_code','member_name','uncompressed_bytes','uncompressed_member_sha256']);writer.writeheader()
        for code,binding in c['members'].items():
            writer.writerow(dict(market=code[-2:].lower(),security_code=code[:6],member_name=binding['name'],uncompressed_bytes=binding['bytes'],uncompressed_member_sha256=binding['sha256']))
    artifact=root/'allcategory.jsonl';date=int(frame().trade_date.iloc[15]);day=pd.to_datetime(str(date)).strftime('%Y-%m-%d')
    artifact.write_text(json.dumps(dict(symbol='000001.SZ',event_date=day,record=dict(category=1,fenhong=0,peigujia=0,songzhuangu=10,peigu=0)))+'\n')
    for key,path in [('inventory',inventory),('allcategory',artifact)]:
        c[key]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    c['allcategory_covered_symbols']=list(m.PREFERRED);c['ancillary_source_binding']={'physical_units':'CANNOT_CONFIRM'}
    import pyarrow as pa
    pa.set_cpu_count(8);pa.set_io_thread_count(8)
    original=pd.read_parquet;observed=[]
    def guarded(*args,**kwargs):
        records=[json.loads(line) for line in Path(c['registry_path']).read_text().splitlines()]
        assert any(e['execution_id']==c['execution_id'] and e['event_type']=='RUN_STARTED' for e in records)
        assert pa.cpu_count()==pa.io_thread_count()==4;observed.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',guarded)
    def forbidden_query(*args,**kwargs):raise AssertionError('NO_QUERY_EXPECTED')
    monkeypatch.setattr(m,'query_events',forbidden_query)
    result=m.run(c,adapter,fixture=True)
    assert result['symbol']=='000001.SZ' and result['decision']=='PREFIX_BLOCKED'
    assert result['query']['query_calls']==0 and result['arrow_pools']==dict(cpu=4,io=4)
    assert result['prefix_artifact'] is None and len(observed)==2
    records=[json.loads(line) for line in Path(c['registry_path']).read_text().splitlines()]
    events=[e for e in records if e['execution_id']==c['execution_id']]
    assert [e['event_type'] for e in events]==['RUN_STARTED','TERMINAL']
    assert all(e['baseline_id']==m.BASELINE for e in events)
    with pytest.raises(FileExistsError):m.run(c,adapter,fixture=True)


@pytest.mark.parametrize('malformed',['month13','category'])
def test_malformed_query_raw_snapshot_survives_normalization_and_terminal(fixture_run,malformed):
    c,adapter=fixture_run;root=Path(c['fixture_root']);c['run_dir']=str(root/'M11_R1_SYNTHETIC_01');c['execution_id']='PREFIX_RESPONSE_SYNTHETIC_01'
    inventory=root/'inventory.csv'
    with inventory.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['market','security_code','member_name','uncompressed_bytes','uncompressed_member_sha256']);writer.writeheader()
        for code,binding in c['members'].items():
            writer.writerow(dict(market=code[-2:].lower(),security_code=code[:6],member_name=binding['name'],uncompressed_bytes=binding['bytes'],uncompressed_member_sha256=binding['sha256']))
    artifact=root/'allcategory.jsonl';artifact.write_text('')
    date=int(frame().trade_date.iloc[15]);day=pd.to_datetime(str(date)).strftime('%Y-%m-%d')
    pd.DataFrame([dict(ts_code='000002.SZ',_xdxr_trade_date=day,xdxr_fenhong=0.,xdxr_peigujia=0.,xdxr_songzhuangu=10.,xdxr_peigu=0.)]).to_parquet(c['actions']['path'],index=False)
    actions_path=Path(c['actions']['path']);c['actions'].update(bytes=actions_path.stat().st_size,sha256=hashlib.sha256(actions_path.read_bytes()).hexdigest())
    for key,path in [('inventory',inventory),('allcategory',artifact)]:
        c[key]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    c.update(allcategory_covered_symbols=list(m.PREFERRED),query_sources={},query_host='SYNTHETIC_NO_NETWORK',query_port=1,ancillary_source_binding={'physical_units':'CANNOT_CONFIRM'})
    class Client:
        def settimeout(self,value):assert value==5
    calls=[]
    class API:
        auto_retry=False;heartbeat=False;client=Client()
        def get_xdxr_info(self,market,code):
            calls.append((market,code))
            return [dict(year=2021,month=13 if malformed=='month13' else 1,day=6,
                category=1 if malformed=='month13' else 'INVALID_CATEGORY',fenhong=0,peigujia=0,songzhuangu=10,peigu=0)]
    @contextlib.contextmanager
    def session(**kwargs):yield API()
    with pytest.raises(Exception):m.run(c,adapter,fixture=True,session_factory=session)
    receipt=json.loads((Path(c['run_dir'])/'RECEIPT.json').read_text())
    assert receipt['query']['connect_calls']==receipt['query']['query_calls']==1
    assert calls==[(0,'000002')]
    binding=receipt['query']['raw_response_snapshot'];raw=Path(binding['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==binding['sha256']
    saved=json.loads(raw)['response'][0]
    assert saved['month']==(13 if malformed=='month13' else 1)
    assert saved['category']==(1 if malformed=='month13' else 'INVALID_CATEGORY')
    assert not (Path(c['run_dir'])/'SELECTED_EVENTS.json').exists()
    events=[json.loads(line) for line in Path(c['registry_path']).read_text().splitlines()]
    own=[e for e in events if e['execution_id']==c['execution_id']]
    assert [e['event_type'] for e in own]==['RUN_STARTED','TERMINAL']
    assert own[-1]['decision'].startswith('FAILED_')
