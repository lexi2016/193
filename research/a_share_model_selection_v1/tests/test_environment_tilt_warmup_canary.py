"""Synthetic rows only; original formula/native engine and guarded IO adapters."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_warmup_canary as m

ROOT = Path(os.environ['FACTOR_FACTORY_ROOT'])
ORIGINAL = ROOT/'workspace/.task_worktrees/s1-industry-month-20261020/research/a_share_model_selection_v1/frozen_m0/research/a_share_model_selection_v1'
V193 = ROOT/'research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/inputs/V193_ORIG_SNAPSHOT_UTF8.txt'
V192 = ROOT/'research/a_v192_signal_pool_return_stability_v1/intake/unpacked/base_bundle/formula/V192ORIG_UTF8.txt'
MAP = {k:k for k in ('date','code','open','high','low','close','volume','amount')}
SHA = 'a'*64


@pytest.fixture(scope='module')
def original():
    sys.path.insert(0, str(ORIGINAL))
    import formula_adapter
    recipe = m.support._load(ORIGINAL/'model_selection.py', 'm30_synthetic_recipe')
    return recipe, formula_adapter, dict(v192=V192.read_text(encoding='utf-8-sig'), v193=V193.read_text(encoding='utf-8-sig'))


@pytest.fixture
def history():
    dates = pd.bdate_range('2019-01-02', periods=420)
    i = np.arange(len(dates)); x = 10 + i*.003 + np.sin(i*.2)
    full = pd.DataFrame(dict(date=dates, code=m.SYMBOL, open=x-.1, high=x+.5, low=x-.5,
        close=x, volume=100+i%17, amount=(100+i%17)*x*100))
    old = full.iloc[100:].reset_index(drop=True)
    cal = pd.DataFrame(dict(date=old.date, calendar_index=range(len(old))))
    return full, old, cal


def cache_for(original, old, calendar):
    recipe, formula, texts = original
    a, _ = m.original_features(recipe, formula, old, calendar, texts)
    return a.loc[np.isfinite(a[list(m.FEATURES)]).all(axis=1)].reset_index(drop=True)


def test_original_formula_warmup_difference_with_exact_A_reproduction(original, history):
    full, old, cal = history
    cache = cache_for(original, old, cal)
    recipe, formula, texts = original
    result = m.run_frames(recipe, formula, old, full, cache, cal, texts)
    assert result['A_cache']['exact'] and result['B_computed']
    assert result['decision'] == 'SINGLE_SYMBOL_DIFFERENT'
    assert result['B_cache']['columns']['DISTW']['byte_difference_count'] > 0
    assert result['universe_parity'] == 'CANNOT_CONFIRM'
    assert result['whole_market_arms'] == result['labels'] == result['fit_calls'] == result['query_calls'] == 0


def test_A_failure_stops_before_B_original_formula(original, history):
    full, old, cal = history; cache = cache_for(original, old, cal)
    cache.loc[0, 'C'] += .01
    recipe, formula, texts = original; calls=[]
    proxy = SimpleNamespace(CORE_OUTPUTS=formula.CORE_OUTPUTS)
    def evaluate(panel, **kwargs):
        calls.append(len(panel)); return formula.formula_feature_frame(panel, **kwargs)
    proxy.formula_feature_frame = evaluate
    result = m.run_frames(recipe, proxy, old, full, cache, cal, texts)
    assert result['decision'] == 'A_REPRODUCTION_FAILED_STOP' and not result['B_computed']
    assert calls == [len(old)]


def test_earlier_bad_OHLC_taint_crosses_into_old_cache(original, history):
    full, old, cal = history; cache = cache_for(original, old, cal)
    full.loc[10, 'low'] = 0
    recipe, formula, texts = original
    result = m.run_frames(recipe, formula, old, full, cache, cal, texts)
    assert result['A_cache']['exact']
    assert all(v['right_finite_count'] == 0 for v in result['B_cache']['columns'].values())
    assert result['B_cache']['columns']['C']['finite_mask_difference_count'] == len(cache)


def feature_frame():
    frame = pd.DataFrame(dict(date=pd.bdate_range('2020-01-01', periods=3), code=m.SYMBOL))
    return frame.assign(**{c:np.ones(3) for c in m.FEATURES})


def test_byte_comparison_signed_zero_nonfinite_and_difference_dates():
    a=feature_frame();b=a.copy()
    a.loc[0, 'C']=0.;b.loc[0, 'C']=-0.
    b.loc[1, 'JVAL']=np.nan;b.loc[2, 'JVAL']=np.inf
    result=m.compare(a,b)
    assert not result['exact'] and result['columns']['C']['byte_difference_count']==1
    assert result['columns']['C']['finite_mask_difference_count']==0
    assert result['columns']['JVAL']['finite_mask_difference_count']==2
    assert result['columns']['JVAL']['first_difference_date']=='2020-01-02'
    assert result['columns']['JVAL']['last_difference_date']=='2020-01-03'
    assert result['columns']['JVAL']['right_nan_count']==result['columns']['JVAL']['right_inf_count']==1


def test_nan_payload_bits_are_not_silently_equated():
    a=feature_frame();b=a.copy()
    a['C']=np.array([0x7ff8000000000001,0,0],dtype='<u8').view('<f8')
    b['C']=np.array([0x7ff8000000000002,0,0],dtype='<u8').view('<f8')
    result=m.compare(a,b)['columns']['C']
    assert result['byte_difference_count']==1 and result['finite_mask_difference_count']==0


def test_missing_reference_key_no_refill():
    a=feature_frame()
    with pytest.raises(m.Blocked,match='REFERENCE_KEYS_MISSING'):
        m.on_keys(a.iloc[1:],a)
    assert not m.compare(a.iloc[1:],a)['keys_order_exact']
    assert not m.compare(a.iloc[::-1],a)['keys_order_exact']


@pytest.mark.parametrize('bad',['other_code','duplicate','unnormalized','empty','oversize'])
def test_fixed_symbol_keys_and_bounds(bad):
    f=feature_frame()
    if bad=='other_code':f.loc[0,'code']='000001.SZ'
    elif bad=='duplicate':f=pd.concat([f,f.iloc[:1]])
    elif bad=='unnormalized':f.loc[0,'date']+=pd.Timedelta(hours=1)
    elif bad=='empty':f=f.iloc[:0]
    with pytest.raises(m.Blocked):m.symbol_frame(f,max_rows=2 if bad=='oversize' else 10000)


def test_unit_composition_uses_explicit_original_normalizer_and_preserves_old(original,history):
    recipe,_,_=original;full,old,_=history
    prefix=full.iloc[:100].copy();raw=old.copy();raw['amount']/=10000
    before=raw.copy(deep=True)
    a,b=m.compose(recipe,prefix,raw,dict(prefix=MAP,old_qfq=MAP),
        dict(old_amount_multiplier=10000.,accepted_old_numeric_over_raw_CNY=.0001),expected_prefix_rows=100)
    expected=recipe.normalize_price_panel(before,MAP,amount_multiplier=10000.)
    pd.testing.assert_frame_equal(a,expected,check_exact=True)
    pd.testing.assert_frame_equal(raw,before,check_exact=True)
    pd.testing.assert_frame_equal(b.loc[b.date>=a.date.min()].reset_index(drop=True),a,check_exact=True)
    with pytest.raises(m.Blocked,match='PREFIX_OVERLAPS'):
        m.compose(recipe,old.iloc[:100],raw,dict(prefix=MAP,old_qfq=MAP),
            dict(old_amount_multiplier=10000.,accepted_old_numeric_over_raw_CNY=.0001),expected_prefix_rows=100)
    with pytest.raises(m.Blocked,match='ROW_BOUND'):
        m.compose(recipe,prefix,raw,dict(prefix=MAP,old_qfq=MAP),
            dict(old_amount_multiplier=10000.,accepted_old_numeric_over_raw_CNY=.0001),expected_prefix_rows=100,max_rows=200)
    expected_prefix=prefix.copy();expected_prefix['amount']*=.0001
    expected_prefix=recipe.normalize_price_panel(expected_prefix,MAP,amount_multiplier=10000.)
    pd.testing.assert_frame_equal(b.iloc[:100].reset_index(drop=True),expected_prefix,check_exact=True)


class Active:
    def __init__(self,active=True):self.active=active
    def require_active(self):
        if not self.active:raise m.Blocked('NO_START')


def bound_adapter(tmp_path, name, frame, suffix='.parquet'):
    path=tmp_path/(name+suffix)
    if suffix=='.parquet':frame.to_parquet(path,index=False)
    else:frame.to_csv(path,index=False)
    config=dict(inputs={name:dict(path=str(path),sha256=m.support.sha(path))})
    adapter=m.BoundInputs(config,Active(),lambda stage:None)
    adapter.accepted={name:dict(stat=m.support.stat_identity(path),sha256=config['inputs'][name]['sha256'])}
    return adapter


def test_filtered_parquet_larger_universe_never_materializes_full_panel(tmp_path):
    n=11000
    frame=pd.DataFrame(dict(code=['000001.SZ']*n+[m.SYMBOL]*3,amount=np.arange(n+3,dtype=float)))
    adapter=bound_adapter(tmp_path,'old_qfq',frame)
    result=adapter.frame('old_qfq',['code','amount'],code_column='code')
    assert len(result)==3 and result.code.eq(m.SYMBOL).all()


@pytest.mark.parametrize('kind',['not_started','not_admitted','drift','too_many','wrong_symbol'])
def test_streaming_admission_and_resource_fail_closed(tmp_path,kind):
    n=10001 if kind=='too_many' else 3
    frame=pd.DataFrame(dict(code=[m.SYMBOL]*n,amount=np.arange(n,dtype=float)))
    if kind=='wrong_symbol':frame.loc[0,'code']='000001.SZ'
    adapter=bound_adapter(tmp_path,'prefix',frame,'.csv')
    if kind=='not_started':adapter.session.active=False
    if kind=='not_admitted':adapter.accepted=None
    if kind=='drift':Path(adapter.config['inputs']['prefix']['path']).write_text('changed')
    with pytest.raises(m.Blocked):adapter.frame('prefix',['code','amount'],code_column='code')


class Guard:
    def __init__(self):self.events=[]
    def claim_no_fit(self,**kwargs):self.events.append(('START',kwargs));return {'claim':'durable'}
    def append_no_fit_terminal(self,claim,**kwargs):self.events.append(('TERMINAL',kwargs))


def test_official_session_durable_start_and_one_terminal_on_failure():
    guard=Guard();session=m.Session(guard,dict(execution_id='synthetic',contract=dict(sha256=SHA)))
    with pytest.raises(m.Blocked):session.require_active()
    session.start();session.require_active();session.finish(False,'SyntheticFailure');session.finish(False,'ignored')
    assert [e[0] for e in guard.events]==['START','TERMINAL']
    assert not guard.events[-1][1]['succeeded']
    assert guard.events[0][1]['arm']=='M30_FORMULA_ONLY'


def config_for(tmp_path):
    file=tmp_path/'synthetic_source';file.write_text('synthetic')
    ref=dict(path=str(file),sha256=m.support.sha(file))
    sources={k:ref.copy() for k in ('recipe','formula','engine','fast_builder','c_source','c_binary','support',
        'resource_probe','guard_adapter','registry_adapter','registry_builder','registry_validator','raw_reader','v192','v193')}
    inputs={k:ref.copy() for k in ('prefix','old_qfq','old_m0','old_calendar','raw_day','fast_bind','unit_ruling','m27_checkpoint','m27_terminal')}
    inputs['unit_ruling']['sha256']=m.UNIT_RULING_SHA256
    contract=tmp_path/'contract.json';contract.write_text('{}')
    return dict(state='FROZEN_READY_FOR_ACTIVATION',policy=m.POLICY,caps=m.CAPS,sources=sources,inputs=inputs,
        authentication={k:dict(mode=mode, **({'expected_bytes':file.stat().st_size} if mode=='HASH_BIND_AFTER_START' else {}))
                        for k,mode in m.AUTHENTICATION_MODES.items()},
        contract=dict(path=str(contract),sha256=m.support.sha(contract)),fast_bind_origins={'original_passport':ref},prefix_terminal_event_sha256=SHA,
        columns=dict(prefix=MAP,old_qfq=MAP),units=copy.deepcopy(m.UNIT_POLICY),
        output_root=str(m.BATCH/'M30_SYNTHETIC_NOT_CREATED'),execution_id='synthetic',
        registry_path=str(m.BATCH.parents[3]/'contracts/EXPERIMENT_REGISTRY.jsonl'),
        executor_owner='/root/environment_tilt_design_writer',python=sys.executable,**m.FLAGS)


@pytest.mark.parametrize('kind',['draft','unit_unknown','old_multiplier','wrong_caps','null_source','owner','no_terminal'])
def test_frozen_admission_configuration_guards(tmp_path,kind):
    c=config_for(tmp_path)
    if kind=='draft':c['state']='DRAFT'
    elif kind=='unit_unknown':c['units']['status']='SEMANTIC_CONFLICT'
    elif kind=='old_multiplier':c['units']['old_amount_multiplier']=1.
    elif kind=='wrong_caps':c['caps']={**m.CAPS,'workers':2}
    elif kind=='null_source':c['sources']['c_source']=None
    elif kind=='owner':c['executor_owner']='other'
    elif kind=='no_terminal':c['prefix_terminal_event_sha256']=None
    with pytest.raises(m.Blocked):m.validate_config(c)


def test_exact_activation_and_source_drift(tmp_path):
    c=config_for(tmp_path);m.validate_config(c)
    a=dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',execution_id='synthetic',scientific_identity=m.scientific_identity(c),
        wrapper_sha256=m.support.sha(m.__file__),config_sha256=m.support.digest(c),contract_sha256=c['contract']['sha256'],review_receipt_sha256=SHA)
    m.require_activation(c,a)
    c=copy.deepcopy(c);c['execution_id']='changed'
    with pytest.raises(m.Blocked,match='ACTIVATION'):m.require_activation(c,a)
    c=config_for(tmp_path);Path(c['sources']['c_source']['path']).write_text('drift')
    with pytest.raises(m.Blocked,match='SOURCE_OR_C_BINARY_DRIFT'):m.original_modules(c)


def unit_frames(ratio=1.):
    dates=pd.to_datetime(m.UNIT_DATES)
    raw=np.asarray([100000003.,200000007.,300000011.,400000013.,500000017.],dtype='<f4').astype('<f8')
    a=pd.DataFrame(dict(date=dates,code=m.SYMBOL,amount=raw*ratio))
    b=pd.DataFrame(dict(date=dates,code=m.SYMBOL,amount=raw))
    return a,b


@pytest.mark.parametrize('ratio',[1.,.0001])
def test_frozen_float32_unit_relation_and_prefix_effective_multiplier(ratio):
    a,b=unit_frames(ratio);result=m.unit_admission(a,b)
    assert result['B_allowed'] and result['accepted_old_numeric_over_raw_CNY']==ratio
    assert result['prefix_amount_multiplier']==ratio*10000
    assert len(result['pairs'])==5 and not result['physical_or_economic_unit_certification']


def test_unit_relation_accepts_only_original_float32_quantization():
    a,b=unit_frames()
    # Within the stored raw-float32 cell; no fitted/relative tolerance.
    a['amount']+=1.
    assert m.unit_admission(a,b)['B_allowed']
    a.loc[0,'amount']+=1000.
    assert not m.unit_admission(a,b)['B_allowed']


@pytest.mark.parametrize('kind',['missing','zero','nan','infinity','negative','wrong_ratio','mixed_ratios','not_raw_float32','duplicate','other_code'])
def test_unit_admission_has_no_date_substitution_scale_fit_or_invalid_pair_skip(kind):
    a,b=unit_frames()
    if kind=='missing':a=a.iloc[1:]
    elif kind=='zero':a.loc[0,'amount']=0.
    elif kind=='nan':a.loc[0,'amount']=np.nan
    elif kind=='infinity':b.loc[0,'amount']=np.inf
    elif kind=='negative':b.loc[0,'amount']=-1.
    elif kind=='wrong_ratio':a['amount']*=2
    elif kind=='mixed_ratios':a.loc[0,'amount']*=.0001
    elif kind=='not_raw_float32':b.loc[0,'amount']+=1.
    elif kind=='duplicate':a=pd.concat([a,a.iloc[:1]])
    elif kind=='other_code':a.loc[0,'code']='000001.SZ'
    if kind in ('duplicate','other_code'):
        with pytest.raises(m.Blocked):m.unit_admission(a,b)
    else:
        result=m.unit_admission(a,b)
        assert result['decision']=='UNIT_ADMISSION_BLOCKED' and not result['B_allowed']


def activation_for(c):
    return dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',execution_id=c['execution_id'],
        scientific_identity=m.scientific_identity(c),wrapper_sha256=m.support.sha(m.__file__),
        config_sha256=m.support.digest(c),contract_sha256=c['contract']['sha256'],review_receipt_sha256=SHA)


def test_registered_actual_entry_archives_unit_block_and_terminal_without_B(tmp_path,monkeypatch,original,history):
    c=config_for(tmp_path);monkeypatch.setattr(m,'BATCH',tmp_path)
    c['output_root']=str(tmp_path/'M30_SYNTHETIC_RUN')
    c['registry_path']=str(m.BATCH.parents[3]/'contracts/EXPERIMENT_REGISTRY.jsonl')
    c['sources']['guard_adapter']=dict(path=__file__,sha256=m.support.sha(__file__))
    full,old,cal=history
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'original_modules',lambda config:original)
    monkeypatch.setattr(m.support,'resource_snapshot',lambda config:{'synthetic_resource':True})
    class Adapter:
        def __init__(self,config,session,check):self.session=session;self.accepted={'synthetic':True}
        def admit(self):self.session.require_active()
        def frame(self,name,*args,**kwargs):
            self.session.require_active()
            assert name in ('prefix','old_qfq')
            return full.iloc[:100].copy() if name=='prefix' else old.copy()
        def raw_amounts(self):return old[['date','code','amount']]
    monkeypatch.setattr(m,'BoundInputs',Adapter)
    guard=Guard();result=m.run_registered(c,activation_for(c),guard)
    assert result['status']=='BLOCKED_NO_RETRY' and not result['result']['B_computed']
    assert not guard.events[-1][1]['succeeded']
    assert (Path(c['output_root'])/'CANARY_RECEIPT.json').is_file()
    with pytest.raises(FileExistsError):m.run_registered(c,activation_for(c),guard)
    assert len(guard.events)==2


def test_registered_read_failure_has_archived_failure_and_one_terminal(tmp_path,monkeypatch,original):
    c=config_for(tmp_path);monkeypatch.setattr(m,'BATCH',tmp_path)
    c['output_root']=str(tmp_path/'M30_SYNTHETIC_FAILURE')
    c['registry_path']=str(m.BATCH.parents[3]/'contracts/EXPERIMENT_REGISTRY.jsonl')
    c['sources']['guard_adapter']=dict(path=__file__,sha256=m.support.sha(__file__))
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'original_modules',lambda config:original)
    monkeypatch.setattr(m.support,'resource_snapshot',lambda config:{'synthetic_resource':True})
    class Adapter:
        def __init__(self,config,session,check):pass
        def admit(self):
            m.support.write_json(Path(c['output_root'])/'AUTHENTICATION_RECEIPT.json',dict(status='FAILED_NO_NUMERIC_READ'))
            raise m.Blocked('SyntheticAdmissionFailure')
    monkeypatch.setattr(m,'BoundInputs',Adapter)
    guard=Guard()
    with pytest.raises(m.Blocked,match='SyntheticAdmissionFailure'):m.run_registered(c,activation_for(c),guard)
    assert [v[0] for v in guard.events]==['START','TERMINAL']
    import json
    saved=json.loads((Path(c['output_root'])/'CANARY_RECEIPT.json').read_text())
    assert saved['status']=='FAILED_OR_INTERRUPTED_NO_RETRY'
    assert saved['authentication_receipt_ref']['sha256']==m.support.sha(Path(c['output_root'])/'AUTHENTICATION_RECEIPT.json')


def test_original_raw_reader_on_synthetic_day_bytes_no_second_parser(tmp_path):
    import struct
    dates=pd.bdate_range('1999-01-01',periods=6408)
    path=tmp_path/'synthetic.day'
    path.write_bytes(b''.join(struct.pack('<IIIIIfII',int(d.strftime('%Y%m%d')),1000,1100,900,1000,
        float(np.float32(100000003.)),100,0) for d in dates))
    source=Path(os.environ['FACTOR_FACTORY_RAW_READER'])
    config=dict(inputs={'raw_day':dict(path=str(path),sha256=m.support.sha(path))},
        sources={'raw_reader':dict(path=str(source),sha256=m.support.sha(source))})
    adapter=m.BoundInputs(config,Active(),lambda stage:None)
    adapter.accepted={'raw_day':dict(stat=m.support.stat_identity(path))}
    result=adapter.raw_amounts()
    assert len(result)==6408 and result.code.eq(m.SYMBOL).all()
    assert result.amount.eq(float(np.float32(100000003.))).all()
    assert result.date.iloc[0]==dates[0] and result.date.iloc[-1]==dates[-1]


@pytest.mark.parametrize('kind',['valid','precision_valid','precision_review_drift','new_use_source_drift','units','prefix_checkpoint','terminal','payload_stat'])
def test_original_passport_fast_bind_and_M27_admission(tmp_path,kind,monkeypatch):
    c=precision_fixture(tmp_path,monkeypatch)[0] if kind.startswith('precision_') else config_for(tmp_path)
    c['fast_bind_origins']={'original_passport':c['contract']}
    files={k:c['inputs'][k] for k in ('prefix','old_qfq','old_m0','old_calendar','raw_day')}
    origins=m23_metadata_fixture(tmp_path,c,files)
    c['fast_bind_origins'].update(origins)
    proof=dict(schema='M30_SYMBOL_FORMULA_FAST_BIND_V1',decision='PASS',symbol=m.SYMBOL,files=files,
        origins=c['fast_bind_origins'],sources=copy.deepcopy(c['sources']),units=copy.deepcopy(c['units']),
        authentication=c['authentication'],prefix_terminal_event_sha256=SHA,
        bound_files={k:dict(sha256=files[k]['sha256'],stat=m.support.stat_identity(files[k]['path'])) for k in ('old_m0','old_calendar')})
    ruling=dict(symbol=m.SYMBOL,unit_check=dict(dates=[int(d.replace('-','')) for d in m.UNIT_DATES],
        candidate_old_numeric_per_raw_CNY=[1,.0001]),**m.FLAGS)
    checkpoint=dict(symbol=m.SYMBOL,status='HISTORICAL_PREFIX_ONLY',block_reasons=[],prefix_rows=5147,
        coverage=dict(history_scope='ALL_ORIGINAL_AVAILABLE_BEFORE_OLD_FIRST'),
        artifacts=[{**files['prefix'], 'amount':'CNY','vol':'shares'}],
        raw_member=dict(sha256=files['raw_day']['sha256'],bytes=205056,name='sh\\lday\\sh600000.day'))
    terminal=dict(execution_id=m.PREFIX_EXECUTION,terminal_event_sha256=SHA,closed=True,decision='HISTORICAL_PREFIX_ONLY')
    if kind=='new_use_source_drift':proof['sources']['c_source']['sha256']='b'*64
    elif kind=='units':ruling['unit_check']['dates'][0]=20210803
    elif kind=='prefix_checkpoint':checkpoint['prefix_rows']=5146
    elif kind=='terminal':terminal['closed']=False
    elif kind=='payload_stat':proof['bound_files']['old_m0']['stat']['bytes']+=1
    elif kind=='precision_review_drift':Path(c['precision_admission']['independent_review']['path']).write_text('{}')
    stages=[];archives=[]
    adapter=m.BoundInputs(c,Active(),stages.append)
    adapter.archive=lambda receipt:archives.append(copy.deepcopy(receipt))
    metadata=dict(fast_bind=proof,unit_ruling=ruling,m27_checkpoint=checkpoint,m27_terminal=terminal)
    adapter.metadata=lambda name:metadata[name]
    if kind in ('valid','precision_valid'):
        adapter.admit();assert set(adapter.accepted)==set(files)
        assert (adapter.precision_evidence is not None)==(kind=='precision_valid')
        assert archives[-1]['status']=='AUTHENTICATED_BEFORE_NUMERIC_READ'
        assert {k:v['mode'] for k,v in adapter.accepted.items()}==m.AUTHENTICATION_MODES
        assert not adapter.accepted['old_m0']['rehashed'] and not adapter.accepted['old_calendar']['rehashed']
        assert not any('hash_chunk_old_m0'==stage or 'hash_chunk_old_calendar'==stage for stage in stages)
        assert adapter.hash_bytes==sum(c['authentication'][k]['expected_bytes'] for k in ('prefix','old_qfq','raw_day'))
    else:
        with pytest.raises((m.Blocked,m.support.Blocked)):adapter.admit()
        if kind in ('payload_stat','precision_review_drift'):
            assert adapter.accepted is None
            assert not any(stage.startswith('before_hash_') for stage in stages)


def m23_metadata_fixture(tmp_path,c,files):
    import json
    config={k:{} for k in ('policy','recipe_sources','preparation_receipt','original_binding','new_use_passport',
        'runtime_versions','resource_probe_source','original_empty_pool_proof')}
    config['cache_files']=dict(m0=files['old_m0'],calendar=files['old_calendar'])
    preflight=dict(status='PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW',
        input_binding_identity=m.support.input_binding_identity(config),
        cache_stats=dict(m0=m.support.stat_identity(files['old_m0']['path']),calendar=m.support.stat_identity(files['old_calendar']['path'])))
    result={}
    for name,value in (('m23_config',config),('m23_preflight',preflight)):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value))
        result[name]=dict(path=str(path),sha256=m.support.sha(path))
    review=dict(verdict='PASS_NARROW_NEW_TIME_USE',preflight_receipt_sha256=result['m23_preflight']['sha256'])
    path=tmp_path/'m23_review.json';path.write_text(json.dumps(review))
    result['m23_review']=dict(path=str(path),sha256=m.support.sha(path))
    return result


@pytest.mark.parametrize('kind',['valid','preflight_sha','input_identity','different_cache','incomplete_stat'])
def test_historical_cache_fast_bind_uses_persisted_SHA_stat_not_new_stat(tmp_path,kind):
    import json
    c=config_for(tmp_path);files={k:c['inputs'][k] for k in m.AUTHENTICATION_MODES}
    refs=m23_metadata_fixture(tmp_path,c,files)
    origins={k:json.loads(Path(v['path']).read_text()) for k,v in refs.items()}
    if kind=='preflight_sha':origins['m23_review']['preflight_receipt_sha256']='b'*64
    elif kind=='input_identity':origins['m23_preflight']['input_binding_identity']='b'*64
    elif kind=='different_cache':files['old_m0']={**files['old_m0'],'sha256':'b'*64}
    elif kind=='incomplete_stat':del origins['m23_preflight']['cache_stats']['calendar']['inode']
    if kind=='valid':
        result=m.historical_cache_bindings(origins,refs,files)
        assert result['old_m0']['stat']==origins['m23_preflight']['cache_stats']['m0']
    else:
        with pytest.raises(m.Blocked):m.historical_cache_bindings(origins,refs,files)


def three_terminal_projections():
    # Exact three projection locations in the reviewer-bound M27 metadata.
    terminal=dict(execution_id=m.PREFIX_EXECUTION,terminal_event_sha256=SHA,
        closed=True,decision='HISTORICAL_PREFIX_ONLY')
    return dict(preserved_checkpoint_terminal_bindings=[{}, {}, {}, copy.deepcopy(terminal)],
        preserved_for_lawful_reuse=[{}, copy.deepcopy(terminal)],
        registry_snapshot=dict(execution_closures=[{}, {}, {}, {}, copy.deepcopy(terminal)]))


def test_same_M27_terminal_identity_at_three_real_metadata_projection_paths():
    result=m.terminal_identity(three_terminal_projections(),SHA)
    assert result['projection_count']==3 and result['distinct_terminal_identity_count']==1
    assert result['closed'] and result['terminal_event_sha256']==SHA


@pytest.mark.parametrize('field,value',[('terminal_event_sha256','b'*64),('closed',False),
    ('closed',1),('decision','BLOCKED'),('status','FAILED')])
def test_duplicate_terminal_projection_cannot_hide_conflicting_identity_or_state(field,value):
    document=three_terminal_projections()
    document['preserved_for_lawful_reuse'][1][field]=value
    with pytest.raises(m.Blocked,match='CONFLICTING'):m.terminal_identity(document,SHA)


def test_terminal_projection_status_conflict_and_missing_terminal_rejected():
    document=three_terminal_projections()
    document['preserved_for_lawful_reuse'][1]['status']='COMPLETED'
    document['registry_snapshot']['execution_closures'][4]['status']='CLOSED'
    with pytest.raises(m.Blocked,match='CONFLICTING'):m.terminal_identity(document,SHA)
    with pytest.raises(m.Blocked):m.terminal_identity({},SHA)


@pytest.mark.parametrize('kind',['missing','fallback','null_bytes','total_bytes','fast_hash'])
def test_explicit_authentication_config_has_no_mode_fallback(tmp_path,kind):
    c=config_for(tmp_path)
    if kind=='missing':del c['authentication']['raw_day']
    elif kind=='fallback':c['authentication']['prefix']['mode']='FAST_BIND_ACCEPTED'
    elif kind=='null_bytes':c['authentication']['old_qfq']['expected_bytes']=None
    elif kind=='total_bytes':c['authentication']['old_qfq']['expected_bytes']=m.CAPS['hash_total_bytes']+1
    elif kind=='fast_hash':c['authentication']['old_m0']['expected_bytes']=1
    with pytest.raises(m.Blocked):m.validate_config(c)


def hash_adapter(tmp_path,check=lambda stage:None):
    path=tmp_path/'hash_synthetic.bin';path.write_bytes(b'x'*(2*m.CAPS['hash_chunk_bytes']+17))
    c=dict(inputs={'prefix':dict(path=str(path),sha256=m.support.sha(path))},
        authentication={'prefix':dict(mode='HASH_BIND_AFTER_START',expected_bytes=path.stat().st_size)})
    return m.BoundInputs(c,Active(),check),path


def test_hash_bind_after_START_bounded_stream_and_observed_identity(tmp_path):
    checks=[];adapter,path=hash_adapter(tmp_path,checks.append)
    result=adapter.hash_bind('prefix')
    assert result['sha256']==adapter.config['inputs']['prefix']['sha256']
    assert result['mode']=='HASH_BIND_AFTER_START' and result['stat_stable']
    assert result['bytes_hashed']==path.stat().st_size==adapter.hash_bytes
    assert result['chunks']==3 and checks.count('after_hash_chunk_prefix')==3


@pytest.mark.parametrize('kind',['before_start','sha_drift','bytes_drift','stat_drift','total_cap','resource','timeout','wrong_mode'])
def test_hash_bind_stops_without_reusing_current_bytes_as_expected_identity(tmp_path,monkeypatch,kind):
    adapter,path=hash_adapter(tmp_path)
    if kind=='before_start':adapter.session.active=False
    elif kind=='sha_drift':adapter.config['inputs']['prefix']['sha256']='b'*64
    elif kind=='bytes_drift':adapter.config['authentication']['prefix']['expected_bytes']+=1
    elif kind=='stat_drift':
        def mutate(stage):
            if stage=='after_hash_chunk_prefix':
                import os
                stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
        adapter.check=mutate
    elif kind=='total_cap':monkeypatch.setattr(m,'CAPS',{**m.CAPS,'hash_total_bytes':1})
    elif kind in ('resource','timeout'):
        def fail(stage):
            if stage=='hash_chunk_prefix':
                if kind=='timeout':raise TimeoutError('synthetic wall cap')
                raise m.Blocked('synthetic resource stop')
        adapter.check=fail
    elif kind=='wrong_mode':adapter.config['authentication']['prefix']['mode']='FAST_BIND_ACCEPTED'
    with pytest.raises((m.Blocked,TimeoutError)):adapter.hash_bind('prefix')
    assert adapter.accepted is None


def test_hash_stream_never_reads_beyond_frozen_bytes_if_file_grows(tmp_path):
    adapter,path=hash_adapter(tmp_path);original_bytes=path.stat().st_size;changed=False
    def grow(stage):
        nonlocal changed
        if stage=='after_hash_chunk_prefix' and not changed:
            with path.open('ab') as handle:handle.write(b'new bytes')
            changed=True
    adapter.check=grow
    with pytest.raises(m.Blocked,match='STAT_CHANGED'):adapter.hash_bind('prefix')
    assert adapter.hash_bytes==original_bytes



def precision_fixture(tmp_path, monkeypatch):
    """Synthetic independent receipt chain; no historical amount values read."""
    import json
    c=config_for(tmp_path)
    raw=np.asarray([123456789.,234567891.,345678913.,456789121.,567891231.],dtype='<f4').astype(float)
    old=np.asarray([float(format(float(np.float32(v/10000)),'.7g'))*10000 for v in raw])
    a=pd.DataFrame(dict(date=pd.to_datetime(m.UNIT_DATES),code=m.SYMBOL,amount=old))
    b=pd.DataFrame(dict(date=pd.to_datetime(m.UNIT_DATES),code=m.SYMBOL,amount=raw))
    pairs=[dict(date=date,code=m.SYMBOL,old_numeric=float(x),raw_CNY=float(y))
           for date,x,y in zip(m.UNIT_DATES,old,raw)]
    def save(name,value):
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=m.support.sha(path))
    old_ref=save('prior_canary',dict(status='BLOCKED_NO_RETRY',input_refs=c['inputs'],
        unit_admission=dict(reason='NO_UNIQUE_FIXED_RATIO_ALL_FIVE',pairs=pairs)))
    source=dict(schema='M40_AMOUNT_PRECISION_SOURCE_REVIEW_V1',
        source_rule_gap_blocks_original_data_reuse=False,
        candidate_rounding=dict(all_five_match=True,universal_tolerance_admitted=False),
        evidence=dict(m30_receipt=old_ref),**m.FLAGS)
    source_ref=save('precision_source',source)
    review_ref=save('precision_review',dict(status='PASS_WITH_LIMITS',**m.FLAGS))
    ruling=dict(schema='M40_CONTROLLER_REUSE_RULING_V1',decision='REUSE_ACCEPTED_ORIGINAL_AMOUNT_VALUES',
        review_ref=review_ref,writer_ref=source_ref,raw_CNY_to_existing_input_numeric_ratio=1,
        old_values_unchanged=True,old_m30_failure_unchanged=True,
        feature_matrix_changes_from_real_history_repairs_not_waived=True,
        original_model_and_evaluation_unchanged=True,**m.FLAGS)
    ruling_ref=save('precision_ruling',ruling)
    refs=dict(source_receipt=source_ref,independent_review=review_ref,controller_ruling=ruling_ref)
    monkeypatch.setattr(m,'M40_SOURCE_RECEIPT_SHA256',source_ref['sha256'])
    monkeypatch.setattr(m,'M40_REVIEW_SHA256',review_ref['sha256'])
    monkeypatch.setattr(m,'M40_CONTROLLER_SHA256',ruling_ref['sha256'])
    units={**m.PRECISION_UNIT_POLICY,'precision_source_receipt_sha256':source_ref['sha256']}
    policy={**m.POLICY,'unit_admission':units}
    monkeypatch.setattr(m,'PRECISION_UNIT_POLICY',units);monkeypatch.setattr(m,'PRECISION_POLICY',policy)
    c.update(policy=copy.deepcopy(policy),units=copy.deepcopy(units),precision_admission=refs)
    return c,a,b


def test_precision_receipt_accepts_fixed_reviewed_rounding_without_changing_original_values(tmp_path,monkeypatch):
    c,a,b=precision_fixture(tmp_path,monkeypatch)
    m.validate_config(c)
    before_a,before_b=a.copy(deep=True),b.copy(deep=True)
    assert not m.unit_admission(a,b)['B_allowed']  # Legacy predicate is unchanged.
    evidence=m.m40_precision_evidence(c)
    result=m.unit_admission(a,b,precision_evidence=evidence)
    assert result['B_allowed'] and result['accepted_old_numeric_over_raw_CNY']==1
    assert result['prefix_amount_multiplier']==10000 and result['old_numeric_values_preserved']
    assert result['tolerance']=='NONE_FIXED_REVIEWED_PAIR_REUSE_ONLY'
    assert not result['physical_or_economic_unit_certification']
    pd.testing.assert_frame_equal(a,before_a,check_exact=True)
    pd.testing.assert_frame_equal(b,before_b,check_exact=True)
    assert m.scientific_identity(c)!=m.scientific_identity({k:v for k,v in c.items() if k!='precision_admission'})


@pytest.mark.parametrize('kind',['no_source','no_review','no_ruling','wrong_review_sha','wrong_ruling_sha',
    'wrong_units','legacy_policy','source_drift','review_drift','ruling_drift','different_payload'])
def test_precision_receipt_missing_identity_units_or_provenance_stops(kind,tmp_path,monkeypatch):
    c,a,b=precision_fixture(tmp_path,monkeypatch)
    if kind.startswith('no_'):del c['precision_admission'][{'no_source':'source_receipt','no_review':'independent_review','no_ruling':'controller_ruling'}[kind]]
    elif kind=='wrong_review_sha':c['precision_admission']['independent_review']['sha256']='b'*64
    elif kind=='wrong_ruling_sha':c['precision_admission']['controller_ruling']['sha256']='b'*64
    elif kind=='wrong_units':c['units']['candidate_old_numeric_over_raw_CNY']=[.0001]
    elif kind=='legacy_policy':c['policy']=m.POLICY
    elif kind.endswith('_drift'):
        key={'source_drift':'source_receipt','review_drift':'independent_review','ruling_drift':'controller_ruling'}[kind]
        Path(c['precision_admission'][key]['path']).write_text('{}')
    elif kind=='different_payload':c['inputs']['old_qfq']={**c['inputs']['old_qfq'],'sha256':'b'*64}
    with pytest.raises((m.Blocked,m.support.Blocked)):
        m.validate_config(c);m.m40_precision_evidence(c)


@pytest.mark.parametrize('kind',['date','code','duplicate','missing','unit_scale','changed_old','changed_raw','nan'])
def test_precision_receipt_never_substitutes_dates_units_or_changes_fixed_values(kind,tmp_path,monkeypatch):
    c,a,b=precision_fixture(tmp_path,monkeypatch);evidence=m.m40_precision_evidence(c)
    if kind=='date':a.loc[0,'date']+=pd.Timedelta(days=1)
    elif kind=='code':a.loc[0,'code']='000001.SZ'
    elif kind=='duplicate':a=pd.concat([a,a.iloc[:1]])
    elif kind=='missing':b=b.iloc[1:]
    elif kind=='unit_scale':a['amount']*=.0001
    elif kind=='changed_old':a.loc[0,'amount']+=.01  # No generic tolerance, however small.
    elif kind=='changed_raw':b.loc[0,'amount']+=.01
    elif kind=='nan':a.loc[0,'amount']=np.nan
    with pytest.raises(m.Blocked):m.unit_admission(a,b,precision_evidence=evidence)


def test_precision_ratio1_composition_preserves_old_numeric_recipe(original,history):
    recipe,_,_=original;full,old,_=history;prefix=full.iloc[:100].copy()
    before=old.copy(deep=True)
    a,b=m.compose(recipe,prefix,old,dict(prefix=MAP,old_qfq=MAP),
        dict(old_amount_multiplier=10000.,accepted_old_numeric_over_raw_CNY=1.),expected_prefix_rows=100)
    pd.testing.assert_frame_equal(old,before,check_exact=True)
    pd.testing.assert_frame_equal(a,recipe.normalize_price_panel(before,MAP,amount_multiplier=10000.),check_exact=True)
    pd.testing.assert_frame_equal(b.iloc[100:].reset_index(drop=True),a,check_exact=True)
    pd.testing.assert_frame_equal(b.iloc[:100].reset_index(drop=True),
        recipe.normalize_price_panel(prefix,MAP,amount_multiplier=10000.),check_exact=True)


def test_precision_registered_path_reaches_formula_and_preserves_material_difference_result(tmp_path,monkeypatch,original):
    c,a,b=precision_fixture(tmp_path,monkeypatch);monkeypatch.setattr(m,'BATCH',tmp_path)
    c['output_root']=str(tmp_path/'M30_SYNTHETIC_PRECISION_RUN')
    c['registry_path']=str(m.BATCH.parents[3]/'contracts/EXPERIMENT_REGISTRY.jsonl')
    c['sources']['guard_adapter']=dict(path=__file__,sha256=m.support.sha(__file__))
    old=a.assign(open=10.,high=11.,low=9.,close=10.,volume=100.)
    prefix=pd.DataFrame(dict(date=pd.bdate_range('2020-01-01',periods=100),code=m.SYMBOL,
        open=10.,high=11.,low=9.,close=10.,volume=100.,amount=1000.))
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'original_modules',lambda config:original)
    monkeypatch.setattr(m.support,'resource_snapshot',lambda config:{'synthetic_resource':True})
    class Adapter:
        def __init__(self,config,session,check):self.session=session;self.config=config;self.accepted={'synthetic':True}
        def admit(self):
            self.session.require_active();self.precision_evidence=m.m40_precision_evidence(self.config)
        def frame(self,name,*args,**kwargs):
            self.session.require_active()
            if name=='prefix':return prefix.copy()
            if name=='old_qfq':return old.copy()
            if name=='old_m0':return feature_frame()
            return pd.DataFrame(dict(date=a.date,calendar_index=np.arange(len(a))))
        def raw_amounts(self):return b.copy()
    original_compose=m.compose
    def compose(*args,**kwargs):return original_compose(*args,expected_prefix_rows=100,**kwargs)
    def run_frames(recipe,formula,old_normalized,full,*args,**kwargs):
        pd.testing.assert_frame_equal(old_normalized,recipe.normalize_price_panel(old,MAP,amount_multiplier=10000.),check_exact=True)
        assert len(full)==105
        return dict(decision='SINGLE_SYMBOL_DIFFERENT',B_computed=True,
                    A_cache={'exact':True},B_cache={'exact':False},universe_parity='CANNOT_CONFIRM')
    monkeypatch.setattr(m,'BoundInputs',Adapter);monkeypatch.setattr(m,'compose',compose);monkeypatch.setattr(m,'run_frames',run_frames)
    guard=Guard();result=m.run_registered(c,activation_for(c),guard)
    assert result['unit_admission']['B_allowed'] and result['result']['decision']=='SINGLE_SYMBOL_DIFFERENT'
    assert result['result']['B_cache']['exact'] is False and guard.events[-1][1]['succeeded']
    assert result['policy_sha256']==m.support.digest(c['policy'])
    with pytest.raises(FileExistsError):m.run_registered(c,activation_for(c),guard)
    assert len(guard.events)==2



def m0_stream_fixture(tmp_path, rows=3000, symbol_every=17):
    import pyarrow.parquet as pq
    i=np.arange(rows)
    frame=pd.DataFrame(dict(date=pd.bdate_range('1990-01-01',periods=rows),
        code=np.where(i%symbol_every==0,m.SYMBOL,'000001.SZ')))
    for j,name in enumerate(m.FEATURES):frame[name]=i*.0001+j
    frame.loc[0,m.FEATURES[0]]=-0.
    if rows>2*symbol_every:
        frame.loc[symbol_every,m.FEATURES[1]]=np.nan
        frame.loc[2*symbol_every,m.FEATURES[2]]=np.inf
    frame=frame.sample(frac=1,random_state=193).reset_index(drop=True)
    path=tmp_path/'old_m0.parquet'
    frame.to_parquet(path,index=False,row_group_size=733)
    # Whole-frame reference is deliberately small and SYNTHETIC only.
    with pq.ParquetFile(path) as reader:reference=reader.read().to_pandas()
    reference=reference.loc[reference.code.eq(m.SYMBOL)].reset_index(drop=True)
    config=dict(inputs={'old_m0':dict(path=str(path),sha256=m.support.sha(path))})
    adapter=m.BoundInputs(config,Active(),lambda stage:None)
    adapter.accepted={'old_m0':dict(stat=m.support.stat_identity(path))}
    return adapter,path,reference


def track_m0_parquet(monkeypatch):
    import pyarrow.parquet as pq
    original=pq.ParquetFile;calls=[]
    class Tracked:
        def __init__(self,path,**kwargs):
            assert kwargs==dict(pre_buffer=False,buffer_size=64*1024)
            self.reader=original(path,**kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):self.reader.close()
        @property
        def metadata(self):return self.reader.metadata
        def iter_batches(self,**kwargs):
            cols=kwargs['columns'];calls.append(tuple(cols))
            assert len(cols)==1 or cols==['date','code']
            assert kwargs['batch_size']==512 and kwargs['use_threads'] is False
            for batch in self.reader.iter_batches(**kwargs):
                assert len(batch)<=512
                yield batch
    monkeypatch.setattr(pq,'ParquetFile',Tracked)
    return calls


def test_m0_column_stream_uses_keys_then_single_columns_preserving_order_and_float_bytes(tmp_path,monkeypatch):
    import pyarrow.dataset as ds
    adapter,path,expected=m0_stream_fixture(tmp_path)
    calls=track_m0_parquet(monkeypatch);stages=[];adapter.check=stages.append
    def no_wide_dataset(*args,**kwargs):raise AssertionError('old_m0 must not use a wide filtered dataset scan')
    monkeypatch.setattr(ds,'dataset',no_wide_dataset)
    result=adapter.frame('old_m0',['date','code',*m.FEATURES],code_column='code')
    pd.testing.assert_frame_equal(result,expected,check_exact=True)
    for column in m.FEATURES:
        np.testing.assert_array_equal(result[column].to_numpy(dtype='<f8').view('<u8'),
                                      expected[column].to_numpy(dtype='<f8').view('<u8'))
    assert calls==[('date','code'),*((name,) for name in m.FEATURES)]
    assert len(result)<3000 and result.code.eq(m.SYMBOL).all()
    assert stages[0]=='before_old_m0_keys' and stages[-1]=='read_old_m0'
    for name in m.FEATURES:
        assert 'before_old_m0_column_'+name in stages and 'after_old_m0_column_'+name in stages
    assert m.CAPS['rss_bytes']==m.CAPS['parent_rss_bytes']==m.CAPS['worker_rss_bytes']==2**31


@pytest.mark.parametrize('kind',['oversize','empty','wrong_schema','missing_code_filter','resource_stop','stat_drift'])
def test_m0_column_stream_bounds_and_admission_stops(kind,tmp_path,monkeypatch):
    adapter,path,expected=m0_stream_fixture(tmp_path,rows=10001 if kind=='oversize' else 40,
                                           symbol_every=1 if kind=='oversize' else 17)
    if kind=='empty':
        frame=pd.read_parquet(path);frame['code']='000001.SZ';frame.to_parquet(path,index=False)
        adapter.accepted['old_m0']['stat']=m.support.stat_identity(path)
    calls=track_m0_parquet(monkeypatch)
    columns=['date','code',*m.FEATURES];code='code'
    if kind=='wrong_schema':columns=columns[:-1]
    elif kind=='missing_code_filter':code=None
    elif kind=='resource_stop':
        def stop(stage):
            if stage=='after_old_m0_keys':raise m.Blocked('SYNTHETIC_RESOURCE_STOP')
        adapter.check=stop
    elif kind=='stat_drift':
        def drift(stage):
            if stage=='after_old_m0_column_'+m.FEATURES[0]:
                stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
        adapter.check=drift
    with pytest.raises(m.Blocked):adapter.frame('old_m0',columns,code_column=code)
    if kind in ('oversize','empty','resource_stop'):assert calls==[('date','code')]
    if kind in ('wrong_schema','missing_code_filter'):assert calls==[]
