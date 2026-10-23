"""Synthetic flags-only diagnostic; no real market payload, network or registry."""
from pathlib import Path
import contextlib
import json
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/a_share_model_selection_v1'))
import environment_tilt_month_bonus_stability as m
from environment_tilt_mac_registry import digest

@pytest.fixture
def sample():
    calendar=pd.DataFrame(dict(date=pd.bdate_range('2026-01-01','2026-05-05')))
    calendar['calendar_index']=range(len(calendar))
    panel=pd.DataFrame([dict(date=pd.Timestamp(d),code=c,close=10.)for s,t,c in m.KEYS for d in (s,t)])
    recipe=m.old.support._load(Path(m.__file__).parent/'frozen_m0/research/a_share_model_selection_v1/model_selection.py','synthetic_diag_original_recipe')
    return panel,calendar,recipe


def test_exact_shared_calendar_and_flags_only(sample):
    panel,calendar,recipe=sample
    result=m.flags_for_keys(panel,calendar,recipe)
    assert len(result)==8 and all(r['producer_label_finite']for r in result)
    assert all(r['reason']=='LABEL_FINITE_PREVIOUS_GAP_NOT_REPRODUCED'for r in result)
    assert all(set(r)=={'date','target_date','code','signal','target','producer_label_present','producer_label_finite','reason','source_status'}for r in result)
    assert not any(k in json.dumps(result)for k in ('label_return','signal_close','target_close'))

@pytest.mark.parametrize('mode,reason',[
    ('target_missing','TARGET_BAR_ABSENT_STATUS_UNCLASSIFIED'),
    ('signal_missing','SIGNAL_BAR_ABSENT'),
    ('target_nan','TARGET_CLOSE_NONFINITE'),
    ('target_zero','TARGET_CLOSE_NONPOSITIVE'),
    ('signal_nan','SIGNAL_CLOSE_NONFINITE'),
    ('signal_zero','SIGNAL_CLOSE_NONPOSITIVE'),
])
def test_actual_reason_no_nextbar_no_fill(sample,mode,reason):
    panel,cal,recipe=sample;s,t,c=m.KEYS[0];day=s if mode.startswith('signal')else t
    mask=(panel.date==pd.Timestamp(day))&(panel.code==c)
    if mode.endswith('missing'):panel=panel.loc[~mask]
    else:panel.loc[mask,'close']=float('nan')if mode.endswith('nan')else 0.
    result=m.flags_for_keys(panel,cal,recipe)
    assert result[0]['reason']==reason and not result[0]['producer_label_finite']
    assert result[0]['source_status']=='CANNOT_CONFIRM_NO_BOUND_STATUS_FIELD'


def test_duplicate_extra_and_calendar_conflicts_stop(sample):
    p,c,r=sample
    with pytest.raises(m.Blocked,match='DUPLICATE'):m.flags_for_keys(pd.concat([p,p.iloc[:1]]),c,r)
    with pytest.raises(m.Blocked,match='UNREQUESTED'):m.flags_for_keys(pd.concat([p,p.iloc[:1].assign(code='OTHER')]),c,r)
    changed=c.loc[c.date!=pd.Timestamp('2026-01-12')].reset_index(drop=True);changed.calendar_index=range(len(changed))
    with pytest.raises(m.Blocked,match='TARGET_CONFLICT'):m.flags_for_keys(p,changed,r)

class Adapter:
    def __init__(self):self.events=[]
    def append_parity_event(self,e,**kw):e['event_sha256']=digest(e);self.events.append(e)
    def _locked(self):return contextlib.nullcontext()
    def _read_validated(self):return None,self.events

@pytest.mark.parametrize('phase',['complete','read_failed','flags_failed'])
def test_start_precedes_access_terminal_truth_and_failure(tmp_path,monkeypatch,phase):
    review=tmp_path/'review.json';review.write_text('{}')
    c=dict(execution_id=m.EXECUTION,run_dir=str(tmp_path/'run'),caps=m.CAPS.copy(),
        diagnostic_dates=dict(start='2026-01-09',end='2026-04-28'),contract_path='synthetic',
        contract_sha256='contract',config_path='synthetic',config_sha256='config',code_composite_sha256='code',
        selection_ref=dict(path='synthetic_selection',sha256='0'*64))
    a=dict(status='APPROVED_FOR_EIGHT_KEY_DIAGNOSTIC',review_receipt=dict(path=str(review),sha256=m.sha(review)),
        **{k:c[k]for k in ('execution_id','contract_sha256','config_sha256','code_composite_sha256')})
    official=Adapter();monkeypatch.setattr(m.resources,'available_memory',lambda:64*2**30)
    def compute(config,life):
        assert len(official.events)==1 and official.events[0]['truth_access']=='NOT_READ'
        life.require();config['_access']=dict(key_count_attempted=True,key_count_returned=True,panel_read_attempted=True,requested_keys=8)
        if phase=='read_failed':raise m.Blocked('synthetic_read_failed')
        config['_access'].update(panel_returned=True,returned_target_dates=[t for s,t,c in m.KEYS])
        if phase=='flags_failed':raise m.Blocked('synthetic_flags_failed')
        config['_access'].update(flags_completed=True,diagnosed_keys=8)
        return dict(status='EIGHT_KEYS_DIAGNOSED',rows=[{}]*8)
    result=m.run(c,a,official,compute)
    assert result['status']==('EIGHT_KEYS_DIAGNOSED'if phase=='complete'else 'STOPPED')
    assert [e['event_type']for e in official.events]==['RUN_STARTED','TERMINAL']
    start,end=official.events
    assert start['target_date_range']['count']==0
    assert end['target_date_range']['count']==(0 if phase=='read_failed'else 8)
    expected={'complete':'SELECTED_H2_PRODUCER_FLAGS_COMPLETED','read_failed':'SELECTED_H2_PANEL_READ_ATTEMPTED_NOT_RETURNED','flags_failed':'SELECTED_H2_PANEL_RETURNED_FLAGS_NOT_COMPLETED'}
    assert end['truth_access']==expected[phase]
    assert end['deterministic_metrics']['diagnosed_keys']==(8 if phase=='complete'else 0)
    assert end['checkpoint_ref']is None and 'NO_MODEL'not in json.dumps(end)and 'NO_OUTCOME_LABELS'not in json.dumps(end)
    assert end['deterministic_metrics']['truth_positive_denominator']is None
    assert end['deterministic_metrics']['selection_slots']is None
    assert (tmp_path/'run/RECEIPT.json').is_file()


def test_activation_required_before_any_output(tmp_path):
    with pytest.raises(m.Blocked,match='ACTIVATION_REQUIRED'):m.run({'run_dir':str(tmp_path/'none')},{})
    assert not (tmp_path/'none').exists()


def test_real_tiny_parquet_exact_filters_before_subset_decode(sample,tmp_path):
    panel,calendar,recipe=sample
    panel=pd.concat([panel,panel.iloc[:1].assign(code='NOT_REQUESTED')],ignore_index=True)
    panel.to_parquet(tmp_path/'panel.parquet',index=False);calendar.to_parquet(tmp_path/'calendar.parquet',index=False)
    def ref(p):return dict(path=str(p),sha256=m.sha(p))
    def meta(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return ref(p)
    p=ref(tmp_path/'panel.parquet');cal=ref(tmp_path/'calendar.parquet')
    prep=meta('prep.json',dict(status='COMPLETED',panel_rows=17,cache_files={p['path']:p['sha256']}))
    review=meta('review.json',dict(verdict='READBACK_AND_REGISTRY_CHAIN_PASS_WITH_LIMITATIONS; BUSINESS_ACCEPTANCE_CANNOT_CONFIRM'))
    selection=meta('selection.json',dict(records=[dict(date=s,h2_date=t,selections={a:[c]for a in ('A_ORIGINAL','C_MONTH_BONUS')})for s,t,c in m.KEYS]))
    c=dict(caps=m.CAPS,inputs=dict(panel=p,calendar=cal),panel_row_binding=dict(file_rows=17,preparation_ref=prep),
        prior_actual_review_ref=review,selection_ref=selection,recipe_ref=dict(path=recipe.__file__,sha256=m.sha(recipe.__file__)))
    class Active:
        def require(self):return True
    result=m.diagnose(c,Active())
    assert len(result['rows'])==8 and all(r['producer_label_finite']for r in result['rows'])
    assert len(result['input_bindings'])==5


def test_predecode_bound_rejects_17_matched_rows_without_close_materialize(sample,tmp_path,monkeypatch):
    panel,calendar,recipe=sample
    panel=pd.concat([panel,panel.iloc[:1]],ignore_index=True)
    path=tmp_path/'duplicate.parquet';panel.to_parquet(path,index=False)
    c=dict(caps=m.CAPS,inputs=dict(panel=dict(path=str(path),sha256=m.sha(path))),panel_row_binding=dict(file_rows=17))
    class Active:
        def require_active(self):return True
    adapter=m.old.BoundFiles(c,Active())
    pairs=sorted({(d,code)for s,t,code in m.KEYS for d in(s,t)})
    filters=[[('date','==',pd.Timestamp(d)),('code','==',code)]for d,code in pairs]
    def forbidden(*a,**kw):raise AssertionError('close materialized before row bound')
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    with pytest.raises(m.Blocked,match='PREDECODE_ROW_BOUND'):m.read_exact_panel(adapter,c,pairs,filters)
    assert c['_access']['matched_panel_rows']==17 and c['_access']['key_count_returned']
    assert not c['_access'].get('panel_read_attempted')and not c['_access'].get('panel_returned')
    assert not c['_access'].get('flags_completed')


def raw_fixture(tmp_path,mode):
    rows=[dict(trade_date=int(t.replace('-','')),ts_code=code,close=10.)for s,t,code in m.KEYS]
    if mode=='absent':rows=rows[:0]
    if mode=='overflow':rows=rows+[dict(rows[0])for _ in range(9)]
    raw=pd.DataFrame(rows,columns=m.RAW_POLICY['columns']).astype({'trade_date':'int64','ts_code':'string','close':'float64'})
    import pyarrow as pa
    import pyarrow.parquet as pq
    p=tmp_path/'raw.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([('trade_date',pa.int64()),('ts_code',pa.string()),('close',pa.float64())])),p)
    def ref(p):return dict(path=str(p),sha256=m.sha(p))
    def meta(name,value):
        f=tmp_path/name;f.write_text(json.dumps(value));return ref(f)
    r=ref(p);projection=meta('projection.json',dict(panels=dict(assembled_qfq=dict(path=str(p),metadata_row_count=len(raw),field_summaries=dict(trade_date=dict(type='int64'))))))
    binding=meta('binding.json',dict(sources=dict(price_panel=dict(**r,receipt=projection,column_map=dict(date='trade_date',code='ts_code')))))
    canonical=meta('canonical.json',dict(rows=[dict(date=s,target_date=t,code=code,target=dict(present=False))for s,t,code in m.KEYS]))
    review=meta('review.json',dict(boundary_status=dict(diagnostic_integrity='PASS_WITH_LIMITATIONS')))
    return dict(caps=m.CAPS,inputs=dict(raw=r),raw_rows=len(raw),original_binding_ref=binding,original_projection_ref=projection,canonical_flags_ref=canonical,canonical_actual_review_ref=review)

@pytest.mark.parametrize('mode',['absent','present'])
def test_raw_exact_targets_distinguish_source_absence_without_prices(tmp_path,mode):
    c=raw_fixture(tmp_path,mode)
    class Active:
        def require(self):return True
    report=m.raw_target_flags(c,Active())
    assert len(report['rows'])==8 and report['new_fit_count']==0
    assert all(r['raw_target_present']==(mode=='present')for r in report['rows'])
    expected='RAW_AND_CANONICAL_TARGET_ABSENT_STATUS_UNCLASSIFIED'if mode=='absent'else 'RAW_TARGET_PRESENT_CANONICAL_ABSENT_REQUIRES_EXACT_FIX'
    assert all(r['reason']==expected for r in report['rows'])
    assert all(r['full_OHLC_normalization_validation']=='NOT_RUN_THREE_COLUMNS_ONLY'for r in report['rows'])
    assert 'label_return'not in json.dumps(report['rows'])and '10.0'not in json.dumps(report['rows'])
    assert c['_access']['flags_completed']


def test_expansion_fixed_ac_joint_prior_year_dplus1_no_market_gate(sample):
    import types
    p,cal,recipe=sample
    cal=pd.DataFrame(dict(date=pd.bdate_range('2026-01-01','2026-07-02')));cal['calendar_index']=range(len(cal))
    frame=pd.DataFrame([dict(date=pd.Timestamp('2026-01-30'),code=f'{i:06d}.SZ',prediction=100-i)for i in range(1,61)])
    lookup=pd.DataFrame(dict(signal_year=[2026],max_prior_year=[2025]))
    seen=[]
    def project(keys,roster,calendar,look):
        seen.extend(keys.code.tolist());assert look.max_prior_year.max()==2025
        out=keys.copy();out['p']=[.1 if c=='000012.SZ'else 0. for c in keys.code];out['missing']=0.;return out
    prior=types.SimpleNamespace(PRIOR_COLUMNS=['p','missing'],project_prior=project)
    result=m.select_ac(frame,cal,lookup,pd.DataFrame(),prior,'2026H1')
    r=next(r for r in result if r['date']=='2026-01-30')
    assert r['prior_anchor_date']=='2026-02-02' and '000012.SZ'in r['selections']['C_MONTH_BONUS']
    assert set(r['selections'])=={'A_ORIGINAL','C_MONTH_BONUS'}and r['selections']['A_ORIGINAL']==frame.code[:10].tolist()
    assert len(seen)==50 and '000051.SZ'not in seen
    assert any(r['reason']=='NO_SAVED_SCORE_DATE_ZERO_POOL_UNCONFIRMED'for r in result)
    lookup.max_prior_year=2026
    with pytest.raises(m.Blocked,match='CUTOFF'):m.select_ac(frame,cal,lookup,pd.DataFrame(),prior,'2026H1')


def paired_rows(fold,n,offset=0,delta=.001):
    start={'2024H1':'2024-01-02','2024H2':'2024-07-01','2025H1':'2025-01-02','2025H2':'2025-07-01'}.get(fold,'2023-07-03')
    return [dict(fold=fold,date=d.strftime('%Y-%m-%d'),calendar_index=offset+i,valid=True,reason=None,missing_codes=[],returns=dict(A_ORIGINAL=-.01+i*.0001,C_MONTH_BONUS=-.01+i*.0001+delta))for i,d in enumerate(pd.bdate_range(start,periods=n))]


def test_pair_missing_keeps_nominal_date_and_ten_denominator():
    codes=[f'{i:06d}.SZ'for i in range(10)]
    r=dict(fold='2024H1',date='2024-01-02',calendar_index=0,reason=None,selections={a:codes for a in m.EXP_POLICY['arms']})
    outcomes={(r['date'],c):.01 for c in codes};outcomes.pop((r['date'],codes[0]))
    rows=m.evaluate_ac([r],outcomes);summary=m.ac_stats(rows)
    assert len(rows)==1 and not rows[0]['valid']and rows[0]['returns']is None
    assert summary['nominal_dates']==1 and summary['valid_common_dates']==0 and not summary['complete_coverage']
    assert summary['full_nominal_mean_delta']is None
    r['selections']['A_ORIGINAL']=codes[:9]
    with pytest.raises(m.Blocked,match='TEN'):m.evaluate_ac([r],outcomes)


def test_bootstrap_preserves_fold_gap_and_short_run_boundaries():
    a=paired_rows('2024H1',40);b=paired_rows('2024H2',40,40)
    assert [len(x)for x in m.contiguous_runs(a+b)]==[40,40]
    broken=[dict(r)for r in a];broken[20]['valid']=False
    assert [len(x)for x in m.contiguous_runs(broken)]==[20,19]
    assert m.paired_interval(broken)['lower']is None
    ci=m.paired_interval(a+b)
    assert ci['run_lengths']==[40,40]and ci['lower']==pytest.approx(.001)
    assert ci==m.paired_interval(a+b)and ci['seed']==20261021 and ci['block']==20 and ci['replicates']==2000


def test_screen_same_version_four_halves_no_aux_partial_or_seen_votes():
    rows={f:paired_rows(f,40,k*40)for k,f in enumerate(m.PRIMARY)}
    rows['2023H2']=paired_rows('2023H2',40,delta=-10.)
    rows['2026H2_PARTIAL_TO_2026_10_15']=paired_rows('2026H2_PARTIAL_TO_2026_10_15',40,delta=-10.)
    report=m.stability_summary(rows)
    assert report['screen_passed']and report['primary']['nominal_dates']==160 and report['positive_halves']==4
    assert report['primary']['descriptive_paired_mean_delta']==pytest.approx(.001)
    assert report['auxiliary_2023H2']['descriptive_paired_mean_delta']<0
    assert report['nav']is None and report['net_return']is None
    rows['2024H1'][0]['valid']=False;rows['2024H1'][0]['returns']=None;rows['2024H1'][0]['reason']='MISSING_LABEL'
    assert not m.stability_summary(rows)['screen_passed']


def test_cvar_is_fractional_tail_mean_not_quantile():
    r=paired_rows('2024H1',21)
    for i,x in enumerate(r):x['returns']=dict(A_ORIGINAL=-1. if i==0 else 0.,C_MONTH_BONUS=-.5 if i==0 else 0.)
    summary=m.ac_stats(r)
    assert summary['arms']['A_ORIGINAL']['cvar5']==pytest.approx(-1/1.05)
    assert summary['cvar5_difference']==pytest.approx(.5/1.05)


def test_expansion_checkpoint_existing_lock_validator_guard(tmp_path):
    class CheckpointAdapter(Adapter):
        def _append_unlocked(self,raw,events,e):e['event_sha256']=digest(e);self.events.append(e)
    a=CheckpointAdapter()
    c=dict(execution_id='SYNTHETIC_EXP',contract_path='contract',contract_sha256='c',config_path='config',config_sha256='i',code_composite_sha256='code',caps=m.EXP_CAPS,diagnostic_dates={},input_versions={})
    life=m.ExpansionLifecycle(a,c);life.start(tmp_path/'receipt')
    out=tmp_path/'cell.json';out.write_text('{}');artifact=dict(path=str(out),sha256=m.sha(out))
    life.checkpoint(artifact,'2024H1')
    assert [e['event_type']for e in a.events]==['RUN_STARTED','CHECKPOINT']
    assert a.events[-1]['input_identity']['result_checkpoint']==artifact
    with pytest.raises(m.Blocked,match='UNIQUE'):life.checkpoint(artifact,'2024H1')
    life.finish(tmp_path/'receipt','COMPLETE')
    with pytest.raises(PermissionError):life.checkpoint(artifact,'2024H2')

@pytest.mark.parametrize('drift',[None,'review','sidecar','replay','stage','row_oversize','row_metadata_mismatch'])
def test_saved_checkpoint_acceptance_chain_precedes_unpack(sample,monkeypatch,drift):
    import copy
    import joblib
    p,cal,recipe=sample
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2026-01-09']*10),code=[f'{i:06d}.SZ'for i in range(10)],prediction=list(range(10))))
    keys=recipe.key_frame_identity(frame)
    identity=dict(arm='M0',horizon=2,fold='2026H1',config_sha256='fitcfg',contract_sha256='contract',data_identity=dict(test_keys=keys))
    docs=dict(metadata=dict(status='COMPLETED',artifact_sha256='checkpointsha',identity=identity),
        review=dict(verdict='METRICS_CONSISTENT',research_decision='M4_GATE_FAILED',artifacts=dict(output_manifest_sha256='outputsha',contract_sha256='contract')),
        lifecycle=dict(verdict='PASS',output_manifest_sha256='outputsha'),output=dict(contract_sha256='contract'),
        stage=dict(contract_sha256='contract',cell_results=[dict(checkpoint_path='checkpoint',status='COMPLETED',fit_called=True)]),
        replay=dict(status='VERIFIED',checkpoint_path='checkpoint',checkpoint_metadata_path='metadata',config_sha256='fitcfg',checkpoint_sha256='checkpointsha',checkpoint_metadata_sha256='metasha',data_identity=identity['data_identity'],prediction_identity={'hash':'same','row_count':len(frame)},replay_prediction_identity={'hash':'same','row_count':len(frame)},top10_parity={'status':'EXACT'},comparison=dict(key_order_equal=True,prediction_equal=True)))
    spec=dict(fold='2026H1',format='original_checkpoint',accepted_identity=dict(test=dict(row_count=len(frame))),artifact=dict(path='checkpoint',sha256='checkpointsha'),metadata=dict(path='metadata',sha256='metasha'),as_run_config_sha256='fitcfg',as_run_contract_sha256='contract',as_run_input_version='V1',acceptance_chain={k:dict(path=v,sha256='outputsha'if k=='output_manifest'else k+'sha')for k,v in [('review','review'),('lifecycle_review','lifecycle'),('stage_manifest','stage'),('output_manifest','output'),('checkpoint_replay','replay')]})
    if drift=='review':docs['review']['verdict']='WRONG'
    if drift=='sidecar':docs['metadata']['identity']['fold']='WRONG'
    if drift=='replay':docs['replay']['checkpoint_sha256']='WRONG'
    if drift=='stage':docs['stage']['cell_results'][0]['status']='FAILED'
    if drift=='row_oversize':
        for field in ['row_count','unique_key_count']:identity['data_identity']['test_keys'][field]=700001
        spec['accepted_identity']['test']['row_count']=700001
        docs['replay']['prediction_identity']['row_count']=700001;docs['replay']['replay_prediction_identity']['row_count']=700001
    if drift=='row_metadata_mismatch':spec['accepted_identity']['test']['row_count']=len(frame)+1
    opened=[]
    class Fake:
        def metadata(self,r):return copy.deepcopy(docs[r['path']])
        def authenticate(self,r):opened.append(r);return {'synthetic':True}
    monkeypatch.setattr(joblib,'load',lambda p:dict(predictions=frame))
    monkeypatch.setattr(m.old.support,'stat_identity',lambda p:{'synthetic':True})
    monkeypatch.setattr(m.old,'restore_model_time',lambda *a,**kw:dict(value='2026-01-08',kind='EXACT_MAX',method='SYNTHETIC_ACTUAL_TIME_PROOF'))
    c=dict(sources=dict(recipe=dict(sha256='recipe')),inputs=dict(calendar=dict(sha256='calendar')))
    if drift:
        with pytest.raises(m.Blocked):m.load_saved_scores(Fake(),c,spec,cal,recipe)
        assert not opened
    else:
        got,proof=m.load_saved_scores(Fake(),c,spec,cal,recipe)
        assert got.equals(frame)and proof['model_train_time']['evidence_sha256']=='metasha'
        assert proof['existing_research_decision']=='M4_GATE_FAILED'and not proof['new_predict_called']
        assert len(opened)==1


@pytest.mark.parametrize('resume',[False,True])
def test_expansion_saves_selection_before_selected_close_read(tmp_path,sample,monkeypatch,resume):
    import types
    recipe=sample[2]
    calendar=pd.DataFrame(dict(date=pd.bdate_range('2023-01-01','2026-10-15')));calendar['calendar_index']=range(len(calendar))
    frame=pd.DataFrame(dict(date=pd.to_datetime(['2023-07-03']*10),code=[f'{i:06d}.SZ'for i in range(10)],prediction=list(range(10))))
    target=calendar.date.iloc[calendar.index[calendar.date==pd.Timestamp('2023-07-03')][0]+2]
    panel=pd.DataFrame([dict(date=day,code=code,close=10.)for day in [pd.Timestamp('2023-07-03'),target]for code in frame.code]);path=tmp_path/'panel.parquet';panel.to_parquet(path,index=False)
    panel_ref=dict(path=str(path),sha256=m.sha(path))
    c=dict(run_dir=str(tmp_path),caps=m.EXP_CAPS,inventory_ref='inventory',version_delta_ref='delta',panel_row_binding=dict(file_rows=20,preparation_ref='prep'),inputs={k:dict(path=k)for k in ['calendar','industry_year','month_year','roster']})
    c['inputs']['panel']=panel_ref;c['_raw_prerequisite_ref']='raw_prior'
    lookup=pd.DataFrame(dict(signal_year=[2023,2024,2025,2026],max_prior_year=[2022,2023,2024,2025]))
    def project(keys,*args):
        out=keys.copy();out['p']=0.;out['missing']=0.;return out
    prior=types.SimpleNamespace(PRIOR_COLUMNS=['p','missing'],prior_lookup=lambda *a:lookup,project_prior=project)
    class Fake:
        def __init__(self,*args):pass
        def modules(self):return recipe,prior
        def metadata(self,ref):
            if ref=='raw_prior':return dict(status='RAW_EIGHT_TARGETS_DIAGNOSED',rows=[{}]*8)
            if ref=='inventory':return dict(cells=[dict(fold=f,as_run_input_version='ORIGINAL_M0_FORMAL_V1_EXACT_S1_PREPARATION_PARITY')for f in ['2023H2',*m.PRIMARY,'2026H1','2026H2_PARTIAL_TO_2026_10_15']])
            if ref=='delta':return dict(pooled_primary=list(m.PRIMARY))
            if ref=='prep':return dict(status='COMPLETED',panel_rows=20,cache_files={str(path):panel_ref['sha256']})
            raise AssertionError(ref)
        def authenticate(self,ref):return m.old.support.stat_identity(path)
        def frame(self,ref,**kw):
            if ref==panel_ref:
                frozen=json.loads((tmp_path/'2023H2_SELECTIONS.json').read_text())
                assert frozen['outcomes_not_read_for_this_fold']and frozen['records'][0]['selections']
                assert kw['columns']==['date','code','close']and all(len(group)==2 for group in kw['filters'])
                raise RuntimeError('SENTINEL_AFTER_FROZEN_SELECTION')
            return calendar.copy()if ref['path']=='calendar'else pd.DataFrame()
    monkeypatch.setattr(m.old,'BoundFiles',Fake)
    monkeypatch.setattr(m,'reuse_diagnostic_bindings',lambda *a:None)
    monkeypatch.setattr(m,'load_saved_scores',lambda *args:(frame.copy(),dict(full_pool_test_keys=recipe.key_frame_identity(frame),model_train_time=dict(value='2023-06-30',kind='EXACT_MAX',method='SYNTHETIC_ACTUAL',evidence_sha256='proof'))))
    if resume:
        records=m.select_ac(frame,calendar,lookup,pd.DataFrame(),prior,'2023H2')
        proof=dict(fold='2023H2')
        saved=tmp_path/'2023H2_SELECTIONS.json'
        saved.write_text(json.dumps(dict(outcomes_not_read_for_this_fold=True,records=records)))
        ref=dict(path=str(saved),sha256=m.sha(saved))
        monkeypatch.setattr(m,'recovered_selections',lambda *args:(records,proof,ref))
        def no_recompute(*args):raise AssertionError('completed selection or score recomputed')
        monkeypatch.setattr(m,'load_saved_scores',no_recompute)
        monkeypatch.setattr(m,'select_ac',no_recompute)
    class Registry:
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return None,[dict(execution_id=m.RAW_EXECUTION,event_type='RUN_STARTED'),dict(execution_id=m.RAW_EXECUTION,event_type='TERMINAL',decision='RAW_EIGHT_TARGETS_DIAGNOSED')]
    with pytest.raises(RuntimeError,match='SENTINEL'):m.compute_expansion(c,type('Life',(),dict(adapter=Registry()))())
    assert c['_access']['label_source_attempted']and not c['_access'].get('label_folds_completed')

@pytest.mark.parametrize('drift',[False,True])
def test_diagnostic_fast_bind_reuses_only_matching_sha_stat(tmp_path,drift):
    p=tmp_path/'bound';p.write_bytes(b'synthetic')
    stat=m.old.support.stat_identity(p);r=dict(path=str(p),sha256=m.sha(p));proof=dict(status='EIGHT_KEYS_DIAGNOSED',input_bindings={str(p):dict(sha256=r['sha256'],stat=stat)})
    if drift:proof['input_bindings'][str(p)]['sha256']='WRONG'
    class Fake:
        def __init__(self):self.bound={}
        def metadata(self,ref):return proof if ref=='receipt'else dict(boundary_status=dict(diagnostic_integrity='PASS_WITH_LIMITATIONS'))
    a=Fake();c=dict(inputs=dict(panel=r,calendar=r),diagnostic_receipt_ref='receipt',diagnostic_actual_review_ref='review')
    if drift:
        with pytest.raises(m.Blocked,match='FAST_BIND_IDENTITY'):m.reuse_diagnostic_bindings(a,c)
        assert not a.bound
    else:
        m.reuse_diagnostic_bindings(a,c);assert a.bound[str(p)]['sha256']==r['sha256']
        assert len(c['_access']['fast_bound_inputs'])==2


def test_raw_lifecycle_counts_only_requested_target_keys():
    c=dict(execution_id=m.RAW_EXECUTION,caps=m.CAPS,diagnostic_dates={},contract_path='c',contract_sha256='h',config_path='i',config_sha256='j',code_composite_sha256='k',selection_ref={},_access=dict(requested_keys=8,flags_completed=True,panel_returned=True,returned_target_dates=[]))
    e=m.RawLifecycle(None,c).event('TERMINAL','RAW_EIGHT_TARGETS_DIAGNOSED','receipt',{})
    assert 'requested_unique_signal_keys'not in e['deterministic_metrics']and e['deterministic_metrics']['requested_unique_target_keys']==8
    assert e['target_date_range']['count']==0 and e['truth_access']=='RAW_TARGET_FLAGS_COMPLETED_NO_RETURN_LABELS'


def test_raw_precounter_rejects_17_before_close_read(tmp_path,monkeypatch):
    c=raw_fixture(tmp_path,'overflow')
    class Active:
        def require(self):return True
    def forbidden(*a,**kw):raise AssertionError('raw close should not materialize')
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    with pytest.raises(m.Blocked,match='RAW_TARGET_PREDECODE_ROW_BOUND'):m.raw_target_flags(c,Active())
    assert c['_access']['matched_panel_rows']==17 and not c['_access'].get('panel_read_attempted')


def test_summary_short_frozen_block_is_cannot_confirm_not_negative_screen():
    rows={f:paired_rows(f,40,k*40)for k,f in enumerate(m.PRIMARY)}
    rows['2024H1']=rows['2024H1'][:19]
    rows['2023H2']=paired_rows('2023H2',40)
    rows['2026H2_PARTIAL_TO_2026_10_15']=paired_rows('2026H2_PARTIAL_TO_2026_10_15',40)
    report=m.stability_summary(rows)
    assert report['primary']['complete_coverage']and report['positive_halves']==4
    assert report['status']=='CANNOT_CONFIRM_STABILITY_FROZEN_BLOCK_INTERVAL'
    assert report['interval']['status']=='CANNOT_CONFIRM_RUN_SHORTER_FROZEN_20'
    assert not report['screen_passed']and report['interval']['lower']is None


def selected_fixture(tmp_path,n=4000,extra=None):
    pairs=[('2024-01-02',f'{i:06d}.SZ')for i in range(n)]
    rows=[dict(date=pd.Timestamp(d),code=code,close=float(i+1))for i,(d,code)in enumerate(pairs)]
    # Cross-product decoys ensure exact pairs, not independent date/code membership.
    rows.extend([dict(date=pd.Timestamp('2024-01-03'),code='000000.SZ',close=-1.),
                 dict(date=pd.Timestamp('2024-01-02'),code='OTHER',close=-2.)])
    if extra:rows.extend(dict(rows[0])for _ in range(extra))
    path=tmp_path/'selected.parquet';pd.DataFrame(rows).to_parquet(path,index=False)
    ref=dict(path=str(path),sha256=m.sha(path))
    c=dict(caps=m.EXP_CAPS,inputs=dict(panel=ref),panel_row_binding=dict(file_rows=len(rows)))
    class Active:
        def require_active(self):return True
    return m.old.BoundFiles(c,Active()),c,pairs


def test_bounded_exact_pairs_4000_real_parquet_subprocess(tmp_path):
    import subprocess
    adapter,c,pairs=selected_fixture(tmp_path)
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    code="""import json,sys
import environment_tilt_month_bonus_stability as m
c=json.load(open(sys.argv[1]))
class Active:
 def require_active(self):return True
pairs=[('2024-01-02',f'{i:06d}.SZ')for i in range(4000)]
p=m.read_selected_panel(m.old.BoundFiles(c,Active()),c,pairs)
assert len(p)==4000 and set(p.code)=={x[1]for x in pairs}
assert p.close.sum()==4000*4001/2
assert c['_access']['selected_predicate_batches']==63
print('SAFE_4000_EXACT_PAIRS')
"""
    import os
    result=subprocess.run([sys.executable,'-B','-c',code,str(config)],capture_output=True,text=True,timeout=45,
        env={**os.environ,'PYTHONPATH':str(Path(m.__file__).parent),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    assert result.returncode==0,(result.returncode,result.stdout,result.stderr)
    assert result.stdout.strip()=='SAFE_4000_EXACT_PAIRS'


def test_bounded_exact_pairs_parity_with_original_predicate(tmp_path):
    import pyarrow.dataset as ds
    adapter,c,pairs=selected_fixture(tmp_path,n=130)
    # Same code at different requested dates; empty and repeated requested keys.
    pairs=pairs+[('2024-01-03','000001.SZ'),pairs[0]]
    original=None
    for day,code in pairs:
        term=(ds.field('date')==pd.Timestamp(day).to_pydatetime())&(ds.field('code')==code)
        original=term if original is None else original|term
    expected=ds.dataset(c['inputs']['panel']['path'],format='parquet').to_table(columns=['date','code','close'],filter=original).to_pandas()
    actual=m.read_selected_panel(adapter,c,pairs)
    pd.testing.assert_frame_equal(actual.sort_values(['date','code']).reset_index(drop=True),expected.sort_values(['date','code']).reset_index(drop=True))


def test_bounded_exact_pairs_global_cap_before_any_close_decode(tmp_path,monkeypatch):
    adapter,c,pairs=selected_fixture(tmp_path,n=5050)
    reads=[]
    def forbidden(*a,**kw):reads.append(kw);raise AssertionError('decoded before total cap')
    monkeypatch.setattr(adapter,'frame',forbidden)
    with pytest.raises(m.Blocked,match='SELECTED_PANEL_PREDECODE_ROW_BOUND'):
        m.read_selected_panel(adapter,c,pairs)
    assert c['_access']['selected_matched_panel_rows']==5050
    assert c['_access']['selected_predicate_batches']==79
    assert not reads and not c['_access'].get('label_source_attempted')


def test_bounded_exact_pairs_rejects_duplicate_keys_across_batches(tmp_path):
    adapter,c,pairs=selected_fixture(tmp_path,n=130,extra=1)
    with pytest.raises(m.Blocked,match='SELECTED_PANEL_COUNT_OR_KEYS_CONFLICT'):
        m.read_selected_panel(adapter,c,pairs)


@pytest.mark.parametrize('drift',['none','sha','policy','input','calendar','checkpoint','score'])
def test_recovered_selection_identity_and_no_other_fold_reuse(tmp_path,drift):
    calendar=pd.DataFrame(dict(date=pd.bdate_range('2023-01-02','2024-01-05')))
    calendar['calendar_index']=range(len(calendar));dates=m.old.calendar_dates(calendar)
    records=[dict(fold='2023H2',date=d.isoformat(),calendar_index=dates.index(d),h2_date=dates[dates.index(d)+2].isoformat(),
        selections={a:[f'{i:06d}.SZ'for i in range(10)]for a in m.EXP_POLICY['arms']})for d in dates if '2023-07-01'<=d.isoformat()<='2023-12-31']
    proof=dict(fold='2023H2',input_version='synthetic',full_pool_test_keys=dict(row_count=1240),
        model_train_time=dict(value='2023-06-30',kind='EXACT_MAX',method='SYNTHETIC_ACTUAL',evidence_sha256='proof'))
    saved=dict(policy=m.EXP_POLICY,provenance=proof,records=records,outcomes_not_read_for_this_fold=True)
    def write(name,data):
        p=tmp_path/name;p.write_text(json.dumps(data));return dict(path=str(p),sha256=m.sha(p))
    ref=write('saved.json',saved)
    c={k:{}for k in ('inputs','input_versions','inventory_ref','recipe_ref','sources','panel_row_binding')};c['policy']=m.EXP_POLICY;c['caps']=m.EXP_CAPS
    prev=write('previous.json',dict(c,execution_id=m.EXP_EXECUTION))
    interrupted=write('interrupted.json',dict(status='HARD_INTERRUPTED',recovery_checkpoint=ref))
    manifest=dict(status='FROZEN_SELECTION_ONLY_RECOVERY',selections={'2023H2':ref},previous_config_ref=prev,interruption_handoff_ref=interrupted)
    if drift=='sha':ref['sha256']='0'*64
    if drift=='policy':c['policy']=dict(m.EXP_POLICY,bonus=6)
    if drift=='input':c['inputs']={'unexpected':{}}
    if drift=='calendar':records[0]['h2_date']='2023-07-06';ref=write('saved2.json',saved);manifest['selections']['2023H2']=ref;interrupted=write('interrupted2.json',dict(status='HARD_INTERRUPTED',recovery_checkpoint=ref));manifest['interruption_handoff_ref']=interrupted
    if drift=='checkpoint':manifest['selections']['2023H2']=write('different.json',saved)
    c['resume_ref']=write('manifest.json',manifest)
    class Active:
        def require_active(self):return True
    adapter=m.old.BoundFiles(c,Active());spec=dict(as_run_input_version='synthetic',accepted_identity=dict(test=dict(row_count=999 if drift=='score'else 1240)))
    if drift!='none':
        with pytest.raises((m.Blocked,m.old.Blocked)):
            m.recovered_selections(adapter,c,'2023H2',calendar,spec)
    else:
        got=m.recovered_selections(adapter,c,'2023H2',calendar,spec)
        assert got==(records,proof,ref)
        assert c['_access']['reused_selection_refs']==[ref]
        assert m.recovered_selections(adapter,c,'2024H1',calendar,spec)is None
