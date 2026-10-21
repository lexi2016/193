"""Synthetic fixed-rule scope; no market payload, network, models or real registry."""
from pathlib import Path
import contextlib
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/a_share_model_selection_v1'))
import environment_tilt_simple_rule_canary as rule


def test_fixed_plus5_top50_original_rank_ties_missing_and_no_refill():
    group=pd.DataFrame(dict(code=[f'{n:06d}.SZ'for n in range(1,61)],prediction=[100-n for n in range(60)]))
    prior={c:(0.,0.)for c in group.code};prior['000012.SZ']=(.1,0.);prior['000011.SZ']=(100.,1.);prior['000051.SZ']=(100.,0.)
    base,candidate,ranked=rule.fixed_rank(group,prior)
    assert base==group.code[:10].tolist()and '000012.SZ'in candidate
    assert '000051.SZ'not in candidate and len(ranked)==50
    assert next(r for r in ranked if r['code']=='000012.SZ')['bonus']==5
    assert next(r for r in ranked if r['code']=='000011.SZ')['bonus']==0
    prior['000012.SZ']=(1e100,0.);assert rule.fixed_rank(group,prior)[1]==candidate
    assert rule.fixed_rank(group.iloc[:9],prior)[1]==group.code[:9].tolist()


def test_existing_index_below_ma20_and_no_lookahead_or_fill():
    cal=pd.DataFrame(dict(date=pd.bdate_range('2025-12-01',periods=30),calendar_index=range(30)))
    index=pd.DataFrame(dict(date=cal.date,code='SH.000300',close=[100.]*20+[99.]*10))
    states=rule.weak_states(index,cal)
    assert states[cal.date.iloc[18].strftime('%Y-%m-%d')]is None
    assert states[cal.date.iloc[19].strftime('%Y-%m-%d')]is False
    key=cal.date.iloc[20].strftime('%Y-%m-%d');assert states[key]is True
    index.loc[29,'close']=999.;assert rule.weak_states(index,cal)[key]is True
    assert rule.weak_states(index.drop(index.index[20]),cal)[key]is None


def test_four_arm_freeze_cash_no_score_unknown_not_zero():
    cal=pd.DataFrame(dict(date=pd.to_datetime(['2026-01-02','2026-01-05','2026-01-06','2026-01-07','2026-01-08']),calendar_index=range(5)))
    codes=[f'{i:06d}.SZ'for i in range(50)]
    predictions=pd.DataFrame([dict(date=d,code=c,prediction=50-i)for d in cal.date[:2]for i,c in enumerate(codes)])
    projection=predictions[['date','code']].copy();projection['industry_month_log_relative_lift']=[.1 if c==codes[11]else 0. for c in projection.code];projection['industry_month_missing']=0.
    states={d.strftime('%Y-%m-%d'):True for d in cal.date};states['2026-01-05']=None
    records=rule.select_all(predictions,cal,states,projection)
    assert records[0]['selections']['B_WEAK_GATE']==[]and records[0]['selections']['D_COMBINED']==[]
    assert len(records[0]['selections']['A_ORIGINAL'])==10
    assert records[1]['reason']=='UNKNOWN_INDEX_STATE_NOT_CASH'
    assert records[2]['reason']=='NO_EXISTING_SCORE_DATE_NOT_CASH'
    assert all(d!='2026-01-05'for d,c in rule.selected_keys(records))


def screen_fixture():
    codes_a=[f'A{i}'for i in range(10)];codes_c=[f'C{i}'for i in range(10)]
    dates=list(pd.bdate_range('2026-01-05',periods=20))+list(pd.bdate_range('2026-04-01',periods=20))
    records=[];outcomes={}
    for n,d in enumerate(dates):
        day=d.strftime('%Y-%m-%d');weak=n in(0,20)
        records.append(dict(date=day,h2_date=day,weak=weak,reason=None,month_rank_changed=True,selections=dict(zip(rule.ARMS,(codes_a,[]if weak else codes_a,codes_c,[]if weak else codes_c)))))
        outcomes.update({(day,c):(-.01 if weak else 0.)for c in codes_a});outcomes.update({(day,c):.002 for c in codes_c})
    return records,outcomes


def test_frozen_screen_full_calendar_two_quarters_cash_and_fractional_tail():
    records,outcomes=screen_fixture();report=rule.evaluate(records,outcomes)
    assert report['screen_passed']and report['full_calendar_complete']and report['nominal_dates']==40
    assert report['arms']['D_COMBINED']['cash_dates']==2 and report['arms']['D_COMBINED']['stock_slots']==380
    assert report['primary_full_calendar_lift']>=.0008 and all(v>0 for v in report['quarter_lifts'].values())
    assert report['arms']['A_ORIGINAL']['cvar5']==-.01 and report['arms']['D_COMBINED']['cvar5']==0.
    assert report['nav']is None and report['portfolio_max_drawdown']is None


@pytest.mark.parametrize('kind',['missing_label','no_score','unknown_index'])
def test_unknown_day_never_becomes_cash_or_deleted_day_pass(kind):
    records,outcomes=screen_fixture()
    if kind=='missing_label':del outcomes[(records[1]['date'],'C1')]
    else:records[1].update(reason=kind,selections=None)
    result=rule.evaluate(records,outcomes)
    assert result['valid_common_dates']==39 and result['nominal_dates']==40
    assert result['primary_full_calendar_lift']is None and not result['screen_passed']
    assert all(s['full_calendar_mean']is None for s in result['arms'].values())
    assert result['dates'][1]['returns']is None


def test_cannot_select_best_arm_or_retune_after_screen_failure():
    records,outcomes=screen_fixture()
    # C remains positive inQ2, but D has negativeQ2: no winner switch to C.
    for r in records:
        if r['date'].startswith('2026-04'):
            r['weak']=False;r['selections']['D_COMBINED']=r['selections']['A_ORIGINAL']
    result=rule.evaluate(records,outcomes)
    assert result['quarter_lifts']['2026Q2']==0 and not result['screen_passed']
    assert result['screen_reason']=='FROZEN_SCREEN_NOT_MET_NO_RETUNING'


def test_registered_failure_terminal_no_real_registry(tmp_path,monkeypatch):
    class Adapter:
        def __init__(self):self.events=[]
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
        def append_parity_event(self,event,execution_id):
            from environment_tilt_mac_registry import digest
            event['event_sha256']=digest(event);self.events.append(event)
    adapter=Adapter();review=tmp_path/'review.json';review.write_text('{}')
    c=dict(run_dir=str(tmp_path/'REGISTERED_01'),execution_id='SIMPLE_RULE_SYNTHETIC',caps=rule.CAPS,diagnostic_dates={'start':'2026-01-01','end':'2026-06-30'},config_path='syntheticconfig',config_sha256='cfg',contract_path='contract',contract_sha256='contract',code_composite_sha256='code')
    proof={k:c[k]for k in('execution_id','config_sha256','contract_sha256','code_composite_sha256')};proof.update(status='APPROVED_FOR_ONE_FIXED_RULE_CANARY',review_receipt=dict(path=str(review),sha256=rule.sha(review)))
    monkeypatch.setattr(rule.resources,'available_memory',lambda:100*2**30)
    def fail(c,life,sink):
        assert [e['event_type']for e in adapter.events]==['RUN_STARTED'];life.require();raise rule.Blocked('SYNTHETIC_DRIFT')
    result=rule.run(c,proof,adapter,fail)
    assert result['status']=='STOPPED'and result['error']=='SYNTHETIC_DRIFT'
    assert [e['event_type']for e in adapter.events]==['RUN_STARTED','TERMINAL']
    with pytest.raises(FileExistsError):rule.run(c,proof,adapter,fail)


@pytest.mark.parametrize('identity_matches',[True,False])
def test_bound_compute_saves_all_selections_before_any_label_panel_read(tmp_path,monkeypatch,identity_matches):
    import joblib
    calendar=pd.DataFrame(dict(date=pd.bdate_range('2025-11-01','2026-07-10')));calendar['calendar_index']=range(len(calendar))
    predictions=pd.DataFrame([dict(date=d,code=f'{i:06d}.SZ',prediction=50-i)for d in pd.to_datetime(['2026-01-05','2026-04-01'])for i in range(50)])
    sidecar=dict(artifact_sha256='score',identity=dict(fold='2026H1',stage='M0_FORMAL',data_identity=dict(test_keys='keys')))
    class Recipe:
        def key_frame_identity(self,p):return 'keys'
        def build_close_return_labels(self,p,horizons,**kwargs):
            assert saved
            k=kwargs['signal_keys'].copy();k['label_return']=.001;return {2:k}
    class Prior:
        def prior_lookup(self,*a):return None
        def project_prior(self,keys,*a):
            p=keys.copy();p['industry_month_log_relative_lift']=0.;p['industry_month_missing']=0.;return p
    saved=[]
    class Bound:
        bound={}
        def __init__(self,*a):pass
        def modules(self):return Recipe(),Prior()
        def authenticate(self,ref):return 'stat'
        def metadata(self,ref):return sidecar
        def frame(self,ref,**kwargs):
            ref=ref['path']
            if ref=='calendar':return calendar.copy()
            if ref=='index':return pd.DataFrame(dict(date=calendar.date,code='SH.000300',close=100.))
            if ref=='panel':
                assert saved==['FOUR_ARM_SELECTIONS.json'];return pd.DataFrame(columns=['date','code','close'])
            return pd.DataFrame()
    monkeypatch.setattr(rule,'score_acceptance',lambda *a:{'synthetic':'accepted'});monkeypatch.setattr(rule,'read_selected_panel',lambda adapter,c,start,end,codes:adapter.frame(c['inputs']['panel']));monkeypatch.setattr(rule.old,'BoundFiles',Bound);monkeypatch.setattr(rule.old,'restore_model_time',lambda *a,**k:{'value':'2025-12-31'})
    monkeypatch.setattr(rule.old,'validate_time_proof',lambda *a:None);monkeypatch.setattr(rule.old.support,'stat_identity',lambda *a:'stat')
    monkeypatch.setattr(joblib,'load',lambda *a:dict(predictions=predictions))
    def sink(name,data):
        saved.append(name);p=tmp_path/name;p.write_text(rule.canonical(data));c['selection_identity_ref']=dict(path='previous02selection',sha256=rule.sha(p)if identity_matches else 'mismatch');return dict(path=str(p),sha256=rule.sha(p))
    c=dict(sources=dict(recipe={'sha256':'recipe'}),score=dict(metadata=dict(path='meta',sha256='sidecar'),artifact=dict(path='noactual',sha256='score')),inputs={k:dict(path=v,sha256='fixture')for k,v in dict(calendar='calendar',index='index',panel='panel',roster='roster',industry_year='annual',month_year='monthly').items()},caps=rule.CAPS)
    if not identity_matches:
        with pytest.raises(rule.Blocked,match='SELECTION_OR_CASH_FLAGS_DRIFT_FROM_EXECUTION02'):rule.compute(c,None,sink)
        return
    result=rule.compute(c,None,sink)
    assert c['_access']['selection_identity_vs_execution02']is True
    assert result['checkpoint_unpacks']==1 and not result['full_calendar_complete']
    assert result['fit_count']==0 and result['network_calls']==0


def acceptance_fixture():
    spec=dict(artifact=dict(path='checkpoint',sha256='cp'),metadata=dict(path='sidecar',sha256='md'),input_version='UNCHANGED_EXISTING_INPUT',acceptance_manifest='manifest',original_accepted_review='review',accepted_output_manifest=dict(path='output',sha256='out'),checkpoint_replay='replay',m0_stage_manifest='stage')
    cell=dict(fold='2026H1',artifact=spec['artifact'],metadata=spec['metadata'],as_run_input_version=spec['input_version'],as_run_contract_sha256='contract',as_run_config_sha256='fitconfig')
    sidecar=dict(identity=dict(data_identity={'test_keys':{'row_count':100}},config_sha256='fitconfig',contract_sha256='contract'))
    docs=dict(manifest=dict(accepted_cell_candidates=[cell],accepted_cell_common_rules=dict(old_outer_acceptance_ref='review')),
        review=dict(verdict='METRICS_CONSISTENT',research_decision='M4_GATE_FAILED',artifacts=dict(contract_sha256='contract',output_manifest_sha256='out')),
        output=dict(contract_sha256='contract'),stage=dict(contract_sha256='contract',cell_results=[dict(cell='M0_FORMAL_M0_H2_2026H1_attempt_01',checkpoint_path='checkpoint',status='COMPLETED',fit_called=True)]),
        replay=dict(status='VERIFIED',schema_version='A_SHARE_SAVED_CHECKPOINT_REPLAY_V1',checkpoint_path='checkpoint',checkpoint_sha256='cp',checkpoint_metadata_path='sidecar',checkpoint_metadata_sha256='md',data_identity=sidecar['identity']['data_identity'],config_sha256='fitconfig',prediction_identity={'hash':'same'},replay_prediction_identity={'hash':'same'},top10_parity={'status':'EXACT'},comparison=dict(key_order_equal=True,prediction_equal=True)))
    class MetadataOnly:
        def metadata(self,ref):return docs[ref if isinstance(ref,str)else ref['path']]
        def authenticate(self,ref):raise AssertionError('score payload must not open while accepting metadata')
    return spec,sidecar,docs,MetadataOnly()


def test_existing_acceptance_chain_bound_without_replaying_or_claiming_promotion():
    spec,sidecar,docs,adapter=acceptance_fixture()
    proof=rule.score_acceptance(adapter,{'score':spec},sidecar)
    assert proof['artifact']==spec['artifact']and proof['existing_research_decision']=='M4_GATE_FAILED'
    assert not proof['new_prediction_or_replay_called']and not proof['business_promotion_claimed']


@pytest.mark.parametrize('drift',['review','version','artifact','replay','stage'])
def test_existing_acceptance_chain_any_mismatch_stops_before_payload(drift):
    spec,sidecar,docs,adapter=acceptance_fixture()
    if drift=='review':docs['review']['verdict']='WRONG'
    if drift=='version':spec['input_version']='OTHER_VERSION'
    if drift=='artifact':docs['manifest']['accepted_cell_candidates'][0]['artifact']={'path':'other','sha256':'other'}
    if drift=='replay':docs['replay']['checkpoint_sha256']='other'
    if drift=='stage':docs['stage']['cell_results'][0]['status']='FAILED'
    with pytest.raises(rule.Blocked):rule.score_acceptance(adapter,{'score':spec},sidecar)


def test_lifecycle_start_and_terminal_distinguish_score_and_selected_truth_slots():
    c=dict(execution_id='synthetic',contract_path='contract',contract_sha256='c',config_path='cfg',config_sha256='cf',code_composite_sha256='code',diagnostic_dates={},caps=rule.CAPS,score={'artifact':'existing'})
    life=rule.RuleLifecycle(None,c)
    start=life.event('RUN_STARTED','RUNNING','receipt',{})
    assert start['truth_access']=='NOT_READ'and start['deterministic_metrics']['selection_slots']is None
    c['_access']=dict(existing_checkpoint_ref={'path':'existing','sha256':'cp'},selected_h2_target_range=dict(count=4,start='2026-01-07',end='2026-04-06',scope='SELECTED_H2_OUTCOME_TARGETS_REQUESTED_NOT_ALLPOOL_TRUTH'),score_checkpoint_entered=True,score_checkpoint_unpacked=True,selected_label_source_entered=True,stock_slots_total=100,stock_slots_by_arm={'A':40,'B':10,'C':40,'D':10},nominal_dates=4,valid_dates=3,requested_label_keys=60,finite_label_keys=59,positive_label_keys=20)
    terminal=life.event('TERMINAL','COMPLETE','receipt',{})
    assert terminal['truth_access']=='SELECTED_H2_OUTCOME_SOURCE_ACCESSED'
    assert terminal['deterministic_metrics']['selection_slots']==100
    assert terminal['deterministic_metrics']['requested_unique_selected_label_keys']==60
    assert terminal['deterministic_metrics']['finite_unique_selected_h2_labels']==59
    assert terminal['deterministic_metrics']['truth_positive_denominator']is None
    assert terminal['deterministic_metrics']['existing_checkpoint_unpacks']==1
    assert 'NO_LABELS'not in terminal['sealed_status']


def test_terminal_core_target_checkpoint_and_null_reason_match_actual_access():
    c=dict(execution_id='synthetic',contract_path='contract',contract_sha256='c',config_path='cfg',config_sha256='cf',code_composite_sha256='code',diagnostic_dates={},caps=rule.CAPS,score={'artifact':'existing'})
    life=rule.RuleLifecycle(None,c);start=life.event('RUN_STARTED','RUNNING','receipt',{})
    assert start['checkpoint_ref']is None and start['target_date_range']['count']==0
    assert 'selection_slots'in start['input_identity']['null_reasons']
    c['_access']=dict(existing_checkpoint_ref=dict(path='existing',sha256='cp'),selected_label_source_entered=True,stock_slots_total=40,selected_h2_target_range=dict(count=2,start='2026-01-07',end='2026-01-08',scope='SELECTED_H2_OUTCOME_TARGETS_REQUESTED_NOT_ALLPOOL_TRUTH'))
    end=life.event('TERMINAL','COMPLETED','receipt',{})
    assert end['checkpoint_ref']==c['_access']['existing_checkpoint_ref']
    assert end['target_date_range']==c['_access']['selected_h2_target_range']
    assert 'NO_OUTCOME_LABELS'not in str(end['target_date_range'])
    assert 'checkpoint_ref'not in end['input_identity']['null_reasons']
    assert 'selection_slots'not in end['input_identity']['null_reasons']
    assert end['deterministic_metrics']['truth_positive_denominator']is None


@pytest.mark.parametrize('failed',[False,True])
def test_registered_completed_and_failed_label_access_events_consistent(tmp_path,monkeypatch,failed):
    class Adapter:
        def __init__(self):self.events=[]
        def _locked(self):return contextlib.nullcontext()
        def _read_validated(self):return b'',self.events
        def append_parity_event(self,event,execution_id):
            from environment_tilt_mac_registry import digest
            event['event_sha256']=digest(event);self.events.append(event)
    adapter=Adapter();review=tmp_path/'review.json';review.write_text('{}')
    c=dict(run_dir=str(tmp_path/'REGISTERED_01'),execution_id='SYNTHETIC_LIFECYCLE',caps=rule.CAPS,diagnostic_dates={},config_path='config',config_sha256='cfg',contract_path='contract',contract_sha256='contract',code_composite_sha256='code',score={'artifact':'existing'})
    proof={k:c[k]for k in('execution_id','config_sha256','contract_sha256','code_composite_sha256')};proof.update(status='APPROVED_FOR_ONE_FIXED_RULE_CANARY',review_receipt=dict(path=str(review),sha256=rule.sha(review)))
    monkeypatch.setattr(rule.resources,'available_memory',lambda:100*2**30)
    def work(c,life,sink):
        life.require();c['_access']=dict(score_checkpoint_entered=True,score_checkpoint_unpacked=True,existing_checkpoint_ref=dict(path='existing',sha256='cp'),selected_label_source_entered=True,stock_slots_total=20,stock_slots_by_arm={'A':10,'B':0,'C':10,'D':0},nominal_dates=1,valid_dates=None if failed else 1,selected_h2_target_range=dict(count=1,start='2026-01-07',end='2026-01-07',scope='SELECTED_H2_OUTCOME_TARGETS_REQUESTED_NOT_ALLPOOL_TRUTH'))
        if failed:raise rule.Blocked('SYNTHETIC_LABEL_READ_FAILURE')
        return dict(status='SMALL_SAMPLE_NOT_YET_WORTH_EXPANDING',nominal_dates=1,valid_common_dates=1)
    result=rule.run(c,proof,adapter,work);start,end=adapter.events
    assert start['truth_access']=='NOT_READ'and start['checkpoint_ref']is None
    assert end['event_type']=='TERMINAL'and end['truth_access']=='SELECTED_H2_OUTCOME_SOURCE_ACCESSED'
    assert end['checkpoint_ref']=={'path':'existing','sha256':'cp'}and end['target_date_range']['start']=='2026-01-07'
    assert end['deterministic_metrics']['selection_slots']==20
    assert 'selection_slots'not in end['input_identity']['null_reasons']and 'checkpoint_ref'not in end['input_identity']['null_reasons']
    assert 'NO_MODEL'not in str(end['input_identity'])and 'NO_OUTCOME_LABELS'not in str(end['target_date_range'])
    assert result['status']==('STOPPED'if failed else'SMALL_SAMPLE_NOT_YET_WORTH_EXPANDING')


def test_runtime_real_time_guard_requires_the_authenticated_sidecar_evidence(tmp_path,monkeypatch):
    import joblib
    calendar=pd.DataFrame(dict(date=pd.to_datetime(['2025-12-29','2025-12-30','2025-12-31','2026-01-05','2026-01-06','2026-01-07']),calendar_index=range(6)))
    keys=dict(date_max='2025-12-29',ordered_keys_sha256='ordered',sorted_keyset_sha256='sorted',row_count=1,unique_key_count=1,schema_version='A_SHARE_DATE_CODE_SORTED_KEYS_V1')
    data=dict(schema_version='A_SHARE_CANARY_CELL_DATA_IDENTITY_V1',feature_columns=list(rule.old.support.FEATURES),training_keys=keys,training_label_vector={'sha256':'past_train_labels'},test_keys={'synthetic':'keys'})
    sidecar=dict(status='COMPLETED',artifact_sha256='score',identity=dict(arm='M0',horizon=2,fold='2026H1',stage='M0_FORMAL',code_sha256={'model_selection':'recipe'},data_identity=data))
    missing=rule.old.restore_model_time(sidecar,tuple(t.date()for t in calendar.date),model_source_sha256='recipe',calendar_sha256='calendar')
    with pytest.raises(rule.old.Blocked,match='MODEL_TRAIN_TIME_PROOF_REQUIRED_NO_PURGE_PROXY'):
        rule.old.validate_time_proof(missing,pd.Timestamp('2026-01-05').date())
    class ReachedAfterRealTimeValidation(Exception):pass
    class Recipe:
        def key_frame_identity(self,frame):return {'synthetic':'keys'}
    class Bound:
        def __init__(self,*a):pass
        def modules(self):return Recipe(),None
        def metadata(self,ref):assert ref['sha256']=='authenticated_sidecar';return sidecar
        def authenticate(self,ref):return 'same_stat'
        def frame(self,ref,**kwargs):
            if ref['path']=='calendar':return calendar.copy()
            assert ref['path']=='index';raise ReachedAfterRealTimeValidation()
    monkeypatch.setattr(rule.old,'BoundFiles',Bound);monkeypatch.setattr(rule,'score_acceptance',lambda *a:{'synthetic':'accepted'})
    monkeypatch.setattr(rule.old.support,'stat_identity',lambda *a:'same_stat')
    monkeypatch.setattr(joblib,'load',lambda *a:dict(predictions=pd.DataFrame(dict(date=pd.to_datetime(['2026-01-05']),code=['fixture.SZ'],prediction=[.01]))))
    c=dict(sources={'recipe':{'sha256':'recipe'}},inputs={'calendar':{'path':'calendar','sha256':'calendar'},'index':{'path':'index'}},score=dict(metadata={'path':'sidecar','sha256':'authenticated_sidecar'},artifact={'path':'score','sha256':'score'}))
    # This calls the real restore+validate pair, not a mocked time validator.
    with pytest.raises(ReachedAfterRealTimeValidation):rule.compute(c,None,lambda *a:None)


def test_filtered_panel_file_metadata_bound_separate_from_decoded_subset(tmp_path,monkeypatch):
    import json
    panel=pd.DataFrame(dict(date=pd.date_range('2026-01-01',periods=10),code='fixture.SZ',close=10.))
    p=tmp_path/'panel.parquet';panel.to_parquet(p,index=False);ref=dict(path=str(p),sha256=rule.sha(p))
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(status='COMPLETED',panel_rows=10,cache_files={str(p):ref['sha256']})))
    c=dict(inputs={'panel':ref},panel_row_binding=dict(file_rows=10,preparation_ref=dict(path=str(proof),sha256=rule.sha(proof))),caps=dict(rule.CAPS,max_panel_rows=3))
    class Session:
        def require_active(self):pass
    adapter=rule.old.BoundFiles(c,Session());seen=[];original=pd.read_parquet
    def read(*a,**kwargs):seen.append(kwargs);return original(*a,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',read)
    # Reproduces02: original helper checks unfilteredmetadata10 againstsubset3.
    with pytest.raises(rule.old.Blocked,match='PARQUET_ROW_RESOURCE_BOUND'):
        adapter.frame(ref,columns=['date','code','close'],filters=[('date','<=',pd.Timestamp('2026-01-02'))],max_rows=3)
    assert not seen
    result=rule.read_selected_panel(adapter,c,pd.Timestamp('2026-01-01'),pd.Timestamp('2026-01-02'),['fixture.SZ'])
    assert len(result)==2 and seen[0]['columns']==['date','code','close']
    assert seen[0]['filters']==[('date','>=',pd.Timestamp('2026-01-01')),('date','<=',pd.Timestamp('2026-01-02')),('code','in',['fixture.SZ'])]
    with pytest.raises(rule.Blocked,match='SELECTED_PANEL_SUBSET_ROW_BOUND'):
        rule.read_selected_panel(adapter,c,pd.Timestamp('2026-01-01'),pd.Timestamp('2026-01-05'),['fixture.SZ'])
