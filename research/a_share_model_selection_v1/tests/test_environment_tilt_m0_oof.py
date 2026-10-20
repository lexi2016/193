"""Only synthetic frames/fake estimator/guard. Original recipe SOURCE is reused."""
from __future__ import annotations

import copy
import importlib.util
import json
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import environment_tilt_m0_oof as m

ORIGINAL = Path('/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/s1-industry-month-20261020/research/a_share_model_selection_v1/frozen_m0/research/a_share_model_selection_v1/model_selection.py')


@pytest.fixture(scope='module')
def recipe():
    # Import defines functions/classes only: no model instance, weights, data or run.
    return m._load(ORIGINAL, 'm13_synthetic_original_recipe')


@pytest.fixture
def frames():
    dates = pd.bdate_range('2021-06-30', '2024-01-03')
    cal = pd.DataFrame({'date': dates, 'calendar_index': np.arange(len(dates))})
    keys = pd.MultiIndex.from_product([dates, ['000001.SZ','600000.SH']], names=['date','code']).to_frame(index=False)
    features = keys.copy()
    for i, name in enumerate(m.FEATURES):
        features[name] = np.arange(len(features)) * .001 + i
    panel = keys.copy(); panel['close'] = np.arange(len(panel)) * .01 + 10
    features = features.loc[features.date.between('2021-07-01', '2023-12-31')].reset_index(drop=True)
    # Match the real wrapper's past-only panel read projection.
    panel = panel.loc[panel.date.between('2021-07-01','2023-07-01',inclusive='left')].reset_index(drop=True)
    return features, panel, cal


def config(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'BATCH', tmp_path)
    probe = Path(m.__file__).with_name('environment_tilt_mac_recovery.py')
    return dict(policy=copy.deepcopy(m.POLICY), policy_sha256=m.POLICY_SHA256,
        output_root=str(tmp_path/'M13_SYNTHETIC'), cache_files={
            k:dict(path=str(tmp_path/(k+'.parquet')), sha256=k*64) for k in ('m0','panel','calendar')},
        state='FROZEN_READY_FOR_ACTIVATION', execution_ids=dict(preflight='SYNTHETIC_PREFLIGHT', fit='SYNTHETIC_FIT'),
        recipe_sources={'model_selection': dict(path=str(ORIGINAL), sha256=m.sha(ORIGINAL))},
        preparation_receipt={}, original_binding={}, new_use_passport=None,
        original_empty_pool_proof={k:dict(path=str(tmp_path/(k+'.json')),sha256='a'*64)
                                  for k in ('ruling','diagnostic','s1_runner')},
        runtime_versions={'python':'SYNTHETIC'},
        resource_probe_source=dict(path=str(probe),sha256=m.sha(probe)), **m.FLAGS)


def test_custom_fold_exact_original_split_weights_and_unlabeled_scores(recipe, frames):
    before = recipe.FORMAL_FOLDS
    prepared, proof = m.prepare_single_cell(recipe, *frames)
    features, panel, calendar = frames
    fold = recipe.Fold('2023H2', pd.Timestamp('2023-07-01'), pd.Timestamp('2023-12-31'), False)
    labels = recipe.build_close_return_labels(panel, horizons=(2,), calendar=calendar,
        signal_start=pd.Timestamp('2021-07-01'), signal_end_exclusive=pd.Timestamp('2023-07-01'))[2]
    train, test = recipe.split_fold(features, labels, fold)
    pd.testing.assert_frame_equal(prepared['train'], train)
    pd.testing.assert_frame_equal(prepared['test'], test)
    np.testing.assert_array_equal(prepared['weights'], recipe.equal_day_weights(train.date))
    assert recipe.FORMAL_FOLDS is before and len(before) == 6
    assert len(recipe.stage_plan('M0_FORMAL')) == 18  # Never executed by wrapper.
    assert not {'label_return','label_end_date','target_date'} & set(test.columns)
    assert proof['max_model_train_label_end'] < proof['earliest_score_date']
    assert proof['cell'] == m.CELL and proof['expected_score_dates'] == test.date.nunique()
    assert proof['excluded_or_purged_train_feature_rows'] > 0
    assert proof['train_labels_only'] and not proof['test_labels_read']
    assert tuple(prepared['feature_columns']) == m.FEATURES


@pytest.mark.parametrize('kind', ['column_order','nonfinite','duplicate','missing_train_date','missing_test_date','calendar_order','calendar_span'])
def test_new_time_use_invalid_inputs_stop_without_pool_change(kind, recipe, frames):
    f, p, c = (x.copy() for x in frames)
    if kind == 'column_order': f = f[['date','code',*reversed(m.FEATURES)]]
    elif kind == 'nonfinite': f.loc[0,m.FEATURES[0]] = np.nan
    elif kind == 'duplicate': f = pd.concat([f,f.iloc[:1]], ignore_index=True)
    elif kind == 'missing_train_date': f = f.loc[f.date != pd.Timestamp('2022-06-01')]
    elif kind == 'missing_test_date': f = f.loc[f.date != pd.Timestamp('2023-08-01')]
    elif kind == 'calendar_order': c = c.iloc[::-1].reset_index(drop=True)
    elif kind == 'calendar_span': c = c.loc[c.date >= pd.Timestamp('2022-01-01')].reset_index(drop=True)
    with pytest.raises((m.Blocked, recipe.ContractError)):
        m.prepare_single_cell(recipe, f, p, c)


def test_whole_common_keys_never_m0_only_or_ten_stock_truncation(recipe, frames):
    f,p,c=frames
    prepared, proof=m.prepare_single_cell(recipe,f,p,c)
    expected=f.loc[f.date.between('2023-07-01','2023-12-31')]
    assert len(prepared['test']) == len(expected)
    assert prepared['test'].code.nunique() == 2
    assert m.POLICY['pool'] == 'EXACT_EXISTING_FIVE_ARM_COMMON_FINITE_CACHE'
    assert m.POLICY['input_version'] == 'S1_PREPARATION_EXISTING_CACHE_NO_PREFIX'


def test_recipe_prep_pool_change_is_rejected(recipe, frames):
    def dropping(**kwargs):
        result=recipe.prepare_cell_data(**kwargs)
        result['test']=result['test'].iloc[:-1]
        return result
    proxy=SimpleNamespace(**{name:getattr(recipe,name) for name in ('Fold','build_close_return_labels','key_frame_identity')},
                          prepare_cell_data=dropping)
    with pytest.raises(m.Blocked,match='POOL'):
        m.prepare_single_cell(proxy,*frames)


def test_fit_calls_original_model_factory_and_roundtrip_once_with_exact_columns(recipe, frames, tmp_path):
    prepared, proof=m.prepare_single_cell(recipe,*frames)
    guard=FakeGuard(); transaction=m.DeferredFitTerminal(guard)
    factory_calls=[]
    def factory(): factory_calls.append(1); return FakeEstimator()
    proxy=SimpleNamespace(model_for_arm=factory,predict_with_roundtrip=recipe.predict_with_roundtrip)
    path=tmp_path/'synthetic.joblib'
    scores=m.fit_single_cell(proxy,prepared,transaction,path,'SYNTHETIC', 'a'*64,'b'*64)
    assert factory_calls == [1] and len(guard.claims) == 1 and not guard.terminals
    assert transaction.entered and transaction.returned and path.is_file()
    result=m.annotate_scores(scores,prepared,m.sha(path),proof,'c'*64)
    assert len(result)==len(prepared['test'])
    assert result.model_train_label_end.eq(proof['max_model_train_label_end']).all()
    assert result.model_sha256.eq(m.sha(path)).all()
    assert not {'label_return','target_date','label_end_date'} & set(result.columns)
    transaction.finish(True)
    assert guard.terminals[0][1]['real_fit_returned_count']==1
    assert guard.terminals[0][1]['terminal_status']=='COMPLETED'
    with pytest.raises(m.Blocked,match='ONE_FIT'):
        transaction.execute_once(fit_callable=lambda:None,execution_id='SYNTHETIC')


class FakeEstimator:
    """Not an actual model: deterministic sentinel for original recipe plumbing."""
    def fit(self,x,y,sample_weight):
        assert x.shape[1]==36 and np.isfinite(x).all()
        self.mean=float(np.average(y,weights=sample_weight))
        return self
    def predict(self,x):
        return x[:,0]*.001+self.mean


class FakeGuard:
    def __init__(self): self.claims=[]; self.terminals=[]
    def claim(self,**identity): self.claims.append(identity); return identity
    def append_terminal(self,claim,**fields): self.terminals.append((claim,fields))


def test_deferred_terminal_failure_consumes_claim_once_and_preserves_debit():
    guard=FakeGuard(); tx=m.DeferredFitTerminal(guard)
    def fail(): raise RuntimeError('synthetic fit failure')
    with pytest.raises(RuntimeError): tx.execute_once(fit_callable=fail,execution_id='SYNTHETIC')
    tx.finish(False,'RuntimeError');tx.finish(False,'RuntimeError')
    assert len(guard.claims)==len(guard.terminals)==1
    assert guard.terminals[0][1]['fit_call_entered'] is True
    assert guard.terminals[0][1]['real_fit_returned_count']==0


def test_terminal_append_error_is_never_blindly_retried():
    guard=FakeGuard()
    def fail(claim,**fields): guard.terminals.append((claim,fields));raise RuntimeError('durability unknown')
    guard.append_terminal=fail;tx=m.DeferredFitTerminal(guard)
    tx.execute_once(fit_callable=lambda:1,execution_id='SYNTHETIC')
    with pytest.raises(RuntimeError):tx.finish(True)
    tx.finish(False,'RuntimeError')
    assert len(guard.terminals)==1 and tx.terminal_attempted


def test_linked_fit_start_reports_already_consumed_training_labels_truthfully():
    assert m.registry_truth_access('RUN_STARTED','SYNTHETIC_PREFLIGHT',{})=='NOT_READ'
    assert m.registry_truth_access('RUN_STARTED','SYNTHETIC_FIT',{})=='NOT_READ'
    assert m.registry_truth_access('RUN_STARTED','SYNTHETIC_FIT_MODEL',{})=='PAST_TRAIN_H2_ONLY_NO_TEST_OUTCOMES'
    assert m.registry_truth_access('TERMINAL','SYNTHETIC_PREFLIGHT',{})=='CANNOT_CONFIRM_TRAIN_LABEL_PREPARATION_NO_TEST_OUTCOMES'
    assert m.registry_truth_access('TERMINAL','SYNTHETIC_PREFLIGHT',{'proof':{'train_labels_only':True}})=='PAST_TRAIN_H2_ONLY_NO_TEST_OUTCOMES'


def test_score_annotation_refuses_missing_duplicate_nonfinite_or_future_keys(recipe, frames):
    prepared, proof=m.prepare_single_cell(recipe,*frames)
    scores=prepared['test'][['date','code']].copy();scores['prediction']=0.
    for broken in (scores.iloc[:-1],pd.concat([scores,scores.iloc[:1]]),scores.assign(prediction=np.inf),
                   scores.assign(date=pd.Timestamp('2024-01-01'))):
        with pytest.raises(m.Blocked,match='SCORE_KEYS'):
            m.annotate_scores(broken,prepared,'a'*64,proof,'b'*64)


def test_narrow_proof_does_not_invent_first_passport(tmp_path,monkeypatch,recipe,frames):
    c=config(tmp_path,monkeypatch);_,proof=m.prepare_single_cell(recipe,*frames)
    status=m.new_time_use_status(c,proof)
    assert status['mode']=='NARROW_NEW_TIME_USE_PROOF_REQUIRES_REVIEW'
    assert status['first_source_pit_certification']=='CANNOT_CONFIRM'
    passport=dict(status='CERTIFIED_FOR_THIS_TIME_USE',policy_sha256=m.POLICY_SHA256,
                  recipe_sources=c['recipe_sources'],cache_files=c['cache_files'],data_identity=proof['data_identity'])
    p=tmp_path/'passport.json';p.write_text(json.dumps(passport))
    c['new_use_passport']=dict(path=str(p),sha256=m.sha(p))
    assert m.new_time_use_status(c,proof)['mode']=='FAST_BIND_EXISTING_VALID_TIME_USE_PASSPORT'
    passport['data_identity']={};p.write_text(json.dumps(passport));c['new_use_passport']['sha256']=m.sha(p)
    with pytest.raises(m.Blocked,match='PASSPORT'):
        m.new_time_use_status(c,proof)


def test_activation_is_phase_config_source_and_preflight_bound(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch)
    contract=tmp_path/'contract.json';contract.write_text('{}')
    c['contract']=dict(path=str(contract),sha256=m.sha(contract))
    cp=tmp_path/'config.json';cp.write_text(json.dumps(c))
    activation=dict(phase='preflight',decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',config_sha256=m.sha(cp),
        policy_sha256=m.POLICY_SHA256,wrapper_sha256=m.sha(m.__file__),contract_sha256=m.sha(contract),
        execution_id=c['execution_ids']['preflight'],reviewer='INDEPENDENT_SYNTHETIC',review_receipt_sha256='c'*64)
    m.require_activation(c,cp,activation,'preflight')
    for key in ('phase','config_sha256','wrapper_sha256','contract_sha256','execution_id','reviewer'):
        bad={**activation,key:''}
        with pytest.raises(m.Blocked):m.require_activation(c,cp,bad,'preflight')
    with pytest.raises(m.Blocked):m.require_activation({**c,'state':'DRAFT'},cp,activation,'preflight')
    fit_activation={**activation,'phase':'fit','execution_id':c['execution_ids']['fit']}
    with pytest.raises(m.Blocked,match='PREFLIGHT'):
        m.require_activation(c,cp,fit_activation,'fit')


def test_policy_caps_one_cell_and_protected_output(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch);m.validate_config(c)
    assert m.POLICY['max_real_fit_debits']==1 and m.POLICY['threads']==4
    for change in ({'output_root':str(tmp_path/'not_owned')},{'production_allowed':True},
                   {'policy':{**c['policy'],'cell':{**m.CELL,'fold':'2022H1'}}}):
        with pytest.raises(m.Blocked):m.validate_config({**c,**change})


def test_durable_json_writer_does_not_create_a_second_ledger(tmp_path):
    p=tmp_path/'receipt.json';m.write_json(p,{'status':'SYNTHETIC'})
    assert json.loads(p.read_text())=={'status':'SYNTHETIC'}
    m.write_json(p,{'status':'UPDATED_OWNED_RECEIPT'})
    assert json.loads(p.read_text())['status']=='UPDATED_OWNED_RECEIPT'
    assert sorted(x.name for x in tmp_path.iterdir())==['receipt.json']


def test_cache_authentication_once_fit_fast_bind_and_past_only_projection(tmp_path,monkeypatch,frames):
    c=config(tmp_path,monkeypatch)
    for name,ref in c['cache_files'].items():
        p=Path(ref['path']);p.write_bytes(('synthetic_'+name).encode());ref['sha256']=m.sha(p)
    prep=tmp_path/'prep.json';prep.write_text(json.dumps(dict(status='COMPLETED',
        feature_columns={'M0':list(m.FEATURES)},cache_files={r['path']:r['sha256'] for r in c['cache_files'].values()})))
    old=tmp_path/'old.json';old.write_text(json.dumps({'schema_version':'SYNTHETIC'}))
    c['preparation_receipt']=dict(path=str(prep),sha256=m.sha(prep));c['original_binding']=dict(path=str(old),sha256=m.sha(old))
    calls=[];f,p,cal=frames
    def read(path,**kwargs):
        calls.append((Path(path).stem,kwargs))
        return {'m0':f,'panel':p,'calendar':cal}[Path(path).stem].copy()
    monkeypatch.setattr(pd,'read_parquet',read)
    _,stats=m.read_bound_caches(c)
    original_sha=m.sha
    payloads={ref['path'] for ref in c['cache_files'].values()}
    def no_payload_rehash(path):
        assert str(path) not in payloads
        return original_sha(path)
    monkeypatch.setattr(m,'sha',no_payload_rehash)
    m.read_bound_caches(c,accepted_stats=stats)
    assert len(calls)==6
    panel_args=dict(calls)['panel']
    assert panel_args['columns']==['date','code','close']
    assert panel_args['filters'][-1]==('date','<',pd.Timestamp('2023-07-01'))
    Path(c['cache_files']['m0']['path']).write_bytes(b'drift')
    with pytest.raises(m.Blocked,match='STAT_DRIFT'):
        m.read_bound_caches(c,accepted_stats=stats)


def test_two_phase_synthetic_orchestration_registers_before_reads_and_terminal_after_artifacts(
        tmp_path,monkeypatch,recipe,frames):
    c=config(tmp_path,monkeypatch);c['python']=sys.executable
    versions={n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','joblib','pyarrow','threadpoolctl')}
    versions['python']=m.platform.python_version();c['runtime_versions']=versions
    c['resource_caps']=dict(preflight_wall_seconds=30,fit_wall_seconds=30)
    contract=tmp_path/'contract.json';contract.write_text('{}')
    c['contract']=dict(path=str(contract),sha256=m.sha(contract))
    cp=tmp_path/'config.json'
    trace=[]
    class PhaseGuard(FakeGuard):
        def claim_no_fit(self,**identity): trace.append('start_'+identity['execution_id']);return identity
        def append_no_fit_terminal(self,claim,**fields): trace.append('end_'+claim['execution_id']);self.nofit=fields
        def claim(self,**identity): trace.append('model_claim');return super().claim(**identity)
        def append_terminal(self,claim,**fields):
            if fields['terminal_status']=='COMPLETED':
                assert (Path(c['output_root'])/'SCORES.csv').is_file()
                assert json.loads((Path(c['output_root'])/'FIT_RECEIPT.json').read_text())['checkpoint_ref']['sha256']
            trace.append('model_terminal');super().append_terminal(claim,**fields)
    guards=[]
    def guard_factory(*args):g=PhaseGuard();guards.append(g);return g
    def read(config,*,accepted_stats=None):
        assert trace[-1].startswith('start_')
        trace.append('cache_read')
        return dict(m0=frames[0],panel=frames[1],calendar=frames[2]),{'synthetic':'unchanged'}
    proxy=SimpleNamespace(**{name:getattr(recipe,name) for name in
        ('Fold','build_close_return_labels','prepare_cell_data','key_frame_identity','predict_with_roundtrip')},
        model_for_arm=lambda:FakeEstimator())
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'make_guard',guard_factory);monkeypatch.setattr(m,'read_bound_caches',read)
    monkeypatch.setattr(m,'load_recipe',lambda c:proxy)
    monkeypatch.setattr(m,'resource_snapshot',lambda c:{'synthetic':True})
    monkeypatch.setattr(m,'load_original_empty_pool_proof',lambda c:None)
    def activation(phase):
        cp.write_text(json.dumps(c))
        a=dict(phase=phase,decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',config_sha256=m.sha(cp),
            policy_sha256=m.POLICY_SHA256,wrapper_sha256=m.sha(m.__file__),contract_sha256=m.sha(contract),
            execution_id=c['execution_ids'][phase],reviewer='SYNTHETIC_INDEPENDENT',review_receipt_sha256='a'*64)
        if phase=='fit':a['preflight_sha256']=c['accepted_preflight']['sha256']
        return a
    result=m.run_phase(c,cp,activation('preflight'),'preflight')
    assert result['status']=='PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW'
    assert not guards[0].claims
    pre=Path(c['output_root'])/'PREFLIGHT_RECEIPT.json'
    c['accepted_preflight']=dict(path=str(pre),sha256=m.sha(pre))
    result=m.run_phase(c,cp,activation('fit'),'fit')
    assert result['status']=='SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION'
    assert len(guards[1].claims)==len(guards[1].terminals)==1
    assert trace.index('model_terminal')<trace.index('end_SYNTHETIC_FIT')
    assert result['proof']==json.loads(pre.read_text())['proof']
    with pytest.raises(m.Blocked,match='NO_AUTOMATIC_REENTRY'):
        m.run_phase(c,cp,activation('fit'),'fit')


def resource_config(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch)
    c['output_root']=str(tmp_path)
    c['resource_caps']=dict(threads=4,rss_bytes=34359738368,min_available_bytes=8589934592,
                            min_disk_bytes=4294967296,preflight_wall_seconds=600,fit_wall_seconds=180)
    return c


@pytest.mark.skipif(sys.platform!='darwin',reason='Current Mac resource smoke requires Darwin')
def test_resource_snapshot_current_mac_exact_runtime_smoke(tmp_path,monkeypatch):
    c=resource_config(tmp_path,monkeypatch)
    # REAL resource_snapshot and original native probe; no psutil mock/install.
    result=m.resource_snapshot(c)
    assert result['platform']=='Darwin'
    assert 0<result['peak_rss_bytes']==result['rss_bytes']<=34359738368
    assert result['available_bytes']>=8589934592 and result['free_disk_bytes']>=4294967296
    assert result['rss_is_current'] is False
    assert result['rss_measurement']=='GETRUSAGE_PEAK_RSS_CONSERVATIVE_CAP'
    assert result['resource_probe_source']==c['resource_probe_source']


@pytest.mark.parametrize('peak,available,disk', [
    (34359738369,8589934592,4294967296),
    (34359738368,8589934591,4294967296),
    (34359738368,8589934592,4294967295),
])
def test_resource_snapshot_preserves_rss_memory_disk_stops(tmp_path,monkeypatch,peak,available,disk):
    c=resource_config(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_load',lambda *args:SimpleNamespace(rss_bytes=lambda:peak,available_memory=lambda:available))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=disk))
    with pytest.raises(m.Blocked,match='RESOURCE_GUARD_STOP'):
        m.resource_snapshot(c)


def test_resource_snapshot_exact_thresholds_and_probe_identity_fail_closed(tmp_path,monkeypatch):
    c=resource_config(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'_load',lambda *args:SimpleNamespace(rss_bytes=lambda:34359738368,available_memory=lambda:8589934592))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=4294967296))
    result=m.resource_snapshot(c)
    assert result['peak_rss_bytes']==34359738368 and not result['rss_is_current']
    bad={**c,'resource_probe_source':{**c['resource_probe_source'],'sha256':'0'*64}}
    with pytest.raises(m.Blocked,match='SHA_DRIFT'):
        m.resource_snapshot(bad)
    with pytest.raises(m.Blocked,match='BINDING'):
        m.resource_snapshot({**c,'resource_probe_source':None})


@pytest.fixture
def bound_empty_frames(frames):
    f, p, _ = frames
    train = list(pd.bdate_range('2021-07-01', '2023-06-30'))
    # Synthetic observed calendar with the exact nominal counts; never real data.
    protected = {pd.Timestamp(d) for d in m.ORIGINAL_ZERO_POOL_DATES} | {train[0], train[-1]}
    remove = [d for d in train if d not in protected][:len(train)-485]
    train = [d for d in train if d not in remove]
    score = list(pd.bdate_range('2023-07-03', '2023-12-29'))
    score = score[:1] + score[len(score)-123:]
    observed = [pd.Timestamp('2021-06-30'), *train, *score, pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')]
    cal = pd.DataFrame(dict(date=observed,calendar_index=np.arange(len(observed))))
    f = f.loc[f.date.isin(train+score) & ~f.date.isin(pd.to_datetime(m.ORIGINAL_ZERO_POOL_DATES))].reset_index(drop=True)
    p = p.loc[p.date.isin(train)].reset_index(drop=True)
    evidence = dict(original_zero_pool_dates=list(m.ORIGINAL_ZERO_POOL_DATES),
        nominal_train_calendar_days=485,nonempty_train_pool_days=480,
        score_calendar_days=124,nonempty_score_pool_days=124,evidence_refs={'synthetic_only':True})
    return f,p,cal,evidence


def test_original_bound_empty_pool_preserves_calendar_keys_split_and_equal_day_weights(recipe,bound_empty_frames):
    f,p,cal,evidence=bound_empty_frames
    prepared,proof=m.prepare_single_cell(recipe,f,p,cal,original_empty_pool_proof=evidence)
    labels=recipe.build_close_return_labels(p,horizons=(2,),calendar=cal,
        signal_start=pd.Timestamp('2021-07-01'),signal_end_exclusive=pd.Timestamp('2023-07-01'))[2]
    train,test=recipe.split_fold(f,labels,recipe.Fold('2023H2',pd.Timestamp('2023-07-01'),pd.Timestamp('2023-12-31'),False))
    pd.testing.assert_frame_equal(prepared['train'],train)
    pd.testing.assert_frame_equal(prepared['test'],test)
    np.testing.assert_array_equal(prepared['weights'],recipe.equal_day_weights(train.date))
    coverage=proof['calendar_pool_coverage']
    assert (coverage['nominal_train_calendar_days'],coverage['nonempty_train_pool_days'],coverage['empty_train_pool_days'])==(485,480,5)
    assert coverage['original_zero_pool_dates']==list(m.ORIGINAL_ZERO_POOL_DATES)
    assert coverage['score_calendar_days']==coverage['nonempty_score_pool_days']==124
    assert coverage['empty_score_pool_dates']==[] and coverage['complete_nominal_partition']
    assert coverage['calendar_dates_removed']==coverage['synthetic_rows_added']==0
    assert proof['expected_train_feature_dates']==480
    assert set(pd.to_datetime(m.ORIGINAL_ZERO_POOL_DATES)).issubset(set(cal.date))
    assert not set(train.date) & set(pd.to_datetime(m.ORIGINAL_ZERO_POOL_DATES))


@pytest.mark.parametrize('kind',['no_evidence','unknown_missing','known_empty_has_rows','score_missing','calendar_missing','proof_dates','proof_counts'])
def test_bound_empty_pool_rejects_any_unaccounted_coverage_change(kind,recipe,bound_empty_frames):
    f,p,cal,evidence=bound_empty_frames
    if kind=='no_evidence': evidence=None
    elif kind=='unknown_missing': f=f.loc[f.date!=pd.Timestamp('2022-06-01')]
    elif kind=='known_empty_has_rows':
        f=pd.concat([f,f.iloc[:2].assign(date=pd.Timestamp(m.ORIGINAL_ZERO_POOL_DATES[0]))],ignore_index=True)
    elif kind=='score_missing': f=f.loc[f.date!=pd.Timestamp('2023-08-01')]
    elif kind=='calendar_missing':
        cal=cal.loc[cal.date!=pd.Timestamp(m.ORIGINAL_ZERO_POOL_DATES[0])].reset_index(drop=True)
        cal['calendar_index']=np.arange(len(cal))
    elif kind=='proof_dates': evidence={**evidence,'original_zero_pool_dates':list(m.ORIGINAL_ZERO_POOL_DATES[:-1])}
    elif kind=='proof_counts': evidence={**evidence,'nominal_train_calendar_days':484}
    with pytest.raises(m.Blocked):
        m.prepare_single_cell(recipe,f,p,cal,original_empty_pool_proof=evidence)


def synthetic_empty_metadata(c,tmp_path):
    source=tmp_path/'synthetic_s1.py';source.write_text('# synthetic producer identity only\n')
    diagnostic=tmp_path/'diagnostic.json';ruling=tmp_path/'ruling.json'
    refs=dict(s1_runner=dict(path=str(source),sha256=m.sha(source)))
    dates=list(m.ORIGINAL_ZERO_POOL_DATES)
    diag=dict(status='DATE_DIAGNOSTIC_COMPLETE_NO_REPAIR',fit_calls=0,phases={},
        authenticated_inputs=[dict(name=k,path=c['cache_files'][k]['path'],
            actual_sha256=c['cache_files'][k]['sha256'],expected_sha256=c['cache_files'][k]['sha256']) for k in ('m0','calendar')])
    for phase,start,end,inclusive,nominal,nonempty,missing in (
        ('train','2021-07-01','2023-07-01','left',485,480,dates),
        ('score','2023-07-01','2023-12-31','both',124,124,[])):
        diag['phases'][phase]=dict(start=start,end=end,inclusive=inclusive,
            expected=dict(count=nominal),actual=dict(unique_date_count=nonempty),
            missing_dates=missing,missing_dates_empty_in_whole_m0_cache=missing,extra_dates=[])
    diagnostic.write_text(json.dumps(diag));refs['diagnostic']=dict(path=str(diagnostic),sha256=m.sha(diagnostic))
    c['preparation_receipt']=dict(path=str(tmp_path/'synthetic_prep.json'),sha256='b'*64)
    decision=dict(schema='ENV_TILT_M22_ORIGINAL_EMPTY_COMMON_POOL_RULING_V1',
        verdict='ORIGINAL_FIVE_ARM_COMMON_POOL_ZERO_KEY_DATES_PROVEN',
        train_start='2021-07-01',train_end_exclusive='2023-07-01',nominal_train_calendar_days=485,
        nonempty_train_pool_days=480,score_calendar_days=124,nonempty_score_pool_days=124,
        original_zero_pool_dates=dates,preparation_sha256=c['preparation_receipt']['sha256'],
        m0_sha256=c['cache_files']['m0']['sha256'],diagnostic_receipt_path=str(diagnostic),
        diagnostic_receipt_sha256=refs['diagnostic']['sha256'],
        producer_source_sha256=dict(model_selection=c['recipe_sources']['model_selection']['sha256'],s1_runner=refs['s1_runner']['sha256']))
    ruling.write_text(json.dumps(decision));refs['ruling']=dict(path=str(ruling),sha256=m.sha(ruling))
    c['original_empty_pool_proof']=refs
    return diag,decision


def test_original_empty_pool_proof_authenticates_metadata_only_and_changes_input_identity(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch);before=m.input_binding_identity(c)
    synthetic_empty_metadata(c,tmp_path)
    result=m.load_original_empty_pool_proof(c)
    assert result['original_zero_pool_dates']==list(m.ORIGINAL_ZERO_POOL_DATES)
    assert result['evidence_refs']==c['original_empty_pool_proof']
    assert before!=m.input_binding_identity(c)
    # No cache/preparation payload needs to exist to authenticate this metadata.
    assert not Path(c['cache_files']['m0']['path']).exists()
    assert not Path(c['preparation_receipt']['path']).exists()


@pytest.mark.parametrize('kind',['metadata_sha','cache_sha','prep_sha','recipe_sha','producer_sha','ruling_dates','diagnostic_missing','diagnostic_extra','score_missing','diagnostic_cache'])
def test_original_empty_pool_proof_binding_conflicts_stop(kind,tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch);diag,ruling=synthetic_empty_metadata(c,tmp_path)
    refs=c['original_empty_pool_proof']
    if kind=='metadata_sha': refs['ruling']['sha256']='0'*64
    elif kind=='cache_sha': c['cache_files']['m0']['sha256']='0'*64
    elif kind=='prep_sha': c['preparation_receipt']['sha256']='0'*64
    elif kind=='recipe_sha': c['recipe_sources']['model_selection']['sha256']='0'*64
    elif kind=='producer_sha': Path(refs['s1_runner']['path']).write_text('# drift\n')
    elif kind=='ruling_dates': ruling['original_zero_pool_dates']=ruling['original_zero_pool_dates'][:-1]
    elif kind=='diagnostic_missing': diag['phases']['train']['missing_dates']=[]
    elif kind=='diagnostic_extra': diag['phases']['train']['extra_dates']=['2022-01-01']
    elif kind=='score_missing': diag['phases']['score']['missing_dates']=['2023-08-01']
    elif kind=='diagnostic_cache': diag['authenticated_inputs'][0]['actual_sha256']='0'*64
    if kind in ('diagnostic_missing','diagnostic_extra','score_missing','diagnostic_cache'):
        Path(refs['diagnostic']['path']).write_text(json.dumps(diag))
        refs['diagnostic']['sha256']=m.sha(refs['diagnostic']['path'])
        ruling['diagnostic_receipt_sha256']=refs['diagnostic']['sha256']
    if kind in ('ruling_dates','diagnostic_missing','diagnostic_extra','score_missing','diagnostic_cache'):
        Path(refs['ruling']['path']).write_text(json.dumps(ruling));refs['ruling']['sha256']=m.sha(refs['ruling']['path'])
    with pytest.raises(m.Blocked): m.load_original_empty_pool_proof(c)


def test_config_requires_exact_empty_pool_proof_references(tmp_path,monkeypatch):
    c=config(tmp_path,monkeypatch)
    for bad in (None,{},dict(allow_missing=True),{**c['original_empty_pool_proof'],'extra':{}}):
        with pytest.raises(m.Blocked,match='EMPTY_POOL_PROOF'):
            m.validate_config({**c,'original_empty_pool_proof':bad})
