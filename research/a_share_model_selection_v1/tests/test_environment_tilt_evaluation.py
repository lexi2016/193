"""Synthetic orchestration only: no real price/score/checkpoint/registry access."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import environment_tilt_evaluation as m

ORIGINAL=Path('/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/s1-industry-month-20261020/research/a_share_model_selection_v1/frozen_m0/research/a_share_model_selection_v1/model_selection.py')
PRIOR=Path('/Users/lexi/Desktop/学习/因子工厂/workspace/.task_worktrees/s1-industry-month-20261020/research/a_share_model_selection_v1/s1_industry_month_prior.py')
SHA='a'*64


@pytest.fixture(scope='module')
def recipe():return m.support._load(ORIGINAL,'m17_synthetic_original_recipe')


@pytest.fixture(scope='module')
def prior():return m.support._load(PRIOR,'m17_synthetic_original_prior')


@pytest.fixture
def calendar():
    days=[date(2021,12,31),date(2022,1,3),date(2022,7,4),date(2023,1,3),date(2023,7,3),
          date(2024,1,3),date(2024,7,3),date(2025,1,3),date(2025,7,3),date(2026,1,5),
          date(2026,7,3),date(2026,10,15),date(2026,10,16),date(2026,10,19)]
    return pd.DataFrame(dict(date=pd.to_datetime(days),calendar_index=range(len(days))))


class SyntheticPrior:
    PRIOR_COLUMNS=('industry_month_log_relative_lift','industry_month_support',
                   'industry_month_support_years','industry_month_missing')
    @classmethod
    def project_prior(cls,keys,roster,calendar,lookup):
        result=keys.copy()
        result[cls.PRIOR_COLUMNS[0]]=[2. if int(code[:6])%2 else 0. for code in keys.code]
        result[cls.PRIOR_COLUMNS[1]]=1.;result[cls.PRIOR_COLUMNS[2]]=10.;result[cls.PRIOR_COLUMNS[3]]=0.
        return result


def roster():
    return pd.DataFrame(dict(code=[f'{i:06}.SZ' for i in range(60)],
                             industry_key=['A' if i%2 else 'B' for i in range(60)]))


def predictions(fold,cal):
    start,end=m.bounds(fold)
    dates=[t for t in cal.date if start<=t.date()<end]
    frame=pd.MultiIndex.from_product([dates,roster().code],names=['date','code']).to_frame(index=False)
    frame['prediction']=[.02-int(c[:6])*.0001 for c in frame.code]
    return frame


def provenance(fold,frame,recipe):
    return dict(fold=fold,model_sha256=SHA,input_version='SYNTHETIC_'+fold,
        model_train_time=dict(value=(date(int(fold[:4])-1,12,30)).isoformat(),kind='EXACT_MAX',
                              method='SYNTHETIC_ACTUAL_USED_TRAIN_KEYS',evidence_sha256='b'*64),
        full_pool_test_keys=recipe.key_frame_identity(frame))


def limits():
    return dict(threads=4,rss_bytes=2**35,min_available_bytes=2**33,min_disk_bytes=2**32,
        max_payload_file_bytes=2**30,max_segment_rows=10000,max_panel_rows=100000,max_aux_rows=10000,
        max_rows_per_date=1000,max_segment_days=500,max_prepared_days=2000,max_artifact_bytes=2**26,
        max_json_record_bytes=2**20,prepare_wall_seconds=60,select_wall_seconds=60,evaluate_wall_seconds=60)


def prepared_records(recipe,calendar):
    breadth=pd.DataFrame(dict(date=calendar.date,market_advancer_share=1.))
    result=[]
    for fold in m.SEGMENTS:
        frame=predictions(fold,calendar)
        result.extend(m.prepare_segment(recipe,SyntheticPrior,calendar,breadth,roster(),None,
                                         frame,provenance(fold,frame,recipe),limits()))
    return result


class MemorySink:
    def __init__(self):self.data={};self.order=[]
    def put(self,name,value):
        assert name not in self.data;self.data[name]=copy.deepcopy(value);self.order.append(name)
        return dict(path='/synthetic/'+name,sha256=m.support.digest(m.jsonable(value)))
    def json(self,name,value):return self.put(name,value)
    def jsonlines(self,name,value):return self.put(name,list(value))


def provider(cal,trace=None,sink=None,partial_negative=False,missing=None):
    dates=m.calendar_dates(cal);pos={d:i for i,d in enumerate(dates)}
    def labels(requested,*,role,cutoff):
        if trace is not None:trace.append((role,cutoff,tuple(requested)))
        if role=='training':
            assert cutoff in {m.bounds(f)[0] for f in m.OUTER}
            assert all(t<cutoff for t,c in requested)
            previous=[f for f in m.OUTER if m.bounds(f)[0]<cutoff]
            if sink:
                assert all('SELECTIONS_'+f+'.jsonl' in sink.data for f in previous)
        values=[]
        for t,c in requested:
            target=dates[pos[t]+2]
            value=.09 if int(c[:6])%2 else 0.
            if partial_negative and t>=date(2026,7,1):value=-value
            if (cutoff and target>=cutoff) or (missing and (t,c)==missing):value=None
            values.append(m.core.Outcome(t,c,value,target))
        return tuple(values)
    return labels


@contextmanager
def checkpoint(fold):yield


def test_full_pool_before_auxiliary_prior_and_no_labels_top50(recipe,calendar):
    frame=predictions('2022H1',calendar)
    b=pd.DataFrame(dict(date=calendar.date,market_advancer_share=1.))
    result=m.prepare_segment(recipe,SyntheticPrior,calendar,b,roster(),None,frame,
                             provenance('2022H1',frame,recipe),limits())
    record=result[0];day=m.restore_day(record)
    assert record['full_pool_rows']==60 and len(day.rows)==50
    assert tuple(r.code for r in day.rows)==tuple(f'{i:06}.SZ' for i in range(50))
    assert record['topk_provenance'].startswith('STABLE_DESC')
    assert day.next_session>day.signal_date and day.h2_session>day.next_session
    assert record['industries']['000001.SZ']=='A'


@pytest.mark.parametrize('kind',['outside_top50_nonfinite','duplicate','missing_key','missing_date','bad_identity','bad_train_time'])
def test_preparation_conflicts_stop_instead_of_rebuilding_pool(kind,recipe,calendar):
    frame=predictions('2022H1',calendar);proof=provenance('2022H1',frame,recipe)
    if kind=='outside_top50_nonfinite':frame.loc[59,'prediction']=np.nan
    elif kind=='duplicate':frame=pd.concat([frame,frame.iloc[:1]])
    elif kind=='missing_key':frame=frame.iloc[:-1]
    elif kind=='missing_date':frame=frame.iloc[:0]
    elif kind=='bad_identity':proof['full_pool_test_keys']={}
    else:proof['model_train_time']['value']='2022-01-03'
    with pytest.raises(m.Blocked):
        m.prepare_segment(recipe,SyntheticPrior,calendar,
            pd.DataFrame(dict(date=calendar.date,market_advancer_share=1.)),roster(),None,frame,proof,limits())


def test_missing_calendar_anchor_is_a_dependency_stop_not_partial_drop(recipe):
    cal=pd.DataFrame(dict(date=pd.to_datetime(['2021-12-31','2022-06-30','2022-07-01']),calendar_index=[0,1,2]))
    frame=predictions('2022H1',cal)
    with pytest.raises(m.Blocked,match='CALENDAR_ANCHOR'):
        m.prepare_segment(recipe,SyntheticPrior,cal,pd.DataFrame(dict(date=cal.date,market_advancer_share=1.)),
            roster(),None,frame,provenance('2022H1',frame,recipe),limits())


def test_original_prior_and_all_observed_breadth_producers_reused(recipe,prior,calendar):
    panel=pd.MultiIndex.from_product([calendar.date,roster().code],names=['date','code']).to_frame(index=False)
    panel['close']=100.+np.repeat(np.arange(len(calendar)),60)
    panel['amount']=1.
    # This code is outside every score pool; it still participates in breadth.
    other=pd.DataFrame(dict(date=calendar.date,code='OUTSIDE',close=100.-np.arange(len(calendar)),amount=1.))
    panel=pd.concat([panel,other],ignore_index=True)
    breadth=recipe._calendar_daily_market(panel,calendar)
    monthly=pd.DataFrame([dict(industry_key=k,year=2020,month=mo,actual_hits=(100 if mo in (1,7) else 1),expected_hits=100)
                          for k in ('A','B') for mo in range(1,13)])
    annual=monthly.groupby(['industry_key','year'],as_index=False)[['actual_hits','expected_hits']].sum()
    lookup=prior.prior_lookup(annual,monthly,[2022])
    frame=predictions('2022H1',calendar)
    result=m.prepare_segment(recipe,prior,calendar,breadth,roster(),lookup,frame,
                             provenance('2022H1',frame,recipe),limits())
    keys=frame.sort_values(['date','prediction','code'],ascending=[True,False,True]).head(50)[['date','code']].reset_index(drop=True)
    projected=prior.project_prior(keys,roster(),calendar,lookup)
    assert [r.prior for r in m.restore_day(result[0]).rows]==projected[prior.PRIOR_COLUMNS[0]].tolist()
    assert m.restore_day(result[0]).breadth==pytest.approx(60/61)


def sidecar(last='2023-06-28'):
    return dict(status='COMPLETED',identity=dict(arm='M0',horizon=2,code_sha256={'model_selection':SHA},
        data_identity=dict(schema_version='A_SHARE_CANARY_CELL_DATA_IDENTITY_V1',
            feature_columns=list(m.support.FEATURES),
            training_keys=dict(schema_version='A_SHARE_DATE_CODE_SORTED_KEYS_V1',date_max=last,
                row_count=10,unique_key_count=10,ordered_keys_sha256='b'*64,sorted_keyset_sha256='c'*64),
            training_label_vector=dict(sha256='d'*64))))


def test_actual_used_train_key_max_h2_restore_and_upper_bound_are_distinct():
    dates=(date(2023,6,28),date(2023,6,29),date(2023,6,30),date(2023,7,3))
    proof=m.restore_model_time(sidecar(),dates,model_source_sha256=SHA,calendar_sha256='e'*64)
    assert proof['value']=='2023-06-30' and proof['kind']=='EXACT_MAX'
    proof['evidence_sha256']='f'*64
    assert m.validate_time_proof(proof,date(2023,7,3))==date(2023,6,30)
    upper={**proof,'kind':'UPPER_BOUND','method':'EVIDENCED_BOUND_FROM_ACTUAL_TRAIN_KEYS'}
    assert m.validate_time_proof(upper,date(2023,7,3))==date(2023,6,30)
    assert upper['kind']=='UPPER_BOUND'
    with pytest.raises(m.Blocked):m.validate_time_proof({**proof,'method':'PURGE_CUTOFF'},date(2023,7,3))
    with pytest.raises(m.Blocked):m.restore_model_time(sidecar(),dates[:2],model_source_sha256=SHA,calendar_sha256='e'*64)
    bad=sidecar();bad['identity']['data_identity']['training_keys']['schema_version']='PRE_PURGE_CANDIDATE_KEYS'
    with pytest.raises(m.Blocked):m.restore_model_time(bad,dates,model_source_sha256=SHA,calendar_sha256='e'*64)


def test_chronological_fit_freezes_previous_lists_before_later_training_labels(recipe,calendar):
    records=prepared_records(recipe,calendar);sink=MemorySink();trace=[]
    result=m.fit_and_freeze(records,m.calendar_dates(calendar),SHA,'b'*64,
                           provider(calendar,trace,sink),sink,checkpoint)
    assert result['scalar_fits']==6 and result['base_fits']==0 and len(trace)==6
    assert all(0<=fit['lambda_value']<=.25 for fit in sink.data['LAMBDAS.json'].values())
    assert all(fit['scored_dates']>=fit['complete_label_dates'] for fit in sink.data['LAMBDAS.json'].values())
    assert len(sink.data['SELECTIONS.jsonl'])==7
    assert sink.order.index('SELECTIONS_2024H1.jsonl')<sink.order.index('TRAIN_REQUEST_2024H2.json')
    for row in sink.data['SELECTIONS.jsonl']:
        pair=m.restore_pair(row)
        assert len(pair.baseline.selected)==len(pair.candidate.selected)==10


def test_missing_early_segment_cannot_shorten_window_or_fix_lambda(recipe,calendar):
    records=[r for r in prepared_records(recipe,calendar) if r['segment']!='2022H1'];trace=[]
    with pytest.raises(m.Blocked,match='TWO_YEAR'):
        m.fit_and_freeze(records,m.calendar_dates(calendar),SHA,'b'*64,provider(calendar,trace),MemorySink(),checkpoint)
    assert trace==[]


def test_isolated_frozen_evaluation_partial_separate_swap_and_industry_hhi(recipe,calendar):
    sink=MemorySink()
    m.fit_and_freeze(prepared_records(recipe,calendar),m.calendar_dates(calendar),SHA,'b'*64,
                     provider(calendar,sink=sink),sink,checkpoint)
    records=sink.data['SELECTIONS.jsonl'];trace=[]
    result=m.evaluate_frozen(records,m.calendar_dates(calendar),SHA,'b'*64,provider(calendar,trace,partial_negative=True))
    assert trace[0][0]=='outer' and trace[0][1] is None
    assert result['state']=='DIRECTION_ONLY' and result['evaluation']['positive_full_folds']==5
    assert result['evaluation']['partial_fold_mean']<0
    for row,paired in zip(result['diagnostics']['dates'],result['evaluation']['dates']):
        assert row['swap_delta']==pytest.approx(paired['delta'])
        assert sum(row['industry_gross_delta_contribution'].values())==pytest.approx(paired['delta'])
        assert 0<=row['industry']['TILT']['hhi']<=1
        assert all(1<=rank<=50 for rank in row['entrants'].values())
    assert result['diagnostics']['optional_risk']['beta'].startswith('CANNOT_CONFIRM')
    # Removing an actually selected candidate key excludes the WHOLE paired day.
    first=m.restore_pair(records[0]);code=first.candidate.selected[0].code
    changed=m.evaluate_frozen(records,m.calendar_dates(calendar),SHA,'b'*64,
        provider(calendar,missing=(first.baseline.signal_date,code)))
    assert changed['evaluation']['dates'][0]['paired_complete'] is False
    assert changed['state']=='CANNOT_CONFIRM'
    assert records==sink.data['SELECTIONS.jsonl']


def test_selection_tamper_fails_before_outer_label_access(recipe,calendar):
    sink=MemorySink();m.fit_and_freeze(prepared_records(recipe,calendar),m.calendar_dates(calendar),SHA,'b'*64,
        provider(calendar,sink=sink),sink,checkpoint)
    records=copy.deepcopy(sink.data['SELECTIONS.jsonl']);records[0]['candidate']['selected'][0]['score']+=1
    trace=[]
    with pytest.raises(m.Blocked,match='HASH'):
        m.evaluate_frozen(records,m.calendar_dates(calendar),SHA,'b'*64,provider(calendar,trace))
    assert trace==[]


class Session:
    def __init__(self,phase,active=True):self.phase,self.active=phase,active
    def require_active(self):
        if not self.active:raise m.Blocked('NOT_REGISTERED')


def test_actual_label_adapter_uses_original_h2_and_never_fills_missing(recipe,calendar):
    config=dict(caps=limits(),inputs={'panel':{'path':'/synthetic/panel','sha256':SHA}})
    adapter=m.BoundFiles(config,Session('select'))
    adapter.modules=lambda:(recipe,SyntheticPrior)
    requested=[(date(2022,1,3),'000001.SZ'),(date(2022,1,3),'MISSING')]
    panel=pd.DataFrame(dict(date=pd.to_datetime(['2022-01-03','2022-07-04','2023-01-03']),
                            code='000001.SZ',close=[10.,12.,15.]))
    reads=[]
    def frame(ref,**kw):reads.append(kw);return panel.copy()
    adapter.frame=frame
    actual=adapter.labels(requested,role='training',cutoff=date(2024,1,1),calendar=calendar)
    assert actual[0].value==.5 and actual[0].label_end==date(2023,1,3)
    assert actual[1].value is None
    assert all(t<pd.Timestamp('2024-01-01') for t in reads[0]['filters'][0][2])
    with pytest.raises(m.Blocked,match='OUTER_TRUTH'):
        adapter.labels(requested,role='outer',cutoff=None,calendar=calendar)
    adapter.session=Session('prepare')
    with pytest.raises(m.Blocked,match='TRAINING_STAGE'):
        adapter.labels(requested,role='training',cutoff=date(2024,1,1),calendar=calendar)


def test_bound_files_require_registered_start_authenticate_once_and_fast_bind(tmp_path):
    p=tmp_path/'synthetic.bin';p.write_bytes(b'SYNTHETIC_NOT_MARKET')
    ref=dict(path=str(p),sha256=m.support.sha(p));config=dict(caps=limits())
    adapter=m.BoundFiles(config,Session('prepare',False))
    with pytest.raises(m.Blocked,match='NOT_REGISTERED'):adapter.authenticate(ref)
    adapter.session.active=True;adapter.authenticate(ref);adapter.authenticate(ref)
    assert len(adapter.bound)==1
    fast=m.BoundFiles(config,Session('select'),adapter.bound);fast.authenticate(ref)
    p.write_bytes(b'CHANGED')
    with pytest.raises(m.Blocked,match='FAST_BIND'):m.BoundFiles(config,Session('select'),adapter.bound).authenticate(ref)


def test_artifact_caps_existing_output_and_accepted_list_hash_guard(tmp_path):
    sink=m.ArtifactSink(tmp_path,limits());ref=sink.jsonlines('LIST.jsonl',[{'x':1}])
    assert m.read_artifact(ref,Session('evaluate'),limits(),lines=True)==[{'x':1}]
    with pytest.raises(m.Blocked,match='NO_OVERWRITE'):sink.jsonlines('LIST.jsonl',[{'x':2}])
    Path(ref['path']).write_text('{"x":2}\n')
    with pytest.raises(m.Blocked,match='SHA_DRIFT'):m.read_artifact(ref,Session('evaluate'),limits(),lines=True)
    tiny=m.ArtifactSink(tmp_path,{**limits(),'max_json_record_bytes':2})
    with pytest.raises(m.Blocked,match='RECORD_SIZE'):tiny.jsonlines('TOO_LARGE.jsonl',[{'x':100}])


def reference(path):return dict(path=str(path),sha256=m.support.sha(path))


def fixture_config(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BATCH',tmp_path)
    stub=tmp_path/'CONTROL.json';stub.write_text('{"decision":"PASS"}')
    ref=reference(stub)
    versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','joblib','pyarrow','threadpoolctl')}
    versions['python']=m.platform.python_version()
    core_review=tmp_path/'SYNTHETIC_CORE_REVIEW.json'
    core_review.write_text(json.dumps(dict(verdict='REVIEWED/PASS',
        source_sha256=m.support.sha(m.core.__file__),config_sha256_unchanged=m.core.CONFIG_SHA256)))
    c=dict(state='FROZEN_READY_FOR_ACTIVATION',policy=copy.deepcopy(m.POLICY),policy_sha256=m.POLICY_SHA256,
        output_root=str(tmp_path/'M17_SYNTHETIC'),python=sys.executable,executor_owner='SYNTHETIC_EXECUTOR',
        sources=dict(recipe=reference(ORIGINAL),prior=reference(PRIOR),core=reference(Path(m.core.__file__)),
                     support=reference(Path(m.support.__file__))),
        inputs={k:ref for k in ('panel','calendar','roster','industry_year','month_year')},
        segments=[dict(fold=f,format='normalized_oof' if f in m.EARLY else 'original_checkpoint',
                       input_version='SYNTHETIC_'+f,artifact=ref,metadata=ref,accepted_review=ref) for f in m.SEGMENTS],
        contract=ref,resource_probe_source=reference(Path(m.support.__file__).with_name('environment_tilt_mac_recovery.py')),
        accepted_core_review=reference(core_review),registry_dependencies={k:ref for k in ('core','adapter','builder','validator')},
        registry_path=str(tmp_path/'NO_ACTUAL_REGISTRY'),runtime_versions=versions,passport=None,caps=limits(),
        execution_ids={phase:'SYNTHETIC_'+phase.upper() for phase in m.PHASES},**m.FLAGS)
    contract=tmp_path/'SYNTHETIC_CONTRACT.json'
    contract.write_text(json.dumps(dict(state='FROZEN_READY_FOR_ACTIVATION',policy_sha256=m.POLICY_SHA256,
        wrapper_sha256=m.support.sha(m.__file__),scientific_identity=m.scientific_identity(c))))
    c['contract']=reference(contract)
    return c


def activation(config,config_path,phase):
    config_path.write_text(json.dumps(config))
    a=dict(phase=phase,decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',execution_id=config['execution_ids'][phase],
        config_sha256=m.support.sha(config_path),wrapper_sha256=m.support.sha(m.__file__),policy_sha256=m.POLICY_SHA256,
        scientific_identity=m.scientific_identity(config),contract_sha256=config['contract']['sha256'],
        reviewer='SYNTHETIC_INDEPENDENT',review_receipt_sha256=SHA)
    if phase!='prepare':a['upstream_receipt_sha256']=config['accepted_prepare' if phase=='select' else 'accepted_select']['sha256']
    return a


@pytest.mark.parametrize('mutation',['draft','null_score','logical_version','missing_segment','changed_policy','unbounded','wrong_owner_path'])
def test_draft_logical_null_missing_or_changed_bindings_cannot_execute(tmp_path,monkeypatch,mutation):
    c=fixture_config(tmp_path,monkeypatch)
    if mutation=='draft':c['state']='DRAFT'
    elif mutation=='null_score':c['segments'][0]['artifact']={'path':None,'sha256':None}
    elif mutation=='logical_version':c['segments'][0]['input_version']={'logical_id':'UNRESOLVED'}
    elif mutation=='missing_segment':c['segments']=c['segments'][1:]
    elif mutation=='changed_policy':c['policy']['core_policy_sha256']='0'*64
    elif mutation=='unbounded':c['caps']['max_artifact_bytes']=float('inf')
    else:c['output_root']=str(tmp_path/'FOREIGN')
    with pytest.raises(m.Blocked):m.validate_config(c)


def test_original_checkpoint_adapter_reads_only_bound_saved_predictions_not_predict(tmp_path,recipe,calendar):
    import joblib
    frame=predictions('2024H1',calendar)
    payload=tmp_path/'SYNTHETIC.joblib';joblib.dump({'predictions':frame,'model':'NO_MODEL_OBJECT'},payload)
    meta=sidecar('2022-07-04')
    meta['identity'].update(stage='M0_FORMAL',fold='2024H1')
    meta['identity']['code_sha256']['model_selection']=m.support.sha(ORIGINAL)
    meta['identity']['data_identity']['test_keys']=recipe.key_frame_identity(frame)
    meta['artifact_sha256']=m.support.sha(payload)
    mp=tmp_path/'SIDECAR.json';mp.write_text(json.dumps(meta))
    rp=tmp_path/'REVIEW.json';rp.write_text('{"decision":"PASS"}')
    c=dict(caps=limits(),sources={'recipe':reference(ORIGINAL)},inputs={'calendar':{'sha256':SHA}})
    adapter=m.BoundFiles(c,Session('prepare'));adapter.modules=lambda:(recipe,SyntheticPrior)
    spec=dict(fold='2024H1',format='original_checkpoint',input_version='SYNTHETIC_ORIGINAL',
              artifact=reference(payload),metadata=reference(mp),accepted_review=reference(rp))
    rp.write_text(json.dumps(dict(decision='PASS',score_binding={k:spec[k] for k in ('fold','artifact','metadata','input_version')})))
    spec['accepted_review']=reference(rp)
    actual,proof=adapter.score_segment(spec,calendar)
    pd.testing.assert_frame_equal(actual,frame)
    assert adapter.checkpoint_unpacks==1
    assert proof['model_train_time']['value']=='2023-07-03'
    assert proof['model_train_time']['kind']=='EXACT_MAX'


def test_normalized_oof_receipt_binding_and_csv_original_float_precision(tmp_path,recipe,calendar):
    frame=predictions('2024H1',calendar)
    sensitive=np.nextafter(.02,1.)
    frame.loc[0,'prediction']=sensitive
    model_sha='e'*64
    frame['model_sha256']=model_sha;frame['model_train_label_end']='2023-07-03'
    config=tmp_path/'FIT_CONFIG.json'
    config.write_text(json.dumps(dict(recipe_sources={'model_selection':reference(ORIGINAL)},
        policy={'cell':dict(arm='M0',horizon=2,fold='2024H1')})))
    frame['config_sha256']=m.support.sha(config)
    score=tmp_path/'SCORES.csv';frame.to_csv(score,index=False)
    data=sidecar('2022-07-04')['identity']['data_identity']
    data['test_keys']=recipe.key_frame_identity(frame)
    fit=tmp_path/'FIT.json'
    fit.write_text(json.dumps(dict(status='SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION',
        proof=dict(feature_columns=list(m.support.FEATURES),params=m.support.PARAMS,
                   data_identity=data,max_model_train_label_end='2023-07-03'),
        scores={'sha256':m.support.sha(score)},checkpoint_ref={'sha256':model_sha},input_binding_identity='d'*64)))
    manifest=tmp_path/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(schema='M17_NORMALIZED_OOF_BINDING_V1',fold='2024H1',
        scores_ref=reference(score),model_recipe_sha256=m.support.sha(ORIGINAL),
        pool_policy='ORIGINAL_FIVE_ARM_COMMON_FINITE',input_version='SYNTHETIC_V1',fit_receipt_ref=reference(fit),
        fit_config_ref=reference(config),fit_input_binding_identity='d'*64)))
    review=tmp_path/'REVIEW.json';review.write_text('{"decision":"PASS"}')
    c=dict(caps=limits(),sources={'recipe':reference(ORIGINAL)},inputs={'calendar':{'sha256':SHA}})
    adapter=m.BoundFiles(c,Session('prepare'));adapter.modules=lambda:(recipe,SyntheticPrior)
    spec=dict(fold='2024H1',format='normalized_oof',input_version='SYNTHETIC_V1',artifact=reference(score),
              metadata=reference(manifest),accepted_review=reference(review))
    review.write_text(json.dumps(dict(decision='PASS',score_binding={k:spec[k] for k in ('fold','artifact','metadata','input_version')})))
    spec['accepted_review']=reference(review)
    actual,proof=adapter.score_segment(spec,calendar)
    assert actual.prediction.iloc[0].hex()==float(sensitive).hex()
    assert proof['model_sha256']==model_sha and adapter.checkpoint_unpacks==0
    broken=copy.deepcopy(spec);broken['input_version']='DIFFERENT_VERSION'
    with pytest.raises(m.Blocked,match='REVIEW|PROVENANCE'):adapter.score_segment(broken,calendar)


def test_full_three_phase_synthetic_lifecycle_and_upstream_authorization(tmp_path,monkeypatch,recipe,calendar):
    c=fixture_config(tmp_path,monkeypatch);cp=tmp_path/'CONFIG.json';events=[];label_calls=[]
    class Guard:
        def claim_no_fit(self,**identity):
            assert not any(e[0]=='START' and e[1]['execution_id']==identity['execution_id'] for e in events)
            events.append(('START',identity));return identity
        def append_no_fit_terminal(self,claim,**fields):events.append(('TERMINAL',claim,fields))
    guard=Guard()
    class Files:
        def __init__(self,config,session,accepted_bound_files=None):
            self.config,self.session=config,session;self.bound={'SYNTHETIC':{'stat':'stable'}};self.checkpoint_unpacks=0
        def modules(self):return recipe,SyntheticPrior
        def context(self):
            self.session.require_active();assert events[-1][0]=='START'
            return calendar,pd.DataFrame(dict(date=calendar.date,market_advancer_share=1.)),roster(),None
        def score_segment(self,spec,cal):
            f=predictions(spec['fold'],cal);return f,provenance(spec['fold'],f,recipe)
        def labels(self,requested,*,role,cutoff,calendar):
            self.session.require_active();label_calls.append((role,cutoff))
            if role=='outer':
                assert self.session.phase=='evaluate'
                previous=json.loads((Path(c['output_root'])/'SELECT_RECEIPT.json').read_text())
                assert previous['status']=='COMPLETED'
            else:
                for fold in m.OUTER:
                    if m.bounds(fold)[0]<cutoff:assert (Path(c['output_root'])/('SELECTIONS_'+fold+'.jsonl')).is_file()
            return provider(calendar)(requested,role=role,cutoff=cutoff)
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'resource_check',lambda config:{'REUSED_R1_RESOURCE_SCOPE':'synthetic'})
    def run(phase):
        return m.run_phase(c,cp,activation(c,cp,phase),phase,guard_factory=lambda *args:guard,adapter_factory=Files)
    a=run('prepare');assert a['status']=='COMPLETED' and label_calls==[]
    c['accepted_prepare']=reference(Path(c['output_root'])/'PREPARE_RECEIPT.json')
    b=run('select');assert b['result']['scalar_fits']==6 and len(label_calls)==6
    c['accepted_select']=reference(Path(c['output_root'])/'SELECT_RECEIPT.json')
    result=run('evaluate');assert result['result']['state']=='DIRECTION_ONLY'
    assert len(label_calls)==7 and label_calls[-1]==('outer',None)
    assert len([e for e in events if e[0]=='START'])==9
    assert len([e for e in events if e[0]=='TERMINAL'])==9
    count=len(events)
    with pytest.raises(m.Blocked,match='NO_AUTOMATIC_REENTRY'):run('evaluate')
    assert len(events)==count
    # A valid select receipt cannot be used with a different prepared receipt.
    altered={**c,'accepted_prepare':{**c['accepted_prepare'],'sha256':'0'*64}}
    with pytest.raises(m.Blocked,match='DIFFERENT_PREPARATION'):
        m.require_activation(altered,cp,activation(altered,cp,'evaluate'),'evaluate')


def test_registered_preparation_failure_has_terminal_and_no_retry(tmp_path,monkeypatch):
    c=fixture_config(tmp_path,monkeypatch);cp=tmp_path/'CONFIG.json';events=[]
    class Guard:
        def claim_no_fit(self,**kw):events.append('START');return kw
        def append_no_fit_terminal(self,claim,**kw):events.append(('TERMINAL',kw))
    class Failed:
        def __init__(self,*args,**kwargs):pass
        def context(self):raise m.Blocked('SYNTHETIC_MISSING_DEPENDENCY')
    monkeypatch.setattr(m.platform,'system',lambda:'Darwin')
    monkeypatch.setattr(m,'resource_check',lambda c:{'synthetic':True})
    a=activation(c,cp,'prepare')
    with pytest.raises(m.Blocked,match='SYNTHETIC_MISSING'):
        m.run_phase(c,cp,a,'prepare',guard_factory=lambda *args:Guard(),adapter_factory=Failed)
    assert events[0]=='START' and events[1][1]['succeeded'] is False
    receipt=json.loads((Path(c['output_root'])/'PREPARE_RECEIPT.json').read_text())
    assert receipt['status']=='FAILED_OR_INTERRUPTED_NO_RETRY'


def test_last_partial_unknown_h2_preserves_lists_and_invalid_pair_without_extra_calendar(recipe,calendar):
    # The actual bound calendar may end at T+1; no future date is synthesized.
    cal=calendar.loc[calendar.date<=pd.Timestamp('2026-10-16')].copy()
    records=prepared_records(recipe,cal)
    last=records[-1]
    assert last['day']['signal_date']=='2026-10-15' and last['day']['h2_session'] is None
    assert last['h2_anchor_state']=='UNOBSERVED_TARGET_INVALID_PAIR'
    sink=MemorySink()
    def outcomes(requested,*,role,cutoff):
        mature=[(t,c) for t,c in requested if t!=date(2026,10,15)]
        values=provider(calendar)(mature,role=role,cutoff=cutoff)
        return (*values,*(m.core.Outcome(t,c,None,None) for t,c in requested if t==date(2026,10,15)))
    m.fit_and_freeze(records,m.calendar_dates(cal),SHA,'b'*64,outcomes,sink,checkpoint)
    selected=sink.data['SELECTIONS.jsonl']
    pair=m.restore_pair(selected[-1])
    assert len(pair.baseline.selected)==len(pair.candidate.selected)==10
    assert pair.baseline.h2_session is pair.candidate.h2_session is None
    result=m.evaluate_frozen(selected,m.calendar_dates(cal),SHA,'b'*64,outcomes)
    final=result['evaluation']['dates'][-1]
    assert not final['paired_complete'] and final['reason']=='h2_calendar_anchor_unavailable'
    assert final['baseline_return'] is final['candidate_return'] is final['delta'] is None
    assert result['state']=='DIRECTION_ONLY'
    assert result['coverage']['nominal_outer_dates']==7
    assert result['coverage']['unobserved_h2_dates']==['2026-10-15']
    assert result['coverage']['last_evaluable_date']=='2026-07-03'


def test_original_label_adapter_unobserved_target_remains_nan_and_none(recipe,calendar):
    cal=calendar.loc[calendar.date<=pd.Timestamp('2026-10-16')].copy()
    adapter=m.BoundFiles(dict(caps=limits(),inputs={'panel':{'path':'/SYNTHETIC','sha256':SHA}}),Session('evaluate'))
    adapter.modules=lambda:(recipe,SyntheticPrior)
    seen=[]
    def frame(ref,**kw):
        seen.extend(kw['filters'][0][2])
        return pd.DataFrame(dict(date=pd.to_datetime(['2026-10-15','2026-10-16']),code='000001.SZ',close=[10.,11.]))
    adapter.frame=frame
    result=adapter.labels([(date(2026,10,15),'000001.SZ')],role='outer',cutoff=None,calendar=cal)
    assert result==(m.core.Outcome(date(2026,10,15),'000001.SZ',None,None),)
    assert seen==[pd.Timestamp('2026-10-15')]
    assert adapter.labels([],role='outer',cutoff=None,calendar=cal)==()


def zero_fixture(recipe,calendar):
    cal=pd.concat([calendar,pd.DataFrame(dict(date=pd.to_datetime(['2022-01-04','2022-01-05']),calendar_index=[0,0]))],ignore_index=True)
    cal=cal.sort_values('date').reset_index(drop=True);cal['calendar_index']=np.arange(len(cal))
    frame=predictions('2022H1',cal)
    frame=frame.loc[frame.date!=pd.Timestamp('2022-01-04')].reset_index(drop=True)
    proof=provenance('2022H1',frame,recipe)
    proof['zero_pool_proof']=dict(zero_pool_dates=['2022-01-04'],nominal_calendar_days=3,nonempty_pool_days=2)
    return cal,frame,proof


def test_version_bound_zero_pool_emits_empty_day_and_keeps_nominal_chronology(recipe,calendar):
    cal,frame,proof=zero_fixture(recipe,calendar)
    records=m.prepare_segment(recipe,SyntheticPrior,cal,
        pd.DataFrame(dict(date=cal.date,market_advancer_share=1.)),roster(),None,frame,proof,limits())
    assert len(records)==3
    empty=records[1]
    assert empty['day']['signal_date']=='2022-01-04' and empty['original_pool_state']=='PROVEN_EMPTY'
    assert empty['full_pool_rows']==0 and m.restore_day(empty).rows==()
    assert m.restore_day(empty).next_session==date(2022,1,5)
    # Genuine empty U_t contributes no fabricated zero observations/weights.
    all_records=prepared_records(recipe,cal)
    all_records=[r for r in all_records if r['segment']!='2022H1']+records
    sink=MemorySink();m.fit_and_freeze(all_records,m.calendar_dates(cal),SHA,'b'*64,provider(cal),sink,checkpoint)
    mask=sink.data['LAMBDAS.json']['2024H1']['train_mask']
    assert ['2022-01-04',False,'incomplete_TopK_labels'] in mask


@pytest.mark.parametrize('kind',['no_proof','other_missing','extra_score','wrong_dates','wrong_counts'])
def test_zero_pool_exact_missing_partition_required(kind,recipe,calendar):
    cal,frame,proof=zero_fixture(recipe,calendar)
    if kind=='no_proof':proof.pop('zero_pool_proof')
    elif kind=='other_missing':
        frame=frame.loc[frame.date!=pd.Timestamp('2022-01-05')]
        proof['full_pool_test_keys']=recipe.key_frame_identity(frame)
    elif kind=='extra_score':
        frame=pd.concat([frame,frame.iloc[:1].assign(date=pd.Timestamp('2022-01-04'))],ignore_index=True)
        proof['full_pool_test_keys']=recipe.key_frame_identity(frame)
    elif kind=='wrong_dates':proof['zero_pool_proof']['zero_pool_dates']=['2022-01-05']
    else:proof['zero_pool_proof']['nominal_calendar_days']=2
    with pytest.raises(m.Blocked):
        m.prepare_segment(recipe,SyntheticPrior,cal,pd.DataFrame(dict(date=cal.date,market_advancer_share=1.)),
                          roster(),None,frame,proof,limits())


def test_zero_pool_metadata_is_bound_to_exact_version_calendar_source_keys_and_review(tmp_path,recipe,calendar):
    cal,frame,provenance=zero_fixture(recipe,calendar)
    full=provenance['full_pool_test_keys']
    evidence=tmp_path/'SYNTHETIC_PRODUCER_EVIDENCE.json';evidence.write_text('{"scope":"synthetic producer partition"}')
    proof_file=tmp_path/'ZERO_POOL.json';review_file=tmp_path/'ZERO_REVIEW.json'
    spec=dict(fold='2022H1',input_version='SYNTHETIC_PREFIX_V1',artifact={'sha256':SHA})
    proof=dict(schema='M17_ORIGINAL_ZERO_POOL_COVERAGE_V1',fold='2022H1',input_version=spec['input_version'],
        pool_policy='ORIGINAL_FIVE_ARM_COMMON_FINITE',score_artifact_sha256=SHA,
        full_pool_test_keys_sha256=m.support.digest(full),calendar_sha256='b'*64,recipe_source_sha256='c'*64,
        nominal_sessions_sha256=m.support.digest(m.jsonable(m.sessions(m.calendar_dates(cal),*m.bounds('2022H1')))),
        nominal_calendar_days=3,nonempty_pool_days=2,zero_pool_dates=['2022-01-04'],producer_evidence_ref=reference(evidence))
    proof_file.write_text(json.dumps(proof))
    spec['zero_pool']=dict(proof=reference(proof_file))
    review_file.write_text(json.dumps(dict(decision='PASS',proof_ref=spec['zero_pool']['proof'])))
    spec['zero_pool']['accepted_review']=reference(review_file)
    config=dict(caps=limits(),inputs={'calendar':{'sha256':'b'*64}},sources={'recipe':{'sha256':'c'*64}})
    adapter=m.BoundFiles(config,Session('prepare'))
    assert adapter.zero_pool(spec,cal,full)['zero_pool_dates']==['2022-01-04']
    for name in ('input_version','calendar_sha256','recipe_source_sha256','full_pool_test_keys_sha256'):
        broken={**proof,name:'wrong'};proof_file.write_text(json.dumps(broken))
        spec['zero_pool']['proof']=reference(proof_file)
        review_file.write_text(json.dumps(dict(decision='PASS',proof_ref=spec['zero_pool']['proof'])))
        spec['zero_pool']['accepted_review']=reference(review_file)
        with pytest.raises(m.Blocked,match='VERSION_BOUND'):
            m.BoundFiles(config,Session('prepare')).zero_pool(spec,cal,full)


def test_activation_requires_independent_review_identity(tmp_path,monkeypatch):
    c=fixture_config(tmp_path,monkeypatch);cp=tmp_path/'CONFIG.json'
    a=activation(c,cp,'prepare')
    m.require_activation(c,cp,a,'prepare')
    with pytest.raises(m.Blocked,match='INDEPENDENT_ACTIVATION'):
        m.require_activation(c,cp,{**a,'review_receipt_sha256':None},'prepare')


def test_original_core_review_cannot_authorize_changed_core(tmp_path,monkeypatch):
    c=fixture_config(tmp_path,monkeypatch)
    ref=c['accepted_core_review'];record=json.loads(Path(ref['path']).read_text())
    record['source_sha256']='0'*64;Path(ref['path']).write_text(json.dumps(record))
    c['accepted_core_review']=reference(Path(ref['path']))
    # Stops before any source data adapter is entered.
    with pytest.raises(m.Blocked,match='CORE_PASS'):
        m.prepare_dependencies(c,SimpleNamespace(),MemorySink())


def test_draft_contract_cannot_be_activated_by_renaming_only_the_config_state(tmp_path,monkeypatch):
    c=fixture_config(tmp_path,monkeypatch);cp=tmp_path/'CONFIG.json'
    p=Path(c['contract']['path']);contract=json.loads(p.read_text())
    contract['state']='DRAFT_DEPENDENCIES_UNBOUND';p.write_text(json.dumps(contract));c['contract']=reference(p)
    with pytest.raises(m.Blocked,match='FINAL_FROZEN_CONTRACT'):
        m.require_activation(c,cp,activation(c,cp,'prepare'),'prepare')


@pytest.mark.parametrize('target',['2024-01-01','2024-01-03'])
def test_filtered_csv_training_target_at_or_after_cutoff_stops_before_read_or_label_computation(
        tmp_path,monkeypatch,target):
    signal=pd.Timestamp('2023-12-29')
    panel=pd.DataFrame(dict(date=[signal,pd.Timestamp(target)],code='000001.SZ',close=[10.,15.]))
    path=tmp_path/'SYNTHETIC_PANEL.csv';panel.to_csv(path,index=False)
    cal=pd.DataFrame(dict(date=pd.to_datetime(['2023-12-28','2023-12-29','2023-12-30',target,'2024-01-04']),
                          calendar_index=range(5)))
    adapter=m.BoundFiles(dict(caps=limits(),inputs={'panel':reference(path)}),Session('select'))
    def forbidden(*args,**kwargs):raise AssertionError('No CSV authentication/parsing or label producer call permitted')
    adapter.authenticate=forbidden
    adapter.modules=forbidden
    monkeypatch.setattr(pd,'read_csv',forbidden)
    with pytest.raises(m.Blocked,match='FILTERED_CSV_UNSUPPORTED'):
        adapter.labels([(signal.date(),'000001.SZ')],role='training',cutoff=date(2024,1,1),calendar=cal)


def test_original_parquet_label_filters_apply_cutoff_and_code_before_h2_builder(tmp_path,recipe):
    dates=pd.to_datetime(['2023-12-27','2023-12-28','2023-12-29','2024-01-01','2024-01-03'])
    cal=pd.DataFrame(dict(date=dates,calendar_index=range(len(dates))))
    panel=pd.DataFrame(dict(date=list(dates)*2,code=['000001.SZ']*5+['OTHER']*5,
                            close=[8.,9.,10.,12.,15.,80.,90.,100.,120.,150.]))
    path=tmp_path/'SYNTHETIC_PANEL.parquet';panel.to_parquet(path,index=False)
    adapter=m.BoundFiles(dict(caps=limits(),inputs={'panel':reference(path)}),Session('select'))
    seen=[]
    def builder(p,**kwargs):
        seen.append(p.copy())
        assert p.date.lt(pd.Timestamp('2024-01-01')).all()
        assert set(p.code)=={'000001.SZ'}
        return recipe.build_close_return_labels(p,**kwargs)
    adapter.modules=lambda:(SimpleNamespace(build_close_return_labels=builder),SyntheticPrior)
    result=adapter.labels([(date(2023,12,27),'000001.SZ'),(date(2023,12,28),'000001.SZ'),
                           (date(2023,12,29),'000001.SZ')],role='training',cutoff=date(2024,1,1),calendar=cal)
    assert len(seen)==1
    assert result[0].value==pytest.approx(10/8-1)
    assert result[1].label_end==date(2024,1,1) and result[1].value is None
    assert result[2].label_end==date(2024,1,3) and result[2].value is None
