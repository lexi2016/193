"""Synthetic original-producer adapter tests; no real payload/model/registry IO."""
import ast
import copy
from datetime import date
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import environment_tilt_early_inputs as m

ROOT=Path('/Users/lexi/Desktop/学习/因子工厂')
ORIGINAL=ROOT/'workspace/.task_worktrees/s1-industry-month-20261020/research/a_share_model_selection_v1/frozen_m0/research/a_share_model_selection_v1'
V193=ROOT/'research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/inputs/V193_ORIG_SNAPSHOT_UTF8.txt'
V192=ROOT/'research/a_v192_signal_pool_return_stability_v1/intake/unpacked/base_bundle/formula/V192ORIG_UTF8.txt'
SHA='a'*64
MAP={k:k for k in ('date','code','open','high','low','close','volume','amount')}


@pytest.fixture(scope='module')
def recipe():return m.support._load(ORIGINAL/'model_selection.py','m26_synthetic_original_recipe')


def pure_source_function(path,name,extra=None):
    # Execute only this original parser definition, never engine/C import/main.
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    scope={'re':re,**(extra or {})}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


@pytest.fixture(scope='module')
def parser():
    def_re=re.compile(r'^\s*([A-Za-z_]\w*)\s*(:=|:)\s*(.*?)\s*;\s*$')
    parse=pure_source_function(ORIGINAL/'formula_adapter.py','parse_definitions',{'DEF_RE':def_re})
    convert=pure_source_function(ORIGINAL/'tdx_ast_engine_injected.py','tdx_expr_to_py')
    return SimpleNamespace(parse_definitions=parse,CORE_OUTPUTS=m.FEATURES),SimpleNamespace(tdx_expr_to_py=convert)


@pytest.fixture
def prices():
    dates=pd.bdate_range('2019-08-01','2020-06-30')
    keys=pd.MultiIndex.from_product([dates,['000001.SZ','600000.SH']],names=['date','code']).to_frame(index=False)
    x=np.arange(len(keys))*.01+10
    panel=keys.assign(open=x,high=x+1,low=x-1,close=x,volume=100.,amount=200000.)
    prefix=panel.loc[panel.date<'2020-03-02'].copy().reset_index(drop=True)
    old=panel.loc[panel.date>='2020-03-02'].copy().reset_index(drop=True);old['amount']/=10000
    proof=dict(status='CERTIFIED_FULL_AVAILABLE_PRE_OLD_HISTORY',prefix_amount_unit='CNY',
        old_amount_unit='10000_CNY',volume_unit='SHARES',by_code={})
    for code,g in prefix.groupby('code'):
        proof['by_code'][code]=dict(old_start='2020-03-02',prefix_rows=len(g),
            prefix_dates_sha256=m.support.digest(m._date_strings(g.date)),available_start='2019-08-01',
            earlier_source_history_omitted=False,source_evidence_sha256=SHA)
    cal=pd.DataFrame(dict(date=sorted(old.date.unique()),calendar_index=range(old.date.nunique())))
    return prefix,old,proof,cal


def compose(recipe,prices):
    p,o,proof,cal=prices
    combined,old,receipt=m.compose_history(recipe,p,o,prefix_map=MAP,old_map=MAP,history_proof=proof,max_rows=10000)
    return combined,old,receipt,cal


def test_two_period_currency_bridge_old_raw_immutability_and_full_history(recipe,prices):
    p,o,_,_=prices;beforep=p.copy(deep=True);beforeo=o.copy(deep=True)
    full,old,receipt,_=compose(recipe,prices)
    assert full.amount.eq(200000.).all() and full.volume.eq(100.).all()
    assert full.date.min()==pd.Timestamp('2019-08-01')
    assert receipt['old_rows_retained_exactly'] and not receipt['old_raw_mutated']
    pd.testing.assert_frame_equal(p,beforep,check_exact=True);pd.testing.assert_frame_equal(o,beforeo,check_exact=True)
    pd.testing.assert_frame_equal(full.loc[full.date>='2020-03-02'].reset_index(drop=True),old,check_exact=True)


@pytest.mark.parametrize('kind',['draft','units','truncated_history','missing_code','overlap','new_code','row_limit'])
def test_history_conflicts_reject_no_fill_or_source_rewrite(kind,recipe,prices):
    p,o,proof,_=copy.deepcopy(prices);maxrows=10000
    if kind=='draft':proof['status']='CANARY_NOT_ACCEPTED'
    elif kind=='units':proof['prefix_amount_unit']='10000_CNY'
    elif kind=='truncated_history':p=p.loc[p.date>='2020-01-01']
    elif kind=='missing_code':proof['by_code'].pop('000001.SZ')
    elif kind=='overlap':p.loc[0,'date']=pd.Timestamp('2020-03-02')
    elif kind=='new_code':p.loc[0,'code']='NEW.SZ'
    else:maxrows=1
    with pytest.raises((m.Blocked,recipe.ContractError)):
        m.compose_history(recipe,p,o,prefix_map=MAP,old_map=MAP,history_proof=proof,max_rows=maxrows)


def test_calendar_equivalent_union_constant_shift_and_original_lag_mapping(recipe,prices):
    full,old,_,oldcal=compose(recipe,prices)
    # Preserve an old strategy/calendar-only final extra session.
    oldcal=pd.concat([oldcal,pd.DataFrame(dict(date=[pd.Timestamp('2020-07-01')],calendar_index=[len(oldcal)]))],ignore_index=True)
    cal,proof=m.extend_calendar(recipe,full,old,oldcal)
    assert proof['old_index_constant_shift']>0 and proof['old_overlap_sequence_exact']
    assert cal.date.max()==pd.Timestamp('2020-07-01')
    lag_new=recipe.session_lag(full,cal,'close',5)
    lag_old=recipe.session_lag(old,oldcal,'close',5)
    n=full.assign(lag=lag_new).merge(old[['date','code']],on=['date','code'])
    mask=old.date.isin(oldcal.date.iloc[5:])
    np.testing.assert_array_equal(n.loc[mask,'lag'],lag_old.loc[mask])


def test_new_observation_inside_old_calendar_span_is_not_a_constant_shift(recipe,prices):
    full,old,_,oldcal=compose(recipe,prices)
    missing=pd.Timestamp('2020-03-10')
    old=old.loc[old.date!=missing]
    oldcal=oldcal.loc[oldcal.date!=missing].reset_index(drop=True);oldcal['calendar_index']=np.arange(len(oldcal))
    with pytest.raises(m.Blocked,match='CALENDAR_SEQUENCE_DRIFT'):
        m.extend_calendar(recipe,full,old,oldcal)


def test_lookback_is_derived_from_frozen_formula_and_original_arm_source(recipe,parser):
    formula,engine=parser
    proof=m.derive_index_lookback(recipe,formula,engine,V192.read_text(),V193.read_text())
    assert proof['calendar_previous_sessions']==19
    assert proof['per_stock_previous_rows']==19
    assert proof['v193_index_dependent_outputs']=={'XG':19}
    assert not proof['v192_index_dependent_outputs'] and not proof['index_filled']
    assert {r['window'] for r in proof['source_m1_uses']}=={1,5,20}


@pytest.mark.parametrize('expression',['EMA(INDEXC,20)','MA(INDEXC,N)','SUM(INDEXC,0)','REF(INDEXC,-1)'])
def test_no_guessed_finite_index_lookback_for_unbounded_dynamic_or_future_dependency(expression,recipe,parser):
    f,e=parser
    with pytest.raises(m.Blocked):m.derive_index_lookback(recipe,f,e,'XG:1;','XG:'+expression+';')


class Formula:
    """Synthetic recursive sentinel, only the wrapper plumbing is under test."""
    CORE_OUTPUTS=m.FEATURES
    def __init__(self,parser):self.parse_definitions=parser.parse_definitions;self.seen=[]
    def formula_feature_frame(self,frame,**kwargs):
        self.seen.append((frame.copy(),kwargs['index_close_by_date'].copy()))
        f=frame.sort_values(['code','date']).reset_index(drop=True)
        out=f[['date','code']].copy()
        recursive=f.groupby('code',sort=False).close.transform(lambda s:s.ewm(alpha=.2,adjust=False).mean())
        for name in m.FEATURES:out[name]=recursive
        out['AUDITM']=0.;out['V192_XG']=1.;out['V193_XG']=1.
        return out


class Budget:
    def __init__(self):self.stages=[]
    def check(self,stage):self.stages.append(stage)


def candidate(recipe,parser,prices,*,workers=2,gap=False,bad=False):
    f,e=parser;formula=Formula(f);full,old,_,oldcal=compose(recipe,prices)
    if bad:
        full.loc[(full.code=='000001.SZ')&(full.date==pd.Timestamp('2019-08-02')),['open','high','low','close']]=0.
    index=pd.Series(3000.+np.arange(full.date.nunique()),index=pd.DatetimeIndex(sorted(full.date.unique())))
    # Pre-index stock warmup is retained but not fabricated into index data.
    index=index.loc[index.index>='2019-09-02']
    if gap:index=index.drop(pd.Timestamp('2019-12-20'))
    def provider(required):
        assert pd.Timestamp('2019-08-01') not in required
        return index
    budget=Budget()
    result=m.materialize_candidate(recipe,formula,e,panel=full,old_panel=old,old_calendar=oldcal,
        index_provider=provider,industry=pd.DataFrame(dict(code=['000001.SZ','600000.SH'],industry_code=['A','B'])),
        texts={'v192':'XG:1;','v193':'XG:MA(INDEXC,20);'},
        output_start='2020-01-01',output_end='2020-06-30',
        caps=dict(max_input_rows=10000,formula_workers=workers,max_formula_batch_rows=300),budget=budget)
    return result,formula,budget


def test_full_recursive_stock_warmup_unfilled_early_index_parallel_parity_and_late_projection(recipe,parser,prices):
    result,formula,budget=candidate(recipe,parser,prices,workers=2)
    serial,_,_=candidate(recipe,parser,prices,workers=1)
    pd.testing.assert_frame_equal(result['m0'],serial['m0'],check_exact=True)
    assert result['early_m0'].date.min()>=pd.Timestamp('2020-01-01')
    assert min(frame.date.min() for frame,index in formula.seen)==pd.Timestamp('2019-08-01')
    assert all(index.index.min()==pd.Timestamp('2019-09-02') for frame,index in formula.seen)
    assert result['provenance']['projection_after_formula_arms_taint_and_common']
    assert budget.stages[-1]=='after_original_common_pool'
    # Removing genuine early recursive history changes later synthetic EMA.
    full=pd.concat([g for g,_ in formula.seen],ignore_index=True)
    truncated=full.loc[full.date>='2020-01-01']
    ff=formula.formula_feature_frame(truncated,index_close_by_date=formula.seen[0][1])
    code='600000.SH';t=pd.Timestamp('2020-01-01')
    assert ff.loc[(ff.code==code)&(ff.date==t),'DISTW'].iloc[0]!=result['early_m0'].loc[(result['early_m0'].code==code)&(result['early_m0'].date==t),'DISTW'].iloc[0]


def test_actual_required_index_warmup_gap_rejects_without_filling(recipe,parser,prices):
    with pytest.raises(m.Blocked,match='INDEX_WARMUP_GAP'):candidate(recipe,parser,prices,gap=True)


def test_bad_ohlc_before_output_and_before_index_still_taints_original_recursive_suffix(recipe,parser,prices):
    result,formula,_=candidate(recipe,parser,prices,bad=True)
    q=result['provenance']['quarantine']['quarantined_keys']
    assert q[0]['date']=='2019-08-02'
    assert set(result['early_m0'].code)=={'600000.SH'}
    assert result['calendar'].date.min()==pd.Timestamp('2019-08-01')
    taint=result['provenance']['taint']['coverage_loss_by_arm']['M0']
    assert taint['tainted_key_count']>len(result['early_m0'])


def test_oversized_whole_stock_history_stops_instead_of_truncation(recipe,parser,prices):
    formula=Formula(parser[0]);full,_,_,_=compose(recipe,prices)
    with pytest.raises(m.Blocked,match='NO_TRUNCATION'):
        m._formula_batches(formula,full,{'v192':'','v193':''},pd.Series(dtype=float),workers=2,max_batch_rows=1)


def model_fixture(recipe):
    dates=pd.bdate_range('2021-06-01','2024-01-03')
    panel=pd.MultiIndex.from_product([dates,['000001.SZ','600000.SH']],names=['date','code']).to_frame(index=False)
    panel['close']=10.+np.arange(len(panel))*.01
    cal=pd.DataFrame(dict(date=dates,calendar_index=range(len(dates))))
    f=panel[['date','code']].copy()
    for i,k in enumerate(m.FEATURES):f[k]=np.arange(len(f))*.01+i
    f=f.loc[f.date<'2024-01-01'].reset_index(drop=True)
    fold=recipe.Fold('2023H2',pd.Timestamp('2023-07-01'),pd.Timestamp('2023-12-31'),False)
    labels=recipe.build_close_return_labels(panel,horizons=(2,),calendar=cal,
        signal_start='2021-07-01',signal_end_exclusive='2023-07-01')[2]
    identity=recipe.prepare_cell_data(features=f,labels=labels,fold=fold,canary=False)['identity']
    side=dict(status='COMPLETED',identity=dict(arm='M0',horizon=2,fold='2023H2',data_identity=identity,
        code_sha256={'model_selection':m.support.sha(recipe.__file__)}))
    return f,panel,cal,{'SYNTHETIC_MODEL':side}


def test_overlap_and_each_model_classified_separately_no_auto_refit(recipe):
    f,p,c,bindings=model_fixture(recipe)
    passed=m.parity_gate(recipe,f,f,p,c,bindings,['SYNTHETIC_MODEL'])
    assert passed['status']=='INPUT_VERSION_PARITY_PASS'
    changed=f.copy();changed.loc[0,m.FEATURES[0]]+=1
    early=m.parity_gate(recipe,changed,f,p,c,bindings,['SYNTHETIC_MODEL'])
    assert early['status']=='INPUT_VERSION_PARITY_FAILED'
    assert early['model_cells']['SYNTHETIC_MODEL']['status']=='UNCHANGED' and early['affected_cell_ids']==[]
    # A changed actual training row affects this particular existing cell.
    changed=f.copy();changed.loc[changed.date==pd.Timestamp('2022-06-01'),m.FEATURES[0]]+=1
    bad=m.parity_gate(recipe,changed,f,p,c,bindings,['SYNTHETIC_MODEL'])
    assert bad['affected_cell_ids']==['SYNTHETIC_MODEL'] and not bad['publish_candidate_passport_allowed']
    assert not bad['automatic_refit'] and not bad['input_versions_mixed'] and bad['old_V1_preserved']


def test_key_order_and_float64_bit_parity_not_tolerance_or_calendar_shift(recipe):
    f,_,_,_=model_fixture(recipe)
    assert m.overlap_parity(recipe,f.iloc[::-1],f)['status']=='DIFFERENT'
    other=f.copy();other.loc[0,m.FEATURES[0]]=np.nextafter(other.loc[0,m.FEATURES[0]],np.inf)
    assert m.overlap_parity(recipe,other,f)['status']=='DIFFERENT'


def test_missing_model_manifest_cell_stops_not_unaffected_inference(recipe):
    f,p,c,b=model_fixture(recipe)
    with pytest.raises(m.Blocked,match='MANIFEST'):
        m.parity_gate(recipe,f,f,p,c,b,['SYNTHETIC_MODEL','MISSING'])


def config(tmp_path):
    stub=tmp_path/'SYNTHETIC.txt';stub.write_text('synthetic source only')
    ref=dict(path=str(stub),sha256=m.support.sha(stub))
    return dict(state='FROZEN_READY_FOR_ACTIVATION',policy=copy.deepcopy(m.POLICY),
        sources={k:ref for k in ('recipe','formula','engine','fast_builder','s1_runner','v192','v193','loader','support','resource_probe')},
        inputs={k:ref for k in ('prefix','old_qfq','old_calendar','index','industry','old_m0','model_manifest','history_proof','accepted_prefix_review','source_plan')},
        accepted_prefix_status='INDEPENDENT_ACCEPTED_FULL_HISTORY_CANARY',expected_cell_ids=['SYNTHETIC'],
        output_root=str(m.BATCH/'M26_SYNTHETIC_NOT_CREATED'),
        caps=dict(max_input_rows=10000,max_formula_batch_rows=300,formula_workers=2,threads=4,
            rss_bytes=2**35,min_available_bytes=2**33,min_disk_bytes=2**32,wall_seconds=600),**m.FLAGS)


def activation(c):return dict(decision='INDEPENDENT_REVIEW_PASS_ACTIVATE',
    scientific_identity=m.support.digest({k:c[k] for k in ('policy','sources','inputs','caps','expected_cell_ids')}),
    wrapper_sha256=m.support.sha(m.__file__),review_receipt_sha256=SHA,execution_id='SYNTHETIC_M26')


@pytest.mark.parametrize('kind',['draft','null_prefix','no_m19','bad_source','workers','duplicate_cell'])
def test_draft_dependency_or_acceptance_gaps_cannot_execute(kind,tmp_path):
    c=config(tmp_path)
    if kind=='draft':c['state']='DRAFT'
    elif kind=='null_prefix':c['inputs']['prefix']=None
    elif kind=='no_m19':c['accepted_prefix_status']='PENDING_5576'
    elif kind=='bad_source':c['sources']['formula']['sha256']='notasha'
    elif kind=='workers':c['caps']['formula_workers']=5
    else:c['expected_cell_ids']=['SYNTHETIC','SYNTHETIC']
    with pytest.raises(m.Blocked):m.require_activation(c,activation(c))


@pytest.mark.parametrize('kind',['not_registered','wrong_execution','output_exists'])
def test_recovery_guard_never_reopens_loader_or_retries_existing_output(kind,tmp_path):
    c=config(tmp_path);calls=[]
    class Session:
        execution_id='WRONG' if kind=='wrong_execution' else 'SYNTHETIC_M26'
        def require_active(self):
            if kind=='not_registered':raise m.Blocked('REGISTERED_START_REQUIRED')
        def output_exists(self):return kind=='output_exists'
    loader=SimpleNamespace(open_once=lambda *args:calls.append(1))
    with pytest.raises(m.Blocked):m.prepare_registered(c,activation(c),Session(),loader)
    assert calls==[]


def test_wrong_model_source_is_unproven_not_an_affected_cell_or_retraining_trigger(recipe):
    f,p,c,b=model_fixture(recipe)
    b['SYNTHETIC_MODEL']['identity']['code_sha256']['model_selection']='0'*64
    with pytest.raises(m.Blocked,match='MODEL_IDENTITY'):
        m.parity_gate(recipe,f,f,p,c,b,['SYNTHETIC_MODEL'])


def test_bound_executor_loader_source_drift_stops_before_any_open(tmp_path):
    c=config(tmp_path);calls=[]
    class Session:
        execution_id='SYNTHETIC_M26'
        def require_active(self):pass
        def output_exists(self):return False
    class Loader:
        def open_once(self,*args):calls.append(1)
    with pytest.raises(m.Blocked,match='LOADER_SOURCE_DRIFT'):
        m.prepare_registered(c,activation(c),Session(),Loader())
    assert calls==[]


def test_m40_ratio1_bridge_preserves_old_numeric_and_same_original_recipe(recipe,prices):
    p,o,proof,_=copy.deepcopy(prices)
    # Existing old numeric is CNY at the original collector scale, left unchanged.
    o['amount']*=10000.
    proof.update(old_amount_unit='ORIGINAL_NUMERIC_CNY_UNCHANGED',amount_bridge=copy.deepcopy(m.M40_AMOUNT_BRIDGE))
    beforep,beforeo=p.copy(deep=True),o.copy(deep=True)
    full,old,r=m.compose_history(recipe,p,o,prefix_map=MAP,old_map=MAP,history_proof=proof,max_rows=10000)
    assert full.amount.eq(200000.*10000.).all()
    assert r['prefix_amount_multiplier']==r['old_amount_multiplier']==10000.
    assert r['normalized_amount_semantics']=='ORIGINAL_MODEL_NUMERIC_SCALE_NOT_CANONICAL_CNY'
    pd.testing.assert_frame_equal(p,beforep,check_exact=True)
    pd.testing.assert_frame_equal(o,beforeo,check_exact=True)
    pd.testing.assert_series_equal(old.amount,beforeo.amount*10000.,check_exact=True)


@pytest.mark.parametrize('field,value',[('raw_CNY_to_old_numeric',.0001),('old_values_preserved',False),('controller_ruling_sha256','0'*64)])
def test_m40_bridge_rejects_unbound_scale_or_rewrite(field,value,recipe,prices):
    p,o,proof,_=copy.deepcopy(prices)
    proof.update(old_amount_unit='ORIGINAL_NUMERIC_CNY_UNCHANGED',amount_bridge=copy.deepcopy(m.M40_AMOUNT_BRIDGE))
    proof['amount_bridge'][field]=value
    with pytest.raises(m.Blocked,match='EXACT_M40_AMOUNT_BRIDGE_REQUIRED'):
        m.compose_history(recipe,p,o,prefix_map=MAP,old_map=MAP,history_proof=proof,max_rows=10000)
