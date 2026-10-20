"""Synthetic G1 rule fixtures only; no models fitted or real outcomes accessed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np
import pandas as pd
import pytest
from g1_diagnostic import *


def fixture(n=50):
    date=pd.Timestamp('2024-01-02')
    pred=pd.DataFrame({'date':[date]*n,'code':[f'{i:06d}.SZ' for i in range(n)],'prediction':np.arange(n,0,-1)/1000})
    prior=pred[['date','code']].copy();prior[PRIOR]=np.tile(np.arange(10,dtype=float),int(np.ceil(n/10)))[:n];prior[MISSING]=0
    return top50(pred),prior


def test_rank_ties_code_order_and_fifty_exact():
    p,_=fixture();p['prediction']=1
    out=top50(p.sample(frac=1,random_state=1));assert out.code.tolist()==sorted(p.code)
    with pytest.raises(ValueError):top50(p.iloc[:49])


def test_linear_quantiles_sizes_and_five_bands():
    p,prior=fixture();keys,status=groups_before_labels(p,prior)
    assert len(status)==5 and status.prelabel_valid.all()
    np.testing.assert_allclose(status.q30,2.7,rtol=0,atol=1e-14);np.testing.assert_allclose(status.q70,6.3,rtol=0,atol=1e-14)
    assert status.high_count.eq(3).all() and status.low_count.eq(3).all()


def test_ties_kept_whole_and_overlap_invalid():
    p,prior=fixture();prior[PRIOR]=1
    keys,status=groups_before_labels(p,prior)
    assert status.prelabel_reason.eq('HIGH_LOW_OVERLAP').all()
    assert keys.high.all() and keys.low.all()


def test_missing_prior_never_low_group_and_known_minimum():
    p,prior=fixture();prior.loc[:4,MISSING]=1;prior.loc[:4,PRIOR]=0
    keys,status=groups_before_labels(p,prior)
    assert status.iloc[0].known_prior_count==5 and status.iloc[0].prelabel_reason=='KNOWN_PRIOR_LT6'
    assert not keys.iloc[:5].low.any()


def test_selected_h2_missing_invalidates_whole_band():
    p,prior=fixture();keys,status=groups_before_labels(p,prior)
    labels=p[['date','code']].copy();labels['label_return']=p.prediction+.001*prior[PRIOR]
    labels.loc[0,'label_return']=np.nan
    dates=pd.DataFrame({'date':[p.date.iloc[0]],'calendar_index':[0],'fold':['2024H1']})
    rows,bands,daily=evaluate_bands(keys,status,labels,dates)
    assert not bands.iloc[0].valid and bands.iloc[0].invalid_reason=='MISSING_SELECTED_H2'
    assert not daily.valid.iloc[0]


def test_unselected_h2_missing_does_not_refill_or_change_groups():
    p,prior=fixture();keys,status=groups_before_labels(p,prior)
    labels=p[['date','code']].copy();labels['label_return']=p.prediction+.001*prior[PRIOR]
    labels.loc[4,'label_return']=np.nan
    dates=pd.DataFrame({'date':[p.date.iloc[0]],'calendar_index':[0],'fold':['2024H1']})
    rows,bands,daily=evaluate_bands(keys,status,labels,dates)
    assert daily.valid.iloc[0] and daily.residual_spread.iloc[0]==pytest.approx(.007)
    assert np.max(bands.reconciliation_error)<1e-12


def daily_fixture():
    rows=[];i=0
    for f in COMPLETE:
        for date in pd.bdate_range(f'{f[:4]}-01-01' if f.endswith('1') else f'{f[:4]}-07-01',periods=30):
            rows.append(dict(date=date,fold=f,calendar_index=i,valid=True,residual_spread=.0008,raw_h2_spread=.001,M0_score_gap=.0002));i+=1
    return pd.DataFrame(rows)


def test_bootstrap_preserves_real_calendar_gaps():
    daily=daily_fixture();daily.loc[daily.calendar_index%5==0,'valid']=False
    result=bootstrap_daily(daily)
    assert result['ci95'] is None and result['valid_blocks']==0


def test_bootstrap_no_fold_crossing_and_deterministic():
    daily=daily_fixture();a=bootstrap_daily(daily,reps=20);b=bootstrap_daily(daily,reps=20)
    assert a==b and a['valid_blocks']==55 and all(x==11 for x in a['blocks_by_fold'].values())


def test_only_one_legal_start_is_technical_cc():
    d=daily_fixture().iloc[:20];r=bootstrap_daily(d);assert r['valid_blocks']==1 and r['ci95'] is None


def test_denominator_remains_original_calendar_and_coverage_first():
    daily=daily_fixture();daily.loc[daily.index%2==0,'valid']=False
    bands=pd.DataFrame([dict(date=r.date,band=b,valid=r.valid,invalid_reason='OK' if r.valid else 'HIGH_LOW_OVERLAP',residual_spread=.0008) for r in daily.itertuples() for b in range(1,6)])
    r=summarize(daily,bands)
    assert r['decision']=='CANNOT_CONFIRM_COVERAGE' and r['overall_coverage']['denominator_dates']==150


def test_information_threshold_five_bp_not_ten_bp():
    daily=daily_fixture();bands=pd.DataFrame([dict(date=r.date,band=b,valid=True,invalid_reason='OK',residual_spread=.0008) for r in daily.itertuples() for b in range(1,6)])
    r=summarize(daily,bands);assert r['decision']=='INFORMATION_GATE_PASS_DEVELOPMENT_ONLY'
    daily.residual_spread=.0001;daily.raw_h2_spread=.0003;r=summarize(daily,bands)
    assert r['decision']=='BELOW_PRACTICAL_INFORMATION_THRESHOLD'
