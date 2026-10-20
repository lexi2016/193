import numpy as np
import pandas as pd
from g1_ci_weight import build_support,corrected_ci,support_status


def frame(f,n,start=0,values=None):
    return pd.DataFrame(dict(fold=f,date=[f'{f}-{i}' for i in range(n)],calendar_index=np.arange(start,start+n),valid=True,residual_spread=np.full(n,2.) if values is None else values))


def test_extreme_constant_folds_fixed_day_weights_zero_width():
    data=pd.concat([frame('a',25,values=np.full(25,-20.)),frame('b',41,100,np.full(41,100.))]);support,_=build_support(data,('a','b'),{'a':25,'b':41})
    point=(25*(-20.)+41*100.)/66;result,draws=corrected_ci(support,{'a':-20.,'b':100.},point,folds=('a','b'),reps=17)
    assert np.ptp(draws)==0;np.testing.assert_allclose(draws,np.full(17,point),atol=1e-13);np.testing.assert_allclose(result['ci95'],[point,point],atol=1e-13)


def test_within_fold_constant_translation_leaves_errors_and_ci_width():
    data=pd.concat([frame('a',25,values=np.sin(np.arange(25))),frame('b',41,100,np.cos(np.arange(41)))])
    means={f:float(data.loc[data.fold==f,'residual_spread'].mean()) for f in ('a','b')};point=(25*means['a']+41*means['b'])/66
    s,_=build_support(data,('a','b'),{'a':25,'b':41});r,d=corrected_ci(s,means,point,folds=('a','b'),reps=101)
    shifted=data.copy();shifted.loc[shifted.fold=='a','residual_spread']+=100.;shifted.loc[shifted.fold=='b','residual_spread']-=5.
    sm={'a':means['a']+100.,'b':means['b']-5.};sp=point+(25*100.-41*5.)/66;t,_=build_support(shifted,('a','b'),{'a':25,'b':41});rr,dd=corrected_ci(t,sm,sp,folds=('a','b'),reps=101)
    np.testing.assert_allclose(dd-dd.mean(),d-d.mean(),atol=5e-14,rtol=1e-12)
    np.testing.assert_allclose(np.diff(rr['ci95']),np.diff(r['ci95']),atol=5e-14)


def test_inverse_coverage_expected_weight_identity_endpoints():
    data=frame('a',27,values=np.arange(27,dtype=float)**2);s,_=build_support(data,('a',),{'a':27});a=s['a'];c=a['coverage'];assert c[0]==1 and c[13]>1
    expected=np.zeros(27)
    for block in a['blocks']:expected[block]+=1/c[block]
    np.testing.assert_allclose(expected,np.ones(27),atol=1e-14)
    mu=float(data.residual_spread.mean());z=data.residual_spread.to_numpy()-mu
    expected_error=sum(np.sum(z[block]/c[block]) for block in a['blocks'])/a['n']
    np.testing.assert_allclose(expected_error,0,atol=1e-13)


def test_uncovered_date_cc_no_drop_no_resampling():
    data=frame('a',25);data.loc[24,'calendar_index']=100;s,rows=build_support(data,('a',),{'a':25});assert len(rows)==25 and rows.legal_block_coverage.iloc[-1]==0
    result,draws=corrected_ci(s,{'a':2.},2.,folds=('a',));assert result['status']=='CANNOT_CONFIRM_BLOCK_COVERAGE' and result['ci95'] is None and draws is None


def test_one_legal_start_is_only_technical_cc():
    s,_=build_support(frame('a',20),('a',),{'a':20});assert support_status(s)=='CANNOT_CONFIRM_INSUFFICIENT_BLOCKS'


def test_gap_and_fold_partial_never_join():
    a=frame('a',40);a.loc[20:,'calendar_index']+=5
    b=frame('b',21,100);partial=frame('partial',30,200)
    s,rows=build_support(pd.concat([a,b,partial]),('a','b'),{'a':40,'b':21})
    assert s['a']['K']==2 and s['b']['K']==2 and set(rows.fold)=={'a','b'}
    for fold,q in s.items():
        for block in q['blocks']:assert len(block)==20 and (np.diff(q['frame'].calendar_index.iloc[block])==1).all()


def test_centered_error_linear_percentiles_around_fixed_point():
    data=frame('a',30,values=np.arange(30,dtype=float));s,_=build_support(data,('a',),{'a':30});result,draws=corrected_ci(s,{'a':14.5},14.5,folds=('a',),reps=31,seed=1)
    np.testing.assert_array_equal(result['ci95'],14.5+np.quantile(draws-draws.mean(),[.025,.975],method='linear'))
