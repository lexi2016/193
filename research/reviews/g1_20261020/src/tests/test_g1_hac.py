import numpy as np
import pytest
from g1_hac_core import mask_matrices,variance,lag_quadratic,real_ci,calibrate,information_gate,select_primary_mask,calibration_gate,SCENARIOS,SEEDS


def example():
    idx=np.r_[np.arange(30),np.arange(31,62),np.arange(63,95),np.arange(97,130),np.arange(132,166)]
    codes=np.repeat(np.arange(5),[30,31,32,33,34]);return mask_matrices(idx,codes)


def test_constant_fold_zero_width_and_translation_invariance():
    m=example();means=np.array([0.,1.,-1.,2.,-2.]);x=means[m['fold_codes']];r=real_ci(x,means,float(x.mean()),m);assert r['variance']==0 and r['ci95'][0]==r['ci95'][1]
    e=np.sin(m['indices']);r=real_ci(x+e,means,0.,m);s=np.array([100.,-20.,5.,9.,-7.]);rr=real_ci(x+e+s[m['fold_codes']],means+s,3.,m);np.testing.assert_allclose(r['variance'],rr['variance'],rtol=1e-13,atol=1e-14)


def test_unit_diagonal_full_mask_and_trace_formula():
    m=example();np.testing.assert_array_equal(np.diag(m['K']),np.ones(m['N']));assert m['N']==160
    expected=m['N']-sum(m['K'][np.ix_(m['fold_codes']==f,m['fold_codes']==f)].sum()/m['counts'][f] for f in range(5));np.testing.assert_allclose(m['D'],expected,atol=1e-12);assert m['c']!=m['N']/(m['N']-5)


def test_gap_not_compressed_and_crossfold_covariance_uses_real_lag():
    m=example();i=29;j=30;assert m['fold_codes'][i]!=m['fold_codes'][j];assert m['indices'][j]-m['indices'][i]==2;assert m['K'][i,j]==.9
    assert m['K'][29,31]!=1-2/20  # true lag3, not row distance2


def test_matrix_matches_original_calendar_lag_quadratic():
    m=example();z=np.cos(m['indices']);np.testing.assert_allclose(z@m['K']@z,lag_quadratic(z,m['indices']),rtol=1e-12,atol=1e-12)


def test_chunk_and_seed_parity_synthetic():
    m=example();a=calibrate(m,'ma5',reps=31,chunk_size=7);b=calibrate(m,'ma5',reps=31,chunk_size=13);assert a['covered']==b['covered'];np.testing.assert_allclose(a['mean_width'],b['mean_width'],rtol=1e-14)


def test_mask_invalid_failclosed():
    with pytest.raises(ValueError):mask_matrices([1,1,2,3,4],[0,1,2,3,4])
    with pytest.raises(ValueError):mask_matrices(np.arange(10),np.zeros(10,dtype=int))


def test_information_gate_preserves_threshold_and_coverage():
    o=dict(overall_coverage={'passed':True},coverage_by_fold={f:{'coverage':.8} for f in ('2024H1','2024H2','2025H1','2025H2','2026H1')},primary_point=.002,complete_fold_positive_count=5)
    assert information_gate({'ci95':[.001,.004]},o)=='INFORMATION_GATE_PASS_DEVELOPMENT_ONLY';assert information_gate({'ci95':[-.001,.004]},o)=='INCONCLUSIVE';o['overall_coverage']['passed']=False;assert information_gate({'ci95':[.001,.004]},o)=='CANNOT_CONFIRM_COVERAGE'


def test_all547_unit_diagonal_partial_excluded_and_identity_failclosed():
    import pandas as pd
    counts=[110,123,106,112,96];den=[117,125,117,126,116];folds=['2024H1','2024H2','2025H1','2025H2','2026H1'];records=[];i=0
    for f,n,d in zip(folds,counts,den):
        for j in range(d):records.append(dict(date=str(i),calendar_index=i+(3 if i>250 else 0),fold=f,valid=j<n));i+=1
    records.append(dict(date='partial',calendar_index=1000,fold='2026H2_PARTIAL_TO_2026_10_15',valid=True));frame=pd.DataFrame(records);primary,selected=select_primary_mask(frame);assert len(primary)==601 and len(selected)==547 and not selected.fold.str.contains('PARTIAL').any()
    codes=np.array([folds.index(f) for f in selected.fold]);m=mask_matrices(selected.calendar_index,codes);np.testing.assert_array_equal(np.diag(m['K']),np.ones(547))
    with pytest.raises(ValueError):select_primary_mask(frame.assign(residual_spread=0.))
    bad=frame.copy();bad.loc[0,'valid']=False
    with pytest.raises(ValueError):select_primary_mask(bad)


def test_calibration_gate_failclosed_on_missing_failed_or_wrong_seed():
    results={s:dict(scenario=s,seed=SEEDS[s],reps=10000,wilson95=[.94,.96],pass_calibration=True) for s in SCENARIOS};assert calibration_gate(results)
    results['ma5']['wilson95'][0]=.929;assert not calibration_gate(results);results['ma5']['wilson95'][0]=.94;results['iid']['seed']+=1;assert not calibration_gate(results)
