import json
import numpy as np
import pandas as pd
import pytest
from g1_qstar_core import (HAC,FOLDS,COUNTS,Z95,build_mask,rng_for,error_draws,simulate_segment,critical_index,order_bound,validation_gate,fixed_real_ci)
from g1_qstar_state import save_checkpoint,load_checkpoint


def mask():
    records=[];index=648
    for f,n,d in zip(FOLDS,COUNTS,[117,125,117,126,116]):
        for j in range(d):records.append(dict(date=str(index),calendar_index=index,fold=f,valid=j<n));index+=1
    return build_mask(pd.DataFrame(records))[2]


def test_seedsequence_phase_scenario_fixed_and_disjoint():
    a=rng_for(0,6);b=np.random.Generator(np.random.PCG64(np.random.SeedSequence(19320261021,spawn_key=(0,6))))
    np.testing.assert_array_equal(a.standard_normal(100),b.standard_normal(100));assert not np.array_equal(rng_for(0,1).standard_normal(10),rng_for(1,1).standard_normal(10))


def test_order_stat_rank_binomial_minimal_and_no_interpolation():
    r=critical_index();assert r['cdf_at_k_minus_2']<r['required_cdf']<=r['cdf_at_k_minus_1'];assert order_bound([8.,1.,3.,4.],3)==4.
    with pytest.raises(ValueError):order_bound([1.,np.inf],1)
    with pytest.raises(ValueError):critical_index(n=1)


def test_ar_burnin_exact_recurrence_and_t5_scale():
    m=mask();rows=2;L=m['full_length'];rng=rng_for(0,6);u=rng.standard_t(5,size=(rows,L+512))*np.sqrt(3./5.);manual=np.zeros_like(u);previous=np.zeros(rows)
    for i in range(L+512):manual[:,i]=.9*previous+np.sqrt(1-.9**2)*u[:,i];previous=manual[:,i]
    np.testing.assert_allclose(error_draws(rng_for(0,6),rows,m,6),manual[:,512:],rtol=1e-14,atol=1e-14)


def test_sigma_full_calendar_and_reverse_fold_pattern():
    m=mask();base=error_draws(rng_for(0,7),2,m,3);expected=base*np.array([2.,1.,2.,1.,2.])[m['full_fold_codes']];np.testing.assert_array_equal(error_draws(rng_for(0,7),2,m,7),expected)
    base=error_draws(rng_for(1,8),2,m,6);expected=base*(1+.75*np.sin(2*np.pi*np.arange(601)/600));np.testing.assert_array_equal(error_draws(rng_for(1,8),2,m,8),expected)


@pytest.mark.parametrize('scenario',[1,3,6,8])
def test_chunk_parity_fixed_math_tiles(scenario):
    m=mask();a,se=simulate_segment(rng_for(0,scenario),96,m,scenario,32);b,sb=simulate_segment(rng_for(0,scenario),96,m,scenario,64);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(se,sb)


def test_atomic_recovery_rng_and_all_arrays_exact(tmp_path):
    m=mask();rng=rng_for(1,8);a,sa=simulate_segment(rng,64,m,8,64);identity=dict(phase=1,scenario=8,total=96,science='frozen',lock_sha='fixedq');save_checkpoint(tmp_path,a,sa,rng,identity,'testcapture');state,y,se=load_checkpoint(tmp_path,identity);restored=rng_for(1,8);restored.bit_generator.state=state['rng_state'];b,sb=simulate_segment(restored,32,m,8,32);whole,sw=simulate_segment(rng_for(1,8),96,m,8,64);np.testing.assert_array_equal(np.r_[y,b],whole);np.testing.assert_array_equal(np.r_[se,sb],sw)
    with pytest.raises(ValueError):load_checkpoint(tmp_path,{**identity,'lock_sha':'tampered'})
    arrays=state['arrays_path'];p=__import__('pathlib').Path(arrays);p.write_bytes(p.read_bytes()+b'corruption')
    with pytest.raises(ValueError):load_checkpoint(tmp_path,identity)


def test_locked_q_validation_gate_failclosed():
    result={s:dict(phase=1,scenario_id=s,reps=10000,pass_validation=True,wilson95=[.94,.96],qstar_lock_sha256='lock',qstar=4.) for s in range(1,9)}
    assert validation_gate(result,'lock','lock',4.);assert not validation_gate(result,'changed','lock',4.);result[5]['phase']=0;assert not validation_gate(result,'lock','lock',4.)


def test_q_floor_and_same_variance_ci_only_widens():
    m=mask();x=np.sin(m['indices'])+np.array([0,1,-1,2,-2])[m['fold_codes']];means=[0,1,-1,2,-2]
    old=HAC.real_ci(x,means,.003,m);new=fixed_real_ci(x,means,.003,m,4.);assert old['variance']==new['variance'] and new['ci95'][0]<=old['ci95'][0] and new['ci95'][1]>=old['ci95'][1]
    with pytest.raises(ValueError):fixed_real_ci(x,means,.003,m,Z95-1e-6)
