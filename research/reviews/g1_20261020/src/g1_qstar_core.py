"""Only the critical value changes; frozen HAC20 estimator is imported verbatim."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.signal import lfilter
from scipy.stats import binom
HAC_SOURCE=Path(__file__).with_name('g1_hac_core.py')
spec=importlib.util.spec_from_file_location('qstar_frozen_hac',HAC_SOURCE);HAC=importlib.util.module_from_spec(spec);spec.loader.exec_module(HAC)
FOLDS=HAC.FOLDS;COUNTS=HAC.COUNTS;Z95=HAC.Z95
SCENARIOS={1:'IID',2:'MA1',3:'MA5',4:'MA5_STD_12121',5:'GAUSSIAN_AR09',6:'T5_AR09',7:'MA5_STD_21212',8:'T5_AR09_SINE_SIGMA'}
TRUE_MEANS=np.array([0.,1.,-1.,2.,-2.]);MICROTILE=32


def build_mask(frame):
    primary,selected=HAC.select_primary_mask(frame);primary=primary.sort_values('calendar_index');indices=primary.calendar_index.to_numpy(dtype=np.int64)
    if not np.array_equal(indices,np.arange(648,1249)):raise ValueError('601original calendar648..1248 identity drift')
    codes=np.array([FOLDS.index(f) for f in selected.fold]);mask=HAC.mask_matrices(selected.calendar_index.to_numpy(),codes)
    mask.update(full_indices=indices,full_fold_codes=np.array([FOLDS.index(f) for f in primary.fold]),valid_positions=mask['indices']-indices[0],full_length=len(indices));return primary,selected,mask


def rng_for(phase,scenario):
    if phase not in (0,1) or scenario not in SCENARIOS:raise ValueError('phase/scenario')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(19320261021,spawn_key=(phase,scenario))))


def error_draws(rng,rows,mask,scenario):
    L=mask['full_length']
    if scenario==1:error=rng.standard_normal((rows,L))
    elif scenario in (2,3,4,7):
        u=rng.standard_normal((rows,L+5))
        if scenario==2:error=(u[:,5:]+.5*u[:,4:-1])/np.sqrt(1.25)
        else:error=sum(u[:,5-j:L+5-j] for j in range(6))/np.sqrt(6.)
        if scenario in (4,7):error*=np.array([1.,2.,1.,2.,1.] if scenario==4 else [2.,1.,2.,1.,2.])[mask['full_fold_codes']]
    elif scenario in (5,6,8):
        u=rng.standard_normal((rows,L+512)) if scenario==5 else rng.standard_t(5,size=(rows,L+512))*np.sqrt(3./5.)
        error=lfilter([np.sqrt(1-.9**2)],[1.,-.9],u,axis=1)[:,512:]
        if scenario==8:error*=1+.75*np.sin(2*np.pi*np.arange(L)/600.)
    else:raise ValueError('scenario')
    return error


def abs_t_and_se(errors,mask):
    samples=errors[:,mask['valid_positions']]+TRUE_MEANS[mask['fold_codes']]
    means=np.column_stack([samples[:,mask['fold_codes']==f].mean(axis=1) for f in range(5)])
    point=samples.mean(axis=1);residual=samples-means[:,mask['fold_codes']]
    variance=HAC.variance(residual,mask)
    if not np.isfinite(variance).all() or np.any(variance<=0):raise ValueError('nonfinite/nonpositive simulated variance')
    se=np.sqrt(variance);truth=float(np.dot(mask['counts']/mask['N'],TRUE_MEANS));Y=np.abs((point-truth)/se)
    if not np.isfinite(Y).all():raise ValueError('nonfinite absolute studentized statistic')
    return Y,se


def simulate_segment(rng,count,mask,scenario,chunk=256):
    if chunk<32 or chunk>4096 or chunk%32:raise ValueError('chunk must be multiple32 in32..4096')
    ys=[];ses=[]
    for start in range(0,count,chunk):
        error=error_draws(rng,min(chunk,count-start),mask,scenario)
        # Fixed math tiles make BLAS shape independent of configurable I/O chunk.
        for j in range(0,len(error),MICROTILE):
            y,se=abs_t_and_se(error[j:j+MICROTILE],mask);ys.append(y);ses.append(se)
    return np.concatenate(ys),np.concatenate(ses)


def critical_index(n=50000,p=.95,m=8,family_alpha=.01):
    threshold=1-family_alpha/m
    if binom.cdf(n-1,n,p)<threshold:raise ValueError('no finite order-statistic upper bound rank')
    lo,hi=1,n
    while lo<hi:
        mid=(lo+hi)//2
        if binom.cdf(mid-1,n,p)>=threshold:hi=mid
        else:lo=mid+1
    return dict(k=lo,n=n,p=p,m=m,family_alpha=family_alpha,cdf_at_k_minus_1=float(binom.cdf(lo-1,n,p)),cdf_at_k_minus_2=float(binom.cdf(lo-2,n,p)),required_cdf=threshold)


def order_bound(Y,k):
    Y=np.asarray(Y,dtype=float)
    if Y.ndim!=1 or not np.isfinite(Y).all() or k<1 or k>len(Y):raise ValueError('invalid finite order statistic input')
    return float(np.partition(Y,k-1)[k-1])


def validation_summary(Y,se,q,scenario):
    if not np.isfinite(q) or q<Z95 or len(Y)!=10000 or not np.isfinite(Y).all() or not np.isfinite(se).all():raise ValueError('validation identity/nonfinite conflict')
    hits=int(np.sum(Y<=q));wilson=HAC.wilson(hits,len(Y))
    return dict(phase=1,scenario_id=scenario,scenario=SCENARIOS[scenario],reps=len(Y),covered=hits,coverage=hits/len(Y),wilson95=wilson,mean_width=float(np.mean(2*q*se)),pass_validation=wilson[0]>=.93,qstar=q)


def validation_gate(results,lock_sha,expected_lock_sha,q):
    if lock_sha!=expected_lock_sha or set(results)!=set(range(1,9)):return False
    return all(r['phase']==1 and r['scenario_id']==s and r['reps']==10000 and r['pass_validation'] and r['wilson95'][0]>=.93 and r.get('qstar_lock_sha256')==expected_lock_sha and r.get('qstar')==q for s,r in results.items())


def fixed_real_ci(x,accepted_means,accepted_point,mask,q):
    if not np.isfinite(q) or q<Z95:raise ValueError('critical value below frozen floor/nonfinite')
    z=np.asarray(x,dtype=float)-np.asarray(accepted_means)[mask['fold_codes']];v=float(HAC.variance(z,mask));se=float(np.sqrt(v));return dict(point=accepted_point,variance=v,se=se,qstar=q,ci95=[float(accepted_point-q*se),float(accepted_point+q*se)],N=mask['N'],D=mask['D'],mask_correction=mask['c'],method='FROZEN_HAC20_WITH_EIGHT_SCENARIO_QSTAR')
