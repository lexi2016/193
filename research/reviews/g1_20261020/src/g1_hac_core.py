"""Frozen Bartlett-20 HAC and mask-only calibration; no project model imports."""
import numpy as np
FOLDS=('2024H1','2024H2','2025H1','2025H2','2026H1')
COUNTS=(110,123,106,112,96)
Z95=1.959963984540054
SCENARIOS=('iid','ma1','ma5','ma5_hetero')
SEEDS=dict(zip(SCENARIOS,(19320261020,19320261021,19320261022,19320261023)))
TRUE_MEANS=np.array([0.,1.,-1.,2.,-2.])


def select_primary_mask(frame):
    # The parser supplies only date/calendar_index/fold/valid: no real residual column.
    required={'date','calendar_index','fold','valid'}
    if set(frame.columns)!=required:raise ValueError('mask parser received forbidden/unknown columns')
    primary=frame.loc[frame.fold.isin(FOLDS)].copy()
    if len(primary)!=601 or primary[['date']].duplicated().any():raise ValueError('primary skeleton identity mismatch')
    if not primary.valid.isin([True,False]).all():raise ValueError('invalid mask boolean')
    selected=primary.loc[primary.valid].sort_values('calendar_index').copy()
    if len(selected)!=547 or tuple(selected.groupby('fold').size().reindex(FOLDS))!=COUNTS:raise ValueError('frozen valid-day identity mismatch')
    return primary,selected


def calibration_gate(results):
    if set(results)!=set(SCENARIOS):return False
    return all(results[s]['scenario']==s and results[s]['seed']==SEEDS[s] and results[s]['reps']==10000 and results[s]['wilson95'][0]>=.93 and results[s]['pass_calibration'] for s in SCENARIOS)


def mask_matrices(indices,fold_codes,nfolds=5,bandwidth=20):
    indices=np.asarray(indices,dtype=np.int64);fold_codes=np.asarray(fold_codes,dtype=np.int64)
    if len(indices)==0 or len(indices)!=len(fold_codes) or np.any(np.diff(indices)<=0):raise ValueError('invalid/nonunique original calendar indices')
    counts=np.bincount(fold_codes,minlength=nfolds)
    if len(counts)!=nfolds or np.any(counts<=0):raise ValueError('invalid fold mask')
    K=np.maximum(1-np.abs(indices[:,None]-indices[None,:])/bandwidth,0.)
    X=np.eye(nfolds)[fold_codes];M=np.eye(len(indices))-X@np.diag(1/counts)@X.T
    D=float(np.trace(K@M))
    if not np.isfinite(D) or D<=0:raise ValueError('nonpositive trace(KM)')
    return dict(K=K,D=D,c=len(indices)/D,counts=counts,fold_codes=fold_codes,indices=indices,N=len(indices),bandwidth=bandwidth)


def variance(residuals,mask):
    z=np.asarray(residuals,dtype=float)
    q=np.sum((z@mask['K'])*z,axis=-1)
    v=mask['c']*q/mask['N']**2
    if np.any(v < -1e-16) or not np.isfinite(v).all():raise ValueError('invalid HAC variance')
    return np.maximum(v,0.)


def lag_quadratic(z,indices,bandwidth=20):
    z=np.asarray(z);total=float(np.dot(z,z));indices=np.asarray(indices)
    for i in range(len(z)):
        for j in range(i+1,len(z)):
            lag=indices[j]-indices[i]
            if lag>=bandwidth:break
            total+=2*(1-lag/bandwidth)*z[i]*z[j]
    return total


def wilson(successes,total):
    p=successes/total;den=1+Z95**2/total;center=(p+Z95**2/(2*total))/den;radius=Z95*np.sqrt(p*(1-p)/total+Z95**2/(4*total**2))/den
    return [float(center-radius),float(center+radius)]


def simulate_chunk(rng,size,mask,scenario):
    # Innovations cover every integer original-calendar slot, then select valid dates.
    L=int(mask['indices'][-1]-mask['indices'][0]+1);prefix=5
    innovations=rng.standard_normal((size,L+prefix))
    if scenario=='iid':error=innovations[:,prefix:]
    elif scenario=='ma1':error=(innovations[:,prefix:]+.5*innovations[:,prefix-1:-1])/np.sqrt(1.25)
    elif scenario in ['ma5','ma5_hetero']:error=sum(innovations[:,prefix-j:L+prefix-j] for j in range(6))/np.sqrt(6.)
    else:raise ValueError('scenario')
    selected=error[:,mask['indices']-mask['indices'][0]]
    if scenario=='ma5_hetero':selected=selected*np.array([1.,2.,1.,2.,1.])[mask['fold_codes']]
    samples=selected+TRUE_MEANS[mask['fold_codes']]
    means=np.column_stack([samples[:,mask['fold_codes']==f].mean(axis=1) for f in range(5)])
    point=samples.mean(axis=1);residuals=samples-means[:,mask['fold_codes']]
    half=Z95*np.sqrt(variance(residuals,mask));target=float(np.dot(mask['counts']/mask['N'],TRUE_MEANS))
    covered=(point-half<=target)&(target<=point+half)
    return covered,2*half


def calibrate(mask,scenario,reps=10000,chunk_size=256,progress=None):
    if chunk_size<1 or chunk_size>4096:raise ValueError('bounded chunk must be1..4096')
    rng=np.random.default_rng(SEEDS[scenario]);covered=[];width=[]
    for start in range(0,reps,chunk_size):
        c,w=simulate_chunk(rng,min(chunk_size,reps-start),mask,scenario);covered.append(c);width.append(w)
        if progress is not None:progress(min(start+chunk_size,reps),reps)
    covered=np.concatenate(covered);width=np.concatenate(width);hits=int(covered.sum());ci=wilson(hits,reps)
    return dict(scenario=scenario,seed=SEEDS[scenario],reps=reps,covered=hits,coverage=hits/reps,wilson95=ci,mean_width=float(width.mean()),pass_calibration=ci[0]>=.93,threshold_wilson_lower=.93,chunk_size=chunk_size)


def real_ci(x,accepted_fold_means,accepted_point,mask):
    z=np.asarray(x,dtype=float)-np.asarray(accepted_fold_means)[mask['fold_codes']]
    v=float(variance(z,mask));se=float(np.sqrt(v));half=Z95*se
    return dict(point=accepted_point,variance=v,se=se,ci95=[float(accepted_point-half),float(accepted_point+half)],N=mask['N'],D=mask['D'],mask_correction=mask['c'],method='BARTLETT20_HAC_TRACE_KM_MASK_CORRECTION',z95=Z95)


def information_gate(ci,original):
    if not original['overall_coverage']['passed'] or any(original['coverage_by_fold'][f]['coverage']<.7 for f in FOLDS):return 'CANNOT_CONFIRM_COVERAGE'
    lo,hi=ci['ci95']
    if original['primary_point']>=.0005 and lo>0 and original['complete_fold_positive_count']>=4:return 'INFORMATION_GATE_PASS_DEVELOPMENT_ONLY'
    if hi<.0005:return 'BELOW_PRACTICAL_INFORMATION_THRESHOLD'
    return 'INCONCLUSIVE'
