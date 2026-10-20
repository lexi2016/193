"""Frozen CI-only correction; all daily values and point estimates are inputs."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

FOLDS=('2024H1','2024H2','2025H1','2025H2','2026H1')
N_BY_FOLD=dict(zip(FOLDS,(110,123,106,112,96)))


def build_support(daily,folds=FOLDS,n_by_fold=N_BY_FOLD,block_length=20):
    """Only map original-calendar blocks; never recompute the point or coverage."""
    if daily[['fold','date']].duplicated().any():raise ValueError('duplicate fold/date')
    support={};rows=[]
    for fold in folds:
        g=daily.loc[(daily.fold==fold)&daily.valid].sort_values('calendar_index').reset_index(drop=True)
        n=len(g)
        if n!=n_by_fold[fold]:raise ValueError('frozen valid-day count drift: '+fold)
        if g.calendar_index.duplicated().any():raise ValueError('duplicate original calendar index')
        blocks=[]
        for start in range(n-block_length+1):
            if np.array_equal(np.diff(g.calendar_index.iloc[start:start+block_length].to_numpy()),np.ones(block_length-1)):
                blocks.append(np.arange(start,start+block_length))
        c=np.zeros(n,dtype=np.int64)
        for block in blocks:c[block]+=1
        support[fold]=dict(frame=g,blocks=blocks,coverage=c,n=n,K=len(blocks),m=math.ceil(n/block_length))
        rows.extend(dict(fold=fold,date=row.date,calendar_index=int(row.calendar_index),legal_block_coverage=int(c[i])) for i,row in g.iterrows())
    return support,pd.DataFrame(rows)


def support_status(support):
    if any(np.any(s['coverage']==0) for s in support.values()):return 'CANNOT_CONFIRM_BLOCK_COVERAGE'
    if any(s['K']<2 for s in support.values()):return 'CANNOT_CONFIRM_INSUFFICIENT_BLOCKS'
    return 'SUPPORTED_TECHNICALLY_ONLY'


def corrected_ci(support,fold_means,point,*,folds=FOLDS,reps=2000,seed=19320261020):
    status=support_status(support)
    if status!='SUPPORTED_TECHNICALLY_ONLY':return dict(status=status,ci95=None,replicates_executed=0),None
    total=sum(support[f]['n'] for f in folds);rng=np.random.default_rng(seed)
    draws=np.empty(reps)
    for rep in range(reps):
        value=0.0
        for f in folds:
            s=support[f];mu=fold_means[f];z=s['frame'].residual_spread.to_numpy(dtype=float)-mu
            starts=rng.integers(0,s['K'],size=s['m'])
            perturbation=sum(float(np.sum(z[s['blocks'][j]]/s['coverage'][s['blocks'][j]])) for j in starts)
            fold_draw=mu+s['K']/(s['m']*s['n'])*perturbation
            value+=(s['n']/total)*fold_draw
        draws[rep]=value
    errors=draws-float(draws.mean())
    bounds=point+np.quantile(errors,[.025,.975],method='linear')
    return dict(status='CI_COMPUTED_FIXED_FOLD_CENTERED_INVERSE_COVERAGE',ci95=[float(v) for v in bounds],replicates_executed=reps,bootstrap_mean=float(draws.mean()),minimum_is_technical_only=True),draws


def information_decision(ci_result,original_metrics):
    if ci_result['ci95'] is None:return ci_result['status']
    # Existing point/fold/coverage gates are copied; none is recomputed here.
    coverage=original_metrics['overall_coverage']['passed'] and all(original_metrics['coverage_by_fold'][f]['coverage']>=.7 for f in FOLDS)
    if not coverage:return 'CANNOT_CONFIRM_COVERAGE'
    lo,hi=ci_result['ci95'];point=original_metrics['primary_point']
    if point>=.0005 and lo>0 and original_metrics['complete_fold_positive_count']>=4:return 'INFORMATION_GATE_PASS_DEVELOPMENT_ONLY'
    if hi<.0005:return 'BELOW_PRACTICAL_INFORMATION_THRESHOLD'
    return 'INCONCLUSIVE'
