"""Frozen G1 Top50 rank-band prior residual diagnostic. No estimator fitting."""
import numpy as np
import pandas as pd
PRIOR='industry_month_log_relative_lift'
MISSING='industry_month_missing'
COMPLETE=('2024H1','2024H2','2025H1','2025H2','2026H1')


def top50(predictions):
    p=predictions[['date','code','prediction']].copy()
    if p.duplicated(['date','code']).any() or not np.isfinite(p.prediction).all():raise ValueError('prediction identity invalid')
    p=p.sort_values(['date','prediction','code'],ascending=[True,False,True],kind='mergesort')
    p['rank']=p.groupby('date').cumcount()+1
    out=p.loc[p['rank']<=50].reset_index(drop=True)
    if not out.groupby('date').size().eq(50).all():raise ValueError('Top50 scoring date lacks fifty candidates')
    return out


def groups_before_labels(top,prior):
    if prior.duplicated(['date','code']).any():raise ValueError('prior duplicate key')
    keys=top.merge(prior[['date','code',PRIOR,MISSING]],on=['date','code'],how='left',validate='one_to_one')
    keys['known_prior']=keys[MISSING].eq(0)&np.isfinite(keys[PRIOR])
    keys['band']=((keys['rank']-1)//10+1).astype(int);keys['high']=False;keys['low']=False
    status=[]
    for (date,band),g in keys.groupby(['date','band'],sort=True):
        known=g.loc[g.known_prior];reason='OK';lo=hi=None
        if len(known)<6:reason='KNOWN_PRIOR_LT6'
        else:
            lo,hi=np.quantile(known[PRIOR].to_numpy(dtype=float),[.3,.7],method='linear')
            high=known[PRIOR]>=hi;low=known[PRIOR]<=lo
            keys.loc[known.index,'high']=high;keys.loc[known.index,'low']=low
            if (high&low).any():reason='HIGH_LOW_OVERLAP'
            elif int(high.sum())<2 or int(low.sum())<2:reason='GROUP_SIZE_LT2'
        h=keys.loc[g.index,'high'];l=keys.loc[g.index,'low']
        status.append(dict(date=date,band=int(band),known_prior_count=len(known),missing_prior_count=len(g)-len(known),q30=lo,q70=hi,high_count=int(h.sum()),low_count=int(l.sum()),prelabel_valid=reason=='OK',prelabel_reason=reason))
    return keys,pd.DataFrame(status)


def evaluate_bands(keys,status,labels,validation_dates):
    rows=keys.merge(labels[['date','code','label_return']],on=['date','code'],how='left',validate='one_to_one')
    diagnostics=[]
    for stat in status.to_dict('records'):
        date,band=stat['date'],stat['band'];g=rows.loc[(rows.date==date)&(rows.band==band)]
        chosen=g.loc[g.high|g.low];missing=int((~np.isfinite(chosen.label_return)).sum())
        reason=stat['prelabel_reason'];valid=stat['prelabel_valid'] and missing==0
        if stat['prelabel_valid'] and missing:reason='MISSING_SELECTED_H2'
        item={**stat,'selected_h2_missing_count':missing,'valid':bool(valid),'invalid_reason':'OK' if valid else reason,
              'residual_spread':None,'raw_h2_spread':None,'M0_score_gap':None,'reconciliation_error':None}
        if valid:
            h=g.loc[g.high];l=g.loc[g.low]
            residual=float((h.label_return-h.prediction).mean()-(l.label_return-l.prediction).mean())
            raw=float(h.label_return.mean()-l.label_return.mean());gap=float(h.prediction.mean()-l.prediction.mean())
            error=abs(residual-(raw-gap))
            if error>1e-12:raise ValueError('residual/raw/score reconciliation conflict')
            item.update(residual_spread=residual,raw_h2_spread=raw,M0_score_gap=gap,reconciliation_error=error)
        diagnostics.append(item)
    bands=pd.DataFrame(diagnostics)
    if bands.duplicated(['date','band']).any():raise ValueError('band key duplication')
    daily=[]
    for row in validation_dates.to_dict('records'):
        g=bands.loc[bands.date==row['date']];valid=len(g)==5 and bool(g.valid.all())
        daily.append({**row,'valid':valid,'valid_band_count':int(g.valid.sum()),
          **{name:float(g[name].mean()) if valid else None for name in ['residual_spread','raw_h2_spread','M0_score_gap']}})
    return rows,bands,pd.DataFrame(daily)


def bootstrap_daily(daily,minimum_blocks=2,reps=2000,seed=19320261020):
    valid=daily.loc[daily.valid].sort_values('calendar_index').reset_index(drop=True)
    blocks=[];by_fold={str(f):0 for f in daily.fold.unique()}
    for i in range(len(valid)-19):
        block=valid.iloc[i:i+20]
        if block.fold.nunique()==1 and np.array_equal(np.diff(block.calendar_index),np.ones(19)):
            blocks.append(np.arange(i,i+20));by_fold[str(block.fold.iloc[0])]+=1
    if len(blocks)<minimum_blocks:return dict(status='CANNOT_CONFIRM_INSUFFICIENT_BLOCKS',valid_blocks=len(blocks),blocks_by_fold=by_fold,minimum_is_technical_only=True,ci95=None)
    rng=np.random.default_rng(seed);values=valid.residual_spread.to_numpy(dtype=float);rep=np.empty(reps)
    for i in range(reps):
        ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),size=int(np.ceil(len(valid)/20)))])[:len(valid)]
        rep[i]=values[ix].mean()
    return dict(status='COMPLETE',valid_blocks=len(blocks),blocks_by_fold=by_fold,minimum_is_technical_only=True,ci95=[float(np.quantile(rep,.025)),float(np.quantile(rep,.975))])


def summarize(daily,bands,minimum_blocks=2):
    primary=daily.loc[daily.fold.isin(COMPLETE)].copy();valid=primary.loc[primary.valid]
    coverage={}
    foldmeans={}
    for f,g in daily.groupby('fold',sort=False):
        v=g.loc[g.valid];coverage[f]=dict(valid_dates=len(v),denominator_dates=len(g),coverage=float(len(v)/len(g)))
        foldmeans[f]=float(v.residual_spread.mean()) if len(v) else None
    overall=float(len(valid)/len(primary));coverage_ok=overall>=.8 and all(coverage[f]['coverage']>=.7 for f in COMPLETE)
    boot=bootstrap_daily(primary,minimum_blocks=minimum_blocks)
    point=float(valid.residual_spread.mean()) if len(valid) else None
    raw=float(valid.raw_h2_spread.mean()) if len(valid) else None;gap=float(valid.M0_score_gap.mean()) if len(valid) else None
    if point is not None and abs(point-(raw-gap))>1e-12:raise ValueError('daily reconciliation conflict')
    positive=sum(foldmeans[f] is not None and foldmeans[f]>0 for f in COMPLETE)
    if not coverage_ok:decision='CANNOT_CONFIRM_COVERAGE'
    elif boot['ci95'] is None:decision='CANNOT_CONFIRM_INSUFFICIENT_BLOCKS'
    elif point>=.0005 and boot['ci95'][0]>0 and positive>=4:decision='INFORMATION_GATE_PASS_DEVELOPMENT_ONLY'
    elif boot['ci95'][1]<.0005:decision='BELOW_PRACTICAL_INFORMATION_THRESHOLD'
    else:decision='INCONCLUSIVE'
    b=bands.merge(daily[['date','fold','valid']].rename(columns={'valid':'complete_day_valid'}),on='date',how='left',validate='many_to_one')
    descriptive={str(band):{'complete_day_residual_spread':float(g.loc[g.complete_day_valid,'residual_spread'].mean()) if g.complete_day_valid.any() else None}
                 for band,g in b.loc[b.fold.isin(COMPLETE)].groupby('band')}
    return dict(decision=decision,evidence_level='DIRECTION_ONLY',fit_calls=0,primary_estimand='mean_daily(mean_five_rank_bands(mean(h2-M0pred|high)-mean(h2-M0pred|low)))',
        prior_column=PRIOR,primary_point=point,raw_h2_spread=raw,M0_score_gap=gap,reconciliation_error=None if point is None else abs(point-(raw-gap)),
        overall_coverage=dict(valid_dates=len(valid),denominator_dates=len(primary),coverage=overall,required=.8,passed=coverage_ok),
        coverage_by_fold=coverage,complete_fold_positive_count=positive,fold_residual_spreads=foldmeans,bootstrap=boot,
        band_descriptive_on_complete_day_set=descriptive,invalid_band_reason_counts={str(k):int(v) for k,v in b.loc[b.fold.isin(COMPLETE),'invalid_reason'].value_counts().items()},
        partial_fold_diagnostic_only={f:{**coverage[f],'residual_spread':foldmeans[f]} for f in coverage if f not in COMPLETE},
        threshold=.0005,threshold_meaning='5bp resource-screening information threshold; NOT Top10 strategy return lift',S1_status_unchanged='S1_DEVELOPMENT_GATE_FAILED',S2_started=False)
