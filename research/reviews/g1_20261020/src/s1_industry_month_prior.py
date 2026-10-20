"""Frozen S1 prior transforms and date-paired acceptance; no data producers."""
from __future__ import annotations
import numpy as np
import pandas as pd

PRIOR_COLUMNS = ('industry_month_log_relative_lift', 'industry_month_support',
                 'industry_month_support_years', 'industry_month_missing')
KAPPA = 100.0
COMPLETE_FOLDS = ('2024H1','2024H2','2025H1','2025H2','2026H1')


def validate_annual(industry, month):
    for frame, keys in [(industry,['industry_key','year']), (month,['industry_key','year','month'])]:
        if not set(keys+['actual_hits','expected_hits']).issubset(frame.columns):
            raise ValueError('annual source fields missing')
        if frame.duplicated(keys).any():
            raise ValueError('duplicate annual keys')
        vals=frame[['actual_hits','expected_hits']].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals<0).any():
            raise ValueError('invalid H/E')
        if not frame.year.between(2006,2025).all():
            raise ValueError('only frozen complete years 2006..2025 allowed')
    if not month.month.between(1,12).all():
        raise ValueError('invalid month')
    totals=month.groupby(['industry_key','year'])[['actual_hits','expected_hits']].sum().sort_index()
    yearly=industry.set_index(['industry_key','year'])[['actual_hits','expected_hits']].sort_index()
    if not totals.index.equals(yearly.index) or not np.allclose(totals,yearly,rtol=1e-12,atol=1e-9):
        raise ValueError('industry/month annual H/E semantic conflict')


def prior_lookup(industry, month, signal_years):
    validate_annual(industry,month)
    output=[]
    for year in sorted(set(int(y) for y in signal_years)):
        annual=industry.loc[industry.year<year].groupby('industry_key')[['actual_hits','expected_hits']].sum()
        annual['L_i']=(annual.actual_hits+KAPPA)/(annual.expected_hits+KAPPA)
        cells=month.loc[month.year<year].groupby(['industry_key','month']).agg(
            H=('actual_hits','sum'),E=('expected_hits','sum'),
            support_years=('expected_hits',lambda s:int((s>0).sum()))).reset_index()
        cells=cells.merge(annual[['L_i']],on='industry_key',validate='many_to_one')
        cells['L_im']=(cells.H+KAPPA*cells.L_i)/(cells.E+KAPPA)
        unknown=cells.industry_key.astype(str).str.upper().str.contains('UNKNOWN')
        missing=unknown | (cells.E<=0)
        cells[PRIOR_COLUMNS[0]]=np.where(missing,0,np.log(cells.L_im/cells.L_i))
        cells[PRIOR_COLUMNS[1]]=np.where(missing,0,cells.E/(cells.E+KAPPA))
        cells[PRIOR_COLUMNS[2]]=np.where(missing,0,cells.support_years)
        cells[PRIOR_COLUMNS[3]]=missing.astype(float)
        cells['signal_year']=year
        cells['max_prior_year']=min(year-1,int(industry.year.max()))
        output.append(cells)
    return pd.concat(output,ignore_index=True)


def project_prior(keys, roster, calendar, lookup):
    """Every row uses D.year cutoff and the next observed calendar session month."""
    if keys.duplicated(['date','code']).any() or roster.code.duplicated().any():
        raise ValueError('ambiguous key/roster join')
    cal=calendar[['date','calendar_index']].sort_values('calendar_index').copy()
    if cal.date.duplicated().any() or cal.calendar_index.duplicated().any():
        raise ValueError('ambiguous calendar')
    if not np.array_equal(cal.calendar_index.to_numpy(),np.arange(len(cal))):
        raise ValueError('calendar indices must be contiguous from zero')
    cal['anchor_date']=pd.to_datetime(cal.date).shift(-1)
    dates=pd.to_datetime(keys.date).dt.normalize()
    if not dates.isin(pd.to_datetime(cal.date)).all():
        raise ValueError('signal date outside calendar')
    frame=keys[['date','code']].copy(); frame['date']=dates
    frame['_row']=np.arange(len(frame));frame['signal_year']=dates.dt.year
    frame=frame.merge(cal[['date','anchor_date']],on='date',how='left',validate='many_to_one')
    # An unavailable D+1 anchor is a metadata failure, never a silent missing prior.
    if frame.anchor_date.isna().any():
        raise ValueError('D+1 anchor unavailable')
    frame['month']=frame.anchor_date.dt.month
    frame=frame.merge(roster[['code','industry_key']],on='code',how='left',validate='many_to_one')
    frame=frame.merge(lookup[['signal_year','industry_key','month',*PRIOR_COLUMNS]],
                      on=['signal_year','industry_key','month'],how='left',validate='many_to_one')
    for col in PRIOR_COLUMNS[:-1]: frame[col]=frame[col].fillna(0).astype(float)
    frame[PRIOR_COLUMNS[-1]]=frame[PRIOR_COLUMNS[-1]].fillna(1).astype(float)
    frame=frame.sort_values('_row')
    if len(frame)!=len(keys) or not np.isfinite(frame[list(PRIOR_COLUMNS)]).all().all():
        raise ValueError('prior must preserve all candidate rows as finite values')
    return frame[['date','code',*PRIOR_COLUMNS]].reset_index(drop=True)


def append_prior(features, projection):
    if not features[['date','code']].reset_index(drop=True).equals(projection[['date','code']].reset_index(drop=True)):
        raise ValueError('prior key order mismatch')
    if set(PRIOR_COLUMNS)&set(features.columns): raise ValueError('prior already present')
    return pd.concat([features.reset_index(drop=True),projection[list(PRIOR_COLUMNS)]],axis=1)


def paired_date_table(m0, s1, calendar, complete_only=True):
    """Full ten-stock outcomes per arm/date; missing outcomes exclude pair, never replace stock."""
    summaries=[]
    for arm,frame in [('M0',m0),('S1',s1)]:
        if frame.duplicated(['fold','date','code']).any(): raise ValueError('duplicate selected key')
        if not frame.groupby(['fold','date']).size().eq(10).all(): raise ValueError('daily Top10 count differs')
        g=frame.groupby(['fold','date']).return_h2.agg(['mean','count'])
        g.loc[g['count']!=10,'mean']=np.nan
        summaries.append(g[['mean']].rename(columns={'mean':arm}))
    if not summaries[0].index.equals(summaries[1].index):
        raise ValueError('selection date coverage differs across arms')
    pair=pd.concat(summaries,axis=1).reset_index()
    pair['date']=pd.to_datetime(pair.date)
    pair=pair.merge(calendar[['date','calendar_index']],on='date',how='left',validate='many_to_one')
    if pair.calendar_index.isna().any(): raise ValueError('pair outside calendar')
    pair=pair.sort_values('calendar_index').reset_index(drop=True)
    if complete_only: pair=pair.loc[pair.fold.isin(COMPLETE_FOLDS)].reset_index(drop=True)
    pair['paired_complete']=pair[['M0','S1']].notna().all(axis=1)
    pair['delta']=pair.S1-pair.M0
    return pair


def block_bootstrap(pair, m0_events, s1_events, reps=2000, seed=19320261020, block=20):
    """Noncircular moving 20 consecutive observed-session blocks; preserve date pairing.

    Blocks cannot cross a missing paired date or fold boundary. Quantile contrasts
    resample the same dates, repeating all ten events on every sampled date.
    """
    pair=pair.loc[pair.paired_complete].reset_index(drop=True)
    blocks=[]
    for start in range(len(pair)-block+1):
        span=pair.iloc[start:start+block]
        if span.fold.nunique()==1 and np.array_equal(np.diff(span.calendar_index),np.ones(block-1)):
            blocks.append(np.arange(start,start+block))
    if not blocks: raise ValueError('insufficient consecutive sessions for frozen block bootstrap')
    def events(frame):
        f=frame.copy();f['date']=pd.to_datetime(f.date)
        grouped={pd.Timestamp(date):g.return_h2.to_numpy(dtype=float) for date,g in f.groupby('date')}
        result=np.stack([grouped[date] for date in pair.date])
        if result.shape!=(len(pair),10) or not np.isfinite(result).all(): raise ValueError('invalid event return cube')
        return result
    a,b=events(m0_events),events(s1_events)
    rng=np.random.default_rng(seed); values=np.empty((reps,3))
    n=len(pair); needed=int(np.ceil(n/block)); delta=pair.delta.to_numpy();
    for i in range(reps):
        ix=np.concatenate([blocks[j] for j in rng.integers(0,len(blocks),size=needed)])[:n]
        values[i]=[delta[ix].mean(),np.median(pair.S1.to_numpy()[ix])-np.median(pair.M0.to_numpy()[ix]),
                   np.quantile(b[ix].reshape(-1),.1)-np.quantile(a[ix].reshape(-1),.1)]
    return {name:{'lower':float(np.quantile(values[:,j],.025)),'upper':float(np.quantile(values[:,j],.975))}
            for j,name in enumerate(['mean_delta','daily_median_delta','event_p10_delta'])}


def acceptance(m0, s1, calendar):
    pair=paired_date_table(m0,s1,calendar)
    valid=pair.loc[pair.paired_complete].copy()
    if valid.empty: raise ValueError('no complete paired dates')
    dates=set(valid.date)
    a=m0.loc[pd.to_datetime(m0.date).isin(dates)].copy();b=s1.loc[pd.to_datetime(s1.date).isin(dates)].copy()
    ci=block_bootstrap(pair,a,b)
    folds={f:float(valid.loc[valid.fold==f,'delta'].mean()) for f in COMPLETE_FOLDS}
    if not all(np.isfinite(list(folds.values()))): raise ValueError('missing complete fold')
    mean=float(valid.delta.mean());median=float(valid.S1.median()-valid.M0.median())
    p10=float(np.quantile(b.return_h2,.1)-np.quantile(a.return_h2,.1))
    gates={'mean_delta_at_least_10bp':mean>=.001,'mean_delta_95pct_lcb_positive':ci['mean_delta']['lower']>0,
           'at_least_four_positive_complete_folds':sum(v>0 for v in folds.values())>=4,
           'daily_median_not_worse':median>=0,'event_p10_not_worse':p10>=0}
    return {'primary_horizon':2,'gate_folds':list(COMPLETE_FOLDS),'paired_dates':len(valid),
            'missing_pair_dates':int((~pair.paired_complete).sum()),'mean_delta':mean,
            'daily_median_delta':median,'event_p10_delta':p10,'fold_mean_deltas':folds,'block_ci95':ci,
            'gates':gates,'decision':'S1_DEVELOPMENT_GATE_PASS' if all(gates.values()) else 'S1_DEVELOPMENT_GATE_FAILED',
            'evidence_level':'DIRECTION_ONLY'}
