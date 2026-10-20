"""Bound single TDX index member decoding and calendar-period price changes."""
import numpy as np
import pandas as pd
DAY_DTYPE=np.dtype([('date','<u4'),('open','<u4'),('high','<u4'),('low','<u4'),('close','<u4'),('amount','<f4'),('volume','<u4'),('reserved','<u4')])


def decode_index(raw):
    if not raw or len(raw)%32:raise ValueError('index member length is not nonempty integral 32-byte records')
    bars=np.frombuffer(raw,dtype=DAY_DTYPE)
    dates=pd.to_datetime(bars['date'].astype(str),format='%Y%m%d',errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:raise ValueError('duplicate/non-increasing date')
    p=np.column_stack([bars[k]/100 for k in ['open','high','low','close']])
    if not np.isfinite(p).all() or (p<=0).any():raise ValueError('nonpositive/nonfinite OHLC')
    if ((p[:,1]<np.max(p[:,[0,2,3]],axis=1))|(p[:,2]>np.min(p[:,[0,1,3]],axis=1))).any():raise ValueError('OHLC ordering conflict')
    return pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'),open=p[:,0],high=p[:,1],low=p[:,2],close=p[:,3]))


def period_returns(frame,frequency):
    """Prior adjacent observed calendar-period endpoint required; never fill."""
    if frequency not in ['Y','M']:raise ValueError('frequency')
    d=frame.copy();d['period']=pd.to_datetime(d.date).dt.to_period(frequency)
    grouped={p:g for p,g in d.groupby('period',sort=True)};out=[]
    for p,g in grouped.items():
        prior=grouped.get(p-1);base=None if prior is None else prior.iloc[-1]
        last=g.iloc[-1];value=None if base is None else float(last.close/base.close-1)
        out.append(dict(period=str(p),year=p.year,month=None if frequency=='Y' else p.month,endpoint_date=last.date,endpoint_close=float(last.close),records=len(g),base_date=None if base is None else base.date,base_close=None if base is None else float(base.close),price_return=value,status='MISSING_ADJACENT_PRIOR_BASE' if base is None else 'PACKAGE_OBSERVED_PRICE_RETURN',direction=None if value is None else 'UP' if value>0 else 'DOWN' if value<0 else 'FLAT'))
    return pd.DataFrame(out)


def parity(index,futu,tolerance=.02):
    a=index[['date','close']].merge(futu[['date','close']],on='date',how='inner',suffixes=('_tdx','_futu'),validate='one_to_one')
    a['absolute_difference']=abs(a.close_tdx-a.close_futu);a['within_tolerance']=a.absolute_difference<=tolerance
    result=dict(intersection_rows=len(a),tolerance_index_points=tolerance,max_abs_difference=None if a.empty else float(a.absolute_difference.max()),out_of_tolerance=int((~a.within_tolerance).sum()),status='CANNOT_CONFIRM_NO_INTERSECTION' if a.empty else 'CANNOT_CONFIRM_SOURCE_PARITY' if (~a.within_tolerance).any() else 'CROSS_SOURCE_INTERSECTION_PARITY_WITH_LIMITS',scope='Auxiliary observed intersection only; does not officially certify historical index data')
    return result,a
