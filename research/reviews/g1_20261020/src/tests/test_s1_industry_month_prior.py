"""Frozen S1 boundary fixtures, no real sources or experimental outcome lookups."""
import numpy as np
import pandas as pd
import pytest
from s1_industry_month_prior import (PRIOR_COLUMNS,prior_lookup,project_prior,append_prior,
    validate_annual,paired_date_table,block_bootstrap,acceptance)


def annual_fixture():
    rows=[]
    for year in [2006,2020,2021,2023,2024,2025]:
        for key in ['CAPCO_CURRENT_RETROSPECTIVE:C40','TDX_RETROSPECTIVE:C40','UNKNOWN:UNKNOWN']:
            for month in range(1,13):
                rows.append(dict(industry_key=key,year=year,month=month,
                    actual_hits=float((month+year%7)*2),expected_hits=float(month*3)))
    monthly=pd.DataFrame(rows)
    annual=monthly.groupby(['industry_key','year'])[['actual_hits','expected_hits']].sum().reset_index()
    return annual,monthly


def test_asof_excludes_entire_current_year_and_future():
    a,b=annual_fixture();x=prior_lookup(a,b,[2021,2024])
    b.loc[b.year>=2021,'actual_hits']*=1000
    a=b.groupby(['industry_key','year'])[['actual_hits','expected_hits']].sum().reset_index()
    y=prior_lookup(a,b,[2021,2024])
    pd.testing.assert_frame_equal(x[x.signal_year==2021],y[y.signal_year==2021])
    assert not np.array_equal(x.loc[x.signal_year==2024,PRIOR_COLUMNS[0]],y.loc[y.signal_year==2024,PRIOR_COLUMNS[0]])
    assert x.groupby('signal_year').max_prior_year.max().to_dict()=={2021:2020,2024:2023}


def test_exact_shrinkage_formula_and_four_columns():
    a,b=annual_fixture();row=prior_lookup(a,b,[2021]).query("industry_key=='CAPCO_CURRENT_RETROSPECTIVE:C40' and month==1").iloc[0]
    aa=a.query("year<2021 and industry_key=='CAPCO_CURRENT_RETROSPECTIVE:C40'")
    bb=b.query("year<2021 and industry_key=='CAPCO_CURRENT_RETROSPECTIVE:C40' and month==1")
    li=(aa.actual_hits.sum()+100)/(aa.expected_hits.sum()+100)
    lim=(bb.actual_hits.sum()+100*li)/(bb.expected_hits.sum()+100)
    assert row[PRIOR_COLUMNS[0]]==pytest.approx(np.log(lim/li))
    assert row[PRIOR_COLUMNS[1]]==pytest.approx(6/106)
    assert row[PRIOR_COLUMNS[2]]==2
    assert len(PRIOR_COLUMNS)==4


def projection_fixture():
    a,b=annual_fixture();lookup=prior_lookup(a,b,[2020,2021])
    calendar=pd.DataFrame({'date':pd.to_datetime(['2020-12-30','2020-12-31','2021-01-04','2021-01-05']), 'calendar_index':range(4)})
    keys=pd.DataFrame({'date':pd.to_datetime(['2020-12-31']*3),'code':['1.SH','2.SZ','3.BJ']})
    roster=pd.DataFrame({'code':['1.SH','2.SZ'],'industry_key':['CAPCO_CURRENT_RETROSPECTIVE:C40','UNKNOWN:UNKNOWN']})
    return keys,roster,calendar,lookup


def test_dplus1_month_december_to_january_and_signal_year_cutoff():
    keys,roster,cal,lookup=projection_fixture();p=project_prior(keys,roster,cal,lookup)
    expected=lookup.query("signal_year==2020 and month==1 and industry_key=='CAPCO_CURRENT_RETROSPECTIVE:C40'").iloc[0]
    assert p.iloc[0][PRIOR_COLUMNS[0]]==expected[PRIOR_COLUMNS[0]]
    assert p.iloc[0][PRIOR_COLUMNS[2]]==1


def test_unknown_and_unmapped_neutral_without_key_loss():
    keys,r,c,l=projection_fixture();p=project_prior(keys,r,c,l)
    assert len(p)==3
    np.testing.assert_array_equal(p.loc[1:,list(PRIOR_COLUMNS)].to_numpy(),[[0,0,0,1],[0,0,0,1]])
    pd.testing.assert_frame_equal(p[['date','code']],keys)


def test_zero_cell_support_neutral():
    a,b=annual_fixture();b.loc[b.month==1,['actual_hits','expected_hits']]=0
    a=b.groupby(['industry_key','year'])[['actual_hits','expected_hits']].sum().reset_index()
    x=prior_lookup(a,b,[2021]).query('month==1')
    np.testing.assert_array_equal(x[list(PRIOR_COLUMNS)].to_numpy(),np.tile([0,0,0,1],(3,1)))


def test_namespace_kept_separate():
    a,b=annual_fixture();b.loc[b.industry_key.str.startswith('TDX'),'actual_hits']*=5
    a=b.groupby(['industry_key','year'])[['actual_hits','expected_hits']].sum().reset_index()
    x=prior_lookup(a,b,[2021]);assert x.industry_key.nunique()==3
    # Absolute scale does not make namespace collapse permissible.
    assert len(x)==36


def test_duplicate_or_conflicting_aggregate_fails_closed():
    a,b=annual_fixture()
    with pytest.raises(ValueError,match='duplicate'):validate_annual(pd.concat([a,a.iloc[:1]]),b)
    a.loc[0,'actual_hits']+=1
    with pytest.raises(ValueError,match='semantic conflict'):validate_annual(a,b)


def test_invalid_annual_values_and_years_fail_closed():
    a,b=annual_fixture();a.loc[0,'expected_hits']=-1
    with pytest.raises(ValueError):prior_lookup(a,b,[2021])
    a,b=annual_fixture();a.loc[0,'year']=2026
    with pytest.raises(ValueError):prior_lookup(a,b,[2021])


def test_key_and_calendar_ambiguity_fail_closed():
    keys,r,c,l=projection_fixture()
    with pytest.raises(ValueError):project_prior(pd.concat([keys,keys.iloc[:1]]),r,c,l)
    with pytest.raises(ValueError):project_prior(keys,pd.concat([r,r.iloc[:1]]),c,l)
    c.loc[1,'calendar_index']=3
    with pytest.raises(ValueError):project_prior(keys,r,c,l)


def test_unavailable_anchor_is_not_silent_missing():
    keys,r,c,l=projection_fixture();keys['date']=c.date.max()
    with pytest.raises(ValueError,match='anchor unavailable'):project_prior(keys,r,c,l)


def test_append_exactly_four_columns_preserves_m0_values_order():
    keys,r,c,l=projection_fixture();features=keys.copy();features['core']=np.array([1.,2.,3.])
    p=project_prior(keys,r,c,l);s=append_prior(features,p)
    pd.testing.assert_frame_equal(features,s[features.columns]);assert list(s.columns)==list(features.columns)+list(PRIOR_COLUMNS)
    with pytest.raises(ValueError):append_prior(features,p.iloc[::-1])


def event_fixture():
    rows=[];cal=[];idx=0
    for fold in ['2024H1','2024H2','2025H1','2025H2','2026H1','2026H2_PARTIAL_TO_2026_10_15']:
        year=int(fold[:4]);month=1 if fold[5]=='1' else 7
        for date in pd.bdate_range(f'{year}-{month:02d}-01',periods=25):
            cal.append(dict(date=date,calendar_index=idx));idx+=1
            for k in range(10):rows.append(dict(date=date,fold=fold,code=str(k),return_h2=float(k/1000)))
    a=pd.DataFrame(rows);b=a.copy();b.return_h2+=.002
    return a,b,pd.DataFrame(cal)


def test_primary_gates_five_complete_folds_ignore_partial_winner():
    a,b,c=event_fixture();b.loc[b.fold.str.startswith('2026H2'),'return_h2']=-100
    result=acceptance(a,b,c)
    assert result['decision']=='S1_DEVELOPMENT_GATE_PASS';assert result['paired_dates']==125
    assert result['mean_delta']==pytest.approx(.002);assert len(result['fold_mean_deltas'])==5


def test_return_threshold_is_001_and_no_median_or_p10_deterioration():
    a,b,c=event_fixture();b.return_h2=a.return_h2+.0009
    result=acceptance(a,b,c)
    assert not result['gates']['mean_delta_at_least_10bp']
    assert result['decision']=='S1_DEVELOPMENT_GATE_FAILED'


def test_missing_outcome_excludes_whole_date_pair_no_refill():
    a,b,c=event_fixture();b.loc[0,'return_h2']=np.nan
    pair=paired_date_table(a,b,c);assert (~pair.paired_complete).sum()==1
    b=b.drop(index=0)
    with pytest.raises(ValueError,match='Top10'):paired_date_table(a,b,c)


def test_bootstrap_seed_and_date_block_alignment():
    a,b,c=event_fixture();pair=paired_date_table(a,b,c)
    x=block_bootstrap(pair,a,b,reps=20);y=block_bootstrap(pair,a,b,reps=20)
    assert x==y
    assert x['mean_delta']['lower']==pytest.approx(.002)
    pair=pair.iloc[::2].copy()
    with pytest.raises(ValueError,match='consecutive'):block_bootstrap(pair,a,b,reps=2)


def test_frozen_route_split_purges_label_maturity_before_any_fit():
    from s1_runner import frozen_route
    m=frozen_route();fold=m.fixed_folds()[0]
    dates=pd.to_datetime(['2023-12-27','2023-12-28','2024-01-02'])
    features=pd.DataFrame({'date':dates,'code':['a']*3,'x':[1.,2.,3.]})
    labels=pd.DataFrame({'date':dates[:2],'code':['a']*2,'label_return':[.1,.2],
      'label_end_date':pd.to_datetime(['2023-12-29','2024-01-02'])})
    train,test=m.split_fold(features,labels,fold)
    assert len(train)==1 and train.date.iloc[0]==dates[0]
    assert 'label_return' not in test
