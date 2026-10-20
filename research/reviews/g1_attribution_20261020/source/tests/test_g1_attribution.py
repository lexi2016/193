"""Synthetic accounting fixtures only; no real inputs or old model imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from g1_attribution_core import audit, PRIOR, MISSING


def fixture():
    specs = [('2024-01-02', '2024H1'), ('2024-01-03', '2024H1'),
             ('2024-07-01', '2024H2'), ('2026-07-01', '2026H2_PARTIAL_TO_2026_10_15')]
    dates = pd.DataFrame([dict(date=pd.Timestamp(d), fold=f, calendar_index=i*2)
                          for i, (d, f) in enumerate(specs)])
    calrows = []
    for i, (d, _) in enumerate(specs):
        calrows += [dict(date=pd.Timestamp(d), calendar_index=i*2),
                    dict(date=pd.Timestamp(d)+pd.Timedelta(hours=12), calendar_index=i*2+1)]
    calendar = pd.DataFrame(calrows)
    roster = pd.DataFrame([dict(code=f'{j:06d}', industry_key=f'I{(j-1)%10}') for j in range(1, 51)])
    lookup = pd.DataFrame([dict(industry_key=f'I{i}', month=m, signal_year=y,
        max_prior_year=y-1, H=100, E=100, support_years=18,
        industry_month_support=.5, industry_month_support_years=18,
        **{PRIOR: i/10, MISSING: 0.0}) for y in [2024, 2026] for m in [1, 7] for i in range(10)])
    keys, bands, daily = [], [], []
    for d in dates.itertuples():
        for j in range(1, 51):
            prior = ((j-1)%10)/10
            keys.append(dict(date=d.date, fold=d.fold, code=f'{j:06d}', prediction=j*.0001,
                rank=j, band=(j-1)//10+1, **{PRIOR: prior, MISSING: 0.0},
                known_prior=True, high=prior>=.63, low=prior<=.27))
        for b in range(1, 6):
            bands.append(dict(date=d.date, band=b, known_prior_count=10, missing_prior_count=0,
                q30=.27, q70=.63, high_count=3, low_count=3, prelabel_valid=True,
                prelabel_reason='OK', selected_h2_missing_count=0, valid=True, invalid_reason='OK'))
    keys = pd.DataFrame(keys)
    rows = keys.copy()
    # The second fold has a different known residual contrast; equal-date != equal-fold.
    multiplier = np.where(rows.fold.eq('2024H2'), 3, 1)
    rows['label_return'] = rows.prediction+rows[PRIOR]*.01*multiplier
    old = pd.DataFrame(bands)
    for i, row in old.iterrows():
        g = rows.loc[rows.date.eq(row.date) & rows.band.eq(row.band)]
        h, l = g.loc[g.high], g.loc[g.low]
        old.loc[i, 'raw_h2_spread'] = h.label_return.mean()-l.label_return.mean()
        old.loc[i, 'M0_score_gap'] = h.prediction.mean()-l.prediction.mean()
        old.loc[i, 'residual_spread'] = (h.label_return-h.prediction).mean()-(l.label_return-l.prediction).mean()
        old.loc[i, 'reconciliation_error'] = 0.0
    statuses = old[['date', 'band', 'known_prior_count', 'missing_prior_count', 'q30', 'q70',
                   'high_count', 'low_count', 'prelabel_valid', 'prelabel_reason']].copy()
    for d in dates.to_dict('records'):
        g = old.loc[old.date.eq(d['date'])]
        daily.append({**d, 'valid': True, 'valid_band_count': 5,
            **{x: g[x].mean() for x in ['raw_h2_spread', 'M0_score_gap', 'residual_spread']}})
    return dict(rows=rows, keys=keys, dates=dates, statuses=statuses, old_bands=old,
                old_daily=pd.DataFrame(daily), calendar=calendar, roster=roster, lookup=lookup)


def invalidate_selected_label(data):
    date = data['dates'].date.iloc[0]
    data['rows'].loc[data['rows'].date.eq(date) & data['rows'].code.eq('000010'), 'label_return'] = np.nan
    mask = data['old_bands'].date.eq(date) & data['old_bands'].band.eq(1)
    data['old_bands'].loc[mask, ['selected_h2_missing_count', 'valid', 'invalid_reason']] = [1, False, 'MISSING_SELECTED_H2']
    data['old_bands'].loc[mask, ['residual_spread', 'raw_h2_spread', 'M0_score_gap', 'reconciliation_error']] = np.nan
    mask = data['old_daily'].date.eq(date)
    data['old_daily'].loc[mask, ['valid', 'valid_band_count']] = [False, 4]
    data['old_daily'].loc[mask, ['residual_spread', 'raw_h2_spread', 'M0_score_gap']] = np.nan


def test_original_equal_date_not_equal_fold_weights_and_components():
    output, summary = audit(**fixture(), expected_primary=(3, 3, 0))
    point = summary['populations'][0]['point_decimal']
    assert point == pytest.approx(.007*5/3)
    folded = output['fold_identity'].set_index('fold')
    assert folded.loc['2024H1', 'global_valid_date_weight'] == pytest.approx(2/3)
    assert folded.loc['2024H2', 'global_valid_date_weight'] == pytest.approx(1/3)
    for name in ['industry_key', 'code', 'fold', 'band']:
        table = output[name+'_contributions']
        assert table.loc[table.population.eq('PRIMARY_COMPLETE_FOLDS'), 'residual_contribution_decimal'].sum() == pytest.approx(point)
    band = output['band_contributions']
    assert band.loc[band.population.eq('PRIMARY_COMPLETE_FOLDS') & band.band.eq(1), 'residual_contribution_decimal'].iloc[0] == pytest.approx(point/5)


def test_missing_selected_label_invalidates_whole_date_without_refill():
    data = fixture(); invalidate_selected_label(data)
    output, summary = audit(**data, expected_primary=(3, 2, 1))
    day = output['daily_identity'].iloc[0]
    assert day.invalid_reason_signature == 'B1:MISSING_SELECTED_H2'
    assert day.actual_failure_MISSING_SELECTED_H2
    assert summary['populations'][0]['point_decimal'] == pytest.approx(.014)
    assert output['validity_partition'].date_count.sum() == 4


def test_nonselected_missing_label_is_descriptive_only():
    data = fixture()
    data['rows'].loc[data['rows'].code.eq('000005'), 'label_return'] = np.nan
    output, _ = audit(**data)
    assert output['daily_identity'].day_valid.all()
    assert output['daily_identity'].any_nonselected_h2_missing.all()
    assert output['daily_identity'].invalid_reason_signature.eq('VALID').all()


def test_missing_prior_exclusion_not_automatic_invalidity():
    data = fixture()
    # One neutral, unselected original prior was unknown, leaving nine known rows.
    for key in ['keys', 'rows']:
        mask = data[key].code.eq('000005')
        data[key].loc[mask, [PRIOR, MISSING, 'known_prior']] = [0.0, 1.0, False]
    data['lookup'].loc[data['lookup'].industry_key.eq('I4'), [PRIOR, MISSING]] = [0.0, 1.0]
    for key in ['statuses', 'old_bands']:
        mask = data[key].band.eq(1)
        data[key].loc[mask, ['known_prior_count', 'missing_prior_count']] = [9, 1]
    # Same I4 code in other bands must also carry the same original missing prior.
    for key in ['keys', 'rows']:
        mask = data[key].code.isin(['000015', '000025', '000035', '000045'])
        data[key].loc[mask, [PRIOR, MISSING, 'known_prior']] = [0.0, 1.0, False]
    for key in ['statuses', 'old_bands']:
        data[key].loc[:, ['known_prior_count', 'missing_prior_count']] = [9, 1]
    output, _ = audit(**data)
    assert output['daily_identity'].day_valid.all()
    assert output['daily_identity'].any_missing_prior.all()


def test_frozen_membership_drift_stops():
    data = fixture(); data['rows'].loc[0, 'high'] = True
    with pytest.raises(ValueError, match='membership identity'):
        audit(**data)


@pytest.mark.parametrize('change', ['missing_band', 'duplicate_band', 'duplicate_roster', 'prior_drift'])
def test_ambiguous_structure_or_metadata_stops(change):
    data = fixture()
    if change == 'missing_band': data['statuses'] = data['statuses'].iloc[1:].copy()
    if change == 'duplicate_band': data['statuses'] = pd.concat([data['statuses'], data['statuses'].iloc[:1]])
    if change == 'duplicate_roster': data['roster'] = pd.concat([data['roster'], data['roster'].iloc[:1]])
    if change == 'prior_drift': data['lookup'].loc[0, PRIOR] += 1
    with pytest.raises(ValueError): audit(**data)


def test_partial_is_separate_and_missing_industry_side_unidentified():
    output, summary = audit(**fixture())
    assert summary['populations'][1]['point_decimal'] == pytest.approx(.007)
    table = output['industry_key_contributions']
    assert table.side_identifiability.eq('UNIDENTIFIED_MISSING_SIDE').all()
    assert table.loc[table.high_weighted_mass.eq(0), 'high_raw_outcome_mean_decimal'].isna().all()
    assert output['prior_cells'].prior_distinct_values.eq(1).all()
    assert output['prior_cells'].identifiability.eq('NO_WITHIN_CELL_PRIOR_VARIATION').all()


def test_tie_expanded_actual_low_group_size_keeps_exact_mass():
    data = fixture()
    # Existing grouping is fixed here with four lows rather than truncating to three.
    for key in ['keys', 'rows']:
        mask = data[key].code.isin([f'{j:06d}' for j in [4, 14, 24, 34, 44]])
        data[key].loc[mask, PRIOR] = 0.0
        data[key].loc[mask, 'low'] = True
    data['lookup'].loc[data['lookup'].industry_key.eq('I3'), PRIOR] = 0.0
    data['rows']['label_return'] = data['rows'].prediction+data['rows'][PRIOR]*.01
    for key in ['statuses', 'old_bands']:
        data[key].loc[:, 'low_count'] = 4
    for i, band in data['old_bands'].iterrows():
        g = data['rows'].loc[data['rows'].date.eq(band.date) & data['rows'].band.eq(band.band)]
        h, l = g.loc[g.high], g.loc[g.low]
        data['old_bands'].loc[i, 'raw_h2_spread'] = h.label_return.mean()-l.label_return.mean()
        data['old_bands'].loc[i, 'M0_score_gap'] = h.prediction.mean()-l.prediction.mean()
        data['old_bands'].loc[i, 'residual_spread'] = .00725
    for i, day in data['old_daily'].iterrows():
        g = data['old_bands'].loc[data['old_bands'].date.eq(day.date)]
        for field in ['raw_h2_spread', 'M0_score_gap', 'residual_spread']:
            data['old_daily'].loc[i, field] = g[field].mean()
    output, summary = audit(**data)
    assert summary['populations'][0]['point_decimal'] == pytest.approx(.00725)
    table = output['code_contributions']
    primary = table.loc[table.population.eq('PRIMARY_COMPLETE_FOLDS')]
    assert primary.low_weighted_mass.sum() == pytest.approx(1)
    assert primary.loc[primary.code.eq('000004'), 'low_weighted_mass'].iloc[0] == pytest.approx(1/20)
    assert output['daily_identity'].any_tie_expansion.all()
