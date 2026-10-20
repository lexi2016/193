"""Descriptive accounting of saved G1. No estimator, inference, or resampling."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR = 'industry_month_log_relative_lift'
MISSING = 'industry_month_missing'
COMPLETE = ('2024H1', '2024H2', '2025H1', '2025H2', '2026H1')
PARTIAL = '2026H2_PARTIAL_TO_2026_10_15'
VALUES = ('raw_outcome', 'M0_score', 'residual')


def require(condition, message):
    if not condition:
        raise ValueError('SEMANTIC_CONFLICT: ' + message)


def close(a, b, tolerance=1e-12):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=tolerance, rtol=0, equal_nan=True)


def unique(frame, keys, name):
    require(not frame.duplicated(keys).any(), name + ' duplicate keys')


def assert_same(left, right, columns, name):
    require(len(left) == len(right), name + ' row count')
    for column in columns:
        a, b = left[column].reset_index(drop=True), right[column].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            require(close(a, b, 0), name + ' exact column ' + column)
        else:
            require(a.equals(b), name + ' exact column ' + column)


def sorted_keys(frame, keys):
    return frame.sort_values(keys, kind='mergesort').reset_index(drop=True)


def enrich(rows, roster, calendar, lookup, tolerance):
    """Join only original producer metadata; reproduce saved neutral missing rule."""
    unique(roster, ['code'], 'roster')
    unique(calendar, ['date'], 'calendar date')
    unique(calendar, ['calendar_index'], 'calendar index')
    unique(lookup, ['industry_key', 'month', 'signal_year'], 'prior lookup')
    cal = calendar.sort_values('calendar_index').copy()
    require(np.array_equal(np.diff(cal.calendar_index), np.ones(len(cal)-1)),
            'calendar index discontinuity')
    cal['anchor_date'] = cal.date.shift(-1)
    result = rows.merge(roster[['code', 'industry_key']], on='code', how='left',
                        validate='many_to_one')
    result['roster_unmatched'] = result.industry_key.isna()
    result = result.merge(cal[['date', 'anchor_date']], on='date', how='left',
                          validate='many_to_one')
    require(result.anchor_date.notna().all(), 'original D+1 anchor unavailable')
    result['signal_year'] = result.date.dt.year
    result['anchor_month'] = result.anchor_date.dt.month
    cols = ['industry_key', 'month', 'signal_year', 'max_prior_year', 'H', 'E',
            'support_years', 'industry_month_support', 'industry_month_support_years',
            PRIOR, MISSING]
    prior = lookup[cols].rename(columns={'month': 'anchor_month', PRIOR: '_lookup_prior',
                                        MISSING: '_lookup_missing'})
    result = result.merge(prior, on=['industry_key', 'anchor_month', 'signal_year'],
                          how='left', validate='many_to_one', indicator='_prior_join')
    result['lookup_unmatched'] = result._prior_join.eq('left_only')
    expected_prior = result._lookup_prior.fillna(0)
    expected_missing = result._lookup_missing.fillna(1)
    require(close(result[PRIOR], expected_prior, tolerance), 'saved prior/lookup parity')
    require(close(result[MISSING], expected_missing, 0), 'saved prior missing parity')
    matched = result.loc[~result.lookup_unmatched]
    require((matched.max_prior_year < matched.signal_year).all(), 'prior cutoff not earlier')
    # This is saved numerical-cutoff metadata, not a fresh PIT certification.
    result['industry_key'] = result.industry_key.fillna('UNMAPPED_ROSTER')
    result['raw_outcome'] = result.label_return
    result['M0_score'] = result.prediction
    result['residual'] = result.label_return - result.prediction
    return result.drop(columns='_prior_join')


def reconcile_bands(rows, statuses, old_bands, dates, old_daily, tolerance):
    unique(statuses, ['date', 'band'], 'prelabel bands')
    unique(old_bands, ['date', 'band'], 'old bands')
    unique(old_daily, ['date'], 'old daily')
    require(len(statuses) == 5*len(dates), 'missing band structure')
    expected = {(d, b) for d in dates.date for b in range(1, 6)}
    require(set(zip(statuses.date, statuses.band)) == expected, 'band grid mismatch')
    left, right = sorted_keys(statuses, ['date', 'band']), sorted_keys(old_bands, ['date', 'band'])
    assert_same(left, right, list(statuses.columns), 'prelabel status preservation')
    indexed = old_bands.set_index(['date', 'band'])
    table = []
    for (date, band), group in rows.groupby(['date', 'band'], sort=True):
        old = indexed.loc[(date, band)]
        known = group[MISSING].eq(0) & np.isfinite(group[PRIOR])
        h, l = group.loc[group.high], group.loc[group.low]
        require(known.equals(group.known_prior), 'known prior flag')
        require(int(known.sum()) == old.known_prior_count, 'known count')
        require(int((~known).sum()) == old.missing_prior_count, 'missing prior count')
        require(len(h) == old.high_count and len(l) == old.low_count, 'side group counts')
        require((~(group.high | group.low) | known).all(), 'unknown prior selected')
        require(np.isfinite(group.prediction).all(), 'nonfinite M0 score')
        overlap = int((group.high & group.low).sum())
        if known.sum() < 6:
            reason = 'KNOWN_PRIOR_LT6'
        elif overlap:
            reason = 'HIGH_LOW_OVERLAP'
        elif min(len(h), len(l)) < 2:
            reason = 'GROUP_SIZE_LT2'
        else:
            reason = 'OK'
        require(reason == old.prelabel_reason, 'prelabel criterion/reason parity')
        require(bool(old.prelabel_valid) == (reason == 'OK'), 'prelabel valid parity')
        selected = group.high | group.low
        absent = ~np.isfinite(group.label_return)
        selected_missing = int((selected & absent).sum())
        valid = reason == 'OK' and not selected_missing
        actual_reason = 'MISSING_SELECTED_H2' if reason == 'OK' and selected_missing else reason
        require(selected_missing == old.selected_h2_missing_count, 'selected label missing count')
        require(valid == bool(old.valid) and actual_reason == old.invalid_reason,
                'original band validity/reason mismatch')
        item = dict(date=date, fold=group.fold.iloc[0], band=int(band), n_high=len(h), n_low=len(l),
                    known_prior_count=int(known.sum()), missing_prior_count=int((~known).sum()),
                    selected_h2_missing_count=selected_missing,
                    nonselected_h2_missing_count=int((~selected & absent).sum()),
                    high_low_overlap_count=overlap, high_tie_expansion=max(0, len(h)-3),
                    low_tie_expansion=max(0, len(l)-3), prelabel_reason=reason,
                    original_invalid_reason=actual_reason, band_valid=valid)
        for side, data in [('high', h), ('low', l)]:
            for field in VALUES:
                # Keep unavailable means absent; never mean-drop a missing selected label.
                finite = np.isfinite(data[field]).all()
                item[side+'_'+field+'_mean_decimal'] = float(data[field].mean()) if len(data) and finite else np.nan
        for field in VALUES:
            item[field+'_spread_decimal'] = (item['high_'+field+'_mean_decimal']-
                                             item['low_'+field+'_mean_decimal']) if valid else np.nan
        item['closure_error_decimal'] = (item['residual_spread_decimal']-
            (item['raw_outcome_spread_decimal']-item['M0_score_spread_decimal'])) if valid else np.nan
        if valid:
            require(abs(item['closure_error_decimal']) <= tolerance, 'band residual closure')
            for field, old_name in [('raw_outcome', 'raw_h2_spread'), ('M0_score', 'M0_score_gap'),
                                    ('residual', 'residual_spread')]:
                require(close(item[field+'_spread_decimal'], old[old_name], tolerance),
                        'saved band '+old_name+' parity')
        table.append(item)
    bands = pd.DataFrame(table)
    old_daily = sorted_keys(old_daily, ['date'])
    dates = sorted_keys(dates, ['date'])
    assert_same(dates, old_daily, list(dates.columns), 'validation calendar preservation')
    daily = []
    for original in old_daily.to_dict('records'):
        group = bands.loc[bands.date.eq(original['date'])].sort_values('band')
        valid = bool(group.band_valid.all())
        require(valid == bool(original['valid']), 'original date validity mismatch')
        require(int(group.band_valid.sum()) == original['valid_band_count'], 'valid band count')
        failures = [f"B{int(x.band)}:{x.original_invalid_reason}"
                    for x in group.itertuples() if not x.band_valid]
        item = dict(date=original['date'], fold=original['fold'], calendar_index=original['calendar_index'],
                    day_valid=valid, valid_band_count=int(group.band_valid.sum()),
                    invalid_reason_signature='|'.join(failures) if failures else 'VALID',
                    any_missing_prior=bool(group.missing_prior_count.gt(0).any()),
                    any_selected_h2_missing=bool(group.selected_h2_missing_count.gt(0).any()),
                    any_nonselected_h2_missing=bool(group.nonselected_h2_missing_count.gt(0).any()),
                    any_tie_expansion=bool((group.high_tie_expansion+group.low_tie_expansion).gt(0).any()))
        for row in group.itertuples():
            item[f'band_{row.band}_valid'] = row.band_valid
            item[f'band_{row.band}_original_reason'] = row.original_invalid_reason
        for reason in ('KNOWN_PRIOR_LT6', 'HIGH_LOW_OVERLAP', 'GROUP_SIZE_LT2', 'MISSING_SELECTED_H2'):
            item['actual_failure_'+reason] = bool(group.original_invalid_reason.eq(reason).any())
        for field, old_name in [('raw_outcome', 'raw_h2_spread'), ('M0_score', 'M0_score_gap'),
                                ('residual', 'residual_spread')]:
            value = float(group[field+'_spread_decimal'].mean()) if valid else np.nan
            require(close(value, original[old_name], tolerance), 'saved daily '+old_name+' parity')
            item[field+'_spread_decimal'] = value
        daily.append(item)
    return bands, pd.DataFrame(daily)


def contributions(rows, daily, population, tolerance):
    valid_dates = daily.loc[daily.day_valid, 'date']
    selected = rows.loc[rows.date.isin(valid_dates) & (rows.high | rows.low)].copy()
    if selected.empty:
        return {}, dict(population=population, valid_dates=0, point_decimal=None)
    require(not (selected.high & selected.low).any(), 'overlap on valid dates')
    selected['side'] = np.where(selected.high, 'high', 'low')
    selected['side_count'] = selected.groupby(['date', 'band', 'side']).code.transform('size')
    selected['mass'] = 1.0/(len(valid_dates)*5*selected.side_count)
    selected['sign'] = np.where(selected.high, 1.0, -1.0)
    for field in VALUES:
        selected[field+'_weighted'] = selected.mass*selected[field]
        selected[field+'_contribution_decimal'] = selected.sign*selected[field+'_weighted']
    require(close(selected.loc[selected.high, 'mass'].sum(), 1, tolerance), 'high mass closure')
    require(close(selected.loc[selected.low, 'mass'].sum(), 1, tolerance), 'low mass closure')
    tables = {}
    for key in ('industry_key', 'code', 'fold', 'band'):
        records = []
        for value, group in selected.groupby(key, sort=True):
            item = dict(population=population, **{key: value}, selected_rows=len(group),
                        distinct_dates=group.date.nunique(), distinct_codes=group.code.nunique())
            for side in ('high', 'low'):
                sub = group.loc[group.side.eq(side)]
                mass = float(sub.mass.sum())
                item[side+'_weighted_mass'] = mass
                item[side+'_rows'] = len(sub)
                for field in VALUES:
                    item[side+'_'+field+'_mean_decimal'] = float(sub[field+'_weighted'].sum()/mass) if mass else np.nan
            item['side_identifiability'] = ('BOTH_SIDES_OBSERVED_DESCRIPTIVE' if
                item['high_weighted_mass'] and item['low_weighted_mass'] else 'UNIDENTIFIED_MISSING_SIDE')
            item['composition_mass_gap'] = item['high_weighted_mass']-item['low_weighted_mass']
            for field in VALUES:
                item[field+'_contribution_decimal'] = float(group[field+'_contribution_decimal'].sum())
            item['closure_error_decimal'] = item['residual_contribution_decimal']-(
                item['raw_outcome_contribution_decimal']-item['M0_score_contribution_decimal'])
            require(abs(item['closure_error_decimal']) <= tolerance, key+' contribution closure')
            records.append(item)
        table = pd.DataFrame(records)
        for field in VALUES:
            require(close(table[field+'_contribution_decimal'].sum(),
                          daily.loc[daily.day_valid, field+'_spread_decimal'].mean(), tolerance),
                    key+' sum contribution closure')
        tables[key+'_contributions'] = table
    point = float(daily.loc[daily.day_valid, 'residual_spread_decimal'].mean())
    code = tables['code_contributions'].residual_contribution_decimal.abs()
    industry = tables['industry_key_contributions'].residual_contribution_decimal.abs()
    def concentration(values):
        total = float(values.sum())
        return dict(absolute_contribution_sum_decimal=total,
                    absolute_share_hhi=float(((values/total)**2).sum()) if total else None,
                    largest_absolute_share=float(values.max()/total) if total else None)
    return tables, dict(population=population, valid_dates=len(valid_dates),
        denominator_dates=len(daily), invalid_dates=int((~daily.day_valid).sum()),
        point_decimal=point, point_bp=point*10000,
        raw_spread_decimal=float(daily.loc[daily.day_valid, 'raw_outcome_spread_decimal'].mean()),
        score_gap_decimal=float(daily.loc[daily.day_valid, 'M0_score_spread_decimal'].mean()),
        selected_rows=len(selected), distinct_codes=selected.code.nunique(),
        repeated_code_rows=len(selected)-selected.code.nunique(),
        code_concentration=concentration(code), industry_concentration=concentration(industry),
        concentration_is_not_independent_N=True)


def prior_tables(rows, daily, tolerance):
    fields = ['industry_key', 'anchor_month', 'signal_year']
    valid = daily[['date', 'day_valid']]
    joined = rows.merge(valid, on='date', validate='many_to_one')
    cells = []
    for key, group in joined.groupby(fields, sort=True):
        distinct = group[PRIOR].nunique(dropna=False)
        require(distinct == 1, 'prior not constant within original industry/month/year cell')
        first = group.iloc[0]
        item = dict(zip(fields, key))
        item.update(observed_rows=len(group), distinct_dates=group.date.nunique(),
                    distinct_codes=group.code.nunique(), prior_distinct_values=int(distinct),
                    high_rows=int(group.high.sum()), low_rows=int(group.low.sum()),
                    repeated_rows=len(group)-1, original_valid_date_rows=int(group.day_valid.sum()),
                    industry_month_log_relative_lift=float(first[PRIOR]),
                    industry_month_missing=float(first[MISSING]),
                    identifiability='NO_WITHIN_CELL_PRIOR_VARIATION')
        for col in ['max_prior_year', 'H', 'E', 'support_years', 'industry_month_support',
                    'industry_month_support_years', 'lookup_unmatched', 'roster_unmatched']:
            item[col] = first[col]
        cells.append(item)
    detail = joined.groupby(['date', 'fold', 'band', 'industry_key'], sort=True).agg(
        rows=('code', 'size'), distinct_codes=('code', 'nunique'),
        prior_distinct_values=(PRIOR, 'nunique'), high_rows=('high', 'sum'), low_rows=('low', 'sum'),
        original_day_valid=('day_valid', 'first')).reset_index()
    detail['identifiability'] = np.where(detail.high_rows.gt(0) & detail.low_rows.gt(0),
        'BOTH_SIDES_SAME_CONSTANT_PRIOR_NOT_IDENTIFIED_EFFECT', 'UNIDENTIFIED_MISSING_SIDE')
    return pd.DataFrame(cells), detail


def audit(rows, keys, dates, statuses, old_bands, old_daily, calendar, roster, lookup,
          tolerance=1e-12, expected_primary=None, expected_point=None):
    for frame in (rows, keys, dates, statuses, old_bands, old_daily, calendar):
        frame['date'] = pd.to_datetime(frame.date)
    for frame, name in ((rows, 'labeled'), (keys, 'frozen')):
        unique(frame, ['date', 'code'], name)
    unique(dates, ['date'], 'validation dates')
    require(set(rows.date) == set(dates.date), 'Top50/calendar date mismatch')
    require(rows.groupby('date').size().eq(50).all(), 'original Top50 size')
    require(rows.groupby('date')['rank'].apply(lambda x: set(x) == set(range(1, 51))).all(), 'original rank identity')
    require(np.array_equal(rows.band, ((rows['rank']-1)//10+1)), 'band/rank identity')
    require(set(dates.fold).issubset(set(COMPLETE) | {PARTIAL}), 'unexpected fold')
    require(rows.merge(dates[['date', 'fold']], on='date', suffixes=('', '_calendar'),
                        validate='many_to_one').eval('fold == fold_calendar').all(), 'fold calendar parity')
    assert_same(sorted_keys(rows, ['date', 'code']), sorted_keys(keys, ['date', 'code']),
                list(keys.columns), 'frozen key and membership identity')
    enriched = enrich(rows, roster, calendar, lookup, tolerance)
    bands, daily = reconcile_bands(enriched, statuses, old_bands, dates, old_daily, tolerance)
    bands = bands.merge(daily[['date', 'day_valid']], on='date', validate='many_to_one')
    primary = daily.loc[daily.fold.isin(COMPLETE)]
    if expected_primary is not None:
        require((len(primary), int(primary.day_valid.sum()), int((~primary.day_valid).sum())) ==
                tuple(expected_primary), 'original 601/547/54 count identity')
    output = dict(date_band_identity=bands, daily_identity=daily)
    summaries = []
    for population, part in [('PRIMARY_COMPLETE_FOLDS', primary),
                             ('PARTIAL_2026H2_DIAGNOSTIC_ONLY', daily.loc[daily.fold.eq(PARTIAL)])]:
        if part.empty:
            continue
        parts, summary = contributions(enriched, part, population, tolerance)
        summaries.append(summary)
        for key, table in parts.items():
            output[key] = pd.concat([output.get(key, pd.DataFrame()), table], ignore_index=True)
    if expected_point is not None:
        require(close(summaries[0]['point_decimal'], expected_point, tolerance), 'accepted point parity')
    fold_rows = []
    for fold, group in daily.groupby('fold', sort=False):
        population = primary if fold in COMPLETE else daily.loc[daily.fold.eq(PARTIAL)]
        item = dict(fold=fold, denominator_dates=len(group), valid_dates=int(group.day_valid.sum()),
                    invalid_dates=int((~group.day_valid).sum()),
                    global_valid_date_weight=float(group.day_valid.sum()/population.day_valid.sum())
                    if population.day_valid.sum() else np.nan)
        for field in VALUES:
            item[field+'_own_mean_decimal'] = float(group.loc[group.day_valid, field+'_spread_decimal'].mean())
            item[field+'_global_contribution_decimal'] = item[field+'_own_mean_decimal']*item['global_valid_date_weight']
        fold_rows.append(item)
    output['fold_identity'] = pd.DataFrame(fold_rows)
    output['band_identity_summary'] = bands.loc[bands.day_valid].groupby(['fold', 'band'], sort=True)[
        [x+'_spread_decimal' for x in VALUES]].mean().reset_index()
    output['validity_partition'] = daily.groupby(['fold', 'day_valid', 'invalid_reason_signature'],
        sort=True).size().rename('date_count').reset_index()
    output['prior_cells'], output['prior_identifiability_cells'] = prior_tables(enriched, daily, tolerance)
    summary = dict(status='DESCRIPTIVE_AUDIT_COMPLETE_PENDING_INDEPENDENT_REVIEW',
        units='Decimal gross close-close H2 returns and frozen M0 predicted return; bp=decimal*10000',
        populations=summaries, fit_calls=0, predict_calls=0, model_loads=0, ci_calls=0,
        evidence_level='DIRECTION_ONLY', scientific_information_gate='INCONCLUSIVE_UNCHANGED',
        S1_status='S1_DEVELOPMENT_GATE_FAILED', S2_started=False,
        primary_calendar_dates=len(primary), primary_valid_dates=int(primary.day_valid.sum()),
        primary_invalid_dates=int((~primary.day_valid).sum()),
        prior_cell_count=len(output['prior_cells']),
        prior_all_cells_constant=bool(output['prior_cells'].prior_distinct_values.eq(1).all()),
        mean_errors_are_descriptive_not_calibration_proof=True,
        limits=['Exposed history only; no causal or conditional predictive conclusion',
                'Current industry retrospective; historical PIT/calendar/universe not certified',
                'Gross outcome, not executable net return; no sealed validation',
                'Repeated codes/cells are not independent observations'])
    return output, summary
