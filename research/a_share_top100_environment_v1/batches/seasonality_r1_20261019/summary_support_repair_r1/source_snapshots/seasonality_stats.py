"""Deterministic seasonality and industry-position summaries for A-share Top100.

All measures are descriptive. Industry labels are the bound current
retrospective classification, and no result is a prediction, causal estimate,
trading rule, or sealed-forward validation.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


EVENT_INDUSTRY_CODES = ("B06", "D44", "D45", "H61", "H62", "G56", "N78", "L72")
EVENT_PROXY_CODES = frozenset(("N78", "L72"))
EVENT_ANCHORS = {"MAY_DAY": (5, 1), "NATIONAL_DAY": (10, 1)}
EVENT_BINS = (
    ("pre_20_11", -20, -11),
    ("pre_10_6", -10, -6),
    ("pre_5_1", -5, -1),
    ("post_0_4", 0, 4),
    ("post_5_9", 5, 9),
    ("post_10_19", 10, 19),
)
POSITION_STATES = ("LOW", "MID", "HIGH", "UNKNOWN")
RS_STATES = ("POS", "NONPOS", "UNKNOWN")
MARKET_STATES = ("AT_OR_ABOVE", "BELOW", "UNKNOWN")
FULL_YEARS = (2022, 2023, 2024, 2025)
PARTIAL_YEARS = (2021, 2026)
BOUND_TOP100_ENVIRONMENT_SHA256 = "628ef44b6a8082e1f49179d943e8d1c062916849c9726190d00d2fefa4afba90"
DEFAULT_BATCH = Path("/Users/lexi/Desktop/学习/因子工厂/research/a_share_top100_environment_v1/batches/seasonality_r1_20261019")


class SeasonalityScopeError(RuntimeError):
    """Raised when a frozen grouping or completeness invariant fails."""


def _date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def validate_frozen_configs(config_a_path: Path, config_b_path: Path) -> dict[str, str]:
    """Fail closed if the saved A/B contracts drift from executable constants."""
    import top100_environment as env

    config_a_path, config_b_path = Path(config_a_path), Path(config_b_path)
    config_a = json.loads(config_a_path.read_text(encoding="utf-8"))
    config_b = json.loads(config_b_path.read_text(encoding="utf-8"))
    expected_bins = [{"name": name, "start_offset": start, "end_offset": end} for name, start, end in EVENT_BINS]
    expected_anchors = {name: {"month": month, "day": day} for name, (month, day) in EVENT_ANCHORS.items()}
    expected_a = {
        "schema_version": "A_SHARE_TOP100_SEASONALITY_CONFIG_V1",
        "month_tenday": {"day_1_to_10": 1, "day_11_to_20": 2, "day_21_to_month_end": 3},
        "full_years": list(FULL_YEARS),
        "partial_years": list(PARTIAL_YEARS),
        "event_anchors": expected_anchors,
        "event_bins": expected_bins,
        "event_industry_codes": list(EVENT_INDUSTRY_CODES),
    }
    for key, expected in expected_a.items():
        if config_a.get(key) != expected:
            raise SeasonalityScopeError(f"A config does not match frozen code constant: {key}")
    expected_b = {
        "schema_version": "A_SHARE_TOP100_INDUSTRY_POSITION_CONFIG_V1",
        "position_window_sessions": 252,
        "position_midrank_denominator": 252,
        "position_low_max_inclusive": 0.20,
        "position_high_min_inclusive": 0.80,
        "rs_windows_sessions": [20, 60],
        "rs_positive_rule": "COMPOUND_INDUSTRY_RETURN_MINUS_COMPOUND_POOL_RETURN_GT_ZERO",
        "state_cutoff": "T_MINUS_1_ONLY",
    }
    for key, expected in expected_b.items():
        if config_b.get(key) != expected:
            raise SeasonalityScopeError(f"B config does not match frozen code constant: {key}")
    return {
        "config_a_sha256": env.sha256_file(config_a_path),
        "config_b_sha256": env.sha256_file(config_b_path),
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _industry_suffix(value: Any) -> str:
    text = str(value).strip()
    return text.rsplit(":", 1)[-1]


def month_tenday(value: Any) -> int:
    """Return the frozen calendar-day segment: 1-10, 11-20, or 21-month-end."""
    day = _date(value).day
    return 1 if day <= 10 else 2 if day <= 20 else 3


def _strict_selector(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) < 100:
        raise SeasonalityScopeError("eligible Top100 day has fewer than 100 valid rows")
    ranked = group.sort_values(["return", "code"], ascending=[False, True], kind="mergesort")
    selected = ranked.iloc[:100].copy()
    if len(selected) != 100 or selected["code"].astype(str).duplicated().any():
        raise SeasonalityScopeError("strict Top100 is not exactly 100 unique codes")
    return selected


def _call_selector(selector: Callable[[pd.DataFrame], Any], group: pd.DataFrame) -> pd.DataFrame:
    selected = selector(group)
    if isinstance(selected, tuple):
        selected = selected[0]
    if not isinstance(selected, pd.DataFrame):
        raise SeasonalityScopeError("bound Top100 selector did not return a DataFrame")
    if len(selected) != 100 or selected["code"].astype(str).duplicated().any():
        raise SeasonalityScopeError("bound selector failed the exact-100 unique-code contract")
    return selected


def _index_returns(calendar: pd.DataFrame, dates: Sequence[pd.Timestamp],
                   index_close: pd.Series) -> dict[pd.Timestamp, float | None]:
    ordered = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    calendar_dates = pd.DatetimeIndex(pd.to_datetime(ordered["date"], errors="raise")).normalize()
    previous = dict(zip(calendar_dates, [None, *calendar_dates[:-1]]))
    close_map: dict[pd.Timestamp, float] = {}
    for raw_date, raw_value in index_close.items():
        date = _date(raw_date)
        value = _float_or_none(raw_value)
        if date in close_map:
            raise SeasonalityScopeError("bound index has duplicate normalized dates")
        if value is not None:
            close_map[date] = value
    result: dict[pd.Timestamp, float | None] = {}
    for raw_date in dates:
        date = _date(raw_date)
        prev_date = previous.get(date)
        current = close_map.get(date)
        prior = close_map.get(prev_date) if prev_date is not None else None
        if current is None or prior is None or current <= 0 or prior <= 0:
            result[date] = None
        else:
            value = current / prior - 1.0
            result[date] = value if math.isfinite(value) else None
    return result


def build_daily_projection(
    observations: pd.DataFrame,
    calendar: pd.DataFrame,
    index_close: pd.Series,
    *,
    selector: Callable[[pd.DataFrame], Any] | None = None,
) -> pd.DataFrame:
    """Build the single traceable date x industry projection used by A and B."""
    required = {"date", "code", "return", "pool_valid", "industry_code", "industry_name", "lag_market"}
    missing = required - set(observations.columns)
    if missing:
        raise SeasonalityScopeError(f"observations lack required columns: {sorted(missing)}")
    if not {"date", "calendar_index"}.issubset(calendar.columns):
        raise SeasonalityScopeError("shared calendar lacks date/calendar_index")
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"], errors="raise").dt.normalize()
    obs["code"] = obs["code"].astype(str)
    obs["industry_code"] = obs["industry_code"].fillna("UNKNOWN").astype(str).str.strip()
    obs.loc[obs["industry_code"].eq(""), "industry_code"] = "UNKNOWN"
    if obs.duplicated(["date", "code"]).any():
        raise SeasonalityScopeError("observations contain duplicate date/code keys")
    valid = obs.loc[obs["pool_valid"].astype(bool)].copy()
    if valid.empty:
        raise SeasonalityScopeError("no valid same-day observed pool rows")
    valid_returns = pd.to_numeric(valid["return"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(valid_returns).all():
        raise SeasonalityScopeError("valid pool rows include nonfinite returns")
    valid["return"] = valid_returns

    ordered_calendar = calendar.copy()
    ordered_calendar["date"] = pd.to_datetime(ordered_calendar["date"], errors="raise").dt.normalize()
    if ordered_calendar["date"].duplicated().any() or ordered_calendar["calendar_index"].duplicated().any():
        raise SeasonalityScopeError("shared calendar date and ordinal must be unique")
    ordered_calendar = ordered_calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    first, last = obs["date"].min(), obs["date"].max()
    dates = [_date(d) for d in ordered_calendar.loc[ordered_calendar["date"].between(first, last), "date"]]
    if not dates:
        raise SeasonalityScopeError("no shared-calendar dates overlap observed panel")
    date_index = {date: i for i, date in enumerate(dates)}
    if not set(obs["date"]).issubset(date_index):
        raise SeasonalityScopeError("observed panel contains dates outside projected calendar span")

    categories = sorted(set(obs["industry_code"].astype(str)) | {"UNKNOWN"})
    if not categories:
        raise SeasonalityScopeError("industry category set is empty")
    name_rows = obs.loc[obs["industry_name"].notna(), ["industry_code", "industry_name"]].copy()
    name_rows["industry_name"] = name_rows["industry_name"].astype(str).str.strip()
    name_rows = name_rows.loc[~name_rows["industry_name"].eq("")].drop_duplicates()
    name_conflicts = name_rows.groupby("industry_code", sort=False)["industry_name"].nunique()
    if (name_conflicts > 1).any():
        raise SeasonalityScopeError("industry name conflicts for a category code")
    name_map = dict(zip(name_rows["industry_code"], name_rows["industry_name"]))

    obs_market = obs.groupby("date", sort=False)["lag_market"].agg(lambda s: set(s.astype(str)))
    lag_map: dict[pd.Timestamp, str] = {}
    for raw_date, values in obs_market.items():
        if len(values) != 1:
            raise SeasonalityScopeError(f"lag_market is not date-constant at {_date(raw_date).date()}")
        lag_map[_date(raw_date)] = next(iter(values))
    same_day_index = _index_returns(ordered_calendar, dates, index_close)

    grouped = { _date(d): frame for d, frame in valid.groupby("date", sort=False, observed=True) }
    select = selector or _strict_selector
    records: list[dict[str, Any]] = []
    for date in dates:
        pool = grouped.get(date)
        pool_n = 0 if pool is None else int(len(pool))
        eligible = pool_n >= 100
        pool_mean = float(pool["return"].mean()) if pool_n else None
        if pool is None:
            by_category: dict[str, pd.DataFrame] = {}
        else:
            by_category = {str(code): group for code, group in pool.groupby("industry_code", sort=False, observed=True)}
        if eligible:
            selected = _call_selector(select, pool)
            selected_by_category = {str(code): group for code, group in selected.groupby("industry_code", sort=False, observed=True)}
            hit_sum = sum(len(frame) for frame in selected_by_category.values())
            if hit_sum != 100:
                raise SeasonalityScopeError("Top100 industry hit cells do not sum to 100")
        else:
            selected_by_category = {}
        date_market = lag_map.get(date, "UNKNOWN")
        if date_market not in MARKET_STATES:
            date_market = "UNKNOWN"
        row_total_n = 0
        row_total_hits = 0
        row_total_expected = 0.0
        for category in categories:
            frame = by_category.get(category)
            n = 0 if frame is None else int(len(frame))
            row_total_n += n
            if n:
                values = frame["return"].to_numpy(dtype=np.float64)
                return_sum: float | None = float(np.sum(values, dtype=np.float64))
                return_square_sum: float | None = float(np.dot(values, values))
                industry_mean: float | None = float(np.mean(values, dtype=np.float64))
                industry_median: float | None = float(np.median(values))
                sample_std: float | None = float(np.std(values, ddof=1, dtype=np.float64)) if n > 1 else None
                daily_excess = industry_mean - pool_mean if pool_mean is not None else None
            else:
                # Empty industry cells carry no return statistic, never a zero return.
                return_sum = return_square_sum = industry_mean = industry_median = sample_std = daily_excess = None
            if eligible:
                hits_frame = selected_by_category.get(category)
                hits = 0 if hits_frame is None else int(len(hits_frame))
                expected = 100.0 * n / pool_n
                top_return_sum = 0.0 if hits_frame is None else float(hits_frame["return"].sum())
                row_total_hits += hits
                row_total_expected += expected
            else:
                hits = expected = top_return_sum = None
            records.append({
                "date": date,
                "calendar_position": date_index[date],
                "year": date.year,
                "month": date.month,
                "ten_day": month_tenday(date),
                "industry_code": category,
                "industry_name": name_map.get(category),
                "industry_n": n,
                "industry_return_sum": return_sum,
                "industry_return_square_sum": return_square_sum,
                "industry_mean_return": industry_mean,
                "industry_median_return": industry_median,
                "industry_sample_std_return": sample_std,
                "industry_excess_vs_pool": daily_excess,
                "top100_hits": hits,
                "expected_top100_hits": expected,
                "top100_return_sum": top_return_sum,
                "date_fullpool_n": pool_n,
                "date_pool_mean_return": pool_mean,
                "same_day_000300_return": same_day_index.get(date),
                "lag_market": date_market,
                "eligible_top100_date": bool(eligible),
            })
        if row_total_n != pool_n:
            raise SeasonalityScopeError(f"industry N does not sum to daily pool at {date.date()}")
        if eligible and (row_total_hits != 100 or not math.isclose(row_total_expected, 100.0, rel_tol=0, abs_tol=1e-9)):
            raise SeasonalityScopeError(f"daily Top100 hits/expected do not equal 100 at {date.date()}")

    result = pd.DataFrame.from_records(records)
    return result.sort_values(["date", "industry_code"], kind="mergesort").reset_index(drop=True)


def _mean_nonnull(values: pd.Series) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    return float(cleaned.mean()) if not cleaned.empty else None


def _sum_nonnull(values: pd.Series) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    return float(cleaned.sum()) if not cleaned.empty else 0.0


def _scope_frames(projection: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("FULL_YEARS_2022_2025", projection.loc[projection["year"].isin(FULL_YEARS)]),
        *((f"YEAR_{year}", projection.loc[projection["year"].eq(year)]) for year in FULL_YEARS),
        *((f"PARTIAL_YEAR_{year}", projection.loc[projection["year"].eq(year)]) for year in PARTIAL_YEARS),
    ]


def summarize_month_tenday(projection: pd.DataFrame) -> pd.DataFrame:
    """Full industry x month x ten-day grid and year-separated summaries."""
    categories = sorted(projection["industry_code"].astype(str).unique())
    names = projection.drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()
    rows: list[dict[str, Any]] = []
    for scope_name, scope in _scope_frames(projection):
        scope_years = tuple(FULL_YEARS) if scope_name == "FULL_YEARS_2022_2025" else tuple(
            [int(scope_name.split("_")[-1])] if scope_name.startswith(("YEAR_", "PARTIAL_YEAR_")) else []
        )
        for category in categories:
            for month in range(1, 13):
                for ten_day in range(1, 4):
                    cell = scope.loc[
                        scope["industry_code"].eq(category) & scope["month"].eq(month) & scope["ten_day"].eq(ten_day)
                    ]
                    nonempty = cell.loc[cell["industry_n"].gt(0)]
                    selected = cell.loc[cell["eligible_top100_date"]]
                    n_total = int(cell["industry_n"].sum())
                    sum_return = _sum_nonnull(cell["industry_return_sum"])
                    sum_square = _sum_nonnull(cell["industry_return_square_sum"])
                    expected = _sum_nonnull(selected["expected_top100_hits"])
                    hits = int(_sum_nonnull(selected["top100_hits"]))
                    rows.append({
                        "period_scope": scope_name,
                        "years_in_scope": ",".join(map(str, scope_years)),
                        "industry_code": category,
                        "industry_name": names.get(category),
                        "month": month,
                        "ten_day": ten_day,
                        "calendar_session_count": int(cell["date"].nunique()),
                        "top100_eligible_date_count": int(selected["date"].nunique()),
                        "industry_support_date_count": int(nonempty["date"].nunique()),
                        "industry_n_sum": n_total,
                        "industry_return_sum": sum_return if n_total else None,
                        "industry_return_square_sum": sum_square if n_total else None,
                        "row_weighted_industry_mean_return": sum_return / n_total if n_total else None,
                        "mean_daily_industry_return_equal_date": _mean_nonnull(nonempty["industry_mean_return"]),
                        "mean_daily_industry_median_equal_date": _mean_nonnull(nonempty["industry_median_return"]),
                        "mean_daily_pool_return_equal_date": _mean_nonnull(nonempty["date_pool_mean_return"]),
                        "mean_daily_excess_equal_date": _mean_nonnull(nonempty["industry_excess_vs_pool"]),
                        "mean_daily_crosssection_sample_std": _mean_nonnull(nonempty["industry_sample_std_return"]),
                        "top100_hits_sum": hits,
                        "expected_top100_hits_sum": expected if int(selected["date"].nunique()) else None,
                        "top100_lift": hits / expected if expected > 0 else None,
                        "weighting_for_daily_return_metrics": "EQUAL_DATE",
                        "classification_basis": "CURRENT_RETROSPECTIVE",
                    })
    return pd.DataFrame(rows)


def _event_dates(calendar: pd.DataFrame, year: int, month: int, day: int) -> tuple[pd.Timestamp | None, int | None]:
    ordered = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(ordered["date"], errors="raise")).normalize()
    anchor = pd.Timestamp(year=year, month=month, day=day)
    pos = int(dates.searchsorted(anchor, side="left"))
    if pos >= len(dates):
        return None, None
    return _date(dates[pos]), pos


def _daily_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    supported = rows.loc[rows["industry_n"].gt(0)]
    eligible = rows.loc[rows["eligible_top100_date"]]
    hits = int(_sum_nonnull(eligible["top100_hits"]))
    expected = _sum_nonnull(eligible["expected_top100_hits"])
    return {
        "support_date_count": int(supported["date"].nunique()),
        "eligible_top100_date_count": int(eligible["date"].nunique()),
        "industry_n_sum": int(rows["industry_n"].sum()),
        "mean_industry_return_equal_date": _mean_nonnull(supported["industry_mean_return"]),
        "mean_industry_excess_equal_date": _mean_nonnull(supported["industry_excess_vs_pool"]),
        "mean_crosssection_sample_std_equal_date": _mean_nonnull(supported["industry_sample_std_return"]),
        "top100_hits_sum": hits,
        "expected_top100_hits_sum": expected if int(eligible["date"].nunique()) else None,
        "top100_lift": hits / expected if expected > 0 else None,
    }


def build_event_windows(projection: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all fixed May/October event bins and their same-year/state baselines."""
    ordered_calendar = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    calendar_dates = [_date(v) for v in ordered_calendar["date"]]
    projection_by_date = { _date(d): f for d, f in projection.groupby("date", sort=False, observed=True) }
    years = range(min(d.year for d in calendar_dates), max(d.year for d in calendar_dates) + 1)
    event_windows: dict[tuple[int, str, str], list[pd.Timestamp]] = {}
    event_meta: dict[tuple[int, str], tuple[pd.Timestamp | None, int | None]] = {}
    all_window_dates_by_year: dict[int, set[pd.Timestamp]] = {year: set() for year in years}
    for year in years:
        for event, (month, day) in EVENT_ANCHORS.items():
            anchor_date, anchor_pos = _event_dates(ordered_calendar, year, month, day)
            event_meta[(year, event)] = (anchor_date, anchor_pos)
            for bin_name, lo, hi in EVENT_BINS:
                positions = list(range(anchor_pos + lo, anchor_pos + hi + 1)) if anchor_pos is not None else []
                bounded = [calendar_dates[p] for p in positions if 0 <= p < len(calendar_dates)]
                event_windows[(year, event, bin_name)] = bounded
                all_window_dates_by_year[year].update(bounded)

    categories = sorted(projection["industry_code"].astype(str).unique())
    names = projection.drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()
    selected_categories = [c for c in categories if _industry_suffix(c) in EVENT_INDUSTRY_CODES]
    rows: list[dict[str, Any]] = []
    market_states = MARKET_STATES
    for year in years:
        for event in EVENT_ANCHORS:
            anchor_date, anchor_pos = event_meta[(year, event)]
            for bin_name, lo, hi in EVENT_BINS:
                window_dates = event_windows[(year, event, bin_name)]
                expected_session_count = hi - lo + 1
                actual_projection_rows = [projection_by_date.get(d) for d in window_dates]
                actual_eligible = sum(
                    bool(frame["eligible_top100_date"].iloc[0]) if frame is not None and len(frame) else False
                    for frame in actual_projection_rows
                )
                if anchor_pos is None:
                    reason = "ANCHOR_AFTER_BOUND_CALENDAR"
                elif len(window_dates) != expected_session_count:
                    reason = "WINDOW_EXTENDS_OUTSIDE_BOUND_CALENDAR"
                elif any(frame is None for frame in actual_projection_rows):
                    reason = "NO_PROJECTION_FOR_SHARED_SESSION"
                elif actual_eligible != expected_session_count:
                    reason = "TOP100_INELIGIBLE_SESSION_IN_WINDOW"
                else:
                    reason = None
                status = "COMPLETE" if reason is None else "INCOMPLETE"
                for category in selected_categories:
                    for market in market_states:
                        window_rows = []
                        for frame in actual_projection_rows:
                            if frame is not None:
                                match = frame.loc[frame["industry_code"].eq(category) & frame["lag_market"].eq(market)]
                                if not match.empty:
                                    window_rows.extend(match.to_dict("records"))
                        window_frame = pd.DataFrame(window_rows, columns=projection.columns)
                        window_metrics = _daily_metrics(window_frame) if not window_frame.empty else _daily_metrics(projection.iloc[0:0])
                        baseline_rows = projection.loc[
                            projection["year"].eq(year)
                            & projection["industry_code"].eq(category)
                            & projection["lag_market"].eq(market)
                            & ~projection["date"].isin(all_window_dates_by_year[year])
                        ]
                        baseline_metrics = _daily_metrics(baseline_rows)
                        window_lift = window_metrics["top100_lift"]
                        baseline_lift = baseline_metrics["top100_lift"]
                        window_excess = window_metrics["mean_industry_excess_equal_date"]
                        baseline_excess = baseline_metrics["mean_industry_excess_equal_date"]
                        rows.append({
                            "year": year,
                            "year_scope": "FULL_YEAR" if year in FULL_YEARS else "PARTIAL_YEAR",
                            "event": event,
                            "anchor_date": anchor_date,
                            "anchor_session_position": anchor_pos,
                            "anchor_semantics": "FIRST_BOUND_SHARED_CALENDAR_DATE_ON_OR_AFTER_FIXED_ANCHOR",
                            "window_bin": bin_name,
                            "offset_start_inclusive": lo,
                            "offset_end_inclusive": hi,
                            "industry_code": category,
                            "industry_name": names.get(category),
                            "industry_role": "MIXED_PROXY" if _industry_suffix(category) in EVENT_PROXY_CODES else "PRIMARY_EVENT_INDUSTRY",
                            "lag_market": market,
                            "expected_session_count": expected_session_count,
                            "bounded_session_count": len(window_dates),
                            "top100_eligible_window_session_count": actual_eligible,
                            "window_status": status,
                            "missing_reason": reason,
                            **{f"window_{k}": v for k, v in window_metrics.items()},
                            "baseline_outside_all_registered_event_windows": True,
                            **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
                            "lift_difference_window_minus_baseline": (
                                float(window_lift - baseline_lift)
                                if window_lift is not None and baseline_lift is not None else None
                            ),
                            "excess_difference_window_minus_baseline": (
                                float(window_excess - baseline_excess)
                                if window_excess is not None and baseline_excess is not None else None
                            ),
                            "inference": "DESCRIPTIVE_ONLY_NO_PVALUE_CI_OR_CAUSAL_CLAIM",
                        })
    detail = pd.DataFrame(rows)
    common_rows: list[dict[str, Any]] = []
    if not detail.empty:
        for (event, category, market), group in detail.groupby(["event", "industry_code", "lag_market"], sort=True):
            complete_years_by_bin: dict[str, set[int]] = {}
            for bin_name, _, _ in EVENT_BINS:
                subset = group.loc[group["window_bin"].eq(bin_name)]
                complete_years_by_bin[bin_name] = set(
                    subset.loc[subset["window_status"].eq("COMPLETE") & subset["baseline_support_date_count"].gt(0), "year"].astype(int)
                )
            common_years = set.intersection(*complete_years_by_bin.values()) if complete_years_by_bin else set()
            for bin_name, _, _ in EVENT_BINS:
                rows_for_bin = group.loc[group["window_bin"].eq(bin_name) & group["year"].isin(common_years)]
                common_rows.append({
                    "event": event,
                    "industry_code": category,
                    "industry_name": names.get(category),
                    "industry_role": "MIXED_PROXY" if _industry_suffix(category) in EVENT_PROXY_CODES else "PRIMARY_EVENT_INDUSTRY",
                    "lag_market": market,
                    "window_bin": bin_name,
                    "common_complete_year_count": len(common_years),
                    "common_complete_years": ",".join(map(str, sorted(common_years))),
                    "mean_annual_window_lift_equal_year": _mean_nonnull(rows_for_bin["window_top100_lift"]),
                    "mean_annual_baseline_lift_equal_year": _mean_nonnull(rows_for_bin["baseline_top100_lift"]),
                    "mean_annual_lift_difference_equal_year": _mean_nonnull(rows_for_bin["lift_difference_window_minus_baseline"]),
                    "mean_annual_excess_difference_equal_year": _mean_nonnull(rows_for_bin["excess_difference_window_minus_baseline"]),
                    "complete_year_window_support": int(rows_for_bin["window_eligible_top100_date_count"].sum()) if not rows_for_bin.empty else 0,
                    "inference": "COMMON_COMPLETE_YEARS_DESCRIPTIVE_ONLY",
                })
    return detail, pd.DataFrame(common_rows)


def _midrank_state(values: np.ndarray, current: float) -> tuple[float | None, str]:
    if len(values) != 252 or not np.isfinite(values).all() or not _finite(current):
        return None, "UNKNOWN"
    lower = int(np.count_nonzero(values < current))
    equal = int(np.count_nonzero(values == current))
    rank = (lower + 0.5 * equal) / 252.0
    if rank <= 0.20:
        state = "LOW"
    elif rank >= 0.80:
        state = "HIGH"
    else:
        state = "MID"
    return float(rank), state


def _relative_compound(industry_returns: np.ndarray, pool_returns: np.ndarray, window: int) -> tuple[float | None, str]:
    if len(industry_returns) != window or len(pool_returns) != window:
        return None, "UNKNOWN"
    if not np.isfinite(industry_returns).all() or not np.isfinite(pool_returns).all():
        return None, "UNKNOWN"
    industry_compound = float(np.prod(1.0 + industry_returns, dtype=np.float64) - 1.0)
    pool_compound = float(np.prod(1.0 + pool_returns, dtype=np.float64) - 1.0)
    difference = industry_compound - pool_compound
    if not math.isfinite(difference):
        return None, "UNKNOWN"
    return difference, "POS" if difference > 0 else "NONPOS"


def build_position_states(projection: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Compute T states only from contiguous shared-calendar values through T-1."""
    ordered_calendar = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    calendar_dates = [_date(v) for v in ordered_calendar["date"]]
    projection_dates = set(pd.to_datetime(projection["date"], errors="raise").dt.normalize())
    start = min(projection_dates)
    end = max(projection_dates)
    dates = [d for d in calendar_dates if start <= d <= end]
    if not dates:
        raise SeasonalityScopeError("position state has no calendar dates")
    records: list[dict[str, Any]] = []
    for category, raw in projection.groupby("industry_code", sort=True, observed=True):
        frame = raw.set_index(pd.to_datetime(raw["date"]).dt.normalize()).reindex(dates)
        industry_returns = pd.to_numeric(frame["industry_mean_return"], errors="coerce").to_numpy(dtype=np.float64)
        pool_returns = pd.to_numeric(frame["date_pool_mean_return"], errors="coerce").to_numpy(dtype=np.float64)
        industry_n = pd.to_numeric(frame["industry_n"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        levels = np.full(len(dates), np.nan, dtype=np.float64)
        segment_ids = np.full(len(dates), -1, dtype=np.int64)
        segment = -1
        last_valid_pos = -2
        last_level = np.nan
        for position, (n, daily_return) in enumerate(zip(industry_n, industry_returns)):
            if n <= 0 or not np.isfinite(daily_return):
                last_valid_pos = -2
                last_level = np.nan
                continue
            if last_valid_pos == position - 1 and np.isfinite(last_level):
                level = last_level * (1.0 + float(daily_return))
            else:
                segment += 1
                level = 100.0
            levels[position] = level
            segment_ids[position] = segment
            last_valid_pos = position
            last_level = level

        for position, date in enumerate(dates):
            if position < len(frame):
                base = frame.iloc[position].to_dict()
            else:
                base = {}
            state_asof_date = dates[position - 1] if position > 0 else None
            if position >= 252:
                level_window = levels[position - 252:position]
                segment_window = segment_ids[position - 252:position]
                contiguous = np.isfinite(level_window).all() and len(set(segment_window.tolist())) == 1 and segment_window[0] >= 0
                if contiguous:
                    percentile, position_state = _midrank_state(level_window, float(level_window[-1]))
                else:
                    percentile, position_state = None, "UNKNOWN"
            else:
                percentile, position_state = None, "UNKNOWN"
            rs20, rs20_state = _relative_compound(
                industry_returns[max(0, position - 20):position],
                pool_returns[max(0, position - 20):position],
                20,
            )
            rs60, rs60_state = _relative_compound(
                industry_returns[max(0, position - 60):position],
                pool_returns[max(0, position - 60):position],
                60,
            )
            records.append({
                "date": date,
                "year": date.year,
                "industry_code": str(category),
                "industry_name": base.get("industry_name"),
                "industry_n": int(industry_n[position]),
                "industry_mean_return": _float_or_none(industry_returns[position]),
                "date_pool_mean_return": _float_or_none(base.get("date_pool_mean_return")),
                "industry_excess_vs_pool": _float_or_none(base.get("industry_excess_vs_pool")),
                "industry_sample_std_return": _float_or_none(base.get("industry_sample_std_return")),
                "top100_hits": _float_or_none(base.get("top100_hits")),
                "expected_top100_hits": _float_or_none(base.get("expected_top100_hits")),
                "eligible_top100_date": bool(base.get("eligible_top100_date", False)),
                "lag_market": str(base.get("lag_market", "UNKNOWN")),
                "industry_equal_weight_level": _float_or_none(levels[position]),
                "level_segment_id": int(segment_ids[position]) if segment_ids[position] >= 0 else None,
                "position_history_n": int(
                    sum(
                        1 for j in range(position - 1, max(-1, position - 253), -1)
                        if np.isfinite(levels[j]) and segment_ids[j] == segment_ids[position - 1]
                    ) if position > 0 and segment_ids[position - 1] >= 0 else 0
                ),
                "position_midrank_asof_t_minus_1": percentile,
                "position_state_asof_t_minus_1": position_state,
                "position_state_asof_date": state_asof_date,
                "rs20_difference_asof_t_minus_1": rs20,
                "rs20_state_asof_t_minus_1": rs20_state,
                "rs20_asof_date": state_asof_date,
                "rs20_support_returns": int(
                    np.count_nonzero(
                        np.isfinite(industry_returns[max(0, position - 20):position])
                        & np.isfinite(pool_returns[max(0, position - 20):position])
                    )
                ),
                "rs60_difference_asof_t_minus_1": rs60,
                "rs60_state_asof_t_minus_1": rs60_state,
                "rs60_asof_date": state_asof_date,
                "rs60_support_returns": int(
                    np.count_nonzero(
                        np.isfinite(industry_returns[max(0, position - 60):position])
                        & np.isfinite(pool_returns[max(0, position - 60):position])
                    )
                ),
                "state_timing": "T_STATE_USES_ONLY_SHARED_SESSIONS_THROUGH_T_MINUS_1",
            })
    result = pd.DataFrame(records)
    result = result.sort_values(["date", "industry_code"], kind="mergesort").reset_index(drop=True)
    known_pos = result.loc[result["position_state_asof_t_minus_1"].ne("UNKNOWN")]
    if not (known_pos["position_state_asof_date"] < known_pos["date"]).all():
        raise SeasonalityScopeError("position state contains same-day or future level information")
    for field in ("rs20_asof_date", "rs60_asof_date"):
        known = result.loc[result[field].notna()]
        if not (known[field] < known["date"]).all():
            raise SeasonalityScopeError(f"{field} contains same-day or future return information")
    return result


def _group_performance(rows: pd.DataFrame) -> dict[str, Any]:
    supported = rows.loc[rows["industry_n"].gt(0)]
    eligible = rows.loc[rows["eligible_top100_date"]]
    hits = int(_sum_nonnull(eligible["top100_hits"]))
    expected = _sum_nonnull(eligible["expected_top100_hits"])
    return {
        "date_count": int(rows["date"].nunique()),
        "industry_support_date_count": int(supported["date"].nunique()),
        "eligible_top100_date_count": int(eligible["date"].nunique()),
        "industry_n_sum": int(rows["industry_n"].sum()),
        "mean_industry_return_equal_date": _mean_nonnull(supported["industry_mean_return"]),
        "mean_industry_excess_equal_date": _mean_nonnull(supported["industry_excess_vs_pool"]),
        "mean_crosssection_sample_std_equal_date": _mean_nonnull(supported["industry_sample_std_return"]),
        "top100_hits_sum": hits,
        "expected_top100_hits_sum": expected if int(eligible["date"].nunique()) else None,
        "top100_lift": hits / expected if expected > 0 else None,
    }


def summarize_position_states(position_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = sorted(position_daily["industry_code"].astype(str).unique())
    names = position_daily.drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()
    rows: list[dict[str, Any]] = []
    rs60_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", position_daily)] + [
        (f"YEAR_{int(y)}", position_daily.loc[position_daily["year"].eq(y)])
        for y in sorted(position_daily["year"].unique())
    ]
    for scope_name, scope in scopes:
        for category in categories:
            for position_state in POSITION_STATES:
                for rs_state in RS_STATES:
                    for market in MARKET_STATES:
                        group = scope.loc[
                            scope["industry_code"].eq(category)
                            & scope["position_state_asof_t_minus_1"].eq(position_state)
                            & scope["rs20_state_asof_t_minus_1"].eq(rs_state)
                            & scope["lag_market"].eq(market)
                        ]
                        rows.append({
                            "period_scope": scope_name,
                            "industry_code": category,
                            "industry_name": names.get(category),
                            "industry_price_position": position_state,
                            "rs20_state": rs_state,
                            "lag_market": market,
                            **_group_performance(group),
                            "state_timing": "T_STATE_USES_ONLY_SHARED_SESSIONS_THROUGH_T_MINUS_1",
                        })
            for rs_state in RS_STATES:
                for market in MARKET_STATES:
                    group = scope.loc[
                        scope["industry_code"].eq(category)
                        & scope["rs60_state_asof_t_minus_1"].eq(rs_state)
                        & scope["lag_market"].eq(market)
                    ]
                    rs60_rows.append({
                        "period_scope": scope_name,
                        "industry_code": category,
                        "industry_name": names.get(category),
                        "rs60_state": rs_state,
                        "lag_market": market,
                        **_group_performance(group),
                        "purpose": "INDEPENDENT_RS60_SENSITIVITY_NO_POSITION_CROSS",
                    })
    return pd.DataFrame(rows), pd.DataFrame(rs60_rows)


def latest_asof_close_states(position_daily: pd.DataFrame, *, asof_date: Any) -> pd.DataFrame:
    """Return a close-through-T snapshot; it is not a pre-open state for T."""
    target = _date(asof_date)
    rows: list[dict[str, Any]] = []
    for category, raw in position_daily.groupby("industry_code", sort=True, observed=True):
        frame = raw.sort_values("date", kind="mergesort").reset_index(drop=True)
        matched = frame.loc[pd.to_datetime(frame["date"]).dt.normalize().eq(target)]
        if matched.empty:
            rows.append({
                "asof_date": target,
                "industry_code": str(category),
                "industry_name": frame["industry_name"].dropna().iloc[0] if frame["industry_name"].notna().any() else None,
                "level": None,
                "position_midrank_asof_close": None,
                "position_state_asof_close": "UNKNOWN",
                "rs20_difference_asof_close": None,
                "rs20_state_asof_close": "UNKNOWN",
                "rs60_difference_asof_close": None,
                "rs60_state_asof_close": "UNKNOWN",
                "state_timing": "ASOF_CLOSE_T_INCLUDED_NOT_OPENING_STATE",
            })
            continue
        pos = int(matched.index[-1])
        levels = pd.to_numeric(frame["industry_equal_weight_level"], errors="coerce").to_numpy(dtype=np.float64)
        # The latest row itself is included because this snapshot is explicitly after T close.
        level_window = levels[max(0, pos - 251):pos + 1]
        segment_ids = pd.to_numeric(frame["level_segment_id"], errors="coerce").to_numpy(dtype=np.float64)
        segment_window = segment_ids[max(0, pos - 251):pos + 1]
        if len(level_window) == 252 and np.isfinite(level_window).all() and len(set(segment_window.tolist())) == 1:
            percentile, position_state = _midrank_state(level_window, float(level_window[-1]))
        else:
            percentile, position_state = None, "UNKNOWN"
        ir = pd.to_numeric(frame["industry_mean_return"], errors="coerce").to_numpy(dtype=np.float64)
        pr = pd.to_numeric(frame["date_pool_mean_return"], errors="coerce").to_numpy(dtype=np.float64) if "date_pool_mean_return" in frame else np.full(len(frame), np.nan)
        rs20, state20 = _relative_compound(ir[max(0, pos - 19):pos + 1], pr[max(0, pos - 19):pos + 1], 20)
        rs60, state60 = _relative_compound(ir[max(0, pos - 59):pos + 1], pr[max(0, pos - 59):pos + 1], 60)
        row = matched.iloc[0]
        rows.append({
            "asof_date": target,
            "industry_code": str(category),
            "industry_name": row["industry_name"],
            "industry_n": int(row["industry_n"]),
            "level": _float_or_none(row["industry_equal_weight_level"]),
            "position_history_n": len(level_window),
            "position_midrank_asof_close": percentile,
            "position_state_asof_close": position_state,
            "rs20_difference_asof_close": rs20,
            "rs20_state_asof_close": state20,
            "rs20_support_returns": min(pos + 1, 20),
            "rs60_difference_asof_close": rs60,
            "rs60_state_asof_close": state60,
            "rs60_support_returns": min(pos + 1, 60),
            "state_timing": "ASOF_CLOSE_T_INCLUDED_NOT_OPENING_STATE",
        })
    return pd.DataFrame(rows)


def summarize_same_day_index_linkage(projection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = sorted(projection["industry_code"].astype(str).unique())
    names = projection.drop_duplicates("industry_code").set_index("industry_code")["industry_name"].to_dict()
    correlations: list[dict[str, Any]] = []
    up_nonup: list[dict[str, Any]] = []
    for category in categories:
        frame = projection.loc[projection["industry_code"].eq(category)].sort_values("date", kind="mergesort")
        paired = frame.loc[frame["industry_mean_return"].notna() & frame["same_day_000300_return"].notna()]
        rho = None
        if (
            len(paired) >= 2
            and paired["industry_mean_return"].nunique(dropna=True) > 1
            and paired["same_day_000300_return"].nunique(dropna=True) > 1
        ):
            value = pd.to_numeric(paired["industry_mean_return"], errors="coerce").corr(
                pd.to_numeric(paired["same_day_000300_return"], errors="coerce"), method="spearman"
            )
            rho = _float_or_none(value)
        correlations.append({
            "industry_code": category,
            "industry_name": names.get(category),
            "paired_day_count": int(len(paired)),
            "same_day_industry_vs_000300_spearman_rho": rho,
            "inference": "SAME_DAY_DESCRIPTIVE_LINKAGE_ONLY_NO_REGRESSION_OR_CAUSAL_CLAIM",
        })
        frame = frame.loc[frame["same_day_000300_return"].notna()].copy()
        frame["index_return_sign"] = np.where(frame["same_day_000300_return"].gt(0), "UP", "NONUP")
        for sign in ("UP", "NONUP"):
            group = frame.loc[frame["index_return_sign"].eq(sign)]
            up_nonup.append({
                "industry_code": category,
                "industry_name": names.get(category),
                "index_return_sign": sign,
                "paired_day_count": int(group["date"].nunique()),
                "industry_support_day_count": int(group.loc[group["industry_n"].gt(0), "date"].nunique()),
                "mean_industry_return_equal_date": _mean_nonnull(group["industry_mean_return"]),
                "mean_industry_excess_equal_date": _mean_nonnull(group["industry_excess_vs_pool"]),
                "mean_top100_lift_equal_date": _mean_nonnull(group.loc[group["eligible_top100_date"], "top100_hits"] / group.loc[group["eligible_top100_date"], "expected_top100_hits"]),
                "inference": "SAME_DAY_DESCRIPTIVE_LINKAGE_ONLY_NO_REGRESSION_OR_CAUSAL_CLAIM",
            })
    return pd.DataFrame(correlations), pd.DataFrame(up_nonup)


def analyze_projection(projection: pd.DataFrame, calendar: pd.DataFrame, *, latest_date: Any = "2026-10-15") -> dict[str, Any]:
    """Create all predeclared A/B descriptive summaries from one projection."""
    seasonality = summarize_month_tenday(projection)
    event_windows, common_years = build_event_windows(projection, calendar)
    position_daily = build_position_states(projection, calendar)
    position_summary, rs60_sensitivity = summarize_position_states(position_daily)
    latest = latest_asof_close_states(position_daily, asof_date=latest_date)
    linkage, up_nonup = summarize_same_day_index_linkage(projection)

    eligible = projection.loc[projection["eligible_top100_date"]].groupby("date", sort=True).agg(
        hits=("top100_hits", "sum"), expected=("expected_top100_hits", "sum"),
    )
    qa = {
        "projection_rows": int(len(projection)),
        "projection_dates": int(projection["date"].nunique()),
        "industry_count_including_unknown": int(projection["industry_code"].nunique()),
        "eligible_top100_date_count": int(projection.loc[projection["eligible_top100_date"], "date"].nunique()),
        "daily_top100_hits_equal_100": bool((eligible["hits"] == 100).all()),
        "daily_expected_hits_equal_100": bool(np.allclose(eligible["expected"].to_numpy(dtype=np.float64), 100.0, rtol=0, atol=1e-9)),
        "full_month_tenday_grid": bool(
            len(seasonality.loc[seasonality["period_scope"].eq("FULL_YEARS_2022_2025")])
            == int(projection["industry_code"].nunique()) * 12 * 3
        ),
        "calendar_bins_predeclared_and_retained": bool(
            set(event_windows["window_bin"].unique()) == {v[0] for v in EVENT_BINS}
        ),
        "position_state_timing_prior_only": bool(
            (position_daily.loc[position_daily["position_state_asof_t_minus_1"].ne("UNKNOWN"), "position_state_asof_date"]
             < position_daily.loc[position_daily["position_state_asof_t_minus_1"].ne("UNKNOWN"), "date"]).all()
        ),
        "rs20_state_timing_prior_only": bool(
            (position_daily.loc[position_daily["rs20_state_asof_t_minus_1"].ne("UNKNOWN"), "rs20_asof_date"]
             < position_daily.loc[position_daily["rs20_state_asof_t_minus_1"].ne("UNKNOWN"), "date"]).all()
        ),
        "rs60_state_timing_prior_only": bool(
            (position_daily.loc[position_daily["rs60_state_asof_t_minus_1"].ne("UNKNOWN"), "rs60_asof_date"]
             < position_daily.loc[position_daily["rs60_state_asof_t_minus_1"].ne("UNKNOWN"), "date"]).all()
        ),
        "zero_n_return_statistics_are_null": bool(
            projection.loc[projection["industry_n"].eq(0), [
                "industry_return_sum", "industry_return_square_sum", "industry_mean_return",
                "industry_median_return", "industry_sample_std_return", "industry_excess_vs_pool",
            ]].isna().all().all()
        ),
        "event_window_complete_count": int(event_windows["window_status"].eq("COMPLETE").sum()),
        "event_window_incomplete_count": int(event_windows["window_status"].eq("INCOMPLETE").sum()),
        "common_complete_year_support_max": int(common_years["common_complete_year_count"].max()) if not common_years.empty else 0,
        "claims": "DIRECTION_ONLY_DESCRIPTIVE; no inference or prospective validity",
        "safety_flags": {
            "research_only": True,
            "production_allowed": False,
            "buy_gate_updated": False,
            "orders_sent": False,
        },
    }
    return {
        "daily_industry_projection": projection,
        "month_tenday": seasonality,
        "event_windows": event_windows,
        "event_common_years": common_years,
        "industry_position_daily": position_daily,
        "industry_position_summary": position_summary,
        "rs60_sensitivity": rs60_sensitivity,
        "latest_asof_close": latest,
        "same_day_index_linkage": linkage,
        "index_up_nonup": up_nonup,
        "qa": qa,
    }


def load_bound_analysis(
    input_binding_path: Path,
    *,
    config_a_path: Path | None = None,
    config_b_path: Path | None = None,
    latest_date: Any = "2026-10-15",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the frozen B0 inputs once via its existing consumer helpers."""
    import top100_environment as env

    binding = json.loads(Path(input_binding_path).read_text(encoding="utf-8"))
    config_sha = validate_frozen_configs(
        config_a_path or DEFAULT_BATCH / "CONFIG_A.json",
        config_b_path or DEFAULT_BATCH / "CONFIG_B.json",
    )
    bound_source_path = Path(__file__).with_name("top100_environment.py")
    actual_bound_source_sha = env.sha256_file(bound_source_path)
    if actual_bound_source_sha != BOUND_TOP100_ENVIRONMENT_SHA256:
        raise SeasonalityScopeError("bound top100_environment.py SHA differs from frozen R1 helper")
    helper = env.load_bound_helpers(binding)
    panel, panel_meta = env.read_frozen_panel(binding, helper)
    calendar, industry, index_close, auxiliary_meta = env.read_frozen_auxiliary_inputs(binding, helper, panel)
    observations, coverage, coverage_year = env.prepare_observations(panel, calendar, industry, index_close)
    projection = build_daily_projection(observations, calendar, index_close, selector=env.select_top100)
    results = analyze_projection(projection, calendar, latest_date=latest_date)
    meta = {
        "panel": panel_meta,
        "auxiliary": auxiliary_meta,
        "coverage_dates": int(len(coverage)),
        "coverage_years": int(len(coverage_year)),
        "source_code_sha256": actual_bound_source_sha,
        "input_binding_sha256": env.sha256_file(Path(input_binding_path)),
        "analysis_code_sha256": env.sha256_file(Path(__file__)),
        **config_sha,
    }
    return results, meta


def render_compact_report(results: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """Render a concise method/coverage report without selecting a winning window."""
    projection = results["daily_industry_projection"]
    qa = results["qa"]
    event_windows = results["event_windows"]
    complete = int(event_windows["window_status"].eq("COMPLETE").sum()) if not event_windows.empty else 0
    incomplete = int(event_windows["window_status"].eq("INCOMPLETE").sum()) if not event_windows.empty else 0
    reasons = (
        event_windows.loc[event_windows["window_status"].eq("INCOMPLETE"), "missing_reason"]
        .fillna("UNKNOWN").value_counts().sort_index().to_dict()
        if not event_windows.empty else {}
    )
    first = _date(projection["date"].min()).date().isoformat()
    last = _date(projection["date"].max()).date().isoformat()
    lines = [
        "# A股日涨幅 Top100：季节性与行业位置描述统计",
        "",
        f"样本日期：{first} 至 {last}；Top100 有效日 {qa['eligible_top100_date_count']} 日。",
        f"行业格数（含 UNKNOWN）：{qa['industry_count_including_unknown']}；日×行业派生投影 {qa['projection_rows']} 行。",
        "",
        "## 固定口径",
        "",
        "- 同日有效池沿用绑定的 QFQ 面板、精确相邻共享日收盘和正成交量条件；收益降序、代码升序选满 100 只。",
        "- 行业分类是当前回溯分类。每个日×行业格保留 N、收益和、平方和、日均值/中位数、Top100 命中与期望命中；N=0 时收益统计为空。",
        "- A 表按自然月 1–10、11–20、21–月底给出全行业完整网格；2022–2025 合并与逐年视图、2021/2026 部分年份分开。",
        "- 五一和国庆锚点分别固定为 5 月 1 日、10 月 1 日在绑定共享日历上的首个不早于锚点的日期；全部预注册前后窗口均保留。窗口基线是同年、同滞后大盘状态下，排除所有预注册事件窗日期后的行业样本。",
        "- 行业等权价格位置链遇缺日或 N=0 即中断并在下一有效日从 100 重启；T 的位置、RS20、RS60 仅使用截至 T−1 的值。RS60 是独立敏感性表。",
        "- 最新状态按收盘后截至日计算，包含该日收盘；不能解释为该日开盘前可用状态。",
        "- 同日 000300 对照只计算 Spearman 与上涨/非上涨日分组；不做回归、显著性检验或因果解释。",
        "",
        "## 完整性",
        "",
        f"Top100 每日命中总数=100：{qa['daily_top100_hits_equal_100']}；期望命中总数=100：{qa['daily_expected_hits_equal_100']}。",
        f"月×旬完整网格：{qa['full_month_tenday_grid']}；T−1 状态时序：position={qa['position_state_timing_prior_only']}、RS20={qa['rs20_state_timing_prior_only']}、RS60={qa['rs60_state_timing_prior_only']}。",
        f"事件窗完整格 {complete}，不完整格 {incomplete}；缺口原因计数：`{json.dumps(reasons, ensure_ascii=False, sort_keys=True)}`。不完整窗保留支持数与原因，不进入共同完整年份比较。",
        "",
        "所有数值均为暴露历史的描述统计，结论上限为 `DIRECTION_ONLY`。这不是预测、策略回测、收益承诺、历史 PIT 行业判断或封存验证。安全状态保持 `research_only=true`、`production_allowed=false`、`buy_gate_updated=false`、`orders_sent=false`。",
        "",
    ]
    if meta:
        lines.extend([
            f"输入绑定 SHA：`{meta.get('input_binding_sha256', 'CANNOT_CONFIRM')}`；分析代码 SHA：`{meta.get('analysis_code_sha256', 'CANNOT_CONFIRM')}`；复用 R1 helper SHA：`{meta.get('source_code_sha256', 'CANNOT_CONFIRM')}`。",
            "",
        ])
    return "\n".join(lines)


def write_outputs(results: Mapping[str, Any], output_dir: Path, *, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write a new immutable output package; refuses to replace any existing output."""
    import top100_environment as env

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"seasonality output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "DAILY_INDUSTRY_PROJECTION.csv": "daily_industry_projection",
        "MONTH_TENDAY_INDUSTRY_STATS.csv": "month_tenday",
        "MAY_OCT_EVENT_WINDOWS.csv": "event_windows",
        "MAY_OCT_COMMON_YEARS.csv": "event_common_years",
        "INDUSTRY_POSITION_DAILY.csv": "industry_position_daily",
        "INDUSTRY_POSITION_STATE_STATS.csv": "industry_position_summary",
        "RS60_SENSITIVITY.csv": "rs60_sensitivity",
        "LATEST_ASOF_CLOSE.csv": "latest_asof_close",
        "SAME_DAY_000300_SPEARMAN.csv": "same_day_index_linkage",
        "INDEX_UP_NONUP.csv": "index_up_nonup",
    }
    paths: list[Path] = []
    for filename, key in names.items():
        path = output_dir / filename
        env.write_csv(path, results[key])
        paths.append(path)
    report_path = output_dir / "COMPACT_REPORT.md"
    env._write_new_or_identical(report_path, render_compact_report(results, meta or {}).encode("utf-8"))
    paths.append(report_path)
    hashes = {path.name: env.sha256_file(path) for path in paths}
    receipt = {
        "schema_version": "A_SHARE_TOP100_SEASONALITY_STATS_RECEIPT_V1",
        "status": "COMPUTED_OUTPUTS_REQUIRE_REGISTERED_LIFECYCLE_AND_INDEPENDENT_REVIEW",
        "output_sha256": hashes,
        "qa": dict(results["qa"]),
        "meta": dict(meta or {}),
        "inference_scope": "DIRECTION_ONLY_DESCRIPTIVE_NO_SEALED_VALIDATION",
        "safety_flags": {
            "research_only": True,
            "production_allowed": False,
            "buy_gate_updated": False,
            "orders_sent": False,
        },
    }
    receipt_path = output_dir / "QA_RECEIPT.json"
    env.write_json(receipt_path, receipt)
    receipt["receipt_sha256"] = env.sha256_file(receipt_path)
    return receipt

