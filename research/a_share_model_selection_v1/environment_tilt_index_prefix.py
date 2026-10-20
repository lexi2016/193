"""Compose a bounded observed CSI 300 index input for historical M0 replay.

This module is a pure projection over already-decoded TDX rows and the existing
Futu CSV rows. It does not read, hash, decode, download, interpolate, rescale, or
compare market inputs. The caller must authenticate the exact bound inputs
after a durable registered RUN_STARTED event and before calling this function.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "ENV_TILT_OBSERVED_INDEX_PREFIX_V1"
SERIES_CODE = "SH.000300"
TDX_START = pd.Timestamp("2005-01-04")
FUTU_START = pd.Timestamp("2021-01-04")
ANALYSIS_CUTOFF = pd.Timestamp("2026-10-15")

TDX_DERIVED_COLUMNS = ("date", "open", "high", "low", "close")
FUTU_CSV_COLUMNS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
)
PRICE_COLUMNS = ("open", "high", "low", "close")

TDX_SOURCE_REF = {
    "source_id": "TDX_OBSERVED_DERIVED_INDEX",
    "path": "/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/history_market_view_20261020/HISTORY_EXPORT_01/INDEX_DECODED_FULL.csv",
    "sha256": "3d9288b7adcbb3ca1227589057b3253899ee3f61811a590ff9ed4718008b335b",
    "upstream_archive_path": "/Users/lexi/Desktop/学习/因子工厂/research/a_share_top100_environment_v1/batches/hsjday_20y_20261019/inputs/hsjday.zip",
    "upstream_archive_sha256": "a56c962378a5d3dd86a90186cc830ee6c8fd968557ce6eddeb310d203c5ff3c0",
    "upstream_member": "sh\\lday\\sh000300.day",
    "upstream_member_sha256": "2e5adb536fa0f03860e70dbcb0e4f67c7151f35d3e7595df174a5ad15a27c9ac",
    "observed_date_range": ["2005-01-04", "2026-10-19"],
    "price_projection": "OHLC_ALREADY_DIVIDED_BY_100_UPSTREAM",
    "official_history": "CANNOT_CONFIRM",
}

FUTU_SOURCE_REF = {
    "source_id": "EXISTING_FUTU_CONTINUATION",
    "path": "/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/index_recovery_futu/FUTU_000300_HISTORY_FULL_20261019_R1/index_000300_daily.csv",
    "sha256": "ed55aaab1691999863461e38482fb66cc7bc0c4da85c0fed2556f0925eadaa70",
    "index_code": "SH.000300",
    "adjustment": "AuType.NONE",
    "date_range": ["2021-01-04", "2026-10-15"],
    "row_count": 1399,
}

REUSED_OVERLAP_REF = {
    "path": "/Users/lexi/Desktop/学习/因子工厂/research/a_share_model_selection_v1/batches/history_market_view_20261020/HISTORY_EXPORT_01/SOURCE_PARITY.json",
    "sha256": "639ef5ee0fb10308016ecac734165dae3dbda48e6634196537d3b1716c58c64f",
    "status": "CROSS_SOURCE_INTERSECTION_PARITY_WITH_LIMITS",
    "intersection_rows": 1399,
    "max_abs_difference": 0.005400000000008731,
    "tolerance_index_points": 0.02,
    "out_of_tolerance": 0,
    "reused_without_recalculation": True,
}


class IndexPrefixError(ValueError):
    """Raised when input rows cannot satisfy the frozen composition contract."""


def _source_dates(frame: pd.DataFrame, source_name: str) -> pd.DatetimeIndex:
    if "date" not in frame.columns:
        raise IndexPrefixError(f"{source_name} is missing the date column")
    if frame["date"].isna().any():
        raise IndexPrefixError(f"{source_name} contains a missing date")
    if any(isinstance(value, Number) for value in frame["date"].tolist()):
        raise IndexPrefixError(f"{source_name} date values must be date-like, not numeric")
    try:
        parsed = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise IndexPrefixError(f"{source_name} contains an invalid date") from exc
    if isinstance(parsed, pd.Series):
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise IndexPrefixError(f"{source_name} dates must not carry a timezone")
        dates = pd.DatetimeIndex(parsed)
    else:
        dates = pd.DatetimeIndex(parsed)
    if dates.isna().any():
        raise IndexPrefixError(f"{source_name} contains a missing date")
    if not dates.equals(dates.normalize()):
        raise IndexPrefixError(f"{source_name} contains a non-daily timestamp")
    if dates.has_duplicates:
        raise IndexPrefixError(f"{source_name} contains duplicate dates")
    if not dates.is_monotonic_increasing:
        raise IndexPrefixError(f"{source_name} dates are not strictly increasing")
    return dates


def _validate_prices(frame: pd.DataFrame, source_name: str) -> None:
    for column in ("close", "open", "high", "low"):
        if column not in frame.columns:
            raise IndexPrefixError(f"{source_name} is missing price column {column}")
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise IndexPrefixError(f"{source_name} has nonfinite or nonpositive {column}")


def _calendar_dates(required_sessions: Iterable[Any]) -> pd.DatetimeIndex:
    values = list(required_sessions)
    if not values:
        raise IndexPrefixError("bound stock calendar is empty")
    calendar = pd.DataFrame({"date": values})
    return _source_dates(calendar, "bound stock calendar")


def compose_observed_index_prefix(
    tdx_derived: pd.DataFrame,
    futu_existing: pd.DataFrame,
    *,
    required_sessions: Iterable[Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a versioned TDX-prefix/Futu-continuation projection and manifest.

    TDX input must be the existing decoded OHLC export. Its OHLC values are
    copied as-is because the upstream TDX projection already applied /100.
    Futu rows from 2021-01-04 through 2026-10-15 are copied without changing
    their values, nulls, columns, or relative order. Source tags live only in
    the returned manifest, so the original Futu CSV schema remains unchanged.

    When ``required_sessions`` is supplied, every bound stock-session date must
    exist in the composite. This checks only the supplied observed calendar; it
    does not certify an official exchange calendar.
    """
    if not isinstance(tdx_derived, pd.DataFrame) or not isinstance(futu_existing, pd.DataFrame):
        raise IndexPrefixError("both inputs must be pandas DataFrames")
    if tdx_derived.columns.has_duplicates or futu_existing.columns.has_duplicates:
        raise IndexPrefixError("source columns must be unique")
    if tuple(tdx_derived.columns) != TDX_DERIVED_COLUMNS:
        raise IndexPrefixError(
            f"TDX derived schema must be exactly {list(TDX_DERIVED_COLUMNS)}"
        )
    if tuple(futu_existing.columns) != FUTU_CSV_COLUMNS:
        raise IndexPrefixError(
            f"Futu schema must be exactly {list(FUTU_CSV_COLUMNS)}"
        )

    tdx_dates = _source_dates(tdx_derived, "TDX derived input")
    futu_dates = _source_dates(futu_existing, "Futu input")
    if len(tdx_derived) == 0 or len(futu_existing) == 0:
        raise IndexPrefixError("both source inputs must be nonempty")
    if tdx_dates[0] < TDX_START:
        raise IndexPrefixError("TDX input starts before the bound observed history")
    if (futu_dates < FUTU_START).any():
        raise IndexPrefixError("Futu input contains rows before the frozen handoff date")

    tdx_mask = (tdx_dates < FUTU_START) & (tdx_dates <= ANALYSIS_CUTOFF)
    futu_mask = (futu_dates >= FUTU_START) & (futu_dates <= ANALYSIS_CUTOFF)
    tdx_prefix = tdx_derived.loc[tdx_mask].copy().reset_index(drop=True)
    futu_continuation = futu_existing.loc[futu_mask].copy().reset_index(drop=True)
    tdx_prefix_dates = tdx_dates[tdx_mask]
    futu_continuation_dates = futu_dates[futu_mask]

    if tdx_prefix.empty:
        raise IndexPrefixError("TDX input has no rows before the Futu handoff date")
    if futu_continuation.empty or futu_continuation_dates[0] != FUTU_START:
        raise IndexPrefixError("Futu continuation does not begin on the frozen handoff date")
    if tdx_prefix_dates[-1] >= futu_continuation_dates[0]:
        raise IndexPrefixError("TDX prefix overlaps the Futu continuation")

    _validate_prices(tdx_prefix, "TDX prefix")
    _validate_prices(futu_continuation, "Futu continuation")
    futu_codes = futu_continuation["code"].astype("string")
    if futu_codes.isna().any() or not futu_codes.eq(SERIES_CODE).all():
        raise IndexPrefixError("Futu continuation code differs from bound SH.000300")

    prefix = pd.DataFrame(index=range(len(tdx_prefix)))
    prefix["date"] = tdx_prefix["date"].to_numpy(copy=True)
    prefix["code"] = SERIES_CODE
    for column in PRICE_COLUMNS:
        prefix[column] = tdx_prefix[column].to_numpy(copy=True)
    prefix["volume"] = None
    prefix["turnover"] = None
    prefix = prefix.loc[:, list(FUTU_CSV_COLUMNS)]

    composite = pd.concat([prefix, futu_continuation], ignore_index=True)
    composite_dates = _source_dates(composite, "composite index")
    if len(composite_dates) != len(tdx_prefix) + len(futu_continuation):
        raise IndexPrefixError("composite row count changed during projection")

    if required_sessions is None:
        calendar_coverage = {
            "status": "NOT_CHECKED_BOUND_STOCK_CALENDAR_NOT_PROVIDED",
            "required_session_count": None,
            "missing_session_count": None,
            "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        }
    else:
        required = _calendar_dates(required_sessions)
        composite_set = set(composite_dates)
        missing = [day.strftime("%Y-%m-%d") for day in required if day not in composite_set]
        if missing:
            sample = ", ".join(missing[:10])
            raise IndexPrefixError(
                "bound stock-calendar sessions are missing from the composite index: " + sample
            )
        calendar_coverage = {
            "status": "PASS_BOUND_OBSERVED_STOCK_CALENDAR_COVERAGE",
            "required_session_count": len(required),
            "missing_session_count": 0,
            "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "CSI300_TDX_OBSERVED_PREFIX_FUTU_CONTINUATION_V1",
        "status": "OBSERVED_COMPOSITE_READY",
        "research_classification": "DIRECTION_ONLY",
        "series_code": SERIES_CODE,
        "analysis_cutoff": ANALYSIS_CUTOFF.strftime("%Y-%m-%d"),
        "handoff_date": FUTU_START.strftime("%Y-%m-%d"),
        "output_schema": list(FUTU_CSV_COLUMNS),
        "consumer_columns": ["date", "code", "close"],
        "source_segments": [
            {
                "source_id": TDX_SOURCE_REF["source_id"],
                "date_predicate": "2005-01-04 <= date < 2021-01-04",
                "row_count": len(tdx_prefix),
                "first_date": tdx_prefix_dates[0].strftime("%Y-%m-%d"),
                "last_date": tdx_prefix_dates[-1].strftime("%Y-%m-%d"),
            },
            {
                "source_id": FUTU_SOURCE_REF["source_id"],
                "date_predicate": "2021-01-04 <= date <= 2026-10-15",
                "row_count": len(futu_continuation),
                "first_date": futu_continuation_dates[0].strftime("%Y-%m-%d"),
                "last_date": futu_continuation_dates[-1].strftime("%Y-%m-%d"),
            },
        ],
        "source_refs": {
            "tdx_derived": dict(TDX_SOURCE_REF),
            "futu_existing": dict(FUTU_SOURCE_REF),
            "reused_overlap_receipt": dict(REUSED_OVERLAP_REF),
            "input_authentication": "MUST_VERIFY_BOUND_SOURCE_HASHES_AFTER_DURABLE_RUN_STARTED_BEFORE_READ",
        },
        "projection": {
            "tdx_scale_applied_here": False,
            "tdx_scale_already_applied_upstream": 0.01,
            "futu_adjustment": "AuType.NONE",
            "futu_row_values_mutated": False,
            "futu_row_order_preserved": True,
            "interpolation": False,
            "forward_fill": False,
            "returns_recomputed": False,
            "price_parity_recomputed": False,
            "source_tags_in_separate_manifest": True,
            "prefix_only_fields_without_tdx_counterpart": ["volume", "turnover"],
            "prefix_series_code_from_bound_identity": SERIES_CODE,
        },
        "calendar_coverage": calendar_coverage,
        "official_history": "CANNOT_CONFIRM",
        "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        "safety": {
            "research_only": True,
            "production_allowed": False,
            "buy_gate_updated": False,
            "orders_sent": False,
        },
    }
    return composite, manifest
