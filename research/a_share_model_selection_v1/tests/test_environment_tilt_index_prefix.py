from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research.a_share_model_selection_v1.environment_tilt_index_prefix import (
    FUTU_CSV_COLUMNS,
    IndexPrefixError,
    SCHEMA_VERSION,
    SERIES_CODE,
    compose_observed_index_prefix,
)


def _tdx(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": day,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
            }
            for day, close in rows
        ],
        columns=["date", "open", "high", "low", "close"],
    )


def _futu(rows: list[tuple[str, float, float | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": day,
                "code": SERIES_CODE,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000 + i,
                "turnover": turnover,
            }
            for i, (day, close, turnover) in enumerate(rows)
        ],
        columns=list(FUTU_CSV_COLUMNS),
    )


def _base_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    tdx = _tdx(
        [
            ("2020-12-30", 3998.0),
            ("2020-12-31", 4000.0),
            ("2021-01-04", 4100.0),  # The boundary belongs to Futu.
            ("2026-10-15", 5000.0),
            ("2026-10-16", 5100.0),  # Beyond the old analysis cutoff.
        ]
    )
    futu = _futu(
        [
            ("2021-01-04", 4100.25, 12.5),
            ("2021-01-05", 4110.5, None),
            ("2026-10-15", 4999.75, 25.0),
            ("2026-10-16", 5100.0, 30.0),
        ]
    )
    return tdx, futu


def test_composes_tdx_prefix_with_futu_boundary_and_cutoff_unchanged() -> None:
    tdx, futu = _base_inputs()
    result, manifest = compose_observed_index_prefix(
        tdx,
        futu,
        required_sessions=["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"],
    )

    assert list(result.columns) == list(FUTU_CSV_COLUMNS)
    assert result["date"].tolist() == [
        "2020-12-30",
        "2020-12-31",
        "2021-01-04",
        "2021-01-05",
        "2026-10-15",
    ]
    # Existing derived TDX values are copied as-is; /100 is not applied twice.
    assert result.loc[:1, "close"].tolist() == [3998.0, 4000.0]
    assert result.loc[:1, "volume"].isna().all()
    assert result.loc[:1, "turnover"].isna().all()
    assert result.loc[:1, "code"].tolist() == [SERIES_CODE, SERIES_CODE]

    # The Futu segment keeps every original column value, null and row order.
    expected_futu = futu.iloc[:3].reset_index(drop=True)
    actual_futu = result.iloc[2:].reset_index(drop=True)
    assert_frame_equal(actual_futu, expected_futu, check_dtype=False)
    assert actual_futu["turnover"].isna().tolist() == expected_futu["turnover"].isna().tolist()

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["source_segments"][0]["row_count"] == 2
    assert manifest["source_segments"][1]["row_count"] == 3
    assert manifest["projection"]["tdx_scale_applied_here"] is False
    assert manifest["projection"]["futu_row_values_mutated"] is False
    assert manifest["projection"]["futu_row_order_preserved"] is True
    assert manifest["projection"]["prefix_only_fields_without_tdx_counterpart"] == ["volume", "turnover"]
    assert manifest["projection"]["price_parity_recomputed"] is False
    assert manifest["calendar_coverage"]["status"] == "PASS_BOUND_OBSERVED_STOCK_CALENDAR_COVERAGE"
    assert manifest["calendar_coverage"]["official_exchange_calendar_parity"] == "CANNOT_CONFIRM"
    assert manifest["official_history"] == "CANNOT_CONFIRM"
    assert "source_id" not in result.columns


def test_without_bound_calendar_manifest_does_not_claim_coverage() -> None:
    tdx, futu = _base_inputs()
    _, manifest = compose_observed_index_prefix(tdx, futu)

    assert manifest["calendar_coverage"]["status"] == "NOT_CHECKED_BOUND_STOCK_CALENDAR_NOT_PROVIDED"
    assert manifest["calendar_coverage"]["required_session_count"] is None
    assert manifest["official_exchange_calendar_parity"] == "CANNOT_CONFIRM"


def test_missing_bound_calendar_session_fails_explicitly() -> None:
    tdx, futu = _base_inputs()

    with pytest.raises(IndexPrefixError, match="bound stock-calendar sessions are missing"):
        compose_observed_index_prefix(
            tdx,
            futu,
            required_sessions=["2020-12-29", "2020-12-30"],
        )


@pytest.mark.parametrize(
    ("source", "rows", "message"),
    [
        ("tdx", [("2020-12-31", 4000.0), ("2020-12-31", 4001.0)], "duplicate dates"),
        ("tdx", [("2020-12-31", 4000.0), ("2020-12-30", 4001.0)], "not strictly increasing"),
        ("tdx", [("2020-12-30", float("inf"))], "nonfinite or nonpositive"),
        ("futu", [("2021-01-04", 4100.0, 1.0), ("2021-01-04", 4101.0, 2.0)], "duplicate dates"),
        ("futu", [("2021-01-04", 4100.0, 1.0), ("2021-01-05", float("nan"), 2.0)], "nonfinite or nonpositive"),
    ],
)
def test_duplicate_order_and_nonfinite_inputs_fail_closed(
    source: str, rows: list[tuple[str, float, float | None]], message: str
) -> None:
    tdx, futu = _base_inputs()
    if source == "tdx":
        tdx = _tdx(rows)  # type: ignore[arg-type]
    else:
        futu = _futu(rows)  # type: ignore[arg-type]

    with pytest.raises(IndexPrefixError, match=message):
        compose_observed_index_prefix(tdx, futu)


def test_missing_dates_wrong_schema_and_wrong_futu_identity_fail_closed() -> None:
    tdx, futu = _base_inputs()

    missing_date = tdx.copy()
    missing_date.loc[0, "date"] = None
    with pytest.raises(IndexPrefixError, match="missing date"):
        compose_observed_index_prefix(missing_date, futu)

    wrong_schema = tdx.rename(columns={"close": "settle"})
    with pytest.raises(IndexPrefixError, match="TDX derived schema must be exactly"):
        compose_observed_index_prefix(wrong_schema, futu)

    wrong_code = futu.copy()
    wrong_code.loc[0, "code"] = "SZ.399300"
    with pytest.raises(IndexPrefixError, match="code differs from bound SH.000300"):
        compose_observed_index_prefix(tdx, wrong_code)


def test_ambiguous_numeric_date_and_non_numeric_price_types_fail_closed() -> None:
    tdx, futu = _base_inputs()

    numeric_date = tdx.copy()
    numeric_date["date"] = numeric_date["date"].astype(object)
    numeric_date.loc[0, "date"] = 20201230
    with pytest.raises(IndexPrefixError, match="date values must be date-like, not numeric"):
        compose_observed_index_prefix(numeric_date, futu)

    non_numeric_price = tdx.copy()
    non_numeric_price["close"] = non_numeric_price["close"].astype(object)
    non_numeric_price.loc[0, "close"] = "unknown"
    with pytest.raises(IndexPrefixError, match="nonfinite or nonpositive close"):
        compose_observed_index_prefix(non_numeric_price, futu)


def test_missing_prefix_or_handoff_row_fails_instead_of_filling() -> None:
    tdx, futu = _base_inputs()
    no_prefix = tdx.loc[tdx["date"] >= "2021-01-04"].reset_index(drop=True)
    with pytest.raises(IndexPrefixError, match="no rows before the Futu handoff"):
        compose_observed_index_prefix(no_prefix, futu)

    missing_handoff = futu.iloc[1:].reset_index(drop=True)
    with pytest.raises(IndexPrefixError, match="does not begin on the frozen handoff date"):
        compose_observed_index_prefix(tdx, missing_handoff)
