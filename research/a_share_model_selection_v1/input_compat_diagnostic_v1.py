from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


DIAGNOSTIC_CODE = "PV_INPUT_COMPAT_DIAGNOSTIC_V1"
EXPERIMENT_ID = "A_SHARE_MODEL_SELECTION_PV_ENVIRONMENT_R1_20261019"
INPUT_BINDING_SHA256 = "6618e8d347216311fee106d3e8a4fc497227b2fc5375ed51029e317ecadcc5c9"
BASE_CONTRACT_SHA256 = "9a7f726b81ae492a28571360cea97dbbc29a139169eeadca05a63ce077f0eab9"
FROZEN_ROUTE_SHA256 = "cf88ddab5ba22c2e786f9717be117d0ea753953d23120e6e12586351c4ab6f4b"
FUTU_RECEIPT_SHA256 = "b1e58e310d55a95f21817cdb523e13a060cdccac96a8e82e23147f0b6cb76c8a"
FUTU_CSV_SHA256 = "ed55aaab1691999863461e38482fb66cc7bc0c4da85c0fed2556f0925eadaa70"
FROZEN_ROUTE_READ_ACTIVATION_NAME = "REVIEW_ACTIVATION_M0_R2.json"
FROZEN_ROUTE_READ_ACTIVATION_SHA256 = "e4e07dcd7485f934eeafdc0d8ff9a24d57113e3bd4c35c4e481f9a83ac9dfae5"
EXPECTED_COLUMNS = ("trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount")
NUMERIC_COLUMNS = ("open", "high", "low", "close", "vol", "amount")
OHLC_COLUMNS = ("open", "high", "low", "close")
EPSILON_MULTIPLIER = 8.0
MAX_RAW_VIOLATING_ROWS = 10
ACTIVATION_NAME = "INPUT_COMPAT_DIAGNOSTIC_ACTIVATION.json"


class DiagnosticError(RuntimeError):
    """A fail-closed diagnostic contract or input error."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
    else:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite_numeric(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    original = frame[column]
    parsed_raw = pd.to_numeric(original, errors="coerce")
    numeric = parsed_raw.to_numpy(dtype=np.float64, na_value=np.nan)
    parsed = pd.Series(numeric, index=frame.index, dtype=np.float64)
    missing = original.isna() | parsed.isna()
    non_numeric = ~original.isna() & parsed.isna()
    nonfinite = pd.Series(np.isinf(numeric), index=frame.index)
    return parsed, missing, pd.DataFrame({"non_numeric": non_numeric, "nonfinite": nonfinite}, index=frame.index)


def _parse_dates_like_consumer(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values.dtype):
        parsed = pd.to_datetime(values.astype("string"), format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.normalize()


def diagnose_price_panel(frame: pd.DataFrame, column_map: Mapping[str, str]) -> dict[str, Any]:
    """Summarize only the eight bound raw columns; never edits or normalizes them."""
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise DiagnosticError(f"PANEL_COLUMNS_MISMATCH:{','.join(map(str, frame.columns))}")
    expected_map = {
        "date": "trade_date", "code": "ts_code", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "vol", "amount": "amount",
    }
    if dict(column_map) != expected_map:
        raise DiagnosticError("PANEL_COLUMN_BINDING_MISMATCH")

    dates = _parse_dates_like_consumer(frame["trade_date"])
    codes = frame["ts_code"]
    date_code_duplicates = frame.duplicated(["trade_date", "ts_code"], keep=False)
    normalized_codes = codes.astype(str).str.strip()
    normalized_date_code_keys = pd.DataFrame(
        {"trade_date": dates, "ts_code": normalized_codes}, index=frame.index
    )
    normalized_date_code_duplicates = normalized_date_code_keys.duplicated(
        ["trade_date", "ts_code"], keep=False
    )
    raw_code_text = codes.astype(str)
    raw_code_whitespace = raw_code_text.ne(normalized_codes)
    row_violation = dates.isna() | codes.isna() | normalized_codes.eq("") | normalized_date_code_duplicates
    numeric: dict[str, pd.Series] = {}
    numeric_report: dict[str, Any] = {}
    for column in NUMERIC_COLUMNS:
        parsed, missing, detail = _finite_numeric(frame, column)
        numeric[column] = parsed
        row_violation |= missing | detail["nonfinite"]
        numeric_report[column] = {
            "original_dtype": str(frame[column].dtype),
            "missing_or_unparseable_count": int(missing.sum()),
            "non_numeric_count": int(detail["non_numeric"].sum()),
            "nonfinite_infinity_count": int(detail["nonfinite"].sum()),
        }

    finite_ohlc = pd.DataFrame({name: numeric[name] for name in OHLC_COLUMNS}).apply(
        lambda col: np.isfinite(col.to_numpy(dtype=np.float64, na_value=np.nan)), axis=0
    ).all(axis=1)
    high_floor = pd.concat([numeric["open"], numeric["close"]], axis=1).max(axis=1, skipna=False)
    low_ceiling = pd.concat([numeric["open"], numeric["close"]], axis=1).min(axis=1, skipna=False)
    high_delta = (high_floor - numeric["high"]).where(finite_ohlc)
    low_delta = (numeric["low"] - low_ceiling).where(finite_ohlc)
    high_bad = finite_ohlc & high_delta.gt(0)
    low_bad = finite_ohlc & low_delta.gt(0)
    high_scale = pd.concat([numeric["open"].abs(), numeric["close"].abs()], axis=1).max(axis=1, skipna=False)
    low_scale = low_ceiling.abs()
    epsilon = np.finfo(np.float64).eps * EPSILON_MULTIPLIER
    high_rel = high_delta / high_scale.where(high_scale.ne(0))
    low_rel = low_delta / low_scale.where(low_scale.ne(0))
    high_eps = high_bad & high_rel.le(epsilon)
    low_eps = low_bad & low_rel.le(epsilon)

    nonpositive = {
        column: int((numeric[column].notna() & np.isfinite(numeric[column]) & numeric[column].le(0)).sum())
        for column in OHLC_COLUMNS
    }
    volume_amount = {
        column: {
            "zero_count": int(numeric[column].eq(0).sum()),
            "negative_count": int(numeric[column].lt(0).sum()),
        }
        for column in ("vol", "amount")
    }
    for mask in (high_bad, low_bad):
        row_violation |= mask
    for column in OHLC_COLUMNS:
        row_violation |= numeric[column].notna() & np.isfinite(numeric[column]) & numeric[column].le(0)
    for column in ("vol", "amount"):
        row_violation |= numeric[column].notna() & np.isfinite(numeric[column]) & numeric[column].lt(0)

    raw_rows: list[dict[str, Any]] = []
    for index in frame.index[row_violation][:MAX_RAW_VIOLATING_ROWS]:
        raw_rows.append({str(name): _json_value(frame.at[index, name]) for name in EXPECTED_COLUMNS})

    return {
        "row_count": int(len(frame)),
        "columns": {str(name): {"original_dtype": str(frame[name].dtype), "null_count": int(frame[name].isna().sum())} for name in EXPECTED_COLUMNS},
        "date": {
            "parse_failure_count": int(dates.isna().sum()),
            "duplicate_row_count": int(frame["trade_date"].duplicated(keep=False).sum()),
            "first_valid": dates.min().date().isoformat() if dates.notna().any() else None,
            "last_valid": dates.max().date().isoformat() if dates.notna().any() else None,
            "distinct_valid_count": int(dates.dropna().nunique()),
        },
        "code": {
            "null_count": int(codes.isna().sum()),
            "distinct_non_null_count": int(codes.dropna().nunique()),
            "duplicate_row_count": int(codes.duplicated(keep=False).sum()),
            "consumer_normalized_distinct_count": int(normalized_codes.nunique(dropna=False)),
            "consumer_normalized_empty_count": int(normalized_codes.eq("").sum()),
            "raw_whitespace_row_count": int(raw_code_whitespace.sum()),
            "raw_whitespace_samples": [
                {
                    "row_index": _json_value(frame.index[position]),
                    "raw_code": _json_value(codes.iloc[position]),
                }
                for position in np.flatnonzero(raw_code_whitespace.to_numpy())[:MAX_RAW_VIOLATING_ROWS]
            ],
        },
        "duplicate_date_code_row_count": int(date_code_duplicates.sum()),
        "normalized_duplicate_date_code_row_count": int(normalized_date_code_duplicates.sum()),
        "numeric_fields": numeric_report,
        "ohlc": {
            "nonpositive_row_count_by_field": nonpositive,
            "high_below_max_open_close_count": int(high_bad.sum()),
            "low_above_min_open_close_count": int(low_bad.sum()),
            "high_violation_max_absolute": float(high_delta[high_bad].max()) if high_bad.any() else 0.0,
            "high_violation_max_relative": float(high_rel[high_bad].max()) if high_bad.any() else 0.0,
            "low_violation_max_absolute": float(low_delta[low_bad].max()) if low_bad.any() else 0.0,
            "low_violation_max_relative": float(low_rel[low_bad].max()) if low_bad.any() else 0.0,
            "strict_violation_epsilon_explainable_count": int(high_eps.sum() + low_eps.sum()),
            "epsilon_relative_bound": epsilon,
            "epsilon_is_diagnostic_only": True,
        },
        "volume_amount": volume_amount,
        "raw_violating_rows_cap": MAX_RAW_VIOLATING_ROWS,
        "raw_violating_rows": raw_rows,
        "calendar_dates_for_coverage": sorted({d.date().isoformat() for d in dates.dropna()}),
    }


def diagnose_calendar(calendar_bytes: bytes, *, panel_dates: set[str], expected: Mapping[str, Any]) -> dict[str, Any]:
    schema = expected.get("schema")
    if schema != ["trade_date", "calendar_index"]:
        raise DiagnosticError("CALENDAR_CONTRACT_SCHEMA_MISMATCH")
    try:
        frame = pd.read_csv(io.BytesIO(calendar_bytes), usecols=schema)
    except Exception as exc:
        raise DiagnosticError(f"CALENDAR_PARSE_FAILED:{type(exc).__name__}") from exc
    if list(frame.columns) != schema:
        raise DiagnosticError("CALENDAR_COLUMNS_MISMATCH")
    dates = _parse_dates_like_consumer(frame["trade_date"])
    indexes = pd.to_numeric(frame["calendar_index"], errors="coerce")
    valid_indexes = indexes.dropna().to_numpy(dtype=np.float64)
    index_contiguous = bool(
        len(valid_indexes) == len(frame)
        and len(valid_indexes) > 0
        and np.array_equal(valid_indexes, np.arange(len(valid_indexes), dtype=np.float64))
    )
    indexed_dates = pd.DataFrame({"calendar_index": indexes, "trade_date": dates}).sort_values(
        "calendar_index", kind="mergesort", na_position="last"
    )["trade_date"]
    calendar_index_date_order_strictly_increasing = bool(
        len(frame) > 0
        and indexes.notna().all()
        and dates.notna().all()
        and not indexes.duplicated().any()
        and not dates.duplicated().any()
        and (indexed_dates.diff().dropna() > pd.Timedelta(0)).all()
    )
    calendar_dates = {d.date().isoformat() for d in dates.dropna()}
    missing = sorted(panel_dates - calendar_dates)
    duplicate_date_count = int(frame["trade_date"].duplicated(keep=False).sum())
    duplicate_index_count = int(frame["calendar_index"].duplicated(keep=False).sum())
    return {
        "row_count": int(len(frame)),
        "schema": list(frame.columns),
        "trade_date_dtype": str(frame["trade_date"].dtype),
        "calendar_index_dtype": str(frame["calendar_index"].dtype),
        "trade_date_parse_failure_count": int(dates.isna().sum()),
        "calendar_index_missing_count": int(indexes.isna().sum()),
        "duplicate_date_row_count": duplicate_date_count,
        "duplicate_index_row_count": duplicate_index_count,
        "calendar_index_min": float(indexes.min()) if indexes.notna().any() else None,
        "calendar_index_max": float(indexes.max()) if indexes.notna().any() else None,
        "calendar_index_contiguous_from_zero": index_contiguous,
        "calendar_index_date_order_strictly_increasing": calendar_index_date_order_strictly_increasing,
        "row_count_matches_metadata": len(frame) == expected.get("rows"),
        "date_range_matches_metadata": (
            [dates.min().date().isoformat(), dates.max().date().isoformat()] == expected.get("date_range")
            if dates.notna().any() else False
        ),
        "panel_distinct_valid_dates": len(panel_dates),
        "panel_dates_missing_from_calendar_count": len(missing),
        "panel_dates_missing_from_calendar_sample": missing[:MAX_RAW_VIOLATING_ROWS],
        "official_exchange_calendar_identity": "CANNOT_CONFIRM",
        "official_calendar_parity": "CANNOT_CONFIRM",
        "expected_metadata_rows": expected.get("rows"),
        "expected_metadata_range": expected.get("date_range"),
    }


def diagnose_futu(csv_bytes: bytes, receipt_bytes: bytes, *, panel_dates: set[str], expected: Mapping[str, Any]) -> dict[str, Any]:
    if _sha256(csv_bytes) != expected.get("csv_sha256"):
        raise DiagnosticError("FUTU_CSV_SHA256_MISMATCH")
    if _sha256(receipt_bytes) != expected.get("receipt_sha256"):
        raise DiagnosticError("FUTU_RECEIPT_SHA256_MISMATCH")
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError("FUTU_RECEIPT_INVALID") from exc
    if receipt.get("normalized_path") != expected.get("csv_name"):
        raise DiagnosticError("FUTU_RECEIPT_CSV_NAME_MISMATCH")
    if receipt.get("normalized_sha256") != expected.get("csv_sha256"):
        raise DiagnosticError("FUTU_RECEIPT_CSV_SHA_MISMATCH")
    if receipt.get("index_code") != expected.get("index_code"):
        raise DiagnosticError("FUTU_RECEIPT_INDEX_CODE_MISMATCH")

    try:
        reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig", errors="strict")))
        header = next(reader)
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise DiagnosticError("FUTU_CSV_HEADER_INVALID") from exc
    if len(header) != len(set(header)) or not {"date", "code"}.issubset(header):
        raise DiagnosticError("FUTU_CSV_DATE_CODE_FIELDS_INVALID")
    if expected.get("header") is not None and header != expected["header"]:
        raise DiagnosticError("FUTU_CSV_HEADER_MISMATCH")
    date_idx, code_idx = header.index("date"), header.index("code")
    valid_dates: list[str] = []
    invalid_date_count = 0
    null_code_count = 0
    unexpected_code_count = 0
    codes: set[str] = set()
    duplicate_dates: set[str] = set()
    seen_dates: set[str] = set()
    row_count = 0
    for row in reader:
        if len(row) != len(header):
            raise DiagnosticError("FUTU_CSV_ROW_WIDTH_MISMATCH")
        row_count += 1
        raw_date, code = row[date_idx].strip(), row[code_idx].strip()
        try:
            parsed = pd.to_datetime(raw_date, errors="raise")
            date = parsed.date().isoformat()
            valid_dates.append(date)
        except (ValueError, TypeError):
            invalid_date_count += 1
            continue
        if date in seen_dates:
            duplicate_dates.add(date)
        seen_dates.add(date)
        if not code:
            null_code_count += 1
        else:
            codes.add(code)
            if code != expected["index_code"]:
                unexpected_code_count += 1

    futu_dates = set(valid_dates)
    missing = sorted(panel_dates - futu_dates)
    extra = sorted(futu_dates - panel_dates)
    return {
        "csv_sha256": _sha256(csv_bytes),
        "receipt_sha256": _sha256(receipt_bytes),
        "header": header,
        "date_and_code_only": True,
        "row_count": row_count,
        "date_parse_failure_count": invalid_date_count,
        "null_code_count": null_code_count,
        "unexpected_code_count": unexpected_code_count,
        "distinct_code_values": sorted(codes),
        "duplicate_date_count": len(duplicate_dates),
        "distinct_valid_date_count": len(futu_dates),
        "first_valid_date": min(futu_dates) if futu_dates else None,
        "last_valid_date": max(futu_dates) if futu_dates else None,
        "panel_date_missing_from_futu_count": len(missing),
        "panel_date_missing_from_futu_sample": missing[:MAX_RAW_VIOLATING_ROWS],
        "futu_date_outside_panel_count": len(extra),
        "futu_date_outside_panel_sample": extra[:MAX_RAW_VIOLATING_ROWS],
        "receipt_row_count_metadata": receipt.get("normalized_row_count"),
        "row_count_matches_receipt_metadata": row_count == receipt.get("normalized_row_count"),
        "codes_match_bound_index_code": codes == {expected["index_code"]},
        "index_code": receipt.get("index_code"),
    }


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label}_LOAD_FAILED:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label}_NOT_OBJECT")
    return value, raw


def validate_activation(
    activation: Mapping[str, Any],
    *,
    contract_sha256: str,
    input_binding_sha256: str,
    code_sha256: str,
    reviewer_receipt_sha256: str | None = None,
) -> None:
    required = {
        "schema_version": "A_SHARE_MODEL_SELECTION_PV_INPUT_COMPAT_ACTIVATION_V1",
        "status": "APPROVED_FOR_INPUT_COMPAT_DIAGNOSTIC_ONLY",
        "diagnostic_code": DIAGNOSTIC_CODE,
        "contract_sha256": contract_sha256,
        "input_binding_sha256": input_binding_sha256,
        "code_sha256": code_sha256,
        "allow_bound_input_read": True,
        "allow_futu_metadata_read": True,
        "allow_model_fit": False,
        "allow_formula_evaluation": False,
        "allow_label_read": False,
        "allow_network": False,
    }
    for key, expected in required.items():
        if activation.get(key) != expected:
            raise DiagnosticError(f"ACTIVATION_MISMATCH:{key}")
    if not isinstance(activation.get("reviewer_identity"), str) or not activation["reviewer_identity"].strip():
        raise DiagnosticError("ACTIVATION_REVIEWER_IDENTITY_MISSING")
    if not isinstance(activation.get("reviewer_receipt_path"), str) or not activation["reviewer_receipt_path"].strip():
        raise DiagnosticError("ACTIVATION_REVIEWER_RECEIPT_PATH_MISSING")
    actual_receipt_sha = activation.get("reviewer_receipt_sha256")
    if (
        not isinstance(actual_receipt_sha, str)
        or len(actual_receipt_sha) != 64
        or any(char not in "0123456789abcdef" for char in actual_receipt_sha)
    ):
        raise DiagnosticError("ACTIVATION_REVIEWER_RECEIPT_SHA_MISSING")
    if reviewer_receipt_sha256 is not None and actual_receipt_sha != reviewer_receipt_sha256:
        raise DiagnosticError("ACTIVATION_REVIEWER_RECEIPT_SHA_MISMATCH")


def _load_frozen_route(project_root: Path) -> Any:
    path = Path(project_root) / "workspace/research/a_share_model_selection_v1/model_selection.py"
    if _sha256_file(path) != FROZEN_ROUTE_SHA256:
        raise DiagnosticError("FROZEN_MODEL_SELECTION_ROUTE_SHA256_MISMATCH")
    spec = importlib.util.spec_from_file_location("pv_input_compat_frozen_model_selection", path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("FROZEN_ROUTE_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The checked-in route is under workspace/research; its production helper
    # computes a root one level too deep for this repository layout.
    module.PROJECT_ROOT = Path(project_root).resolve()
    return module


def _initial_receipt(execution_id: str, *, project_root: Path, contract_sha: str, binding_sha: str, code_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "A_SHARE_MODEL_SELECTION_PV_INPUT_COMPAT_RECEIPT_V1",
        "diagnostic_code": DIAGNOSTIC_CODE,
        "experiment_id": EXPERIMENT_ID,
        "execution_id": execution_id,
        "status": "INITIAL_RECEIPT_DURABLE_GUARD_PENDING",
        "created_at_utc": _utc_now(),
        "project_root": str(Path(project_root).resolve()),
        "diagnostic_contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "diagnostic_code_sha256": code_sha,
        "source_read_started": False,
        "source_read_completed": False,
        "official_registry_run_started": False,
        "official_registry_terminal": "PENDING",
        "fit_started": False,
        "predict_started": False,
        "truth_access": "NOT_READ",
        "sealed_status": "NOT_ENTERED",
        "security_flags": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
    }


def run_diagnostic(
    project_root: Path,
    *,
    execution_id: str | None = None,
    guard_factory: Callable[..., Any] | None = None,
    route_loader: Callable[[Path], Any] = _load_frozen_route,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    batch = root / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
    diagnostic_contract_path = batch / "INPUT_COMPAT_DIAGNOSTIC_CONTRACT.json"
    binding_path = batch / "INPUT_BINDING.json"
    base_contract_path = batch / "CONTRACT.json"
    activation_path = batch / ACTIVATION_NAME
    diagnostic_contract, diagnostic_contract_bytes = _read_json(diagnostic_contract_path, "DIAGNOSTIC_CONTRACT")
    binding, binding_bytes = _read_json(binding_path, "INPUT_BINDING")
    activation, activation_bytes = _read_json(activation_path, "ACTIVATION")
    contract_sha = _sha256(diagnostic_contract_bytes)
    binding_sha = _sha256(binding_bytes)
    code_sha = _sha256_file(Path(__file__).resolve())

    if diagnostic_contract.get("diagnostic_code") != DIAGNOSTIC_CODE or diagnostic_contract.get("experiment_id") != EXPERIMENT_ID:
        raise DiagnosticError("DIAGNOSTIC_CONTRACT_IDENTITY_MISMATCH")
    if diagnostic_contract.get("expected_input_binding_sha256") != INPUT_BINDING_SHA256 or binding_sha != INPUT_BINDING_SHA256:
        raise DiagnosticError("INPUT_BINDING_SHA256_MISMATCH")
    if diagnostic_contract.get("base_contract_sha256") != BASE_CONTRACT_SHA256 or _sha256_file(base_contract_path) != BASE_CONTRACT_SHA256:
        raise DiagnosticError("BASE_CONTRACT_SHA256_MISMATCH")
    if diagnostic_contract.get("frozen_route_sha256") != FROZEN_ROUTE_SHA256:
        raise DiagnosticError("FROZEN_ROUTE_SHA256_CONTRACT_MISMATCH")
    frozen_read_activation = diagnostic_contract.get("frozen_route_read_activation")
    expected_frozen_read_activation = {
        "path": FROZEN_ROUTE_READ_ACTIVATION_NAME,
        "sha256": FROZEN_ROUTE_READ_ACTIVATION_SHA256,
    }
    if frozen_read_activation != expected_frozen_read_activation:
        raise DiagnosticError("FROZEN_ROUTE_READ_ACTIVATION_CONTRACT_MISMATCH")
    validate_activation(
        activation,
        contract_sha256=contract_sha,
        input_binding_sha256=binding_sha,
        code_sha256=code_sha,
    )
    reviewer_receipt_path = Path(str(activation["reviewer_receipt_path"]))
    if not reviewer_receipt_path.is_absolute():
        reviewer_receipt_path = batch / reviewer_receipt_path
    try:
        reviewer_receipt_path = reviewer_receipt_path.resolve(strict=True)
        reviewer_receipt_path.relative_to(batch.resolve())
    except (OSError, ValueError) as exc:
        raise DiagnosticError("ACTIVATION_REVIEWER_RECEIPT_OUTSIDE_BATCH_OR_MISSING") from exc
    if _sha256_file(reviewer_receipt_path) != activation["reviewer_receipt_sha256"]:
        raise DiagnosticError("ACTIVATION_REVIEWER_RECEIPT_SHA_MISMATCH")

    execution_id = execution_id or f"PV_INPUT_COMPAT_DIAGNOSTIC_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    if not execution_id or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-." for ch in execution_id):
        raise DiagnosticError("EXECUTION_ID_INVALID")
    output_dir = batch / "input_compat_diagnostic" / execution_id
    receipt_path = output_dir / "receipt.json"
    receipt = _initial_receipt(execution_id, project_root=root, contract_sha=contract_sha, binding_sha=binding_sha, code_sha=code_sha)
    _atomic_write(receipt_path, receipt, exclusive=True)

    claim = None
    claim_attempted = False
    terminal_attempted = False
    terminal_returned = False
    route = None
    try:
        route = route_loader(root)
        if guard_factory is None:
            guard_factory = route.make_official_fit_guard
        guard = guard_factory(
            contract_path=diagnostic_contract_path,
            binding_path=binding_path,
            stage="M0_INPUT_COMPAT_DIAGNOSTIC_CANARY",
            artifact_paths=(base_contract_path, diagnostic_contract_path, binding_path, activation_path, receipt_path),
        )
        route.set_registry_execution_context(
            guard,
            execution_id,
            {
                "truth_access": "NOT_READ",
                "sealed_status": "NOT_ENTERED",
                "signal_date_range": binding.get("sources", {}).get("price_panel", {}).get("dates", {}),
                "target_date_range": {"status": "NOT_USED_BY_DIAGNOSTIC"},
                "runtime": {"python": sys.version.split()[0], "diagnostic_only": True},
                "device": "CPU_RESEARCH_ONLY",
                "checkpoint_ref": None,
                "code_sha256": code_sha,
                "config_sha256": contract_sha,
                "input_sha256": binding_sha,
                "input_paths": [
                    str(binding.get("sources", {}).get("price_panel", {}).get("path", "")),
                    str(batch / diagnostic_contract.get("futu", {}).get("receipt_path", "")),
                    str(batch / diagnostic_contract.get("futu", {}).get("csv_path", "")),
                ],
                "artifact_paths": [str(receipt_path)],
                "deterministic_metrics": {"truth_positive_denominator": None, "selection_slots": None},
            },
        )
        claim_attempted = True
        claim = guard.claim_no_fit(
            execution_id=execution_id,
            arm=DIAGNOSTIC_CODE,
            contract_sha256=contract_sha,
            input_binding_sha256=binding_sha,
        )
        receipt["official_registry_run_started"] = True
        receipt["status"] = "RUN_STARTED_NO_FIT"
        _atomic_write(receipt_path, receipt)

        receipt["source_read_started"] = True
        _atomic_write(receipt_path, receipt)
        frozen_read_activation_path = batch / FROZEN_ROUTE_READ_ACTIVATION_NAME
        if _sha256_file(frozen_read_activation_path) != FROZEN_ROUTE_READ_ACTIVATION_SHA256:
            raise DiagnosticError("FROZEN_ROUTE_READ_ACTIVATION_SHA256_MISMATCH")
        panel, _ = route.read_bound_panel(
            contract_path=base_contract_path,
            binding_path=binding_path,
            activation_path=frozen_read_activation_path,
            stage="M0_CANARY",
            arms=("M0",),
            preparation_claim=claim,
        )
        column_map = binding["sources"]["price_panel"]["column_map"]
        panel_metrics = diagnose_price_panel(panel, column_map)
        panel_dates = set(panel_metrics.pop("calendar_dates_for_coverage"))

        calendar_source = binding["sources"]["trading_date_index"]
        calendar_bytes = route.read_bound_archive_member(calendar_source)
        calendar_metrics = diagnose_calendar(calendar_bytes, panel_dates=panel_dates, expected=calendar_source)

        futu = diagnostic_contract["futu"]
        futu_receipt_path = batch / futu["receipt_path"]
        futu_csv_path = batch / futu["csv_path"]
        futu_receipt_bytes = futu_receipt_path.read_bytes()
        futu_csv_bytes = futu_csv_path.read_bytes()
        futu_metrics = diagnose_futu(
            futu_csv_bytes,
            futu_receipt_bytes,
            panel_dates=panel_dates,
            expected={
                "receipt_sha256": futu["receipt_sha256"],
                "csv_sha256": futu["csv_sha256"],
                "csv_name": Path(futu["csv_path"]).name,
                "index_code": futu["index_code"],
            },
        )
        receipt.update({
            "status": "DIAGNOSTIC_COMPLETE_NO_FIT",
            "source_read_completed": True,
            "official_registry_terminal": "PENDING",
            "panel": panel_metrics,
            "calendar": calendar_metrics,
            "futu_index_metadata": futu_metrics,
            "semantic_decision": "DIAGNOSTIC_ONLY_NO_DATA_REPAIR_OR_MODEL_DECISION",
        })
        _atomic_write(receipt_path, receipt)
        terminal_attempted = True
        guard.append_no_fit_terminal(claim, succeeded=True)
        terminal_returned = True
        receipt["official_registry_terminal"] = "ENGINEERING_CANARY_ONLY"
        receipt["official_registry_terminal_append_returned"] = True
        _atomic_write(receipt_path, receipt)
        return receipt
    except Exception as exc:
        receipt["status"] = "CANNOT_CONFIRM"
        receipt["error_type"] = type(exc).__name__
        receipt["error_code"] = str(exc).split(":", 1)[0]
        if claim is not None and not terminal_attempted:
            receipt["official_registry_terminal"] = "ERROR_PENDING"
            try:
                _atomic_write(receipt_path, receipt)
            except OSError:
                pass
            terminal_attempted = True
            try:
                guard.append_no_fit_terminal(claim, succeeded=False, error_type=type(exc).__name__)
                receipt["official_registry_terminal"] = "CANNOT_CONFIRM_ENGINEERING_CANARY"
                receipt["official_registry_terminal_append_returned"] = True
            except Exception as terminal_exc:
                receipt["official_registry_terminal"] = "CANNOT_CONFIRM_TERMINAL_APPEND"
                receipt["terminal_append_error_type"] = type(terminal_exc).__name__
        elif claim is None:
            receipt["official_registry_run_started"] = "CANNOT_CONFIRM" if claim_attempted else False
            receipt["official_registry_terminal"] = "CANNOT_CONFIRM" if claim_attempted else "NOT_STARTED"
        else:
            if terminal_returned:
                receipt["official_registry_terminal"] = "ENGINEERING_CANARY_ONLY"
                receipt["official_registry_terminal_append_returned"] = True
            else:
                receipt["official_registry_terminal"] = "CANNOT_CONFIRM_TERMINAL_APPEND"
                receipt["terminal_append_returned"] = False
        try:
            _atomic_write(receipt_path, receipt)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="No-fit diagnostic of the exact bound PV panel, calendar and Futu date/code metadata.")
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = run_diagnostic(args.project_root)
    except Exception as exc:
        print(json.dumps({"status": "CANNOT_CONFIRM", "error_type": type(exc).__name__, "error_code": str(exc).split(":", 1)[0]}, sort_keys=True))
        return 2
    print(json.dumps({"status": receipt["status"], "execution_id": receipt["execution_id"], "receipt_path": str(Path(args.project_root).resolve() / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/input_compat_diagnostic" / receipt["execution_id"] / "receipt.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
