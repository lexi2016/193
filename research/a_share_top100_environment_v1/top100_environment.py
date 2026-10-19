"""Deterministic descriptive statistics for the observed A-share daily Top100.

This route reads only the bound five-column QFQ panel projection, the bound
shared calendar, the retrospective CAPCO classification member, and the bound
CSI 300 index series. It does not call the PV model runner or build labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path("/Users/lexi/Desktop/学习/因子工厂")
DEFAULT_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/top100_environment_r1_20261019"
DEFAULT_REPO = ROOT / "workspace"
EXPERIMENT_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019"
ROUTE_ID = "a-share-top100-environment-descriptive-stats-r1-20261019"
TRACK_ID = "A_SHARE_TOP100_ENVIRONMENT_RESEARCH"
RUN_IDS = {
    "fast_bind": f"{EXPERIMENT_ID}_FAST_BIND_01",
    "canary": f"{EXPERIMENT_ID}_CANARY_01",
    "full_stats": f"{EXPERIMENT_ID}_FULL_STATS_01",
    "review_sample": f"{EXPERIMENT_ID}_REVIEW_SAMPLE_01",
}
SAFETY_FLAGS = {
    "research_only": True,
    "production_allowed": False,
    "buy_gate_updated": False,
    "orders_sent": False,
}
REASON_COLUMNS = (
    "close_nonfinite_rows",
    "close_nonpositive_rows",
    "missing_previous_row_rows",
    "previous_close_nonfinite_rows",
    "previous_close_nonpositive_rows",
    "volume_nonfinite_rows",
    "volume_zero_rows",
    "volume_negative_rows",
    "amount_nonfinite_rows",
    "amount_negative_rows",
    "amount_zero_with_positive_volume_rows",
)
DIMENSION_SETS = (
    "industry",
    "industry_month",
    "industry_lag_market",
    "industry_month_lag_market",
)
SUMMARY_DIMENSIONS = ("all", "month", "lag_market", "month_lag_market")


class ScopeError(RuntimeError):
    """A frozen input, scope, or data invariant failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (Path,)):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default) + "\n"
    _write_new_or_identical(path, payload.encode("utf-8"))


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace existing output: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
        finally:
            tmp.unlink(missing_ok=True)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    _write_new_or_identical(path, payload)


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bound helper: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def load_bound_helpers(binding: Mapping[str, Any]):
    path = Path(binding["shared_dependencies"]["model_selection"]["path"])
    expected = str(binding["shared_dependencies"]["model_selection"]["sha256"])
    if sha256_file(path) != expected:
        raise ScopeError("shared model_selection helper SHA drifted")
    return _load_module(path, "a_share_top100_bound_helpers")


def load_official_registry(binding: Mapping[str, Any], *, contract_path: Path,
                           input_binding_path: Path, config_path: Path,
                           code_path: Path, test_path: Path):
    deps = binding["shared_dependencies"]
    adapter_path = Path(deps["registry_adapter"]["path"])
    core_path = Path(deps["execution_guard_core"]["path"])
    builder_path = Path(deps["registry_builder"]["path"])
    validator_path = Path(deps["registry_validator"]["path"])
    for name, path in (
        ("registry_adapter", adapter_path),
        ("execution_guard_core", core_path),
        ("registry_builder", builder_path),
        ("registry_validator", validator_path),
    ):
        if sha256_file(path) != deps[name]["sha256"]:
            raise ScopeError(f"shared {name} SHA drifted")

    adapter_dir = adapter_path.parent
    if str(adapter_dir) not in sys.path:
        sys.path.insert(0, str(adapter_dir))
    adapter_module = _load_module(adapter_path, "a_share_top100_official_registry_adapter")
    core = sys.modules.get("execution_guard_core")
    if core is None:
        core = _load_module(core_path, "execution_guard_core")
        sys.modules["execution_guard_core"] = core

    profile = {
        "track_id": TRACK_ID,
        "route_id": ROUTE_ID,
        "registration_mode": "LIVE_EXECUTION",
        "start_stage": "RUN_STARTED",
        "terminal_stage": "TERMINAL",
        "occurred_at_reason": "LIVE_DESCRIPTIVE_EXECUTION_UTC",
        "baseline_id": "OBSERVED_QFQ_DAILY_TOP100_DESCRIPTION_V1",
        "one_variable": "FROZEN_DESCRIPTIVE_GROUPING_AND_MEASUREMENT_CONTRACT",
        "signal_scope": "OBSERVED_A_SHARE_QFQ_DAILY_CLOSE_RETURN",
        "target_scope": "NO_FORWARD_TARGET_OR_LABEL",
        "device": "CPU_RESEARCH_ONLY",
        "evidence_level": "ENGINEERING_CANARY_ONLY",
    }

    class DescriptiveRegistryAdapter(adapter_module.OfficialRegistryAdapter):
        def __init__(self, **kwargs):
            self._route_context: dict[str, dict[str, Any]] = {}
            super().__init__(**kwargs)

        def set_route_context(self, execution_id: str, context: Mapping[str, Any]) -> None:
            self._route_context[str(execution_id)] = dict(context)

        def _event_spec(self, event: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
            spec = super()._event_spec(event, event_type=event_type)
            context = self._route_context.get(str(event.get("execution_id", "")))
            if context is None:
                raise ScopeError("registry event lacks frozen descriptive context")
            spec.update({
                "experiment_id": EXPERIMENT_ID,
                "contract_sha256": context["contract_sha256"],
                "code_sha256": context["code_sha256"],
                "config_sha256": context["config_sha256"],
                "input_sha256": context["input_binding_sha256"],
                "input_binding_sha256": context["input_binding_sha256"],
                "input_paths": list(context["input_paths"]),
                "signal_date_range": context.get("signal_date_range"),
                "target_date_range": None,
                "truth_access": context.get("truth_access", "NOT_OPENED"),
                "sealed_status": context.get("sealed_status", "NOT_ENTERED"),
                "runtime": context["runtime"],
                "device": "CPU_RESEARCH_ONLY",
                "artifact_path": list(context["artifact_paths"]),
                "artifact_paths": list(context["artifact_paths"]),
                "checkpoint_ref": None,
                "deterministic_metrics": dict(context["deterministic_metrics"]),
                "decision": "RUNNING" if event_type == "RUN_STARTED" else str(event.get("terminal_status", "CANNOT_CONFIRM")),
                "stop_reason": "NONE" if event_type == "RUN_STARTED" else str(event.get("error_type") or "COMPLETED_DESCRIPTIVE_ONLY"),
                "evidence_level": context["evidence_level"],
                "input_identity": {
                    "experiment_id": EXPERIMENT_ID,
                    "execution_id": str(event["execution_id"]),
                    "contract_sha256": context["contract_sha256"],
                    "input_binding_sha256": context["input_binding_sha256"],
                    "input_component_sha256": context["input_component_sha256"],
                    "code_sha256_by_file": context["code_sha256_by_file"],
                    "null_reasons": dict(context.get("null_reasons", {})),
                },
                **SAFETY_FLAGS,
            })
            if event_type == "RUN_STARTED":
                spec["target_date_range"] = None
                spec["input_identity"]["null_reasons"].setdefault(
                    "target_date_range", "DESCRIPTIVE_ROUTE_HAS_NO_FORWARD_TARGET_DATE_RANGE"
                )
            if spec.get("checkpoint_ref") is None:
                spec["input_identity"]["null_reasons"].setdefault(
                    "checkpoint_ref", "NO_MODEL_CHECKPOINT_IN_DESCRIPTIVE_ROUTE"
                )
            return spec

    artifact_paths = tuple(Path(p).resolve() for p in (
        contract_path, input_binding_path, config_path, code_path, test_path,
        Path(binding["baseline_record_path"]),
        Path(json.loads(Path(contract_path).read_text(encoding="utf-8")).get(
            "frozen_identity_path", Path(contract_path).with_name("FROZEN_IDENTITY.json")
        )),
    ))
    adapter = DescriptiveRegistryAdapter(
        registry_path=Path(binding["registry_path"]),
        contract_path=contract_path,
        input_binding_path=input_binding_path,
        artifact_paths=artifact_paths,
        builder_path=builder_path,
        validator_path=validator_path,
        event_profile=profile,
    )
    return adapter, core


def _file_manifest(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path).resolve()): sha256_file(Path(path)) for path in paths}


def verify_frozen_identity(*, contract_path: Path, binding_path: Path,
                           config_path: Path, code_path: Path, test_path: Path,
                           expected_baseline: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    identities = {
        "contract_sha256": sha256_file(contract_path),
        "input_binding_sha256": sha256_file(binding_path),
        "config_sha256": sha256_file(config_path),
        "code_sha256": sha256_file(code_path),
        "test_sha256": sha256_file(test_path),
    }
    expected = contract["frozen_identity"]
    for key, actual in identities.items():
        if key == "contract_sha256":
            continue
        if expected.get(key) != actual:
            raise ScopeError(f"frozen {key} mismatch")
    identity_path = Path(contract.get("frozen_identity_path", Path(contract_path).with_name("FROZEN_IDENTITY.json")))
    frozen_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if frozen_identity.get("schema_version") != "A_SHARE_TOP100_FROZEN_IDENTITY_V1":
        raise ScopeError("frozen identity receipt schema mismatch")
    for key, actual in identities.items():
        if frozen_identity.get(key) != actual:
            raise ScopeError(f"frozen identity receipt {key} mismatch")
    if contract.get("status") != "FROZEN_DESCRIPTIVE_ONLY":
        raise ScopeError("contract is not frozen descriptive-only")
    if binding.get("experiment_id") != EXPERIMENT_ID:
        raise ScopeError("input binding experiment identity mismatch")
    if expected_baseline and contract.get("baseline_commit") != expected_baseline:
        raise ScopeError("baseline commit changed")
    if contract.get("input_binding_path") != str(Path(binding_path).resolve()):
        raise ScopeError("contract input binding path mismatch")
    if contract.get("config_path") != str(Path(config_path).resolve()):
        raise ScopeError("contract config path mismatch")
    return contract, binding, {**config, **identities}


def _strict_numeric(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    parse_error = series.notna() & numeric.isna()
    return numeric, parse_error


def read_frozen_panel(binding: Mapping[str, Any], helper: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = binding["sources"]["price_panel"]
    path = Path(source["path"])
    if helper.sha256_file(path) != source["sha256"]:
        raise ScopeError("bound QFQ panel SHA drifted")
    column_map = source["column_map"]
    projected = [column_map[key] for key in ("date", "code", "close", "volume", "amount")]
    raw = pd.read_parquet(path, columns=projected)
    if len(raw) != int(source["rows"]):
        raise ScopeError("projected QFQ panel row count differs from frozen binding")
    rename = {column_map[key]: key for key in ("date", "code", "close", "volume", "amount")}
    out = raw.rename(columns=rename)[list(rename.values())].copy()
    out.columns = ["date", "code", "close", "volume", "amount"]
    out["date"] = helper._parse_bound_dates(out["date"], field_name="QFQ trade_date")
    out["code"] = out["code"].astype("string")
    if out["date"].isna().any() or out["code"].isna().any() or out["code"].eq("").any():
        raise ScopeError("QFQ panel has missing date or code identity")
    if out["code"].ne(out["code"].str.strip()).any():
        raise ScopeError("QFQ code contains surrounding whitespace; exact join is ambiguous")
    if out.duplicated(["date", "code"]).any():
        raise ScopeError("QFQ panel date/code key is not unique")

    parse_errors: dict[str, int] = {}
    for field in ("close", "volume", "amount"):
        values, bad = _strict_numeric(out[field])
        parse_errors[field] = int(bad.sum())
        out[field] = values
    amount_multiplier = float(source["units"].get("amount_multiplier_to_cny", 10000.0))
    if not np.isfinite(amount_multiplier) or amount_multiplier <= 0:
        raise ScopeError("bound amount conversion to CNY is invalid")
    out["amount"] = out["amount"] * amount_multiplier
    out = out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    return out, {
        "source_rows": int(len(out)),
        "distinct_codes": int(out["code"].nunique()),
        "observed_dates": int(out["date"].nunique()),
        "first_date": out["date"].min().date().isoformat(),
        "last_date": out["date"].max().date().isoformat(),
        "parse_error_counts": parse_errors,
        "projection_columns": ["trade_date", "ts_code", "close", "vol", "amount"],
        "amount_multiplier_to_cny": amount_multiplier,
    }


def read_frozen_auxiliary_inputs(binding: Mapping[str, Any], helper: Any,
                                 panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    calendar_source = binding["sources"]["trading_date_index"]
    calendar_raw = pd.read_csv(io.BytesIO(helper.read_bound_archive_member(calendar_source)))
    calendar = helper.normalize_calendar(calendar_raw, panel)

    industry_source = binding["sources"]["industry_classification"]
    industry_raw = pd.read_csv(io.BytesIO(helper.read_bound_archive_member(industry_source)))
    industry = helper.normalize_industry_columns(industry_raw)
    industry["code"] = industry["code"].astype("string")
    if industry["code"].isna().any() or industry["code"].eq("").any():
        raise ScopeError("industry member has an empty exact-code key")
    if industry["code"].ne(industry["code"].str.strip()).any():
        raise ScopeError("industry code contains surrounding whitespace; exact join is ambiguous")
    industry["industry_code"] = industry["industry_code"].astype("string").str.strip()
    industry.loc[industry["industry_code"].isna() | industry["industry_code"].eq(""), "industry_code"] = pd.NA
    name_field = "industry_name" if "industry_name" in industry_raw.columns else ("name" if "name" in industry_raw.columns else None)
    industry["industry_name"] = pd.NA
    if name_field:
        code_field = next(c for c in industry_raw.columns if c in {"ts_code", "code"})
        raw_name = industry_raw[[code_field, name_field]].copy()
        raw_name.columns = ["code", "industry_name"]
        raw_name["code"] = raw_name["code"].astype("string")
        raw_name["industry_name"] = raw_name["industry_name"].astype("string").str.strip()
        raw_name.loc[raw_name["industry_name"].eq(""), "industry_name"] = pd.NA
        name_conflicts = raw_name.groupby("code", dropna=False, sort=False)["industry_name"].nunique()
        if (name_conflicts > 1).any():
            raise ScopeError("industry name field conflicts for an exact code")
        raw_name = raw_name.dropna(subset=["industry_name"]).drop_duplicates("code")
        industry["industry_name"] = industry["code"].map(dict(zip(raw_name["code"], raw_name["industry_name"])))

    normalized_values = industry[["code", "industry_code", "industry_name"]].copy()
    conflicts = normalized_values.groupby("code", dropna=False, sort=False).agg(
        category_count=("industry_code", "nunique"),
        name_count=("industry_name", "nunique"),
    )
    if (conflicts["category_count"] > 1).any() or (conflicts["name_count"] > 1).any():
        raise ScopeError("industry member has conflicting categories or names for an exact code")
    normalized_values = normalized_values.drop_duplicates(["code", "industry_code", "industry_name"])
    if normalized_values["code"].duplicated().any():
        raise ScopeError("industry member has ambiguous duplicate exact-code rows")
    industry = normalized_values.reset_index(drop=True)
    named_categories = industry.dropna(subset=["industry_code", "industry_name"])
    category_name_counts = named_categories.groupby("industry_code", sort=False)["industry_name"].nunique()
    if (category_name_counts > 1).any():
        raise ScopeError("industry name field conflicts within an exact category code")

    index_source = binding["sources"]["index_series"]
    panel_dates = set(pd.DatetimeIndex(panel["date"].drop_duplicates()).normalize())
    index_close = helper.load_bound_index_series(index_source, panel_dates=panel_dates)
    meta = {
        "calendar_rows": int(len(calendar)),
        "calendar_first_date": calendar["date"].min().date().isoformat(),
        "calendar_last_date": calendar["date"].max().date().isoformat(),
        "industry_rows_after_exact_code_dedupe": int(len(industry)),
        "industry_unique_codes": int(industry["code"].nunique()),
        "industry_name_source_field": name_field,
        "industry_name_values_retained_from_member": bool(name_field),
        "index_rows": int(len(index_close)),
        "index_first_date": index_close.index.min().date().isoformat(),
        "index_last_date": index_close.index.max().date().isoformat(),
    }
    return calendar, industry, index_close, meta


def market_regime_by_date(calendar: pd.DataFrame, panel_dates: Sequence[pd.Timestamp],
                          index_close: pd.Series) -> dict[pd.Timestamp, str]:
    ordered = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    dates = pd.DatetimeIndex(ordered["date"]).as_unit("ns").normalize().as_unit("ns")
    date_to_pos = {date: pos for pos, date in enumerate(dates)}
    index_dates = pd.DatetimeIndex(pd.to_datetime(index_close.index)).as_unit("ns").normalize().as_unit("ns")
    index_values = pd.to_numeric(index_close, errors="coerce").to_numpy(dtype=np.float64)
    if index_dates.duplicated().any():
        raise ScopeError("index series contains duplicate dates")
    order = np.argsort(index_dates.asi8, kind="mergesort")
    index_dates = index_dates[order]
    index_values = index_values[order]
    index_ns = index_dates.asi8
    result: dict[pd.Timestamp, str] = {}
    for raw_date in pd.DatetimeIndex(panel_dates).as_unit("ns").normalize().as_unit("ns"):
        date = pd.Timestamp(raw_date).normalize()
        position = date_to_pos.get(date)
        if position is None or position < 1:
            result[date] = "UNKNOWN"
            continue
        t_minus_1 = dates[position - 1]
        p1 = np.searchsorted(index_ns, t_minus_1.value, side="left")
        if p1 >= len(index_ns) or index_ns[p1] != t_minus_1.value:
            result[date] = "UNKNOWN"
            continue
        close_t_minus_1 = float(index_values[p1])
        prior_values = index_values[:p1]
        history = prior_values[-20:]
        if (not np.isfinite(close_t_minus_1) or close_t_minus_1 <= 0 or
                len(history) != 20 or not np.isfinite(history).all() or (history <= 0).any()):
            result[date] = "UNKNOWN"
            continue
        reference_mean = float(np.mean(history, dtype=np.float64))
        result[date] = "AT_OR_ABOVE" if close_t_minus_1 >= reference_mean else "BELOW"
    return result


def prepare_observations(panel: pd.DataFrame, calendar: pd.DataFrame, industry: pd.DataFrame,
                         index_close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"date", "code", "close", "volume", "amount"}
    if not required.issubset(panel.columns):
        raise ScopeError(f"panel lacks fields: {sorted(required - set(panel.columns))}")
    if not {"date", "calendar_index"}.issubset(calendar.columns):
        raise ScopeError("calendar lacks normalized date/index fields")
    if calendar["date"].duplicated().any() or calendar["calendar_index"].duplicated().any():
        raise ScopeError("calendar date and ordinal must be unique")
    ordered_calendar = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    expected_previous = dict(zip(ordered_calendar["date"], ordered_calendar["date"].shift(1)))
    calendar_index_by_date = dict(zip(ordered_calendar["date"], ordered_calendar["calendar_index"]))

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.normalize()
    work["expected_previous_date"] = work["date"].map(expected_previous)
    work["calendar_index"] = work["date"].map(calendar_index_by_date)
    if work["calendar_index"].isna().any():
        raise ScopeError("panel date is outside the shared calendar")
    work = work.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    groups = work.groupby("code", sort=False, observed=True)
    work["previous_row_date"] = groups["date"].shift(1)
    work["previous_row_close"] = groups["close"].shift(1)
    work["missing_previous_row"] = (
        work["expected_previous_date"].isna() |
        work["previous_row_date"].ne(work["expected_previous_date"])
    )
    work["previous_date_matches_calendar"] = (
        work["expected_previous_date"].notna() &
        work["previous_row_date"].eq(work["expected_previous_date"])
    )
    work["previous_close"] = work["previous_row_close"].where(~work["missing_previous_row"])
    close = work["close"].to_numpy(dtype=np.float64)
    previous_close = work["previous_close"].to_numpy(dtype=np.float64, na_value=np.nan)
    volume = work["volume"].to_numpy(dtype=np.float64, na_value=np.nan)
    amount = work["amount"].to_numpy(dtype=np.float64, na_value=np.nan)

    work["close_nonfinite"] = ~np.isfinite(close)
    work["close_nonpositive"] = np.isfinite(close) & (close <= 0)
    work["previous_close_nonfinite"] = (~work["missing_previous_row"].to_numpy()) & ~np.isfinite(previous_close)
    work["previous_close_nonpositive"] = np.isfinite(previous_close) & (previous_close <= 0)
    work["volume_nonfinite"] = ~np.isfinite(volume)
    work["volume_zero"] = np.isfinite(volume) & (volume == 0)
    work["volume_negative"] = np.isfinite(volume) & (volume < 0)
    work["amount_nonfinite"] = ~np.isfinite(amount)
    work["amount_negative"] = np.isfinite(amount) & (amount < 0)
    work["amount_zero_with_positive_volume"] = np.isfinite(amount) & (amount == 0) & np.isfinite(volume) & (volume > 0)
    work["pool_valid"] = (
        ~work["missing_previous_row"] &
        np.isfinite(close) & (close > 0) &
        np.isfinite(previous_close) & (previous_close > 0) &
        np.isfinite(volume) & (volume > 0)
    )
    invalid_flags = (
        work["close_nonfinite"] | work["close_nonpositive"] |
        work["missing_previous_row"] | work["previous_close_nonfinite"] |
        work["previous_close_nonpositive"] | work["volume_nonfinite"] |
        work["volume_zero"] | work["volume_negative"]
    )
    if not invalid_flags.eq(~work["pool_valid"]).all():
        raise ScopeError("pool exclusion reason union does not reconcile to the observed pool")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        returns = close / previous_close - 1.0
    if (~np.isfinite(returns[work["pool_valid"].to_numpy()])).any():
        raise ScopeError("valid close inputs produced a nonfinite return")
    work["return"] = np.where(work["pool_valid"], returns, np.nan)

    category_map = dict(zip(industry["code"].astype(str), industry["industry_code"]))
    named_categories = industry.dropna(subset=["industry_code", "industry_name"]).drop_duplicates("industry_code")
    name_map = dict(zip(named_categories["industry_code"].astype(str), named_categories["industry_name"]))
    work["industry_code"] = work["code"].astype(str).map(category_map)
    work["industry_code"] = work["industry_code"].astype("string").str.strip()
    work.loc[work["industry_code"].isna() | work["industry_code"].eq(""), "industry_code"] = "UNKNOWN"
    work["industry_code"] = work["industry_code"].astype(str)
    work["industry_name"] = work["industry_code"].map(name_map)
    work["lag_market"] = work["date"].map(market_regime_by_date(calendar, work["date"].drop_duplicates(), index_close)).fillna("UNKNOWN")
    work.loc[work["lag_market"].isin(["", "<NA>"]), "lag_market"] = "UNKNOWN"
    work["lag_market"] = work["lag_market"].astype(str)
    work["year"] = work["date"].dt.year.astype(int)
    work["month"] = work["date"].dt.month.astype(int)

    bool_fields = {
        "close_nonfinite": "close_nonfinite_rows",
        "close_nonpositive": "close_nonpositive_rows",
        "missing_previous_row": "missing_previous_row_rows",
        "previous_close_nonfinite": "previous_close_nonfinite_rows",
        "previous_close_nonpositive": "previous_close_nonpositive_rows",
        "volume_nonfinite": "volume_nonfinite_rows",
        "volume_zero": "volume_zero_rows",
        "volume_negative": "volume_negative_rows",
        "amount_nonfinite": "amount_nonfinite_rows",
        "amount_negative": "amount_negative_rows",
        "amount_zero_with_positive_volume": "amount_zero_with_positive_volume_rows",
    }
    coverage = work.groupby("date", sort=True, observed=True).agg(
        source_rows=("code", "size"),
        valid_pool_observations=("pool_valid", "sum"),
        **{output: (field, "sum") for field, output in bool_fields.items()},
    ).reset_index()
    coverage["excluded_from_pool_rows"] = coverage["source_rows"] - coverage["valid_pool_observations"]
    coverage["reason_union_rows"] = coverage["excluded_from_pool_rows"]
    coverage["eligible_for_top100"] = coverage["valid_pool_observations"] >= 100
    coverage["excluded_because_pool_below_100"] = ~coverage["eligible_for_top100"]
    coverage["year"] = coverage["date"].dt.year.astype(int)
    if int(coverage["source_rows"].sum()) != len(work):
        raise ScopeError("daily coverage rows do not reconcile to input rows")
    if int(coverage["valid_pool_observations"].sum()) != int(work["pool_valid"].sum()):
        raise ScopeError("daily eligible-pool observations do not reconcile")
    if int(coverage["reason_union_rows"].sum()) != int((~work["pool_valid"]).sum()):
        raise ScopeError("daily reason union does not reconcile")

    coverage_year = coverage.groupby("year", sort=True).agg(
        source_dates=("date", "nunique"),
        eligible_dates=("eligible_for_top100", "sum"),
        below_100_dates=("excluded_because_pool_below_100", "sum"),
        source_rows=("source_rows", "sum"),
        valid_pool_observations=("valid_pool_observations", "sum"),
        excluded_from_pool_rows=("excluded_from_pool_rows", "sum"),
        reason_union_rows=("reason_union_rows", "sum"),
        **{output: (output, "sum") for output in bool_fields.values()},
    ).reset_index()
    return work, coverage, coverage_year


def _cell_key(period_scope: str, year: int | None, dimension_set: str,
              industry_code: str, month: int | None, market: str | None) -> tuple[Any, ...]:
    return (period_scope, year, dimension_set, industry_code, month, market)


def _add_cell(acc: dict[tuple[Any, ...], dict[str, Any]], *, period_scope: str,
              year: int | None, dimension_set: str, industry_code: str,
              month: int | None, market: str | None, exposure: int,
              hits: int, expected: float, date_year: int) -> None:
    key = _cell_key(period_scope, year, dimension_set, industry_code, month, market)
    row = acc.setdefault(key, {
        "exposure_count": 0,
        "hits_count": 0,
        "expected_hits": 0.0,
        "eligible_date_count": 0,
        "support_years": set(),
    })
    row["exposure_count"] += int(exposure)
    row["hits_count"] += int(hits)
    row["expected_hits"] += float(expected)
    row["eligible_date_count"] += 1
    row["support_years"].add(int(date_year))


def _industry_cell_rows(acc: Mapping[tuple[Any, ...], Mapping[str, Any]],
                        name_map: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for key, values in acc.items():
        period_scope, year, dimension_set, category, month, market = key
        expected = float(values["expected_hits"])
        support_years = sorted(values["support_years"])
        rows.append({
            "period_scope": period_scope,
            "year": year,
            "dimension_set": dimension_set,
            "industry_code": category,
            "industry_name": name_map.get(category),
            "month": month,
            "lag_market": market,
            "exposure_count": int(values["exposure_count"]),
            "hits_count": int(values["hits_count"]),
            "expected_hits": expected,
            "lift": float(values["hits_count"]) / expected if expected > 0 else None,
            "eligible_date_count": int(values["eligible_date_count"]),
            "support_year_count": len(support_years),
            "support_years": ",".join(map(str, support_years)),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["_year_sort"] = frame["year"].fillna(0).astype(int)
    frame["_month_sort"] = frame["month"].fillna(0).astype(int)
    frame["_market_sort"] = frame["lag_market"].fillna("").astype(str)
    frame = frame.sort_values(
        ["period_scope", "dimension_set", "_year_sort", "industry_code", "_month_sort", "_market_sort"],
        kind="mergesort",
    ).drop(columns=["_year_sort", "_month_sort", "_market_sort"]).reset_index(drop=True)
    return frame


def _validate_industry_parity(cells: pd.DataFrame, eligible_days: int) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for dimension in DIMENSION_SETS:
        all_rows = cells[(cells["period_scope"] == "ALL") & (cells["dimension_set"] == dimension)]
        expected_total = float(all_rows["expected_hits"].sum())
        hits_total = int(all_rows["hits_count"].sum())
        checks[f"{dimension}_all_expected_hits_equal_100_per_day"] = bool(abs(expected_total - 100.0 * eligible_days) <= 1e-9)
        checks[f"{dimension}_all_hits_equal_100_per_day"] = bool(hits_total == 100 * eligible_days)

        group_fields = ["dimension_set", "industry_code", "month", "lag_market"]
        by_all = all_rows.groupby(group_fields, dropna=False, sort=True).agg(
            exposure_count=("exposure_count", "sum"), hits_count=("hits_count", "sum"),
            expected_hits=("expected_hits", "sum"), eligible_date_count=("eligible_date_count", "sum"),
        )
        yearly_rows = cells[(cells["period_scope"] == "YEAR") & (cells["dimension_set"] == dimension)]
        by_year = yearly_rows.groupby(group_fields, dropna=False, sort=True).agg(
            exposure_count=("exposure_count", "sum"), hits_count=("hits_count", "sum"),
            expected_hits=("expected_hits", "sum"), eligible_date_count=("eligible_date_count", "sum"),
        )
        by_all, by_year = by_all.align(by_year, join="outer", axis=0)
        parity = True
        for field in ("exposure_count", "hits_count", "eligible_date_count"):
            left = by_all[field].fillna(0).to_numpy(dtype=np.float64)
            right = by_year[field].fillna(0).to_numpy(dtype=np.float64)
            parity = parity and bool(np.array_equal(left, right))
        left = by_all["expected_hits"].fillna(0).to_numpy(dtype=np.float64)
        right = by_year["expected_hits"].fillna(0).to_numpy(dtype=np.float64)
        parity = parity and bool(np.allclose(left, right, rtol=0, atol=1e-9))
        checks[f"{dimension}_aggregate_equals_sum_of_years"] = bool(parity)
        if not all(checks[key] for key in checks if key.startswith(f"{dimension}_")):
            raise ScopeError(f"industry aggregate parity failed for {dimension}")
    return checks


def _summarize_daily_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    metric_columns = [
        "valid_pool_size", "top100_cutoff_return", "top100_mean_return",
        "top100_median_return", "top100_p90_return", "top100_max_return",
        "pool_positive_share", "top100_industry_hhi", "top100_industry_max_share",
    ]
    summaries: list[dict[str, Any]] = []
    dimensions: list[tuple[str, list[str]]] = [
        ("all", []),
        ("month", ["month"]),
        ("lag_market", ["lag_market"]),
        ("month_lag_market", ["month", "lag_market"]),
    ]
    for period_scope in ("ALL", "YEAR"):
        for dimension, keys in dimensions:
            grouping = list(keys)
            if period_scope == "YEAR":
                grouping = ["year", *grouping]
            if not grouping:
                grouped = [((), daily)]
            else:
                grouped = daily.groupby(grouping, dropna=False, sort=True, observed=True)
            for key, frame in grouped:
                key_tuple = key if isinstance(key, tuple) else (key,)
                values = dict(zip(grouping, key_tuple))
                row: dict[str, Any] = {
                    "period_scope": period_scope,
                    "dimension": dimension,
                    "year": values.get("year"),
                    "month": values.get("month"),
                    "lag_market": values.get("lag_market"),
                    "eligible_date_count": int(len(frame)),
                    "weighting": "EQUAL_DAILY",
                }
                for metric in metric_columns:
                    row[f"mean_{metric}"] = float(frame[metric].mean())
                summaries.append(row)
    result = pd.DataFrame(summaries)
    result["_year_sort"] = result["year"].fillna(0).astype(int)
    result["_month_sort"] = result["month"].fillna(0).astype(int)
    result["_market_sort"] = result["lag_market"].fillna("").astype(str)
    return result.sort_values(
        ["period_scope", "dimension", "_year_sort", "_month_sort", "_market_sort"],
        kind="mergesort",
    ).drop(columns=["_year_sort", "_month_sort", "_market_sort"]).reset_index(drop=True)


def _validate_summary_year_parity(summaries: pd.DataFrame) -> bool:
    if summaries.empty:
        return False
    metric_columns = [column for column in summaries.columns if column.startswith("mean_")]
    summary_groups = {
        "all": [],
        "month": ["month"],
        "lag_market": ["lag_market"],
        "month_lag_market": ["month", "lag_market"],
    }
    for dimension, key_fields in summary_groups.items():
        aggregate = summaries.loc[(summaries["period_scope"] == "ALL") & (summaries["dimension"] == dimension)]
        yearly = summaries.loc[(summaries["period_scope"] == "YEAR") & (summaries["dimension"] == dimension)]
        for row in aggregate.itertuples(index=False):
            matched = yearly
            for field in key_fields:
                matched = matched.loc[matched[field].eq(getattr(row, field))]
            if matched.empty:
                return False
            support = matched["eligible_date_count"].to_numpy(dtype=np.float64)
            if float(support.sum()) != float(row.eligible_date_count):
                return False
            for metric in metric_columns:
                weighted = float(np.average(matched[metric].to_numpy(dtype=np.float64), weights=support))
                if not np.isclose(weighted, float(getattr(row, metric)), rtol=0, atol=1e-12):
                    return False
    return True


def select_top100(group: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return the deterministic daily Top100, total cutoff ties, and excluded tied rows."""
    if len(group) < 100:
        raise ScopeError("same-day valid pool has fewer than 100 observations")
    ranked = group.sort_values(["return", "code"], ascending=[False, True], kind="mergesort")
    selected = ranked.iloc[:100].copy()
    if len(selected) != 100 or selected["code"].duplicated().any():
        raise ScopeError("Top100 does not contain exactly 100 unique codes")
    cutoff = float(selected.iloc[-1]["return"])
    boundary_ties = int(group["return"].eq(cutoff).sum())
    selected_at_cutoff = int(selected["return"].eq(cutoff).sum())
    tie_codes = sorted(group.loc[group["return"].eq(cutoff), "code"].astype(str))
    selected_tie_codes = selected.loc[selected["return"].eq(cutoff), "code"].astype(str).tolist()
    if selected_tie_codes != tie_codes[:selected_at_cutoff]:
        raise ScopeError("Top100 cutoff tie ordering is not code-ascending")
    return selected, boundary_ties, boundary_ties - selected_at_cutoff


def compute_statistics(observations: pd.DataFrame, coverage: pd.DataFrame,
                       coverage_year: pd.DataFrame) -> dict[str, Any]:
    valid = observations.loc[observations["pool_valid"]].copy()
    if valid.empty:
        raise ScopeError("observed pool is empty")
    eligible_dates = set(coverage.loc[coverage["eligible_for_top100"], "date"])
    daily_rows: list[dict[str, Any]] = []
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    selected_slot_total = 0
    tie_days = 0
    tie_rows = 0
    tie_excluded_rows = 0
    for date, group in valid.groupby("date", sort=True, observed=True):
        date = pd.Timestamp(date).normalize()
        if date not in eligible_dates:
            continue
        if len(group) < 100:
            raise ScopeError("eligible date has fewer than 100 valid pool observations")
        selected, boundary_ties, excluded_ties = select_top100(group)
        pool_codes = set(group["code"].astype(str))
        if not set(selected["code"].astype(str)).issubset(pool_codes):
            raise ScopeError("Top100 contains a code outside its same-day observed pool")
        cutoff = float(selected.iloc[-1]["return"])
        if boundary_ties > 1:
            tie_days += 1
            tie_rows += boundary_ties
            tie_excluded_rows += excluded_ties
        # Recheck the exact deterministic boundary ordering.
        if len(selected) < 100 or float(selected.iloc[-1]["return"]) != cutoff:
            raise ScopeError("Top100 boundary is not reproducible")

        values = selected["return"].to_numpy(dtype=np.float64)
        pool_returns = group["return"].to_numpy(dtype=np.float64)
        counts = group.groupby("industry_code", sort=True, observed=True).size()
        hit_counts = selected.groupby("industry_code", sort=True, observed=True).size()
        expected_sum = 0.0
        for category, exposure in counts.items():
            category = str(category)
            hit_count = int(hit_counts.get(category, 0))
            expected = 100.0 * int(exposure) / len(group)
            expected_sum += expected
            for scope, month, market in (
                ("industry", None, None),
                ("industry_month", int(date.month), None),
                ("industry_lag_market", None, str(group.iloc[0]["lag_market"])),
                ("industry_month_lag_market", int(date.month), str(group.iloc[0]["lag_market"])),
            ):
                _add_cell(cells, period_scope="ALL", year=None, dimension_set=scope,
                          industry_code=category, month=month, market=market,
                          exposure=int(exposure), hits=hit_count, expected=expected, date_year=int(date.year))
                _add_cell(cells, period_scope="YEAR", year=int(date.year), dimension_set=scope,
                          industry_code=category, month=month, market=market,
                          exposure=int(exposure), hits=hit_count, expected=expected, date_year=int(date.year))
        if abs(expected_sum - 100.0) > 1e-10:
            raise ScopeError("same-day industry expected-hit denominator does not sum to 100")
        industry_shares = selected.groupby("industry_code", sort=True, observed=True).size().to_numpy(dtype=np.float64) / 100.0
        row = {
            "date": date,
            "year": int(date.year),
            "month": int(date.month),
            "lag_market": str(group.iloc[0]["lag_market"]),
            "valid_pool_size": int(len(group)),
            "top100_slots": 100,
            "top100_cutoff_return": cutoff,
            "top100_mean_return": float(np.mean(values, dtype=np.float64)),
            "top100_median_return": float(np.median(values)),
            "top100_p90_return": float(np.quantile(values, 0.9, method="linear")),
            "top100_max_return": float(np.max(values)),
            "pool_positive_share": float(np.count_nonzero(pool_returns > 0) / len(pool_returns)),
            "top100_industry_hhi": float(np.sum(industry_shares * industry_shares)),
            "top100_industry_max_share": float(np.max(industry_shares)),
            "cutoff_tie_pool_count": boundary_ties,
            "cutoff_tie_excluded_count": excluded_ties,
        }
        daily_rows.append(row)
        selected_slot_total += 100

    if not daily_rows:
        raise ScopeError("no date has at least 100 valid same-day observations")
    daily = pd.DataFrame(daily_rows).sort_values("date", kind="mergesort").reset_index(drop=True)
    name_rows = observations.dropna(subset=["industry_name"]).drop_duplicates(["industry_code", "industry_name"])
    name_conflicts = name_rows.groupby("industry_code", sort=False)["industry_name"].nunique()
    if (name_conflicts > 1).any():
        raise ScopeError("industry names conflict within a category code")
    name_rows = name_rows.drop_duplicates("industry_code")
    cells_frame = _industry_cell_rows(cells, dict(zip(name_rows["industry_code"].astype(str), name_rows["industry_name"])))
    summaries = _summarize_daily_metrics(daily)
    coverage_reason_fields = [
        "close_nonfinite", "close_nonpositive", "missing_previous_row", "previous_close_nonfinite",
        "previous_close_nonpositive", "volume_nonfinite", "volume_zero", "volume_negative",
        "amount_nonfinite", "amount_negative", "amount_zero_with_positive_volume",
    ]
    coverage_reason_columns = {
        "close_nonfinite": "close_nonfinite_rows",
        "close_nonpositive": "close_nonpositive_rows",
        "missing_previous_row": "missing_previous_row_rows",
        "previous_close_nonfinite": "previous_close_nonfinite_rows",
        "previous_close_nonpositive": "previous_close_nonpositive_rows",
        "volume_nonfinite": "volume_nonfinite_rows",
        "volume_zero": "volume_zero_rows",
        "volume_negative": "volume_negative_rows",
        "amount_nonfinite": "amount_nonfinite_rows",
        "amount_negative": "amount_negative_rows",
        "amount_zero_with_positive_volume": "amount_zero_with_positive_volume_rows",
    }
    coverage_by_industry_day = observations.groupby(["date", "industry_code"], sort=True, observed=True).agg(
        source_rows=("code", "size"),
        valid_pool_observations=("pool_valid", "sum"),
        **{coverage_reason_columns[field]: (field, "sum") for field in coverage_reason_fields},
    ).reset_index()
    coverage_by_industry_day["excluded_from_pool_rows"] = (
        coverage_by_industry_day["source_rows"] - coverage_by_industry_day["valid_pool_observations"]
    )
    coverage_by_industry_day["reason_union_rows"] = coverage_by_industry_day["excluded_from_pool_rows"]
    coverage_by_industry_day["year"] = coverage_by_industry_day["date"].dt.year.astype(int)
    eligible_date_set = set(coverage.loc[coverage["eligible_for_top100"], "date"])
    coverage_by_industry_day["eligible_top100_date"] = coverage_by_industry_day["date"].isin(eligible_date_set)
    coverage_by_industry_year = coverage_by_industry_day.groupby(["year", "industry_code"], sort=True).agg(
        source_dates=("date", "nunique"),
        eligible_top100_dates=("eligible_top100_date", "sum"),
        source_rows=("source_rows", "sum"),
        valid_pool_observations=("valid_pool_observations", "sum"),
        excluded_from_pool_rows=("excluded_from_pool_rows", "sum"),
        reason_union_rows=("reason_union_rows", "sum"),
        **{column: (column, "sum") for column in coverage_reason_columns.values()},
    ).reset_index()
    coverage_by_industry_all = coverage_by_industry_day.groupby("industry_code", sort=True).agg(
        source_dates=("date", "nunique"),
        eligible_top100_dates=("eligible_top100_date", "sum"),
        source_rows=("source_rows", "sum"),
        valid_pool_observations=("valid_pool_observations", "sum"),
        excluded_from_pool_rows=("excluded_from_pool_rows", "sum"),
        reason_union_rows=("reason_union_rows", "sum"),
        **{column: (column, "sum") for column in coverage_reason_columns.values()},
    ).reset_index()
    coverage_fields = [column for column in coverage_by_industry_all.columns if column != "industry_code"]
    coverage_year_sum = coverage_by_industry_year.groupby("industry_code", sort=True)[coverage_fields].sum().sort_index()
    coverage_all_by_code = coverage_by_industry_all.set_index("industry_code")[coverage_fields].sort_index()
    industry_coverage_annual_parity = coverage_all_by_code.equals(coverage_year_sum)
    eligible_day_count = len(daily)
    if eligible_day_count != len(eligible_dates):
        raise ScopeError("daily Top100 summary count differs from eligible date coverage")
    checks = {
        "same_day_pool_selection": True,
        "valid_rows_use_exact_previous_shared_calendar_date": bool(
            observations.loc[observations["pool_valid"], "previous_date_matches_calendar"].all()
        ),
        "exactly_100_codes_per_eligible_day": bool((daily["top100_slots"] == 100).all()),
        "dates_below_100_excluded": bool(not set(coverage.loc[~coverage["eligible_for_top100"], "date"]) & set(daily["date"])),
        "coverage_source_rows_reconciled": int(coverage["source_rows"].sum()) == len(observations),
        "coverage_pool_rows_reconciled": int(coverage["valid_pool_observations"].sum()) == int(observations["pool_valid"].sum()),
        "reason_union_reconciled": int(coverage["reason_union_rows"].sum()) == int((~observations["pool_valid"]).sum()),
        "industry_coverage_source_rows_reconciled": int(coverage_by_industry_day["source_rows"].sum()) == len(observations),
        "industry_coverage_pool_rows_reconciled": int(coverage_by_industry_day["valid_pool_observations"].sum()) == int(observations["pool_valid"].sum()),
        "industry_coverage_reason_union_reconciled": int(coverage_by_industry_day["reason_union_rows"].sum()) == int((~observations["pool_valid"]).sum()),
        "industry_coverage_year_source_rows_reconciled": int(coverage_by_industry_year["source_rows"].sum()) == len(observations),
        "industry_coverage_all_equals_sum_of_years": bool(industry_coverage_annual_parity),
        "daily_environment_aggregate_equals_equal_date_weighted_years": _validate_summary_year_parity(summaries),
    }
    checks.update(_validate_industry_parity(cells_frame, eligible_day_count))
    if not all(checks.values()):
        raise ScopeError("one or more deterministic completeness checks failed")
    year_date_count = int(coverage_year["eligible_dates"].sum())
    if year_date_count != eligible_day_count:
        raise ScopeError("annual eligible dates do not reconcile to aggregate date count")
    return {
        "daily_environment": daily,
        "coverage_by_day": coverage,
        "coverage_by_year": coverage_year,
        "coverage_by_industry_day": coverage_by_industry_day,
        "coverage_by_industry_year": coverage_by_industry_year,
        "industry_cells": cells_frame,
        "month_market_environment": summaries,
        "qa": {
            "quality_checks": checks,
            "eligible_date_count": eligible_day_count,
            "excluded_below_100_date_count": int((~coverage["eligible_for_top100"]).sum()),
            "selected_slot_count": int(selected_slot_total),
            "industry_expected_hits_all_scope": {
                scope: float(cells_frame.loc[(cells_frame["period_scope"] == "ALL") & (cells_frame["dimension_set"] == scope), "expected_hits"].sum())
                for scope in DIMENSION_SETS
            },
            "industry_hits_all_scope": {
                scope: int(cells_frame.loc[(cells_frame["period_scope"] == "ALL") & (cells_frame["dimension_set"] == scope), "hits_count"].sum())
                for scope in DIMENSION_SETS
            },
            "tie_days": tie_days,
            "cutoff_tie_rows_summed_per_day": tie_rows,
            "cutoff_tie_excluded_rows": tie_excluded_rows,
            "reason_counts_overlap": True,
            "reason_union_count": int(coverage["reason_union_rows"].sum()),
        },
    }


def _date_range(frame: pd.DataFrame) -> dict[str, Any] | None:
    if frame.empty:
        return None
    dates = pd.DatetimeIndex(frame["date"].drop_duplicates()).sort_values()
    return {"count": int(len(dates)), "min": dates.min().date().isoformat(), "max": dates.max().date().isoformat(),
            "scope": "OBSERVED_EXPOSED_HISTORY_DESCRIPTIVE_ONLY"}


def _summary_markdown(stats: Mapping[str, Any], source_meta: Mapping[str, Any], binding: Mapping[str, Any],
                      contract: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    daily = stats["daily_environment"]
    cells = stats["industry_cells"]
    summary = stats["month_market_environment"]
    coverage = stats["coverage_by_year"]
    eligible = len(daily)
    mean_return = float(daily["top100_mean_return"].mean()) if eligible else float("nan")
    mean_pool_positive = float(daily["pool_positive_share"].mean()) if eligible else float("nan")
    mean_hhi = float(daily["top100_industry_hhi"].mean()) if eligible else float("nan")
    mean_maxshare = float(daily["top100_industry_max_share"].mean()) if eligible else float("nan")
    one = cells[(cells["period_scope"] == "ALL") & (cells["dimension_set"] == "industry")].copy()
    one = one.sort_values(["lift", "exposure_count", "industry_code"], ascending=[False, False, True], kind="mergesort")
    over = one[one["lift"] > 1.0].head(10)
    months = summary[(summary["period_scope"] == "ALL") & (summary["dimension"] == "month")].sort_values("month")
    markets = summary[(summary["period_scope"] == "ALL") & (summary["dimension"] == "lag_market")].sort_values("lag_market")
    month_metric = "mean_top100_mean_return"
    lines = [
        "# A股每日涨幅 Top100 环境描述统计",
        "",
        f"范围：{source_meta['first_date']} 至 {source_meta['last_date']}；原始观察 {source_meta['source_rows']:,} 行、{source_meta['distinct_codes']:,} 个代码、{source_meta['observed_dates']:,} 个面板日期。合格日期 {eligible:,} 天，每天严格 100 个观测槽，共 {stats['qa']['selected_slot_count']:,} 槽。",
        "",
        "这是一份已暴露历史的同日涨幅描述统计。Top100 依当天实现收益降序、代码升序排序；它不表示可执行策略、预测、因果关系或经济总收益。月份按当天月份号（1–12）分组；市场状态按 T-1 指数收盘相对截至 T-2 的 20 个已观察收盘均值，分类仅为 AT_OR_ABOVE / BELOW / UNKNOWN，不代表牛熊。月与市场的整体命中 lift 恒为 1，未作为机会结论。",
        "",
        "## 样本与覆盖",
        "",
        f"- 等权日均 Top100 平均涨幅：{mean_return:.4%}；全合格池上涨占比：{mean_pool_positive:.2%}。",
        f"- Top100 行业集中度日均 HHI：{mean_hhi:.4f}；最大行业份额日均：{mean_maxshare:.2%}。UNKNOWN 作为独立行业组参与分母与集中度。",
        f"- N<100 的日期排除 {stats['qa']['excluded_below_100_date_count']} 天；理由计数允许交叠，原因并集 {stats['qa']['reason_union_count']:,} 行与池外行数对账。成交额异常只作覆盖说明，不作筛选条件。",
        "",
        "| 年份 | 面板日期 | Top100 合格日 | 面板行 | 有效池观察 | 池外行 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"| {int(row.year)} | {int(row.source_dates)} | {int(row.eligible_dates)} | {int(row.source_rows):,} | {int(row.valid_pool_observations):,} | {int(row.excluded_from_pool_rows):,} |")
    lines += [
        "",
        "## 行业过度出现",
        "",
        "lift = 行业在每日 Top100 中的命中数 ÷ Σ(100 × 当日该行业有效池数量 / 当日有效池总数)。lift>1 只表示相对当日池暴露的历史代表性较高；支持度列给出日期、年份、暴露数，不使用显著性检验或置信区间。",
        "",
        "| 行业代码 | 行业名（原成员） | 命中 | 期望 | lift | 暴露 | 有效日 | 年份数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in over.itertuples(index=False):
        name = row.industry_name if pd.notna(row.industry_name) else "—"
        lines.append(f"| {row.industry_code} | {name} | {int(row.hits_count)} | {row.expected_hits:.2f} | {row.lift:.3f} | {int(row.exposure_count):,} | {int(row.eligible_date_count)} | {int(row.support_year_count)} |")
    if over.empty:
        lines.append("本样本没有行业 lift 高于 1；所有行业行仍保留在机器表中。")
    lines += [
        "",
        "## 月份与滞后大盘状态",
        "",
        "以下均为合格日期等权均值，涨幅未 winsorize。全量组合、年内版本和行业×月份×市场状态单元见同目录机器表。",
        "",
        "| 月份 | 日期数 | Top100 平均涨幅 | 中位数均值 | P90 均值 | 全池上涨占比 | HHI 均值 | 最大行业份额均值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in months.itertuples(index=False):
        lines.append(f"| {int(row.month):02d} | {int(row.eligible_date_count)} | {getattr(row, month_metric):.3%} | {row.mean_top100_median_return:.3%} | {row.mean_top100_p90_return:.3%} | {row.mean_pool_positive_share:.2%} | {row.mean_top100_industry_hhi:.4f} | {row.mean_top100_industry_max_share:.2%} |")
    lines += [
        "",
        "| T-1 指数状态 | 日期数 | Top100 平均涨幅 | 中位数均值 | P90 均值 | 全池上涨占比 | HHI 均值 | 最大行业份额均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in markets.itertuples(index=False):
        lines.append(f"| {row.lag_market} | {int(row.eligible_date_count)} | {getattr(row, month_metric):.3%} | {row.mean_top100_median_return:.3%} | {row.mean_top100_p90_return:.3%} | {row.mean_pool_positive_share:.2%} | {row.mean_top100_industry_hhi:.4f} | {row.mean_top100_industry_max_share:.2%} |")
    lines += [
        "",
        "## 年度一致性与边界",
        "",
        "下表仅展示全期 lift 排名前列行业的逐年 lift；年份数与 lift>1 年数直接列出，不作稳定性判决。全体行业、月份、滞后市场及交叉单元的逐年明细见 `INDUSTRY_CELLS.csv`；月、状态、月×状态等权日均值见 `MONTH_MARKET_ENVIRONMENT.csv`。",
        "",
        "| 行业代码 | 全期 lift | 有暴露年份 | lift>1 年数 | 逐年 lift（有效日/暴露） |",
        "|---|---:|---:|---:|---|",
    ]
    yearly_industry = cells[(cells["period_scope"] == "YEAR") & (cells["dimension_set"] == "industry")]
    for row in over.itertuples(index=False):
        by_year = yearly_industry.loc[yearly_industry["industry_code"].eq(row.industry_code)].sort_values("year")
        year_values = ", ".join(
            f"{int(y.year)}={y.lift:.2f} ({int(y.eligible_date_count)}日/{int(y.exposure_count)})"
            for y in by_year.itertuples(index=False)
        )
        above_count = int((by_year["lift"] > 1.0).sum())
        lines.append(f"| {row.industry_code} | {row.lift:.3f} | {int(row.support_year_count)} | {above_count} | {year_values or '—'} |")
    lines += [
        "",
        "原因计数可重叠；并集与池外行数逐日对账。",
        "",
        "| 年份 | 非有限收盘 | 非正收盘 | 缺前日记录 | 前收盘异常 | 非法成交量 | 零量 | 负量 | 非有限金额 | 负金额 | 零额正量 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {int(row.year)} | {int(row.close_nonfinite_rows)} | {int(row.close_nonpositive_rows)} | "
            f"{int(row.missing_previous_row_rows)} | {int(row.previous_close_nonfinite_rows + row.previous_close_nonpositive_rows)} | "
            f"{int(row.volume_nonfinite_rows)} | {int(row.volume_zero_rows)} | {int(row.volume_negative_rows)} | "
            f"{int(row.amount_nonfinite_rows)} | {int(row.amount_negative_rows)} | {int(row.amount_zero_with_positive_volume_rows)} |"
        )
    lines += [
        "",
        f"- 2021 与 2026 均为部分年份；源面板在 2021-08-02 出现覆盖扩张，原因未知。面板可见范围不证明历史全 A 股、ST、上市/退市或可交易资格完整。",
        f"- 行业来自 {binding['sources']['industry_classification']['pit_status']}；只保留原成员可提供的行业名称，不外补。",
        f"- CSI 300 指数为 {binding['sources']['index_series']['source_identity_status']}；原始 TDX 身份和官方交易所日历 parity 均 CANNOT_CONFIRM。",
        "- 价格为前复权 QFQ；不是认证经济总收益序列。本报告状态 DIRECTION_ONLY / RESEARCH_ONLY。",
        "- 无 bootstrap、CI 或 p-value；全部历史 exposed sample，不读取未来标签，不训练、不回测、不改变交易门。",
        "",
        "## 冻结身份",
        "",
        f"- contract SHA-256: `{hashes['contract_sha256']}`",
        f"- input binding SHA-256: `{hashes['input_binding_sha256']}`",
        f"- config SHA-256: `{hashes['config_sha256']}`",
        f"- code SHA-256: `{hashes['code_sha256']}`",
        f"- outcome label: `{contract['outcome_label']}`",
        "",
    ]
    return "\n".join(lines)


def build_registry_context(*, stage: str, contract: Mapping[str, Any], binding: Mapping[str, Any],
                           hashes: Mapping[str, str], code_path: Path, test_path: Path,
                           baseline_path: Path, output_paths: Sequence[Path],
                           signal_range: Mapping[str, Any] | None,
                           metrics: Mapping[str, Any] | None = None,
                           terminal: bool = False) -> dict[str, Any]:
    input_sources = binding["sources"]
    input_paths = [
        input_sources["price_panel"]["path"],
        *input_sources["trading_date_index"]["container_parts"],
        *input_sources["industry_classification"]["container_parts"],
        input_sources["index_series"]["path"],
        input_sources["index_series"]["receipt_path"],
        binding["source_binding_path"],
        binding["shared_dependencies"]["model_selection"]["path"],
        binding["shared_dependencies"]["registry_adapter"]["path"],
        binding["shared_dependencies"]["execution_guard_core"]["path"],
        binding["shared_dependencies"]["registry_builder"]["path"],
        binding["shared_dependencies"]["registry_validator"]["path"],
    ]
    artifacts = [
        str(Path(contract["contract_path"]).resolve()),
        str(Path(binding["input_binding_path"]).resolve()),
        str(Path(contract["config_path"]).resolve()),
        str(Path(contract.get("frozen_identity_path", Path(contract["contract_path"]).with_name("FROZEN_IDENTITY.json"))).resolve()),
        str(Path(code_path).resolve()),
        str(Path(test_path).resolve()),
        str(Path(baseline_path).resolve()),
        *[str(Path(path).resolve()) for path in output_paths if Path(path).is_file()],
    ]
    null_reasons = {
        "deterministic_metrics.truth_positive_denominator": "NO_BINARY_FORWARD_TRUTH_LABEL_IN_DESCRIPTIVE_ROUTE",
        "target_date_range": "NO_FORWARD_TARGET_DATE_RANGE_IN_DESCRIPTIVE_ROUTE",
        "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_DESCRIPTIVE_ROUTE",
    }
    if not terminal:
        null_reasons["signal_date_range"] = "DATE_RANGE_NOT_READ_AT_RUN_STARTED"
    code_shas = {
        str(Path(code_path).resolve()): hashes["code_sha256"],
        str(Path(test_path).resolve()): hashes["test_sha256"],
    }
    det_metrics = {
        "truth_positive_denominator": None,
        "selection_slots": int((metrics or {}).get("selection_slots", 0)),
        "source_rows": (metrics or {}).get("source_rows"),
        "valid_pool_observations": (metrics or {}).get("valid_pool_observations"),
        "eligible_date_count": (metrics or {}).get("eligible_date_count", 0),
        "tie_days": (metrics or {}).get("tie_days"),
        "execution_kind": "NO_FIT_ENGINEERING_CANARY",
        "planned_real_fit_count": 0,
        "real_fit_budget_debit": 0,
        "real_fit_returned_count": 0,
        "fit_call_entered": False,
    }
    for key, value in list(det_metrics.items()):
        if value is None:
            null_reasons[f"deterministic_metrics.{key}"] = "VALUE_NOT_AVAILABLE_AT_THIS_LIFECYCLE_STAGE"
    runtime = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    return {
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "input_paths": list(dict.fromkeys(map(str, map(Path, input_paths)))),
        "artifact_paths": list(dict.fromkeys(artifacts)),
        "input_component_sha256": {
            "price_panel": input_sources["price_panel"]["sha256"],
            "calendar_member": input_sources["trading_date_index"]["member_sha256"],
            "industry_member": input_sources["industry_classification"]["member_sha256"],
            "index_series": input_sources["index_series"]["sha256"],
            "model_selection_helper": binding["shared_dependencies"]["model_selection"]["sha256"],
            "registry_adapter": binding["shared_dependencies"]["registry_adapter"]["sha256"],
            "execution_guard_core": binding["shared_dependencies"]["execution_guard_core"]["sha256"],
            "registry_builder": binding["shared_dependencies"]["registry_builder"]["sha256"],
            "registry_validator": binding["shared_dependencies"]["registry_validator"]["sha256"],
        },
        "code_sha256_by_file": code_shas,
        "signal_date_range": signal_range,
        "truth_access": "OPENED_EXPOSED_HISTORY" if terminal else "NOT_OPENED",
        "sealed_status": "EXPOSED" if terminal else "NOT_ENTERED",
        "runtime": runtime,
        "evidence_level": "DIRECTION_ONLY" if stage == "FULL_STATS" else "ENGINEERING_CANARY_ONLY",
        "deterministic_metrics": det_metrics,
        "null_reasons": null_reasons,
    }


def _claim_stage(adapter: Any, core: Any, *, stage: str, execution_id: str,
                 contract_sha256: str, input_binding_sha256: str,
                 context: Mapping[str, Any]):
    adapter.set_route_context(execution_id, context)
    event = {
        "experiment_id": EXPERIMENT_ID,
        "execution_id": execution_id,
        "arm": stage,
        "contract_sha256": contract_sha256,
        "input_binding_sha256": input_binding_sha256,
        "execution_kind": "NO_FIT_ENGINEERING_CANARY",
        "planned_real_fit_count": 0,
        "real_fit_budget_debit": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    adapter.claim_run_started(event, max_real_fit_debits=0)
    return core.ExecutionClaim(
        experiment_id=EXPERIMENT_ID,
        execution_id=execution_id,
        arm=stage,
        contract_sha256=contract_sha256,
        input_binding_sha256=input_binding_sha256,
        execution_kind="NO_FIT_ENGINEERING_CANARY",
        real_fit_budget_debit=0,
    )


def _terminalize_stage(adapter: Any, claim: Any, *, terminal_status: str,
                       metrics: Mapping[str, Any], error: BaseException | None = None) -> None:
    context = adapter._route_context[claim.execution_id]
    updated = dict(context)
    det = dict(context["deterministic_metrics"])
    det.update(metrics)
    det.update({
        "execution_kind": "NO_FIT_ENGINEERING_CANARY",
        "planned_real_fit_count": 0,
        "real_fit_budget_debit": 0,
        "real_fit_returned_count": 0,
        "fit_call_entered": False,
    })
    updated["deterministic_metrics"] = det
    updated["truth_access"] = "OPENED_EXPOSED_HISTORY"
    updated["sealed_status"] = "EXPOSED"
    updated["null_reasons"] = dict(updated.get("null_reasons", {}))
    updated["null_reasons"].pop("signal_date_range", None)
    adapter.set_route_context(claim.execution_id, updated)
    event = {
        "experiment_id": claim.experiment_id,
        "execution_id": claim.execution_id,
        "arm": claim.arm,
        "contract_sha256": claim.contract_sha256,
        "input_binding_sha256": claim.input_binding_sha256,
        "execution_kind": claim.execution_kind,
        "terminal_status": terminal_status,
        "real_fit_returned_count": 0,
        "fit_call_entered": False,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error_type": None if error is None else f"{type(error).__name__}:{str(error)[:240]}",
    }
    adapter.terminalize(claim, event)


def _registry_global_counts(adapter: Any) -> dict[str, Any]:
    with adapter._locked():
        _, events = adapter._read_validated()
    return {
        "routes": len({str(event.get("route_id")) for event in events}),
        "events": len(events),
        "executions": len({str(event.get("execution_id")) for event in events}),
        "corrections": sum(event.get("event_type") == "CORRECTION" for event in events),
        "unregistered_route_count": None,
        "unregistered_route_count_reason": "GLOBAL_ROUTE_INVENTORY_NOT_AUDITED_IN_THIS_BATCH",
        "contract_only_count": None,
        "contract_only_count_reason": "GLOBAL_CONTRACT_ONLY_INVENTORY_NOT_AUDITED_IN_THIS_BATCH",
    }


def _stage_files_exist(output_dir: Path) -> bool:
    return any(output_dir.iterdir()) if output_dir.exists() else False


def run_all(*, contract_path: Path, binding_path: Path, config_path: Path,
            code_path: Path, test_path: Path, output_dir: Path,
            baseline_path: Path, expected_baseline: str) -> dict[str, Any]:
    contract, binding, hashes = verify_frozen_identity(
        contract_path=contract_path, binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
        expected_baseline=expected_baseline,
    )
    if _stage_files_exist(output_dir):
        raise ScopeError(f"output directory is not empty: {output_dir}")
    helper = load_bound_helpers(binding)
    adapter, core = load_official_registry(
        binding, contract_path=contract_path, input_binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
    )
    existing = list(adapter.list_events(EXPERIMENT_ID))
    if existing:
        raise ScopeError("this frozen execution identity already has registry events; no retry is allowed")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_clock = time.monotonic()

    # FAST_BIND: the durable RUN_STARTED precedes every bound data read.
    fast_context = build_registry_context(
        stage="FAST_BIND", contract=contract, binding=binding, hashes=hashes,
        code_path=code_path, test_path=test_path, baseline_path=baseline_path,
        output_paths=(), signal_range=None, metrics={"selection_slots": 0}, terminal=False,
    )
    fast_claim = _claim_stage(
        adapter, core, stage="FAST_BIND", execution_id=RUN_IDS["fast_bind"],
        contract_sha256=hashes["contract_sha256"], input_binding_sha256=hashes["input_binding_sha256"],
        context=fast_context,
    )
    try:
        panel, panel_meta = read_frozen_panel(binding, helper)
        calendar, industry, index_close, auxiliary_meta = read_frozen_auxiliary_inputs(binding, helper, panel)
        input_receipt_path = output_dir / "FAST_BIND_RECEIPT.json"
        fast_receipt = {
            "schema_version": "A_SHARE_TOP100_FAST_BIND_RECEIPT_V1",
            "execution_id": RUN_IDS["fast_bind"],
            "experiment_id": EXPERIMENT_ID,
            "route_id": ROUTE_ID,
            "input_binding_sha256": hashes["input_binding_sha256"],
            "contract_sha256": hashes["contract_sha256"],
            "config_sha256": hashes["config_sha256"],
            "code_sha256": hashes["code_sha256"],
            "panel": panel_meta,
            "auxiliary": auxiliary_meta,
            "security_flags": SAFETY_FLAGS,
            "status": "FAST_BIND_COMPLETE",
        }
        write_json(input_receipt_path, fast_receipt)
        fast_context = build_registry_context(
            stage="FAST_BIND", contract=contract, binding=binding, hashes=hashes,
            code_path=code_path, test_path=test_path, baseline_path=baseline_path,
            output_paths=(input_receipt_path,),
            signal_range={"count": panel_meta["observed_dates"], "min": panel_meta["first_date"],
                          "max": panel_meta["last_date"], "scope": "OBSERVED_EXPOSED_HISTORY_DESCRIPTIVE_ONLY"},
            metrics={"selection_slots": 0, "source_rows": panel_meta["source_rows"],
                     "valid_pool_observations": None, "eligible_date_count": 0},
            terminal=True,
        )
        adapter.set_route_context(RUN_IDS["fast_bind"], fast_context)
        _terminalize_stage(adapter, fast_claim, terminal_status="COMPLETED", metrics={
            "source_rows": panel_meta["source_rows"],
            "observed_date_count": panel_meta["observed_dates"],
            "distinct_code_count": panel_meta["distinct_codes"],
            "calendar_row_count": auxiliary_meta["calendar_rows"],
            "industry_code_count": auxiliary_meta["industry_unique_codes"],
            "index_row_count": auxiliary_meta["index_rows"],
        })
    except BaseException as exc:
        _terminalize_stage(adapter, fast_claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise

    # The canary date is frozen in the contract. Only its date and previous
    # shared-calendar date are prepared after this RUN_STARTED is durable.
    canary_date = pd.Timestamp(contract["canary"]["date"]).normalize()
    calendar_order = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    canary_calendar = calendar_order.loc[calendar_order["date"].eq(canary_date)]
    if canary_calendar.empty:
        raise ScopeError("frozen canary date is absent from the shared calendar")
    canary_position = int(canary_calendar["calendar_index"].iloc[0])
    previous_calendar = calendar_order.loc[calendar_order["calendar_index"].eq(canary_position - 1), "date"]
    if previous_calendar.empty:
        raise ScopeError("frozen canary has no preceding shared-calendar date")
    previous_date = pd.Timestamp(previous_calendar.iloc[0]).normalize()
    canary_context = build_registry_context(
        stage="CANARY", contract=contract, binding=binding, hashes=hashes,
        code_path=code_path, test_path=test_path, baseline_path=baseline_path,
        output_paths=(output_dir / "FAST_BIND_RECEIPT.json",),
        signal_range={"count": 1, "min": canary_date.date().isoformat(), "max": canary_date.date().isoformat(),
                      "scope": "FROZEN_SINGLE_DATE_REAL_DATA_CANARY"},
        metrics={"selection_slots": 0, "source_rows": None,
                 "valid_pool_observations": None, "eligible_date_count": 0},
        terminal=False,
    )
    canary_claim = _claim_stage(
        adapter, core, stage="CANARY", execution_id=RUN_IDS["canary"],
        contract_sha256=hashes["contract_sha256"], input_binding_sha256=hashes["input_binding_sha256"],
        context=canary_context,
    )
    try:
        if canary_date not in set(panel["date"]):
            raise ScopeError("frozen canary date is absent from the bound panel")
        canary_panel = panel.loc[panel["date"].isin({previous_date, canary_date})].copy()
        canary_observations, canary_coverage, canary_coverage_year = prepare_observations(
            canary_panel, calendar, industry, index_close
        )
        canary_day_coverage = canary_coverage.loc[canary_coverage["date"].eq(canary_date)].copy()
        if canary_day_coverage.empty or int(canary_day_coverage["valid_pool_observations"].iloc[0]) < 100:
            raise ScopeError("frozen canary date has fewer than 100 valid same-day observations")
        canary_stats = compute_statistics(
            canary_observations.loc[canary_observations["date"].eq(canary_date)].copy(),
            canary_day_coverage,
            canary_coverage_year.loc[canary_coverage_year["year"].eq(canary_date.year)].copy(),
        )
        canary_receipt_path = output_dir / "CANARY_RECEIPT.json"
        canary_daily = canary_stats["daily_environment"].iloc[0]
        canary_receipt = {
            "schema_version": "A_SHARE_TOP100_CANARY_RECEIPT_V1",
            "execution_id": RUN_IDS["canary"],
            "experiment_id": EXPERIMENT_ID,
            "date": canary_date.date().isoformat(),
            "valid_pool_size": int(canary_daily["valid_pool_size"]),
            "selection_slots": int(canary_daily["top100_slots"]),
            "cutoff_tie_pool_count": int(canary_daily["cutoff_tie_pool_count"]),
            "cutoff_tie_excluded_count": int(canary_daily["cutoff_tie_excluded_count"]),
            "industry_expected_sum": 100.0,
            "same_pool_check": True,
            "quality_checks": canary_stats["qa"]["quality_checks"],
            "result_scope": "ONE_DATE_ENGINEERING_CANARY_ONLY",
            "security_flags": SAFETY_FLAGS,
        }
        write_json(canary_receipt_path, canary_receipt)
        canary_context = build_registry_context(
            stage="CANARY", contract=contract, binding=binding, hashes=hashes,
            code_path=code_path, test_path=test_path, baseline_path=baseline_path,
            output_paths=(output_dir / "FAST_BIND_RECEIPT.json", canary_receipt_path),
            signal_range={"count": 1, "min": canary_date.date().isoformat(), "max": canary_date.date().isoformat(),
                          "scope": "FROZEN_SINGLE_DATE_REAL_DATA_CANARY"},
            metrics={"selection_slots": 100, "source_rows": int(canary_day_coverage["source_rows"].iloc[0]),
                     "valid_pool_observations": int(canary_daily["valid_pool_size"]), "eligible_date_count": 1,
                     "tie_days": int(canary_daily["cutoff_tie_pool_count"] > 1)},
            terminal=True,
        )
        adapter.set_route_context(RUN_IDS["canary"], canary_context)
        _terminalize_stage(adapter, canary_claim, terminal_status="COMPLETED", metrics={
            "selection_slots": 100,
            "source_rows": int(canary_day_coverage["source_rows"].iloc[0]),
            "valid_pool_observations": int(canary_daily["valid_pool_size"]),
            "eligible_date_count": 1,
            "cutoff_tie_pool_count": int(canary_daily["cutoff_tie_pool_count"]),
            "cutoff_tie_excluded_count": int(canary_daily["cutoff_tie_excluded_count"]),
        })
    except BaseException as exc:
        _terminalize_stage(adapter, canary_claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise

    # Full history descriptive evaluation is registered before full-panel date,
    # pool, return, classification, or Top100 calculations begin.
    full_context = build_registry_context(
        stage="FULL_STATS", contract=contract, binding=binding, hashes=hashes,
        code_path=code_path, test_path=test_path, baseline_path=baseline_path,
        output_paths=(output_dir / "FAST_BIND_RECEIPT.json", output_dir / "CANARY_RECEIPT.json"),
        signal_range={"count": panel_meta["observed_dates"],
                      "min": panel_meta["first_date"],
                      "max": panel_meta["last_date"],
                      "scope": "ALL_BOUND_OBSERVED_PANEL_DATES"},
        metrics={"selection_slots": 0, "source_rows": panel_meta["source_rows"],
                 "valid_pool_observations": None, "eligible_date_count": 0},
        terminal=False,
    )
    full_claim = _claim_stage(
        adapter, core, stage="FULL_STATS", execution_id=RUN_IDS["full_stats"],
        contract_sha256=hashes["contract_sha256"], input_binding_sha256=hashes["input_binding_sha256"],
        context=full_context,
    )
    try:
        observations, coverage, coverage_year = prepare_observations(panel, calendar, industry, index_close)
        stats = compute_statistics(observations, coverage, coverage_year)
        output_files = {
            "COVERAGE_BY_DAY.csv": stats["coverage_by_day"],
            "COVERAGE_BY_YEAR.csv": stats["coverage_by_year"],
            "COVERAGE_BY_INDUSTRY_DAY.csv": stats["coverage_by_industry_day"],
            "COVERAGE_BY_INDUSTRY_YEAR.csv": stats["coverage_by_industry_year"],
            "DAILY_ENVIRONMENT.csv": stats["daily_environment"],
            "INDUSTRY_CELLS.csv": stats["industry_cells"],
            "MONTH_MARKET_ENVIRONMENT.csv": stats["month_market_environment"],
        }
        for name, frame in output_files.items():
            write_csv(output_dir / name, frame)
        report = _summary_markdown(stats, {
            "source_rows": int(len(observations)),
            "distinct_codes": int(observations["code"].nunique()),
            "observed_dates": int(observations["date"].nunique()),
            "first_date": observations["date"].min().date().isoformat(),
            "last_date": observations["date"].max().date().isoformat(),
        }, binding, contract, hashes)
        report_path = output_dir / "COMPACT_REPORT.md"
        _write_new_or_identical(report_path, report.encode("utf-8"))
        output_hashes = _file_manifest([*(output_dir / name for name in output_files), report_path])
        elapsed = time.monotonic() - start_clock
        full_receipt_path = output_dir / "FULL_STATS_RECEIPT.json"
        full_receipt = {
            "schema_version": "A_SHARE_TOP100_FULL_STATS_RECEIPT_V1",
            "experiment_id": EXPERIMENT_ID,
            "route_id": ROUTE_ID,
            "execution_id": RUN_IDS["full_stats"],
            "status": "DIRECTION_ONLY",
            "data_scope": "ALL_EXPOSED_BOUND_PANEL_HISTORY",
            "source_rows": int(len(observations)),
            "valid_pool_observations": int(observations["pool_valid"].sum()),
            "eligible_date_count": int(stats["qa"]["eligible_date_count"]),
            "excluded_below_100_date_count": int(stats["qa"]["excluded_below_100_date_count"]),
            "selection_slots": int(stats["qa"]["selected_slot_count"]),
            "signal_date_range": _date_range(observations),
            "input_binding_sha256": hashes["input_binding_sha256"],
            "contract_sha256": hashes["contract_sha256"],
            "config_sha256": hashes["config_sha256"],
            "code_sha256": hashes["code_sha256"],
            "output_sha256": output_hashes,
            "quality_checks": stats["qa"]["quality_checks"],
            "industry_expected_hits_all_scope": stats["qa"]["industry_expected_hits_all_scope"],
            "industry_hits_all_scope": stats["qa"]["industry_hits_all_scope"],
            "tie_days": stats["qa"]["tie_days"],
            "cutoff_tie_rows_summed_per_day": stats["qa"]["cutoff_tie_rows_summed_per_day"],
            "cutoff_tie_excluded_rows": stats["qa"]["cutoff_tie_excluded_rows"],
            "reason_counts_overlap": True,
            "reason_union_count": stats["qa"]["reason_union_count"],
            "runtime_seconds_from_fastbind_start": elapsed,
            "safety_flags": SAFETY_FLAGS,
            "limitations": contract["limitations"],
        }
        write_json(full_receipt_path, full_receipt)
        terminal_artifacts = [*(output_dir / name for name in output_files), report_path, full_receipt_path,
                              output_dir / "FAST_BIND_RECEIPT.json", output_dir / "CANARY_RECEIPT.json"]
        full_context = build_registry_context(
            stage="FULL_STATS", contract=contract, binding=binding, hashes=hashes,
            code_path=code_path, test_path=test_path, baseline_path=baseline_path,
            output_paths=terminal_artifacts,
            signal_range=_date_range(observations),
            metrics={"selection_slots": int(stats["qa"]["selected_slot_count"]),
                     "source_rows": int(len(observations)),
                     "valid_pool_observations": int(observations["pool_valid"].sum()),
                     "eligible_date_count": int(stats["qa"]["eligible_date_count"]),
                     "tie_days": int(stats["qa"]["tie_days"])},
            terminal=True,
        )
        adapter.set_route_context(RUN_IDS["full_stats"], full_context)
        _terminalize_stage(adapter, full_claim, terminal_status="COMPLETED", metrics={
            "selection_slots": int(stats["qa"]["selected_slot_count"]),
            "source_rows": int(len(observations)),
            "valid_pool_observations": int(observations["pool_valid"].sum()),
            "eligible_date_count": int(stats["qa"]["eligible_date_count"]),
            "excluded_below_100_date_count": int(stats["qa"]["excluded_below_100_date_count"]),
            "tie_days": int(stats["qa"]["tie_days"]),
            "cutoff_tie_rows_summed_per_day": int(stats["qa"]["cutoff_tie_rows_summed_per_day"]),
            "cutoff_tie_excluded_rows": int(stats["qa"]["cutoff_tie_excluded_rows"]),
        })
    except BaseException as exc:
        _terminalize_stage(adapter, full_claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise

    global_counts = _registry_global_counts(adapter)
    scoped = list(adapter.list_events(EXPERIMENT_ID))
    scoped_counts = {
        "route_id": ROUTE_ID,
        "routes": len({str(event.get("route_id")) for event in scoped}),
        "events": len(scoped),
        "executions": len({str(event.get("execution_id")) for event in scoped}),
        "corrections": sum(event.get("event_type") == "CORRECTION" for event in scoped),
        "open_execution_ids": sorted({str(event.get("execution_id")) for event in scoped if event.get("event_type") == "RUN_STARTED"} -
                                       {str(event.get("execution_id")) for event in scoped if event.get("event_type") == "TERMINAL"}),
    }
    index_path = binding["machine_index_path"]
    index = {
        "schema_version": "A_SHARE_TOP100_ENVIRONMENT_INDEX_V1",
        "experiment_id": EXPERIMENT_ID,
        "route_id": ROUTE_ID,
        "status": "DIRECTION_ONLY",
        "batch_path": str(output_dir.resolve()),
        "baseline_commit": contract["baseline_commit"],
        "registry_path": str(Path(binding["registry_path"]).resolve()),
        "registry_sha256": sha256_file(Path(binding["registry_path"])),
        "registry_counts_global": global_counts,
        "registry_counts_scoped": scoped_counts,
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "output_sha256": output_hashes,
        "safety_flags": SAFETY_FLAGS,
        "registered_execution_ids": [RUN_IDS[key] for key in ("fast_bind", "canary", "full_stats")],
        "planned_review_execution_id": RUN_IDS["review_sample"],
        "unregistered_route_count_this_batch": 0,
        "contract_only_execution_count_this_batch": 0,
    }
    write_json(Path(index_path), index)
    return {
        "full_receipt": full_receipt,
        "index": index,
        "scoped_counts": scoped_counts,
        "global_counts": global_counts,
        "elapsed_seconds": time.monotonic() - start_clock,
    }


def _review_context(*, contract: Mapping[str, Any], binding: Mapping[str, Any],
                    hashes: Mapping[str, str], code_path: Path, test_path: Path,
                    baseline_path: Path, receipt_path: Path | None = None,
                    metrics: Mapping[str, Any] | None = None, terminal: bool = False) -> dict[str, Any]:
    date = pd.Timestamp(contract["canary"]["date"]).normalize().date().isoformat()
    return build_registry_context(
        stage="REVIEW_REAL_SAMPLE", contract=contract, binding=binding, hashes=hashes,
        code_path=code_path, test_path=test_path, baseline_path=baseline_path,
        output_paths=() if receipt_path is None else (receipt_path,),
        signal_range={"count": 1, "min": date, "max": date, "scope": "FROZEN_SINGLE_DATE_INDEPENDENT_REVIEW_SAMPLE"},
        metrics={"selection_slots": 0, "source_rows": None,
                 "valid_pool_observations": None, "eligible_date_count": 0, **dict(metrics or {})},
        terminal=terminal,
    )


def register_review_start(*, contract_path: Path, binding_path: Path, config_path: Path,
                          code_path: Path, test_path: Path, baseline_path: Path,
                          expected_baseline: str) -> dict[str, Any]:
    """Append a review RUN_STARTED event before an independent real-data canary."""
    contract, binding, hashes = verify_frozen_identity(
        contract_path=contract_path, binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
        expected_baseline=expected_baseline,
    )
    adapter, core = load_official_registry(
        binding, contract_path=contract_path, input_binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
    )
    if any(event.get("execution_id") == RUN_IDS["review_sample"] for event in adapter.list_events(EXPERIMENT_ID)):
        raise ScopeError("independent review sample execution is already registered; no retry is allowed")
    context = _review_context(
        contract=contract, binding=binding, hashes=hashes, code_path=code_path,
        test_path=test_path, baseline_path=baseline_path,
    )
    claim = _claim_stage(
        adapter, core, stage="REVIEW_REAL_SAMPLE", execution_id=RUN_IDS["review_sample"],
        contract_sha256=hashes["contract_sha256"], input_binding_sha256=hashes["input_binding_sha256"],
        context=context,
    )
    return {"execution_id": claim.execution_id, "event": "RUN_STARTED", "status": "REGISTERED"}


def run_review_sample(*, contract_path: Path, binding_path: Path, config_path: Path,
                      code_path: Path, test_path: Path, baseline_path: Path,
                      expected_baseline: str) -> dict[str, Any]:
    """Read-only one-date independent sample; refuses to run without its own registered start."""
    contract, binding, hashes = verify_frozen_identity(
        contract_path=contract_path, binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
        expected_baseline=expected_baseline,
    )
    adapter, _ = load_official_registry(
        binding, contract_path=contract_path, input_binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
    )
    review_events = [event for event in adapter.list_events(EXPERIMENT_ID)
                     if event.get("execution_id") == RUN_IDS["review_sample"]]
    if len(review_events) != 1 or review_events[0].get("event_type") != "RUN_STARTED":
        raise ScopeError("independent review sample must have exactly one registered RUN_STARTED before data access")
    if (review_events[0].get("contract_sha256") != hashes["contract_sha256"] or
            review_events[0].get("input_binding_sha256") != hashes["input_binding_sha256"]):
        raise ScopeError("independent review RUN_STARTED identity differs from the frozen input")

    helper = load_bound_helpers(binding)
    panel, panel_meta = read_frozen_panel(binding, helper)
    calendar, industry, index_close, _ = read_frozen_auxiliary_inputs(binding, helper, panel)
    canary_date = pd.Timestamp(contract["canary"]["date"]).normalize()
    calendar_order = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    canary_calendar = calendar_order.loc[calendar_order["date"].eq(canary_date)]
    if canary_calendar.empty:
        raise ScopeError("frozen review sample date is absent from the shared calendar")
    position = int(canary_calendar["calendar_index"].iloc[0])
    previous_rows = calendar_order.loc[calendar_order["calendar_index"].eq(position - 1), "date"]
    if previous_rows.empty:
        raise ScopeError("frozen review sample date has no preceding shared-calendar date")
    previous_date = pd.Timestamp(previous_rows.iloc[0]).normalize()
    if canary_date not in set(panel["date"]):
        raise ScopeError("frozen review sample date is absent from the bound panel")
    sample_panel = panel.loc[panel["date"].isin({previous_date, canary_date})].copy()
    observations, coverage, coverage_year = prepare_observations(sample_panel, calendar, industry, index_close)
    day_coverage = coverage.loc[coverage["date"].eq(canary_date)].copy()
    if day_coverage.empty or int(day_coverage["valid_pool_observations"].iloc[0]) < 100:
        raise ScopeError("frozen review sample date has fewer than 100 valid same-day observations")
    sample_stats = compute_statistics(
        observations.loc[observations["date"].eq(canary_date)].copy(), day_coverage,
        coverage_year.loc[coverage_year["year"].eq(canary_date.year)].copy(),
    )
    daily = sample_stats["daily_environment"].iloc[0]
    unknown = sample_stats["industry_cells"].query(
        "period_scope == 'ALL' and dimension_set == 'industry' and industry_code == 'UNKNOWN'"
    )
    receipt = {
        "schema_version": "A_SHARE_TOP100_REVIEW_SAMPLE_RECEIPT_V1",
        "experiment_id": EXPERIMENT_ID,
        "route_id": ROUTE_ID,
        "execution_id": RUN_IDS["review_sample"],
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "test_sha256": hashes["test_sha256"],
        "date": canary_date.date().isoformat(),
        "panel_source_rows": panel_meta["source_rows"],
        "day_source_rows": int(day_coverage["source_rows"].iloc[0]),
        "valid_pool_size": int(daily["valid_pool_size"]),
        "selection_slots": int(daily["top100_slots"]),
        "cutoff_tie_pool_count": int(daily["cutoff_tie_pool_count"]),
        "cutoff_tie_excluded_count": int(daily["cutoff_tie_excluded_count"]),
        "industry_expected_hits_sum": 100.0,
        "unknown_industry_exposure": 0 if unknown.empty else int(unknown["exposure_count"].iloc[0]),
        "unknown_industry_hits": 0 if unknown.empty else int(unknown["hits_count"].iloc[0]),
        "quality_checks": sample_stats["qa"]["quality_checks"],
        "same_pool_check": True,
        "status": "PASS_SAMPLE" if all(sample_stats["qa"]["quality_checks"].values()) else "FAIL_SAMPLE",
        "evidence_level": "ENGINEERING_CANARY_ONLY",
        "security_flags": SAFETY_FLAGS,
    }
    return receipt


def terminalize_review_sample(*, contract_path: Path, binding_path: Path, config_path: Path,
                              code_path: Path, test_path: Path, baseline_path: Path,
                              receipt_path: Path, expected_baseline: str) -> dict[str, Any]:
    """Append review terminal using a reviewer receipt; the independent reviewer never writes registry."""
    contract, binding, hashes = verify_frozen_identity(
        contract_path=contract_path, binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
        expected_baseline=expected_baseline,
    )
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "A_SHARE_TOP100_REVIEW_SAMPLE_RECEIPT_V1":
        raise ScopeError("review sample receipt schema mismatch")
    if receipt.get("execution_id") != RUN_IDS["review_sample"]:
        raise ScopeError("review sample receipt execution identity mismatch")
    for key in ("contract_sha256", "input_binding_sha256", "config_sha256", "code_sha256", "test_sha256"):
        identity_key = "input_binding_sha256" if key == "input_binding_sha256" else key
        if receipt.get(key) != hashes[identity_key]:
            raise ScopeError(f"review sample receipt {key} mismatch")
    if receipt.get("date") != pd.Timestamp(contract["canary"]["date"]).normalize().date().isoformat():
        raise ScopeError("review sample receipt date differs from the frozen canary date")
    if receipt.get("status") not in {"PASS_SAMPLE", "FAIL_SAMPLE"}:
        raise ScopeError("review sample receipt status is invalid")

    adapter, core = load_official_registry(
        binding, contract_path=contract_path, input_binding_path=binding_path,
        config_path=config_path, code_path=code_path, test_path=test_path,
    )
    review_events = [event for event in adapter.list_events(EXPERIMENT_ID)
                     if event.get("execution_id") == RUN_IDS["review_sample"]]
    if len(review_events) != 1 or review_events[0].get("event_type") != "RUN_STARTED":
        raise ScopeError("review terminal requires exactly one unmatched RUN_STARTED")
    metrics = {
        "selection_slots": int(receipt.get("selection_slots", 0)),
        "source_rows": int(receipt.get("day_source_rows", 0)),
        "valid_pool_observations": int(receipt.get("valid_pool_size", 0)),
        "eligible_date_count": 1 if receipt.get("status") == "PASS_SAMPLE" else 0,
        "tie_days": int(int(receipt.get("cutoff_tie_pool_count", 0)) > 1),
    }
    context = _review_context(
        contract=contract, binding=binding, hashes=hashes, code_path=code_path,
        test_path=test_path, baseline_path=baseline_path, receipt_path=receipt_path,
        metrics=metrics, terminal=True,
    )
    adapter.set_route_context(RUN_IDS["review_sample"], context)
    claim = core.ExecutionClaim(
        experiment_id=EXPERIMENT_ID,
        execution_id=RUN_IDS["review_sample"],
        arm="REVIEW_REAL_SAMPLE",
        contract_sha256=hashes["contract_sha256"],
        input_binding_sha256=hashes["input_binding_sha256"],
        execution_kind="NO_FIT_ENGINEERING_CANARY",
        real_fit_budget_debit=0,
    )
    status = "COMPLETED" if receipt["status"] == "PASS_SAMPLE" and all(receipt["quality_checks"].values()) else "FAILED_NO_FIT"
    error = None if status == "COMPLETED" else ScopeError("independent review sample quality gate failed")
    _terminalize_stage(adapter, claim, terminal_status=status, metrics=metrics, error=error)

    scoped = [event for event in adapter.list_events(EXPERIMENT_ID)]
    global_counts = _registry_global_counts(adapter)
    index_path = Path(binding["machine_index_path"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("experiment_id") != EXPERIMENT_ID or index.get("contract_sha256") != hashes["contract_sha256"]:
        raise ScopeError("machine index identity drift prevents review count update")
    index["registry_sha256"] = sha256_file(Path(binding["registry_path"]))
    index["registry_counts_global"] = global_counts
    index["registry_counts_scoped"] = {
        "route_id": ROUTE_ID,
        "routes": len({str(event.get("route_id")) for event in scoped}),
        "events": len(scoped),
        "executions": len({str(event.get("execution_id")) for event in scoped}),
        "corrections": sum(event.get("event_type") == "CORRECTION" for event in scoped),
        "open_execution_ids": sorted({str(event.get("execution_id")) for event in scoped if event.get("event_type") == "RUN_STARTED"} -
                                       {str(event.get("execution_id")) for event in scoped if event.get("event_type") == "TERMINAL"}),
    }
    index["registered_execution_ids"] = [*RUN_IDS.values()]
    index["review_sample_receipt_path"] = str(receipt_path)
    index["review_sample_receipt_sha256"] = sha256_file(receipt_path)
    index_payload = json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temp_path = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    with temp_path.open("x", encoding="utf-8") as handle:
        handle.write(index_payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, index_path)
    directory_fd = os.open(index_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {"execution_id": RUN_IDS["review_sample"], "terminal_status": status,
            "scoped_events": len(scoped), "registry_sha256": index["registry_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_frozen_args(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--contract", type=Path, required=True)
        command_parser.add_argument("--input-binding", type=Path, required=True)
        command_parser.add_argument("--config", type=Path, required=True)
        command_parser.add_argument("--baseline-record", type=Path, required=True)
        command_parser.add_argument("--expected-baseline", required=True)

    run = subparsers.add_parser("run-all")
    add_frozen_args(run)
    run.add_argument("--output-dir", type=Path, required=True)
    review_start = subparsers.add_parser("review-start")
    add_frozen_args(review_start)
    review_sample = subparsers.add_parser("review-sample")
    add_frozen_args(review_sample)
    review_terminal = subparsers.add_parser("review-terminal")
    add_frozen_args(review_terminal)
    review_terminal.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    code_path = Path(__file__).resolve()
    test_path = code_path.parent / "tests/test_top100_environment.py"
    common = {
        "contract_path": args.contract.resolve(),
        "binding_path": args.input_binding.resolve(),
        "config_path": args.config.resolve(),
        "code_path": code_path,
        "test_path": test_path,
        "baseline_path": args.baseline_record.resolve(),
        "expected_baseline": args.expected_baseline,
    }
    if args.command == "run-all":
        result = run_all(**common, output_dir=args.output_dir.resolve())
        output = {
            "status": "DIRECTION_ONLY",
            "eligible_dates": result["full_receipt"]["eligible_date_count"],
            "selection_slots": result["full_receipt"]["selection_slots"],
            "scoped_registry_counts": result["scoped_counts"],
            "global_registry_counts": result["global_counts"],
            "index_path": result["index"]["batch_path"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    elif args.command == "review-start":
        output = register_review_start(**common)
    elif args.command == "review-sample":
        output = run_review_sample(**common)
    else:
        output = terminalize_review_sample(**common, receipt_path=args.receipt.resolve())
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
