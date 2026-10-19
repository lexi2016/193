"""Repair the reviewed first-day market label from frozen aggregate artifacts.

This bounded correction reads only the seven R1 aggregate CSVs and the bound
CSI 300 series. It never reads the QFQ panel or recomputes returns, ranking, or
Top100 industry hits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import top100_environment as env


ROOT = Path("/Users/lexi/Desktop/学习/因子工厂")
TASK_WORKTREE = ROOT / "workspace/.task_worktrees/a-top100-environment-20261019"
ROUTE_DIR = TASK_WORKTREE / "research/a_share_top100_environment_v1"
BASE_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/top100_environment_r1_20261019"
REPAIR_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/market_boundary_repair_r1"
CONTRACT_PATH = REPAIR_BATCH / "CONTRACT_R2.json"
INPUT_BINDING_PATH = REPAIR_BATCH / "INPUT_BINDING_R2.json"
CONFIG_PATH = REPAIR_BATCH / "CONFIG_R2.json"
IDENTITY_PATH = REPAIR_BATCH / "FROZEN_IDENTITY_R2.json"
REPAIR_SOURCE = ROUTE_DIR / "market_boundary_repair.py"
MAIN_SOURCE = ROUTE_DIR / "top100_environment.py"
MAIN_TEST = ROUTE_DIR / "tests/test_top100_environment.py"
REPAIR_TEST = ROUTE_DIR / "tests/test_market_boundary_repair.py"
ORIGINAL_BINDING_PATH = BASE_BATCH / "INPUT_BINDING.json"
ORIGINAL_CONTRACT_PATH = BASE_BATCH / "CONTRACT.json"
ORIGINAL_CONFIG_PATH = BASE_BATCH / "CONFIG.json"
ORIGINAL_IDENTITY_PATH = BASE_BATCH / "FROZEN_IDENTITY.json"
BASE_REPORT_RECEIPT_PATH = BASE_BATCH / "outputs/REPORT_REPAIR_RECEIPT.json"
MACHINE_INDEX_PATH = ROOT / "contracts/A_SHARE_TOP100_ENVIRONMENT_INDEX_20261019.json"
BASELINE_PATH = BASE_BATCH / "BASELINE_AND_DEPENDENCIES.json"
OUTPUT_DIR = REPAIR_BATCH / "outputs_attempt_02"
DELTA_PATH = OUTPUT_DIR / "BOUNDARY_DELTA.json"
RECEIPT_PATH = OUTPUT_DIR / "MARKET_BOUNDARY_REPAIR_RECEIPT.json"
REPORT_PATH = OUTPUT_DIR / "COMPACT_REPORT.md"

EXPERIMENT_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019"
ROUTE_ID = "a-share-top100-environment-descriptive-stats-r1-20261019"
REPAIR_EXECUTION_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019_MARKET_BOUNDARY_REPAIR_02"
REVIEW_EXECUTION_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019_INCREMENTAL_REVIEW_02"
REQUIRED_TABLES = (
    "COVERAGE_BY_DAY.csv",
    "COVERAGE_BY_YEAR.csv",
    "COVERAGE_BY_INDUSTRY_DAY.csv",
    "COVERAGE_BY_INDUSTRY_YEAR.csv",
    "DAILY_ENVIRONMENT.csv",
    "INDUSTRY_CELLS.csv",
    "MONTH_MARKET_ENVIRONMENT.csv",
)
PRESERVED_TABLES = frozenset({
    "COVERAGE_BY_DAY.csv",
    "COVERAGE_BY_YEAR.csv",
    "COVERAGE_BY_INDUSTRY_DAY.csv",
    "COVERAGE_BY_INDUSTRY_YEAR.csv",
})
CHANGED_CELL_DIMENSIONS = frozenset({"industry_lag_market", "industry_month_lag_market"})
SAFETY_FLAGS = {
    "research_only": True,
    "production_allowed": False,
    "buy_gate_updated": False,
    "orders_sent": False,
}


class RepairError(RuntimeError):
    """A frozen repair input or lossless-repair invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RepairError(f"expected JSON object: {path}")
    return value


def write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _test_bundle_sha256(main_test_sha: str, repair_test_sha: str) -> str:
    payload = json.dumps(
        {str(MAIN_TEST): main_test_sha, str(REPAIR_TEST): repair_test_sha},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def frozen_identity() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    contract = read_json(CONTRACT_PATH)
    binding = read_json(INPUT_BINDING_PATH)
    config = read_json(CONFIG_PATH)
    identity = read_json(IDENTITY_PATH)
    main_test_sha = sha256_file(MAIN_TEST)
    repair_test_sha = sha256_file(REPAIR_TEST)
    hashes = {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "input_binding_sha256": sha256_file(INPUT_BINDING_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": sha256_file(REPAIR_SOURCE),
        "main_code_sha256": sha256_file(MAIN_SOURCE),
        "main_test_sha256": main_test_sha,
        "repair_test_sha256": repair_test_sha,
        "test_sha256": _test_bundle_sha256(main_test_sha, repair_test_sha),
    }
    expected = identity.get("frozen_identity", {})
    for key in ("config_sha256", "code_sha256", "main_code_sha256", "main_test_sha256",
                "repair_test_sha256", "test_sha256"):
        if expected.get(key) != hashes[key]:
            raise RepairError(f"frozen repair identity drift: {key}")
    if identity.get("contract_sha256") != hashes["contract_sha256"]:
        raise RepairError("frozen repair contract SHA drift")
    if identity.get("input_binding_sha256") != hashes["input_binding_sha256"]:
        raise RepairError("frozen repair input-binding SHA drift")
    if contract.get("input_binding_sha256") != hashes["input_binding_sha256"]:
        raise RepairError("repair contract input-binding SHA mismatch")
    if contract.get("frozen_identity") != expected:
        raise RepairError("repair contract and identity receipt disagree")
    if contract.get("status") != "FROZEN_DERIVED_ARTIFACT_REPAIR_ONLY":
        raise RepairError("repair contract is not frozen to derived-artifact repair")
    if binding.get("experiment_id") != EXPERIMENT_ID or binding.get("route_id") != ROUTE_ID:
        raise RepairError("repair input binding identity mismatch")
    if contract.get("input_binding_path") != str(INPUT_BINDING_PATH.resolve()):
        raise RepairError("repair contract input-binding path mismatch")
    if contract.get("config_path") != str(CONFIG_PATH.resolve()):
        raise RepairError("repair contract config path mismatch")
    if contract.get("frozen_identity_path") != str(IDENTITY_PATH.resolve()):
        raise RepairError("repair contract identity path mismatch")
    return contract, binding, config, hashes


def _record_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _cell_key(row: Mapping[str, Any], *, include_market: bool) -> tuple[Any, ...]:
    fields = ["period_scope", "year", "dimension_set", "industry_code", "month", "industry_name"]
    if include_market:
        fields.append("lag_market")
    normalized = []
    for field in fields:
        value = row.get(field)
        if pd.isna(value):
            normalized.append(None)
        elif field in {"year", "month"}:
            normalized.append(int(value))
        else:
            normalized.append(str(value))
    return tuple(normalized)


def _sort_cells(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_year_sort"] = ordered["year"].fillna(0).astype(int)
    ordered["_month_sort"] = ordered["month"].fillna(0).astype(int)
    ordered["_market_sort"] = ordered["lag_market"].fillna("").astype(str)
    return ordered.sort_values(
        ["period_scope", "dimension_set", "_year_sort", "industry_code", "_month_sort", "_market_sort"],
        kind="mergesort",
    ).drop(columns=["_year_sort", "_month_sort", "_market_sort"]).reset_index(drop=True)


def _support_years(value: Any) -> set[int]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {int(part) for part in str(value).split(",") if part.strip()}


def migrate_unknown_market_cells(
    cells: pd.DataFrame,
    *,
    affected_year: int,
    affected_month: int,
    valid_pool_size: int,
    eligible_days: int,
) -> pd.DataFrame:
    """Move the sole UNKNOWN-day cohort into BELOW without rebuilding hit counts."""
    required = {
        "period_scope", "year", "dimension_set", "industry_code", "industry_name", "month",
        "lag_market", "exposure_count", "hits_count", "expected_hits", "lift",
        "eligible_date_count", "support_year_count", "support_years",
    }
    if not required.issubset(cells.columns):
        raise RepairError(f"industry-cell schema is incomplete: {sorted(required - set(cells.columns))}")
    key_columns = ["period_scope", "year", "dimension_set", "industry_code", "industry_name", "month", "lag_market"]
    if cells.duplicated(key_columns, keep=False).any():
        raise RepairError("industry-cell aggregate keys are not unique")

    move_mask = cells["dimension_set"].isin(CHANGED_CELL_DIMENSIONS) & cells["lag_market"].eq("UNKNOWN")
    moving = cells.loc[move_mask].copy()
    if moving.empty:
        raise RepairError("no UNKNOWN industry-market cells exist for the reviewed boundary day")

    for dimension in CHANGED_CELL_DIMENSIONS:
        dimension_rows = moving.loc[moving["dimension_set"].eq(dimension)]
        if set(dimension_rows["period_scope"].astype(str)) != {"ALL", "YEAR"}:
            raise RepairError(f"UNKNOWN cohort lacks ALL/YEAR cells for {dimension}")
        for period_scope, group in dimension_rows.groupby("period_scope", sort=True):
            if not group["eligible_date_count"].eq(1).all():
                raise RepairError(f"UNKNOWN {dimension}/{period_scope} cannot be split losslessly")
            if not group["support_year_count"].eq(1).all() or not group["support_years"].astype(str).eq(str(affected_year)).all():
                raise RepairError(f"UNKNOWN {dimension}/{period_scope} support years do not match the affected date")
            if period_scope == "ALL" and group["year"].notna().any():
                raise RepairError(f"ALL-scope UNKNOWN {dimension} unexpectedly has a year key")
            if period_scope == "YEAR" and not group["year"].eq(affected_year).all():
                raise RepairError(f"YEAR-scope UNKNOWN {dimension} includes another year")
            if dimension == "industry_month_lag_market" and not group["month"].eq(affected_month).all():
                raise RepairError("UNKNOWN month-market cells include another month")
            if int(group["exposure_count"].sum()) != valid_pool_size:
                raise RepairError(f"UNKNOWN {dimension}/{period_scope} exposure does not equal the affected-day pool")
            if int(group["hits_count"].sum()) != 100 or not np.isclose(
                float(group["expected_hits"].sum()), 100.0, rtol=0, atol=1e-9
            ):
                raise RepairError(f"UNKNOWN {dimension}/{period_scope} hits or denominator do not equal one day")

    rows = cells.to_dict(orient="records")
    moving_rows = [row.copy() for row in rows if row["dimension_set"] in CHANGED_CELL_DIMENSIONS and row["lag_market"] == "UNKNOWN"]
    result_rows = [row.copy() for row in rows if not (row["dimension_set"] in CHANGED_CELL_DIMENSIONS and row["lag_market"] == "UNKNOWN")]
    targets: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(result_rows):
        if row["dimension_set"] in CHANGED_CELL_DIMENSIONS and row["lag_market"] == "BELOW":
            key = _cell_key(row, include_market=False)
            if key in targets:
                raise RepairError("multiple BELOW destination cells share one aggregate key")
            targets[key] = index

    for source in moving_rows:
        destination = source.copy()
        destination["lag_market"] = "BELOW"
        key = _cell_key(destination, include_market=False)
        target_index = targets.get(key)
        if target_index is None:
            result_rows.append(destination)
            targets[key] = len(result_rows) - 1
            continue
        target = result_rows[target_index]
        if _record_value(target.get("industry_name")) != _record_value(source.get("industry_name")):
            raise RepairError("industry-name mismatch prevents a lossless market-cell merge")
        for metric in ("exposure_count", "hits_count", "eligible_date_count"):
            target[metric] = int(target[metric]) + int(source[metric])
        target["expected_hits"] = float(target["expected_hits"]) + float(source["expected_hits"])
        years = _support_years(target["support_years"]) | _support_years(source["support_years"])
        target["support_years"] = ",".join(map(str, sorted(years)))
        target["support_year_count"] = len(years)
        target["lift"] = float(target["hits_count"]) / float(target["expected_hits"]) if target["expected_hits"] > 0 else np.nan

    repaired = pd.DataFrame(result_rows, columns=cells.columns)
    repaired = _sort_cells(repaired)
    if repaired["dimension_set"].isin(CHANGED_CELL_DIMENSIONS).where(
        repaired["lag_market"].eq("UNKNOWN"), False
    ).any():
        raise RepairError("UNKNOWN market cells remain after migration")

    def affected_row_mask(frame: pd.DataFrame) -> pd.Series:
        market_key = frame["lag_market"].isin({"UNKNOWN", "BELOW"})
        year_key = frame["period_scope"].eq("ALL") | frame["year"].eq(affected_year)
        lag_scope = frame["dimension_set"].eq("industry_lag_market") & year_key
        month_scope = (
            frame["dimension_set"].eq("industry_month_lag_market")
            & year_key
            & frame["month"].eq(affected_month)
        )
        return market_key & (lag_scope | month_scope)

    original_unaffected = _sort_cells(cells.loc[~affected_row_mask(cells)].copy())
    repaired_unaffected = _sort_cells(repaired.loc[~affected_row_mask(repaired)].copy())
    pd.testing.assert_frame_equal(original_unaffected, repaired_unaffected, check_dtype=True)
    for dimension in ("industry", "industry_month"):
        before = _sort_cells(cells.loc[cells["dimension_set"].eq(dimension)].copy())
        after = _sort_cells(repaired.loc[repaired["dimension_set"].eq(dimension)].copy())
        pd.testing.assert_frame_equal(before, after, check_dtype=True)
    env._validate_industry_parity(repaired, eligible_days=eligible_days)
    return repaired


def _summary_affected_mask(frame: pd.DataFrame, affected_year: int, affected_month: int) -> pd.Series:
    market = frame["lag_market"].isin({"UNKNOWN", "BELOW"})
    year_key = frame["period_scope"].eq("ALL") | frame["year"].eq(affected_year)
    lag_scope = frame["dimension"].eq("lag_market") & year_key
    month_scope = (
        frame["dimension"].eq("month_lag_market")
        & year_key
        & frame["month"].eq(affected_month)
    )
    return market & (lag_scope | month_scope)


def _validate_summary_delta(old: pd.DataFrame, new: pd.DataFrame, *, affected_year: int, affected_month: int) -> None:
    required = {"period_scope", "dimension", "year", "month", "lag_market", "eligible_date_count", "weighting"}
    if not required.issubset(old.columns) or list(old.columns) != list(new.columns):
        raise RepairError("monthly/market summary schema changed")
    old_unknown = old.loc[old["lag_market"].eq("UNKNOWN") & old["dimension"].isin({"lag_market", "month_lag_market"})]
    if len(old_unknown) != 4 or not old_unknown["eligible_date_count"].eq(1).all():
        raise RepairError("UNKNOWN summaries do not represent exactly the one reviewed day")
    if new["lag_market"].eq("UNKNOWN").any() and new["dimension"].isin({"lag_market", "month_lag_market"}).any():
        remaining = new.loc[new["lag_market"].eq("UNKNOWN") & new["dimension"].isin({"lag_market", "month_lag_market"})]
        if not remaining.empty:
            raise RepairError("UNKNOWN market summary remains after the one-day correction")
    key_columns = ["period_scope", "dimension", "year", "month", "lag_market"]
    old_unaffected = old.loc[~_summary_affected_mask(old, affected_year, affected_month)].sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    new_unaffected = new.loc[~_summary_affected_mask(new, affected_year, affected_month)].sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(old_unaffected, new_unaffected, check_dtype=True)


def _write_bytes_new(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace existing output: {path}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _make_context(
    *, contract: Mapping[str, Any], binding: Mapping[str, Any], hashes: Mapping[str, str],
    stage: str, terminal: bool,
) -> dict[str, Any]:
    table_paths = [str(Path(item["path"]).resolve()) for item in binding["table_inputs"].values()]
    index = binding["index_source"]
    input_paths = [*table_paths, str(Path(index["path"]).resolve()), str(Path(index["receipt_path"]).resolve()), str(Path(binding["source_input_binding_path"]).resolve())]
    artifacts = [
        str(CONTRACT_PATH.resolve()), str(INPUT_BINDING_PATH.resolve()), str(CONFIG_PATH.resolve()), str(IDENTITY_PATH.resolve()),
        str(REPAIR_SOURCE.resolve()), str(MAIN_SOURCE.resolve()), str(MAIN_TEST.resolve()), str(REPAIR_TEST.resolve()),
        str(ORIGINAL_CONTRACT_PATH.resolve()), str(ORIGINAL_BINDING_PATH.resolve()), str(ORIGINAL_CONFIG_PATH.resolve()),
        str(ORIGINAL_IDENTITY_PATH.resolve()), str(BASE_REPORT_RECEIPT_PATH.resolve()), str(BASELINE_PATH.resolve()),
        *[str(Path(path).resolve()) for path in binding.get("source_snapshot_paths", [])], *table_paths,
        str(Path(index["path"]).resolve()), str(Path(index["receipt_path"]).resolve()),
    ]
    null_reasons = {
        "deterministic_metrics.truth_positive_denominator": "NO_BINARY_FORWARD_TRUTH_LABEL_IN_DESCRIPTIVE_ROUTE",
        "deterministic_metrics.source_rows": "REPAIR_DOES_NOT_READ_OR_RECOMPUTE_PANEL_ROWS",
        "deterministic_metrics.valid_pool_observations": "REPAIR_COPIES_EXISTING_POOL_METRICS",
        "target_date_range": "NO_FORWARD_TARGET_IN_DESCRIPTIVE_ROUTE",
        "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_DESCRIPTIVE_ROUTE",
    }
    metrics = {
        "truth_positive_denominator": None,
        "selection_slots": 0,
        "source_rows": None,
        "valid_pool_observations": None,
        "eligible_date_count": 0,
        "execution_kind": "NO_FIT_ENGINEERING_CANARY",
        "planned_real_fit_count": 0,
        "real_fit_budget_debit": 0,
        "real_fit_returned_count": 0,
        "fit_call_entered": False,
        "input_aggregate_table_count": len(binding["table_inputs"]),
        "index_source_row_count": None,
        "derived_label_delta_count": 0,
    }
    for key, value in metrics.items():
        if value is None:
            null_reasons[f"deterministic_metrics.{key}"] = "NOT_AVAILABLE_AT_RUN_START" if not terminal else "NOT_RECOMPUTED_BY_THIS_REPAIR"
    return {
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "input_paths": list(dict.fromkeys(input_paths)),
        "artifact_paths": list(dict.fromkeys(artifacts)),
        "input_component_sha256": {
            **{f"aggregate:{name}": item["sha256"] for name, item in binding["table_inputs"].items()},
            "index_series": index["sha256"],
            "index_receipt": index["receipt_sha256"],
            "source_input_binding": binding["source_input_binding_sha256"],
            "model_selection_helper": binding["shared_dependencies"]["model_selection"]["sha256"],
            "registry_adapter": binding["shared_dependencies"]["registry_adapter"]["sha256"],
            "execution_guard_core": binding["shared_dependencies"]["execution_guard_core"]["sha256"],
            "registry_builder": binding["shared_dependencies"]["registry_builder"]["sha256"],
            "registry_validator": binding["shared_dependencies"]["registry_validator"]["sha256"],
        },
        "code_sha256_by_file": {
            str(REPAIR_SOURCE.resolve()): hashes["code_sha256"],
            str(MAIN_SOURCE.resolve()): hashes["main_code_sha256"],
            str(MAIN_TEST.resolve()): hashes["main_test_sha256"],
            str(REPAIR_TEST.resolve()): hashes["repair_test_sha256"],
        },
        "signal_date_range": {"count": 1, "min": contract["affected_date"], "max": contract["affected_date"], "scope": "SINGLE_REVIEWED_BOUNDARY_DATE"},
        "truth_access": "OPENED_EXPOSED_HISTORY" if terminal else "NOT_OPENED",
        "sealed_status": "EXPOSED" if terminal else "NOT_ENTERED",
        "runtime": {"python": sys.executable, "python_version": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "evidence_level": "DIRECTION_ONLY" if stage == "MARKET_BOUNDARY_REPAIR" else "ENGINEERING_CANARY_ONLY",
        "deterministic_metrics": metrics,
        "null_reasons": null_reasons,
    }


def _open_adapter(binding: Mapping[str, Any]):
    base_binding = read_json(Path(binding["source_input_binding_path"]))
    if sha256_file(Path(binding["source_input_binding_path"])) != binding["source_input_binding_sha256"]:
        raise RepairError("source input binding SHA drifted")
    adapter, core = env.load_official_registry(
        base_binding,
        contract_path=CONTRACT_PATH,
        input_binding_path=INPUT_BINDING_PATH,
        config_path=CONFIG_PATH,
        code_path=REPAIR_SOURCE,
        test_path=REPAIR_TEST,
    )
    return adapter, core, base_binding


def _register_start(adapter: Any, core: Any, *, execution_id: str, stage: str,
                    hashes: Mapping[str, str], contract: Mapping[str, Any], binding: Mapping[str, Any],
                    context_override: Mapping[str, Any] | None = None) -> Any:
    if any(event.get("execution_id") == execution_id for event in adapter.list_events(EXPERIMENT_ID)):
        raise RepairError(f"execution ID already exists; no retry is allowed: {execution_id}")
    context = dict(context_override) if context_override is not None else _make_context(
        contract=contract, binding=binding, hashes=hashes, stage=stage, terminal=False
    )
    return env._claim_stage(
        adapter, core, stage=stage, execution_id=execution_id,
        contract_sha256=hashes["contract_sha256"],
        input_binding_sha256=hashes["input_binding_sha256"], context=context,
    )


def _as_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _table_inputs(binding: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, bytes]]:
    tables: dict[str, pd.DataFrame] = {}
    preserved_bytes: dict[str, bytes] = {}
    for name in REQUIRED_TABLES:
        item = binding["table_inputs"].get(name)
        if not isinstance(item, Mapping):
            raise RepairError(f"missing frozen aggregate binding: {name}")
        payload = Path(item["path"]).read_bytes()
        if sha256_bytes(payload) != item["sha256"]:
            raise RepairError(f"source aggregate SHA drift: {name}")
        frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        tables[name] = frame
        if name in PRESERVED_TABLES:
            preserved_bytes[name] = payload
    return tables, preserved_bytes


def _classify_boundary(config: Mapping[str, Any], index_close: pd.Series) -> dict[str, Any]:
    target = _as_date(config["affected_date"])
    t_minus_1 = _as_date(config["t_minus_1_date"])
    if target <= t_minus_1:
        raise RepairError("boundary T date must follow the bound shared-calendar T-1 date")
    if index_close.index.has_duplicates:
        raise RepairError("bound index contains duplicate observation dates")
    prior = index_close.loc[index_close.index < t_minus_1].sort_index().tail(20)
    prior_values = prior.to_numpy(dtype=np.float64)
    if len(prior_values) != 20 or not np.isfinite(prior_values).all() or (prior_values <= 0).any():
        raise RepairError("the nearest 20 bound index rows before T-1 do not all have valid positive closes")
    close_t_minus_1 = float(index_close.get(t_minus_1, np.nan))
    if not np.isfinite(close_t_minus_1) or close_t_minus_1 <= 0:
        raise RepairError("bound index lacks a valid T-1 close")
    reference_mean = float(np.mean(prior_values, dtype=np.float64))
    expected_close = float(config["reviewed_t_minus_1_index_close"])
    expected_mean = float(config["reviewed_prior_20_mean"])
    if not np.isclose(close_t_minus_1, expected_close, rtol=0, atol=1e-8):
        raise RepairError("bound T-1 index close differs from independent review evidence")
    if not np.isclose(reference_mean, expected_mean, rtol=0, atol=1e-8):
        raise RepairError("bound prior-20 index mean differs from independent review evidence")
    minimal_calendar = pd.DataFrame({"date": [t_minus_1, target], "calendar_index": [0, 1]})
    label = env.market_regime_by_date(minimal_calendar, [target], index_close)[target]
    return {
        "affected_date": target.date().isoformat(),
        "t_minus_1_date": t_minus_1.date().isoformat(),
        "prior_observed_close_count": int(len(prior)),
        "prior_20_observed_close_mean": reference_mean,
        "t_minus_1_index_close": close_t_minus_1,
        "classification": label,
    }


def _repair_frames(tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], index_close: pd.Series):
    daily = tables["DAILY_ENVIRONMENT.csv"].copy()
    cells = tables["INDUSTRY_CELLS.csv"].copy()
    old_summary = tables["MONTH_MARKET_ENVIRONMENT.csv"].copy()
    target = _as_date(config["affected_date"])
    target_rows = daily.loc[daily["date"].eq(target)]
    if len(target_rows) != 1:
        raise RepairError("affected daily date is not unique in the frozen table")
    if str(target_rows["lag_market"].iloc[0]) != str(config["expected_old_label"]):
        raise RepairError("affected daily label differs from the frozen old value")
    unknown_daily = daily.loc[daily["lag_market"].eq("UNKNOWN")]
    if len(unknown_daily) != 1 or not unknown_daily["date"].eq(target).all():
        raise RepairError("UNKNOWN daily labels include dates outside the reviewed one-day cohort")
    if int(target_rows["top100_slots"].iloc[0]) != 100:
        raise RepairError("affected date does not preserve exactly 100 original slots")
    boundary = _classify_boundary(config, index_close)
    if boundary["classification"] != config["expected_new_label"]:
        raise RepairError("bound index did not produce the frozen expected boundary label")

    daily_new = daily.copy()
    daily_new.loc[daily_new["date"].eq(target), "lag_market"] = boundary["classification"]
    daily_unaffected = daily.loc[~daily["date"].eq(target)].reset_index(drop=True)
    daily_new_unaffected = daily_new.loc[~daily_new["date"].eq(target)].reset_index(drop=True)
    pd.testing.assert_frame_equal(daily_unaffected, daily_new_unaffected, check_dtype=True)
    daily_target_before = daily.loc[daily["date"].eq(target)].drop(columns=["lag_market"]).reset_index(drop=True)
    daily_target_after = daily_new.loc[daily_new["date"].eq(target)].drop(columns=["lag_market"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily_target_before, daily_target_after, check_dtype=True)

    cells_new = migrate_unknown_market_cells(
        cells,
        affected_year=target.year,
        affected_month=target.month,
        valid_pool_size=int(target_rows["valid_pool_size"].iloc[0]),
        eligible_days=len(daily),
    )
    summary_new = env._summarize_daily_metrics(daily_new)
    _validate_summary_delta(old_summary, summary_new, affected_year=target.year, affected_month=target.month)
    return {**tables, "DAILY_ENVIRONMENT.csv": daily_new, "INDUSTRY_CELLS.csv": cells_new,
            "MONTH_MARKET_ENVIRONMENT.csv": summary_new}, boundary


def _row_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _record_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _render_corrected_report(tables: Mapping[str, pd.DataFrame], original_binding: Mapping[str, Any],
                             original_contract: Mapping[str, Any], hashes: Mapping[str, str],
                             boundary: Mapping[str, Any], delta_name: str) -> str:
    daily = tables["DAILY_ENVIRONMENT.csv"]
    coverage_day = tables["COVERAGE_BY_DAY.csv"]
    coverage_year = tables["COVERAGE_BY_YEAR.csv"]
    cells = tables["INDUSTRY_CELLS.csv"]
    summaries = tables["MONTH_MARKET_ENVIRONMENT.csv"]
    stats = {
        "daily_environment": daily,
        "coverage_by_day": coverage_day,
        "coverage_by_year": coverage_year,
        "industry_cells": cells,
        "month_market_environment": summaries,
        "qa": {
            "selected_slot_count": int(daily["top100_slots"].sum()),
            "excluded_below_100_date_count": int((~coverage_day["eligible_for_top100"].astype(bool)).sum()),
            "reason_union_count": int(coverage_day["reason_union_rows"].sum()),
        },
    }
    panel = original_binding["sources"]["price_panel"]
    metadata = {
        "source_rows": int(panel["rows"]),
        "distinct_codes": int(panel["distinct_codes"]),
        "observed_dates": int(panel["dates"]["observed_date_count"]),
        "first_date": str(panel["dates"]["first"]),
        "last_date": str(panel["dates"]["last"]),
    }
    display_contract = {"outcome_label": "DESCRIPTIVE_ONLY_NO_TARGET_LABEL"}
    text = env._summary_markdown(stats, metadata, original_binding, display_contract, hashes)
    text += "\n## 市场边界修复\n\n"
    text += (
        f"- `{boundary['affected_date']}` 的市场状态由 `UNKNOWN` 更正为 `{boundary['classification']}`："
        f"T-1 `{boundary['t_minus_1_date']}` 指数收盘 {boundary['t_minus_1_index_close']:.4f}，"
        f"低于此前 {boundary['prior_observed_close_count']} 个有效指数收盘均值 {boundary['prior_20_observed_close_mean']:.5f}。\n"
        f"- 仅将既有行业×市场、行业×月份×市场汇总中的该日 cohort 并入对应 `BELOW` 组；"
        f"不重算收益、排名、Top100 命中或覆盖。逐表变更见 `{delta_name}`。\n"
        f"- 原 R1 完整统计终态仍为 `FAILED_NO_FIT`；原 R1 输出和报告均保留。本文件由独立边界修复批次生成。\n"
    )
    return text


def _build_delta(*, binding: Mapping[str, Any], original_tables: Mapping[str, pd.DataFrame],
                 corrected_tables: Mapping[str, pd.DataFrame], raw_inputs: Mapping[str, bytes],
                 boundary: Mapping[str, Any], target: pd.Timestamp, output_dir: Path) -> dict[str, Any]:
    table_hashes = {}
    for name in REQUIRED_TABLES:
        item = binding["table_inputs"][name]
        after = sha256_file(output_dir / name)
        table_hashes[name] = {
            "before_sha256": item["sha256"],
            "after_sha256": after,
            "byte_identical": item["sha256"] == after,
        }
    old_daily = original_tables["DAILY_ENVIRONMENT.csv"]
    new_daily = corrected_tables["DAILY_ENVIRONMENT.csv"]
    old_cells = original_tables["INDUSTRY_CELLS.csv"]
    new_cells = corrected_tables["INDUSTRY_CELLS.csv"]
    old_summary = original_tables["MONTH_MARKET_ENVIRONMENT.csv"]
    new_summary = corrected_tables["MONTH_MARKET_ENVIRONMENT.csv"]
    unknown_cells = old_cells.loc[old_cells["dimension_set"].isin(CHANGED_CELL_DIMENSIONS) & old_cells["lag_market"].eq("UNKNOWN")]
    cell_change_records = []
    for old in unknown_cells.to_dict(orient="records"):
        moved = old.copy()
        moved["lag_market"] = "BELOW"
        key = _cell_key(moved, include_market=True)
        matches = [row for row in new_cells.to_dict(orient="records") if _cell_key(row, include_market=True) == key]
        if len(matches) != 1:
            raise RepairError("cannot reconcile one migrated industry-market cell")
        before_below = old_cells.loc[
            old_cells.apply(lambda row: _cell_key(row, include_market=True) == key, axis=1)
        ]
        cell_change_records.append({
            "dimension_set": old["dimension_set"], "period_scope": old["period_scope"],
            "year": _record_value(old.get("year")), "month": _record_value(old.get("month")),
            "industry_code": old["industry_code"], "from_market": "UNKNOWN", "to_market": "BELOW",
            "unknown_before": _row_records(pd.DataFrame([old]))[0],
            "below_before": None if before_below.empty else _row_records(before_below)[0],
            "below_after": _row_records(pd.DataFrame(matches))[0],
        })
    old_summary_affected = old_summary.loc[_summary_affected_mask(old_summary, target.year, target.month)]
    new_summary_affected = new_summary.loc[_summary_affected_mask(new_summary, target.year, target.month)]
    daily_before = old_daily.loc[old_daily["date"].eq(target)]
    daily_after = new_daily.loc[new_daily["date"].eq(target)]
    return {
        "schema_version": "A_SHARE_TOP100_MARKET_BOUNDARY_DELTA_V1",
        "date": target.date().isoformat(),
        "market_label": {"before": "UNKNOWN", "after": boundary["classification"]},
        "index_evidence": dict(boundary),
        "daily_environment": {
            "changed_row_count": 1,
            "before": _row_records(daily_before)[0],
            "after": _row_records(daily_after)[0],
        },
        "industry_cells": {
            "migrated_unknown_cell_count": len(cell_change_records),
            "changed_cells": cell_change_records,
        },
        "month_market_environment": {
            "affected_group_row_count_before": len(old_summary_affected),
            "affected_group_row_count_after": len(new_summary_affected),
            "before": _row_records(old_summary_affected),
            "after": _row_records(new_summary_affected),
        },
        "tables": table_hashes,
        "preservation_checks": {
            name: table_hashes[name]["byte_identical"] for name in sorted(PRESERVED_TABLES)
        },
        "source_panel_read": False,
        "returns_recomputed": False,
        "ranking_recomputed": False,
        "industry_hits_recomputed": False,
    }


def _update_machine_index(*, receipt: Mapping[str, Any], hashes: Mapping[str, str],
                          stage: str, adapter: Any) -> dict[str, Any]:
    if not MACHINE_INDEX_PATH.is_file():
        raise RepairError("existing machine index is missing; refuse to create a parallel index")
    index = read_json(MACHINE_INDEX_PATH)
    if index.get("experiment_id") != EXPERIMENT_ID or index.get("route_id") != ROUTE_ID:
        raise RepairError("machine index identity drift")
    with adapter._locked():
        _, events = adapter._read_validated()
    scoped = [event for event in events if event.get("route_id") == ROUTE_ID]
    index["registry_sha256"] = sha256_file(Path(read_json(INPUT_BINDING_PATH)["registry_path"]))
    index["registry_counts_global"] = env._registry_global_counts(adapter)
    index["registry_counts_scoped"] = {
        "route_id": ROUTE_ID,
        "routes": len({str(event.get("route_id")) for event in scoped}),
        "events": len(scoped),
        "executions": len({str(event.get("execution_id")) for event in scoped}),
        "corrections": sum(event.get("event_type") == "CORRECTION" for event in scoped),
        "open_execution_ids": sorted(
            {str(event.get("execution_id")) for event in scoped if event.get("event_type") == "RUN_STARTED"}
            - {str(event.get("execution_id")) for event in scoped if event.get("event_type") == "TERMINAL"}
        ),
    }
    repair = index.setdefault("market_boundary_repair", {})
    repair.update({
        "execution_id": REPAIR_EXECUTION_ID,
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "receipt_path": str(RECEIPT_PATH.resolve()) if stage == "MARKET_BOUNDARY_REPAIR" else receipt.get("repair_receipt_path"),
        "receipt_sha256": sha256_file(RECEIPT_PATH) if stage == "MARKET_BOUNDARY_REPAIR" and RECEIPT_PATH.exists() else receipt.get("repair_receipt_sha256"),
        "status": "COMPLETED" if stage == "MARKET_BOUNDARY_REPAIR" else repair.get("status", "COMPLETED"),
    })
    if stage == "INCREMENTAL_REVIEW_02":
        repair["incremental_review_execution_id"] = REVIEW_EXECUTION_ID
        repair["incremental_review_status"] = receipt.get("review_verdict", "RUNNING")
        repair["incremental_review_record_path"] = receipt.get("review_record_path")
        repair["incremental_review_record_sha256"] = receipt.get("review_record_sha256")
    existing_ids = list(index.get("registered_execution_ids", []))
    for execution_id in (REPAIR_EXECUTION_ID, REVIEW_EXECUTION_ID):
        if execution_id not in existing_ids and (execution_id == REPAIR_EXECUTION_ID or stage == "INCREMENTAL_REVIEW_02"):
            existing_ids.append(execution_id)
    index["registered_execution_ids"] = existing_ids
    index["status"] = "CANNOT_CONFIRM"
    index["status_reason"] = "market boundary repair recorded; independent incremental review pending or failed"
    payload = (json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = MACHINE_INDEX_PATH.with_name(f".{MACHINE_INDEX_PATH.name}.{time.time_ns()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, MACHINE_INDEX_PATH)
    directory_fd = os.open(MACHINE_INDEX_PATH.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return index


def run_repair() -> dict[str, Any]:
    contract, binding, config, hashes = frozen_identity()
    if set(binding["table_inputs"]) != set(REQUIRED_TABLES):
        raise RepairError("repair input binding must contain exactly the seven R1 aggregate tables")
    if OUTPUT_DIR.exists():
        raise RepairError("repair outputs already exist; no retry or overwrite is allowed")
    adapter, core, original_binding = _open_adapter(binding)
    claim = _register_start(
        adapter, core, execution_id=REPAIR_EXECUTION_ID, stage="MARKET_BOUNDARY_REPAIR",
        hashes=hashes, contract=contract, binding=binding,
    )
    started = time.monotonic()
    stage_dir = REPAIR_BATCH / f".outputs_staging_{REPAIR_EXECUTION_ID}"
    if stage_dir.exists():
        exc = RepairError("repair staging directory already exists; refusing reuse")
        env._terminalize_stage(adapter, claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise exc
    stage_dir.mkdir(parents=True)
    try:
        tables, preserved_bytes = _table_inputs(binding)
        helper = env.load_bound_helpers(original_binding)
        index_source = binding["index_source"]
        index_close = helper.load_bound_index_series(index_source, panel_dates=set())
        original_tables = dict(tables)
        repaired_tables, boundary = _repair_frames(tables, config, index_close)
        repaired_tables.update({
            name: original_tables[name] for name in PRESERVED_TABLES
        })
        for name, payload in preserved_bytes.items():
            _write_bytes_new(stage_dir / name, payload)
        for name in ("DAILY_ENVIRONMENT.csv", "INDUSTRY_CELLS.csv", "MONTH_MARKET_ENVIRONMENT.csv"):
            env.write_csv(stage_dir / name, repaired_tables[name])

        delta = _build_delta(
            binding=binding, original_tables=original_tables, corrected_tables=repaired_tables,
            raw_inputs=preserved_bytes, boundary=boundary,
            target=_as_date(config["affected_date"]), output_dir=stage_dir,
        )
        write_new_json(stage_dir / "BOUNDARY_DELTA.json", delta)
        report_text = _render_corrected_report(
            repaired_tables, original_binding, read_json(ORIGINAL_CONTRACT_PATH), hashes,
            boundary, "BOUNDARY_DELTA.json",
        )
        _write_bytes_new(stage_dir / "COMPACT_REPORT.md", report_text.encode("utf-8"))
        output_hashes = {path.name: sha256_file(path) for path in sorted(stage_dir.iterdir()) if path.is_file()}
        output_bytes = {path.name: path.stat().st_size for path in sorted(stage_dir.iterdir()) if path.is_file()}
        receipt = {
            "schema_version": "A_SHARE_TOP100_MARKET_BOUNDARY_REPAIR_RECEIPT_V1",
            "experiment_id": EXPERIMENT_ID,
            "route_id": ROUTE_ID,
            "execution_id": REPAIR_EXECUTION_ID,
            "status": "REPAIRED_FROM_EXISTING_AGGREGATES",
            "contract_sha256": hashes["contract_sha256"],
            "input_binding_sha256": hashes["input_binding_sha256"],
            "config_sha256": hashes["config_sha256"],
            "code_sha256": hashes["code_sha256"],
            "main_code_sha256": hashes["main_code_sha256"],
            "test_sha256": hashes["test_sha256"],
            "input_table_sha256": {name: item["sha256"] for name, item in binding["table_inputs"].items()},
            "input_index_sha256": index_source["sha256"],
            "input_index_receipt_sha256": index_source["receipt_sha256"],
            "boundary": boundary,
            "delta_path": str((OUTPUT_DIR / "BOUNDARY_DELTA.json").resolve()),
            "delta_sha256": output_hashes["BOUNDARY_DELTA.json"],
            "output_sha256": output_hashes,
            "output_bytes": output_bytes,
            "validation": {
                "source_panel_read": False,
                "returns_recomputed": False,
                "ranking_recomputed": False,
                "industry_hits_recomputed": False,
                "daily_market_label_delta_count": delta["daily_environment"]["changed_row_count"],
                "unknown_industry_cells_migrated": delta["industry_cells"]["migrated_unknown_cell_count"],
                "industry_and_industry_month_values_unchanged": True,
                "coverage_tables_byte_identical": all(delta["preservation_checks"].values()),
                "summary_rebuilt_from_daily_aggregate": True,
                "only_affected_market_groups_changed": True,
                "annual_industry_parity": env._validate_industry_parity(repaired_tables["INDUSTRY_CELLS.csv"], len(repaired_tables["DAILY_ENVIRONMENT.csv"])),
                "safety_flags": SAFETY_FLAGS,
            },
            "preserved_prior_failures": {
                "full_stats_execution_id": "A_SHARE_TOP100_ENVIRONMENT_R1_20261019_FULL_STATS_01",
                "full_stats_terminal": "FAILED_NO_FIT",
                "r1_review_execution_id": "A_SHARE_TOP100_ENVIRONMENT_R1_20261019_REVIEW_SAMPLE_01",
                "r1_review_terminal": "FAILED_NO_FIT",
            },
            "runtime_seconds": round(time.monotonic() - started, 6),
        }
        receipt_payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _write_bytes_new(stage_dir / "MARKET_BOUNDARY_REPAIR_RECEIPT.json", receipt_payload)
        if OUTPUT_DIR.exists():
            raise RepairError("new repair output directory appeared; refusing to overwrite")
        os.rename(stage_dir, OUTPUT_DIR)
        context = dict(adapter._route_context[REPAIR_EXECUTION_ID])
        context["artifact_paths"] = list(dict.fromkeys([
            *context["artifact_paths"], *[str((OUTPUT_DIR / name).resolve()) for name in output_hashes],
            str(RECEIPT_PATH.resolve()), str(DELTA_PATH.resolve()), str(REPORT_PATH.resolve()),
        ]))
        context["deterministic_metrics"].update({
            "derived_label_delta_count": 1,
            "migrated_industry_market_cell_count": delta["industry_cells"]["migrated_unknown_cell_count"],
            "repair_output_table_count": len(REQUIRED_TABLES),
            "selection_slots": 0,
            "eligible_date_count": 0,
        })
        adapter.set_route_context(REPAIR_EXECUTION_ID, context)
        env._terminalize_stage(adapter, claim, terminal_status="COMPLETED", metrics={
            "selection_slots": 0,
            "derived_label_delta_count": 1,
            "migrated_industry_market_cell_count": delta["industry_cells"]["migrated_unknown_cell_count"],
            "repair_output_table_count": len(REQUIRED_TABLES),
            "repair_report_bytes": output_bytes["COMPACT_REPORT.md"],
        })
        updated_index = _update_machine_index(receipt={**receipt, "repair_receipt_path": str(RECEIPT_PATH),
                                                       "repair_receipt_sha256": sha256_file(RECEIPT_PATH)},
                                              hashes=hashes, stage="MARKET_BOUNDARY_REPAIR", adapter=adapter)
        return {"receipt": receipt, "receipt_path": str(RECEIPT_PATH),
                "receipt_sha256": sha256_file(RECEIPT_PATH), "delta_path": str(DELTA_PATH),
                "delta_sha256": sha256_file(DELTA_PATH), "machine_index_sha256": sha256_file(MACHINE_INDEX_PATH),
                "registry_sha256": sha256_file(Path(binding["registry_path"])),
                "registry_counts_scoped": updated_index["registry_counts_scoped"]}
    except BaseException as exc:
        current_events = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == REPAIR_EXECUTION_ID]
        if len(current_events) == 1 and current_events[0].get("event_type") == "RUN_STARTED":
            env._terminalize_stage(adapter, claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise


def register_incremental_review_start() -> dict[str, Any]:
    contract, binding, _, hashes = frozen_identity()
    if not RECEIPT_PATH.is_file():
        raise RepairError("boundary repair receipt is missing; cannot start incremental review")
    receipt = read_json(RECEIPT_PATH)
    if receipt.get("execution_id") != REPAIR_EXECUTION_ID or receipt.get("status") != "REPAIRED_FROM_EXISTING_AGGREGATES":
        raise RepairError("repair receipt identity or status mismatch")
    adapter, core, _ = _open_adapter(binding)
    context = _make_context(contract=contract, binding=binding, hashes=hashes,
                            stage="INCREMENTAL_REVIEW_02", terminal=False)
    context["artifact_paths"] = list(dict.fromkeys([
        *context["artifact_paths"], str(RECEIPT_PATH.resolve()), str(DELTA_PATH.resolve()),
        str(REPORT_PATH.resolve()), *[str((OUTPUT_DIR / name).resolve()) for name in REQUIRED_TABLES],
    ]))
    context["input_paths"] = list(dict.fromkeys([
        *context["input_paths"], *[str((OUTPUT_DIR / name).resolve()) for name in REQUIRED_TABLES],
        str(RECEIPT_PATH.resolve()), str(DELTA_PATH.resolve()),
    ]))
    context["input_component_sha256"].update({
        f"repaired:{name}": receipt["output_sha256"][name] for name in REQUIRED_TABLES
    })
    context["input_component_sha256"]["repair_receipt"] = sha256_file(RECEIPT_PATH)
    claim = _register_start(
        adapter, core, execution_id=REVIEW_EXECUTION_ID, stage="INCREMENTAL_REVIEW_02",
        hashes=hashes, contract=contract, binding=binding, context_override=context,
    )
    index = _update_machine_index(receipt={"repair_receipt_path": str(RECEIPT_PATH),
                                          "repair_receipt_sha256": sha256_file(RECEIPT_PATH)},
                                  hashes=hashes, stage="INCREMENTAL_REVIEW_02", adapter=adapter)
    review_paths = [str(path.resolve()) for path in (
        REPAIR_SOURCE, MAIN_SOURCE, MAIN_TEST, REPAIR_TEST, DELTA_PATH, RECEIPT_PATH,
        REPORT_PATH, *(OUTPUT_DIR / name for name in REQUIRED_TABLES),
    )]
    return {"execution_id": claim.execution_id, "event": "RUN_STARTED", "status": "REGISTERED",
            "review_scope": "market-boundary code and directly affected aggregate tables only",
            "review_input_paths": review_paths,
            "machine_index_sha256": sha256_file(MACHINE_INDEX_PATH),
            "registry_sha256": sha256_file(Path(binding["registry_path"])),
            "registry_counts_scoped": index["registry_counts_scoped"]}


def terminalize_incremental_review(receipt_path: Path) -> dict[str, Any]:
    contract, binding, _, hashes = frozen_identity()
    receipt_path = Path(receipt_path).resolve()
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != "A_SHARE_TOP100_INCREMENTAL_REVIEW_HANDOFF_V1":
        raise RepairError("incremental review handoff schema mismatch")
    if receipt.get("execution_id") != REVIEW_EXECUTION_ID:
        raise RepairError("incremental review handoff execution identity mismatch")
    for field in ("contract_sha256", "input_binding_sha256", "config_sha256", "code_sha256", "test_sha256"):
        if receipt.get(field) != hashes[field]:
            raise RepairError(f"incremental review handoff {field} mismatch")
    verdict = receipt.get("review_verdict")
    checks = receipt.get("checks")
    if verdict not in {"PASS", "FAIL"} or not isinstance(checks, dict) or not checks:
        raise RepairError("incremental review handoff lacks a terminal verdict or checks")
    adapter, core, _ = _open_adapter(binding)
    starts = [event for event in adapter.list_events(EXPERIMENT_ID)
              if event.get("execution_id") == REVIEW_EXECUTION_ID]
    if len(starts) != 1 or starts[0].get("event_type") != "RUN_STARTED":
        raise RepairError("incremental review terminal requires exactly one unmatched RUN_STARTED")
    metrics = {
        "selection_slots": 0,
        "review_checks_passed": sum(value is True for value in checks.values()),
        "review_checks_total": len(checks),
    }
    context = _make_context(contract=contract, binding=binding, hashes=hashes,
                            stage="INCREMENTAL_REVIEW_02", terminal=True)
    context["artifact_paths"] = list(dict.fromkeys([
        *context["artifact_paths"], str(RECEIPT_PATH.resolve()), str(DELTA_PATH.resolve()),
        str(REPORT_PATH.resolve()), str(receipt_path),
    ]))
    context["input_paths"] = list(dict.fromkeys([
        *context["input_paths"], *[str((OUTPUT_DIR / name).resolve()) for name in REQUIRED_TABLES],
    ]))
    adapter.set_route_context(REVIEW_EXECUTION_ID, context)
    claim = core.ExecutionClaim(
        experiment_id=EXPERIMENT_ID, execution_id=REVIEW_EXECUTION_ID,
        arm="INCREMENTAL_REVIEW_02", contract_sha256=hashes["contract_sha256"],
        input_binding_sha256=hashes["input_binding_sha256"],
        execution_kind="NO_FIT_ENGINEERING_CANARY", real_fit_budget_debit=0,
    )
    passed = verdict == "PASS" and all(value is True for value in checks.values())
    status = "COMPLETED" if passed else "FAILED_NO_FIT"
    error = None if passed else RepairError("independent incremental boundary review failed")
    env._terminalize_stage(adapter, claim, terminal_status=status, metrics=metrics, error=error)
    index = _update_machine_index(receipt={"review_verdict": verdict,
                                          "review_record_path": str(receipt_path),
                                          "review_record_sha256": sha256_file(receipt_path)},
                                  hashes=hashes, stage="INCREMENTAL_REVIEW_02", adapter=adapter)
    return {"execution_id": REVIEW_EXECUTION_ID, "terminal_status": status,
            "registry_sha256": sha256_file(Path(binding["registry_path"])),
            "registry_counts_scoped": index["registry_counts_scoped"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("review-start")
    terminal = sub.add_parser("review-terminal")
    terminal.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_repair()
    elif args.command == "review-start":
        result = register_incremental_review_start()
    else:
        result = terminalize_incremental_review(args.receipt)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
