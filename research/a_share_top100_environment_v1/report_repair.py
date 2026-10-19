"""Render the frozen descriptive report from already-produced aggregate tables.

This repair path never reads the shared market panel or recomputes Top100 rows.
It validates the seven existing statistics tables, renders the missing Markdown
report, and records its own zero-fit registry lifecycle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path("/Users/lexi/Desktop/学习/因子工厂")
REPO = ROOT / "workspace"
TASK_WORKTREE = REPO / ".task_worktrees/a-top100-environment-20261019"
BATCH = ROOT / "research/a_share_top100_environment_v1/batches/top100_environment_r1_20261019"
OUTPUTS = BATCH / "outputs"
REPAIR_CONTRACT = BATCH / "REPORT_REPAIR_CONTRACT_R3.json"
REPAIR_BINDING = BATCH / "REPORT_REPAIR_INPUT_BINDING_R2.json"
REPAIR_CONFIG = BATCH / "REPORT_REPAIR_CONFIG.json"
REPAIR_IDENTITY = BATCH / "REPORT_REPAIR_FROZEN_IDENTITY_R3.json"
REPAIR_SOURCE = TASK_WORKTREE / "research/a_share_top100_environment_v1/report_repair.py"
MAIN_SOURCE = TASK_WORKTREE / "research/a_share_top100_environment_v1/top100_environment.py"
TEST_SOURCE = TASK_WORKTREE / "research/a_share_top100_environment_v1/tests/test_top100_environment.py"
ORIGINAL_CONTRACT = BATCH / "CONTRACT.json"
ORIGINAL_BINDING = BATCH / "INPUT_BINDING.json"
ORIGINAL_CONFIG = BATCH / "CONFIG.json"
ORIGINAL_IDENTITY = BATCH / "FROZEN_IDENTITY.json"
REPORT_PATH = OUTPUTS / "COMPACT_REPORT.md"
RECEIPT_PATH = OUTPUTS / "REPORT_REPAIR_RECEIPT.json"
EXECUTION_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019_REPORT_REPAIR_02"
EXPERIMENT_ID = "A_SHARE_TOP100_ENVIRONMENT_R1_20261019"
REQUIRED_TABLES = (
    "COVERAGE_BY_DAY.csv",
    "COVERAGE_BY_YEAR.csv",
    "COVERAGE_BY_INDUSTRY_DAY.csv",
    "COVERAGE_BY_INDUSTRY_YEAR.csv",
    "DAILY_ENVIRONMENT.csv",
    "INDUSTRY_CELLS.csv",
    "MONTH_MARKET_ENVIRONMENT.csv",
)
SAFETY_FLAGS = {
    "research_only": True,
    "production_allowed": False,
    "buy_gate_updated": False,
    "orders_sent": False,
}


class RepairError(RuntimeError):
    """A frozen repair input or aggregate-output invariant failed."""


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


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_new_or_identical(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace existing output: {path}")
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_repair_identity() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    contract = read_json(REPAIR_CONTRACT)
    binding = read_json(REPAIR_BINDING)
    config = read_json(REPAIR_CONFIG)
    identity = read_json(REPAIR_IDENTITY)
    hashes = {
        "contract_sha256": sha256_file(REPAIR_CONTRACT),
        "input_binding_sha256": sha256_file(REPAIR_BINDING),
        "config_sha256": sha256_file(REPAIR_CONFIG),
        "code_sha256": sha256_file(REPAIR_SOURCE),
        "main_code_sha256": sha256_file(MAIN_SOURCE),
        "test_sha256": sha256_file(TEST_SOURCE),
    }
    expected = identity.get("frozen_identity", {})
    for key in ("contract_sha256", "input_binding_sha256", "config_sha256", "code_sha256"):
        if expected.get(key) != hashes[key]:
            raise RepairError(f"report repair frozen identity drift: {key}")
    original_identity = read_json(ORIGINAL_IDENTITY)
    original_hashes = {
        "contract_sha256": sha256_file(ORIGINAL_CONTRACT),
        "input_binding_sha256": sha256_file(ORIGINAL_BINDING),
        "config_sha256": sha256_file(ORIGINAL_CONFIG),
        "code_sha256": hashes["main_code_sha256"],
        "test_sha256": hashes["test_sha256"],
    }
    original_expected = original_identity.get("frozen_identity", original_identity)
    for key, value in original_hashes.items():
        if original_expected.get(key) != value:
            raise RepairError(f"original statistics identity drift: {key}")
    if contract.get("original_full_stats_status") != "FAILED_NO_FIT":
        raise RepairError("report repair must preserve the original failed full-stats terminal")
    return {"contract": contract, "config": config}, binding, hashes


def make_context(*, contract: Mapping[str, Any], binding: Mapping[str, Any], hashes: Mapping[str, str],
                 runtime: Mapping[str, Any]) -> dict[str, Any]:
    inputs = binding["table_inputs"]
    source_paths = [str(Path(item["path"]).resolve()) for item in inputs.values()]
    artifacts = [
        str(REPAIR_CONTRACT.resolve()), str(REPAIR_BINDING.resolve()), str(REPAIR_CONFIG.resolve()),
        str(REPAIR_IDENTITY.resolve()), str(REPAIR_SOURCE.resolve()), str(MAIN_SOURCE.resolve()),
        str(TEST_SOURCE.resolve()), str(ORIGINAL_CONTRACT.resolve()), str(ORIGINAL_BINDING.resolve()),
        str(ORIGINAL_CONFIG.resolve()), str(ORIGINAL_IDENTITY.resolve()),
        str(Path(binding["baseline_record_path"]).resolve()),
        *source_paths,
    ]
    return {
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "input_paths": source_paths,
        "artifact_paths": artifacts,
        "input_component_sha256": {name: item["sha256"] for name, item in inputs.items()},
        "code_sha256_by_file": {
            str(REPAIR_SOURCE.resolve()): hashes["code_sha256"],
            str(MAIN_SOURCE.resolve()): hashes["main_code_sha256"],
            str(TEST_SOURCE.resolve()): hashes["test_sha256"],
        },
        "signal_date_range": contract["parent_signal_date_range"],
        "truth_access": "NOT_OPENED",
        "sealed_status": "NOT_ENTERED",
        "runtime": dict(runtime),
        "evidence_level": "DIRECTION_ONLY",
        "deterministic_metrics": {
            "truth_positive_denominator": None,
            "selection_slots": 0,
            "source_rows": None,
            "valid_pool_observations": None,
            "eligible_date_count": None,
            "execution_kind": "NO_FIT_ENGINEERING_CANARY",
            "planned_real_fit_count": 0,
            "real_fit_budget_debit": 0,
            "real_fit_returned_count": 0,
            "fit_call_entered": False,
            "repair_input_table_count": len(inputs),
        },
        "null_reasons": {
            "deterministic_metrics.truth_positive_denominator": "DESCRIPTIVE_REPORT_HAS_NO_BINARY_FORWARD_TRUTH_LABEL",
            "deterministic_metrics.source_rows": "REPORT_REPAIR_READS_AGGREGATE_TABLES_ONLY",
            "deterministic_metrics.valid_pool_observations": "REPORT_REPAIR_DOES_NOT_RECOMPUTE_POOL",
            "deterministic_metrics.eligible_date_count": "NOT_READ_BEFORE_REPORT_REPAIR_RUN_START",
            "target_date_range": "NO_FORWARD_TARGET_IN_DESCRIPTIVE_ROUTE",
            "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_DESCRIPTIVE_ROUTE",
            "signal_date_range": "FROZEN_FROM_PARENT_CONTRACT_METADATA",
        },
    }


def validate_and_read_tables(binding: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, item in binding["table_inputs"].items():
        path = Path(item["path"])
        if not path.is_file():
            raise RepairError(f"missing frozen aggregate table: {name}")
        if sha256_file(path) != item["sha256"]:
            raise RepairError(f"aggregate table SHA drift: {name}")
        table = pd.read_csv(path)
        if "date" in table.columns:
            table["date"] = pd.to_datetime(table["date"], errors="raise")
        tables[name] = table

    required = {
        "COVERAGE_BY_DAY.csv": {"date", "year", "source_rows", "valid_pool_observations", "excluded_from_pool_rows", "reason_union_rows", "eligible_for_top100"},
        "COVERAGE_BY_YEAR.csv": {"year", "source_dates", "eligible_dates", "source_rows", "valid_pool_observations", "excluded_from_pool_rows", "reason_union_rows"},
        "COVERAGE_BY_INDUSTRY_DAY.csv": {"date", "industry_code", "source_rows", "valid_pool_observations", "excluded_from_pool_rows", "reason_union_rows"},
        "COVERAGE_BY_INDUSTRY_YEAR.csv": {"year", "industry_code", "source_rows", "valid_pool_observations", "excluded_from_pool_rows", "reason_union_rows"},
        "DAILY_ENVIRONMENT.csv": {"date", "valid_pool_size", "top100_slots", "top100_cutoff_return", "top100_mean_return", "top100_median_return", "top100_p90_return", "top100_max_return", "pool_positive_share", "top100_industry_hhi", "top100_industry_max_share", "cutoff_tie_pool_count", "cutoff_tie_excluded_count"},
        "INDUSTRY_CELLS.csv": {"period_scope", "dimension_set", "industry_code", "hits_count", "expected_hits", "exposure_count", "eligible_date_count", "support_year_count", "lift"},
        "MONTH_MARKET_ENVIRONMENT.csv": {"period_scope", "dimension", "eligible_date_count", "mean_top100_mean_return", "mean_top100_median_return", "mean_top100_p90_return", "mean_pool_positive_share", "mean_top100_industry_hhi", "mean_top100_industry_max_share"},
    }
    for name, columns in required.items():
        frame = tables[name]
        missing = sorted(columns.difference(frame.columns))
        if missing:
            raise RepairError(f"aggregate table schema mismatch {name}: {missing}")

    by_day = tables["COVERAGE_BY_DAY.csv"]
    by_year = tables["COVERAGE_BY_YEAR.csv"]
    daily = tables["DAILY_ENVIRONMENT.csv"]
    industry_day = tables["COVERAGE_BY_INDUSTRY_DAY.csv"]
    industry_year = tables["COVERAGE_BY_INDUSTRY_YEAR.csv"]
    cells = tables["INDUSTRY_CELLS.csv"]
    if by_day["date"].duplicated().any() or by_year["year"].duplicated().any() or daily["date"].duplicated().any():
        raise RepairError("daily or annual aggregate keys are not unique")
    if industry_day.duplicated(["date", "industry_code"]).any() or industry_year.duplicated(["year", "industry_code"]).any():
        raise RepairError("industry coverage aggregate keys are not unique")
    if not daily["top100_slots"].eq(100).all():
        raise RepairError("an eligible date does not contain exactly 100 slots")
    if int(daily["top100_slots"].sum()) != 100 * len(daily):
        raise RepairError("daily selected-slot count does not reconcile")
    if int(by_day["eligible_for_top100"].astype(bool).sum()) != len(daily):
        raise RepairError("coverage and daily report date counts disagree")
    if set(daily["date"]) != set(by_day.loc[by_day["eligible_for_top100"].astype(bool), "date"]):
        raise RepairError("daily report dates differ from eligible coverage dates")
    if int(by_day["source_rows"].sum()) != int(binding["source_panel_rows"]):
        raise RepairError("coverage source rows do not match frozen panel metadata")
    if int(by_year["source_rows"].sum()) != int(binding["source_panel_rows"]):
        raise RepairError("annual source rows do not match frozen panel metadata")
    observed_dates = int(by_day["source_dates"].sum()) if "source_dates" in by_day.columns else len(by_day)
    if observed_dates != int(binding["source_observed_dates"]):
        raise RepairError("daily coverage dates do not match frozen panel metadata")
    if int(by_year["source_dates"].sum()) != len(by_day):
        raise RepairError("annual dates do not reconcile to daily coverage")
    if not by_day["reason_union_rows"].eq(by_day["excluded_from_pool_rows"]).all():
        raise RepairError("daily reason union does not reconcile to excluded rows")
    if not by_year["reason_union_rows"].eq(by_year["excluded_from_pool_rows"]).all():
        raise RepairError("annual reason union does not reconcile to excluded rows")
    if int(industry_day["source_rows"].sum()) != int(binding["source_panel_rows"]):
        raise RepairError("industry-by-day source rows do not reconcile")
    if int(industry_year["source_rows"].sum()) != int(binding["source_panel_rows"]):
        raise RepairError("industry-by-year source rows do not reconcile")
    for scope in ("industry", "industry_month", "industry_lag_market", "industry_month_lag_market"):
        all_scope = cells.loc[cells["period_scope"].eq("ALL") & cells["dimension_set"].eq(scope)]
        if abs(float(all_scope["expected_hits"].sum()) - 100.0 * len(daily)) > 1e-6:
            raise RepairError(f"all-scope industry expected hits do not reconcile: {scope}")
        if int(all_scope["hits_count"].sum()) != 100 * len(daily):
            raise RepairError(f"all-scope industry hits do not reconcile: {scope}")
    for year, group in daily.groupby(daily["date"].dt.year):
        if (group["top100_slots"] != 100).any() or group["date"].nunique() != len(group):
            raise RepairError(f"annual daily output failed slot/date checks: {year}")
    return tables


class ReportContractView(dict):
    """Expose display-only metadata without mutating the executed contract."""

    def __init__(self, original: Mapping[str, Any], outcome_label: str):
        super().__init__(original)
        self._display_outcome_label = outcome_label

    def __getitem__(self, key: str):
        if key == "outcome_label" and key not in self:
            return self._display_outcome_label
        return super().__getitem__(key)


def render_report(tables: Mapping[str, pd.DataFrame], binding: Mapping[str, Any],
                  repair_contract: Mapping[str, Any], repair_config: Mapping[str, Any],
                  main_module: Any) -> tuple[str, dict[str, Any]]:
    by_day = tables["COVERAGE_BY_DAY.csv"]
    by_year = tables["COVERAGE_BY_YEAR.csv"]
    daily = tables["DAILY_ENVIRONMENT.csv"]
    cells = tables["INDUSTRY_CELLS.csv"]
    summary = tables["MONTH_MARKET_ENVIRONMENT.csv"]
    stats = {
        "daily_environment": daily,
        "coverage_by_day": by_day,
        "coverage_by_year": by_year,
        "industry_cells": cells,
        "month_market_environment": summary,
        "qa": {
            "eligible_date_count": int(len(daily)),
            "excluded_below_100_date_count": int((~by_day["eligible_for_top100"].astype(bool)).sum()),
            "selected_slot_count": int(daily["top100_slots"].sum()),
            "reason_union_count": int(by_day["reason_union_rows"].sum()),
            "reason_counts_overlap": True,
        },
    }
    metadata = {
        "source_rows": int(by_day["source_rows"].sum()),
        "distinct_codes": int(binding["source_distinct_codes"]),
        "observed_dates": int(len(by_day)),
        "first_date": by_day["date"].min().date().isoformat(),
        "last_date": by_day["date"].max().date().isoformat(),
    }
    original_contract = read_json(ORIGINAL_CONTRACT)
    original_binding = read_json(ORIGINAL_BINDING)
    identity_document = read_json(ORIGINAL_IDENTITY)
    original_identity = identity_document.get("frozen_identity", identity_document)
    hashes = {
        "contract_sha256": original_identity["contract_sha256"],
        "input_binding_sha256": original_identity["input_binding_sha256"],
        "config_sha256": original_identity["config_sha256"],
        "code_sha256": original_identity["code_sha256"],
    }
    contract_view = ReportContractView(original_contract, str(repair_config["display_only_outcome_label"]))
    text = main_module._summary_markdown(stats, metadata, original_binding, contract_view, hashes)
    text += "\n## 报告修复记录\n\n"
    text += (
        f"- 本报告由 `{EXECUTION_ID}` 从上述 7 张已存在统计表生成；完整统计执行 "
        f"`{repair_contract['parent_full_stats_execution_id']}` 保留原 terminal `FAILED_NO_FIT` "
        f"（`KeyError:'outcome_label'`），未被改写。\n"
        "- 修复只读取已落盘的汇总表，不重新读取共享行情面板、不重算个股收益/排序/行业命中，"
        "也未修改原始统计合同。\n"
        f"- `outcome_label` 是仅用于报告展示的元数据：`{repair_config['display_only_outcome_label']}`。\n"
    )
    facts = {
        "source_rows": metadata["source_rows"],
        "observed_dates": metadata["observed_dates"],
        "eligible_dates": stats["qa"]["eligible_date_count"],
        "selection_slots_from_existing_table": stats["qa"]["selected_slot_count"],
        "excluded_below_100_date_count": stats["qa"]["excluded_below_100_date_count"],
        "reason_union_count": stats["qa"]["reason_union_count"],
    }
    return text, facts


def build_repair_receipt(*, tables: Mapping[str, pd.DataFrame], facts: Mapping[str, Any],
                         binding: Mapping[str, Any], repair_contract: Mapping[str, Any],
                         hashes: Mapping[str, str], output_sha256: str,
                         output_bytes: int, runtime_seconds: float) -> dict[str, Any]:
    return {
        "schema_version": "A_SHARE_TOP100_REPORT_REPAIR_RECEIPT_V1",
        "experiment_id": EXPERIMENT_ID,
        "route_id": repair_contract["route_id"],
        "execution_id": EXECUTION_ID,
        "status": "REPORT_REPAIRED_FROM_EXISTING_AGGREGATES",
        "parent_full_stats_execution_id": repair_contract["parent_full_stats_execution_id"],
        "parent_full_stats_terminal": "FAILED_NO_FIT",
        "parent_full_stats_stop_reason": "KeyError:'outcome_label'",
        "contract_sha256": hashes["contract_sha256"],
        "input_binding_sha256": hashes["input_binding_sha256"],
        "config_sha256": hashes["config_sha256"],
        "code_sha256": hashes["code_sha256"],
        "parent_statistics_code_sha256": hashes["main_code_sha256"],
        "test_sha256": hashes["test_sha256"],
        "input_table_sha256": {name: item["sha256"] for name, item in binding["table_inputs"].items()},
        "output_sha256": {str(REPORT_PATH.resolve()): output_sha256},
        "output_bytes": int(output_bytes),
        "validated_facts_from_aggregate_tables": dict(facts),
        "validation": {
            "table_count": len(tables),
            "source_panel_reread": False,
            "return_or_rank_recomputed": False,
            "report_tables_hash_matched": True,
            "daily_100_slot_parity": True,
            "coverage_and_eligible_date_parity": True,
            "all_dimension_expected_hits_parity": True,
            "safety_flags": SAFETY_FLAGS,
        },
        "runtime_seconds": round(float(runtime_seconds), 6),
    }


def run() -> dict[str, Any]:
    repair_meta, repair_binding, hashes = load_repair_identity()
    repair_contract = repair_meta["contract"]
    repair_config = repair_meta["config"]
    if set(repair_binding["table_inputs"]) != set(REQUIRED_TABLES):
        raise RepairError("repair input binding must contain exactly the seven frozen aggregate tables")
    if REPORT_PATH.exists() or RECEIPT_PATH.exists():
        raise RepairError("report repair outputs already exist; refusing a second execution")

    main_module = load_module(MAIN_SOURCE, "a_share_top100_main_for_report_repair")
    original_binding = read_json(ORIGINAL_BINDING)
    adapter, core = main_module.load_official_registry(
        original_binding,
        contract_path=REPAIR_CONTRACT,
        input_binding_path=REPAIR_BINDING,
        config_path=REPAIR_CONFIG,
        code_path=REPAIR_SOURCE,
        test_path=TEST_SOURCE,
    )
    prior_events = list(adapter.list_events(EXPERIMENT_ID))
    full_terminal = [
        event for event in prior_events
        if event.get("execution_id") == repair_contract["parent_full_stats_execution_id"]
        and event.get("event_type") == "TERMINAL"
    ]
    if len(full_terminal) != 1 or full_terminal[0].get("terminal_status") != "FAILED_NO_FIT":
        raise RepairError("original FULL_STATS failure is not present as one validated terminal")

    import numpy as np

    context = make_context(
        contract=repair_contract, binding=repair_binding, hashes=hashes,
        runtime={"python": sys.executable, "python_version": sys.version.split()[0],
                 "pandas": pd.__version__, "numpy": np.__version__},
    )
    claim = main_module._claim_stage(
        adapter, core, stage="REPORT_REPAIR", execution_id=EXECUTION_ID,
        contract_sha256=hashes["contract_sha256"],
        input_binding_sha256=hashes["input_binding_sha256"], context=context,
    )
    started = time.monotonic()
    try:
        tables = validate_and_read_tables(repair_binding)
        report_text, facts = render_report(tables, repair_binding, repair_contract, repair_config, main_module)
        report_payload = report_text.encode("utf-8")
        write_new_or_identical(REPORT_PATH, report_payload)
        output_sha = sha256_file(REPORT_PATH)
        receipt = build_repair_receipt(
            tables=tables, facts=facts, binding=repair_binding, repair_contract=repair_contract,
            hashes=hashes, output_sha256=output_sha, output_bytes=len(report_payload),
            runtime_seconds=time.monotonic() - started,
        )
        receipt_payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        write_new_or_identical(RECEIPT_PATH, receipt_payload)
        terminal_context = dict(adapter._route_context[EXECUTION_ID])
        terminal_context["artifact_paths"] = [
            *terminal_context["artifact_paths"], str(REPORT_PATH.resolve()), str(RECEIPT_PATH.resolve())
        ]
        terminal_context["signal_date_range"] = {
            **repair_contract["parent_signal_date_range"],
            "scope": "PARENT_EXPOSED_STATS_RENDERED_FROM_PREEXISTING_AGGREGATES",
        }
        adapter.set_route_context(EXECUTION_ID, terminal_context)
        metrics = {
            "selection_slots": 0,
            "report_input_table_count": len(tables),
            "report_date_count": facts["observed_dates"],
            "report_eligible_date_count": facts["eligible_dates"],
            "report_selected_slot_count_from_existing_tables": facts["selection_slots_from_existing_table"],
            "report_output_bytes": len(report_payload),
        }
        main_module._terminalize_stage(adapter, claim, terminal_status="COMPLETED", metrics=metrics)
        return receipt
    except BaseException as exc:
        main_module._terminalize_stage(adapter, claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
        raise


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
