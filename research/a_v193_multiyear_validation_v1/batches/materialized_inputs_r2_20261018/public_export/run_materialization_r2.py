#!/usr/bin/env python3
"""Bounded R2 no-fit input materialization and date/key coverage projection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
TRACK_REL = Path("research/a_v193_multiyear_validation_v1")
BATCH_REL = TRACK_REL / "batches/materialized_inputs_r2_20261018"
BATCH = ROOT / BATCH_REL
CONTRACT = BATCH / "private/R2_CONTRACT.json"
BINDING = BATCH / "private/R2_INPUT_BINDING.json"
INTENT = BATCH / "private/RUN_INTENT.json"
START_RECEIPT = BATCH / "private/RUN_STARTED_RECEIPT.json"
TERMINAL_RECEIPT = BATCH / "private/TERMINAL_RECEIPT.json"
REGISTRY = ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
RUNNER = Path(__file__).resolve()
ROUTE_ID = "A_V193_MATERIALIZED_INPUTS_R2_20261018"
TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
EXPERIMENT_ID = ROUTE_ID
ARM = "R2_NOFIT_INPUT_MATERIALIZATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, value: Any) -> None:
    write_exclusive(path, json_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(name: str, rows: list[dict[str, Any]], *, metadata: dict[str, Any]) -> dict[str, str]:
    table_dir = BATCH / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / f"{name}.json"
    csv_path = table_dir / f"{name}.csv"
    md_path = table_dir / f"{name}.md"
    write_json_exclusive(json_path, {"schema_version": "R2_TABLE_V1", "metadata": metadata, "rows": rows})
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
        handle.flush()
        os.fsync(handle.fileno())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        cells = []
        for key in fields:
            value = row.get(key)
            cell = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            cells.append(cell.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    write_exclusive(md_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return {"json": str(json_path.relative_to(BATCH)), "csv": str(csv_path.relative_to(BATCH)), "md": str(md_path.relative_to(BATCH))}


def sync_dir() -> None:
    for name in ("private", "inputs", "tables", "public_export"):
        (BATCH / name).mkdir(parents=True, exist_ok=True)


def registry_events() -> list[dict[str, Any]]:
    return [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_adapter(artifact_paths: Iterable[Path]):
    sys.path.insert(0, str(ADAPTER_DIR))
    from execution_guard_core import ExecutionGuard
    from registry_adapter import OfficialRegistryAdapter

    adapter = OfficialRegistryAdapter(
        registry_path=REGISTRY,
        contract_path=CONTRACT,
        input_binding_path=BINDING,
        artifact_paths=tuple(artifact_paths),
        event_profile={
            "track_id": TRACK_ID,
            "route_id": ROUTE_ID,
            "registration_mode": "LIVE_EXECUTION",
            "start_stage": "V193_R2_NOFIT_MATERIALIZATION_STARTED",
            "terminal_stage": "V193_R2_NOFIT_MATERIALIZATION_TERMINAL",
            "occurred_at_reason": "LIVE_R2_NOFIT_EXECUTION_UTC",
            "baseline_id": "V193_S0_NATIVE_IDENTITY_CANNOT_CONFIRM",
            "one_variable": "R2_MATERIALIZE_INPUTS_AND_PROJECT_TRADE_DATE_CODE_COVERAGE",
            "signal_scope": "NO_SIGNAL_SCAN",
            "target_scope": "NO_LABEL_OR_TARGET_SCAN",
            "device": "CPU_PYARROW_DATE_CODE_PROJECTION",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
        },
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def verify_frozen_binding() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = read_json(CONTRACT)
    binding = read_json(BINDING)
    contract_sha = sha256_file(CONTRACT)
    binding_sha = sha256_file(BINDING)
    runner_sha = sha256_file(RUNNER)
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("frozen input binding does not match contract bytes")
    if binding.get("runner_sha256") != runner_sha or contract.get("runner_sha256") != runner_sha:
        raise RuntimeError("frozen identity does not match runner bytes")
    for receipt in binding.get("reused_receipts", []):
        receipt_path = ROOT / receipt["path"]
        if not receipt_path.is_file() or sha256_file(receipt_path) != receipt["sha256"]:
            raise RuntimeError(f"bound receipt identity mismatch: {receipt['path']}")
    if contract.get("experiment_scope") != "NO_FIT_NO_BACKTEST_NO_PREDICTION_NO_LABEL_READ":
        raise RuntimeError("contract scope is not the approved no-fit scope")
    return contract, binding, contract_sha, binding_sha


def _phase_start_body() -> None:
    sync_dir()
    if any(path.exists() for path in (INTENT, START_RECEIPT, TERMINAL_RECEIPT)):
        raise FileExistsError("R2 lifecycle files already exist; automatic re-entry is blocked")
    contract, binding, contract_sha, binding_sha = verify_frozen_binding()
    if any(event.get("route_id") == ROUTE_ID for event in registry_events()):
        raise RuntimeError("R2 route already has registry history; automatic re-entry is blocked")
    execution_id = "A_V193_MATERIALIZED_INPUTS_R2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    before = REGISTRY.read_bytes()
    before_events = registry_events()
    intent = {
        "schema_version": "R2_RUN_INTENT_V1",
        "execution_id": execution_id,
        "experiment_id": EXPERIMENT_ID,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "arm": ARM,
        "contract_path": str(CONTRACT),
        "contract_sha256": contract_sha,
        "input_binding_path": str(BINDING),
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "source_reads_allowed_only_after_durable_start": True,
        "registry_before": {"sha256": hashlib.sha256(before).hexdigest(), "bytes": len(before), "event_count": len(before_events)},
        "materialization_input_count": len(binding["materialization_inputs"]),
        "panel_projection_source_count": len(binding["panel_sources"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
    }
    write_json_exclusive(INTENT, intent)
    immutable_artifacts = [
        RUNNER, CONTRACT, BINDING,
        *(ROOT / item["path"] for item in binding["reused_receipts"]),
        INTENT,
    ]
    adapter, guard = make_adapter(immutable_artifacts)
    claim = guard.claim_no_fit(
        execution_id=execution_id,
        arm=ARM,
        contract_sha256=contract_sha,
        input_binding_sha256=binding_sha,
    )
    matching = [
        event for event in adapter.list_events(EXPERIMENT_ID)
        if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"
    ]
    if len(matching) != 1:
        raise RuntimeError("durable RUN_STARTED readback is not exactly one event")
    write_json_exclusive(START_RECEIPT, {
        "schema_version": "R2_RUN_STARTED_RECEIPT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": intent["runner_sha256"],
        "registry_event": matching[0],
        "durable_start_readback": True,
        "fit_calls": 0,
    })
    print(json.dumps({"phase": "start", "execution_id": execution_id, "run_started": True, "fit_calls": 0}))


def copy_source(item: dict[str, Any]) -> dict[str, Any]:
    source = Path(item["source_path"])
    target = BATCH / item["target_relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_sha = item.get("expected_bytes"), item.get("expected_sha256")
    base = {
        "logical_name": item["logical_name"], "source_path": str(source), "source_drive_id": item.get("drive_id"),
        "target_relative_path": str(target.relative_to(BATCH)), "expected_bytes": expected_size, "expected_sha256": expected_sha,
    }
    if not source.is_file():
        return {**base, "status": "CANNOT_CONFIRM_SOURCE_MISSING", "source_bytes": None, "source_sha256": None, "materialized_bytes": None, "materialized_sha256": None}
    source_bytes = source.stat().st_size
    if expected_size is not None and source_bytes != expected_size:
        return {**base, "status": "CANNOT_CONFIRM_SOURCE_SIZE_MISMATCH", "source_bytes": source_bytes, "source_sha256": None, "materialized_bytes": None, "materialized_sha256": None}
    source_sha = sha256_file(source)
    if expected_sha is not None and source_sha != expected_sha:
        return {**base, "status": "CANNOT_CONFIRM_SOURCE_SHA_MISMATCH", "source_bytes": source_bytes, "source_sha256": source_sha, "materialized_bytes": None, "materialized_sha256": None}
    if target.exists():
        target_size, target_sha = target.stat().st_size, sha256_file(target)
        status = "BYTE_IDENTICAL_ALREADY_MATERIALIZED" if target_size == source_bytes and target_sha == source_sha else "CANNOT_CONFIRM_TARGET_EXISTS_WITH_DIFFERENT_BYTES"
        return {**base, "status": status, "source_bytes": source_bytes, "source_sha256": source_sha, "materialized_bytes": target_size, "materialized_sha256": target_sha}
    digest, written = hashlib.sha256(), 0
    with source.open("rb") as src, target.open("xb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            written += len(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    materialized_sha = sha256_file(target)
    copied_sha = digest.hexdigest()
    valid = written == source_bytes and copied_sha == source_sha and materialized_sha == source_sha
    return {**base, "status": "MATERIALIZED_BYTE_IDENTICAL" if valid else "CANNOT_CONFIRM_COPY_PARITY", "source_bytes": source_bytes, "source_sha256": source_sha, "materialized_bytes": written, "materialized_sha256": materialized_sha}


def manifest_claims(manifest: Any) -> list[dict[str, Any]]:
    selected_tokens = ("date", "schema", "column", "field", "source", "version", "row", "market", "coverage", "dependency")
    denied_tokens = ("formula_text", "formula_source_text", "drive_id", "credential", "token")
    rows: list[dict[str, Any]] = []
    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                low = path.lower()
                if any(token in low for token in denied_tokens):
                    continue
                if isinstance(child, (dict, list)):
                    visit(child, path)
                elif any(token in low for token in selected_tokens):
                    rows.append({"row_type": "manifest_claim", "dependency": path, "source": "INPUT_DATA_MANIFEST", "value": child, "status": "DECLARED_CLAIM_ONLY"})
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")
    visit(manifest)
    return rows


def build_identity_rows(inventory: list[dict[str, Any]], binding: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {row["logical_name"]: row for row in inventory}
    snapshot, cand3 = by_name.get("v193_original_snapshot", {}), by_name.get("v192_cand3_comparison", {})
    alias_equal = (
        snapshot.get("status") == "MATERIALIZED_BYTE_IDENTICAL"
        and cand3.get("status") == "MATERIALIZED_BYTE_IDENTICAL"
        and snapshot.get("source_sha256") == cand3.get("source_sha256")
    )
    rows = [
        {"row_type": "baseline_identity", "dependency": "193_S0 native baseline source and installed provenance",
         "requirement": "Required before any model or evaluation freeze",
         "historical_availability": "V193 snapshot and V192 CAND3 comparison texts are physically available",
         "observed": "V193 snapshot byte SHA equals V192 CAND3 comparison byte SHA" if alias_equal else "snapshot/CAND3 byte identity not established",
         "status": "CANNOT_CONFIRM_NATIVE_V193_IDENTITY" if alias_equal else "CANNOT_CONFIRM_IDENTITY_COMPARISON",
         "evidence": {"v193_snapshot_sha256": snapshot.get("materialized_sha256"), "v192_cand3_sha256": cand3.get("materialized_sha256"), "byte_equal": bool(alias_equal)}},
        {"row_type": "dependency_gate", "dependency": "Native V193 producer, exact formula version, and execution output lineage",
         "requirement": "Required to assert S0 is the native historical baseline",
         "historical_availability": "CANNOT_CONFIRM from the bound source receipts",
         "observed": "Text hash equality identifies an alias match only; it does not establish producer or execution semantics",
         "status": "CANNOT_CONFIRM", "evidence": {"bound_receipt": "SOURCE_METADATA_RECEIPT_V1"}},
        {"row_type": "dependency_gate", "dependency": "Historical candidate aggregation dependencies including EPPY and BKEXIST",
         "requirement": "Required to reproduce the original candidate stream",
         "historical_availability": "No V193-native producer output or PIT-bound dependency map is present in the R2 source binding",
         "observed": "No dependency values or candidate outputs read in this batch",
         "status": "CANNOT_CONFIRM", "evidence": {"values_read": False, "candidate_rows_read": False}},
        {"row_type": "dependency_gate", "dependency": "Raw market OHLCV date/security keys",
         "requirement": "Required physical history for later feature calculation",
         "historical_availability": "20261015 QFQ, selector, and backtest panel files are bound to the old recovery receipt",
         "observed": "Only trade_date and ts_code are projected for annual physical coverage",
         "status": "PHYSICAL_COVERAGE_ONLY", "evidence": {"projected_fields": ["trade_date", "ts_code"], "value_fields_read": []}},
        {"row_type": "dependency_gate", "dependency": "Market context for the approved one-candidate research design",
         "requirement": "Optional extra information for the later candidate",
         "historical_availability": "Selector schema receipt lists market_context_source; exact source semantics and PIT mapping are not certified",
         "observed": "No market context values read", "status": "CANNOT_CONFIRM", "evidence": {"values_read": False}},
        {"row_type": "dependency_gate", "dependency": "Historical listing, delisting, ST, suspension, limit, corporate-action, and source-vintage state",
         "requirement": "Required for point-in-time candidate and return eligibility",
         "historical_availability": "Not established by date/code-only projections",
         "observed": "No eligibility or target columns read", "status": "CANNOT_CONFIRM", "evidence": {"pit_certified": False}},
        {"row_type": "dependency_gate", "dependency": "V193 formula-derived features and warmup parity",
         "requirement": "Required before minimum 2-year qualified training and complete warmup can be assessed",
         "historical_availability": "Selector/backtest schemas exist in prior V192 projections; native V193 feature parity and complete warmup are unverified",
         "observed": "No feature values or warmup calculations read", "status": "CANNOT_CONFIRM", "evidence": {"feature_values_read": False, "warmup_certified": False}},
        {"row_type": "dependency_gate", "dependency": "T5 D+1 open to D+6 open and short T2 D+1 open to D+3 open target values",
         "requirement": "For a later separately registered evaluation only",
         "historical_availability": "Not assessed in this batch", "observed": "No target or label values read",
         "status": "NOT_READ_BY_SCOPE", "evidence": {"target_values_read": False, "labels_read": False}},
    ]
    if any(row.get("logical_name") == "input_data_manifest" and row.get("status") == "MATERIALIZED_BYTE_IDENTICAL" for row in inventory):
        manifest_path = BATCH / next(row["target_relative_path"] for row in inventory if row["logical_name"] == "input_data_manifest")
        rows.extend(manifest_claims(read_json(manifest_path)))
    return rows


def parse_day(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.strftime("%Y%m%d"))
    if isinstance(value, date):
        return int(value.strftime("%Y%m%d"))
    if isinstance(value, (int, float)):
        number = int(value)
        return number if 19000101 <= number <= 21001231 else None
    digits = "".join(char for char in str(value).strip() if char.isdigit())
    if len(digits) < 8:
        return None
    number = int(digits[:8])
    return number if 19000101 <= number <= 21001231 else None


def project_panel(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(source["path"])
    result_meta = {
        "source_id": source["source_id"], "snapshot_asof": source["snapshot_asof"], "source_path": str(path),
        "expected_bytes": source["expected_bytes"], "expected_sha256": source["expected_sha256"],
        "expected_rows_from_receipt": source["expected_rows"], "date_field": source["date_field"],
        "security_field": source["security_field"], "reused_receipt": source["reused_receipt"],
        "reused_receipt_sha256": source["reused_receipt_sha256"], "data_fields_read": ["trade_date", "ts_code"],
        "value_fields_read": [], "label_prediction_weight_fields_read": [],
        "v193_trainable_coverage_status": "CANNOT_CONFIRM",
        "v193_trainable_coverage_reason": "Date/security row coverage does not establish native formula features, PIT eligibility, labels, source vintage, or complete warmup.",
    }
    if not path.is_file():
        return [], {**result_meta, "status": "CANNOT_CONFIRM_SOURCE_MISSING", "actual_bytes": None, "actual_sha256": None, "actual_rows": None}
    actual_bytes = path.stat().st_size
    if actual_bytes != source["expected_bytes"]:
        return [], {**result_meta, "status": "CANNOT_CONFIRM_SOURCE_SIZE_MISMATCH", "actual_bytes": actual_bytes, "actual_sha256": None, "actual_rows": None}
    actual_sha = sha256_file(path)
    if actual_sha != source["expected_sha256"]:
        return [], {**result_meta, "status": "CANNOT_CONFIRM_SOURCE_SHA_MISMATCH", "actual_bytes": actual_bytes, "actual_sha256": actual_sha, "actual_rows": None}
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if source["date_field"] not in names or source["security_field"] not in names:
        return [], {**result_meta, "status": "CANNOT_CONFIRM_REQUIRED_KEY_MISSING", "actual_bytes": actual_bytes, "actual_sha256": actual_sha, "actual_rows": parquet.metadata.num_rows}
    actual_rows = int(parquet.metadata.num_rows)
    yearly: dict[int, dict[str, Any]] = {}
    missing_date_rows = 0
    for batch in parquet.iter_batches(columns=[source["date_field"], source["security_field"]], batch_size=250_000):
        for day_value, security in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            day = parse_day(day_value)
            if day is None:
                missing_date_rows += 1
                continue
            year = day // 10000
            bucket = yearly.setdefault(year, {"rows": 0, "dates": set(), "securities": set(), "first_trade_date": day, "last_trade_date": day, "null_security_rows": 0})
            bucket["rows"] += 1
            bucket["dates"].add(day)
            bucket["first_trade_date"] = min(bucket["first_trade_date"], day)
            bucket["last_trade_date"] = max(bucket["last_trade_date"], day)
            if security is None:
                bucket["null_security_rows"] += 1
            else:
                bucket["securities"].add(str(security))
    rows = []
    for year in sorted(yearly):
        bucket = yearly[year]
        rows.append({
            "source_id": source["source_id"], "snapshot_asof": source["snapshot_asof"],
            "source_sha256": actual_sha, "source_bytes": actual_bytes, "source_row_count": actual_rows,
            "year": year, "row_count": bucket["rows"], "unique_trade_dates": len(bucket["dates"]),
            "unique_securities": len(bucket["securities"]), "first_trade_date": bucket["first_trade_date"],
            "last_trade_date": bucket["last_trade_date"], "null_trade_date_rows_global": missing_date_rows,
            "null_security_rows": bucket["null_security_rows"], "physical_coverage_status": "VERIFIED_DATE_CODE_PROJECTION",
            "coverage_interpretation": source["coverage_interpretation"], "v193_trainable_coverage_status": "CANNOT_CONFIRM",
        })
    status = "VERIFIED_DATE_CODE_PROJECTION" if actual_rows == source["expected_rows"] else "VERIFIED_DATE_CODE_PROJECTION_ROWCOUNT_DIFFERS_FROM_PRIOR_RECEIPT"
    return rows, {**result_meta, "status": status, "actual_bytes": actual_bytes, "actual_sha256": actual_sha, "actual_rows": actual_rows, "missing_trade_date_rows": missing_date_rows}


def append_fsync(path: Path, text: str) -> None:
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def summarize_registry() -> dict[str, Any]:
    raw = REGISTRY.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        starts = {(str(row.get("route_id")), str(row.get("execution_id"))) for row in rows if row.get("event_type") == "RUN_STARTED"}
        terminals = {(str(row.get("route_id")), str(row.get("execution_id"))) for row in rows if row.get("event_type") == "TERMINAL"}
        return {"route_count": len({str(row.get("route_id")) for row in rows if row.get("route_id")}),
                "event_count": len(rows), "execution_count": len({str(row.get("execution_id")) for row in rows if row.get("execution_id")}),
                "correction_count": sum(row.get("event_type") == "CORRECTION" for row in rows),
                "open_execution_count": len(starts - terminals)}
    track_rows = [row for row in events if row.get("track_id") == TRACK_ID]
    route_rows = [row for row in events if row.get("route_id") == ROUTE_ID]
    return {
        "registry_path": "contracts/EXPERIMENT_REGISTRY.jsonl", "registry_sha256": hashlib.sha256(raw).hexdigest(),
        "registry_bytes": len(raw), "global": summary(events), "track": summary(track_rows), "route": summary(route_rows),
        "track_known_contract_only_study_units": 1,
        "global_unregistered_route_count": None, "global_unregistered_route_count_reason": "Not audited in this bounded batch.",
        "global_contract_only_count": None, "global_contract_only_count_reason": "Not audited in this bounded batch.",
    }


def _phase_finish_body() -> None:
    contract, binding, contract_sha, binding_sha = verify_frozen_binding()
    if not INTENT.is_file():
        raise FileNotFoundError("RUN_INTENT missing; phase start must be completed first")
    intent = read_json(INTENT)
    execution_id = intent["execution_id"]
    if TERMINAL_RECEIPT.exists():
        raise FileExistsError("R2 terminal already exists; automatic re-entry is blocked")
    started = read_json(START_RECEIPT)
    if started.get("execution_id") != execution_id or started.get("contract_sha256") != contract_sha or started.get("input_binding_sha256") != binding_sha:
        raise RuntimeError("durable start receipt does not match the frozen identity")

    inventory = [copy_source(item) for item in binding["materialization_inputs"]]
    write_table("01_materialization_inventory", inventory, metadata={
        "copy_mode": "raw byte stream with source and target SHA-256 and byte checks",
        "formula_text_parsed": False, "source_reads_started_after_registry_start": True,
    })
    identity_rows = build_identity_rows(inventory, binding)
    write_table("02_s0_identity_dependency", identity_rows, metadata={
        "baseline_identity_status": "CANNOT_CONFIRM_NATIVE_V193_IDENTITY", "source_formula_text_parsed": False,
        "target_or_label_values_read": False, "field_and_pit_admission": "CANNOT_CONFIRM",
        "approved_research_definition": contract["approved_research_definition"],
    })

    all_coverage_rows: list[dict[str, Any]] = []
    panel_summaries: list[dict[str, Any]] = []
    for source in binding["panel_sources"]:
        print(json.dumps({"phase": "finish", "panel": source["source_id"], "status": "starting", "data_fields": ["trade_date", "ts_code"]}), flush=True)
        rows, summary = project_panel(source)
        all_coverage_rows.extend(rows)
        panel_summaries.append(summary)
        print(json.dumps({"phase": "finish", "panel": source["source_id"], "status": summary["status"], "years": len(rows)}), flush=True)
    write_table("03_annual_panel_coverage", all_coverage_rows, metadata={
        "panel_summaries": panel_summaries, "projection_fields": ["trade_date", "ts_code"], "value_fields_read": [],
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
        "physical_coverage_is_not_v193_trainable_coverage": True, "v193_trainable_coverage_status": "CANNOT_CONFIRM",
        "outer_half_years": contract["approved_research_definition"]["outer_target_half_years"],
    })

    safe_inventory = [
        {key: row.get(key) for key in ("logical_name", "target_relative_path", "expected_bytes", "expected_sha256", "source_bytes", "source_sha256", "materialized_bytes", "materialized_sha256", "status")}
        for row in inventory
    ]
    safe_identity = [
        {key: row.get(key) for key in ("row_type", "dependency", "requirement", "historical_availability", "observed", "status", "evidence")}
        for row in identity_rows if row.get("row_type") != "manifest_claim"
    ]
    coverage = read_json(BATCH / "tables/03_annual_panel_coverage.json")
    safe_panel_summaries = [
        {key: summary.get(key) for key in (
            "source_id", "snapshot_asof", "expected_bytes", "expected_sha256", "expected_rows_from_receipt",
            "date_field", "security_field", "reused_receipt", "reused_receipt_sha256", "data_fields_read",
            "value_fields_read", "label_prediction_weight_fields_read", "status", "actual_bytes",
            "actual_sha256", "actual_rows", "missing_trade_date_rows", "v193_trainable_coverage_status",
            "v193_trainable_coverage_reason",
        )}
        for summary in coverage["metadata"]["panel_summaries"]
    ]
    safe_coverage_metadata = {
        key: value for key, value in coverage["metadata"].items() if key != "panel_summaries"
    }
    safe_coverage_metadata["panel_summaries"] = safe_panel_summaries
    core_ok = (
        all(row.get("status") == "MATERIALIZED_BYTE_IDENTICAL" for row in inventory)
        and len(all_coverage_rows) > 0
        and all(summary.get("status") == "VERIFIED_DATE_CODE_PROJECTION" for summary in panel_summaries)
    )
    public_json = {
        "schema_version": "V193_R2_PUBLIC_SAFE_AGGREGATE_V1", "batch_date": "2026-10-18",
        "status": "NO_FIT_INPUT_MATERIALIZATION_AND_PHYSICAL_COVERAGE_ONLY",
        "required_materialization_and_projection_status": "VERIFIED" if core_ok else "CANNOT_CONFIRM",
        "baseline_identity_status": "CANNOT_CONFIRM_NATIVE_V193_IDENTITY",
        "source_text_sha256_equal_to_v192_cand3": bool(next((row.get("evidence", {}).get("byte_equal") for row in safe_identity if row.get("row_type") == "baseline_identity"), False)),
        "materialization_inventory": safe_inventory, "identity_dependency": safe_identity,
        "annual_panel_coverage": {"metadata": safe_coverage_metadata, "rows": coverage["rows"]},
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
        "research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False,
    }
    write_json_exclusive(BATCH / "public_export/SAFE_AGGREGATE.json", public_json)
    progress = [
        "# V193 R2 materialization progress", "",
        "This export records a bounded no-fit materialization and a physical date/security-key projection over frozen 20261015 panels.", "",
        f"- Byte-identical materialized source documents: {sum(row.get('status') == 'MATERIALIZED_BYTE_IDENTICAL' for row in inventory)} of {len(inventory)}.",
        "- The V193 snapshot text has the same SHA-256 as the V192 CAND3 comparison text." if public_json["source_text_sha256_equal_to_v192_cand3"] else "- V193 snapshot and V192 CAND3 byte equality is CANNOT_CONFIRM.",
        "- Native V193 producer identity remains CANNOT_CONFIRM.",
        "- Annual panel rows report physical row, date, and security coverage only. V193 trainable coverage remains CANNOT_CONFIRM because feature parity, PIT eligibility, complete warmup, and label gates were not evaluated.",
        "- The only panel columns read by this batch were trade_date and ts_code.",
        "- Fit calls: 0; backtests: 0; predictions: 0; labels read: 0; weights read: 0.",
        "- Research-only flags: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.", "",
        "The public aggregate contains no Drive IDs, account-local source paths, formula text, raw market rows, labels, predictions, weights, or run logs.", "",
        "The included runner source depends on the project's existing OfficialRegistryAdapter and ExecutionGuard; this export is not a standalone runtime package.",
    ]
    write_exclusive(BATCH / "public_export/PROGRESS_REPORT.md", ("\n".join(progress) + "\n").encode("utf-8"))
    shutil.copyfile(RUNNER, BATCH / "public_export/run_materialization_r2.py")
    write_exclusive(BATCH / "public_export/DEPENDENCIES.md", (
        "The runner reuses OfficialRegistryAdapter and ExecutionGuard from "
        "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity. "
        "The public copy is for source review only and is not standalone: execution also requires the private frozen contract, input binding, and local source files. "
        "It contains no Drive IDs or account-local source paths.\n"
    ).encode("utf-8"))
    readme_prefix = [
        "## V193 R2 materialized inputs (2026-10-18)", "",
        "The earlier public report reflects research evidence only through 2026-10-18 06:09 UTC and remains historical context.",
        "The R2 batch materialized the bound small source documents and added annual physical date/security coverage for frozen 20261015 panels; native V193 baseline identity and trainable coverage remain CANNOT_CONFIRM.",
        "",
        "- [R2 progress report](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/PROGRESS_REPORT.md)",
        "- [R2 safe aggregate JSON](research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/public_export/SAFE_AGGREGATE.json)",
    ]
    write_exclusive(BATCH / "public_export/README_PREFIX.md", ("\n".join(readme_prefix) + "\n").encode("utf-8"))

    output_paths = [
        RUNNER, CONTRACT, BINDING, INTENT, START_RECEIPT,
        *(BATCH / f"tables/{name}.{ext}" for name in (
            "01_materialization_inventory", "02_s0_identity_dependency", "03_annual_panel_coverage"
        ) for ext in ("json", "csv", "md")),
        BATCH / "public_export/SAFE_AGGREGATE.json", BATCH / "public_export/PROGRESS_REPORT.md",
        BATCH / "public_export/run_materialization_r2.py", BATCH / "public_export/DEPENDENCIES.md",
        BATCH / "public_export/README_PREFIX.md",
    ]
    output_paths = [path for path in output_paths if path.is_file()]
    adapter, guard = make_adapter(output_paths)
    from execution_guard_core import ExecutionClaim, NO_FIT_ENGINEERING_KIND
    claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, NO_FIT_ENGINEERING_KIND, 0)
    guard.append_no_fit_terminal(claim, succeeded=core_ok, error_type=None if core_ok else "CANNOT_CONFIRM_REQUIRED_INPUT_OR_PANEL")

    reg = summarize_registry()
    terminal_events = [
        event for event in adapter.list_events(EXPERIMENT_ID)
        if event.get("execution_id") == execution_id and event.get("event_type") == "TERMINAL"
    ]
    terminal_data = {
        "schema_version": "R2_TERMINAL_RECEIPT_V1", "execution_id": execution_id, "route_id": ROUTE_ID,
        "track_id": TRACK_ID, "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER), "terminal_event": terminal_events[0] if len(terminal_events) == 1 else None,
        "registry_summary_after_terminal": reg, "materialization_inventory_table": "tables/01_materialization_inventory.json",
        "identity_dependency_table": "tables/02_s0_identity_dependency.json", "annual_coverage_table": "tables/03_annual_panel_coverage.json",
        "public_export": "public_export/SAFE_AGGREGATE.json", "fit_calls": 0, "backtests": 0, "predictions": 0,
        "labels_read": 0, "weights_read": 0, "data_fields_read": ["trade_date", "ts_code"], "value_fields_read": [],
        "terminal_status": "ENGINEERING_CANARY_ONLY" if core_ok else "CANNOT_CONFIRM_ENGINEERING_CANARY",
    }
    write_json_exclusive(TERMINAL_RECEIPT, terminal_data)
    index = {
        "schema_version": "R2_EVIDENCE_INDEX_V1", "classification": "NAVIGATION_SNAPSHOT_NOT_AUTHORITY",
        "batch": str(BATCH_REL), "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER), "registry": reg,
        "route_event_execution_correction_counts": {"global": reg["global"], "track": reg["track"], "route": reg["route"]},
        "terminal_receipt": "private/TERMINAL_RECEIPT.json",
        "tables": {"materialization_inventory": "tables/01_materialization_inventory.json",
                   "s0_identity_dependency": "tables/02_s0_identity_dependency.json",
                   "annual_panel_coverage": "tables/03_annual_panel_coverage.json"},
        "global_unregistered_route_count": None, "global_contract_only_count": None,
    }
    write_json_exclusive(BATCH / "EVIDENCE_INDEX.json", index)
    sync_receipt = {
        "schema_version": "R2_REGISTRY_SYNC_RECEIPT_V1", "registry": reg,
        "sync_targets": ["CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md"],
        "batch_local_evidence_index": "EVIDENCE_INDEX.json",
        "machine_index_note": "EVIDENCE_INDEX.json is NAVIGATION_SNAPSHOT_NOT_AUTHORITY; global authority indexes were not edited.",
        "root_append_status": "PENDING",
    }
    write_json_exclusive(BATCH / "private/REGISTRY_SYNC_RECEIPT.json", sync_receipt)
    current_append = (
        "\n\n## 2026-10-18 V193 R2 no-fit materialization\n"
        f"- 状态：{terminal_data['terminal_status']}；小原件与 20261015 冻结 panel 的 trade_date/ts_code 年度物理覆盖见本批表。193_S0 原生身份及 V193 可训练覆盖保持 CANNOT_CONFIRM。\n"
        f"- Registry：contracts/EXPERIMENT_REGISTRY.jsonl SHA-256 {reg['registry_sha256']}；V193 track routes/events/executions/corrections/open={reg['track']['route_count']}/{reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['correction_count']}/{reg['track']['open_execution_count']}；global 未登记 route 与 contract-only 库存未审计，保持 CANNOT_CONFIRM。\n"
        "- 边界：fit/backtest/prediction/label/weight reads 均为 0；研究只读，未改变 production/BUY/orders。\n"
        f"- 证据导航索引：{BATCH_REL}/EVIDENCE_INDEX.json（NAVIGATION_SNAPSHOT_NOT_AUTHORITY）。\n"
    )
    append_fsync(ROOT / "CURRENT_STATE.md", current_append)
    append_fsync(ROOT / "HK_OHLCV_RESEARCH_STATE.md", (
        "\n\n## 2026-10-18 global registry sync note for V193 R2\n"
        f"- Global EXPERIMENT_REGISTRY SHA-256 {reg['registry_sha256']}; route/event/execution/correction counts {reg['global']['route_count']}/{reg['global']['event_count']}/{reg['global']['execution_count']}/{reg['global']['correction_count']}; V193 track open executions {reg['track']['open_execution_count']}. This A-share R2 activity made no HK track claim or HK coverage change.\n"
    ))
    append_fsync(ROOT / "PROGRESS.md", (
        "\n\n## 2026-10-18 V193 R2 materialization registry sync\n"
        f"- 20261015 frozen panels received date/security-key annual physical coverage only; native V193 identity and trainable coverage remain CANNOT_CONFIRM. Registry SHA-256 {reg['registry_sha256']}; V193 track events/executions/open {reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['open_execution_count']}. See {BATCH_REL}/EVIDENCE_INDEX.json.\n"
    ))
    sync_receipt["root_append_status"] = "APPENDED_FSYNC"
    sync_receipt["root_file_sha256"] = {
        "CURRENT_STATE.md": sha256_file(ROOT / "CURRENT_STATE.md"),
        "HK_OHLCV_RESEARCH_STATE.md": sha256_file(ROOT / "HK_OHLCV_RESEARCH_STATE.md"),
        "PROGRESS.md": sha256_file(ROOT / "PROGRESS.md"),
    }
    write_json_exclusive(BATCH / "private/REGISTRY_SYNC_RECEIPT_FINAL.json", sync_receipt)
    write_json_exclusive(BATCH / "private/POST_SYNC_READBACK.json", {
        "registry": summarize_registry(), "root_file_sha256": sync_receipt["root_file_sha256"],
        "terminal_receipt_sha256": sha256_file(TERMINAL_RECEIPT),
        "evidence_index_sha256": sha256_file(BATCH / "EVIDENCE_INDEX.json"),
        "sync_receipt_sha256": sha256_file(BATCH / "private/REGISTRY_SYNC_RECEIPT_FINAL.json"),
        "status_files_append_fsynced": True,
    })
    print(json.dumps({"phase": "finish", "execution_id": execution_id, "terminal_status": terminal_data["terminal_status"],
                      "registry_sha256": reg["registry_sha256"], "track_open_execution_count": reg["track"]["open_execution_count"],
                      "coverage_rows": len(all_coverage_rows), "fit_calls": 0}))
    if not core_ok:
        raise RuntimeError("R2 terminalized CANNOT_CONFIRM; inspect materialization and panel summaries")


def close_open_claim_on_error(exc: BaseException) -> None:
    if not INTENT.is_file():
        return
    intent = read_json(INTENT)
    execution_id = str(intent.get("execution_id") or "")
    if not execution_id:
        return
    contract_sha = str(intent.get("contract_sha256") or "")
    binding_sha = str(intent.get("input_binding_sha256") or "")
    artifacts = [RUNNER, CONTRACT, BINDING, INTENT, START_RECEIPT]
    adapter, guard = make_adapter(artifacts)
    events = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == execution_id]
    if not any(event.get("event_type") == "RUN_STARTED" for event in events) or any(event.get("event_type") == "TERMINAL" for event in events):
        return
    from execution_guard_core import ExecutionClaim, NO_FIT_ENGINEERING_KIND
    claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, NO_FIT_ENGINEERING_KIND, 0)
    guard.append_no_fit_terminal(claim, succeeded=False, error_type=type(exc).__name__)


def phase_start() -> None:
    try:
        _phase_start_body()
    except BaseException as exc:
        close_open_claim_on_error(exc)
        failure_path = BATCH / "private/FAILURE_DIAGNOSTIC.json"
        if INTENT.exists() and not failure_path.exists():
            write_json_exclusive(failure_path, {
                "phase": "start", "error_type": type(exc).__name__, "error": str(exc),
                "execution_id": read_json(INTENT).get("execution_id"),
                "registry_open_claim_closed": TERMINAL_RECEIPT.exists(),
            })
        raise


def phase_finish() -> None:
    try:
        _phase_finish_body()
    except BaseException as exc:
        close_open_claim_on_error(exc)
        failure_path = BATCH / "private/FAILURE_DIAGNOSTIC.json"
        if INTENT.exists() and not failure_path.exists():
            write_json_exclusive(failure_path, {
                "phase": "finish", "error_type": type(exc).__name__, "error": str(exc),
                "execution_id": read_json(INTENT).get("execution_id"),
                "registry_open_claim_closed": True,
            })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "finish"))
    args = parser.parse_args()
    phase_start() if args.phase == "start" else phase_finish()


if __name__ == "__main__":
    main()
