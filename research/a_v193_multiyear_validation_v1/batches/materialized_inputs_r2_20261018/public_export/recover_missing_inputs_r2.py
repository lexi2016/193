#!/usr/bin/env python3
"""Recover only the two R2 originals that failed the first byte check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
TRACK_REL = Path("research/a_v193_multiyear_validation_v1")
BATCH_REL = TRACK_REL / "batches/materialized_inputs_r2_20261018"
BATCH = ROOT / BATCH_REL
CONT = BATCH / "continuation_r1"
PRIVATE = CONT / "private"
CONTRACT = PRIVATE / "CONTRACT.json"
BINDING = PRIVATE / "INPUT_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
REGISTRY = ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
RUNNER = Path(__file__).resolve()
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
ROUTE_ID = "A_V193_R2_SOURCE_CONTINUATION_20261018"
TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
EXPERIMENT_ID = ROUTE_ID
ARM = "R2_RECOVER_TWO_MISSING_ORIGINALS"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_identity() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = read_json(CONTRACT)
    binding = read_json(BINDING)
    contract_sha = sha256_file(CONTRACT)
    binding_sha = sha256_file(BINDING)
    runner_sha = sha256_file(RUNNER)
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("continuation binding does not match frozen contract")
    if contract.get("runner_sha256") != runner_sha or binding.get("runner_sha256") != runner_sha:
        raise RuntimeError("continuation runner identity mismatch")
    if contract.get("source_count") != 2 or len(binding.get("sources", [])) != 2:
        raise RuntimeError("continuation scope must bind exactly two unfinished originals")
    continuation = contract.get("continuation_of", {})
    if continuation.get("terminal_receipt_sha256") != sha256_file(BATCH / "private/TERMINAL_RECEIPT.json"):
        raise RuntimeError("base terminal receipt changed since continuation contract")
    if continuation.get("materialization_inventory_sha256") != sha256_file(BATCH / "tables/01_materialization_inventory.json"):
        raise RuntimeError("base inventory changed since continuation contract")
    return contract, binding, contract_sha, binding_sha


def make_adapter(artifact_paths: list[Path]):
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
            "start_stage": "V193_R2_SOURCE_CONTINUATION_STARTED",
            "terminal_stage": "V193_R2_SOURCE_CONTINUATION_TERMINAL",
            "occurred_at_reason": "LIVE_R2_SOURCE_RECOVERY_UTC",
            "baseline_id": "V193_S0_NATIVE_IDENTITY_CANNOT_CONFIRM",
            "one_variable": "R2_RECOVER_ONLY_TWO_MISSING_SMALL_ORIGINALS",
            "signal_scope": "NO_SIGNAL_SCAN",
            "target_scope": "NO_LABEL_OR_TARGET_SCAN",
            "device": "CPU_RAW_BYTE_VALIDATION",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
        },
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def base_artifacts() -> list[Path]:
    return [
        RUNNER, CONTRACT, BINDING, INTENT, START_RECEIPT,
        BATCH / "private/TERMINAL_RECEIPT.json",
        BATCH / "tables/01_materialization_inventory.json",
        BATCH / "tables/02_s0_identity_dependency.json",
        BATCH / "tables/03_annual_panel_coverage.json",
    ]


def start() -> None:
    contract, binding, contract_sha, binding_sha = frozen_identity()
    for path in (INTENT, START_RECEIPT, PRIVATE / "TERMINAL_RECEIPT.json"):
        if path.exists():
            raise FileExistsError(f"continuation lifecycle path already exists: {path.name}")
    existing = []
    if REGISTRY.is_file():
        existing = [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(event.get("route_id") == ROUTE_ID for event in existing):
        raise RuntimeError("continuation route already has registry history; re-entry blocked")
    execution_id = ROUTE_ID + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw = REGISTRY.read_bytes()
    intent = {
        "schema_version": "R2_SOURCE_CONTINUATION_INTENT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "experiment_id": EXPERIMENT_ID,
        "arm": ARM,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "registry_before": {"sha256": sha256_bytes(raw), "bytes": len(raw), "event_count": len(existing)},
        "source_count": 2,
        "source_reads_allowed_only_after_durable_start": True,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_exclusive(INTENT, intent)
    immutable = [path for path in base_artifacts() if path.is_file()]
    adapter, guard = make_adapter(immutable)
    claim = guard.claim_no_fit(
        execution_id=execution_id,
        arm=ARM,
        contract_sha256=contract_sha,
        input_binding_sha256=binding_sha,
    )
    events = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
    if len(events) != 1:
        raise RuntimeError("continuation RUN_STARTED durable readback not exactly one")
    write_exclusive(START_RECEIPT, {
        "schema_version": "R2_SOURCE_CONTINUATION_RUN_STARTED_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "event": events[0],
        "durable_start_readback": True,
        "fit_calls": 0,
    })
    print(json.dumps({"phase": "start", "execution_id": execution_id, "registered": True, "source_count": 2, "fit_calls": 0}))


def registry_summary() -> dict[str, Any]:
    raw = REGISTRY.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    def summarize(items: list[dict[str, Any]]) -> dict[str, int]:
        starts = {(str(item.get("route_id")), str(item.get("execution_id"))) for item in items if item.get("event_type") == "RUN_STARTED"}
        terminals = {(str(item.get("route_id")), str(item.get("execution_id"))) for item in items if item.get("event_type") == "TERMINAL"}
        return {
            "route_count": len({item.get("route_id") for item in items if item.get("route_id")}),
            "event_count": len(items),
            "execution_count": len({item.get("execution_id") for item in items if item.get("execution_id")}),
            "correction_count": sum(item.get("event_type") == "CORRECTION" for item in items),
            "open_execution_count": len(starts - terminals),
        }

    track = [event for event in events if event.get("track_id") == TRACK_ID]
    route = [event for event in events if event.get("route_id") == ROUTE_ID]
    return {
        "registry_path": "contracts/EXPERIMENT_REGISTRY.jsonl",
        "registry_sha256": sha256_bytes(raw),
        "registry_bytes": len(raw),
        "global": summarize(events),
        "track": summarize(track),
        "route": summarize(route),
        "global_unregistered_route_count": None,
        "global_unregistered_route_count_reason": "Not audited in this bounded continuation.",
        "global_contract_only_count": None,
        "global_contract_only_count_reason": "Not audited in this bounded continuation.",
    }


def append_fsync(path: Path, text: str) -> None:
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def finish() -> None:
    contract, binding, contract_sha, binding_sha = frozen_identity()
    intent = read_json(INTENT)
    execution_id = intent["execution_id"]
    start_receipt = read_json(START_RECEIPT)
    if start_receipt.get("execution_id") != execution_id or start_receipt.get("contract_sha256") != contract_sha or start_receipt.get("input_binding_sha256") != binding_sha:
        raise RuntimeError("continuation start receipt identity mismatch")
    receipt_path = PRIVATE / "FETCH_READBACK.json"
    if receipt_path.exists() or (PRIVATE / "TERMINAL_RECEIPT.json").exists():
        raise FileExistsError("continuation already has fetch or terminal receipt; re-entry blocked")

    inventory = []
    all_match = True
    for source in binding["sources"]:
        target = BATCH / source["target_relative_path"]
        row = {key: source.get(key) for key in ("logical_name", "target_relative_path", "expected_bytes", "expected_sha256")}
        if not target.is_file():
            status, size, digest = "CANNOT_CONFIRM_NO_RAW_BYTES_WRITTEN", None, None
        else:
            size, digest = target.stat().st_size, sha256_file(target)
            status = "MATERIALIZED_BYTE_IDENTICAL" if size == source["expected_bytes"] and digest == source["expected_sha256"] else "CANNOT_CONFIRM_TARGET_BYTES_MISMATCH"
        all_match = all_match and status == "MATERIALIZED_BYTE_IDENTICAL"
        inventory.append({**row, "actual_bytes": size, "actual_sha256": digest, "status": status})
    PRIVATE.mkdir(parents=True, exist_ok=True)
    write_exclusive(receipt_path, {
        "schema_version": "R2_SOURCE_CONTINUATION_FETCH_READBACK_V1",
        "execution_id": execution_id,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "source_inventory": inventory,
        "all_expected_raw_bytes_verified": all_match,
        "no_formula_or_market_values_read": True,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
    })

    public_aggregate = {
        "schema_version": "V193_R2_SOURCE_CONTINUATION_SAFE_AGGREGATE_V1",
        "status": "VERIFIED_TWO_MISSING_ORIGINALS" if all_match else "CANNOT_CONFIRM_TWO_MISSING_ORIGINALS",
        "base_materialization_status": "3_OF_5_BYTE_IDENTICAL",
        "continuation_expected_files": len(inventory),
        "continuation_verified_files": sum(row["status"] == "MATERIALIZED_BYTE_IDENTICAL" for row in inventory),
        "files": [{key: row.get(key) for key in ("logical_name", "target_relative_path", "expected_bytes", "expected_sha256", "actual_bytes", "actual_sha256", "status")} for row in inventory],
        "baseline_identity_status": "CANNOT_CONFIRM_NATIVE_V193_IDENTITY",
        "v193_trainable_coverage_status": "CANNOT_CONFIRM",
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
        "research_only": True,
        "production_allowed": False,
        "buy_gate_updated": False,
        "orders_sent": False,
    }
    public_aggregate_path = BATCH / "public_export/CONTINUATION_SAFE_AGGREGATE.json"
    public_report_path = BATCH / "public_export/PROGRESS_REPORT_R2_COMPLETE.md"
    public_source_path = BATCH / "public_export/recover_missing_inputs_r2.py"
    public_dependency_path = BATCH / "public_export/CONTINUATION_DEPENDENCIES.md"
    write_exclusive(public_aggregate_path, public_aggregate)
    report = (
        "# V193 R2 source recovery continuation\n\n"
        "The first registered no-fit materialization preserved three byte-identical originals; two JSON sources failed the local source SHA check and were not copied. This continuation retries only those two originals through their bound Drive IDs and records the exact expected and actual size/SHA results.\n\n"
        f"- Continuation files verified: {public_aggregate['continuation_verified_files']} of 2.\n"
        "- The original three-file inventory and prior terminal receipt remain unchanged.\n"
        "- Native V193 source identity and trainable coverage remain CANNOT_CONFIRM.\n"
        "- Formula text, raw market rows, labels, predictions, weights, and OHLCV values were not read.\n"
        "- Fit calls, backtests, predictions, label reads, and weight reads: all 0.\n"
        "- Research-only flags remain research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.\n"
        "\nThe safe aggregate contains file identities and byte counts only; no private Drive IDs or account-local source paths.\n"
    )
    public_report_path.write_text(report, encoding="utf-8", newline="\n")
    with public_report_path.open("rb") as handle:
        os.fsync(handle.fileno())
    import shutil
    shutil.copyfile(RUNNER, public_source_path)
    with public_source_path.open("rb") as handle:
        os.fsync(handle.fileno())
    dependency = (
        "# R2 source continuation dependencies\n\n"
        "This source depends on the existing project OfficialRegistryAdapter and ExecutionGuard at `workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity`. The public source is not standalone; execution also requires the private frozen contract and input binding.\n"
    )
    public_dependency_path.write_text(dependency, encoding="utf-8", newline="\n")
    with public_dependency_path.open("rb") as handle:
        os.fsync(handle.fileno())

    artifact_paths = [
        RUNNER, CONTRACT, BINDING, INTENT, START_RECEIPT, receipt_path,
        BATCH / "private/TERMINAL_RECEIPT.json",
        BATCH / "tables/01_materialization_inventory.json",
        BATCH / "tables/02_s0_identity_dependency.json",
        BATCH / "tables/03_annual_panel_coverage.json",
        *(BATCH / source["target_relative_path"] for source in binding["sources"] if (BATCH / source["target_relative_path"]).is_file()),
        public_aggregate_path, public_report_path, public_source_path, public_dependency_path,
    ]
    adapter, guard = make_adapter([path for path in artifact_paths if path.is_file()])
    from execution_guard_core import ExecutionClaim, NO_FIT_ENGINEERING_KIND
    claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, NO_FIT_ENGINEERING_KIND, 0)
    guard.append_no_fit_terminal(claim, succeeded=all_match, error_type=None if all_match else "CANNOT_CONFIRM_EXPECTED_BYTES")
    terminals = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == execution_id and event.get("event_type") == "TERMINAL"]
    if len(terminals) != 1:
        raise RuntimeError("continuation terminal readback not exactly one")
    reg = registry_summary()
    write_exclusive(PRIVATE / "TERMINAL_RECEIPT.json", {
        "schema_version": "R2_SOURCE_CONTINUATION_TERMINAL_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "source_inventory": inventory,
        "terminal_event": terminals[0],
        "registry_summary_after_terminal": reg,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
    })

    index_path = BATCH / "EVIDENCE_INDEX.json"
    index_before = contract.get("evidence_index_sha256_before")
    if index_path.is_file() and sha256_file(index_path) == index_before:
        index = read_json(index_path)
        index["registry"] = reg
        index["route_event_execution_correction_counts"] = {"global": reg["global"], "track": reg["track"], "route": reg["route"]}
        index["continuation"] = {
            "classification": "NOFIT_SOURCE_RECOVERY_CONTINUATION",
            "route_id": ROUTE_ID,
            "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha,
            "terminal_receipt": "continuation_r1/private/TERMINAL_RECEIPT.json",
            "inventory": "continuation_r1/private/FETCH_READBACK.json",
            "verified_sources": sum(row["status"] == "MATERIALIZED_BYTE_IDENTICAL" for row in inventory),
            "expected_sources": len(inventory),
        }
        temp_index = index_path.with_suffix(".json.tmp")
        temp_index.write_bytes(json_bytes(index))
        with temp_index.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_index, index_path)
        index_sync_status = "UPDATED_IF_BASELINE_UNCHANGED"
    else:
        index_sync_status = "CANNOT_CONFIRM_INDEX_CHANGED; original EVIDENCE_INDEX preserved"

    root_append = (
        "\n\n## 2026-10-18 V193 R2 missing-source continuation\n"
        f"- Continuation route {ROUTE_ID}: verified {sum(row['status'] == 'MATERIALIZED_BYTE_IDENTICAL' for row in inventory)}/{len(inventory)} previously missing small JSON originals; original R2 inventory and terminal remain unchanged.\n"
        f"- Registry SHA-256 {reg['registry_sha256']}; V193 track route/event/execution/correction/open={reg['track']['route_count']}/{reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['correction_count']}/{reg['track']['open_execution_count']}. Native S0 identity and trainable coverage remain CANNOT_CONFIRM.\n"
        f"- Continuation evidence: {BATCH_REL}/continuation_r1/private/TERMINAL_RECEIPT.json; no-fit, no label/weight/value reads.\n"
    )
    append_fsync(ROOT / "CURRENT_STATE.md", root_append)
    append_fsync(ROOT / "HK_OHLCV_RESEARCH_STATE.md", (
        "\n\n## 2026-10-18 global registry sync note for V193 R2 source continuation\n"
        f"- Registry SHA-256 {reg['registry_sha256']}; global routes/events/executions/corrections={reg['global']['route_count']}/{reg['global']['event_count']}/{reg['global']['execution_count']}/{reg['global']['correction_count']}. A-share input recovery only; no HK track state or coverage change.\n"
    ))
    append_fsync(ROOT / "PROGRESS.md", (
        "\n\n## 2026-10-18 V193 R2 missing-source continuation\n"
        f"- Registered continuation verified {sum(row['status'] == 'MATERIALIZED_BYTE_IDENTICAL' for row in inventory)}/{len(inventory)} previously missing JSON originals. Registry SHA-256 {reg['registry_sha256']}; V193 track events/executions/open={reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['open_execution_count']}. Native identity/trainable coverage remain CANNOT_CONFIRM.\n"
    ))
    sync = {
        "schema_version": "R2_SOURCE_CONTINUATION_SYNC_V1",
        "registry": registry_summary(),
        "index_sync_status": index_sync_status,
        "root_file_sha256": {name: sha256_file(ROOT / name) for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md")},
        "root_append_status": "APPENDED_FSYNC",
    }
    write_exclusive(PRIVATE / "REGISTRY_SYNC_RECEIPT.json", sync)
    write_exclusive(PRIVATE / "POST_SYNC_READBACK.json", {
        "registry": registry_summary(),
        "terminal_receipt_sha256": sha256_file(PRIVATE / "TERMINAL_RECEIPT.json"),
        "fetch_readback_sha256": sha256_file(receipt_path),
        "root_file_sha256": sync["root_file_sha256"],
        "index_sync_status": index_sync_status,
    })
    print(json.dumps({"phase": "finish", "execution_id": execution_id, "verified": sum(row["status"] == "MATERIALIZED_BYTE_IDENTICAL" for row in inventory), "expected": len(inventory), "terminalized": True, "open_execution_count": reg["route"]["open_execution_count"], "registry_sha256": reg["registry_sha256"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "finish"))
    args = parser.parse_args()
    start() if args.phase == "start" else finish()


if __name__ == "__main__":
    main()
