#!/usr/bin/env python3
"""Capture two bound R2 JSON payloads under their actual byte digests."""
from __future__ import annotations

import argparse
import base64
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
CAPTURE = BATCH / "continuation_capture_r1"
PRIVATE = CAPTURE / "private"
CONTRACT = PRIVATE / "CAPTURE_CONTRACT.json"
BINDING = PRIVATE / "CAPTURE_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
REGISTRY = ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
RUNNER = Path(__file__).resolve()
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
ROUTE_ID = "A_V193_R2_SOURCE_PAYLOAD_CAPTURE_20261018"
TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
EXPERIMENT_ID = ROUTE_ID
ARM = "R2_CAPTURE_CURRENT_JSON_BYTES_BY_ACTUAL_SHA"
MAX_SOURCE_BYTES = 16384


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
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
    if contract.get("runner_sha256") != runner_sha or binding.get("runner_sha256") != runner_sha:
        raise RuntimeError("capture runner identity mismatch")
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("capture binding contract identity mismatch")
    if contract.get("base_continuation_terminal_sha256") != sha256_file(
        BATCH / "continuation_r1/private/TERMINAL_RECEIPT.json"
    ):
        raise RuntimeError("prior continuation terminal changed")
    if len(binding.get("sources", [])) != 2:
        raise RuntimeError("capture scope must bind exactly two sources")
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
            "start_stage": "R2_CURRENT_SOURCE_CAPTURE_STARTED",
            "terminal_stage": "R2_CURRENT_SOURCE_CAPTURE_TERMINAL",
            "occurred_at_reason": "LIVE_R2_SOURCE_CAPTURE_UTC",
            "baseline_id": "V193_S0_NATIVE_IDENTITY_CANNOT_CONFIRM",
            "one_variable": "CAPTURE_RAW_JSON_BYTES_AND_COMPARE_TO_HISTORICAL_DIGESTS",
            "signal_scope": "NO_SIGNAL_SCAN",
            "target_scope": "NO_LABEL_OR_TARGET_SCAN",
            "device": "CPU_RAW_BYTE_VALIDATION",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
        },
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


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
        "global_unregistered_route_count_reason": "Not audited in this bounded capture.",
        "global_contract_only_count": None,
        "global_contract_only_count_reason": "Not audited in this bounded capture.",
    }


def append_fsync(path: Path, text: str) -> None:
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def start() -> None:
    contract, binding, contract_sha, binding_sha = frozen_identity()
    for path in (INTENT, START_RECEIPT, PRIVATE / "TERMINAL_RECEIPT.json"):
        if path.exists():
            raise FileExistsError(f"capture lifecycle path already exists: {path.name}")
    raw = REGISTRY.read_bytes()
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(event.get("route_id") == ROUTE_ID for event in events):
        raise RuntimeError("capture route already has registry history; re-entry blocked")
    execution_id = ROUTE_ID + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    intent = {
        "schema_version": "R2_CURRENT_SOURCE_CAPTURE_INTENT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "experiment_id": EXPERIMENT_ID,
        "arm": ARM,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "registry_before": {"sha256": sha256_bytes(raw), "bytes": len(raw), "event_count": len(events)},
        "source_count": 2,
        "source_reads_allowed_only_after_durable_start": True,
        "manifest_reused_from_execution_id": binding.get("manifest_reused_from_execution_id"),
        "manifest_fetches_in_this_execution": 0,
        "independent_receipt_fetches_allowed": 1,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_exclusive(INTENT, intent)
    immutable = [RUNNER, CONTRACT, BINDING, INTENT]
    adapter, guard = make_adapter(immutable)
    guard.claim_no_fit(execution_id=execution_id, arm=ARM, contract_sha256=contract_sha, input_binding_sha256=binding_sha)
    starts = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
    if len(starts) != 1:
        raise RuntimeError("capture RUN_STARTED durable readback not exactly one")
    write_exclusive(START_RECEIPT, {
        "schema_version": "R2_CURRENT_SOURCE_CAPTURE_RUN_STARTED_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha256_file(RUNNER),
        "event": starts[0],
        "durable_start_readback": True,
        "fit_calls": 0,
    })
    print(json.dumps({"phase": "start", "execution_id": execution_id, "registered": True, "source_count": 2, "fit_calls": 0}))


def capture(logical_name: str, encoded: str) -> None:
    contract, binding, contract_sha, binding_sha = frozen_identity()
    intent = read_json(INTENT)
    execution_id = intent["execution_id"]
    start_receipt = read_json(START_RECEIPT)
    if start_receipt.get("execution_id") != execution_id or start_receipt.get("contract_sha256") != contract_sha or start_receipt.get("input_binding_sha256") != binding_sha:
        raise RuntimeError("capture start receipt identity mismatch")
    sources = {item["logical_name"]: item for item in binding["sources"]}
    if logical_name not in sources:
        raise ValueError("logical source is outside the frozen capture binding")
    source = sources[logical_name]
    data = base64.b64decode(encoded, validate=True)
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("captured source byte count outside frozen small-JSON bound")
    digest = sha256_bytes(data)
    capture_dir = CAPTURE / "private/captures"
    target = capture_dir / f"{logical_name}__sha256_{digest}.json"
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError("same digest capture path contains different bytes")
    else:
        write_bytes_exclusive(target, data)
    readback = target.read_bytes()
    if readback != data or sha256_bytes(readback) != digest:
        raise RuntimeError("raw capture readback hash mismatch")
    parse_error = None
    top_level_type = None
    try:
        parsed = json.loads(readback.decode("utf-8"))
        top_level_type = type(parsed).__name__
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parse_error = type(exc).__name__
    receipt = {
        "schema_version": "R2_CURRENT_SOURCE_CAPTURE_READBACK_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "logical_name": logical_name,
        "capture_path": str(target.relative_to(BATCH)),
        "actual_bytes": len(readback),
        "actual_sha256": digest,
        "historical_expected_bytes": source["expected_bytes"],
        "historical_expected_sha256": source["expected_sha256"],
        "size_matches_historical": len(readback) == source["expected_bytes"],
        "sha256_matches_historical": digest == source["expected_sha256"],
        "raw_readback_matches_input": True,
        "json_parse_status": "PASS" if parse_error is None else "FAIL",
        "json_top_level_type": top_level_type,
        "json_error_type": parse_error,
        "source_origin": source["source_origin"],
        "fit_calls": 0,
        "labels_read": 0,
        "weights_read": 0,
    }
    receipt_path = PRIVATE / f"{logical_name}_CAPTURE_READBACK.json"
    if not receipt_path.exists():
        write_exclusive(receipt_path, receipt)
    elif read_json(receipt_path) != receipt:
        raise RuntimeError("capture readback receipt exists with different content")
    print(json.dumps({"phase":"capture", "logical_name":logical_name, "path":str(target.relative_to(BATCH)), "bytes":len(readback), "sha256":digest, "historical_sha256_match":receipt["sha256_matches_historical"], "json_parse_status":receipt["json_parse_status"]}, sort_keys=True))


def finish() -> None:
    contract, binding, contract_sha, binding_sha = frozen_identity()
    intent = read_json(INTENT)
    execution_id = intent["execution_id"]
    receipts = []
    for source in binding["sources"]:
        path = PRIVATE / f"{source['logical_name']}_CAPTURE_READBACK.json"
        receipts.append(read_json(path) if path.is_file() else None)
    actual_complete = all(item is not None and item.get("raw_readback_matches_input") is True and item.get("json_parse_status") == "PASS" for item in receipts)
    expected_identity = actual_complete and all(item.get("sha256_matches_historical") and item.get("size_matches_historical") for item in receipts if item)
    capture_index = {
        "schema_version": "R2_CURRENT_SOURCE_CAPTURE_INDEX_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "sources": receipts,
        "actual_bytes_persisted_and_readback": actual_complete,
        "historical_source_identity": "CONFIRMED" if expected_identity else "CANNOT_CONFIRM",
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
    }
    index_path = PRIVATE / "CAPTURE_INDEX.json"
    write_exclusive(index_path, capture_index)
    public_dir = CAPTURE / "public_export"
    public_dir.mkdir(parents=True, exist_ok=True)
    public_summary = {
        "schema_version": "R2_CURRENT_SOURCE_CAPTURE_SAFE_SUMMARY_V1",
        "capture_status": "RAW_BYTES_CAPTURED" if actual_complete else "CANNOT_CONFIRM_RAW_BYTES_CAPTURE",
        "historical_identity_status": "CONFIRMED" if expected_identity else "CANNOT_CONFIRM",
        "sources": [None if item is None else {key:item.get(key) for key in ("logical_name","capture_path","actual_bytes","actual_sha256","historical_expected_bytes","historical_expected_sha256","size_matches_historical","sha256_matches_historical","json_parse_status")} for item in receipts],
        "fit_calls": 0,"backtests": 0,"predictions": 0,"labels_read": 0,"weights_read": 0,
        "research_only": True,"production_allowed": False,"buy_gate_updated": False,"orders_sent": False,
    }
    summary_path = public_dir / "CURRENT_SOURCE_CAPTURE_SAFE_SUMMARY.json"
    write_exclusive(summary_path, public_summary)
    report_path = public_dir / "CURRENT_SOURCE_CAPTURE_REPORT.md"
    rows = []
    for item in receipts:
        if item is None:
            rows.append("- One bound source remains uncaptured; no retry is authorized by this capture contract.")
        else:
            rows.append(f"- `{item['logical_name']}`: {item['actual_bytes']} bytes, actual SHA-256 `{item['actual_sha256']}`, historical SHA-256 `{item['historical_expected_sha256']}`, match={str(item['sha256_matches_historical']).lower()}, JSON parse={item['json_parse_status']}; saved at `{item['capture_path']}`.")
    report = (
        "# V193 R2 current source byte capture\n\n"
        "This bounded no-fit capture stores the current raw JSON payloads under filenames containing each payload's actual SHA-256. It preserves the bytes without normalizing them and separately compares current hashes with the historical R2 expected hashes. A matching current capture does not retroactively certify the older V193 source identity.\n\n"
        + "\n".join(rows) + "\n\n"
        + f"Raw capture/readback status: {'PASS' if actual_complete else 'CANNOT_CONFIRM'}. Historical identity: {'CONFIRMED' if expected_identity else 'CANNOT_CONFIRM'}. Fit calls, backtests, predictions, label reads, and weight reads: all 0.\n"
        + "Research-only flags: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.\n"
    )
    write_bytes_exclusive(report_path, report.encode("utf-8"))
    dep_path = public_dir / "CAPTURE_DEPENDENCIES.md"
    write_bytes_exclusive(dep_path, (
        "# Capture dependencies\n\n"
        "This runner uses the project's existing OfficialRegistryAdapter and ExecutionGuard from `workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity`. It captures only the two JSON payloads bound in its private input binding.\n"
    ).encode("utf-8"))
    public_runner = public_dir / "capture_r2_payloads.py"
    write_bytes_exclusive(public_runner, RUNNER.read_bytes())
    artifact_paths = [RUNNER, CONTRACT, BINDING, INTENT, START_RECEIPT, index_path, *[PRIVATE / f"{source['logical_name']}_CAPTURE_READBACK.json" for source in binding["sources"]], *(BATCH / item["capture_path"] for item in receipts if item), summary_path, report_path, dep_path, public_runner]
    adapter, guard = make_adapter([path for path in artifact_paths if path.is_file()])
    from execution_guard_core import ExecutionClaim, NO_FIT_ENGINEERING_KIND
    claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, NO_FIT_ENGINEERING_KIND, 0)
    guard.append_no_fit_terminal(claim, succeeded=actual_complete, error_type=None if actual_complete else "CANNOT_CONFIRM_RAW_CAPTURE_OR_JSON_READBACK")
    terminals = [event for event in adapter.list_events(EXPERIMENT_ID) if event.get("execution_id") == execution_id and event.get("event_type") == "TERMINAL"]
    if len(terminals) != 1:
        raise RuntimeError("capture terminal readback not exactly one")
    reg = registry_summary()
    write_exclusive(PRIVATE / "TERMINAL_RECEIPT.json", {
        "schema_version":"R2_CURRENT_SOURCE_CAPTURE_TERMINAL_V1","execution_id":execution_id,"route_id":ROUTE_ID,
        "contract_sha256":contract_sha,"input_binding_sha256":binding_sha,"runner_sha256":sha256_file(RUNNER),
        "capture_index_sha256":sha256_file(index_path),"terminal_event":terminals[0],"registry_summary_after_terminal":reg,
        "historical_source_identity":"CONFIRMED" if expected_identity else "CANNOT_CONFIRM",
        "fit_calls":0,"backtests":0,"predictions":0,"labels_read":0,"weights_read":0,
    })
    evidence_path = BATCH / "EVIDENCE_INDEX.json"
    if sha256_file(evidence_path) == contract.get("evidence_index_sha256_before"):
        evidence = read_json(evidence_path)
        evidence["registry"] = reg
        evidence["route_event_execution_correction_counts"] = {"global":reg["global"],"track":reg["track"],"route":reg["route"]}
        evidence["payload_capture"] = {"classification":"NOFIT_CURRENT_BYTES_CAPTURE","route_id":ROUTE_ID,"contract_sha256":contract_sha,"input_binding_sha256":binding_sha,"terminal_receipt":"continuation_capture_r1/private/TERMINAL_RECEIPT.json","capture_index":"continuation_capture_r1/private/CAPTURE_INDEX.json","actual_bytes_persisted":actual_complete,"historical_identity":"CONFIRMED" if expected_identity else "CANNOT_CONFIRM"}
        tmp=evidence_path.with_suffix(".json.tmp"); tmp.write_bytes(json_bytes(evidence))
        with tmp.open("rb") as handle: os.fsync(handle.fileno())
        os.replace(tmp,evidence_path); evidence_status="UPDATED_IF_BASELINE_UNCHANGED"
    else: evidence_status="CANNOT_CONFIRM_INDEX_CHANGED; original EVIDENCE_INDEX preserved"
    root_line=(
        "\n\n## 2026-10-18 V193 R2 current-source byte capture\n"
        f"- Current payload capture route {ROUTE_ID}: raw JSON bytes were persisted under actual-SHA filenames and locally parsed/read back when available; historical SHA identity remains {'CONFIRMED' if expected_identity else 'CANNOT_CONFIRM'}. This does not change native V193 identity or trainable coverage.\n"
        f"- Registry SHA-256 {reg['registry_sha256']}; V193 track route/event/execution/correction/open={reg['track']['route_count']}/{reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['correction_count']}/{reg['track']['open_execution_count']}.\n"
        f"- Evidence: {BATCH_REL}/continuation_capture_r1/private/TERMINAL_RECEIPT.json; no fit, backtest, prediction, label, weight, or market-value reads.\n"
    )
    append_fsync(ROOT / "CURRENT_STATE.md",root_line)
    append_fsync(ROOT / "HK_OHLCV_RESEARCH_STATE.md",(
        "\n\n## 2026-10-18 global registry sync note for V193 R2 current-source capture\n"
        f"- Registry SHA-256 {reg['registry_sha256']}; global routes/events/executions/corrections={reg['global']['route_count']}/{reg['global']['event_count']}/{reg['global']['execution_count']}/{reg['global']['correction_count']}. A-share source byte capture only; no HK state or coverage change.\n"
    ))
    append_fsync(ROOT / "PROGRESS.md",(
        "\n\n## 2026-10-18 V193 R2 current-source byte capture\n"
        f"- Registered capture route {ROUTE_ID}; raw payload capture/readback={'PASS' if actual_complete else 'CANNOT_CONFIRM'}, historical identity={'CONFIRMED' if expected_identity else 'CANNOT_CONFIRM'}. Registry SHA-256 {reg['registry_sha256']}; V193 track events/executions/open={reg['track']['event_count']}/{reg['track']['execution_count']}/{reg['track']['open_execution_count']}. Native identity/trainable coverage remain CANNOT_CONFIRM.\n"
    ))
    sync={"schema_version":"R2_CURRENT_SOURCE_CAPTURE_SYNC_V1","registry":registry_summary(),"evidence_index_status":evidence_status,"root_file_sha256":{name:sha256_file(ROOT/name) for name in ("CURRENT_STATE.md","HK_OHLCV_RESEARCH_STATE.md","PROGRESS.md")},"root_append_status":"APPENDED_FSYNC"}
    write_exclusive(PRIVATE/"REGISTRY_SYNC_RECEIPT.json",sync)
    write_exclusive(PRIVATE/"POST_SYNC_READBACK.json",{"registry":registry_summary(),"terminal_receipt_sha256":sha256_file(PRIVATE/"TERMINAL_RECEIPT.json"),"capture_index_sha256":sha256_file(index_path),"root_file_sha256":sync["root_file_sha256"],"evidence_index_status":evidence_status})
    print(json.dumps({"phase":"finish","execution_id":execution_id,"actual_capture_complete":actual_complete,"historical_identity":"CONFIRMED" if expected_identity else "CANNOT_CONFIRM","open_route_execution_count":reg["route"]["open_execution_count"],"track_counts":reg["track"],"registry_sha256":reg["registry_sha256"],"evidence_index_status":evidence_status}))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("start","capture","finish"))
    parser.add_argument("logical_name",nargs="?")
    parser.add_argument("base64_payload",nargs="?")
    args=parser.parse_args()
    if args.phase=="start": start()
    elif args.phase=="capture":
        if not args.logical_name or not args.base64_payload: raise SystemExit("capture requires logical_name and base64_payload")
        capture(args.logical_name,args.base64_payload)
    else: finish()


if __name__=="__main__":
    main()
