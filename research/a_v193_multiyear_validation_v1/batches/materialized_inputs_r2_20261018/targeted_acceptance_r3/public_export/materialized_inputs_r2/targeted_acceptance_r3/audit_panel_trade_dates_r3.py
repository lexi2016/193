"""Bounded no-fit strict-calendar audit over frozen R2 panel date columns."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from calendar_dates import parse_calendar_day

ROOT = Path(__file__).resolve().parents[5]
BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3")
BATCH = ROOT / BATCH_REL
PRIVATE = BATCH / "private"
TABLES = BATCH / "tables"
CONTRACT = PRIVATE / "CONTRACT.json"
BINDING = PRIVATE / "INPUT_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
TERMINAL_RECEIPT = PRIVATE / "TERMINAL_RECEIPT.json"
RESULT_JSON = TABLES / "01_strict_calendar_date_audit.json"
REGISTRY = ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
RUNNER = Path(__file__).resolve()
HELPER = RUNNER.with_name("calendar_dates.py")
R2_BINDING = ROOT / "research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/private/R2_INPUT_BINDING.json"
R2_COVERAGE = ROOT / "research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/tables/03_annual_panel_coverage.json"
TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
ROUTE_ID = "A_V193_R2_TARGETED_DATE_TOKEN_AUDIT_R3_20261018"
EXPERIMENT_ID = ROUTE_ID
ARM = "R3_STRICT_CALENDAR_DATE_TOKEN_AUDIT_NOFIT"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def write_json_exclusive(path: Path, value: Any) -> None:
    write_exclusive(path, json_bytes(value))


def append_fsync(path: Path, text: str) -> None:
    with path.open("ab") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def registry_summary(raw: bytes) -> dict[str, Any]:
    events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    track = [event for event in events if event.get("track_id") == TRACK_ID]
    starts = {(event.get("route_id"), event.get("execution_id")) for event in track
              if event.get("event_type") == "RUN_STARTED"}
    terminals = {(event.get("route_id"), event.get("execution_id")) for event in track
                 if event.get("event_type") == "TERMINAL"}
    return {
        "registry_sha256": sha_bytes(raw),
        "track_id": TRACK_ID,
        "track_route_count": len({event.get("route_id") for event in track}),
        "track_event_count": len(track),
        "track_execution_count": len({event.get("execution_id") for event in track}),
        "track_correction_count": sum(event.get("event_type") == "CORRECTION" for event in track),
        "track_open_execution_count": len(starts - terminals),
        "global_unregistered_route_count": None,
        "global_contract_only_count": None,
        "global_inventory_status": "CANNOT_CONFIRM_NOT_AUDITED_IN_THIS_SCOPE",
    }


def sync_root_status(status: str, execution_id: str) -> dict[str, Any]:
    reg = registry_summary(REGISTRY.read_bytes())
    note = (
        f"\n\n## 2026-10-18 V193 R2 R3 strict date-token audit\n"
        f"- Status: {status}; execution `{execution_id}` read only the `trade_date` column from three frozen 20261015 panels, reused prior R2 identity receipts without rehashing panel contents, and used strict calendar-day validation. No fit, backtest, prediction, label, weight, or value field was read. V193 trainable coverage remains CANNOT_CONFIRM.\n"
        f"- Registry SHA-256 `{reg['registry_sha256']}`; V193 routes/events/executions/corrections/open: {reg['track_route_count']}/{reg['track_event_count']}/{reg['track_execution_count']}/{reg['track_correction_count']}/{reg['track_open_execution_count']}. Global unregistered-route and contract-only inventory remains null / CANNOT_CONFIRM_NOT_AUDITED.\n"
        f"- Evidence: `{BATCH_REL}/EVIDENCE_INDEX.json`.\n"
    )
    for rel in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md"):
        append_fsync(ROOT / rel, note)
    root_hashes = {rel: sha_file(ROOT / rel) for rel in
                   ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md")}
    readback = {"schema_version": "V193_R2_R3_ROOT_SYNC_READBACK_V1",
                "registry": reg, "root_file_sha256": root_hashes,
                "append_fsync": True}
    write_json_exclusive(PRIVATE / "POST_SYNC_READBACK.json", readback)
    return readback


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path.name}")
    return value


def verify_identity() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = load_json(CONTRACT)
    binding = load_json(BINDING)
    contract_sha = sha_file(CONTRACT)
    binding_sha = sha_file(BINDING)
    runner_sha = sha_file(RUNNER)
    helper_sha = sha_file(HELPER)
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("R3 binding does not match frozen contract")
    if contract.get("runner_sha256") != runner_sha or binding.get("runner_sha256") != runner_sha:
        raise RuntimeError("R3 runner SHA does not match frozen binding")
    if contract.get("calendar_helper_sha256") != helper_sha or binding.get("calendar_helper_sha256") != helper_sha:
        raise RuntimeError("strict calendar helper SHA does not match frozen binding")
    if contract.get("experiment_scope") != "NO_FIT_NO_BACKTEST_NO_PREDICTION_NO_LABEL_OR_WEIGHT_READ":
        raise RuntimeError("R3 contract scope is not the authorized no-fit scope")
    if binding.get("base_r2_input_binding_sha256") != sha_file(R2_BINDING):
        raise RuntimeError("frozen R2 source identity binding changed")
    if binding.get("base_r2_annual_coverage_sha256") != sha_file(R2_COVERAGE):
        raise RuntimeError("frozen R2 annual coverage table changed")
    old_binding = load_json(R2_BINDING)
    source_by_id = {x["source_id"]: x for x in old_binding["panel_sources"]}
    for source in binding.get("panels", []):
        original = source_by_id.get(source.get("source_id"))
        if not original:
            raise RuntimeError("R3 source does not resolve in frozen R2 binding")
        for key in ("path", "expected_sha256", "expected_bytes", "date_field", "reused_receipt_sha256"):
            if source.get(key) != original.get(key):
                raise RuntimeError(f"R3 source binding differs from frozen R2 field: {key}")
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
            "start_stage": "V193_R2_R3_STRICT_DATE_AUDIT_STARTED",
            "terminal_stage": "V193_R2_R3_STRICT_DATE_AUDIT_TERMINAL",
            "occurred_at_reason": "LIVE_R3_NOFIT_DATE_TOKEN_CHECK_UTC",
            "baseline_id": "V193_S0_NATIVE_IDENTITY_CANNOT_CONFIRM",
            "one_variable": "STRICT_CALENDAR_VALIDATION_OF_EXISTING_TRADE_DATE_TOKENS",
            "signal_scope": "NO_SIGNAL_SCAN",
            "target_scope": "NO_LABEL_OR_TARGET_SCAN",
            "device": "CPU_PYARROW_TRADE_DATE_ONLY_PROJECTION",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
        },
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def start() -> None:
    for path in (INTENT, START_RECEIPT, TERMINAL_RECEIPT, RESULT_JSON):
        if path.exists():
            raise FileExistsError(f"R3 lifecycle evidence already exists: {path.name}")
    contract, binding, contract_sha, binding_sha = verify_identity()
    before = REGISTRY.read_bytes()
    events_before = [json.loads(line) for line in before.decode("utf-8").splitlines() if line.strip()]
    if any(event.get("route_id") == ROUTE_ID for event in events_before):
        raise RuntimeError("R3 route already has registry history; re-entry is blocked")
    execution_id = "A_V193_R2_R3_DATE_AUDIT_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    intent = {
        "schema_version": "V193_R2_R3_DATE_AUDIT_INTENT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER),
        "calendar_helper_sha256": sha_file(HELPER),
        "registry_before": {"sha256": sha_bytes(before), "bytes": len(before), "event_count": len(events_before)},
        "source_reads_start_only_after_durable_RUN_STARTED": True,
        "source_content_rehash": False,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_exclusive(INTENT, intent)
    artifacts = [RUNNER, HELPER, CONTRACT, BINDING, INTENT, R2_BINDING, R2_COVERAGE]
    artifacts.extend(ROOT / item["path"] for item in binding.get("reused_receipts", []))
    adapter, guard = make_adapter(artifacts)
    claim = guard.claim_no_fit(execution_id=execution_id, arm=ARM,
                               contract_sha256=contract_sha, input_binding_sha256=binding_sha)
    matches = [event for event in adapter.list_events(EXPERIMENT_ID)
               if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
    if len(matches) != 1:
        raise RuntimeError("durable R3 RUN_STARTED readback is not exactly one event")
    write_json_exclusive(START_RECEIPT, {
        "schema_version": "V193_R2_R3_DATE_AUDIT_START_RECEIPT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER),
        "calendar_helper_sha256": sha_file(HELPER),
        "registry_event": matches[0],
        "durable_start_readback": True,
        "fit_calls": 0,
    })
    print(json.dumps({"phase": "start", "execution_id": execution_id,
                      "run_started": True, "fit_calls": 0}, sort_keys=True))


def raw_token_key(value: Any) -> tuple[str, str, str]:
    kind = type(value)
    return kind.__module__, kind.__name__, repr(value)


def inspect_panel(source: dict[str, Any]) -> dict[str, Any]:
    path = Path(source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"bound frozen panel missing: {source['source_id']}")
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if source["date_field"] not in parquet.schema_arrow.names:
        raise KeyError(f"bound date column absent: {source['source_id']}")
    seen: dict[tuple[str, str, str], Any] = {}
    null_count = 0
    for batch in parquet.iter_batches(columns=[source["date_field"]], batch_size=250_000):
        for token in batch.column(0).to_pylist():
            if token is None:
                null_count += 1
                continue
            seen.setdefault(raw_token_key(token), token)
    invalid_unique = sum(parse_calendar_day(token) is None for token in seen.values())
    return {
        "panel_id": source["source_id"],
        "input_identity_status": "FAST_BIND_PRIOR_R2_RECEIPT_NO_REHASH",
        "source_sha256_reused_from_r2": source["expected_sha256"],
        "source_bytes_reused_from_r2": source["expected_bytes"],
        "identity_receipt_sha256": source["reused_receipt_sha256"],
        "date_field": source["date_field"],
        "unique_raw_nonnull_date_token_count": len(seen),
        "date_null_row_count": null_count,
        "invalid_unique_date_token_count": invalid_unique,
        "invalid_token_values_recorded": False,
        "date_values_or_rows_persisted": False,
        "status": "PASS_INVALID_ZERO" if invalid_unique == 0 else "HOLD_INVALID_DATE_TOKENS",
    }


def audit() -> None:
    contract, binding, contract_sha, binding_sha = verify_identity()
    start_receipt = load_json(START_RECEIPT)
    execution_id = start_receipt["execution_id"]
    if start_receipt.get("durable_start_readback") is not True:
        raise RuntimeError("R3 durable start receipt missing")
    try:
        results = [inspect_panel(panel) for panel in binding["panels"]]
        invalid_total = sum(item["invalid_unique_date_token_count"] for item in results)
        summary_status = "PASS_INVALID_ZERO" if invalid_total == 0 else "HOLD_INVALID_DATE_TOKENS"
        error_type = None if invalid_total == 0 else "INVALID_DATE_TOKENS"
        completed = True
    except Exception as exc:
        results = []
        summary_status = "CANNOT_CONFIRM_PANEL_DATE_AUDIT"
        error_type = type(exc).__name__
        completed = False
        failure = {"error_type": error_type, "error": str(exc)}
        write_json_exclusive(PRIVATE / "AUDIT_ERROR.json", failure)
    result = {
        "schema_version": "V193_R2_R3_STRICT_CALENDAR_DATE_AUDIT_V1",
        "execution_id": execution_id,
        "status": summary_status,
        "as_of_snapshot": "20261015",
        "panels": results,
        "interpretation": "Invalid token count zero means the strict parser found no malformed unique date tokens among the frozen R2 date columns; it does not confirm V193 native features, PIT eligibility, signal coverage, warmup, or trainable coverage.",
        "panel_source_content_rehashed": False,
        "panel_columns_read": ["trade_date"],
        "values_or_rows_persisted": False,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
    }
    write_json_exclusive(RESULT_JSON, result)
    artifacts = [RUNNER, HELPER, CONTRACT, BINDING, INTENT, START_RECEIPT,
                 R2_BINDING, R2_COVERAGE, RESULT_JSON]
    artifacts.extend(ROOT / item["path"] for item in binding.get("reused_receipts", []))
    adapter, guard = make_adapter(artifacts)
    from execution_guard_core import ExecutionClaim, NO_FIT_ENGINEERING_KIND
    claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha,
                           NO_FIT_ENGINEERING_KIND, 0)
    guard.append_no_fit_terminal(claim, succeeded=completed and summary_status == "PASS_INVALID_ZERO",
                                 error_type=error_type)
    terminal_events = [event for event in adapter.list_events(EXPERIMENT_ID)
                       if event.get("execution_id") == execution_id and event.get("event_type") == "TERMINAL"]
    if len(terminal_events) != 1:
        raise RuntimeError("R3 terminal registry readback is not exactly one event")
    terminal = {
        "schema_version": "V193_R2_R3_DATE_AUDIT_TERMINAL_RECEIPT_V1",
        "execution_id": execution_id,
        "route_id": ROUTE_ID,
        "track_id": TRACK_ID,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER),
        "calendar_helper_sha256": sha_file(HELPER),
        "terminal_event": terminal_events[0],
        "status": summary_status,
        "fit_calls": 0,
        "backtests": 0,
        "predictions": 0,
        "labels_read": 0,
        "weights_read": 0,
        "panel_content_rehashed": False,
    }
    write_json_exclusive(TERMINAL_RECEIPT, terminal)
    post_sync = sync_root_status(summary_status, execution_id)
    print(json.dumps({"phase": "audit", "execution_id": execution_id,
                      "status": summary_status, "panels": len(results),
                      "invalid_unique_date_tokens": sum(x["invalid_unique_date_token_count"] for x in results),
                      "terminal_event_count": len(terminal_events), "fit_calls": 0,
                      "registry_sha256": post_sync["registry"]["registry_sha256"],
                      "track_open_execution_count": post_sync["registry"]["track_open_execution_count"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("start", "audit"))
    args = parser.parse_args()
    if args.phase == "start":
        start()
    else:
        audit()


if __name__ == "__main__":
    main()
