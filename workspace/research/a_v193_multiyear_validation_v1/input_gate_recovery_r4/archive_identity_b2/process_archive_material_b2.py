"""R4 B2: inspect only the already-captured archive members, never the ZIP.

The original archive analyzer remains immutable. This bounded adapter supplies
the field name expected by its report builder and feeds that builder one in-
memory snapshot per captured file, so every on-disk member is read once.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[5]
WORK_DIR = Path(__file__).resolve().parent
BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/archive_identity_b2")
BATCH = ROOT / BATCH_REL
PRIVATE = BATCH / "private"
PUBLIC = BATCH / "public_export"
CONTRACT = PRIVATE / "CONTRACT.json"
BINDING = PRIVATE / "INPUT_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
MATERIAL_REPORT = PRIVATE / "MATERIAL_ASSESSMENT_B2.json"
DIAGNOSTIC_RECEIPT = PRIVATE / "DIAGNOSTIC_RECEIPT.json"
TERMINAL_RECEIPT = PRIVATE / "TERMINAL_RECEIPT.json"
PREFLIGHT_FAILURE = PRIVATE / "PREFLIGHT_FAILURE.json"
REGISTRY_SYNC_RECEIPT = PRIVATE / "REGISTRY_SYNC_RECEIPT.json"
SAFE_AGGREGATE = PUBLIC / "SAFE_AGGREGATE.json"
SOURCE_ALLOWLIST = PUBLIC / "SOURCE_ALLOWLIST.json"
EVIDENCE_INDEX = BATCH / "EVIDENCE_INDEX.json"
RUNNER = Path(__file__).resolve()
TEST = RUNNER.with_name("test_archive_material_b2.py")
ANALYZER = WORK_DIR.parent / "archive_identity_b" / "extract_archive_identity.py"
REGISTRY = ROOT / "contracts" / "EXPERIMENT_REGISTRY.jsonl"
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"

B1_BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/archive_identity_b")
B1_PRIVATE = ROOT / B1_BATCH_REL / "private"
SOURCE_BINDING = B1_PRIVATE / "ARCHIVE_SOURCE_BINDING.json"
MEMBER_INVENTORY = B1_PRIVATE / "MEMBER_INVENTORY.json"
CAPTURE_DIR = B1_PRIVATE / "captures"
A2_BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/a2")
A2_DIAGNOSTIC = ROOT / A2_BATCH_REL / "private/DIAGNOSTIC_RECEIPT.json"
A2_SAFE_AGGREGATE = ROOT / A2_BATCH_REL / "public_export/SAFE_AGGREGATE.json"
A2_DIAGNOSTIC_SHA256 = "14137412bbfb1833db794eb4303e92490e69b330b3b6bd271c4dba348bb55642"
A2_SAFE_AGGREGATE_SHA256 = "a271dd8ce7cec6aa7c139d818c6d2442ab79ecb3122b6609776c0428d75c4c5c"

TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
ROUTE_ID = "A_V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_20261018"
EXPERIMENT_ID = ROUTE_ID
ARM = "R4_REUSE_19_CAPTURED_MEMBERS_STATIC_MATERIAL_REVIEW_NOFIT"
NO_FIT_KIND = "NO_FIT_ENGINEERING_CANARY"
PROFILE = {
    "track_id": TRACK_ID,
    "route_id": ROUTE_ID,
    "registration_mode": "LIVE_EXECUTION",
    "start_stage": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_STARTED",
    "terminal_stage": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_TERMINAL",
    "occurred_at_reason": "LIVE_R4_CAPTURED_MEMBER_STATIC_REVIEW_UTC",
    "baseline_id": "R4_B1_CAPTURE_REUSE_STATIC_MATERIAL_ASSESSMENT",
    "one_variable": "REPAIR_CAPTURE_MEMBER_FIELD_BINDING_AND_REUSE_EXISTING_CAPTURED_BYTES",
    "signal_scope": "NO_SIGNAL_SCAN;CAPTURED_SOURCE_METADATA_ONLY",
    "target_scope": "NO_TARGET_OR_LABEL_SCAN",
    "device": "CPU_STDLIB_STATIC_TEXT_AND_AST_SUMMARY",
    "evidence_level": "ENGINEERING_CANARY_ONLY",
}
SAFETY_FLAGS = {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False}
REGISTRY_LOCK_PATH = REGISTRY.with_name(f".{REGISTRY.name}.lock")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def append_fsync(path: Path, text: str) -> None:
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must be a JSON object")
    return value


def load_analyzer():
    spec = importlib.util.spec_from_file_location("r4_archive_identity_analyzer_b2", ANALYZER)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the frozen B1 archive analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_member_records(captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bridge B1's original_member_name field to its report builder's member_name."""
    normalized = []
    for source in captures:
        member = dict(source)
        original_name = member.get("original_member_name")
        if not isinstance(original_name, str) or not original_name:
            raise ValueError("captured source binding is missing original_member_name")
        member["member_name"] = original_name
        normalized.append(member)
    return normalized


class _CachedCapturePath:
    """One-use in-memory view used by the frozen report builder."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.read_count = 0

    def read_bytes(self) -> bytes:
        self.read_count += 1
        if self.read_count != 1:
            raise RuntimeError("static report attempted to consume one captured payload more than once")
        return self._payload


def build_material_report_from_payloads(analyzer, captures: Sequence[Mapping[str, Any]],
                                        payloads: Mapping[str, bytes], archive_sha: str,
                                        archive_size: int) -> tuple[dict[str, Any], int]:
    """Reuse the frozen B1 report builder after binding its expected member-name key."""
    normalized = normalize_member_records(captures)
    cached_paths = {name: _CachedCapturePath(data) for name, data in payloads.items()}
    original_candidate_path = analyzer._candidate_capture_path

    def in_memory_path(_capture_dir: Path, filename: str):
        try:
            return cached_paths[filename]
        except KeyError as exc:
            raise RuntimeError("report requested a capture outside the frozen input binding") from exc

    analyzer._candidate_capture_path = in_memory_path
    try:
        report = analyzer.build_material_report(normalized, CAPTURE_DIR, archive_sha, archive_size)
    finally:
        analyzer._candidate_capture_path = original_candidate_path
    consumed = sum(path.read_count for path in cached_paths.values())
    if consumed != len(cached_paths) or any(path.read_count != 1 for path in cached_paths.values()):
        raise RuntimeError("frozen report builder did not consume each in-memory capture exactly once")
    return report, consumed


def _load_components():
    sys.path.insert(0, str(ADAPTER_DIR))
    from execution_guard_core import ExecutionGuard
    from registry_adapter import OfficialRegistryAdapter
    return ExecutionGuard, OfficialRegistryAdapter


def make_adapter(artifact_paths: Sequence[Path]):
    ExecutionGuard, OfficialRegistryAdapter = _load_components()
    adapter = OfficialRegistryAdapter(
        registry_path=REGISTRY, contract_path=CONTRACT, input_binding_path=BINDING,
        artifact_paths=tuple(artifact_paths), event_profile=PROFILE,
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def registry_summary(raw: bytes, events: list[dict[str, Any]]) -> dict[str, Any]:
    track = [event for event in events if event.get("track_id") == TRACK_ID]
    starts = {(event.get("route_id"), event.get("execution_id")) for event in track
              if event.get("event_type") == "RUN_STARTED"}
    terminals = {(event.get("route_id"), event.get("execution_id")) for event in track
                 if event.get("event_type") in {"TERMINAL", "RUN_TERMINATED"}}
    return {
        "registry_sha256": sha_bytes(raw), "registry_bytes": len(raw),
        "global_route_count": len({event.get("route_id") for event in events}),
        "global_event_count": len(events),
        "global_execution_count": len({event.get("execution_id") for event in events}),
        "global_correction_count": sum(event.get("event_type") == "CORRECTION" for event in events),
        "track_id": TRACK_ID, "track_route_count": len({event.get("route_id") for event in track}),
        "track_event_count": len(track), "track_execution_count": len({event.get("execution_id") for event in track}),
        "track_correction_count": sum(event.get("event_type") == "CORRECTION" for event in track),
        "track_open_execution_count": len(starts - terminals),
        "global_unregistered_route_count": None, "global_contract_only_count": None,
        "global_inventory_status": "CANNOT_CONFIRM_NOT_AUDITED_IN_THIS_SCOPE",
        "full_legacy_schema_status": "NOT_CLAIMED",
    }


def read_validated_registry(adapter):
    with adapter._locked():
        raw, events = adapter._read_validated()
    return raw, events, registry_summary(raw, events)


def verify_contract_binding() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    contract = load_json_object(CONTRACT)
    binding = load_json_object(BINDING)
    contract_sha, binding_sha = sha_file(CONTRACT), sha_file(BINDING)
    if contract.get("schema_version") != "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_CONTRACT_V1":
        raise RuntimeError("B2 contract schema mismatch")
    if binding.get("schema_version") != "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_INPUT_BINDING_V1":
        raise RuntimeError("B2 input binding schema mismatch")
    if contract.get("runner_sha256") != sha_file(RUNNER) or contract.get("test_sha256") != sha_file(TEST):
        raise RuntimeError("B2 runner/test SHA mismatch")
    if contract.get("analyzer_path") != ANALYZER.relative_to(ROOT).as_posix() or contract.get("analyzer_sha256") != sha_file(ANALYZER):
        raise RuntimeError("frozen B1 analyzer identity changed")
    if binding.get("contract_sha256") != contract_sha or binding.get("runner_sha256") != sha_file(RUNNER):
        raise RuntimeError("B2 contract/input binding/runner identity mismatch")
    if binding.get("test_sha256") != sha_file(TEST) or binding.get("analyzer_sha256") != sha_file(ANALYZER):
        raise RuntimeError("B2 input binding code identity mismatch")
    if contract.get("safety_flags") != SAFETY_FLAGS or binding.get("safety_flags") != SAFETY_FLAGS:
        raise RuntimeError("B2 safety flags mismatch")
    for field in ("source_binding_sha256", "member_inventory_sha256", "expected_registry_before_sha256",
                  "expected_member_count", "expected_archive_sha256", "expected_archive_size_bytes"):
        binding_field = {
            "source_binding_sha256": "source_binding_sha256",
            "member_inventory_sha256": "member_inventory_sha256",
            "expected_registry_before_sha256": "expected_registry_before_sha256",
            "expected_member_count": "member_count",
            "expected_archive_sha256": "archive_raw_sha256",
            "expected_archive_size_bytes": "archive_size_bytes",
        }[field]
        if contract.get(field) != binding.get(binding_field):
            raise RuntimeError(f"B2 contract and input binding differ for {field}")
    if binding.get("phase_a_diagnostic_sha256") != A2_DIAGNOSTIC_SHA256 or sha_file(A2_DIAGNOSTIC) != A2_DIAGNOSTIC_SHA256:
        raise RuntimeError("A2 diagnostic receipt hash differs from the frozen B2 binding")
    if binding.get("phase_a_safe_aggregate_sha256") != A2_SAFE_AGGREGATE_SHA256 or sha_file(A2_SAFE_AGGREGATE) != A2_SAFE_AGGREGATE_SHA256:
        raise RuntimeError("A2 safe aggregate hash differs from the frozen B2 binding")
    if contract.get("phase_a_diagnostic_sha256") != A2_DIAGNOSTIC_SHA256 or contract.get("phase_a_safe_aggregate_sha256") != A2_SAFE_AGGREGATE_SHA256:
        raise RuntimeError("B2 contract does not freeze the A2 diagnostic receipt identities")
    phase_a = load_json_object(A2_DIAGNOSTIC)
    if (phase_a.get("status") != "MISSING_CONSTANT_REPRODUCER_MATCHES_2_OF_2"
            or phase_a.get("source_count") != 2
            or phase_a.get("corrected_missing_constant_reproducer_matches_historical_expected_count") != 2
            or phase_a.get("raw_matches_historical_expected_count") != 0
            or phase_a.get("current_capture_sha_matches_count") != 2):
        raise RuntimeError("A2 SHA diagnosis does not satisfy the bound 2/2 result")
    source_binding_bytes = SOURCE_BINDING.read_bytes()
    inventory_bytes = MEMBER_INVENTORY.read_bytes()
    if sha_bytes(source_binding_bytes) != binding.get("source_binding_sha256"):
        raise RuntimeError("B1 archive source binding SHA mismatch")
    if sha_bytes(inventory_bytes) != binding.get("member_inventory_sha256"):
        raise RuntimeError("B1 member inventory SHA mismatch")
    source_binding = json.loads(source_binding_bytes)
    inventory = json.loads(inventory_bytes)
    if not isinstance(source_binding, dict) or not isinstance(inventory, dict):
        raise RuntimeError("B1 source binding/inventory must be JSON objects")
    if source_binding.get("archive_source_path") != binding.get("archive_source_path"):
        raise RuntimeError("B2 source binding archive identity differs")
    if source_binding.get("archive_raw_sha256") != binding.get("archive_raw_sha256"):
        raise RuntimeError("B2 current archive SHA differs from B1 binding")
    if source_binding.get("archive_size_bytes") != binding.get("archive_size_bytes"):
        raise RuntimeError("B2 current archive size differs from B1 binding")
    if source_binding.get("member_inventory_sha256") != inventory.get("inventory_sha256"):
        raise RuntimeError("B1 source binding and inventory digest differ")
    captures = source_binding.get("members")
    inventory_members = inventory.get("members")
    if not isinstance(captures, list) or len(captures) != 19 or not isinstance(inventory_members, list):
        raise RuntimeError("B1 member input inventory shape/count mismatch")
    if binding.get("members") != captures:
        raise RuntimeError("B2 member binding differs from the immutable B1 source binding")
    captured_inventory = [member for member in inventory_members if member.get("status") == "CAPTURED"]
    if len(captured_inventory) != len(captures):
        raise RuntimeError("B1 captured member counts disagree")
    by_ordinal = {member.get("ordinal"): member for member in captured_inventory}
    for member in captures:
        inventory_member = by_ordinal.get(member.get("ordinal"))
        if inventory_member is None:
            raise RuntimeError("B1 source binding member has no inventory row")
        for field in ("captured_sha256", "captured_bytes", "safe_basename", "category", "central_directory_crc32"):
            if member.get(field) != inventory_member.get(field):
                raise RuntimeError("B1 source binding and inventory member differ")
        basename = member.get("safe_basename")
        if not isinstance(basename, str) or PurePosixPath(basename).name != basename or "/" in basename or "\\" in basename:
            raise RuntimeError("captured member basename is unsafe")
        stat_result = (CAPTURE_DIR / basename).lstat()
        if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size != member.get("captured_bytes"):
            raise RuntimeError("captured member size/type differs from binding")
    _, _ = _load_components()
    adapter, guard = make_adapter((RUNNER, TEST, ANALYZER, CONTRACT, BINDING))
    with adapter._locked():
        raw, events = adapter._read_validated()
    summary = registry_summary(raw, events)
    if summary["registry_sha256"] != contract.get("expected_registry_before_sha256"):
        raise RuntimeError("registry baseline changed before B2 start")
    if summary["track_open_execution_count"] != 0:
        raise RuntimeError("V193 track has an open execution")
    if any(event.get("route_id") == ROUTE_ID or event.get("experiment_id") == EXPERIMENT_ID for event in events):
        raise RuntimeError("B2 route already has registry history")
    if guard.accounting().run_started_count or guard.accounting().terminal_count:
        raise RuntimeError("B2 no-fit guard has prior route history")
    return contract, binding, source_binding, inventory, phase_a, contract_sha, binding_sha


def _append_prestart_failure(execution_id: str, error_type: str, contract_sha: str, binding_sha: str) -> dict[str, Any]:
    ExecutionGuard, OfficialRegistryAdapter = _load_components()
    failure = {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_PREFLIGHT_FAILURE_V1",
        "route_id": ROUTE_ID, "execution_id": execution_id, "track_id": TRACK_ID,
        "event_type": "RUN_TERMINATED", "registration_mode": "BACKFILLED_FROM_ARTIFACT",
        "occurred_at": None,
        "occurred_at_null_reason": "No durable RUN_STARTED occurred; failure time is not inferred.",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "recorded_at_is_not_execution_time": True,
        "error_type": error_type, "captured_member_reads": 0, "zip_reads": 0,
        "fit_calls": 0, "backtests": 0, "predictions": 0, "market_rows_read": 0,
        "labels_read": 0, "weights_read": 0, "safety_flags": SAFETY_FLAGS,
    }
    write_json_exclusive(PREFLIGHT_FAILURE, failure)
    artifacts = (RUNNER, TEST, ANALYZER, CONTRACT, BINDING, PREFLIGHT_FAILURE)
    adapter = OfficialRegistryAdapter(registry_path=REGISTRY, contract_path=CONTRACT,
        input_binding_path=BINDING, artifact_paths=artifacts, event_profile=PROFILE)
    spec = {
        "schema_version": "1.0", "registry_event_id": f"{execution_id}_RUN_TERMINATED",
        "experiment_id": EXPERIMENT_ID, "arm": ARM, "registration_mode": "BACKFILLED_FROM_ARTIFACT",
        "route_id": ROUTE_ID, "execution_id": execution_id, "track_id": TRACK_ID,
        "stage": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_PREFLIGHT_FAILURE", "event_type": "RUN_TERMINATED",
        "occurred_at": None, "occurred_at_reason": failure["occurred_at_null_reason"],
        "contract_path": str(CONTRACT), "artifact_paths": [str(path) for path in artifacts],
        "input_paths": [str(BINDING)], "input_sha256": binding_sha, "input_binding_sha256": binding_sha,
        "code_sha256": sha_file(RUNNER), "config_sha256": contract_sha,
        "baseline_id": PROFILE["baseline_id"], "one_variable": PROFILE["one_variable"],
        "signal_date_range": {"count": 0, "scope": PROFILE["signal_scope"]},
        "target_date_range": {"count": 0, "scope": PROFILE["target_scope"]},
        "truth_access": "NOT_READ", "sealed_status": "NOT_ENTERED",
        "runtime": {"python": str(Path(sys.executable).resolve())}, "device": PROFILE["device"],
        "checkpoint_ref": None,
        "deterministic_metrics": {"truth_positive_denominator": None, "selection_slots": None,
            "execution_kind": NO_FIT_KIND, "planned_real_fit_count": 0, "real_fit_budget_debit": 0,
            "real_fit_returned_count": 0, "fit_call_entered": False},
        "decision": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "stop_reason": error_type,
        "evidence_level": "ENGINEERING_CANARY_ONLY",
        "input_identity": {"experiment_id": EXPERIMENT_ID, "arm": ARM, "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha, "null_reasons": {}},
        **SAFETY_FLAGS,
    }
    with adapter._locked():
        raw, events = adapter._read_validated()
        if any(event.get("execution_id") == execution_id for event in events):
            raise RuntimeError("B2 preflight failure execution already exists")
        adapter._append_unlocked(raw, events, spec)
    with adapter._locked():
        _, events = adapter._read_validated()
    return next(event for event in reversed(events)
                if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_TERMINATED")


def _read_capture_payloads_once(captures: Sequence[Mapping[str, Any]], read_state: dict[str, int] | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
    state = read_state if read_state is not None else {"count": 0, "bytes": 0}
    payloads: dict[str, bytes] = {}
    total = 0
    for member in captures:
        basename = str(member["safe_basename"])
        payload = (CAPTURE_DIR / basename).read_bytes()
        state["count"] += 1
        state["bytes"] += len(payload)
        if len(payload) != member["captured_bytes"] or sha_bytes(payload) != member["captured_sha256"]:
            raise RuntimeError("captured member readback does not match the frozen binding")
        payloads[basename] = payload
        total += len(payload)
    if len(payloads) != len(captures):
        raise RuntimeError("captured member read count was not exactly the bound count")
    return payloads, {"disk_member_read_count": len(payloads), "disk_member_bytes_read": total,
                      "all_member_sha256_and_size_match": True}


def add_scoped_lineage_findings(report: dict[str, Any], captures: Sequence[Mapping[str, Any]],
                                inventory: Mapping[str, Any], binding: Mapping[str, Any],
                                phase_a: Mapping[str, Any]) -> None:
    by_name = {member.get("original_member_name"): member for member in captures}
    orig = by_name.get("formulas/V193_ORIG_SNAPSHOT_UTF8.txt", {})
    fixed = by_name.get("formulas/V193_FIXED_SNAPSHOT_UTF8.txt", {})
    report["scope"] = "CAPTURED_MEMBER_STATIC_IDENTITY_AND_DEPENDENCY_ONLY; NO_ZIP_PANEL_PIT_OR_SELECTOR_VALUE_READ"
    report["source_identity_review"] = {
        "current_zip_identity": {
            "raw_sha256": binding.get("archive_raw_sha256"),
            "size_bytes": binding.get("archive_size_bytes"),
            "historical_zip_sha256": None,
            "historical_zip_identity": "CANNOT_CONFIRM_NO_PRIOR_ZIP_SHA_WAS_SUPPLIED",
        },
        "v193_orig_snapshot": {
            "captured_raw_sha256": orig.get("captured_sha256"),
            "captured_bytes": orig.get("captured_bytes"),
            "r2_s0_cand3_byte_comparison": "MATCH_REPORTED_BY_INDEPENDENT_LINEAGE_REVIEW; CURRENT_B2_BINDS_CAPTURED_BYTES",
            "comparison_source_sha256": orig.get("captured_sha256"),
            "native_formula_engine_identity": "NOT_PROVEN",
        },
        "v193_fixed_snapshot": {
            "captured_raw_sha256": fixed.get("captured_sha256"),
            "captured_bytes": fixed.get("captured_bytes"),
            "status": "CAPTURED_TEXT_VARIANT; FORMULA_ENGINE_PARITY_NOT_RUN",
        },
        "archive_assertions": {
            "embedded_verification_status_fields_are_unverified": True,
            "current_zip_does_not_authenticate_a_historical_zip_identity": True,
        },
    }
    report["phase_a_sha_diagnostic_reuse"] = {
        "status": "A2_EXACT_MISSING_SHA256_ROUND_CONSTANT_REPRODUCER_MATCHED_BOTH_HISTORICAL_DIGESTS",
        "source_content_reread_by_b2": False,
        "current_raw_sha_binding_remains_authoritative_for_the_captured_bytes": True,
        "old_receiver_implementation_identity": "CANNOT_CONFIRM_NOT_AUTHENTICATED_BY_REPRODUCTION_ALONE",
        "meaning": "The two historical digest mismatches are explained by the exact faulty reproducer; this does not authenticate an old receiver deployment or establish native formula identity.",
        "evidence_reference": {"path": A2_DIAGNOSTIC.relative_to(ROOT).as_posix(),
                                "sha256": A2_DIAGNOSTIC_SHA256,
                                "safe_aggregate_path": A2_SAFE_AGGREGATE.relative_to(ROOT).as_posix(),
                                "safe_aggregate_sha256": A2_SAFE_AGGREGATE_SHA256},
        "source_results": [{
            "logical_name": item.get("logical_name"),
            "current_raw_sha256": item.get("raw_bytes_sha256"),
            "historical_expected_sha256": item.get("historical_expected_sha256"),
            "raw_matches_historical_expected": item.get("raw_matches_historical_expected_sha"),
            "current_capture_matches": item.get("raw_matches_current_capture_sha"),
            "fault_reproducer_matches_historical_expected": item.get("reproducer_matches_historical_expected_sha"),
            "bytes": item.get("size_bytes"),
        } for item in phase_a.get("sources", [])],
        "source_count": phase_a.get("source_count"),
        "raw_historical_match_count": phase_a.get("raw_matches_historical_expected_count"),
        "fault_reproducer_historical_match_count": phase_a.get("corrected_missing_constant_reproducer_matches_historical_expected_count"),
        "current_capture_match_count": phase_a.get("current_capture_sha_matches_count"),
    }
    report["dependency_review"] = {
        "provenance": "READ_ONLY_LINEAGE_REVIEW_FORWARDED_BY_PARENT; NO_LOCAL_RECEIPT_PATH_OR_SHA_WAS_SUPPLIED; NO_RUNTIME_EXECUTION",
        "s0_environment_names_observed": ["OHLCV", "AMOUNT", "DRAWNULL"],
        "formula_name_environment_observed": [],
        "INDEXC_or_MONTH_bound_as_input": False,
        "INDEXC_or_MONTH_binding_status": "NOT_BOUND_IN_CAPTURED_STATIC_ENVIRONMENT",
        "eppy_role": "STEP1_LEGACY_COMPARISON_HELPER; NOT_PRESENT_IN_CAPTURED_FORMULA_TEXT",
        "eppy_admitted_as_s0_dependency": False,
        "bkexist_role": "INTERNAL_BOOLEAN_OR_DERIVATION; NOT_AN_EXTERNAL_INPUT",
        "actual_input_availability_or_parity": "NOT_CHECKED",
        "pit_status": "CANNOT_CONFIRM_REQUIRED_SOURCE_PUBLICATION_REVISION_VINTAGE_AND_ADJUSTMENT_ASOF_METADATA_NOT_BOUND",
        "selector_2022_status": "CANNOT_CONFIRM_PHYSICAL_HISTORY_EXISTS_BUT_INCREMENTAL_SELECTOR_MECHANISM_DOES_NOT_FILL_OLDER_YEARS",
        "fit_gate": "CLOSED_UNTIL_NATIVE_IDENTITY_SELECTOR_2022_AND_PIT_INPUT_BINDINGS_ARE_RESOLVED",
    }
    report["referenced_metadata_presence"] = {
        "formula_identity_and_canary_audit_json": "NOT_IN_CURRENT_25_MEMBER_ARCHIVE_INVENTORY; OUTSIDE_SCOPE_EXISTENCE_NOT_INFERRED",
        "environment_json": "PRESENT_IN_INVENTORY_BUT_REJECTED_BY_IDENTITY_JSON_BASENAME_ALLOWLIST; CONTENT_NOT_READ",
        "native_generation_or_direct_pit_receipt": "NO_DIRECT_POINTER_IN_CAPTURED_BINDINGS_OR_BOUNDED_METADATA_AUDIT",
    }
    report["b1_inventory_reuse"] = {
        "inventory_sha256": inventory.get("inventory_sha256"),
        "member_count": inventory.get("member_count"),
        "captured_member_count": inventory.get("captured_member_count"),
        "rejected_member_count": inventory.get("rejected_member_count"),
        "inventory_reopened_by_b2": True,
        "original_zip_reopened_by_b2": False,
    }


def _write_root_sync(status: str, execution_id: str, diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    adapter, _ = make_adapter((RUNNER, TEST, ANALYZER, CONTRACT, BINDING, INTENT, START_RECEIPT,
                               MATERIAL_REPORT, DIAGNOSTIC_RECEIPT, TERMINAL_RECEIPT))
    raw, events, summary = read_validated_registry(adapter)
    route_events = [event for event in events if event.get("route_id") == ROUTE_ID]
    starts = sum(event.get("event_type") == "RUN_STARTED" for event in route_events)
    terminals = sum(event.get("event_type") in {"TERMINAL", "RUN_TERMINATED"} for event in route_events)
    if starts > 1 or terminals != 1:
        raise RuntimeError("B2 route lifecycle did not close exactly once")
    report = load_json_object(MATERIAL_REPORT) if MATERIAL_REPORT.is_file() else {}
    safe = {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_SAFE_AGGREGATE_V1",
        "classification": "SAFE_AGGREGATE_NO_FORMULA_TEXT_OR_PRIVATE_MEMBER_CONTENT",
        "status": status,
        "captured_member_count": diagnostic.get("captured_member_read_count", 0),
        "captured_member_bytes": diagnostic.get("captured_member_bytes_read", 0),
        "member_readback_sha256_verified": diagnostic.get("all_member_sha256_and_size_match", False),
        "current_archive_sha256": diagnostic.get("current_archive_sha256"),
        "current_archive_size_bytes": diagnostic.get("current_archive_size_bytes"),
        "historical_archive_identity": "CANNOT_CONFIRM_NO_PRIOR_ZIP_SHA_WAS_SUPPLIED",
        "v193_orig_r2_s0_cand3_identity": report.get("source_identity_review", {}).get("v193_orig_snapshot", {}).get("r2_s0_cand3_byte_comparison", "CANNOT_CONFIRM"),
        "native_formula_engine_identity": "CANNOT_CONFIRM",
        "static_dependency_status": "BOUNDED_STATIC_REFERENCES_RECORDED; ACTUAL_INPUT_PARITY_NOT_CHECKED",
        "pit_status": report.get("dependency_review", {}).get("pit_status", "CANNOT_CONFIRM"),
        "selector_2022_status": report.get("dependency_review", {}).get("selector_2022_status", "CANNOT_CONFIRM"),
        "original_zip_reopened": False,
        "panel_reread": False,
        "source_code_executed": False,
        "market_rows_labels_weights_fit_backtest_prediction_read": False,
        "registry": {key: summary[key] for key in (
            "global_route_count", "global_event_count", "global_execution_count", "global_correction_count",
            "track_route_count", "track_event_count", "track_execution_count", "track_correction_count",
            "track_open_execution_count", "global_unregistered_route_count", "global_contract_only_count",
            "global_inventory_status")},
        "safety": SAFETY_FLAGS,
    }
    write_json_exclusive(SAFE_AGGREGATE, safe)
    source_allowlist = {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_SOURCE_ALLOWLIST_V1",
        "classification": "SOURCE_CODE_DEPENDENCIES_ONLY_NO_CAPTURED_SOURCE_CONTENT",
        "implementation_files": [
            {"path": RUNNER.relative_to(ROOT).as_posix(), "sha256": sha_file(RUNNER), "role": "B2 one-read wrapper, binding validation, registration and report annotation"},
            {"path": TEST.relative_to(ROOT).as_posix(), "sha256": sha_file(TEST), "role": "synthetic regression controls; no archive source reads"},
            {"path": ANALYZER.relative_to(ROOT).as_posix(), "sha256": sha_file(ANALYZER), "role": "frozen B1 static report builder; imported by B2 without editing"},
            {"path": (ADAPTER_DIR / "execution_guard_core.py").relative_to(ROOT).as_posix(), "sha256": sha_file(ADAPTER_DIR / "execution_guard_core.py"), "role": "existing no-fit execution lifecycle"},
            {"path": (ADAPTER_DIR / "registry_adapter.py").relative_to(ROOT).as_posix(), "sha256": sha_file(ADAPTER_DIR / "registry_adapter.py"), "role": "existing official append/validate/lock lifecycle"},
        ],
        "original_zip_read": False,
        "member_source_content_published": False,
        "safety": SAFETY_FLAGS,
    }
    write_json_exclusive(SOURCE_ALLOWLIST, source_allowlist)
    note = (
        "\n\n## 2026-10-18 V193 R4 B2 captured-member static identity review\n"
        f"- Status: {status}; reused only the 19 B1 captured members and their bound inventory/source-binding metadata; the original ZIP and R2/R3 panels were not reopened. Each captured member was read once and checked against its bound size/SHA.\n"
        f"- V193 original snapshot raw SHA {diagnostic.get('v193_orig_raw_sha256')}; independent lineage review reports byte identity with the R2 S0/CAND3 candidate. That does not prove installed native formula-engine identity; the fixed snapshot remains text-only and untested for parity.\n"
        f"- The two historical JSON digest conflicts are explained by the A2 missing-round-constant reproducer (2/2 historical digests; raw bytes match current captures 2/2, raw matches old expected 0/2). This authenticates neither the old receiver call path nor native V193 identity.\n"
        f"- Static dependency observations: OHLCV/AMOUNT/DRAWNULL names only; no bound formula-name environment or INDEXC/MONTH input. EPPY is a legacy step-1 comparison helper and absent from the captured formula; BKEXIST is an internal OR-derived boolean. Actual input parity is not checked.\n"
        f"- Selector-2022: physical historical data exists, but the incremental selector mechanism does not backfill older years; completeness remains CANNOT_CONFIRM. Required native identity/PIT source publication, revision, vintage and adjustment/as-of metadata are not bound; fit remains blocked.\n"
        f"- Registry SHA-256 {summary['registry_sha256']}; global route/event/execution/correction {summary['global_route_count']}/{summary['global_event_count']}/{summary['global_execution_count']}/{summary['global_correction_count']}; V193 route/event/execution/correction/open {summary['track_route_count']}/{summary['track_event_count']}/{summary['track_execution_count']}/{summary['track_correction_count']}/{summary['track_open_execution_count']}. Full global inventory remains CANNOT_CONFIRM. Evidence index: {BATCH_REL.as_posix()}/EVIDENCE_INDEX.json.\n"
        "- Safety: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.\n"
    )
    for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md"):
        append_fsync(ROOT / name, note)
    root_hashes = {name: sha_file(ROOT / name) for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md")}
    raw_registry, _, final_summary = read_validated_registry(adapter)
    sync = {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_REGISTRY_SYNC_V1",
        "route_id": ROUTE_ID, "execution_id": execution_id,
        "route_start_count": starts, "route_terminal_count": terminals,
        "registry": final_summary, "contract_sha256": sha_file(CONTRACT),
        "input_binding_sha256": sha_file(BINDING), "runner_sha256": sha_file(RUNNER),
        "test_sha256": sha_file(TEST), "analyzer_sha256": sha_file(ANALYZER),
        "source_binding_sha256": sha_file(SOURCE_BINDING), "member_inventory_sha256": sha_file(MEMBER_INVENTORY),
        "root_file_sha256_after_append": root_hashes,
        "registry_readback_matches_summary": sha_bytes(raw_registry) == final_summary["registry_sha256"],
        "root_append_fsync": True,
    }
    write_json_exclusive(REGISTRY_SYNC_RECEIPT, sync)
    artifact_paths = (CONTRACT, BINDING, INTENT, START_RECEIPT, MATERIAL_REPORT,
                      DIAGNOSTIC_RECEIPT, TERMINAL_RECEIPT, REGISTRY_SYNC_RECEIPT,
                      SAFE_AGGREGATE, SOURCE_ALLOWLIST, RUNNER, TEST, ANALYZER)
    artifacts = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
                 for path in artifact_paths if path.is_file()]
    index = {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_EVIDENCE_INDEX_V1",
        "classification": "NAVIGATION_NOT_AUTHORITY", "batch": BATCH_REL.as_posix(), "status": status,
        "route_id": ROUTE_ID, "execution_id": execution_id,
        "original_zip_content_in_index": False, "captured_member_content_in_index": False,
        "artifacts": artifacts, "registry_summary": final_summary,
        "registry_sync_receipt": {"path": REGISTRY_SYNC_RECEIPT.relative_to(ROOT).as_posix(), "sha256": sha_file(REGISTRY_SYNC_RECEIPT)},
        "safe_aggregate": {"path": SAFE_AGGREGATE.relative_to(ROOT).as_posix(), "sha256": sha_file(SAFE_AGGREGATE)},
        "source_allowlist": {"path": SOURCE_ALLOWLIST.relative_to(ROOT).as_posix(), "sha256": sha_file(SOURCE_ALLOWLIST)},
        "safety": SAFETY_FLAGS,
    }
    write_json_exclusive(EVIDENCE_INDEX, index)
    return {"registry": final_summary, "root_file_sha256": root_hashes,
            "safe_aggregate_sha256": sha_file(SAFE_AGGREGATE),
            "source_allowlist_sha256": sha_file(SOURCE_ALLOWLIST),
            "evidence_index_sha256": sha_file(EVIDENCE_INDEX),
            "registry_sync_receipt_sha256": sha_file(REGISTRY_SYNC_RECEIPT)}


def run() -> dict[str, Any]:
    outputs = (INTENT, START_RECEIPT, MATERIAL_REPORT, DIAGNOSTIC_RECEIPT,
               PREFLIGHT_FAILURE, TERMINAL_RECEIPT, REGISTRY_SYNC_RECEIPT,
               SAFE_AGGREGATE, SOURCE_ALLOWLIST, EVIDENCE_INDEX)
    if any(path.exists() for path in outputs):
        raise FileExistsError("B2 output exists; route re-entry blocked")
    execution_id = ROUTE_ID + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    contract_sha = binding_sha = None
    try:
        contract, binding, source_binding, inventory, phase_a, contract_sha, binding_sha = verify_contract_binding()
    except Exception as exc:
        if CONTRACT.is_file() and BINDING.is_file():
            contract_sha, binding_sha = sha_file(CONTRACT), sha_file(BINDING)
            event = _append_prestart_failure(execution_id, type(exc).__name__, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {"schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False, "registry_event": event,
                "captured_member_reads": 0, "original_zip_reads": 0})
            diag = {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "captured_member_read_count": 0,
                    "captured_member_bytes_read": 0, "all_member_sha256_and_size_match": False,
                    "current_archive_sha256": None, "current_archive_size_bytes": None}
            sync = _write_root_sync("CANNOT_CONFIRM_PREFLIGHT_FAILURE", execution_id, diag)
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry": sync["registry"], "evidence_index_sha256": sync["evidence_index_sha256"]}
        raise

    adapter, guard = make_adapter((RUNNER, TEST, ANALYZER, CONTRACT, BINDING, INTENT))
    _, _, preflight_summary = read_validated_registry(adapter)
    write_json_exclusive(INTENT, {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_RUN_INTENT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "analyzer_sha256": sha_file(ANALYZER), "preflight_registry": preflight_summary,
        "source_binding_sha256": binding.get("source_binding_sha256"),
        "member_inventory_sha256": binding.get("member_inventory_sha256"),
        "bound_member_count": len(source_binding["members"]),
        "captured_member_reads_planned": len(source_binding["members"]),
        "original_zip_reads_planned": 0, "panel_rereads_planned": 0,
        "read_member_payloads_only_after_durable_RUN_STARTED": True,
        "fit_backtest_prediction_selector_pit_numeric_reads_planned": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    try:
        claim = guard.claim_no_fit(execution_id=execution_id, arm=ARM,
                                   contract_sha256=contract_sha, input_binding_sha256=binding_sha)
    except Exception as exc:
        # Do not start member reads if the official durable start was rejected.
        starts = [event for event in adapter.list_events(EXPERIMENT_ID)
                  if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
        if not starts:
            event = _append_prestart_failure(execution_id, type(exc).__name__, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {"schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False, "registry_event": event,
                "captured_member_reads": 0, "original_zip_reads": 0})
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry_event_written": True, "captured_member_reads": 0}
        raise
    starts = [event for event in adapter.list_events(EXPERIMENT_ID)
              if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
    start_ok = len(starts) == 1
    write_json_exclusive(START_RECEIPT, {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_RUN_STARTED_RECEIPT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "durable_start_readback": start_ok, "registry_event": starts[0] if start_ok else None,
        "captured_member_reads_before_start": 0, "original_zip_reads_before_start": 0,
    })

    completed = False
    error_type = None
    diagnostic: dict[str, Any] = {}
    report: dict[str, Any] = {}
    captured_reads = 0
    captured_bytes_read = 0
    read_state = {"count": 0, "bytes": 0}
    try:
        if not start_ok:
            raise RuntimeError("durable RUN_STARTED readback was not exactly one")
        payloads, read_metrics = _read_capture_payloads_once(source_binding["members"], read_state)
        captured_reads = read_metrics["disk_member_read_count"]
        captured_bytes_read = read_metrics["disk_member_bytes_read"]
        analyzer = load_analyzer()
        report, in_memory_reads = build_material_report_from_payloads(
            analyzer, source_binding["members"], payloads,
            str(source_binding["archive_raw_sha256"]), int(source_binding["archive_size_bytes"]))
        if in_memory_reads != len(source_binding["members"]):
            raise RuntimeError("static report input consumption count mismatch")
        add_scoped_lineage_findings(report, source_binding["members"], inventory, binding, phase_a)
        report.update({
            "schema_version": "V193_R4_ARCHIVE_MATERIAL_ASSESSMENT_B2_V1",
            "status": "CAPTURED_MEMBER_STATIC_REVIEW_COMPLETED_NATIVE_IDENTITY_PIT_SELECTOR_UNCONFIRMED",
            "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
            "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
            "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
            "analyzer_path": ANALYZER.relative_to(ROOT).as_posix(), "analyzer_sha256": sha_file(ANALYZER),
            "source_binding_sha256": binding.get("source_binding_sha256"),
            "member_inventory_sha256": binding.get("member_inventory_sha256"),
            "disk_member_read_count": read_metrics["disk_member_read_count"],
            "disk_member_bytes_read": read_metrics["disk_member_bytes_read"],
            "static_builder_in_memory_member_consumptions": in_memory_reads,
            "original_zip_reads": 0, "panel_rereads": 0, "source_code_executed": False,
            "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
            "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
            "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
            "safety_flags": SAFETY_FLAGS,
        })
        write_json_exclusive(MATERIAL_REPORT, report)
        completed = True
        diagnostic = {
            "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_DIAGNOSTIC_V1",
            "status": report["status"], "execution_id": execution_id, "route_id": ROUTE_ID,
            "captured_member_read_count": read_metrics["disk_member_read_count"],
            "captured_member_bytes_read": read_metrics["disk_member_bytes_read"],
            "all_member_sha256_and_size_match": True,
            "current_archive_sha256": binding["archive_raw_sha256"],
            "current_archive_size_bytes": binding["archive_size_bytes"],
            "v193_orig_raw_sha256": report["source_identity_review"]["v193_orig_snapshot"]["captured_raw_sha256"],
            "source_binding_sha256": binding["source_binding_sha256"],
            "member_inventory_sha256": binding["member_inventory_sha256"],
            "material_report_path": MATERIAL_REPORT.relative_to(ROOT).as_posix(),
            "material_report_sha256": sha_file(MATERIAL_REPORT),
            "original_zip_reads": 0, "panel_rereads": 0, "source_code_executed": False,
            "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
            "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
            "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
            "safety_flags": SAFETY_FLAGS,
        }
    except Exception as exc:
        error_type = type(exc).__name__
        diagnostic = {
            "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_DIAGNOSTIC_V1",
            "status": "CANNOT_CONFIRM", "execution_id": execution_id, "route_id": ROUTE_ID,
            "error_type": error_type, "captured_member_read_count": read_state["count"],
            "captured_member_bytes_read": read_state["bytes"], "all_member_sha256_and_size_match": False,
            "current_archive_sha256": binding.get("archive_raw_sha256"),
            "current_archive_size_bytes": binding.get("archive_size_bytes"),
            "original_zip_reads": 0, "panel_rereads": 0, "source_code_executed": False,
            "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
            "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
            "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
            "safety_flags": SAFETY_FLAGS,
        }
        if MATERIAL_REPORT.parent.exists() and not MATERIAL_REPORT.exists():
            write_json_exclusive(MATERIAL_REPORT, {**diagnostic, "partial_report": report})
    write_json_exclusive(DIAGNOSTIC_RECEIPT, diagnostic)
    terminal_adapter, terminal_guard = make_adapter((RUNNER, TEST, ANALYZER, CONTRACT, BINDING, INTENT,
        START_RECEIPT, MATERIAL_REPORT, DIAGNOSTIC_RECEIPT))
    terminal_guard.append_no_fit_terminal(claim, succeeded=completed, error_type=None if completed else error_type)
    terminal_events = [event for event in terminal_adapter.list_events(EXPERIMENT_ID)
                       if event.get("execution_id") == execution_id and event.get("event_type") == "TERMINAL"]
    if len(terminal_events) != 1:
        raise RuntimeError("B2 terminal registry readback was not exactly one")
    final_status = diagnostic.get("status", "CANNOT_CONFIRM")
    write_json_exclusive(TERMINAL_RECEIPT, {
        "schema_version": "V193_R4_ARCHIVE_MEMBER_STATIC_IDENTITY_B2_TERMINAL_V1",
        "status": final_status, "execution_id": execution_id, "route_id": ROUTE_ID,
        "track_id": TRACK_ID, "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST), "analyzer_sha256": sha_file(ANALYZER),
        "run_started": True, "terminal_event": terminal_events[0],
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT),
        "captured_member_reads": diagnostic.get("captured_member_read_count"),
        "original_zip_reads": 0, "panel_rereads": 0, "fit_calls": 0,
        "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
        "backtest_calls": 0, "prediction_calls": 0,
        "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
    })
    sync = _write_root_sync(final_status, execution_id, diagnostic)
    return {
        "status": final_status, "execution_id": execution_id,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "analyzer_sha256": sha_file(ANALYZER),
        "material_report_sha256": sha_file(MATERIAL_REPORT),
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT),
        "terminal_receipt_sha256": sha_file(TERMINAL_RECEIPT),
        "registry": sync["registry"], "registry_sync_receipt_sha256": sync["registry_sync_receipt_sha256"],
        "safe_aggregate_sha256": sync["safe_aggregate_sha256"],
        "source_allowlist_sha256": sync["source_allowlist_sha256"],
        "evidence_index_sha256": sync["evidence_index_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run",))
    args = parser.parse_args()
    if args.phase == "run":
        print(json.dumps(run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
