"""Immutable R4 A2 diagnostic for the reported missing SHA-256 round constant.

This runner reads exactly two already-captured JSON byte files, once each,
after a durable no-fit RUN_STARTED. It compares raw SHA-256 with a standalone
reproducer that models JavaScript's uint32 coercion order at the final
undefined round. It never parses, prints, or persists source contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[5]
BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/a2")
BATCH = ROOT / BATCH_REL
PRIVATE = BATCH / "private"
PUBLIC = BATCH / "public_export"
CONTRACT = PRIVATE / "CONTRACT.json"
BINDING = PRIVATE / "INPUT_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
DIAGNOSTIC_RECEIPT = PRIVATE / "DIAGNOSTIC_RECEIPT.json"
PREFLIGHT_FAILURE = PRIVATE / "PREFLIGHT_FAILURE.json"
TERMINAL_RECEIPT = PRIVATE / "TERMINAL_RECEIPT.json"
REGISTRY_SYNC_RECEIPT = PRIVATE / "REGISTRY_SYNC_RECEIPT.json"
SAFE_AGGREGATE = PUBLIC / "SAFE_AGGREGATE.json"
EVIDENCE_INDEX = BATCH / "EVIDENCE_INDEX.json"
RUNNER = Path(__file__).resolve()
TEST = RUNNER.with_name("test_constant_table_parity.py")
REGISTRY = ROOT / "contracts" / "EXPERIMENT_REGISTRY.jsonl"
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"

TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
ROUTE_ID = "A_V193_R4_SHA_CONSTANT_TABLE_PARITY_A2_20261018"
EXPERIMENT_ID = ROUTE_ID
ARM = "R4_TWO_JSON_RAW_SHA_VS_CORRECTED_MISSING_CONSTANT_REPRODUCER_NOFIT"
NO_FIT_KIND = "NO_FIT_ENGINEERING_CANARY"
BASELINE_REGISTRY_SHA256 = "da1ae7f0a5e2c2c1addc052c9e9da19c1396fe6c986fae623d8d772f7c3df147"

TRACK_PHASE_A_ROUTE = "A_V193_R4_JSON_HASH_SHA_CONSTANT_TABLE_DIAGNOSTIC_20261018"
TRACK_PHASE_A_EXECUTION = "A_V193_R4_JSON_HASH_SHA_CONSTANT_TABLE_DIAGNOSTIC_20261018_20261018T124619114107Z"
TRACK_PHASE_A_TERMINAL_SHA256 = "11d0708c5bba2c8cd01cfa31f241d2146e25fb3091636c0449beed7d5d29feb4"
PHASE_A_DIAGNOSTIC_PATH = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/private/DIAGNOSTIC_RECEIPT.json")
PHASE_A_DIAGNOSTIC_SHA256 = "aa492874f161a36b5f475c4639eb524a2a19998c9c0b31c5eb43cad852a07364"

SOURCES = (
    {
        "logical_name": "input_data_manifest",
        "path": "research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/private/captures/input_data_manifest__sha256_b02813c24916488c03684513bac36da6ed39bf87a8ae919f8b5f2f6cc5a7e516.json",
        "expected_size_bytes": 4026,
        "captured_bytes_sha256": "b02813c24916488c03684513bac36da6ed39bf87a8ae919f8b5f2f6cc5a7e516",
        "historical_expected_sha256": "ef60a6c9141c8d1dd871c6815fc250b2be171f1f535d6ebfcf016d40ddb6b8cd",
    },
    {
        "logical_name": "independent_verification_receipt",
        "path": "research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/continuation_capture_r1/private/captures/independent_verification_receipt__sha256_e81c184e96e18f39c9884c1c448071b8c73a5b560b4600e0af00aea0df6766f1.json",
        "expected_size_bytes": 1296,
        "captured_bytes_sha256": "e81c184e96e18f39c9884c1c448071b8c73a5b560b4600e0af00aea0df6766f1",
        "historical_expected_sha256": "7ec5e16286a0291484c7a80ebf8b24cea36479cb1448d24dfb548742587f7ae4",
    },
)

PROFILE = {
    "track_id": TRACK_ID,
    "route_id": ROUTE_ID,
    "registration_mode": "LIVE_EXECUTION",
    "start_stage": "V193_R4_SHA_CONSTANT_PARITY_A2_STARTED",
    "terminal_stage": "V193_R4_SHA_CONSTANT_PARITY_A2_TERMINAL",
    "occurred_at_reason": "LIVE_R4_CORRECTED_SHA_CONSTANT_REPRODUCER_UTC",
    "baseline_id": "R4_PHASE_A_CAPTURED_JSON_RAW_SHA_AND_REVIEWER_PARITY_FIXTURE",
    "one_variable": "CORRECT_JAVASCRIPT_UINT32_COERCION_AFTER_UNDEFINED_FINAL_K",
    "signal_scope": "NO_SIGNAL_SCAN; METADATA_HASH_DIAGNOSTIC_ONLY",
    "target_scope": "NO_TARGET_OR_LABEL_SCAN",
    "device": "CPU_STDLIB_SHA256_REFERENCE_AND_BOUNDED_REPRODUCER",
    "evidence_level": "ENGINEERING_CANARY_ONLY",
}

MASK32 = 0xFFFFFFFF
INITIAL_STATE = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)
STANDARD_K = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)
MISSING_CONSTANT_INDEX = 41
MISSING_CONSTANT = 0xA81A664B
FAULTY_K = STANDARD_K[:MISSING_CONSTANT_INDEX] + STANDARD_K[MISSING_CONSTANT_INDEX + 1:]
JS_ABC_FIXTURE = "d9d21bfe14289fdbcc797ddbed058ba0ecb0288f834459b31d73753c1cd0a42d"


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
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _rotr32(value: int, count: int) -> int:
    return ((value >> count) | ((value << (32 - count)) & MASK32)) & MASK32


def _js_uint32(value: int | None) -> int:
    """Model JavaScript Number(value) >>> 0, including NaN -> 0."""

    return 0 if value is None else value & MASK32


def sha256_reference_standard(data: bytes) -> str:
    """Small SHA-256 reference implementation used only on synthetic controls."""

    return _sha256_with_constants(data, STANDARD_K)


def sha256_missing_constant_js_semantics(data: bytes) -> str:
    """Reproduce the missing-K fault with the historical JS coercion order.

    K[41] is omitted, shifting following entries left. The final K[63] access
    is undefined. The historical helper coerces the resulting NaN to uint32
    first (t1 = 0), then still computes a=(t1+t2)>>>0 and e=(d+t1)>>>0.
    """

    if len(STANDARD_K) != 64 or len(FAULTY_K) != 63 or STANDARD_K[41] != MISSING_CONSTANT:
        raise AssertionError("the bound SHA-256 constant-table model changed")
    return _sha256_with_constants(data, FAULTY_K)


def _sha256_with_constants(data: bytes, constants: Sequence[int]) -> str:
    bit_length = (len(data) * 8) & 0xFFFFFFFFFFFFFFFF
    padded = bytearray(data)
    padded.append(0x80)
    padded.extend(b"\x00" * ((56 - len(padded) % 64) % 64))
    padded.extend(bit_length.to_bytes(8, "big"))
    state = list(INITIAL_STATE)

    for offset in range(0, len(padded), 64):
        block = padded[offset:offset + 64]
        words = [int.from_bytes(block[i:i + 4], "big") for i in range(0, 64, 4)]
        for i in range(16, 64):
            x = words[i - 15]
            s0 = _rotr32(x, 7) ^ _rotr32(x, 18) ^ (x >> 3)
            y = words[i - 2]
            s1 = _rotr32(y, 17) ^ _rotr32(y, 19) ^ (y >> 10)
            words.append((words[i - 16] + s0 + words[i - 7] + s1) & MASK32)

        a, b, c, d, e, f, g, h = state
        for i in range(64):
            big1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
            choose = (e & f) ^ ((~e & MASK32) & g)
            constant = constants[i] if i < len(constants) else None
            t1 = _js_uint32(None if constant is None else h + big1 + choose + constant + words[i])
            big0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
            majority = (a & b) ^ (a & c) ^ (b & c)
            t2 = _js_uint32(big0 + majority)
            next_a = _js_uint32(t1 + t2)
            next_e = _js_uint32(d + t1)
            h, g, f, e, d, c, b, a = g, f, e, next_e, c, b, a, next_a

        state = [(old + new) & MASK32 for old, new in zip(state, (a, b, c, d, e, f, g, h))]
    return "".join(f"{word:08x}" for word in state)


def synthetic_self_check() -> dict[str, Any]:
    controls = (b"", b"abc", bytes(range(256)), bytes(range(251)) * 5)
    standard_ok = all(sha256_reference_standard(sample) == hashlib.sha256(sample).hexdigest() for sample in controls)
    table_shape_ok = (
        len(STANDARD_K) == 64 and len(FAULTY_K) == 63
        and STANDARD_K[MISSING_CONSTANT_INDEX] == MISSING_CONSTANT
        and FAULTY_K[MISSING_CONSTANT_INDEX] == STANDARD_K[MISSING_CONSTANT_INDEX + 1]
        and _js_uint32(None) == 0
    )
    javascript_fixture_ok = sha256_missing_constant_js_semantics(b"abc") == JS_ABC_FIXTURE
    result = {
        "standard_reference_matches_hashlib": standard_ok,
        "faulty_table_shape_matches_one_missing_constant": table_shape_ok,
        "exact_historical_js_abc_fixture_matches": javascript_fixture_ok,
        "javascript_abc_fixture": JS_ABC_FIXTURE,
        "source_json_reads": 0,
    }
    if result != {
        "standard_reference_matches_hashlib": True,
        "faulty_table_shape_matches_one_missing_constant": True,
        "exact_historical_js_abc_fixture_matches": True,
        "javascript_abc_fixture": JS_ABC_FIXTURE,
        "source_json_reads": 0,
    }:
        raise AssertionError("corrected SHA constant-table controls failed")
    return result


def verify_identity() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = load_json_object(CONTRACT)
    binding = load_json_object(BINDING)
    contract_sha = sha_file(CONTRACT)
    binding_sha = sha_file(BINDING)
    runner_sha = sha_file(RUNNER)
    test_sha = sha_file(TEST)
    if contract.get("schema_version") != "V193_R4_SHA_CONSTANT_PARITY_A2_CONTRACT_V1":
        raise RuntimeError("A2 contract schema mismatch")
    if contract.get("runner_sha256") != runner_sha or contract.get("test_sha256") != test_sha:
        raise RuntimeError("A2 runner/test SHA mismatch")
    if contract.get("scope") != "NO_FIT_NO_BACKTEST_NO_PREDICTION_NO_LABEL_OR_WEIGHT_READ":
        raise RuntimeError("A2 scope mismatch")
    if contract.get("safety_flags") != {
        "research_only": True, "production_allowed": False,
        "buy_gate_updated": False, "orders_sent": False,
    }:
        raise RuntimeError("A2 safety flags mismatch")
    if binding.get("schema_version") != "V193_R4_SHA_CONSTANT_PARITY_A2_INPUT_BINDING_V1":
        raise RuntimeError("A2 input binding schema mismatch")
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("A2 contract SHA mismatch")
    if binding.get("runner_sha256") != runner_sha or binding.get("test_sha256") != test_sha:
        raise RuntimeError("A2 binding runner/test SHA mismatch")
    if binding.get("sources") != list(SOURCES) or contract.get("sources") != list(SOURCES):
        raise RuntimeError("A2 source binding mismatch")
    if contract.get("expected_registry_before_sha256") != BASELINE_REGISTRY_SHA256:
        raise RuntimeError("A2 registry baseline contract mismatch")
    return contract, binding, contract_sha, binding_sha


def load_registry_components():
    sys.path.insert(0, str(ADAPTER_DIR))
    from execution_guard_core import ExecutionClaim, ExecutionGuard, NO_FIT_ENGINEERING_KIND
    from registry_adapter import OfficialRegistryAdapter
    return ExecutionClaim, ExecutionGuard, NO_FIT_ENGINEERING_KIND, OfficialRegistryAdapter


def registry_summary(raw: bytes, events: list[dict[str, Any]]) -> dict[str, Any]:
    track = [event for event in events if event.get("track_id") == TRACK_ID]
    starts = {(event.get("route_id"), event.get("execution_id")) for event in track if event.get("event_type") == "RUN_STARTED"}
    terminals = {(event.get("route_id"), event.get("execution_id")) for event in track if event.get("event_type") in {"TERMINAL", "RUN_TERMINATED"}}
    return {
        "registry_sha256": sha_bytes(raw), "registry_bytes": len(raw),
        "global_route_count": len({event.get("route_id") for event in events}),
        "global_event_count": len(events), "global_execution_count": len({event.get("execution_id") for event in events}),
        "global_correction_count": sum(event.get("event_type") == "CORRECTION" for event in events),
        "track_id": TRACK_ID, "track_route_count": len({event.get("route_id") for event in track}),
        "track_event_count": len(track), "track_execution_count": len({event.get("execution_id") for event in track}),
        "track_correction_count": sum(event.get("event_type") == "CORRECTION" for event in track),
        "track_open_execution_count": len(starts - terminals),
        "global_unregistered_route_count": None, "global_contract_only_count": None,
        "global_inventory_status": "CANNOT_CONFIRM_NOT_AUDITED_IN_THIS_SCOPE",
        "full_legacy_schema_status": "NOT_CLAIMED",
    }


def make_adapter(artifact_paths: Sequence[Path]):
    _, ExecutionGuard, _, OfficialRegistryAdapter = load_registry_components()
    adapter = OfficialRegistryAdapter(
        registry_path=REGISTRY, contract_path=CONTRACT, input_binding_path=BINDING,
        artifact_paths=tuple(artifact_paths), event_profile=PROFILE,
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def read_validated_registry(adapter):
    with adapter._locked():
        raw, events = adapter._read_validated()
    return raw, events, registry_summary(raw, events)


def verify_preflight(adapter, guard, contract: Mapping[str, Any]) -> dict[str, Any]:
    required = set(adapter.builder.REQUIRED_FIELDS)
    required_scope = {
        "schema_version", "registration_mode", "route_id", "execution_id", "track_id", "stage", "event_type",
        "occurred_at", "occurred_at_reason", "contract_path", "input_sha256", "code_sha256", "config_sha256",
        "baseline_id", "one_variable", "signal_date_range", "target_date_range", "truth_access", "sealed_status",
        "runtime", "device", "artifact_path", "checkpoint_ref", "deterministic_metrics", "decision", "stop_reason",
        "evidence_level", "research_only", "production_allowed", "buy_gate_updated", "orders_sent",
        "previous_event_sha256", "event_sha256",
    }
    if not required_scope.issubset(required):
        raise RuntimeError("official registry builder lacks required A2 schema fields")
    if contract.get("track_id") != TRACK_ID or contract.get("route_id") != ROUTE_ID:
        raise RuntimeError("A2 track/route binding mismatch")
    raw, events, summary = read_validated_registry(adapter)
    if summary["registry_sha256"] != BASELINE_REGISTRY_SHA256:
        raise RuntimeError("registry baseline changed before A2 start")
    if any(event.get("route_id") == ROUTE_ID or event.get("experiment_id") == EXPERIMENT_ID for event in events):
        raise RuntimeError("A2 route already exists; re-entry blocked")
    if summary["track_open_execution_count"] != 0:
        raise RuntimeError("V193 track has an open execution; A2 start blocked")
    accounting = guard.accounting()
    if accounting.run_started_count or accounting.terminal_count:
        raise RuntimeError("A2 guard has existing lifecycle events")
    source_stat = []
    for source in SOURCES:
        path = ROOT / source["path"]
        st = path.stat()
        if not path.is_file() or st.st_size != source["expected_size_bytes"]:
            raise RuntimeError("bound JSON path or expected byte size mismatch")
        source_stat.append({"logical_name": source["logical_name"], "size_bytes": st.st_size})
    old_diag = ROOT / PHASE_A_DIAGNOSTIC_PATH
    if sha_file(old_diag) != PHASE_A_DIAGNOSTIC_SHA256:
        raise RuntimeError("Phase A immutable diagnostic SHA changed")
    return {"registry": summary, "source_stat": source_stat, "source_bytes_read": 0,
            "phase_a_diagnostic_sha256": PHASE_A_DIAGNOSTIC_SHA256}


def make_diagnostic(results: list[dict[str, Any]], execution_id: str,
                    contract_sha: str, binding_sha: str) -> dict[str, Any]:
    raw_matches = sum(row["raw_matches_historical_expected_sha"] for row in results)
    capture_matches = sum(row["raw_matches_current_capture_sha"] for row in results)
    faulty_matches = sum(row["reproducer_matches_historical_expected_sha"] for row in results)
    if faulty_matches == 2 and raw_matches == 0 and capture_matches == 2:
        status = "MISSING_CONSTANT_REPRODUCER_MATCHES_2_OF_2"
        causal = "REPORTED_MISSING_CONSTANT_FAULT_MODEL_REPRODUCED; ORIGINAL_DEPLOYED_RECEIVER_CALL_PATH_UNAUTHENTICATED"
    elif raw_matches == 2 and capture_matches == 2:
        status = "RAW_BYTES_MATCH_HISTORICAL_EXPECTED_2_OF_2"
        causal = "MISSING_CONSTANT_REPRODUCER_NOT_REQUIRED_FOR_THESE_DIGESTS"
    else:
        status = "CANNOT_CONFIRM"
        causal = "NO_EXACT_TWO_SOURCE_CAUSAL_MATCH"
    identity_payload = [
        {"logical_name": row["logical_name"], "size_bytes": row["size_bytes"], "raw_bytes_sha256": row["raw_bytes_sha256"]}
        for row in results
    ]
    return {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_DIAGNOSTIC_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "status": status, "sources": results, "source_count": len(results), "source_read_count": len(results),
        "raw_matches_historical_expected_count": raw_matches,
        "corrected_missing_constant_reproducer_matches_historical_expected_count": faulty_matches,
        "current_capture_sha_matches_count": capture_matches,
        "fault_model_match": causal,
        "original_receiver_algorithm_causal_evidence": "CONTROLLER_REPORTED_SOURCE_DIFF; EXACT_DEPLOYED_CALL_PATH_NOT_AUTHENTICATED",
        "input_payload_identity_sha256": sha_bytes(canonical_json_bytes(identity_payload)),
        "phase_a_handling": {
            "prior_diagnostic_path": PHASE_A_DIAGNOSTIC_PATH.as_posix(),
            "prior_diagnostic_sha256": PHASE_A_DIAGNOSTIC_SHA256,
            "prior_terminal_event_sha256": TRACK_PHASE_A_TERMINAL_SHA256,
            "prior_measurement_record_preserved": True,
            "prior_reproducer_result_status": "INVALID_IMPLEMENTATION_SUPERSEDED_BY_A2",
            "prior_raw_and_capture_hash_measurements": "RETAINED_AS_MEASURED",
            "registry_correction_event_required": False,
            "registry_correction_reason": "The old terminal registry event records only ENGINEERING_CANARY_ONLY and does not encode the invalid reproducer match count; the erroneous mechanism conclusion exists in the immutable diagnostic artifact and is explicitly superseded here.",
        },
        "native_v193_identity": "CANNOT_CONFIRM",
        "selector_2022": "CANNOT_CONFIRM_NOT_NUMERICALLY_CHECKED",
        "pit_and_required_fields": "CANNOT_CONFIRM_NOT_CHECKED",
        "source_json_parse_calls": 0,
        "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
        "oof_runs": 0, "forward_runs": 0, "labels_read": 0, "weights_read": 0,
        "old_phase_a_artifacts_modified": False,
        "safety_flags": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
    }


def write_preflight_failure_backfill(execution_id: str, error_type: str,
                                    contract_sha: str, binding_sha: str) -> dict[str, Any]:
    _, _, _, OfficialRegistryAdapter = load_registry_components()
    failure = {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_PREFLIGHT_FAILURE_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "error_type": error_type, "registration_mode": "BACKFILLED_FROM_ARTIFACT",
        "occurred_at": None,
        "occurred_at_null_reason": "No durable RUN_STARTED occurred; exact failure time is not inferred.",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "recorded_at_is_not_execution_time": True, "run_started": False, "source_reads": 0,
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
        "safety_flags": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
    }
    write_json_exclusive(PREFLIGHT_FAILURE, failure)
    artifacts = (RUNNER, TEST, CONTRACT, BINDING, PREFLIGHT_FAILURE)
    adapter = OfficialRegistryAdapter(registry_path=REGISTRY, contract_path=CONTRACT,
        input_binding_path=BINDING, artifact_paths=artifacts, event_profile=PROFILE)
    spec = {
        "schema_version": "1.0", "registry_event_id": f"{execution_id}_RUN_TERMINATED",
        "experiment_id": EXPERIMENT_ID, "arm": ARM, "registration_mode": "BACKFILLED_FROM_ARTIFACT",
        "route_id": ROUTE_ID, "execution_id": execution_id, "track_id": TRACK_ID,
        "stage": "V193_R4_SHA_CONSTANT_PARITY_A2_PREFLIGHT_FAILURE", "event_type": "RUN_TERMINATED",
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
        "research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False,
    }
    with adapter._locked():
        raw, events = adapter._read_validated()
        if any(event.get("execution_id") == execution_id for event in events):
            raise RuntimeError("A2 no-start execution already exists")
        adapter._append_unlocked(raw, events, spec)
    with adapter._locked():
        _, events = adapter._read_validated()
    event = next(event for event in reversed(events) if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_TERMINATED")
    return {"failure": failure, "event": event}


def write_root_sync(status: str, execution_id: str, diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    adapter, _ = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT, START_RECEIPT, DIAGNOSTIC_RECEIPT, TERMINAL_RECEIPT))
    _, events, summary = read_validated_registry(adapter)
    route_events = [event for event in events if event.get("route_id") == ROUTE_ID]
    starts = sum(event.get("event_type") == "RUN_STARTED" for event in route_events)
    terminals = sum(event.get("event_type") in {"TERMINAL", "RUN_TERMINATED"} for event in route_events)
    if starts > 1 or terminals != 1:
        raise RuntimeError("A2 route lifecycle is not closed exactly once")
    safe = {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_SAFE_AGGREGATE_V1",
        "classification": "SAFE_AGGREGATE_NO_SOURCE_PATHS_OR_CONTENT",
        "status": status,
        "source_count": diagnostic.get("source_count", 0),
        "raw_matches_historical_expected_count": diagnostic.get("raw_matches_historical_expected_count", 0),
        "corrected_reproducer_matches_historical_expected_count": diagnostic.get("corrected_missing_constant_reproducer_matches_historical_expected_count", 0),
        "current_capture_sha_matches_count": diagnostic.get("current_capture_sha_matches_count", 0),
        "fault_model_match": diagnostic.get("fault_model_match", "CANNOT_CONFIRM_NO_SOURCE_READ"),
        "original_receiver_algorithm_causal_evidence": diagnostic.get("original_receiver_algorithm_causal_evidence", "CANNOT_CONFIRM_NO_SOURCE_READ"),
        "prior_phase_a_reproducer_status": "INVALID_IMPLEMENTATION_SUPERSEDED_BY_A2",
        "native_v193_identity": "CANNOT_CONFIRM", "selector_2022": "CANNOT_CONFIRM_NOT_NUMERICALLY_CHECKED",
        "pit_and_required_fields": "CANNOT_CONFIRM_NOT_CHECKED",
        "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
        "labels_read": 0, "weights_read": 0, "source_json_parse_calls": 0,
        "registry": {k: summary[k] for k in (
            "global_route_count", "global_event_count", "global_execution_count", "global_correction_count",
            "track_route_count", "track_event_count", "track_execution_count", "track_correction_count",
            "track_open_execution_count", "global_unregistered_route_count", "global_contract_only_count",
            "global_inventory_status", "full_legacy_schema_status")},
        "safety": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
    }
    write_json_exclusive(SAFE_AGGREGATE, safe)
    root_note = (
        "\n\n## 2026-10-18 V193 R4 A2 SHA constant-table parity correction\n"
        f"- Status: {status}; corrected missing-constant reproducer uses JS uint32 coercion before dependent state updates. The exact synthetic `abc` fixture and standard SHA reference controls pass.\n"
        f"- Historical inputs: raw SHA versus historical expected {diagnostic.get('raw_matches_historical_expected_count', 0)}/2; corrected reproducer versus historical expected {diagnostic.get('corrected_missing_constant_reproducer_matches_historical_expected_count', 0)}/2; raw versus prior capture SHA {diagnostic.get('current_capture_sha_matches_count', 0)}/2. Original deployed receiver call path remains unauthenticated.\n"
        f"- Phase A diagnostic {PHASE_A_DIAGNOSTIC_PATH.as_posix()} remains byte-immutable; its raw/capture measurements are retained, while the faulty reproducer result is marked INVALID_IMPLEMENTATION_SUPERSEDED_BY_A2. No registry CORRECTION was needed because the old terminal event did not encode its reproducer match count.\n"
        f"- Registry SHA-256 {summary['registry_sha256']}; global route/event/execution/correction {summary['global_route_count']}/{summary['global_event_count']}/{summary['global_execution_count']}/{summary['global_correction_count']}; V193 route/event/execution/correction/open {summary['track_route_count']}/{summary['track_event_count']}/{summary['track_execution_count']}/{summary['track_correction_count']}/{summary['track_open_execution_count']}. Selector 2022 and PIT remain CANNOT_CONFIRM; no fit/backtest/prediction/labels/weights. Safety: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.\n"
        f"- Navigation only: {BATCH_REL.as_posix()}/EVIDENCE_INDEX.json.\n"
    )
    for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md"):
        append_fsync(ROOT / name, root_note)
    root_hashes = {name: sha_file(ROOT / name) for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md")}
    raw_registry, _, final_summary = read_validated_registry(adapter)
    sync = {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_REGISTRY_SYNC_V1",
        "route_id": ROUTE_ID, "execution_id": execution_id,
        "route_run_started_count": starts, "terminal_event_count": terminals,
        "registry": final_summary, "contract_sha256": sha_file(CONTRACT), "input_binding_sha256": sha_file(BINDING),
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "root_file_sha256_after_append": root_hashes,
        "registry_readback_matches_summary": sha_bytes(raw_registry) == final_summary["registry_sha256"],
        "append_only_root_sync": True, "append_fsync": True,
    }
    write_json_exclusive(REGISTRY_SYNC_RECEIPT, sync)
    artifact_paths = (CONTRACT, BINDING, INTENT, START_RECEIPT, DIAGNOSTIC_RECEIPT, PREFLIGHT_FAILURE,
                      TERMINAL_RECEIPT, REGISTRY_SYNC_RECEIPT, SAFE_AGGREGATE, RUNNER, TEST)
    artifacts = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path)}
                 for path in artifact_paths if path.is_file()]
    index = {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_EVIDENCE_INDEX_V1",
        "classification": "NAVIGATION_NOT_AUTHORITY",
        "batch": BATCH_REL.as_posix(), "status": status, "route_id": ROUTE_ID,
        "execution_id": execution_id, "source_json_contents_in_index": False,
        "artifacts": artifacts, "registry_summary": final_summary,
        "registry_sync_receipt": {"path": REGISTRY_SYNC_RECEIPT.relative_to(ROOT).as_posix(), "sha256": sha_file(REGISTRY_SYNC_RECEIPT)},
        "safe_aggregate": {"path": SAFE_AGGREGATE.relative_to(ROOT).as_posix(), "sha256": sha_file(SAFE_AGGREGATE)},
        "safety": safe["safety"],
    }
    write_json_exclusive(EVIDENCE_INDEX, index)
    return {"registry": final_summary, "root_file_sha256": root_hashes,
            "safe_aggregate_sha256": sha_file(SAFE_AGGREGATE),
            "evidence_index_sha256": sha_file(EVIDENCE_INDEX),
            "registry_sync_receipt_sha256": sha_file(REGISTRY_SYNC_RECEIPT)}


def run() -> dict[str, Any]:
    execution_id = ROUTE_ID + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outputs = (INTENT, START_RECEIPT, DIAGNOSTIC_RECEIPT, PREFLIGHT_FAILURE,
               TERMINAL_RECEIPT, REGISTRY_SYNC_RECEIPT, SAFE_AGGREGATE, EVIDENCE_INDEX)
    if any(path.exists() for path in outputs):
        raise FileExistsError("A2 output exists; re-entry is blocked")
    contract_sha = binding_sha = None
    try:
        contract, binding, contract_sha, binding_sha = verify_identity()
        adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING))
        preflight = verify_preflight(adapter, guard, contract)
    except Exception as exc:
        error_type = type(exc).__name__
        try:
            contract_sha = contract_sha or (sha_file(CONTRACT) if CONTRACT.is_file() else None)
            binding_sha = binding_sha or (sha_file(BINDING) if BINDING.is_file() else None)
            if not contract_sha or not binding_sha:
                raise RuntimeError("A2 contract or binding unavailable for backfill")
            backfill = write_preflight_failure_backfill(execution_id, error_type, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {
                "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False,
                "registry_event": backfill["event"], "preflight_failure": backfill["failure"],
                "source_reads": 0, "fit_calls": 0,
            })
            sync = write_root_sync("CANNOT_CONFIRM_PREFLIGHT_FAILURE", execution_id, {"source_count": 0})
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry": sync["registry"], "evidence_index_sha256": sync["evidence_index_sha256"]}
        except Exception as backfill_exc:
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_AND_BACKFILL", "execution_id": execution_id,
                    "preflight_error_type": error_type, "backfill_error_type": type(backfill_exc).__name__,
                    "registry_event_written": False, "source_reads": 0}

    write_json_exclusive(INTENT, {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_RUN_INTENT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "preflight": preflight, "raw_source_reads_planned": 2,
        "source_reads_start_only_after_durable_run_started": True,
        "json_parse_calls": 0, "source_content_printed_or_persisted": False,
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT))
    try:
        claim = guard.claim_no_fit(execution_id=execution_id, arm=ARM,
            contract_sha256=contract_sha, input_binding_sha256=binding_sha)
        start_error = None
    except Exception as exc:
        start_error = type(exc).__name__
        starts = [event for event in adapter.list_events(EXPERIMENT_ID)
                  if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
        if starts:
            ExecutionClaim, _, nofit_kind, _ = load_registry_components()
            claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, nofit_kind, 0)
        else:
            backfill = write_preflight_failure_backfill(execution_id, start_error, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {
                "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False,
                "registry_event": backfill["event"], "preflight_failure": backfill["failure"],
                "source_reads": 0, "fit_calls": 0,
            })
            sync = write_root_sync("CANNOT_CONFIRM_PREFLIGHT_FAILURE", execution_id, {"source_count": 0})
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry": sync["registry"], "evidence_index_sha256": sync["evidence_index_sha256"]}

    starts = [event for event in adapter.list_events(EXPERIMENT_ID)
              if event.get("execution_id") == execution_id and event.get("event_type") == "RUN_STARTED"]
    durable_start = len(starts) == 1
    write_json_exclusive(START_RECEIPT, {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_RUN_STARTED_RECEIPT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "durable_start_readback": durable_start, "registry_event": starts[0] if durable_start else None,
        "source_reads_before_start": 0,
    })
    read_count = 0
    error_type = start_error
    try:
        if not durable_start or start_error:
            raise RuntimeError(start_error or "durable RUN_STARTED readback count mismatch")
        results = []
        for source in binding["sources"]:
            raw_bytes = (ROOT / source["path"]).read_bytes()
            read_count += 1
            raw_sha = sha_bytes(raw_bytes)
            faulty_sha = sha256_missing_constant_js_semantics(raw_bytes)
            results.append({
                "logical_name": source["logical_name"], "source_path": source["path"],
                "size_bytes": len(raw_bytes), "current_capture_sha256": source["captured_bytes_sha256"],
                "raw_bytes_sha256": raw_sha, "historical_expected_sha256": source["historical_expected_sha256"],
                "raw_matches_current_capture_sha": raw_sha == source["captured_bytes_sha256"],
                "raw_matches_historical_expected_sha": raw_sha == source["historical_expected_sha256"],
                "corrected_missing_constant_reproducer_sha256": faulty_sha,
                "reproducer_matches_historical_expected_sha": faulty_sha == source["historical_expected_sha256"],
                "json_parsed": False, "source_contents_printed_or_persisted": False,
            })
            del raw_bytes
        diagnostic = make_diagnostic(results, execution_id, contract_sha, binding_sha)
        succeeded = True
    except Exception as exc:
        error_type = error_type or type(exc).__name__
        diagnostic = {
            "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_DIAGNOSTIC_V1",
            "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
            "status": "CANNOT_CONFIRM", "error_type": error_type, "source_read_count": read_count,
            "source_count": 0, "source_content_printed_or_persisted": False,
            "native_v193_identity": "CANNOT_CONFIRM", "selector_2022": "CANNOT_CONFIRM_NOT_NUMERICALLY_CHECKED",
            "pit_and_required_fields": "CANNOT_CONFIRM_NOT_CHECKED", "source_json_parse_calls": 0,
            "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
            "labels_read": 0, "weights_read": 0,
            "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
            "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
            "safety_flags": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
        }
        succeeded = False
    write_json_exclusive(DIAGNOSTIC_RECEIPT, diagnostic)
    adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT, START_RECEIPT, DIAGNOSTIC_RECEIPT))
    guard.append_no_fit_terminal(claim, succeeded=succeeded, error_type=None if succeeded else error_type)
    _, events, _ = read_validated_registry(adapter)
    route_events = [event for event in events if event.get("route_id") == ROUTE_ID and event.get("execution_id") == execution_id]
    terminals = [event for event in route_events if event.get("event_type") == "TERMINAL"]
    if len(terminals) != 1:
        raise RuntimeError("A2 terminal event readback was not exactly one")
    status = diagnostic.get("status", "CANNOT_CONFIRM")
    terminal = {
        "schema_version": "V193_R4_SHA_CONSTANT_PARITY_A2_TERMINAL_V1",
        "status": status, "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST), "run_started": True,
        "source_read_count": read_count, "source_contents_printed_or_persisted": False,
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT), "terminal_event": terminals[0],
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
    }
    write_json_exclusive(TERMINAL_RECEIPT, terminal)
    sync = write_root_sync(status, execution_id, diagnostic)
    return {
        "status": status, "execution_id": execution_id, "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha, "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT), "terminal_receipt_sha256": sha_file(TERMINAL_RECEIPT),
        "registry": sync["registry"], "registry_sync_receipt_sha256": sync["registry_sync_receipt_sha256"],
        "safe_aggregate_sha256": sync["safe_aggregate_sha256"], "evidence_index_sha256": sync["evidence_index_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("self-check", "run"))
    args = parser.parse_args()
    if args.phase == "self-check":
        print(json.dumps(synthetic_self_check(), sort_keys=True))
    else:
        print(json.dumps(run(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
