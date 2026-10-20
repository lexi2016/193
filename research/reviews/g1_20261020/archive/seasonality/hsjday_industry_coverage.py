from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path("/LOCAL_ONLY/factor_factory")
WORKTREE = ROOT / "workspace/.task_worktrees/a-hsjday-20y-20261019"
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
SCRIPT_PATH = Path(__file__).resolve()
TEST_PATH = WORKTREE / "research/a_share_top100_environment_v1/tests/test_hsjday_industry_coverage.py"
BATCH = ROOT / "research/a_share_top100_environment_v1/batches/hsjday_20y_20261019/panel_dependency_01/industry_coverage_02"
CONTRACT_PATH = BATCH / "CONTRACT.json"
CONFIG_PATH = BATCH / "CONFIG.json"
INPUT_BINDING_PATH = BATCH / "INPUT_BINDING.json"
BASELINE_PATH = BATCH / "BASELINE_AND_DEPENDENCIES.json"
IDENTITY_PATH = BATCH / "FROZEN_IDENTITY.json"
OUTPUT_DIR = BATCH / "outputs"
RUN_RECEIPT_PATH = OUTPUT_DIR / "RUN_RECEIPT.json"
PROGRESS_PATH = OUTPUT_DIR / "PROGRESS.jsonl"
REGISTRY_CLOSEOUT_PATH = OUTPUT_DIR / "REGISTRY_CLOSEOUT.json"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "OUTPUT_MANIFEST.json"
READER_CANARY_PATH = OUTPUT_DIR / "INDUSTRY_READER_CANARY.json"
REGISTRY_PATH = ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"

PRIOR_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/hsjday_20y_20261019/panel_dependency_01"
PRIOR_INPUT_BINDING_PATH = PRIOR_BATCH / "INPUT_BINDING.json"
PRIOR_INPUT_BINDING_SHA256 = "ec9ef9aa67ad9aa7d0e9f4ec744bc6f08e0bc60a955e7b346c9ee284d210770c"
PRIOR_OUTPUT_MANIFEST_PATH = PRIOR_BATCH / "outputs/OUTPUT_MANIFEST.json"
PRIOR_OUTPUT_MANIFEST_SHA256 = "ad9371810c772298960d5c65cc5e13d3dfe3fb087d039b926dde38de32a8daf6"
PRIOR_INDUSTRY_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/hsjday_20y_20261019/panel_dependency_01/industry_coverage_01"
PRIOR_INDUSTRY_IDENTITY_PATH = PRIOR_INDUSTRY_BATCH / "FROZEN_IDENTITY.json"
PRIOR_INDUSTRY_RECEIPT_PATH = PRIOR_INDUSTRY_BATCH / "outputs/RUN_RECEIPT.json"
PRIOR_INDUSTRY_MANIFEST_PATH = PRIOR_INDUSTRY_BATCH / "outputs/OUTPUT_MANIFEST.json"
PRIOR_INDUSTRY_IDENTITY_SHA256 = "fe890af9f46652ac68bd370150563febbc07e11da462cfb51dac8da9d1d2937d"
PRIOR_INDUSTRY_RECEIPT_SHA256 = "e9f5eb0424262e884cdd7769d70f97537df92a1bb672f7b9eb3cd4e90c2fff84"
PRIOR_INDUSTRY_MANIFEST_SHA256 = "22434acabd9a71f7f74e26d87d4023da74b7bc815291aec8c70006142862c627"
PRIOR_FAILED_SOURCE_PATH = BATCH / "PRIOR_FAILED_SOURCE.py"
PRIOR_FAILED_TEST_PATH = BATCH / "PRIOR_FAILED_TEST.py"
PRIOR_FAILED_SOURCE_SHA256 = "32cb19964925d5aed5f886f3ee9e11d15b8ee649a9b712403d4091b07f66be16"
PRIOR_FAILED_TEST_SHA256 = "174944fd9c574a3fbc00e62eb580a85b695d542519b61267df2485c9cd3c03f3"

INVENTORY_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/hsjday_20y_20261019/outputs/A_SHARE_TDX_20Y_DATA_INVENTORY_20261019_HSJDAY_ZIP_INVENTORY_03/FINAL_01"
INVENTORY_MANIFEST_PATH = INVENTORY_BATCH / "OUTPUT_MANIFEST.json"
INVENTORY_MANIFEST_SHA256 = "6399db4da271a6af1523421d1cc6cf6d9849dae9c6b2732ca6a65c5691572f29"
ROSTER_PATH = INVENTORY_BATCH / "FILE_INVENTORY_INDIVIDUAL_STOCK.csv"
ROSTER_SHA256 = "12f748784ee33e66337ed28331c994bdd81abecf765e47feb26d36f3706bbf1c"

SEASONALITY_BINDING_PATH = ROOT / "research/a_share_top100_environment_v1/batches/seasonality_r1_20261019/INPUT_BINDING.json"
EXPECTED_INDUSTRY_MEMBER = "v192_external_v2_20261017/industry_cap_inputs_20261017/industry_current_retrospective.csv"
EXPECTED_INDUSTRY_MEMBER_SHA256 = "c9044f6d2413aad5c0e0cead62d591a4b2bdbc8f10cc5878fdfc9b1497f5e1ff"
EXPECTED_INDUSTRY_MEMBER_BYTES = 3_101_881
MODEL_SELECTION_PATH = ROOT / "workspace/research/a_share_model_selection_v1/model_selection.py"
MODEL_SELECTION_SHA256 = "0aa1428f2b62447e35ecbebce4843613dd01111cfb4340f38221e022eecbd40c"

RUNTIME_POINTERS_PATH = PRIOR_BATCH / "outputs/RUNTIME_POINTERS.jsonl"
RUNTIME_POINTERS_SHA256 = "1040424502c8858f1287e70900c088a322060244adb2dce8c45490a6336513c5"
SELECTOR_POINTER_PATH = Path("/LOCAL_ONLY/private_home/Library/Application Support/com.lexi.v35.local/state/workspace/数据/data_layer_interface_current/CURRENT_SELECTOR_READY_PANEL.txt")
SELECTOR_POINTER_SHA256 = "9dc6eaaa4b5122aa466dd4550de4871b7bb8ed35f1615f48065899b1633744cd"
SELECTOR_PATH = Path("/LOCAL_ONLY/private_home/Library/Application Support/com.lexi.v35.local/state/workspace/数据/selector_ready_panels/data_layer_daily_formula_increment_update_v1/selector_ready_current_to_20261019_20261019_171031/selector_ready_current_to_20261019.parquet")
SELECTOR_BYTES = 1_391_079_536
SELECTOR_MTIME_NS = 1_792_401_238_218_079_092
SELECTOR_ROWS = 3_036_316
SELECTOR_COLUMNS = 207
SELECTOR_RUNTIME = Path("/LOCAL_ONLY/private_home/Library/Application Support/com.lexi.v4.local/runtime/venvs/v3.5-mac/bin/python")

EXPERIMENT_ID = "A_SHARE_TDX_20Y_PANEL_DEPENDENCY_20261019"
EXECUTION_ID = "A_SHARE_TDX_20Y_DATA_INVENTORY_20261019_HSJDAY_INDUSTRY_COVERAGE_02"
ROUTE_ID = "a-share-tdx-local-20y-panel-dependency-20261019"
TRACK_ID = "A_SHARE_TDX_LOCAL_20Y_PANEL_DEPENDENCY"
STAGE = "HSJDAY_CURRENT_INDUSTRY_COVERAGE"
BASELINE_ID = "HSJDAY_CURRENT_CLASSIFICATION_RETROSPECTIVE_20261019"
ONE_VARIABLE = "CURRENT_CAPCO_COVERAGE_WITH_EXISTING_TDX_SUPPLEMENT_FOR_GAPS"
SAFETY_FLAGS = {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("pandas", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.executable, "python_version": sys.version.split()[0], "packages": packages}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_registry_helper(shared: Mapping[str, Any]):
    entry = shared["official_registry_helper"]
    helper_path = Path(entry["path"])
    if sha256_file(helper_path) != entry["sha256"]:
        raise RuntimeError("frozen OfficialRegistryAdapter helper SHA changed")
    spec = importlib.util.spec_from_file_location("hsjday_industry_registry", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load OfficialRegistryAdapter helper")
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    env.EXPERIMENT_ID, env.ROUTE_ID, env.TRACK_ID = EXPERIMENT_ID, ROUTE_ID, TRACK_ID
    adapter, core = env.load_official_registry(read_json(INPUT_BINDING_PATH), contract_path=CONTRACT_PATH,
        input_binding_path=INPUT_BINDING_PATH, config_path=CONFIG_PATH, code_path=SCRIPT_PATH, test_path=TEST_PATH)
    adapter.event_profile.update({"track_id": TRACK_ID, "route_id": ROUTE_ID, "baseline_id": BASELINE_ID,
        "one_variable": ONE_VARIABLE, "signal_scope": "ZIP_CONFIRMED_INDIVIDUAL_STOCK_CODES",
        "target_scope": "CURRENT_CLASSIFICATION_ONLY_NO_FORWARD_LABEL", "evidence_level": "ENGINEERING_DEPENDENCY_CHECKPOINT"})
    return env, adapter, core


def _context() -> dict[str, Any]:
    return {
        "contract_sha256": sha256_file(CONTRACT_PATH), "input_binding_sha256": sha256_file(INPUT_BINDING_PATH),
        "config_sha256": sha256_file(CONFIG_PATH), "code_sha256": sha256_file(SCRIPT_PATH),
        "input_paths": [str(SEASONALITY_BINDING_PATH), str(ROSTER_PATH), str(SELECTOR_POINTER_PATH), str(SELECTOR_PATH)],
        "artifact_paths": [str(p.resolve()) for p in (CONTRACT_PATH, CONFIG_PATH, INPUT_BINDING_PATH, BASELINE_PATH, IDENTITY_PATH,
            SCRIPT_PATH, TEST_PATH, PRIOR_INPUT_BINDING_PATH, PRIOR_OUTPUT_MANIFEST_PATH, INVENTORY_MANIFEST_PATH, ROSTER_PATH, RUNTIME_POINTERS_PATH)
            if p.is_file()],
        "input_component_sha256": {"prior_dependency_input_binding_sha256": PRIOR_INPUT_BINDING_SHA256,
            "prior_dependency_output_manifest_sha256": PRIOR_OUTPUT_MANIFEST_SHA256,
            "inventory_manifest_sha256": INVENTORY_MANIFEST_SHA256, "zip_confirmed_roster_sha256": ROSTER_SHA256,
            "runtime_pointer_receipt_sha256": RUNTIME_POINTERS_SHA256,
            "seasonality_binding_sha256": None, "seasonality_binding_sha256_null_reason": "READ_AFTER_RUN_STARTED",
            "selector_target_sha256": None, "selector_target_sha256_null_reason": "NOT_REHASHED; reuse frozen pointer/footer identity"},
        "code_sha256_by_file": {str(SCRIPT_PATH.resolve()): sha256_file(SCRIPT_PATH), str(TEST_PATH.resolve()): sha256_file(TEST_PATH),
            str(MODEL_SELECTION_PATH.resolve()): MODEL_SELECTION_SHA256},
        "signal_date_range": {"start": "CURRENT_CLASSIFICATION_SNAPSHOT", "end": "2026-10-19", "basis": "current retrospective only"},
        "truth_access": "NOT_READ", "sealed_status": "NOT_ENTERED", "runtime": runtime_metadata(),
        "evidence_level": "ENGINEERING_DEPENDENCY_CHECKPOINT",
        "deterministic_metrics": {"execution_kind": "NO_FIT_ENGINEERING_CANARY", "planned_real_fit_count": 0,
            "real_fit_budget_debit": 0, "real_fit_returned_count": 0, "fit_call_entered": False,
            "truth_positive_denominator": None, "selection_slots": 0},
        "null_reasons": {"checkpoint_ref": "OUTPUT_NOT_YET_WRITTEN", "target_date_range": "NO_FORWARD_LABEL",
            "deterministic_metrics.truth_positive_denominator": "NO_LABEL_OR_TARGET"},
    }


def _candidate_progress_preflight(adapter: Any) -> None:
    context = _context()
    preflight_id = EXECUTION_ID + "_PREFLIGHT"
    adapter.set_route_context(preflight_id, context)
    start = adapter._event_spec({"experiment_id": EXPERIMENT_ID, "execution_id": preflight_id, "arm": STAGE,
        "contract_sha256": context["contract_sha256"], "input_binding_sha256": context["input_binding_sha256"],
        "execution_kind": "NO_FIT_ENGINEERING_CANARY", "planned_real_fit_count": 0, "real_fit_budget_debit": 0}, event_type="RUN_STARTED")
    progress = dict(start)
    progress.update({"registry_event_id": f"{preflight_id}_PROGRESS", "event_type": "PROGRESS", "stage": "REGISTRY_PROGRESS_DRY_RUN",
        "occurred_at": utc_now(), "decision": "RUNNING", "checkpoint_ref": str(PROGRESS_PATH)})
    with adapter._locked():
        raw, events = adapter._read_validated()
    payload = adapter.builder.materialize([start, progress], starting_previous_sha=events[-1]["event_sha256"])
    adapter.validator.validate_registry_bytes(raw + adapter.builder._registry_payload(payload), source="hsjday-industry-progress-preflight")


def freeze_contract() -> None:
    if any(p.exists() for p in (CONTRACT_PATH, CONFIG_PATH, INPUT_BINDING_PATH, BASELINE_PATH, IDENTITY_PATH)):
        raise FileExistsError("industry coverage frozen identity exists; refusing rewrite")
    if not SCRIPT_PATH.is_file() or not TEST_PATH.is_file() or not MODEL_SELECTION_PATH.is_file():
        raise FileNotFoundError("industry coverage script/test/existing source reader missing")
    for path, expected in ((PRIOR_INPUT_BINDING_PATH, PRIOR_INPUT_BINDING_SHA256), (PRIOR_OUTPUT_MANIFEST_PATH, PRIOR_OUTPUT_MANIFEST_SHA256),
                           (INVENTORY_MANIFEST_PATH, INVENTORY_MANIFEST_SHA256), (ROSTER_PATH, ROSTER_SHA256),
                           (RUNTIME_POINTERS_PATH, RUNTIME_POINTERS_SHA256),
                           (PRIOR_INDUSTRY_IDENTITY_PATH, PRIOR_INDUSTRY_IDENTITY_SHA256),
                           (PRIOR_INDUSTRY_RECEIPT_PATH, PRIOR_INDUSTRY_RECEIPT_SHA256),
                           (PRIOR_INDUSTRY_MANIFEST_PATH, PRIOR_INDUSTRY_MANIFEST_SHA256),
                           (PRIOR_FAILED_SOURCE_PATH, PRIOR_FAILED_SOURCE_SHA256),
                           (PRIOR_FAILED_TEST_PATH, PRIOR_FAILED_TEST_SHA256)):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"prior frozen input identity changed: {path}")
    prior_industry_identity = read_json(PRIOR_INDUSTRY_IDENTITY_PATH)
    prior_industry_receipt = read_json(PRIOR_INDUSTRY_RECEIPT_PATH)
    if (prior_industry_identity.get("execution_id") != "A_SHARE_TDX_20Y_DATA_INVENTORY_20261019_HSJDAY_INDUSTRY_COVERAGE_01"
            or prior_industry_identity.get("code_sha256") != PRIOR_FAILED_SOURCE_SHA256
            or prior_industry_receipt.get("status") != "CANNOT_CONFIRM"
            or prior_industry_receipt.get("code_sha256") != PRIOR_FAILED_SOURCE_SHA256
            or "different industry projection" not in str((prior_industry_receipt.get("failure") or {}).get("message", ""))):
        raise RuntimeError("prior terminal industry failure differs from the bound projection-contract defect")
    if sha256_file(MODEL_SELECTION_PATH) != MODEL_SELECTION_SHA256:
        raise RuntimeError("existing bound industry reader changed before freeze")
    prior_binding = read_json(PRIOR_INPUT_BINDING_PATH)
    shared = prior_binding["shared_dependencies"]
    if sha256_file(Path(shared["official_registry_helper"]["path"])) != shared["official_registry_helper"]["sha256"]:
        raise RuntimeError("official registry helper does not match prior frozen binding")
    binding = {
        "schema_version": "HSJDAY_INDUSTRY_COVERAGE_INPUT_BINDING_V1", "experiment_id": EXPERIMENT_ID,
        "execution_id": EXECUTION_ID, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_path": str(CONTRACT_PATH), "config_path": str(CONFIG_PATH), "baseline_record_path": str(BASELINE_PATH),
        "registry_path": str(REGISTRY_PATH), "known_inputs": {
            "prior_dependency_binding": {"path": str(PRIOR_INPUT_BINDING_PATH), "sha256": PRIOR_INPUT_BINDING_SHA256},
            "prior_dependency_output_manifest": {"path": str(PRIOR_OUTPUT_MANIFEST_PATH), "sha256": PRIOR_OUTPUT_MANIFEST_SHA256},
            "zip_confirmed_inventory_manifest": {"path": str(INVENTORY_MANIFEST_PATH), "sha256": INVENTORY_MANIFEST_SHA256},
            "zip_confirmed_individual_stock_roster": {"path": str(ROSTER_PATH), "sha256": ROSTER_SHA256,
                "classification_basis": "R3 frozen prefix rules; not historical issuer-status proof"},
            "seasonality_industry_input_binding": {"path": str(SEASONALITY_BINDING_PATH), "sha256": None,
                "sha256_null_reason": "READ_AFTER_RUN_STARTED; member identity is separately frozen from controller-provided binding"},
            "seasonality_industry_member": {"member": EXPECTED_INDUSTRY_MEMBER, "sha256": EXPECTED_INDUSTRY_MEMBER_SHA256,
                "bytes": EXPECTED_INDUSTRY_MEMBER_BYTES, "classification_lock": "CURRENT_CLASSIFICATION_RETROSPECTIVE"},
            "runtime_selector_pointer_receipt": {"path": str(RUNTIME_POINTERS_PATH), "sha256": RUNTIME_POINTERS_SHA256,
                "pointer_path": str(SELECTOR_POINTER_PATH), "pointer_sha256": SELECTOR_POINTER_SHA256,
                "target_path": str(SELECTOR_PATH), "target_size": SELECTOR_BYTES, "target_mtime_ns": SELECTOR_MTIME_NS,
                "target_rows": SELECTOR_ROWS, "target_columns": SELECTOR_COLUMNS,
                "target_sha256": None, "target_sha256_null_reason": "Existing 1.39GB producer output; reuse frozen pointer/footer identity, do not rehash"},
            "prior_industry_attempt": {"execution_id": prior_industry_identity["execution_id"],
                "identity_path": str(PRIOR_INDUSTRY_IDENTITY_PATH), "identity_sha256": PRIOR_INDUSTRY_IDENTITY_SHA256,
                "run_receipt_path": str(PRIOR_INDUSTRY_RECEIPT_PATH), "run_receipt_sha256": PRIOR_INDUSTRY_RECEIPT_SHA256,
                "output_manifest_path": str(PRIOR_INDUSTRY_MANIFEST_PATH), "output_manifest_sha256": PRIOR_INDUSTRY_MANIFEST_SHA256,
                "failed_source_snapshot_path": str(PRIOR_FAILED_SOURCE_PATH), "failed_source_sha256": PRIOR_FAILED_SOURCE_SHA256,
                "failed_test_snapshot_path": str(PRIOR_FAILED_TEST_PATH), "failed_test_sha256": PRIOR_FAILED_TEST_SHA256,
                "failure": "Reader returns full schema names and projected row values; prior caller compared the schema to the projection exactly."},
        },
        "shared_dependencies": shared,
        "safety_flags": SAFETY_FLAGS,
        "pre_run_context": {"source_member_rows_read": False, "selector_rows_read": False,
            "scope": "Only prior manifests/metadata, source reader code and task-provided bound member identity were used to freeze this stage."},
    }
    config = {
        "schema_version": "HSJDAY_INDUSTRY_COVERAGE_CONFIG_V1", "one_variable": ONE_VARIABLE,
        "classification_lock": "CURRENT_CLASSIFICATION_RETROSPECTIVE",
        "primary_classification": "CAPCO current retrospective source member bound in seasonality input binding",
        "supplement_rule": "Only ZIP roster codes absent from CAPCO may use existing selector-ready TDX industry output; store classification_system per row and aggregate by system+code, never merge taxonomy codes.",
        "selector_reader_contract": "Existing _tabular_rows returns full Parquet schema names; its row iterator projects only requested columns. Resolve requested fields as a subset of schema, then read only those projected row keys.",
        "exact_member": EXPECTED_INDUSTRY_MEMBER, "exact_member_sha256": EXPECTED_INDUSTRY_MEMBER_SHA256,
        "selector_columns": ["ts_code", "trade_date", "name", "tdx_industry_code", "tdx_industry_name"],
        "selector_scan_policy": "Read only the five declared columns from the already-resolved selector parquet if CAPCO has roster gaps or CAPCO lacks stock names; do not read OHLCV or labels/returns.",
        "exact_code_join": "Prefer exact ts_code = six_digit_security_code + '.' + market.upper(); six-digit-only source codes may join only when unique across SH/SZ roster; otherwise UNKNOWN.",
        "unknown_policy": "Missing/ambiguous code, blank category or absent label remains UNKNOWN with a reason; no inferred labels or cross-system translation.",
        "expected_outputs": ["RUN_RECEIPT.json", "PROGRESS.jsonl", "ROUTE_CONTEXT.json", "REGISTRY_CLOSEOUT.json", "INDUSTRY_READER_CANARY.json",
            "INDUSTRY_ROSTER.csv", "INDUSTRY_SOURCE_COVERAGE.json", "INDUSTRY_GAPS.csv", "INDUSTRY_COUNTS_BY_SYSTEM.csv", "OUTPUT_MANIFEST.json"],
    }
    contract = {
        "schema_version": "HSJDAY_INDUSTRY_COVERAGE_CONTRACT_V1", "experiment_id": EXPERIMENT_ID,
        "execution_id": EXECUTION_ID, "route_id": ROUTE_ID, "track_id": TRACK_ID, "stage": STAGE,
        "baseline_id": BASELINE_ID, "one_variable": ONE_VARIABLE,
        "goal": "Measure and label current retrospective industry coverage for ZIP-confirmed A-share stock codes using the already-bound CAPCO member, supplementing only missing codes from the existing current selector producer and preserving UNKNOWN.",
        "hypothesis": "The bound CAPCO member maps a measurable subset of the ZIP roster; existing selector TDX industry fields can fill some absent codes without merging classification systems.",
        "incremental_correction": "Fix the prior exact-equality check between the full Parquet schema and the requested row projection; preserve all other frozen coverage rules.",
        "prediction": "Produce one per-code row for every R3 individual_stock member with code, optional name, classification system, code/label, source/as-of, and explicit UNKNOWN reason; publish coverage counts by system.",
        "rejection_condition": "Member SHA, helper SHA, roster SHA, selector pointer, or selector footer metadata mismatch; stop and preserve CANNOT_CONFIRM.",
        "signal_date_range": {"start": "CURRENT_CLASSIFICATION_SNAPSHOT", "end": "2026-10-19"}, "target_date_range": None,
        "truth_access": "NOT_READ", "sealed_status": "NOT_ENTERED",
        "pass_threshold": "All ZIP roster rows are represented exactly once in output; known labels carry source system/path/as-of, unresolved rows remain UNKNOWN; no taxonomy blending.",
        "sealed_data": "NOT_ENTERED", "writer": "zip_inventory_writer",
        "allowed_paths": [str(SCRIPT_PATH), str(TEST_PATH), str(BATCH / "**"), str(REGISTRY_PATH)],
        "forbidden_paths": ["phase-01 frozen dependency files", "R3 FINAL_01 inventory outputs", "selector/backtest/QFQ/current pointer writes", "ZIP/source archive writes", "market data retrieval", "Top100 or returns", "V4/services/schedules/UI"],
        "stop_rule": "Stop on any identity mismatch or failed member parse; do not mutate source or substitute another classification system.", "safety_flags": SAFETY_FLAGS,
    }
    baseline = {
        "schema_version": "HSJDAY_INDUSTRY_COVERAGE_BASELINE_V1", "baseline_commit": "20610d06fc1767bdd8dd9c6c54d48776b4b874d8",
        "baseline_branch": "codex/a-hsjday-20y-20261019", "task_worktree": str(WORKTREE),
        "prior_phase_01_status": "CANNOT_CONFIRM; no industry member was extracted; do not redo date, calendar, footer or XDXR stages.",
        "prior_phase_01_manifest_sha256": PRIOR_OUTPUT_MANIFEST_SHA256, "R3_inventory_manifest_sha256": INVENTORY_MANIFEST_SHA256,
        "prior_industry_attempt": {"execution_id": prior_industry_identity["execution_id"], "status": prior_industry_receipt["status"],
            "failure": prior_industry_receipt["failure"], "output_manifest_sha256": PRIOR_INDUSTRY_MANIFEST_SHA256,
            "failed_code_sha256": PRIOR_FAILED_SOURCE_SHA256, "failed_test_sha256": PRIOR_FAILED_TEST_SHA256},
        "R3_roster_sha256": ROSTER_SHA256, "model_selection_reader_sha256": MODEL_SELECTION_SHA256,
        "runtime_selector_pointer_sha256": SELECTOR_POINTER_SHA256, "runtime_selector_metadata": {"bytes": SELECTOR_BYTES, "mtime_ns": SELECTOR_MTIME_NS, "rows": SELECTOR_ROWS, "columns": SELECTOR_COLUMNS},
        "not_deployed": ["no source/cache write", "no selector/backtest/current pointer write", "no market data retrieval", "no training or return calculation"],
        "runtime": runtime_metadata(), "shared_dependencies": shared,
    }
    write_new(CONTRACT_PATH, canonical_bytes(contract))
    write_new(CONFIG_PATH, canonical_bytes(config))
    write_new(INPUT_BINDING_PATH, canonical_bytes(binding))
    write_new(BASELINE_PATH, canonical_bytes(baseline))
    identity = {"schema_version": "HSJDAY_INDUSTRY_COVERAGE_IDENTITY_V1", "execution_id": EXECUTION_ID,
        "contract_sha256": sha256_file(CONTRACT_PATH), "config_sha256": sha256_file(CONFIG_PATH),
        "input_binding_sha256": sha256_file(INPUT_BINDING_PATH), "baseline_and_dependencies_sha256": sha256_file(BASELINE_PATH),
        "code_sha256": sha256_file(SCRIPT_PATH), "test_sha256": sha256_file(TEST_PATH),
        "model_selection_reader_sha256": MODEL_SELECTION_SHA256, "roster_sha256": ROSTER_SHA256,
        "industry_member_sha256": EXPECTED_INDUSTRY_MEMBER_SHA256, "selector_pointer_sha256": SELECTOR_POINTER_SHA256}
    write_new(IDENTITY_PATH, canonical_bytes(identity))


def verify_frozen() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, config, binding, baseline, identity = (read_json(p) for p in (CONTRACT_PATH, CONFIG_PATH, INPUT_BINDING_PATH, BASELINE_PATH, IDENTITY_PATH))
    actual = {"contract_sha256": sha256_file(CONTRACT_PATH), "config_sha256": sha256_file(CONFIG_PATH),
        "input_binding_sha256": sha256_file(INPUT_BINDING_PATH), "baseline_and_dependencies_sha256": sha256_file(BASELINE_PATH),
        "code_sha256": sha256_file(SCRIPT_PATH), "test_sha256": sha256_file(TEST_PATH),
        "model_selection_reader_sha256": sha256_file(MODEL_SELECTION_PATH)}
    if any(identity.get(k) != v for k, v in actual.items()) or contract.get("execution_id") != EXECUTION_ID or binding.get("execution_id") != EXECUTION_ID:
        raise RuntimeError("frozen industry coverage identity mismatch")
    if Path(sys.executable).resolve() != SELECTOR_RUNTIME.resolve():
        raise RuntimeError("industry stage must use the controller-identified producer interpreter")
    return contract, config, binding


def _claim(env: Any, adapter: Any, core: Any, context: dict[str, Any]):
    with adapter._locked():
        _, events = adapter._read_validated()
    scoped = [e for e in events if e.get("execution_id") == EXECUTION_ID]
    if any(e.get("event_type") in {"RUN_STARTED", "TERMINAL"} for e in scoped):
        raise RuntimeError("industry execution already started or terminal; refusing repeat")
    _candidate_progress_preflight(adapter)
    claim = env._claim_stage(adapter, core, stage=STAGE, execution_id=EXECUTION_ID,
        contract_sha256=context["contract_sha256"], input_binding_sha256=context["input_binding_sha256"], context=context)
    return claim


def _append_progress(adapter: Any, phase: str, metrics: Mapping[str, Any]) -> None:
    with adapter._locked():
        _, events = adapter._read_validated()
    sequence = sum(e.get("event_type") == "PROGRESS" for e in events if e.get("execution_id") == EXECUTION_ID) + 1
    context = dict(adapter._route_context[EXECUTION_ID])
    context["artifact_paths"] = [*context.get("artifact_paths", []), str(PROGRESS_PATH)]
    adapter.set_route_context(EXECUTION_ID, context)
    spec = adapter._event_spec({"experiment_id": EXPERIMENT_ID, "execution_id": EXECUTION_ID, "arm": STAGE,
        "contract_sha256": context["contract_sha256"], "input_binding_sha256": context["input_binding_sha256"],
        "execution_kind": "NO_FIT_ENGINEERING_CANARY", "planned_real_fit_count": 0, "real_fit_budget_debit": 0}, event_type="RUN_STARTED")
    spec.update({"registry_event_id": f"{EXECUTION_ID}_PROGRESS_{sequence:04d}", "event_type": "PROGRESS", "stage": phase,
        "occurred_at": utc_now(), "decision": "RUNNING", "stop_reason": "NONE", "checkpoint_ref": str(PROGRESS_PATH),
        "deterministic_metrics": {**spec["deterministic_metrics"], **dict(metrics)}, "input_identity": {**spec["input_identity"], "progress_phase": phase}})
    with adapter._locked():
        raw, all_events = adapter._read_validated()
        if not any(e.get("execution_id") == EXECUTION_ID and e.get("event_type") == "RUN_STARTED" for e in all_events):
            raise RuntimeError("cannot append industry progress before RUN_STARTED")
        adapter._append_unlocked(raw, all_events, spec)
    append_jsonl(PROGRESS_PATH, {"sequence": sequence, "phase": phase, "occurred_at_utc": utc_now(), **dict(metrics)})


def _registry_terminal(adapter: Any, claim: Any, status: str, metrics: Mapping[str, Any], failure: BaseException | None) -> None:
    context = dict(adapter._route_context[EXECUTION_ID])
    context["artifact_paths"] = [*context.get("artifact_paths", []), str(RUN_RECEIPT_PATH)]
    context["deterministic_metrics"] = {**context.get("deterministic_metrics", {}), **dict(metrics)}
    adapter.set_route_context(EXECUTION_ID, context)
    adapter.terminalize(claim, {"experiment_id": EXPERIMENT_ID, "execution_id": EXECUTION_ID, "arm": STAGE,
        "contract_sha256": context["contract_sha256"], "input_binding_sha256": context["input_binding_sha256"],
        "execution_kind": "NO_FIT_ENGINEERING_CANARY", "terminal_status": status,
        "real_fit_returned_count": 0, "fit_call_entered": False, "finished_at": utc_now(),
        "error_type": None if failure is None else f"{type(failure).__name__}:{str(failure)[:240]}"})


def _closeout() -> dict[str, Any]:
    binding = read_json(INPUT_BINDING_PATH)
    spec = importlib.util.spec_from_file_location("hsjday_industry_closeout_registry", Path(binding["shared_dependencies"]["official_registry_helper"]["path"]))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official registry helper for closeout")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    env.EXPERIMENT_ID, env.ROUTE_ID, env.TRACK_ID = EXPERIMENT_ID, ROUTE_ID, TRACK_ID
    adapter, _ = env.load_official_registry(binding, contract_path=CONTRACT_PATH, input_binding_path=INPUT_BINDING_PATH,
        config_path=CONFIG_PATH, code_path=SCRIPT_PATH, test_path=TEST_PATH)
    with adapter._locked(): raw, events = adapter._read_validated()
    scoped = [e for e in events if e.get("execution_id") == EXECUTION_ID]
    route_events = [e for e in events if e.get("route_id") == ROUTE_ID]
    close = {"schema_version": "HSJDAY_INDUSTRY_COVERAGE_REGISTRY_CLOSEOUT_V1", "registry_path": str(REGISTRY_PATH),
        "registry_sha256": hashlib.sha256(raw).hexdigest(), "global_route_count": len({e.get("route_id") for e in events}),
        "global_event_count": len(events), "global_execution_count": len({e.get("execution_id") for e in events}),
        "global_correction_count": sum(e.get("event_type") == "CORRECTION" for e in events),
        "route_event_count": len(route_events), "route_execution_count": len({e.get("execution_id") for e in route_events}),
        "route_correction_count": sum(e.get("route_id") == ROUTE_ID and e.get("event_type") == "CORRECTION" for e in events),
        "execution_run_started_count": sum(e.get("event_type") == "RUN_STARTED" for e in scoped),
        "execution_progress_count": sum(e.get("event_type") == "PROGRESS" for e in scoped),
        "execution_terminal_count": sum(e.get("event_type") == "TERMINAL" for e in scoped),
        "run_started_event_sha256": next((e.get("event_sha256") for e in scoped if e.get("event_type") == "RUN_STARTED"), None),
        "terminal_event_sha256": next((e.get("event_sha256") for e in reversed(scoped) if e.get("event_type") == "TERMINAL"), None),
        "route_local_unregistered_route_count": 0, "route_local_contract_only_count": 0,
        "global_unregistered_route_count": None, "global_unregistered_route_count_reason": "GLOBAL_ROUTE_CATALOG_NOT_AUDITED",
        "global_contract_only_count": None, "global_contract_only_count_reason": "GLOBAL_CONTRACT_ONLY_CATALOG_NOT_AUDITED",
        "external_or_git_persistence": "CANNOT_CONFIRM"}
    write_new(REGISTRY_CLOSEOUT_PATH, canonical_bytes(close))
    return close


def _append_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def _blank(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in {"", "nan", "nat", "none", "null", "unknown", "未知", "未分类"}


def _read_roster() -> list[dict[str, str]]:
    if sha256_file(ROSTER_PATH) != ROSTER_SHA256:
        raise RuntimeError("R3 individual-stock roster SHA changed")
    rows: list[dict[str, str]] = []
    with ROSTER_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"market", "security_code", "classification"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"R3 roster schema missing fields: {required - set(reader.fieldnames or [])}")
        for row in reader:
            if row.get("classification") != "individual_stock":
                continue
            market = str(row.get("market") or "").strip().lower()
            code = str(row.get("security_code") or "").strip()
            if market not in {"sh", "sz"} or not re.fullmatch(r"\d{6}", code):
                raise RuntimeError(f"unexpected individual-stock roster key: {market}/{code}")
            rows.append({"market": market, "security_code": code, "code": f"{code}.{market.upper()}",
                "member_name": str(row.get("member_name") or ""), "classification_basis": str(row.get("classification_basis") or "")})
    return rows


def _read_bound_capco(source: Mapping[str, Any], model: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source.get("member") != EXPECTED_INDUSTRY_MEMBER or source.get("member_sha256") != EXPECTED_INDUSTRY_MEMBER_SHA256:
        raise RuntimeError("seasonality binding CAPCO member identity differs from frozen source")
    raw = model.read_bound_archive_member(source)
    if len(raw) != EXPECTED_INDUSTRY_MEMBER_BYTES or hashlib.sha256(raw).hexdigest() != EXPECTED_INDUSTRY_MEMBER_SHA256:
        raise RuntimeError("bound CAPCO member size/SHA differs")
    import pandas as pd  # type: ignore
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    normalized = model.normalize_industry_columns(frame)
    label_candidates = [name for name in ("industry_name", "CAPCO_NAME", "capco_name", "industry_label", "industry") if name in frame.columns]
    label_field = label_candidates[0] if label_candidates else None
    stock_name_candidates = [name for name in ("stock_name", "security_name", "company_name") if name in frame.columns]
    stock_name_field = stock_name_candidates[0] if stock_name_candidates else ("name" if "name" in frame.columns and label_field else None)
    rows = []
    for i, item in normalized.iterrows():
        original = frame.iloc[i]
        rows.append({"source_code": str(item["code"]), "industry_code": str(item["industry_code"]),
            "industry_label": str(original[label_field]) if label_field else "",
            "stock_name": str(original[stock_name_field]) if stock_name_field else "", "source_row": int(i) + 2})
    metadata = {"member": source["member"], "member_sha256": hashlib.sha256(raw).hexdigest(), "member_bytes": len(raw),
        "input_binding_as_of": next((source[key] for key in ("as_of", "as_of_date", "effective_date", "snapshot_date") if source.get(key)), None),
        "source_columns": list(frame.columns), "code_field": str(normalized.columns[0]), "category_field": str(normalized.columns[1]),
        "label_field": label_field, "stock_name_field": stock_name_field, "row_count": len(frame)}
    return rows, metadata


def _reader_canary() -> dict[str, Any]:
    code = (
        "import importlib.util, sys; import pandas as pd; "
        f"path={str(MODEL_SELECTION_PATH)!r}; "
        "spec=importlib.util.spec_from_file_location('hsjday_industry_reader_canary', path); "
        "assert spec and spec.loader; module=importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name]=module; spec.loader.exec_module(module); "
        "frame=pd.DataFrame([{'code':'000001.SZ','CAPCO':'A01'}]); "
        "result=module.normalize_industry_columns(frame); "
        "assert result.iloc[0]['code']=='000001.SZ' and result.iloc[0]['industry_code']=='A01'; "
        "print('INDUSTRY_READER_CANARY_PASS')"
    )
    import subprocess
    completed = subprocess.run([str(SELECTOR_RUNTIME), "-c", code], cwd=str(WORKTREE), capture_output=True,
        text=True, timeout=60, check=False)
    payload = {"schema_version": "HSJDAY_INDUSTRY_READER_CANARY_V1", "execution_id": EXECUTION_ID,
        "interpreter": str(SELECTOR_RUNTIME), "working_directory": str(WORKTREE),
        "synthetic_input": {"code": "000001.SZ", "CAPCO": "A01"}, "returncode": completed.returncode,
        "stdout": completed.stdout[-1200:], "stderr": completed.stderr[-1200:],
        "passed": completed.returncode == 0 and "INDUSTRY_READER_CANARY_PASS" in completed.stdout,
        "model_selection_sha256": MODEL_SELECTION_SHA256, "safety_flags": SAFETY_FLAGS}
    write_new(READER_CANARY_PATH, canonical_bytes(payload))
    if not payload["passed"]:
        raise RuntimeError(f"existing industry-reader subprocess canary failed: {completed.stderr[-500:] or completed.stdout[-500:]}")
    return payload


def _join_capco(roster: list[dict[str, str]], capco_rows: list[dict[str, Any]], meta: Mapping[str, Any]) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_key_roster: dict[str, set[str]] = defaultdict(set)
    for row in roster: raw_key_roster[row["security_code"]].add(row["code"])
    for row in capco_rows:
        source_code = row["source_code"].strip()
        by_source[source_code].append(row)
    output = []
    missing: set[str] = set()
    duplicate_source_rows = sum(max(0, len(rows) - 1) for rows in by_source.values())
    ambiguous_keys = 0
    for stock in roster:
        canonical, digits = stock["code"], stock["security_code"]
        matches = by_source.get(canonical, [])
        join_basis = "EXACT_CANONICAL_TS_CODE"
        if not matches and len(raw_key_roster[digits]) == 1:
            matches = by_source.get(digits, [])
            join_basis = "EXACT_SIX_DIGIT_CODE_UNIQUE_ACROSS_ROSTER" if matches else "NO_EXACT_CODE_MATCH"
        elif not matches and len(raw_key_roster[digits]) > 1 and by_source.get(digits):
            ambiguous_keys += 1
            matches = []
            join_basis = "AMBIGUOUS_SIX_DIGIT_CODE_ACROSS_EXCHANGES"
        if not matches or len(matches) != 1 or _blank(matches[0]["industry_code"]):
            missing.add(canonical)
            reason = "CAPCO_CODE_NOT_FOUND" if not matches else ("CAPCO_DUPLICATE_SOURCE_CODE" if len(matches) > 1 else "CAPCO_CATEGORY_BLANK")
            output.append({**stock, "name": "", "classification_system": "UNKNOWN", "industry_code": "", "industry_label": "",
                "classification_source": "", "as_of": "", "mapping_status": "UNKNOWN", "unknown_reason": reason,
                "join_basis": join_basis, "capco_source_code": "", "capco_source_row": ""})
            continue
        match = matches[0]
        output.append({**stock, "name": match["stock_name"], "classification_system": "CAPCO_CURRENT_RETROSPECTIVE",
            "industry_code": match["industry_code"], "industry_label": match["industry_label"],
            "classification_source": f"BOUND_MEMBER:{meta['member']}", "as_of": meta.get("input_binding_as_of") or "",
            "mapping_status": "MAPPED" if not _blank(match["industry_label"]) else "MAPPED_CODE_LABEL_UNKNOWN",
            "unknown_reason": "" if not _blank(match["industry_label"]) else "CAPCO_LABEL_FIELD_ABSENT_OR_EMPTY",
            "join_basis": join_basis, "capco_source_code": match["source_code"], "capco_source_row": match["source_row"]})
    return output, missing, {"duplicate_source_code_extra_rows": duplicate_source_rows, "ambiguous_six_digit_code_count": ambiguous_keys,
        "source_unique_code_count": len(by_source), "roster_canonical_code_count": len(roster)}


def _resolve_projection_fields(schema_fields: Iterable[str], requested_fields: Iterable[str]) -> dict[str, str]:
    by_casefold: dict[str, str] = {}
    ambiguous: set[str] = set()
    for field in schema_fields:
        key = str(field).casefold()
        if key in by_casefold and by_casefold[key] != field:
            ambiguous.add(key)
        else:
            by_casefold[key] = str(field)
    requested = list(requested_fields)
    ambiguous_required = [field for field in requested if field.casefold() in ambiguous]
    missing = [field for field in requested if field.casefold() not in by_casefold]
    if ambiguous_required or missing:
        raise RuntimeError(f"existing selector schema cannot resolve required projection: missing={missing!r} ambiguous={ambiguous_required!r}")
    return {field: by_casefold[field.casefold()] for field in requested}


def _selector_snapshot(missing: set[str], roster: list[dict[str, str]], pointer_lines: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selector_rows: dict[str, dict[str, Any]] = {}
    if not missing and all(row.get("name") for row in roster):
        return selector_rows, {"selector_read": False, "reason": "NO_CAPCO_ROSTER_GAPS"}
    pointer_row = next((row for row in pointer_lines if row.get("pointer_path") == str(SELECTOR_POINTER_PATH)), None)
    if not pointer_row or pointer_row.get("status") != "RESOLVED" or pointer_row.get("target_paths") != [str(SELECTOR_PATH)] or pointer_row.get("pointer_sha256") != SELECTOR_POINTER_SHA256:
        raise RuntimeError("frozen selector pointer receipt does not resolve to the expected target")
    pointer_raw = SELECTOR_POINTER_PATH.read_bytes()
    if hashlib.sha256(pointer_raw).hexdigest() != SELECTOR_POINTER_SHA256 or pointer_raw.decode("utf-8").strip() != str(SELECTOR_PATH):
        raise RuntimeError("current selector pointer content/hash differs from phase-01 resolved identity")
    stat = SELECTOR_PATH.stat()
    if stat.st_size != SELECTOR_BYTES or stat.st_mtime_ns != SELECTOR_MTIME_NS:
        raise RuntimeError("current selector target size/mtime differs from phase-01 metadata")
    from research.a_share_top100_environment_v1.hsjday_panel_dependency import _tabular_rows
    fields = ["ts_code", "trade_date", "name", "tdx_industry_code", "tdx_industry_name"]
    loaded_fields, row_iterator = _tabular_rows(SELECTOR_PATH, fields)
    selected_fields = _resolve_projection_fields(loaded_fields, fields)
    possible_codes = {row["code"] for row in roster}
    by_digits: dict[str, list[str]] = defaultdict(list)
    for row in roster:
        by_digits[row["security_code"]].append(row["code"])
    ambiguous_latest: set[str] = set()
    for row in row_iterator:
        code = str(row.get(selected_fields["ts_code"]) or "").strip()
        if code not in possible_codes and len(by_digits.get(code, [])) != 1:
            continue
        canonical = code if code in possible_codes else by_digits[code][0]
        if not canonical:
            continue
        trade_date = str(row.get(selected_fields["trade_date"]) or "").strip()
        candidate = {"trade_date": trade_date, "name": str(row.get(selected_fields["name"]) or "").strip(),
            "industry_code": str(row.get(selected_fields["tdx_industry_code"]) or "").strip(),
            "industry_label": str(row.get(selected_fields["tdx_industry_name"]) or "").strip()}
        previous = selector_rows.get(canonical)
        if previous is None or trade_date > previous["trade_date"]:
            selector_rows[canonical] = candidate
            ambiguous_latest.discard(canonical)
        elif trade_date == previous["trade_date"] and candidate != previous:
            ambiguous_latest.add(canonical)
    return selector_rows, {"selector_read": True, "selector_pointer_sha256": SELECTOR_POINTER_SHA256,
        "selector_target_path": str(SELECTOR_PATH), "selector_target_bytes": stat.st_size,
        "selector_target_mtime_ns": stat.st_mtime_ns, "selector_target_rows_from_prior_footer": SELECTOR_ROWS,
        "selector_schema_column_count": len(loaded_fields), "selector_fields_read": list(selected_fields.values()),
        "selector_fields_requested": fields, "missing_codes_requested": len(missing), "zip_codes_for_name_lookup": len(roster),
        "missing_codes_with_any_snapshot_row": len(missing & selector_rows.keys()), "zip_codes_with_any_snapshot_row": len(selector_rows),
        "ambiguous_latest_code_count": len(ambiguous_latest), "ambiguous_latest_codes": sorted(ambiguous_latest)}


def _output_manifest(status: str) -> dict[str, Any]:
    artifacts = [p for p in sorted(OUTPUT_DIR.iterdir()) if p.is_file() and p.name != "OUTPUT_MANIFEST.json"]
    manifest = {"schema_version": "HSJDAY_INDUSTRY_COVERAGE_OUTPUT_MANIFEST_V1", "experiment_id": EXPERIMENT_ID,
        "execution_id": EXECUTION_ID, "route_id": ROUTE_ID, "status": status,
        "safety_flags": SAFETY_FLAGS,
        "artifacts": [{"path": p.relative_to(BATCH).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in artifacts],
        "manifest_scope": "Execution outputs only; source archive member and runtime selector remain read-only."}
    write_new(OUTPUT_MANIFEST_PATH, canonical_bytes(manifest))
    return manifest


def run() -> dict[str, Any]:
    contract, config, binding = verify_frozen()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env, adapter, core = _load_registry_helper(binding["shared_dependencies"])
    context = _context()
    claim = _claim(env, adapter, core, context)
    started = utc_now()
    append_jsonl(PROGRESS_PATH, {"sequence": 0, "phase": "RUN_STARTED", "execution_id": EXECUTION_ID,
        "occurred_at_utc": started, "contract_sha256": context["contract_sha256"]})
    status, failure = "CANNOT_CONFIRM", None
    metrics: dict[str, Any] = {"truth_positive_denominator": None, "selection_slots": 0, "fit_call_entered": False}
    seasonality_binding_sha: str | None = None
    try:
        seasonality_binding_sha = sha256_file(SEASONALITY_BINDING_PATH)
        seasonality = read_json(SEASONALITY_BINDING_PATH)
        source = seasonality.get("sources", {}).get("industry_classification")
        if not isinstance(source, dict):
            raise RuntimeError("seasonality input binding lacks sources.industry_classification")
        if sha256_file(MODEL_SELECTION_PATH) != seasonality.get("shared_dependencies", {}).get("model_selection", {}).get("sha256"):
            raise RuntimeError("existing industry reader SHA does not match seasonality binding")
        if sha256_file(MODEL_SELECTION_PATH) != MODEL_SELECTION_SHA256:
            raise RuntimeError("existing industry reader differs from frozen SHA")
        canary = _reader_canary()
        metrics.update({"reader_canary_passed": canary["passed"], "reader_canary_returncode": canary["returncode"]})
        _append_progress(adapter, "EXISTING_INDUSTRY_READER_CANARY_PASSED", {"reader_canary_passed": True, "reader_canary_returncode": canary["returncode"]})
        helper_spec = importlib.util.spec_from_file_location("hsjday_bound_model_selection", MODEL_SELECTION_PATH)
        if helper_spec is None or helper_spec.loader is None:
            raise ImportError("cannot load existing bound industry reader")
        model = importlib.util.module_from_spec(helper_spec)
        sys.modules[helper_spec.name] = model
        helper_spec.loader.exec_module(model)
        roster = _read_roster()
        capco_rows, capco_meta = _read_bound_capco(source, model)
        roster_rows, missing_capco, capco_stats = _join_capco(roster, capco_rows, capco_meta)
        metrics.update({"seasonality_binding_sha256": seasonality_binding_sha, "industry_member_sha256_verified": capco_meta["member_sha256"],
            "model_selection_reader_sha256": MODEL_SELECTION_SHA256})
        _append_progress(adapter, "CAPCO_MEMBER_MAPPED_TO_ZIP_ROSTER", {"zip_individual_stock_count": len(roster),
            "capco_member_row_count": capco_meta["row_count"], "capco_unique_source_codes": capco_stats["source_unique_code_count"],
            "capco_matched_codes": len(roster) - len(missing_capco), "capco_missing_codes": len(missing_capco)})
        runtime_pointer_receipt = _load_jsonl(RUNTIME_POINTERS_PATH)
        selector_rows, selector_meta = _selector_snapshot(missing_capco, roster_rows, runtime_pointer_receipt)
        ambiguous_latest = set(selector_meta.get("ambiguous_latest_codes", []))
        roster_names = {row["code"]: row["name"] for row in roster_rows}
        supplemented = 0
        for row in roster_rows:
            code = row["code"]
            candidate = selector_rows.get(code)
            if candidate:
                row["name"] = row["name"] or candidate["name"]
            if code not in missing_capco:
                continue
            if candidate and candidate["industry_code"] and code not in ambiguous_latest:
                row.update({"name": row["name"] or candidate["name"], "classification_system": "TDX_INDUSTRY_CURRENT",
                    "industry_code": candidate["industry_code"], "industry_label": candidate["industry_label"],
                    "classification_source": str(SELECTOR_PATH), "as_of": candidate["trade_date"],
                    "mapping_status": "SUPPLEMENTED_FROM_EXISTING_SELECTOR" if candidate["industry_label"] else "SUPPLEMENTED_CODE_LABEL_UNKNOWN",
                    "unknown_reason": "" if candidate["industry_label"] else "TDX_INDUSTRY_LABEL_EMPTY"})
                supplemented += 1
            else:
                if candidate:
                    row["name"] = row["name"] or candidate["name"]
                if code in ambiguous_latest:
                    row["unknown_reason"] = "AMBIGUOUS_LATEST_SELECTOR_INDUSTRY_ROW"
                elif not candidate:
                    row["unknown_reason"] = "NO_MATCH_IN_CAPCO_OR_SELECTOR"
                elif not candidate["industry_code"]:
                    row["unknown_reason"] = "NO_TDX_INDUSTRY_CODE_IN_SELECTOR"
        unknown_count = sum(row["classification_system"] == "UNKNOWN" for row in roster_rows)
        code_only_count = sum(row["mapping_status"].endswith("LABEL_UNKNOWN") or row["mapping_status"] == "MAPPED_CODE_LABEL_UNKNOWN" for row in roster_rows)
        counts = Counter((row["classification_system"], row["industry_code"], row["industry_label"]) for row in roster_rows if row["classification_system"] != "UNKNOWN")
        classification_rows = [{"classification_system": system, "industry_code": code, "industry_label": label, "stock_count": count}
            for (system, code, label), count in sorted(counts.items())]
        gap_rows = [{key: row.get(key, "") for key in ("code", "market", "security_code", "name", "classification_system", "industry_code", "industry_label", "classification_source", "as_of", "mapping_status", "unknown_reason", "join_basis")}
            for row in roster_rows if row["classification_system"] == "UNKNOWN" or row["unknown_reason"]]
        summary = {"schema_version": "HSJDAY_INDUSTRY_SOURCE_COVERAGE_V1", "status": "CURRENT_RETROSPECTIVE_MAPPED_WITH_UNKNOWN" if unknown_count else "CURRENT_RETROSPECTIVE_MAPPED",
            "execution_id": EXECUTION_ID, "classification_lock": "CURRENT_CLASSIFICATION_RETROSPECTIVE",
            "zip_individual_stock_count": len(roster), "capco_source_row_count": capco_meta["row_count"],
            "capco_unique_source_code_count": capco_stats["source_unique_code_count"], "capco_matched_roster_code_count": len(roster) - len(missing_capco),
            "capco_missing_roster_code_count": len(missing_capco), "tdx_supplemented_missing_codes": supplemented,
            "remaining_unknown_codes": unknown_count, "codes_with_unknown_or_empty_label": code_only_count,
            "classification_counts_by_system": {system: sum(v for (s, _, _), v in counts.items() if s == system) for system in sorted({k[0] for k in counts})},
            "capco_source_metadata": capco_meta, "capco_integrity": capco_stats, "selector_source_metadata": selector_meta,
            "unknown_reason_counts": dict(Counter(row["unknown_reason"] for row in roster_rows if row["unknown_reason"])),
            "limitations": ["Current retrospective classifications are not historical PIT industry truth.", "CAPCO and TDX labels remain separate taxonomies.", "ZIP prefix classification is not proof of historical issuer status; BJ unknown instruments are excluded from this A-share roster."],
            "safety_flags": SAFETY_FLAGS}
        write_new(OUTPUT_DIR / "INDUSTRY_SOURCE_COVERAGE.json", canonical_bytes(summary))
        _append_csv(OUTPUT_DIR / "INDUSTRY_ROSTER.csv", roster_rows, ["code", "market", "security_code", "member_name", "classification_basis", "name", "classification_system", "industry_code", "industry_label", "classification_source", "as_of", "mapping_status", "unknown_reason", "join_basis", "capco_source_code", "capco_source_row"])
        _append_csv(OUTPUT_DIR / "INDUSTRY_GAPS.csv", gap_rows, ["code", "market", "security_code", "name", "classification_system", "industry_code", "industry_label", "classification_source", "as_of", "mapping_status", "unknown_reason", "join_basis"])
        _append_csv(OUTPUT_DIR / "INDUSTRY_COUNTS_BY_SYSTEM.csv", classification_rows, ["classification_system", "industry_code", "industry_label", "stock_count"])
        metrics.update({"zip_individual_stock_count": len(roster), "capco_matched_roster_code_count": len(roster) - len(missing_capco),
            "tdx_supplemented_missing_codes": supplemented, "remaining_unknown_codes": unknown_count, "classification_system_count": len({row["classification_system"] for row in roster_rows if row["classification_system"] != "UNKNOWN"})})
        _append_progress(adapter, "INDUSTRY_COVERAGE_TERMINAL_ARTIFACTS_WRITTEN", metrics)
        _update_context = dict(context)
        _update_context["input_component_sha256"] = {**_update_context["input_component_sha256"],
            "seasonality_binding_sha256": seasonality_binding_sha,
            "seasonality_member_sha256_verified": EXPECTED_INDUSTRY_MEMBER_SHA256,
            "model_selection_reader_sha256": MODEL_SELECTION_SHA256,
            "selector_pointer_sha256": SELECTOR_POINTER_SHA256 if selector_meta.get("selector_read") else None}
        _update_context["artifact_paths"] = [*_update_context["artifact_paths"], str(OUTPUT_DIR / "INDUSTRY_SOURCE_COVERAGE.json"),
            str(OUTPUT_DIR / "INDUSTRY_ROSTER.csv"), str(OUTPUT_DIR / "INDUSTRY_GAPS.csv"), str(OUTPUT_DIR / "INDUSTRY_COUNTS_BY_SYSTEM.csv"), str(OUTPUT_DIR / "ROUTE_CONTEXT.json")]
        _update_context["deterministic_metrics"] = {**_update_context["deterministic_metrics"], **metrics}
        adapter.set_route_context(EXECUTION_ID, _update_context)
        write_new(OUTPUT_DIR / "ROUTE_CONTEXT.json", canonical_bytes(_update_context))
    except BaseException as exc:
        failure = exc
        metrics.update({"failure_type": type(exc).__name__, "failure_message": str(exc)[:900]})
        write_new(OUTPUT_DIR / "FAILURE.json", canonical_bytes({"execution_id": EXECUTION_ID, "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:900], "occurred_at_utc": utc_now(), "safety_flags": SAFETY_FLAGS}))
        failure_context = dict(adapter._route_context[EXECUTION_ID])
        failure_context["artifact_paths"] = [*failure_context.get("artifact_paths", []), str(OUTPUT_DIR / "FAILURE.json"), str(OUTPUT_DIR / "ROUTE_CONTEXT.json")]
        failure_context["deterministic_metrics"] = {**failure_context.get("deterministic_metrics", {}), **metrics}
        adapter.set_route_context(EXECUTION_ID, failure_context)
        write_new(OUTPUT_DIR / "ROUTE_CONTEXT.json", canonical_bytes(failure_context))
    receipt = {"schema_version": "HSJDAY_INDUSTRY_COVERAGE_RUN_RECEIPT_V1", "execution_id": EXECUTION_ID,
        "experiment_id": EXPERIMENT_ID, "route_id": ROUTE_ID, "run_started_at_utc": started, "run_finished_at_utc": utc_now(),
        "status": status, "contract_sha256": sha256_file(CONTRACT_PATH), "config_sha256": sha256_file(CONFIG_PATH),
        "input_binding_sha256": sha256_file(INPUT_BINDING_PATH), "code_sha256": sha256_file(SCRIPT_PATH),
        "seasonality_binding_sha256": seasonality_binding_sha,
        "seasonality_binding_sha256_null_reason": None if seasonality_binding_sha else "BINDING_UNAVAILABLE_BEFORE_TERMINAL",
        "roster_sha256": ROSTER_SHA256,
        "industry_member_sha256_bound": EXPECTED_INDUSTRY_MEMBER_SHA256,
        "industry_member_sha256_verified": metrics.get("industry_member_sha256_verified"), "metrics": metrics,
        "failure": None if failure is None else {"type": type(failure).__name__, "message": str(failure)[:900], "failure_type": "CANNOT_CONFIRM"},
        "truth_access": "NOT_READ", "sealed_status": "NOT_ENTERED", "safety_flags": SAFETY_FLAGS}
    write_new(RUN_RECEIPT_PATH, canonical_bytes(receipt))
    _registry_terminal(adapter, claim, status, metrics, failure)
    close = _closeout()
    manifest = _output_manifest(status)
    return {"status": status, "metrics": metrics, "registry_closeout": close,
        "output_manifest_sha256": sha256_file(OUTPUT_MANIFEST_PATH), "output_count": len(manifest["artifacts"])}


def _update_context(adapter: Any, context: dict[str, Any], components: Mapping[str, Any], metrics: Mapping[str, Any], extra_paths: Iterable[Path] = ()) -> None:
    context = dict(context)
    context["input_component_sha256"] = {**context.get("input_component_sha256", {}), **dict(components)}
    context["artifact_paths"] = [*context.get("artifact_paths", []), *(str(p) for p in extra_paths)]
    context["deterministic_metrics"] = {**context.get("deterministic_metrics", {}), **dict(metrics)}
    adapter.set_route_context(EXECUTION_ID, context)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run: parser.error("choose exactly one of --freeze or --run")
    if args.freeze:
        freeze_contract(); return 0
    result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "CANNOT_CONFIRM" else 1


if __name__ == "__main__":
    raise SystemExit(main())
