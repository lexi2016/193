"""Local diagnostic lifecycle through the existing official locked adapter.

Import has no registry, data, SDK, or network side effects.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

FLAGS = dict(research_only=True, production_allowed=False,
             buy_gate_updated=False, orders_sent=False)
EXPERIMENT = "A_SHARE_ENVIRONMENT_TILT_MAC_DIAGNOSTIC_20261021"
ROUTE = "a-share-environment-tilt-mac-diagnostic-20261021"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(event):
    return hashlib.sha256(canonical({k: v for k, v in event.items()
                                    if k != "event_sha256"}).encode()).hexdigest()


def load_adapter(config):
    sys.dont_write_bytecode = True
    deps = config["registry_dependencies"]
    for item in deps.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("REGISTRY_DEPENDENCY_DRIFT")
    adapter_path = Path(deps["registry_adapter"]["path"]).resolve()
    core = sys.modules.get("execution_guard_core")
    if core and Path(core.__file__).resolve() != Path(deps["execution_guard_core"]["path"]).resolve():
        raise ValueError("REGISTRY_IMPORT_COLLISION")
    sys.path.insert(0, str(adapter_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("tilt_mac_official_adapter", adapter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.OfficialRegistryAdapter(registry_path=Path(config["registry_path"]),
        contract_path=Path(config["contract_path"]), input_binding_path=Path(config["config_path"]),
        builder_path=Path(deps["registry_builder"]["path"]),
        validator_path=Path(deps["registry_validator"]["path"]))


class Lifecycle:
    def __init__(self, adapter, config):
        self.adapter, self.config = adapter, config
        self.started = self.finished = False

    def event(self, kind, decision, receipt, metrics):
        c = self.config
        return dict(schema_version="1.0", registry_event_id=c["execution_id"] + "_" + kind,
            route_id=ROUTE, experiment_id=EXPERIMENT, execution_id=c["execution_id"],
            track_id="A_SHARE_PANEL_MODEL_SELECTION_RESEARCH", stage="MAC_RAW_ACTION_QFQ_DIAGNOSTIC",
            event_type=kind, registration_mode="LIVE_EXECUTION", arm="FIXED_SIX_LOCAL_SOURCE",
            occurred_at=datetime.now(timezone.utc).isoformat(), occurred_at_reason="LIVE_EXECUTION_UTC",
            contract_path=c["contract_path"], contract_sha256=c["contract_sha256"],
            input_binding_sha256=c["config_sha256"], input_sha256=c["config_sha256"],
            code_sha256=c["code_composite_sha256"], config_sha256=c["config_sha256"],
            baseline_id="PV_ENVIRONMENT_R1_M0_FORMAL_EXACT_4THREAD",
            one_variable="LOCAL_RAW_ACTION_QFQ_SEMANTIC_DIAGNOSTIC_NO_PREFIX_EXTENSION",
            signal_date_range=c["diagnostic_dates"],
            target_date_range={"count": 0, "scope": "NO_OUTCOME_LABELS"},
            truth_access="NOT_READ" if kind == "RUN_STARTED" else
                "NO_OUTCOME_TRUTH_SOURCE_IDENTITY_PROBE_ONLY_NO_LABELS_OR_SCORES",
            sealed_status="NOT_ENTERED", runtime=dict(python=sys.executable,
                version=platform.python_version(), active_concurrency_limit=c["caps"]["threads"],
                arrow_pools=c.get("arrow_pools"), arrow_pools_reason=None if c.get("arrow_pools")
                else "NOT_CONFIGURED_BEFORE_SOURCE_READS"), device="CPU_RESEARCH_ONLY",
            artifact_paths=[c["contract_path"], c["config_path"], str(receipt)], checkpoint_ref=None,
            deterministic_metrics=dict(truth_positive_denominator=None, selection_slots=None,
                execution_kind="NO_FIT_ENGINEERING", planned_real_fit_count=0,
                real_fit_budget_debit=0, real_fit_returned_count=0, fit_call_entered=False, **metrics),
            decision=decision, stop_reason="NONE" if kind == "RUN_STARTED" else decision,
            evidence_level="ENGINEERING_CANARY_ONLY",
            input_identity=dict(input_binding_sha256=c["config_sha256"],
                null_reasons={"checkpoint_ref": "NO_MODEL", "truth_positive_denominator": "NO_CLASSIFICATION",
                              "selection_slots": "NO_SELECTION"}), **FLAGS)

    def start(self, receipt):
        if self.started:
            raise PermissionError("DUPLICATE_START")
        self.adapter.append_parity_event(self.event("RUN_STARTED", "RUNNING", receipt, {}),
                                         execution_id=self.config["execution_id"])
        self.started = True
        self.require()

    def require(self):
        if not self.started or self.finished:
            raise PermissionError("REGISTERED_ACTIVE_RUN_REQUIRED")
        # Reuse the official adapter's same OS lock and validated raw event view.
        with self.adapter._locked():
            _, events = self.adapter._read_validated()
            same = [e for e in events if e.get("execution_id") == self.config["execution_id"]]
            starts = [e for e in same if e.get("event_type") == "RUN_STARTED"]
            if len(starts) != 1 or any(e.get("event_type") == "TERMINAL" for e in same):
                raise PermissionError("DURABLE_ACTIVE_START_REQUIRED")
            e = starts[0]
            if (digest(e) != e["event_sha256"] or e.get("contract_sha256") != self.config["contract_sha256"]
                    or e.get("input_binding_sha256") != self.config["config_sha256"]):
                raise PermissionError("DURABLE_START_IDENTITY_MISMATCH")
            return e

    def finish(self, receipt, decision, metrics=None):
        if not self.started or self.finished:
            raise PermissionError("ACTIVE_START_REQUIRED_FOR_TERMINAL")
        self.adapter.append_parity_event(self.event("TERMINAL", decision, receipt, metrics or {}),
                                         execution_id=self.config["execution_id"])
        self.finished = True
