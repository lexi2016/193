"""Review-gated, one-off recovery of SH.000300 daily history through Futu.

The runner has two separately registered phases.  ``canary`` requests at most
five bars; ``full`` refuses to run without the exact successful canary receipt.
It never calls a model, account, trade, or snapshot API and does not claim
official exchange-calendar parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


ROUTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTE_ROOT.parents[2]
BATCH_ROOT = PROJECT_ROOT / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
CONTRACT_PATH = BATCH_ROOT / "FUTU_INDEX_HISTORY_RECOVERY_CONTRACT.json"
INPUT_BINDING_PATH = BATCH_ROOT / "FUTU_INDEX_HISTORY_RECOVERY_INPUT_BINDING.json"
ACTIVATION_PATH = BATCH_ROOT / "FUTU_INDEX_HISTORY_RECOVERY_ACTIVATION.json"
OUTPUT_ROOT = BATCH_ROOT / "index_recovery_futu"
REGISTRY_PATH = PROJECT_ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
REGISTRY_ADAPTER_DIR = PROJECT_ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"

EXPERIMENT_ID = "A_SHARE_INDEX_FUTU_HISTORY_RECOVERY_PV_ENVIRONMENT_R1_20261019"
ROUTE_ID = "a-share-index-futu-history-recovery-pv-environment-r1-20261019"
INDEX_CODE = "SH.000300"
LOCAL_OPEND_HOST = "127.0.0.1"
LOCAL_OPEND_PORT = 11111
CANARY_START, CANARY_END = "2021-01-04", "2021-01-08"
FULL_START, FULL_END = "2021-01-01", "2026-10-15"
CANARY_COUNT = 5
FULL_PAGE_SIZE = 1000
FULL_MAX_PAGES = 3
EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")
RAW_COLUMNS = ("code", "time_key", "open", "close", "high", "low", "volume", "turnover",
               "change_rate", "last_close")
CANONICAL_COLUMNS = ("date", "code", "open", "high", "low", "close", "volume", "turnover")


class RecoveryBlocked(RuntimeError):
    """Frozen source, review activation, or phase prerequisite is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryBlocked(f"required control file is missing: {Path(path).name}") from exc
    except json.JSONDecodeError as exc:
        raise RecoveryBlocked(f"control JSON is invalid: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise RecoveryBlocked(f"expected JSON object: {Path(path).name}")
    return value


def _bound_path(batch_root: Path, binding: Mapping[str, Any], name: str) -> Path:
    entry = binding.get("source_bindings", {}).get(name)
    if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
        raise RecoveryBlocked(f"input binding is missing source identity: {name}")
    path = (batch_root / str(entry["path"])).resolve()
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise RecoveryBlocked(f"bound source is missing or drifted: {name}")
    return path


def validate_source_bindings(*, contract_path: Path, input_binding_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    contract_path, input_binding_path = Path(contract_path).resolve(), Path(input_binding_path).resolve()
    contract, binding = read_json(contract_path), read_json(input_binding_path)
    if contract.get("status") != "FROZEN_AWAITING_PRE_RUN_REVIEW":
        raise RecoveryBlocked("history-recovery contract is not frozen for review")
    if binding.get("status") != "STATIC_SOURCE_BINDING_READY":
        raise RecoveryBlocked("history-recovery input binding is not ready")
    if binding.get("experiment_id") != EXPERIMENT_ID or binding.get("route_id") != ROUTE_ID:
        raise RecoveryBlocked("history-recovery input binding identifies another route")
    batch_root = contract_path.parent
    source_paths = {
        name: _bound_path(batch_root, binding, name)
        for name in ("model_contract", "model_input_binding", "capability_probe_contract",
                     "capability_probe_input_binding", "capability_probe_receipt")
    }
    for name, path in source_paths.items():
        contract_ref = contract.get("source_bindings", {}).get(name)
        bound = binding.get("source_bindings", {}).get(name)
        if (not isinstance(contract_ref, Mapping) or not isinstance(bound, Mapping)
                or any(contract_ref.get(key) != bound.get(key) for key in ("path", "sha256"))):
            raise RecoveryBlocked(f"contract/input binding source mismatch: {name}")
    probe_contract = read_json(source_paths["capability_probe_contract"])
    probe_receipt = read_json(source_paths["capability_probe_receipt"])
    if probe_contract.get("status") != "FROZEN_AWAITING_PRE_RUN_REVIEW":
        raise RecoveryBlocked("source Futu capability contract is not frozen")
    stock = probe_receipt.get("stock_metadata", {})
    quota = probe_receipt.get("history_kl_quota", {})
    if (probe_receipt.get("status") != "CAPABILITY_CONFIRMED"
            or stock.get("code") != INDEX_CODE or stock.get("status") != "API_OK"
            or stock.get("security_type") != "IDX" or quota.get("status") != "API_OK"):
        raise RecoveryBlocked("bound Futu capability receipt does not confirm SH.000300 index history capability")
    if contract.get("api_scope", {}).get("history_request", {}).get("code") != INDEX_CODE:
        raise RecoveryBlocked("contract does not pin the exact index identifier")
    return contract, binding, source_paths


def validate_activation(*, contract: Mapping[str, Any], activation: Mapping[str, Any],
                        contract_sha256: str, binding_sha256: str, code_sha256: str,
                        activation_path: Path) -> None:
    if activation.get("status") != "APPROVED_FUTU_HISTORY_CANARY_AND_CONDITIONAL_FULL":
        raise RecoveryBlocked("independent Futu history activation is absent")
    writer = str(contract.get("writer_identity", ""))
    reviewer = str(activation.get("reviewer_identity", ""))
    if not writer or not reviewer or writer == reviewer:
        raise RecoveryBlocked("activation must identify an independent reviewer")
    expected = {"contract_sha256": contract_sha256, "input_binding_sha256": binding_sha256,
                "code_sha256": code_sha256}
    if any(activation.get(key) != value for key, value in expected.items()):
        raise RecoveryBlocked("activation hashes do not match the reviewed history-recovery inputs")
    receipt_path, receipt_sha = activation.get("reviewer_receipt_path"), activation.get("reviewer_receipt_sha256")
    if not receipt_path or not receipt_sha:
        raise RecoveryBlocked("activation must identify the independent reviewer receipt")
    receipt = Path(str(receipt_path)).expanduser().resolve()
    if not receipt.is_file() or sha256_file(receipt) != receipt_sha:
        raise RecoveryBlocked("independent reviewer receipt is missing or drifted")
    if activation.get("allow_history_canary") is not True:
        raise RecoveryBlocked("activation must explicitly allow the one-time five-bar history canary")
    if activation.get("allow_full_history_after_canary") is not True:
        raise RecoveryBlocked("activation must explicitly allow conditional full history after canary")
    for key in ("allow_snapshot_request", "allow_account_or_trade_api"):
        if activation.get(key) is not False:
            raise RecoveryBlocked(f"activation must keep {key} closed")
    if activation_path.resolve() == receipt:
        raise RecoveryBlocked("reviewer receipt cannot be the activation document itself")


def _quota_snapshot(response: Any, *, ret_ok: Any) -> dict[str, Any]:
    result = {"status": "INVALID_RESPONSE", "used_quota": None, "remain_quota": None, "quota_total": None}
    if not isinstance(response, tuple) or len(response) != 2:
        return result
    ret_code, payload = response
    if ret_code != ret_ok:
        result["status"] = "API_ERROR"
        return result
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        return result
    values: list[int] = []
    for raw in payload[:2]:
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError):
            return result
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            return result
        values.append(int(number))
    result.update(status="API_OK", used_quota=values[0], remain_quota=values[1],
                  quota_total=values[0] + values[1])
    return result


def _api_page(response: Any, *, ret_ok: Any) -> tuple[str, pd.DataFrame | None, Any, str | None]:
    if not isinstance(response, tuple) or len(response) != 3:
        return "INVALID_RESPONSE", None, None, "request_history_kline must return a 3-tuple"
    ret_code, data, page_req_key = response
    if ret_code != ret_ok:
        message = str(data)[:500] if data is not None else "history API returned non-RET_OK"
        return "API_ERROR", None, None, message
    if not isinstance(data, pd.DataFrame):
        return "INVALID_RESPONSE", None, None, "history API data was not a DataFrame"
    return "API_OK", data.copy(), page_req_key, None


def _normalize_page(data: pd.DataFrame, *, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = sorted(set(("code", "time_key", "open", "high", "low", "close")) - set(data.columns))
    if missing:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), {"status": "INVALID_RESPONSE", "missing_columns": missing}
    frame = data.copy()
    frame["code"] = frame["code"].astype(str)
    parsed = pd.to_datetime(frame["time_key"], errors="coerce")
    frame["date"] = parsed.dt.strftime("%Y-%m-%d")
    frame["date"] = frame["date"].fillna("")
    numeric = [column for column in ("open", "high", "low", "close", "volume", "turnover") if column in frame]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    result = pd.DataFrame({column: frame[column] if column in frame else pd.NA for column in CANONICAL_COLUMNS})
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    date_values = pd.to_datetime(result["date"], errors="coerce")
    code_valid = result["code"].eq(INDEX_CODE)
    date_valid = date_values.notna() & date_values.between(start_ts, end_ts)
    prices = result[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    price_valid = prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
    ohlc_valid = (prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
                  & prices["low"].le(prices[["open", "close", "high"]].min(axis=1)))
    valid = code_valid & date_valid & price_valid & ohlc_valid
    summary = {
        "status": "VALID" if bool(valid.all()) and len(result) > 0 else "INVALID_ROWS",
        "row_count": int(len(result)),
        "invalid_row_count": int((~valid).sum()),
        "invalid_code_count": int((~code_valid).sum()),
        "invalid_date_count": int((~date_valid).sum()),
        "invalid_ohlc_count": int((~(price_valid & ohlc_valid)).sum()),
    }
    if result.empty:
        summary["status"] = "EMPTY"
    normalized = result.loc[valid, list(CANONICAL_COLUMNS)].copy()
    normalized = normalized.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume", "turnover"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized, summary


def _date_gap_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    dates = sorted(set(str(value) for value in frame.get("date", pd.Series(dtype=str)) if value))
    intervals: list[dict[str, Any]] = []
    for before, after in zip(dates, dates[1:]):
        gap = (pd.Timestamp(after) - pd.Timestamp(before)).days
        if gap > 1:
            intervals.append({"from": before, "to": after, "calendar_days_between": gap - 1})
    return {
        "status": "CANNOT_CONFIRM_SESSION_GAPS_WITHOUT_BOUND_CANONICAL_CALENDAR",
        "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        "observed_date_count": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "nonconsecutive_calendar_date_intervals": intervals,
        "interval_semantics": "Diagnostic spans only; weekends and exchange holidays are not asserted to be missing sessions.",
    }


def _history_guard(*, contract_path: Path, binding_path: Path, registry_path: Path,
                   artifact_paths: Sequence[Path], stage: str):
    sys.path.insert(0, str(REGISTRY_ADAPTER_DIR))
    from execution_guard_core import ExecutionGuard
    from registry_adapter import OfficialRegistryAdapter

    class HistoryRegistryAdapter(OfficialRegistryAdapter):
        def __init__(self, **kwargs: Any) -> None:
            self._contexts: dict[str, dict[str, Any]] = {}
            super().__init__(**kwargs)

        def set_execution_context(self, execution_id: str, context: Mapping[str, Any]) -> None:
            if not execution_id:
                raise RecoveryBlocked("registry context requires an execution id")
            self._contexts[str(execution_id)] = dict(context)

        def update_execution_context(self, execution_id: str, **updates: Any) -> None:
            if str(execution_id) not in self._contexts:
                raise RecoveryBlocked(f"registry context was not initialized: {execution_id}")
            self._contexts[str(execution_id)].update(updates)

        def _event_spec(self, event: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
            spec = super()._event_spec(event, event_type=event_type)
            execution_id = str(event.get("execution_id", ""))
            context = self._contexts.get(execution_id)
            if context is None:
                raise RecoveryBlocked(f"refusing un-enriched registry event: {execution_id}")
            for field in ("truth_access", "sealed_status", "signal_date_range", "target_date_range",
                          "runtime", "device", "checkpoint_ref", "code_sha256", "config_sha256",
                          "input_sha256"):
                if field in context:
                    spec[field] = context[field]
            spec["input_paths"] = list(context.get("input_paths", ()))
            paths = list(spec.get("artifact_paths") or [])
            paths.extend(str(value) for value in context.get("artifact_paths", ()))
            spec["artifact_paths"] = list(dict.fromkeys(paths))
            spec["artifact_path"] = spec["artifact_paths"]
            identity = dict(spec.get("input_identity") or {})
            identity.update(dict(context.get("input_identity") or {}))
            metrics = dict(spec.get("deterministic_metrics") or {})
            metrics.update(dict(context.get("deterministic_metrics") or {}))
            spec["input_identity"], spec["deterministic_metrics"] = identity, metrics
            return spec

    profile = {
        "track_id": "A_SHARE_PANEL_MODEL_SELECTION_RESEARCH",
        "route_id": ROUTE_ID,
        "registration_mode": "LIVE_EXECUTION",
        "start_stage": f"FUTU_HISTORY_{stage.upper()}_RUN_STARTED",
        "terminal_stage": f"FUTU_HISTORY_{stage.upper()}_TERMINAL",
        "occurred_at_reason": "LIVE_FUTU_OPEND_READ_ONLY_INDEX_HISTORY_UTC",
        "baseline_id": "FUTU_SH_000300_DAILY_NONE_BOUNDED_HISTORY_RECOVERY",
        "one_variable": "FUTU_SH_000300_HISTORICAL_INDEX_SERIES_RETRIEVAL",
        "signal_scope": "NO_MODEL_SIGNAL_IN_SOURCE_RECOVERY",
        "target_scope": "NO_LABEL_OR_TARGET_IN_SOURCE_RECOVERY",
        "device": "LOCAL_OPEND_QUOTE_CONTEXT_NO_MODEL_FIT",
        "evidence_level": "ENGINEERING_CANARY_ONLY",
    }
    backend = HistoryRegistryAdapter(registry_path=registry_path, contract_path=contract_path,
                                     input_binding_path=binding_path, artifact_paths=artifact_paths,
                                     event_profile=profile)
    return ExecutionGuard(backend, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def _registry_context(*, contract_sha: str, binding_sha: str, code_sha: str,
                      contract_path: Path, binding_path: Path, source_paths: Mapping[str, Path],
                      artifact_paths: Sequence[Path], phase: str, status: str,
                      quota: Mapping[str, Any] | None = None,
                      row_count: int | None = None, page_count: int | None = None,
                      normalized_sha: str | None = None) -> dict[str, Any]:
    return {
        "truth_access": "FUTU_HISTORICAL_INDEX_MARKET_DATA_ONLY_NO_LABELS",
        "sealed_status": "NOT_ENTERED_SOURCE_RECOVERY_ONLY",
        "signal_date_range": {"count": 0, "scope": "NO_MODEL_SIGNAL_IN_SOURCE_RECOVERY"},
        "target_date_range": {"count": 0, "scope": "NO_LABEL_OR_TARGET_IN_SOURCE_RECOVERY"},
        "runtime": {"python": sys.executable, "python_version": sys.version.split()[0],
                    "pandas": pd.__version__},
        "device": "LOCAL_OPEND_QUOTE_CONTEXT_NO_MODEL_FIT",
        "checkpoint_ref": None,
        "code_sha256": code_sha,
        "config_sha256": contract_sha,
        "input_sha256": binding_sha,
        "input_paths": [str(Path(contract_path).resolve()), str(Path(binding_path).resolve()),
                        *[str(path) for path in source_paths.values()]],
        "artifact_paths": [str(Path(path).resolve()) for path in artifact_paths],
        "deterministic_metrics": {
            "truth_positive_denominator": None,
            "selection_slots": None,
            "phase": phase,
            "status": status,
            "index_bar_count": row_count,
            "api_page_count": page_count,
            "history_kline_request_count": page_count,
            "quota_total": quota.get("quota_total") if quota else None,
            "quota_used": quota.get("used_quota") if quota else None,
            "quota_remaining": quota.get("remain_quota") if quota else None,
            "normalized_sha256": normalized_sha,
            "fit_call_count": 0,
            "snapshot_request_count": 0,
            "account_or_trade_api_call_count": 0,
        },
        "input_identity": {
            "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha,
            "code_sha256": code_sha,
            "source_bindings": {name: {"path": str(path), "sha256": sha256_file(path)}
                                for name, path in source_paths.items()},
            "null_reasons": {
                "deterministic_metrics.truth_positive_denominator": "NO_LABELS_IN_SOURCE_RECOVERY",
                "deterministic_metrics.selection_slots": "NO_SELECTION_IN_SOURCE_RECOVERY",
                "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_SOURCE_RECOVERY",
            },
        },
    }


def _validate_canary_receipt(path: Path, *, output_root: Path, expected_sha256: str,
                             contract_sha256: str, binding_sha256: str) -> dict[str, Any]:
    path = Path(path).resolve()
    output_root = Path(output_root).resolve()
    if output_root not in path.parents:
        raise RecoveryBlocked("canary receipt must be under this batch's Futu recovery output root")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RecoveryBlocked("exact canary receipt is missing or its SHA does not match")
    receipt = read_json(path)
    if (receipt.get("phase") != "canary" or receipt.get("status") != "CANARY_PASS"
            or receipt.get("contract_sha256") != contract_sha256
            or receipt.get("input_binding_sha256") != binding_sha256
            or receipt.get("code_sha256") != sha256_file(Path(__file__).resolve())
            or receipt.get("index_code") != INDEX_CODE
            or receipt.get("requested_bounds") != {"start": CANARY_START, "end": CANARY_END}
            or receipt.get("api_plan", {}).get("max_count_per_page") != CANARY_COUNT
            or receipt.get("page_count", 0) != 1
            or receipt.get("row_validation", {}).get("status") != "VALID"
            or receipt.get("normalized_sha256") is None
            or receipt.get("normalized_row_count", 0) <= 0):
        raise RecoveryBlocked("canary receipt does not prove the frozen valid five-bar phase")
    normalized_name = receipt.get("normalized_path")
    if not normalized_name:
        raise RecoveryBlocked("canary receipt lacks its normalized data artifact")
    normalized_path = path.parent / str(normalized_name)
    if not normalized_path.is_file() or sha256_file(normalized_path) != receipt["normalized_sha256"]:
        raise RecoveryBlocked("canary normalized data artifact is missing or drifted")
    return receipt


def run_history_recovery(*, phase: str, execution_id: str, contract_path: Path = CONTRACT_PATH,
                         input_binding_path: Path = INPUT_BINDING_PATH,
                         activation_path: Path = ACTIVATION_PATH,
                         output_root: Path = OUTPUT_ROOT, registry_path: Path = REGISTRY_PATH,
                         canary_receipt_path: Path | None = None,
                         canary_receipt_sha256: str | None = None,
                         futu_api: Any = None,
                         quote_context_factory: Callable[..., Any] | None = None,
                         guard_factory: Callable[..., Any] | None = None,
                         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict[str, Any]:
    """Run one review-activated canary or one canary-gated full retrieval."""
    phase = str(phase).lower()
    if phase not in {"canary", "full"}:
        raise RecoveryBlocked("phase must be canary or full")
    if not EXECUTION_ID_RE.fullmatch(str(execution_id)):
        raise RecoveryBlocked("execution_id contains unsupported characters")
    contract_path, input_binding_path, activation_path = map(
        lambda path: Path(path).resolve(), (contract_path, input_binding_path, activation_path))
    contract, binding, source_paths = validate_source_bindings(
        contract_path=contract_path, input_binding_path=input_binding_path)
    contract_sha, binding_sha, code_sha = sha256_file(contract_path), sha256_file(input_binding_path), sha256_file(Path(__file__).resolve())
    if not activation_path.is_file():
        raise RecoveryBlocked("independent Futu history activation is absent")
    activation = read_json(activation_path)
    validate_activation(contract=contract, activation=activation, contract_sha256=contract_sha,
                        binding_sha256=binding_sha, code_sha256=code_sha, activation_path=activation_path)
    canary_receipt: dict[str, Any] | None = None
    if phase == "full":
        if canary_receipt_path is None or not canary_receipt_sha256:
            raise RecoveryBlocked("full phase requires the exact successful canary receipt path and SHA")
        canary_receipt = _validate_canary_receipt(
            canary_receipt_path, output_root=output_root, expected_sha256=canary_receipt_sha256,
            contract_sha256=contract_sha, binding_sha256=binding_sha)
    output_root = Path(output_root).resolve()
    execution_dir = output_root / str(execution_id)
    receipt_path = execution_dir / "receipt.json"
    if execution_dir.exists() or receipt_path.exists():
        raise RecoveryBlocked("execution output already exists; use a fresh execution identity")
    start_date, end_date, max_count, max_pages = (
        (CANARY_START, CANARY_END, CANARY_COUNT, 1) if phase == "canary"
        else (FULL_START, FULL_END, FULL_PAGE_SIZE, FULL_MAX_PAGES)
    )

    if futu_api is None:
        futu_api = importlib.import_module("futu")
    factory = quote_context_factory or futu_api.OpenQuoteContext
    artifact_paths = [Path(__file__).resolve(), contract_path, input_binding_path,
                      *source_paths.values()]
    guard = (guard_factory or _history_guard)(
        contract_path=contract_path, binding_path=input_binding_path, registry_path=Path(registry_path),
        artifact_paths=artifact_paths, stage=phase)
    if guard.accounting().dangling_execution_ids:
        raise RecoveryBlocked("prior history-recovery execution is dangling; diagnose before another call")
    if phase == "full":
        list_events = getattr(guard.backend, "list_events", None)
        if not callable(list_events):
            raise RecoveryBlocked("full phase requires official canary lifecycle readback")
        canary_id = str(canary_receipt.get("execution_id")) if canary_receipt else ""
        canary_events = [event for event in list_events(EXPERIMENT_ID)
                         if str(event.get("execution_id")) == canary_id]
        starts = [event for event in canary_events if event.get("event_type") == "RUN_STARTED"]
        terminals = [event for event in canary_events if event.get("event_type") == "TERMINAL"]
        if (len(starts) != 1 or len(terminals) != 1
                or terminals[0].get("terminal_status") != "ENGINEERING_CANARY_ONLY"):
            raise RecoveryBlocked("full phase requires one completed no-fit registry canary lifecycle")
    initial = _registry_context(contract_sha=contract_sha, binding_sha=binding_sha, code_sha=code_sha,
                                contract_path=contract_path, binding_path=input_binding_path,
                                source_paths=source_paths, artifact_paths=(),
                                phase=phase, status="RUNNING")
    setter = getattr(guard.backend, "set_execution_context", None)
    updater = getattr(guard.backend, "update_execution_context", None)
    if not callable(setter) or not callable(updater):
        raise RecoveryBlocked("official no-fit registry adapter lacks event context enrichment")
    setter(str(execution_id), initial)
    claim = guard.claim_no_fit(execution_id=str(execution_id), arm=f"FUTU_SH_000300_HISTORY_{phase.upper()}",
                               contract_sha256=contract_sha, input_binding_sha256=binding_sha)

    created_at = now().astimezone(timezone.utc).isoformat()
    planned = {
        "schema_version": "FUTU_INDEX_HISTORY_RECOVERY_RECEIPT_V1",
        "status": "RUN_STARTED",
        "experiment_id": EXPERIMENT_ID,
        "route_id": ROUTE_ID,
        "execution_id": str(execution_id),
        "phase": phase,
        "created_at_utc": created_at,
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "code_sha256": code_sha,
        "index_code": INDEX_CODE,
        "requested_bounds": {"start": start_date, "end": end_date},
        "api_plan": {
            "quote_context": {"host": LOCAL_OPEND_HOST, "port": LOCAL_OPEND_PORT},
            "quota_calls": 1,
            "history_kline_calls_max": max_pages,
            "max_count_per_page": max_count,
            "retries": 0,
            "autype": "AuType.NONE",
            "ktype": "KLType.K_DAY",
        },
        "canary_receipt": ({"path": str(Path(canary_receipt_path).resolve()),
                             "sha256": canary_receipt_sha256} if canary_receipt else None),
        "security_flags": {"research_only": True, "production_allowed": False,
                           "buy_gate_updated": False, "orders_sent": False},
    }
    pages: list[pd.DataFrame] = []
    raw_page_artifacts: list[dict[str, Any]] = []
    page_count = 0
    quota: dict[str, Any] = {"status": "NOT_CALLED", "used_quota": None,
                             "remain_quota": None, "quota_total": None}
    api_status = "NOT_CALLED"
    error_type: str | None = None
    error_message: str | None = None
    normalized = pd.DataFrame(columns=CANONICAL_COLUMNS)
    row_validation: dict[str, Any] = {"status": "NOT_RUN", "row_count": 0}
    status = "CANNOT_CONFIRM"
    context = None
    try:
        execution_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(receipt_path, planned)
        context = factory(host=LOCAL_OPEND_HOST, port=LOCAL_OPEND_PORT)
        quota = _quota_snapshot(context.get_history_kl_quota(get_detail=False), ret_ok=futu_api.RET_OK)
        if quota["status"] != "API_OK":
            api_status = "QUOTA_CANNOT_CONFIRM"
            raise RuntimeError("Futu history quota status could not be confirmed")
        # A remaining quota of zero does not block this single frozen canary;
        # the API response is the authority and no retry is attempted.
        page_req_key = None
        for page_index in range(max_pages):
            page_count += 1
            response = context.request_history_kline(
                INDEX_CODE, start=start_date, end=end_date,
                ktype=futu_api.KLType.K_DAY, autype=futu_api.AuType.NONE,
                max_count=max_count, page_req_key=page_req_key)
            api_status, raw, next_key, error_message = _api_page(response, ret_ok=futu_api.RET_OK)
            if api_status != "API_OK" or raw is None:
                raise RuntimeError(error_message or "history API returned an invalid page")
            if phase == "canary" and len(raw) > CANARY_COUNT:
                api_status = "CANARY_RESPONSE_EXCEEDS_FROZEN_MAX_COUNT"
                raise RuntimeError("canary API response exceeded the five-bar limit")
            raw_path = execution_dir / f"futu_history_raw_page_{page_index + 1:03d}.csv"
            raw.to_csv(raw_path, index=False)
            raw_page_artifacts.append({"path": raw_path.name, "sha256": sha256_file(raw_path),
                                      "row_count": int(len(raw))})
            page_normalized, validation = _normalize_page(raw, start=start_date, end=end_date)
            if validation["status"] != "VALID":
                row_validation = validation
                raise RuntimeError(f"history page failed row validation: {validation['status']}")
            pages.append(page_normalized)
            page_req_key = next_key
            if phase == "canary" or page_req_key is None:
                break
        if phase == "full" and page_req_key is not None:
            api_status = "PAGE_LIMIT_EXCEEDED"
            raise RuntimeError("history pagination key remains after the frozen three-page limit")
        normalized = pd.concat(pages, ignore_index=True) if pages else normalized
        duplicates = normalized["date"].duplicated(keep=False)
        if bool(duplicates.any()):
            row_validation = {"status": "DUPLICATE_DATES", "duplicate_date_count": int(duplicates.sum())}
            raise RuntimeError("history pages contained duplicate dates")
        normalized = normalized.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
        if normalized.empty:
            row_validation = {"status": "EMPTY", "row_count": 0}
            raise RuntimeError("history API returned no bars")
        row_validation = {"status": "VALID", "row_count": int(len(normalized)),
                          "duplicate_date_count": 0, "invalid_row_count": 0}
        status = "CANARY_PASS" if phase == "canary" else "SERIES_RECOVERED_CALENDAR_PARITY_CANNOT_CONFIRM"
    except BaseException as exc:
        status = "CANARY_CANNOT_CONFIRM" if phase == "canary" else "CANNOT_CONFIRM"
        error_type = type(exc).__name__
        if error_message is None:
            error_message = str(exc)[:500]
    finally:
        if context is not None:
            close = getattr(context, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    error_type = error_type or type(exc).__name__
                    error_message = error_message or str(exc)[:500]
                    status = "CANARY_CANNOT_CONFIRM" if phase == "canary" else "CANNOT_CONFIRM"

    normalized_path = execution_dir / "index_000300_daily.csv"
    try:
        if not normalized.empty:
            normalized.to_csv(normalized_path, index=False, columns=CANONICAL_COLUMNS)
    except Exception as exc:
        status = "CANARY_CANNOT_CONFIRM" if phase == "canary" else "CANNOT_CONFIRM"
        error_type = error_type or type(exc).__name__
        error_message = error_message or f"normalized CSV write failed: {str(exc)[:400]}"
    normalized_sha = sha256_file(normalized_path) if normalized_path.is_file() else None
    gaps = _date_gap_diagnostics(normalized)
    result = {
        **planned,
        "status": status,
        "finished_at_utc": now().astimezone(timezone.utc).isoformat(),
        "futu_sdk_version": str(getattr(futu_api, "__version__", "UNKNOWN")),
        "quota": quota,
        "api_status": api_status,
        "page_count": page_count,
        "raw_page_artifacts": raw_page_artifacts,
        "normalized_path": normalized_path.name if normalized_path.is_file() else None,
        "normalized_sha256": normalized_sha,
        "normalized_row_count": int(len(normalized)),
        "returned_dates": sorted(set(normalized["date"].astype(str))) if not normalized.empty else [],
        "date_gaps": gaps,
        "row_validation": row_validation,
        "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        "error_type": error_type,
        "error_message": error_message,
        "fit_calls": 0,
        "snapshot_requests": 0,
        "account_or_trade_api_calls": 0,
    }
    try:
        atomic_write_json(receipt_path, result)
    except Exception as exc:
        status = "CANARY_CANNOT_CONFIRM" if phase == "canary" else "CANNOT_CONFIRM"
        error_type = error_type or type(exc).__name__
        error_message = error_message or f"receipt write failed: {str(exc)[:400]}"
        result.update(status=status, error_type=error_type, error_message=error_message)
    updated = _registry_context(contract_sha=contract_sha, binding_sha=binding_sha, code_sha=code_sha,
                                contract_path=contract_path, binding_path=input_binding_path,
                                source_paths=source_paths,
                                artifact_paths=[receipt_path, *[
                                    execution_dir / item["path"] for item in raw_page_artifacts],
                                    *([normalized_path] if normalized_path.is_file() else [])],
                                phase=phase, status=status, quota=quota,
                                row_count=int(len(normalized)), page_count=page_count,
                                normalized_sha=normalized_sha)
    updater(str(execution_id), **updated)
    succeeded = status in {"CANARY_PASS", "SERIES_RECOVERED_CALENDAR_PARITY_CANNOT_CONFIRM"}
    guard.append_no_fit_terminal(claim, succeeded=succeeded, error_type=error_type or (None if succeeded else status))
    result["receipt_path"] = str(receipt_path.resolve())
    result["receipt_sha256"] = sha256_file(receipt_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("canary", "full"))
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--input-binding", type=Path, default=INPUT_BINDING_PATH)
    parser.add_argument("--activation", type=Path, default=ACTIVATION_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--canary-receipt-sha256")
    parser.add_argument("--futu-sdk-path", type=Path,
                        help="optional import root for the already installed Futu SDK")
    args = parser.parse_args(argv)
    if args.futu_sdk_path:
        sys.path.insert(0, str(args.futu_sdk_path.resolve()))
    result = run_history_recovery(
        phase=args.phase, execution_id=args.execution_id, contract_path=args.contract,
        input_binding_path=args.input_binding, activation_path=args.activation,
        output_root=args.output_root, registry_path=args.registry,
        canary_receipt_path=args.canary_receipt, canary_receipt_sha256=args.canary_receipt_sha256)
    print(json.dumps({key: result.get(key) for key in (
        "status", "phase", "execution_id", "receipt_path", "receipt_sha256",
        "normalized_row_count", "page_count", "official_exchange_calendar_parity", "fit_calls")},
        ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {"CANARY_PASS", "SERIES_RECOVERED_CALENDAR_PARITY_CANNOT_CONFIRM"} else 2


if __name__ == "__main__":  # pragma: no cover - actual OpenD access is activation-gated
    raise SystemExit(main())
