"""Review-gated, read-only Futu capability probe for Shanghai index metadata.

The one-shot probe uses only ``get_stock_basicinfo`` for ``SH.000300`` and
``get_history_kl_quota(get_detail=False)``.  It does not request history bars,
quotes, account identity, holdings, or orders.  A durable no-fit registry
claim is written before a local OpenD connection is created.
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
from typing import Any, Callable, Mapping


ROUTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROUTE_ROOT.parents[2]
BATCH_ROOT = PROJECT_ROOT / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
MODEL_CONTRACT_PATH = BATCH_ROOT / "CONTRACT.json"
INPUT_BINDING_PATH = BATCH_ROOT / "INPUT_BINDING.json"
PROBE_CONTRACT_PATH = BATCH_ROOT / "FUTU_INDEX_CAPABILITY_PROBE_CONTRACT.json"
ACTIVATION_PATH = BATCH_ROOT / "FUTU_INDEX_CAPABILITY_PROBE_ACTIVATION.json"
DEFAULT_OUTPUT_PATH = BATCH_ROOT / "FUTU_INDEX_CAPABILITY_PROBE_RECEIPT_20261019.json"
REGISTRY_PATH = PROJECT_ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
REGISTRY_ADAPTER_DIR = PROJECT_ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"

PROBE_EXPERIMENT_ID = "A_SHARE_INDEX_FUTU_CAPABILITY_PROBE_PV_ENVIRONMENT_R1_20261019"
PROBE_ROUTE_ID = "a-share-index-futu-capability-probe-pv-environment-r1-20261019"
INDEX_CODE = "SH.000300"
LOCAL_OPEND_HOST = "127.0.0.1"
LOCAL_OPEND_PORT = 11111
EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


class ProbeBlocked(RuntimeError):
    """The exact reviewed probe contract or activation is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
        raise ProbeBlocked(f"required probe control file is missing: {Path(path).name}") from exc
    except json.JSONDecodeError as exc:
        raise ProbeBlocked(f"probe control JSON is invalid: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ProbeBlocked(f"expected JSON object: {Path(path).name}")
    return value


def resolve_bound_source_paths(probe_contract_path: Path, probe_contract: Mapping[str, Any]) -> tuple[Path, Path]:
    sources = probe_contract.get("source_bindings", {})
    model = sources.get("model_contract", {})
    binding = sources.get("input_binding", {})
    if not model.get("path") or not model.get("sha256") or not binding.get("path") or not binding.get("sha256"):
        raise ProbeBlocked("Futu probe source bindings are incomplete")
    parent = Path(probe_contract_path).resolve().parent
    model_path = (parent / str(model["path"])).resolve()
    binding_path = (parent / str(binding["path"])).resolve()
    if not model_path.is_file() or sha256_file(model_path) != model["sha256"]:
        raise ProbeBlocked("source model-selection contract is missing or drifted")
    if not binding_path.is_file() or sha256_file(binding_path) != binding["sha256"]:
        raise ProbeBlocked("source input binding is missing or drifted")
    return model_path, binding_path


def probe_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def validate_activation(
    probe_contract: Mapping[str, Any],
    activation: Mapping[str, Any],
    *,
    probe_contract_sha256: str,
    binding_sha256: str,
    code_sha256: str,
) -> None:
    if probe_contract.get("status") != "FROZEN_AWAITING_PRE_RUN_REVIEW":
        raise ProbeBlocked("Futu capability probe contract is not frozen for review")
    if activation.get("status") != "APPROVED_FOR_FUTU_METADATA_AND_QUOTA_ONLY":
        raise ProbeBlocked("independent probe activation is absent")
    writer = str(probe_contract.get("writer_identity", ""))
    reviewer = str(activation.get("reviewer_identity", ""))
    if not writer or not reviewer or writer == reviewer:
        raise ProbeBlocked("activation must identify an independent reviewer")
    expected = {
        "probe_contract_sha256": probe_contract_sha256,
        "input_binding_sha256": binding_sha256,
        "code_sha256": code_sha256,
    }
    if any(activation.get(field) != value for field, value in expected.items()):
        raise ProbeBlocked("activation hashes do not match the reviewed probe")
    receipt_path = activation.get("reviewer_receipt_path")
    receipt_sha = activation.get("reviewer_receipt_sha256")
    if not receipt_path or not receipt_sha:
        raise ProbeBlocked("reviewer receipt path and SHA are required")
    receipt = Path(str(receipt_path))
    if not receipt.is_file() or sha256_file(receipt) != receipt_sha:
        raise ProbeBlocked("independent reviewer receipt is missing or drifted")
    if activation.get("allow_local_quote_metadata_calls") is not True:
        raise ProbeBlocked("activation must explicitly allow the two local quote metadata calls")
    for field in ("allow_history_kline_request", "allow_snapshot_request", "allow_account_or_trade_api"):
        if activation.get(field) is not False:
            raise ProbeBlocked(f"activation must keep {field} closed")


def _status_name(ret_code: Any, ret_ok: Any) -> str:
    return "API_OK" if ret_code == ret_ok else "API_ERROR"


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _finite_nonnegative_int(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def extract_stock_metadata(data: Any, *, ret_code: Any, ret_ok: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": _status_name(ret_code, ret_ok), "code": INDEX_CODE,
                              "name": None, "security_type": None}
    if ret_code != ret_ok:
        return result
    if not hasattr(data, "columns") or not hasattr(data, "loc"):
        result["status"] = "INVALID_RESPONSE"
        return result
    required = {"code", "name", "stock_type"}
    if not required.issubset(set(data.columns)):
        result["status"] = "INVALID_RESPONSE"
        return result
    rows = data.loc[data["code"].astype(str).eq(INDEX_CODE), ["name", "stock_type"]]
    if len(rows) != 1:
        result["status"] = "CANNOT_CONFIRM_EXACT_INDEX_ROW"
        return result
    result["name"] = _clean_text(rows.iloc[0]["name"])
    result["security_type"] = _clean_text(rows.iloc[0]["stock_type"])
    if result["name"] is None or result["security_type"] is None:
        result["status"] = "CANNOT_CONFIRM_METADATA_FIELDS"
    return result


def extract_history_quota(response: Any, *, ret_ok: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "INVALID_RESPONSE", "quota_total": None}
    if not isinstance(response, tuple) or len(response) != 2:
        return result
    ret_code, payload = response
    result["status"] = _status_name(ret_code, ret_ok)
    if ret_code != ret_ok:
        return result
    if not isinstance(payload, (tuple, list)) or len(payload) < 2:
        result["status"] = "INVALID_RESPONSE"
        return result
    used = _finite_nonnegative_int(payload[0])
    remaining = _finite_nonnegative_int(payload[1])
    if used is None or remaining is None:
        result["status"] = "INVALID_RESPONSE"
        return result
    result["quota_total"] = used + remaining
    return result


def _registry_guard(*, contract_path: Path, binding_path: Path, registry_path: Path,
                    artifact_path: Path, stage: str):
    sys.path.insert(0, str(REGISTRY_ADAPTER_DIR))
    from execution_guard_core import ExecutionGuard
    from registry_adapter import OfficialRegistryAdapter

    profile = {
        "track_id": "A_SHARE_PANEL_MODEL_SELECTION_RESEARCH",
        "route_id": PROBE_ROUTE_ID,
        "registration_mode": "LIVE_EXECUTION",
        "start_stage": f"{stage}_RUN_STARTED",
        "terminal_stage": f"{stage}_TERMINAL",
        "occurred_at_reason": "LOCAL_FUTU_OPEND_READ_ONLY_CAPABILITY_PROBE_UTC",
        "baseline_id": "FUTU_SH_000300_CAPABILITY_METADATA_ONLY",
        "one_variable": "FUTU_SH_INDEX_SECURITY_TYPE_AND_KLINE_QUOTA_CAPABILITY",
        "signal_scope": "NO_MARKET_SIGNAL",
        "target_scope": "NO_LABEL_OR_TARGET",
        "device": "LOCAL_OPEND_QUOTE_CONTEXT_NO_MODEL_FIT",
        "evidence_level": "ENGINEERING_CANARY_ONLY",
    }

    class FutuProbeAdapter(OfficialRegistryAdapter):
        def __init__(self, **kwargs: Any) -> None:
            self._contexts: dict[str, dict[str, Any]] = {}
            super().__init__(**kwargs)

        def set_execution_context(self, execution_id: str, context: Mapping[str, Any]) -> None:
            self._contexts[str(execution_id)] = dict(context)

        def update_execution_context(self, execution_id: str, **updates: Any) -> None:
            if str(execution_id) not in self._contexts:
                raise ProbeBlocked(f"registry context was not initialized: {execution_id}")
            self._contexts[str(execution_id)].update(updates)

        def _event_spec(self, event: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
            spec = super()._event_spec(event, event_type=event_type)
            execution_id = str(event.get("execution_id", ""))
            context = self._contexts.get(execution_id)
            if context is None:
                raise ProbeBlocked(f"refusing un-enriched registry event: {execution_id}")
            for field in ("truth_access", "sealed_status", "signal_date_range", "target_date_range",
                          "runtime", "device", "checkpoint_ref", "code_sha256", "config_sha256",
                          "input_sha256"):
                if field in context:
                    spec[field] = context[field]
            spec["input_paths"] = list(context.get("input_paths", ()))
            spec["artifact_paths"] = list(dict.fromkeys([
                *spec.get("artifact_paths", ()), *context.get("artifact_paths", ())]))
            spec["artifact_path"] = spec["artifact_paths"]
            metrics = dict(spec.get("deterministic_metrics") or {})
            metrics.update(dict(context.get("deterministic_metrics") or {}))
            spec["deterministic_metrics"] = metrics
            identity = dict(spec.get("input_identity") or {})
            identity.update(dict(context.get("input_identity") or {}))
            spec["input_identity"] = identity
            return spec

    adapter = FutuProbeAdapter(registry_path=registry_path, contract_path=contract_path,
                               input_binding_path=binding_path,
                               artifact_paths=(Path(__file__).resolve(), artifact_path),
                               event_profile=profile)
    return ExecutionGuard(adapter, experiment_id=PROBE_EXPERIMENT_ID, max_real_fit_debits=0)


def _event_context(*, contract_sha256: str, binding_sha256: str, output_path: Path,
                   status: str, stock: Mapping[str, Any] | None = None,
                   quota: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "truth_positive_denominator": None,
        "selection_slots": None,
        "quote_metadata_call_count": 1 if stock is not None else 0,
        "history_quota_call_count": 1 if quota is not None else 0,
        "history_kline_request_count": 0,
        "snapshot_request_count": 0,
        "account_or_trade_api_call_count": 0,
        "probe_status": status,
    }
    if stock is not None:
        metrics["stock_metadata_status"] = str(stock.get("status"))
        metrics["security_type"] = stock.get("security_type")
    if quota is not None:
        metrics["history_quota_status"] = str(quota.get("status"))
        metrics["history_quota_total"] = quota.get("quota_total")
    return {
        "truth_access": "LOCAL_FUTU_METADATA_AND_QUOTA_ONLY",
        "sealed_status": "NOT_ENTERED",
        "signal_date_range": {"count": 0, "scope": "NO_MARKET_SIGNAL"},
        "target_date_range": {"count": 0, "scope": "NO_LABEL_OR_TARGET"},
        "runtime": {"python": sys.executable, "python_version": sys.version.split()[0]},
        "device": "LOCAL_OPEND_QUOTE_CONTEXT_NO_MODEL_FIT",
        "checkpoint_ref": None,
        "code_sha256": probe_code_sha256(),
        "config_sha256": contract_sha256,
        "input_sha256": binding_sha256,
        "input_paths": [str(PROBE_CONTRACT_PATH.resolve()), str(INPUT_BINDING_PATH.resolve())],
        "artifact_paths": [str(Path(output_path).resolve())],
        "deterministic_metrics": metrics,
        "input_identity": {
            "experiment_id": PROBE_EXPERIMENT_ID,
            "probe_contract_sha256": contract_sha256,
            "input_binding_sha256": binding_sha256,
            "null_reasons": {
                "deterministic_metrics.truth_positive_denominator": "NO_BINARY_LABEL_IN_CAPABILITY_PROBE",
                "deterministic_metrics.selection_slots": "NO_SELECTION_IN_CAPABILITY_PROBE",
                "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_NO_FIT_CAPABILITY_PROBE",
            },
        },
    }


def run_capability_probe(
    *,
    execution_id: str,
    contract_path: Path = PROBE_CONTRACT_PATH,
    activation_path: Path = ACTIVATION_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    registry_path: Path = REGISTRY_PATH,
    futu_api: Any = None,
    quote_context_factory: Callable[..., Any] | None = None,
    guard_factory: Callable[..., Any] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Execute one reviewed pair of read-only API calls and register both ends."""
    if not EXECUTION_ID_RE.fullmatch(str(execution_id)):
        raise ProbeBlocked("execution_id contains unsupported characters")
    contract_path = Path(contract_path).resolve()
    activation_path = Path(activation_path).resolve()
    output_path = Path(output_path).resolve()
    probe_contract = read_json(contract_path)
    model_contract_path, binding_path = resolve_bound_source_paths(contract_path, probe_contract)
    if not activation_path.is_file():
        raise ProbeBlocked("independent probe activation is absent")
    activation = read_json(activation_path)
    contract_sha = sha256_file(contract_path)
    binding_sha = sha256_file(binding_path)
    code_sha = probe_code_sha256()
    validate_activation(probe_contract, activation, probe_contract_sha256=contract_sha,
                        binding_sha256=binding_sha, code_sha256=code_sha)
    if output_path.exists():
        raise ProbeBlocked("capability probe output already exists; use a fresh execution/output identity")

    if futu_api is None:
        futu_api = importlib.import_module("futu")
    factory = quote_context_factory or futu_api.OpenQuoteContext
    guard = (guard_factory or _registry_guard)(
        contract_path=contract_path,
        binding_path=binding_path,
        registry_path=registry_path,
        artifact_path=output_path,
        stage="FUTU_INDEX_CAPABILITY_PROBE",
    )
    if guard.accounting().dangling_execution_ids:
        raise ProbeBlocked("a prior Futu capability execution is dangling; diagnose before another probe")
    initial_context = _event_context(contract_sha256=contract_sha, binding_sha256=binding_sha,
                                    output_path=output_path, status="RUNNING")
    initial_context["input_paths"] = [str(contract_path), str(binding_path), str(model_contract_path)]
    set_context = getattr(guard.backend, "set_execution_context", None)
    if not callable(set_context):
        raise ProbeBlocked("official no-fit registry adapter lacks context enrichment")
    set_context(str(execution_id), initial_context)

    created = now().astimezone(timezone.utc).isoformat()
    planned = {
        "schema_version": "FUTU_INDEX_CAPABILITY_PROBE_RECEIPT_V1",
        "status": "PLANNED_BEFORE_RUN_STARTED",
        "experiment_id": PROBE_EXPERIMENT_ID,
        "route_id": PROBE_ROUTE_ID,
        "execution_id": str(execution_id),
        "created_at_utc": created,
        "probe_contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "model_contract_sha256": sha256_file(model_contract_path),
        "api_plan": {
            "quote_context": {"host": LOCAL_OPEND_HOST, "port": LOCAL_OPEND_PORT},
            "calls": [
                {"method": "get_stock_basicinfo", "market": "SH", "code_list": [INDEX_CODE]},
                {"method": "get_history_kl_quota", "get_detail": False},
            ],
            "history_kline_requests": 0,
            "snapshot_requests": 0,
            "account_or_trade_api_calls": 0,
        },
        "security_flags": {"research_only": True, "production_allowed": False,
                           "buy_gate_updated": False, "orders_sent": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, planned)
    claim = guard.claim_no_fit(execution_id=str(execution_id), arm="FUTU_SH_000300_CAPABILITY_ONLY",
                               contract_sha256=contract_sha, input_binding_sha256=binding_sha)
    context = None
    stock: dict[str, Any] = {"status": "NOT_CALLED", "code": INDEX_CODE, "name": None, "security_type": None}
    quota: dict[str, Any] = {"status": "NOT_CALLED", "quota_total": None}
    status = "CANNOT_CONFIRM"
    error_type: str | None = None
    try:
        context = factory(host=LOCAL_OPEND_HOST, port=LOCAL_OPEND_PORT)
        stock_ret, stock_data = context.get_stock_basicinfo(
            futu_api.Market.SH, code_list=[INDEX_CODE]
        )
        stock = extract_stock_metadata(stock_data, ret_code=stock_ret, ret_ok=futu_api.RET_OK)
        quota = extract_history_quota(context.get_history_kl_quota(get_detail=False), ret_ok=futu_api.RET_OK)
        status = "CAPABILITY_CONFIRMED" if stock["status"] == "API_OK" and quota["status"] == "API_OK" else "CANNOT_CONFIRM"
    except BaseException as exc:
        error_type = type(exc).__name__
        status = "CANNOT_CONFIRM"
    finally:
        if context is not None:
            close = getattr(context, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    error_type = error_type or type(exc).__name__
                    status = "CANNOT_CONFIRM"

    result = {
        **planned,
        "status": status,
        "finished_at_utc": now().astimezone(timezone.utc).isoformat(),
        "futu_sdk_version": str(getattr(futu_api, "__version__", "UNKNOWN")),
        "stock_metadata": stock,
        "history_kl_quota": quota,
        "error_type": error_type,
        "fit_calls": 0,
        "history_kline_requests": 0,
        "snapshot_requests": 0,
        "account_or_trade_api_calls": 0,
    }
    atomic_write_json(output_path, result)
    updated_context = _event_context(contract_sha256=contract_sha, binding_sha256=binding_sha,
                                     output_path=output_path, status=status, stock=stock, quota=quota)
    updated_context["input_paths"] = [str(contract_path), str(binding_path), str(model_contract_path)]
    updater = getattr(guard.backend, "update_execution_context", None)
    if not callable(updater):
        raise ProbeBlocked("official no-fit registry adapter lacks context update support")
    updater(str(execution_id), **updated_context)
    guard.append_no_fit_terminal(claim, succeeded=status == "CAPABILITY_CONFIRMED",
                                 error_type=error_type or (None if status == "CAPABILITY_CONFIRMED" else status))
    result["output_path"] = str(output_path)
    result["output_sha256"] = sha256_file(output_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--contract", type=Path, default=PROBE_CONTRACT_PATH)
    parser.add_argument("--activation", type=Path, default=ACTIVATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--futu-sdk-path", type=Path,
                        help="optional import root for the already installed Futu SDK")
    args = parser.parse_args(argv)
    if args.futu_sdk_path:
        sys.path.insert(0, str(args.futu_sdk_path.resolve()))
    result = run_capability_probe(execution_id=args.execution_id, contract_path=args.contract,
                                  activation_path=args.activation, output_path=args.output,
                                  registry_path=args.registry)
    print(json.dumps({key: result.get(key) for key in (
        "status", "execution_id", "output_path", "output_sha256", "stock_metadata",
        "history_kl_quota", "fit_calls", "history_kline_requests", "account_or_trade_api_calls")},
        ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "CAPABILITY_CONFIRMED" else 2


if __name__ == "__main__":  # pragma: no cover - actual OpenD access is reviewer-gated
    raise SystemExit(main())
