"""Frozen M0-M4 A-share close-return model-selection route.

This module contains deterministic transforms and a fail-closed runner.  The
runner will not load the bound parquet or call ``fit`` until a separate
review-activation receipt matches the exact contract, input binding, and code
SHA.  Formula and auxiliary-source adapters are intentionally supplied through
the binding layer; this module never substitutes another data producer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import contextlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import zipfile
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


ROUTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (candidate for candidate in ROUTE_ROOT.parents
     if (candidate / "workspace/research/a_share_model_selection_v1").is_dir()
     and (candidate / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/CONTRACT_R2.json").is_file()),
    ROUTE_ROOT.parents[2],
)
FROZEN_ROUTE_DEPENDENCY_ROOT = PROJECT_ROOT / "workspace/research/a_share_model_selection_v1"
BATCH_ROOT = PROJECT_ROOT / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
CONTRACT_PATH = BATCH_ROOT / "CONTRACT_R2.json"
INPUT_BINDING_PATH = BATCH_ROOT / "INPUT_BINDING_R2.json"
ACTIVATION_PATH = BATCH_ROOT / "REVIEW_ACTIVATION.json"
EXPERIMENT_ID = "A_SHARE_MODEL_SELECTION_PV_ENVIRONMENT_R1_20261019"
CORE_FEATURES = (
    "DISTW", "DISTB", "C", "YEL0", "LCOUNT", "HIST0", "AMTR", "VOLR",
    "VMA20R", "VSPHHV", "FCOUNT", "SFCOUNT", "RET20P", "GAIN60",
    "RANGE60", "RANGE120", "JVAL", "BKRSI", "CPOS", "UPSH", "WASHSCR",
    "B1PRE", "BKEXIST", "FRESH", "CONT0", "B1SIG", "B2SIG", "B3SIG",
    "BREAKOUT", "BKRES", "RTETYPE", "RISK0", "HEALTH", "LOWEXT", "KQUAL",
    "LIQSTD",
)
HORIZONS = (1, 2, 5)
ARMS = ("M0", "M1", "M2", "M3", "M4")
FORMAL_FOLDS = (
    ("2024H1", "2024-01-01", "2024-06-30", False),
    ("2024H2", "2024-07-01", "2024-12-31", False),
    ("2025H1", "2025-01-01", "2025-06-30", False),
    ("2025H2", "2025-07-01", "2025-12-31", False),
    ("2026H1", "2026-01-01", "2026-06-30", False),
    ("2026H2_PARTIAL_TO_2026_10_15", "2026-07-01", "2026-10-15", True),
)
MODEL_PARAMS = {
    "loss": "squared_error",
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 200,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 193,
}
KEYS = ("date", "code")
KEY_IDENTITY_SCHEMA = "A_SHARE_DATE_CODE_SORTED_KEYS_V1"
MATRIX_IDENTITY_SCHEMA = "A_SHARE_FLOAT64_MATRIX_V1"
REPLAY_RECEIPT_SCHEMA = "A_SHARE_SAVED_CHECKPOINT_REPLAY_V1"
INDEXED_ARMS = ("M1", "M2", "M3", "M4")
OHLC_COLUMNS = ("open", "high", "low", "close")
STAGE_SPECS = {
    "M0_CANARY": {"arms": ("M0",), "input_arms": ("M0",), "requires_common_keys": False,
                   "folds": ("2024H1",), "horizons": HORIZONS, "canary": True},
    "M0_FORMAL": {"arms": ("M0",), "input_arms": ARMS, "requires_common_keys": True,
                   "folds": tuple(item[0] for item in FORMAL_FOLDS), "horizons": HORIZONS, "canary": False},
    "M4_CANARY": {"arms": ("M4",), "input_arms": ARMS, "requires_common_keys": True,
                   "folds": ("2024H1",), "horizons": HORIZONS, "canary": True},
    "INDEXED_FORMAL": {"arms": INDEXED_ARMS, "input_arms": ARMS, "requires_common_keys": True,
                       "folds": tuple(item[0] for item in FORMAL_FOLDS), "horizons": HORIZONS, "canary": False},
}


class ContractError(ValueError):
    """A frozen contract or bound input is incomplete or inconsistent."""


class ExecutionBlocked(RuntimeError):
    """Real-input access or fitting was attempted before independent review."""


@dataclass(frozen=True)
class Fold:
    fold_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    partial: bool


def validate_index_source_policy(
    contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    contract_path: Path,
    binding_path: Path,
    binding_sha256: str,
) -> None:
    """Require indexed stages to use the exact documented Futu replacement."""
    contract_file = Path(contract_path)
    binding_file = Path(binding_path)
    policy = contract.get("inputs", {}).get("official_index_history")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("replacement_policy"), Mapping):
        raise ExecutionBlocked("indexed stage contract lacks an explicit replacement index source policy")
    if contract_file.name != "CONTRACT_R2.json" or binding_file.name != "INPUT_BINDING_R2.json":
        raise ExecutionBlocked("indexed stage requires the exact R2 contract and input binding files")
    if policy.get("original_tdx_identity") != "CANNOT_CONFIRM":
        raise ExecutionBlocked("R2 contract must retain CANNOT_CONFIRM for original TDX identity")
    if policy.get("official_exchange_calendar_parity") != "CANNOT_CONFIRM":
        raise ExecutionBlocked("R2 contract must retain CANNOT_CONFIRM for official calendar parity")

    replacement = policy["replacement_policy"]
    expected_binding_path = replacement.get("binding_path")
    if (expected_binding_path != binding_file.name or
            binding_file.resolve() != (contract_file.resolve().parent / str(expected_binding_path)).resolve()):
        raise ExecutionBlocked("replacement index policy points to a different input binding path")
    if replacement.get("binding_sha256") != binding_sha256:
        raise ExecutionBlocked("replacement index policy does not bind the exact R2 input binding SHA")

    index_source = binding.get("sources", {}).get("index_series", {})
    if not isinstance(index_source, Mapping) or index_source.get("status") != "BOUND":
        raise ExecutionBlocked("replacement index binding is missing or not bound")
    for policy_key, source_key in (
        ("source_type", "source_type"),
        ("index_code", "index_code"),
        ("column_map", "column_map"),
        ("source_columns", "source_columns"),
        ("source_sha256", "sha256"),
        ("receipt_sha256", "receipt_sha256"),
    ):
        if replacement.get(policy_key) != index_source.get(source_key):
            raise ExecutionBlocked(f"replacement index source {policy_key} differs from the binding")
    for policy_key, source_key in (("source_path", "path"), ("receipt_path", "receipt_path")):
        expected_path = replacement.get(policy_key)
        actual_path = index_source.get(source_key)
        if (not expected_path or not actual_path or
                (contract_file.resolve().parent / str(expected_path)).resolve() != Path(str(actual_path)).resolve()):
            raise ExecutionBlocked(f"replacement index source {policy_key} differs from the binding")
    if replacement.get("source_type") != "FUTU_RECOVERED_REPLACEMENT":
        raise ExecutionBlocked("indexed stage source must be labeled FUTU_RECOVERED_REPLACEMENT")
    identity = str(index_source.get("source_identity_status", ""))
    if "ORIGINAL_TDX_FILE_IDENTITY_CANNOT_CONFIRM" not in identity:
        raise ExecutionBlocked("replacement binding must retain CANNOT_CONFIRM for original TDX identity")
    if index_source.get("official_exchange_calendar_parity") != "CANNOT_CONFIRM":
        raise ExecutionBlocked("replacement binding must retain CANNOT_CONFIRM for official calendar parity")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def validate_review_activation(
    contract: Mapping[str, Any],
    binding: Mapping[str, Any],
    activation: Mapping[str, Any],
    *,
    contract_sha256: str,
    binding_sha256: str,
    code_sha256: str,
    stage: str,
    arms: Sequence[str],
    contract_path: Path | None = None,
    binding_path: Path | None = None,
) -> None:
    """Require a reviewer-owned receipt before any panel read or real fit."""
    if contract.get("status") != "FROZEN_AWAITING_PRE_RUN_REVIEW":
        raise ExecutionBlocked("contract status is not the frozen pre-run status")
    if binding.get("status") not in {"INPUT_BINDING_READY_PARTIAL", "INPUT_BINDING_READY"}:
        raise ExecutionBlocked("input binding status changed; a new review is required")
    readiness = binding.get("readiness_by_arm", {})
    input_arms = STAGE_SPECS[stage]["input_arms"]
    for arm in input_arms:
        arm_state = readiness.get(arm, {})
        if arm_state.get("status") not in {"READY_WITH_SCOPE_LIMITATIONS", "READY_AFTER_INDEX_BINDING"}:
            raise ExecutionBlocked(f"{arm} is not input-ready: {arm_state.get('status', 'MISSING_READINESS')}")
    if any(arm in INDEXED_ARMS for arm in input_arms):
        if contract_path is None or binding_path is None:
            raise ExecutionBlocked("indexed stage source policy requires explicit contract and binding paths")
        validate_index_source_policy(
            contract, binding, contract_path=contract_path, binding_path=binding_path,
            binding_sha256=binding_sha256,
        )
    allowed_arms = activation.get("allowed_arms", [])
    allowed_stages = activation.get("allowed_stages", [])
    if not set(arms).issubset(set(allowed_arms)) or not set(input_arms).issubset(set(allowed_arms)) or stage not in allowed_stages:
        raise ExecutionBlocked("activation does not cover the requested stage and arm set")
    activation_status = activation.get("status")
    if activation_status not in {"APPROVED_FOR_M0_ONLY", "APPROVED_FOR_INDEXED_ARMS"}:
        raise ExecutionBlocked("review activation is absent or not approved")
    if any(arm in INDEXED_ARMS for arm in input_arms) and activation_status != "APPROVED_FOR_INDEXED_ARMS":
        raise ExecutionBlocked("indexed arms require the separate indexed-stage activation")
    reviewer = str(activation.get("reviewer_identity", ""))
    governance = contract.get("governance", {})
    governance_writer = governance.get("writer_identity", "") if isinstance(governance, Mapping) else ""
    writer = str(contract.get("writer_identity") or governance_writer)
    if not reviewer or not writer or reviewer == writer:
        raise ExecutionBlocked("activation must identify an independent reviewer")
    if activation.get("contract_sha256") != contract_sha256:
        raise ExecutionBlocked("reviewed contract SHA does not match")
    if activation.get("input_binding_sha256") != binding_sha256:
        raise ExecutionBlocked("reviewed input binding SHA does not match")
    if activation.get("code_sha256") != code_sha256:
        raise ExecutionBlocked("reviewed code SHA does not match")
    receipt_path = activation.get("reviewer_receipt_path")
    receipt_sha = activation.get("reviewer_receipt_sha256")
    if not receipt_path or not receipt_sha:
        raise ExecutionBlocked("reviewer receipt path and SHA are required")
    if not Path(receipt_path).is_file() or sha256_file(Path(receipt_path)) != receipt_sha:
        raise ExecutionBlocked("independent reviewer receipt is missing or has drifted")
    if activation.get("allow_real_data_read") is not True or activation.get("allow_fit") is not True:
        raise ExecutionBlocked("activation must explicitly open both real-data and fit gates")
    if any(arm in INDEXED_ARMS and readiness.get(arm, {}).get("status") != "READY_AFTER_INDEX_BINDING"
           for arm in input_arms):
        raise ExecutionBlocked("indexed arm stage requires the reviewed index binding")


def _load_approved_inputs(
    *,
    contract_path: Path = CONTRACT_PATH,
    binding_path: Path = INPUT_BINDING_PATH,
    activation_path: Path = ACTIVATION_PATH,
    stage: str,
    arms: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only small control files before the exact reviewed gate passes."""
    contract = load_json(contract_path)
    binding = load_json(binding_path)
    activation = load_json(activation_path)
    validate_review_activation(
        contract,
        binding,
        activation,
        contract_sha256=sha256_file(contract_path),
        binding_sha256=sha256_file(binding_path),
        code_sha256=route_code_sha256(),
        stage=stage,
        arms=arms,
        contract_path=contract_path,
        binding_path=binding_path,
    )
    return contract, binding, activation


def read_bound_panel(*, contract_path: Path = CONTRACT_PATH, binding_path: Path = INPUT_BINDING_PATH,
                     activation_path: Path = ACTIVATION_PATH,
                     stage: str = "M0_CANARY", arms: Sequence[str] = ("M0",),
                     preparation_claim: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Open the authoritative parquet only after the reviewed activation gate."""
    if preparation_claim is None:
        raise ExecutionBlocked("bound panel reads require a durable no-fit RUN_STARTED claim")
    contract, binding, _ = _load_approved_inputs(
        contract_path=contract_path, binding_path=binding_path, activation_path=activation_path,
        stage=stage, arms=arms,
    )
    source = binding.get("sources", {}).get("price_panel", {})
    if source.get("status") != "BOUND" or not source.get("path") or not source.get("sha256"):
        raise ContractError("price panel path/SHA binding is incomplete")
    panel_path = Path(source["path"])
    if sha256_file(panel_path) != source["sha256"]:
        raise ContractError("price panel SHA drifted from reviewed binding")
    # This is the first numeric panel read in the route.
    column_map = source.get("column_map", {})
    columns = list(dict.fromkeys(column_map.values())) if column_map else source.get("required_columns")
    raw = pd.read_parquet(panel_path, columns=columns)
    return raw, {"contract": contract, "binding": binding}


def read_bound_archive_member(source: Mapping[str, Any]) -> bytes:
    """Read one SHA-bound member from the already-bound split archive parts."""
    parts = source.get("container_parts", [])
    member = source.get("member")
    expected = source.get("member_sha256")
    if not parts or not member or not expected:
        raise ContractError("archive member binding is incomplete")
    fd, tmp_name = tempfile.mkstemp(prefix="model-selection-bound-", suffix=".zip")
    os.close(fd)
    archive_path = Path(tmp_name)
    try:
        with archive_path.open("wb") as target:
            for part in parts:
                with Path(part).open("rb") as handle:
                    shutil.copyfileobj(handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        with zipfile.ZipFile(archive_path, "r") as archive:
            raw = archive.read(str(member))
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ContractError("bound archive member SHA drifted")
        return raw
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ContractError(f"bound archive member could not be read: {type(exc).__name__}") from exc
    finally:
        if archive_path.exists():
            archive_path.unlink()


def normalize_industry_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Bind only the recorded exact-code key and CAPCO category aliases."""
    code_candidates = [name for name in ("ts_code", "code") if name in frame.columns]
    category_candidates = [name for name in ("industry_code", "CAPCO", "capco_code") if name in frame.columns]
    if len(code_candidates) != 1 or len(category_candidates) != 1:
        raise ContractError("industry member must expose one exact-code key and one CAPCO category field")
    return frame.rename(columns={code_candidates[0]: "code", category_candidates[0]: "industry_code"})[["code", "industry_code"]]


def load_stage_inputs(*, stage: str, contract_path: Path = CONTRACT_PATH,
                      binding_path: Path = INPUT_BINDING_PATH, activation_path: Path = ACTIVATION_PATH,
                      preparation_claim: Any = None
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[int, pd.DataFrame], dict[str, Any]]:
    """Read only exact bound inputs after the requested stage has reviewer activation."""
    if stage not in STAGE_SPECS:
        raise ContractError(f"unknown model-selection stage: {stage}")
    if preparation_claim is None:
        raise ExecutionBlocked("stage input reads require a durable no-fit RUN_STARTED claim")
    stage_arms = STAGE_SPECS[stage]["input_arms"]
    panel_raw, records = read_bound_panel(contract_path=contract_path, binding_path=binding_path,
                                          activation_path=activation_path, stage=stage, arms=stage_arms,
                                          preparation_claim=preparation_claim)
    binding = records["binding"]
    price_binding = binding["sources"]["price_panel"]
    amount_unit = price_binding.get("units", {}).get("amount_input")
    if amount_unit == "10000_CNY":
        amount_multiplier = 10000.0
    elif amount_unit == "CNY":
        amount_multiplier = 1.0
    else:
        raise ContractError(f"unsupported bound amount unit: {amount_unit}")
    panel = normalize_price_panel(panel_raw, price_binding["column_map"], amount_multiplier=amount_multiplier)
    if price_binding.get("rows") is not None and len(panel) != int(price_binding["rows"]):
        raise ContractError("normalized price row count differs from the bound panel metadata")
    feature_panel, quarantine = derive_price_feature_view(panel)
    _validate_quarantine_binding(quarantine, binding.get("consumer_compatibility", {}))

    calendar_source = binding["sources"].get("trading_date_index", {})
    calendar_raw = pd.read_csv(io.BytesIO(read_bound_archive_member(calendar_source)))
    calendar = normalize_calendar(calendar_raw, panel)
    formula_sources = binding.get("formula_inputs", {})
    if not {"v192", "v193"}.issubset(formula_sources):
        raise ContractError("both exact V192 and V193 formula snapshots must be bound")
    from formula_adapter import exact_formula_source, formula_feature_frame
    v192_text, _ = exact_formula_source(Path(formula_sources["v192"]["path"]), formula_sources["v192"]["sha256"])
    v193_text, _ = exact_formula_source(Path(formula_sources["v193"]["path"]), formula_sources["v193"]["sha256"])

    index_close = None
    indexed = any(arm in INDEXED_ARMS for arm in stage_arms)
    if indexed:
        index_source = binding["sources"].get("index_series", {})
        index_close = load_bound_index_series(index_source, panel_dates=set(panel["date"]))

    formula_features = formula_feature_frame(
        feature_panel, v193_text=v193_text, v192_text=v192_text,
        index_close_by_date=index_close,
        include_v192_trigger=True,
        include_v193_trigger=indexed,
        include_auditm=indexed,
    )
    industry = None
    if STAGE_SPECS[stage]["requires_common_keys"]:
        industry_source = binding["sources"].get("industry_classification", {})
        industry = normalize_industry_columns(pd.read_csv(io.BytesIO(read_bound_archive_member(industry_source))))
    requested = ARMS if STAGE_SPECS[stage]["requires_common_keys"] else ("M0",)
    arm_frames = build_nested_arm_frames(feature_panel, formula_features, calendar, index_close_by_date=index_close,
                                         industry_classification=industry, requested_arms=requested)
    formula_features, arm_frames, compatibility_sidecar = apply_ohlc_taint_masks(
        formula_features, arm_frames, calendar, quarantine,
        industry_classification=industry,
    )
    labels: dict[int, pd.DataFrame] = {}
    records.update({"panel": panel, "feature_panel": feature_panel,
                    "feature_view_quarantine": quarantine,
                    "input_compat_sidecar": compatibility_sidecar,
                    "calendar": calendar, "formula_features": formula_features,
                    "index_close": index_close, "industry": industry})
    return panel, calendar, arm_frames, labels, records


def normalize_price_panel(raw: pd.DataFrame, column_map: Mapping[str, str], *, amount_multiplier: float = 1.0) -> pd.DataFrame:
    """Project exact date/code/close/amount inputs without imputing values."""
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    if set(column_map) != required:
        raise ContractError("price column map must bind exactly date, code, OHLC, volume, and amount")
    absent = [column_map[key] for key in required if column_map[key] not in raw]
    if absent:
        raise ContractError(f"bound price columns are absent: {absent}")
    out = raw[[column_map[k] for k in ("date", "code", "open", "high", "low", "close", "volume", "amount")]].copy()
    out.columns = ["date", "code", "open", "high", "low", "close", "volume", "amount"]
    date_values = out["date"]
    if pd.api.types.is_numeric_dtype(date_values.dtype):
        date_values = pd.to_datetime(
            date_values.astype("string"), format="%Y%m%d", errors="coerce"
        )
    else:
        date_values = pd.to_datetime(date_values, errors="coerce")
    out["date"] = date_values.dt.normalize()
    if out["date"].isna().any() or out["code"].isna().any():
        raise ContractError("missing date/code identity")
    out["code"] = out["code"].astype(str).str.strip()
    if out["code"].eq("").any() or out.duplicated(list(KEYS)).any():
        raise ContractError("empty code or duplicate date/code identity")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        raw_values = out[col]
        out[col] = pd.to_numeric(raw_values, errors="coerce").astype(float)
        if (raw_values.notna() & out[col].isna()).any():
            raise ContractError(f"non-numeric {col}")
        if np.isinf(out[col].to_numpy(dtype=np.float64)).any():
            raise ContractError(f"infinite {col}")
    if not np.isfinite(amount_multiplier) or amount_multiplier <= 0:
        raise ContractError("amount_multiplier must be positive and finite")
    out["amount"] *= float(amount_multiplier)
    price = out[list(OHLC_COLUMNS)]
    if ((price["high"] < price[["low", "open", "close"]].max(axis=1)) |
            (price["low"] > price[["high", "open", "close"]].min(axis=1))).any():
        raise ContractError("invalid OHLC range in bound panel")
    if (out[["volume", "amount"]].lt(0).any(axis=1)).any():
        raise ContractError("negative volume or amount in bound panel")
    return out.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def _parse_bound_dates(values: pd.Series, *, field_name: str) -> pd.Series:
    """Parse YYYYMMDD ordinals explicitly; never let pandas treat them as nanoseconds."""
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numeric = pd.to_numeric(values, errors="coerce")
        numbers = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        if (not np.isfinite(numbers).all() or
                not np.equal(numbers, np.floor(numbers)).all() or
                (numbers < np.iinfo(np.int64).min).any() or
                (numbers > np.iinfo(np.int64).max).any()):
            raise ContractError(f"{field_name} must be integer YYYYMMDD values")
        text = pd.Series(numbers.astype(np.int64), index=values.index).astype("string")
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        text = values.astype("string").str.strip()
        yyyymmdd = text.str.fullmatch(r"\d{8}", na=False)
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        if yyyymmdd.any():
            parsed.loc[yyyymmdd] = pd.to_datetime(
                text.loc[yyyymmdd], format="%Y%m%d", errors="coerce"
            )
        if (~yyyymmdd).any():
            parsed.loc[~yyyymmdd] = pd.to_datetime(text.loc[~yyyymmdd], errors="coerce")
    if parsed.isna().any():
        raise ContractError(f"{field_name} contains an invalid date")
    return parsed.dt.normalize()


def derive_price_feature_view(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mask only nonpositive OHLC in a derived view while retaining all raw rows and keys."""
    if not set(KEYS + OHLC_COLUMNS).issubset(panel.columns):
        raise ContractError("price feature view requires date/code and all OHLC fields")
    if panel.duplicated(list(KEYS)).any():
        raise ContractError("price feature view requires unique date/code keys")
    feature_panel = panel.copy(deep=True)
    nonpositive = panel[list(OHLC_COLUMNS)].le(0).any(axis=1)
    quarantine_rows: list[dict[str, Any]] = []
    for row_index in panel.index[nonpositive]:
        reasons = [f"NON_POSITIVE_{column.upper()}" for column in OHLC_COLUMNS
                   if pd.notna(panel.at[row_index, column]) and panel.at[row_index, column] <= 0]
        quarantine_rows.append({
            "date": pd.Timestamp(panel.at[row_index, "date"]).normalize().date().isoformat(),
            "code": str(panel.at[row_index, "code"]),
            "reason_codes": reasons,
        })
    feature_panel.loc[nonpositive, list(OHLC_COLUMNS)] = np.nan
    feature_panel = feature_panel.reset_index(drop=True)
    source_rows = int(len(panel))
    quarantined_rows = int(nonpositive.sum())
    duplicate_keys = int(panel.duplicated(list(KEYS), keep=False).sum())
    manifest = {
        "source_row_count": source_rows,
        "accepted_row_count": source_rows - quarantined_rows,
        "quarantined_row_count": quarantined_rows,
        "accepted_plus_quarantined_row_count": source_rows,
        "row_count_conserved": source_rows == (source_rows - quarantined_rows) + quarantined_rows,
        "duplicate_date_code_key_count": duplicate_keys,
        "quarantined_keys": sorted(quarantine_rows, key=lambda item: (item["date"], item["code"])),
        "feature_view_policy": "MASK_ALL_OHLC_FIELDS_ON_ANY_NONPOSITIVE_OHLC_ROW; RAW_PANEL_UNCHANGED",
    }
    if manifest["accepted_plus_quarantined_row_count"] != source_rows or duplicate_keys:
        raise ContractError("quarantine manifest does not conserve unique source rows")
    if len(feature_panel) != len(panel) or not feature_panel[list(KEYS)].equals(panel.reset_index(drop=True)[list(KEYS)]):
        raise ContractError("derived feature view changed source rows or keys")
    return feature_panel, manifest


def _validate_quarantine_binding(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    if not policy:
        return
    expected_rows = policy.get("expected_quarantined_row_count")
    if expected_rows is not None and int(manifest["quarantined_row_count"]) != int(expected_rows):
        raise ContractError("quarantined OHLC row count differs from the frozen consumer binding")
    expected_keys = policy.get("expected_quarantine_keys")
    if expected_keys is not None and manifest["quarantined_keys"] != expected_keys:
        raise ContractError("quarantined OHLC keys or reason codes differ from the frozen consumer binding")
    expected_source_rows = policy.get("expected_source_row_count")
    if expected_source_rows is not None and int(manifest["source_row_count"]) != int(expected_source_rows):
        raise ContractError("quarantine source row count differs from the frozen consumer binding")


def load_bound_index_series(index_source: Mapping[str, Any], *, panel_dates: set[pd.Timestamp]) -> pd.Series:
    """Read only bound date/code/close columns from a SHA-bound replacement index source."""
    if index_source.get("status") != "BOUND" or not index_source.get("path") or not index_source.get("sha256"):
        raise ContractError("indexed arms require a SHA-bound index series")
    column_map = index_source.get("column_map")
    if not isinstance(column_map, Mapping) or set(column_map) != {"date", "code", "close"}:
        raise ContractError("index source must bind exactly date, code, and close columns")
    source_columns = index_source.get("source_columns")
    expected_columns = [str(column_map[key]) for key in ("date", "code", "close")]
    if source_columns != expected_columns or len(set(expected_columns)) != 3:
        raise ContractError("index source columns do not match the exact date/code/close map")
    index_code = index_source.get("index_code")
    if not isinstance(index_code, str) or not index_code:
        raise ContractError("index source code identity is missing")
    index_path = Path(str(index_source["path"]))
    if sha256_file(index_path) != index_source["sha256"]:
        raise ContractError("bound index series SHA drifted")
    receipt_path = Path(str(index_source.get("receipt_path", "")))
    receipt_sha256 = index_source.get("receipt_sha256")
    if not receipt_path.is_file() or not receipt_sha256 or sha256_file(receipt_path) != receipt_sha256:
        raise ContractError("bound index recovery receipt is missing or its SHA drifted")
    receipt = load_json(receipt_path)
    if (receipt.get("normalized_sha256") != index_source["sha256"] or
            receipt.get("index_code") != index_code or
            Path(str(receipt.get("normalized_path", ""))).name != index_path.name or
            (index_source.get("row_count") is not None and
             int(receipt.get("normalized_row_count", -1)) != int(index_source["row_count"]))):
        raise ContractError("index recovery receipt does not bind the normalized source identity")
    expected_header = index_source.get("csv_header")
    if expected_header is not None:
        header = pd.read_csv(index_path, nrows=0).columns.tolist()
        if header != expected_header:
            raise ContractError("index CSV header differs from the bound source schema")
    date_column, code_column, close_column = expected_columns
    index_frame = pd.read_csv(
        index_path,
        usecols=source_columns,
        dtype={date_column: "string", code_column: "string"},
    )
    if list(index_frame.columns) != source_columns:
        raise ContractError("index CSV selected columns do not match the bound order")
    codes = index_frame[code_column]
    if codes.isna().any() or not codes.eq(index_code).all():
        raise ContractError("index CSV contains a code outside its exact bound identity")
    dates = _parse_bound_dates(index_frame[date_column], field_name="index date")
    close_raw = index_frame[close_column]
    close = pd.to_numeric(close_raw, errors="coerce")
    close_values = close.to_numpy(dtype=np.float64, na_value=np.nan)
    if (close.isna().any() or not np.isfinite(close_values).all() or (close_values <= 0).any()):
        raise ContractError("index close contains a missing, nonfinite, or nonpositive value")
    if dates.duplicated().any():
        raise ContractError("index series contains duplicate dates")
    if index_source.get("row_count") is not None and len(index_frame) != int(index_source["row_count"]):
        raise ContractError("index row count differs from the bound source metadata")
    expected_range = index_source.get("date_range")
    if expected_range is not None and [dates.min().date().isoformat(), dates.max().date().isoformat()] != expected_range:
        raise ContractError("index date range differs from the bound source metadata")
    missing_index_dates = set(pd.DatetimeIndex(panel_dates).normalize()) - set(dates)
    if missing_index_dates:
        raise ContractError(f"index series misses {len(missing_index_dates)} observed panel dates")
    index_frame = pd.DataFrame({"date": dates, "close": close_values}).sort_values("date", kind="mergesort")
    return pd.Series(index_frame["close"].to_numpy(dtype=np.float64),
                      index=pd.DatetimeIndex(index_frame["date"]), dtype=np.float64)


def normalize_calendar(calendar: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Validate the bound shared date index and every panel date's membership."""
    if not {"trade_date", "calendar_index"}.issubset(calendar.columns):
        raise ContractError("calendar must bind trade_date and calendar_index")
    out = calendar[["trade_date", "calendar_index"]].copy()
    out["date"] = _parse_bound_dates(out.pop("trade_date"), field_name="calendar trade_date")
    raw_ordinals = out["calendar_index"]
    if pd.api.types.is_bool_dtype(raw_ordinals.dtype):
        raise ContractError("calendar ordinal must be an integer")
    ordinals = pd.to_numeric(raw_ordinals, errors="coerce")
    ordinal_values = ordinals.to_numpy(dtype=np.float64, na_value=np.nan)
    if (not np.isfinite(ordinal_values).all() or
            not np.equal(ordinal_values, np.floor(ordinal_values)).all() or
            (ordinal_values < np.iinfo(np.int64).min).any() or
            (ordinal_values > np.iinfo(np.int64).max).any()):
        raise ContractError("calendar ordinal must be an integer")
    out["calendar_index"] = ordinal_values.astype(np.int64)
    out = out.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    if out["date"].duplicated().any() or out["calendar_index"].duplicated().any():
        raise ContractError("calendar date and ordinal must be unique")
    if not np.array_equal(out["calendar_index"].to_numpy(), np.arange(len(out), dtype=np.int64)):
        raise ContractError("calendar ordinal must be contiguous from zero")
    date_ordinals = out["date"].astype("int64").to_numpy()
    if len(date_ordinals) > 1 and not (np.diff(date_ordinals) > 0).all():
        raise ContractError("calendar dates must strictly increase with calendar_index")
    absent = set(pd.to_datetime(panel["date"]).dt.normalize()) - set(out["date"])
    if absent:
        raise ContractError(f"panel dates absent from bound calendar: {len(absent)}")
    return out[["date", "calendar_index"]]


def apply_ohlc_taint_masks(
    formula_features: pd.DataFrame,
    arm_frames: Mapping[str, pd.DataFrame],
    calendar: pd.DataFrame,
    quarantine: Mapping[str, Any],
    *,
    industry_classification: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Mask recursive and exact-window dependencies without changing source keys."""
    if not set(KEYS + CORE_FEATURES).issubset(formula_features.columns):
        raise ContractError("formula feature frame is missing frozen core outputs")
    if not {"date", "calendar_index"}.issubset(calendar.columns):
        raise ContractError("OHLC taint masking requires the normalized session calendar")
    if formula_features.duplicated(list(KEYS)).any():
        raise ContractError("formula feature frame has duplicate date/code keys")

    normalized_calendar = calendar[["date", "calendar_index"]].copy()
    normalized_calendar["date"] = pd.to_datetime(normalized_calendar["date"]).dt.normalize()
    date_to_index = dict(zip(normalized_calendar["date"], normalized_calendar["calendar_index"].astype(int)))
    if len(date_to_index) != len(normalized_calendar):
        raise ContractError("OHLC taint calendar contains duplicate dates")

    raw_quarantined = quarantine.get("quarantined_keys")
    if not isinstance(raw_quarantined, list):
        raise ContractError("OHLC quarantine keys must be a list")
    taint_anchors: list[tuple[pd.Timestamp, str, int, tuple[str, ...]]] = []
    first_bad_by_code: dict[str, int] = {}
    for item in raw_quarantined:
        if not isinstance(item, Mapping):
            raise ContractError("OHLC quarantine entry must be an object")
        date = pd.Timestamp(item.get("date")).normalize()
        code = str(item.get("code", "")).strip()
        reason_codes = tuple(sorted(str(value) for value in item.get("reason_codes", ())))
        if not code or not reason_codes or date not in date_to_index:
            raise ContractError("OHLC quarantine entry has an unbound date, code, or reason")
        session = int(date_to_index[date])
        taint_anchors.append((date, code, session, reason_codes))
        first_bad_by_code[code] = min(first_bad_by_code.get(code, session), session)
    taint_anchors.sort(key=lambda item: (item[2], item[1]))

    industry_for_code: dict[str, str] = {}
    codes_by_industry: dict[str, set[str]] = {}
    if any(arm in {"M3", "M4"} for arm in arm_frames):
        if industry_classification is None or not {"code", "industry_code"}.issubset(industry_classification.columns):
            raise ContractError("M3/M4 OHLC taint masking requires the bound industry map")
        mapping = industry_classification[["code", "industry_code"]].copy()
        mapping["code"] = mapping["code"].astype(str).str.strip()
        mapping["industry_code"] = mapping["industry_code"].astype("string").str.strip()
        if mapping["code"].eq("").any() or mapping["code"].duplicated().any():
            raise ContractError("OHLC taint industry map must have one row per exact code")
        for code, industry in mapping.itertuples(index=False, name=None):
            if pd.isna(industry) or not str(industry).strip():
                continue
            industry_text = str(industry).strip()
            industry_for_code[str(code)] = industry_text
            codes_by_industry.setdefault(industry_text, set()).add(str(code))

    # These frames are transient preparation products. Mask them in place to
    # avoid holding another full multi-year panel copy beside the raw outcomes.
    formula_out = formula_features
    formula_dates = pd.to_datetime(formula_out["date"]).dt.normalize()
    formula_sessions = formula_dates.map(date_to_index)
    if formula_sessions.isna().any():
        raise ContractError("formula feature frame contains dates outside the bound calendar")
    for code, first_bad_session in first_bad_by_code.items():
        suffix = formula_out["code"].astype(str).str.strip().eq(code) & (formula_sessions >= first_bad_session)
        formula_out.loc[suffix, list(CORE_FEATURES)] = np.nan

    frame_out: dict[str, pd.DataFrame] = {}
    entries_by_arm: dict[str, dict[tuple[str, str], dict[str, set[str]]]] = {}

    def mark_rows(arm: str, frame: pd.DataFrame, mask: np.ndarray, reason: str,
                  requested_fields: Sequence[str]) -> None:
        fields = [field for field in requested_fields if field in frame.columns]
        if not fields:
            return
        frame.loc[mask, fields] = np.nan
        entries = entries_by_arm[arm]
        selected = frame.loc[mask, list(KEYS)]
        for date_value, code_value in selected.itertuples(index=False, name=None):
            key = (pd.Timestamp(date_value).normalize().date().isoformat(), str(code_value).strip())
            entry = entries.setdefault(key, {"reason_codes": set(), "masked_feature_fields": set()})
            entry["reason_codes"].add(reason)
            entry["masked_feature_fields"].update(fields)

    for arm, source_frame in arm_frames.items():
        if arm not in ARMS:
            raise ContractError(f"unknown arm in OHLC taint input: {arm}")
        if not set(KEYS + CORE_FEATURES).issubset(source_frame.columns):
            raise ContractError(f"{arm} frame is missing frozen core feature outputs")
        if source_frame.duplicated(list(KEYS)).any():
            raise ContractError(f"{arm} frame has duplicate date/code keys")
        frame = source_frame
        frame_dates = pd.to_datetime(frame["date"]).dt.normalize()
        frame_sessions = frame_dates.map(date_to_index)
        if frame_sessions.isna().any():
            raise ContractError(f"{arm} frame contains dates outside the bound calendar")
        frame_sessions = frame_sessions.to_numpy(dtype=np.int64)
        frame_codes = frame["code"].astype(str).str.strip()
        entries_by_arm[arm] = {}

        for code, first_bad_session in first_bad_by_code.items():
            suffix = frame_codes.eq(code).to_numpy() & (frame_sessions >= first_bad_session)
            mark_rows(arm, frame, suffix, "RECURSIVE_FORMULA_TAINT_SUFFIX", CORE_FEATURES)

        if arm in INDEXED_ARMS:
            required = ("stock_rs5", "market_advancer_share")
            if not set(required).issubset(frame.columns):
                raise ContractError(f"{arm} frame is missing OHLC-dependent M1 features")
            for _, code, session, _ in taint_anchors:
                market_mask = (frame_sessions >= session) & (frame_sessions <= session + 1)
                mark_rows(arm, frame, market_mask,
                          "MARKET_RET1_DEPENDENCY_TAINT_T_TO_T_PLUS_1", ("market_advancer_share",))
                stock_mask = frame_codes.eq(code).to_numpy() & (frame_sessions >= session) & (frame_sessions <= session + 5)
                mark_rows(arm, frame, stock_mask,
                          "STOCK_RETURN_5_DEPENDENCY_TAINT_T_TO_T_PLUS_5", ("stock_rs5",))

        if arm in {"M3", "M4"}:
            required = ("industry_advancer_share", "industry_mean_ret5", "stock_relative_industry_ret5",
                        "industry_mean_ret20")
            if not set(required).issubset(frame.columns):
                raise ContractError(f"{arm} frame is missing OHLC-dependent industry features")
            for _, code, session, _ in taint_anchors:
                industry = industry_for_code.get(code)
                if industry is None:
                    continue
                member_codes = codes_by_industry[industry]
                member_mask = frame_codes.isin(member_codes).to_numpy()
                advancer_mask = member_mask & (frame_sessions >= session) & (frame_sessions <= session + 1)
                mark_rows(arm, frame, advancer_mask,
                          "INDUSTRY_RET1_DEPENDENCY_TAINT_T_TO_T_PLUS_1", ("industry_advancer_share",))
                ret5_mask = member_mask & (frame_sessions >= session) & (frame_sessions <= session + 5)
                mark_rows(arm, frame, ret5_mask,
                          "INDUSTRY_RET5_DEPENDENCY_TAINT_T_TO_T_PLUS_5",
                          ("industry_mean_ret5", "stock_relative_industry_ret5"))
                ret20_mask = member_mask & (frame_sessions >= session) & (frame_sessions <= session + 20)
                mark_rows(arm, frame, ret20_mask,
                          "INDUSTRY_RET20_DEPENDENCY_TAINT_T_TO_T_PLUS_20", ("industry_mean_ret20",))
        frame_out[arm] = frame

    derived_by_arm: dict[str, list[dict[str, Any]]] = {}
    coverage_by_arm: dict[str, dict[str, Any]] = {}
    for arm in sorted(entries_by_arm):
        entries = [
            {"date": date, "code": code,
             "reason_codes": sorted(value["reason_codes"]),
             "masked_feature_fields": sorted(value["masked_feature_fields"])}
            for (date, code), value in sorted(entries_by_arm[arm].items())
        ]
        derived_by_arm[arm] = entries
        by_date: dict[str, int] = {}
        by_code: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        for entry in entries:
            by_date[entry["date"]] = by_date.get(entry["date"], 0) + 1
            by_code[entry["code"]] = by_code.get(entry["code"], 0) + 1
            for reason in entry["reason_codes"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        coverage_by_arm[arm] = {
            "tainted_key_count": len(entries),
            "tainted_date_count": len(by_date),
            "tainted_code_count": len(by_code),
            "by_date": [{"date": date, "key_count": by_date[date]} for date in sorted(by_date)],
            "by_code": [{"code": code, "key_count": by_code[code]} for code in sorted(by_code)],
            "reason_counts": dict(sorted(reason_counts.items())),
        }

    sidecar = {
        "schema_version": "A_SHARE_MODEL_SELECTION_INPUT_COMPAT_SIDECAR_V1",
        "status": "COMPLETE" if taint_anchors else "NO_OHLC_QUARANTINE_ROWS",
        "source_row_accounting": {
            "source_row_count": int(quarantine.get("source_row_count", -1)),
            "accepted_row_count": int(quarantine.get("accepted_row_count", -1)),
            "quarantined_row_count": int(quarantine.get("quarantined_row_count", -1)),
            "accepted_plus_quarantined_row_count": int(quarantine.get("accepted_plus_quarantined_row_count", -1)),
            "row_count_conserved": bool(quarantine.get("row_count_conserved", False)),
            "duplicate_date_code_key_count": int(quarantine.get("duplicate_date_code_key_count", -1)),
        },
        "original_quarantined_keys": raw_quarantined,
        "taint_policy": {
            "core_formula": "RECURSIVE_CODE_SUFFIX_FROM_EARLIEST_NONPOSITIVE_OHLC_SESSION",
            "market_advancer_share": "ALL_CODES_AT_T_AND_T_PLUS_1",
            "stock_rs5": "SAME_CODE_AT_T_THROUGH_T_PLUS_5",
            "industry_advancer_share": "SAME_KNOWN_INDUSTRY_AT_T_AND_T_PLUS_1",
            "industry_mean_ret5_and_stock_relative_industry_ret5": "SAME_KNOWN_INDUSTRY_AT_T_THROUGH_T_PLUS_5",
            "industry_mean_ret20": "SAME_KNOWN_INDUSTRY_AT_T_THROUGH_T_PLUS_20",
            "industry_amount_activity_and_one_hot": "UNCHANGED",
            "market_amount_and_index_features": "UNCHANGED",
            "labels": "RAW_NORMALIZED_PANEL_CLOSE_ONLY",
        },
        "derived_tainted_keys_by_arm": derived_by_arm,
        "coverage_loss_by_arm": coverage_by_arm,
    }
    return formula_out, frame_out, sidecar


def _date_maps(calendar: pd.DataFrame) -> tuple[dict[pd.Timestamp, int], dict[int, pd.Timestamp]]:
    return (dict(zip(calendar["date"], calendar["calendar_index"].astype(int))),
            dict(zip(calendar["calendar_index"].astype(int), calendar["date"])))


def session_lag(panel: pd.DataFrame, calendar: pd.DataFrame, value_column: str, lag: int) -> pd.Series:
    """Return the same code's value exactly ``lag`` bound sessions earlier.

    Missing stock rows at that exact session remain missing; this never shifts
    across a suspension by taking the previous available stock row.
    """
    if lag <= 0 or value_column not in panel:
        raise ContractError("session lag requires a positive lag and a known source column")
    date_to_index, index_to_date = _date_maps(calendar)
    source = panel[["date", "code", value_column]].copy()
    if source.duplicated(["date", "code"]).any():
        raise ContractError("duplicate date/code keys in session lag source")
    source["_session"] = pd.to_datetime(source["date"]).dt.normalize().map(date_to_index)
    target_dates = [
        index_to_date.get(int(value) + lag) if pd.notna(value) else pd.NaT
        for value in source["_session"].to_numpy()
    ]
    source["date"] = pd.Series(target_dates, dtype=panel["date"].dtype).array
    source = source.rename(columns={value_column: f"{value_column}_lag{lag}"}).drop(columns="_session")
    source = source.loc[source["date"].notna()]
    left = panel[["date", "code"]].copy()
    left["_position"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(source, on=["date", "code"], how="left", validate="one_to_one", sort=False)
    merged = merged.sort_values("_position", kind="mergesort")
    return pd.Series(merged[f"{value_column}_lag{lag}"].to_numpy(), index=panel.index)


def build_close_return_labels(panel: pd.DataFrame, horizons: Sequence[int] = HORIZONS,
                              *, calendar: pd.DataFrame | None = None,
                              signal_start: pd.Timestamp | str | None = None,
                              signal_end_exclusive: pd.Timestamp | str | None = None,
                              signal_keys: pd.DataFrame | None = None) -> dict[int, pd.DataFrame]:
    """Create close(T+h)/close(T)-1 labels on the shared observed date axis.

    A missing target close stays NaN.  Labels never determine the scoring pool
    and never cause a lower-ranked candidate to replace an unknown selection.
    """
    if not set(KEYS + ("close",)).issubset(panel.columns):
        raise ContractError("panel must contain date, code, close")
    if panel.duplicated(list(KEYS)).any():
        raise ContractError("duplicate date/code keys")
    if calendar is None:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).dt.normalize().unique()))
        position = {date: i for i, date in enumerate(dates)}
        index_to_date = dict(enumerate(dates))
    else:
        date_to_index, index_to_date = _date_maps(calendar)
        dates = pd.DatetimeIndex(calendar["date"])
        if not set(pd.to_datetime(panel["date"]).dt.normalize()).issubset(date_to_index):
            raise ContractError("label panel contains dates outside the bound shared calendar")
        position = date_to_index
    indexed = panel[["date", "code", "close"]].copy()
    indexed["date"] = pd.to_datetime(indexed["date"]).dt.normalize()
    indexed["_pos"] = indexed["date"].map(position)
    base_rows = indexed
    if signal_keys is not None:
        if not set(KEYS).issubset(signal_keys.columns) or signal_keys.duplicated(list(KEYS)).any():
            raise ContractError("selected label keys must have unique date/code pairs")
        selected_keys = signal_keys[["date", "code"]].copy()
        selected_keys["date"] = pd.to_datetime(selected_keys["date"], errors="coerce").dt.normalize()
        selected_keys["code"] = selected_keys["code"].astype(str)
        base_rows = indexed.merge(selected_keys, on=list(KEYS), how="inner", validate="one_to_one", sort=False)
    else:
        if signal_start is not None:
            base_rows = base_rows.loc[base_rows["date"] >= pd.Timestamp(signal_start).normalize()]
        if signal_end_exclusive is not None:
            base_rows = base_rows.loc[base_rows["date"] < pd.Timestamp(signal_end_exclusive).normalize()]
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        if int(h) <= 0:
            raise ValueError("horizon must be positive")
        target_pos = base_rows["_pos"] + int(h)
        target_dates = target_pos.map(lambda p: index_to_date.get(int(p), pd.NaT))
        base = base_rows[["date", "code", "close"]].rename(columns={"close": "signal_close"}).copy()
        base["target_date"] = target_dates.to_numpy()
        target = indexed[["date", "code", "close"]].rename(
            columns={"date": "target_date", "close": "target_close"}
        )
        labels = base.merge(target, on=["target_date", "code"], how="left", validate="many_to_one", sort=False)
        valid = (labels["signal_close"] > 0) & (labels["target_close"] > 0)
        labels["label_return"] = np.where(valid, labels["target_close"] / labels["signal_close"] - 1.0, np.nan)
        labels["label_end_date"] = labels["target_date"]
        labels = labels[["date", "code", "target_date", "label_end_date", "label_return"]]
        out[int(h)] = labels.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)
    return out


def _session_return(panel: pd.DataFrame, calendar: pd.DataFrame, lag: int) -> pd.Series:
    prior = session_lag(panel, calendar, "close", lag)
    current = pd.to_numeric(panel["close"], errors="coerce")
    return current.div(prior.where(prior > 0)).sub(1.0)


def _calendar_daily_market(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    rows = panel[["date", "code", "amount"]].copy()
    rows["ret1"] = _session_return(panel, calendar, 1)
    daily = rows.groupby("date", sort=True).agg(
        market_advancer_share=("ret1", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan),
        market_total_amount=("amount", "sum"),
    )
    daily = daily.reindex(pd.DatetimeIndex(calendar["date"]))
    daily.index.name = "date"
    ma20 = daily["market_total_amount"].rolling(20, min_periods=20).mean()
    daily["market_amount_activity"] = daily["market_total_amount"].div(ma20.where(ma20 > 0))
    return daily.reset_index()


def build_m1_features(panel: pd.DataFrame, calendar: pd.DataFrame,
                      index_close_by_date: pd.Series) -> pd.DataFrame:
    """Build only the frozen M1 market/index features on the shared date axis."""
    index = pd.Series(index_close_by_date.to_numpy(dtype=np.float64),
                       index=pd.to_datetime(index_close_by_date.index).normalize(), dtype=np.float64)
    if index.index.duplicated().any() or (index.dropna() <= 0).any():
        raise ContractError("index close dates must be unique and positive")
    if panel["amount"].isna().any():
        raise ContractError("market amount activity cannot be aggregated with missing panel amount")
    index = index.reindex(pd.DatetimeIndex(calendar["date"]))
    index.name = "index_close"
    index_frame = index.rename_axis("date").reset_index()
    index_ma5 = index.rolling(5, min_periods=5).mean()
    index_ma20 = index.rolling(20, min_periods=20).mean()
    index_features = pd.DataFrame({
        "date": index.index,
        "index_ma5_position": index.div(index_ma5.where(index_ma5 > 0)).sub(1.0).to_numpy(),
        "index_ma20_position": index.div(index_ma20.where(index_ma20 > 0)).sub(1.0).to_numpy(),
        "index_return_1": index.pct_change(1, fill_method=None).to_numpy(),
        "index_return_5": index.pct_change(5, fill_method=None).to_numpy(),
    })
    stock = panel[["date", "code", "close"]].copy()
    stock["stock_return_5"] = _session_return(panel, calendar, 5).to_numpy(dtype=np.float64)
    stock = stock.merge(index_features[["date", "index_return_5"]], on="date", how="left", validate="many_to_one")
    stock["stock_rs5"] = stock["stock_return_5"] - stock["index_return_5"]
    market = _calendar_daily_market(panel, calendar)
    return (stock[["date", "code", "stock_rs5"]]
            .merge(index_features, on="date", how="left", validate="many_to_one")
            .merge(market, on="date", how="left", validate="many_to_one")
            .drop(columns=["market_total_amount"]))


def _industry_token(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"industry_cat_{digest}"


def build_m3_features(panel: pd.DataFrame, calendar: pd.DataFrame,
                      industry_classification: pd.DataFrame) -> pd.DataFrame:
    """Build retrospective industry aggregates while preserving missing codes."""
    required = {"code", "industry_code"}
    if not required.issubset(industry_classification.columns):
        raise ContractError("industry source must bind code and industry_code")
    mapping = industry_classification[["code", "industry_code"]].copy()
    mapping["code"] = mapping["code"].astype(str).str.strip()
    mapping["industry_code"] = mapping["industry_code"].astype("string").str.strip()
    if mapping["code"].eq("").any() or mapping["code"].duplicated().any():
        raise ContractError("industry code mapping must be one row per exact stock code")
    if panel["amount"].isna().any():
        raise ContractError("industry amount activity cannot be aggregated with missing panel amount")
    rows = panel[["date", "code", "close", "amount"]].copy().merge(
        mapping, on="code", how="left", validate="many_to_one", sort=False
    )
    rows["ret1"] = _session_return(panel, calendar, 1).to_numpy(dtype=np.float64)
    rows["ret5"] = _session_return(panel, calendar, 5).to_numpy(dtype=np.float64)
    rows["ret20"] = _session_return(panel, calendar, 20).to_numpy(dtype=np.float64)
    known = rows.loc[rows["industry_code"].notna()].copy()
    if known.empty:
        raise ContractError("industry source has no exact panel-code matches")
    daily = known.groupby(["industry_code", "date"], sort=True).agg(
        industry_mean_ret5=("ret5", "mean"),
        industry_mean_ret20=("ret20", "mean"),
        industry_advancer_share=("ret1", lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan),
        industry_total_amount=("amount", "sum"),
    )
    categories = sorted(mapping["industry_code"].dropna().astype(str).unique())
    full_index = pd.MultiIndex.from_product([categories, pd.DatetimeIndex(calendar["date"])],
                                            names=["industry_code", "date"])
    daily = daily.reindex(full_index).reset_index()
    daily = daily.sort_values(["industry_code", "date"], kind="mergesort")
    daily["industry_amount_activity"] = daily.groupby("industry_code", sort=False)["industry_total_amount"].transform(
        lambda values: values.div(values.rolling(20, min_periods=20).mean().where(values.rolling(20, min_periods=20).mean() > 0))
    )
    out = rows[["date", "code", "industry_code", "ret5"]].merge(
        daily.drop(columns="industry_total_amount"), on=["date", "industry_code"], how="left", validate="many_to_one"
    )
    out["stock_relative_industry_ret5"] = out["ret5"] - out["industry_mean_ret5"]
    one_hot = pd.DataFrame(index=out.index)
    for category in categories:
        one_hot[_industry_token(category)] = out["industry_code"].eq(category).astype(np.float64)
    result = out[["date", "code", "industry_mean_ret5", "industry_mean_ret20",
                  "industry_advancer_share", "industry_amount_activity",
                  "stock_relative_industry_ret5"]].copy()
    return pd.concat([result.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1)


def build_m4_features(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Build the user-defined observed-market amount proxy, not official 0AMV."""
    if panel["amount"].isna().any():
        raise ContractError("aggregate amount proxy cannot use missing panel amount")
    first_date = pd.to_datetime(panel["date"]).min()
    last_date = pd.to_datetime(panel["date"]).max()
    session_dates = pd.DatetimeIndex(calendar.loc[calendar["date"].between(first_date, last_date), "date"])
    daily = panel.groupby("date", sort=True)["amount"].sum().reindex(session_dates)
    if daily.isna().any():
        raise ContractError("aggregate amount proxy requires an observed total on every bound calendar session")
    daily.index.name = "date"
    values = daily.to_numpy(dtype=np.float64)
    smooth = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        smooth[i] = value if i == 0 or not np.isfinite(smooth[i - 1]) else 0.1 * value + 0.9 * smooth[i - 1]
    proxy = pd.Series(smooth, index=daily.index, dtype=np.float64)
    ma20 = proxy.rolling(20, min_periods=20).mean()
    ma5 = proxy.rolling(5, min_periods=5).mean()
    ma13 = proxy.rolling(13, min_periods=13).mean()
    features = pd.DataFrame({
        "date": proxy.index,
        "amv_proxy_sma10": proxy.to_numpy(),
        "amv_proxy_change5": proxy.div(proxy.shift(5).where(proxy.shift(5) > 0)).sub(1.0).to_numpy(),
        "amv_proxy_vs_ma20": proxy.div(ma20.where(ma20 > 0)).sub(1.0).to_numpy(),
        "amv_proxy_ma5_vs_ma13": ma5.div(ma13.where(ma13 > 0)).sub(1.0).to_numpy(),
    })
    return panel[["date", "code"]].merge(features, on="date", how="left", validate="many_to_one")


def build_nested_arm_frames(panel: pd.DataFrame, formula_features: pd.DataFrame, calendar: pd.DataFrame,
                            *, index_close_by_date: pd.Series | None = None,
                            industry_classification: pd.DataFrame | None = None,
                            requested_arms: Sequence[str] = ARMS) -> dict[str, pd.DataFrame]:
    """Build only nested arm prefixes whose bound dependencies are available."""
    requested = tuple(requested_arms)
    if not requested or requested[0] != "M0" or requested != ARMS[:len(requested)]:
        raise ContractError("requested arms must be a non-empty nested prefix beginning at M0")
    if not set({"date", "code", *CORE_FEATURES}).issubset(formula_features.columns):
        raise ContractError("formula feature frame does not contain the frozen M0 core")
    m0 = formula_features[["date", "code", *CORE_FEATURES]].copy()
    if m0.duplicated(list(KEYS)).any():
        raise ContractError("formula feature frame has duplicate keys")
    frames: dict[str, pd.DataFrame] = {"M0": m0.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)}
    if len(requested) == 1:
        return frames
    if index_close_by_date is None:
        raise ContractError("M1 and dependent arms require the bound 000300 index series")
    m1_features = build_m1_features(panel, calendar, index_close_by_date)
    m1 = m0.merge(m1_features, on=list(KEYS), how="left", validate="one_to_one")
    frames["M1"] = m1
    if len(requested) == 2:
        return frames
    if "AUDITM" not in formula_features:
        raise ContractError("M2 requires AUDITM output from the exact V193 formula source")
    month = m1[["date", "code"]].copy()
    month_number = pd.to_datetime(month["date"]).dt.month
    for number in range(1, 13):
        month[f"month_{number:02d}"] = (month_number == number).astype(np.float64)
    audit = formula_features[["date", "code", "AUDITM"]].copy()
    month = month.merge(audit, on=list(KEYS), how="left", validate="one_to_one")
    frames["M2"] = m1.merge(month, on=list(KEYS), how="left", validate="one_to_one")
    if len(requested) == 3:
        return frames
    if industry_classification is None:
        raise ContractError("M3 and M4 require the bound retrospective industry map")
    industry = build_m3_features(panel, calendar, industry_classification)
    frames["M3"] = frames["M2"].merge(industry, on=list(KEYS), how="left", validate="one_to_one")
    if len(requested) == 4:
        return frames
    proxy = build_m4_features(panel, calendar)
    frames["M4"] = frames["M3"].merge(proxy, on=list(KEYS), how="left", validate="one_to_one")
    return frames


def assemble_common_arm_features(arm_frames: Mapping[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, ...]]]:
    """Apply a strict shared feature-complete key intersection to every arm."""
    if set(arm_frames) != set(ARMS):
        raise ContractError(f"expected exact arm set {ARMS}")
    normalized: dict[str, pd.DataFrame] = {}
    key_sets: list[set[tuple[pd.Timestamp, str]]] = []
    for arm in ARMS:
        frame = arm_frames[arm].copy()
        if not set(KEYS).issubset(frame.columns) or frame.duplicated(list(KEYS)).any():
            raise ContractError(f"{arm} has missing keys or duplicate rows")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["code"] = frame["code"].astype(str)
        feature_cols = [c for c in frame.columns if c not in KEYS]
        if not feature_cols:
            raise ContractError(f"{arm} has no features")
        for col in feature_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        finite = np.isfinite(frame[feature_cols].to_numpy(dtype=np.float64)).all(axis=1)
        frame = frame.loc[finite].sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
        keys = set(zip(frame["date"], frame["code"]))
        key_sets.append(keys)
        normalized[arm] = frame
    common = set.intersection(*key_sets)
    if not common:
        raise ContractError("no common feature-complete scoring keys across M0-M4")
    keys_frame = pd.DataFrame(sorted(common), columns=KEYS)
    result: dict[str, pd.DataFrame] = {}
    columns: dict[str, tuple[str, ...]] = {}
    for arm, frame in normalized.items():
        merged = keys_frame.merge(frame, on=list(KEYS), how="left", validate="one_to_one", sort=False)
        cols = tuple(c for c in merged.columns if c not in KEYS)
        result[arm] = merged.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
        columns[arm] = cols
    return result, columns


def model_for_arm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def equal_day_weights(dates: pd.Series) -> np.ndarray:
    """Give each signal date equal total weight and normalize mean weight to 1."""
    normalized = pd.to_datetime(dates).dt.normalize()
    counts = normalized.value_counts(sort=False)
    weights = normalized.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=np.float64)
    if not len(weights):
        raise ContractError("cannot weight an empty training set")
    return weights * (len(weights) / weights.sum())


def fixed_folds() -> tuple[Fold, ...]:
    return tuple(Fold(name, pd.Timestamp(start), pd.Timestamp(end), partial)
                 for name, start, end, partial in FORMAL_FOLDS)


def stage_plan(stage: str, *, attempt: str = "attempt_01") -> tuple[dict[str, Any], ...]:
    """Return the frozen work cells for one approved stage."""
    if stage not in STAGE_SPECS:
        raise ContractError(f"unknown model-selection stage: {stage}")
    spec = STAGE_SPECS[stage]
    folds = {fold.fold_id: fold for fold in fixed_folds()}
    cells = []
    for arm in spec["arms"]:
        for horizon in spec["horizons"]:
            for fold_id in spec["folds"]:
                cells.append({"stage": stage, "arm": arm, "horizon": int(horizon),
                              "fold": folds[fold_id], "canary": bool(spec["canary"]),
                              "attempt": attempt})
    return tuple(cells)


def arm_source_identity(binding: Mapping[str, Any], arm: str) -> dict[str, str]:
    """Return only component SHAs that can change this arm's fitted values."""
    sources = binding.get("sources", {})
    formulas = binding.get("formula_inputs", {})
    identity: dict[str, str] = {}
    for key in ("price_panel",):
        value = sources.get(key, {})
        if value.get("status") != "BOUND" or not value.get("sha256"):
            raise ContractError(f"required source SHA missing for {arm}: {key}")
        identity[key] = str(value["sha256"])
    for key in ("v192", "v193"):
        value = formulas.get(key, {})
        if not value.get("sha256"):
            raise ContractError(f"required formula SHA missing for {arm}: {key}")
        identity[f"formula_{key}"] = str(value["sha256"])
    calendar = sources.get("trading_date_index", {})
    if not calendar.get("member_sha256"):
        raise ContractError(f"bound calendar member SHA missing for {arm}")
    identity["calendar_member"] = str(calendar["member_sha256"])
    engine = binding.get("formula_engine", {})
    for key in ("tdx_ast_engine_sha256", "fast_builder_c_sha256", "fast_loops_dylib_sha256"):
        if not engine.get(key):
            raise ContractError(f"formula runtime identity missing: {key}")
        identity[f"formula_runtime_{key}"] = str(engine[key])
    if arm in INDEXED_ARMS:
        index = sources.get("index_series", {})
        if index.get("status") != "BOUND" or not index.get("sha256"):
            raise ContractError("M1-M4 checkpoint identity requires a SHA-bound 000300 series")
        identity["index_series"] = str(index["sha256"])
    if arm in {"M3", "M4"}:
        industry = sources.get("industry_classification", {})
        if not industry.get("member_sha256"):
            raise ContractError(f"industry mapping SHA missing for {arm}")
        identity["industry_member"] = str(industry["member_sha256"])
    return identity


def key_frame_identity(frame: pd.DataFrame) -> dict[str, Any]:
    """Return a versioned, order-independent identity for exact date/code keys."""
    if not set(KEYS).issubset(frame.columns):
        raise ContractError("key identity frame must contain date and code")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    codes = frame["code"]
    if dates.isna().any() or codes.isna().any():
        raise ContractError("key identity frame contains a missing date or code")
    code_values = codes.astype(str)
    if code_values.str.strip().eq("").any():
        raise ContractError("key identity frame contains an empty code")
    ordered = [[date.strftime("%Y-%m-%d"), code]
               for date, code in zip(dates, code_values)]
    if len(set((row[0], row[1]) for row in ordered)) != len(ordered):
        raise ContractError("key identity frame contains duplicate date/code keys")
    sorted_keys = sorted(ordered, key=lambda row: (row[0], row[1]))
    return {
        "schema_version": KEY_IDENTITY_SCHEMA,
        "encoding": "ISO-8601-date-and-exact-UTF8-code; sorted by date then code",
        "row_count": int(len(ordered)),
        "unique_key_count": int(len(sorted_keys)),
        "unique_date_count": int(dates.nunique()),
        "date_min": dates.min().strftime("%Y-%m-%d") if len(dates) else None,
        "date_max": dates.max().strftime("%Y-%m-%d") if len(dates) else None,
        "sorted_keyset_sha256": canonical_sha256({
            "schema_version": KEY_IDENTITY_SCHEMA, "keys": sorted_keys,
        }),
        "ordered_keys_sha256": canonical_sha256({
            "schema_version": f"{KEY_IDENTITY_SCHEMA}_ORDERED", "keys": ordered,
        }),
    }


def numeric_matrix_identity(values: Any, columns: Sequence[str]) -> dict[str, Any]:
    """Hash a finite numeric matrix by explicit little-endian float64 bytes."""
    array = np.asarray(values, dtype="<f8", order="C")
    if array.ndim != 2:
        raise ContractError("matrix identity input must be two-dimensional")
    if array.shape[1] != len(columns):
        raise ContractError("matrix identity feature order does not match matrix width")
    if not np.isfinite(array).all():
        raise ContractError("matrix identity input contains a non-finite value")
    payload = {
        "schema_version": MATRIX_IDENTITY_SCHEMA,
        "dtype": "float64-little-endian-C-order",
        "shape": [int(array.shape[0]), int(array.shape[1])],
        "columns": [str(column) for column in columns],
        "values_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def numeric_vector_identity(values: Any, *, name: str) -> dict[str, Any]:
    """Hash one finite numeric training vector with a versioned representation."""
    array = np.asarray(values, dtype="<f8", order="C").reshape(-1, 1)
    result = numeric_matrix_identity(array, (str(name),))
    return {**result, "vector_name": str(name), "row_count": int(array.shape[0])}


def cell_data_identity(*, train: pd.DataFrame, test: pd.DataFrame,
                       feature_columns: Sequence[str], x: np.ndarray, y: np.ndarray,
                       weights: np.ndarray, xt: np.ndarray) -> dict[str, Any]:
    """Bind the exact finite train and unlabeled test slices passed to a cell."""
    columns = tuple(str(column) for column in feature_columns)
    if not columns or any(column not in train or column not in test for column in columns):
        raise ContractError("checkpoint evidence feature order is incomplete")
    x_array, y_array, weight_array, xt_array = (np.asarray(value) for value in (x, y, weights, xt))
    if (x_array.ndim != 2 or xt_array.ndim != 2 or y_array.ndim != 1 or weight_array.ndim != 1 or
            len(x_array) != len(train) or len(y_array) != len(train) or len(weight_array) != len(train) or
            len(xt_array) != len(test)):
        raise ContractError("checkpoint evidence matrices/vectors do not align to their exact key rows")
    train_keys = key_frame_identity(train)
    test_keys = key_frame_identity(test)
    return {
        "schema_version": "A_SHARE_CANARY_CELL_DATA_IDENTITY_V1",
        "training_keys": train_keys,
        "test_keys": test_keys,
        "feature_columns": list(columns),
        "training_feature_matrix": numeric_matrix_identity(x, columns),
        "training_label_vector": numeric_vector_identity(y, name="label_return"),
        "training_sample_weight_vector": numeric_vector_identity(weights, name="sample_weight"),
        "test_feature_matrix": numeric_matrix_identity(xt, columns),
    }


def checkpoint_identity(*, stage: str, fold: Fold, arm: str, horizon: int, attempt: str,
                        contract_sha256: str, code_shas: Mapping[str, str],
                        binding: Mapping[str, Any], feature_names: Sequence[str],
                        data_identity: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in STAGE_SPECS or arm not in STAGE_SPECS[stage]["arms"] or horizon not in HORIZONS:
        raise ContractError("checkpoint cell is outside the frozen stage plan")
    if tuple(str(value) for value in feature_names) != tuple(data_identity.get("feature_columns", ())):
        raise ContractError("checkpoint feature order differs from the training/test data identity")
    config = {"model_params": MODEL_PARAMS, "feature_names": tuple(feature_names),
              "label": "close(T+h)/close(T)-1", "horizon": int(horizon),
              "fold": [fold.fold_id, fold.start.isoformat(), fold.end.isoformat()],
              "data_identity": dict(data_identity)}
    return {
        "stage": stage,
        "fold": fold.fold_id,
        "arm": arm,
        "horizon": int(horizon),
        "attempt": attempt,
        "contract_sha256": contract_sha256,
        "input_components": arm_source_identity(binding, arm),
        "code_sha256": dict(sorted(code_shas.items())),
        "data_identity": dict(data_identity),
        "config_sha256": canonical_sha256(config),
    }


def load_reusable_checkpoint(artifact_path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reuse only a complete artifact with exact component identities and SHA."""
    artifact_path = Path(artifact_path)
    metadata_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    if not artifact_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = load_json(metadata_path)
        stored_identity = metadata.get("identity")
        expected_identity = dict(identity)
        if not isinstance(stored_identity, dict):
            return None
        stored_identity = dict(stored_identity)
        # An attempt number changes only after diagnosis; the completed artifact
        # remains reusable when every behavioral/source identity is unchanged.
        stored_identity.pop("attempt", None)
        expected_identity.pop("attempt", None)
        if metadata.get("status") != "COMPLETED" or stored_identity != expected_identity:
            return None
        if metadata.get("artifact_sha256") != sha256_file(artifact_path):
            return None
        import joblib as _joblib
        payload = _joblib.load(artifact_path)
        if not isinstance(payload, dict) or "model" not in payload or "predictions" not in payload:
            return None
        return payload
    except (OSError, ValueError, EOFError, KeyError):
        return None


def save_checkpoint(artifact_path: Path, payload: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically persist one completed fit result and its input identity."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{artifact_path.name}.", dir=artifact_path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(dict(payload), tmp_path)
        with tmp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, artifact_path)
        dir_fd = os.open(artifact_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    metadata = {"status": "COMPLETED", "identity": dict(identity), "artifact_sha256": sha256_file(artifact_path)}
    atomic_write_json(artifact_path.with_suffix(artifact_path.suffix + ".json"), metadata)
    return metadata


def code_identity() -> dict[str, str]:
    """Hash the route code that determines formula, feature, and fit behavior."""
    paths = {
        "model_selection": Path(__file__).resolve(),
        "formula_adapter": FROZEN_ROUTE_DEPENDENCY_ROOT / "formula_adapter.py",
        "tdx_ast_engine_injected": FROZEN_ROUTE_DEPENDENCY_ROOT / "tdx_ast_engine_injected.py",
        "fast_builder_c": PROJECT_ROOT / "workspace/research/a_v193_multiyear_validation_v1/s0_recovery_r7/private/engine_runtime/fast_builder_c.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def route_code_sha256() -> str:
    return canonical_sha256(code_identity())


def _registry_date_range(values: Sequence[Any] | pd.Series | pd.Index, *, scope: str) -> dict[str, Any] | None:
    dates = pd.to_datetime(pd.Series(values), errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        return None
    return {"count": int(len(dates)), "min": dates.iloc[0].strftime("%Y-%m-%d"),
            "max": dates.iloc[-1].strftime("%Y-%m-%d"), "scope": scope}


def _registry_input_paths(binding: Mapping[str, Any], *, indexed: bool) -> list[str]:
    sources = binding.get("sources", {})
    paths: list[str] = []
    panel = sources.get("price_panel", {})
    if panel.get("path"):
        paths.append(str(panel["path"]))
    calendar = sources.get("trading_date_index", {})
    paths.extend(str(path) for path in calendar.get("container_parts", ()))
    for value in binding.get("formula_inputs", {}).values():
        if isinstance(value, Mapping) and value.get("path"):
            paths.append(str(value["path"]))
    if indexed:
        index = sources.get("index_series", {})
        if index.get("path"):
            paths.append(str(index["path"]))
        industry = sources.get("industry_classification", {})
        paths.extend(str(path) for path in industry.get("container_parts", ()))
    return list(dict.fromkeys(paths))


def _registry_input_component_shas(binding: Mapping[str, Any], *, arms: Sequence[str]) -> dict[str, str]:
    sources = binding.get("sources", {})
    formulas = binding.get("formula_inputs", {})
    output: dict[str, str] = {}
    for key in ("price_panel", "trading_date_index"):
        value = sources.get(key, {})
        sha = value.get("sha256") or value.get("member_sha256")
        if sha:
            output[key] = str(sha)
    for key in ("v192", "v193"):
        value = formulas.get(key, {})
        if isinstance(value, Mapping) and value.get("sha256"):
            output[f"formula_{key}"] = str(value["sha256"])
    if any(arm in INDEXED_ARMS for arm in arms):
        value = sources.get("index_series", {})
        if value.get("sha256"):
            output["index_series"] = str(value["sha256"])
    if any(arm in {"M3", "M4"} for arm in arms):
        value = sources.get("industry_classification", {})
        if value.get("member_sha256"):
            output["industry_classification"] = str(value["member_sha256"])
    return output


def build_registry_event_context(*, stage: str, binding: Mapping[str, Any],
                                 contract_sha256: str, binding_sha256: str,
                                 code_shas: Mapping[str, str], config_sha256: str,
                                 arms: Sequence[str], truth_access: str,
                                 signal_dates: Sequence[Any] | pd.Series | pd.Index = (),
                                 target_dates: Sequence[Any] | pd.Series | pd.Index = (),
                                 checkpoint_path: Path | None = None,
                                 artifact_paths: Sequence[Path] = (),
                                 deterministic_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build truthful, stage-specific registry evidence without changing guard accounting."""
    code_map = dict(sorted((str(key), str(value)) for key, value in code_shas.items()))
    signal_range = _registry_date_range(signal_dates, scope="model_signal_dates")
    target_range = _registry_date_range(target_dates, scope="observed_label_end_dates")
    checkpoint_ref: dict[str, str] | None = None
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        checkpoint_ref = {"path": str(Path(checkpoint_path).resolve()),
                          "sha256": sha256_file(Path(checkpoint_path))}
    metric_values = {
        "truth_positive_denominator": None,
        "selection_slots": None,
    }
    metric_values.update(dict(deterministic_metrics or {}))
    null_reasons: dict[str, str] = {}
    if metric_values.get("truth_positive_denominator") is None:
        null_reasons["deterministic_metrics.truth_positive_denominator"] = (
            "NO_FROZEN_BINARY_TRUTH_LABEL_FOR_REGRESSION_RETURNS")
    if metric_values.get("selection_slots") is None:
        null_reasons["deterministic_metrics.selection_slots"] = "NO_SELECTION_OCCURRED_IN_THIS_LIFECYCLE_EVENT"
    if signal_range is None:
        null_reasons["signal_date_range"] = "NO_SIGNAL_ROWS_AVAILABLE_AT_EVENT_START"
    if target_range is None:
        null_reasons["target_date_range"] = "NO_LABEL_END_ROWS_AVAILABLE_AT_EVENT_START"
    if checkpoint_ref is None:
        null_reasons["checkpoint_ref"] = "NO_MODEL_CHECKPOINT_IN_THIS_LIFECYCLE_EVENT"
    indexed = any(arm in INDEXED_ARMS for arm in arms)
    paths = [str(Path(path).resolve()) for path in artifact_paths]
    return {
        "truth_access": truth_access,
        "sealed_status": "EXPOSED_HISTORY",
        "signal_date_range": signal_range,
        "target_date_range": target_range,
        "runtime": {"python": sys.executable, "python_version": sys.version.split()[0],
                    "numpy": np.__version__, "pandas": pd.__version__},
        "device": "CPU_RESEARCH_ONLY",
        "checkpoint_ref": checkpoint_ref,
        "code_sha256": canonical_sha256(code_map),
        "config_sha256": str(config_sha256),
        "input_sha256": binding_sha256,
        "input_paths": _registry_input_paths(binding, indexed=indexed),
        "artifact_paths": paths,
        "deterministic_metrics": metric_values,
        "input_identity": {
            "contract_sha256": contract_sha256,
            "input_binding_sha256": binding_sha256,
            "input_component_sha256": _registry_input_component_shas(binding, arms=arms),
            "code_sha256_by_file": code_map,
            "null_reasons": null_reasons,
        },
    }


def make_official_fit_guard(*, contract_path: Path, binding_path: Path, stage: str,
                            artifact_paths: Sequence[Path] = (),
                            registry_path: Path | None = None):
    """Create the sole allowed real-fit guard over the existing official registry."""
    registry_dir = PROJECT_ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"
    sys.path.insert(0, str(registry_dir))
    from execution_guard_core import ExecutionGuard
    from registry_adapter import OfficialRegistryAdapter

    class PVRegistryAdapter(OfficialRegistryAdapter):
        """Route-local event enrichment over the existing locked append path."""

        def __init__(self, **kwargs: Any) -> None:
            self._execution_context: dict[str, dict[str, Any]] = {}
            super().__init__(**kwargs)

        def set_execution_context(self, execution_id: str, context: Mapping[str, Any]) -> None:
            if not execution_id:
                raise ExecutionBlocked("registry context requires an execution id")
            self._execution_context[str(execution_id)] = dict(context)

        def update_execution_context(self, execution_id: str, **updates: Any) -> None:
            if str(execution_id) not in self._execution_context:
                raise ExecutionBlocked(f"registry context was not initialized: {execution_id}")
            self._execution_context[str(execution_id)].update(updates)

        def _event_spec(self, event: Mapping[str, Any], *, event_type: str) -> dict[str, Any]:
            spec = super()._event_spec(event, event_type=event_type)
            execution_id = str(event.get("execution_id", ""))
            context = self._execution_context.get(execution_id)
            if context is None:
                raise ExecutionBlocked(f"refusing un-enriched registry event: {execution_id}")

            for field in ("truth_access", "sealed_status", "signal_date_range", "target_date_range",
                          "runtime", "device", "checkpoint_ref", "code_sha256", "config_sha256",
                          "input_sha256"):
                if field in context:
                    spec[field] = context[field]

            input_paths = context.get("input_paths")
            if input_paths is not None:
                spec["input_paths"] = [str(path) for path in input_paths]

            artifact_paths = list(spec.get("artifact_paths") or [])
            artifact_paths.extend(str(path) for path in context.get("artifact_paths", ()))
            artifact_paths = list(dict.fromkeys(artifact_paths))
            spec["artifact_paths"] = artifact_paths
            spec["artifact_path"] = artifact_paths

            input_identity = dict(spec.get("input_identity") or {})
            input_identity.update(dict(context.get("input_identity") or {}))
            artifact_sha256: dict[str, str] = {}
            for raw_path in artifact_paths:
                path = Path(raw_path)
                if path.is_file():
                    artifact_sha256[str(path)] = sha256_file(path)
            input_identity["artifact_sha256"] = artifact_sha256
            null_reasons = dict(input_identity.get("null_reasons") or {})
            metrics = dict(spec.get("deterministic_metrics") or {})
            metrics.update(dict(context.get("deterministic_metrics") or {}))
            if metrics.get("truth_positive_denominator") is None:
                null_reasons.setdefault(
                    "deterministic_metrics.truth_positive_denominator",
                    "NO_FROZEN_BINARY_TRUTH_LABEL_FOR_REGRESSION_RETURNS",
                )
            if metrics.get("selection_slots") is None:
                null_reasons.setdefault(
                    "deterministic_metrics.selection_slots",
                    "NO_SELECTION_OCCURRED_IN_THIS_LIFECYCLE_EVENT",
                )
            if spec.get("checkpoint_ref") is None:
                null_reasons.setdefault("checkpoint_ref", "NO_MODEL_CHECKPOINT_IN_THIS_LIFECYCLE_EVENT")
            input_identity["null_reasons"] = null_reasons
            spec["input_identity"] = input_identity
            spec["deterministic_metrics"] = metrics
            return spec

    profile = {
        "track_id": "A_SHARE_PANEL_MODEL_SELECTION_RESEARCH",
        "route_id": "a-share-model-selection-pv-environment-r1-20261019",
        "registration_mode": "LIVE_EXECUTION",
        "start_stage": f"{stage}_RUN_STARTED",
        "terminal_stage": f"{stage}_TERMINAL",
        "occurred_at_reason": "LIVE_MODEL_SELECTION_FIT_UTC",
        "baseline_id": "M0_SHARED_V192_V193_CORE_CLOSE_RETURN_TOP10",
        "one_variable": "NESTED_FEATURE_ARM_BY_FROZEN_STAGE_PLAN",
        "signal_scope": "A_SHARE_CLOSE_RETURN_TOP10_RESEARCH_ONLY",
        "target_scope": "CLOSE_T_PLUS_H_CLOSE_RETURN_LABELS",
        "device": "CPU_RESEARCH_ONLY",
        "evidence_level": "ENGINEERING_CANARY_ONLY" if "CANARY" in stage else "DIRECTION_ONLY",
    }
    adapter = PVRegistryAdapter(
        registry_path=Path(registry_path) if registry_path is not None else PROJECT_ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl",
        contract_path=contract_path,
        input_binding_path=binding_path,
        artifact_paths=(Path(__file__).resolve(), contract_path, binding_path, *artifact_paths),
        event_profile=profile,
    )
    return ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=96)


def set_registry_execution_context(guard: Any, execution_id: str, context: Mapping[str, Any]) -> None:
    """Bind truthful stage evidence before the official guard claims a lifecycle."""
    backend = getattr(guard, "backend", None)
    setter = getattr(backend, "set_execution_context", None)
    if not callable(setter):
        raise ExecutionBlocked("official guard is missing the route-local event enrichment adapter")
    setter(execution_id, context)


def update_registry_execution_context(guard: Any, execution_id: str, **updates: Any) -> None:
    backend = getattr(guard, "backend", None)
    updater = getattr(backend, "update_execution_context", None)
    if not callable(updater):
        raise ExecutionBlocked("official guard is missing the route-local event enrichment adapter")
    updater(execution_id, **updates)


def _finite_arm_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    features = tuple(column for column in frame.columns if column not in KEYS)
    if not features:
        raise ContractError("arm feature frame is empty")
    out = frame.copy()
    for column in features:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(out[list(features)].to_numpy(dtype=np.float64)).all(axis=1)
    return out.loc[finite].sort_values(list(KEYS), kind="mergesort").reset_index(drop=True), features


def prepare_cell_data(*, features: pd.DataFrame, labels: pd.DataFrame, fold: Fold,
                      canary: bool) -> dict[str, Any]:
    """Materialize the exact finite train/test matrices consumed by one cell."""
    clean_features, feature_columns = _finite_arm_frame(features)
    train, test = split_fold(clean_features, labels, fold)
    if canary:
        train, test = canary_subset(train, test)
    x = train[list(feature_columns)].to_numpy(dtype=np.float64)
    y = pd.to_numeric(train["label_return"], errors="raise").to_numpy(dtype=np.float64)
    weights = equal_day_weights(train["date"])
    xt = test[list(feature_columns)].to_numpy(dtype=np.float64)
    if (not len(x) or not len(xt) or not np.isfinite(x).all() or
            not np.isfinite(y).all() or not np.isfinite(weights).all() or not np.isfinite(xt).all()):
        raise ContractError("non-finite or empty training/test matrix")
    identity = cell_data_identity(
        train=train, test=test, feature_columns=feature_columns, x=x, y=y, weights=weights, xt=xt,
    )
    return {"train": train, "test": test, "feature_columns": feature_columns,
            "x": x, "y": y, "weights": weights, "xt": xt, "identity": identity,
            "training_label_preparation_slots": int(len(labels))}


def prediction_frame_identity(predictions: pd.DataFrame) -> dict[str, Any]:
    """Hash date/code/prediction rows in their exact stored order."""
    if not {"date", "code", "prediction"}.issubset(predictions.columns):
        raise ContractError("candidate predictions are missing date/code/prediction")
    keys = key_frame_identity(predictions)
    values = pd.to_numeric(predictions["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    if len(values) != keys["row_count"] or not np.isfinite(values).all():
        raise ContractError("candidate predictions contain missing or non-finite scores")
    ordered_rows = [
        [row[0], row[1], float(value).hex()]
        for row, value in zip(
            [[date, code] for date, code in zip(
                pd.to_datetime(predictions["date"]).dt.normalize().dt.strftime("%Y-%m-%d"),
                predictions["code"].astype(str),
            )],
            values,
        )
    ]
    payload = {"schema_version": "A_SHARE_ORDERED_PREDICTIONS_HEX_FLOAT_V1", "rows": ordered_rows}
    return {"schema_version": payload["schema_version"], "row_count": int(len(values)),
            "keyset_sha256": keys["sorted_keyset_sha256"],
            "ordered_rows_sha256": canonical_sha256(payload)}


def _same_key_order(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        return key_frame_identity(left)["ordered_keys_sha256"] == key_frame_identity(right)["ordered_keys_sha256"]
    except ContractError:
        return False


def verify_saved_checkpoint_replay(*, checkpoint_path: Path, replay_receipt_path: Path,
                                   identity: Mapping[str, Any], execution_id: str,
                                   provenance: str = "FIT_TIME_SAVED_MODEL_REPLAY") -> dict[str, Any]:
    """Reload a durable checkpoint and require exact key, score, and Top10 parity."""
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    checkpoint_sha_before = sha256_file(checkpoint_path)
    metadata_sha_before = sha256_file(metadata_path)
    metadata = load_json(metadata_path)
    if metadata.get("status") != "COMPLETED" or metadata.get("artifact_sha256") != checkpoint_sha_before:
        raise ContractError("saved checkpoint metadata is not complete or its SHA does not match")
    if metadata.get("identity") != dict(identity):
        raise ContractError("saved checkpoint metadata identity differs from the planned cell")
    payload = joblib.load(checkpoint_path)
    if not isinstance(payload, dict) or not {"model", "predictions", "feature_columns", "replay_fixture"}.issubset(payload):
        raise ContractError("saved checkpoint lacks the immutable replay fixture")
    fixture = payload["replay_fixture"]
    if not isinstance(fixture, Mapping) or not {"test_keys", "test_features"}.issubset(fixture):
        raise ContractError("saved checkpoint replay fixture is incomplete")
    test_keys = fixture["test_keys"]
    if not isinstance(test_keys, pd.DataFrame) or not set(KEYS).issubset(test_keys.columns):
        raise ContractError("saved replay keys are invalid")
    feature_columns = tuple(str(column) for column in payload["feature_columns"])
    if feature_columns != tuple(identity["data_identity"]["feature_columns"]):
        raise ContractError("saved replay feature order differs from the fit identity")
    matrix = np.asarray(fixture["test_features"], dtype=np.float64, order="C")
    matrix_identity = numeric_matrix_identity(matrix, feature_columns)
    expected_matrix = identity["data_identity"]["test_feature_matrix"]
    if matrix_identity != expected_matrix or len(matrix) != len(test_keys):
        raise ContractError("saved test feature matrix differs from the fit identity")
    if key_frame_identity(test_keys) != identity["data_identity"]["test_keys"]:
        raise ContractError("saved test keys differ from the fit identity")
    original = payload["predictions"]
    if not isinstance(original, pd.DataFrame) or not {"date", "code", "prediction"}.issubset(original.columns):
        raise ContractError("saved candidate predictions are invalid")
    if not _same_key_order(test_keys, original):
        raise ContractError("saved prediction key order differs from the persisted test matrix")
    if key_frame_identity(original) != identity["data_identity"]["test_keys"]:
        raise ContractError("saved prediction keys differ from the fitted test key identity")

    replay_scores = np.asarray(payload["model"].predict(matrix), dtype=np.float64)
    original_scores = pd.to_numeric(original["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    if len(replay_scores) != len(original_scores) or not np.isfinite(replay_scores).all():
        raise ContractError("reloaded checkpoint produced missing or non-finite predictions")
    replay = test_keys.loc[:, list(KEYS)].reset_index(drop=True).copy()
    replay["code"] = replay["code"].astype(str)
    replay["prediction"] = replay_scores
    expected = original.loc[:, [*KEYS, "prediction"]].reset_index(drop=True).copy()
    expected["date"] = pd.to_datetime(expected["date"], errors="coerce").dt.normalize()
    expected["code"] = expected["code"].astype(str)
    expected["prediction"] = original_scores
    replay["date"] = pd.to_datetime(replay["date"], errors="coerce").dt.normalize()
    joined = replay.merge(expected, on=list(KEYS), how="outer", validate="one_to_one",
                          suffixes=("_replay", "_saved"), indicator=True, sort=False)
    if len(joined) != len(replay) or not joined["_merge"].eq("both").all():
        raise ContractError("replay and saved candidate keys do not form a one-to-one join")
    joined_scores = joined[["prediction_replay", "prediction_saved"]].to_numpy(dtype=np.float64)
    if not np.array_equal(replay_scores, original_scores) or not np.array_equal(joined_scores[:, 0], joined_scores[:, 1]):
        raise ContractError("saved checkpoint replay prediction parity failed at tolerance 0")
    original_top = stable_top10(expected)
    replay_top = stable_top10(replay)
    top_columns = ["date", "code", "rank"]
    if not original_top[top_columns].equals(replay_top[top_columns]):
        raise ContractError("saved checkpoint replay Top10 key/rank parity failed")
    original_top_sha = key_frame_identity(original_top[top_columns])["ordered_keys_sha256"]
    replay_top_sha = key_frame_identity(replay_top[top_columns])["ordered_keys_sha256"]
    checkpoint_sha_after = sha256_file(checkpoint_path)
    metadata_sha_after = sha256_file(metadata_path)
    if checkpoint_sha_after != checkpoint_sha_before or metadata_sha_after != metadata_sha_before:
        raise ContractError("checkpoint or metadata changed during saved-model replay")
    receipt = {
        "schema_version": REPLAY_RECEIPT_SCHEMA,
        "status": "VERIFIED",
        "provenance": provenance,
        "execution_id": str(execution_id),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_metadata_path": str(metadata_path.resolve()),
        "checkpoint_metadata_sha256": metadata_sha_before,
        "config_sha256": str(identity["config_sha256"]),
        "data_identity": dict(identity["data_identity"]),
        "prediction_identity": prediction_frame_identity(original),
        "replay_prediction_identity": prediction_frame_identity(replay),
        "top10_parity": {"status": "EXACT", "count": int(len(original_top)),
                          "saved_ordered_key_sha256": original_top_sha,
                          "replay_ordered_key_sha256": replay_top_sha},
        "comparison": {"join_validation": "ONE_TO_ONE", "rtol": 0.0, "atol": 0.0,
                       "prediction_equal": True, "key_order_equal": True},
        "immutability": {"checkpoint_unchanged": True, "metadata_unchanged": True},
    }
    atomic_write_json(Path(replay_receipt_path), receipt)
    return receipt


def _archive_member_bytes(archive_path: Path, member_name: str) -> bytes:
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            matches = [name for name in archive.namelist()
                       if PurePosixPath(name).name == member_name]
            if len(matches) != 1:
                raise ContractError(f"source archive must contain one {member_name}")
            return archive.read(matches[0])
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ContractError(f"source archive member unavailable: {member_name}") from exc


def _create_exclusive_json(path: Path, payload: Mapping[str, Any]) -> tuple[int, int]:
    """Create one receipt without replacing a file created by another run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ExecutionBlocked(f"checkpoint replay receipt path already exists: {path}") from exc
    created = os.fstat(fd)
    owner = (int(created.st_dev), int(created.st_ino))
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return owner
    except BaseException:
        # If receipt initialization itself fails, remove only the inode this
        # call created; an existing or concurrently replaced path is untouched.
        try:
            current = path.stat()
            if (int(current.st_dev), int(current.st_ino)) == owner:
                path.unlink()
        except OSError:
            pass
        raise


def _update_owned_json(path: Path, payload: Mapping[str, Any], owner: tuple[int, int]) -> tuple[int, int]:
    """Atomically update only the receipt inode created by this execution."""
    path = Path(path)
    try:
        current = path.stat()
    except FileNotFoundError as exc:
        raise ExecutionBlocked("checkpoint replay receipt ownership was lost") from exc
    if (int(current.st_dev), int(current.st_ino)) != owner:
        raise ExecutionBlocked("checkpoint replay receipt was replaced after this execution claimed it")
    atomic_write_json(path, payload)
    updated = path.stat()
    return int(updated.st_dev), int(updated.st_ino)


@contextlib.contextmanager
def _pinned_route_from_archives(*, source_archive_path: Path,
                                expected_source_archive_sha256: str,
                                dependency_archive_path: Path,
                                expected_dependency_archive_sha256: str,
                                expected_code_shas: Mapping[str, str],
                                project_root: Path,
                                contract_path: Path,
                                binding_path: Path,
                                activation_path: Path) -> Iterator[Any]:
    """Load only SHA-bound route/runtime source from reviewed archives."""
    source_archive_path = Path(source_archive_path)
    dependency_archive_path = Path(dependency_archive_path)
    if sha256_file(source_archive_path) != expected_source_archive_sha256:
        raise ExecutionBlocked("reviewed route source archive SHA mismatch")
    if sha256_file(dependency_archive_path) != expected_dependency_archive_sha256:
        raise ExecutionBlocked("reviewed dependency source archive SHA mismatch")
    temp = tempfile.TemporaryDirectory(prefix="pv-checkpoint-replay-source-")
    isolated_project_root = Path(temp.name) / "project"
    source_root = isolated_project_root / "workspace/research/a_share_model_selection_v1"
    runtime_root = (isolated_project_root / "workspace/research/a_v193_multiyear_validation_v1"
                    / "s0_recovery_r7/private/engine_runtime")
    source_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    try:
        members = {
            "model_selection": (source_archive_path, "model_selection.py"),
            "formula_adapter": (dependency_archive_path, "formula_adapter.py"),
            "tdx_ast_engine_injected": (dependency_archive_path, "tdx_ast_engine_injected.py"),
        }
        for component, (archive_path, filename) in members.items():
            raw = _archive_member_bytes(archive_path, filename)
            actual_sha = hashlib.sha256(raw).hexdigest()
            if actual_sha != str(expected_code_shas.get(component, "")):
                raise ExecutionBlocked(f"reviewed source component SHA mismatch: {component}")
            (source_root / filename).write_bytes(raw)
        runtime_source = (Path(project_root).resolve() / "workspace/research/a_v193_multiyear_validation_v1"
                          / "s0_recovery_r7/private/engine_runtime/fast_builder_c.py")
        try:
            runtime_bytes = runtime_source.read_bytes()
        except OSError as exc:
            raise ExecutionBlocked("SHA-bound fast_builder_c runtime is unavailable") from exc
        runtime_sha = hashlib.sha256(runtime_bytes).hexdigest()
        if runtime_sha != str(expected_code_shas.get("fast_builder_c", "")):
            raise ExecutionBlocked("reviewed source component SHA mismatch: fast_builder_c")
        (runtime_root / "fast_builder_c.py").write_bytes(runtime_bytes)
        saved_path = list(sys.path)
        dependency_names = ("formula_adapter", "tdx_ast_engine_injected")
        saved_modules = {name: sys.modules.pop(name, None) for name in dependency_names}
        sys.path.insert(0, str(source_root))
        try:
            source_path = source_root / "model_selection.py"
            spec = importlib.util.spec_from_file_location("pv_pinned_model_selection", source_path)
            if spec is None or spec.loader is None:
                raise ExecutionBlocked("reviewed model-selection source could not be loaded")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.PROJECT_ROOT = isolated_project_root
            module.FROZEN_ROUTE_DEPENDENCY_ROOT = source_root
            module.BATCH_ROOT = Path(contract_path).resolve().parent
            module.CONTRACT_PATH = Path(contract_path).resolve()
            module.INPUT_BINDING_PATH = Path(binding_path).resolve()
            module.ACTIVATION_PATH = Path(activation_path).resolve()
            if module.code_identity() != dict(expected_code_shas):
                raise ExecutionBlocked("loaded reviewed route identity differs from the original manifest")
            yield module
        finally:
            sys.path[:] = saved_path
            sys.modules.pop("pv_pinned_model_selection", None)
            for name in dependency_names:
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
    finally:
        temp.cleanup()


def _read_registered_events(registry_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(registry_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExecutionBlocked(f"official registry JSONL is invalid at line {line_number}") from exc
            if not isinstance(event, dict):
                raise ExecutionBlocked(f"official registry event is not an object at line {line_number}")
            events.append(event)
    return events


def _verify_original_fit_event_pair(events: Sequence[Mapping[str, Any]], *, execution_id: str,
                                    stage: str, experiment_id: str,
                                    expected_route_code_sha256: str,
                                    contract_sha256: str, binding_sha256: str) -> dict[str, Any]:
    matching = [dict(event) for event in events
                if event.get("experiment_id") == experiment_id and event.get("execution_id") == execution_id]
    starts = [event for event in matching if event.get("event_type") == "RUN_STARTED"]
    terminals = [event for event in matching if event.get("event_type") == "TERMINAL"]
    if len(matching) != 2 or len(starts) != 1 or len(terminals) != 1:
        raise ExecutionBlocked(f"original fit lifecycle is not an exact start/terminal pair: {execution_id}")
    start, terminal = starts[0], terminals[0]
    for event, expected_stage in ((start, f"{stage}_RUN_STARTED"), (terminal, f"{stage}_TERMINAL")):
        if (event.get("stage") != expected_stage or event.get("route_id") != "a-share-model-selection-pv-environment-r1-20261019" or
                event.get("registration_mode") != "LIVE_EXECUTION" or
                event.get("code_sha256") != expected_route_code_sha256 or
                event.get("contract_sha256") != contract_sha256 or
                event.get("input_binding_sha256") != binding_sha256):
            raise ExecutionBlocked(f"original fit event identity mismatch: {execution_id}")
        event_sha = event.get("event_sha256")
        if not isinstance(event_sha, str) or len(event_sha) != 64:
            raise ExecutionBlocked(f"original fit event SHA is missing: {execution_id}")
    terminal_metrics = dict(terminal.get("deterministic_metrics") or {})
    fit_counts = (
        terminal_metrics.get("planned_real_fit_count"),
        terminal_metrics.get("real_fit_budget_debit"),
        terminal_metrics.get("real_fit_returned_count"),
    )
    if (terminal.get("decision") != "COMPLETED" or
            terminal_metrics.get("execution_kind") != "REAL_FIT" or
            terminal_metrics.get("fit_call_entered") is not True or
            any(not isinstance(count, int) or isinstance(count, bool) or count != 1
                for count in fit_counts)):
        raise ExecutionBlocked(f"original fit terminal does not confirm one returned fit: {execution_id}")
    metrics = dict(start.get("deterministic_metrics") or {})
    return {
        "execution_id": execution_id,
        "run_started_event_sha256": str(start["event_sha256"]),
        "terminal_event_sha256": str(terminal["event_sha256"]),
        "training_rows_from_start_event": metrics.get("training_rows"),
        "training_label_preparation_slots": metrics.get("training_label_preparation_slots"),
        "start_artifact_sha256": dict((start.get("input_identity") or {}).get("artifact_sha256") or {}),
        "terminal_artifact_sha256": dict((terminal.get("input_identity") or {}).get("artifact_sha256") or {}),
    }


def _prepare_pinned_cell_data(route_module: Any, *, features: pd.DataFrame,
                              labels: pd.DataFrame, fold: Any, canary: bool) -> dict[str, Any]:
    clean, columns = route_module._finite_arm_frame(features)
    train, test = route_module.split_fold(clean, labels, fold)
    if canary:
        train, test = route_module.canary_subset(train, test)
    x = train[list(columns)].to_numpy(dtype=np.float64)
    y = pd.to_numeric(train["label_return"], errors="raise").to_numpy(dtype=np.float64)
    weights = route_module.equal_day_weights(train["date"])
    xt = test[list(columns)].to_numpy(dtype=np.float64)
    if (not len(x) or not len(xt) or not np.isfinite(x).all() or
            not np.isfinite(y).all() or not np.isfinite(weights).all() or not np.isfinite(xt).all()):
        raise ContractError("reconstructed train or test matrix is empty or non-finite")
    return {
        "train": train, "test": test, "feature_columns": tuple(columns),
        "x": x, "y": y, "weights": weights, "xt": xt,
        "identity": cell_data_identity(train=train, test=test, feature_columns=columns,
                                        x=x, y=y, weights=weights, xt=xt),
        "training_label_preparation_slots": int(len(labels)),
    }


def _preflight_checkpoint_replay_audit(*, stage: str, attempt: str,
                                       manifest_path: Path, expected_manifest_sha256: str,
                                       source_archive_path: Path, expected_source_archive_sha256: str,
                                       dependency_archive_path: Path, expected_dependency_archive_sha256: str,
                                       contract_path: Path, binding_path: Path, activation_path: Path,
                                       registry_path: Path | None, execution_id: str,
                                       guard: Any) -> dict[str, Any]:
    """Validate pinned artifacts after the no-fit intent has been claimed."""
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ExecutionBlocked("original stage manifest SHA mismatch")
    original_manifest = load_json(manifest_path)
    if (original_manifest.get("stage") != stage or original_manifest.get("attempt") != attempt or
            int(original_manifest.get("fit_calls", -1)) != 3):
        raise ExecutionBlocked("original stage manifest does not identify the requested three-cell canary")
    contract_sha256, binding_sha256 = sha256_file(contract_path), sha256_file(binding_path)
    if (original_manifest.get("contract_sha256") != contract_sha256 or
            original_manifest.get("input_binding_sha256") != binding_sha256):
        raise ExecutionBlocked("original manifest contract or input-binding SHA mismatch")
    expected_code_shas = original_manifest.get("code_sha256")
    if not isinstance(expected_code_shas, dict):
        raise ExecutionBlocked("original manifest lacks a per-file code identity")
    expected_route_code_sha256 = canonical_sha256(expected_code_shas)
    source_archive_sha_before = sha256_file(source_archive_path)
    dependency_archive_sha_before = sha256_file(dependency_archive_path)
    if (source_archive_sha_before != expected_source_archive_sha256 or
            dependency_archive_sha_before != expected_dependency_archive_sha256):
        raise ExecutionBlocked("reviewed source archive SHA mismatch")
    output_manifest_path = Path(str(original_manifest.get("output_manifest_path", ""))).resolve()
    if not output_manifest_path.is_file():
        raise ExecutionBlocked("original stage output manifest is missing")
    output_manifest_sha_before = sha256_file(output_manifest_path)
    output_manifest = load_json(output_manifest_path)
    if (output_manifest.get("stage") != stage or output_manifest.get("attempt") != attempt or
            output_manifest.get("contract_sha256") != contract_sha256 or
            output_manifest.get("input_binding_sha256") != binding_sha256):
        raise ExecutionBlocked("original output manifest identity mismatch")
    manifest_artifacts = original_manifest.get("output_artifacts")
    output_artifacts = output_manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict) or output_artifacts != manifest_artifacts:
        raise ExecutionBlocked("stage and output manifests disagree on artifact identities")
    selected_info = output_artifacts.get("selected_keys")
    if not isinstance(selected_info, Mapping) or not selected_info.get("path") or not selected_info.get("sha256"):
        raise ExecutionBlocked("original output manifest lacks the selected-key artifact identity")
    selected_keys_path = Path(str(selected_info["path"])).resolve()
    if sha256_file(selected_keys_path) != selected_info["sha256"]:
        raise ExecutionBlocked("selected-key artifact SHA differs from the output manifest")

    planned_cells = {item.get("cell"): item for item in original_manifest.get("cell_results", [])
                     if isinstance(item, Mapping)}
    expected_plan = stage_plan(stage, attempt=attempt)
    if len(planned_cells) != len(expected_plan) or len(expected_plan) != 3:
        raise ExecutionBlocked("original manifest does not contain the exact canary cell plan")
    cell_records: list[dict[str, Any]] = []
    immutable_paths = [manifest_path, output_manifest_path, selected_keys_path,
                       source_archive_path, dependency_archive_path,
                       contract_path, binding_path, activation_path]
    for cell in expected_plan:
        cell_key = f"{stage}_{cell['arm']}_H{cell['horizon']}_{cell['fold'].fold_id}_{attempt}"
        record = planned_cells.get(cell_key)
        if not isinstance(record, Mapping) or record.get("status") != "COMPLETED" or record.get("fit_called") is not True:
            raise ExecutionBlocked(f"original manifest does not confirm the completed fit cell {cell_key}")
        checkpoint_path = Path(str(record.get("checkpoint_path", ""))).resolve()
        if checkpoint_path.name != f"{cell_key}.joblib" or not checkpoint_path.is_file():
            raise ExecutionBlocked(f"original checkpoint path is invalid for {cell_key}")
        metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        if not metadata_path.is_file():
            raise ExecutionBlocked(f"original checkpoint metadata is missing for {cell_key}")
        immutable_paths.extend((checkpoint_path, metadata_path))
        cell_records.append({"cell": cell_key, "arm": cell["arm"], "horizon": int(cell["horizon"]),
                             "fold": cell["fold"], "canary": bool(cell["canary"]),
                             "checkpoint_path": checkpoint_path, "metadata_path": metadata_path})
    immutable_sha_before = {str(path): sha256_file(path) for path in immutable_paths}

    accounting = guard.accounting()
    prior_dangling = set(accounting.dangling_execution_ids) - {str(execution_id)}
    if prior_dangling:
        raise ExecutionBlocked(f"official registry has dangling executions before checkpoint replay: {sorted(prior_dangling)}")
    registry_before = _read_registered_events(
        Path(registry_path) if registry_path is not None else PROJECT_ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
    )
    expected_fit_events: dict[str, dict[str, Any]] = {}
    for cell_record in cell_records:
        fit_execution_id = str(cell_record["cell"])
        fit_events = _verify_original_fit_event_pair(
            registry_before, execution_id=fit_execution_id, stage=stage,
            experiment_id=EXPERIMENT_ID, expected_route_code_sha256=expected_route_code_sha256,
            contract_sha256=contract_sha256, binding_sha256=binding_sha256,
        )
        checkpoint_path = Path(cell_record["checkpoint_path"])
        metadata_path = Path(cell_record["metadata_path"])
        terminal_artifact_sha = fit_events["terminal_artifact_sha256"]
        if (terminal_artifact_sha.get(str(checkpoint_path)) != sha256_file(checkpoint_path) or
                terminal_artifact_sha.get(str(metadata_path)) != sha256_file(metadata_path)):
            raise ExecutionBlocked(f"original fit terminal artifact SHA mismatch: {fit_execution_id}")
        expected_fit_events[fit_execution_id] = fit_events
    return {
        "contract_sha256": contract_sha256, "binding_sha256": binding_sha256,
        "expected_code_shas": expected_code_shas,
        "expected_route_code_sha256": expected_route_code_sha256,
        "source_archive_sha_before": source_archive_sha_before,
        "dependency_archive_sha_before": dependency_archive_sha_before,
        "output_manifest_path": output_manifest_path,
        "output_manifest_sha_before": output_manifest_sha_before,
        "output_manifest": output_manifest, "manifest_artifacts": manifest_artifacts,
        "selected_info": selected_info, "selected_keys_path": selected_keys_path,
        "cell_records": cell_records, "immutable_paths": immutable_paths,
        "immutable_sha_before": immutable_sha_before,
        "expected_fit_events": expected_fit_events,
    }


def run_checkpoint_replay_audit(*, stage: str, attempt: str,
                                manifest_path: Path, expected_manifest_sha256: str,
                                source_archive_path: Path, expected_source_archive_sha256: str,
                                dependency_archive_path: Path, expected_dependency_archive_sha256: str,
                                contract_path: Path, binding_path: Path, activation_path: Path,
                                receipt_path: Path, execution_id: str,
                                registry_path: Path | None = None) -> dict[str, Any]:
    """No-fit reconstruct a canary's cells and replay its immutable checkpoints.

    The official no-fit intent is claimed before artifact preflight. Bound input
    reads occur only after that ``RUN_STARTED`` is durably claimed.
    The receipt records reconstructed evidence and never rewrites the old run.
    """
    if stage not in {"M0_CANARY", "M4_CANARY"}:
        raise ContractError("checkpoint replay audit is limited to the frozen M0/M4 canary stages")
    manifest_path = Path(manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    contract_path, binding_path, activation_path = (
        Path(contract_path).resolve(), Path(binding_path).resolve(), Path(activation_path).resolve()
    )
    source_archive_path = Path(source_archive_path).resolve()
    dependency_archive_path = Path(dependency_archive_path).resolve()
    nofit_stage = f"{stage}_CHECKPOINT_REPLAY_VERIFICATION"
    requested_contract_sha = sha256_file(contract_path) if contract_path.is_file() else None
    requested_binding_sha = sha256_file(binding_path) if binding_path.is_file() else None
    claim_contract_path = contract_path if requested_contract_sha else Path(CONTRACT_PATH).resolve()
    claim_binding_path = binding_path if requested_binding_sha else Path(INPUT_BINDING_PATH).resolve()
    claim_contract_sha = requested_contract_sha or sha256_file(claim_contract_path)
    claim_binding_sha = requested_binding_sha or sha256_file(claim_binding_path)
    audit_config_sha256 = canonical_sha256({
        "schema_version": REPLAY_RECEIPT_SCHEMA, "stage": stage, "attempt": attempt,
        "execution_id": str(execution_id),
        "manifest_sha256": expected_manifest_sha256,
        "source_archive_sha256": expected_source_archive_sha256,
        "dependency_archive_sha256": expected_dependency_archive_sha256,
    })
    receipt = {
        "schema_version": REPLAY_RECEIPT_SCHEMA,
        "status": "NO_FIT_INTENT_PREPARED",
        "provenance": "PROVENANCE_RECONSTRUCTED",
        "original_fit_time_training_key_hash": "CANNOT_CONFIRM",
        "execution_id": execution_id,
        "stage": stage,
        "attempt": attempt,
        "original_manifest_path": str(manifest_path),
        "expected_original_manifest_sha256": expected_manifest_sha256,
        "expected_source_archive_sha256": expected_source_archive_sha256,
        "expected_dependency_archive_sha256": expected_dependency_archive_sha256,
        "requested_control_file_identity": {
            "contract_path": str(contract_path), "contract_sha256": requested_contract_sha,
            "input_binding_path": str(binding_path), "input_binding_sha256": requested_binding_sha,
        },
        "null_reasons": {
            **({"requested_control_file_identity.contract_sha256": "REQUESTED_CONTRACT_FILE_MISSING"}
               if requested_contract_sha is None else {}),
            **({"requested_control_file_identity.input_binding_sha256": "REQUESTED_INPUT_BINDING_FILE_MISSING"}
               if requested_binding_sha is None else {}),
            "original_manifest_sha256": "EXPECTED_SHA_NOT_YET_VERIFIED",
            "source_archive_sha256": "EXPECTED_SHA_NOT_YET_VERIFIED",
            "dependency_archive_sha256": "EXPECTED_SHA_NOT_YET_VERIFIED",
        },
        "source_archive_path": str(source_archive_path),
        "dependency_archive_path": str(dependency_archive_path),
        "preflight_status": "NOT_STARTED",
        "cells": [],
        "security_flags": {"research_only": True, "production_allowed": False,
                           "buy_gate_updated": False, "orders_sent": False},
    }
    receipt_owner = _create_exclusive_json(receipt_path, receipt)
    verification_guard = make_official_fit_guard(
        contract_path=claim_contract_path, binding_path=claim_binding_path, stage=nofit_stage,
        artifact_paths=(receipt_path,),
        registry_path=registry_path,
    )
    registry_context = build_registry_event_context(
        stage=nofit_stage, binding={},
        contract_sha256=claim_contract_sha, binding_sha256=claim_binding_sha,
        code_shas=code_identity(), config_sha256=audit_config_sha256,
        arms=STAGE_SPECS[stage]["input_arms"], truth_access="NO_FIT_AUDIT_INTENT_UNVERIFIED",
        artifact_paths=(receipt_path,),
        deterministic_metrics={"fit_calls": 0, "cells_planned": None, "cells_verified": 0,
                               "training_label_preparation_slots_by_horizon": None,
                               "truth_positive_denominator": None, "selection_slots": None},
    )
    registry_context["input_paths"] = [str(path) for path in (
        manifest_path, source_archive_path, dependency_archive_path,
        contract_path, binding_path, activation_path,
    )]
    registry_context["input_identity"]["requested_control_file_identity"] = {
        "contract_path": str(contract_path), "contract_sha256": requested_contract_sha,
        "input_binding_path": str(binding_path), "input_binding_sha256": requested_binding_sha,
    }
    registry_context["input_identity"]["null_reasons"].update(receipt["null_reasons"])
    registry_context["input_identity"]["preflight_status"] = "NOT_STARTED"
    set_registry_execution_context(verification_guard, execution_id, registry_context)
    claim = verification_guard.claim_no_fit(
        execution_id=execution_id, arm=f"{stage}:CHECKPOINT_REPLAY_RECONSTRUCTION",
        contract_sha256=claim_contract_sha, input_binding_sha256=claim_binding_sha,
    )
    input_read_started = False
    terminalized = False
    terminal_attempted = False
    try:
        receipt.update({"status": "RUNNING", "preflight_status": "IN_PROGRESS",
                        "registry_claim": {"execution_id": execution_id,
                                           "contract_sha256": claim_contract_sha,
                                           "input_binding_sha256": claim_binding_sha,
                                           "fit_budget_debit": 0}})
        receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
        audit = _preflight_checkpoint_replay_audit(
            stage=stage, attempt=attempt,
            manifest_path=manifest_path, expected_manifest_sha256=expected_manifest_sha256,
            source_archive_path=source_archive_path, expected_source_archive_sha256=expected_source_archive_sha256,
            dependency_archive_path=dependency_archive_path,
            expected_dependency_archive_sha256=expected_dependency_archive_sha256,
            contract_path=contract_path, binding_path=binding_path, activation_path=activation_path,
            registry_path=registry_path, execution_id=execution_id, guard=verification_guard,
        )
        contract_sha256, binding_sha256 = audit["contract_sha256"], audit["binding_sha256"]
        expected_code_shas = audit["expected_code_shas"]
        expected_route_code_sha256 = audit["expected_route_code_sha256"]
        source_archive_sha_before = audit["source_archive_sha_before"]
        dependency_archive_sha_before = audit["dependency_archive_sha_before"]
        output_manifest_path = audit["output_manifest_path"]
        output_manifest_sha_before = audit["output_manifest_sha_before"]
        selected_info, selected_keys_path = audit["selected_info"], audit["selected_keys_path"]
        cell_records = audit["cell_records"]
        immutable_paths, immutable_sha_before = audit["immutable_paths"], audit["immutable_sha_before"]
        expected_fit_events = audit["expected_fit_events"]
        receipt.update({
            "preflight_status": "VERIFIED",
            "original_manifest_sha256": expected_manifest_sha256,
            "contract_sha256": contract_sha256, "input_binding_sha256": binding_sha256,
            "source_archive_sha256": source_archive_sha_before,
            "dependency_archive_sha256": dependency_archive_sha_before,
            "original_code_sha256": expected_code_shas,
            "output_manifest_path": str(output_manifest_path),
            "output_manifest_sha256": output_manifest_sha_before,
            "selected_key_path": str(selected_keys_path),
            "selected_key_sha256": str(selected_info["sha256"]),
            "immutable_artifact_sha256_before": immutable_sha_before,
            "null_reasons": {},
        })
        verified_context = build_registry_event_context(
            stage=nofit_stage, binding=load_json(binding_path),
            contract_sha256=contract_sha256, binding_sha256=binding_sha256,
            code_shas=code_identity(), config_sha256=audit_config_sha256,
            arms=STAGE_SPECS[stage]["input_arms"], truth_access="BOUND_INPUT_READ_NOT_STARTED",
            artifact_paths=tuple(Path(path) for path in immutable_paths) + (receipt_path,),
            deterministic_metrics={"fit_calls": 0, "cells_planned": len(cell_records), "cells_verified": 0,
                                   "training_label_preparation_slots_by_horizon": None,
                                   "truth_positive_denominator": None, "selection_slots": None},
        )
        update_registry_execution_context(verification_guard, execution_id, **verified_context)
        receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
        with _pinned_route_from_archives(
            source_archive_path=source_archive_path,
            expected_source_archive_sha256=expected_source_archive_sha256,
            dependency_archive_path=dependency_archive_path,
            expected_dependency_archive_sha256=expected_dependency_archive_sha256,
            expected_code_shas=expected_code_shas,
            project_root=PROJECT_ROOT,
            contract_path=contract_path,
            binding_path=binding_path,
            activation_path=activation_path,
        ) as pinned_route:
            if sha256_file(activation_path) != immutable_sha_before[str(activation_path)]:
                raise ExecutionBlocked("activation changed before pinned source validation")
            input_read_started = True
            panel, calendar, arm_frames, _, records = pinned_route.load_stage_inputs(
                stage=stage, contract_path=contract_path, binding_path=binding_path,
                activation_path=activation_path, preparation_claim=claim,
            )
            labels_by_horizon: dict[int, pd.DataFrame] = {}
            cells_receipts: list[dict[str, Any]] = []
            folds = {fold.fold_id: fold for fold in pinned_route.fixed_folds()}
            if not STAGE_SPECS[stage]["requires_common_keys"]:
                selected_frames = {"M0": arm_frames["M0"]}
            else:
                selected_frames, _ = pinned_route.assemble_common_arm_features(arm_frames)
            for cell_record in cell_records:
                horizon, fold, arm = int(cell_record["horizon"]), cell_record["fold"], cell_record["arm"]
                if horizon not in labels_by_horizon:
                    labels_by_horizon[horizon] = pinned_route.build_close_return_labels(
                        panel, horizons=(horizon,), calendar=calendar,
                        signal_start=fold.start - pd.DateOffset(years=2),
                        signal_end_exclusive=fold.start,
                    )[horizon]
                prepared = _prepare_pinned_cell_data(
                    pinned_route, features=selected_frames[arm], labels=labels_by_horizon[horizon],
                    fold=fold, canary=bool(cell_record["canary"]),
                )
                checkpoint_path = Path(cell_record["checkpoint_path"])
                metadata = load_json(Path(cell_record["metadata_path"]))
                actual_key_rows = prepared["identity"]["training_keys"]["row_count"]
                expected_fit = expected_fit_events[str(cell_record["cell"])]
                if (expected_fit.get("training_rows_from_start_event") is not None and
                        int(expected_fit["training_rows_from_start_event"]) != actual_key_rows):
                    raise ExecutionBlocked(f"reconstructed train row count differs from original start event for {cell_record['cell']}")
                original_identity_train, original_identity_test = pinned_route.split_fold(
                    selected_frames[arm], labels_by_horizon[horizon], fold,
                )
                if bool(cell_record["canary"]):
                    original_identity_train, _ = pinned_route.canary_subset(
                        original_identity_train, original_identity_test,
                    )
                original_raw_key_identity = key_frame_identity(original_identity_train)
                original_key_sha = canonical_sha256({
                    "keys": [[pd.Timestamp(date).strftime("%Y-%m-%d"), str(code)]
                             for date, code in original_identity_train[list(KEYS)].itertuples(index=False, name=None)]
                })
                expected_old_identity = pinned_route.checkpoint_identity(
                    stage=stage, fold=fold, arm=arm, horizon=horizon, attempt=attempt,
                    contract_sha256=contract_sha256,
                    code_shas=expected_code_shas,
                    binding=load_json(binding_path), feature_names=prepared["feature_columns"],
                    training_key_sha256=original_key_sha,
                )
                if metadata.get("identity") != expected_old_identity:
                    raise ExecutionBlocked(f"reconstructed identity differs from immutable checkpoint config for {cell_record['cell']}")
                if metadata.get("artifact_sha256") != sha256_file(checkpoint_path):
                    raise ExecutionBlocked(f"immutable checkpoint hash differs from its sidecar for {cell_record['cell']}")
                payload = joblib.load(checkpoint_path)
                if tuple(payload.get("feature_columns", ())) != tuple(prepared["feature_columns"]):
                    raise ExecutionBlocked(f"reconstructed feature order differs from checkpoint for {cell_record['cell']}")
                if not _same_key_order(prepared["test"], payload.get("predictions", pd.DataFrame())):
                    raise ExecutionBlocked(f"reconstructed test key order differs from saved predictions for {cell_record['cell']}")
                original = payload["predictions"].copy()
                replay_scores = np.asarray(payload["model"].predict(prepared["xt"]), dtype=np.float64)
                original_scores = pd.to_numeric(original["prediction"], errors="coerce").to_numpy(dtype=np.float64)
                if len(replay_scores) != len(original_scores) or not np.isfinite(replay_scores).all() or not np.array_equal(replay_scores, original_scores):
                    raise ExecutionBlocked(f"reconstructed saved-model replay differs at tolerance 0 for {cell_record['cell']}")
                replay = prepared["test"].loc[:, list(KEYS)].reset_index(drop=True).copy()
                replay["date"] = pd.to_datetime(replay["date"]).dt.normalize()
                replay["code"] = replay["code"].astype(str)
                replay["prediction"] = replay_scores
                expected_predictions = original.loc[:, [*KEYS, "prediction"]].copy()
                expected_predictions["date"] = pd.to_datetime(expected_predictions["date"]).dt.normalize()
                expected_predictions["code"] = expected_predictions["code"].astype(str)
                expected_predictions["prediction"] = original_scores
                join = replay.merge(expected_predictions, on=list(KEYS), how="outer", validate="one_to_one",
                                    suffixes=("_replay", "_saved"), indicator=True, sort=False)
                if len(join) != len(replay) or not join["_merge"].eq("both").all():
                    raise ExecutionBlocked(f"reconstructed prediction keys do not join one-to-one for {cell_record['cell']}")
                replay_top = pinned_route.stable_top10(replay)
                selected_keys = pd.read_csv(selected_keys_path)
                selected_keys["date"] = pd.to_datetime(selected_keys["date"], errors="coerce").dt.normalize()
                saved_top = selected_keys.loc[
                    selected_keys["model_arm"].eq(arm) & selected_keys["model_horizon"].eq(horizon) &
                    selected_keys["fold"].eq(fold.fold_id),
                    ["date", "code", "rank", "prediction"],
                ].sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True)
                replay_top = replay_top[["date", "code", "rank", "prediction"]].sort_values(
                    ["date", "rank"], kind="mergesort").reset_index(drop=True)
                if (not saved_top[["date", "code", "rank"]].equals(replay_top[["date", "code", "rank"]]) or
                        not np.array_equal(saved_top["prediction"].to_numpy(dtype=np.float64),
                                           replay_top["prediction"].to_numpy(dtype=np.float64))):
                    raise ExecutionBlocked(f"reconstructed top10 differs from original selected keys for {cell_record['cell']}")
                cells_receipts.append({
                    "cell": cell_record["cell"], "arm": arm, "horizon": horizon,
                    "fold": fold.fold_id, "fit_execution_id": expected_fit["execution_id"],
                    "fit_run_started_event_sha256": expected_fit["run_started_event_sha256"],
                    "fit_terminal_event_sha256": expected_fit["terminal_event_sha256"],
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": immutable_sha_before[str(checkpoint_path)],
                    "checkpoint_metadata_sha256": immutable_sha_before[str(cell_record["metadata_path"])],
                    "reconstructed_training_rows": int(len(prepared["train"])),
                    "reconstructed_original_raw_training_rows": int(len(original_identity_train)),
                    "training_label_preparation_slots": int(prepared["training_label_preparation_slots"]),
                    "reconstructed_original_checkpoint_training_key_identity": original_raw_key_identity,
                    "reconstructed_original_checkpoint_training_key_sha256": original_key_sha,
                    "reconstructed_finite_fit_training_key_identity": prepared["identity"]["training_keys"],
                    "reconstructed_test_key_identity": prepared["identity"]["test_keys"],
                    "feature_columns": list(prepared["feature_columns"]),
                    "training_feature_matrix_sha256": prepared["identity"]["training_feature_matrix"]["sha256"],
                    "training_label_vector_sha256": prepared["identity"]["training_label_vector"]["sha256"],
                    "test_feature_matrix_sha256": prepared["identity"]["test_feature_matrix"]["sha256"],
                    "saved_candidate_prediction_identity": prediction_frame_identity(original),
                    "replay_prediction_identity": prediction_frame_identity(replay),
                    "prediction_join_validation": "ONE_TO_ONE",
                    "prediction_comparison": {"rtol": 0.0, "atol": 0.0, "status": "EXACT"},
                    "top10_parity": "EXACT",
                })
            immutable_sha_after = {str(path): sha256_file(path) for path in immutable_paths}
            if immutable_sha_after != immutable_sha_before:
                raise ExecutionBlocked("an original stage/source/checkpoint artifact changed during reconstruction")
            if sha256_file(manifest_path) != expected_manifest_sha256:
                raise ExecutionBlocked("original stage manifest changed during reconstruction")
            receipt.update({
                "status": "VERIFIED_RECONSTRUCTED_REPLAY",
                "expected_no_fit_terminal_status": "ENGINEERING_CANARY_ONLY",
                "cells": cells_receipts,
                "training_label_preparation_slots_by_horizon": {
                    str(horizon): int(len(labels)) for horizon, labels in labels_by_horizon.items()
                },
                "fit_time_training_key_hash": "CANNOT_CONFIRM",
                "reconstructed_training_key_hashes_are_retrospective": True,
                "fit_budget_debit": 0,
                "fit_calls": 0,
                "immutable_artifact_sha256_after": immutable_sha_after,
                "source_archive_sha256_after": sha256_file(source_archive_path),
                "dependency_archive_sha256_after": sha256_file(dependency_archive_path),
                "output_manifest_sha256_after": sha256_file(output_manifest_path),
                "selected_key_sha256_after": sha256_file(selected_keys_path),
                "limitations": [
                    "Reconstructed train/test identities describe the replay reconstruction, not contemporaneous fit-time capture.",
                    "Original fit-time train-key SHA remains CANNOT_CONFIRM.",
                    "No model acceptance or sealed-forward status is changed by this engineering receipt.",
                ],
            })
            registry_after = _read_registered_events(
                Path(registry_path) if registry_path is not None else PROJECT_ROOT / "contracts/EXPERIMENT_REGISTRY.jsonl"
            )
            for fit_execution_id, original_events in expected_fit_events.items():
                again = _verify_original_fit_event_pair(
                    registry_after, execution_id=fit_execution_id, stage=stage,
                    experiment_id=EXPERIMENT_ID, expected_route_code_sha256=expected_route_code_sha256,
                    contract_sha256=contract_sha256, binding_sha256=binding_sha256,
                )
                if (again["run_started_event_sha256"] != original_events["run_started_event_sha256"] or
                        again["terminal_event_sha256"] != original_events["terminal_event_sha256"]):
                    raise ExecutionBlocked("original fit registry evidence changed during no-fit verification")
            receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
            update_registry_execution_context(
                verification_guard, execution_id,
                truth_access="BOUND_INPUT_READ_AND_CHECKPOINT_REPLAY_COMPLETED_NO_FIT",
                deterministic_metrics={"fit_calls": 0, "cells_planned": 3, "cells_verified": len(cells_receipts),
                                       "training_label_preparation_slots_by_horizon": {
                                           str(horizon): int(len(value))
                                           for horizon, value in labels_by_horizon.items()},
                                       "truth_positive_denominator": None, "selection_slots": None,
                                       "checkpoint_replay_status": "VERIFIED_RECONSTRUCTED_REPLAY"},
            )
            terminal_attempted = True
            verification_guard.append_no_fit_terminal(claim, succeeded=True)
            terminalized = True
            return receipt
    except BaseException as exc:
        if not terminalized:
            if terminal_attempted:
                receipt.update({"status": "CANNOT_CONFIRM_NO_FIT_TERMINAL",
                                "error_type": type(exc).__name__, "error": str(exc)[:500],
                                "terminal_outcome": "CANNOT_CONFIRM", "fit_budget_debit": 0,
                                "fit_calls": 0, "no_fit_terminal_status": "CANNOT_CONFIRM",
                                "terminal_error_type": type(exc).__name__,
                                "terminal_error": str(exc)[:500]})
                try:
                    receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
                except BaseException as receipt_exc:
                    exc.add_note(f"Could not persist checkpoint replay failure receipt: {type(receipt_exc).__name__}: {receipt_exc}")
                raise

            finalization_errors: list[BaseException] = []
            try:
                update_registry_execution_context(
                    verification_guard, execution_id,
                    truth_access=("BOUND_INPUT_READ_OR_RECONSTRUCTION_FAILED" if input_read_started
                                  else "BOUND_INPUT_READ_NOT_COMPLETED"),
                    deterministic_metrics={"fit_calls": 0, "checkpoint_replay_status": "CANNOT_CONFIRM",
                                           "truth_positive_denominator": None, "selection_slots": None},
                )
            except BaseException as context_exc:
                finalization_errors.append(context_exc)
            receipt.update({"status": "CANNOT_CONFIRM_RECONSTRUCTED_REPLAY",
                            "error_type": type(exc).__name__, "error": str(exc)[:500],
                            "preflight_status": ("VERIFIED" if input_read_started else "FAILED"),
                            "fit_budget_debit": 0, "fit_calls": 0,
                            "expected_no_fit_terminal_status": "CANNOT_CONFIRM_ENGINEERING_CANARY"})
            try:
                receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
            except BaseException as receipt_exc:
                finalization_errors.append(receipt_exc)
            terminal_attempted = True
            try:
                verification_guard.append_no_fit_terminal(claim, succeeded=False,
                                                          error_type=type(exc).__name__)
                terminalized = True
            except BaseException as terminal_exc:
                receipt.update({"status": "CANNOT_CONFIRM_NO_FIT_TERMINAL",
                                "no_fit_terminal_status": "CANNOT_CONFIRM",
                                "terminal_error_type": type(terminal_exc).__name__,
                                "terminal_error": str(terminal_exc)[:500]})
                try:
                    receipt_owner = _update_owned_json(receipt_path, receipt, receipt_owner)
                except BaseException as receipt_exc:
                    exc.add_note(f"Could not persist no-fit terminal failure: {type(receipt_exc).__name__}: {receipt_exc}")
                exc.add_note(f"No-fit terminalization was not confirmed: {type(terminal_exc).__name__}: {terminal_exc}")
                raise exc from terminal_exc
            if finalization_errors:
                finalization_detail = "; ".join(
                    f"{type(error).__name__}: {error}" for error in finalization_errors
                )
                exc.add_note(f"No-fit failure finalization encountered: {finalization_detail}")
        raise


def _fit_one_cell(*, cell: Mapping[str, Any], features: pd.DataFrame, labels: pd.DataFrame,
                  output_root: Path, identity: Mapping[str, Any], guard: Any,
                  contract_sha256: str, binding_sha256: str,
                  binding: Mapping[str, Any], prepared: Mapping[str, Any] | None = None) -> dict[str, Any]:
    arm = str(cell["arm"])
    fold: Fold = cell["fold"]
    horizon = int(cell["horizon"])
    cell_key = f"{cell['stage']}_{arm}_H{horizon}_{fold.fold_id}_{cell['attempt']}"
    checkpoint_path = Path(output_root) / "checkpoints" / f"{cell_key}.joblib"
    replay_receipt_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".replay.json")
    candidates = sorted(checkpoint_path.parent.glob(f"{cell['stage']}_{arm}_H{horizon}_{fold.fold_id}_*.joblib"))
    cached = None
    cached_path = None
    for candidate in candidates:
        cached = load_reusable_checkpoint(candidate, identity)
        if cached is not None:
            cached_path = candidate
            break
    if cached is not None:
        return {"cell": cell_key, "arm": arm, "horizon": horizon, "fold": fold.fold_id,
                "canary": bool(cell["canary"]), "status": "REUSED_CHECKPOINT", "predictions": cached["predictions"],
                "model": cached["model"], "fit_called": False, "checkpoint_path": str(cached_path)}
    if guard is None:
        raise ExecutionBlocked("real fit requires the existing official registry guard")
    metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if checkpoint_path.exists() or metadata_path.exists() or replay_receipt_path.exists():
        raise ExecutionBlocked(f"checkpoint path already exists but is not reusable: {checkpoint_path.name}")
    # The official registry hashes artifact paths for RUN_STARTED, so persist a
    # small pre-fit intent at those exact paths before claiming the fit debit.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(checkpoint_path, {"status": "PLANNED_BEFORE_FIT", "cell": cell_key,
                                        "identity": dict(identity)})
    atomic_write_json(metadata_path, {"status": "PLANNED_BEFORE_FIT", "identity": dict(identity),
                                      "artifact_sha256": sha256_file(checkpoint_path)})
    atomic_write_json(replay_receipt_path, {
        "schema_version": REPLAY_RECEIPT_SCHEMA,
        "status": "PLANNED_BEFORE_FIT",
        "provenance": "FIT_TIME_SAVED_MODEL_REPLAY",
        "execution_id": f"{cell['stage']}_{arm}_H{horizon}_{fold.fold_id}_{cell['attempt']}",
        "config_sha256": str(identity["config_sha256"]),
    })
    prepared_data = dict(prepared or prepare_cell_data(
        features=features, labels=labels, fold=fold, canary=bool(cell["canary"]),
    ))
    if prepared_data.get("identity") != identity.get("data_identity"):
        raise ContractError(f"prepared train/test identity differs from checkpoint plan in {cell_key}")
    train, test = prepared_data["train"], prepared_data["test"]
    feature_columns = tuple(prepared_data["feature_columns"])
    x, y, weights, xt = (prepared_data[key] for key in ("x", "y", "weights", "xt"))
    model = model_for_arm()
    execution_id = f"{cell['stage']}_{arm}_H{horizon}_{fold.fold_id}_{cell['attempt']}"
    registry_context = build_registry_event_context(
        stage=str(cell["stage"]), binding=binding,
        contract_sha256=contract_sha256, binding_sha256=binding_sha256,
        code_shas=identity["code_sha256"], config_sha256=str(identity["config_sha256"]),
        arms=(arm,), truth_access="TRAIN_LABELS_READ_PAST_ONLY",
        signal_dates=train["date"], target_dates=train["label_end_date"],
        checkpoint_path=checkpoint_path,
        artifact_paths=(checkpoint_path, metadata_path, replay_receipt_path),
        deterministic_metrics={"training_rows": int(len(train)),
                               "training_signal_dates": int(train["date"].nunique()),
                               "test_prediction_rows": int(len(test)),
                               "test_signal_dates": int(test["date"].nunique()),
                               "training_keyset_sha256": identity["data_identity"]["training_keys"]["sorted_keyset_sha256"],
                               "training_key_order_sha256": identity["data_identity"]["training_keys"]["ordered_keys_sha256"],
                               "training_unique_keys": identity["data_identity"]["training_keys"]["unique_key_count"],
                               "training_signal_date_min": identity["data_identity"]["training_keys"]["date_min"],
                               "training_signal_date_max": identity["data_identity"]["training_keys"]["date_max"],
                               "training_feature_matrix_sha256": identity["data_identity"]["training_feature_matrix"]["sha256"],
                               "training_label_vector_sha256": identity["data_identity"]["training_label_vector"]["sha256"],
                               "test_keyset_sha256": identity["data_identity"]["test_keys"]["sorted_keyset_sha256"],
                               "test_feature_matrix_sha256": identity["data_identity"]["test_feature_matrix"]["sha256"],
                               "training_label_preparation_slots": int(prepared_data["training_label_preparation_slots"]),
                               "checkpoint_replay_status": "PENDING_BEFORE_FIT"},
    )
    set_registry_execution_context(guard, execution_id, registry_context)

    def fit_and_predict() -> pd.DataFrame:
        with threadpool_limits(limits=4):
            model.fit(x, y, sample_weight=weights)
        score = np.asarray(model.predict(xt), dtype=np.float64)
        if len(score) != len(test) or not np.isfinite(score).all():
            raise ContractError("fit produced missing or non-finite predictions")
        predictions = pd.DataFrame({"date": test["date"].to_numpy(), "code": test["code"].astype(str).to_numpy(),
                                    "prediction": score})
        with tempfile.NamedTemporaryFile(prefix="model-selection-roundtrip-", suffix=".joblib", delete=False) as handle:
            replay_path = Path(handle.name)
        try:
            joblib.dump(model, replay_path)
            replay_model = joblib.load(replay_path)
            replay = np.asarray(replay_model.predict(xt), dtype=np.float64)
            if not np.array_equal(score, replay):
                raise ContractError("saved-model prediction parity failed")
        finally:
            if replay_path.exists():
                replay_path.unlink()
        # Persist the test matrix with the estimator, then reload the actual
        # durable checkpoint and write its replay receipt before the fit terminal.
        save_checkpoint(checkpoint_path, {"model": model, "predictions": predictions,
                                          "feature_columns": feature_columns,
                                          "replay_fixture": {
                                              "test_keys": test.loc[:, list(KEYS)].reset_index(drop=True),
                                              "test_features": np.asarray(xt, dtype=np.float64, order="C"),
                                          }}, identity)
        replay_receipt = verify_saved_checkpoint_replay(
            checkpoint_path=checkpoint_path, replay_receipt_path=replay_receipt_path,
            identity=identity, execution_id=execution_id,
        )
        update_registry_execution_context(
            guard, execution_id,
            checkpoint_ref={"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
            deterministic_metrics={
                "training_rows": int(len(train)),
                "training_signal_dates": int(train["date"].nunique()),
                "test_prediction_rows": int(len(test)),
                "test_signal_dates": int(test["date"].nunique()),
                "training_keyset_sha256": identity["data_identity"]["training_keys"]["sorted_keyset_sha256"],
                "training_key_order_sha256": identity["data_identity"]["training_keys"]["ordered_keys_sha256"],
                "training_unique_keys": identity["data_identity"]["training_keys"]["unique_key_count"],
                "training_signal_date_min": identity["data_identity"]["training_keys"]["date_min"],
                "training_signal_date_max": identity["data_identity"]["training_keys"]["date_max"],
                "training_feature_matrix_sha256": identity["data_identity"]["training_feature_matrix"]["sha256"],
                "training_label_vector_sha256": identity["data_identity"]["training_label_vector"]["sha256"],
                "test_keyset_sha256": identity["data_identity"]["test_keys"]["sorted_keyset_sha256"],
                "test_feature_matrix_sha256": identity["data_identity"]["test_feature_matrix"]["sha256"],
                "training_label_preparation_slots": int(prepared_data["training_label_preparation_slots"]),
                "checkpoint_replay_status": replay_receipt["status"],
                "checkpoint_replay_prediction_sha256": replay_receipt["replay_prediction_identity"]["ordered_rows_sha256"],
                "checkpoint_replay_top10_status": replay_receipt["top10_parity"]["status"],
            },
        )
        return predictions

    arm_identity = f"{cell['stage']}:{arm}:H{horizon}:{fold.fold_id}"
    predictions = guard.execute_once(
        execution_id=execution_id,
        arm=arm_identity,
        contract_sha256=contract_sha256,
        input_binding_sha256=binding_sha256,
        fit_callable=fit_and_predict,
    )
    return {"cell": cell_key, "arm": arm, "horizon": horizon, "fold": fold.fold_id,
            "canary": bool(cell["canary"]), "status": "COMPLETED", "predictions": predictions,
            "model": model, "fit_called": True, "checkpoint_path": str(checkpoint_path)}


def run_stage_cells(*, stage: str, arm_frames: Mapping[str, pd.DataFrame],
                    labels_by_cell: Mapping[tuple[str, int], pd.DataFrame], output_root: Path,
                    contract_sha256: str, binding_sha256: str, binding: Mapping[str, Any],
                    guard: Any, code_shas: Mapping[str, str] | None = None,
                    attempt: str = "attempt_01", cell_guard_factory: Any = None) -> dict[str, Any]:
    """Run one exact stage plan; each uncached fit is registered before ``fit``."""
    cells = stage_plan(stage, attempt=attempt)
    if guard is None:
        raise ExecutionBlocked("stage runner requires a reviewer-approved official lifecycle")
    accounting = guard.accounting()
    if accounting.dangling_execution_ids:
        raise ExecutionBlocked(f"dangling registered fits require diagnosis: {accounting.dangling_execution_ids}")
    used_arms = set(STAGE_SPECS[stage]["arms"])
    if not STAGE_SPECS[stage]["requires_common_keys"]:
        selected_frames = {"M0": arm_frames["M0"]}
    else:
        if not set(ARMS).issubset(arm_frames):
            raise ContractError("indexed stage requires all five arms for the frozen common feature key set")
        selected_frames, _ = assemble_common_arm_features({arm: arm_frames[arm] for arm in ARMS})
    output = []
    folds = {fold.fold_id: fold for fold in fixed_folds()}
    for cell in cells:
        arm = cell["arm"]
        fold = folds[cell["fold"].fold_id]
        label_key = (fold.fold_id, int(cell["horizon"]))
        if label_key not in labels_by_cell:
            raise ContractError(f"missing purged training labels for {label_key}")
        prepared = prepare_cell_data(
            features=selected_frames[arm], labels=labels_by_cell[label_key], fold=fold,
            canary=bool(cell["canary"]),
        )
        feature_names = prepared["feature_columns"]
        identity = checkpoint_identity(
            stage=stage, fold=fold, arm=arm, horizon=cell["horizon"], attempt=cell["attempt"],
            contract_sha256=contract_sha256, code_shas=code_shas or code_identity(),
            binding=binding, feature_names=feature_names,
            data_identity=prepared["identity"],
        )
        checkpoint_path = Path(output_root) / "checkpoints" / f"{stage}_{arm}_H{cell['horizon']}_{fold.fold_id}_{cell['attempt']}.joblib"
        cell_guard = cell_guard_factory(cell, checkpoint_path) if cell_guard_factory is not None else guard
        output.append(_fit_one_cell(cell=cell, features=selected_frames[arm],
                                    labels=labels_by_cell[label_key],
                                    output_root=output_root, identity=identity, guard=cell_guard,
                                    contract_sha256=contract_sha256, binding_sha256=binding_sha256,
                                    binding=binding, prepared=prepared))
    return {"stage": stage, "cells_planned": len(cells), "fit_calls": sum(item["fit_called"] for item in output),
            "results": output, "state": "ENGINEERING_CANARY_ONLY" if STAGE_SPECS[stage]["canary"] else "DIRECTION_ONLY"}


def load_m0_formal_baseline(*, arm_frames: Mapping[str, pd.DataFrame], labels_by_cell: Mapping[tuple[str, int], pd.DataFrame],
                            output_root: Path, binding: Mapping[str, Any], contract_sha256: str,
                            code_shas: Mapping[str, str]) -> dict[tuple[str, int], dict[str, Any]]:
    """Load exact M0 checkpoints only when trained on the current M0-M4 key pool."""
    common_frames, _ = assemble_common_arm_features({arm: arm_frames[arm] for arm in ARMS})
    features = common_frames["M0"]
    baseline: dict[tuple[str, int], dict[str, Any]] = {}
    for fold in fixed_folds():
        for horizon in HORIZONS:
            label_key = (fold.fold_id, horizon)
            if label_key not in labels_by_cell:
                raise ContractError(f"M0 baseline requires training labels for {label_key}")
            prepared = prepare_cell_data(features=features, labels=labels_by_cell[label_key],
                                         fold=fold, canary=False)
            train, test = prepared["train"], prepared["test"]
            feature_names = prepared["feature_columns"]
            identity = checkpoint_identity(
                stage="M0_FORMAL", fold=fold, arm="M0", horizon=horizon, attempt="attempt_01",
                contract_sha256=contract_sha256, code_shas=code_shas, binding=binding,
                feature_names=feature_names,
                data_identity=prepared["identity"],
            )
            pattern = f"M0_FORMAL_M0_H{horizon}_{fold.fold_id}_*.joblib"
            candidates = sorted((Path(output_root) / "checkpoints").glob(pattern))
            matched_path = None
            match = None
            for candidate in candidates:
                match = load_reusable_checkpoint(candidate, identity)
                if match is not None:
                    matched_path = candidate
                    break
            if match is None:
                raise ExecutionBlocked(f"missing reusable M0 formal checkpoint for {fold.fold_id} h={horizon}")
            predictions = match["predictions"].copy()
            predictions["date"] = pd.to_datetime(predictions["date"]).dt.normalize()
            predictions["code"] = predictions["code"].astype(str)
            expected_keys = set(zip(pd.to_datetime(test["date"]).dt.normalize(), test["code"].astype(str)))
            actual_keys = set(zip(predictions["date"], predictions["code"]))
            if actual_keys != expected_keys:
                raise ExecutionBlocked(f"M0 formal predictions do not cover the current test feature set for {fold.fold_id} h={horizon}")
            baseline[label_key] = {"cell": f"M0_FORMAL_M0_H{horizon}_{fold.fold_id}", "arm": "M0",
                                   "horizon": horizon, "fold": fold.fold_id, "canary": False,
                                   "status": "REUSED_CHECKPOINT", "predictions": predictions,
                                   "model": match["model"], "fit_called": False,
                                   "checkpoint_path": str(matched_path)}
    return baseline


def run_reviewed_stage(*, stage: str, output_root: Path, attempt: str = "attempt_01",
                       contract_path: Path = CONTRACT_PATH, binding_path: Path = INPUT_BINDING_PATH,
                       activation_path: Path = ACTIVATION_PATH) -> dict[str, Any]:
    """Gate, register, materialize, and execute one explicitly activated stage."""
    if stage not in STAGE_SPECS:
        raise ContractError(f"unknown model-selection stage: {stage}")
    contract = load_json(contract_path)
    binding = load_json(binding_path)
    activation = load_json(activation_path)
    contract_sha = sha256_file(contract_path)
    binding_sha = sha256_file(binding_path)
    validate_review_activation(contract, binding, activation,
                               contract_sha256=contract_sha,
                               binding_sha256=binding_sha,
                               code_sha256=route_code_sha256(),
                               stage=stage,
                               arms=STAGE_SPECS[stage]["input_arms"],
                               contract_path=contract_path,
                               binding_path=binding_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{stage}_{attempt}_stage_manifest.json"
    if manifest_path.exists():
        raise ExecutionBlocked(f"stage attempt already has a manifest: {manifest_path.name}")
    cells = stage_plan(stage, attempt=attempt)
    preparation_receipt_path = output_root / f"{stage}_{attempt}_preparation_receipt.json"
    if preparation_receipt_path.exists():
        raise ExecutionBlocked(f"stage attempt already has a preparation receipt: {preparation_receipt_path.name}")
    compatibility_sidecar_path = output_root / f"{stage}_{attempt}_input_compatibility_sidecar.json"
    if compatibility_sidecar_path.exists():
        raise ExecutionBlocked(f"stage attempt already has an input compatibility sidecar: {compatibility_sidecar_path.name}")
    manifest = {
        "schema_version": "A_SHARE_MODEL_SELECTION_STAGE_MANIFEST_V1",
        "status": "PLANNED",
        "stage": stage,
        "attempt": attempt,
        "planned_cells": [{"arm": cell["arm"], "horizon": cell["horizon"],
                           "fold": cell["fold"].fold_id, "canary": cell["canary"]} for cell in cells],
        "input_arms": list(STAGE_SPECS[stage]["input_arms"]),
        "requires_common_feature_keys": bool(STAGE_SPECS[stage]["requires_common_keys"]),
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "code_sha256": code_identity(),
        "input_compatibility_sidecar_path": str(compatibility_sidecar_path.resolve()),
        "fit_calls": 0,
        "security_flags": {"research_only": True, "production_allowed": False,
                           "buy_gate_updated": False, "orders_sent": False},
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(preparation_receipt_path, {
        "schema_version": "A_SHARE_MODEL_SELECTION_PREPARATION_RECEIPT_V1",
        "stage": stage,
        "attempt": attempt,
        "planned_cells": manifest["planned_cells"],
        "input_arms": manifest["input_arms"],
        "requires_common_feature_keys": manifest["requires_common_feature_keys"],
        "contract_sha256": contract_sha,
        "input_binding_sha256": binding_sha,
        "code_sha256": manifest["code_sha256"],
        "security_flags": manifest["security_flags"],
    })
    atomic_write_json(compatibility_sidecar_path, {
        "schema_version": "A_SHARE_MODEL_SELECTION_INPUT_COMPAT_SIDECAR_V1",
        "status": "AWAITING_BOUND_INPUT_READ",
        "input_binding_sha256": binding_sha,
        "reason": "NO_PANEL_OR_FEATURE_DATA_READ_BEFORE_DURABLE_PREPARATION_CLAIM",
    })
    prep_guard = make_official_fit_guard(contract_path=contract_path, binding_path=binding_path,
                                         stage=f"{stage}_DATA_PREP",
                                         artifact_paths=(preparation_receipt_path, compatibility_sidecar_path))
    if prep_guard.accounting().dangling_execution_ids:
        raise ExecutionBlocked("a prior registered run is dangling; diagnose before resuming")
    prep_execution_id = f"{stage}_DATA_PREP_{attempt}"
    prep_registry_context = build_registry_event_context(
        stage=f"{stage}_DATA_PREP", binding=binding,
        contract_sha256=contract_sha, binding_sha256=binding_sha,
        code_shas=code_identity(), config_sha256=contract_sha,
        arms=STAGE_SPECS[stage]["input_arms"], truth_access="BOUND_INPUT_READ_NOT_STARTED",
        artifact_paths=(preparation_receipt_path, compatibility_sidecar_path),
        deterministic_metrics={"training_label_preparation_slots_by_cell_key": None,
                               "selection_slots": None},
    )
    set_registry_execution_context(prep_guard, prep_execution_id, prep_registry_context)
    prep_claim = prep_guard.claim_no_fit(execution_id=prep_execution_id,
                                         arm=f"{stage}:DATA_PREPARATION",
                                         contract_sha256=contract_sha,
                                         input_binding_sha256=binding_sha)
    label_processing_started = False
    try:
        panel, calendar, arm_frames, labels, records = load_stage_inputs(
            stage=stage, contract_path=contract_path, binding_path=binding_path,
            activation_path=activation_path, preparation_claim=prep_claim,
        )
        compatibility_sidecar = records["input_compat_sidecar"]
        compatibility_sidecar.update({
            "status": "COMPLETE",
            "input_binding_sha256": binding_sha,
            "contract_sha256": contract_sha,
            "code_sha256": manifest["code_sha256"],
        })
        atomic_write_json(compatibility_sidecar_path, compatibility_sidecar)
        labels_by_cell: dict[tuple[str, int], pd.DataFrame] = {}
        fold_lookup = {fold.fold_id: fold for fold in fixed_folds()}
        label_processing_started = True
        for cell in cells:
            label_key = (cell["fold"].fold_id, int(cell["horizon"]))
            if label_key not in labels_by_cell:
                fold = fold_lookup[label_key[0]]
                labels_by_cell[label_key] = build_close_return_labels(
                    panel, horizons=(label_key[1],), calendar=calendar,
                    signal_start=fold.start - pd.DateOffset(years=2),
                    signal_end_exclusive=fold.start,
                )[label_key[1]]
        manifest.update({"status": "INPUTS_READY", "panel_rows": int(len(panel)),
                         "calendar_dates": int(len(calendar)), "feature_rows_by_arm": {
                             arm: int(len(frame)) for arm, frame in arm_frames.items()},
                         "input_compatibility_sidecar": {
                             "path": str(compatibility_sidecar_path.resolve()),
                             "sha256": sha256_file(compatibility_sidecar_path),
                             "source_row_count": compatibility_sidecar["source_row_accounting"]["source_row_count"],
                             "quarantined_row_count": compatibility_sidecar["source_row_accounting"]["quarantined_row_count"],
                             "coverage_loss_by_arm": compatibility_sidecar["coverage_loss_by_arm"],
                         },
                         "training_label_slots_by_cell_key": {
                             f"{fold_id}:H{horizon}": int(len(frame))
                             for (fold_id, horizon), frame in labels_by_cell.items()},
                         "formula_source_sha256": {
                             key: str(binding["formula_inputs"][key]["sha256"]) for key in ("v192", "v193")}})
        atomic_write_json(manifest_path, manifest)
        training_dates = pd.concat([frame["date"] for frame in labels_by_cell.values()], ignore_index=True)
        target_dates = pd.concat([frame["label_end_date"] for frame in labels_by_cell.values()], ignore_index=True)
        prep_registry_context = build_registry_event_context(
            stage=f"{stage}_DATA_PREP", binding=binding,
            contract_sha256=contract_sha, binding_sha256=binding_sha,
            code_shas=code_identity(), config_sha256=contract_sha,
            arms=STAGE_SPECS[stage]["input_arms"], truth_access="TRAIN_LABELS_READ_PAST_ONLY",
            signal_dates=training_dates, target_dates=target_dates,
            artifact_paths=(preparation_receipt_path, compatibility_sidecar_path),
            deterministic_metrics={"training_label_preparation_slots_by_cell_key": {
                                       f"{fold_id}:H{horizon}": int(len(frame))
                                       for (fold_id, horizon), frame in labels_by_cell.items()},
                                   "training_label_cells": int(len(labels_by_cell)),
                                   "selection_slots": None},
        )
        set_registry_execution_context(prep_guard, prep_execution_id, prep_registry_context)
    except BaseException as exc:
        try:
            update_registry_execution_context(
                prep_guard, prep_execution_id,
                truth_access=("TRAIN_LABELS_READ_OR_ATTEMPTED" if label_processing_started
                              else "BOUND_INPUT_READ_OR_FEATURE_PREPARATION_FAILED"),
            )
            prep_guard.append_no_fit_terminal(prep_claim, succeeded=False, error_type=type(exc).__name__)
        finally:
            manifest.update({"status": "CANNOT_CONFIRM_INPUT_PREPARATION", "error_type": type(exc).__name__})
            atomic_write_json(manifest_path, manifest)
        raise
    prep_guard.append_no_fit_terminal(prep_claim, succeeded=True)

    def cell_guard_factory(cell: Mapping[str, Any], checkpoint_path: Path):
        return make_official_fit_guard(
            contract_path=contract_path,
            binding_path=binding_path,
            stage=str(cell["stage"]),
            artifact_paths=(checkpoint_path,
                            checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
                            checkpoint_path.with_suffix(checkpoint_path.suffix + ".replay.json")),
        )

    baseline: dict[tuple[str, int], dict[str, Any]] = {}
    if stage == "INDEXED_FORMAL":
        try:
            baseline = load_m0_formal_baseline(
                arm_frames=arm_frames, labels_by_cell=labels_by_cell, output_root=output_root,
                binding=binding, contract_sha256=contract_sha, code_shas=code_identity(),
            )
        except BaseException as exc:
            manifest.update({"status": "CANNOT_CONFIRM_M0_BASELINE", "error_type": type(exc).__name__})
            atomic_write_json(manifest_path, manifest)
            raise

    result = run_stage_cells(stage=stage, arm_frames=arm_frames,
                             labels_by_cell=labels_by_cell, output_root=output_root,
                             contract_sha256=contract_sha, binding_sha256=binding_sha,
                             binding=binding, guard=prep_guard, code_shas=code_identity(),
                             attempt=attempt, cell_guard_factory=cell_guard_factory)
    stage_results = list(result["results"])
    if stage == "INDEXED_FORMAL":
        stage_results.extend(baseline.values())

    evaluation_guard = make_official_fit_guard(
        contract_path=contract_path, binding_path=binding_path,
        stage=f"{stage}_EVALUATION", artifact_paths=(),
    )
    if evaluation_guard.accounting().dangling_execution_ids:
        raise ExecutionBlocked("a prior evaluation registration is dangling; diagnose before resuming")
    evaluation_execution_id = f"{stage}_EVALUATION_{attempt}"
    evaluation_registry_context = build_registry_event_context(
        stage=f"{stage}_EVALUATION", binding=binding,
        contract_sha256=contract_sha, binding_sha256=binding_sha,
        code_shas=code_identity(), config_sha256=contract_sha,
        arms=STAGE_SPECS[stage]["input_arms"], truth_access="TEST_LABEL_JOIN_NOT_STARTED",
        artifact_paths=(),
        deterministic_metrics={"selection_slots": None, "known_label_slots": None,
                               "unknown_label_slots": None},
    )
    set_registry_execution_context(evaluation_guard, evaluation_execution_id, evaluation_registry_context)
    evaluation_claim = evaluation_guard.claim_no_fit(
        execution_id=evaluation_execution_id, arm=f"{stage}:ISOLATED_EVALUATION",
        contract_sha256=contract_sha, input_binding_sha256=binding_sha,
    )
    try:
        outputs = write_stage_outputs(
            stage=stage, attempt=attempt, results=stage_results, panel=panel, calendar=calendar,
            arm_frames=arm_frames, records=records, output_root=output_root,
            contract_sha256=contract_sha, binding_sha256=binding_sha,
        )
        evaluation_artifacts = [
            Path(info["path"]) for info in outputs["artifacts"].values() if info.get("path")
        ]
        evaluation_registry_context = build_registry_event_context(
            stage=f"{stage}_EVALUATION", binding=binding,
            contract_sha256=contract_sha, binding_sha256=binding_sha,
            code_shas=code_identity(), config_sha256=contract_sha,
            arms=STAGE_SPECS[stage]["input_arms"], truth_access="EXPOSED_TEST_LABELS_READ_AFTER_KEYS_FROZEN",
            signal_dates=outputs["signal_dates"], target_dates=outputs["target_dates"],
            artifact_paths=tuple(evaluation_artifacts) + (Path(outputs["output_manifest_path"]),),
            deterministic_metrics={"selection_slots": int(outputs["selection_slots"]),
                                   "known_label_slots": int(outputs["known_label_slots"]),
                                   "unknown_label_slots": int(outputs["unknown_label_slots"]),
                                   "truth_positive_denominator": None},
        )
        set_registry_execution_context(evaluation_guard, evaluation_execution_id, evaluation_registry_context)
        evaluation_guard.append_no_fit_terminal(evaluation_claim, succeeded=True)
    except BaseException as exc:
        try:
            update_registry_execution_context(
                evaluation_guard, evaluation_execution_id,
                truth_access="EXPOSED_TEST_LABELS_READ_OR_ATTEMPTED_AFTER_KEYS_FROZEN",
            )
            evaluation_guard.append_no_fit_terminal(evaluation_claim, succeeded=False, error_type=type(exc).__name__)
        finally:
            manifest.update({"status": "CANNOT_CONFIRM_EVALUATION", "error_type": type(exc).__name__})
            atomic_write_json(manifest_path, manifest)
        raise
    manifest.update({"status": result["state"], "fit_calls": result["fit_calls"],
                     "cell_results": [{"cell": item["cell"], "status": item["status"],
                                       "fit_called": item["fit_called"],
                                       "checkpoint_path": item["checkpoint_path"]}
                                      for item in stage_results],
                     "output_manifest_path": outputs["output_manifest_path"],
                     "output_artifacts": outputs["artifacts"],
                     "m4_vs_m0_acceptance": outputs["acceptance"]})
    atomic_write_json(manifest_path, manifest)
    return {"stage": stage, "attempt": attempt, "state": result["state"],
            "fit_calls": result["fit_calls"], "manifest_path": str(manifest_path),
            "preparation_receipt_path": str(preparation_receipt_path),
            "output_manifest_path": outputs["output_manifest_path"],
            "checkpoint_paths": [item["checkpoint_path"] for item in stage_results]}


def split_fold(features: pd.DataFrame, labels: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join outcomes to past training signals only; return unlabeled test features."""
    signal_date = pd.to_datetime(features["date"]).dt.normalize()
    start = fold.start - pd.DateOffset(years=2)
    train_mask = (signal_date >= start) & (signal_date < fold.start)
    train_features = features.loc[train_mask].copy()
    train = train_features.merge(labels, on=list(KEYS), how="left", validate="one_to_one", sort=False)
    label_end = pd.to_datetime(train["label_end_date"], errors="coerce")
    train_mask = (label_end < fold.start) & np.isfinite(pd.to_numeric(train["label_return"], errors="coerce"))
    train = train.loc[train_mask].copy().sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    test_mask = signal_date.between(fold.start, fold.end, inclusive="both")
    test = features.loc[test_mask].copy().sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    if train.empty or test.empty:
        raise ContractError(f"empty purged train or test for {fold.fold_id}")
    if (pd.to_datetime(train["label_end_date"]) >= fold.start).any():
        raise ContractError("purge invariant failed: training label reaches test start")
    if {"label_return", "target_date", "label_end_date"}.intersection(test.columns):
        raise ContractError("test features must remain unlabeled until selection keys are frozen")
    return train, test


def predict_with_roundtrip(model: Any, train: pd.DataFrame, test: pd.DataFrame,
                           feature_columns: Sequence[str], model_path: Path, *, guard: Any,
                           execution_id: str, arm: str, contract_sha256: str,
                           input_binding_sha256: str) -> pd.DataFrame:
    """Register one frozen fit before fitting, then verify saved-model parity."""
    if not feature_columns or any(c not in train or c not in test for c in feature_columns):
        raise ContractError("feature order is incomplete")
    x = train[list(feature_columns)].to_numpy(dtype=np.float64)
    y = pd.to_numeric(train["label_return"], errors="raise").to_numpy(dtype=np.float64)
    weights = equal_day_weights(train["date"])
    xt = test[list(feature_columns)].to_numpy(dtype=np.float64)
    if guard is None:
        raise ExecutionBlocked("fit requires the existing official registry guard")
    def fit_once() -> np.ndarray:
        with threadpool_limits(limits=4):
            model.fit(x, y, sample_weight=weights)
        score = np.asarray(model.predict(xt), dtype=np.float64)
        return score
    score = guard.execute_once(execution_id=execution_id, arm=arm,
                               contract_sha256=contract_sha256,
                               input_binding_sha256=input_binding_sha256,
                               fit_callable=fit_once)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": tuple(feature_columns)}, model_path)
    reloaded = joblib.load(model_path)
    replay = np.asarray(reloaded["model"].predict(xt), dtype=np.float64)
    if not np.array_equal(score, replay):
        raise ContractError("saved-model prediction parity failed")
    prediction = test[["date", "code"]].copy()
    prediction["prediction"] = score
    return prediction


def stable_top10(predictions: pd.DataFrame) -> pd.DataFrame:
    """Top 10 within each day, stable ties by ascending code; never refill."""
    required = {"date", "code", "prediction"}
    if not required.issubset(predictions):
        raise ContractError("prediction keys/score missing")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame[np.isfinite(frame["prediction"].to_numpy(dtype=np.float64))]
    frame["code"] = frame["code"].astype(str)
    frame = frame.sort_values(["date", "prediction", "code"], ascending=[True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby("date", sort=False).cumcount() + 1
    return frame.loc[frame["rank"] <= 10].reset_index(drop=True)


def freeze_selection_keys(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Freeze rank lists before any test-label join."""
    selected = []
    for item in results:
        top = stable_top10(item["predictions"])
        if top.empty:
            raise ContractError(f"no ranked candidates for {item.get('cell')}")
        top["model_arm"] = str(item["arm"])
        top["model_horizon"] = int(item["horizon"])
        top["fold"] = str(item["fold"])
        top["canary"] = bool(item["canary"])
        top["cell"] = str(item["cell"])
        selected.append(top[["date", "code", "model_arm", "model_horizon", "fold", "canary", "cell", "rank", "prediction"]])
    if not selected:
        raise ContractError("stage produced no selection lists")
    result = pd.concat(selected, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if result.duplicated(["date", "code", "model_arm", "model_horizon", "fold"]).any():
        raise ContractError("duplicate model selection key after ranking")
    return result.sort_values(["model_arm", "model_horizon", "fold", "date", "rank"], kind="mergesort").reset_index(drop=True)


def attach_selected_returns(selected_keys: pd.DataFrame, panel: pd.DataFrame, calendar: pd.DataFrame,
                            formula_features: pd.DataFrame, arm_frames: Mapping[str, pd.DataFrame],
                            industry: pd.DataFrame | None) -> pd.DataFrame:
    """Join close outcomes and diagnostics only after ranks have been frozen."""
    keys = selected_keys.copy()
    selected_pairs = keys[["date", "code"]].drop_duplicates().reset_index(drop=True)
    for horizon, labels in build_close_return_labels(panel, horizons=HORIZONS, calendar=calendar,
                                                     signal_keys=selected_pairs).items():
        outcomes = labels[["date", "code", "target_date", "label_return"]].rename(
            columns={"target_date": f"target_date_h{horizon}", "label_return": f"actual_close_return_h{horizon}"}
        )
        keys = keys.merge(outcomes, on=list(KEYS), how="left", validate="many_to_one", sort=False)
    feature_columns = [name for name in CORE_FEATURES + ("V192_XG", "V193_XG") if name in formula_features.columns]
    keys = keys.merge(formula_features[["date", "code", *feature_columns]], on=list(KEYS), how="left",
                      validate="many_to_one", sort=False)
    prices = panel[["date", "code", "close", "amount", "volume"]].rename(
        columns={"close": "signal_close", "amount": "signal_amount_cny", "volume": "signal_volume_shares"}
    )
    keys = keys.merge(prices, on=list(KEYS), how="left", validate="many_to_one", sort=False)
    # Use the bound index and observed market context for every arm's displayed row.
    context = arm_frames.get("M1")
    if context is not None:
        context_columns = [name for name in context.columns if name.startswith(("index_", "stock_rs5", "market_"))]
        keys = keys.merge(context[["date", "code", *context_columns]], on=list(KEYS), how="left",
                          validate="many_to_one", sort=False)
    if industry is not None:
        if not {"code", "industry_code"}.issubset(industry.columns):
            raise ContractError("industry output context is not bound to exact stock code")
        keys = keys.merge(industry[["code", "industry_code"]], on="code", how="left",
                          validate="many_to_one", sort=False)
    keys["signal_month"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m")
    if "index_ma20_position" in keys:
        position = pd.to_numeric(keys["index_ma20_position"], errors="coerce")
        keys["market_regime_index_ma20"] = np.select(
            [position < 0, position >= 0], ["BELOW", "AT_OR_ABOVE"], default="UNKNOWN"
        )
    else:
        keys["market_regime_index_ma20"] = "UNKNOWN"
    return keys.sort_values(["model_arm", "model_horizon", "fold", "date", "rank"],
                            kind="mergesort").reset_index(drop=True)


def summarize_selected_returns(selected_returns: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize slots and descriptive subgroups while retaining unknown labels."""
    summaries: list[dict[str, Any]] = []
    subgroups: list[dict[str, Any]] = []
    group_columns = {
        "month": "signal_month",
        "market_regime": "market_regime_index_ma20",
        "industry": "industry_code",
    }
    for (arm, model_h, fold, canary), selected in selected_returns.groupby(
            ["model_arm", "model_horizon", "fold", "canary"], sort=True):
        for outcome_h in HORIZONS:
            column = f"actual_close_return_h{outcome_h}"
            values = pd.to_numeric(selected[column], errors="coerce")
            complete_dates = []
            excluded_date_count = 0
            excluded_slot_count = 0
            for date, day_rows in selected.assign(_outcome=values).groupby("date", sort=True):
                day_values = day_rows["_outcome"].dropna().to_numpy(dtype=np.float64)
                if len(day_rows) == 10 and len(day_values) == 10:
                    complete_dates.append(float(day_values.mean()))
                else:
                    excluded_date_count += 1
                    excluded_slot_count += len(day_rows)
            known = values.dropna().to_numpy(dtype=np.float64)
            summaries.append({
                "model_arm": str(arm), "model_horizon": int(model_h), "fold": str(fold),
                "outcome_horizon": int(outcome_h), "canary": bool(canary),
                "selected_slots": int(len(selected)), "known_label_slots": int(values.notna().sum()),
                "unknown_label_slots": int(values.isna().sum()),
                "complete_top10_dates": int(len(complete_dates)),
                "excluded_dates": int(excluded_date_count),
                "excluded_slots": int(excluded_slot_count),
                "complete_top10_mean_return": float(np.mean(complete_dates)) if complete_dates else None,
                "known_selected_event_mean": float(known.mean()) if len(known) else None,
                "known_selected_event_median": float(np.median(known)) if len(known) else None,
                "known_selected_event_p10": float(np.quantile(known, .10, method="linear")) if len(known) else None,
            })
            for group_type, group_column in group_columns.items():
                if group_column not in selected:
                    grouped = [("UNKNOWN", selected)]
                else:
                    grouped = [(str(name) if pd.notna(name) else "UNKNOWN", frame)
                               for name, frame in selected.groupby(group_column, dropna=False, sort=True)]
                for group_value, frame in grouped:
                    group_values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=np.float64)
                    subgroups.append({
                        "model_arm": str(arm), "model_horizon": int(model_h), "fold": str(fold),
                        "outcome_horizon": int(outcome_h), "canary": bool(canary),
                        "group_type": group_type, "group_value": group_value,
                        "selected_slots": int(len(frame)), "known_label_slots": int(len(group_values)),
                        "unknown_label_slots": int(len(frame) - len(group_values)),
                        "known_event_mean": float(group_values.mean()) if len(group_values) else None,
                        "known_event_median": float(np.median(group_values)) if len(group_values) else None,
                        "known_event_p10": float(np.quantile(group_values, .10, method="linear")) if len(group_values) else None,
                    })
    return summaries, subgroups


def paired_stage_acceptance(selected_returns: pd.DataFrame, panel: pd.DataFrame,
                            calendar: pd.DataFrame) -> dict[str, Any]:
    """Evaluate frozen M4-vs-M0 gates on shared complete Top10 dates only."""
    comparisons: dict[int, dict[str, pd.DataFrame]] = {horizon: {} for horizon in HORIZONS}
    for horizon in HORIZONS:
        for fold in fixed_folds():
            current = selected_returns.loc[
                selected_returns["fold"].eq(fold.fold_id) &
                selected_returns["model_horizon"].eq(horizon) &
                selected_returns["model_arm"].isin(["M0", "M4"])
            ]
            m0 = current.loc[current["model_arm"].eq("M0"), ["date", "code", "rank", "prediction"]]
            m4 = current.loc[current["model_arm"].eq("M4"), ["date", "code", "rank", "prediction"]]
            union = pd.concat([m0[["date", "code"]], m4[["date", "code"]]], ignore_index=True).drop_duplicates()
            labels = build_close_return_labels(panel, horizons=(horizon,), calendar=calendar,
                                               signal_keys=union)[horizon]
            paired = paired_common_dates(m0, m4, labels)
            comparisons[horizon][fold.fold_id] = paired
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        result = assess_horizon(comparisons[horizon])
        if result.get("per_fold"):
            bootstrap = paired_within_fold_bootstrap({
                fold_id: frame["delta_M4_minus_M0"].to_numpy(dtype=np.float64)
                for fold_id, frame in comparisons[horizon].items()
                if "delta_M4_minus_M0" in frame
            })
        else:
            bootstrap = {"status": "CANNOT_CONFIRM_INCOMPLETE_PAIRED_FOLDS"}
        output[f"h{horizon}"] = {**result, "paired_10_day_block_bootstrap": bootstrap}
    return output


def write_csv_artifact(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Durably write a CSV selection artifact and return its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return {"path": str(path), "rows": int(len(frame)), "sha256": sha256_file(path)}


def build_formula_comparator_keys(formula_features: pd.DataFrame,
                                  folds: Sequence[Fold]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Freeze formula-trigger comparator lists separately from model candidates."""
    available = [name for name in ("V192_XG", "V193_XG") if name in formula_features]
    rows = formula_features[["date", "code", "FRESH", "CONT0", "RISK0", *available]].copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    selected: list[pd.DataFrame] = []
    status: list[dict[str, Any]] = []
    for fold in folds:
        for column, version in (("V192_XG", "V192"), ("V193_XG", "V193")):
            if column not in rows:
                status.append({"formula_version": version, "fold": fold.fold_id,
                               "status": "CANNOT_CONFIRM_TRIGGER_NOT_BOUND", "selected_slots": 0})
                continue
            period = rows.loc[rows["date"].between(fold.start, fold.end, inclusive="both")]
            ranked = formula_trigger_diagnostic(period, column)
            ranked["formula_version"] = version
            ranked["fold"] = fold.fold_id
            if not ranked.empty:
                selected.append(ranked[["date", "code", "formula_version", "fold", "rank", "selector_score"]])
            status.append({"formula_version": version, "fold": fold.fold_id,
                           "status": "COMPARATOR_READY", "selected_slots": int(len(ranked))})
    columns = ["date", "code", "formula_version", "fold", "rank", "selector_score"]
    return (pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns), status)


def attach_comparator_returns(keys: pd.DataFrame, panel: pd.DataFrame,
                              calendar: pd.DataFrame) -> pd.DataFrame:
    """Join outcomes after separate formula-trigger comparator ranks are fixed."""
    output = keys.copy()
    if output.empty:
        for horizon in HORIZONS:
            output[f"actual_close_return_h{horizon}"] = pd.Series(dtype=np.float64)
            output[f"target_date_h{horizon}"] = pd.Series(dtype="datetime64[ns]")
        return output
    selected_pairs = output[["date", "code"]].drop_duplicates().reset_index(drop=True)
    for horizon, labels in build_close_return_labels(panel, horizons=HORIZONS, calendar=calendar,
                                                     signal_keys=selected_pairs).items():
        outcomes = labels[["date", "code", "target_date", "label_return"]].rename(
            columns={"target_date": f"target_date_h{horizon}", "label_return": f"actual_close_return_h{horizon}"}
        )
        output = output.merge(outcomes, on=list(KEYS), how="left", validate="many_to_one", sort=False)
    return output


def same_date_same_n_metrics(model_returns: pd.DataFrame, comparator_returns: pd.DataFrame) -> list[dict[str, Any]]:
    """Report only equal-size date pairs; never refill sparse formula triggers."""
    output: list[dict[str, Any]] = []
    if comparator_returns.empty:
        return output
    for (version, fold), comparator in comparator_returns.groupby(["formula_version", "fold"], sort=True):
        formula_counts = comparator.groupby("date").size().rename("formula_n")
        for (arm, model_h, model_fold), model in model_returns.groupby(
                ["model_arm", "model_horizon", "fold"], sort=True):
            if str(model_fold) != str(fold):
                continue
            model_counts = model.groupby("date").size().rename("model_n")
            counts = pd.concat([model_counts, formula_counts], axis=1).fillna(0).astype(int)
            comparable_dates = counts.loc[(counts["model_n"] == counts["formula_n"]) & (counts["model_n"] > 0)].index
            for horizon in HORIZONS:
                model_means: list[float] = []
                formula_means: list[float] = []
                complete_dates = 0
                outcome_column = f"actual_close_return_h{horizon}"
                for date in comparable_dates:
                    m = model.loc[model["date"].eq(date)]
                    f = comparator.loc[comparator["date"].eq(date)]
                    mv = pd.to_numeric(m.get(outcome_column), errors="coerce")
                    fv = pd.to_numeric(f.get(outcome_column), errors="coerce")
                    n = int(counts.loc[date, "model_n"])
                    if len(mv) == n and len(fv) == n and mv.notna().all() and fv.notna().all():
                        model_means.append(float(mv.mean()))
                        formula_means.append(float(fv.mean()))
                        complete_dates += 1
                model_mean = float(np.mean(model_means)) if model_means else None
                formula_mean = float(np.mean(formula_means)) if formula_means else None
                output.append({
                    "model_arm": str(arm), "model_horizon": int(model_h), "formula_version": str(version),
                    "fold": str(fold), "outcome_horizon": int(horizon),
                    "same_date_same_n_dates": int(len(comparable_dates)),
                    "complete_same_n_label_dates": int(complete_dates),
                    "model_daily_mean_return": model_mean,
                    "formula_daily_mean_return": formula_mean,
                    "model_minus_formula": model_mean - formula_mean if model_mean is not None and formula_mean is not None else None,
                    "no_refill": True,
                })
    return output


def write_stage_outputs(*, stage: str, attempt: str, results: Sequence[Mapping[str, Any]],
                        panel: pd.DataFrame, calendar: pd.DataFrame,
                        arm_frames: Mapping[str, pd.DataFrame], records: Mapping[str, Any],
                        output_root: Path, contract_sha256: str, binding_sha256: str) -> dict[str, Any]:
    """Persist ranked lists first, then join actual selected-stock outcomes."""
    artifact_root = Path(output_root) / "stage_outputs" / f"{stage}_{attempt}"
    artifact_root.mkdir(parents=True, exist_ok=False)
    keys = freeze_selection_keys(results)
    keys_artifact = write_csv_artifact(artifact_root / "selected_keys_before_labels.csv", keys)
    selected_returns = attach_selected_returns(
        keys, panel, calendar, records["formula_features"], arm_frames, records.get("industry")
    )
    summary, subgroups = summarize_selected_returns(selected_returns)
    artifacts = {
        "selected_keys": keys_artifact,
        "selected_returns": write_csv_artifact(artifact_root / "selected_returns.csv", selected_returns),
        "selection_summary": write_csv_artifact(artifact_root / "selection_summary.csv", pd.DataFrame(summary)),
        "subgroup_summary": write_csv_artifact(artifact_root / "subgroup_summary.csv", pd.DataFrame(subgroups)),
    }
    formula_keys, formula_status = build_formula_comparator_keys(records["formula_features"], fixed_folds())
    formula_keys = formula_keys.loc[formula_keys["fold"].isin(set(keys["fold"]))].reset_index(drop=True)
    artifacts["formula_comparator_keys"] = write_csv_artifact(
        artifact_root / "formula_comparator_keys_before_labels.csv", formula_keys
    )
    formula_returns = attach_comparator_returns(formula_keys, panel, calendar)
    artifacts["formula_comparator_returns"] = write_csv_artifact(
        artifact_root / "formula_comparator_returns.csv", formula_returns
    )
    same_n = same_date_same_n_metrics(selected_returns, formula_returns)
    metrics: dict[str, Any] = {
        "schema_version": "A_SHARE_MODEL_SELECTION_STAGE_METRICS_V1",
        "stage": stage,
        "attempt": attempt,
        "contract_sha256": contract_sha256,
        "input_binding_sha256": binding_sha256,
        "selected_key_sha256": keys_artifact["sha256"],
        "selection_summary_rows": len(summary),
        "subgroup_summary_rows": len(subgroups),
        "formula_comparator_status": formula_status,
        "same_date_same_n": same_n,
        "security_flags": {"research_only": True, "production_allowed": False,
                            "buy_gate_updated": False, "orders_sent": False},
    }
    if stage == "INDEXED_FORMAL":
        metrics["m4_vs_m0_acceptance"] = paired_stage_acceptance(selected_returns, panel, calendar)
    metrics_path = artifact_root / "metrics.json"
    atomic_write_json(metrics_path, metrics)
    artifacts["metrics"] = {"path": str(metrics_path), "sha256": sha256_file(metrics_path)}
    manifest = {
        "schema_version": "A_SHARE_MODEL_SELECTION_OUTPUT_MANIFEST_V1",
        "status": "DIRECTION_ONLY" if not STAGE_SPECS[stage]["canary"] else "ENGINEERING_CANARY_ONLY",
        "stage": stage,
        "attempt": attempt,
        "contract_sha256": contract_sha256,
        "input_binding_sha256": binding_sha256,
        "artifacts": artifacts,
        "fit_calls": sum(bool(item.get("fit_called")) for item in results),
        "security_flags": metrics["security_flags"],
    }
    manifest_path = artifact_root / "output_manifest.json"
    atomic_write_json(manifest_path, manifest)
    selection_slots = int(len(keys))
    known_label_slots = 0
    target_dates: list[str] = []
    for horizon in HORIZONS:
        selected_for_horizon = selected_returns.loc[selected_returns["model_horizon"].eq(horizon)]
        known_label_slots += int(pd.to_numeric(
            selected_for_horizon[f"actual_close_return_h{horizon}"], errors="coerce").notna().sum())
        target_col = f"target_date_h{horizon}"
        if target_col in selected_for_horizon:
            target_dates.extend(pd.to_datetime(selected_for_horizon[target_col], errors="coerce")
                                .dropna().dt.strftime("%Y-%m-%d").tolist())
    return {"output_dir": str(artifact_root), "output_manifest_path": str(manifest_path),
            "artifacts": artifacts, "acceptance": metrics.get("m4_vs_m0_acceptance"),
            "selection_slots": selection_slots, "known_label_slots": known_label_slots,
            "unknown_label_slots": max(0, selection_slots - known_label_slots),
            "signal_dates": pd.to_datetime(keys["date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique().tolist()
            if not keys.empty else [],
            "target_dates": target_dates}


def selection_metrics(top10: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """Return selected-slot and known-label counts without replacing unknowns."""
    joined = top10.merge(labels, on=list(KEYS), how="left", validate="one_to_one", sort=False)
    values = pd.to_numeric(joined["label_return"], errors="coerce")
    per_date: list[dict[str, Any]] = []
    for date, group in joined.assign(_label=values).groupby("date", sort=True):
        known = group["_label"].dropna().to_numpy(dtype=np.float64)
        per_date.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "selected_slots": int(len(group)),
            "known_label_slots": int(len(known)),
            "unknown_label_slots": int(group["_label"].isna().sum()),
            "top10_mean_return": float(known.mean()) if len(known) == len(group) and len(group) else None,
            "complete_top10": bool(len(group) == 10 and len(known) == 10),
            "known_event_returns": known.tolist(),
        })
    all_known = values.dropna().to_numpy(dtype=np.float64)
    return {
        "selected_slots": int(len(joined)),
        "known_label_slots": int(values.notna().sum()),
        "unknown_label_slots": int(values.isna().sum()),
        "event_p10_known_labels": float(np.quantile(all_known, 0.10, method="linear")) if len(all_known) else None,
        "per_date": per_date,
    }


def paired_common_dates(m0: pd.DataFrame, m4: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Pair metrics only when both unfilled Top10 lists have 10 known labels."""
    outputs = []
    event_values: dict[str, dict[pd.Timestamp, list[float]]] = {"M0": {}, "M4": {}}
    for arm, picks in (("M0", m0), ("M4", m4)):
        joined = picks.merge(labels, on=list(KEYS), how="left", validate="one_to_one", sort=False)
        for date, group in joined.groupby("date", sort=True):
            vals = pd.to_numeric(group["label_return"], errors="coerce")
            normalized_date = pd.Timestamp(date).normalize()
            event_values[arm][normalized_date] = vals.dropna().to_numpy(dtype=np.float64).tolist()
            complete = len(group) == 10 and bool(vals.notna().all())
            outputs.append((arm, normalized_date, int(len(group)), int(vals.notna().sum()),
                            float(vals.mean()) if complete else np.nan,
                            0 if complete else int(len(group)), 0 if complete else 1))
    summary = pd.DataFrame(outputs, columns=[
        "arm", "date", "selected_slots", "known_label_slots", "daily_mean_return",
        "excluded_slots", "excluded_dates",
    ])
    pivot = summary.pivot(index="date", columns="arm", values="daily_mean_return")
    common = pivot.dropna(subset=["M0", "M4"]).copy()
    common["delta_M4_minus_M0"] = common["M4"] - common["M0"]
    common = common.reset_index()
    common["M0_event_returns"] = common["date"].map(lambda date: event_values["M0"][pd.Timestamp(date).normalize()])
    common["M4_event_returns"] = common["date"].map(lambda date: event_values["M4"][pd.Timestamp(date).normalize()])
    per_arm_counts = summary.groupby("arm").agg(
        selected_slots=("selected_slots", "sum"),
        known_label_slots=("known_label_slots", "sum"),
        date_count=("date", "nunique"),
        complete_date_count=("daily_mean_return", "count"),
        excluded_dates=("excluded_dates", "sum"),
        excluded_slots=("excluded_slots", "sum"),
    ).to_dict(orient="index")
    for arm, counts in per_arm_counts.items():
        counts["unknown_label_slots"] = int(counts["selected_slots"] - counts["known_label_slots"])
        counts["excluded_dates"] = int(counts["excluded_dates"])
        counts["excluded_slots"] = int(counts["excluded_slots"])
    paired_excluded_dates = pivot.index.difference(common["date"])
    paired_excluded = summary.loc[summary["date"].isin(paired_excluded_dates)]
    paired_excluded_slots_by_arm = {
        arm: int(paired_excluded.loc[paired_excluded["arm"].eq(arm), "selected_slots"].sum())
        for arm in ("M0", "M4")
    }
    common.attrs["per_arm_counts"] = per_arm_counts
    common.attrs["paired_exclusion_counts"] = {
        "candidate_dates": int(len(pivot.index)),
        "paired_complete_dates": int(len(common)),
        "paired_excluded_dates": int(len(paired_excluded_dates)),
        "paired_excluded_slots_by_arm": paired_excluded_slots_by_arm,
        "paired_excluded_slots": int(sum(paired_excluded_slots_by_arm.values())),
    }
    return common


def paired_within_fold_bootstrap(fold_deltas: Mapping[str, Sequence[float]], *,
                                 block_days: int = 10, replicates: int = 1000,
                                 seed: int = 193) -> dict[str, Any]:
    """Auxiliary paired moving-block bootstrap; blocks never cross fold seams."""
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in fold_deltas.items()}
    if set(arrays) != {fold.fold_id for fold in fixed_folds()}:
        return {"status": "CANNOT_CONFIRM_FOLD_COVERAGE"}
    if any(values.ndim != 1 or not np.isfinite(values).all() or len(values) < block_days
           for values in arrays.values()):
        return {"status": "CANNOT_CONFIRM_TOO_FEW_COMPLETE_DATES"}
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        sampled: list[np.ndarray] = []
        for values in arrays.values():
            need = len(values)
            pieces = []
            while sum(len(piece) for piece in pieces) < need:
                start = int(rng.integers(0, len(values) - block_days + 1))
                pieces.append(values[start:start + block_days])
            sampled.append(np.concatenate(pieces)[:need])
        merged = np.concatenate(sampled)
        samples[rep] = float(merged.mean())
    return {
        "status": "AUXILIARY_ONLY",
        "statistic": "mean_paired_daily_top10_return_delta_M4_minus_M0",
        "block_unit": "consecutive_dates_within_each_fold",
        "block_days": block_days,
        "folds": list(arrays),
        "replicates": replicates,
        "seed": seed,
        "ci_95_percentile": [float(x) for x in np.quantile(samples, [0.025, 0.975], method="linear")],
        "used_as_acceptance_gate": False,
    }


def assess_horizon(fold_comparisons: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Evaluate one horizon against the frozen M4-vs-M0 paired gates."""
    expected = [fold.fold_id for fold in fixed_folds()]
    if list(fold_comparisons) != expected:
        return {"status": "CANNOT_CONFIRM_FOLD_COVERAGE", "passed": False}
    per_fold: dict[str, dict[str, Any]] = {}
    deltas: dict[str, Sequence[float]] = {}
    for fold_id, frame in fold_comparisons.items():
        if not {"M0", "M4", "delta_M4_minus_M0"}.issubset(frame.columns) or frame.empty:
            return {"status": f"CANNOT_CONFIRM_INCOMPLETE_{fold_id}", "passed": False}
        per_arm_counts = frame.attrs.get("per_arm_counts", {})
        paired_exclusions = frame.attrs.get("paired_exclusion_counts", {})
        per_fold[fold_id] = {
            "paired_complete_dates": int(len(frame)),
            "m0_mean": float(frame["M0"].mean()),
            "m4_mean": float(frame["M4"].mean()),
            "mean_delta": float(frame["delta_M4_minus_M0"].mean()),
            "partial_fold": fold_id == expected[-1],
            "per_arm_label_and_exclusion_counts": per_arm_counts,
            "paired_exclusion_counts": paired_exclusions,
        }
        deltas[fold_id] = frame["delta_M4_minus_M0"].to_numpy(dtype=np.float64)
    concat = pd.concat(list(fold_comparisons.values()), ignore_index=True)
    mean_delta = float(concat["delta_M4_minus_M0"].mean())
    median_delta = float(concat["M4"].median() - concat["M0"].median())
    events_m0 = np.asarray([value for batch in concat["M0_event_returns"] for value in batch], dtype=np.float64)
    events_m4 = np.asarray([value for batch in concat["M4_event_returns"] for value in batch], dtype=np.float64)
    if not len(events_m0) or not len(events_m4):
        return {"status": "CANNOT_CONFIRM_EMPTY_COMMON_SELECTED_EVENTS", "passed": False}
    p10_m0 = float(np.quantile(events_m0, 0.10, method="linear"))
    p10_m4 = float(np.quantile(events_m4, 0.10, method="linear"))
    positive_folds = sum(value["mean_delta"] > 0.0 for value in per_fold.values())
    gates = {
        "overall_date_equal_mean_delta_gt_zero": mean_delta > 0.0,
        "at_least_4_of_6_fold_mean_deltas_gt_zero": positive_folds >= 4,
        "median_daily_top10_return_not_lower": median_delta >= 0.0,
        "selected_event_p10_not_lower": p10_m4 >= p10_m0,
    }
    return {
        "status": "DIRECTION_ONLY",
        "passed": bool(all(gates.values())),
        "gates": gates,
        "positive_paired_folds": int(positive_folds),
        "overall_mean_daily_delta": mean_delta,
        "median_daily_return_delta": median_delta,
        "m0_selected_event_p10": p10_m0,
        "m4_selected_event_p10": p10_m4,
        "per_fold": per_fold,
        "sealed_forward": False,
        "breakthrough_verified": False,
    }


def formula_trigger_diagnostic(rows: pd.DataFrame, trigger_column: str) -> pd.DataFrame:
    """Separate comparator ranking; it never modifies model candidate pools."""
    required = {"date", "code", trigger_column, "FRESH", "CONT0", "RISK0"}
    if not required.issubset(rows):
        raise ContractError(f"trigger diagnostic missing fields: {sorted(required - set(rows.columns))}")
    active = rows.loc[rows[trigger_column].fillna(False).astype(bool)].copy()
    if active.empty:
        active["rank"] = pd.Series(dtype=np.int64)
        return active
    active["selector_score"] = np.maximum(
        pd.to_numeric(active["FRESH"], errors="coerce").fillna(0),
        pd.to_numeric(active["CONT0"], errors="coerce").fillna(0),
    ) - pd.to_numeric(active["RISK0"], errors="coerce")
    active = active.sort_values(["date", "selector_score", "code"],
                                ascending=[True, False, True], kind="mergesort")
    active["rank"] = active.groupby("date", sort=False).cumcount() + 1
    return active.loc[active["rank"] <= 10].reset_index(drop=True)


def canary_subset(train: pd.DataFrame, test: pd.DataFrame, *, max_codes: int = 200,
                  train_dates: int = 60, test_dates: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic M4 canary slice; feature warmup must precede this slice."""
    codes = sorted(train["code"].astype(str).unique())[:max_codes]
    dates = sorted(pd.to_datetime(train["date"]).dt.normalize().unique())[-train_dates:]
    test_date_values = sorted(pd.to_datetime(test["date"]).dt.normalize().unique())[:test_dates]
    if len(codes) < 1 or len(dates) < train_dates or len(test_date_values) < test_dates:
        raise ContractError("canary lacks 200-or-available codes, 60 train dates, or 10 test dates")
    train_out = train.loc[train["code"].astype(str).isin(codes) &
                          pd.to_datetime(train["date"]).dt.normalize().isin(dates)].copy()
    test_out = test.loc[test["code"].astype(str).isin(codes) &
                        pd.to_datetime(test["date"]).dt.normalize().isin(test_date_values)].copy()
    if train_out.empty or test_out.empty:
        raise ContractError("canary subset is empty")
    return train_out.reset_index(drop=True), test_out.reset_index(drop=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one checkpoint atomically without changing prior successful jobs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "read-panel"))
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--binding", type=Path, default=INPUT_BINDING_PATH)
    parser.add_argument("--activation", type=Path, default=ACTIVATION_PATH)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        contract, binding = load_json(args.contract), load_json(args.binding)
        print(json.dumps({"contract_status": contract.get("status"), "binding_status": binding.get("status"),
                          "real_data_read": False, "fit_started": False}, sort_keys=True))
        return 0
    # This command deliberately fails closed while pre-run review is pending.
    read_bound_panel(contract_path=args.contract, binding_path=args.binding, activation_path=args.activation)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI in preflight checks
    raise SystemExit(main())
