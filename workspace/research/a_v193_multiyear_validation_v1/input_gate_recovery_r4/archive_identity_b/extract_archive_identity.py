"""Bounded, read-only extraction and static audit of the supplied ZIP bundle.

The ZIP path is opened once after a durable no-fit RUN_STARTED. Its bytes are
held in memory for SHA-256 and central-directory access. This code never runs
bundle code and never extracts through filesystem paths from the archive.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[5]
WORK_DIR = Path(__file__).resolve().parent
BATCH_REL = Path("research/a_v193_multiyear_validation_v1/batches/input_gate_recovery_r4_20261018/archive_identity_b")
BATCH = ROOT / BATCH_REL
PRIVATE = BATCH / "private"
CAPTURE_DIR = PRIVATE / "captures"
PUBLIC = BATCH / "public_export"
CONTRACT = PRIVATE / "CONTRACT.json"
BINDING = PRIVATE / "INPUT_BINDING.json"
INTENT = PRIVATE / "RUN_INTENT.json"
START_RECEIPT = PRIVATE / "RUN_STARTED_RECEIPT.json"
ARCHIVE_RECEIPT = PRIVATE / "ARCHIVE_CAPTURE_RECEIPT.json"
MEMBER_INVENTORY = PRIVATE / "MEMBER_INVENTORY.json"
SOURCE_BINDING = PRIVATE / "ARCHIVE_SOURCE_BINDING.json"
MATERIAL_REPORT = PRIVATE / "MATERIAL_ASSESSMENT.json"
DIAGNOSTIC_RECEIPT = PRIVATE / "DIAGNOSTIC_RECEIPT.json"
PREFLIGHT_FAILURE = PRIVATE / "PREFLIGHT_FAILURE.json"
TERMINAL_RECEIPT = PRIVATE / "TERMINAL_RECEIPT.json"
REGISTRY_SYNC_RECEIPT = PRIVATE / "REGISTRY_SYNC_RECEIPT.json"
SAFE_AGGREGATE = PUBLIC / "SAFE_AGGREGATE.json"
EVIDENCE_INDEX = BATCH / "EVIDENCE_INDEX.json"
RUNNER = Path(__file__).resolve()
TEST = RUNNER.with_name("test_archive_identity.py")
REGISTRY = ROOT / "contracts" / "EXPERIMENT_REGISTRY.jsonl"
ADAPTER_DIR = ROOT / "workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity"

ZIP_SOURCE = Path(
    "/Users/lexi/Library/CloudStorage/GoogleDrive-huangwin1@gmail.com/我的云端硬盘/"
    "FMLA04_v1.9.2_原版公式与Mac深化研究_20261015/"
    "FMLA04_V192ORIG_Research_Upgrade_Deliverables_20261016/"
    "192_193_Verification_Optimization_20261016_v2/"
    "code_and_formulas_package_20261016_v2.zip"
)
EXPECTED_ZIP_SIZE = 194062
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024
MAX_TOTAL_DECOMPRESSED_BYTES = 2 * 1024 * 1024
MAX_SINGLE_READ_OVERFLOW_PROBE = 1

TRACK_ID = "A_SHARE_V193_MULTIYEAR_VALIDATION"
ROUTE_ID = "A_V193_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_20261018"
EXPERIMENT_ID = ROUTE_ID
ARM = "R4_BOUNDED_PACKAGE_IDENTITY_AND_SOURCE_RESTORE_NOFIT"
NO_FIT_KIND = "NO_FIT_ENGINEERING_CANARY"
PRIOR_R3_REGISTRY_SHA = "8cff734998138e939beb648e24a1c711e3ae411544790eb1d969aa07d243e6d6"
R3_INDEX_PATH = "research/a_v193_multiyear_validation_v1/batches/materialized_inputs_r2_20261018/targeted_acceptance_r3/EVIDENCE_INDEX.json"
R3_INDEX_SHA = "e7ef90672943d99b00bf9674a5072604fd5109d9f982c94c2f3d03322cc5e887"

PROFILE = {
    "track_id": TRACK_ID,
    "route_id": ROUTE_ID,
    "registration_mode": "LIVE_EXECUTION",
    "start_stage": "V193_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_STARTED",
    "terminal_stage": "V193_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_TERMINAL",
    "occurred_at_reason": "LIVE_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_UTC",
    "baseline_id": "R4_PHASE_A_R3_REUSE_ZIP_PACKAGE_IDENTITY_RECOVERY",
    "one_variable": "LIMITED_READ_ONLY_ZIP_MEMBER_IDENTITY_AND_SOURCE_PROVENANCE_AUDIT",
    "signal_scope": "NO_SIGNAL_SCAN;METADATA_AND_SOURCE_ONLY",
    "target_scope": "NO_TARGET_OR_LABEL_SCAN",
    "device": "CPU_STDLIB_ZIP_CRC_AND_STATIC_SOURCE_ANALYSIS",
    "evidence_level": "ENGINEERING_CANARY_ONLY",
}

ALLOWED_SOURCE_EXTENSIONS = {".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".js", ".ts"}
ALLOWED_FORMULA_EXTENSIONS = {".formula", ".tdx", ".tn6", ".tn8", ".tne"}
ALLOWED_DOC_EXTENSIONS = {".txt", ".md", ".rst"}
ALLOWED_JSON_NAME_TOKENS = {
    "formula", "identity", "audit", "canary", "manifest", "verification",
    "receipt", "provenance", "source", "metadata", "readme", "version",
}
FORBIDDEN_DATA_EXTENSIONS = {
    ".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".npy", ".npz",
    ".pkl", ".pickle", ".pt", ".pth", ".onnx", ".safetensors", ".h5", ".hdf5",
    ".db", ".sqlite", ".zip", ".7z", ".rar", ".tar", ".gz", ".xlsx", ".xls",
}
FORBIDDEN_NAME_TOKENS = {
    "label", "labels", "weight", "weights", "secret", "secrets", "credential",
    "credentials", "password", "passwd", "token", "auth", "privatekey", "apikey",
    "cookie", "position", "order", "account", "holding", "rawdata", "marketdata",
}
SECRET_BYTE_MARKERS = (
    re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
)
S0_TOKENS = ("C", "H", "L", "O", "VOL", "AMOUNT", "INDEXC", "MONTH", "EPPY", "BKEXIST", "ST", "XG")
WINDOW_CALLS = {"REF", "MA", "EMA", "SMA", "HHV", "LLV", "COUNT", "SUM", "EVERY", "EXIST", "FILTER"}


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


def _name_tokens(name: str) -> set[str]:
    return {token.casefold() for token in re.split(r"[^A-Za-z0-9]+", name) if token}


def safe_member_name(name: str) -> tuple[bool, str | None]:
    if not name or "\x00" in name:
        return False, "empty_or_nul_member_name"
    if "\\" in name:
        return False, "backslash_member_name"
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return False, "absolute_or_drive_member_name"
    if any(part in {"..", ".", ""} for part in name.split("/")):
        return False, "dot_or_parent_path_component"
    return True, None


def classify_member(info: zipfile.ZipInfo) -> tuple[str, str | None]:
    """Classify by central-directory metadata only; this does not open content."""

    name = info.orig_filename
    safe, reason = safe_member_name(name)
    if not safe:
        return "rejected", reason
    if info.is_dir():
        return "rejected", "directory_entry"
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        return "rejected", "symbolic_link_member"
    if info.flag_bits & 0x1:
        return "rejected", "encrypted_member"
    if info.file_size < 0 or info.compress_size < 0:
        return "rejected", "negative_central_directory_size"
    tokens = _name_tokens(name)
    if tokens & FORBIDDEN_NAME_TOKENS:
        return "rejected", "forbidden_name_token"
    suffix = PurePosixPath(name).suffix.casefold()
    if suffix in FORBIDDEN_DATA_EXTENSIONS:
        return "rejected", "forbidden_data_or_nested_archive_type"
    if info.file_size > MAX_MEMBER_BYTES:
        return "rejected", "member_size_limit_exceeded"
    if suffix in ALLOWED_SOURCE_EXTENSIONS:
        category = "formula" if "formula" in tokens or "formula" in name.casefold().split("/") else "source_code"
        return category, None
    if suffix in ALLOWED_FORMULA_EXTENSIONS:
        return "formula", None
    if suffix == ".json":
        if tokens & ALLOWED_JSON_NAME_TOKENS:
            return "identity_audit_json", None
        return "rejected", "json_name_not_identity_or_audit_allowlisted"
    if suffix in ALLOWED_DOC_EXTENSIONS:
        return "readme_or_text", None
    return "rejected", "unsupported_member_type"


def _safe_basename(index: int, member_name: str) -> str:
    base = PurePosixPath(member_name).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")[:80] or "member"
    name_sha = sha_bytes(member_name.encode("utf-8", errors="surrogatepass"))[:12]
    return f"member_{index:04d}_{name_sha}_{safe}"


def _secret_signature(data: bytes) -> bool:
    return any(pattern.search(data) is not None for pattern in SECRET_BYTE_MARKERS)


def _candidate_capture_path(capture_dir: Path, filename: str) -> Path:
    resolved_dir = capture_dir.resolve()
    candidate = (resolved_dir / filename).resolve()
    if candidate.parent != resolved_dir:
        raise RuntimeError("safe basename escaped the capture directory")
    return candidate


def process_archive_bytes(archive_bytes: bytes, capture_dir: Path) -> dict[str, Any]:
    """Inventory a ZIP in memory and capture only bounded allowlisted members."""

    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive_byte_limit_exceeded")
    capture_dir.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    decompressed_budget_used = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
        infos = archive.infolist()
        for ordinal, info in enumerate(infos, start=1):
            category, reject_reason = classify_member(info)
            record = {
                "ordinal": ordinal,
                "member_name": info.orig_filename,
                "member_name_sha256": sha_bytes(info.orig_filename.encode("utf-8", errors="surrogatepass")),
                "declared_size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "central_directory_crc32": f"{info.CRC:08x}",
                "compression_method": int(info.compress_type),
                "general_purpose_flags": int(info.flag_bits),
                "external_attributes": int(info.external_attr),
                "category": category,
                "status": "PENDING" if reject_reason is None else "REJECTED",
                "reason": reject_reason,
                "captured_sha256": None,
                "captured_bytes": None,
                "safe_basename": None,
                "crc_verified": False,
            }
            inventory.append(record)
            if reject_reason is not None:
                continue
            read_limit = min(info.file_size + MAX_SINGLE_READ_OVERFLOW_PROBE, MAX_MEMBER_BYTES + 1)
            remaining = MAX_TOTAL_DECOMPRESSED_BYTES - decompressed_budget_used
            if read_limit > remaining:
                record.update(status="REJECTED", reason="total_decompressed_limit_exceeded")
                continue
            try:
                with archive.open(info, "r") as stream:
                    member_bytes = stream.read(read_limit)
            except Exception as exc:
                decompressed_budget_used += read_limit
                record.update(status="REJECTED", reason=f"member_read_error_{type(exc).__name__}")
                continue
            decompressed_budget_used += len(member_bytes)
            actual_crc = zlib.crc32(member_bytes) & 0xFFFFFFFF
            if len(member_bytes) > MAX_MEMBER_BYTES:
                record.update(status="REJECTED", reason="actual_member_size_limit_exceeded")
                continue
            if len(member_bytes) != info.file_size:
                record.update(status="REJECTED", reason="declared_actual_size_mismatch")
                continue
            if actual_crc != info.CRC:
                record.update(status="REJECTED", reason="member_crc_mismatch")
                continue
            if _secret_signature(member_bytes):
                record.update(status="REJECTED", reason="sensitive_content_signature_detected")
                continue
            basename = _safe_basename(ordinal, info.orig_filename)
            target = _candidate_capture_path(capture_dir, basename)
            with target.open("xb") as handle:
                handle.write(member_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            member_sha = sha_bytes(member_bytes)
            record.update(status="CAPTURED", reason=None, captured_sha256=member_sha,
                          captured_bytes=len(member_bytes), safe_basename=basename, crc_verified=True)
            captures.append({
                "ordinal": ordinal,
                "original_member_name": info.orig_filename,
                "category": category,
                "central_directory_crc32": f"{info.CRC:08x}",
                "captured_sha256": member_sha,
                "captured_bytes": len(member_bytes),
                "safe_basename": basename,
                "capture_path": target.relative_to(capture_dir.resolve()).as_posix(),
                "zip_readback_crc_verified": True,
            })
    inventory_sha = sha_bytes(canonical_json_bytes(inventory))
    return {
        "inventory": inventory,
        "captures": captures,
        "inventory_sha256": inventory_sha,
        "member_count": len(inventory),
        "captured_member_count": len(captures),
        "rejected_member_count": sum(record["status"] == "REJECTED" for record in inventory),
        "captured_member_bytes": sum(record["captured_bytes"] or 0 for record in inventory),
        "decompressed_bytes_processed": decompressed_budget_used,
    }


def _walk_json(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            yield item_path, key_text, item
            yield from _walk_json(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            yield item_path, "[]", item
            yield from _walk_json(item, item_path)


def _static_text_summary(member: Mapping[str, Any], data: bytes) -> dict[str, Any]:
    name = str(member["original_member_name"])
    suffix = PurePosixPath(name).suffix.casefold()
    base = {
        "member_name": name,
        "category": member["category"],
        "captured_sha256": member["captured_sha256"],
        "captured_bytes": len(data),
    }
    try:
        text = data.decode("utf-8-sig")
        encoding = "UTF8_OR_UTF8_BOM"
    except UnicodeDecodeError:
        base.update(text_decode="CANNOT_CONFIRM_UTF8", encoding=None)
        return base
    lines = text.splitlines()
    base.update(text_decode="PASS", encoding=encoding, line_count=len(lines))
    if suffix == ".py":
        try:
            tree = ast.parse(text, filename=PurePosixPath(name).name)
            functions = [
                {"name": node.name, "line_start": node.lineno, "line_end": getattr(node, "end_lineno", node.lineno)}
                for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = sorted({node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)})
            exec_calls = sum(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "compile"}
                for node in ast.walk(tree)
            )
            base.update(python_ast="PASS", functions=functions, classes=classes, import_from_modules=imports,
                        executable_dynamic_call_sites=exec_calls, code_executed=False)
        except (SyntaxError, ValueError) as exc:
            base.update(python_ast="CANNOT_CONFIRM", ast_error_type=type(exc).__name__, code_executed=False)
    if member["category"] in {"formula", "source_code", "readme_or_text"}:
        token_records = {}
        for token in S0_TOKENS:
            rx = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", re.IGNORECASE)
            found_lines = [index for index, line in enumerate(lines, start=1) if rx.search(line)]
            if found_lines:
                token_records[token] = {"line_count": len(found_lines), "first_line": min(found_lines), "last_line": max(found_lines)}
        bk_subsignals = sorted({
            token for token in re.findall(r"\bBK[A-Z0-9_]*\b", text.upper()) if token != "BKEXIST"
        })
        window_args = []
        for line_no, line in enumerate(lines, start=1):
            for match in re.finditer(r"\b([A-Z][A-Z0-9_]*)\s*\([^()\r\n]*,\s*(\d{1,5})\s*\)", line.upper()):
                if match.group(1) in WINDOW_CALLS:
                    window_args.append({"line": line_no, "function": match.group(1), "argument": int(match.group(2))})
        base.update(static_token_mentions=token_records, bk_subsignal_names=bk_subsignals,
                    explicit_window_argument_candidates=window_args,
                    formula_text_saved_only_in_private_capture=True)
    if suffix == ".json":
        try:
            value = json.loads(text)
            entries = list(_walk_json(value))
            top_keys = sorted(str(key) for key in value.keys()) if isinstance(value, dict) else []
            hash_claims = []
            assertion_keys = []
            identity_keys = []
            for path, key, item in entries:
                key_low = key.casefold()
                if "all_passed" in key_low or key_low in {"passed", "pass", "status"}:
                    assertion_keys.append({"path": path, "key": key})
                if any(token in key_low for token in ("identity", "formula", "version", "source")):
                    identity_keys.append({"path": path, "key": key, "value_type": type(item).__name__})
                if any(token in key_low for token in ("sha256", "sha_256", "digest", "hash")) and isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                    hash_claims.append({"path": path, "key": key, "claimed_sha256": item.lower()})
            base.update(json_parse="PASS", top_level_keys=top_keys, nested_key_count=len(entries),
                        assertion_fields=assertion_keys, identity_related_fields=identity_keys,
                        embedded_sha256_claims=hash_claims,
                        all_passed_or_status_claims_are_unverified=True)
        except (json.JSONDecodeError, UnicodeError) as exc:
            base.update(json_parse="CANNOT_CONFIRM", json_error_type=type(exc).__name__)
    return base


def build_material_report(captures: Sequence[Mapping[str, Any]], capture_dir: Path,
                          archive_sha: str, archive_size: int) -> dict[str, Any]:
    summaries = []
    for member in captures:
        data = _candidate_capture_path(capture_dir, str(member["safe_basename"])).read_bytes()
        if sha_bytes(data) != member["captured_sha256"]:
            raise RuntimeError("captured member changed after bounded extraction")
        summaries.append(_static_text_summary(member, data))
    captured_hashes = {item["captured_sha256"] for item in captures}
    for summary in summaries:
        for claim in summary.get("embedded_sha256_claims", []):
            claim["matches_any_captured_member_sha256"] = claim["claimed_sha256"] in captured_hashes
    formula_summaries = [item for item in summaries if item.get("category") == "formula"]
    all_text_summaries = [item for item in summaries if item.get("text_decode") == "PASS"]
    aggregate_mentions: dict[str, int] = {token: 0 for token in S0_TOKENS}
    aggregate_bk_names: set[str] = set()
    aggregate_windows = []
    for summary in formula_summaries:
        for token, details in summary.get("static_token_mentions", {}).items():
            aggregate_mentions[token] += int(details["line_count"])
        aggregate_bk_names.update(summary.get("bk_subsignal_names", []))
        aggregate_windows.extend(summary.get("explicit_window_argument_candidates", []))
    package_files = [
        {"member_name": item["member_name"], "category": item["category"],
         "captured_sha256": item["captured_sha256"], "captured_bytes": item["captured_bytes"]}
        for item in captures
    ]
    identity_json = [item for item in summaries if item.get("category") == "identity_audit_json"]
    referenced_all_passed = any(item.get("assertion_fields") for item in identity_json)
    return {
        "schema_version": "V193_R4_ARCHIVE_MATERIAL_ASSESSMENT_V1",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "historical_archive_sha256": None,
        "historical_archive_sha256_null_reason": "No prior ZIP digest was provided; current digest binds only this one local read.",
        "package_content_identity": "BOUND_TO_CURRENT_ZIP_SHA_AND_MEMBER_CRC_SHA; HISTORICAL_ZIP_IDENTITY_NOT_AUTHENTICATED",
        "native_installed_v193_identity": "NOT_PROVEN_BY_DELIVERABLE_PACKAGE_ALONE",
        "native_identity_reason": "This bounded package read can bind source files and embedded claims to the current ZIP bytes, but no live installed-formula hash or native formula-engine readback is included in this scope.",
        "embedded_all_passed_or_status_claim_present": referenced_all_passed,
        "embedded_claims_are_evidence": False,
        "embedded_claim_reason": "An all_passed/status field inside a V192/V193 delivery artifact is a package assertion, not independent native-identity evidence.",
        "member_summaries": summaries,
        "captured_package_files": package_files,
        "identity_json_count": len(identity_json),
        "s0_static_formula_member_count": len(formula_summaries),
        "s0_static_token_mentions_by_formula_line": aggregate_mentions,
        "eppy_present_in_captured_formula_text": aggregate_mentions.get("EPPY", 0) > 0,
        "eppy_admitted_as_s0_dependency": False,
        "bkexist_present_in_captured_formula_text": aggregate_mentions.get("BKEXIST", 0) > 0,
        "bk_subsignal_names_in_captured_formula_text": sorted(aggregate_bk_names),
        "bkexist_interpretation": "TREAT_AS_DERIVED_INTERNAL_BOOLEAN_ONLY; NOT_AN_EXTERNAL_INPUT" if aggregate_mentions.get("BKEXIST", 0) else "CANNOT_CONFIRM_NOT_FOUND_IN_CAPTURED_FORMULA_TEXT",
        "explicit_window_argument_candidates": aggregate_windows,
        "static_source_availability": "REFERENCED_FIELDS_ONLY; ACTUAL_INPUT_AVAILABILITY_AND_PARITY_NOT_CHECKED",
        "pit_status": "CANNOT_CONFIRM_SOURCE_PUBLICATION_REVISION_VINTAGE_AVAILABLE_AT_ADJUSTMENT_OR_ASOF",
        "selector_2022_status": "CANNOT_CONFIRM_NOT_NUMERICALLY_CHECKED",
        "prior_r2_r3_evidence_reuse": {
            "status": "REUSED_NOT_REREAD",
            "r3_evidence_index_path": R3_INDEX_PATH,
            "r3_evidence_index_sha256": R3_INDEX_SHA,
            "panel_identity_and_date_schema_recomputed": False,
        },
        "candidate_reusable_functions": [
            {"member_name": item["member_name"], "function": func["name"],
             "line_start": func["line_start"], "line_end": func["line_end"],
             "code_executed": False}
            for item in summaries for func in item.get("functions", [])
            if any(token in func["name"].casefold() for token in ("formula", "factor", "feature", "build", "select", "score", "filter"))
        ],
        "candidate_code_text_member_count": sum(item.get("category") in {"source_code", "formula"} for item in summaries),
        "decoded_text_member_count": len(all_text_summaries),
        "no_data_or_model_member_content_read": True,
        "no_source_code_executed": True,
        "no_market_rows_labels_weights_or_values_read": True,
        "captured_member_readback_sha256_verified": True,
    }


def synthetic_self_check() -> dict[str, Any]:
    """Exercise path filtering, CRC-checked capture, and the size guard in memory."""

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("docs/README.md", b"synthetic safe text")
        archive.writestr("../escape.py", b"must not be captured")
    with tempfile.TemporaryDirectory() as temporary_directory:
        result = process_archive_bytes(archive_buffer.getvalue(), Path(temporary_directory) / "captures")
    safe_capture = result["captured_member_count"] == 1
    traversal_rejected = any(
        item["member_name"] == "../escape.py" and item["status"] == "REJECTED"
        for item in result["inventory"]
    )
    limit_guard = False
    with tempfile.TemporaryDirectory() as temporary_directory:
        try:
            process_archive_bytes(b"0" * (MAX_ARCHIVE_BYTES + 1), Path(temporary_directory) / "captures")
        except ValueError as exc:
            limit_guard = str(exc) == "archive_byte_limit_exceeded"
    checks = {
        "safe_member_captured": safe_capture,
        "traversal_member_rejected": traversal_rejected,
        "archive_size_limit_rejected": limit_guard,
        "archive_source_path_reads": 0,
    }
    if checks != {
        "safe_member_captured": True,
        "traversal_member_rejected": True,
        "archive_size_limit_rejected": True,
        "archive_source_path_reads": 0,
    }:
        raise AssertionError("synthetic archive controls failed")
    return checks


def verify_identity() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = load_json_object(CONTRACT)
    binding = load_json_object(BINDING)
    contract_sha = sha_file(CONTRACT)
    binding_sha = sha_file(BINDING)
    runner_sha = sha_file(RUNNER)
    test_sha = sha_file(TEST)
    if contract.get("schema_version") != "V193_R4_ARCHIVE_IDENTITY_CONTRACT_V1":
        raise RuntimeError("archive identity contract schema mismatch")
    if contract.get("runner_sha256") != runner_sha or contract.get("test_sha256") != test_sha:
        raise RuntimeError("archive identity runner/test SHA mismatch")
    if contract.get("scope") != "NO_FIT_NO_BACKTEST_NO_PREDICTION_NO_SELECTOR_2022_NUMERIC_OR_PIT_VALUE_READ":
        raise RuntimeError("archive identity scope mismatch")
    if binding.get("schema_version") != "V193_R4_ARCHIVE_IDENTITY_INPUT_BINDING_V1":
        raise RuntimeError("archive identity input binding schema mismatch")
    if binding.get("contract_sha256") != contract_sha:
        raise RuntimeError("archive identity binding contract SHA mismatch")
    if binding.get("runner_sha256") != runner_sha or binding.get("test_sha256") != test_sha:
        raise RuntimeError("archive identity binding code SHA mismatch")
    if binding.get("archive_source_path") != str(ZIP_SOURCE) or binding.get("historical_archive_sha256") is not None:
        raise RuntimeError("archive source path or no-historical-hash binding mismatch")
    if binding.get("expected_archive_size_bytes") != EXPECTED_ZIP_SIZE:
        raise RuntimeError("archive expected size binding mismatch")
    flags = {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False}
    if contract.get("safety_flags") != flags:
        raise RuntimeError("archive identity safety flags mismatch")
    return contract, binding, contract_sha, binding_sha


def load_registry_components():
    sys.path.insert(0, str(ADAPTER_DIR))
    from execution_guard_core import ExecutionClaim, ExecutionGuard, NO_FIT_ENGINEERING_KIND
    from registry_adapter import OfficialRegistryAdapter
    return ExecutionClaim, ExecutionGuard, NO_FIT_ENGINEERING_KIND, OfficialRegistryAdapter


def make_adapter(artifact_paths: Sequence[Path]):
    _, ExecutionGuard, _, OfficialRegistryAdapter = load_registry_components()
    adapter = OfficialRegistryAdapter(
        registry_path=REGISTRY, contract_path=CONTRACT, input_binding_path=BINDING,
        artifact_paths=tuple(artifact_paths), event_profile=PROFILE,
    )
    return adapter, ExecutionGuard(adapter, experiment_id=EXPERIMENT_ID, max_real_fit_debits=0)


def registry_summary(raw: bytes, events: list[dict[str, Any]]) -> dict[str, Any]:
    track = [event for event in events if event.get("track_id") == TRACK_ID]
    starts = {(e.get("route_id"), e.get("execution_id")) for e in track if e.get("event_type") == "RUN_STARTED"}
    terminals = {(e.get("route_id"), e.get("execution_id")) for e in track if e.get("event_type") in {"TERMINAL", "RUN_TERMINATED"}}
    return {
        "registry_sha256": sha_bytes(raw), "registry_bytes": len(raw),
        "global_route_count": len({e.get("route_id") for e in events}),
        "global_event_count": len(events), "global_execution_count": len({e.get("execution_id") for e in events}),
        "global_correction_count": sum(e.get("event_type") == "CORRECTION" for e in events),
        "track_id": TRACK_ID, "track_route_count": len({e.get("route_id") for e in track}),
        "track_event_count": len(track), "track_execution_count": len({e.get("execution_id") for e in track}),
        "track_correction_count": sum(e.get("event_type") == "CORRECTION" for e in track),
        "track_open_execution_count": len(starts - terminals),
        "global_unregistered_route_count": None, "global_contract_only_count": None,
        "global_inventory_status": "CANNOT_CONFIRM_NOT_AUDITED_IN_THIS_SCOPE",
        "full_legacy_schema_status": "NOT_CLAIMED",
    }


def read_validated_registry(adapter):
    with adapter._locked():
        raw, events = adapter._read_validated()
    return raw, events, registry_summary(raw, events)


def verify_preflight(adapter, guard, contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("track_id") != TRACK_ID or contract.get("route_id") != ROUTE_ID:
        raise RuntimeError("archive identity registry binding mismatch")
    _, events, summary = read_validated_registry(adapter)
    if summary["registry_sha256"] != PRIOR_R3_REGISTRY_SHA:
        raise RuntimeError("registry baseline changed before archive identity start")
    if any(e.get("route_id") == ROUTE_ID or e.get("experiment_id") == EXPERIMENT_ID for e in events):
        raise RuntimeError("archive identity route already has registry history")
    if summary["track_open_execution_count"] != 0:
        raise RuntimeError("V193 track has an open execution; archive identity start blocked")
    accounting = guard.accounting()
    if accounting.run_started_count or accounting.terminal_count:
        raise RuntimeError("archive identity guard has prior lifecycle events")
    stat_result = ZIP_SOURCE.lstat()
    if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size != EXPECTED_ZIP_SIZE:
        raise RuntimeError("archive path is not a regular file of the bound expected size")
    return {"registry": summary, "zip_stat_size_bytes": stat_result.st_size,
            "zip_historical_sha256": None, "zip_stat_checked_before_start": True}


def _event_preflight_failure(execution_id: str, error_type: str, contract_sha: str, binding_sha: str) -> dict[str, Any]:
    _, _, _, OfficialRegistryAdapter = load_registry_components()
    failure = {
        "schema_version": "V193_R4_ARCHIVE_IDENTITY_PREFLIGHT_FAILURE_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "error_type": error_type, "event_type": "RUN_TERMINATED",
        "registration_mode": "BACKFILLED_FROM_ARTIFACT", "occurred_at": None,
        "occurred_at_null_reason": "No durable RUN_STARTED occurred; exact failure time is not inferred.",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "recorded_at_is_not_execution_time": True, "source_reads": 0, "run_started": False,
        "fit_calls": 0, "backtests": 0, "predictions": 0, "labels_read": 0, "weights_read": 0,
        "safety_flags": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
    }
    write_json_exclusive(PREFLIGHT_FAILURE, failure)
    artifacts = (RUNNER, TEST, CONTRACT, BINDING, PREFLIGHT_FAILURE)
    adapter = OfficialRegistryAdapter(registry_path=REGISTRY, contract_path=CONTRACT,
                                      input_binding_path=BINDING, artifact_paths=artifacts,
                                      event_profile=PROFILE)
    spec = {
        "schema_version": "1.0", "registry_event_id": f"{execution_id}_RUN_TERMINATED",
        "experiment_id": EXPERIMENT_ID, "arm": ARM, "registration_mode": "BACKFILLED_FROM_ARTIFACT",
        "route_id": ROUTE_ID, "execution_id": execution_id, "track_id": TRACK_ID,
        "stage": "V193_R4_ARCHIVE_PREFLIGHT_FAILURE_BEFORE_START", "event_type": "RUN_TERMINATED",
        "occurred_at": None, "occurred_at_reason": failure["occurred_at_null_reason"],
        "contract_path": str(CONTRACT), "artifact_paths": [str(p) for p in artifacts],
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
        if any(e.get("execution_id") == execution_id for e in events):
            raise RuntimeError("archive identity no-start execution already exists")
        adapter._append_unlocked(raw, events, spec)
    with adapter._locked():
        _, events = adapter._read_validated()
    event = next(e for e in reversed(events) if e.get("execution_id") == execution_id and e.get("event_type") == "RUN_TERMINATED")
    return {"failure": failure, "event": event}


def _capture_archive_once() -> bytes:
    with ZIP_SOURCE.open("rb") as handle:
        return handle.read(MAX_ARCHIVE_BYTES + 1)


def _write_root_sync(status: str, execution_id: str, diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    adapter, _ = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT, TERMINAL_RECEIPT))
    _, events, summary = read_validated_registry(adapter)
    route_events = [e for e in events if e.get("route_id") == ROUTE_ID]
    starts = sum(e.get("event_type") == "RUN_STARTED" for e in route_events)
    terminals = sum(e.get("event_type") in {"TERMINAL", "RUN_TERMINATED"} for e in route_events)
    if starts > 1 or terminals != 1:
        raise RuntimeError("archive identity lifecycle did not close exactly once")
    safe = {
        "schema_version": "V193_R4_ARCHIVE_IDENTITY_SAFE_AGGREGATE_V1",
        "classification": "SAFE_AGGREGATE_NO_ARCHIVE_MEMBER_NAMES_OR_FORMULA_OR_PRIVATE_IDS",
        "status": status,
        "archive_bytes": diagnostic.get("archive_size_bytes"),
        "archive_sha256": diagnostic.get("archive_sha256"),
        "historical_archive_sha256": None,
        "historical_archive_sha256_status": "NOT_PROVIDED",
        "member_count": diagnostic.get("member_count", 0),
        "captured_member_count": diagnostic.get("captured_member_count", 0),
        "rejected_member_count": diagnostic.get("rejected_member_count", 0),
        "captured_member_bytes": diagnostic.get("captured_member_bytes", 0),
        "native_installed_v193_identity": diagnostic.get("native_installed_v193_identity", "CANNOT_CONFIRM"),
        "s0_static_dependency_recovery": diagnostic.get("s0_static_dependency_recovery", "CANNOT_CONFIRM"),
        "pit_status": diagnostic.get("pit_status", "CANNOT_CONFIRM"),
        "selector_2022_status": diagnostic.get("selector_2022_status", "CANNOT_CONFIRM"),
        "prior_r2_r3_reuse": "REUSED_NOT_REREAD",
        "code_executed": False, "market_data_labels_weights_read": False,
        "registry": {
            "global_route_count": summary["global_route_count"], "global_event_count": summary["global_event_count"],
            "global_execution_count": summary["global_execution_count"], "global_correction_count": summary["global_correction_count"],
            "track_route_count": summary["track_route_count"], "track_event_count": summary["track_event_count"],
            "track_execution_count": summary["track_execution_count"], "track_correction_count": summary["track_correction_count"],
            "track_open_execution_count": summary["track_open_execution_count"],
            "global_unregistered_route_count": None, "global_contract_only_count": None,
            "global_inventory_status": summary["global_inventory_status"],
        },
        "safety": {"research_only": True, "production_allowed": False, "buy_gate_updated": False, "orders_sent": False},
    }
    write_json_exclusive(SAFE_AGGREGATE, safe)
    note = (
        "\n\n## 2026-10-18 V193 R4 bounded formula-package identity recovery\n"
        f"- Status: {status}; one bound local ZIP read after durable RUN_STARTED, with a bounded central-directory inventory and selective <=128 KiB/member, <=2 MiB total text/code/identity captures. No bundle code ran, and no market rows, labels, weights, or PIT/selector values were read.\n"
        f"- ZIP size/SHA-256: {diagnostic.get('archive_size_bytes')}/{diagnostic.get('archive_sha256')}; no historical ZIP digest was supplied. Members inventoried/captured/rejected: {diagnostic.get('member_count', 0)}/{diagnostic.get('captured_member_count', 0)}/{diagnostic.get('rejected_member_count', 0)}.\n"
        f"- Native V193 identity: {diagnostic.get('native_installed_v193_identity', 'CANNOT_CONFIRM')}; S0 static references: {diagnostic.get('s0_static_dependency_recovery', 'CANNOT_CONFIRM')}; PIT {diagnostic.get('pit_status', 'CANNOT_CONFIRM')}; selector 2022 {diagnostic.get('selector_2022_status', 'CANNOT_CONFIRM')}. R2/R3 panel/date/schema receipts were reused without panel rereads.\n"
        f"- Registry SHA-256 {summary['registry_sha256']}; global route/event/execution/correction {summary['global_route_count']}/{summary['global_event_count']}/{summary['global_execution_count']}/{summary['global_correction_count']}; V193 route/event/execution/correction/open {summary['track_route_count']}/{summary['track_event_count']}/{summary['track_execution_count']}/{summary['track_correction_count']}/{summary['track_open_execution_count']}. Full global inventory remains CANNOT_CONFIRM. Evidence index: {BATCH_REL.as_posix()}/EVIDENCE_INDEX.json.\n"
        "- Safety: research_only=true, production_allowed=false, buy_gate_updated=false, orders_sent=false.\n"
    )
    for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md"):
        append_fsync(ROOT / name, note)
    root_hashes = {name: sha_file(ROOT / name) for name in ("CURRENT_STATE.md", "HK_OHLCV_RESEARCH_STATE.md", "PROGRESS.md")}
    raw_registry, _, final_summary = read_validated_registry(adapter)
    sync = {
        "schema_version": "V193_R4_ARCHIVE_IDENTITY_REGISTRY_SYNC_V1",
        "route_id": ROUTE_ID, "execution_id": execution_id,
        "route_start_count": starts, "route_terminal_count": terminals,
        "registry": final_summary, "contract_sha256": sha_file(CONTRACT),
        "input_binding_sha256": sha_file(BINDING), "runner_sha256": sha_file(RUNNER),
        "test_sha256": sha_file(TEST), "archive_source_binding_sha256": sha_file(SOURCE_BINDING),
        "root_file_sha256_after_append": root_hashes,
        "registry_readback_matches_summary": sha_bytes(raw_registry) == final_summary["registry_sha256"],
        "root_append_fsync": True,
    }
    write_json_exclusive(REGISTRY_SYNC_RECEIPT, sync)
    artifact_paths = (CONTRACT, BINDING, INTENT, START_RECEIPT, ARCHIVE_RECEIPT,
                      MEMBER_INVENTORY, SOURCE_BINDING, MATERIAL_REPORT, DIAGNOSTIC_RECEIPT,
                      TERMINAL_RECEIPT, REGISTRY_SYNC_RECEIPT, SAFE_AGGREGATE, RUNNER, TEST)
    artifacts = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)}
                 for p in artifact_paths if p.is_file()]
    index = {
        "schema_version": "V193_R4_ARCHIVE_IDENTITY_EVIDENCE_INDEX_V1",
        "classification": "NAVIGATION_NOT_AUTHORITY",
        "batch": BATCH_REL.as_posix(), "status": status,
        "route_id": ROUTE_ID, "execution_id": execution_id,
        "source_archive_content_in_index": False,
        "artifacts": artifacts,
        "registry_summary": final_summary,
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
    outputs = (INTENT, START_RECEIPT, ARCHIVE_RECEIPT, MEMBER_INVENTORY, SOURCE_BINDING,
               MATERIAL_REPORT, DIAGNOSTIC_RECEIPT, PREFLIGHT_FAILURE, TERMINAL_RECEIPT,
               REGISTRY_SYNC_RECEIPT, SAFE_AGGREGATE, EVIDENCE_INDEX)
    if any(path.exists() for path in outputs):
        raise FileExistsError("archive identity output exists; route re-entry blocked")
    contract_sha = binding_sha = None
    try:
        contract, binding, contract_sha, binding_sha = verify_identity()
        adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING))
        preflight = verify_preflight(adapter, guard, contract)
    except Exception as exc:
        error_type = type(exc).__name__
        try:
            if contract_sha is None and CONTRACT.is_file(): contract_sha = sha_file(CONTRACT)
            if binding_sha is None and BINDING.is_file(): binding_sha = sha_file(BINDING)
            if contract_sha is None or binding_sha is None:
                raise RuntimeError("contract/binding unavailable for preflight backfill")
            backfill = _event_preflight_failure(execution_id, error_type, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {"schema_version": "V193_R4_ARCHIVE_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False, "registry_event": backfill["event"],
                "preflight_failure": backfill["failure"], "archive_bytes_read": 0})
            fail_diag = {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "archive_size_bytes": None,
                         "archive_sha256": None, "member_count": 0, "captured_member_count": 0,
                         "rejected_member_count": 0, "captured_member_bytes": 0,
                         "native_installed_v193_identity": "CANNOT_CONFIRM", "s0_static_dependency_recovery": "CANNOT_CONFIRM",
                         "pit_status": "CANNOT_CONFIRM", "selector_2022_status": "CANNOT_CONFIRM"}
            sync = _write_root_sync("CANNOT_CONFIRM_PREFLIGHT_FAILURE", execution_id, fail_diag)
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry": sync["registry"], "evidence_index_sha256": sync["evidence_index_sha256"]}
        except Exception as backfill_exc:
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_AND_BACKFILL", "execution_id": execution_id,
                    "preflight_error_type": error_type, "backfill_error_type": type(backfill_exc).__name__,
                    "registry_event_written": False, "archive_bytes_read": 0}

    write_json_exclusive(INTENT, {
        "schema_version": "V193_R4_ARCHIVE_IDENTITY_RUN_INTENT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "preflight": preflight,
        "archive_raw_reads_planned": 1,
        "central_directory_and_member_reads_start_only_after_durable_RUN_STARTED": True,
        "maximum_archive_bytes": MAX_ARCHIVE_BYTES,
        "maximum_member_bytes": MAX_MEMBER_BYTES,
        "maximum_total_decompressed_bytes": MAX_TOTAL_DECOMPRESSED_BYTES,
        "bundle_code_execution": False,
        "market_data_label_weight_or_PIT_value_reads": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT))
    try:
        claim = guard.claim_no_fit(execution_id=execution_id, arm=ARM,
                                   contract_sha256=contract_sha, input_binding_sha256=binding_sha)
        start_error = None
    except Exception as exc:
        starts = [e for e in adapter.list_events(EXPERIMENT_ID) if e.get("execution_id") == execution_id and e.get("event_type") == "RUN_STARTED"]
        if starts:
            ExecutionClaim, _, nofit, _ = load_registry_components()
            claim = ExecutionClaim(EXPERIMENT_ID, execution_id, ARM, contract_sha, binding_sha, nofit, 0)
            start_error = type(exc).__name__
        else:
            error_type = type(exc).__name__
            backfill = _event_preflight_failure(execution_id, error_type, contract_sha, binding_sha)
            write_json_exclusive(TERMINAL_RECEIPT, {"schema_version": "V193_R4_ARCHIVE_TERMINAL_V1",
                "status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                "route_id": ROUTE_ID, "run_started": False, "registry_event": backfill["event"],
                "preflight_failure": backfill["failure"], "archive_bytes_read": 0})
            return {"status": "CANNOT_CONFIRM_PREFLIGHT_FAILURE", "execution_id": execution_id,
                    "registry_event_written": True, "archive_bytes_read": 0}
    starts = [e for e in adapter.list_events(EXPERIMENT_ID) if e.get("execution_id") == execution_id and e.get("event_type") == "RUN_STARTED"]
    start_ok = len(starts) == 1
    write_json_exclusive(START_RECEIPT, {
        "schema_version": "V193_R4_ARCHIVE_RUN_STARTED_RECEIPT_V1",
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "durable_start_readback": start_ok, "registry_event": starts[0] if start_ok else None,
        "archive_raw_reads_before_start": 0,
    })
    diagnostic: dict[str, Any]
    completed = False
    try:
        if not start_ok or start_error:
            raise RuntimeError(start_error or "durable_start_readback_count_mismatch")
        # Exactly one archive-path open/read; all ZIP/member access uses this in-memory buffer.
        archive_bytes = _capture_archive_once()
        archive_size = len(archive_bytes)
        archive_sha = sha_bytes(archive_bytes)
        write_json_exclusive(ARCHIVE_RECEIPT, {
            "schema_version": "V193_R4_ARCHIVE_CAPTURE_RECEIPT_V1",
            "execution_id": execution_id, "route_id": ROUTE_ID,
            "archive_bytes_read": archive_size, "archive_sha256": archive_sha,
            "historical_archive_sha256": None,
            "historical_archive_sha256_null_reason": "No historical SHA was supplied.",
            "archive_source_open_count": 1,
            "archive_byte_read_call_count": 1,
            "archive_bytes_copied": False,
            "expected_size_bytes": EXPECTED_ZIP_SIZE,
            "size_matches_frozen_expected": archive_size == EXPECTED_ZIP_SIZE,
            "zip_central_directory_read_after_start": True,
        })
        if archive_size > MAX_ARCHIVE_BYTES:
            raise ValueError("archive_byte_limit_exceeded")
        if archive_size != EXPECTED_ZIP_SIZE:
            raise ValueError("archive_size_mismatch")
        processed = process_archive_bytes(archive_bytes, CAPTURE_DIR)
        write_json_exclusive(MEMBER_INVENTORY, {
            "schema_version": "V193_R4_ARCHIVE_MEMBER_INVENTORY_V1",
            "execution_id": execution_id, "archive_sha256": archive_sha,
            "archive_size_bytes": archive_size,
            "inventory_sha256": processed["inventory_sha256"],
            "member_count": processed["member_count"],
            "captured_member_count": processed["captured_member_count"],
            "rejected_member_count": processed["rejected_member_count"],
            "captured_member_bytes": processed["captured_member_bytes"],
            "decompressed_bytes_processed": processed["decompressed_bytes_processed"],
            "maximum_member_bytes": MAX_MEMBER_BYTES,
            "maximum_total_decompressed_bytes": MAX_TOTAL_DECOMPRESSED_BYTES,
            "members": processed["inventory"],
        })
        source_binding = {
            "schema_version": "V193_R4_ARCHIVE_SOURCE_BINDING_V1",
            "execution_id": execution_id, "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha,
            "archive_source_path": str(ZIP_SOURCE),
            "archive_size_bytes": archive_size,
            "archive_raw_sha256": archive_sha,
            "historical_archive_sha256": None,
            "historical_archive_sha256_null_reason": "No prior ZIP SHA was supplied; the current raw SHA binds this one read only.",
            "archive_bytes_read_once_after_run_started": True,
            "member_inventory_sha256": processed["inventory_sha256"],
            "capture_directory_relative": CAPTURE_DIR.relative_to(ROOT).as_posix(),
            "members": processed["captures"],
            "captured_member_count": processed["captured_member_count"],
            "captured_member_bytes": processed["captured_member_bytes"],
        }
        write_json_exclusive(SOURCE_BINDING, source_binding)
        report = build_material_report(processed["captures"], CAPTURE_DIR, archive_sha, archive_size)
        report.update({
            "execution_id": execution_id,
            "contract_sha256": contract_sha,
            "input_binding_sha256": binding_sha,
            "archive_source_binding_sha256": sha_file(SOURCE_BINDING),
            "member_inventory_sha256": processed["inventory_sha256"],
            "member_count": processed["member_count"],
            "captured_member_count": processed["captured_member_count"],
            "rejected_member_count": processed["rejected_member_count"],
            "captured_member_bytes": processed["captured_member_bytes"],
            "decompressed_bytes_processed": processed["decompressed_bytes_processed"],
        })
        # The raw current ZIP hash authenticates package bytes only. No old SHA exists.
        if processed["member_count"] > 0 and processed["captured_member_count"] > 0:
            status = "PACKAGE_MEMBERS_BOUNDED_AND_BOUND_NATIVE_IDENTITY_PIT_UNCONFIRMED"
        else:
            status = "CANNOT_CONFIRM_NO_ALLOWLISTED_MEMBERS_CAPTURED"
        report["status"] = status
        report["s0_static_dependency_recovery"] = (
            "STATIC_FORMULA_REFERENCES_RECOVERED; ACTUAL_INPUT_PARITY_NOT_CHECKED"
            if report["s0_static_formula_member_count"] > 0 else "CANNOT_CONFIRM_NO_FORMULA_MEMBER_CAPTURED"
        )
        write_json_exclusive(MATERIAL_REPORT, report)
        diagnostic = {
            "schema_version": "V193_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_V1",
            "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
            "status": status, "archive_size_bytes": archive_size, "archive_sha256": archive_sha,
            "historical_archive_sha256": None,
            "archive_capture_receipt_path": ARCHIVE_RECEIPT.relative_to(ROOT).as_posix(),
            "archive_capture_receipt_sha256": sha_file(ARCHIVE_RECEIPT),
            "member_inventory_path": MEMBER_INVENTORY.relative_to(ROOT).as_posix(),
            "member_inventory_sha256": sha_file(MEMBER_INVENTORY),
            "archive_source_binding_path": SOURCE_BINDING.relative_to(ROOT).as_posix(),
            "archive_source_binding_sha256": sha_file(SOURCE_BINDING),
            "material_report_path": MATERIAL_REPORT.relative_to(ROOT).as_posix(),
            "material_report_sha256": sha_file(MATERIAL_REPORT),
            "member_count": processed["member_count"],
            "captured_member_count": processed["captured_member_count"],
            "rejected_member_count": processed["rejected_member_count"],
            "captured_member_bytes": processed["captured_member_bytes"],
            "native_installed_v193_identity": report["native_installed_v193_identity"],
            "s0_static_dependency_recovery": report["s0_static_dependency_recovery"],
            "pit_status": report["pit_status"],
            "selector_2022_status": report["selector_2022_status"],
            "code_executed": False, "market_rows_read": 0, "labels_read": 0,
            "weights_read": 0, "fit_calls": 0, "backtest_calls": 0,
            "prediction_calls": 0, "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
        }
        completed = True
    except Exception as exc:
        error_type = type(exc).__name__
        diagnostic = {
            "schema_version": "V193_R4_ARCHIVE_IDENTITY_DIAGNOSTIC_V1",
            "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
            "status": "CANNOT_CONFIRM", "error_type": error_type,
            "archive_size_bytes": None, "archive_sha256": None,
            "member_count": 0, "captured_member_count": 0,
            "rejected_member_count": 0, "captured_member_bytes": 0,
            "native_installed_v193_identity": "CANNOT_CONFIRM",
            "s0_static_dependency_recovery": "CANNOT_CONFIRM",
            "pit_status": "CANNOT_CONFIRM", "selector_2022_status": "CANNOT_CONFIRM",
            "archive_source_open_count": 1, "code_executed": False,
            "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
            "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
        }
        completed = False
    write_json_exclusive(DIAGNOSTIC_RECEIPT, diagnostic)
    adapter, guard = make_adapter((RUNNER, TEST, CONTRACT, BINDING, INTENT, START_RECEIPT,
                                   ARCHIVE_RECEIPT, MEMBER_INVENTORY, SOURCE_BINDING,
                                   MATERIAL_REPORT, DIAGNOSTIC_RECEIPT))
    guard.append_no_fit_terminal(claim, succeeded=True, error_type=None if completed else error_type)
    terminals = [e for e in adapter.list_events(EXPERIMENT_ID) if e.get("execution_id") == execution_id and e.get("event_type") == "TERMINAL"]
    if len(terminals) != 1:
        raise RuntimeError("archive identity terminal registry readback was not exactly one")
    final_status = diagnostic.get("status", "CANNOT_CONFIRM") if completed else "CANNOT_CONFIRM"
    terminal = {
        "schema_version": "V193_R4_ARCHIVE_TERMINAL_V1", "status": final_status,
        "execution_id": execution_id, "route_id": ROUTE_ID, "track_id": TRACK_ID,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "run_started": True, "terminal_event": terminals[0],
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT),
        "archive_source_binding_sha256": sha_file(SOURCE_BINDING) if SOURCE_BINDING.is_file() else None,
        "market_rows_read": 0, "labels_read": 0, "weights_read": 0,
        "fit_calls": 0, "backtest_calls": 0, "prediction_calls": 0,
        "selector_2022_numeric_checks": 0, "pit_numeric_checks": 0,
    }
    write_json_exclusive(TERMINAL_RECEIPT, terminal)
    sync = _write_root_sync(final_status, execution_id, diagnostic)
    return {
        "status": final_status, "execution_id": execution_id,
        "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
        "runner_sha256": sha_file(RUNNER), "test_sha256": sha_file(TEST),
        "diagnostic_receipt_sha256": sha_file(DIAGNOSTIC_RECEIPT),
        "terminal_receipt_sha256": sha_file(TERMINAL_RECEIPT),
        "registry": sync["registry"],
        "registry_sync_receipt_sha256": sync["registry_sync_receipt_sha256"],
        "safe_aggregate_sha256": sync["safe_aggregate_sha256"],
        "evidence_index_sha256": sync["evidence_index_sha256"],
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
