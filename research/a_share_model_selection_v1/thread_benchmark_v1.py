"""One-fit, supervisor-controlled M0/H1 12-thread engineering benchmark.

This tool is deliberately bound to one frozen completed M0/H1/2026H1 cell.
It does not evaluate returns, select a new research route, or alter the frozen
training implementation. A supervisor process owns the official registry and
only releases the one fit after a worker has reconstructed and verified every
input identity against the completed four-thread checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits


TOOL_SCHEMA = "A_SHARE_M0_THREAD_BENCHMARK_V1"
EXPECTED_MODEL_SHA256 = "0aa1428f2b62447e35ecbebce4843613dd01111cfb4340f38221e022eecbd40c"
EXPECTED_ROUTE_SHA256 = "d11d65942d6c5a3eb7edc02a01b4b8e312304c167824ea6b0dd83f85a54022bb"
TARGET_CELL = "M0_FORMAL_M0_H1_2026H1_attempt_01"
TARGET_FOLD = "2026H1"
SOURCE_FOLDS = ("2024H1", "2024H2", "2025H1", "2025H2")
FIT_THREADS = 12
BASELINE_THREADS = 4
MIN_AVAILABLE_BYTES = 8 * 1024**3
MAX_WORKER_RSS_BYTES = 12 * 1024**3
MAX_SWAP_INCREASE_BYTES = 256 * 1024**2
SAMPLE_SECONDS = 1.0
BENCHMARK_EXPERIMENT_ID = "A_SHARE_M0_THREAD_BENCHMARK_R1_20261019"
BENCHMARK_ROUTE_ID = "a-share-m0-thread-benchmark-r1-20261019"


class BenchmarkError(RuntimeError):
    """Frozen identity, input, resource, or comparison gate failed."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"expected JSON object: {path}")
    return value


def config_identity(contract: Mapping[str, Any]) -> str:
    without_self = dict(contract)
    without_self.pop("config_sha256", None)
    return canonical_sha256(without_self)


def resolve_under(root: Path, relative: str) -> Path:
    path = (Path(root) / relative).resolve()
    resolved_root = Path(root).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise BenchmarkError(f"contract path escapes project root: {relative}")
    return path


def verify_pinned_hash(path: Path, expected: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise BenchmarkError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def load_frozen_route(project_root: Path, contract: Mapping[str, Any]):
    route_path = Path(project_root) / "workspace/research/a_share_model_selection_v1/model_selection.py"
    if (contract.get("frozen_route", {}).get("model_selection_sha256") != EXPECTED_MODEL_SHA256 or
            contract.get("frozen_route", {}).get("route_code_sha256") != EXPECTED_ROUTE_SHA256):
        raise BenchmarkError("benchmark contract frozen-route pins differ from the writer pins")
    verify_pinned_hash(route_path, EXPECTED_MODEL_SHA256, "frozen model_selection.py")
    frozen_route_dir = str(route_path.resolve().parent)
    if frozen_route_dir not in sys.path:
        sys.path.insert(0, frozen_route_dir)
    spec = importlib.util.spec_from_file_location("pv_benchmark_frozen_model_selection", route_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError("could not load the pinned frozen model-selection route")
    route = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = route
    spec.loader.exec_module(route)
    route_sha = route.route_code_sha256()
    if route_sha != EXPECTED_ROUTE_SHA256:
        raise BenchmarkError(f"frozen route composite SHA mismatch: {route_sha}")
    code_identity = route.code_identity()
    expected_code = contract["frozen_route"]["code_sha256"]
    if code_identity != expected_code:
        raise BenchmarkError("frozen route component SHA map differs from benchmark contract")
    return route


def verify_identity(expected: Mapping[str, Any], actual: Mapping[str, Any], *, name: str) -> None:
    if dict(expected) != dict(actual):
        raise BenchmarkError(f"{name} identity mismatch")


def rank_all(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "code", "prediction"}
    if not required.issubset(predictions.columns):
        raise BenchmarkError("prediction frame lacks date/code/prediction")
    frame = predictions.loc[:, ["date", "code", "prediction"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if frame["date"].isna().any() or frame["code"].isna().any():
        raise BenchmarkError("prediction frame contains invalid keys")
    score = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all() or frame.duplicated(["date", "code"]).any():
        raise BenchmarkError("prediction frame contains non-finite scores or duplicate keys")
    frame = frame.sort_values(["date", "prediction", "code"],
                              ascending=[True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby("date", sort=False).cumcount() + 1
    return frame.reset_index(drop=True)


def ranked_key_identity(ranked: pd.DataFrame) -> str:
    rows = [
        [date.strftime("%Y-%m-%d"), str(code), int(rank)]
        for date, code, rank in zip(ranked["date"], ranked["code"], ranked["rank"])
    ]
    return canonical_sha256({"schema_version": "A_SHARE_DAILY_FULL_RANK_KEYS_V1", "rows": rows})


def top10_key_identity(ranked: pd.DataFrame) -> str:
    selected = ranked.loc[ranked["rank"] <= 10, ["date", "code", "rank"]]
    rows = [
        [date.strftime("%Y-%m-%d"), str(code), int(rank)]
        for date, code, rank in zip(selected["date"], selected["code"], selected["rank"])
    ]
    return canonical_sha256({"schema_version": "A_SHARE_DAILY_TOP10_KEYS_V1", "rows": rows})


def compare_prediction_frames(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    keys = ["date", "code"]
    required = set(keys + ["prediction"])
    if not required.issubset(candidate.columns) or not required.issubset(reference.columns):
        raise BenchmarkError("candidate/reference predictions lack keys or scores")
    left = candidate.loc[:, keys + ["prediction"]].copy()
    right = reference.loc[:, keys + ["prediction"]].copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    left["code"] = left["code"].astype(str)
    right["code"] = right["code"].astype(str)
    if (left["date"].isna().any() or right["date"].isna().any() or
            left.duplicated(keys).any() or right.duplicated(keys).any()):
        raise BenchmarkError("candidate/reference prediction keys are invalid or duplicated")
    same_key_order = left[keys].equals(right[keys])
    left_score = pd.to_numeric(left["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    right_score = pd.to_numeric(right["prediction"], errors="coerce").to_numpy(dtype=np.float64)
    if len(left_score) != len(right_score):
        raise BenchmarkError("candidate/reference prediction counts differ")
    finite = np.isfinite(left_score).all() and np.isfinite(right_score).all()
    max_abs_diff = float(np.max(np.abs(left_score - right_score))) if finite and len(left_score) else None
    score_exact = bool(finite and np.array_equal(left_score, right_score))
    left_rank = rank_all(left)
    right_rank = rank_all(right)
    rank_order_exact = left_rank[["date", "code", "rank"]].equals(right_rank[["date", "code", "rank"]])
    top10_exact = (left_rank.loc[left_rank["rank"] <= 10, ["date", "code", "rank"]]
                   .reset_index(drop=True)
                   .equals(right_rank.loc[right_rank["rank"] <= 10, ["date", "code", "rank"]]
                           .reset_index(drop=True)))
    return {
        "row_count": int(len(left)),
        "key_order_exact": bool(same_key_order),
        "prediction_array_equal": score_exact and same_key_order,
        "max_abs_diff": max_abs_diff,
        "full_daily_rank_order_exact": bool(rank_order_exact and same_key_order),
        "candidate_ranked_key_sha256": ranked_key_identity(left_rank),
        "reference_ranked_key_sha256": ranked_key_identity(right_rank),
        "top10_keys_exact": bool(top10_exact and same_key_order),
        "candidate_top10_key_sha256": top10_key_identity(left_rank),
        "reference_top10_key_sha256": top10_key_identity(right_rank),
    }


def parse_vm_stat(text: str) -> tuple[int, dict[str, int]]:
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", text, flags=re.I)
    if not page_match:
        raise BenchmarkError("vm_stat output has no page size")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for raw_line in text.splitlines():
        match = re.match(r"\s*([^:]+):\s*([\d.]+)", raw_line)
        if match:
            label = match.group(1).strip().lower().replace(" ", "_")
            pages[label] = int(float(match.group(2)))
    return page_size, pages


def parse_swap_used_bytes(text: str) -> int:
    match = re.search(r"used\s*=\s*([\d.]+)\s*([KMGTP])", text, flags=re.I)
    if not match:
        raise BenchmarkError("swapusage output has no used quantity")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def parse_memory_pressure(value: str) -> str:
    raw = value.strip()
    if raw == "0":
        return "normal"
    if raw.isdigit():
        return "warning_or_critical"
    raise BenchmarkError(f"unrecognized vm.memory_pressure value: {raw!r}")


def parse_rss_kilobytes(value: str) -> int | None:
    match = re.search(r"\b(\d+)\b", value.strip())
    return int(match.group(1)) * 1024 if match else None


def _read_command(args: Sequence[str]) -> str:
    result = subprocess.run(list(args), check=True, capture_output=True, text=True)
    return result.stdout.strip()


def probe_host_resources(worker_pid: int | None = None) -> dict[str, Any]:
    if sys.platform != "darwin":
        raise BenchmarkError("resource gate is supported only on macOS")
    vm_text = _read_command(["/usr/bin/vm_stat"])
    page_size, pages = parse_vm_stat(vm_text)
    free_pages = pages.get("pages_free", 0)
    available_bytes = free_pages * page_size
    pressure = parse_memory_pressure(_read_command(["/usr/sbin/sysctl", "-n", "vm.memory_pressure"]))
    swap = parse_swap_used_bytes(_read_command(["/usr/sbin/sysctl", "vm.swapusage"]))
    rss = None
    if worker_pid is not None:
        rss_text = _read_command(["/bin/ps", "-o", "rss=", "-p", str(int(worker_pid))])
        rss = parse_rss_kilobytes(rss_text)
    return {
        "pressure_state": pressure,
        "vm_memory_pressure_raw": _read_command(["/usr/sbin/sysctl", "-n", "vm.memory_pressure"]),
        "available_bytes_estimate": int(available_bytes),
        "available_memory_formula": "pages_free times vm_stat page size; caches are not assumed reclaimable",
        "swap_used_bytes": int(swap),
        "worker_rss_bytes": rss,
        "sampled_at_epoch_seconds": time.time(),
    }


def resource_abort_reason(snapshot: Mapping[str, Any], *, initial_swap_used_bytes: int,
                          during_run: bool) -> str | None:
    if snapshot.get("pressure_state") != "normal":
        return "MEMORY_PRESSURE_NOT_NORMAL"
    rss = snapshot.get("worker_rss_bytes")
    if rss is not None and int(rss) > MAX_WORKER_RSS_BYTES:
        return "WORKER_RSS_EXCEEDED_12_GIB"
    if (during_run and int(snapshot.get("swap_used_bytes", 0)) - int(initial_swap_used_bytes)
            >= MAX_SWAP_INCREASE_BYTES):
        return "SWAP_INCREASE_REACHED_256_MIB"
    return None


def terminate_only_worker(process: subprocess.Popen, *, grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _write_joblib_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(dict(payload), tmp_name)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return sha256_file(path)


def _bound_path(project_root: Path, entry: Mapping[str, Any]) -> Path:
    return resolve_under(project_root, str(entry["path"]))


def _load_checkpoint_fixture(route: Any, project_root: Path, artifact: Mapping[str, Any],
                             expected_code_shas: Mapping[str, str], expected_contract_sha: str,
                             expected_input_components: Mapping[str, str],
                             *, feature_buffer: np.ndarray | None = None,
                             date_buffer: np.ndarray | None = None,
                             code_buffer: np.ndarray | None = None,
                             offset: int = 0,
                             fold: Any = None) -> dict[str, Any]:
    checkpoint_path = _bound_path(project_root, artifact["checkpoint"])
    metadata_path = _bound_path(project_root, artifact["metadata"])
    replay_path = _bound_path(project_root, artifact["replay"])
    verify_pinned_hash(checkpoint_path, str(artifact["checkpoint_sha256"]), "checkpoint")
    verify_pinned_hash(metadata_path, str(artifact["metadata"]["sha256"]), "checkpoint metadata")
    verify_pinned_hash(replay_path, str(artifact["replay"]["sha256"]), "checkpoint replay receipt")
    metadata = load_json(metadata_path)
    replay = load_json(replay_path)
    if metadata.get("status") != "COMPLETED" or metadata.get("artifact_sha256") != artifact["checkpoint_sha256"]:
        raise BenchmarkError("checkpoint metadata is incomplete or its artifact SHA differs")
    identity = metadata.get("identity", {})
    if identity.get("code_sha256") != dict(expected_code_shas):
        raise BenchmarkError("checkpoint frozen code identity mismatch")
    if identity.get("contract_sha256") != expected_contract_sha:
        raise BenchmarkError("checkpoint frozen contract identity mismatch")
    verify_identity(expected_input_components, identity.get("input_components", {}),
                    name="checkpoint bound input components")
    if replay.get("status") != "VERIFIED":
        raise BenchmarkError("checkpoint lacks an exact saved-model replay receipt")
    if replay.get("checkpoint_sha256") != artifact["checkpoint_sha256"]:
        raise BenchmarkError("saved-model replay receipt names a different checkpoint")
    if replay.get("prediction_identity") != replay.get("replay_prediction_identity"):
        raise BenchmarkError("checkpoint replay predictions are not exactly identical")
    if artifact.get("fold") and (
            identity.get("stage") != "M0_FORMAL" or identity.get("fold") != artifact["fold"] or
            identity.get("arm") != "M0" or identity.get("horizon") != 1):
        raise BenchmarkError("checkpoint is not the contract-bound M0/H1 fold fixture")
    payload = joblib.load(checkpoint_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("replay_fixture"), dict):
        raise BenchmarkError("checkpoint lacks a saved replay fixture")
    fixture = payload["replay_fixture"]
    columns = tuple(str(name) for name in payload.get("feature_columns", ()))
    if columns != tuple(identity.get("data_identity", {}).get("feature_columns", ())):
        raise BenchmarkError("checkpoint feature order differs from its metadata")
    keys = fixture.get("test_keys")
    values = np.asarray(fixture.get("test_features"), dtype=np.float64)
    if not isinstance(keys, pd.DataFrame) or values.ndim != 2 or len(keys) != len(values):
        raise BenchmarkError("checkpoint replay fixture has invalid keys/matrix")
    data_identity = identity.get("data_identity", {})
    if route.key_frame_identity(keys) != data_identity.get("test_keys"):
        raise BenchmarkError("checkpoint replay keys differ from metadata")
    if route.numeric_matrix_identity(values, columns) != data_identity.get("test_feature_matrix"):
        raise BenchmarkError("checkpoint replay matrix differs from metadata")
    if values.dtype != np.dtype("float64") or not values.flags.c_contiguous:
        raise BenchmarkError("checkpoint fixture matrix is not C-contiguous float64")
    if fold is not None:
        dates = pd.to_datetime(keys["date"], errors="coerce").dt.normalize()
        if dates.isna().any() or dates.lt(fold.start).any() or dates.gt(fold.end).any():
            raise BenchmarkError(f"{fold.fold_id} source fixture contains out-of-fold dates")
    if feature_buffer is not None or date_buffer is not None or code_buffer is not None:
        if feature_buffer is None or date_buffer is None or code_buffer is None:
            raise BenchmarkError("fixture reconstruction buffers must be supplied together")
        end = offset + len(values)
        if end > len(feature_buffer) or end > len(date_buffer) or end > len(code_buffer):
            raise BenchmarkError("source fixture exceeds the preallocated reconstruction buffers")
        feature_buffer[offset:end, :] = values
        date_buffer[offset:end] = pd.to_datetime(keys["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        code_buffer[offset:end] = keys["code"].astype(str).to_numpy(dtype=object)
    result = {
        "metadata": metadata,
        "replay": replay,
        "payload": payload,
        "columns": columns,
        "keys": keys,
        "values": values,
    }
    return result


def _load_and_verify_inputs(project_root: Path, contract: Mapping[str, Any], run_dir: Path,
                            *, preparation_claim: Mapping[str, Any]) -> dict[str, Any]:
    route = load_frozen_route(project_root, contract)
    frozen = contract["frozen_inputs"]
    contract_path = resolve_under(project_root, frozen["contract_r2"]["path"])
    binding_path = resolve_under(project_root, frozen["input_binding_r2"]["path"])
    activation_path = resolve_under(project_root, frozen["activation"]["path"])
    contract_sha = verify_pinned_hash(contract_path, frozen["contract_r2"]["sha256"], "CONTRACT_R2")
    binding_sha = verify_pinned_hash(binding_path, frozen["input_binding_r2"]["sha256"], "INPUT_BINDING_R2")
    verify_pinned_hash(activation_path, frozen["activation"]["sha256"], "review activation")
    binding = load_json(binding_path)
    frozen_contract = load_json(contract_path)
    if binding.get("status") != frozen["input_binding_r2"]["status"]:
        raise BenchmarkError("bound input status differs from the frozen benchmark contract")

    stage_manifest_path = resolve_under(project_root, frozen["stage_manifest"]["path"])
    stage_manifest = load_json(stage_manifest_path)
    verify_pinned_hash(stage_manifest_path, frozen["stage_manifest"]["sha256"], "formal stage manifest")
    if (stage_manifest.get("stage") != "M0_FORMAL" or stage_manifest.get("attempt") != "attempt_01" or
            stage_manifest.get("fit_calls") != 18 or stage_manifest.get("status") != "DIRECTION_ONLY"):
        raise BenchmarkError("completed M0 formal baseline manifest does not match the frozen target")
    if stage_manifest.get("code_sha256", {}).get("model_selection") != EXPECTED_MODEL_SHA256:
        raise BenchmarkError("formal baseline used a different frozen model-selection source")
    if stage_manifest.get("code_sha256") != route.code_identity():
        raise BenchmarkError("formal baseline code component identity differs from pinned route")
    if stage_manifest.get("contract_sha256") != contract_sha:
        raise BenchmarkError("formal baseline contract SHA differs from current frozen contract")
    if stage_manifest.get("input_binding_sha256") != binding_sha:
        raise BenchmarkError("formal baseline input binding SHA differs from current binding")
    sidecar_spec = frozen["input_compatibility_sidecar"]
    sidecar_path = resolve_under(project_root, sidecar_spec["path"])
    verify_pinned_hash(sidecar_path, sidecar_spec["sha256"], "formal input-compatibility sidecar")
    if (stage_manifest.get("input_compatibility_sidecar", {}).get("sha256") != sidecar_spec["sha256"] or
            stage_manifest.get("input_compatibility_sidecar_path") != str(sidecar_path)):
        raise BenchmarkError("formal stage manifest compatibility-sidecar identity differs")

    output_manifest_path = resolve_under(project_root, frozen["output_manifest"]["path"])
    output_manifest = load_json(output_manifest_path)
    verify_pinned_hash(output_manifest_path, frozen["output_manifest"]["sha256"], "formal output manifest")
    if output_manifest.get("status") != "DIRECTION_ONLY" or output_manifest.get("fit_calls") != 18:
        raise BenchmarkError("formal output manifest is not the completed 18-fit baseline")
    selected_keys_spec = frozen["output_manifest"]["selected_keys"]
    selected_keys_path = resolve_under(project_root, selected_keys_spec["path"])
    verify_pinned_hash(selected_keys_path, selected_keys_spec["sha256"], "formal selected-key output")

    fold_lookup = {item.fold_id: item for item in route.fixed_folds()}
    target_fold = fold_lookup[TARGET_FOLD]
    target_artifact = frozen["target_checkpoint"]
    expected_input_components = route.arm_source_identity(binding, "M0")
    target = _load_checkpoint_fixture(route, project_root, target_artifact,
                                      route.code_identity(), contract_sha,
                                      expected_input_components)
    target_metadata = target["metadata"]
    if target_metadata["identity"].get("stage") != "M0_FORMAL" or target_metadata["identity"].get("fold") != TARGET_FOLD:
        raise BenchmarkError("target checkpoint is not the M0/H1/2026H1 formal cell")
    if target_metadata["identity"].get("arm") != "M0" or target_metadata["identity"].get("horizon") != 1:
        raise BenchmarkError("target checkpoint is not the M0/H1 cell")

    source_artifacts = frozen["source_checkpoints"]
    if tuple(entry["fold"] for entry in source_artifacts) != SOURCE_FOLDS:
        raise BenchmarkError("contract source checkpoint fold sequence differs")
    source_metadata = []
    source_rows = 0
    for artifact in source_artifacts:
        metadata_path = _bound_path(project_root, artifact["metadata"])
        verify_pinned_hash(metadata_path, str(artifact["metadata"]["sha256"]),
                           f"{artifact['fold']} checkpoint metadata")
        metadata = load_json(metadata_path)
        keys_identity = metadata.get("identity", {}).get("data_identity", {}).get("test_keys", {})
        rows = int(keys_identity.get("row_count", -1))
        if rows <= 0:
            raise BenchmarkError(f"{artifact['fold']} source fixture has no authenticated test rows")
        source_metadata.append((artifact, metadata, rows))
        source_rows += rows
    target_rows = int(target["metadata"]["identity"]["data_identity"]["test_keys"]["row_count"])
    all_rows = source_rows + target_rows
    feature_columns = target["columns"]
    feature_buffer = np.empty((all_rows, len(feature_columns)), dtype=np.float64, order="C")
    date_buffer = np.empty(all_rows, dtype="datetime64[ns]")
    code_buffer = np.empty(all_rows, dtype=object)

    source_fixtures = []
    offset = 0
    for artifact, metadata, rows in source_metadata:
        loaded = _load_checkpoint_fixture(
            route, project_root, artifact, route.code_identity(), contract_sha,
            expected_input_components, feature_buffer=feature_buffer, date_buffer=date_buffer,
            code_buffer=code_buffer, offset=offset, fold=fold_lookup[artifact["fold"]],
        )
        if len(loaded["values"]) != rows or loaded["columns"] != feature_columns:
            raise BenchmarkError(f"{artifact['fold']} source feature fixture dimensions differ")
        source_fixtures.append({
            "fold": artifact["fold"],
            "rows": int(len(loaded["keys"])),
            "keyset_sha256": metadata["identity"]["data_identity"]["test_keys"]["sorted_keyset_sha256"],
            "feature_matrix_sha256": metadata["identity"]["data_identity"]["test_feature_matrix"]["sha256"],
        })
        offset += rows
        del loaded["payload"], loaded["keys"], loaded["values"], loaded
        gc.collect()

    target_offset = offset
    target_end = target_offset + target_rows
    feature_buffer[target_offset:target_end, :] = target["values"]
    date_buffer[target_offset:target_end] = pd.to_datetime(target["keys"]["date"], errors="coerce").to_numpy(
        dtype="datetime64[ns]")
    code_buffer[target_offset:target_end] = target["keys"]["code"].astype(str).to_numpy(dtype=object)
    if target_end != all_rows:
        raise BenchmarkError("source and target fixtures do not fill the preallocated feature buffers")

    # Historical M0/H1 checkpoint test fixtures are the exact feature matrices
    # consumed by the same frozen formal route. Load each checkpoint once,
    # verify its sidecar and replay identity, copy only its matrix/keys, then
    # release its estimator and prediction objects before loading the next one.
    feature_frame = pd.DataFrame(feature_buffer, columns=list(feature_columns), copy=False)
    feature_frame.insert(0, "code", code_buffer)
    feature_frame.insert(0, "date", date_buffer)

    raw_panel, records = route.read_bound_panel(
        contract_path=contract_path, binding_path=binding_path, activation_path=activation_path,
        stage="M0_FORMAL", arms=route.STAGE_SPECS["M0_FORMAL"]["input_arms"],
        preparation_claim=preparation_claim,
    )
    if records.get("binding") != binding:
        raise BenchmarkError("bound panel reader returned a different R2 input binding")
    price_binding = binding["sources"]["price_panel"]
    amount_unit = price_binding.get("units", {}).get("amount_input")
    if amount_unit == "10000_CNY":
        amount_multiplier = 10000.0
    elif amount_unit == "CNY":
        amount_multiplier = 1.0
    else:
        raise BenchmarkError(f"unsupported bound amount unit: {amount_unit}")
    panel = route.normalize_price_panel(raw_panel, price_binding["column_map"],
                                        amount_multiplier=amount_multiplier)
    del raw_panel, records
    calendar_source = binding["sources"]["trading_date_index"]
    calendar_raw = pd.read_csv(io.BytesIO(route.read_bound_archive_member(calendar_source)))
    calendar = route.normalize_calendar(calendar_raw, panel)
    del calendar_raw
    labels = route.build_close_return_labels(
        panel, horizons=(1,), calendar=calendar,
        signal_start=target_fold.start - pd.DateOffset(years=2),
        signal_end_exclusive=target_fold.start,
    )[1]
    del panel, calendar
    gc.collect()

    target_predictions = target["payload"].get("predictions")
    if not isinstance(target_predictions, pd.DataFrame):
        raise BenchmarkError("target checkpoint has no saved four-thread predictions")
    target_replay_identity = target["replay"]["prediction_identity"]
    if route.prediction_frame_identity(target_predictions) != target_replay_identity:
        raise BenchmarkError("saved four-thread prediction identity differs from its replay receipt")
    if route.key_frame_identity(target_predictions[["date", "code"]]) != target_metadata["identity"]["data_identity"]["test_keys"]:
        raise BenchmarkError("saved four-thread prediction key order differs from target checkpoint metadata")
    target_data_identity = target_metadata["identity"]["data_identity"]
    if not np.array_equal(
            feature_buffer[target_offset:target_end, :],
            np.asarray(target["values"], dtype=np.float64)):
        raise BenchmarkError("copied target test features differ from the frozen test fixture")
    target["payload"].pop("model", None)
    target["payload"] = {"predictions": target_predictions}
    del target["keys"], target["values"]
    gc.collect()

    prepared = route.prepare_cell_data(features=feature_frame, labels=labels,
                                      fold=target_fold, canary=False)
    del feature_frame, feature_buffer, date_buffer, code_buffer, labels
    gc.collect()
    expected_identity = target_data_identity
    verify_identity(expected_identity, prepared["identity"], name="M0/H1/2026H1 train/test")
    reconstructed_identity = route.checkpoint_identity(
        stage="M0_FORMAL", fold=target_fold, arm="M0", horizon=1, attempt="attempt_01",
        contract_sha256=contract_sha, code_shas=route.code_identity(), binding=binding,
        feature_names=prepared["feature_columns"], data_identity=prepared["identity"],
    )
    verify_identity(target_metadata["identity"], reconstructed_identity,
                    name="M0/H1/2026H1 complete checkpoint/config")
    if prepared["feature_columns"] != feature_columns:
        raise BenchmarkError("target feature column order differs from the original checkpoint")
    if prepared["identity"]["training_keys"]["date_max"] >= target_fold.start.strftime("%Y-%m-%d"):
        raise BenchmarkError("strict label-end cutoff failed: training signal reached the target period")
    if (pd.to_datetime(prepared["train"]["label_end_date"], errors="coerce") >= target_fold.start).any():
        raise BenchmarkError("strict label_end_date < 2026-01-01 was violated")
    if prepared["training_label_preparation_slots"] != int(frozen["expected_training_label_preparation_slots"]):
        raise BenchmarkError("broad label-preparation slot count differs from the frozen baseline")
    if len(prepared["train"]) != int(frozen["expected_training_rows"]):
        raise BenchmarkError("target train row count differs from the frozen baseline")

    predictions = target["payload"]["predictions"]
    baseline_ranked = rank_all(predictions)
    selected_keys = pd.read_csv(selected_keys_path)
    selected_keys = selected_keys.loc[
        (selected_keys["model_arm"] == "M0") &
        (pd.to_numeric(selected_keys["model_horizon"], errors="coerce") == 1) &
        (selected_keys["fold"].astype(str) == TARGET_FOLD),
        ["date", "code", "rank"],
    ].copy()
    selected_keys["date"] = pd.to_datetime(selected_keys["date"], errors="coerce").dt.normalize()
    selected_keys["rank"] = pd.to_numeric(selected_keys["rank"], errors="coerce").astype("int64")
    baseline_top10 = baseline_ranked.loc[
        baseline_ranked["rank"] <= 10, ["date", "code", "rank"]
    ].reset_index(drop=True)
    if not baseline_top10.equals(selected_keys.sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True)):
        raise BenchmarkError("four-thread checkpoint Top10 differs from frozen selected-key output")

    training_date_min = prepared["identity"]["training_keys"]["date_min"]
    training_date_max = prepared["identity"]["training_keys"]["date_max"]
    target_date_min = pd.to_datetime(prepared["train"]["label_end_date"]).min().strftime("%Y-%m-%d")
    target_date_max = pd.to_datetime(prepared["train"]["label_end_date"]).max().strftime("%Y-%m-%d")
    result = {
        "route": route,
        "route_code_sha256": route.route_code_sha256(),
        "code_identity": route.code_identity(),
        "frozen_contract": frozen_contract,
        "contract_sha256": contract_sha,
        "binding": binding,
        "binding_sha256": binding_sha,
        "target_fold": target_fold,
        "benchmark_identity": reconstructed_identity,
        "prepared": prepared,
        "target": {"metadata": target_metadata, "replay": target["replay"]},
        "baseline_predictions": predictions,
        "baseline_ranked": baseline_ranked,
        "source_fixtures": source_fixtures,
        "selected_keys_sha256": selected_keys_spec["sha256"],
        "manifest_sha256": frozen["stage_manifest"]["sha256"],
        "output_manifest_sha256": frozen["output_manifest"]["sha256"],
        "training_date_min": training_date_min,
        "training_date_max": training_date_max,
        "target_date_min": target_date_min,
        "target_date_max": target_date_max,
    }
    return result


def make_benchmark_guard(route: Any, *, contract_path: Path, binding_path: Path,
                         stage: str, artifact_paths: Sequence[Path] = (),
                         registry_path: Path | None = None):
    """Reuse the official locked adapter, with a separate one-fit experiment."""
    base = route.make_official_fit_guard(
        contract_path=contract_path, binding_path=binding_path, stage=stage,
        artifact_paths=artifact_paths, registry_path=registry_path,
    )
    base.backend.event_profile.update({
        "route_id": BENCHMARK_ROUTE_ID,
        "one_variable": "fit_threadpool_limit_4_to_12",
        "baseline_id": TARGET_CELL,
        "evidence_level": "ENGINEERING_CANARY_ONLY",
    })
    return type(base)(base.backend, experiment_id=BENCHMARK_EXPERIMENT_ID,
                      max_real_fit_debits=1)


def validate_benchmark_retry_budget(guard: Any) -> None:
    """Closed zero-fit checks do not spend the one durable real-fit claim."""
    if guard.accounting().dangling_execution_ids:
        raise BenchmarkError("official registry contains a dangling execution; refusing benchmark")
    for event in guard.backend.list_events(guard.experiment_id):
        if event.get("event_type") != "RUN_STARTED":
            continue
        if (event.get("execution_kind") != "NO_FIT_ENGINEERING_CANARY" or
                event.get("real_fit_budget_debit") != 0):
            raise BenchmarkError("the one-run benchmark execution budget has already been consumed")


def validate_worker_claim(events: Sequence[Mapping[str, Any]], message: Mapping[str, Any],
                          *, phase: str, parent_pid: int, contract_sha: str,
                          binding_sha: str) -> Mapping[str, Any]:
    """A local sentinel cannot authorize a read/fit: require an open registry claim."""
    execution_id = message.get("execution_id")
    nonce = message.get("nonce")
    if message.get("phase") != phase or not execution_id or not isinstance(nonce, str) or len(nonce) != 32:
        raise BenchmarkError("worker authorization channel is missing a valid claim")
    scoped = [event for event in events if event.get("execution_id") == execution_id]
    starts = [event for event in scoped if event.get("event_type") == "RUN_STARTED"]
    if len(starts) != 1 or any(event.get("event_type") == "TERMINAL" for event in scoped):
        raise BenchmarkError("worker authorization requires one open official RUN_STARTED")
    event = starts[0]
    identity = event.get("input_identity", {})
    metrics = event.get("deterministic_metrics", {})
    kind = "REAL_FIT" if phase == "FIT" else "NO_FIT_ENGINEERING_CANARY"
    if (event.get("experiment_id") != BENCHMARK_EXPERIMENT_ID or
            event.get("route_id") != BENCHMARK_ROUTE_ID or
            event.get("contract_sha256") != contract_sha or
            event.get("input_binding_sha256") != binding_sha or
            metrics.get("execution_kind") != kind or
            metrics.get("real_fit_budget_debit") != (1 if phase == "FIT" else 0) or
            identity.get("benchmark_supervisor_pid") != parent_pid or
            identity.get("worker_authorization_sha256") != hashlib.sha256(nonce.encode()).hexdigest()):
        raise BenchmarkError("worker authorization differs from its official claim")
    return event


def receive_worker_claim(project_root: Path, contract_path: Path,
                         contract: Mapping[str, Any], *, phase: str) -> Mapping[str, Any]:
    if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
        raise BenchmarkError("standalone worker has no inherited supervisor pipe")
    line = sys.stdin.readline(8192)
    try:
        message = json.loads(line)
    except (ValueError, TypeError) as exc:
        raise BenchmarkError("worker authorization pipe has no claim") from exc
    if not isinstance(message, dict):
        raise BenchmarkError("worker authorization must be an object")
    route = load_frozen_route(project_root, contract)
    binding = contract["frozen_inputs"]["input_binding_r2"]
    guard = make_benchmark_guard(route, contract_path=contract_path,
                                 binding_path=resolve_under(project_root, binding["path"]),
                                 stage=f"M0_THREAD_BENCHMARK_CANARY_{phase}")
    return validate_worker_claim(
        guard.backend.list_events(BENCHMARK_EXPERIMENT_ID), message,
        phase=phase, parent_pid=os.getppid(), contract_sha=sha256_file(contract_path),
        binding_sha=binding["sha256"],
    )


def send_worker_claim(process: subprocess.Popen, claim: Any, nonce: str, *, phase: str) -> None:
    if process.stdin is None:
        raise BenchmarkError("supervisor worker pipe is unavailable")
    process.stdin.write((json.dumps({"execution_id": claim.execution_id,
                                    "nonce": nonce, "phase": phase}) + "\n").encode())
    process.stdin.flush()


def _worker_main(project_root: Path, contract_path: Path, run_dir: Path) -> int:
    status_path = run_dir / "worker_status.json"
    fit_started = False
    worker_start_wall = time.perf_counter()
    try:
        contract = load_json(contract_path)
        if contract.get("schema_version") != TOOL_SCHEMA:
            raise BenchmarkError("benchmark contract schema mismatch")
        if config_identity(contract) != contract.get("config_sha256"):
            raise BenchmarkError("benchmark contract config SHA mismatch")
        verify_pinned_hash(Path(__file__).resolve(), contract["writer_tool_sha256"], "benchmark writer tool")
        preparation_claim = receive_worker_claim(project_root, contract_path, contract, phase="PREFLIGHT")
        atomic_write_json(status_path, {"state": "INPUT_READ_STARTED", "fit_calls": 0})
        inputs = _load_and_verify_inputs(project_root, contract, run_dir,
                                         preparation_claim=preparation_claim)
        prepared = inputs["prepared"]
        preflight = {
            "status": "READY_TO_FIT",
            "route_code_sha256": inputs["route_code_sha256"],
            "frozen_code_identity": inputs["code_identity"],
            "contract_sha256": inputs["contract_sha256"],
            "input_binding_sha256": inputs["binding_sha256"],
            "target_cell": TARGET_CELL,
            "training_rows": int(len(prepared["train"])),
            "training_label_preparation_slots": int(prepared["training_label_preparation_slots"]),
            "training_signal_dates": int(prepared["train"]["date"].nunique()),
            "training_date_min": inputs["training_date_min"],
            "training_date_max": inputs["training_date_max"],
            "target_date_min": inputs["target_date_min"],
            "target_date_max": inputs["target_date_max"],
            "training_keyset_sha256": prepared["identity"]["training_keys"]["sorted_keyset_sha256"],
            "training_key_order_sha256": prepared["identity"]["training_keys"]["ordered_keys_sha256"],
            "training_matrix_sha256": prepared["identity"]["training_feature_matrix"]["sha256"],
            "training_label_vector_sha256": prepared["identity"]["training_label_vector"]["sha256"],
            "sample_weight_vector_sha256": prepared["identity"]["training_sample_weight_vector"]["sha256"],
            "test_rows": int(len(prepared["test"])),
            "test_keyset_sha256": prepared["identity"]["test_keys"]["sorted_keyset_sha256"],
            "test_matrix_sha256": prepared["identity"]["test_feature_matrix"]["sha256"],
            "feature_columns": list(prepared["feature_columns"]),
            "source_fixtures": inputs["source_fixtures"],
            "strict_label_end_cutoff": "label_end_date < 2026-01-01",
            "fit_calls": 0,
            "truth_access": "TRAIN_LABELS_READ_PAST_ONLY",
            "sealed_status": "EXPOSED_HISTORY",
        }
        atomic_write_json(run_dir / "preflight.json", preflight)
        atomic_write_json(status_path, {"state": "READY_TO_FIT", "fit_calls": 0})
        receive_worker_claim(project_root, contract_path, contract, phase="FIT")
        atomic_write_json(status_path, {"state": "FIT_CLAIMED", "fit_calls": 0})
        route = inputs["route"]
        model = route.model_for_arm()
        x, y, weights, xt = (prepared[key] for key in ("x", "y", "weights", "xt"))
        fit_thread_info: list[dict[str, Any]]
        before_fit = threadpool_info()
        fit_start_wall = time.perf_counter()
        fit_start_cpu = time.process_time()
        with threadpool_limits(limits=FIT_THREADS):
            fit_thread_info = threadpool_info()
            fit_method = model.fit
            atomic_write_json(status_path, {"state": "FIT_STARTED", "fit_calls": 1,
                                            "fit_started_at_epoch_seconds": time.time()})
            fit_started = True
            fit_method(x, y, sample_weight=weights)
        fit_seconds = time.perf_counter() - fit_start_wall
        fit_cpu_seconds = time.process_time() - fit_start_cpu
        prediction_thread_info = threadpool_info()
        predict_start = time.perf_counter()
        score = np.asarray(model.predict(xt), dtype=np.float64)
        predict_seconds = time.perf_counter() - predict_start
        candidate_predictions = prepared["test"][["date", "code"]].copy()
        candidate_predictions["prediction"] = score
        comparison = compare_prediction_frames(candidate_predictions, inputs["baseline_predictions"])
        if (not comparison["prediction_array_equal"] or not comparison["full_daily_rank_order_exact"] or
                not comparison["top10_keys_exact"]):
            atomic_write_json(run_dir / "result.json", {
                "status": "PARITY_REJECTED",
                "target_cell": TARGET_CELL,
                "fit_calls": 1,
                "benchmark_fit_threads": FIT_THREADS,
                "fit_seconds": fit_seconds,
                "fit_process_cpu_seconds": fit_cpu_seconds,
                "prediction_seconds": predict_seconds,
                "comparison": comparison,
            })
            raise BenchmarkError("12-thread predictions/rank ordering/Top10 failed strict four-thread parity")
        candidate_ranked = rank_all(candidate_predictions)
        baseline_top10 = inputs["baseline_ranked"].loc[
            inputs["baseline_ranked"]["rank"] <= 10, ["date", "code", "rank"]
        ].reset_index(drop=True)
        selected_keys_path = resolve_under(project_root, contract["frozen_inputs"]["output_manifest"]["selected_keys"]["path"])
        selected_output = pd.read_csv(selected_keys_path)
        selected_output = selected_output.loc[
            (selected_output["model_arm"] == "M0") &
            (pd.to_numeric(selected_output["model_horizon"], errors="coerce") == 1) &
            (selected_output["fold"].astype(str) == TARGET_FOLD),
            ["date", "code", "rank"],
        ].copy()
        selected_output["date"] = pd.to_datetime(selected_output["date"], errors="coerce").dt.normalize()
        selected_output["rank"] = pd.to_numeric(selected_output["rank"], errors="coerce").astype("int64")
        candidate_top10 = candidate_ranked.loc[
            candidate_ranked["rank"] <= 10, ["date", "code", "rank"]
        ].reset_index(drop=True)
        frozen_output_exact = candidate_top10.equals(
            selected_output.sort_values(["date", "rank"], kind="mergesort").reset_index(drop=True))
        if not frozen_output_exact or not candidate_top10.equals(baseline_top10):
            atomic_write_json(run_dir / "result.json", {
                "status": "OUTPUT_PARITY_REJECTED",
                "target_cell": TARGET_CELL,
                "fit_calls": 1,
                "benchmark_fit_threads": FIT_THREADS,
                "fit_seconds": fit_seconds,
                "fit_process_cpu_seconds": fit_cpu_seconds,
                "prediction_seconds": predict_seconds,
                "comparison": comparison,
                "frozen_selected_output_top10_exact": bool(frozen_output_exact),
            })
            raise BenchmarkError("12-thread Top10 differs from checkpoint or completed formal selected-key output")

        checkpoint_path = run_dir / "candidate_checkpoint.joblib"
        checkpoint_sha = _write_joblib_atomic(checkpoint_path, {
            "schema_version": TOOL_SCHEMA,
            "model": model,
            "feature_columns": tuple(prepared["feature_columns"]),
            "identity": inputs["benchmark_identity"],
            "data_identity": prepared["identity"],
            "benchmark_config_sha256": contract["config_sha256"],
            "contract_file_sha256": sha256_file(contract_path),
            "fit_threads": FIT_THREADS,
            "baseline_threads": BASELINE_THREADS,
        })
        predictions_path = run_dir / "candidate_predictions.csv"
        candidate_predictions.to_csv(predictions_path, index=False, date_format="%Y-%m-%d")
        top10_path = run_dir / "candidate_top10.csv"
        candidate_top10.to_csv(top10_path, index=False, date_format="%Y-%m-%d")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = {
            "status": "COMPLETED",
            "target_cell": TARGET_CELL,
            "fit_calls": 1,
            "benchmark_config_sha256": contract["config_sha256"],
            "frozen_route_code_sha256": inputs["route_code_sha256"],
            "frozen_checkpoint_config_sha256": inputs["benchmark_identity"]["config_sha256"],
            "baseline_fit_threads": BASELINE_THREADS,
            "benchmark_fit_threads": FIT_THREADS,
            "prediction_threads_controlled": False,
            "model_params": dict(route.MODEL_PARAMS),
            "fit_seconds": fit_seconds,
            "worker_wall_seconds": time.perf_counter() - worker_start_wall,
            "fit_process_cpu_seconds": fit_cpu_seconds,
            "prediction_seconds": predict_seconds,
            "fit_threadpool_info_before": before_fit,
            "fit_threadpool_info_during": fit_thread_info,
            "prediction_threadpool_info": prediction_thread_info,
            "comparison": comparison,
            "frozen_selected_output_top10_exact": frozen_output_exact,
            "source_fixtures": inputs["source_fixtures"],
            "training_rows": int(len(prepared["train"])),
            "training_label_preparation_slots": int(prepared["training_label_preparation_slots"]),
            "training_keyset_sha256": prepared["identity"]["training_keys"]["sorted_keyset_sha256"],
            "training_key_order_sha256": prepared["identity"]["training_keys"]["ordered_keys_sha256"],
            "training_matrix_sha256": prepared["identity"]["training_feature_matrix"]["sha256"],
            "training_label_vector_sha256": prepared["identity"]["training_label_vector"]["sha256"],
            "test_matrix_sha256": prepared["identity"]["test_feature_matrix"]["sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "top10_path": str(top10_path),
            "top10_sha256": sha256_file(top10_path),
            "worker_ru_maxrss_bytes": int(usage.ru_maxrss),
            "worker_user_cpu_seconds": float(usage.ru_utime),
            "worker_system_cpu_seconds": float(usage.ru_stime),
        }
        atomic_write_json(run_dir / "result.json", result)
        atomic_write_json(status_path, {"state": "COMPLETED", "fit_calls": 1,
                                        "checkpoint_sha256": checkpoint_sha})
        return 0
    except BaseException as exc:
        atomic_write_json(status_path, {
            "state": "FAILED",
            "fit_calls": 1 if fit_started else 0,
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        })
        return 1


def _guard_context(route: Any, contract: Mapping[str, Any], binding: Mapping[str, Any],
                   frozen_contract_sha: str, binding_sha: str, run_dir: Path,
                   *, stage: str, truth_access: str,
                   metrics: Mapping[str, Any] | None = None,
                   artifact_paths: Sequence[Path] | None = None) -> dict[str, Any]:
    code_shas = dict(route.code_identity())
    code_shas["thread_benchmark_v1"] = sha256_file(Path(__file__).resolve())
    return route.build_registry_event_context(
        stage=stage, binding=binding, contract_sha256=frozen_contract_sha,
        binding_sha256=binding_sha, code_shas=code_shas,
        config_sha256=str(contract["config_sha256"]), arms=("M0",),
        truth_access=truth_access, artifact_paths=tuple(artifact_paths) if artifact_paths is not None else (
            run_dir / "receipt.json", run_dir / "preflight.json", run_dir / "result.json",
            run_dir / "candidate_checkpoint.joblib", run_dir / "candidate_predictions.csv",
            run_dir / "candidate_top10.csv",
        ), deterministic_metrics=dict(metrics or {}),
    )


def _record_no_fit_terminal(route: Any, guard: Any, claim: Any, execution_id: str,
                            metrics: Mapping[str, Any], *, succeeded: bool) -> None:
    route.update_registry_execution_context(
        guard, execution_id, truth_access=(
            "TRAIN_LABELS_READ_PAST_ONLY" if metrics.get("truth_access") == "TRAIN_LABELS_READ_PAST_ONLY"
            else "BOUND_INPUT_READ_NOT_STARTED"),
        deterministic_metrics=dict(metrics),
    )
    guard.append_no_fit_terminal(claim, succeeded=succeeded,
                                 error_type=None if succeeded else str(metrics.get("error_type", "BenchmarkError")))


def _record_fit_terminal(route: Any, guard: Any, claim: Any, execution_id: str,
                         metrics: Mapping[str, Any], *, succeeded: bool,
                         fit_call_entered: bool, returned_count: int,
                         error_type: str | None = None) -> None:
    updates: dict[str, Any] = {"deterministic_metrics": dict(metrics)}
    checkpoint_path = metrics.get("checkpoint_path")
    checkpoint_sha = metrics.get("checkpoint_sha256")
    if checkpoint_path and checkpoint_sha:
        updates["checkpoint_ref"] = {"path": str(checkpoint_path), "sha256": str(checkpoint_sha)}
    route.update_registry_execution_context(guard, execution_id, **updates)
    guard.append_terminal(
        claim,
        terminal_status=("COMPLETED" if succeeded else
                         "FAILED_DURING_FIT_CALL" if fit_call_entered else "ABORTED_BEFORE_FIT"),
        fit_call_entered=fit_call_entered,
        real_fit_returned_count=returned_count,
        error_type=None if succeeded else (error_type or "BenchmarkError"),
    )


def _terminate_and_collect(process: subprocess.Popen, *, grace_seconds: float = 5.0) -> int:
    terminate_only_worker(process, grace_seconds=grace_seconds)
    return int(process.returncode if process.returncode is not None else -1)


def run_benchmark(project_root: Path, contract_path: Path, *, execution_id: str | None = None) -> Path:
    project_root = Path(project_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = load_json(contract_path)
    if contract.get("schema_version") != TOOL_SCHEMA:
        raise BenchmarkError("benchmark contract schema mismatch")
    if config_identity(contract) != contract.get("config_sha256"):
        raise BenchmarkError("benchmark contract config SHA mismatch")
    if (contract["registry"]["experiment_id"] != BENCHMARK_EXPERIMENT_ID or
            contract["registry"]["route_id"] != BENCHMARK_ROUTE_ID):
        raise BenchmarkError("benchmark requires its independent registry experiment and route")
    verify_pinned_hash(Path(__file__).resolve(), contract["writer_tool_sha256"], "benchmark writer tool")
    contract_file_sha = sha256_file(contract_path)
    started_wall = time.perf_counter()
    run_id = execution_id or f"thread12-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", run_id):
        raise BenchmarkError("execution ID must contain only ASCII letters, digits, underscore, or hyphen")
    run_dir = resolve_under(project_root, f"{contract['output_root']}/{run_id}")
    if run_dir.exists():
        raise BenchmarkError(f"run output already exists; refusing overwrite: {run_dir}")
    receipt_path = run_dir / "receipt.json"
    checkpoint_path = run_dir / "candidate_checkpoint.joblib"

    route = load_frozen_route(project_root, contract)
    frozen_inputs = contract["frozen_inputs"]
    contract_r2_path = resolve_under(project_root, frozen_inputs["contract_r2"]["path"])
    binding_path = resolve_under(project_root, frozen_inputs["input_binding_r2"]["path"])
    frozen_contract_sha = sha256_file(contract_r2_path)
    binding_sha = sha256_file(binding_path)
    if frozen_contract_sha != frozen_inputs["contract_r2"]["sha256"] or binding_sha != frozen_inputs["input_binding_r2"]["sha256"]:
        raise BenchmarkError("frozen contract or input binding SHA drifted")
    binding = load_json(binding_path)
    registry_contract_sha = contract_file_sha
    worker_nonce = uuid.uuid4().hex
    worker_authorization = {
        "benchmark_supervisor_pid": os.getpid(),
        "worker_authorization_sha256": hashlib.sha256(worker_nonce.encode()).hexdigest(),
    }

    prep_stage = "M0_THREAD_BENCHMARK_CANARY_PREFLIGHT"
    execution_prefix = f"M0_THREAD_BENCHMARK_{run_id}_"
    prep_execution_id = f"{execution_prefix}PREFLIGHT"
    fit_execution_id = f"{execution_prefix}FIT"
    prep_guard = make_benchmark_guard(route,
        contract_path=contract_path, binding_path=binding_path, stage=prep_stage,
        artifact_paths=(receipt_path, run_dir / "preflight.json", run_dir / "worker_status.json"),
    )
    validate_benchmark_retry_budget(prep_guard)

    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(receipt_path, {
        "schema_version": TOOL_SCHEMA,
        "status": "PREFLIGHT_PENDING",
        "execution_id": run_id,
        "contract_sha256": contract_file_sha,
        "config_sha256": contract["config_sha256"],
        "fit_calls": 0,
        "security_flags": {"research_only": True, "production_allowed": False,
                           "buy_gate_updated": False, "orders_sent": False},
    })
    atomic_write_json(checkpoint_path, {"status": "PLANNED_BEFORE_FIT", "fit_calls": 0})
    atomic_write_json(run_dir / "preflight.json", {"status": "PLANNED_BEFORE_PREFLIGHT"})
    atomic_write_json(run_dir / "worker_status.json", {"state": "PLANNED_BEFORE_PREFLIGHT", "fit_calls": 0})
    prep_context = _guard_context(route, contract, binding, registry_contract_sha, binding_sha,
                                  run_dir, stage=prep_stage, truth_access="BOUND_INPUT_READ_NOT_STARTED",
                                  artifact_paths=(receipt_path, run_dir / "preflight.json",
                                                  run_dir / "worker_status.json"))
    prep_context["input_identity"].update(worker_authorization)
    route.set_registry_execution_context(prep_guard, prep_execution_id, prep_context)
    prep_claim = prep_guard.claim_no_fit(
        execution_id=prep_execution_id, arm="M0:H1:2026H1:RESOURCE_AND_INPUT_PREFLIGHT",
        contract_sha256=registry_contract_sha, input_binding_sha256=binding_sha,
    )
    fit_claim = None
    fit_guard = None
    prep_terminalized = False
    last_state: dict[str, Any] = {"state": "NOT_STARTED", "fit_calls": 0}
    resource_summary: dict[str, Any] = {}
    resource_samples: list[dict[str, Any]] = []
    process: subprocess.Popen | None = None
    try:
        initial_snapshot = probe_host_resources()
        resource_summary["initial_snapshot"] = initial_snapshot
        if (initial_snapshot["pressure_state"] != "normal" or
                int(initial_snapshot["available_bytes_estimate"]) < MIN_AVAILABLE_BYTES):
            raise BenchmarkError("resource preflight requires normal pressure and at least 8 GiB available")
        process = subprocess.Popen([
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--project-root", str(project_root), "--contract", str(contract_path),
            "--run-dir", str(run_dir),
        ], cwd=str(project_root), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        send_worker_claim(process, prep_claim, worker_nonce, phase="PREFLIGHT")
        while process.poll() is None:
            status_path = run_dir / "worker_status.json"
            if status_path.is_file():
                last_state = load_json(status_path)
                if last_state.get("state") == "FAILED":
                    break
                if last_state.get("state") == "READY_TO_FIT":
                    break
            snapshot = probe_host_resources(process.pid)
            resource_samples.append(snapshot)
            resource_summary["last_snapshot"] = snapshot
            abort = resource_abort_reason(snapshot, initial_swap_used_bytes=int(initial_snapshot["swap_used_bytes"]),
                                          during_run=True)
            if abort:
                _terminate_and_collect(process)
                resource_summary["resource_abort_reason"] = abort
                raise BenchmarkError(abort)
            time.sleep(SAMPLE_SECONDS)
        if process.poll() is not None and not (run_dir / "worker_status.json").is_file():
            raise BenchmarkError(f"preflight worker exited without status ({process.returncode})")
        last_state = load_json(run_dir / "worker_status.json")
        if last_state.get("state") != "READY_TO_FIT":
            raise BenchmarkError(f"input preflight failed: {last_state.get('error_type')}: {last_state.get('error')}")
        preflight = load_json(run_dir / "preflight.json")
        resource_summary.update({
            "worker_rss_sample_count_before_fit": len(resource_samples),
            "worker_sampled_max_rss_before_fit_bytes": max(
                (int(item["worker_rss_bytes"]) for item in resource_samples
                 if item.get("worker_rss_bytes") is not None), default=None),
        })
        route.update_registry_execution_context(
            prep_guard, prep_execution_id,
            truth_access="TRAIN_LABELS_READ_PAST_ONLY",
            signal_date_range={"count": int(preflight["training_signal_dates"]),
                               "min": preflight["training_date_min"],
                               "max": preflight["training_date_max"],
                               "scope": "M0_H1_2026H1_training_signal_dates"},
            deterministic_metrics={
                "fit_calls": 0, "training_rows": preflight["training_rows"],
                "training_label_preparation_slots": preflight["training_label_preparation_slots"],
                "training_keyset_sha256": preflight["training_keyset_sha256"],
                "training_matrix_sha256": preflight["training_matrix_sha256"],
                "input_parity": "EXACT",
            },
        )
        prep_guard.append_no_fit_terminal(prep_claim, succeeded=True)
        prep_terminalized = True

        fit_stage = "M0_THREAD_BENCHMARK_CANARY"
        fit_guard = make_benchmark_guard(route,
            contract_path=contract_path, binding_path=binding_path, stage=fit_stage,
            artifact_paths=(receipt_path, run_dir / "preflight.json", run_dir / "result.json",
                            checkpoint_path, run_dir / "candidate_predictions.csv",
                            run_dir / "candidate_top10.csv"),
        )
        if fit_guard.accounting().dangling_execution_ids:
            raise BenchmarkError("official registry contains a dangling execution before fit")
        fit_context = _guard_context(
            route, contract, binding, registry_contract_sha, binding_sha, run_dir,
            stage=fit_stage, truth_access="TRAIN_LABELS_READ_PAST_ONLY",
            metrics={
                "fit_calls": 1, "target_cell": TARGET_CELL,
                "training_rows": preflight["training_rows"],
                "training_label_preparation_slots": preflight["training_label_preparation_slots"],
                "training_keyset_sha256": preflight["training_keyset_sha256"],
                "training_key_order_sha256": preflight["training_key_order_sha256"],
                "training_matrix_sha256": preflight["training_matrix_sha256"],
                "training_label_vector_sha256": preflight["training_label_vector_sha256"],
                "sample_weight_vector_sha256": preflight["sample_weight_vector_sha256"],
                "test_keyset_sha256": preflight["test_keyset_sha256"],
                "test_matrix_sha256": preflight["test_matrix_sha256"],
                "fit_threads": FIT_THREADS, "baseline_threads": BASELINE_THREADS,
            },
            artifact_paths=(receipt_path, run_dir / "preflight.json", run_dir / "result.json",
                            checkpoint_path, run_dir / "candidate_predictions.csv",
                            run_dir / "candidate_top10.csv"),
        )
        fit_context["signal_date_range"] = {
            "count": int(preflight["training_signal_dates"]),
            "min": preflight["training_date_min"],
            "max": preflight["training_date_max"],
            "scope": "M0_H1_2026H1_training_signal_dates",
        }
        fit_context["target_date_range"] = {
            "min": preflight["target_date_min"],
            "max": preflight["target_date_max"],
            "scope": "M0_H1_2026H1_training_label_end_dates",
        }
        atomic_write_json(run_dir / "result.json", {"status": "PLANNED_BEFORE_FIT"})
        (run_dir / "candidate_predictions.csv").write_text(
            "date,code,prediction\n", encoding="utf-8")
        (run_dir / "candidate_top10.csv").write_text(
            "date,code,rank\n", encoding="utf-8")
        fit_context["input_identity"].update(worker_authorization)
        route.set_registry_execution_context(fit_guard, fit_execution_id, fit_context)
        fit_claim = fit_guard.claim(
            execution_id=fit_execution_id, arm="M0:H1:2026H1:THREADS_12",
            contract_sha256=registry_contract_sha, input_binding_sha256=binding_sha,
        )
        atomic_write_json(run_dir / "supervisor.json", {
            "fit_execution_claimed": True,
            "fit_execution_id": fit_execution_id,
            "fit_claimed_at_epoch_seconds": time.time(),
        })
        send_worker_claim(process, fit_claim, worker_nonce, phase="FIT")
        while process.poll() is None:
            status_path = run_dir / "worker_status.json"
            if status_path.is_file():
                last_state = load_json(status_path)
                if last_state.get("state") in {"COMPLETED", "FAILED"}:
                    break
            snapshot = probe_host_resources(process.pid)
            resource_samples.append(snapshot)
            resource_summary["last_snapshot"] = snapshot
            abort = resource_abort_reason(snapshot, initial_swap_used_bytes=int(initial_snapshot["swap_used_bytes"]),
                                          during_run=True)
            if abort:
                _terminate_and_collect(process)
                resource_summary["resource_abort_reason"] = abort
                break
            time.sleep(SAMPLE_SECONDS)
        last_state = load_json(run_dir / "worker_status.json") if (run_dir / "worker_status.json").is_file() else last_state
        if last_state.get("state") in {"COMPLETED", "FAILED"}:
            process.wait(timeout=10)
        resource_summary.update({
            "worker_pid": int(process.pid),
            "worker_returncode": int(process.returncode if process.returncode is not None else -1),
            "worker_ru_maxrss_children_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
            "worker_children_user_cpu_seconds": float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime),
            "worker_children_system_cpu_seconds": float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime),
            "worker_rss_sample_count": len(resource_samples),
            "worker_sampled_max_rss_bytes": max(
                (int(item["worker_rss_bytes"]) for item in resource_samples
                 if item.get("worker_rss_bytes") is not None), default=None),
            "fit_claimed": True,
            "parent_total_wall_seconds": time.perf_counter() - started_wall,
        })
        succeeded = bool(last_state.get("state") == "COMPLETED" and not resource_summary.get("resource_abort_reason"))
        result = load_json(run_dir / "result.json") if (run_dir / "result.json").is_file() else {}
        terminal_metrics = {**result, **resource_summary,
                            "fit_calls": int(last_state.get("fit_calls", 0)),
                            "target_cell": TARGET_CELL}
        _record_fit_terminal(
            route, fit_guard, fit_claim, fit_execution_id, terminal_metrics,
            succeeded=succeeded, fit_call_entered=bool(last_state.get("fit_calls")),
            returned_count=1 if succeeded else 0,
            error_type=None if succeeded else str(last_state.get("error_type") or resource_summary.get("resource_abort_reason") or "BenchmarkError"),
        )
        receipt = {
            "schema_version": TOOL_SCHEMA,
            "status": "ENGINEERING_CANARY_ONLY" if succeeded else "CANNOT_CONFIRM",
            "benchmark_execution_status": "COMPLETED" if succeeded else "CANNOT_CONFIRM",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
            "execution_id": run_id,
            "preflight_execution_id": prep_execution_id,
            "fit_execution_id": fit_execution_id,
            "contract_sha256": contract_file_sha,
            "config_sha256": contract["config_sha256"],
            "target_cell": TARGET_CELL,
            "fit_calls": int(last_state.get("fit_calls", 0)),
            "input_parity": "EXACT",
            "fit_result": result,
            "resource_monitor": resource_summary,
            "baseline_timing": contract["timing_baseline"],
            "baseline_fit_duration": {
                "status": "CANNOT_CONFIRM",
                "reason": "FROZEN_REGISTRY_TIMESTAMPS_COVER_THE_FULL_FIT_PREDICT_SAVE_REPLAY_LIFECYCLE_NOT_FIT_ONLY",
            },
            "security_flags": {"research_only": True, "production_allowed": False,
                               "buy_gate_updated": False, "orders_sent": False},
            "speedup_claim": "NOT_CLAIMED_NO_COMPARABLE_FROZEN_BASELINE_DURATION",
        }
        atomic_write_json(receipt_path, receipt)
        return receipt_path
    except BaseException as exc:
        if process is not None and process.poll() is None:
            _terminate_and_collect(process)
        status_path = run_dir / "worker_status.json"
        if status_path.is_file():
            last_state = load_json(status_path)
        if fit_claim is None and not prep_terminalized:
            metrics = {
                "fit_calls": int(last_state.get("fit_calls", 0)),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
                "resource_abort_reason": resource_summary.get("resource_abort_reason"),
            }
            if last_state.get("state") == "READY_TO_FIT":
                truth_access = "TRAIN_LABELS_READ_PAST_ONLY"
            elif last_state.get("state") == "INPUT_READ_STARTED":
                truth_access = "BOUND_INPUT_READ_OR_FEATURE_PREPARATION_FAILED"
            else:
                truth_access = "BOUND_INPUT_READ_NOT_STARTED"
            route.update_registry_execution_context(prep_guard, prep_execution_id,
                                                    truth_access=truth_access,
                                                    deterministic_metrics=metrics)
            prep_guard.append_no_fit_terminal(prep_claim, succeeded=False,
                                              error_type=type(exc).__name__)
            prep_terminalized = True
        else:
            if fit_claim is not None:
                worker_result = (load_json(run_dir / "result.json")
                                 if (run_dir / "result.json").is_file() else {})
                fit_metrics = {"fit_calls": int(last_state.get("fit_calls", 0)),
                               "error_type": type(exc).__name__, "error": str(exc)[:1000],
                               **resource_summary, **worker_result}
                _record_fit_terminal(route, fit_guard, fit_claim, fit_execution_id, fit_metrics,
                                     succeeded=False,
                                     fit_call_entered=bool(last_state.get("fit_calls")),
                                     returned_count=0, error_type=type(exc).__name__)
        atomic_write_json(receipt_path, {
            "schema_version": TOOL_SCHEMA,
            "status": "CANNOT_CONFIRM",
            "benchmark_execution_status": "CANNOT_CONFIRM",
            "evidence_level": "ENGINEERING_CANARY_ONLY",
            "execution_id": run_id,
            "preflight_execution_id": prep_execution_id,
            "fit_execution_id": fit_execution_id if fit_claim is not None else None,
            "contract_sha256": contract_file_sha,
            "fit_calls": int(last_state.get("fit_calls", 0)),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "fit_result": (load_json(run_dir / "result.json")
                           if (run_dir / "result.json").is_file() else None),
            "resource_monitor": resource_summary,
            "security_flags": {"research_only": True, "production_allowed": False,
                               "buy_gate_updated": False, "orders_sent": False},
        })
        return receipt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--execution-id")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.contract is None or args.run_dir is None:
            parser.error("--worker requires --contract and --run-dir")
        return _worker_main(args.project_root.resolve(), args.contract.resolve(), args.run_dir.resolve())
    contract_path = args.contract or (args.project_root / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/thread_benchmark_r1/CONTRACT.json")
    receipt = run_benchmark(args.project_root, contract_path, execution_id=args.execution_id)
    print(str(receipt))
    return 0 if load_json(receipt).get("benchmark_execution_status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
