"""Thin registered IO/output adapter for the frozen M26 producer.

Import and the default CLI do not read input payloads or the registry.  This
module never fits, predicts, downloads, changes original producer globals, or
admits a candidate passport.  DRAFT dependencies must be resolved separately.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import environment_tilt_early_inputs as producer

support = producer.support
FLAGS = producer.FLAGS
PRICE_MAP = dict(date="trade_date", code="ts_code", open="open", high="high",
                 low="low", close="close", volume="vol", amount="amount")
ORIGINAL_FOLDS = ("2024H1", "2024H2", "2025H1", "2025H2", "2026H1",
                  "2026H2_PARTIAL_TO_2026_10_15")
ORIGINAL_CELLS = tuple(f"M0:H{h}:{f}" for f in ORIGINAL_FOLDS for h in (1, 2, 5))


class Blocked(ValueError):
    pass


def checked_json(ref):
    return support.checked_json(ref)


def _same_ref(left, right):
    return all(left.get(k) == right.get(k) for k in ("path", "sha256"))


def _stat_matches(path, recorded):
    observed = support.stat_identity(path)
    if not recorded or any(observed.get(k) != v for k, v in recorded.items()
                           if k in ("device", "inode", "bytes", "mtime_ns")):
        raise Blocked("FAST_BIND_STAT_DRIFT_NO_REHASH_FALLBACK")
    if not {"bytes", "mtime_ns"}.issubset(recorded):
        raise Blocked("FAST_BIND_SIZE_AND_MTIME_REQUIRED")
    return observed


def authenticate_unchanged(config, name, session):
    """Exact original S1 stat binding or independently accepted M23 cache stat."""
    session.require_active()
    io = config["runtime"]["io"]
    ref = config["inputs"][name]
    if name == "old_qfq":
        s1 = checked_json(io["s1_binding"])
        original = checked_json(s1["old_binding"])
        if not _same_ref(s1["old_binding"], io["original_binding"]):
            raise Blocked("S1_ORIGINAL_BINDING_CONFLICT")
        if (not _same_ref(original["sources"]["price_panel"], ref)
                or s1["panel_stat"]["sha256"] != ref["sha256"]):
            raise Blocked("S1_PRICE_FAST_BIND_IDENTITY_CONFLICT")
        recorded = {k: s1["panel_stat"][k] for k in ("bytes", "mtime_ns")}
    elif name in ("old_calendar", "old_m0"):
        pre = checked_json(io["accepted_cache_preflight"])
        prior = checked_json(io["accepted_cache_config"])
        review = checked_json(io["accepted_cache_review"])
        key = "calendar" if name == "old_calendar" else "m0"
        if (pre.get("status") != "PREFLIGHT_READY_FOR_INDEPENDENT_TIME_USE_REVIEW"
                or review.get("verdict") != "PASS_NARROW_NEW_TIME_USE"
                or review.get("preflight_receipt_sha256") != io["accepted_cache_preflight"]["sha256"]
                or not _same_ref(prior["cache_files"][key], ref)):
            raise Blocked("ACCEPTED_CACHE_FAST_BIND_CONFLICT")
        recorded = pre["cache_stats"][key]
    else:
        raise Blocked("UNCHANGED_INPUT_NOT_ALLOWLISTED")
    return dict(name=name, path=ref["path"], sha256=ref["sha256"],
                mode="ACCEPTED_UNCHANGED_INPUT_FAST_BIND_NOT_PIT_CERTIFICATION",
                stat=_stat_matches(ref["path"], recorded))


def normalized_m24_sidecar(fit, config, review):
    if (fit.get("status") != "SINGLE_CELL_SCORES_COMPLETE_NO_PERFORMANCE_EVALUATION"
            or review.get("verdict") != "PASS_SINGLE_CELL_FIT_RECEIPT_AND_SCORE_IDENTITY"
            or fit.get("policy_sha256") != config.get("policy_sha256")
            or fit.get("metrics", {}).get("fit_calls") != 1
            or fit.get("metrics", {}).get("roundtrip_exact") is not True):
        raise Blocked("M24_ACCEPTED_INPUT_IDENTITY_REQUIRED")
    return dict(status="COMPLETED", artifact_sha256=fit["checkpoint_ref"]["sha256"],
        identity=dict(arm="M0", fold="2023H2", horizon=2,
            code_sha256={"model_selection": config["recipe_sources"]["model_selection"]["sha256"]},
            data_identity=fit["proof"]["data_identity"], stage="ACCEPTED_M24_SINGLE_CELL"))


def load_model_manifest(ref, expected_cell_ids):
    """Only metadata sidecars/accepted receipts; never open a model or scores."""
    manifest = checked_json(ref)
    cells = manifest.get("cells", {})
    original = set(ORIGINAL_CELLS)
    additional = set(manifest.get("additional_accepted_cell_ids", []))
    if (manifest.get("status") != "BOUND_COMPLETE_ACCEPTED_M0_INPUT_IDENTITIES"
            or set(manifest.get("original_cell_ids", [])) != original
            or additional != {"M0:H2:2023H2"}
            or len(expected_cell_ids) != len(set(expected_cell_ids))
            or set(cells) != original | additional or set(cells) != set(expected_cell_ids)
            or set(manifest.get("evidence_by_cell", {})) != set(cells)):
        raise Blocked("COMPLETE_18_ORIGINAL_PLUS_ACCEPTED_M24_MANIFEST_REQUIRED")
    for cell_id, evidence in manifest["evidence_by_cell"].items():
        if evidence["kind"] == "ORIGINAL_SIDECAR":
            actual = checked_json(evidence["sidecar_ref"])
        elif evidence["kind"] == "M24_ACCEPTED_RECEIPT":
            fit = checked_json(evidence["fit_ref"])
            review = checked_json(evidence["review_ref"])
            if review.get("fit_receipt_sha256") != evidence["fit_ref"]["sha256"]:
                raise Blocked("M24_REVIEW_RECEIPT_BINDING_CONFLICT")
            actual = normalized_m24_sidecar(fit, checked_json(evidence["config_ref"]), review)
        else:
            raise Blocked("UNKNOWN_MODEL_METADATA_KIND")
        if actual != cells[cell_id] or actual.get("status") != "COMPLETED":
            raise Blocked("ACCEPTED_MODEL_METADATA_DRIFT")
        identity = actual["identity"]
        if cell_id != f"M0:H{identity['horizon']}:{identity['fold']}" or identity["arm"] != "M0":
            raise Blocked("MODEL_CELL_NAME_IDENTITY_CONFLICT")
    return cells


def load_original_modules(sources):
    """Load exact frozen files without rewriting formula/fold/root globals."""
    sys.dont_write_bytecode = True
    def load(key, name):
        ref = sources[key]
        if support.sha(ref["path"]) != ref["sha256"]:
            raise Blocked("ORIGINAL_COMPONENT_DRIFT")
        existing = sys.modules.get(name)
        if existing is not None:
            if Path(existing.__file__).resolve() != Path(ref["path"]).resolve():
                raise Blocked("ORIGINAL_MODULE_IMPORT_COLLISION")
            return existing
        return support._load(ref["path"], name)
    engine = load("engine", "tdx_ast_engine_injected")
    formula = load("formula", "formula_adapter")
    recipe = load("recipe", "m29_original_recipe")
    fast = engine.fbc
    if (Path(fast.__file__).resolve() != Path(sources["fast_builder"]["path"]).resolve()
            or support.sha(fast.__file__) != sources["fast_builder"]["sha256"]):
        raise Blocked("ORIGINAL_ENGINE_RUNTIME_DRIFT")
    return dict(recipe=recipe, formula=formula, engine=engine, fast_builder=fast)


class OriginalLoader:
    """Original S1 IO and exact old FAST_BIND; changed prefix parts hash once."""
    def __init__(self):
        self.opened = False
        self.admissions = []

    def open_once(self, config, activation, session):
        session.require_active()
        if self.opened:
            raise Blocked("LOADER_NO_AUTOMATIC_REENTRY")
        self.opened = True
        io = config["runtime"]["io"]
        unit = checked_json(io["unit_ruling"])
        if (unit.get("decision") != "PASS_FOR_FROZEN_M26_UNIT_BRIDGE"
                or unit.get("old_multiplier") != 10000 or unit.get("prefix_multiplier") != 1):
            raise Blocked("M26_EFFECTIVE_UNIT_BRIDGE_NOT_REVIEWED")
        original = checked_json(io["original_binding"])
        price = original["sources"]["price_panel"]
        if (not _same_ref(price, config["inputs"]["old_qfq"])
                or price["column_map"] != PRICE_MAP
                or price["units"].get("amount_input") != "10000_CNY"
                or price["units"].get("volume") != "shares"):
            raise Blocked("EXACT_ORIGINAL_PRICE_MAP_AND_CONSUMER_SCALE_REQUIRED")
        modules = load_original_modules(config["sources"])
        import pandas as pd
        from environment_tilt_mac_recovery import configure_arrow_pools
        self.arrow_pools = configure_arrow_pools(4)
        loaded = {}
        for name, columns in (("old_qfq", list(PRICE_MAP.values())),
                              ("old_calendar", ["date", "calendar_index"]),
                              ("old_m0", ["date", "code", *producer.FEATURES])):
            binding = authenticate_unchanged(config, name, session)
            loaded[name] = pd.read_parquet(binding["path"], columns=columns)
            _stat_matches(binding["path"], binding["stat"])
            self.admissions.append(binding)
            session.check_resources()
        prefix_manifest = checked_json(config["inputs"]["prefix"])
        if (prefix_manifest.get("schema") != "M26_FULL_AVAILABLE_PREFIX_PARTS_V1"
                or prefix_manifest.get("scope") != "FULL_AVAILABLE_PRE_OLD_HISTORY"):
            raise Blocked("FULL_HISTORY_PREFIX_PART_MANIFEST_REQUIRED")
        parts, seen, count = [], set(), 0
        for part in prefix_manifest["parts"]:
            session.require_active()
            if part["symbol"] in seen:
                raise Blocked("DUPLICATE_PREFIX_SYMBOL_PART")
            seen.add(part["symbol"])
            ref = part["artifact"]
            before = support.stat_identity(ref["path"])
            if support.sha(ref["path"]) != ref["sha256"]:
                raise Blocked("CHANGED_PREFIX_PART_SHA_DRIFT")
            frame = pd.read_csv(ref["path"], usecols=list(PRICE_MAP.values()),
                                dtype={"ts_code": "string"}, float_precision="round_trip")
            if not frame.ts_code.eq(part["symbol"]).all() or len(frame) != ref["rows"]:
                raise Blocked("PREFIX_PART_SYMBOL_OR_ROW_BINDING_CONFLICT")
            _stat_matches(ref["path"], before)
            count += len(frame)
            if count + len(loaded["old_qfq"]) > config["caps"]["max_input_rows"]:
                raise Blocked("PREFIX_INPUT_ROW_BOUND")
            parts.append(frame)
            self.admissions.append(dict(path=ref["path"], sha256=ref["sha256"],
                                        mode="CHANGED_PREFIX_HASH_ONCE", stat=before))
            session.check_resources()
        prefix = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(PRICE_MAP.values()))
        if not _same_ref(config["inputs"]["industry"], io["original_binding"]):
            raise Blocked("INDUSTRY_ORIGINAL_SOURCE_BINDING_REQUIRED")
        import io as byte_io
        industry = modules["recipe"].normalize_industry_columns(pd.read_csv(byte_io.BytesIO(
            modules["recipe"].read_bound_archive_member(original["sources"]["industry_classification"]))))
        index_source = io["index_source"]
        if not _same_ref(index_source, config["inputs"]["index"]):
            raise Blocked("INDEX_BINDING_CONFLICT")
        index_called = False
        def index_provider(required):
            nonlocal index_called
            session.require_active()
            if index_called:
                raise Blocked("INDEX_PROVIDER_NO_REENTRY")
            index_called = True
            return modules["recipe"].load_bound_index_series(index_source, panel_dates=set(required))
        texts = {name: modules["formula"].exact_formula_source(Path(config["sources"][name]["path"]),
            config["sources"][name]["sha256"])[0] for name in ("v192", "v193")}
        return dict(recipe=modules["recipe"], formula=modules["formula"], engine=modules["engine"],
            source_modules=modules, texts=texts, old_raw=loaded["old_qfq"], prefix_raw=prefix,
            old_map=dict(PRICE_MAP), prefix_map=dict(PRICE_MAP),
            old_calendar=loaded["old_calendar"], old_m0=loaded["old_m0"], industry=industry,
            history_proof=checked_json(config["inputs"]["history_proof"]),
            model_bindings=load_model_manifest(config["inputs"]["model_manifest"], config["expected_cell_ids"]),
            index_provider=index_provider, admission=dict(
                scientific_identity=activation["scientific_identity"], bound_inputs=config["inputs"]))


def make_official_guard(config, config_path, receipt_path, scientific_identity):
    """Reuse the existing core+official adapter atomic claim/terminal machinery."""
    runtime = config["runtime"]
    deps = runtime["registry_dependencies"]
    for ref in deps.values():
        if support.sha(ref["path"]) != ref["sha256"]:
            raise Blocked("OFFICIAL_REGISTRY_COMPONENT_DRIFT")
    existing = sys.modules.get("execution_guard_core")
    if existing and Path(existing.__file__).resolve() != Path(deps["core"]["path"]).resolve():
        raise Blocked("OFFICIAL_CORE_IMPORT_COLLISION")
    core = existing or support._load(deps["core"]["path"], "execution_guard_core")
    module = support._load(deps["adapter"]["path"], "m29_official_adapter")
    class Adapter(module.OfficialRegistryAdapter):
        def _event_spec(self, event, *, event_type):
            spec = super()._event_spec(event, event_type=event_type)
            spec.update(input_sha256=scientific_identity,
                code_sha256=runtime["code_composite_sha256"], config_sha256=support.sha(config_path),
                truth_access="NOT_READ" if event_type == "RUN_STARTED" else
                    "PAST_TRAIN_LABELS_PER_ACCEPTED_CELL_INPUT_PARITY_ONLY_NO_TEST_PERFORMANCE")
            if receipt_path.exists():
                spec["input_identity"]["phase_receipt_sha256"] = support.sha(receipt_path)
            return spec
    profile = dict(track_id="A_SHARE_PANEL_MODEL_SELECTION_RESEARCH",
        route_id="a-share-environment-tilt-original-producer-early-inputs",
        registration_mode="LIVE_EXECUTION", start_stage="M26_INPUT_PREPARATION_STARTED",
        terminal_stage="M26_INPUT_PREPARATION_TERMINAL", occurred_at_reason="LIVE_EXECUTION_UTC",
        baseline_id="ORIGINAL_S1_M0_FORMAL_ACCEPTED_18_PLUS_M24_INPUT_IDENTITIES",
        one_variable="SEPARATE_CERTIFIED_FULL_PRE_OLD_HISTORY_ORIGINAL_PRODUCER_INPUT_VERSION",
        signal_scope="2020_2023_EXPORT_PLUS_UNCHANGED_OLD_OVERLAP",
        target_scope="PAST_TRAIN_LABELS_ONLY_FOR_ACCEPTED_MODEL_PARITY",
        device="CPU_RESEARCH_ONLY", evidence_level="DIRECTION_ONLY")
    adapter = Adapter(registry_path=Path(runtime["registry_path"]),
        contract_path=Path(runtime["contract"]["path"]), input_binding_path=Path(config_path),
        artifact_paths=(receipt_path,), builder_path=Path(deps["builder"]["path"]),
        validator_path=Path(deps["validator"]["path"]), event_profile=profile)
    return core.ExecutionGuard(adapter, experiment_id="A_SHARE_ENV_TILT_EARLY_INPUTS_M26",
                               max_real_fit_debits=0)


class RegisteredSession:
    def __init__(self, config, activation, guard, claim):
        self.config, self.guard, self.claim = config, guard, claim
        self.execution_id = activation["execution_id"]
        self.root = Path(config["output_root"])
        self.closed = False

    def require_active(self):
        rows = [e for e in self.guard.backend.list_events(self.guard.experiment_id)
                if e["execution_id"] == self.execution_id]
        if (self.closed or len(rows) != 1 or rows[0]["event_type"] != "RUN_STARTED"
                or rows[0]["contract_sha256"] != self.claim.contract_sha256
                or rows[0]["input_binding_sha256"] != self.claim.input_binding_sha256):
            raise Blocked("DURABLE_ACTIVE_REGISTERED_SESSION_REQUIRED")

    def output_exists(self):
        return any(p.name not in (".owner.lock", "RECEIPT.json") for p in self.root.iterdir())

    def check_resources(self):
        return producer.NativeBudget(self.config).check("runtime_io_boundary")


def _metadata(value):
    """Keep frame payloads out of JSON provenance, not out of candidate files."""
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return dict(rows=len(value), columns=list(value.columns), frame_values="NOT_EMBEDDED")
    if isinstance(value, dict):
        return {str(k): _metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def publish_candidate(result, config):
    """Persist an isolated candidate only on full parity; never active admission."""
    root = Path(config["output_root"])
    refs = {}
    def record(path):
        refs[path.name] = dict(path=str(path), sha256=support.sha(path), bytes=path.stat().st_size)
        if sum(p.stat().st_size for p in root.iterdir() if p.is_file()) > config["runtime"]["max_output_bytes"]:
            raise Blocked("CANDIDATE_OUTPUT_BYTE_LIMIT")
    for name, body in (("PARITY.json", result["parity"]),
                       ("PROVENANCE.json", dict(composition=result["composition"],
                           candidate=result["candidate"]["provenance"]))):
        path = root / name
        if path.exists():
            raise Blocked("OUTPUT_EXISTS_NO_OVERWRITE")
        support.write_json(path, _metadata(body)); record(path)
    if result["parity"]["status"] == "INPUT_VERSION_PARITY_PASS":
        for name, key in (("CANONICAL_PANEL.parquet", "panel"), ("CALENDAR.parquet", "calendar"),
                          ("M0_FULL_COMMON.parquet", "m0"), ("M0_EARLY_2020_2023.parquet", "early_m0")):
            path = root / name
            with path.open("xb") as handle:
                result["candidate"][key].to_parquet(handle, index=False)
                handle.flush(); os.fsync(handle.fileno())
            record(path)
    manifest = dict(status="CANDIDATE_ONLY_PENDING_INDEPENDENT_REVIEW" if
        result["parity"]["status"] == "INPUT_VERSION_PARITY_PASS" else "INPUT_VERSION_PARITY_FAILED_NO_PUBLISH",
        artifacts=dict(refs), scientific_identity=result["scientific_identity"], active_passport_admitted=False,
        input_versions_mixed=False, old_V1_preserved=True, base_fits=0, **FLAGS)
    support.write_json(root / "OUTPUT_MANIFEST.json", manifest)
    record(root / "OUTPUT_MANIFEST.json")
    return manifest


def run_registered(config_path, activation_path):
    """Future explicit entry; DRAFT config rejects before output/registry/data IO."""
    config_path, activation_path = Path(config_path), Path(activation_path)
    config = json.loads(config_path.read_text())
    activation = json.loads(activation_path.read_text())
    identity = producer.require_activation(config, activation)
    runtime = config["runtime"]
    if activation.get("config_sha256") != support.sha(config_path):
        raise Blocked("WHOLE_CONFIG_ACTIVATION_REQUIRED_INCLUDING_IO_AND_LIFECYCLE")
    if (runtime.get("state") != "FROZEN" or any(type(runtime.get(k)) is not int or runtime[k] <= 0
            for k in ("max_output_bytes", "output_wall_seconds"))):
        raise Blocked("FINAL_RUNTIME_AND_OUTPUT_BUDGET_REQUIRED")
    if support.sha(__file__) != config["sources"]["loader"]["sha256"]:
        raise Blocked("RUNTIME_SOURCE_IDENTITY_DRIFT")
    checked_json(runtime["contract"]); checked_json(runtime["review_receipt"])
    if activation.get("review_receipt_sha256") != runtime["review_receipt"]["sha256"]:
        raise Blocked("RUNTIME_REVIEW_BINDING_CONFLICT")
    root = Path(config["output_root"])
    root.mkdir(exist_ok=False)
    with (root / ".owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rp = root / "RECEIPT.json"
        support.write_json(rp, dict(status="INTENT_NOT_STARTED", **FLAGS))
        guard = make_official_guard(config, config_path, rp, identity)
        claim = guard.claim_no_fit(execution_id=activation["execution_id"], arm="ORIGINAL_M26_INPUTS_NO_FIT",
            contract_sha256=runtime["contract"]["sha256"], input_binding_sha256=support.sha(config_path))
        session = RegisteredSession(config, activation, guard, claim)
        loader = OriginalLoader()
        started = time.monotonic()
        report = dict(status="RUNNING", execution_id=session.execution_id, scientific_identity=identity, **FLAGS)
        error = None
        try:
            result = producer.prepare_registered(config, activation, session, loader)
            with support.deadline(runtime["output_wall_seconds"]):
                manifest = publish_candidate(result, config)
            report.update(status=result["parity"]["status"], parity=result["parity"],
                output_manifest=manifest, admissions=loader.admissions, arrow_pools=loader.arrow_pools,
                resource_after=session.check_resources(), fits=0, performance_evaluation=False)
        except BaseException as exc:
            error = exc
            report.update(status="FAILED_NO_RETRY", error_type=type(exc).__name__, error=str(exc)[:1000])
        finally:
            report.update(elapsed_seconds=time.monotonic() - started, active_passport_admitted=False)
            support.write_json(rp, _metadata(report))
            try:
                guard.append_no_fit_terminal(claim, succeeded=report["status"] == "INPUT_VERSION_PARITY_PASS",
                    error_type=type(error).__name__ if error else None)
                session.closed = True
            except BaseException as terminal_error:
                report["terminal_status"] = "CANNOT_CONFIRM"
                report["terminal_error"] = repr(terminal_error)
                support.write_json(rp, _metadata(report))
                raise
        if error is not None:
            raise error
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        config = json.loads(args.config.read_text())
        print(json.dumps(dict(status="M29_METADATA_ONLY_NOT_ACTIVATED", config_state=config.get("state"))))
        return
    if args.activation is None:
        parser.error("--execute requires --activation")
    print(support.canonical(_metadata(run_registered(args.config, args.activation))))


if __name__ == "__main__":
    main()
