"""Registered FAST_BIND, real-data canary, and full descriptive run.

This orchestration is bound separately from the statistics implementation.
It uses the existing locked official registry adapter and the B0 reader.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path("/Users/lexi/Desktop/学习/因子工厂")
WORKTREE = ROOT / "workspace/.task_worktrees/a-top100-environment-20261019"
MODULE_DIR = WORKTREE / "research/a_share_top100_environment_v1"
PREVIOUS_BATCH = ROOT / "research/a_share_top100_environment_v1/batches/seasonality_r1_20261019"
BATCH = PREVIOUS_BATCH / "attempt_02"
OUTPUTS = BATCH / "outputs"
CONTRACT_PATHS = {
    "A": BATCH / "CONTRACT_A.json",
    "B": BATCH / "CONTRACT_B.json",
}
CONFIG_PATHS = {
    "A": PREVIOUS_BATCH / "CONFIG_A.json",
    "B": PREVIOUS_BATCH / "CONFIG_B.json",
}
BINDING_PATH = PREVIOUS_BATCH / "INPUT_BINDING.json"
IDENTITY_PATH = BATCH / "FROZEN_IDENTITY.json"
FAST_BIND_REUSE_RECEIPT_PATH = BATCH / "FAST_BIND_REUSE_RECEIPT.json"
RUNNER_PATH = Path(__file__).resolve()
ANALYSIS_PATH = MODULE_DIR / "seasonality_stats.py"
TEST_PATH = MODULE_DIR / "tests/test_seasonality_stats.py"
READER_PATH = MODULE_DIR / "top100_environment.py"
EXECUTION_ID_PREFIX = "A_SHARE_TOP100_SEASONALITY_INDUSTRY_POSITION_R1_20261019"
CANARY_DATE = "2021-08-02"

sys.path.insert(0, str(MODULE_DIR))
import seasonality_stats as stats  # noqa: E402
import top100_environment as env  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing execution artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return hashlib.sha256(raw).hexdigest()


def _ensure_frozen_identity() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = _read_json(IDENTITY_PATH)
    binding = _read_json(BINDING_PATH)
    contracts = {key: _read_json(path) for key, path in CONTRACT_PATHS.items()}
    source_contract_path = Path(binding["source_binding_path"]).parent / "CONTRACT.json"
    source_config_path = Path(binding["source_binding_path"]).parent / "CONFIG.json"
    actual = {
        "code_sha256": identity.get("code_sha256"),
        "analysis_module_sha256": sha256_file(ANALYSIS_PATH),
        "execution_runner_sha256": sha256_file(RUNNER_PATH),
        "test_sha256": sha256_file(TEST_PATH),
        "reused_r1_reader_sha256": sha256_file(READER_PATH),
        "input_binding_sha256": sha256_file(BINDING_PATH),
        "config_a_sha256": sha256_file(CONFIG_PATHS["A"]),
        "config_b_sha256": sha256_file(CONFIG_PATHS["B"]),
        "contract_a_sha256": sha256_file(CONTRACT_PATHS["A"]),
        "contract_b_sha256": sha256_file(CONTRACT_PATHS["B"]),
        "composite_contract_sha256": sha256_file(BATCH / "CONTRACT.json"),
        "source_input_binding_sha256": sha256_file(Path(binding["source_binding_path"])),
        "source_config_sha256": sha256_file(source_config_path),
        "source_contract_sha256": sha256_file(source_contract_path),
    }
    code_bundle = canonical_sha256({
        "analysis_module": actual["analysis_module_sha256"],
        "execution_runner": actual["execution_runner_sha256"],
        "reused_r1_reader": actual["reused_r1_reader_sha256"],
    })
    actual["code_sha256"] = code_bundle
    for key, value in actual.items():
        if identity.get(key) != value:
            raise RuntimeError(f"frozen identity mismatch: {key}")

    for key, contract in contracts.items():
        expected_variable = identity.get("one_variables", {}).get(contract.get("route_id"))
        expected = {
            "one_variable": expected_variable,
            "analysis_code_sha256": actual["analysis_module_sha256"],
            "execution_runner_sha256": actual["execution_runner_sha256"],
            "code_sha256": actual["code_sha256"],
            "test_sha256": actual["test_sha256"],
            "input_binding_sha256": actual["input_binding_sha256"],
            "config_sha256": actual[f"config_{key.lower()}_sha256"],
        }
        if contract.get("route_id") not in identity.get("one_variables", {}):
            raise RuntimeError(f"{key} route is absent from frozen one-variable map")
        for field, value in expected.items():
            if contract.get(field) != value:
                raise RuntimeError(f"{key} contract identity mismatch: {field}")
    expected_reader = stats.BOUND_TOP100_ENVIRONMENT_SHA256
    if actual["reused_r1_reader_sha256"] != expected_reader:
        raise RuntimeError("R1 reader no longer matches the bound frozen SHA")

    # These values are reused from the prior receipt. The full panel is deliberately
    # not rehashed here; the bound reader validates it once after canary registration.
    sources = binding["sources"]
    reused_input_shas = {
        "price_panel": sources["price_panel"]["sha256"],
        "shared_calendar_member": sources["trading_date_index"]["member_sha256"],
        "industry_classification_member": sources["industry_classification"]["member_sha256"],
        "index_series": sources["index_series"]["sha256"],
    }
    if identity.get("input_sha256_reused_without_rehash") != reused_input_shas:
        raise RuntimeError("bound source input SHA mapping drifted")
    return identity, binding, contracts


def _setup_routes(identity: Mapping[str, Any], binding: Mapping[str, Any],
                  contracts: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for key in ("A", "B"):
        contract_path = CONTRACT_PATHS[key]
        config_path = CONFIG_PATHS[key]
        adapter, core = env.load_official_registry(
            binding,
            contract_path=contract_path,
            input_binding_path=BINDING_PATH,
            config_path=config_path,
            code_path=ANALYSIS_PATH,
            test_path=TEST_PATH,
        )
        adapter.event_profile["route_id"] = str(contracts[key]["route_id"])
        adapter.event_profile["one_variable"] = str(contracts[key]["one_variable"])
        if adapter.registry_path != Path(binding["registry_path"]).resolve():
            raise RuntimeError("official adapter resolved a different registry path")
        routes[key] = {
            "adapter": adapter,
            "core": core,
            "contract": dict(contracts[key]),
            "contract_path": contract_path,
            "config_path": config_path,
            "route_id": str(contracts[key]["route_id"]),
            "one_variable": str(contracts[key]["one_variable"]),
            "experiment_id": str(contracts[key]["experiment_id"]),
        }
    experiment_ids = {route["experiment_id"] for route in routes.values()}
    if len(experiment_ids) != 1:
        raise RuntimeError("A/B contracts must share the frozen experiment batch identity")
    return routes


def _hashes(identity: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, str]:
    key = "A" if route["route_id"] == _read_json(CONTRACT_PATHS["A"])["route_id"] else "B"
    return {
        "contract_sha256": sha256_file(route["contract_path"]),
        "input_binding_sha256": sha256_file(BINDING_PATH),
        "config_sha256": sha256_file(route["config_path"]),
        "code_sha256": str(identity["code_sha256"]),
        "test_sha256": str(identity["test_sha256"]),
    }


def _context(route: Mapping[str, Any], identity: Mapping[str, Any], binding: Mapping[str, Any],
             *, stage: str, signal_range: Mapping[str, Any] | None,
             metrics: Mapping[str, Any] | None, terminal: bool,
             output_paths: Sequence[Path] = ()) -> dict[str, Any]:
    hashes = _hashes(identity, route)
    contract = dict(route["contract"])
    contract.update({
        "contract_path": str(route["contract_path"]),
        "config_path": str(route["config_path"]),
        "frozen_identity_path": str(IDENTITY_PATH),
    })
    context = env.build_registry_context(
        stage=stage,
        contract=contract,
        binding=binding,
        hashes=hashes,
        code_path=ANALYSIS_PATH,
        test_path=TEST_PATH,
        baseline_path=Path(binding["baseline_record_path"]),
        output_paths=tuple(output_paths),
        signal_range=signal_range,
        metrics=metrics,
        terminal=terminal,
    )
    context["code_sha256"] = identity["code_sha256"]
    context["code_sha256_by_file"] = {
        str(ANALYSIS_PATH.resolve()): identity["analysis_module_sha256"],
        str(RUNNER_PATH.resolve()): identity["execution_runner_sha256"],
        str(READER_PATH.resolve()): identity["reused_r1_reader_sha256"],
        str(TEST_PATH.resolve()): identity["test_sha256"],
    }
    context["input_component_sha256"]["seasonality_analysis_module"] = identity["analysis_module_sha256"]
    context["input_component_sha256"]["seasonality_execution_runner"] = identity["execution_runner_sha256"]
    context["input_component_sha256"]["reused_top100_environment_reader"] = identity["reused_r1_reader_sha256"]
    context["route_id"] = route["route_id"]
    context["one_variable"] = route["one_variable"]
    context["experiment_id"] = route["experiment_id"]
    if stage in {"CANARY", "FULL_STATS"}:
        reuse_path = str(FAST_BIND_REUSE_RECEIPT_PATH.resolve())
        context["input_paths"] = list(dict.fromkeys([*context["input_paths"], reuse_path]))
        context["input_component_sha256"]["reused_fastbind_receipt"] = identity["reused_fastbind_receipt_sha256"]
    return context


def _execution_id(route: Mapping[str, Any], stage: str) -> str:
    return str(route["contract"]["execution_ids"][stage])


def _set_experiment(route: Mapping[str, Any]) -> None:
    env.EXPERIMENT_ID = route["experiment_id"]


def _start_pair(routes: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any],
                binding: Mapping[str, Any], stage: str, signal_range: Mapping[str, Any] | None,
                metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    claims: dict[str, Any] = {}
    try:
        for key in ("A", "B"):
            route = routes[key]
            _set_experiment(route)
            execution_id = _execution_id(route, stage)
            context = _context(route, identity, binding, stage=stage, signal_range=signal_range,
                               metrics=metrics, terminal=False)
            claim = env._claim_stage(
                route["adapter"], route["core"], stage=stage, execution_id=execution_id,
                contract_sha256=context["contract_sha256"],
                input_binding_sha256=context["input_binding_sha256"],
                context=context,
            )
            claims[key] = claim
    except BaseException as exc:
        for key, claim in claims.items():
            try:
                route = routes[key]
                _set_experiment(route)
                context = _context(route, identity, binding, stage=stage, signal_range=signal_range,
                                   metrics=metrics, terminal=True)
                route["adapter"].set_route_context(claim.execution_id, context)
                env._terminalize_stage(route["adapter"], claim, terminal_status="FAILED_NO_FIT", metrics={}, error=exc)
            except BaseException:
                pass
        raise
    return claims


def _terminalize_pair(routes: Mapping[str, Mapping[str, Any]], claims: Mapping[str, Any],
                      identity: Mapping[str, Any], binding: Mapping[str, Any],
                      stage: str, signal_range: Mapping[str, Any] | None,
                      *, terminal_status: str, metrics: Mapping[str, Any],
                      output_paths: Sequence[Path], error: BaseException | None = None) -> None:
    errors = []
    for key in ("A", "B"):
        route = routes[key]
        claim = claims[key]
        try:
            _set_experiment(route)
            context = _context(route, identity, binding, stage=stage, signal_range=signal_range,
                               metrics=metrics, terminal=True, output_paths=output_paths)
            route["adapter"].set_route_context(claim.execution_id, context)
            context = route["adapter"]._route_context[claim.execution_id]
            terminal_context = dict(context)
            deterministic = dict(context["deterministic_metrics"])
            deterministic.update(metrics)
            deterministic.update({
                "execution_kind": "NO_FIT_ENGINEERING_CANARY",
                "planned_real_fit_count": 0,
                "real_fit_budget_debit": 0,
                "real_fit_returned_count": 0,
                "fit_call_entered": False,
            })
            terminal_context["deterministic_metrics"] = deterministic
            terminal_context["truth_access"] = "NOT_OPENED" if stage == "FAST_BIND" else "OPENED_EXPOSED_HISTORY"
            terminal_context["sealed_status"] = "NOT_ENTERED" if stage == "FAST_BIND" else "EXPOSED"
            terminal_context["null_reasons"] = dict(terminal_context.get("null_reasons", {}))
            terminal_context["null_reasons"].pop("signal_date_range", None)
            route["adapter"].set_route_context(claim.execution_id, terminal_context)
            route["adapter"].terminalize(claim, {
                "experiment_id": claim.experiment_id,
                "execution_id": claim.execution_id,
                "arm": claim.arm,
                "contract_sha256": claim.contract_sha256,
                "input_binding_sha256": claim.input_binding_sha256,
                "execution_kind": claim.execution_kind,
                "terminal_status": terminal_status,
                "real_fit_returned_count": 0,
                "fit_call_entered": False,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_type": None if error is None else f"{type(error).__name__}:{str(error)[:240]}",
            })
        except BaseException as exc:
            errors.append(f"{key}:{type(exc).__name__}:{str(exc)[:180]}")
    if errors:
        raise RuntimeError("one or more route terminal events failed: " + "; ".join(errors))


def _verify_terminal_pair(routes: Mapping[str, Mapping[str, Any]], stage: str) -> None:
    for key in ("A", "B"):
        route = routes[key]
        adapter = route["adapter"]
        events = list(adapter.list_events(route["experiment_id"]))
        execution_id = _execution_id(route, stage)
        matching = [event for event in events if event.get("execution_id") == execution_id]
        starts = [event for event in matching if event.get("event_type") == "RUN_STARTED"]
        terminals = [event for event in matching if event.get("event_type") == "TERMINAL"]
        if len(starts) != 1 or len(terminals) != 1 or terminals[0].get("terminal_status") != "COMPLETED":
            raise RuntimeError(f"{key} {stage} does not have exactly one COMPLETED registered lifecycle")
        if starts[0].get("route_id") != route["route_id"] or terminals[0].get("route_id") != route["route_id"]:
            raise RuntimeError(f"{key} {stage} event was written under the wrong route")


def _verify_reused_fastbind(routes: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any]) -> None:
    receipt = _read_json(FAST_BIND_REUSE_RECEIPT_PATH)
    if sha256_file(FAST_BIND_REUSE_RECEIPT_PATH) != identity["reused_fastbind_receipt_sha256"]:
        raise RuntimeError("reused FAST_BIND receipt SHA mismatch")
    if receipt.get("input_binding_sha256") != identity["input_binding_sha256"]:
        raise RuntimeError("reused FAST_BIND input binding differs from the frozen retry")
    prior_identity = _read_json(PREVIOUS_BATCH / "FROZEN_IDENTITY.json")
    if receipt.get("source_code_sha256") != prior_identity.get("code_sha256"):
        raise RuntimeError("reused FAST_BIND source identity mismatch")
    for key in ("A", "B"):
        route = routes[key]
        execution_id = _execution_id(route, "FAST_BIND")
        expected = receipt["route_bindings"][key]
        if expected.get("execution_id") != execution_id or expected.get("route_id") != route["route_id"]:
            raise RuntimeError(f"{key} reused FAST_BIND route identity mismatch")
        events = [e for e in route["adapter"].list_events(route["experiment_id"])
                  if e.get("execution_id") == execution_id]
        starts = [e for e in events if e.get("event_type") == "RUN_STARTED"]
        terminals = [e for e in events if e.get("event_type") == "TERMINAL"]
        if len(starts) != 1 or len(terminals) != 1:
            raise RuntimeError(f"{key} reused FAST_BIND lifecycle is incomplete or duplicated")
        start, terminal = starts[0], terminals[0]
        for event in (start, terminal):
            if event.get("route_id") != route["route_id"]:
                raise RuntimeError(f"{key} reused FAST_BIND event route mismatch")
            if event.get("input_binding_sha256") != identity["input_binding_sha256"]:
                raise RuntimeError(f"{key} reused FAST_BIND input binding mismatch")
            if event.get("config_sha256") != expected["config_sha256"]:
                raise RuntimeError(f"{key} reused FAST_BIND config mismatch")
            if event.get("code_sha256") != receipt["source_code_sha256"]:
                raise RuntimeError(f"{key} reused FAST_BIND code identity mismatch")
        if terminal.get("decision") != "COMPLETED" or terminal.get("contract_sha256") != expected["contract_sha256"]:
            raise RuntimeError(f"{key} prior FAST_BIND was not a matching completed execution")
        if start.get("event_sha256") != expected["run_started_event_sha256"] or terminal.get("event_sha256") != expected["terminal_event_sha256"]:
            raise RuntimeError(f"{key} reused FAST_BIND event SHA mismatch")


def _write_fastbind_receipt(identity: Mapping[str, Any], binding: Mapping[str, Any],
                            checks: Mapping[str, Any]) -> Path:
    path = OUTPUTS / "FAST_BIND_RECEIPT.json"
    _write_new_json(path, {
        "schema_version": "A_SHARE_TOP100_SEASONALITY_FAST_BIND_RECEIPT_V1",
        "status": "PASS",
        "identity_checks": dict(checks),
        "identity_sha256": sha256_file(IDENTITY_PATH),
        "input_binding_sha256": identity["input_binding_sha256"],
        "source_input_sha256_reused_without_rehash": identity["input_sha256_reused_without_rehash"],
        "raw_panel_read": False,
        "raw_panel_rehashed": False,
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "safety_flags": {
            "research_only": True,
            "production_allowed": False,
            "buy_gate_updated": False,
            "orders_sent": False,
        },
    })
    return path


def _run_fastbind(routes: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any],
                  binding: Mapping[str, Any]) -> None:
    signal_range = {
        "count": int(binding["sources"]["price_panel"]["dates"]["observed_date_count"]),
        "min": str(binding["sources"]["price_panel"]["dates"]["first"]),
        "max": str(binding["sources"]["price_panel"]["dates"]["last"]),
        "scope": "BOUND_METADATA_ONLY_BEFORE_RAW_PANEL_READ",
    }
    claims = _start_pair(routes, identity, binding, "FAST_BIND", signal_range,
                         {"selection_slots": 0, "source_rows": None, "valid_pool_observations": None,
                          "eligible_date_count": 0})
    try:
        checks = {
            "analysis_module_sha256_matches": sha256_file(ANALYSIS_PATH) == identity["analysis_module_sha256"],
            "execution_runner_sha256_matches": sha256_file(RUNNER_PATH) == identity["execution_runner_sha256"],
            "test_sha256_matches": sha256_file(TEST_PATH) == identity["test_sha256"],
            "reused_r1_reader_sha256_matches": sha256_file(READER_PATH) == identity["reused_r1_reader_sha256"],
            "input_binding_sha256_matches": sha256_file(BINDING_PATH) == identity["input_binding_sha256"],
            "config_a_sha256_matches": sha256_file(CONFIG_PATHS["A"]) == identity["config_a_sha256"],
            "config_b_sha256_matches": sha256_file(CONFIG_PATHS["B"]) == identity["config_b_sha256"],
            "contract_a_sha256_matches": sha256_file(CONTRACT_PATHS["A"]) == identity["contract_a_sha256"],
            "contract_b_sha256_matches": sha256_file(CONTRACT_PATHS["B"]) == identity["contract_b_sha256"],
            "source_contract_sha256_matches": sha256_file(Path(binding["source_binding_path"]).parent / "CONTRACT.json") == identity["source_contract_sha256"],
            "source_config_sha256_matches": sha256_file(Path(binding["source_binding_path"]).parent / "CONFIG.json") == identity["source_config_sha256"],
            "source_input_binding_sha256_matches": sha256_file(Path(binding["source_binding_path"])) == identity["source_input_binding_sha256"],
            "frozen_configs_match_analysis_constants": False,
            "bound_reader_matches_frozen_constant": stats.BOUND_TOP100_ENVIRONMENT_SHA256 == identity["reused_r1_reader_sha256"],
            "all_safety_flags_research_only": binding["safety_flags"] == {
                "research_only": True, "production_allowed": False,
                "buy_gate_updated": False, "orders_sent": False,
            },
        }
        stats.validate_frozen_configs(CONFIG_PATHS["A"], CONFIG_PATHS["B"])
        checks["frozen_configs_match_analysis_constants"] = True
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise RuntimeError("FAST_BIND identity checks failed: " + ", ".join(failed))
        # The existing B0 helper's dependency SHA guards run before any raw panel read.
        env.load_bound_helpers(binding)
        receipt = _write_fastbind_receipt(identity, binding, checks)
        _terminalize_pair(routes, claims, identity, binding, "FAST_BIND", signal_range,
                          terminal_status="COMPLETED",
                          metrics={"identity_checks_passed": True, "selection_slots": 0,
                                   "source_rows": None, "valid_pool_observations": None,
                                   "eligible_date_count": 0},
                          output_paths=(receipt,))
    except BaseException as exc:
        try:
            _terminalize_pair(routes, claims, identity, binding, "FAST_BIND", signal_range,
                              terminal_status="FAILED_NO_FIT", metrics={}, output_paths=(), error=exc)
        except BaseException:
            pass
        raise
    _verify_terminal_pair(routes, "FAST_BIND")


def _canary_previous_date(calendar: pd.DataFrame, canary: pd.Timestamp) -> pd.Timestamp:
    ordered = calendar.sort_values("calendar_index", kind="mergesort").reset_index(drop=True)
    dates = pd.to_datetime(ordered["date"], errors="raise").dt.normalize()
    matches = np.flatnonzero(dates.eq(canary).to_numpy())
    if len(matches) != 1 or int(matches[0]) == 0:
        raise RuntimeError("frozen canary date has no unique immediately preceding shared session")
    return pd.Timestamp(dates.iloc[int(matches[0]) - 1]).normalize()


def _run_canary(routes: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any],
                binding: Mapping[str, Any], helper: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                  pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                  dict[str, Any]]:
    signal_range = {
        "count": 1,
        "min": CANARY_DATE,
        "max": CANARY_DATE,
        "scope": "FROZEN_SINGLE_DATE_REAL_DATA_CANARY",
    }
    claims = _start_pair(routes, identity, binding, "CANARY", signal_range,
                         {"selection_slots": 0, "source_rows": None, "valid_pool_observations": None,
                          "eligible_date_count": 0})
    try:
        panel, panel_meta = env.read_frozen_panel(binding, helper)
        calendar, industry, index_close, aux_meta = env.read_frozen_auxiliary_inputs(binding, helper, panel)
        canary_date = pd.Timestamp(CANARY_DATE).normalize()
        previous_date = _canary_previous_date(calendar, canary_date)
        sample = panel.loc[panel["date"].isin({previous_date, canary_date})].copy()
        observations, coverage, coverage_year = env.prepare_observations(sample, calendar, industry, index_close)
        observed_date = observations.loc[observations["date"].eq(canary_date)]
        canary_projection = stats.build_daily_projection(
            observed_date, calendar, index_close, selector=env.select_top100,
        )
        canary_day = canary_projection.loc[canary_projection["date"].eq(canary_date)].copy()
        if canary_day.empty or not bool(canary_day["eligible_top100_date"].all()):
            raise RuntimeError("frozen canary date did not have an eligible Top100 universe")
        hits = int(canary_day["top100_hits"].sum())
        expected = float(canary_day["expected_top100_hits"].sum())
        valid_pool_observations = int(canary_day["date_fullpool_n"].iloc[0])
        if hits != 100 or not np.isclose(expected, 100.0, rtol=0, atol=1e-9):
            raise RuntimeError(f"canary Top100 reconciliation failed: hits={hits}, expected={expected}")
        if not bool(canary_day.loc[canary_day["industry_n"].eq(0), [
            "industry_return_sum", "industry_return_square_sum", "industry_mean_return",
            "industry_median_return", "industry_sample_std_return", "industry_excess_vs_pool",
        ]].isna().all().all()):
            raise RuntimeError("canary zero-N cells did not remain null")
        canary_position = stats.build_position_states(canary_projection, calendar)
        position_unknown = bool(
            canary_position["position_state_asof_t_minus_1"].eq("UNKNOWN").all()
            and canary_position["rs20_state_asof_t_minus_1"].eq("UNKNOWN").all()
            and canary_position["rs60_state_asof_t_minus_1"].eq("UNKNOWN").all()
        )
        if not position_unknown:
            raise RuntimeError("canary state warmup did not remain UNKNOWN")
        day_coverage = coverage.loc[coverage["date"].eq(canary_date)]
        if len(day_coverage) != 1:
            raise RuntimeError("canary coverage row is missing or duplicated")
        coverage_row = day_coverage.iloc[0]
        metrics = {
            "date": CANARY_DATE,
            "previous_shared_calendar_date": previous_date.date().isoformat(),
            "panel_source_rows": int(panel_meta["source_rows"]),
            "sample_source_rows": int(len(sample)),
            "valid_pool_observations": valid_pool_observations,
            "selection_slots": hits,
            "top100_hits": hits,
            "expected_top100_hits": expected,
            "industry_count_including_unknown": int(canary_day["industry_code"].nunique()),
            "position_and_rs_unknown_when_warmup_insufficient": position_unknown,
            "same_day_000300_return_available": bool(canary_day["same_day_000300_return"].notna().all()),
            "lag_market_values": sorted(canary_day["lag_market"].astype(str).unique().tolist()),
            "coverage_source_rows": int(coverage_row["source_rows"]),
            "coverage_valid_pool_observations": int(coverage_row["valid_pool_observations"]),
            "quality_checks": {
                "strict_top100_hits_equal_100": hits == 100,
                "daily_expected_hits_equal_100": bool(np.isclose(expected, 100.0, rtol=0, atol=1e-9)),
                "zero_n_return_statistics_null": True,
                "t_minus_1_state_unknown_when_252_20_60_warmup_unavailable": position_unknown,
            },
            "panel_meta": panel_meta,
            "auxiliary_meta": aux_meta,
            "coverage_year_rows_in_two_date_canary_slice": int(len(coverage_year)),
            "safety_flags": {
                "research_only": True,
                "production_allowed": False,
                "buy_gate_updated": False,
                "orders_sent": False,
            },
        }
        if not all(metrics["quality_checks"].values()):
            raise RuntimeError("one or more frozen canary quality checks failed")
        canary_output = OUTPUTS / "canary"
        if canary_output.exists() and any(canary_output.iterdir()):
            raise FileExistsError(f"canary output directory is not empty: {canary_output}")
        canary_output.mkdir(parents=True, exist_ok=True)
        env.write_csv(canary_output / "CANARY_DAILY_INDUSTRY.csv", canary_day)
        env.write_csv(canary_output / "CANARY_POSITION_STATES.csv", canary_position)
        receipt_path = canary_output / "CANARY_RECEIPT.json"
        env.write_json(receipt_path, {
            "schema_version": "A_SHARE_TOP100_SEASONALITY_CANARY_RECEIPT_V1",
            "status": "PASS",
            "identity": {
                "frozen_identity_sha256": sha256_file(IDENTITY_PATH),
                "code_sha256": identity["code_sha256"],
                "input_binding_sha256": identity["input_binding_sha256"],
                "contract_a_sha256": identity["contract_a_sha256"],
                "contract_b_sha256": identity["contract_b_sha256"],
            },
            "metrics": metrics,
            "output_sha256": {
                "CANARY_DAILY_INDUSTRY.csv": sha256_file(canary_output / "CANARY_DAILY_INDUSTRY.csv"),
                "CANARY_POSITION_STATES.csv": sha256_file(canary_output / "CANARY_POSITION_STATES.csv"),
            },
        })
        output_paths = (
            canary_output / "CANARY_DAILY_INDUSTRY.csv",
            canary_output / "CANARY_POSITION_STATES.csv",
            receipt_path,
        )
        event_metrics = {
            "selection_slots": hits,
            "source_rows": int(coverage_row["source_rows"]),
            "valid_pool_observations": valid_pool_observations,
            "eligible_date_count": 1,
            "top100_hits": hits,
            "expected_top100_hits": expected,
            "position_and_rs_unknown_when_warmup_insufficient": position_unknown,
        }
        _terminalize_pair(routes, claims, identity, binding, "CANARY", signal_range,
                          terminal_status="COMPLETED", metrics=event_metrics,
                          output_paths=output_paths)
        _verify_terminal_pair(routes, "CANARY")
        return panel, calendar, industry, index_close, canary_day, canary_position, metrics
    except BaseException as exc:
        try:
            _terminalize_pair(routes, claims, identity, binding, "CANARY", signal_range,
                              terminal_status="FAILED_NO_FIT", metrics={}, output_paths=(), error=exc)
        except BaseException:
            pass
        raise


def _run_full(routes: Mapping[str, Mapping[str, Any]], identity: Mapping[str, Any],
              binding: Mapping[str, Any], panel: pd.DataFrame, calendar: pd.DataFrame,
              industry: pd.DataFrame, index_close: pd.Series, canary_metrics: Mapping[str, Any]) -> dict[str, Any]:
    _verify_terminal_pair(routes, "CANARY")
    signal_range = {
        "count": int(panel["date"].nunique()),
        "min": panel["date"].min().date().isoformat(),
        "max": panel["date"].max().date().isoformat(),
        "scope": "ALL_BOUND_OBSERVED_PANEL_DATES",
    }
    claims = _start_pair(routes, identity, binding, "FULL_STATS", signal_range,
                         {"selection_slots": 0, "source_rows": int(len(panel)),
                          "valid_pool_observations": None, "eligible_date_count": 0})
    try:
        observations, coverage, coverage_year = env.prepare_observations(panel, calendar, industry, index_close)
        projection = stats.build_daily_projection(
            observations, calendar, index_close, selector=env.select_top100,
        )
        results = stats.analyze_projection(projection, calendar, latest_date=signal_range["max"])
        qa = dict(results["qa"])
        required_checks = (
            "daily_top100_hits_equal_100",
            "daily_expected_hits_equal_100",
            "full_month_tenday_grid",
            "calendar_bins_predeclared_and_retained",
            "position_state_timing_prior_only",
            "rs20_state_timing_prior_only",
            "rs60_state_timing_prior_only",
            "zero_n_return_statistics_are_null",
        )
        failed_checks = [name for name in required_checks if not qa.get(name, False)]
        if failed_checks:
            raise RuntimeError("full statistics acceptance checks failed: " + ", ".join(failed_checks))
        meta = {
            "panel": {
                "source_rows": int(len(panel)),
                "distinct_codes": int(panel["code"].nunique()),
                "observed_dates": int(panel["date"].nunique()),
                "first_date": panel["date"].min().date().isoformat(),
                "last_date": panel["date"].max().date().isoformat(),
            },
            "auxiliary": {
                "calendar_rows": int(len(calendar)),
                "industry_rows": int(len(industry)),
                "index_rows": int(len(index_close)),
                "index_first_date": index_close.index.min().date().isoformat(),
                "index_last_date": index_close.index.max().date().isoformat(),
            },
            "coverage_dates": int(len(coverage)),
            "coverage_years": int(len(coverage_year)),
            "source_code_sha256": identity["reused_r1_reader_sha256"],
            "input_binding_sha256": identity["input_binding_sha256"],
            "analysis_code_sha256": identity["analysis_module_sha256"],
            "execution_runner_sha256": identity["execution_runner_sha256"],
            "code_sha256": identity["code_sha256"],
            "config_a_sha256": identity["config_a_sha256"],
            "config_b_sha256": identity["config_b_sha256"],
            "canary_receipt": str(OUTPUTS / "canary/CANARY_RECEIPT.json"),
            "canary_receipt_sha256": sha256_file(OUTPUTS / "canary/CANARY_RECEIPT.json"),
            "canary_metrics": dict(canary_metrics),
        }
        output_dir = OUTPUTS / "full"
        receipt = stats.write_outputs(results, output_dir, meta=meta)
        full_receipt_path = output_dir / "QA_RECEIPT.json"
        output_paths = tuple(sorted(path for path in output_dir.iterdir() if path.is_file()))
        output_hashes = {path.name: sha256_file(path) for path in output_paths}
        event_metrics = {
            "selection_slots": int(qa["eligible_top100_date_count"]) * 100,
            "source_rows": int(len(observations)),
            "valid_pool_observations": int(observations["pool_valid"].sum()),
            "eligible_date_count": int(qa["eligible_top100_date_count"]),
            "truth_positive_denominator": None,
            "projection_rows": int(qa["projection_rows"]),
            "projection_dates": int(qa["projection_dates"]),
            "event_window_complete_count": int(qa["event_window_complete_count"]),
            "event_window_incomplete_count": int(qa["event_window_incomplete_count"]),
        }
        _terminalize_pair(routes, claims, identity, binding, "FULL_STATS", signal_range,
                          terminal_status="COMPLETED", metrics=event_metrics,
                          output_paths=output_paths)
        _verify_terminal_pair(routes, "FULL_STATS")
        return {
            "status": "DIRECTION_ONLY_PENDING_INDEPENDENT_REVIEW",
            "identity": {key: identity[key] for key in (
                "code_sha256", "analysis_module_sha256", "execution_runner_sha256",
                "test_sha256", "reused_r1_reader_sha256", "input_binding_sha256",
                "config_a_sha256", "config_b_sha256", "contract_a_sha256",
                "contract_b_sha256", "composite_contract_sha256",
            )},
            "routes": {
                key: {
                    "route_id": routes[key]["route_id"],
                    "experiment_id": routes[key]["experiment_id"],
                    "one_variable": routes[key]["one_variable"],
                    "execution_ids": {
                        stage.lower(): _execution_id(routes[key], stage)
                        for stage in ("FAST_BIND", "CANARY", "FULL_STATS")
                    },
                    "status": "COMPLETED",
                }
                for key in ("A", "B")
            },
            "signal_date_range": signal_range,
            "source_scope": {
                "source_rows": int(len(observations)),
                "distinct_codes": int(observations["code"].nunique()),
                "observed_dates": int(observations["date"].nunique()),
                "eligible_top100_dates": int(qa["eligible_top100_date_count"]),
                "selection_slots": int(qa["eligible_top100_date_count"]) * 100,
            },
            "qa": qa,
            "output_sha256": output_hashes,
            "qa_receipt_sha256": sha256_file(full_receipt_path),
            "canary_receipt_sha256": sha256_file(OUTPUTS / "canary/CANARY_RECEIPT.json"),
            "safety_flags": {
                "research_only": True,
                "production_allowed": False,
                "buy_gate_updated": False,
                "orders_sent": False,
            },
            "limitations": [
                "Industry classification is current retrospective, not historical PIT.",
                "The 000300 series is a bound Futu-recovered replacement; official TDX source identity and exchange-calendar parity remain CANNOT_CONFIRM.",
                "QFQ close returns are descriptive price-return proxies, not certified economic total returns.",
                "Observed history is exposed and no sealed validation or prediction is claimed.",
            ],
        }
    except BaseException as exc:
        try:
            _terminalize_pair(routes, claims, identity, binding, "FULL_STATS", signal_range,
                              terminal_status="FAILED_NO_FIT", metrics={}, output_paths=(), error=exc)
        except BaseException:
            pass
        raise


def main() -> int:
    if OUTPUTS.exists() and any(OUTPUTS.iterdir()):
        raise FileExistsError(f"output root is not empty; refusing to overwrite: {OUTPUTS}")
    identity, binding, contracts = _ensure_frozen_identity()
    routes = _setup_routes(identity, binding, contracts)
    _verify_reused_fastbind(routes, identity)
    helpers = env.load_bound_helpers(binding)
    panel, calendar, industry, index_close, canary_day, canary_position, canary_metrics = _run_canary(
        routes, identity, binding, helpers,
    )
    summary = _run_full(routes, identity, binding, panel, calendar, industry, index_close, canary_metrics)
    summary_path = OUTPUTS / "EXECUTION_SUMMARY.json"
    _write_new_json(summary_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "routes": {key: value["status"] for key, value in summary["routes"].items()},
        "signal_date_range": summary["signal_date_range"],
        "source_scope": summary["source_scope"],
        "qa": summary["qa"],
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "qa_receipt_sha256": summary["qa_receipt_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
