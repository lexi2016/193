"""Registered summary-only repair from the frozen attempt_02 event-window CSV."""
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
BATCH = ROOT / "research/a_share_top100_environment_v1/batches/seasonality_r1_20261019"
REPAIR = BATCH / "summary_support_repair_r1"
CONTRACT_PATH = REPAIR / "CONTRACT_03.json"
IDENTITY_PATH = REPAIR / "FROZEN_IDENTITY_03.json"
RECOVERY_RECEIPT_PATH = REPAIR / "FAILED_START_RECOVERY_RECEIPT_01.json"
RUNNER_PATH = Path(__file__).resolve()
INPUT_BINDING_PATH = BATCH / "INPUT_BINDING.json"
CONFIG_PATH = BATCH / "CONFIG_A.json"
CODE_PATH = MODULE_DIR / "seasonality_stats.py"
TEST_PATH = MODULE_DIR / "tests/test_seasonality_stats.py"
EVENT_CSV = BATCH / "attempt_02/outputs/full/MAY_OCT_EVENT_WINDOWS.csv"
OLD_SUMMARY_CSV = BATCH / "attempt_02/outputs/full/MAY_OCT_COMMON_YEARS.csv"
OUTPUT_DIR = REPAIR / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "MAY_OCT_COMMON_YEARS_SUPPORT_REPAIRED.csv"
RECEIPT_PATH = OUTPUT_DIR / "SUMMARY_SUPPORT_REPAIR_RECEIPT.json"
BASELINE_COMMIT = "20610d06fc1767bdd8dd9c6c54d48776b4b874d8"
EXPERIMENT_ID = "A_SHARE_TOP100_SEASONALITY_INDUSTRY_POSITION_R1_20261019"
ROUTE_ID = "a-share-top100-seasonality-descriptive-r1-20261019"
ONE_VARIABLE = "PER_BIN_FINITE_PAIRED_SUPPORT_SUMMARY_CORRECTION_ONLY"
REPAIR_EXECUTION_ID = "A_SHARE_TOP100_SEASONALITY_INDUSTRY_POSITION_R1_20261019_SEASONALITY_SUPPORT_REPAIR_02"
REVIEW_EXECUTION_ID = "A_SHARE_TOP100_SEASONALITY_INDUSTRY_POSITION_R1_20261019_SEASONALITY_INCREMENTAL_RESULT_REVIEW_02"

sys.path.insert(0, str(MODULE_DIR))
import seasonality_stats as stats  # noqa: E402
import top100_environment as env  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    return hashlib.sha256(raw).hexdigest()


def _verify_frozen() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, binding, hashes = env.verify_frozen_identity(
        contract_path=CONTRACT_PATH,
        binding_path=INPUT_BINDING_PATH,
        config_path=CONFIG_PATH,
        code_path=CODE_PATH,
        test_path=TEST_PATH,
        expected_baseline=BASELINE_COMMIT,
    )
    identity = _read_json(IDENTITY_PATH)
    checks = {
        "runner_sha256": sha256_file(RUNNER_PATH) == identity["runner_sha256"],
        "event_csv_sha256": sha256_file(EVENT_CSV) == contract["input_sha256"],
        "old_summary_sha256": sha256_file(OLD_SUMMARY_CSV) == contract["prior_summary_sha256"],
        "failed_start_recovery_sha256": sha256_file(RECOVERY_RECEIPT_PATH) == contract["recovery_receipt_sha256"],
        "experiment_id": contract["experiment_id"] == EXPERIMENT_ID,
        "route_id": contract["route_id"] == ROUTE_ID,
        "safety_flags": binding["safety_flags"] == {
            "research_only": True,
            "production_allowed": False,
            "buy_gate_updated": False,
            "orders_sent": False,
        },
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("summary repair frozen identity mismatch: " + ", ".join(failed))
    return contract, binding, {**hashes, **identity, "identity_checks": checks}


def _load_adapter(contract: Mapping[str, Any], binding: Mapping[str, Any]):
    env.EXPERIMENT_ID = EXPERIMENT_ID
    adapter, core = env.load_official_registry(
        binding,
        contract_path=CONTRACT_PATH,
        input_binding_path=INPUT_BINDING_PATH,
        config_path=CONFIG_PATH,
        code_path=CODE_PATH,
        test_path=TEST_PATH,
    )
    adapter.event_profile["route_id"] = ROUTE_ID
    adapter.event_profile["one_variable"] = ONE_VARIABLE
    if adapter.registry_path != Path(binding["registry_path"]).resolve():
        raise RuntimeError("official adapter resolved a different registry path")
    return adapter, core


def _context(contract: Mapping[str, Any], binding: Mapping[str, Any], identity: Mapping[str, Any],
             *, stage: str, terminal: bool, metrics: Mapping[str, Any] | None = None,
             output_paths: Sequence[Path] = ()) -> dict[str, Any]:
    runtime = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    null_reasons = {
        "deterministic_metrics.truth_positive_denominator": "NO_BINARY_FORWARD_TRUTH_LABEL_IN_DESCRIPTIVE_ROUTE",
        "target_date_range": "NO_FORWARD_TARGET_DATE_RANGE_IN_DESCRIPTIVE_ROUTE",
        "checkpoint_ref": "NO_MODEL_CHECKPOINT_IN_DESCRIPTIVE_ROUTE",
    }
    signal_range = dict(contract["signal_date_range"])
    if not terminal:
        null_reasons["deterministic_metrics.source_rows"] = "INPUT_NOT_READ_AT_RUN_STARTED"
        null_reasons["deterministic_metrics.summary_rows"] = "OUTPUT_NOT_CREATED_AT_RUN_STARTED"
    component_shas = {
        "event_window_csv": contract["input_sha256"],
        "prior_common_year_summary_csv": contract["prior_summary_sha256"],
        "input_binding": identity["input_binding_sha256"],
        "analysis_module": identity["code_sha256"],
        "test_source": identity["test_sha256"],
        "runner": identity["runner_sha256"],
        "config": identity["config_sha256"],
        "failed_start_recovery_receipt": contract["recovery_receipt_sha256"],
    }
    artifact_paths = [
        str(CONTRACT_PATH.resolve()), str(IDENTITY_PATH.resolve()),
        str(INPUT_BINDING_PATH.resolve()), str(CONFIG_PATH.resolve()),
        str(CODE_PATH.resolve()), str(TEST_PATH.resolve()), str(RUNNER_PATH.resolve()),
        str(RECOVERY_RECEIPT_PATH.resolve()),
        str(EVENT_CSV.resolve()), str(OLD_SUMMARY_CSV.resolve()),
        str(Path(binding["baseline_record_path"]).resolve()),
        *[str(path.resolve()) for path in output_paths if path.is_file()],
    ]
    return {
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "input_binding_sha256": sha256_file(INPUT_BINDING_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "code_sha256": identity["code_sha256"],
        "input_paths": [str(EVENT_CSV.resolve()), str(OLD_SUMMARY_CSV.resolve())],
        "artifact_paths": list(dict.fromkeys(artifact_paths)),
        "input_component_sha256": component_shas,
        "code_sha256_by_file": {
            str(CODE_PATH.resolve()): identity["code_sha256"],
            str(TEST_PATH.resolve()): identity["test_sha256"],
            str(RUNNER_PATH.resolve()): identity["runner_sha256"],
        },
        "signal_date_range": signal_range,
        "target_date_range": None,
        "truth_access": "OPENED_EXPOSED_HISTORY" if terminal else "NOT_OPENED",
        "sealed_status": "EXPOSED" if terminal else "NOT_ENTERED",
        "runtime": runtime,
        "evidence_level": "DIRECTION_ONLY",
        "deterministic_metrics": {
            "execution_kind": "NO_FIT_ENGINEERING_CANARY",
            "truth_positive_denominator": None,
            "selection_slots": 0,
            "source_rows": (metrics or {}).get("source_rows"),
            "summary_rows": (metrics or {}).get("summary_rows"),
            "lift_support_pair_count": (metrics or {}).get("lift_support_pair_count"),
            "excess_support_pair_count": (metrics or {}).get("excess_support_pair_count"),
            "cross_bin_comparable_rows": (metrics or {}).get("cross_bin_comparable_rows"),
            "fit_call_entered": False,
            "planned_real_fit_count": 0,
            "real_fit_budget_debit": 0,
            "real_fit_returned_count": 0,
        },
        "null_reasons": null_reasons,
    }


def _register_start(adapter: Any, core: Any, contract: Mapping[str, Any], binding: Mapping[str, Any],
                    identity: Mapping[str, Any], *, execution_id: str, stage: str) -> Any:
    existing = [event for event in adapter.list_events(EXPERIMENT_ID)
                if event.get("execution_id") == execution_id]
    if existing:
        raise RuntimeError(f"refusing duplicate registered execution: {execution_id}")
    context = _context(contract, binding, identity, stage=stage, terminal=False)
    return env._claim_stage(
        adapter, core, stage=stage, execution_id=execution_id,
        contract_sha256=sha256_file(CONTRACT_PATH),
        input_binding_sha256=sha256_file(INPUT_BINDING_PATH),
        context=context,
    )


def _register_review_start(adapter: Any, core: Any, contract: Mapping[str, Any], binding: Mapping[str, Any],
                           identity: Mapping[str, Any]) -> dict[str, Any]:
    if any(event.get("execution_id") == REVIEW_EXECUTION_ID for event in adapter.list_events(EXPERIMENT_ID)):
        raise RuntimeError("refusing duplicate incremental result review start")
    context = _context(contract, binding, identity, stage="INDEPENDENT_RESULT_REVIEW", terminal=False,
                       metrics={"summary_rows": None}, output_paths=(OUTPUT_CSV, RECEIPT_PATH))
    claim = env._claim_stage(
        adapter, core, stage="INDEPENDENT_RESULT_REVIEW", execution_id=REVIEW_EXECUTION_ID,
        contract_sha256=sha256_file(CONTRACT_PATH),
        input_binding_sha256=sha256_file(INPUT_BINDING_PATH),
        context=context,
    )
    return {"execution_id": claim.execution_id, "event": "RUN_STARTED", "status": "REGISTERED"}


def run() -> dict[str, Any]:
    contract, binding, identity = _verify_frozen()
    original_full_files = {name: sha256_file(BATCH / "attempt_02/outputs/full" / name)
                           for name in contract["original_full_output_sha256"]}
    if original_full_files != contract["original_full_output_sha256"]:
        raise RuntimeError("frozen attempt_02 output hashes drifted before summary repair")
    if OUTPUT_CSV.exists() or RECEIPT_PATH.exists():
        raise FileExistsError("refusing to replace an existing repair output or receipt")
    adapter, core = _load_adapter(contract, binding)
    claim = _register_start(adapter, core, contract, binding, identity,
                            execution_id=REPAIR_EXECUTION_ID, stage="SUMMARY_SUPPORT_REPAIR")
    metrics: dict[str, Any] = {}
    try:
        source = pd.read_csv(EVENT_CSV, float_precision="round_trip")
        result = stats.summarize_event_common_years(source)
        if result.empty:
            raise RuntimeError("frozen event-window CSV produced an empty summary")

        old = pd.read_csv(OLD_SUMMARY_CSV, dtype={"common_complete_years": "string"})
        keys = ["event", "industry_code", "industry_name", "lag_market", "window_bin"]
        if not set(keys + ["common_complete_years"]).issubset(old.columns):
            raise RuntimeError("prior common-year table lacks the frozen calendar-candidate comparison fields")
        result = stats.normalize_common_year_summary_key_types(result, keys)
        old = stats.normalize_common_year_summary_key_types(old, keys)
        candidate_comparison = result[keys + ["common_complete_years"]].merge(
            old[keys + ["common_complete_years"]], on=keys, how="outer", suffixes=("_new", "_old"), indicator=True
        )
        if not candidate_comparison["_merge"].eq("both").all():
            raise RuntimeError("new summary row keys differ from the frozen prior summary")
        if not candidate_comparison["common_complete_years_new"].fillna("").eq(
            candidate_comparison["common_complete_years_old"].fillna("")
        ).all():
            raise RuntimeError("all-six-bin calendar-complete candidate years changed")

        for row in result.itertuples(index=False):
            if row.lift_pair_support_count != len([year for year in row.lift_pair_support_years.split(",") if year]):
                raise RuntimeError("lift paired-support count does not match its year list")
            if row.excess_pair_support_count != len([year for year in row.excess_pair_support_years.split(",") if year]):
                raise RuntimeError("excess paired-support count does not match its year list")
            if row.lift_pair_support_count:
                if not np.isclose(
                    row.mean_annual_lift_difference_equal_year,
                    row.mean_annual_window_lift_equal_year - row.mean_annual_baseline_lift_equal_year,
                    rtol=1e-12, atol=1e-12,
                ):
                    raise RuntimeError("paired lift mean difference fails window-minus-baseline reconciliation")
            elif not row.lift_empty_support_reason:
                raise RuntimeError("empty lift support lacks an explicit reason")
            if not row.excess_pair_support_count and not row.excess_empty_support_reason:
                raise RuntimeError("empty excess support lacks an explicit reason")

        support_sets = result.groupby(keys[:3] + ["lag_market"], dropna=False)
        comparable_expected: dict[tuple[Any, ...], tuple[bool, bool]] = {}
        for group_key, group in support_sets:
            lift_sets = [tuple(filter(None, value.split(","))) for value in group["lift_pair_support_years"]]
            excess_sets = [tuple(filter(None, value.split(","))) for value in group["excess_pair_support_years"]]
            lift_same = bool(lift_sets) and all(values and values == lift_sets[0] for values in lift_sets)
            excess_same = bool(excess_sets) and all(values and values == excess_sets[0] for values in excess_sets)
            comparable_expected[tuple(group_key if isinstance(group_key, tuple) else (group_key,))] = (lift_same, excess_same)
        for group_key, group in support_sets:
            expected = comparable_expected[tuple(group_key if isinstance(group_key, tuple) else (group_key,))]
            if not group["lift_cross_bin_comparable"].eq(expected[0]).all():
                raise RuntimeError("lift cross-bin comparability flag does not match support sets")
            if not group["excess_cross_bin_comparable"].eq(expected[1]).all():
                raise RuntimeError("excess cross-bin comparability flag does not match support sets")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_bytes = result.to_csv(index=False, na_rep="", float_format="%.17g").encode("utf-8")
        with OUTPUT_CSV.open("xb") as handle:
            handle.write(csv_bytes)
            handle.flush()
        output_sha = hashlib.sha256(csv_bytes).hexdigest()
        metrics = {
            "source_rows": int(len(source)),
            "summary_rows": int(len(result)),
            "lift_support_pair_count": int(result["lift_pair_support_count"].sum()),
            "excess_support_pair_count": int(result["excess_pair_support_count"].sum()),
            "cross_bin_comparable_rows": int(result["cross_bin_comparable"].sum()),
        }
        after_full_files = {name: sha256_file(BATCH / "attempt_02/outputs/full" / name)
                            for name in contract["original_full_output_sha256"]}
        if after_full_files != original_full_files:
            raise RuntimeError("a frozen attempt_02 output changed during summary repair")
        receipt = {
            "schema_version": "A_SHARE_TOP100_SEASONALITY_SUMMARY_SUPPORT_REPAIR_RECEIPT_V1",
            "status": "PASS",
            "execution_id": REPAIR_EXECUTION_ID,
            "route_id": ROUTE_ID,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "identity_sha256": sha256_file(IDENTITY_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "analysis_sha256": sha256_file(CODE_PATH),
            "test_sha256": sha256_file(TEST_PATH),
            "input_event_csv_path": str(EVENT_CSV),
            "input_event_csv_sha256": sha256_file(EVENT_CSV),
            "input_rows": int(len(source)),
            "prior_summary_path": str(OLD_SUMMARY_CSV),
            "prior_summary_sha256": sha256_file(OLD_SUMMARY_CSV),
            "output_path": str(OUTPUT_CSV),
            "output_sha256": output_sha,
            "metrics": metrics,
            "checks": {
                "all_six_bin_calendar_candidate_years_unchanged": True,
                "lift_means_use_same_finite_paired_years": True,
                "lift_difference_equals_window_minus_baseline": True,
                "excess_has_independent_finite_paired_support": True,
                "support_counts_match_year_lists": True,
                "empty_support_has_reason": True,
                "cross_bin_flags_match_effective_support_sets": True,
                "original_attempt_02_outputs_unchanged": True,
                "raw_panel_reread": False,
                "training_or_prediction": False,
            },
            "original_full_output_sha256_before_and_after": {
                "before": original_full_files,
                "after": after_full_files,
            },
            "runtime": {
                "python": sys.executable,
                "python_version": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "safety_flags": binding["safety_flags"],
        }
        receipt_sha = _write_new_json(RECEIPT_PATH, receipt)
        env._terminalize_stage(adapter, claim, terminal_status="COMPLETED", metrics=metrics)
        terminal_events = [event for event in adapter.list_events(EXPERIMENT_ID)
                           if event.get("execution_id") == REPAIR_EXECUTION_ID and event.get("event_type") == "TERMINAL"]
        if len(terminal_events) != 1 or terminal_events[0].get("terminal_status") != "COMPLETED":
            raise RuntimeError("summary repair terminal registry event did not verify")
        review = _register_review_start(adapter, core, contract, binding, identity)
        return {
            "execution_id": REPAIR_EXECUTION_ID,
            "terminal_status": terminal_events[0]["terminal_status"],
            "terminal_event_sha256": terminal_events[0]["event_sha256"],
            "receipt_path": str(RECEIPT_PATH),
            "receipt_sha256": receipt_sha,
            "output_path": str(OUTPUT_CSV),
            "output_sha256": output_sha,
            "metrics": metrics,
            "review_start": review,
            "review_execution_id": REVIEW_EXECUTION_ID,
            "registry_sha256": sha256_file(Path(binding["registry_path"])),
        }
    except BaseException as exc:
        env._terminalize_stage(adapter, claim, terminal_status="FAILED_NO_FIT", metrics=metrics, error=exc)
        raise


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True, indent=2))
