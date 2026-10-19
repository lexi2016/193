from __future__ import annotations

import pandas as pd
import pytest
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import joblib
import numpy as np

import thread_benchmark_v1 as benchmark


def _predictions(values: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [pd.Timestamp("2026-01-05")] * len(values),
        "code": [code for code, _ in values],
        "prediction": [score for _, score in values],
    })


def test_exact_data_identity_passes_and_shuffled_key_identity_blocks_before_fit() -> None:
    expected = {"ordered": "abc", "matrix": "123", "columns": ["C", "RET20P"]}
    benchmark.verify_identity(expected, dict(expected), name="synthetic cell")

    class FitCounter:
        called = False

        def fit(self) -> None:
            self.called = True

    model = FitCounter()
    shuffled = {"ordered": "cba", "matrix": "123", "columns": ["C", "RET20P"]}
    with pytest.raises(benchmark.BenchmarkError, match="identity mismatch"):
        benchmark.verify_identity(expected, shuffled, name="synthetic cell")
        model.fit()
    assert model.called is False


def test_sha_drift_blocks_before_fit(tmp_path) -> None:
    bound_file = tmp_path / "synthetic.bin"
    bound_file.write_bytes(b"source bytes")

    class FitCounter:
        called = False

        def fit(self) -> None:
            self.called = True

    model = FitCounter()
    with pytest.raises(benchmark.BenchmarkError, match="SHA-256 mismatch"):
        benchmark.verify_pinned_hash(bound_file, "0" * 64, "synthetic source")
        model.fit()
    assert model.called is False


def test_prediction_array_full_rank_and_top10_parity_are_all_strict() -> None:
    baseline = _predictions([(f"{i:06d}.SH", 20.0 - i) for i in range(11)])
    exact = benchmark.compare_prediction_frames(baseline, baseline.copy())
    assert exact["prediction_array_equal"] is True
    assert exact["full_daily_rank_order_exact"] is True
    assert exact["top10_keys_exact"] is True
    assert exact["max_abs_diff"] == 0.0

    rank_only_drift = baseline.copy()
    rank_only_drift.loc[10, "prediction"] = -0.5
    drift = benchmark.compare_prediction_frames(rank_only_drift, baseline)
    assert drift["prediction_array_equal"] is False
    assert drift["max_abs_diff"] > 0.0

    top10_drift = baseline.copy()
    top10_drift.loc[10, "prediction"] = 30.0
    top10_comparison = benchmark.compare_prediction_frames(top10_drift, baseline)
    assert top10_comparison["full_daily_rank_order_exact"] is False
    assert top10_comparison["top10_keys_exact"] is False


def test_rank_ties_are_resolved_by_ascending_code_and_duplicate_keys_fail() -> None:
    tied = _predictions([("000002.SH", 1.0), ("000001.SH", 1.0), ("000003.SH", 0.5)])
    ranked = benchmark.rank_all(tied)
    assert ranked[["code", "rank"]].values.tolist() == [
        ["000001.SH", 1], ["000002.SH", 2], ["000003.SH", 3],
    ]
    duplicate = pd.concat([tied, tied.iloc[[0]]], ignore_index=True)
    with pytest.raises(benchmark.BenchmarkError, match="duplicate keys"):
        benchmark.rank_all(duplicate)


def test_resource_gates_stop_on_pressure_swap_or_rss_only() -> None:
    healthy = {
        "pressure_state": "normal",
        "worker_rss_bytes": benchmark.MAX_WORKER_RSS_BYTES,
        "swap_used_bytes": 1000,
    }
    assert benchmark.resource_abort_reason(
        healthy, initial_swap_used_bytes=1000, during_run=True,
    ) is None
    assert benchmark.resource_abort_reason(
        {**healthy, "pressure_state": "warning_or_critical"},
        initial_swap_used_bytes=1000, during_run=True,
    ) == "MEMORY_PRESSURE_NOT_NORMAL"
    assert benchmark.resource_abort_reason(
        {**healthy, "worker_rss_bytes": benchmark.MAX_WORKER_RSS_BYTES + 1},
        initial_swap_used_bytes=1000, during_run=True,
    ) == "WORKER_RSS_EXCEEDED_12_GIB"
    assert benchmark.resource_abort_reason(
        {**healthy, "swap_used_bytes": 1000 + benchmark.MAX_SWAP_INCREASE_BYTES},
        initial_swap_used_bytes=1000, during_run=True,
    ) == "SWAP_INCREASE_REACHED_256_MIB"


def test_termination_only_touches_the_worker_process_given_to_supervisor() -> None:
    class FakeWorker:
        returncode = None
        terminated = False
        killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    worker = FakeWorker()
    unrelated = FakeWorker()
    benchmark.terminate_only_worker(worker, grace_seconds=0.01)
    assert worker.terminated is True
    assert unrelated.terminated is False
    assert unrelated.killed is False


class _FakeRoute:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update_registry_execution_context(self, guard, execution_id, **updates):
        self.updates.append({"execution_id": execution_id, **updates})


class _FakeGuard:
    def __init__(self) -> None:
        self.no_fit_terminals: list[tuple] = []
        self.fit_terminals: list[tuple] = []
        self.events: list[str] = []

    def claim_no_fit(self, *, execution_id, arm, contract_sha256, input_binding_sha256):
        self.events.append("RUN_STARTED_NO_FIT")
        return f"{execution_id}-claim"

    def append_no_fit_terminal(self, claim, *, succeeded, error_type=None):
        self.events.append("TERMINAL_NO_FIT")
        self.no_fit_terminals.append((claim, succeeded, error_type))

    def append_terminal(self, claim, **kwargs):
        self.events.append("TERMINAL_FIT")
        self.fit_terminals.append((claim, kwargs))


def test_official_registry_terminal_helpers_write_one_truthful_terminal() -> None:
    route = _FakeRoute()
    guard = _FakeGuard()
    prep_claim = guard.claim_no_fit(
        execution_id="prep-id", arm="M0", contract_sha256="a" * 64,
        input_binding_sha256="b" * 64,
    )
    benchmark._record_no_fit_terminal(
        route, guard, prep_claim, "prep-id",
        {"fit_calls": 0, "truth_access": "TRAIN_LABELS_READ_PAST_ONLY"},
        succeeded=True,
    )
    assert len(guard.no_fit_terminals) == 1
    assert guard.no_fit_terminals[0] == ("prep-id-claim", True, None)
    assert guard.events == ["RUN_STARTED_NO_FIT", "TERMINAL_NO_FIT"]
    assert route.updates[-1]["execution_id"] == "prep-id"
    assert route.updates[-1]["truth_access"] == "TRAIN_LABELS_READ_PAST_ONLY"

    benchmark._record_fit_terminal(
        route, guard, "fit-claim", "fit-id",
        {"fit_calls": 1, "checkpoint_path": "/tmp/candidate.joblib",
         "checkpoint_sha256": "a" * 64},
        succeeded=True, fit_call_entered=True, returned_count=1,
    )
    assert len(guard.fit_terminals) == 1
    assert guard.events[-1] == "TERMINAL_FIT"
    assert guard.fit_terminals[0][0] == "fit-claim"
    assert guard.fit_terminals[0][1]["terminal_status"] == "COMPLETED"
    assert guard.fit_terminals[0][1]["fit_call_entered"] is True
    assert guard.fit_terminals[0][1]["real_fit_returned_count"] == 1
    assert route.updates[-1]["checkpoint_ref"]["sha256"] == "a" * 64


def test_host_parsers_are_fail_closed_on_unrecognized_resource_output() -> None:
    page_size, pages = benchmark.parse_vm_stat(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 10.\nPages inactive: 5.\nPages speculative: 2.\n"
    )
    assert page_size == 16384
    assert pages["pages_free"] == 10
    assert benchmark.parse_swap_used_bytes(
        "vm.swapusage: total = 5.00G  used = 128.00M  free = 4.88G"
    ) == 128 * 1024**2
    assert benchmark.parse_memory_pressure("0") == "normal"
    with pytest.raises(benchmark.BenchmarkError, match="unrecognized"):
        benchmark.parse_memory_pressure("unknown")
    assert benchmark.parse_rss_kilobytes("12345") == 12345 * 1024
    assert benchmark.parse_rss_kilobytes("") is None


def test_checkpoint_loader_accepts_frozen_nested_sidecar_hashes(tmp_path) -> None:
    keys = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "code": ["000001.SZ"]})
    values = np.array([[1.0]], dtype=np.float64)
    checkpoint = tmp_path / "checkpoint.joblib"
    joblib.dump({"feature_columns": ["feature"],
                 "replay_fixture": {"test_keys": keys, "test_features": values}}, checkpoint)
    checkpoint_sha = benchmark.sha256_file(checkpoint)
    metadata = {"status": "COMPLETED", "artifact_sha256": checkpoint_sha,
                "identity": {"code_sha256": {"code": "c"}, "contract_sha256": "contract",
                             "input_components": {"panel": "input"},
                             "data_identity": {"feature_columns": ["feature"],
                                               "test_keys": {"keys": "k"},
                                               "test_feature_matrix": {"matrix": "m"}}}}
    replay = {"status": "VERIFIED", "checkpoint_sha256": checkpoint_sha,
              "prediction_identity": "p", "replay_prediction_identity": "p"}
    (tmp_path / "meta.json").write_text(json.dumps(metadata))
    (tmp_path / "replay.json").write_text(json.dumps(replay))
    artifact = {"checkpoint": {"path": "checkpoint.joblib"}, "checkpoint_sha256": checkpoint_sha,
                "metadata": {"path": "meta.json", "sha256": benchmark.sha256_file(tmp_path / "meta.json")},
                "replay": {"path": "replay.json", "sha256": benchmark.sha256_file(tmp_path / "replay.json")}}
    route = SimpleNamespace(key_frame_identity=lambda _: {"keys": "k"},
                            numeric_matrix_identity=lambda *_: {"matrix": "m"})
    loaded = benchmark._load_checkpoint_fixture(route, tmp_path, artifact,
                                                {"code": "c"}, "contract", {"panel": "input"})
    assert np.array_equal(loaded["values"], values)
    artifact["metadata"]["sha256"] = "0" * 64
    with pytest.raises(benchmark.BenchmarkError, match="metadata SHA-256 mismatch"):
        benchmark._load_checkpoint_fixture(route, tmp_path, artifact,
                                           {"code": "c"}, "contract", {"panel": "input"})


def test_one_fit_budget_is_separate_from_exhausted_formal_budget(tmp_path) -> None:
    core_dir = Path("/Users/lexi/Desktop/学习/因子工厂/workspace/hkconnect_price_semantics_v2_20260919/zero_label_parity")
    sys.path.insert(0, str(core_dir))
    from execution_guard_core import ExecutionGuard, RealFitBudgetExceeded

    class Backend:
        event_profile = {"route_id": "formal"}
        events = [{"experiment_id": "formal", "event_type": "RUN_STARTED", "real_fit_budget_debit": 1}
                  for _ in range(96)]

        def list_events(self, experiment_id):
            return [e for e in self.events if e["experiment_id"] == experiment_id]

        def claim_run_started(self, event, *, max_real_fit_debits):
            used = sum(e["real_fit_budget_debit"] for e in self.list_events(event["experiment_id"]))
            if used + event["real_fit_budget_debit"] > max_real_fit_debits:
                raise RealFitBudgetExceeded("exhausted")
            self.events.append(dict(event))

    backend = Backend()
    formal = ExecutionGuard(backend, experiment_id="formal", max_real_fit_debits=96)
    route = SimpleNamespace(make_official_fit_guard=lambda **_: formal)
    guard = benchmark.make_benchmark_guard(route, contract_path=tmp_path / "contract",
                                           binding_path=tmp_path / "binding", stage="TEST")
    guard.claim(execution_id="benchmark-1", arm="M0", contract_sha256="a", input_binding_sha256="b")
    with pytest.raises(RealFitBudgetExceeded):
        guard.claim(execution_id="benchmark-2", arm="M0", contract_sha256="a", input_binding_sha256="b")
    assert len(backend.list_events("formal")) == 96
    assert len(backend.list_events(benchmark.BENCHMARK_EXPERIMENT_ID)) == 1
    assert backend.event_profile["route_id"] == benchmark.BENCHMARK_ROUTE_ID


def test_closed_zero_fit_preflight_allows_retry_but_fit_claim_or_open_run_blocks() -> None:
    events = [
        {"event_type": "RUN_STARTED", "execution_id": "M0_THREAD_BENCHMARK_old_PREFLIGHT",
         "execution_kind": "NO_FIT_ENGINEERING_CANARY", "real_fit_budget_debit": 0},
        {"event_type": "TERMINAL", "execution_id": "M0_THREAD_BENCHMARK_old_PREFLIGHT",
         "fit_call_entered": False, "real_fit_returned_count": 0},
    ]
    accounting = SimpleNamespace(dangling_execution_ids=())
    guard = SimpleNamespace(accounting=lambda: accounting, experiment_id="benchmark",
                            backend=SimpleNamespace(list_events=lambda _: events))
    benchmark.validate_benchmark_retry_budget(guard)
    accounting.dangling_execution_ids = ("M0_THREAD_BENCHMARK_open_PREFLIGHT",)
    with pytest.raises(benchmark.BenchmarkError, match="dangling execution"):
        benchmark.validate_benchmark_retry_budget(guard)
    accounting.dangling_execution_ids = ()
    events.extend([
        {"event_type": "RUN_STARTED", "execution_id": "M0_THREAD_BENCHMARK_old_FIT",
         "execution_kind": "REAL_FIT", "real_fit_budget_debit": 1},
        {"event_type": "TERMINAL", "execution_id": "M0_THREAD_BENCHMARK_old_FIT",
         "fit_call_entered": False, "real_fit_returned_count": 0},
    ])
    with pytest.raises(benchmark.BenchmarkError, match="budget has already been consumed"):
        benchmark.validate_benchmark_retry_budget(guard)


def test_worker_rejects_missing_forged_closed_or_wrong_parent_claims() -> None:
    nonce = "a" * 32
    message = {"execution_id": "fit", "phase": "FIT", "nonce": nonce}
    event = {"execution_id": "fit", "event_type": "RUN_STARTED",
             "experiment_id": benchmark.BENCHMARK_EXPERIMENT_ID,
             "route_id": benchmark.BENCHMARK_ROUTE_ID,
             "contract_sha256": "contract", "input_binding_sha256": "binding",
             "deterministic_metrics": {"execution_kind": "REAL_FIT", "real_fit_budget_debit": 1},
             "input_identity": {"benchmark_supervisor_pid": 123,
                                "worker_authorization_sha256": hashlib.sha256(nonce.encode()).hexdigest()}}
    kwargs = dict(phase="FIT", parent_pid=123, contract_sha="contract", binding_sha="binding")
    assert benchmark.validate_worker_claim([event], message, **kwargs) == event
    for events, payload, arguments in [([], message, kwargs),
            ([event], {**message, "nonce": "b" * 32}, kwargs),
            ([event], message, {**kwargs, "parent_pid": 124}),
            ([event], message, {**kwargs, "contract_sha": "different"}),
            ([event, {"execution_id": "fit", "event_type": "TERMINAL"}], message, kwargs)]:
        with pytest.raises(benchmark.BenchmarkError):
            benchmark.validate_worker_claim(events, payload, **arguments)


def test_direct_worker_cannot_read_inputs_with_only_local_sentinels(tmp_path, monkeypatch) -> None:
    (tmp_path / "FIT_GO").write_text("go")
    (tmp_path / "supervisor.json").write_text('{"fit_execution_claimed": true}')
    contract = {"schema_version": benchmark.TOOL_SCHEMA, "writer_tool_sha256": "tool"}
    contract["config_sha256"] = benchmark.config_identity(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(benchmark, "verify_pinned_hash", lambda *a: "tool")
    monkeypatch.setattr(benchmark, "stat", SimpleNamespace(S_ISFIFO=lambda mode: False))
    monkeypatch.setattr(benchmark.sys, "stdin", SimpleNamespace(fileno=lambda: 0))
    calls = []
    monkeypatch.setattr(benchmark, "_load_and_verify_inputs", lambda *a, **k: calls.append("read"))
    assert benchmark._worker_main(tmp_path, contract_path, tmp_path) == 1
    assert calls == []
    assert benchmark.load_json(tmp_path / "worker_status.json")["fit_calls"] == 0


def test_resource_preflight_does_not_count_inactive_pages_as_free(monkeypatch) -> None:
    outputs = {"/usr/bin/vm_stat": "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 1000000.\n",
               "vm.memory_pressure": "0", "vm.swapusage": "used = 128.00M"}
    monkeypatch.setattr(benchmark.sys, "platform", "darwin")
    monkeypatch.setattr(benchmark, "_read_command", lambda a: outputs[a[-1]])
    assert benchmark.probe_host_resources()["available_bytes_estimate"] == 10 * 16384


def test_fit_claim_aborted_before_entry_closes_without_claiming_a_fit() -> None:
    route, guard = _FakeRoute(), _FakeGuard()
    benchmark._record_fit_terminal(route, guard, "claim", "fit", {"fit_calls": 0},
                                   succeeded=False, fit_call_entered=False, returned_count=0)
    terminal = guard.fit_terminals[0][1]
    assert terminal["terminal_status"] == "ABORTED_BEFORE_FIT"
    assert terminal["fit_call_entered"] is False


def test_worker_model_initialization_failure_after_claim_records_zero_fits(tmp_path, monkeypatch) -> None:
    contract = {"schema_version": benchmark.TOOL_SCHEMA, "writer_tool_sha256": "tool"}
    contract["config_sha256"] = benchmark.config_identity(contract)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(benchmark, "verify_pinned_hash", lambda *a: "tool")
    phases = []
    monkeypatch.setattr(benchmark, "receive_worker_claim",
                        lambda *a, phase: phases.append(phase) or {"official_claim": phase})
    frame = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "code": ["000001.SZ"]})
    keys = {"sorted_keyset_sha256": "k", "ordered_keys_sha256": "o"}
    matrix = {"sha256": "x"}
    prepared = {"train": frame, "test": frame, "training_label_preparation_slots": 1,
                "feature_columns": ["f"], "identity": {
                    "training_keys": keys, "test_keys": keys,
                    "training_feature_matrix": matrix, "test_feature_matrix": matrix,
                    "training_label_vector": matrix, "training_sample_weight_vector": matrix}}

    def model_initialization_failure():
        raise RuntimeError("synthetic model initialization failed before fit")

    inputs = {"prepared": prepared, "route": SimpleNamespace(model_for_arm=model_initialization_failure),
              "route_code_sha256": "r", "code_identity": {}, "contract_sha256": "c",
              "binding_sha256": "b", "training_date_min": "2024-01-02", "training_date_max": "2025-12-30",
              "target_date_min": "2024-01-03", "target_date_max": "2025-12-31", "source_fixtures": []}
    monkeypatch.setattr(benchmark, "_load_and_verify_inputs", lambda *a, **k: inputs)
    assert benchmark._worker_main(tmp_path, contract_path, tmp_path) == 1
    assert phases == ["PREFLIGHT", "FIT"]
    status = benchmark.load_json(tmp_path / "worker_status.json")
    assert status["state"] == "FAILED"
    assert status["fit_calls"] == 0
    assert status["error"] == "synthetic model initialization failed before fit"
    route, guard = _FakeRoute(), _FakeGuard()
    benchmark._record_fit_terminal(route, guard, "claim", "fit", status,
                                   succeeded=False, fit_call_entered=bool(status["fit_calls"]), returned_count=0)
    assert guard.fit_terminals[0][1]["terminal_status"] == "ABORTED_BEFORE_FIT"
