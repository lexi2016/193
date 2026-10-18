from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import futu_index_history_recovery_v1 as recovery


PROJECT_ROOT = recovery.PROJECT_ROOT
BATCH_ROOT = recovery.BATCH_ROOT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(date: str, *, code: str = "SH.000300", open_: float = 10.0,
         high: float = 12.0, low: float = 9.0, close: float = 11.0) -> dict[str, Any]:
    return {"code": code, "time_key": f"{date} 00:00:00", "open": open_, "close": close,
            "high": high, "low": low, "volume": 100, "turnover": 1000.0,
            "change_rate": 1.0, "last_close": 10.0}


class FakeRegistryGuard:
    def __init__(self, events: list[tuple[Any, ...]]) -> None:
        self.events = events
        self.backend = self
        self.context: dict[str, dict[str, Any]] = {}

    def accounting(self):
        return SimpleNamespace(dangling_execution_ids=())

    def set_execution_context(self, execution_id: str, context: dict[str, Any]) -> None:
        self.context[execution_id] = dict(context)

    def update_execution_context(self, execution_id: str, **updates: Any) -> None:
        self.context[execution_id].update(updates)

    def claim_no_fit(self, **kwargs: Any):
        self.events.append(("RUN_STARTED", kwargs["execution_id"]))
        return SimpleNamespace(execution_id=kwargs["execution_id"])

    def append_no_fit_terminal(self, claim: Any, *, succeeded: bool, error_type: str | None = None) -> None:
        self.events.append(("TERMINAL", claim.execution_id, succeeded, error_type))

    def list_events(self, experiment_id: str):
        result = []
        for event in self.events:
            if event[0] == "RUN_STARTED":
                result.append({"event_type": "RUN_STARTED", "execution_id": event[1]})
            elif event[0] == "TERMINAL":
                result.append({"event_type": "TERMINAL", "execution_id": event[1],
                               "terminal_status": "ENGINEERING_CANARY_ONLY" if event[2]
                               else "CANNOT_CONFIRM_ENGINEERING_CANARY"})
        return result


class FakeContext:
    def __init__(self, events: list[tuple[Any, ...]], pages: list[tuple[Any, Any, Any]],
                 quota: tuple[int, int] = (7, 93)) -> None:
        self.events, self.pages, self.requests = events, pages, []
        self.quota = quota
        self.closed = False

    def get_history_kl_quota(self, *, get_detail: bool):
        self.events.append(("quota", get_detail))
        return (0, (*self.quota, []))

    def request_history_kline(self, code: str, **kwargs: Any):
        self.events.append(("history", code, kwargs.copy()))
        self.requests.append(kwargs.copy())
        if not self.pages:
            return (0, pd.DataFrame(columns=recovery.RAW_COLUMNS), None)
        return self.pages.pop(0)

    def close(self) -> None:
        self.closed = True


class FutuHistoryRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="futu-history-recovery-test-")
        self.root = Path(self.temp.name)
        self.batch = self.root / "batch"
        self.batch.mkdir()
        for name in (
            "CONTRACT.json", "INPUT_BINDING.json", "FUTU_INDEX_CAPABILITY_PROBE_CONTRACT.json",
            "FUTU_INDEX_CAPABILITY_PROBE_RECEIPT_20261019.json",
        ):
            shutil.copy2(BATCH_ROOT / name, self.batch / name)
        self.contract_path = self.batch / "FUTU_INDEX_HISTORY_RECOVERY_CONTRACT.json"
        self.binding_path = self.batch / "FUTU_INDEX_HISTORY_RECOVERY_INPUT_BINDING.json"
        shutil.copy2(BATCH_ROOT / self.contract_path.name, self.contract_path)
        shutil.copy2(BATCH_ROOT / self.binding_path.name, self.binding_path)
        self.reviewer_receipt = self.batch / "reviewer_receipt.json"
        self.reviewer_receipt.write_text('{"status":"REVIEWED"}\n', encoding="utf-8")
        self.activation_path = self.batch / "FUTU_INDEX_HISTORY_RECOVERY_ACTIVATION.json"
        self.activation_path.write_text(json.dumps({
            "status": "APPROVED_FUTU_HISTORY_CANARY_AND_CONDITIONAL_FULL",
            "reviewer_identity": "independent_reviewer",
            "reviewer_receipt_path": str(self.reviewer_receipt),
            "reviewer_receipt_sha256": _sha(self.reviewer_receipt),
            "contract_sha256": _sha(self.contract_path),
            "input_binding_sha256": _sha(self.binding_path),
            "code_sha256": _sha(Path(recovery.__file__)),
            "allow_history_canary": True,
            "allow_full_history_after_canary": True,
            "allow_snapshot_request": False,
            "allow_account_or_trade_api": False,
        }), encoding="utf-8")
        self.events: list[tuple[Any, ...]] = []
        self.guard = FakeRegistryGuard(self.events)
        self.guard_factory_calls = []
        self.materialized_registry_events: list[dict[str, Any]] = []
        self.futu_api = SimpleNamespace(
            RET_OK=0,
            KLType=SimpleNamespace(K_DAY="K_DAY"),
            AuType=SimpleNamespace(NONE="NONE"),
            __version__="synthetic",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _guard_factory(self, **kwargs: Any):
        self.guard_factory_calls.append(kwargs)
        return self.guard

    def _builder_materializing_guard_factory(self, **kwargs: Any):
        guard = recovery._history_guard(**kwargs)
        backend = guard.backend
        events: list[dict[str, Any]] = []

        def list_events(_experiment_id: str):
            return list(events)

        def claim_run_started(event: dict[str, Any], *, max_real_fit_debits: int) -> None:
            self.events.append(("BUILDER_MATERIALIZE_START", event["execution_id"]))
            spec = backend._event_spec(event, event_type="RUN_STARTED")
            built = backend.builder.materialize([spec], starting_previous_sha="0" * 64)
            self.assertTrue(all(Path(path).is_file() for path in spec["artifact_paths"]))
            self.assertFalse(any(str(path).endswith("receipt.json") for path in spec["artifact_paths"]))
            events.extend(built)
            self.materialized_registry_events.extend(built)

        def terminalize(claim: Any, event: dict[str, Any]) -> None:
            self.events.append(("BUILDER_MATERIALIZE_TERMINAL", event["execution_id"]))
            spec = backend._event_spec(event, event_type="TERMINAL")
            built = backend.builder.materialize(
                [spec], starting_previous_sha=events[-1]["event_sha256"])
            events.extend(built)
            self.materialized_registry_events.extend(built)

        backend.list_events = list_events
        backend.claim_run_started = claim_run_started
        backend.terminalize = terminalize
        return guard

    def _run(self, *, phase: str, execution_id: str, pages: list[tuple[Any, Any, Any]],
             canary_receipt_path: Path | None = None, canary_receipt_sha256: str | None = None,
             quota: tuple[int, int] = (7, 93)):
        context = FakeContext(self.events, pages, quota=quota)
        def context_factory(**kwargs: Any):
            self.events.append(("context", kwargs.copy()))
            return context
        result = recovery.run_history_recovery(
            phase=phase, execution_id=execution_id,
            contract_path=self.contract_path, input_binding_path=self.binding_path,
            activation_path=self.activation_path, output_root=self.root / "out",
            registry_path=self.root / "registry.jsonl", canary_receipt_path=canary_receipt_path,
            canary_receipt_sha256=canary_receipt_sha256, futu_api=self.futu_api,
            quote_context_factory=context_factory, guard_factory=self._guard_factory,
            now=lambda: datetime(2026, 10, 19, tzinfo=timezone.utc),
        )
        self.assertTrue(context.closed)
        return result, context

    def test_canary_runs_after_registry_start_and_writes_bound_receipt(self) -> None:
        result, context = self._run(phase="canary", execution_id="CANARY_001", pages=[
            (0, pd.DataFrame([_row("2021-01-04"), _row("2021-01-05")]), None),
        ])
        self.assertEqual(result["status"], "CANARY_PASS")
        self.assertEqual(result["normalized_row_count"], 2)
        self.assertEqual(len(context.requests), 1)
        self.assertEqual(context.requests[0]["max_count"], 5)
        self.assertEqual(context.requests[0]["start"], "2021-01-04")
        self.assertEqual(context.requests[0]["end"], "2021-01-08")
        self.assertIsNone(context.requests[0]["page_req_key"])
        self.assertLess(self.events.index(("RUN_STARTED", "CANARY_001")),
                        next(i for i, item in enumerate(self.events) if item[0] == "context"))
        receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
        self.assertEqual(receipt["official_exchange_calendar_parity"], "CANNOT_CONFIRM")
        self.assertEqual(receipt["quota"]["quota_total"], 100)
        self.assertEqual(len(receipt["raw_page_artifacts"]), 1)
        self.assertTrue((Path(result["receipt_path"]).parent / receipt["normalized_path"]).is_file())
        self.assertEqual(self.events[-1][0], "TERMINAL")

    def test_invalid_canary_cannot_unlock_full_phase(self) -> None:
        result, _ = self._run(phase="canary", execution_id="CANARY_BAD", pages=[
            (0, pd.DataFrame([_row("2021-01-04", high=8.0)]), None),
        ])
        self.assertEqual(result["status"], "CANARY_CANNOT_CONFIRM")
        before_calls = len(self.guard_factory_calls)
        with self.assertRaises(recovery.RecoveryBlocked):
            recovery.run_history_recovery(
                phase="full", execution_id="FULL_BLOCKED", contract_path=self.contract_path,
                input_binding_path=self.binding_path, activation_path=self.activation_path,
                output_root=self.root / "out", registry_path=self.root / "registry.jsonl",
                canary_receipt_path=Path(result["receipt_path"]), canary_receipt_sha256=result["receipt_sha256"],
                futu_api=self.futu_api, guard_factory=self._guard_factory,
            )
        self.assertEqual(len(self.guard_factory_calls), before_calls)

    def test_one_canary_request_is_allowed_when_reported_remaining_quota_is_zero(self) -> None:
        result, context = self._run(phase="canary", execution_id="CANARY_QUOTA_ZERO", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), None),
        ], quota=(100, 0))
        self.assertEqual(result["status"], "CANARY_PASS")
        self.assertEqual(result["quota"]["remain_quota"], 0)
        self.assertEqual(len(context.requests), 1)
        self.assertEqual(context.requests[0]["max_count"], 5)

    def test_history_api_denial_is_not_retried(self) -> None:
        result, context = self._run(phase="canary", execution_id="CANARY_DENIED", pages=[
            (1, "quota denied", None),
        ])
        self.assertEqual(result["status"], "CANARY_CANNOT_CONFIRM")
        self.assertEqual(result["api_status"], "API_ERROR")
        self.assertEqual(len(context.requests), 1)

    def test_official_builder_materializes_existing_stable_start_and_terminal_artifacts(self) -> None:
        result, _ = self._run_with_builder_materializer(phase="canary", execution_id="BUILDER_OK", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), None),
        ])
        self.assertEqual(result["status"], "CANARY_PASS")
        self.assertEqual([event[0] for event in self.events if event[0].startswith("BUILDER_MATERIALIZE")],
                         ["BUILDER_MATERIALIZE_START", "BUILDER_MATERIALIZE_TERMINAL"])
        self.assertEqual(self.materialized_registry_events[0]["decision"], "RUNNING")
        self.assertEqual(self.materialized_registry_events[1]["decision"], "ENGINEERING_CANARY_ONLY")

    def test_official_builder_materializes_failed_canary_terminal_after_receipt(self) -> None:
        result, _ = self._run_with_builder_materializer(phase="canary", execution_id="BUILDER_FAIL", pages=[
            (1, "quota denied", None),
        ])
        self.assertEqual(result["status"], "CANARY_CANNOT_CONFIRM")
        self.assertEqual([event[0] for event in self.events if event[0].startswith("BUILDER_MATERIALIZE")],
                         ["BUILDER_MATERIALIZE_START", "BUILDER_MATERIALIZE_TERMINAL"])
        self.assertEqual(self.materialized_registry_events[0]["decision"], "RUNNING")
        self.assertEqual(self.materialized_registry_events[1]["decision"],
                         "CANNOT_CONFIRM_ENGINEERING_CANARY")
        self.assertTrue(Path(result["receipt_path"]).is_file())

    def _run_with_builder_materializer(self, *, phase: str, execution_id: str,
                                       pages: list[tuple[Any, Any, Any]]):
        context = FakeContext(self.events, pages)
        def context_factory(**kwargs: Any):
            self.events.append(("context", kwargs.copy()))
            return context
        result = recovery.run_history_recovery(
            phase=phase, execution_id=execution_id,
            contract_path=self.contract_path, input_binding_path=self.binding_path,
            activation_path=self.activation_path, output_root=self.root / "out_builder",
            registry_path=self.root / "registry_builder.jsonl", futu_api=self.futu_api,
            quote_context_factory=context_factory,
            guard_factory=self._builder_materializing_guard_factory,
            now=lambda: datetime(2026, 10, 19, tzinfo=timezone.utc),
        )
        self.assertTrue(context.closed)
        return result, context

    def test_full_phase_reuses_exact_page_key_and_caps_pages(self) -> None:
        canary, _ = self._run(phase="canary", execution_id="CANARY_OK", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), None),
        ])
        page_key = object()
        full, context = self._run(phase="full", execution_id="FULL_001", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), page_key),
            (0, pd.DataFrame([_row("2021-01-05")]), None),
        ], canary_receipt_path=Path(canary["receipt_path"]),
           canary_receipt_sha256=canary["receipt_sha256"])
        self.assertEqual(full["status"], "SERIES_RECOVERED_CALENDAR_PARITY_CANNOT_CONFIRM")
        self.assertEqual(full["page_count"], 2)
        self.assertIsNone(context.requests[0]["page_req_key"])
        self.assertIs(context.requests[1]["page_req_key"], page_key)
        self.assertTrue(all(call["max_count"] == 1000 for call in context.requests))
        self.assertEqual(full["official_exchange_calendar_parity"], "CANNOT_CONFIRM")
        self.assertEqual(len(full["raw_page_artifacts"]), 2)

    def test_full_phase_stops_after_three_pages_if_key_remains(self) -> None:
        canary, _ = self._run(phase="canary", execution_id="CANARY_OK2", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), None),
        ])
        keys = [object(), object(), object()]
        full, context = self._run(phase="full", execution_id="FULL_LIMIT", pages=[
            (0, pd.DataFrame([_row("2021-01-04")]), keys[0]),
            (0, pd.DataFrame([_row("2021-01-05")]), keys[1]),
            (0, pd.DataFrame([_row("2021-01-06")]), keys[2]),
        ], canary_receipt_path=Path(canary["receipt_path"]),
           canary_receipt_sha256=canary["receipt_sha256"])
        self.assertEqual(full["status"], "CANNOT_CONFIRM")
        self.assertEqual(full["api_status"], "PAGE_LIMIT_EXCEEDED")
        self.assertEqual(len(context.requests), 3)
        self.assertEqual(context.requests[1]["page_req_key"], keys[0])
        self.assertEqual(context.requests[2]["page_req_key"], keys[1])

    def test_activation_hash_mismatch_blocks_before_guard_or_api(self) -> None:
        activation = json.loads(self.activation_path.read_text(encoding="utf-8"))
        activation["code_sha256"] = "0" * 64
        self.activation_path.write_text(json.dumps(activation), encoding="utf-8")
        with self.assertRaises(recovery.RecoveryBlocked):
            recovery.run_history_recovery(
                phase="canary", execution_id="BLOCKED", contract_path=self.contract_path,
                input_binding_path=self.binding_path, activation_path=self.activation_path,
                output_root=self.root / "out", registry_path=self.root / "registry.jsonl",
                futu_api=self.futu_api, guard_factory=self._guard_factory,
            )
        self.assertEqual(self.guard_factory_calls, [])
        self.assertFalse((self.root / "out").exists())

    def test_normalizer_rejects_wrong_symbol_bad_dates_and_inconsistent_ohlc(self) -> None:
        frame = pd.DataFrame([_row("2021-01-04"), _row("2021-01-05", code="SZ.000300"),
                              _row("2021-01-09"), _row("2021-01-06", high=8.0)])
        normalized, summary = recovery._normalize_page(frame, start="2021-01-04", end="2021-01-08")
        self.assertEqual(summary["status"], "INVALID_ROWS")
        self.assertEqual(summary["invalid_row_count"], 3)
        self.assertEqual(len(normalized), 1)

    def test_full_phase_requires_canary_sha_and_no_quota_detail_request(self) -> None:
        with self.assertRaises(recovery.RecoveryBlocked):
            recovery.run_history_recovery(
                phase="full", execution_id="NO_CANARY", contract_path=self.contract_path,
                input_binding_path=self.binding_path, activation_path=self.activation_path,
                output_root=self.root / "out", registry_path=self.root / "registry.jsonl",
                futu_api=self.futu_api, guard_factory=self._guard_factory,
            )
        self.assertEqual(self.guard_factory_calls, [])
        quota = recovery._quota_snapshot((0, (3, 0, ["ignored-detail"])), ret_ok=0)
        self.assertEqual(quota, {"status": "API_OK", "used_quota": 3,
                                 "remain_quota": 0, "quota_total": 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
