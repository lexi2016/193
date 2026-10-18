from __future__ import annotations

import json
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import futu_index_capability_probe_v1 as probe


def fixture_contract(root: Path) -> tuple[Path, Path, Path]:
    model_contract = root / "model_contract.json"
    binding = root / "binding.json"
    model_contract.write_text('{"status":"FROZEN_AWAITING_PRE_RUN_REVIEW"}\n', encoding="utf-8")
    binding.write_text('{"status":"INPUT_BINDING_READY_PARTIAL"}\n', encoding="utf-8")
    probe_contract = root / "probe_contract.json"
    probe_contract.write_text(json.dumps({
        "status": "FROZEN_AWAITING_PRE_RUN_REVIEW",
        "writer_identity": "model_selection_writer",
        "source_bindings": {
            "model_contract": {"path": model_contract.name, "sha256": probe.sha256_file(model_contract)},
            "input_binding": {"path": binding.name, "sha256": probe.sha256_file(binding)},
        },
    }), encoding="utf-8")
    receipt = root / "reviewer_receipt.json"
    receipt.write_text('{"decision":"REVIEWED"}\n', encoding="utf-8")
    activation = root / "activation.json"
    activation.write_text(json.dumps({
        "status": "APPROVED_FOR_FUTU_METADATA_AND_QUOTA_ONLY",
        "reviewer_identity": "independent_reviewer",
        "probe_contract_sha256": probe.sha256_file(probe_contract),
        "input_binding_sha256": probe.sha256_file(binding),
        "code_sha256": probe.probe_code_sha256(),
        "reviewer_receipt_path": str(receipt),
        "reviewer_receipt_sha256": probe.sha256_file(receipt),
        "allow_local_quote_metadata_calls": True,
        "allow_history_kline_request": False,
        "allow_snapshot_request": False,
        "allow_account_or_trade_api": False,
    }), encoding="utf-8")
    return probe_contract, activation, binding


class FakeGuard:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.backend = self
        self.context: dict[str, object] = {}
        self.claim = None

    def accounting(self):
        return SimpleNamespace(dangling_execution_ids=())

    def set_execution_context(self, execution_id: str, context):
        self.context[execution_id] = dict(context)

    def update_execution_context(self, execution_id: str, **updates):
        self.context[execution_id].update(updates)

    def claim_no_fit(self, **kwargs):
        self.order.append("RUN_STARTED")
        self.claim = kwargs
        return SimpleNamespace(execution_id=kwargs["execution_id"])

    def append_no_fit_terminal(self, claim, *, succeeded: bool, error_type=None):
        self.order.append("TERMINAL")
        self.terminal = {"succeeded": succeeded, "error_type": error_type}


class CapabilityProbeTests(unittest.TestCase):
    def test_requires_reviewer_activation_before_any_context_or_registry_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            contract, activation, binding = fixture_contract(root)
            order: list[str] = []
            api = SimpleNamespace(Market=SimpleNamespace(SH="SH"), RET_OK=0, __version__="test")
            with self.assertRaisesRegex(probe.ProbeBlocked, "independent probe activation is absent"):
                probe.run_capability_probe(
                    execution_id="FUTU_TEST_NO_ACTIVATION",
                    contract_path=contract,
                    activation_path=root / "missing_activation.json",
                    output_path=root / "out.json",
                    futu_api=api,
                    quote_context_factory=lambda **kwargs: order.append("CONTEXT") or None,
                    guard_factory=lambda **kwargs: order.append("GUARD") or FakeGuard(order),
                )
            self.assertEqual(order, [])

    def test_two_exact_read_only_calls_are_registered_before_opend_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            contract, activation, binding = fixture_contract(root)
            order: list[str] = []
            guard = FakeGuard(order)
            market_sh = object()
            api = SimpleNamespace(Market=SimpleNamespace(SH=market_sh), RET_OK=0,
                                  __version__="10.11.7108")

            class QuoteContext:
                def __init__(self, *, host: str, port: int) -> None:
                    order.append("CONTEXT")
                    self.asserted_address = (host, port)

                def get_stock_basicinfo(self, market, *, code_list):
                    order.append("STOCK_BASICINFO")
                    self.asserted_market = market
                    self.asserted_code_list = list(code_list)
                    return 0, pd.DataFrame([{"code": "SH.000300", "name": "CSI 300", "stock_type": "IDX"}])

                def get_history_kl_quota(self, *, get_detail):
                    order.append("HISTORY_QUOTA")
                    self.asserted_detail = get_detail
                    # The detail list is intentionally sensitive and must be discarded.
                    return 0, (25, 75, [{"account_identity": "must-not-persist"}])

                def close(self):
                    order.append("CLOSE")

            result = probe.run_capability_probe(
                execution_id="FUTU_TEST_CAPABILITY_01",
                contract_path=contract,
                activation_path=activation,
                output_path=root / "out.json",
                registry_path=root / "registry.jsonl",
                futu_api=api,
                quote_context_factory=QuoteContext,
                guard_factory=lambda **kwargs: guard,
                now=lambda: datetime(2026, 10, 19, tzinfo=timezone.utc),
            )

            self.assertEqual(result["status"], "CAPABILITY_CONFIRMED")
            self.assertEqual(order, ["RUN_STARTED", "CONTEXT", "STOCK_BASICINFO",
                                     "HISTORY_QUOTA", "CLOSE", "TERMINAL"])
            self.assertEqual(guard.terminal, {"succeeded": True, "error_type": None})
            self.assertEqual(guard.context["FUTU_TEST_CAPABILITY_01"]["deterministic_metrics"]["history_kline_request_count"], 0)
            saved = json.loads((root / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["stock_metadata"], {
                "status": "API_OK", "code": "SH.000300", "name": "CSI 300", "security_type": "IDX"})
            self.assertEqual(saved["history_kl_quota"], {"status": "API_OK", "quota_total": 100})
            self.assertNotIn("must-not-persist", json.dumps(saved))
            self.assertNotIn("account_identity", json.dumps(saved))

    def test_invalid_or_ambiguous_responses_remain_cannot_confirm(self) -> None:
        duplicates = pd.DataFrame([
            {"code": "SH.000300", "name": "A", "stock_type": "IDX"},
            {"code": "SH.000300", "name": "B", "stock_type": "IDX"},
        ])
        self.assertEqual(probe.extract_stock_metadata(duplicates, ret_code=0, ret_ok=0)["status"],
                         "CANNOT_CONFIRM_EXACT_INDEX_ROW")
        self.assertEqual(probe.extract_stock_metadata(pd.DataFrame(), ret_code=0, ret_ok=0)["status"],
                         "INVALID_RESPONSE")
        self.assertEqual(probe.extract_history_quota((0, (1, float("nan"), [])), ret_ok=0)["status"],
                         "INVALID_RESPONSE")
        self.assertEqual(probe.extract_history_quota((1, (1, 2, [])), ret_ok=0),
                         {"status": "API_ERROR", "quota_total": None})


if __name__ == "__main__":
    unittest.main()
