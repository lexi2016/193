"""Synthetic regression tests for the R4 B2 report-field adapter."""
from __future__ import annotations

import hashlib
import unittest

from process_archive_material_b2 import (
    build_material_report_from_payloads,
    load_analyzer,
    normalize_member_records,
)


class B2MemberNameAdapterTests(unittest.TestCase):
    def test_b1_original_member_name_is_mapped_to_report_builder_member_name(self) -> None:
        records = [{
            "original_member_name": "formulas/base.formula",
            "category": "formula",
            "safe_basename": "member_1_base.formula",
            "captured_sha256": "a" * 64,
            "captured_bytes": 1,
        }]
        mapped = normalize_member_records(records)
        self.assertEqual(mapped[0]["original_member_name"], "formulas/base.formula")
        self.assertEqual(mapped[0]["member_name"], "formulas/base.formula")

    def test_report_builder_consumes_only_bound_in_memory_bytes_and_emits_member_names(self) -> None:
        payloads = {
            "member_1_base.formula": b"XG:C>MA(C,5);\n",
            "member_2_identity.json": b'{"all_passed":true}',
        }
        captures = [
            {
                "ordinal": 1, "original_member_name": "formulas/base.formula",
                "category": "formula", "safe_basename": "member_1_base.formula",
                "captured_sha256": hashlib.sha256(payloads["member_1_base.formula"]).hexdigest(),
                "captured_bytes": len(payloads["member_1_base.formula"],),
            },
            {
                "ordinal": 2, "original_member_name": "audit/identity.json",
                "category": "identity_audit_json", "safe_basename": "member_2_identity.json",
                "captured_sha256": hashlib.sha256(payloads["member_2_identity.json"]).hexdigest(),
                "captured_bytes": len(payloads["member_2_identity.json"]),
            },
        ]
        report, consumed = build_material_report_from_payloads(
            load_analyzer(), captures, payloads, "b" * 64, 194062)
        self.assertEqual(consumed, 2)
        self.assertEqual(
            [item["member_name"] for item in report["captured_package_files"]],
            ["formulas/base.formula", "audit/identity.json"],
        )
        self.assertEqual(report["s0_static_formula_member_count"], 1)
        self.assertTrue(report["embedded_claims_are_evidence"] is False)

    def test_member_report_cannot_read_an_in_memory_payload_twice(self) -> None:
        from process_archive_material_b2 import _CachedCapturePath

        cached = _CachedCapturePath(b"single-use")
        self.assertEqual(cached.read_bytes(), b"single-use")
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            cached.read_bytes()


if __name__ == "__main__":
    unittest.main()
