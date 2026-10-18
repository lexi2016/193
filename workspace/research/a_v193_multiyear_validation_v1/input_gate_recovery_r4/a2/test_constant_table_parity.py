"""Parity controls for the R4 A2 missing SHA-256 round-constant model."""
from __future__ import annotations

import hashlib
import unittest

from compare_json_hash_constant_parity import (
    FAULTY_K,
    JS_ABC_FIXTURE,
    MISSING_CONSTANT,
    MISSING_CONSTANT_INDEX,
    STANDARD_K,
    _js_uint32,
    sha256_missing_constant_js_semantics,
    sha256_reference_standard,
    synthetic_self_check,
)


class MissingConstantParityTests(unittest.TestCase):
    def test_standard_reference_matches_hashlib_on_block_boundaries(self) -> None:
        for sample in (b"", b"abc", bytes(range(64)), bytes(range(256)), bytes(range(251)) * 5):
            with self.subTest(size=len(sample)):
                self.assertEqual(sha256_reference_standard(sample), hashlib.sha256(sample).hexdigest())

    def test_fault_table_has_one_missing_constant_and_final_undefined_round(self) -> None:
        self.assertEqual(len(STANDARD_K), 64)
        self.assertEqual(len(FAULTY_K), 63)
        self.assertEqual(STANDARD_K[MISSING_CONSTANT_INDEX], MISSING_CONSTANT)
        self.assertEqual(MISSING_CONSTANT, 0xA81A664B)
        self.assertEqual(FAULTY_K[MISSING_CONSTANT_INDEX], STANDARD_K[MISSING_CONSTANT_INDEX + 1])
        self.assertEqual(_js_uint32(None), 0)

    def test_corrected_reproducer_matches_exact_historical_javascript_abc_fixture(self) -> None:
        self.assertEqual(
            sha256_missing_constant_js_semantics(b"abc"),
            JS_ABC_FIXTURE,
        )
        self.assertNotEqual(JS_ABC_FIXTURE, hashlib.sha256(b"abc").hexdigest())

    def test_synthetic_self_check_proves_fixture_and_does_not_read_sources(self) -> None:
        self.assertEqual(
            synthetic_self_check(),
            {
                "standard_reference_matches_hashlib": True,
                "faulty_table_shape_matches_one_missing_constant": True,
                "exact_historical_js_abc_fixture_matches": True,
                "javascript_abc_fixture": JS_ABC_FIXTURE,
                "source_json_reads": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
