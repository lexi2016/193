"""Synthetic checks for native formula inputs and indexed closure wiring."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROUTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (candidate for candidate in ROUTE_ROOT.parents
     if (candidate / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/INPUT_BINDING_R2.json").is_file()),
    ROUTE_ROOT.parents[2],
)
BATCH_ROOT = PROJECT_ROOT / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
FROZEN_ENGINE_ROOT = PROJECT_ROOT / "workspace/research/a_share_model_selection_v1"
sys.path.insert(0, str(FROZEN_ENGINE_ROOT))
sys.path.insert(0, str(ROUTE_ROOT))

import formula_adapter as adapter  # noqa: E402


class SharedCoreInputCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = json.loads((BATCH_ROOT / "INPUT_BINDING_R2.json").read_text(encoding="utf-8"))
        formula_inputs = cls.binding["formula_inputs"]
        cls.v192_text, _ = adapter.exact_formula_source(
            Path(formula_inputs["v192"]["path"]), formula_inputs["v192"]["sha256"],
        )
        cls.v193_text, _ = adapter.exact_formula_source(
            Path(formula_inputs["v193"]["path"]), formula_inputs["v193"]["sha256"],
        )

    @staticmethod
    def _core_source(*, omit: set[str] | None = None, overrides: dict[str, str] | None = None) -> str:
        omitted = set(omit or ())
        expressions = dict.fromkeys(adapter.CORE_OUTPUTS, "1")
        expressions.update(overrides or {})
        return "\n".join(
            f"{name}:={expression};"
            for name, expression in expressions.items()
            if name not in omitted and name not in adapter.BASE_INPUTS
        ) + "\n"

    def test_native_close_input_is_accepted_without_formula_definition(self) -> None:
        v192 = self._core_source()
        v193 = self._core_source()

        adapter.verify_shared_core(v192, v193)

    def test_missing_derived_core_output_is_rejected(self) -> None:
        v192 = self._core_source()
        v193 = self._core_source(omit={"DISTW"})

        with self.assertRaisesRegex(adapter.FormulaBindingError, r"missing_v193=\['DISTW'\]"):
            adapter.verify_shared_core(v192, v193)

    def test_mismatched_shared_expression_is_rejected(self) -> None:
        v192 = self._core_source()
        v193 = self._core_source(overrides={"DISTW": "2"})

        with self.assertRaisesRegex(adapter.FormulaBindingError, r"mismatches=\['DISTW'\]"):
            adapter.verify_shared_core(v192, v193)

    def test_quoted_s_and_st_names_are_not_unbound_formula_dependencies(self) -> None:
        source = 'XG:NAMELIKE("ST") OR CODELIKE(\'S\');\n'

        self.assertEqual(adapter.extract_closure(source, ("XG",)), source)

    def test_unquoted_unknown_formula_name_still_fails_closed(self) -> None:
        for unknown in ("UNBOUND_FLAG", "S", "ST"):
            source = f"XG:NAMELIKE('ST') AND {unknown};\n"
            with self.subTest(unknown=unknown):
                with self.assertRaisesRegex(adapter.FormulaBindingError, rf"unbound names: \['{unknown}'\]"):
                    adapter.extract_closure(source, ("XG",))

    @staticmethod
    def _synthetic_price_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for code_index, code in enumerate(("000001", "000002", "600000")):
            phase = np.arange(len(dates), dtype=np.float64)
            close = (10.0 + 8.0 * code_index) * np.exp(
                np.cumsum(0.0003 + 0.0012 * np.sin(phase / 9.0 + code_index))
            )
            open_ = close * (1.0 + 0.001 * np.sin(phase / 5.0))
            rows.extend({
                "date": date,
                "code": code,
                "open": float(open_value),
                "high": float(max(open_value, close_value) * 1.01),
                "low": float(min(open_value, close_value) * 0.99),
                "close": float(close_value),
                "volume": float(100_000.0 + 1_000.0 * (i % 31)),
                "amount": float(close_value * (100_000.0 + 1_000.0 * (i % 31))),
            } for i, (date, open_value, close_value) in enumerate(zip(dates, open_, close)))
        return pd.DataFrame(rows)

    def test_bound_formula_text_runs_with_synthetic_m0_and_m4_inputs(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=320)
        panel = self._synthetic_price_panel(dates)
        expected_month = np.tile(dates.month.to_numpy(dtype=np.float64), 3)
        expected = panel.assign(
            trade_date=pd.to_datetime(panel["date"]).dt.normalize(),
            ts_code=panel["code"].astype(str),
        ).sort_values(["ts_code", "trade_date"], kind="stable")["close"].to_numpy()

        with patch.object(adapter, "_evaluate_closure", wraps=adapter._evaluate_closure) as evaluate:
            m0 = adapter.formula_feature_frame(
                panel,
                v193_text=self.v193_text,
                v192_text=self.v192_text,
                index_close_by_date=None,
                include_v192_trigger=True,
                include_v193_trigger=False,
                include_auditm=False,
            )
            m0_calls = list(evaluate.call_args_list)

        self.assertEqual(len(m0), len(panel))
        self.assertTrue(set(adapter.CORE_OUTPUTS).issubset(m0.columns))
        self.assertIn("V192_XG", m0.columns)
        self.assertNotIn("V193_XG", m0.columns)
        self.assertNotIn("AUDITM", m0.columns)
        np.testing.assert_allclose(m0["C"].to_numpy(), expected, equal_nan=True)
        self.assertTrue(all("MONTH" in call.args[2] for call in m0_calls))
        for call in m0_calls:
            self.assertIsNone(call.args[2]["INDEXC"])
            np.testing.assert_array_equal(call.args[2]["MONTH"], expected_month)

        index_close = pd.Series(
            4_000.0 * np.exp(np.cumsum(0.0002 + 0.002 * np.sin(np.arange(len(dates)) / 11.0))),
            index=dates,
        )
        with patch.object(adapter, "_evaluate_closure", wraps=adapter._evaluate_closure) as evaluate:
            m4 = adapter.formula_feature_frame(
                panel,
                v193_text=self.v193_text,
                v192_text=self.v192_text,
                index_close_by_date=index_close,
                include_v192_trigger=True,
                include_v193_trigger=True,
                include_auditm=True,
            )
            indexed_calls = list(evaluate.call_args_list)

        self.assertEqual(len(m4), len(panel))
        self.assertTrue(set(adapter.CORE_OUTPUTS + ("V192_XG", "V193_XG", "AUDITM")).issubset(m4.columns))
        np.testing.assert_allclose(m4["C"].to_numpy(), expected, equal_nan=True)
        self.assertTrue(all("MONTH" in call.args[2] for call in indexed_calls))
        expected_index = np.tile(index_close.to_numpy(dtype=np.float64), 3)
        for call in indexed_calls:
            injected_index = call.args[2]["INDEXC"]
            self.assertIsNotNone(injected_index)
            self.assertEqual(len(injected_index), len(panel))
            np.testing.assert_allclose(injected_index, expected_index)
            np.testing.assert_array_equal(call.args[2]["MONTH"], expected_month)
        v193_xg_closure = adapter.extract_closure(self.v193_text, ("XG",))
        v193_xg_inputs = set(adapter._formula_identifiers(v193_xg_closure))
        self.assertIn("INDEXC", v193_xg_inputs)
        self.assertIn("MONTH", v193_xg_inputs)


if __name__ == "__main__":
    unittest.main()
