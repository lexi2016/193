"""Synthetic, no-fit regression checks for the frozen M0 canary boundary."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROUTE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (candidate for candidate in ROUTE_ROOT.parents
     if (candidate / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019/CONTRACT_R2.json").is_file()),
    ROUTE_ROOT.parents[2],
)
BATCH_ROOT = PROJECT_ROOT / "research/a_share_model_selection_v1/batches/pv_environment_r1_20261019"
sys.path.insert(0, str(ROUTE_ROOT))

import model_selection as route  # noqa: E402


class M0CanaryBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = BATCH_ROOT / "CONTRACT_R2.json"
        cls.binding_path = BATCH_ROOT / "INPUT_BINDING_R2.json"
        cls.contract = route.load_json(cls.contract_path)
        cls.binding = route.load_json(cls.binding_path)

    @staticmethod
    def _synthetic_price_panel(trade_dates: object) -> pd.DataFrame:
        row_count = len(trade_dates)
        return pd.DataFrame({
            "trade_date": trade_dates,
            "ts_code": ["000001"] * row_count,
            "open": [10.0] * row_count,
            "high": [11.0] * row_count,
            "low": [9.0] * row_count,
            "close": [10.0] * row_count,
            "vol": [100.0] * row_count,
            "amount": [1000.0] * row_count,
        })

    def _normalize_synthetic_price_panel(self, trade_dates: object) -> pd.DataFrame:
        return route.normalize_price_panel(
            self._synthetic_price_panel(trade_dates),
            self.binding["sources"]["price_panel"]["column_map"],
        )

    def test_normalize_price_panel_decodes_int64_yyyymmdd_dates(self) -> None:
        raw_dates = pd.Series([20210506, 20210507], dtype="int64")

        normalized = self._normalize_synthetic_price_panel(raw_dates)

        self.assertEqual(normalized["date"].tolist(), [
            pd.Timestamp("2021-05-06"), pd.Timestamp("2021-05-07"),
        ])
        self.assertFalse(normalized.duplicated(list(route.KEYS)).any())

    def test_normalize_price_panel_preserves_timestamp_and_iso_dates(self) -> None:
        expected = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        cases = {
            "timestamps": pd.Series([
                pd.Timestamp("2024-01-02 09:30:00"), pd.Timestamp("2024-01-03 15:00:00"),
            ]),
            "iso_strings": pd.Series(["2024-01-02", "2024-01-03"], dtype="string"),
        }

        for input_kind, raw_dates in cases.items():
            with self.subTest(input_kind=input_kind):
                normalized = self._normalize_synthetic_price_panel(raw_dates)
                self.assertEqual(normalized["date"].tolist(), expected)

    def test_normalize_price_panel_rejects_invalid_numeric_date_and_duplicate_key(self) -> None:
        invalid_date = self._synthetic_price_panel(pd.Series([20210229], dtype="int64"))
        with self.subTest(case="invalid_date"):
            with self.assertRaisesRegex(route.ContractError, "missing date/code identity"):
                route.normalize_price_panel(invalid_date, self.binding["sources"]["price_panel"]["column_map"])

        duplicate_dates = self._synthetic_price_panel(pd.Series([20210506, 20210506], dtype="int64"))
        with self.subTest(case="duplicate_key"):
            with self.assertRaisesRegex(route.ContractError, "duplicate date/code identity"):
                route.normalize_price_panel(duplicate_dates, self.binding["sources"]["price_panel"]["column_map"])

    def test_current_binding_and_stage_plan_keep_m0_canary_independent(self) -> None:
        readiness = self.binding["readiness_by_arm"]
        self.assertEqual(readiness["M0"]["status"], "READY_WITH_SCOPE_LIMITATIONS")
        self.assertFalse(readiness["M0"]["uses_index_series"])
        for arm in route.INDEXED_ARMS:
            self.assertEqual(readiness[arm]["status"], "READY_AFTER_INDEX_BINDING")
            self.assertTrue(any("ORIGINAL_TDX_FILE_IDENTITY_CANNOT_CONFIRM" in item
                                for item in readiness[arm]["scope_limitations"]))

        m0_canary = route.stage_plan("M0_CANARY")
        self.assertEqual([(cell["arm"], cell["horizon"], cell["fold"].fold_id)
                          for cell in m0_canary],
                         [("M0", 1, "2024H1"), ("M0", 2, "2024H1"), ("M0", 5, "2024H1")])
        self.assertEqual(len(route.stage_plan("M0_FORMAL")), 18)
        self.assertEqual(len(route.stage_plan("M4_CANARY")), 3)
        self.assertEqual(len(route.stage_plan("INDEXED_FORMAL")), 72)
        self.assertEqual(len(m0_canary) + len(route.stage_plan("M0_FORMAL")) +
                         len(route.stage_plan("M4_CANARY")) +
                         len(route.stage_plan("INDEXED_FORMAL")), 96)
        # M0 canary opens only its own input arm. Formal M0 shares the all-arm key set.
        self.assertEqual(route.STAGE_SPECS["M0_CANARY"]["input_arms"], ("M0",))
        self.assertEqual(route.STAGE_SPECS["M0_FORMAL"]["input_arms"], route.ARMS)
        self.assertTrue(route.STAGE_SPECS["M0_FORMAL"]["requires_common_keys"])

    def test_missing_review_receipt_blocks_before_any_parquet_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            activation_path = Path(tmp) / "activation.json"
            activation_path.write_text("{}\n", encoding="utf-8")
            with patch.object(route.pd, "read_parquet") as read_parquet:
                with self.assertRaises(route.ExecutionBlocked):
                    route.read_bound_panel(
                        contract_path=self.contract_path,
                        binding_path=self.binding_path,
                        activation_path=activation_path,
                        stage="M0_CANARY",
                        arms=("M0",),
                        preparation_claim={"execution_id": "SYNTHETIC_NO_FIT"},
                    )
                read_parquet.assert_not_called()

    def test_actual_frozen_contract_accepts_exact_independent_m0_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "synthetic_reviewer_receipt.json"
            receipt_path.write_text('{"status":"REVIEWED_SYNTHETIC_ACTIVATION_SMOKE"}\n', encoding="utf-8")
            activation = {
                "status": "APPROVED_FOR_M0_ONLY",
                "reviewer_identity": "independent_reviewer_identity",
                "allowed_arms": ["M0"],
                "allowed_stages": ["M0_CANARY"],
                "contract_sha256": route.sha256_file(self.contract_path),
                "input_binding_sha256": route.sha256_file(self.binding_path),
                "code_sha256": route.route_code_sha256(),
                "reviewer_receipt_path": str(receipt_path),
                "reviewer_receipt_sha256": route.sha256_file(receipt_path),
                "allow_real_data_read": True,
                "allow_fit": True,
            }
            route.validate_review_activation(
                self.contract, self.binding, activation,
                contract_sha256=route.sha256_file(self.contract_path),
                binding_sha256=route.sha256_file(self.binding_path),
                code_sha256=route.route_code_sha256(), stage="M0_CANARY", arms=("M0",),
            )

    def test_actual_governance_writer_cannot_approve_its_own_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt_path = Path(tmp) / "synthetic_reviewer_receipt.json"
            receipt_path.write_text('{"status":"REVIEWED_SYNTHETIC_ACTIVATION_SMOKE"}\n', encoding="utf-8")
            activation = {
                "status": "APPROVED_FOR_M0_ONLY",
                "reviewer_identity": self.contract["governance"]["writer_identity"],
                "allowed_arms": ["M0"],
                "allowed_stages": ["M0_CANARY"],
                "contract_sha256": route.sha256_file(self.contract_path),
                "input_binding_sha256": route.sha256_file(self.binding_path),
                "code_sha256": route.route_code_sha256(),
                "reviewer_receipt_path": str(receipt_path),
                "reviewer_receipt_sha256": route.sha256_file(receipt_path),
                "allow_real_data_read": True,
                "allow_fit": True,
            }
            with self.assertRaises(route.ExecutionBlocked):
                route.validate_review_activation(
                    self.contract, self.binding, activation,
                    contract_sha256=route.sha256_file(self.contract_path),
                    binding_sha256=route.sha256_file(self.binding_path),
                    code_sha256=route.route_code_sha256(), stage="M0_CANARY", arms=("M0",),
                )

    def test_m0_nested_frame_builds_without_index_series(self) -> None:
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        keys = pd.DataFrame({"date": [dates[0], dates[1]], "code": ["000001", "000001"]})
        formula_features = keys.copy()
        for feature in route.CORE_FEATURES:
            formula_features[feature] = np.array([1.0, 2.0])
        panel = keys.assign(close=[10.0, 11.0], amount=[100.0, 110.0])
        calendar = pd.DataFrame({"date": dates, "calendar_index": [0, 1]})

        frames = route.build_nested_arm_frames(
            panel,
            formula_features,
            calendar,
            index_close_by_date=None,
            requested_arms=("M0",),
        )
        self.assertEqual(tuple(frames), ("M0",))
        self.assertEqual(set(route.KEYS + route.CORE_FEATURES).difference(frames["M0"].columns), set())
        self.assertEqual(len(frames["M0"]), 2)

    def test_close_labels_use_shared_calendar_and_do_not_skip_missing_stock_sessions(self) -> None:
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        calendar = pd.DataFrame({"date": dates, "calendar_index": range(len(dates))})
        panel = pd.DataFrame([
            {"date": date, "code": "000001", "close": close}
            for date, close in zip(dates, [10.0, 11.0, 12.0, 13.0])
        ] + [
            {"date": dates[0], "code": "000002", "close": 20.0},
            {"date": dates[2], "code": "000002", "close": 24.0},
            {"date": dates[3], "code": "000002", "close": 25.0},
        ]).sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)

        labels = route.build_close_return_labels(panel, horizons=(1, 2), calendar=calendar)
        b_h1 = labels[1].loc[labels[1]["code"].eq("000002")].set_index("date")
        b_h2 = labels[2].loc[labels[2]["code"].eq("000002")].set_index("date")
        self.assertTrue(np.isnan(b_h1.loc[dates[0], "label_return"]))
        self.assertEqual(b_h1.loc[dates[0], "target_date"], dates[1])
        self.assertAlmostEqual(b_h2.loc[dates[0], "label_return"], 0.2)
        self.assertTrue(np.isnan(b_h1.loc[dates[3], "label_return"]))

    def test_fold_purges_labels_reaching_test_start_and_keeps_test_unlabeled(self) -> None:
        fold = route.Fold("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"), False)
        features = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2023-12-28", "2024-01-02", "2024-02-01"]),
            "code": ["000001"] * 4,
            "f0": [0.0, 1.0, 2.0, 3.0],
        })
        labels = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-03", "2023-12-28"]),
            "code": ["000001"] * 2,
            "target_date": pd.to_datetime(["2022-01-04", "2024-01-01"]),
            "label_end_date": pd.to_datetime(["2022-01-04", "2024-01-01"]),
            "label_return": [0.01, 0.02],
        })
        train, test = route.split_fold(features, labels, fold)
        self.assertEqual(train["date"].tolist(), [pd.Timestamp("2022-01-03")])
        self.assertEqual(train["label_end_date"].max(), pd.Timestamp("2022-01-04"))
        self.assertEqual(test["date"].tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")])
        self.assertFalse({"label_return", "target_date", "label_end_date"}.intersection(test.columns))

    def test_top10_ties_are_code_stable_and_nonfinite_rows_are_not_refilled(self) -> None:
        date = pd.Timestamp("2024-03-01")
        tied = pd.DataFrame({
            "date": [date] * 11,
            "code": [f"{number:06d}" for number in range(11)],
            "prediction": [5.0] * 10 + [4.0],
        })
        selected = route.stable_top10(tied)
        self.assertEqual(selected["code"].tolist(), [f"{number:06d}" for number in range(10)])
        self.assertEqual(selected["rank"].tolist(), list(range(1, 11)))

        sparse = tied.iloc[:10].copy()
        sparse.loc[sparse.index[-1], "prediction"] = np.nan
        sparse_selected = route.stable_top10(sparse)
        self.assertEqual(len(sparse_selected), 9)
        self.assertNotIn("000010", sparse_selected["code"].tolist())

    def test_canary_subset_uses_deterministic_codes_and_recent_training_dates(self) -> None:
        train_dates = pd.date_range("2023-10-01", periods=62, freq="B")
        test_dates = pd.date_range("2024-01-02", periods=12, freq="B")
        codes = [f"{number:06d}" for number in range(201)]
        train = pd.MultiIndex.from_product([train_dates, codes], names=["date", "code"]).to_frame(index=False)
        test = pd.MultiIndex.from_product([test_dates, codes], names=["date", "code"]).to_frame(index=False)
        train["f0"] = 1.0
        test["f0"] = 2.0

        train_slice, test_slice = route.canary_subset(train, test)
        self.assertEqual(train_slice["code"].nunique(), 200)
        self.assertEqual(test_slice["code"].nunique(), 200)
        self.assertEqual(sorted(train_slice["code"].unique()), codes[:200])
        self.assertEqual(train_slice["date"].nunique(), 60)
        self.assertEqual(train_slice["date"].min(), train_dates[2])
        self.assertEqual(train_slice["date"].max(), train_dates[-1])
        self.assertEqual(test_slice["date"].nunique(), 10)
        self.assertEqual(test_slice["date"].min(), test_dates[0])
        self.assertEqual(test_slice["date"].max(), test_dates[9])

    def test_m0_checkpoint_identity_is_reusable_after_later_index_binding(self) -> None:
        fold = route.stage_plan("M0_CANARY")[0]["fold"]
        before = route.checkpoint_identity(
            stage="M0_CANARY", fold=fold, arm="M0", horizon=1, attempt="attempt_01",
            contract_sha256="b" * 64, code_shas={"model_selection.py": "a" * 64},
            binding=self.binding, feature_names=route.CORE_FEATURES,
            training_key_sha256="c" * 64,
        )
        after_binding = copy.deepcopy(self.binding)
        after_binding["sources"]["index_series"] = {"status": "BOUND", "sha256": "d" * 64}
        after = route.checkpoint_identity(
            stage="M0_CANARY", fold=fold, arm="M0", horizon=1, attempt="attempt_02",
            contract_sha256="b" * 64, code_shas={"model_selection.py": "a" * 64},
            binding=after_binding, feature_names=route.CORE_FEATURES,
            training_key_sha256="c" * 64,
        )
        self.assertEqual(before["input_components"], after["input_components"])
        self.assertNotIn("index_series", before["input_components"])
        self.assertNotIn("index_series", after["input_components"])

        context = route.build_registry_event_context(
            stage="M0_CANARY", binding=after_binding,
            contract_sha256="b" * 64, binding_sha256="e" * 64,
            code_shas={"model_selection.py": "a" * 64}, config_sha256="f" * 64,
            arms=("M0",), truth_access="LABELS_FOR_TRAINING_ONLY",
        )
        self.assertNotIn("index_series", context["input_identity"]["input_component_sha256"])
        self.assertFalse(any("index_000300" in path for path in context["input_paths"]))


if __name__ == "__main__":
    unittest.main()
