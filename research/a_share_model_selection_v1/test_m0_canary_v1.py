"""Synthetic, no-fit regression checks for the frozen M0 canary boundary."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
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


class FixedScoreModel:
    """Predict-only synthetic estimator; replay tests never fit a model."""

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64)[:, 0] * 0.5 + np.asarray(matrix, dtype=np.float64)[:, 1]


class SyntheticNoFitGuard:
    """In-memory protocol stub for exercising the whole no-fit audit lifecycle."""

    def __init__(self, registry_path: Path, *, claim_error: BaseException | None = None,
                 terminal_error: BaseException | None = None) -> None:
        self.registry_path = Path(registry_path)
        self.claim_error = claim_error
        self.terminal_error = terminal_error
        self.context: dict[str, object] = {}
        self.execution_id: str | None = None
        self.claim = None
        self.terminal: dict[str, object] | None = None
        self.backend = self

    def set_execution_context(self, execution_id: str, context: object) -> None:
        self.context[execution_id] = dict(context)

    def update_execution_context(self, execution_id: str, **updates: object) -> None:
        self.context[execution_id].update(updates)

    def accounting(self) -> SimpleNamespace:
        dangling = (self.execution_id,) if self.execution_id else ()
        return SimpleNamespace(dangling_execution_ids=dangling)

    def _append_event(self, event: dict[str, object]) -> None:
        with self.registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def claim_no_fit(self, *, execution_id: str, arm: str, contract_sha256: str,
                     input_binding_sha256: str) -> SimpleNamespace:
        if self.claim_error is not None:
            raise self.claim_error
        self.execution_id = execution_id
        self.claim = SimpleNamespace(execution_id=execution_id, arm=arm,
                                     contract_sha256=contract_sha256,
                                     input_binding_sha256=input_binding_sha256)
        self._append_event({"event_type": "RUN_STARTED", "execution_id": execution_id,
                            "execution_kind": "NO_FIT_ENGINEERING_CANARY", "real_fit_budget_debit": 0})
        return self.claim

    def append_no_fit_terminal(self, claim: SimpleNamespace, *, succeeded: bool,
                               error_type: str | None = None) -> None:
        if self.terminal_error is not None:
            raise self.terminal_error
        self.terminal = {"succeeded": succeeded, "error_type": error_type}
        self._append_event({"event_type": "TERMINAL", "execution_id": claim.execution_id,
                            "terminal_status": "ENGINEERING_CANARY_ONLY" if succeeded
                            else "CANNOT_CONFIRM_ENGINEERING_CANARY",
                            "error_type": error_type, "execution_kind": "NO_FIT_ENGINEERING_CANARY",
                            "real_fit_budget_debit": 0})


class SyntheticPinnedRoute:
    """Small deterministic route surface for a no-input/no-fit replay integration test."""

    def __init__(self, features: pd.DataFrame, panel: pd.DataFrame,
                 code_shas: dict[str, str]) -> None:
        self.features = features
        self.panel = panel
        self.code_shas = code_shas

    def load_stage_inputs(self, **_: object) -> tuple[object, object, dict[str, pd.DataFrame], object, list[object]]:
        return self.panel, None, {"M0": self.features}, None, []

    def fixed_folds(self) -> tuple[object, ...]:
        return route.fixed_folds()

    def build_close_return_labels(self, panel: pd.DataFrame, *, horizons: tuple[int, ...], **_: object) -> dict[int, pd.DataFrame]:
        rows = self.features.loc[self.features["date"] < pd.Timestamp("2024-01-01"), ["date", "code"]].copy()
        out = {}
        for horizon in horizons:
            labels = rows.copy()
            labels["target_date"] = labels["date"] + pd.Timedelta(days=horizon)
            labels["label_end_date"] = labels["target_date"]
            labels["label_return"] = np.arange(len(labels), dtype=np.float64) / max(1, len(labels))
            out[int(horizon)] = labels
        return out

    @staticmethod
    def _finite_arm_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
        return route._finite_arm_frame(frame)

    @staticmethod
    def split_fold(features: pd.DataFrame, labels: pd.DataFrame, fold: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        return route.split_fold(features, labels, fold)

    @staticmethod
    def canary_subset(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return route.canary_subset(train, test)

    @staticmethod
    def equal_day_weights(values: object) -> np.ndarray:
        return route.equal_day_weights(values)

    @staticmethod
    def stable_top10(predictions: pd.DataFrame) -> pd.DataFrame:
        return route.stable_top10(predictions)

    def checkpoint_identity(self, *, stage: str, fold: object, arm: str, horizon: int,
                            attempt: str, contract_sha256: str,
                            code_shas: dict[str, str], binding: dict[str, object],
                            feature_names: tuple[str, ...], training_key_sha256: str) -> dict[str, object]:
        return {"stage": stage, "fold": fold.fold_id, "arm": arm, "horizon": horizon,
                "attempt": attempt, "contract_sha256": contract_sha256,
                "code_sha256": dict(sorted(code_shas.items())),
                "input_components": {"synthetic_binding": binding.get("fixture_id", "FIXTURE")},
                "feature_names": list(feature_names), "training_key_sha256": training_key_sha256}


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
        train = pd.DataFrame({
            "date": pd.to_datetime(["2023-12-01", "2023-12-01", "2023-12-04"]),
            "code": ["000001", "000002", "000001"], "f0": [1.0, 2.0, 3.0],
            "label_return": [0.01, 0.02, -0.01],
        })
        test = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                             "code": ["000001", "000002"], "f0": [4.0, 5.0]})
        x = train[["f0"]].to_numpy(dtype=np.float64)
        y = train["label_return"].to_numpy(dtype=np.float64)
        weights = route.equal_day_weights(train["date"])
        xt = test[["f0"]].to_numpy(dtype=np.float64)
        data_identity = route.cell_data_identity(train=train, test=test, feature_columns=("f0",),
                                                 x=x, y=y, weights=weights, xt=xt)
        before = route.checkpoint_identity(
            stage="M0_CANARY", fold=fold, arm="M0", horizon=1, attempt="attempt_01",
            contract_sha256="b" * 64, code_shas={"model_selection.py": "a" * 64},
            binding=self.binding, feature_names=("f0",),
            data_identity=data_identity,
        )
        after_binding = copy.deepcopy(self.binding)
        after_binding["sources"]["index_series"] = {"status": "BOUND", "sha256": "d" * 64}
        after = route.checkpoint_identity(
            stage="M0_CANARY", fold=fold, arm="M0", horizon=1, attempt="attempt_02",
            contract_sha256="b" * 64, code_shas={"model_selection.py": "a" * 64},
            binding=after_binding, feature_names=("f0",),
            data_identity=data_identity,
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

    def test_sorted_key_identity_is_order_independent_and_order_hash_is_explicit(self) -> None:
        rows = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
                             "code": ["000002", "000001"]})
        first = route.key_frame_identity(rows)
        reordered = route.key_frame_identity(rows.iloc[::-1].reset_index(drop=True))

        self.assertEqual(first["schema_version"], route.KEY_IDENTITY_SCHEMA)
        self.assertEqual(first["row_count"], 2)
        self.assertEqual(first["unique_key_count"], 2)
        self.assertEqual(first["sorted_keyset_sha256"], reordered["sorted_keyset_sha256"])
        self.assertNotEqual(first["ordered_keys_sha256"], reordered["ordered_keys_sha256"])
        self.assertEqual(first["date_min"], "2024-01-02")
        self.assertEqual(first["date_max"], "2024-01-03")

    def test_saved_checkpoint_replay_is_no_fit_exact_and_preserves_artifacts(self) -> None:
        test_dates = pd.to_datetime(["2024-01-02"] * 12 + ["2024-01-03"] * 12)
        test_keys = pd.DataFrame({"date": test_dates,
                                  "code": [f"{number:06d}" for number in range(12)] * 2})
        xt = np.column_stack((np.arange(24, dtype=np.float64),
                              np.linspace(0.0, 1.0, 24, dtype=np.float64)))
        test_features = test_keys.assign(f0=xt[:, 0], f1=xt[:, 1])
        train = pd.DataFrame({"date": pd.to_datetime(["2023-12-01", "2023-12-01", "2023-12-04"]),
                              "code": ["000001", "000002", "000001"],
                              "f0": [1.0, 2.0, 3.0], "f1": [0.0, 1.0, 2.0],
                              "label_return": [0.01, 0.02, -0.01]})
        x = train[["f0", "f1"]].to_numpy(dtype=np.float64)
        y = train["label_return"].to_numpy(dtype=np.float64)
        weights = route.equal_day_weights(train["date"])
        data_identity = route.cell_data_identity(train=train, test=test_features, feature_columns=("f0", "f1"),
                                                 x=x, y=y, weights=weights, xt=xt)
        fold = route.stage_plan("M0_CANARY")[0]["fold"]
        identity = route.checkpoint_identity(
            stage="M0_CANARY", fold=fold, arm="M0", horizon=1, attempt="attempt_synthetic",
            contract_sha256=route.sha256_file(self.contract_path),
            code_shas={"model_selection": route.sha256_file(Path(route.__file__))},
            binding=self.binding, feature_names=("f0", "f1"), data_identity=data_identity,
        )
        model = FixedScoreModel()
        predictions = test_keys.copy()
        predictions["prediction"] = model.predict(xt)
        payload = {"model": model, "predictions": predictions,
                   "feature_columns": ("f0", "f1"),
                   "replay_fixture": {"test_keys": test_keys, "test_features": xt}}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "synthetic_checkpoint.joblib"
            replay_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".replay.json")
            route.save_checkpoint(checkpoint_path, payload, identity)
            checkpoint_sha = route.sha256_file(checkpoint_path)
            metadata_sha = route.sha256_file(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"))

            receipt = route.verify_saved_checkpoint_replay(
                checkpoint_path=checkpoint_path, replay_receipt_path=replay_path,
                identity=identity, execution_id="SYNTHETIC_NO_FIT_REPLAY",
                provenance="RECONSTRUCTED_FROM_EXISTING_ARTIFACTS",
            )

            self.assertEqual(receipt["status"], "VERIFIED")
            self.assertEqual(receipt["provenance"], "RECONSTRUCTED_FROM_EXISTING_ARTIFACTS")
            self.assertEqual(receipt["comparison"], {"join_validation": "ONE_TO_ONE", "rtol": 0.0,
                                                       "atol": 0.0, "prediction_equal": True,
                                                       "key_order_equal": True})
            self.assertEqual(receipt["top10_parity"]["status"], "EXACT")
            self.assertEqual(route.sha256_file(checkpoint_path), checkpoint_sha)
            self.assertEqual(route.sha256_file(checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")),
                             metadata_sha)
            self.assertEqual(route.load_json(replay_path)["checkpoint_sha256"], checkpoint_sha)

    def test_original_fit_registry_pair_is_bound_to_exact_source_and_rows(self) -> None:
        code_sha = "a" * 64
        common = {
            "experiment_id": route.EXPERIMENT_ID,
            "execution_id": "M0_CANARY_M0_H1_2024H1_attempt_04",
            "route_id": "a-share-model-selection-pv-environment-r1-20261019",
            "registration_mode": "LIVE_EXECUTION",
            "code_sha256": code_sha,
            "contract_sha256": "b" * 64,
            "input_binding_sha256": "c" * 64,
            "event_sha256": "d" * 64,
        }
        start = {**common, "event_type": "RUN_STARTED", "stage": "M0_CANARY_RUN_STARTED",
                 "deterministic_metrics": {"training_rows": 11951}}
        terminal = {**common, "event_type": "TERMINAL", "stage": "M0_CANARY_TERMINAL",
                    "event_sha256": "e" * 64, "decision": "COMPLETED",
                    "deterministic_metrics": {
                        "execution_kind": "REAL_FIT", "fit_call_entered": True,
                        "planned_real_fit_count": 1, "real_fit_budget_debit": 1,
                        "real_fit_returned_count": 1,
                    },
                    "input_identity": {"artifact_sha256": {"checkpoint": "f" * 64}}}

        summary = route._verify_original_fit_event_pair(
            [start, terminal], execution_id=common["execution_id"], stage="M0_CANARY",
            experiment_id=route.EXPERIMENT_ID, expected_route_code_sha256=code_sha,
            contract_sha256="b" * 64, binding_sha256="c" * 64,
        )

        self.assertEqual(summary["run_started_event_sha256"], "d" * 64)
        self.assertEqual(summary["terminal_event_sha256"], "e" * 64)
        self.assertEqual(summary["training_rows_from_start_event"], 11951)
        self.assertEqual(summary["terminal_artifact_sha256"]["checkpoint"], "f" * 64)
        rejected_terminals = [
            {**terminal, "decision": "FAILED"},
            {**terminal, "deterministic_metrics": {**terminal["deterministic_metrics"],
                                                     "execution_kind": "NO_FIT"}},
            {**terminal, "deterministic_metrics": {**terminal["deterministic_metrics"],
                                                     "fit_call_entered": False}},
            {**terminal, "deterministic_metrics": {**terminal["deterministic_metrics"],
                                                     "planned_real_fit_count": 0}},
            {**terminal, "deterministic_metrics": {**terminal["deterministic_metrics"],
                                                     "real_fit_budget_debit": 0}},
            {**terminal, "deterministic_metrics": {**terminal["deterministic_metrics"],
                                                     "real_fit_returned_count": 0}},
            {**terminal, "deterministic_metrics": {}},
        ]
        for rejected_terminal in rejected_terminals:
            with self.subTest(metrics=rejected_terminal.get("deterministic_metrics"),
                              decision=rejected_terminal.get("decision")):
                with self.assertRaisesRegex(route.ExecutionBlocked, "does not confirm one returned fit"):
                    route._verify_original_fit_event_pair(
                        [start, rejected_terminal], execution_id=common["execution_id"], stage="M0_CANARY",
                        experiment_id=route.EXPERIMENT_ID, expected_route_code_sha256=code_sha,
                        contract_sha256="b" * 64, binding_sha256="c" * 64,
                    )
        terminal["code_sha256"] = "9" * 64
        with self.assertRaisesRegex(route.ExecutionBlocked, "identity mismatch"):
            route._verify_original_fit_event_pair(
                [start, terminal], execution_id=common["execution_id"], stage="M0_CANARY",
                experiment_id=route.EXPERIMENT_ID, expected_route_code_sha256=code_sha,
                contract_sha256="b" * 64, binding_sha256="c" * 64,
            )

    def test_historical_audit_claims_no_fit_before_manifest_scope_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "stage_manifest.json"
            manifest_path.write_text('{"stage":"M4_CANARY","attempt":"attempt_02","fit_calls":0}\n',
                                     encoding="utf-8")
            source_archive = Path(tmp) / "source.zip"
            dependency_archive = Path(tmp) / "dependency.zip"
            for archive_path in (source_archive, dependency_archive):
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("placeholder.txt", "synthetic")
            registry_path = Path(tmp) / "registry.jsonl"
            registry_path.write_text("", encoding="utf-8")
            guard = SyntheticNoFitGuard(registry_path)
            with patch.object(route, "make_official_fit_guard", return_value=guard) as make_guard:
                with self.assertRaisesRegex(route.ExecutionBlocked, "original stage manifest"):
                    route.run_checkpoint_replay_audit(
                        stage="M0_CANARY", attempt="attempt_04",
                        manifest_path=manifest_path,
                        expected_manifest_sha256=route.sha256_file(manifest_path),
                        source_archive_path=source_archive,
                        expected_source_archive_sha256=route.sha256_file(source_archive),
                        dependency_archive_path=dependency_archive,
                        expected_dependency_archive_sha256=route.sha256_file(dependency_archive),
                        contract_path=self.contract_path, binding_path=self.binding_path,
                        activation_path=BATCH_ROOT / "REVIEW_ACTIVATION_CANARY_R4.json",
                        receipt_path=Path(tmp) / "receipt.json",
                        execution_id="SYNTHETIC_WRONG_SCOPE",
                        registry_path=registry_path,
                    )
                make_guard.assert_called_once()
            receipt = route.load_json(Path(tmp) / "receipt.json")
            self.assertEqual(receipt["status"], "CANNOT_CONFIRM_RECONSTRUCTED_REPLAY")
            self.assertEqual(receipt["preflight_status"], "FAILED")
            self.assertEqual(receipt["error"], "original stage manifest does not identify the requested three-cell canary")
            self.assertEqual(guard.terminal, {"succeeded": False, "error_type": "ExecutionBlocked"})
            self.assertEqual([json.loads(line)["event_type"] for line in registry_path.read_text().splitlines()],
                             ["RUN_STARTED", "TERMINAL"])

    def test_pinned_archive_loader_preserves_engine_runtime_relative_layout(self) -> None:
        model_source = b'''from pathlib import Path\nimport hashlib\nPROJECT_ROOT = Path(".")\nFROZEN_ROUTE_DEPENDENCY_ROOT = Path(".")\ndef _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()\ndef code_identity():\n return {"model_selection":_sha(Path(__file__)), "formula_adapter":_sha(FROZEN_ROUTE_DEPENDENCY_ROOT/"formula_adapter.py"), "tdx_ast_engine_injected":_sha(FROZEN_ROUTE_DEPENDENCY_ROOT/"tdx_ast_engine_injected.py"), "fast_builder_c":_sha(PROJECT_ROOT/"workspace/research/a_v193_multiyear_validation_v1/s0_recovery_r7/private/engine_runtime/fast_builder_c.py")}\ndef runtime_value():\n import tdx_ast_engine_injected as engine\n return engine.fbc.RUNTIME_VALUE\n'''
        formula_source = b"FORMULA_VALUE = 'frozen'\n"
        engine_source = b'''from pathlib import Path\nimport importlib.util\n_runtime_root = Path(__file__).resolve().parents[2] / "research" / "a_v193_multiyear_validation_v1" / "s0_recovery_r7" / "private" / "engine_runtime"\nspec_fb = importlib.util.spec_from_file_location("fbc", _runtime_root / "fast_builder_c.py")\nfbc = importlib.util.module_from_spec(spec_fb)\nspec_fb.loader.exec_module(fbc)\n'''
        fast_builder_source = b"RUNTIME_VALUE = 'sha-bound-layout-ok'\n"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            project_root = tmp_path / "original_project"
            runtime_source = (project_root / "workspace/research/a_v193_multiyear_validation_v1"
                              / "s0_recovery_r7/private/engine_runtime/fast_builder_c.py")
            runtime_source.parent.mkdir(parents=True)
            runtime_source.write_bytes(fast_builder_source)
            source_archive = tmp_path / "source.zip"
            dependency_archive = tmp_path / "dependencies.zip"
            with zipfile.ZipFile(source_archive, "w") as archive:
                archive.writestr("model_selection.py", model_source)
            with zipfile.ZipFile(dependency_archive, "w") as archive:
                archive.writestr("formula_adapter.py", formula_source)
                archive.writestr("tdx_ast_engine_injected.py", engine_source)
            code_shas = {
                "model_selection": hashlib.sha256(model_source).hexdigest(),
                "formula_adapter": hashlib.sha256(formula_source).hexdigest(),
                "tdx_ast_engine_injected": hashlib.sha256(engine_source).hexdigest(),
                "fast_builder_c": hashlib.sha256(fast_builder_source).hexdigest(),
            }
            with route._pinned_route_from_archives(
                source_archive_path=source_archive,
                expected_source_archive_sha256=route.sha256_file(source_archive),
                dependency_archive_path=dependency_archive,
                expected_dependency_archive_sha256=route.sha256_file(dependency_archive),
                expected_code_shas=code_shas, project_root=project_root,
                contract_path=self.contract_path, binding_path=self.binding_path,
                activation_path=BATCH_ROOT / "REVIEW_ACTIVATION_CANARY_R4.json",
            ) as pinned:
                self.assertEqual(pinned.code_identity(), code_shas)
                self.assertEqual(pinned.runtime_value(), "sha-bound-layout-ok")

    def test_frozen_archive_import_resolves_sha_bound_native_runtime(self) -> None:
        manifest_path = (PROJECT_ROOT / "workspace/research/a_share_model_selection_v1/runs"
                         / "pv_environment_r1_20261019_m0_canary_r01"
                         / "M0_CANARY_attempt_04_stage_manifest.json")
        original_manifest = route.load_json(manifest_path)
        expected_code_shas = {
            "model_selection": "995c28d43b6f4cab6a2624d19e450ebea3527544d3ed3dac4240cb28fe4dc52b",
            "formula_adapter": "862d4835e3e0ee0e47833e7be57c0dfc584b30cddbbc5a43176b3d60b2654f94",
            "tdx_ast_engine_injected": "d4af457232e14dc54c38d8b5593a3a8ce8783d4805d85976deccd073ba1dea9c",
            "fast_builder_c": "a66bc2d0bf3eb6b4e999bb2adedd8bc6d1f3b63fe9646b2b05a5c91f12a611b0",
        }
        self.assertEqual(original_manifest["code_sha256"], expected_code_shas)

        source_archive = BATCH_ROOT / "PRE_SESSION_LAG_DELTA_SHARED_SOURCE.zip"
        dependency_archive = BATCH_ROOT / "M4_CANARY_ATTEMPT02_FROZEN_SOURCE.zip"
        runtime_root = (PROJECT_ROOT / "workspace/research/a_v193_multiyear_validation_v1"
                        / "s0_recovery_r7/private/engine_runtime")
        native_library = runtime_root / "fast_loops.dylib"
        runtime_manifest = route.load_json(runtime_root / "ENGINE_RUNTIME_MANIFEST.json")

        with route._pinned_route_from_archives(
            source_archive_path=source_archive,
            expected_source_archive_sha256="48d4c9ce11488ab92b81030f040018973d7887f8db4d74785ba112b24dbec2df",
            dependency_archive_path=dependency_archive,
            expected_dependency_archive_sha256="a4212594043691240119f6ee040284df4090f4ffa839570eae702b233e526c5e",
            expected_code_shas=expected_code_shas, project_root=PROJECT_ROOT,
            contract_path=BATCH_ROOT / "CONTRACT_R2.json",
            binding_path=BATCH_ROOT / "INPUT_BINDING_R2.json",
            activation_path=BATCH_ROOT / "REVIEW_ACTIVATION_M0_R2.json",
        ) as pinned:
            engine = importlib.import_module("tdx_ast_engine_injected")
            adapter = importlib.import_module("formula_adapter")
            self.assertEqual(pinned.code_identity(), expected_code_shas)
            self.assertTrue(Path(engine.__file__).resolve().is_relative_to(
                pinned.FROZEN_ROUTE_DEPENDENCY_ROOT.resolve()))
            self.assertTrue(Path(adapter.__file__).resolve().is_relative_to(
                pinned.FROZEN_ROUTE_DEPENDENCY_ROOT.resolve()))
            self.assertEqual(Path(engine.fbc.__file__).resolve(),
                             (pinned.PROJECT_ROOT / "workspace/research/a_v193_multiyear_validation_v1"
                              / "s0_recovery_r7/private/engine_runtime/fast_builder_c.py").resolve())
            self.assertEqual(Path(engine.fbc._lib._name).resolve(), native_library.resolve())
            self.assertTrue(native_library.is_file())
            self.assertEqual(route.sha256_file(native_library),
                             runtime_manifest["derived_files"]["fast_loops_dylib_sha256"])

    def test_historical_audit_reconstructs_finite_fit_slice_without_rewriting_raw_commitment(self) -> None:
        attempt = "attempt_synthetic"
        stage = "M0_CANARY"
        train_dates = pd.bdate_range("2023-09-01", periods=60)
        test_dates = pd.bdate_range("2024-01-02", periods=10)
        codes = [f"SYN{value:03d}" for value in range(200)]
        dates = train_dates.append(test_dates)
        features = pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), len(codes)),
            "code": np.tile(codes, len(dates)),
            "f0": np.tile(np.arange(len(codes), dtype=np.float64), len(dates)),
            "f1": np.repeat(np.arange(len(dates), dtype=np.float64), len(codes)),
        })
        high_precision_score = 0.009955564222698996
        first_test_date = features["date"].eq(test_dates[0])
        features.loc[first_test_date, "f0"] = 0.0
        features.loc[first_test_date, "f1"] = -np.arange(len(codes), dtype=np.float64)
        exact_score_row = first_test_date & features["code"].eq(codes[0])
        features.loc[exact_score_row, "f1"] = high_precision_score
        features.loc[(features["date"] == train_dates[0]) & features["code"].eq(codes[0]), "f1"] = np.nan
        panel = features[["date", "code"]].assign(close=10.0)
        code_shas = {"model_selection": "1" * 64, "formula_adapter": "2" * 64,
                     "tdx_ast_engine_injected": "3" * 64, "fast_builder_c": "4" * 64}
        pinned = SyntheticPinnedRoute(features, panel, code_shas)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            contract_path = tmp_path / "CONTRACT_R2.json"
            binding_path = tmp_path / "INPUT_BINDING_R2.json"
            activation_path = tmp_path / "REVIEW_ACTIVATION_R2.json"
            contract_path.write_text('{"status":"SYNTHETIC"}\n', encoding="utf-8")
            binding_path.write_text('{"fixture_id":"FIXTURE"}\n', encoding="utf-8")
            activation_path.write_text('{"activation":"SYNTHETIC"}\n', encoding="utf-8")
            contract_sha = route.sha256_file(contract_path)
            binding_sha = route.sha256_file(binding_path)
            source_archive = tmp_path / "source.zip"
            dependency_archive = tmp_path / "dependency.zip"
            for archive_path in (source_archive, dependency_archive):
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("placeholder.txt", "synthetic archive fixture")

            labels_by_horizon: dict[int, pd.DataFrame] = {}
            expected_fit_events = []
            cell_records = []
            selected_rows = []
            plan = route.stage_plan(stage, attempt=attempt)
            for cell in plan:
                horizon, fold = int(cell["horizon"]), cell["fold"]
                labels = pinned.build_close_return_labels(panel, horizons=(horizon,))[horizon]
                labels_by_horizon[horizon] = labels
                prepared = route.prepare_cell_data(features=features, labels=labels, fold=fold, canary=True)
                raw_train, raw_test = route.split_fold(features, labels, fold)
                raw_train, _ = route.canary_subset(raw_train, raw_test)
                raw_keys_sha = route.canonical_sha256({
                    "keys": [[pd.Timestamp(date).strftime("%Y-%m-%d"), str(code)]
                             for date, code in raw_train[list(route.KEYS)].itertuples(index=False, name=None)]
                })
                identity = pinned.checkpoint_identity(
                    stage=stage, fold=fold, arm=cell["arm"], horizon=horizon, attempt=attempt,
                    contract_sha256=contract_sha, code_shas=code_shas,
                    binding=json.loads(binding_path.read_text()),
                    feature_names=prepared["feature_columns"], training_key_sha256=raw_keys_sha,
                )
                cell_key = f"{stage}_{cell['arm']}_H{horizon}_{fold.fold_id}_{attempt}"
                checkpoint_path = (tmp_path / "checkpoints" / f"{cell_key}.joblib").resolve()
                model = FixedScoreModel()
                predictions = prepared["test"].loc[:, list(route.KEYS)].reset_index(drop=True)
                predictions["prediction"] = model.predict(prepared["xt"])
                route.save_checkpoint(checkpoint_path, {"model": model, "predictions": predictions,
                                                        "feature_columns": prepared["feature_columns"]}, identity)
                metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
                cell_records.append({"cell": cell_key, "status": "COMPLETED", "fit_called": True,
                                     "checkpoint_path": str(checkpoint_path)})
                top = route.stable_top10(predictions)
                top["model_arm"], top["model_horizon"], top["fold"] = cell["arm"], horizon, fold.fold_id
                selected_rows.append(top[["date", "code", "rank", "prediction", "model_arm",
                                          "model_horizon", "fold"]])
                expected_fit_events.extend([
                    {"experiment_id": route.EXPERIMENT_ID, "execution_id": cell_key,
                     "route_id": "a-share-model-selection-pv-environment-r1-20261019",
                     "registration_mode": "LIVE_EXECUTION", "code_sha256": route.canonical_sha256(code_shas),
                     "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
                     "event_sha256": ("a" if horizon == 1 else "b" if horizon == 2 else "c") * 64,
                     "event_type": "RUN_STARTED", "stage": f"{stage}_RUN_STARTED",
                     "deterministic_metrics": {"training_rows": len(prepared["train"])}},
                    {"experiment_id": route.EXPERIMENT_ID, "execution_id": cell_key,
                     "route_id": "a-share-model-selection-pv-environment-r1-20261019",
                     "registration_mode": "LIVE_EXECUTION", "code_sha256": route.canonical_sha256(code_shas),
                     "contract_sha256": contract_sha, "input_binding_sha256": binding_sha,
                     "event_sha256": ("d" if horizon == 1 else "e" if horizon == 2 else "f") * 64,
                     "event_type": "TERMINAL", "stage": f"{stage}_TERMINAL",
                     "decision": "COMPLETED",
                     "deterministic_metrics": {
                         "execution_kind": "REAL_FIT", "fit_call_entered": True,
                         "planned_real_fit_count": 1, "real_fit_budget_debit": 1,
                         "real_fit_returned_count": 1,
                     },
                     "input_identity": {"artifact_sha256": {
                         str(checkpoint_path.resolve()): route.sha256_file(checkpoint_path),
                         str(metadata_path.resolve()): route.sha256_file(metadata_path),
                     }}},
                ])

            selected_keys_path = tmp_path / "selected_keys.csv"
            selected_frame = pd.concat(selected_rows, ignore_index=True)
            selected_frame.to_csv(selected_keys_path, index=False)
            selected_row = (selected_frame["date"].eq(test_dates[0]) &
                            selected_frame["code"].eq(codes[0]) &
                            selected_frame["model_arm"].eq("M0") &
                            selected_frame["model_horizon"].eq(1) &
                            selected_frame["fold"].eq("2024H1"))
            self.assertEqual(int(selected_row.sum()), 1)
            self.assertEqual(float(selected_frame.loc[selected_row, "prediction"].iloc[0]),
                             high_precision_score)
            default_read = pd.read_csv(selected_keys_path)
            default_read["date"] = pd.to_datetime(default_read["date"]).dt.normalize()
            round_trip_read = pd.read_csv(selected_keys_path, float_precision="round_trip")
            round_trip_read["date"] = pd.to_datetime(round_trip_read["date"]).dt.normalize()
            selected_identity = (default_read["date"].eq(test_dates[0]) &
                                 default_read["code"].eq(codes[0]) &
                                 default_read["model_arm"].eq("M0") &
                                 default_read["model_horizon"].eq(1) &
                                 default_read["fold"].eq("2024H1"))
            selected_identity_round_trip = (round_trip_read["date"].eq(test_dates[0]) &
                                            round_trip_read["code"].eq(codes[0]) &
                                            round_trip_read["model_arm"].eq("M0") &
                                            round_trip_read["model_horizon"].eq(1) &
                                            round_trip_read["fold"].eq("2024H1"))
            self.assertFalse(np.array_equal(
                default_read.loc[selected_identity, "prediction"].to_numpy(dtype=np.float64),
                np.array([high_precision_score], dtype=np.float64),
            ))
            self.assertTrue(np.array_equal(
                round_trip_read.loc[selected_identity_round_trip, "prediction"].to_numpy(dtype=np.float64),
                np.array([high_precision_score], dtype=np.float64),
            ))
            artifacts = {"selected_keys": {"path": str(selected_keys_path),
                                            "sha256": route.sha256_file(selected_keys_path)}}
            output_manifest_path = tmp_path / "output_manifest.json"
            route.atomic_write_json(output_manifest_path, {"stage": stage, "attempt": attempt,
                                                            "contract_sha256": contract_sha,
                                                            "input_binding_sha256": binding_sha,
                                                            "artifacts": artifacts})
            manifest_path = tmp_path / "stage_manifest.json"
            route.atomic_write_json(manifest_path, {"stage": stage, "attempt": attempt, "fit_calls": 3,
                                                     "contract_sha256": contract_sha,
                                                     "input_binding_sha256": binding_sha,
                                                     "code_sha256": code_shas,
                                                     "output_manifest_path": str(output_manifest_path),
                                                     "output_artifacts": artifacts,
                                                     "cell_results": cell_records})
            registry_path = tmp_path / "registry.jsonl"
            registry_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                                               for event in expected_fit_events), encoding="utf-8")
            guard = SyntheticNoFitGuard(registry_path)

            @contextlib.contextmanager
            def pinned_loader(**_: object):
                yield pinned

            with patch.object(route, "make_official_fit_guard", return_value=guard), \
                    patch.object(route, "_pinned_route_from_archives", new=pinned_loader):
                receipt = route.run_checkpoint_replay_audit(
                    stage=stage, attempt=attempt, manifest_path=manifest_path,
                    expected_manifest_sha256=route.sha256_file(manifest_path),
                    source_archive_path=source_archive,
                    expected_source_archive_sha256=route.sha256_file(source_archive),
                    dependency_archive_path=dependency_archive,
                    expected_dependency_archive_sha256=route.sha256_file(dependency_archive),
                    contract_path=contract_path, binding_path=binding_path,
                    activation_path=activation_path, receipt_path=tmp_path / "replay_receipt.json",
                    execution_id="SYNTHETIC_M0_NO_FIT_AUDIT", registry_path=registry_path,
                )

            self.assertEqual(receipt["status"], "VERIFIED_RECONSTRUCTED_REPLAY")
            self.assertEqual(receipt["fit_calls"], 0)
            self.assertEqual(len(receipt["cells"]), 3)
            for cell in receipt["cells"]:
                self.assertEqual(cell["reconstructed_original_raw_training_rows"], 12000)
                self.assertEqual(cell["reconstructed_training_rows"], 11999)
                self.assertEqual(cell["reconstructed_original_checkpoint_training_key_identity"]["row_count"], 12000)
                self.assertEqual(cell["reconstructed_finite_fit_training_key_identity"]["row_count"], 11999)
            self.assertEqual(guard.terminal, {"succeeded": True, "error_type": None})

    def test_historical_audit_reports_terminal_failure_without_suppressing_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text('{"stage":"M0_CANARY","attempt":"attempt_04","fit_calls":3}\n',
                                     encoding="utf-8")
            source_archive = tmp_path / "source.zip"
            dependency_archive = tmp_path / "dependency.zip"
            for archive_path in (source_archive, dependency_archive):
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("placeholder.txt", "synthetic")
            registry_path = tmp_path / "registry.jsonl"
            registry_path.write_text("", encoding="utf-8")
            guard = SyntheticNoFitGuard(registry_path, terminal_error=RuntimeError("terminal append failed"))
            with patch.object(route, "make_official_fit_guard", return_value=guard):
                with self.assertRaisesRegex(route.ExecutionBlocked, "original stage manifest SHA mismatch") as caught:
                    route.run_checkpoint_replay_audit(
                        stage="M0_CANARY", attempt="attempt_04", manifest_path=manifest_path,
                        expected_manifest_sha256="0" * 64,
                        source_archive_path=source_archive,
                        expected_source_archive_sha256=route.sha256_file(source_archive),
                        dependency_archive_path=dependency_archive,
                        expected_dependency_archive_sha256=route.sha256_file(dependency_archive),
                        contract_path=self.contract_path, binding_path=self.binding_path,
                        activation_path=BATCH_ROOT / "REVIEW_ACTIVATION_CANARY_R4.json",
                        receipt_path=tmp_path / "receipt.json", execution_id="SYNTHETIC_TERMINAL_FAIL",
                        registry_path=registry_path,
                    )
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertEqual(str(caught.exception.__cause__), "terminal append failed")
            receipt = route.load_json(tmp_path / "receipt.json")
            self.assertEqual(receipt["status"], "CANNOT_CONFIRM_NO_FIT_TERMINAL")
            self.assertEqual(receipt["error"], "original stage manifest SHA mismatch")
            self.assertEqual(receipt["terminal_error"], "terminal append failed")
            self.assertNotIn("original_manifest_sha256", receipt)
            events = [json.loads(line) for line in registry_path.read_text().splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["RUN_STARTED"])

    def test_receipt_creation_is_exclusive_and_claim_failure_does_not_update_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            existing_receipt = tmp_path / "existing.json"
            existing_receipt.write_bytes(b"preserve this receipt\n")
            with self.assertRaisesRegex(route.ExecutionBlocked, "receipt path already exists"):
                route._create_exclusive_json(existing_receipt, {"status": "replacement"})
            self.assertEqual(existing_receipt.read_bytes(), b"preserve this receipt\n")

            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            source_archive, dependency_archive = tmp_path / "source.zip", tmp_path / "dependency.zip"
            for archive_path in (source_archive, dependency_archive):
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("placeholder.txt", "synthetic")
            registry_path = tmp_path / "registry.jsonl"
            registry_path.write_text("", encoding="utf-8")
            duplicate = SyntheticNoFitGuard(registry_path, claim_error=route.ExecutionBlocked("duplicate execution"))
            owned_receipt = tmp_path / "owned.json"
            with patch.object(route, "make_official_fit_guard", return_value=duplicate):
                with self.assertRaisesRegex(route.ExecutionBlocked, "duplicate execution"):
                    route.run_checkpoint_replay_audit(
                        stage="M0_CANARY", attempt="attempt_04", manifest_path=manifest_path,
                        expected_manifest_sha256="0" * 64,
                        source_archive_path=source_archive,
                        expected_source_archive_sha256=route.sha256_file(source_archive),
                        dependency_archive_path=dependency_archive,
                        expected_dependency_archive_sha256=route.sha256_file(dependency_archive),
                        contract_path=self.contract_path, binding_path=self.binding_path,
                        activation_path=BATCH_ROOT / "REVIEW_ACTIVATION_CANARY_R4.json",
                        receipt_path=owned_receipt, execution_id="SYNTHETIC_DUPLICATE",
                        registry_path=registry_path,
                    )
            unchanged_receipt = route.load_json(owned_receipt)
            self.assertEqual(unchanged_receipt["status"], "NO_FIT_INTENT_PREPARED")
            self.assertEqual(unchanged_receipt["preflight_status"], "NOT_STARTED")
            self.assertNotIn("error", unchanged_receipt)
            self.assertEqual(duplicate.terminal, None)
            self.assertEqual(registry_path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
