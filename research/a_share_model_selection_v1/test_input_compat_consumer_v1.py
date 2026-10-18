"""Synthetic consumer checks; all dates, codes, prices, and index rows are invented."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROUTE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUTE_ROOT))
import model_selection as route  # noqa: E402


class InputCompatibilityConsumerTests(unittest.TestCase):
    @staticmethod
    def _calendar(periods: int = 23) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=periods)
        return pd.DataFrame({"date": dates, "calendar_index": np.arange(periods, dtype=np.int64)})

    @staticmethod
    def _panel(rows: list[tuple[int, str, float, float, float, float]]) -> pd.DataFrame:
        raw = pd.DataFrame(rows, columns=("trade_date", "ts_code", "open", "high", "low", "close"))
        raw["vol"] = 100.0
        raw["amount"] = 1000.0
        return route.normalize_price_panel(raw, {
            "date": "trade_date", "code": "ts_code", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "vol", "amount": "amount",
        })

    @staticmethod
    def _finite_feature_frames(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        formula = panel[["date", "code"]].copy()
        for feature in route.CORE_FEATURES:
            formula[feature] = 50.0
        m0 = formula[["date", "code", *route.CORE_FEATURES]].copy()
        m1 = m0.copy()
        m1["stock_rs5"] = 0.1
        m1["index_return_1"] = 0.02
        m1["index_return_5"] = 0.05
        m1["index_ma5_position"] = 0.01
        m1["market_advancer_share"] = 0.6
        m1["market_amount_activity"] = 1.2
        m2 = m1.copy()
        m2["AUDITM"] = 12.0
        m2["month_01"] = 1.0
        m3 = m2.copy()
        m3["industry_advancer_share"] = 0.55
        m3["industry_mean_ret5"] = 0.04
        m3["stock_relative_industry_ret5"] = 0.03
        m3["industry_mean_ret20"] = 0.08
        m3["industry_amount_activity"] = 1.1
        m3["industry_cat_synthetic"] = 1.0
        m4 = m3.copy()
        m4["amv_proxy_sma10"] = 1.0
        return formula, {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": m4}

    @staticmethod
    def _synthetic_index_gate_files(root: Path, *, use_r2_policy: bool) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
        binding_path = root / "INPUT_BINDING_R2.json"
        contract_path = root / ("CONTRACT_R2.json" if use_r2_policy else "CONTRACT.json")
        source_path = root / "synthetic_index.csv"
        receipt_path = root / "synthetic_recovery_receipt.json"
        source = {
            "status": "BOUND",
            "source_type": "FUTU_RECOVERED_REPLACEMENT",
            "path": str(source_path),
            "receipt_path": str(receipt_path),
            "index_code": "SYNTH.IDX",
            "column_map": {"date": "session", "code": "ticker", "close": "closing"},
            "source_columns": ["session", "ticker", "closing"],
            "sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "source_identity_status": "FUTU_RECOVERED_REPLACEMENT; ORIGINAL_TDX_FILE_IDENTITY_CANNOT_CONFIRM",
            "official_exchange_calendar_parity": "CANNOT_CONFIRM",
        }
        binding = {
            "status": "INPUT_BINDING_READY_PARTIAL",
            "sources": {"index_series": source},
            "readiness_by_arm": {
                arm: {"status": "READY_AFTER_INDEX_BINDING"} for arm in route.ARMS
            },
        }
        binding_path.write_text(json.dumps(binding, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        binding_sha = route.sha256_file(binding_path)
        legacy_policy = "CANNOT_CONFIRM: original TDX index history is unavailable"
        contract: dict[str, object] = {
            "status": "FROZEN_AWAITING_PRE_RUN_REVIEW",
            "governance": {"writer_identity": "synthetic_writer"},
            "inputs": {"official_index_history": legacy_policy},
        }
        if use_r2_policy:
            contract["inputs"] = {
                "official_index_history": {
                    "original_tdx_identity": "CANNOT_CONFIRM",
                    "original_tdx_evidence": legacy_policy,
                    "official_exchange_calendar_parity": "CANNOT_CONFIRM",
                    "replacement_policy": {
                        "source_type": source["source_type"],
                        "binding_path": binding_path.name,
                        "binding_sha256": binding_sha,
                        "index_code": source["index_code"],
                        "column_map": source["column_map"],
                        "source_columns": source["source_columns"],
                        "source_path": source_path.name,
                        "source_sha256": source["sha256"],
                        "receipt_path": receipt_path.name,
                        "receipt_sha256": source["receipt_sha256"],
                    },
                }
            }
        contract_path.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return contract_path, binding_path, contract, binding

    @staticmethod
    def _synthetic_activation(
        root: Path,
        contract_path: Path,
        binding_path: Path,
        *,
        stage: str,
        arms: tuple[str, ...],
        status: str,
    ) -> dict[str, object]:
        receipt_path = root / "synthetic_reviewer_receipt.json"
        receipt_path.write_text('{"status":"SYNTHETIC_REVIEW_ONLY"}\n', encoding="utf-8")
        return {
            "status": status,
            "reviewer_identity": "synthetic_independent_reviewer",
            "allowed_arms": list(route.ARMS if "M4" in arms else ("M0",)),
            "allowed_stages": [stage],
            "contract_sha256": route.sha256_file(contract_path),
            "input_binding_sha256": route.sha256_file(binding_path),
            "code_sha256": route.route_code_sha256(),
            "reviewer_receipt_path": str(receipt_path),
            "reviewer_receipt_sha256": route.sha256_file(receipt_path),
            "allow_real_data_read": True,
            "allow_fit": True,
        }

    def test_indexed_stage_rejects_legacy_contract_but_m0_keeps_its_no_index_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract_path, binding_path, contract, binding = self._synthetic_index_gate_files(
                root, use_r2_policy=False,
            )
            activation = self._synthetic_activation(
                root, contract_path, binding_path, stage="M4_CANARY", arms=("M4",),
                status="APPROVED_FOR_INDEXED_ARMS",
            )
            with self.assertRaisesRegex(route.ExecutionBlocked, "explicit replacement index source policy"):
                route.validate_review_activation(
                    contract, binding, activation,
                    contract_sha256=route.sha256_file(contract_path),
                    binding_sha256=route.sha256_file(binding_path),
                    code_sha256=route.route_code_sha256(), stage="M4_CANARY", arms=("M4",),
                    contract_path=contract_path, binding_path=binding_path,
                )

            m0_activation = self._synthetic_activation(
                root, contract_path, binding_path, stage="M0_CANARY", arms=("M0",),
                status="APPROVED_FOR_M0_ONLY",
            )
            route.validate_review_activation(
                contract, binding, m0_activation,
                contract_sha256=route.sha256_file(contract_path),
                binding_sha256=route.sha256_file(binding_path),
                code_sha256=route.route_code_sha256(), stage="M0_CANARY", arms=("M0",),
                contract_path=contract_path, binding_path=binding_path,
            )

    def test_indexed_stage_accepts_exact_r2_policy_and_rejects_binding_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contract_path, binding_path, contract, binding = self._synthetic_index_gate_files(
                root, use_r2_policy=True,
            )
            activation = self._synthetic_activation(
                root, contract_path, binding_path, stage="M4_CANARY", arms=("M4",),
                status="APPROVED_FOR_INDEXED_ARMS",
            )
            route.validate_review_activation(
                contract, binding, activation,
                contract_sha256=route.sha256_file(contract_path),
                binding_sha256=route.sha256_file(binding_path),
                code_sha256=route.route_code_sha256(), stage="M4_CANARY", arms=("M4",),
                contract_path=contract_path, binding_path=binding_path,
            )

            binding["synthetic_binding_drift"] = True
            binding_path.write_text(json.dumps(binding, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            drifted_activation = self._synthetic_activation(
                root, contract_path, binding_path, stage="M4_CANARY", arms=("M4",),
                status="APPROVED_FOR_INDEXED_ARMS",
            )
            with self.assertRaisesRegex(route.ExecutionBlocked, "exact R2 input binding SHA"):
                route.validate_review_activation(
                    contract, binding, drifted_activation,
                    contract_sha256=route.sha256_file(contract_path),
                    binding_sha256=route.sha256_file(binding_path),
                    code_sha256=route.route_code_sha256(), stage="M4_CANARY", arms=("M4",),
                    contract_path=contract_path, binding_path=binding_path,
                )

    def test_calendar_parses_int64_yyyymmdd_and_rejects_invalid_order_or_fraction(self) -> None:
        raw_calendar = pd.DataFrame({
            "trade_date": pd.Series([20240102, 20240103, 20240104], dtype="int64"),
            "calendar_index": pd.Series([0, 1, 2], dtype="int64"),
        })
        panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"])})

        normalized = route.normalize_calendar(raw_calendar, panel)

        self.assertEqual(normalized["date"].dt.strftime("%Y-%m-%d").tolist(),
                         ["2024-01-02", "2024-01-03", "2024-01-04"])
        self.assertEqual(normalized["calendar_index"].tolist(), [0, 1, 2])
        reversed_dates = raw_calendar.copy()
        reversed_dates["trade_date"] = pd.Series([20240103, 20240102, 20240104], dtype="int64")
        with self.assertRaisesRegex(route.ContractError, "strictly increase"):
            route.normalize_calendar(reversed_dates, panel)
        fractional_index = raw_calendar.copy()
        fractional_index["calendar_index"] = [0.0, 1.5, 2.0]
        with self.assertRaisesRegex(route.ContractError, "ordinal must be an integer"):
            route.normalize_calendar(fractional_index, panel)

    def test_session_lag_keeps_out_of_calendar_rows_unknown_and_preserves_left_order(self) -> None:
        calendar = self._calendar(3)
        dates = calendar["date"].tolist()
        panel = pd.DataFrame([
            (dates[2], "SYN_A", 12.0),
            (pd.Timestamp("2024-01-01"), "SYN_A", 9.0),
            (dates[0], "SYN_B", 20.0),
            (dates[1], "SYN_A", 11.0),
            (pd.Timestamp("2024-01-08"), "SYN_A", 13.0),
            (dates[2], "SYN_B", 21.0),
            (dates[0], "SYN_A", 10.0),
        ], columns=("date", "code", "close"), index=[9, 4, 11, 2, 7, 1, 8])
        before = panel.copy(deep=True)

        lagged = route.session_lag(panel, calendar, "close", 1)

        self.assertEqual(len(lagged), len(panel))
        self.assertEqual(lagged.index.tolist(), panel.index.tolist())
        self.assertEqual(lagged.iloc[0], 11.0)
        self.assertTrue(lagged.iloc[[1, 2, 4, 5, 6]].isna().all())
        self.assertEqual(lagged.iloc[3], 10.0)
        self.assertTrue(panel.equals(before))

    def test_session_lag_rejects_duplicate_source_keys_at_calendar_boundary(self) -> None:
        calendar = self._calendar(3)
        boundary = calendar["date"].iloc[-1]
        panel = pd.DataFrame({
            "date": [calendar["date"].iloc[0], boundary, boundary],
            "code": ["SYN_DUP", "SYN_DUP", "SYN_DUP"],
            "close": [9.0, 10.0, 11.0],
        })

        with self.assertRaisesRegex(route.ContractError, "duplicate date/code"):
            route.session_lag(panel, calendar, "close", 1)

    def test_selected_and_paired_summaries_report_whole_date_exclusions(self) -> None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        selected_rows = []
        arm_picks: dict[str, list[dict[str, object]]] = {"M0": [], "M4": []}
        label_rows = []
        for arm in ("M0", "M4"):
            for date_index, date in enumerate(dates):
                for rank in range(1, 11):
                    code = f"SYN_{arm}_{date_index}_{rank:02d}"
                    missing_m0 = arm == "M0" and date_index == 1 and rank == 1
                    selected_row: dict[str, object] = {
                        "model_arm": arm, "model_horizon": 1, "fold": "2024H1", "canary": False,
                        "date": date, "code": code,
                        "actual_close_return_h1": np.nan if missing_m0 else (0.01 if arm == "M0" else 0.02),
                        "actual_close_return_h2": 0.03, "actual_close_return_h5": 0.05,
                    }
                    selected_rows.append(selected_row)
                    arm_picks[arm].append({"date": date, "code": code, "rank": rank, "prediction": float(11-rank)})
                    if not missing_m0:
                        label_rows.append({"date": date, "code": code, "label_return": 0.01 if arm == "M0" else 0.02})

        selected = pd.DataFrame(selected_rows)
        summaries, _ = route.summarize_selected_returns(selected)
        m0_summary = next(row for row in summaries
                          if row["model_arm"] == "M0" and row["outcome_horizon"] == 1)
        m4_summary = next(row for row in summaries
                          if row["model_arm"] == "M4" and row["outcome_horizon"] == 1)
        self.assertEqual((m0_summary["known_label_slots"], m0_summary["unknown_label_slots"]), (19, 1))
        self.assertEqual((m0_summary["excluded_dates"], m0_summary["excluded_slots"]), (1, 10))
        self.assertEqual((m4_summary["known_label_slots"], m4_summary["unknown_label_slots"]), (20, 0))
        self.assertEqual((m4_summary["excluded_dates"], m4_summary["excluded_slots"]), (0, 0))

        paired = route.paired_common_dates(
            pd.DataFrame(arm_picks["M0"]), pd.DataFrame(arm_picks["M4"]), pd.DataFrame(label_rows)
        )
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired.attrs["per_arm_counts"]["M0"], {
            "selected_slots": 20, "known_label_slots": 19, "unknown_label_slots": 1,
            "date_count": 2, "complete_date_count": 1, "excluded_dates": 1, "excluded_slots": 10,
        })
        self.assertEqual(paired.attrs["per_arm_counts"]["M4"], {
            "selected_slots": 20, "known_label_slots": 20, "unknown_label_slots": 0,
            "date_count": 2, "complete_date_count": 2, "excluded_dates": 0, "excluded_slots": 0,
        })
        self.assertEqual(paired.attrs["paired_exclusion_counts"], {
            "candidate_dates": 2, "paired_complete_dates": 1, "paired_excluded_dates": 1,
            "paired_excluded_slots_by_arm": {"M0": 10, "M4": 10}, "paired_excluded_slots": 20,
        })

        folds = {fold.fold_id: paired.copy() for fold in route.fixed_folds()}
        assessment = route.assess_horizon(folds)
        self.assertEqual(assessment["per_fold"]["2024H1"]["paired_exclusion_counts"],
                         paired.attrs["paired_exclusion_counts"])
        self.assertEqual(assessment["per_fold"]["2024H1"][
            "per_arm_label_and_exclusion_counts"]["M0"]["unknown_label_slots"], 1)

    def test_bound_index_uses_mapped_string_columns_and_exact_code(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=5)
        index_rows = pd.DataFrame({
            "session": dates.strftime("%Y-%m-%d"),
            "ticker": ["SYNTH.IDX"] * 5,
            "opening": [100.0, 101.0, 102.0, 103.0, 104.0],
            "closing": [100.0, 101.5, 102.5, 103.5, 104.5],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "synthetic_index.csv"
            index_rows.to_csv(csv_path, index=False)
            csv_sha = route.sha256_file(csv_path)
            receipt_path = root / "synthetic_receipt.json"
            receipt_path.write_text(json.dumps({
                "normalized_sha256": csv_sha,
                "index_code": "SYNTH.IDX",
                "normalized_path": csv_path.name,
                "normalized_row_count": 5,
            }), encoding="utf-8")
            source = {
                "status": "BOUND",
                "path": str(csv_path),
                "sha256": csv_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": route.sha256_file(receipt_path),
                "source_columns": ["session", "ticker", "closing"],
                "csv_header": ["session", "ticker", "opening", "closing"],
                "column_map": {"date": "session", "code": "ticker", "close": "closing"},
                "index_code": "SYNTH.IDX",
                "row_count": 5,
                "date_range": [dates[0].date().isoformat(), dates[-1].date().isoformat()],
            }

            loaded = route.load_bound_index_series(source, panel_dates={dates[0], dates[-1]})

            self.assertEqual(loaded.index.tolist(), list(dates))
            self.assertEqual(loaded.tolist(), [100.0, 101.5, 102.5, 103.5, 104.5])
            wrong_code = index_rows.copy()
            wrong_code.loc[2, "ticker"] = "OTHER.IDX"
            wrong_code.to_csv(csv_path, index=False)
            bad_sha = route.sha256_file(csv_path)
            receipt_path.write_text(json.dumps({
                "normalized_sha256": bad_sha,
                "index_code": "SYNTH.IDX",
                "normalized_path": csv_path.name,
                "normalized_row_count": 5,
            }), encoding="utf-8")
            source["sha256"] = bad_sha
            source["receipt_sha256"] = route.sha256_file(receipt_path)
            with self.assertRaisesRegex(route.ContractError, "outside its exact bound identity"):
                route.load_bound_index_series(source, panel_dates={dates[0]})

    def test_five_synthetic_rows_are_quarantined_without_dropping_or_rewriting_raw_values(self) -> None:
        panel = self._panel([
            (20240102, "SYN_A", -1.0, 0.0, -2.0, -0.5),
            (20240103, "SYN_B", -2.0, -1.0, -4.0, -3.0),
            (20240104, "SYN_C", 10.0, 11.0, 0.0, 10.0),
            (20240105, "SYN_D", 10.0, 12.0, 0.0, 0.0),
            (20240108, "SYN_E", 0.0, 1.0, -1.0, 1.0),
        ])
        raw_close = panel["close"].copy(deep=True)
        feature_panel, quarantine = route.derive_price_feature_view(panel)

        self.assertEqual(quarantine["source_row_count"], 5)
        self.assertEqual(quarantine["quarantined_row_count"], 5)
        self.assertEqual(quarantine["accepted_row_count"] + quarantine["quarantined_row_count"], 5)
        self.assertEqual(quarantine["duplicate_date_code_key_count"], 0)
        self.assertTrue(quarantine["row_count_conserved"])
        self.assertEqual(len(feature_panel), 5)
        self.assertTrue(feature_panel[list(route.OHLC_COLUMNS)].isna().all().all())
        self.assertTrue(panel["close"].equals(raw_close))
        self.assertEqual(feature_panel[["date", "code"]].to_dict("records"),
                         panel[["date", "code"]].to_dict("records"))
        route._validate_quarantine_binding(quarantine, {
            "expected_source_row_count": 5,
            "expected_quarantined_row_count": 5,
            "expected_quarantine_keys": [
                {"date": "2024-01-02", "code": "SYN_A",
                 "reason_codes": ["NON_POSITIVE_OPEN", "NON_POSITIVE_HIGH", "NON_POSITIVE_LOW", "NON_POSITIVE_CLOSE"]},
                {"date": "2024-01-03", "code": "SYN_B",
                 "reason_codes": ["NON_POSITIVE_OPEN", "NON_POSITIVE_HIGH", "NON_POSITIVE_LOW", "NON_POSITIVE_CLOSE"]},
                {"date": "2024-01-04", "code": "SYN_C", "reason_codes": ["NON_POSITIVE_LOW"]},
                {"date": "2024-01-05", "code": "SYN_D", "reason_codes": ["NON_POSITIVE_LOW", "NON_POSITIVE_CLOSE"]},
                {"date": "2024-01-08", "code": "SYN_E", "reason_codes": ["NON_POSITIVE_OPEN", "NON_POSITIVE_LOW"]},
            ],
        })

    def test_labels_continue_to_use_raw_close_and_keep_unknown_target_behavior(self) -> None:
        panel = self._panel([
            (20240102, "SYN_LABEL", 10.0, 11.0, 9.0, 10.0),
            (20240103, "SYN_LABEL", 0.0, 12.0, 0.0, 11.0),
            (20240104, "SYN_LABEL", 11.0, 13.0, 10.0, 12.0),
            (20240105, "SYN_LABEL", 12.0, 14.0, 11.0, 13.0),
            (20240108, "SYN_LABEL", 13.0, 14.0, 0.0, 0.0),
        ])
        raw_before = panel.copy(deep=True)
        feature_panel, _ = route.derive_price_feature_view(panel)
        calendar = self._calendar(5)

        labels = route.build_close_return_labels(panel, horizons=(1,), calendar=calendar)[1]
        by_date = labels.set_index("date")["label_return"]

        self.assertAlmostEqual(by_date.loc[pd.Timestamp("2024-01-02")], 0.1)
        self.assertAlmostEqual(by_date.loc[pd.Timestamp("2024-01-03")], 12.0 / 11.0 - 1.0)
        self.assertTrue(np.isnan(by_date.loc[pd.Timestamp("2024-01-05")]))
        self.assertTrue(panel.equals(raw_before))
        self.assertTrue(feature_panel.loc[feature_panel["date"].eq(pd.Timestamp("2024-01-03")), list(route.OHLC_COLUMNS)].isna().to_numpy().all())
        self.assertTrue(feature_panel.loc[feature_panel["date"].eq(pd.Timestamp("2024-01-08")), list(route.OHLC_COLUMNS)].isna().to_numpy().all())

    def test_five_row_taint_fixture_enforces_core_suffix_and_t_plus_one_market_boundary(self) -> None:
        dates = self._calendar()
        by_session = dates["date"].tolist()
        panel = self._panel([
            (20240102, "SYN_A", 10.0, 11.0, 9.0, 10.0),
            (20240103, "SYN_A", 0.0, 11.0, 0.0, 10.0),
            (20240104, "SYN_A", 11.0, 12.0, 10.0, 11.5),
            (20240104, "SYN_B", 11.0, 12.0, 10.0, 11.5),
            (20240105, "SYN_B", 11.5, 12.5, 10.5, 12.0),
        ])
        calendar = dates.rename(columns={"date": "date"})
        formula, frames = self._finite_feature_frames(panel)
        _, quarantine = route.derive_price_feature_view(panel)
        industries = pd.DataFrame({"code": ["SYN_A", "SYN_B"], "industry_code": ["SYNTH_GROUP", "SYNTH_GROUP"]})

        masked_formula, masked_frames, sidecar = route.apply_ohlc_taint_masks(
            formula, frames, calendar, quarantine, industry_classification=industries,
        )

        core = route.CORE_FEATURES[0]
        self.assertTrue(np.isfinite(masked_formula.loc[(masked_formula["code"] == "SYN_A") &
                                                       (masked_formula["date"] == by_session[0]), core].iloc[0]))
        self.assertTrue(masked_formula.loc[(masked_formula["code"] == "SYN_A") &
                                           (masked_formula["date"] >= by_session[1]), list(route.CORE_FEATURES)].isna().all().all())
        self.assertTrue(np.isfinite(masked_formula.loc[(masked_formula["code"] == "SYN_B") &
                                                       (masked_formula["date"] == by_session[2]), core].iloc[0]))
        for arm, frame in masked_frames.items():
            pre_bad = frame.loc[(frame["code"] == "SYN_A") & (frame["date"] == by_session[0]), list(route.CORE_FEATURES)]
            suffix = frame.loc[(frame["code"] == "SYN_A") & (frame["date"] >= by_session[1]), list(route.CORE_FEATURES)]
            self.assertTrue(np.isfinite(pre_bad.to_numpy(dtype=np.float64)).all(), arm)
            self.assertTrue(suffix.isna().all().all(), arm)
        m1 = masked_frames["M1"].set_index(["date", "code"])
        self.assertTrue(np.isnan(m1.loc[(by_session[2], "SYN_B"), "market_advancer_share"]))
        self.assertTrue(np.isfinite(m1.loc[(by_session[3], "SYN_B"), "market_advancer_share"]))
        self.assertTrue(np.isfinite(m1.loc[(by_session[2], "SYN_B"), "index_return_1"]))
        self.assertTrue(np.isfinite(m1.loc[(by_session[2], "SYN_B"), "market_amount_activity"]))
        m3 = masked_frames["M3"].set_index(["date", "code"])
        self.assertTrue(np.isnan(m3.loc[(by_session[2], "SYN_B"), "industry_advancer_share"]))
        self.assertTrue(np.isfinite(m3.loc[(by_session[3], "SYN_B"), "industry_advancer_share"]))
        self.assertTrue(np.isnan(m3.loc[(by_session[3], "SYN_B"), "industry_mean_ret5"]))
        self.assertTrue(np.isfinite(m3.loc[(by_session[3], "SYN_B"), "industry_amount_activity"]))
        self.assertTrue(np.isfinite(m3.loc[(by_session[3], "SYN_B"), "industry_cat_synthetic"]))
        self.assertTrue(any("RECURSIVE_FORMULA_TAINT_SUFFIX" in item["reason_codes"]
                            for item in sidecar["derived_tainted_keys_by_arm"]["M3"]))
        m4 = masked_frames["M4"].set_index(["date", "code"])
        self.assertTrue(np.isnan(m4.loc[(by_session[3], "SYN_B"), "industry_mean_ret20"]))
        self.assertTrue(np.isfinite(m4.loc[(by_session[3], "SYN_B"), "industry_amount_activity"]))
        self.assertTrue(sidecar["source_row_accounting"]["row_count_conserved"])
        self.assertEqual(sidecar["coverage_loss_by_arm"]["M3"]["tainted_key_count"], 4)

    def test_five_row_taint_fixture_enforces_t_plus_five_and_t_plus_twenty_endpoints(self) -> None:
        calendar = self._calendar(23)
        dates = calendar["date"].tolist()
        synthetic_rows = [
            (dates[1].strftime("%Y%m%d"), "SYN_A", 0.0, 11.0, 0.0, 10.0),
            (dates[6].strftime("%Y%m%d"), "SYN_A", 10.0, 12.0, 9.0, 11.0),
            (dates[7].strftime("%Y%m%d"), "SYN_A", 10.0, 12.0, 9.0, 11.0),
            (dates[21].strftime("%Y%m%d"), "SYN_B", 10.0, 12.0, 9.0, 11.0),
            (dates[22].strftime("%Y%m%d"), "SYN_B", 10.0, 12.0, 9.0, 11.0),
        ]
        panel = self._panel(synthetic_rows)
        formula, frames = self._finite_feature_frames(panel)
        _, quarantine = route.derive_price_feature_view(panel)
        industries = pd.DataFrame({"code": ["SYN_A", "SYN_B"], "industry_code": ["SYNTH_GROUP", "SYNTH_GROUP"]})

        _, masked_frames, _ = route.apply_ohlc_taint_masks(
            formula, frames, calendar, quarantine, industry_classification=industries,
        )

        m1 = masked_frames["M1"].set_index(["date", "code"])
        for session, code in ((6, "SYN_A"), (7, "SYN_A"), (21, "SYN_B"), (22, "SYN_B")):
            self.assertTrue(np.isfinite(m1.loc[(dates[session], code), "market_advancer_share"]))
        self.assertTrue(np.isnan(m1.loc[(dates[6], "SYN_A"), "stock_rs5"]))
        self.assertTrue(np.isfinite(m1.loc[(dates[7], "SYN_A"), "stock_rs5"]))
        m3 = masked_frames["M3"].set_index(["date", "code"])
        self.assertTrue(np.isnan(m3.loc[(dates[6], "SYN_A"), "industry_mean_ret5"]))
        self.assertTrue(np.isfinite(m3.loc[(dates[7], "SYN_A"), "industry_mean_ret5"]))
        self.assertTrue(np.isnan(m3.loc[(dates[21], "SYN_B"), "industry_mean_ret20"]))
        self.assertTrue(np.isfinite(m3.loc[(dates[22], "SYN_B"), "industry_mean_ret20"]))
        self.assertTrue(np.isfinite(m3.loc[(dates[6], "SYN_A"), "industry_advancer_share"]))

    def test_unknown_industry_is_not_tainted_or_assigned_a_synthetic_group(self) -> None:
        calendar = self._calendar()
        dates = calendar["date"].tolist()
        panel = self._panel([
            (20240102, "SYN_A", 10.0, 11.0, 9.0, 10.0),
            (20240103, "SYN_A", 0.0, 11.0, 0.0, 10.0),
            (20240104, "SYN_B", 10.0, 12.0, 9.0, 11.0),
            (20240104, "SYN_UNKNOWN", 10.0, 12.0, 9.0, 11.0),
            (20240105, "SYN_UNKNOWN", 10.0, 12.0, 9.0, 11.0),
        ])
        formula, frames = self._finite_feature_frames(panel)
        _, quarantine = route.derive_price_feature_view(panel)
        industries = pd.DataFrame({"code": ["SYN_A", "SYN_B"], "industry_code": ["SYNTH_GROUP", "SYNTH_GROUP"]})

        _, masked_frames, sidecar = route.apply_ohlc_taint_masks(
            formula, frames, calendar, quarantine, industry_classification=industries,
        )

        m3 = masked_frames["M3"].set_index(["date", "code"])
        self.assertTrue(np.isnan(m3.loc[(dates[2], "SYN_UNKNOWN"), "market_advancer_share"]))
        self.assertTrue(np.isfinite(m3.loc[(dates[2], "SYN_UNKNOWN"), "industry_mean_ret5"]))
        self.assertTrue(np.isfinite(m3.loc[(dates[3], "SYN_UNKNOWN"), "industry_mean_ret20"]))
        unknown_entries = [entry for entry in sidecar["derived_tainted_keys_by_arm"]["M3"]
                           if entry["code"] == "SYN_UNKNOWN"]
        self.assertEqual(len(unknown_entries), 1)
        self.assertEqual(unknown_entries[0]["masked_feature_fields"], ["market_advancer_share"])
        self.assertEqual(unknown_entries[0]["reason_codes"],
                         ["MARKET_RET1_DEPENDENCY_TAINT_T_TO_T_PLUS_1"])


if __name__ == "__main__":
    unittest.main()
