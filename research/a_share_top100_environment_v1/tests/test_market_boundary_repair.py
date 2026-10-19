import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import market_boundary_repair as repair


class MarketBoundaryRepairTests(unittest.TestCase):
    def test_table_reader_preserves_to_csv_float_round_trip_exactly(self) -> None:
        expected = 112778.70569152481
        payload = pd.DataFrame({"value": [expected]}).to_csv(index=False).encode("utf-8")
        default = pd.read_csv(io.BytesIO(payload)).iloc[0, 0]
        self.assertNotEqual(float(default).hex(), float(expected).hex())

        with tempfile.TemporaryDirectory() as temp_dir:
            table_inputs = {}
            for name in repair.REQUIRED_TABLES:
                table_payload = payload if name == "DAILY_ENVIRONMENT.csv" else b"value\n1\n"
                path = Path(temp_dir) / name
                path.write_bytes(table_payload)
                table_inputs[name] = {"path": str(path), "sha256": hashlib.sha256(table_payload).hexdigest()}

            tables, _ = repair._table_inputs({"table_inputs": table_inputs})

        actual = float(tables["DAILY_ENVIRONMENT.csv"].iloc[0, 0])
        self.assertEqual(actual.hex(), float(expected).hex())

    def test_context_accepts_string_source_snapshot_paths(self) -> None:
        snapshot = "/tmp/top100_environment_pre_boundary_fix.py"
        binding = {
            "table_inputs": {"DAILY_ENVIRONMENT.csv": {"path": "/tmp/daily.csv", "sha256": "daily"}},
            "index_source": {
                "path": "/tmp/index.csv", "receipt_path": "/tmp/index-receipt.json",
                "sha256": "index", "receipt_sha256": "receipt",
            },
            "source_input_binding_path": "/tmp/source-binding.json",
            "source_input_binding_sha256": "source-binding",
            "source_snapshot_paths": [snapshot],
            "shared_dependencies": {
                name: {"sha256": name} for name in (
                    "model_selection", "registry_adapter", "execution_guard_core",
                    "registry_builder", "registry_validator",
                )
            },
        }
        hashes = {
            "contract_sha256": "contract", "input_binding_sha256": "binding",
            "config_sha256": "config", "code_sha256": "repair",
            "main_code_sha256": "main", "main_test_sha256": "main-test",
            "repair_test_sha256": "repair-test",
        }

        context = repair._make_context(
            contract={"affected_date": "2021-05-07"}, binding=binding, hashes=hashes,
            stage="MARKET_BOUNDARY_REPAIR", terminal=False,
        )

        self.assertIn(str(Path(snapshot).resolve()), context["artifact_paths"])

    def test_full_synthetic_repair_changes_only_reviewed_market_cohort(self) -> None:
        dates = pd.to_datetime(["2021-05-06", "2021-05-07", "2022-05-06", "2022-05-09"])
        daily = pd.DataFrame({
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "lag_market": ["BELOW", "UNKNOWN", "BELOW", "BELOW"],
            "valid_pool_size": [100, 100, 100, 100],
            "top100_slots": [100, 100, 100, 100],
            "top100_cutoff_return": [0.02, 0.03, 0.01, 0.015],
            "top100_mean_return": [0.03, 0.04, 0.02, 0.025],
            "top100_median_return": [0.025, 0.035, 0.015, 0.02],
            "top100_p90_return": [0.05, 0.06, 0.04, 0.045],
            "top100_max_return": [0.08, 0.09, 0.07, 0.075],
            "pool_positive_share": [0.55, 0.60, 0.52, 0.54],
            "top100_industry_hhi": [0.10, 0.11, 0.09, 0.095],
            "top100_industry_max_share": [0.20, 0.21, 0.18, 0.19],
        })
        cells = self._cells()
        tables = {
            "COVERAGE_BY_DAY.csv": pd.DataFrame({"date": dates, "source_rows": [100] * 4}),
            "COVERAGE_BY_YEAR.csv": pd.DataFrame({"year": [2021, 2022], "source_rows": [200, 200]}),
            "COVERAGE_BY_INDUSTRY_DAY.csv": pd.DataFrame({"date": dates, "source_rows": [100] * 4}),
            "COVERAGE_BY_INDUSTRY_YEAR.csv": pd.DataFrame({"year": [2021, 2022], "source_rows": [200, 200]}),
            "DAILY_ENVIRONMENT.csv": daily,
            "INDUSTRY_CELLS.csv": cells,
            "MONTH_MARKET_ENVIRONMENT.csv": repair.env._summarize_daily_metrics(daily),
        }
        original_tables = {name: frame.copy(deep=True) for name, frame in tables.items()}
        prior_dates = pd.date_range("2021-04-01", periods=20, freq="D")
        index_dates = prior_dates.append(pd.DatetimeIndex([pd.Timestamp("2021-05-06"), pd.Timestamp("2021-05-07")]))
        index_close = pd.Series(np.r_[np.full(20, 100.0), 95.0, 9999.0], index=index_dates)
        config = {
            "affected_date": "2021-05-07",
            "t_minus_1_date": "2021-05-06",
            "expected_old_label": "UNKNOWN",
            "expected_new_label": "BELOW",
            "reviewed_t_minus_1_index_close": 95.0,
            "reviewed_prior_20_mean": 100.0,
        }

        corrected, boundary = repair._repair_frames(tables, config, index_close)

        self.assertEqual(boundary["classification"], "BELOW")
        target = corrected["DAILY_ENVIRONMENT.csv"].loc[
            corrected["DAILY_ENVIRONMENT.csv"]["date"].eq(pd.Timestamp("2021-05-07"))
        ].iloc[0]
        self.assertEqual(target["lag_market"], "BELOW")
        for name in repair.PRESERVED_TABLES:
            assert_frame_equal(corrected[name], original_tables[name])
        for name in ("industry", "industry_month"):
            before = repair._sort_cells(original_tables["INDUSTRY_CELLS.csv"].loc[
                original_tables["INDUSTRY_CELLS.csv"]["dimension_set"].eq(name)
            ].copy())
            after = repair._sort_cells(corrected["INDUSTRY_CELLS.csv"].loc[
                corrected["INDUSTRY_CELLS.csv"]["dimension_set"].eq(name)
            ].copy())
            assert_frame_equal(before, after)
        self.assertFalse(corrected["INDUSTRY_CELLS.csv"].loc[
            corrected["INDUSTRY_CELLS.csv"]["dimension_set"].isin(repair.CHANGED_CELL_DIMENSIONS), "lag_market"
        ].eq("UNKNOWN").any())
        self.assertFalse(corrected["MONTH_MARKET_ENVIRONMENT.csv"].loc[
            corrected["MONTH_MARKET_ENVIRONMENT.csv"]["dimension"].isin({"lag_market", "month_lag_market"}), "lag_market"
        ].eq("UNKNOWN").any())

    def test_repair_boundary_uses_bound_t_minus_1_without_calendar_t_minus_2(self) -> None:
        prior_dates = pd.date_range("2021-04-01", periods=20, freq="D")
        t_minus_1 = pd.Timestamp("2021-04-23")
        target = pd.Timestamp("2021-04-26")
        index_dates = prior_dates.append(pd.DatetimeIndex([t_minus_1, target]))
        index_close = pd.Series(np.r_[np.full(20, 100.0), 95.0, 9999.0], index=index_dates)
        config = {
            "affected_date": target.date().isoformat(),
            "t_minus_1_date": t_minus_1.date().isoformat(),
            "reviewed_t_minus_1_index_close": 95.0,
            "reviewed_prior_20_mean": 100.0,
        }

        boundary = repair._classify_boundary(config, index_close)
        self.assertEqual(boundary["classification"], "BELOW")
        self.assertEqual(boundary["prior_observed_close_count"], 20)

        changed_future = index_close.copy()
        changed_future.loc[target] = 1.0
        self.assertEqual(repair._classify_boundary(config, changed_future), boundary)

        insufficient = index_close.drop(index_dates[0])
        with self.assertRaises(repair.RepairError):
            repair._classify_boundary(config, insufficient)

        invalid_nearest = index_close.copy()
        invalid_nearest.loc[prior_dates[-1]] = np.nan
        with self.assertRaises(repair.RepairError):
            repair._classify_boundary(config, invalid_nearest)

    @staticmethod
    def _cells() -> pd.DataFrame:
        rows = []

        def add(dimension, period, year, market, exposure, hits, expected, dates, years, month=None):
            rows.append({
                "period_scope": period,
                "year": year,
                "dimension_set": dimension,
                "industry_code": "CAPCO:AA01",
                "industry_name": "行业A",
                "month": month,
                "lag_market": market,
                "exposure_count": exposure,
                "hits_count": hits,
                "expected_hits": float(expected),
                "lift": float(hits / expected) if expected else None,
                "eligible_date_count": dates,
                "support_year_count": len(years.split(",")),
                "support_years": years,
            })

        for dimension, month in (("industry", None), ("industry_month", 5)):
            add(dimension, "ALL", None, None, 400, 400, 400, 4, "2021,2022", month)
            add(dimension, "YEAR", 2021, None, 200, 200, 200, 2, "2021", month)
            add(dimension, "YEAR", 2022, None, 200, 200, 200, 2, "2022", month)

        for dimension, month in (("industry_lag_market", None), ("industry_month_lag_market", 5)):
            add(dimension, "ALL", None, "BELOW", 300, 300, 300, 3, "2021,2022", month)
            add(dimension, "ALL", None, "UNKNOWN", 100, 100, 100, 1, "2021", month)
            add(dimension, "YEAR", 2021, "BELOW", 100, 100, 100, 1, "2021", month)
            add(dimension, "YEAR", 2021, "UNKNOWN", 100, 100, 100, 1, "2021", month)
            add(dimension, "YEAR", 2022, "BELOW", 200, 200, 200, 2, "2022", month)
        return pd.DataFrame(rows)

    def test_moves_only_boundary_unknown_cells_and_unions_support(self) -> None:
        original = self._cells()
        repaired = repair.migrate_unknown_market_cells(
            original, affected_year=2021, affected_month=5, valid_pool_size=100, eligible_days=4
        )

        self.assertFalse(repaired.loc[repaired["dimension_set"].isin(repair.CHANGED_CELL_DIMENSIONS), "lag_market"].eq("UNKNOWN").any())
        below = repaired.loc[
            repaired["period_scope"].eq("ALL")
            & repaired["dimension_set"].eq("industry_lag_market")
            & repaired["lag_market"].eq("BELOW")
        ].iloc[0]
        self.assertEqual(int(below["exposure_count"]), 400)
        self.assertEqual(int(below["hits_count"]), 400)
        self.assertEqual(float(below["expected_hits"]), 400.0)
        self.assertEqual(int(below["eligible_date_count"]), 4)
        self.assertEqual(int(below["support_year_count"]), 2)
        self.assertEqual(below["support_years"], "2021,2022")

        for dimension in ("industry", "industry_month"):
            before = repair._sort_cells(original.loc[original["dimension_set"].eq(dimension)].copy())
            after = repair._sort_cells(repaired.loc[repaired["dimension_set"].eq(dimension)].copy())
            pd.testing.assert_frame_equal(before, after)

    def test_refuses_to_move_an_unknown_cell_that_spans_multiple_dates(self) -> None:
        cells = self._cells()
        mask = cells["dimension_set"].eq("industry_lag_market") & cells["period_scope"].eq("ALL") & cells["lag_market"].eq("UNKNOWN")
        cells.loc[mask, "eligible_date_count"] = 2
        with self.assertRaises(repair.RepairError):
            repair.migrate_unknown_market_cells(
                cells, affected_year=2021, affected_month=5, valid_pool_size=100, eligible_days=4
            )


if __name__ == "__main__":
    unittest.main()
