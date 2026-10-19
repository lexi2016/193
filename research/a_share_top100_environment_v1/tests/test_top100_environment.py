import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import top100_environment as env


class Top100EnvironmentTests(unittest.TestCase):
    def test_module_loader_registers_dataclass_module_during_import(self) -> None:
        source = "from dataclasses import dataclass\n@dataclass\nclass BoundRecord:\n    value: int\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bound_record.py"
            path.write_text(source, encoding="utf-8")
            module = env._load_module(path, "top100_test_bound_record")
            self.assertEqual(module.BoundRecord(7).value, 7)

    @staticmethod
    def _panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        calendar = pd.DataFrame({"date": dates, "calendar_index": [0, 1, 2]})
        rows = []
        for i in range(101):
            code = f"{i:03d}"
            for date in dates[1:]:
                close = 100.0 if date == dates[1] else (120.0 if i < 99 else 101.0)
                amount = 0.0 if i == 100 else 1000.0
                rows.append({"date": date, "code": code, "close": close, "volume": 1.0, "amount": amount})
        # This security skips the immediately preceding shared-calendar day.
        rows.extend([
            {"date": dates[0], "code": "gap", "close": 100.0, "volume": 1.0, "amount": 1000.0},
            {"date": dates[2], "code": "gap", "close": 110.0, "volume": 1.0, "amount": 1000.0},
        ])
        rows.extend([
            {"date": dates[1], "code": "volzero", "close": 100.0, "volume": 1.0, "amount": 1000.0},
            {"date": dates[2], "code": "volzero", "close": 110.0, "volume": 0.0, "amount": 1000.0},
        ])
        panel = pd.DataFrame(rows).sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
        industry = pd.DataFrame({
            "code": [f"{i:03d}" for i in range(100)],
            "industry_code": ["A"] * 100,
            "industry_name": ["原始行业名"] * 100,
        })
        index_close = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
        return panel, calendar, industry, index_close

    def test_pool_gap_zero_volume_unknown_and_amount_diagnostics(self) -> None:
        panel, calendar, industry, index_close = self._panel()
        observations, coverage, coverage_year = env.prepare_observations(panel, calendar, industry, index_close)
        target = pd.Timestamp("2024-01-04")
        day = observations.loc[observations["date"].eq(target)]
        self.assertEqual(int(day["pool_valid"].sum()), 101)
        self.assertTrue(bool(day.loc[day["code"].eq("gap"), "missing_previous_row"].iloc[0]))
        self.assertFalse(bool(day.loc[day["code"].eq("gap"), "pool_valid"].iloc[0]))
        self.assertFalse(bool(day.loc[day["code"].eq("volzero"), "pool_valid"].iloc[0]))
        self.assertEqual(day.loc[day["code"].eq("100"), "industry_code"].iloc[0], "UNKNOWN")
        self.assertTrue(bool(day.loc[day["code"].eq("100"), "pool_valid"].iloc[0]))
        self.assertEqual(int(coverage.loc[coverage["date"].eq(target), "amount_zero_with_positive_volume_rows"].iloc[0]), 1)
        self.assertEqual(int(coverage_year["source_rows"].sum()), len(panel))

    def test_top100_tie_uses_code_ascending_and_unknown_stays_in_denominator(self) -> None:
        panel, calendar, industry, index_close = self._panel()
        observations, coverage, coverage_year = env.prepare_observations(panel, calendar, industry, index_close)
        target = pd.Timestamp("2024-01-04")
        day = observations.loc[observations["date"].eq(target) & observations["pool_valid"]]
        selected, tie_count, excluded_ties = env.select_top100(day)
        self.assertEqual(len(selected), 100)
        self.assertEqual(tie_count, 2)
        self.assertEqual(excluded_ties, 1)
        self.assertIn("099", set(selected["code"]))
        self.assertNotIn("100", set(selected["code"]))

        stats = env.compute_statistics(observations, coverage, coverage_year)
        target_cells = stats["industry_cells"].query("period_scope == 'ALL' and dimension_set == 'industry'").set_index("industry_code")
        self.assertAlmostEqual(float(target_cells.loc["A", "expected_hits"]), 100.0 * 100.0 / 101.0, places=10)
        self.assertEqual(int(target_cells.loc["UNKNOWN", "exposure_count"]), 1)
        self.assertEqual(int(target_cells.loc["UNKNOWN", "hits_count"]), 0)
        self.assertTrue(all(stats["qa"]["quality_checks"].values()))

    def test_lag_index_uses_t_minus_1_and_twenty_nearest_valid_closes(self) -> None:
        dates = pd.date_range("2024-01-01", periods=24, freq="D")
        calendar = pd.DataFrame({"date": dates, "calendar_index": np.arange(len(dates))})
        values = np.full(len(dates), 100.0)
        values[-2] = 105.0  # T-1
        values[-1] = 10000.0  # T must not enter its own regime.
        target = dates[-1]
        values[0] = np.nan  # Missing data outside the nearest 20 rows does not affect the reference window.
        index_close = pd.Series(values.copy(), index=dates)
        regime = env.market_regime_by_date(calendar, [target], index_close)[target]
        values[-1] = 1.0
        changed_future = pd.Series(values, index=dates)
        self.assertEqual(env.market_regime_by_date(calendar, [target], changed_future)[target], regime)
        self.assertEqual(regime, "AT_OR_ABOVE")

        values[-2] = 95.0
        below = pd.Series(values, index=dates)
        self.assertEqual(env.market_regime_by_date(calendar, [target], below)[target], "BELOW")

    def test_missing_close_inside_nearest_twenty_is_unknown_without_backfill(self) -> None:
        dates = pd.date_range("2024-01-01", periods=24, freq="D")
        calendar = pd.DataFrame({"date": dates, "calendar_index": np.arange(len(dates))})
        values = np.full(len(dates), 100.0)
        values[-2] = 105.0  # T-1
        values[-1] = 10000.0  # T must not enter its own regime.
        values[-12] = np.nan  # One of the nearest 20 pre-T-1 rows is invalid.
        target = dates[-1]
        index_close = pd.Series(values, index=dates)
        self.assertEqual(env.market_regime_by_date(calendar, [target], index_close)[target], "UNKNOWN")

    def test_first_panel_day_uses_twenty_index_closes_before_t_minus_1(self) -> None:
        prior_dates = pd.date_range("2021-04-15", periods=20, freq="D")
        t_minus_1 = pd.Timestamp("2021-05-06")
        target = pd.Timestamp("2021-05-07")
        calendar = pd.DataFrame({"date": [t_minus_1, target], "calendar_index": [0, 1]})
        index_dates = prior_dates.append(pd.DatetimeIndex([t_minus_1, target]))
        index_values = np.r_[np.full(20, 100.0), 99.0, 9999.0]
        index_close = pd.Series(index_values, index=index_dates)

        regime = env.market_regime_by_date(calendar, [target], index_close)[target]
        self.assertEqual(regime, "BELOW")

        no_future_leak = index_close.copy()
        no_future_leak.loc[target] = 1.0
        self.assertEqual(env.market_regime_by_date(calendar, [target], no_future_leak)[target], regime)

        insufficient_history = index_close.loc[index_dates[1:]].copy()
        self.assertEqual(
            env.market_regime_by_date(calendar, [target], insufficient_history)[target],
            "UNKNOWN",
        )


if __name__ == "__main__":
    unittest.main()
