import unittest

import numpy as np
import pandas as pd

import seasonality_stats as stats


class SeasonalityStatsTests(unittest.TestCase):
    @staticmethod
    def _observations() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        dates = pd.date_range("2022-05-01", periods=3, freq="D")
        calendar = pd.DataFrame({"date": dates, "calendar_index": np.arange(len(dates))})
        rows = []
        for date in dates:
            for i in range(100):
                rows.append({
                    "date": date,
                    "code": f"A{i:03d}",
                    "return": 0.10 - i * 0.001,
                    "pool_valid": True,
                    "industry_code": "A",
                    "industry_name": "Industry A",
                    "lag_market": "AT_OR_ABOVE",
                })
            if date == dates[0]:
                rows.append({
                    "date": date,
                    "code": "Z999",
                    "return": -0.50,
                    "pool_valid": True,
                    "industry_code": "UNKNOWN",
                    "industry_name": None,
                    "lag_market": "AT_OR_ABOVE",
                })
        index_close = pd.Series([100.0, 101.0, 102.0], index=dates)
        return pd.DataFrame(rows), calendar, index_close

    def test_projection_keeps_unknown_denominator_and_zero_n_returns_unknown(self) -> None:
        observations, calendar, index_close = self._observations()
        projection = stats.build_daily_projection(observations, calendar, index_close)
        first = projection.loc[projection["date"].eq(pd.Timestamp("2022-05-01"))].set_index("industry_code")
        self.assertEqual(int(first["top100_hits"].sum()), 100)
        self.assertAlmostEqual(float(first["expected_top100_hits"].sum()), 100.0, places=12)
        self.assertEqual(int(first.loc["UNKNOWN", "industry_n"]), 1)
        self.assertEqual(int(first.loc["UNKNOWN", "top100_hits"]), 0)
        self.assertAlmostEqual(float(first.loc["UNKNOWN", "expected_top100_hits"]), 100.0 / 101.0, places=12)

        second = projection.loc[projection["date"].eq(pd.Timestamp("2022-05-02"))].set_index("industry_code")
        self.assertEqual(int(second.loc["UNKNOWN", "industry_n"]), 0)
        for field in (
            "industry_return_sum", "industry_return_square_sum", "industry_mean_return",
            "industry_median_return", "industry_sample_std_return", "industry_excess_vs_pool",
        ):
            self.assertTrue(pd.isna(second.loc["UNKNOWN", field]), field)
        self.assertEqual(float(second.loc["UNKNOWN", "expected_top100_hits"]), 0.0)

    def test_month_tenday_grid_is_complete_and_partial_years_are_separate(self) -> None:
        observations, calendar, index_close = self._observations()
        projection = stats.build_daily_projection(observations, calendar, index_close)
        seasonal = stats.summarize_month_tenday(projection)
        all_full = seasonal.loc[seasonal["period_scope"].eq("FULL_YEARS_2022_2025")]
        self.assertEqual(len(all_full), 2 * 12 * 3)
        self.assertEqual(set(all_full["industry_code"]), {"A", "UNKNOWN"})
        zero = all_full.loc[
            all_full["industry_code"].eq("UNKNOWN")
            & all_full["month"].eq(5)
            & all_full["ten_day"].eq(1)
        ].iloc[0]
        self.assertEqual(int(zero["industry_n_sum"]), 1)
        no_support = all_full.loc[
            all_full["industry_code"].eq("UNKNOWN")
            & all_full["month"].eq(1)
            & all_full["ten_day"].eq(1)
        ].iloc[0]
        self.assertEqual(int(no_support["industry_n_sum"]), 0)
        self.assertTrue(pd.isna(no_support["row_weighted_industry_mean_return"]))
        self.assertIn("PARTIAL_YEAR_2021", set(seasonal["period_scope"]))
        self.assertIn("PARTIAL_YEAR_2026", set(seasonal["period_scope"]))

    def test_event_bins_use_fixed_anchor_offsets_and_require_complete_sessions(self) -> None:
        dates = pd.date_range("2022-04-10", "2022-05-20", freq="D")
        calendar = pd.DataFrame({"date": dates, "calendar_index": np.arange(len(dates))})
        rows = []
        for date in dates:
            for category, count, daily_mean, expected in (("B06", 20, 0.001, 20), ("OTHER", 80, 0.0005, 80)):
                rows.append({
                    "date": date,
                    "calendar_position": int((date - dates[0]).days),
                    "year": date.year,
                    "month": date.month,
                    "ten_day": stats.month_tenday(date),
                    "industry_code": category,
                    "industry_name": category,
                    "industry_n": count,
                    "industry_return_sum": daily_mean * count,
                    "industry_return_square_sum": 0.00002,
                    "industry_mean_return": daily_mean,
                    "industry_median_return": 0.001,
                    "industry_sample_std_return": 0.001,
                    "industry_excess_vs_pool": 0.0005,
                    "top100_hits": expected,
                    "expected_top100_hits": expected,
                    "top100_return_sum": daily_mean * expected,
                    "date_fullpool_n": 100,
                    "date_pool_mean_return": 0.0005,
                    "same_day_000300_return": 0.001,
                    "lag_market": "AT_OR_ABOVE",
                    "eligible_top100_date": True,
                })
        projection = pd.DataFrame(rows)
        detail, common = stats.build_event_windows(projection, calendar)
        may_post = detail.loc[
            detail["event"].eq("MAY_DAY")
            & detail["window_bin"].eq("post_0_4")
            & detail["industry_code"].eq("B06")
            & detail["lag_market"].eq("AT_OR_ABOVE")
        ].iloc[0]
        self.assertEqual(pd.Timestamp(may_post["anchor_date"]), pd.Timestamp("2022-05-01"))
        self.assertEqual(may_post["window_status"], "COMPLETE")
        self.assertEqual(int(may_post["bounded_session_count"]), 5)
        self.assertEqual(int(may_post["window_top100_hits_sum"]), 100)
        self.assertAlmostEqual(float(may_post["window_top100_lift"]), 1.0, places=12)
        self.assertTrue((common["common_complete_year_count"] <= 1).all())

        bundle = stats.analyze_projection(projection, calendar, latest_date=dates[-1])
        self.assertTrue(bundle["qa"]["daily_top100_hits_equal_100"])
        self.assertTrue(bundle["qa"]["full_month_tenday_grid"])
        self.assertTrue(bundle["qa"]["position_state_timing_prior_only"])
        self.assertEqual(bundle["latest_asof_close"].iloc[0]["state_timing"], "ASOF_CLOSE_T_INCLUDED_NOT_OPENING_STATE")

        missing_date = pd.Timestamp("2022-05-03")
        incomplete = projection.loc[~projection["date"].eq(missing_date)]
        detail_missing, _ = stats.build_event_windows(incomplete, calendar)
        failed = detail_missing.loc[
            detail_missing["event"].eq("MAY_DAY")
            & detail_missing["window_bin"].eq("post_0_4")
            & detail_missing["industry_code"].eq("B06")
            & detail_missing["lag_market"].eq("AT_OR_ABOVE")
        ].iloc[0]
        self.assertEqual(failed["window_status"], "INCOMPLETE")
        self.assertEqual(failed["missing_reason"], "NO_PROJECTION_FOR_SHARED_SESSION")

    @staticmethod
    def _position_projection(periods: int = 270) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.date_range("2022-01-01", periods=periods, freq="D")
        calendar = pd.DataFrame({"date": dates, "calendar_index": np.arange(periods)})
        rows = []
        for i, date in enumerate(dates):
            daily_return = 0.001
            rows.append({
                "date": date,
                "year": date.year,
                "industry_code": "A",
                "industry_name": "Industry A",
                "industry_n": 10,
                "industry_mean_return": daily_return,
                "date_pool_mean_return": 0.0,
                "industry_excess_vs_pool": daily_return,
                "industry_sample_std_return": 0.01,
                "top100_hits": 10,
                "expected_top100_hits": 10.0,
                "eligible_top100_date": True,
                "lag_market": "AT_OR_ABOVE",
            })
        return pd.DataFrame(rows), calendar

    def test_position_and_relative_strength_at_t_use_only_through_t_minus_1(self) -> None:
        projection, calendar = self._position_projection()
        position = stats.build_position_states(projection, calendar)
        target = pd.Timestamp("2022-09-10")
        original = position.loc[position["date"].eq(target)].iloc[0]
        self.assertEqual(original["position_state_asof_t_minus_1"], "HIGH")
        self.assertEqual(original["rs20_state_asof_t_minus_1"], "POS")
        self.assertEqual(original["rs60_state_asof_t_minus_1"], "POS")
        self.assertEqual(original["position_history_n"], 252)
        self.assertLess(pd.Timestamp(original["position_state_asof_date"]), target)

        changed = projection.copy()
        changed.loc[changed["date"].eq(target), "industry_mean_return"] = 0.50
        changed.loc[changed["date"].eq(target), "industry_excess_vs_pool"] = 0.50
        altered = stats.build_position_states(changed, calendar).loc[
            lambda frame: frame["date"].eq(target)
        ].iloc[0]
        self.assertEqual(altered["position_midrank_asof_t_minus_1"], original["position_midrank_asof_t_minus_1"])
        self.assertEqual(altered["rs20_difference_asof_t_minus_1"], original["rs20_difference_asof_t_minus_1"])

    def test_gap_restarts_equal_weight_level_and_asof_snapshot_is_close_only(self) -> None:
        projection, calendar = self._position_projection()
        gap_date = pd.Timestamp("2022-09-10")
        projection.loc[projection["date"].eq(gap_date), "industry_n"] = 0
        projection.loc[projection["date"].eq(gap_date), "industry_mean_return"] = np.nan
        projection.loc[projection["date"].eq(gap_date), "industry_excess_vs_pool"] = np.nan
        position = stats.build_position_states(projection, calendar)
        after = position.loc[position["date"].eq(gap_date + pd.Timedelta(days=1))].iloc[0]
        self.assertEqual(after["industry_equal_weight_level"], 100.0)
        self.assertEqual(after["position_state_asof_t_minus_1"], "UNKNOWN")

        asof = stats.latest_asof_close_states(position, asof_date=calendar["date"].iloc[-1])
        row = asof.iloc[0]
        self.assertEqual(row["state_timing"], "ASOF_CLOSE_T_INCLUDED_NOT_OPENING_STATE")
        self.assertEqual(row["position_history_n"], 252)


if __name__ == "__main__":
    unittest.main()
