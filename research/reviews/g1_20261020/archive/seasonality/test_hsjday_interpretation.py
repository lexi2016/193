import json
import unittest

import numpy as np
import pandas as pd

from research.a_share_top100_environment_v1 import hsjday_interpretation as h


class InterpretationTests(unittest.TestCase):
    def arrays(self, cell_hits=13, expected=10, sector_hits=100, sector_expected=100):
        return [np.full(20, x, dtype=float) for x in [cell_hits, expected, sector_hits, sector_expected]]

    def test_pooled_and_half_baseline_use_counts_not_average_lifts(self):
        hits, expected, ih, ie = self.arrays()
        hits[:2] = [90, 10]; expected[:2] = [10, 90]
        ih[:2] = [100, 100]; ie[:2] = [10, 90]
        result = h.period_metrics(hits, expected, ih, ie, h.YEARS <= 2007)
        self.assertEqual(result['lift'], 1)
        self.assertEqual(result['industry_lift'], 2)
        self.assertEqual(result['seasonal_ratio'], .5)
        self.assertNotAlmostEqual(result['industry_lift'], (10 + 100/90)/2)

    def test_consistent_absolute_and_seasonal_advantage_pass(self):
        m, annual = h.cell_metrics(*self.arrays())
        self.assertTrue(m['stable_absolute']); self.assertTrue(m['stable_seasonal'])
        self.assertEqual(m['years_lift_above_1'], 20)
        self.assertEqual(m['years_seasonal_above_1'], 20)
        self.assertEqual(len(annual), 20)
        self.assertAlmostEqual(m['seasonal_ratio'], 1.3)

    def test_relative_strength_below_market_is_not_seasonal_market_pass(self):
        m, _ = h.cell_metrics(*self.arrays(cell_hits=8, sector_hits=50))
        self.assertTrue(m['stable_seasonal_only'])
        self.assertFalse(m['stable_absolute']); self.assertFalse(m['stable_seasonal'])
        self.assertEqual(m['lift'], .8); self.assertEqual(m['seasonal_ratio'], 1.6)

    def test_persistent_sector_strength_is_not_calendar_increment(self):
        m, _ = h.cell_metrics(*self.arrays(sector_hits=200))
        self.assertTrue(m['stable_absolute']); self.assertFalse(m['stable_seasonal_only'])
        self.assertAlmostEqual(m['seasonal_ratio'], .65)

    def test_two_remove_three_mechanisms_remove_different_years(self):
        hits, expected, ih, ie = self.arrays(cell_hits=12)
        hits[:4] = [20, 18, 17, 16]; ih[:6] = [200, 200, 200, 20, 20, 20]
        m, _ = h.cell_metrics(hits, expected, ih, ie)
        self.assertEqual(json.loads(m['absolute_drop3_removed_years']), [2006, 2007, 2008])
        self.assertEqual(json.loads(m['seasonal_drop3_removed_years']), [2009, 2010, 2011])
        mask = np.ones(20, dtype=bool); mask[[3, 4, 5]] = False
        expected_s = (hits[mask].sum()/expected[mask].sum())/(ih[mask].sum()/ie[mask].sum())
        self.assertAlmostEqual(m['seasonal_drop3_seasonal_ratio'], expected_s)
        score = hits-expected*(ih/ie)
        self.assertAlmostEqual(m['seasonal_drop3_top3_positive_excess_share'], np.maximum(score[[3,4,5]],0).sum()/np.maximum(score,0).sum())

    def test_tied_excess_removes_earliest_years_deterministically(self):
        m, _ = h.cell_metrics(*self.arrays())
        self.assertEqual(json.loads(m['absolute_drop3_removed_years']), [2006, 2007, 2008])
        self.assertEqual(json.loads(m['seasonal_drop3_removed_years']), [2006, 2007, 2008])

    def test_few_year_inflation_and_positive_excess_denominator(self):
        values = self.arrays(cell_hits=4)
        values[0][:3] = 100
        m, _ = h.cell_metrics(*values)
        self.assertTrue(m['few_year_inflated_absolute']); self.assertTrue(m['few_year_inflated_seasonal'])
        self.assertFalse(m['stable_absolute'])
        self.assertAlmostEqual(m['absolute_drop3_lift'], .4)
        self.assertEqual(m['absolute_drop3_top3_positive_excess_share'], 1)
        self.assertEqual(m['seasonal_drop3_top3_positive_excess_share'], 1)

    def test_intermediate_stage_pattern_is_not_forced_into_inflated_group(self):
        values = self.arrays(cell_hits=8)
        values[0][:10] = 20
        m, _ = h.cell_metrics(*values)
        self.assertEqual(m['half1_lift'], 2); self.assertEqual(m['half2_lift'], .8)
        self.assertFalse(m['stable_absolute']); self.assertFalse(m['few_year_inflated_absolute'])
        self.assertGreater(m['absolute_drop3_lift'], 1)

    def test_zero_support_keeps_undefined_metrics_and_no_unsafe_division(self):
        m, annual = h.cell_metrics(*self.arrays(cell_hits=0, expected=0))
        self.assertFalse(m['eligible']); self.assertIsNone(m['lift']); self.assertIsNone(m['seasonal_ratio'])
        self.assertIsNone(m['absolute_drop3_top3_positive_excess_share'])
        self.assertIsNone(m['seasonal_drop3_top3_positive_excess_share'])
        self.assertTrue(annual.lift.isna().all())
        self.assertTrue(all(v is None for v in json.loads(m['annual_lifts_json']).values()))

    def test_zero_industry_lift_does_not_create_infinite_seasonality(self):
        m, annual = h.cell_metrics(*self.arrays(cell_hits=0, sector_hits=0))
        self.assertEqual(m['lift'], 0); self.assertIsNone(m['seasonal_ratio'])
        self.assertEqual(m['seasonal_years_supported'], 0)
        self.assertFalse(m['stable_seasonal_only'])
        self.assertTrue(annual.seasonal_ratio.isna().all())

    def test_exact_expected_and_lift_thresholds(self):
        m, _ = h.cell_metrics(*self.arrays(cell_hits=6, expected=5))
        self.assertEqual(m['expected_hits'], 100); self.assertEqual(m['lift'], 1.2)
        self.assertTrue(m['stable_absolute'])
        below, _ = h.cell_metrics(*self.arrays(cell_hits=6, expected=4.999))
        self.assertFalse(below['eligible']); self.assertFalse(below['stable_absolute'])
        no_lift, _ = h.cell_metrics(*self.arrays(cell_hits=5.999, expected=5))
        self.assertFalse(no_lift['stable_absolute'])

    def test_unknown_classification_cannot_be_promoted(self):
        m, _ = h.cell_metrics(*self.arrays(), explainable=False)
        self.assertTrue(m['eligible']); self.assertFalse(m['stable_absolute']); self.assertFalse(m['stable_seasonal'])

    def test_invalid_counts_and_alignment_fail_closed(self):
        values = self.arrays(); values[0][0] = np.nan
        with self.assertRaises(ValueError): h.cell_metrics(*values)
        values = self.arrays(); values[0][0] = -1
        with self.assertRaises(ValueError): h.cell_metrics(*values)
        values = self.arrays(); values[0] = values[0][:-1]
        with self.assertRaises(ValueError): h.cell_metrics(*values)
        values = self.arrays(); values[1][0] = 0
        with self.assertRaises(ValueError): h.cell_metrics(*values)

    def test_mapping_preserves_taxonomies_and_missing_name_is_explicit(self):
        dictionary = pd.DataFrame([{'industry_key': 'CAPCO:F51', 'classification_system': 'CAPCO', 'industry_code': 'F51', 'industry_label': 'CAPCO:F51'}, {'industry_key':'CSRC:F51','classification_system':'CSRC','industry_code':'F51','industry_label':'批发业'}, {'industry_key':'UNKNOWN','classification_system':'UNKNOWN','industry_code':'','industry_label':'UNKNOWN'}])
        supplied = [{'industry_key': key, 'classification_system': system, 'industry_code': 'F51', 'chinese_name': '批发业', 'mapping_status': 'MAPPED', 'source_url':'https://official.example/source'} for key, system in [('CAPCO:F51','CAPCO'),('CSRC:F51','CSRC')]]
        result = h.normalize_mapping(dictionary, supplied)
        self.assertEqual(len(result), 3); self.assertEqual(result.industry_key.nunique(), 3)
        self.assertEqual(result.iloc[-1].chinese_name, '中文名称未知'); self.assertFalse(result.iloc[-1].explainable)
        supplied[0]['classification_system'] = 'CSRC'
        with self.assertRaisesRegex(ValueError, 'taxonomy mismatch'): h.normalize_mapping(dictionary, supplied)


if __name__ == '__main__':
    unittest.main()
