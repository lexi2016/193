"""Focused behavior tests for strict R3 calendar-day parsing."""
from datetime import date, datetime
from pathlib import Path
import sys
import unittest

from calendar_dates import parse_calendar_day

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_materialization_r2  # noqa: E402


class ParseCalendarDayTests(unittest.TestCase):
    def test_reproduces_permissive_legacy_parser_bug(self):
        self.assertEqual(run_materialization_r2.parse_day("2026-02-30"), 20260230)
        self.assertEqual(run_materialization_r2.parse_day("2026-13-01"), 20261301)

    def test_accepts_existing_compact_date_tokens(self):
        self.assertEqual(parse_calendar_day(20261015), 20261015)
        self.assertEqual(parse_calendar_day("20261015"), 20261015)
        self.assertEqual(parse_calendar_day("2026-10-15"), 20261015)

    def test_accepts_valid_leap_day_and_date_objects(self):
        self.assertEqual(parse_calendar_day("20240229"), 20240229)
        self.assertEqual(parse_calendar_day("2024-02-29"), 20240229)
        self.assertEqual(parse_calendar_day(date(2024, 2, 29)), 20240229)
        self.assertEqual(parse_calendar_day(datetime(2024, 2, 29)), 20240229)

    def test_rejects_invalid_leap_days_months_and_days(self):
        for value in ("20230229", "2023-02-29", "2026-02-30", "2026-13-01", "20261301", "2026-04-31"):
            with self.subTest(value=value):
                self.assertIsNone(parse_calendar_day(value))

    def test_rejects_repairable_but_noncanonical_tokens(self):
        for value in ("2026/10/15", "2026-2-3", "x20261015", "20261015x", " 20261015", "20261015 "):
            with self.subTest(value=value):
                self.assertIsNone(parse_calendar_day(value))

    def test_rejects_non_date_or_non_midnight_values(self):
        self.assertIsNone(parse_calendar_day(None))
        self.assertIsNone(parse_calendar_day(True))
        self.assertIsNone(parse_calendar_day(202610150.0))
        self.assertIsNone(parse_calendar_day(datetime(2026, 10, 15, 9, 30)))
        self.assertIsNone(parse_calendar_day("18991231"))


if __name__ == "__main__":
    unittest.main()
