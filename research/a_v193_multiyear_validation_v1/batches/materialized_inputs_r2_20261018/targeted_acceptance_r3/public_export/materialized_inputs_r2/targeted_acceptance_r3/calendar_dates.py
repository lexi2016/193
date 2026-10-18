"""Strict parser for date-only values used in the bounded R3 panel audit."""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Any


def parse_calendar_day(value: Any) -> int | None:
    """Return YYYYMMDD for a valid calendar-day token; never repair malformed input."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        if value.time() != time.min:
            return None
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    elif isinstance(value, int):
        raw = str(value)
        if len(raw) != 8 or not raw.isascii() or not raw.isdigit():
            return None
        try:
            parsed = date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            return None
    elif isinstance(value, str):
        if value != value.strip():
            return None
        if len(value) == 8 and value.isascii() and value.isdigit():
            year, month, day = int(value[:4]), int(value[4:6]), int(value[6:8])
        elif len(value) == 10 and value[4] == "-" and value[7] == "-":
            parts = value.split("-")
            if len(parts) != 3 or any(len(part) != size or not part.isascii() or not part.isdigit()
                                      for part, size in zip(parts, (4, 2, 2))):
                return None
            year, month, day = map(int, parts)
        else:
            return None
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
    else:
        return None

    if not 1900 <= parsed.year <= 2100:
        return None
    return parsed.year * 10000 + parsed.month * 100 + parsed.day
