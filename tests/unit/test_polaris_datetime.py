"""Tests for Polaris datetime parsing and timezone handling.

These tests cover the critical timezone pipeline:
  _parse_time_string → parse_datetime → format_datetime_for_api

The core bug these prevent: a user says "11:30 AM KST" and the event
lands at 8:30 PM because the timezone abbreviation was silently discarded.
"""

import pytest
from datetime import datetime, time, date, timedelta
import zoneinfo

from polaris.tools.utils import (
    _parse_time_string,
    parse_datetime,
    format_datetime_for_api,
)


# ===========================================================================
# _parse_time_string
# ===========================================================================


class TestParseTimeString:
    """Tests for _parse_time_string(time_str) -> (time, Optional[tz])."""

    # --- Basic time parsing ---

    def test_24h_with_minutes(self):
        t, tz = _parse_time_string("14:30")
        assert t == time(14, 30)
        assert tz is None

    def test_24h_no_minutes(self):
        t, tz = _parse_time_string("9")
        assert t == time(9, 0)
        assert tz is None

    def test_12h_am(self):
        t, tz = _parse_time_string("9:30 AM")
        assert t == time(9, 30)
        assert tz is None

    def test_12h_pm(self):
        t, tz = _parse_time_string("2:30 PM")
        assert t == time(14, 30)
        assert tz is None

    def test_12h_pm_no_space(self):
        t, tz = _parse_time_string("2:30PM")
        assert t == time(14, 30)
        assert tz is None

    def test_hour_only_pm(self):
        t, tz = _parse_time_string("2pm")
        assert t == time(14, 0)
        assert tz is None

    def test_hour_only_am(self):
        t, tz = _parse_time_string("9am")
        assert t == time(9, 0)
        assert tz is None

    # --- Noon/midnight edge cases ---

    def test_noon_12pm(self):
        t, tz = _parse_time_string("12:00 PM")
        assert t == time(12, 0)

    def test_midnight_12am(self):
        t, tz = _parse_time_string("12:00 AM")
        assert t == time(0, 0)

    def test_12pm_no_minutes(self):
        t, tz = _parse_time_string("12pm")
        assert t == time(12, 0)

    def test_12am_no_minutes(self):
        t, tz = _parse_time_string("12am")
        assert t == time(0, 0)

    # --- Timezone abbreviation extraction ---

    def test_kst(self):
        t, tz = _parse_time_string("11:30 AM KST")
        assert t == time(11, 30)
        assert tz == "Asia/Seoul"

    def test_pst(self):
        t, tz = _parse_time_string("2:30 PM PST")
        assert t == time(14, 30)
        assert tz == "America/Los_Angeles"

    def test_est(self):
        t, tz = _parse_time_string("9:00 AM EST")
        assert t == time(9, 0)
        assert tz == "America/New_York"

    def test_utc(self):
        t, tz = _parse_time_string("14:30 UTC")
        assert t == time(14, 30)
        assert tz == "UTC"

    def test_jst(self):
        t, tz = _parse_time_string("10:00 AM JST")
        assert t == time(10, 0)
        assert tz == "Asia/Tokyo"

    def test_cet(self):
        t, tz = _parse_time_string("8:00 PM CET")
        assert t == time(20, 0)
        assert tz == "Europe/Berlin"

    def test_gmt(self):
        t, tz = _parse_time_string("3:00 PM GMT")
        assert t == time(15, 0)
        assert tz == "Europe/London"

    def test_case_insensitive_tz(self):
        t, tz = _parse_time_string("11:30 AM kst")
        assert tz == "Asia/Seoul"

    def test_mixed_case_tz(self):
        t, tz = _parse_time_string("11:30 AM Kst")
        assert tz == "Asia/Seoul"

    # --- Unknown timezone → None ---

    def test_unknown_tz_returns_none(self):
        t, tz = _parse_time_string("11:30 AM XYZ")
        assert t == time(11, 30)
        assert tz is None

    def test_unknown_long_tz_returns_none(self):
        t, tz = _parse_time_string("9:00 AM FAKE")
        assert t == time(9, 0)
        assert tz is None

    # --- AM/PM not confused with timezone ---

    def test_pm_not_treated_as_timezone(self):
        """'PM' should be parsed as AM/PM indicator, not timezone."""
        t, tz = _parse_time_string("2 PM")
        assert t == time(14, 0)
        assert tz is None

    def test_am_not_treated_as_timezone(self):
        t, tz = _parse_time_string("9 AM")
        assert t == time(9, 0)
        assert tz is None

    # --- Whitespace handling ---

    def test_leading_trailing_whitespace(self):
        t, tz = _parse_time_string("  11:30 AM KST  ")
        assert t == time(11, 30)
        assert tz == "Asia/Seoul"

    # --- Error cases ---

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_time_string("")

    def test_no_digits_raises(self):
        with pytest.raises(ValueError):
            _parse_time_string("noon")

    # --- CST maps to America/Chicago (known ambiguity) ---

    def test_cst_maps_to_chicago(self):
        t, tz = _parse_time_string("3:00 PM CST")
        assert tz == "America/Chicago"


# ===========================================================================
# parse_datetime
# ===========================================================================


class TestParseDatetime:
    """Tests for parse_datetime(date_str, time_str, timezone)."""

    # --- Timezone from time string overrides config default ---

    def test_tz_from_time_string_overrides_config(self):
        """'11:30 AM KST' with config='UTC' should produce Asia/Seoul time."""
        dt = parse_datetime("2026-01-25", "11:30 AM KST", timezone="UTC")
        assert dt.hour == 11
        assert dt.minute == 30
        assert str(dt.tzinfo) == "Asia/Seoul"

    def test_pst_overrides_config(self):
        dt = parse_datetime("2026-01-25", "2:30 PM PST", timezone="Asia/Seoul")
        assert dt.hour == 14
        assert dt.minute == 30
        assert str(dt.tzinfo) == "America/Los_Angeles"

    def test_no_tz_in_time_uses_config(self):
        """Plain time without abbreviation uses the configured timezone."""
        dt = parse_datetime("2026-01-25", "11:30 AM", timezone="Asia/Seoul")
        assert dt.hour == 11
        assert dt.minute == 30
        assert str(dt.tzinfo) == "Asia/Seoul"

    def test_no_tz_in_time_uses_utc_default(self):
        dt = parse_datetime("2026-01-25", "9:00 AM", timezone="UTC")
        assert str(dt.tzinfo) == "UTC"

    # --- Result is always timezone-aware ---

    def test_result_is_timezone_aware(self):
        dt = parse_datetime("2026-01-25", "11:30 AM", timezone="America/New_York")
        assert dt.tzinfo is not None

    def test_result_aware_with_tz_in_time(self):
        dt = parse_datetime("2026-01-25", "11:30 AM KST")
        assert dt.tzinfo is not None
        assert str(dt.tzinfo) == "Asia/Seoul"

    # --- Relative dates ---

    def test_today(self):
        dt = parse_datetime("today", "9:00 AM", timezone="UTC")
        today = datetime.now(zoneinfo.ZoneInfo("UTC")).date()
        assert dt.date() == today

    def test_tomorrow(self):
        dt = parse_datetime("tomorrow", "9:00 AM", timezone="UTC")
        tomorrow = (datetime.now(zoneinfo.ZoneInfo("UTC")) + timedelta(days=1)).date()
        assert dt.date() == tomorrow

    def test_yesterday(self):
        dt = parse_datetime("yesterday", "9:00 AM", timezone="UTC")
        yesterday = (datetime.now(zoneinfo.ZoneInfo("UTC")) - timedelta(days=1)).date()
        assert dt.date() == yesterday

    # --- ISO date formats ---

    def test_iso_date(self):
        dt = parse_datetime("2026-03-15", "2:00 PM", timezone="UTC")
        assert dt.date() == date(2026, 3, 15)
        assert dt.hour == 14

    def test_us_date_format(self):
        dt = parse_datetime("01/25/2026", "11:30 AM", timezone="UTC")
        assert dt.date() == date(2026, 1, 25)

    # --- No time provided, relative date → midnight ---

    def test_no_time_relative_date_defaults_midnight(self):
        """Relative dates without a time component default to midnight."""
        dt = parse_datetime("tomorrow", timezone="UTC")
        assert dt.hour == 0
        assert dt.minute == 0

    # --- Next day-of-week ---

    def test_next_monday(self):
        dt = parse_datetime("next monday", "9:00 AM", timezone="UTC")
        assert dt.weekday() == 0  # Monday

    def test_next_friday(self):
        dt = parse_datetime("next friday", "10:00 AM", timezone="UTC")
        assert dt.weekday() == 4  # Friday

    def test_next_week(self):
        dt = parse_datetime("next week", timezone="UTC")
        today = datetime.now(zoneinfo.ZoneInfo("UTC")).date()
        expected = today + timedelta(days=7)
        assert dt.date() == expected

    # --- The exact bug scenario ---

    def test_bug_scenario_kst_with_utc_config(self):
        """
        Reproduces the original bug: user says '11:30 AM KST', config is UTC.
        Before fix: 11:30 was treated as UTC → showed as 8:30 PM in KST calendar.
        After fix: 11:30 is correctly stamped as Asia/Seoul.
        """
        dt = parse_datetime("2026-01-25", "11:30 AM KST", timezone="UTC")
        assert dt == datetime(2026, 1, 25, 11, 30, tzinfo=zoneinfo.ZoneInfo("Asia/Seoul"))

    def test_bug_scenario_api_output(self):
        """Full pipeline: parse → format for API should produce +09:00 offset."""
        dt = parse_datetime("2026-01-25", "11:30 AM KST", timezone="UTC")
        api_str = format_datetime_for_api(dt, timezone="UTC")
        assert api_str == "2026-01-25T11:30:00+09:00"


# ===========================================================================
# format_datetime_for_api
# ===========================================================================


class TestFormatDatetimeForApi:
    """Tests for format_datetime_for_api(dt, timezone)."""

    def test_aware_datetime_ignores_timezone_param(self):
        """If dt is already timezone-aware, the timezone param is irrelevant."""
        tz = zoneinfo.ZoneInfo("Asia/Seoul")
        dt = datetime(2026, 1, 25, 11, 30, tzinfo=tz)
        result = format_datetime_for_api(dt, timezone="America/New_York")
        assert result == "2026-01-25T11:30:00+09:00"

    def test_naive_datetime_uses_timezone_param(self):
        """Naive datetime gets the specified timezone attached."""
        dt = datetime(2026, 1, 25, 11, 30)
        result = format_datetime_for_api(dt, timezone="Asia/Seoul")
        assert result == "2026-01-25T11:30:00+09:00"

    def test_naive_utc(self):
        dt = datetime(2026, 1, 25, 11, 30)
        result = format_datetime_for_api(dt, timezone="UTC")
        assert result == "2026-01-25T11:30:00+00:00"

    def test_naive_us_eastern(self):
        dt = datetime(2026, 1, 25, 14, 30)
        result = format_datetime_for_api(dt, timezone="America/New_York")
        assert "+05:00" not in result  # Not +05, should be -05 in January
        assert "-05:00" in result

    def test_invalid_timezone_appends_z(self):
        dt = datetime(2026, 1, 25, 11, 30)
        result = format_datetime_for_api(dt, timezone="InvalidTZ")
        assert result.endswith("Z")

    def test_utc_aware_uses_plus_zero(self):
        """ZoneInfo('UTC') produces +00:00, not Z."""
        tz = zoneinfo.ZoneInfo("UTC")
        dt = datetime(2026, 1, 25, 11, 30, tzinfo=tz)
        result = format_datetime_for_api(dt)
        assert "+00:00" in result

    # --- Full pipeline integration ---

    def test_pipeline_kst_event(self):
        """11:30 AM KST → parse → format → correct API string."""
        dt = parse_datetime("2026-01-25", "11:30 AM KST", timezone="UTC")
        end_dt = dt + timedelta(minutes=150)
        start_str = format_datetime_for_api(dt)
        end_str = format_datetime_for_api(end_dt)
        assert start_str == "2026-01-25T11:30:00+09:00"
        assert end_str == "2026-01-25T14:00:00+09:00"

    def test_pipeline_pst_event(self):
        """3:00 PM PST → correct -08:00 offset."""
        dt = parse_datetime("2026-01-25", "3:00 PM PST", timezone="UTC")
        result = format_datetime_for_api(dt)
        assert result == "2026-01-25T15:00:00-08:00"

    def test_pipeline_no_tz_uses_config(self):
        """Plain '2:00 PM' with config Asia/Seoul → +09:00."""
        dt = parse_datetime("2026-01-25", "2:00 PM", timezone="Asia/Seoul")
        result = format_datetime_for_api(dt)
        assert result == "2026-01-25T14:00:00+09:00"
