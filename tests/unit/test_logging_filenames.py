"""Tests for log filename generation, parsing, and display formatting."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from sentinel.logging import generate_log_filename, parse_log_filename, parse_log_filename_parts


class TestGenerateLogFilename:
    """Tests for generate_log_filename."""

    def test_generates_filename_with_issue_key(self) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = generate_log_filename(ts, issue_key="PROJ-123")
        assert result == "PROJ-123_20250115-103045_a1.log"

    def test_generates_filename_without_issue_key(self) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = generate_log_filename(ts)
        assert result == "20250115-103045_a1.log"

    def test_generates_filename_with_custom_attempt(self) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = generate_log_filename(ts, issue_key="TEST-1", attempt=3)
        assert result == "TEST-1_20250115-103045_a3.log"

    def test_default_attempt_is_one(self) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 45, tzinfo=UTC)
        result = generate_log_filename(ts, issue_key="TEST-1")
        assert "_a1.log" in result

    def test_handles_issue_keys_with_hyphens(self) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = generate_log_filename(ts, issue_key="MY-PROJECT-42")
        assert result == "MY-PROJECT-42_20250601-120000_a1.log"

    def test_handles_issue_keys_with_underscores(self) -> None:
        """Issue keys with underscores should be handled correctly.

        Although standard Jira keys use hyphens (e.g. PROJ-123), the
        parser relies on rsplit('_', 1) to separate the issue key from
        the timestamp, so underscores in the key are safe as long as the
        timestamp format uses hyphens internally.
        """
        ts = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        result = generate_log_filename(ts, issue_key="MY_PROJ-42")
        assert result == "MY_PROJ-42_20250601-120000_a1.log"

    def test_none_issue_key_same_as_no_key(self) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 45, tzinfo=UTC)
        assert generate_log_filename(ts, issue_key=None) == generate_log_filename(ts)


class TestParseLogFilename:
    """Tests for parse_log_filename."""

    def test_parses_new_format_with_issue_key(self) -> None:
        result = parse_log_filename("PROJ-123_20250115-103045_a1.log")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45

    def test_parses_new_format_without_issue_key(self) -> None:
        result = parse_log_filename("20250115-103045_a1.log")
        assert result is not None
        assert result.year == 2025

    def test_parses_legacy_format(self) -> None:
        result = parse_log_filename("20250115_103045.log")
        assert result is not None
        assert result.year == 2025

    def test_returns_none_for_invalid_filename(self) -> None:
        assert parse_log_filename("not-a-log-file.log") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_log_filename("") is None

    def test_returns_none_for_random_text(self) -> None:
        assert parse_log_filename("random_text") is None

    def test_handles_issue_keys_with_hyphens(self) -> None:
        result = parse_log_filename("MY-PROJECT-42_20250601-120000_a1.log")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6

    def test_filename_without_log_extension(self) -> None:
        """Filenames without .log extension still parse when the format matches.

        removesuffix is a no-op here, but the timestamp portion is still
        parseable because the _a{N} suffix is stripped first.
        """
        result = parse_log_filename("PROJ-123_20250115-103045_a1")
        assert result is not None
        assert result.year == 2025


class TestParseLogFilenameParts:
    """Tests for parse_log_filename_parts."""

    def test_parses_full_new_format(self) -> None:
        result = parse_log_filename_parts("PROJ-123_20250115-103045_a2.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "PROJ-123"
        assert ts.year == 2025
        assert ts.month == 1
        assert ts.day == 15
        assert ts.hour == 10
        assert ts.minute == 30
        assert ts.second == 45
        assert attempt == 2

    def test_parses_format_without_issue_key(self) -> None:
        result = parse_log_filename_parts("20250115-103045_a1.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key is None
        assert ts.year == 2025
        assert attempt == 1

    def test_parses_legacy_format(self) -> None:
        result = parse_log_filename_parts("20250115_103045.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key is None
        assert ts.year == 2025
        assert attempt == 1  # Legacy defaults to attempt 1

    def test_handles_issue_keys_with_hyphens(self) -> None:
        result = parse_log_filename_parts("MY-PROJECT-42_20250601-120000_a3.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "MY-PROJECT-42"
        assert attempt == 3

    def test_handles_issue_keys_with_underscores(self) -> None:
        """Issue keys containing underscores are correctly separated from the timestamp.

        The parser uses rsplit('_', 1) which splits on the *last* underscore,
        so underscores within the issue key portion are preserved.
        """
        result = parse_log_filename_parts("MY_PROJ-42_20250601-120000_a1.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "MY_PROJ-42"
        assert ts.month == 6
        assert attempt == 1

    def test_attempt_number_zero(self) -> None:
        """Attempt number 0 should be parsed correctly even though convention is 1-based."""
        result = parse_log_filename_parts("TEST-1_20250115-103045_a0.log")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "TEST-1"
        assert attempt == 0

    def test_large_attempt_number(self) -> None:
        result = parse_log_filename_parts("TEST-1_20250115-103045_a99.log")
        assert result is not None
        _, _, attempt = result
        assert attempt == 99

    def test_returns_none_for_invalid_filename(self) -> None:
        assert parse_log_filename_parts("not-a-log-file.log") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert parse_log_filename_parts("") is None

    def test_filename_without_log_extension(self) -> None:
        """Filenames without .log extension still parse when the format matches.

        removesuffix is a no-op here, but the timestamp portion is still
        parseable because the _a{N} suffix is stripped first.
        """
        result = parse_log_filename_parts("PROJ-123_20250115-103045_a1")
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "PROJ-123"
        assert attempt == 1

    def test_timestamp_has_utc_timezone(self) -> None:
        result = parse_log_filename_parts("TEST-1_20250115-103045_a1.log")
        assert result is not None
        _, ts, _ = result
        assert ts.tzinfo == UTC

    def test_legacy_timestamp_has_utc_timezone(self) -> None:
        result = parse_log_filename_parts("20250115_103045.log")
        assert result is not None
        _, ts, _ = result
        assert ts.tzinfo == UTC

    def test_roundtrip_with_generate(self) -> None:
        """Verify that generate -> parse_parts roundtrip preserves data."""
        original_ts = datetime(2025, 3, 20, 14, 30, 0, tzinfo=UTC)
        filename = generate_log_filename(
            original_ts, issue_key="DS-976", attempt=2,
        )
        result = parse_log_filename_parts(filename)
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key == "DS-976"
        assert ts == original_ts
        assert attempt == 2

    def test_roundtrip_without_issue_key(self) -> None:
        """Verify roundtrip without issue key."""
        original_ts = datetime(2025, 3, 20, 14, 30, 0, tzinfo=UTC)
        filename = generate_log_filename(original_ts, attempt=5)
        result = parse_log_filename_parts(filename)
        assert result is not None
        issue_key, ts, attempt = result
        assert issue_key is None
        assert ts == original_ts
        assert attempt == 5


class TestFormatLogDisplayName:
    """Tests for SentinelStateAccessor._format_log_display_name.

    Uses a minimal mock setup to exercise the display-name formatting logic
    without requiring a full Sentinel application instance.
    """

    @pytest.fixture()
    def state_accessor(self) -> MagicMock:
        """Create a mock with _format_log_display_name bound from the real class."""
        from sentinel.dashboard.state import SentinelStateAccessor

        mock = MagicMock(spec=SentinelStateAccessor)
        mock._format_log_display_name = (
            SentinelStateAccessor._format_log_display_name.__get__(
                mock, SentinelStateAccessor,
            )
        )
        return mock

    def test_new_format_with_issue_key(self, state_accessor: MagicMock) -> None:
        result = state_accessor._format_log_display_name(
            "PROJ-123_20250115-103045_a1.log",
        )
        assert result == "PROJ-123 2025-01-15 10:30:45 (attempt 1)"

    def test_new_format_without_issue_key(self, state_accessor: MagicMock) -> None:
        result = state_accessor._format_log_display_name("20250115-103045_a2.log")
        assert result == "2025-01-15 10:30:45 (attempt 2)"

    def test_legacy_format(self, state_accessor: MagicMock) -> None:
        result = state_accessor._format_log_display_name("20250115_103045.log")
        assert result == "2025-01-15 10:30:45"

    def test_unrecognized_filename_returned_as_is(
        self, state_accessor: MagicMock,
    ) -> None:
        result = state_accessor._format_log_display_name("unknown-file.txt")
        assert result == "unknown-file.txt"

    def test_multi_hyphen_issue_key(self, state_accessor: MagicMock) -> None:
        result = state_accessor._format_log_display_name(
            "MY-PROJECT-42_20250601-120000_a3.log",
        )
        assert result == "MY-PROJECT-42 2025-06-01 12:00:00 (attempt 3)"
