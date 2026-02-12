"""Tests for log filename generation and parsing (DS-966)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from sentinel.config import Config as SentinelConfig
from sentinel.dashboard.state import SentinelStateAccessor, SentinelStateProvider
from sentinel.logging import (
    LOG_FILENAME_EXTENSION,
    LogFilenameParts,
    _parse_log_timestamp,
    _strip_log_extension,
    generate_log_filename,
    parse_log_filename,
    parse_log_filename_parts,
)


class TestGenerateLogFilename:
    """Tests for generate_log_filename function."""

    def test_generates_filename_with_issue_key(self) -> None:
        """Should generate filename with issue key in correct format."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(timestamp, issue_key="DS-123", attempt=1)
        assert filename == "DS-123_20240115-103045_a1.log"

    def test_generates_filename_without_issue_key(self) -> None:
        """Should generate filename without issue key when not provided."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(timestamp, issue_key=None, attempt=1)
        assert filename == "20240115-103045_a1.log"

    def test_generates_filename_with_different_attempt_numbers(self) -> None:
        """Should generate filename with different attempt numbers."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

        filename1 = generate_log_filename(timestamp, issue_key="DS-123", attempt=1)
        filename2 = generate_log_filename(timestamp, issue_key="DS-123", attempt=2)
        filename3 = generate_log_filename(timestamp, issue_key="DS-123", attempt=10)

        assert filename1 == "DS-123_20240115-103045_a1.log"
        assert filename2 == "DS-123_20240115-103045_a2.log"
        assert filename3 == "DS-123_20240115-103045_a10.log"


class TestParseLogFilename:
    """Tests for parse_log_filename function."""

    def test_round_trip_with_issue_key(self) -> None:
        """Generate a filename and parse it back, timestamp should match."""
        original_timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(
            original_timestamp, issue_key="DS-123", attempt=1
        )

        parsed_timestamp = parse_log_filename(filename)

        assert parsed_timestamp is not None
        assert parsed_timestamp == original_timestamp

    def test_round_trip_without_issue_key(self) -> None:
        """Generate a filename without issue key and parse it back."""
        original_timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(original_timestamp, issue_key=None, attempt=1)

        parsed_timestamp = parse_log_filename(filename)

        assert parsed_timestamp is not None
        assert parsed_timestamp == original_timestamp

    def test_parses_new_format_with_issue_key(self) -> None:
        """Should parse new format with issue key correctly."""
        filename = "DS-123_20240115-103045_a1.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_parses_new_format_without_issue_key(self) -> None:
        """Should parse new format without issue key correctly."""
        filename = "20240115-103045_a2.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_parses_new_format_with_different_attempts(self) -> None:
        """Should parse new format with various attempt numbers."""
        filename1 = "DS-123_20240115-103045_a1.log"
        filename2 = "DS-123_20240115-103045_a5.log"
        filename3 = "DS-123_20240115-103045_a100.log"

        parsed1 = parse_log_filename(filename1)
        parsed2 = parse_log_filename(filename2)
        parsed3 = parse_log_filename(filename3)

        expected = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        assert parsed1 == expected
        assert parsed2 == expected
        assert parsed3 == expected

    def test_parses_legacy_format(self) -> None:
        """Should parse legacy format YYYYMMDD_HHMMSS.log correctly."""
        filename = "20240115_103045.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    @pytest.mark.parametrize(
        "filename",
        [
            pytest.param("not-a-log-file.txt", id="wrong-extension"),
            pytest.param("invalid_format.log", id="no-timestamp"),
            pytest.param("20240115.log", id="date-only-no-time"),
            pytest.param("DS-123_invalid_a1.log", id="non-numeric-timestamp"),
            pytest.param("random.log", id="arbitrary-name"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_returns_none_for_invalid_filename(self, filename: str) -> None:
        """Should return None for filenames that don't match any expected format."""
        parsed = parse_log_filename(filename)
        assert parsed is None

    def test_handles_issue_keys_with_hyphens(self) -> None:
        """Should handle issue keys that contain hyphens (normal case)."""
        filename = "PROJ-123_20240115-103045_a1.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_handles_issue_keys_with_underscores(self) -> None:
        """Should handle issue keys that contain underscores via greedy regex."""
        filename = "MY_PROJ-123_20240115-103045_a1.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_new_format_timestamp_without_attempt_suffix_returns_none(self) -> None:
        """Filename with valid new-format timestamp but missing _a{N} suffix.

        Verifies that ``parse_log_filename`` returns ``None`` for a filename
        like ``DS-123_20240115-103045.log`` which has a new-style timestamp
        (dash-separated) but lacks the required attempt suffix.  This filename
        does **not** match the legacy format either (which expects
        ``YYYYMMDD_HHMMSS.log``), so the result should be ``None``.
        """
        filename = "DS-123_20240115-103045.log"
        parsed = parse_log_filename(filename)

        assert parsed is None

    def test_missing_attempt_suffix_parts_none_and_no_legacy_fallback(self) -> None:
        """Verify parse_log_filename_parts returns None while parse_log_filename also returns None.

        DS-985 follow-up: when a filename has a valid new-format timestamp
        (dash-separated) but is missing the ``_a{N}`` suffix,
        ``parse_log_filename_parts`` returns ``None`` and
        ``parse_log_filename`` falls through to legacy parsing.  Because
        the dash-separated timestamp (``20240115-103045``) does not match
        the legacy ``YYYYMMDD_HHMMSS`` format (underscore-separated),
        ``parse_log_filename`` also returns ``None``.

        This test makes the interplay between both functions explicit for
        a filename **without** an issue key prefix.
        """
        filename = "20240115-103045.log"

        # parse_log_filename_parts should not match (no _a{N} suffix)
        parts = parse_log_filename_parts(filename)
        assert parts is None

        # parse_log_filename should also return None: the dash-separated
        # timestamp does not match the legacy underscore-separated format
        parsed_dt = parse_log_filename(filename)
        assert parsed_dt is None


class TestParseLogFilenameParts:
    """Tests for parse_log_filename_parts function.

    Directly tests the public API contract: each call returns a
    ``LogFilenameParts`` named tuple for valid new-format filenames, or
    ``None`` for unrecognised formats.  This strengthens the contract
    for any future consumers of the function beyond
    ``_format_log_display_name``.
    """

    def test_parses_filename_with_issue_key(self) -> None:
        """Should return LogFilenameParts with issue_key for new format with issue key."""
        result = parse_log_filename_parts("DS-123_20240115-103045_a1.log")

        assert result is not None
        assert isinstance(result, LogFilenameParts)
        assert result.issue_key == "DS-123"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 1

    def test_handles_issue_keys_with_hyphens(self) -> None:
        result = parse_log_filename_parts("MY-PROJECT-42_20250601-120000_a3.log")
        assert result is not None
        assert isinstance(result, LogFilenameParts)
        assert result.issue_key == "MY-PROJECT-42"
        assert result.timestamp == "20250601-120000"
        assert result.attempt == 3

    def test_parses_filename_without_issue_key(self) -> None:
        """Should return LogFilenameParts with None issue_key for format without issue key."""
        result = parse_log_filename_parts("20240115-103045_a2.log")

        assert result is not None
        assert isinstance(result, LogFilenameParts)
        assert result.issue_key is None
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 2

    def test_parses_multi_digit_attempt(self) -> None:
        """Should correctly parse multi-digit attempt numbers."""
        result = parse_log_filename_parts("DS-456_20240115-103045_a100.log")

        assert result is not None
        assert isinstance(result, LogFilenameParts)
        assert result.issue_key == "DS-456"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 100

    def test_returns_none_for_legacy_format(self) -> None:
        """Should return None for legacy format (no attempt suffix)."""
        result = parse_log_filename_parts("20240115_103045.log")

        assert result is None

    @pytest.mark.parametrize(
        "filename",
        [
            pytest.param("not-a-log-file.txt", id="wrong-extension"),
            pytest.param("invalid_format.log", id="no-timestamp"),
            pytest.param("20240115.log", id="date-only-no-time"),
            pytest.param("DS-123_invalid_a1.log", id="non-numeric-timestamp"),
            pytest.param("random.log", id="arbitrary-name"),
            pytest.param("", id="empty-string"),
        ],
    )
    def test_returns_none_for_invalid_filename(self, filename: str) -> None:
        """Should return None for filenames that don't match new format."""
        result = parse_log_filename_parts(filename)
        assert result is None

    def test_round_trip_with_generate(self) -> None:
        """Should decompose a generated filename back into its parts."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(timestamp, issue_key="PROJ-789", attempt=3)

        result = parse_log_filename_parts(filename)

        assert result is not None
        assert result.issue_key == "PROJ-789"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 3

    def test_handles_issue_keys_with_underscores(self) -> None:
        """Should correctly parse issue keys containing underscores via greedy regex."""
        result = parse_log_filename_parts("MY_PROJ-123_20240115-103045_a1.log")

        assert result is not None
        assert result.issue_key == "MY_PROJ-123"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 1

    def test_timestamp_and_attempt_always_non_none_on_match(self) -> None:
        """When a match is found, timestamp and attempt should always be non-None."""
        result = parse_log_filename_parts("20240115-103045_a1.log")

        assert result is not None
        # Named attributes from LogFilenameParts are always non-None when a match is found
        assert isinstance(result.timestamp, str)
        assert isinstance(result.attempt, int)
        assert len(result.timestamp) > 0
        assert result.attempt >= 1

    def test_returns_none_for_missing_attempt_suffix(self) -> None:
        """Filename with valid new-format timestamp but missing _a{N} suffix.

        ``parse_log_filename_parts`` should return ``None`` when the attempt
        suffix is absent, even though the timestamp portion looks valid.
        """
        result = parse_log_filename_parts("DS-123_20240115-103045.log")

        assert result is None

    def test_returns_none_for_bare_timestamp_without_attempt_suffix(self) -> None:
        """Bare timestamp without issue key or _a{N} suffix returns None.

        DS-985 follow-up edge case: a filename containing only a valid
        new-format timestamp (dash-separated ``YYYYMMDD-HHMMSS``) but
        lacking the ``_a{N}`` attempt suffix should return ``None``.
        This complements ``test_returns_none_for_missing_attempt_suffix``
        which tests the same scenario *with* an issue key prefix.
        """
        result = parse_log_filename_parts("20240115-103045.log")

        assert result is None

    def test_handles_issue_keys_with_multiple_underscores(self) -> None:
        """Issue keys with multiple underscores are preserved by rsplit('_', 1).

        The greedy rsplit on the last underscore before the timestamp means
        *all* leading underscores in the issue-key portion are kept intact.
        This provides regression protection for the greedy capture behavior.
        """
        result = parse_log_filename_parts("MY_PROJ_V2-123_20240115-103045_a1.log")
        assert result is not None
        assert result.issue_key == "MY_PROJ_V2-123"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 1

    def test_roundtrip_issue_key_with_underscores(self) -> None:
        """Roundtrip generate -> parse preserves issue keys containing underscores."""
        original_ts = datetime(2025, 7, 4, 18, 0, 0, tzinfo=UTC)
        filename = generate_log_filename(
            original_ts, issue_key="MY_PROJ-123", attempt=2,
        )
        result = parse_log_filename_parts(filename)
        assert result is not None
        assert result.issue_key == "MY_PROJ-123"
        assert result.to_datetime() == original_ts
        assert result.attempt == 2

    def test_roundtrip_issue_key_with_multiple_underscores(self) -> None:
        """Roundtrip generate -> parse preserves multi-underscore issue keys.

        Addresses PR #993 review comment: extends roundtrip coverage to
        multi-underscore keys (e.g. MY_PROJ_V2-123) beyond the single-
        underscore case already covered by test_roundtrip_issue_key_with_underscores.
        """
        original_ts = datetime(2026, 3, 15, 9, 0, 0, tzinfo=UTC)
        filename = generate_log_filename(
            original_ts, issue_key="MY_PROJ_V2-123", attempt=1,
        )
        result = parse_log_filename_parts(filename)
        assert result is not None
        assert result.issue_key == "MY_PROJ_V2-123"
        assert result.to_datetime() == original_ts
        assert result.attempt == 1

    def test_roundtrip_issue_key_with_triple_underscore_segments(self) -> None:
        """Roundtrip generate -> parse preserves triple-underscore issue keys.

        DS-996 follow-up: extends roundtrip coverage to triple-underscore
        keys (e.g. FOO_BAR_BAZ-99) beyond the double-underscore case
        already covered by test_roundtrip_issue_key_with_multiple_underscores.
        Complements the non-roundtrip test_issue_key_triple_underscore_segments
        which validates direct parsing only.
        """
        original_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        filename = generate_log_filename(
            original_ts, issue_key="FOO_BAR_BAZ-99", attempt=7,
        )
        result = parse_log_filename_parts(filename)
        assert result is not None
        assert result.issue_key == "FOO_BAR_BAZ-99"
        assert result.to_datetime() == original_ts
        assert result.attempt == 7

    def test_attempt_number_zero(self) -> None:
        """Attempt number 0 should be parsed correctly even though convention is 1-based."""
        result = parse_log_filename_parts("TEST-1_20250115-103045_a0.log")
        assert result is not None
        assert result.issue_key == "TEST-1"
        assert result.attempt == 0

    def test_parses_filename_without_log_extension(self) -> None:
        """Filename without .log extension should still match via removesuffix no-op.

        ``removesuffix`` is a no-op when the suffix is absent, so the regex
        still matches the bare stem.  This verifies that the function does not
        require the ``.log`` extension to produce a result.

        DS-989 follow-up: this is confirmed as desired behaviour.
        ``parse_log_filename_parts`` intentionally does *not* validate the
        file extension — it focuses solely on the stem format.  Extension
        validation is the caller's responsibility (e.g. ``get_log_files``
        uses ``step_dir.glob("*.log")`` to pre-filter).  Adding extension
        enforcement here would break callers that strip the extension before
        calling this function.
        """
        result = parse_log_filename_parts("DS-123_20240115-103045_a1")

        assert result is not None
        assert result.issue_key == "DS-123"
        assert result.timestamp == "20240115-103045"
        assert result.attempt == 1

    @pytest.mark.parametrize(
        ("filename", "expected_issue_key", "expected_timestamp", "expected_attempt"),
        [
            pytest.param(
                "DS-123_20240115-103045_a1",
                "DS-123",
                "20240115-103045",
                1,
                id="with-issue-key",
            ),
            pytest.param(
                "20240115-103045_a2",
                None,
                "20240115-103045",
                2,
                id="without-issue-key",
            ),
            pytest.param(
                "MY_PROJ-99_20260101-000000_a5",
                "MY_PROJ-99",
                "20260101-000000",
                5,
                id="underscore-issue-key",
            ),
            pytest.param(
                "FOO_BAR_BAZ-7_20250601-120000_a10",
                "FOO_BAR_BAZ-7",
                "20250601-120000",
                10,
                id="multi-underscore-issue-key",
            ),
        ],
    )
    def test_parses_filename_without_log_extension_parametrized(
        self,
        filename: str,
        expected_issue_key: str | None,
        expected_timestamp: str,
        expected_attempt: int,
    ) -> None:
        """Filenames without .log extension should still match via removesuffix no-op.

        Extends the single-case test above to cover multiple filename shapes —
        with issue key, without issue key, and with underscore-containing issue
        keys — verifying that ``removesuffix`` being a no-op does not break
        any of the regex capture groups.  (DS-988)
        """
        result = parse_log_filename_parts(filename)

        assert result is not None
        assert result.issue_key == expected_issue_key
        assert result.timestamp == expected_timestamp
        assert result.attempt == expected_attempt

    @pytest.mark.parametrize(
        "stem",
        [
            pytest.param("DS-123_20240115-103045_a1", id="with-issue-key"),
            pytest.param("20240115-103045_a2", id="without-issue-key"),
            pytest.param("MY_PROJ-99_20260101-000000_a5", id="underscore-issue-key"),
        ],
    )
    def test_no_log_extension_matches_with_extension(self, stem: str) -> None:
        """Parsing with and without .log extension should yield identical results.

        This ensures that ``removesuffix('.log')`` is the *only* difference
        between the two code paths, and that the resulting ``LogFilenameParts``
        are field-for-field identical.  (DS-988)
        """
        with_ext = parse_log_filename_parts(f"{stem}.log")
        without_ext = parse_log_filename_parts(stem)

        assert with_ext is not None, f"Expected match for {stem}.log"
        assert without_ext is not None, f"Expected match for {stem}"
        assert with_ext == without_ext, (
            f"Results differ for '{stem}': with_ext={with_ext}, without_ext={without_ext}"
        )

    def test_to_datetime_returns_correct_utc_datetime(self) -> None:
        """LogFilenameParts.to_datetime() should return the correct UTC datetime."""
        result = parse_log_filename_parts("DS-123_20240115-103045_a1.log")

        assert result is not None
        dt = result.to_datetime()
        assert dt == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_to_datetime_raises_value_error_for_invalid_timestamp(self) -> None:
        """LogFilenameParts.to_datetime() should raise ValueError for an invalid timestamp.

        Since NamedTuple does not enforce field types at runtime, a
        LogFilenameParts constructed directly (not via parse_log_filename_parts)
        could contain a malformed timestamp string. Verify that the error
        propagates cleanly.  (DS-986: defensive edge case.)
        """
        bad_parts = LogFilenameParts(issue_key="DS-999", timestamp="not-a-date", attempt=1)
        with pytest.raises(ValueError):
            bad_parts.to_datetime()

    def test_to_datetime_raises_type_error_for_none_timestamp(self) -> None:
        """LogFilenameParts.to_datetime() should raise TypeError for a None timestamp.

        Since NamedTuple does not enforce field types at runtime, a
        LogFilenameParts constructed directly (not via parse_log_filename_parts)
        could carry None as the timestamp.  Passing None to strptime raises
        TypeError rather than ValueError.  (DS-999: complement to the existing
        ValueError test for malformed timestamp strings.)
        """
        none_parts = LogFilenameParts(issue_key="DS-999", timestamp=None, attempt=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            none_parts.to_datetime()

    def test_direct_construction_does_not_enforce_attempt_type(self) -> None:
        """NamedTuple does not enforce ``attempt`` as int at construction time.

        Callers constructing LogFilenameParts directly (bypassing
        parse_log_filename_parts) can accidentally pass a ``str``
        for ``attempt``.  This test documents the lack of runtime
        type enforcement and validates that the contract documented
        in the class docstring (DS-986) accurately describes the
        behaviour.
        """
        # Deliberately pass a str where int is annotated — NamedTuple allows it
        parts = LogFilenameParts(issue_key="DS-999", timestamp="20240115-103045", attempt="3")  # type: ignore[arg-type]
        # The field stores whatever was passed, without int conversion
        assert parts.attempt == "3"  # type: ignore[comparison-overlap]
        assert not isinstance(parts.attempt, int)

    # ------------------------------------------------------------------
    # Consistency: parse_log_filename and parse_log_filename_parts agree
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "filename",
        [
            pytest.param(
                "PROJ-123_20250115-103045_a1.log",
                id="new-format-with-issue-key",
            ),
            pytest.param(
                "20250601-120000_a2.log",
                id="new-format-without-issue-key",
            ),
            pytest.param(
                "MY_PROJ-42_20260301-090500_a3.log",
                id="underscore-issue-key",
            ),
        ],
    )
    def test_timestamp_matches_parse_log_filename(self, filename: str) -> None:
        """Timestamp from parse_log_filename_parts matches parse_log_filename.

        Both functions must agree on the extracted datetime for the same
        input, ensuring that callers migrating from the datetime-only API
        see identical results.

        Note: only new-format filenames are tested here because
        ``parse_log_filename_parts`` returns ``None`` for legacy-format
        filenames (``YYYYMMDD_HHMMSS.log``), so there is no ``to_datetime()``
        result to compare.
        """
        parts_result = parse_log_filename_parts(filename)
        dt_result = parse_log_filename(filename)
        assert parts_result is not None, f"parse_log_filename_parts returned None for {filename}"
        assert dt_result is not None, f"parse_log_filename returned None for {filename}"
        assert parts_result.to_datetime() == dt_result, (
            f"Timestamp mismatch for {filename}: "
            f"parts={parts_result.to_datetime()}, parse={dt_result}"
        )

    # ------------------------------------------------------------------
    # Edge cases: unusual issue key formats (DS-981)
    # ------------------------------------------------------------------

    def test_issue_key_single_underscore_prefix(self) -> None:
        """Issue key with a leading underscore-separated segment.

        Validates that the greedy rsplit correctly isolates the timestamp
        even when the issue key itself contains underscores.
        """
        result = parse_log_filename_parts("A_B-1_20250601-120000_a1.log")
        assert result is not None
        assert result.issue_key == "A_B-1"
        assert result.to_datetime() == datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
        assert result.attempt == 1

    def test_issue_key_triple_underscore_segments(self) -> None:
        """Issue key with three underscore-separated segments.

        Stress-tests that rsplit('_', 1) only splits on the rightmost
        underscore, preserving the entire multi-underscore prefix.
        """
        result = parse_log_filename_parts("FOO_BAR_BAZ-99_20260101-000000_a7.log")
        assert result is not None
        assert result.issue_key == "FOO_BAR_BAZ-99"
        assert result.to_datetime() == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert result.attempt == 7

    # ------------------------------------------------------------------
    # Guarantee: non-None timestamp and attempt on successful parse
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "filename",
        [
            "PROJ-1_20250115-103045_a1.log",
            "MY_PROJ-42_20250601-120000_a3.log",
            "MY_PROJ_V2-123_20240115-103045_a1.log",
            "20260301-090500_a5.log",
        ],
    )
    def test_timestamp_and_attempt_always_non_none(self, filename: str) -> None:
        """Returned timestamp and attempt are always non-None on a successful parse.

        This explicitly asserts the type-precision guarantee documented in
        the docstring: only *issue_key* may be None; *timestamp* (str)
        and *attempt* (int) are always present when the function returns a
        LogFilenameParts rather than None.  The to_datetime() convenience
        method always yields a timezone-aware UTC datetime.
        """
        result = parse_log_filename_parts(filename)
        assert result is not None, f"Expected a match for {filename}"
        assert isinstance(result, LogFilenameParts), f"Expected LogFilenameParts for {filename}"
        assert isinstance(result.timestamp, str), f"Expected str, got {type(result.timestamp)} for {filename}"
        assert isinstance(result.attempt, int), f"Expected int, got {type(result.attempt)} for {filename}"
        dt = result.to_datetime()
        assert isinstance(dt, datetime), f"Expected datetime from to_datetime(), got {type(dt)} for {filename}"
        assert dt.tzinfo is not None, f"Expected timezone-aware datetime for {filename}"


class TestLogFileNamePartsStr:
    """Tests for LogFilenameParts.__str__ convenience method (DS-989)."""

    def test_str_with_issue_key(self) -> None:
        """__str__ should reconstruct filename stem with issue key."""
        parts = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        assert parts is not None
        assert str(parts) == "DS-123_20240115-103045_a1"

    def test_str_without_issue_key(self) -> None:
        """__str__ should reconstruct filename stem without issue key."""
        parts = parse_log_filename_parts("20240115-103045_a2.log")
        assert parts is not None
        assert str(parts) == "20240115-103045_a2"

    def test_str_with_underscore_issue_key(self) -> None:
        """__str__ should preserve underscore-containing issue keys."""
        parts = parse_log_filename_parts("MY_PROJ-123_20240115-103045_a1.log")
        assert parts is not None
        assert str(parts) == "MY_PROJ-123_20240115-103045_a1"

    def test_str_with_multi_underscore_issue_key(self) -> None:
        """__str__ should preserve multi-underscore issue keys."""
        parts = parse_log_filename_parts("MY_PROJ_V2-123_20240115-103045_a1.log")
        assert parts is not None
        assert str(parts) == "MY_PROJ_V2-123_20240115-103045_a1"

    def test_str_with_multi_digit_attempt(self) -> None:
        """__str__ should correctly represent multi-digit attempt numbers."""
        parts = parse_log_filename_parts("DS-456_20240115-103045_a100.log")
        assert parts is not None
        assert str(parts) == "DS-456_20240115-103045_a100"

    def test_str_roundtrip_with_generate(self) -> None:
        """__str__ output plus .log extension should match generate_log_filename."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(timestamp, issue_key="PROJ-789", attempt=3)
        parts = parse_log_filename_parts(filename)

        assert parts is not None
        assert str(parts) + ".log" == filename

    def test_str_roundtrip_without_issue_key(self) -> None:
        """__str__ output plus .log extension should match generate for no-issue-key case."""
        timestamp = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        filename = generate_log_filename(timestamp, issue_key=None, attempt=1)
        parts = parse_log_filename_parts(filename)

        assert parts is not None
        assert str(parts) + ".log" == filename

    def test_str_direct_construction(self) -> None:
        """__str__ should work with directly constructed LogFilenameParts."""
        parts = LogFilenameParts(issue_key="DS-999", timestamp="20240115-103045", attempt=1)
        assert str(parts) == "DS-999_20240115-103045_a1"

    def test_str_direct_construction_no_issue_key(self) -> None:
        """__str__ should work with directly constructed LogFilenameParts without issue key."""
        parts = LogFilenameParts(issue_key=None, timestamp="20240115-103045", attempt=1)
        assert str(parts) == "20240115-103045_a1"

    def test_str_explicit_none_check_empty_string_issue_key(self) -> None:
        """__str__ should include empty-string issue_key prefix since it is not None.

        DS-1006: verifies that the explicit ``is not None`` check in
        ``__str__`` distinguishes between ``None`` (omit prefix) and an
        empty string (include prefix).  While ``parse_log_filename_parts``
        never produces an empty-string ``issue_key``, directly constructed
        instances could carry one — the ``is not None`` guard preserves
        the value rather than silently dropping it via truthiness.
        """
        parts = LogFilenameParts(issue_key="", timestamp="20240115-103045", attempt=1)
        assert str(parts) == "_20240115-103045_a1"

    def test_str_single_underscore_issue_key_display_name_pattern(self) -> None:
        """__str__ reconstructs single-underscore issue key stem per DS-989 requirement.

        DS-1006 item #2: verifies the exact pattern originally requested by
        DS-989 ticket item #1 — ``MY_PROJ-123_20240115-103045_a1`` — is
        correctly reconstructed by ``__str__``.  This complements the
        ``TestFormatLogDisplayName.test_formats_underscore_issue_key`` test
        (DS-1002) that validates the same pattern at the display formatting
        layer.
        """
        parts = parse_log_filename_parts("MY_PROJ-123_20240115-103045_a1.log")
        assert parts is not None
        assert str(parts) == "MY_PROJ-123_20240115-103045_a1"


class TestLogFileNamePartsRepr:
    """Tests for LogFilenameParts.__repr__ debugging convenience (DS-994)."""

    def test_repr_with_issue_key(self) -> None:
        """__repr__ should show all fields with type-clear formatting."""
        parts = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        assert parts is not None
        assert repr(parts) == (
            "LogFilenameParts(issue_key='DS-123', "
            "timestamp='20240115-103045', attempt=1)"
        )

    def test_repr_without_issue_key(self) -> None:
        """__repr__ should show None for missing issue_key."""
        parts = parse_log_filename_parts("20240115-103045_a2.log")
        assert parts is not None
        assert repr(parts) == (
            "LogFilenameParts(issue_key=None, "
            "timestamp='20240115-103045', attempt=2)"
        )

    def test_repr_with_underscore_issue_key(self) -> None:
        """__repr__ should preserve underscore-containing issue keys."""
        parts = parse_log_filename_parts("MY_PROJ-123_20240115-103045_a1.log")
        assert parts is not None
        assert repr(parts) == (
            "LogFilenameParts(issue_key='MY_PROJ-123', "
            "timestamp='20240115-103045', attempt=1)"
        )

    def test_repr_direct_construction(self) -> None:
        """__repr__ should work with directly constructed LogFilenameParts."""
        parts = LogFilenameParts(issue_key="DS-999", timestamp="20240115-103045", attempt=1)
        assert repr(parts) == (
            "LogFilenameParts(issue_key='DS-999', "
            "timestamp='20240115-103045', attempt=1)"
        )

    def test_repr_differs_from_str(self) -> None:
        """__repr__ should differ from __str__ (str reconstructs filename stem)."""
        parts = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        assert parts is not None
        assert repr(parts) != str(parts)
        # __str__ gives filename stem, __repr__ gives LogFilenameParts(...)
        assert str(parts) == "DS-123_20240115-103045_a1"
        assert "LogFilenameParts(" in repr(parts)


class TestLogFileNamePartsToDatetimeCache:
    """Tests for LogFilenameParts.to_datetime caching via _parse_log_timestamp (DS-994)."""

    def test_to_datetime_returns_same_object_on_repeated_calls(self) -> None:
        """Module-level cached _parse_log_timestamp returns the same object for the same timestamp."""
        parts = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        assert parts is not None

        dt1 = parts.to_datetime()
        dt2 = parts.to_datetime()

        # Should be the same object (cached at module level), not just equal
        assert dt1 is dt2

    def test_to_datetime_cache_returns_correct_value(self) -> None:
        """Cached to_datetime should still return the correct datetime value."""
        parts = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        assert parts is not None

        dt = parts.to_datetime()
        assert dt == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_same_timestamp_different_instances_share_cache(self) -> None:
        """Different LogFilenameParts with the same timestamp share the module-level cache."""
        parts1 = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        parts2 = parse_log_filename_parts("DS-456_20240115-103045_a2.log")
        assert parts1 is not None
        assert parts2 is not None

        dt1 = parts1.to_datetime()
        dt2 = parts2.to_datetime()

        # Same timestamp string -> same cached object
        assert dt1 is dt2

    def test_different_timestamps_have_different_results(self) -> None:
        """Different timestamp strings produce different datetime results."""
        parts1 = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
        parts2 = parse_log_filename_parts("DS-456_20250601-120000_a2.log")
        assert parts1 is not None
        assert parts2 is not None

        dt1 = parts1.to_datetime()
        dt2 = parts2.to_datetime()

        assert dt1 != dt2
        assert dt1 == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
        assert dt2 == datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


class TestLogFileNamePartsParsedDatetimeProperty:
    """Tests for LogFilenameParts.parsed_datetime property (DS-998)."""

    def test_parsed_datetime_returns_correct_utc_datetime(self) -> None:
        """parsed_datetime property should return the correct UTC datetime."""
        result = parse_log_filename_parts("DS-123_20240115-103045_a1.log")

        assert result is not None
        dt = result.parsed_datetime
        assert dt == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_parsed_datetime_matches_to_datetime(self) -> None:
        """parsed_datetime property should return the same result as to_datetime()."""
        result = parse_log_filename_parts("DS-123_20240115-103045_a1.log")

        assert result is not None
        assert result.parsed_datetime == result.to_datetime()
        # Should be the same cached object
        assert result.parsed_datetime is result.to_datetime()

    def test_parsed_datetime_without_issue_key(self) -> None:
        """parsed_datetime should work for filenames without issue key."""
        result = parse_log_filename_parts("20240115-103045_a2.log")

        assert result is not None
        assert result.parsed_datetime == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_parsed_datetime_raises_value_error_for_invalid_timestamp(self) -> None:
        """parsed_datetime should raise ValueError for malformed timestamp.

        Mirrors the to_datetime() error test: directly-constructed
        LogFilenameParts with an invalid timestamp should propagate the
        error from strptime.
        """
        bad_parts = LogFilenameParts(issue_key="DS-999", timestamp="not-a-date", attempt=1)
        with pytest.raises(ValueError):
            _ = bad_parts.parsed_datetime

    def test_parsed_datetime_direct_construction(self) -> None:
        """parsed_datetime should work with directly constructed LogFilenameParts."""
        parts = LogFilenameParts(
            issue_key="DS-999", timestamp="20240115-103045", attempt=1,
        )
        assert parts.parsed_datetime == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)


class TestLegacyFormatUsesCache:
    """Tests verifying parse_log_filename legacy fallback delegates to _parse_log_timestamp (DS-998)."""

    def test_legacy_format_returns_correct_datetime(self) -> None:
        """Legacy format should still parse correctly after refactor."""
        filename = "20240115_103045.log"
        parsed = parse_log_filename(filename)

        assert parsed is not None
        assert parsed == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

    def test_legacy_format_delegates_to_cache(self) -> None:
        """Legacy fallback in parse_log_filename should use _parse_log_timestamp.

        DS-998: verify the legacy strptime call is now centralised through
        the same _parse_log_timestamp helper used by to_datetime().
        """
        with patch(
            "sentinel.logging._parse_log_timestamp",
            wraps=_parse_log_timestamp,
        ) as mock_cache:
            result = parse_log_filename("20240115_103045.log")
            assert result is not None
            assert result == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
            # _parse_log_timestamp should have been called for the legacy path
            mock_cache.assert_called_once()

    def test_legacy_format_invalid_returns_none(self) -> None:
        """Invalid legacy format should still return None."""
        assert parse_log_filename("not_valid.log") is None


class TestStripLogExtension:
    """Tests for _strip_log_extension shared helper (DS-1010)."""

    def test_strips_log_extension(self) -> None:
        """Should strip .log extension when present."""
        assert _strip_log_extension("DS-123_20240115-103045_a1.log") == "DS-123_20240115-103045_a1"

    def test_no_op_without_extension(self) -> None:
        """Should return filename unchanged when .log extension is absent."""
        assert _strip_log_extension("DS-123_20240115-103045_a1") == "DS-123_20240115-103045_a1"

    def test_strips_legacy_format_extension(self) -> None:
        """Should strip .log extension from legacy format filenames."""
        assert _strip_log_extension("20240115_103045.log") == "20240115_103045"

    def test_empty_string(self) -> None:
        """Should handle empty string without error."""
        assert _strip_log_extension("") == ""

    def test_only_extension(self) -> None:
        """Should return empty string when filename is just the extension."""
        assert _strip_log_extension(LOG_FILENAME_EXTENSION) == ""

    def test_does_not_strip_partial_extension(self) -> None:
        """Should not strip partial matches like .logfile."""
        assert _strip_log_extension("test.logfile") == "test.logfile"

    def test_does_not_strip_different_extension(self) -> None:
        """Should not strip non-.log extensions like .txt."""
        assert _strip_log_extension("test.txt") == "test.txt"


class TestStripLogExtensionUsedByBothPaths:
    """Verify both parse_log_filename_parts and parse_log_filename use _strip_log_extension (DS-1010)."""

    def test_new_format_uses_strip_log_extension(self) -> None:
        """parse_log_filename_parts should delegate to _strip_log_extension.

        DS-1010: verifies the new-format pipeline calls the shared helper
        instead of inlining .removesuffix().
        """
        with patch(
            "sentinel.logging._strip_log_extension",
            wraps=_strip_log_extension,
        ) as mock_strip:
            result = parse_log_filename_parts("DS-123_20240115-103045_a1.log")
            assert result is not None
            assert result.issue_key == "DS-123"
            mock_strip.assert_called_once_with("DS-123_20240115-103045_a1.log")

    def test_legacy_fallback_uses_strip_log_extension(self) -> None:
        """parse_log_filename legacy fallback should delegate to _strip_log_extension.

        DS-1010: verifies the legacy fallback path calls the shared helper
        instead of inlining .removesuffix().
        """
        with patch(
            "sentinel.logging._strip_log_extension",
            wraps=_strip_log_extension,
        ) as mock_strip:
            result = parse_log_filename("20240115_103045.log")
            assert result is not None
            assert result == datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)
            # _strip_log_extension should be called at least twice:
            # once by parse_log_filename_parts (which returns None for legacy),
            # once by the legacy fallback in parse_log_filename.
            assert mock_strip.call_count == 2


class TestFormatLogDisplayName:
    """Tests for _format_log_display_name method."""

    @pytest.fixture()
    def accessor(self) -> SentinelStateAccessor:
        """Create a minimal SentinelStateAccessor for testing.

        Uses ``MagicMock(spec=...)`` for resilient mocking — if
        ``SentinelStateProvider`` or ``SentinelConfig`` signatures
        change, the mock will raise ``AttributeError`` for any
        newly-accessed attribute that doesn't exist on the spec,
        surfacing breakage immediately instead of silently passing.
        """
        mock_sentinel = MagicMock(spec=SentinelStateProvider)
        mock_sentinel.config = MagicMock(spec=SentinelConfig)
        return SentinelStateAccessor(sentinel=mock_sentinel)

    def test_formats_new_format_with_issue_key(self, accessor: SentinelStateAccessor) -> None:
        """Should format new format with issue key correctly."""
        filename = "DS-123_20240115-103045_a1.log"
        result = accessor._format_log_display_name(filename)

        assert result == "DS-123 2024-01-15 10:30:45 (attempt 1)"

    def test_formats_new_format_without_issue_key(self, accessor: SentinelStateAccessor) -> None:
        """Should format new format without issue key correctly."""
        filename = "20240115-103045_a2.log"
        result = accessor._format_log_display_name(filename)

        assert result == "2024-01-15 10:30:45 (attempt 2)"

    def test_formats_legacy_format(self, accessor: SentinelStateAccessor) -> None:
        """Should format legacy format correctly."""
        filename = "20240115_103045.log"
        result = accessor._format_log_display_name(filename)

        assert result == "2024-01-15 10:30:45"

    def test_returns_filename_as_is_for_invalid_format(
        self, accessor: SentinelStateAccessor
    ) -> None:
        """Should return the filename as-is for invalid formats."""
        invalid_filename = "invalid_format.log"
        result = accessor._format_log_display_name(invalid_filename)

        assert result == invalid_filename

    def test_formats_with_higher_attempt_numbers(
        self, accessor: SentinelStateAccessor
    ) -> None:
        """Should format filenames with higher attempt numbers correctly."""
        filename1 = "DS-456_20240115-103045_a5.log"
        filename2 = "DS-456_20240115-103045_a10.log"

        result1 = accessor._format_log_display_name(filename1)
        result2 = accessor._format_log_display_name(filename2)

        assert result1 == "DS-456 2024-01-15 10:30:45 (attempt 5)"
        assert result2 == "DS-456 2024-01-15 10:30:45 (attempt 10)"

    def test_formats_underscore_issue_key(
        self, accessor: SentinelStateAccessor
    ) -> None:
        """Should format display name for issue keys containing underscores.

        DS-1002: validates that the display formatting layer correctly handles
        underscore-containing issue keys end-to-end.  The underlying parsing
        (``TestParseLogFilenameParts.test_handles_issue_keys_with_underscores``)
        already covered ``MY_PROJ-123`` at the ``LogFilenameParts`` level, but
        this test closes the coverage gap at the ``_format_log_display_name``
        layer identified during code review of DS-989 PR #1001.
        """
        filename = "MY_PROJ-123_20240115-103045_a1.log"
        result = accessor._format_log_display_name(filename)

        assert result == "MY_PROJ-123 2024-01-15 10:30:45 (attempt 1)"

    def test_formats_multi_underscore_issue_key(
        self, accessor: SentinelStateAccessor
    ) -> None:
        """Should format display name for issue keys with multiple underscores.

        DS-989 follow-up: extends ``test_formats_underscore_issue_key`` to verify
        that multi-underscore issue keys (e.g., MY_PROJ_V2-123) are correctly
        handled end-to-end through the display formatting layer, ensuring the
        greedy regex captures all underscore segments.
        """
        filename = "MY_PROJ_V2-123_20240115-103045_a1.log"
        result = accessor._format_log_display_name(filename)

        assert result == "MY_PROJ_V2-123 2024-01-15 10:30:45 (attempt 1)"

    def test_multi_hyphen_issue_key(self, accessor: SentinelStateAccessor) -> None:
        result = accessor._format_log_display_name(
            "MY-PROJECT-42_20250601-120000_a3.log",
        )
        assert result == "MY-PROJECT-42 2025-06-01 12:00:00 (attempt 3)"

    def test_prioritises_parse_log_filename_parts_over_parse_log_filename(
        self, accessor: SentinelStateAccessor
    ) -> None:
        """Verify _format_log_display_name calls parse_log_filename_parts first.

        DS-979 consolidation: for new-format filenames the method should
        resolve via ``parse_log_filename_parts`` without falling through to
        ``parse_log_filename``, avoiding redundant regex evaluation on the
        same filename.  ``parse_log_filename`` is still legitimately imported
        in ``state.py`` for legacy-format fallback, so a module-level import
        check (via ``dir()`` / ``hasattr()``) is **not** appropriate here.

        Trade-off note: this test uses ``unittest.mock.patch(wraps=...)``
        to verify call behaviour at runtime.  A simpler ``dir()`` /
        ``hasattr()`` approach would catch module-level imports but would
        **not** catch function-local imports — an acceptable limitation
        when the concern is redundant regex evaluation at the module level.
        Since ``state.py`` legitimately imports both functions, we instead
        verify that for new-format filenames ``parse_log_filename_parts``
        is called and ``parse_log_filename`` is **not**.
        """
        with (
            patch(
                "sentinel.dashboard.state.parse_log_filename_parts",
                wraps=parse_log_filename_parts,
            ) as mock_parts,
            patch(
                "sentinel.dashboard.state.parse_log_filename",
                wraps=parse_log_filename,
            ) as mock_legacy,
        ):
            result = accessor._format_log_display_name(
                "PROJ-1_20250115-103045_a1.log",
            )
            assert result == "PROJ-1 2025-01-15 10:30:45 (attempt 1)"
            mock_parts.assert_called_once_with("PROJ-1_20250115-103045_a1.log")
            mock_legacy.assert_not_called()
