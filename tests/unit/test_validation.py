"""Tests for validation module.

Tests for the validation helper functions that were consolidated from routes.py
into a dedicated sentinel.validation module.
"""

import logging

import pytest

from sentinel.validation import (
    MAX_CACHE_MAXSIZE,
    MAX_CACHE_TTL,
    MAX_TOGGLE_COOLDOWN,
    MIN_CACHE_MAXSIZE,
    MIN_CACHE_TTL,
    MIN_TOGGLE_COOLDOWN,
    validate_positive_float,
    validate_positive_int,
    validate_strictly_positive_float,
)


class TestValidatePositiveFloat:
    """Tests for validate_positive_float helper function."""

    def test_valid_positive_value(self) -> None:
        """Test parsing a valid positive float."""
        result = validate_positive_float("TEST_VAR", "5.5", 2.0, 0.0, 100.0)
        assert result == 5.5

    def test_valid_zero_value(self) -> None:
        """Test that zero is accepted (non-negative)."""
        result = validate_positive_float("TEST_VAR", "0.0", 2.0, 0.0, 100.0)
        assert result == 0.0

    def test_negative_value_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_float("TEST_VAR", "-1.0", 2.0, 0.0, 100.0)
        assert result == 2.0
        assert "TEST_VAR must be non-negative" in caplog.text

    def test_invalid_string_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid string values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_float("TEST_VAR", "not-a-number", 2.0, 0.0, 100.0)
        assert result == 2.0
        assert "TEST_VAR must be a valid float" in caplog.text

    def test_value_below_min_clamps_to_min(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values below minimum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_positive_float("TEST_VAR", "0.5", 2.0, 1.0, 100.0)
        assert result == 1.0
        assert "below minimum" in caplog.text

    def test_value_above_max_clamps_to_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values above maximum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_positive_float("TEST_VAR", "150.0", 2.0, 0.0, 100.0)
        assert result == 100.0
        assert "exceeds maximum" in caplog.text

    def test_boundary_min_value_accepted(self) -> None:
        """Test that minimum boundary value is accepted."""
        result = validate_positive_float("TEST_VAR", "1.0", 2.0, 1.0, 100.0)
        assert result == 1.0

    def test_boundary_max_value_accepted(self) -> None:
        """Test that maximum boundary value is accepted."""
        result = validate_positive_float("TEST_VAR", "100.0", 2.0, 0.0, 100.0)
        assert result == 100.0


class TestValidateStrictlyPositiveFloat:
    """Tests for validate_strictly_positive_float helper function."""

    def test_valid_positive_value(self) -> None:
        """Test parsing a valid positive float."""
        result = validate_strictly_positive_float("TEST_VAR", "5.5", 2.0, 0.1, 100.0)
        assert result == 5.5

    def test_zero_value_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that zero is rejected (strictly positive required)."""
        with caplog.at_level(logging.ERROR):
            result = validate_strictly_positive_float("TEST_VAR", "0.0", 2.0, 0.1, 100.0)
        assert result == 2.0
        assert "TEST_VAR must be positive (greater than 0)" in caplog.text

    def test_negative_value_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_strictly_positive_float("TEST_VAR", "-1.0", 2.0, 0.1, 100.0)
        assert result == 2.0
        assert "TEST_VAR must be positive (greater than 0)" in caplog.text

    def test_invalid_string_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid string values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_strictly_positive_float("TEST_VAR", "not-a-number", 2.0, 0.1, 100.0)
        assert result == 2.0
        assert "TEST_VAR must be a valid float" in caplog.text

    def test_value_below_min_clamps_to_min(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values below minimum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_strictly_positive_float("TEST_VAR", "0.05", 2.0, 0.1, 100.0)
        assert result == 0.1
        assert "below minimum" in caplog.text

    def test_value_above_max_clamps_to_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values above maximum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_strictly_positive_float("TEST_VAR", "150.0", 2.0, 0.1, 100.0)
        assert result == 100.0
        assert "exceeds maximum" in caplog.text

    def test_small_positive_value_accepted(self) -> None:
        """Test that small positive values are accepted."""
        result = validate_strictly_positive_float("TEST_VAR", "0.001", 2.0, 0.001, 100.0)
        assert result == 0.001


class TestValidatePositiveInt:
    """Tests for validate_positive_int helper function."""

    def test_valid_positive_value(self) -> None:
        """Test parsing a valid positive integer."""
        result = validate_positive_int("TEST_VAR", "42", 10, 1, 1000)
        assert result == 42

    def test_zero_value_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that zero is rejected (strictly positive required)."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_int("TEST_VAR", "0", 10, 1, 1000)
        assert result == 10
        assert "TEST_VAR must be positive (greater than 0)" in caplog.text

    def test_negative_value_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_int("TEST_VAR", "-5", 10, 1, 1000)
        assert result == 10
        assert "TEST_VAR must be positive (greater than 0)" in caplog.text

    def test_invalid_string_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid string values return default with error log."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_int("TEST_VAR", "not-a-number", 10, 1, 1000)
        assert result == 10
        assert "TEST_VAR must be a valid integer" in caplog.text

    def test_float_string_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that float strings are rejected for integer validation."""
        with caplog.at_level(logging.ERROR):
            result = validate_positive_int("TEST_VAR", "3.14", 10, 1, 1000)
        assert result == 10
        assert "TEST_VAR must be a valid integer" in caplog.text

    def test_value_below_min_clamps_to_min(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values below minimum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            # Value is positive but below min_val
            result = validate_positive_int("TEST_VAR", "5", 10, 10, 1000)
        assert result == 10
        assert "below minimum" in caplog.text

    def test_value_above_max_clamps_to_max(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that values above maximum are clamped with warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_positive_int("TEST_VAR", "5000", 10, 1, 1000)
        assert result == 1000
        assert "exceeds maximum" in caplog.text

    def test_boundary_min_value_accepted(self) -> None:
        """Test that minimum boundary value is accepted."""
        result = validate_positive_int("TEST_VAR", "1", 10, 1, 1000)
        assert result == 1

    def test_boundary_max_value_accepted(self) -> None:
        """Test that maximum boundary value is accepted."""
        result = validate_positive_int("TEST_VAR", "1000", 10, 1, 1000)
        assert result == 1000


class TestBoundsConstants:
    """Tests for bounds constants exported from validation module."""

    def test_toggle_cooldown_bounds(self) -> None:
        """Test toggle cooldown bounds constants."""
        assert MIN_TOGGLE_COOLDOWN == 0.0
        assert MAX_TOGGLE_COOLDOWN == 86400.0  # 24 hours

    def test_cache_ttl_bounds(self) -> None:
        """Test cache TTL bounds constants."""
        assert MIN_CACHE_TTL == 1
        assert MAX_CACHE_TTL == 604800  # 1 week

    def test_cache_maxsize_bounds(self) -> None:
        """Test cache maxsize bounds constants."""
        assert MIN_CACHE_MAXSIZE == 1
        assert MAX_CACHE_MAXSIZE == 1000000  # 1 million


class TestValidationModuleAll:
    """Tests for __all__ definition in validation module."""

    def test_all_contains_expected_symbols(self) -> None:
        """Test that __all__ contains all expected public symbols."""
        from sentinel.validation import __all__

        expected_symbols = [
            "MIN_TOGGLE_COOLDOWN",
            "MAX_TOGGLE_COOLDOWN",
            "MIN_CACHE_TTL",
            "MAX_CACHE_TTL",
            "MIN_CACHE_MAXSIZE",
            "MAX_CACHE_MAXSIZE",
            "validate_positive_float",
            "validate_strictly_positive_float",
            "validate_positive_int",
        ]

        for symbol in expected_symbols:
            assert symbol in __all__, f"Expected {symbol} to be in __all__"

    def test_all_symbols_are_importable(self) -> None:
        """Test that all symbols in __all__ can be imported."""
        from sentinel import validation
        from sentinel.validation import __all__

        for symbol in __all__:
            assert hasattr(validation, symbol), f"Symbol {symbol} in __all__ but not importable"
