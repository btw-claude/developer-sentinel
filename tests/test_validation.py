"""Tests for validation module.

Tests for the validation helper functions that were consolidated from routes.py
into a dedicated sentinel.validation module.

Added tests for deprecation warnings on backwards compatibility aliases.
"""

import logging
import warnings

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


class TestBackwardsCompatibility:
    """Tests to ensure backwards compatibility with routes.py imports."""

    def test_routes_module_imports_validation_functions(self) -> None:
        """Test that routes module can import validation functions."""
        from sentinel.dashboard.routes import _validate_positive_float, _validate_positive_int

        # Verify they are callable
        assert callable(_validate_positive_float)
        assert callable(_validate_positive_int)

    def test_routes_module_imports_bounds_constants(self) -> None:
        """Test that routes module can import bounds constants with deprecation warning."""
        from sentinel.dashboard import routes

        # Verify they have correct values and emit deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert routes._MIN_TOGGLE_COOLDOWN == MIN_TOGGLE_COOLDOWN
            assert routes._MAX_TOGGLE_COOLDOWN == MAX_TOGGLE_COOLDOWN
            assert routes._MIN_CACHE_TTL == MIN_CACHE_TTL
            assert routes._MAX_CACHE_TTL == MAX_CACHE_TTL
            assert routes._MIN_CACHE_MAXSIZE == MIN_CACHE_MAXSIZE
            assert routes._MAX_CACHE_MAXSIZE == MAX_CACHE_MAXSIZE

            # Verify deprecation warnings were emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 6

    def test_routes_validation_functions_work_same_as_module(self) -> None:
        """Test that routes validation functions behave same as module functions."""
        from sentinel.dashboard.routes import _validate_positive_float as routes_float
        from sentinel.dashboard.routes import _validate_positive_int as routes_int

        # Test that they produce same results (with deprecation warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert routes_float("TEST", "5.0", 2.0, 0.0, 10.0) == validate_positive_float(
                "TEST", "5.0", 2.0, 0.0, 10.0
            )
            assert routes_int("TEST", "5", 2, 1, 10) == validate_positive_int("TEST", "5", 2, 1, 10)

            # Verify deprecation warnings were emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 2


class TestDeprecationWarnings:
    """Tests for deprecation warnings on backwards compatibility aliases."""

    def test_validate_positive_float_emits_deprecation_warning(self) -> None:
        """Test that _validate_positive_float emits deprecation warning."""
        from sentinel.dashboard.routes import _validate_positive_float

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_positive_float("TEST", "5.0", 2.0, 0.0, 10.0)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "_validate_positive_float is deprecated" in str(deprecation_warnings[0].message)
            assert "sentinel.validation" in str(deprecation_warnings[0].message)

    def test_validate_positive_int_emits_deprecation_warning(self) -> None:
        """Test that _validate_positive_int emits deprecation warning."""
        from sentinel.dashboard.routes import _validate_positive_int

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_positive_int("TEST", "5", 2, 1, 10)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "_validate_positive_int is deprecated" in str(deprecation_warnings[0].message)
            assert "sentinel.validation" in str(deprecation_warnings[0].message)

    def test_bounds_constants_emit_deprecation_warnings(self) -> None:
        """Test that accessing deprecated bounds constants emits warnings."""
        from sentinel.dashboard import routes

        deprecated_constants = [
            ("_MIN_TOGGLE_COOLDOWN", "MIN_TOGGLE_COOLDOWN"),
            ("_MAX_TOGGLE_COOLDOWN", "MAX_TOGGLE_COOLDOWN"),
            ("_MIN_CACHE_TTL", "MIN_CACHE_TTL"),
            ("_MAX_CACHE_TTL", "MAX_CACHE_TTL"),
            ("_MIN_CACHE_MAXSIZE", "MIN_CACHE_MAXSIZE"),
            ("_MAX_CACHE_MAXSIZE", "MAX_CACHE_MAXSIZE"),
        ]

        for deprecated_name, canonical_name in deprecated_constants:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                getattr(routes, deprecated_name)

                deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(deprecation_warnings) == 1, f"Expected 1 warning for {deprecated_name}"
                warning_msg = str(deprecation_warnings[0].message)
                assert deprecated_name in warning_msg
                assert canonical_name in warning_msg
                assert "sentinel.validation" in warning_msg


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
