"""Tests for configuration module."""

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sentinel.config import (
    MAX_PORT,
    MIN_PORT,
    VALID_AGENT_TYPES,
    VALID_CURSOR_MODES,
    VALID_LOG_LEVELS,
    Config,
    _parse_bool,
    _parse_non_negative_float,
    _parse_port,
    _parse_positive_int,
    _validate_agent_type,
    _validate_cursor_mode,
    _validate_log_level,
    load_config,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self) -> None:
        config = Config()
        assert config.poll_interval == 60
        assert config.max_issues_per_poll == 50
        assert config.max_eager_iterations == 10
        assert config.max_concurrent_executions == 1
        assert config.log_level == "INFO"
        assert config.log_json is False
        assert config.orchestrations_dir == Path("./orchestrations")

    def test_custom_values(self) -> None:
        config = Config(
            poll_interval=120,
            max_issues_per_poll=100,
            log_level="DEBUG",
        )
        assert config.poll_interval == 120
        assert config.max_issues_per_poll == 100
        assert config.log_level == "DEBUG"

    def test_frozen_immutable(self) -> None:
        """Config should be immutable after creation."""
        config = Config()
        with pytest.raises(FrozenInstanceError):
            config.poll_interval = 120  # type: ignore[misc]


class TestParsePositiveInt:
    """Tests for _parse_positive_int helper function."""

    def test_valid_positive_integer(self) -> None:
        assert _parse_positive_int("42", "TEST_VAR", 10) == 42
        assert _parse_positive_int("1", "TEST_VAR", 10) == 1
        assert _parse_positive_int("1000", "TEST_VAR", 10) == 1000

    def test_invalid_non_numeric(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_positive_int("abc", "TEST_VAR", 10)
        assert result == 10
        assert "Invalid TEST_VAR: 'abc' is not a valid integer" in caplog.text

    def test_invalid_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_positive_int("0", "TEST_VAR", 10)
        assert result == 10
        assert "Invalid TEST_VAR: 0 is not positive" in caplog.text

    def test_invalid_negative(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_positive_int("-5", "TEST_VAR", 10)
        assert result == 10
        assert "Invalid TEST_VAR: -5 is not positive" in caplog.text

    def test_invalid_float_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_positive_int("3.14", "TEST_VAR", 10)
        assert result == 10
        assert "Invalid TEST_VAR: '3.14' is not a valid integer" in caplog.text

    def test_invalid_empty_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_positive_int("", "TEST_VAR", 10)
        assert result == 10


class TestValidateLogLevel:
    """Tests for _validate_log_level helper function."""

    def test_valid_log_levels(self) -> None:
        for level in VALID_LOG_LEVELS:
            assert _validate_log_level(level) == level

    def test_case_insensitive(self) -> None:
        assert _validate_log_level("debug") == "DEBUG"
        assert _validate_log_level("Info") == "INFO"
        assert _validate_log_level("WARNING") == "WARNING"

    def test_invalid_log_level(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _validate_log_level("INVALID")
        assert result == "INFO"
        assert "Invalid SENTINEL_LOG_LEVEL: 'INVALID' is not valid" in caplog.text

    def test_invalid_with_custom_default(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _validate_log_level("INVALID", default="WARNING")
        assert result == "WARNING"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_MAX_ISSUES", raising=False)
        monkeypatch.delenv("SENTINEL_LOG_LEVEL", raising=False)

        config = load_config()

        assert config.poll_interval == 60
        assert config.max_issues_per_poll == 50
        assert config.log_level == "INFO"

    def test_loads_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "30")
        monkeypatch.setenv("SENTINEL_MAX_ISSUES", "25")
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SENTINEL_ORCHESTRATIONS_DIR", "/custom/path")

        config = load_config()

        assert config.poll_interval == 30
        assert config.max_issues_per_poll == 25
        assert config.log_level == "DEBUG"
        assert config.orchestrations_dir == Path("/custom/path")

    def test_loads_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_LOG_LEVEL", raising=False)

        # Create a test .env file
        env_file = tmp_path / ".env"
        env_file.write_text("SENTINEL_POLL_INTERVAL=45\nSENTINEL_LOG_LEVEL=WARNING\n")

        config = load_config(env_file)

        assert config.poll_interval == 45
        assert config.log_level == "WARNING"

    def test_invalid_poll_interval_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.poll_interval == 60
        assert "Invalid SENTINEL_POLL_INTERVAL" in caplog.text

    def test_negative_poll_interval_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "-10")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.poll_interval == 60
        assert "not positive" in caplog.text

    def test_zero_max_issues_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_MAX_ISSUES", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_issues_per_poll == 50
        assert "not positive" in caplog.text

    def test_invalid_log_level_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "TRACE")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.log_level == "INFO"
        assert "Invalid SENTINEL_LOG_LEVEL" in caplog.text

    def test_log_level_normalized_to_uppercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "debug")

        config = load_config()

        assert config.log_level == "DEBUG"

    def test_log_json_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env var
        monkeypatch.delenv("SENTINEL_LOG_JSON", raising=False)

        config = load_config()

        assert config.log_json is False

    def test_log_json_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "true")

        config = load_config()

        assert config.log_json is True

    def test_log_json_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "1")

        config = load_config()

        assert config.log_json is True

    def test_log_json_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "yes")

        config = load_config()

        assert config.log_json is True

    def test_log_json_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "TRUE")

        config = load_config()

        assert config.log_json is True

    def test_log_json_false_for_other_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "false")

        config = load_config()

        assert config.log_json is False


class TestParseNonNegativeFloat:
    """Tests for _parse_non_negative_float helper function."""

    def test_valid_positive_float(self) -> None:
        assert _parse_non_negative_float("3.14", "TEST_VAR", 1.0) == 3.14
        assert _parse_non_negative_float("1.0", "TEST_VAR", 1.0) == 1.0
        assert _parse_non_negative_float("100.5", "TEST_VAR", 1.0) == 100.5

    def test_valid_zero(self) -> None:
        """Test that zero is accepted (non-negative)."""
        assert _parse_non_negative_float("0", "TEST_VAR", 1.0) == 0.0
        assert _parse_non_negative_float("0.0", "TEST_VAR", 1.0) == 0.0

    def test_valid_integer_string(self) -> None:
        """Test that integer strings are accepted."""
        assert _parse_non_negative_float("5", "TEST_VAR", 1.0) == 5.0

    def test_invalid_negative(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_non_negative_float("-1.5", "TEST_VAR", 1.0)
        assert result == 1.0
        assert "Invalid TEST_VAR: -1.500000 is negative" in caplog.text

    def test_invalid_non_numeric(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_non_negative_float("abc", "TEST_VAR", 1.0)
        assert result == 1.0
        assert "Invalid TEST_VAR: 'abc' is not a valid number" in caplog.text

    def test_invalid_empty_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_non_negative_float("", "TEST_VAR", 1.0)
        assert result == 1.0


class TestParseBool:
    """Tests for _parse_bool helper function."""

    def test_true_values(self) -> None:
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True
        assert _parse_bool("Yes") is True
        assert _parse_bool("YES") is True

    def test_false_values(self) -> None:
        assert _parse_bool("false") is False
        assert _parse_bool("False") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False
        assert _parse_bool("") is False
        assert _parse_bool("anything") is False


class TestPortConstants:
    """Tests for port validation constants."""

    def test_min_port_value(self) -> None:
        """Test that MIN_PORT is 1."""
        assert MIN_PORT == 1

    def test_max_port_value(self) -> None:
        """Test that MAX_PORT is 65535."""
        assert MAX_PORT == 65535


class TestParsePort:
    """Tests for _parse_port helper function."""

    def test_valid_port_numbers(self) -> None:
        """Test parsing valid port numbers within range."""
        assert _parse_port(str(MIN_PORT), "TEST_VAR", 8080) == MIN_PORT
        assert _parse_port("80", "TEST_VAR", 8080) == 80
        assert _parse_port("443", "TEST_VAR", 8080) == 443
        assert _parse_port("8080", "TEST_VAR", 8080) == 8080
        assert _parse_port(str(MAX_PORT), "TEST_VAR", 8080) == MAX_PORT

    def test_invalid_port_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that port 0 is rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("0", "TEST_VAR", 8080)
        assert result == 8080
        assert f"Invalid TEST_VAR: 0 is not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_negative(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative port numbers are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("-1", "TEST_VAR", 8080)
        assert result == 8080
        assert f"Invalid TEST_VAR: -1 is not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_too_high(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that port numbers above MAX_PORT are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port(str(MAX_PORT + 1), "TEST_VAR", 8080)
        assert result == 8080
        assert f"Invalid TEST_VAR: {MAX_PORT + 1} is not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_way_too_high(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that very large port numbers are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("100000", "TEST_VAR", 8080)
        assert result == 8080
        assert f"Invalid TEST_VAR: 100000 is not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_non_numeric(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that non-numeric values are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("abc", "TEST_VAR", 8080)
        assert result == 8080
        assert "Invalid TEST_VAR: 'abc' is not a valid integer" in caplog.text

    def test_invalid_empty_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty strings are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("", "TEST_VAR", 8080)
        assert result == 8080

    def test_invalid_float_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that float strings are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("8080.5", "TEST_VAR", 8080)
        assert result == 8080
        assert "Invalid TEST_VAR: '8080.5' is not a valid integer" in caplog.text


class TestDashboardPortConfig:
    """Tests for dashboard_port configuration with port range validation."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that dashboard_port defaults to 8080."""
        monkeypatch.delenv("SENTINEL_DASHBOARD_PORT", raising=False)

        config = load_config()

        assert config.dashboard_port == 8080

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that dashboard_port loads from environment variable."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "3000")

        config = load_config()

        assert config.dashboard_port == 3000

    def test_invalid_port_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that port 0 uses the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard_port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative ports use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "-1")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard_port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_too_high_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that ports above MAX_PORT use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MAX_PORT + 1))

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard_port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_non_numeric_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that non-numeric values use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard_port == 8080
        assert "Invalid SENTINEL_DASHBOARD_PORT" in caplog.text

    def test_valid_port_boundary_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that MIN_PORT is accepted."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MIN_PORT))

        config = load_config()

        assert config.dashboard_port == MIN_PORT

    def test_valid_port_boundary_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that MAX_PORT is accepted."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MAX_PORT))

        config = load_config()

        assert config.dashboard_port == MAX_PORT


class TestMaxEagerIterations:
    """Tests for max_eager_iterations configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_eager_iterations defaults to 10."""
        monkeypatch.delenv("SENTINEL_MAX_EAGER_ITERATIONS", raising=False)

        config = load_config()

        assert config.max_eager_iterations == 10

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_eager_iterations loads from environment variable."""
        monkeypatch.setenv("SENTINEL_MAX_EAGER_ITERATIONS", "5")

        config = load_config()

        assert config.max_eager_iterations == 5

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_EAGER_ITERATIONS", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_eager_iterations == 10
        assert "Invalid SENTINEL_MAX_EAGER_ITERATIONS" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_MAX_EAGER_ITERATIONS", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_eager_iterations == 10
        assert "not positive" in caplog.text

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_EAGER_ITERATIONS", "-5")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_eager_iterations == 10
        assert "not positive" in caplog.text


class TestAttemptCountsTtlConfig:
    """Tests for attempt_counts_ttl configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that attempt_counts_ttl defaults to 3600 seconds (1 hour)."""
        monkeypatch.delenv("SENTINEL_ATTEMPT_COUNTS_TTL", raising=False)

        config = load_config()

        assert config.attempt_counts_ttl == 3600

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that attempt_counts_ttl loads from environment variable."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "7200")

        config = load_config()

        assert config.attempt_counts_ttl == 7200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.attempt_counts_ttl == 3600
        assert "Invalid SENTINEL_ATTEMPT_COUNTS_TTL" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.attempt_counts_ttl == 3600
        assert "not positive" in caplog.text

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "-100")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.attempt_counts_ttl == 3600
        assert "not positive" in caplog.text


class TestMaxQueueSizeConfig:
    """Tests for max_queue_size configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_queue_size defaults to 100."""
        monkeypatch.delenv("SENTINEL_MAX_QUEUE_SIZE", raising=False)

        config = load_config()

        assert config.max_queue_size == 100

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_queue_size loads from environment variable."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "200")

        config = load_config()

        assert config.max_queue_size == 200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_queue_size == 100
        assert "Invalid SENTINEL_MAX_QUEUE_SIZE" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_queue_size == 100
        assert "not positive" in caplog.text

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "-50")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.max_queue_size == 100
        assert "not positive" in caplog.text


class TestDisableStreamingLogsConfig:
    """Tests for disable_streaming_logs configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs defaults to False."""
        monkeypatch.delenv("SENTINEL_DISABLE_STREAMING_LOGS", raising=False)

        config = load_config()

        assert config.disable_streaming_logs is False

    def test_enabled_with_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with 'true'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "true")

        config = load_config()

        assert config.disable_streaming_logs is True

    def test_enabled_with_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with '1'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "1")

        config = load_config()

        assert config.disable_streaming_logs is True

    def test_enabled_with_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with 'yes'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "yes")

        config = load_config()

        assert config.disable_streaming_logs is True

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "TRUE")

        config = load_config()

        assert config.disable_streaming_logs is True

    def test_false_for_other_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs is False for other values."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "false")

        config = load_config()

        assert config.disable_streaming_logs is False

    def test_false_for_empty_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs is False for empty value."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "")

        config = load_config()

        assert config.disable_streaming_logs is False


class TestValidateAgentType:
    """Tests for _validate_agent_type helper function."""

    def test_valid_agent_types(self) -> None:
        """Test that all valid agent types are accepted."""
        for agent_type in VALID_AGENT_TYPES:
            assert _validate_agent_type(agent_type) == agent_type

    def test_case_insensitive(self) -> None:
        """Test that agent type validation is case-insensitive."""
        assert _validate_agent_type("CLAUDE") == "claude"
        assert _validate_agent_type("Cursor") == "cursor"
        assert _validate_agent_type("CURSOR") == "cursor"

    def test_mixed_case_input(self) -> None:
        """Test that mixed case input is correctly normalized."""
        assert _validate_agent_type("CuRsOr") == "cursor"
        assert _validate_agent_type("ClAuDe") == "claude"
        assert _validate_agent_type("cLaUdE") == "claude"
        assert _validate_agent_type("cUrSoR") == "cursor"

    def test_invalid_agent_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid agent type returns default and logs warning."""
        with caplog.at_level(logging.WARNING):
            result = _validate_agent_type("invalid")
        assert result == "claude"
        assert "Invalid SENTINEL_DEFAULT_AGENT_TYPE: 'invalid' is not valid" in caplog.text


class TestValidateCursorMode:
    """Tests for _validate_cursor_mode helper function."""

    def test_valid_cursor_modes(self) -> None:
        """Test that all valid cursor modes are accepted."""
        for mode in VALID_CURSOR_MODES:
            assert _validate_cursor_mode(mode) == mode

    def test_case_insensitive(self) -> None:
        """Test that cursor mode validation is case-insensitive."""
        assert _validate_cursor_mode("AGENT") == "agent"
        assert _validate_cursor_mode("Plan") == "plan"
        assert _validate_cursor_mode("ASK") == "ask"

    def test_invalid_cursor_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid cursor mode returns default and logs warning."""
        with caplog.at_level(logging.WARNING):
            result = _validate_cursor_mode("invalid")
        assert result == "agent"
        assert "Invalid SENTINEL_CURSOR_DEFAULT_MODE: 'invalid' is not valid" in caplog.text

    def test_invalid_with_custom_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that invalid cursor mode uses custom default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_cursor_mode("invalid", default="plan")
        assert result == "plan"


class TestCursorConfig:
    """Tests for Cursor CLI configuration."""

    def test_default_agent_type_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_agent_type defaults to 'claude'."""
        monkeypatch.delenv("SENTINEL_DEFAULT_AGENT_TYPE", raising=False)

        config = load_config()

        assert config.default_agent_type == "claude"

    def test_default_agent_type_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_agent_type loads from environment variable."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "cursor")

        config = load_config()

        assert config.default_agent_type == "cursor"

    def test_cursor_path_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_path defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CURSOR_PATH", raising=False)

        config = load_config()

        assert config.cursor_path == ""

    def test_cursor_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_path loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_PATH", "/usr/local/bin/cursor")

        config = load_config()

        assert config.cursor_path == "/usr/local/bin/cursor"

    def test_cursor_default_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_model defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CURSOR_DEFAULT_MODEL", raising=False)

        config = load_config()

        assert config.cursor_default_model == ""

    def test_cursor_default_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_model loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODEL", "gpt-4")

        config = load_config()

        assert config.cursor_default_model == "gpt-4"

    def test_cursor_default_mode_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode defaults to 'agent'."""
        monkeypatch.delenv("SENTINEL_CURSOR_DEFAULT_MODE", raising=False)

        config = load_config()

        assert config.cursor_default_mode == "agent"

    def test_cursor_default_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "plan")

        config = load_config()

        assert config.cursor_default_mode == "plan"

    def test_cursor_default_mode_ask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode can be set to 'ask'."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "ask")

        config = load_config()

        assert config.cursor_default_mode == "ask"

    def test_cursor_default_mode_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "PLAN")

        config = load_config()

        assert config.cursor_default_mode == "plan"

    def test_cursor_default_mode_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid cursor_default_mode uses the default."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "invalid")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.cursor_default_mode == "agent"
        assert "Invalid SENTINEL_CURSOR_DEFAULT_MODE" in caplog.text

    def test_default_agent_type_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_agent_type parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "CURSOR")

        config = load_config()

        assert config.default_agent_type == "cursor"

    def test_default_agent_type_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid default_agent_type uses the default."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "invalid")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.default_agent_type == "claude"
        assert "Invalid SENTINEL_DEFAULT_AGENT_TYPE" in caplog.text


class TestInterMessageTimesThresholdConfig:
    """Tests for inter_message_times_threshold configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that inter_message_times_threshold defaults to 100."""
        monkeypatch.delenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", raising=False)

        config = load_config()

        assert config.inter_message_times_threshold == 100

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that inter_message_times_threshold loads from environment variable."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "200")

        config = load_config()

        assert config.inter_message_times_threshold == 200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.inter_message_times_threshold == 100
        assert "Invalid SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.inter_message_times_threshold == 100
        assert "not positive" in caplog.text


class TestToggleCooldownConfig:
    """Tests for toggle_cooldown_seconds configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that toggle_cooldown_seconds defaults to 2.0."""
        monkeypatch.delenv("SENTINEL_TOGGLE_COOLDOWN", raising=False)

        config = load_config()

        assert config.toggle_cooldown_seconds == 2.0

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that toggle_cooldown_seconds loads from environment variable."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "5.0")

        config = load_config()

        assert config.toggle_cooldown_seconds == 5.0

    def test_zero_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that zero is allowed (to disable cooldown)."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "0")

        config = load_config()

        assert config.toggle_cooldown_seconds == 0.0

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "-1.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.toggle_cooldown_seconds == 2.0
        assert "negative" in caplog.text

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.toggle_cooldown_seconds == 2.0
        assert "Invalid SENTINEL_TOGGLE_COOLDOWN" in caplog.text


class TestRateLimitCacheTtlConfig:
    """Tests for rate_limit_cache_ttl configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_ttl defaults to 3600."""
        monkeypatch.delenv("SENTINEL_RATE_LIMIT_CACHE_TTL", raising=False)

        config = load_config()

        assert config.rate_limit_cache_ttl == 3600

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_ttl loads from environment variable."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_TTL", "7200")

        config = load_config()

        assert config.rate_limit_cache_ttl == 7200

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_TTL", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.rate_limit_cache_ttl == 3600
        assert "not positive" in caplog.text


class TestRateLimitCacheMaxsizeConfig:
    """Tests for rate_limit_cache_maxsize configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_maxsize defaults to 10000."""
        monkeypatch.delenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", raising=False)

        config = load_config()

        assert config.rate_limit_cache_maxsize == 10000

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_maxsize loads from environment variable."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", "50000")

        config = load_config()

        assert config.rate_limit_cache_maxsize == 50000

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.rate_limit_cache_maxsize == 10000
        assert "not positive" in caplog.text
