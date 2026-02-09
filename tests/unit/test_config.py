"""Tests for configuration module."""

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sentinel.config import (
    DEFAULT_GREEN_THRESHOLD,
    DEFAULT_YELLOW_THRESHOLD,
    MAX_PORT,
    MIN_PORT,
    VALID_AGENT_TYPES,
    VALID_CURSOR_MODES,
    VALID_LOG_LEVELS,
    CircuitBreakerConfig,
    CodexConfig,
    Config,
    CursorConfig,
    DashboardConfig,
    ExecutionConfig,
    GitHubConfig,
    HealthCheckConfig,
    JiraConfig,
    LoggingConfig,
    PollingConfig,
    RateLimitConfig,
    ServiceHealthGateConfig,
    _format_bound_message,
    _format_number,
    _parse_bool,
    _parse_bounded_float,
    _parse_non_negative_float,
    _parse_port,
    _parse_positive_int,
    _validate_agent_type,
    _validate_branch_name,
    _validate_cursor_mode,
    _validate_log_level,
    load_config,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self) -> None:
        config = Config()
        # Test sub-configs
        assert config.polling.interval == 60
        assert config.polling.max_issues_per_poll == 50
        assert config.execution.max_concurrent_executions == 1
        assert config.logging_config.level == "INFO"
        assert config.logging_config.json is False
        assert config.execution.orchestrations_dir == Path("./orchestrations")

    def test_custom_values(self) -> None:
        config = Config(
            polling=PollingConfig(interval=120, max_issues_per_poll=100),
            logging_config=LoggingConfig(level="DEBUG"),
        )
        # Test sub-configs
        assert config.polling.interval == 120
        assert config.polling.max_issues_per_poll == 100
        assert config.logging_config.level == "DEBUG"

    def test_frozen_immutable(self) -> None:
        """Config should be immutable after creation."""
        config = Config()
        with pytest.raises(FrozenInstanceError):
            config.polling = PollingConfig(interval=120)  # type: ignore[misc]


class TestSubConfigs:
    """Tests for sub-config dataclasses."""

    def test_jira_config_defaults(self) -> None:
        jira = JiraConfig()
        assert jira.base_url == ""
        assert jira.email == ""
        assert jira.api_token == ""
        assert jira.epic_link_field == "customfield_10014"
        assert jira.configured is False

    def test_jira_config_configured(self) -> None:
        jira = JiraConfig(
            base_url="https://test.atlassian.net",
            email="test@example.com",
            api_token="token",
        )
        assert jira.configured is True

    def test_github_config_defaults(self) -> None:
        github = GitHubConfig()
        assert github.token == ""
        assert github.api_url == ""
        assert github.configured is False

    def test_github_config_configured(self) -> None:
        github = GitHubConfig(token="gh_token")
        assert github.configured is True

    def test_dashboard_config_defaults(self) -> None:
        dashboard = DashboardConfig()
        assert dashboard.enabled is False
        assert dashboard.port == 8080
        assert dashboard.host == "127.0.0.1"
        assert dashboard.toggle_cooldown_seconds == 2.0
        assert dashboard.rate_limit_cache_ttl == 3600
        assert dashboard.rate_limit_cache_maxsize == 10000
        assert dashboard.success_rate_green_threshold == DEFAULT_GREEN_THRESHOLD
        assert dashboard.success_rate_yellow_threshold == DEFAULT_YELLOW_THRESHOLD

    def test_rate_limit_config_defaults(self) -> None:
        rate_limit = RateLimitConfig()
        assert rate_limit.enabled is True
        assert rate_limit.per_minute == 60
        assert rate_limit.per_hour == 1000
        assert rate_limit.strategy == "queue"
        assert rate_limit.warning_threshold == 0.2

    def test_circuit_breaker_config_defaults(self) -> None:
        circuit_breaker = CircuitBreakerConfig()
        assert circuit_breaker.enabled is True
        assert circuit_breaker.failure_threshold == 5
        assert circuit_breaker.recovery_timeout == 30.0
        assert circuit_breaker.half_open_max_calls == 3

    def test_health_check_config_defaults(self) -> None:
        health_check = HealthCheckConfig()
        assert health_check.enabled is True
        assert health_check.timeout == 5.0

    def test_service_health_gate_config_defaults(self) -> None:
        shg = ServiceHealthGateConfig()
        assert shg.enabled is True
        assert shg.failure_threshold == 3
        assert shg.initial_probe_interval == 30.0
        assert shg.max_probe_interval == 300.0
        assert shg.probe_backoff_factor == 2.0
        assert shg.probe_timeout == 5.0

    def test_service_health_gate_config_frozen(self) -> None:
        """ServiceHealthGateConfig should be immutable after creation."""
        shg = ServiceHealthGateConfig()
        with pytest.raises(FrozenInstanceError):
            shg.enabled = False  # type: ignore[misc]

    def test_execution_config_defaults(self) -> None:
        execution = ExecutionConfig()
        assert execution.orchestrations_dir == Path("./orchestrations")
        assert execution.agent_workdir == Path("./workdir")
        assert execution.agent_logs_dir == Path("./logs")
        assert execution.orchestration_logs_dir is None
        assert execution.max_concurrent_executions == 1
        assert execution.cleanup_workdir_on_success is True
        assert execution.disable_streaming_logs is False
        assert execution.subprocess_timeout == 60.0
        assert execution.default_base_branch == "main"
        assert execution.attempt_counts_ttl == 3600
        assert execution.max_queue_size == 100
        assert execution.inter_message_times_threshold == 100
        assert execution.max_recent_executions == 10

    def test_cursor_config_defaults(self) -> None:
        cursor = CursorConfig()
        assert cursor.default_agent_type == "claude"
        assert cursor.path == ""
        assert cursor.default_model == ""
        assert cursor.default_mode == "agent"

    def test_codex_config_defaults(self) -> None:
        codex = CodexConfig()
        assert codex.path == ""
        assert codex.default_model == ""

    def test_logging_config_defaults(self) -> None:
        logging_cfg = LoggingConfig()
        assert logging_cfg.level == "INFO"
        assert logging_cfg.json is False

    def test_polling_config_defaults(self) -> None:
        polling = PollingConfig()
        assert polling.interval == 60
        assert polling.max_issues_per_poll == 50


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

    def test_loads_defaults_without_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clear any existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_MAX_ISSUES", raising=False)
        monkeypatch.delenv("SENTINEL_LOG_LEVEL", raising=False)

        # Use an empty .env file to prevent load_dotenv() from searching parent directories
        empty_env_file = tmp_path / ".env"
        empty_env_file.touch()

        config = load_config(empty_env_file)

        assert config.polling.interval == 60
        assert config.polling.max_issues_per_poll == 50
        assert config.logging_config.level == "INFO"

    def test_loads_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "30")
        monkeypatch.setenv("SENTINEL_MAX_ISSUES", "25")
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SENTINEL_ORCHESTRATIONS_DIR", "/custom/path")

        config = load_config()

        assert config.polling.interval == 30
        assert config.polling.max_issues_per_poll == 25
        assert config.logging_config.level == "DEBUG"
        assert config.execution.orchestrations_dir == Path("/custom/path")

    def test_loads_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_LOG_LEVEL", raising=False)

        # Create a test .env file
        env_file = tmp_path / ".env"
        env_file.write_text("SENTINEL_POLL_INTERVAL=45\nSENTINEL_LOG_LEVEL=WARNING\n")

        config = load_config(env_file)

        assert config.polling.interval == 45
        assert config.logging_config.level == "WARNING"

    def test_invalid_poll_interval_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.polling.interval == 60
        assert "Invalid SENTINEL_POLL_INTERVAL" in caplog.text

    def test_negative_poll_interval_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "-10")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.polling.interval == 60
        assert "not positive" in caplog.text

    def test_zero_max_issues_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_MAX_ISSUES", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.polling.max_issues_per_poll == 50
        assert "not positive" in caplog.text

    def test_invalid_log_level_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "TRACE")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.logging_config.level == "INFO"
        assert "Invalid SENTINEL_LOG_LEVEL" in caplog.text

    def test_log_level_normalized_to_uppercase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "debug")

        config = load_config()

        assert config.logging_config.level == "DEBUG"

    def test_log_json_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env var
        monkeypatch.delenv("SENTINEL_LOG_JSON", raising=False)

        config = load_config()

        assert config.logging_config.json is False

    def test_log_json_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "true")

        config = load_config()

        assert config.logging_config.json is True

    def test_log_json_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "1")

        config = load_config()

        assert config.logging_config.json is True

    def test_log_json_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "yes")

        config = load_config()

        assert config.logging_config.json is True

    def test_log_json_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "TRUE")

        config = load_config()

        assert config.logging_config.json is True

    def test_log_json_false_for_other_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_LOG_JSON", "false")

        config = load_config()

        assert config.logging_config.json is False


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
        assert "Invalid TEST_VAR: -1.5 must be >= 0" in caplog.text

    def test_invalid_non_numeric(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_non_negative_float("abc", "TEST_VAR", 1.0)
        assert result == 1.0
        assert "Invalid TEST_VAR: 'abc' is not a valid number" in caplog.text

    def test_invalid_empty_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = _parse_non_negative_float("", "TEST_VAR", 1.0)
        assert result == 1.0


class TestDefaultThresholdConstants:
    """Tests for DEFAULT_GREEN_THRESHOLD and DEFAULT_YELLOW_THRESHOLD constants."""

    def test_green_threshold_value(self) -> None:
        """Test that DEFAULT_GREEN_THRESHOLD is 90.0."""
        assert DEFAULT_GREEN_THRESHOLD == 90.0

    def test_yellow_threshold_value(self) -> None:
        """Test that DEFAULT_YELLOW_THRESHOLD is 70.0."""
        assert DEFAULT_YELLOW_THRESHOLD == 70.0

    def test_green_greater_than_yellow(self) -> None:
        """Test that green threshold is greater than yellow threshold."""
        assert DEFAULT_GREEN_THRESHOLD > DEFAULT_YELLOW_THRESHOLD


class TestParseBoundedFloat:
    """Tests for _parse_bounded_float helper function."""

    def test_valid_value_no_bounds(self) -> None:
        """Test parsing a valid float with no bounds."""
        assert _parse_bounded_float("3.14", "TEST_VAR", 1.0) == 3.14

    def test_valid_value_within_bounds(self) -> None:
        """Test parsing a valid float within specified bounds."""
        assert _parse_bounded_float("0.5", "TEST_VAR", 0.2, min_val=0.0, max_val=1.0) == 0.5

    def test_value_at_min_bound(self) -> None:
        """Test parsing a value exactly at the minimum bound."""
        assert _parse_bounded_float("0.0", "TEST_VAR", 0.5, min_val=0.0, max_val=1.0) == 0.0

    def test_value_at_max_bound(self) -> None:
        """Test parsing a value exactly at the maximum bound."""
        assert _parse_bounded_float("1.0", "TEST_VAR", 0.5, min_val=0.0, max_val=1.0) == 1.0

    def test_value_below_min_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that a value below min returns the default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("-0.5", "TEST_VAR", 0.2, min_val=0.0, max_val=1.0)
        assert result == 0.2
        assert "Invalid TEST_VAR" in caplog.text

    def test_value_above_max_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that a value above max returns the default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("1.5", "TEST_VAR", 0.2, min_val=0.0, max_val=1.0)
        assert result == 0.2
        assert "Invalid TEST_VAR" in caplog.text

    def test_non_numeric_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that non-numeric input returns the default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("abc", "TEST_VAR", 1.0, min_val=0.0)
        assert result == 1.0
        assert "Invalid TEST_VAR: 'abc' is not a valid number" in caplog.text

    def test_empty_string_returns_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty string returns the default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("", "TEST_VAR", 1.0)
        assert result == 1.0

    def test_min_only_bound(self) -> None:
        """Test with only a minimum bound specified."""
        assert _parse_bounded_float("100.0", "TEST_VAR", 1.0, min_val=0.0) == 100.0

    def test_max_only_bound(self) -> None:
        """Test with only a maximum bound specified."""
        assert _parse_bounded_float("-5.0", "TEST_VAR", 1.0, max_val=100.0) == -5.0

    def test_negative_below_zero_min(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative value with min_val=0.0 logs 'must be >=' warning message."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("-1.0", "TEST_VAR", 5.0, min_val=0.0)
        assert result == 5.0
        assert "must be >= 0" in caplog.text

    def test_above_max_only_logs_must_be_lte(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that exceeding max_val with no min_val logs 'must be <=' warning message."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("150.0", "TEST_VAR", 50.0, max_val=100.0)
        assert result == 50.0
        assert "must be <= 100" in caplog.text

    def test_both_bounds_logs_range_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that violating bounds with both min and max set logs 'is not in range' message."""
        with caplog.at_level(logging.WARNING):
            result = _parse_bounded_float("1.5", "TEST_VAR", 0.5, min_val=0.0, max_val=1.0)
        assert result == 0.5
        assert "is not in range 0 to 1" in caplog.text


class TestFormatBoundMessage:
    """Tests for _format_bound_message helper function."""

    def test_both_bounds(self) -> None:
        """Test message when both min and max bounds are set."""
        assert _format_bound_message(min_val=0.0, max_val=1.0) == "is not in range 0 to 1"

    def test_only_min_bound(self) -> None:
        """Test message when only min bound is set."""
        assert _format_bound_message(min_val=0.0) == "must be >= 0"

    def test_only_max_bound(self) -> None:
        """Test message when only max bound is set."""
        assert _format_bound_message(max_val=100.0) == "must be <= 100"

    def test_no_bounds(self) -> None:
        """Test message when no bounds are set."""
        assert _format_bound_message() == "is out of range"

    def test_both_bounds_negative(self) -> None:
        """Test message with negative bound values."""
        assert _format_bound_message(min_val=-10.0, max_val=-1.0) == "is not in range -10 to -1"

    def test_only_min_bound_positive(self) -> None:
        """Test message when only min bound is set to a positive value."""
        assert _format_bound_message(min_val=5.0) == "must be >= 5"

    def test_only_max_bound_zero(self) -> None:
        """Test message when only max bound is set to zero."""
        assert _format_bound_message(max_val=0.0) == "must be <= 0"

    def test_both_bounds_with_decimals(self) -> None:
        """Test message when bounds have non-whole decimal values."""
        assert _format_bound_message(min_val=0.1, max_val=0.9) == "is not in range 0.1 to 0.9"

    def test_mixed_whole_and_decimal_bounds(self) -> None:
        """Test message with one whole and one decimal bound."""
        assert _format_bound_message(min_val=0.0, max_val=0.5) == "is not in range 0 to 0.5"


class TestFormatNumber:
    """Tests for _format_number helper function."""

    def test_whole_number_zero(self) -> None:
        """Test that 0.0 is formatted as '0'."""
        assert _format_number(0.0) == "0"

    def test_whole_number_positive(self) -> None:
        """Test that positive whole floats are formatted without decimal."""
        assert _format_number(5.0) == "5"

    def test_whole_number_negative(self) -> None:
        """Test that negative whole floats are formatted without decimal."""
        assert _format_number(-10.0) == "-10"

    def test_decimal_value(self) -> None:
        """Test that non-whole floats retain their decimal representation."""
        assert _format_number(0.5) == "0.5"

    def test_negative_decimal_value(self) -> None:
        """Test that negative non-whole floats retain their decimal representation."""
        assert _format_number(-3.14) == "-3.14"

    def test_large_whole_number(self) -> None:
        """Test that large whole floats are formatted without decimal."""
        assert _format_number(1000.0) == "1000"

    def test_small_decimal_value(self) -> None:
        """Test that small decimal values are preserved."""
        assert _format_number(0.1) == "0.1"

    def test_positive_infinity(self) -> None:
        """Test that positive infinity is formatted as 'inf' without error."""
        assert _format_number(float("inf")) == "inf"

    def test_negative_infinity(self) -> None:
        """Test that negative infinity is formatted as '-inf' without error."""
        assert _format_number(float("-inf")) == "-inf"

    def test_nan(self) -> None:
        """Test that NaN is formatted as 'nan' without error."""
        assert _format_number(float("nan")) == "nan"


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
        assert (
            f"Invalid TEST_VAR: 0 is not a valid port (must be {MIN_PORT}-{MAX_PORT})"
            in caplog.text
        )

    def test_invalid_port_negative(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that negative port numbers are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("-1", "TEST_VAR", 8080)
        assert result == 8080
        assert (
            f"Invalid TEST_VAR: -1 is not a valid port (must be {MIN_PORT}-{MAX_PORT})"
            in caplog.text
        )

    def test_invalid_port_too_high(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that port numbers above MAX_PORT are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port(str(MAX_PORT + 1), "TEST_VAR", 8080)
        assert result == 8080
        assert (
            f"Invalid TEST_VAR: {MAX_PORT + 1} is not a valid port (must be {MIN_PORT}-{MAX_PORT})"
            in caplog.text
        )

    def test_invalid_port_way_too_high(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that very large port numbers are rejected."""
        with caplog.at_level(logging.WARNING):
            result = _parse_port("100000", "TEST_VAR", 8080)
        assert result == 8080
        assert (
            f"Invalid TEST_VAR: 100000 is not a valid port (must be {MIN_PORT}-{MAX_PORT})"
            in caplog.text
        )

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

        assert config.dashboard.port == 8080

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that dashboard_port loads from environment variable."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "3000")

        config = load_config()

        assert config.dashboard.port == 3000

    def test_invalid_port_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that port 0 uses the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative ports use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "-1")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_port_too_high_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that ports above MAX_PORT use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MAX_PORT + 1))

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.port == 8080
        assert f"not a valid port (must be {MIN_PORT}-{MAX_PORT})" in caplog.text

    def test_invalid_non_numeric_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that non-numeric values use the default."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.port == 8080
        assert "Invalid SENTINEL_DASHBOARD_PORT" in caplog.text

    def test_valid_port_boundary_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that MIN_PORT is accepted."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MIN_PORT))

        config = load_config()

        assert config.dashboard.port == MIN_PORT

    def test_valid_port_boundary_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that MAX_PORT is accepted."""
        monkeypatch.setenv("SENTINEL_DASHBOARD_PORT", str(MAX_PORT))

        config = load_config()

        assert config.dashboard.port == MAX_PORT


class TestAttemptCountsTtlConfig:
    """Tests for attempt_counts_ttl configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that attempt_counts_ttl defaults to 3600 seconds (1 hour)."""
        monkeypatch.delenv("SENTINEL_ATTEMPT_COUNTS_TTL", raising=False)

        config = load_config()

        assert config.execution.attempt_counts_ttl == 3600

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that attempt_counts_ttl loads from environment variable."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "7200")

        config = load_config()

        assert config.execution.attempt_counts_ttl == 7200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.attempt_counts_ttl == 3600
        assert "Invalid SENTINEL_ATTEMPT_COUNTS_TTL" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.attempt_counts_ttl == 3600
        assert "not positive" in caplog.text

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_ATTEMPT_COUNTS_TTL", "-100")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.attempt_counts_ttl == 3600
        assert "not positive" in caplog.text


class TestMaxQueueSizeConfig:
    """Tests for max_queue_size configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_queue_size defaults to 100."""
        monkeypatch.delenv("SENTINEL_MAX_QUEUE_SIZE", raising=False)

        config = load_config()

        assert config.execution.max_queue_size == 100

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_queue_size loads from environment variable."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "200")

        config = load_config()

        assert config.execution.max_queue_size == 200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.max_queue_size == 100
        assert "Invalid SENTINEL_MAX_QUEUE_SIZE" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.max_queue_size == 100
        assert "not positive" in caplog.text

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_MAX_QUEUE_SIZE", "-50")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.max_queue_size == 100
        assert "not positive" in caplog.text


class TestDisableStreamingLogsConfig:
    """Tests for disable_streaming_logs configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs defaults to False."""
        monkeypatch.delenv("SENTINEL_DISABLE_STREAMING_LOGS", raising=False)

        config = load_config()

        assert config.execution.disable_streaming_logs is False

    def test_enabled_with_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with 'true'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "true")

        config = load_config()

        assert config.execution.disable_streaming_logs is True

    def test_enabled_with_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with '1'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "1")

        config = load_config()

        assert config.execution.disable_streaming_logs is True

    def test_enabled_with_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs can be enabled with 'yes'."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "yes")

        config = load_config()

        assert config.execution.disable_streaming_logs is True

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "TRUE")

        config = load_config()

        assert config.execution.disable_streaming_logs is True

    def test_false_for_other_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs is False for other values."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "false")

        config = load_config()

        assert config.execution.disable_streaming_logs is False

    def test_false_for_empty_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that disable_streaming_logs is False for empty value."""
        monkeypatch.setenv("SENTINEL_DISABLE_STREAMING_LOGS", "")

        config = load_config()

        assert config.execution.disable_streaming_logs is False


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

        assert config.cursor.default_agent_type == "claude"

    def test_default_agent_type_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_agent_type loads from environment variable."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "cursor")

        config = load_config()

        assert config.cursor.default_agent_type == "cursor"

    def test_cursor_path_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_path defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CURSOR_PATH", raising=False)

        config = load_config()

        assert config.cursor.path == ""

    def test_cursor_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_path loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_PATH", "/usr/local/bin/cursor")

        config = load_config()

        assert config.cursor.path == "/usr/local/bin/cursor"

    def test_cursor_default_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_model defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CURSOR_DEFAULT_MODEL", raising=False)

        config = load_config()

        assert config.cursor.default_model == ""

    def test_cursor_default_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_model loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODEL", "gpt-4")

        config = load_config()

        assert config.cursor.default_model == "gpt-4"

    def test_cursor_default_mode_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode defaults to 'agent'."""
        monkeypatch.delenv("SENTINEL_CURSOR_DEFAULT_MODE", raising=False)

        config = load_config()

        assert config.cursor.default_mode == "agent"

    def test_cursor_default_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "plan")

        config = load_config()

        assert config.cursor.default_mode == "plan"

    def test_cursor_default_mode_ask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode can be set to 'ask'."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "ask")

        config = load_config()

        assert config.cursor.default_mode == "ask"

    def test_cursor_default_mode_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cursor_default_mode parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "PLAN")

        config = load_config()

        assert config.cursor.default_mode == "plan"

    def test_cursor_default_mode_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid cursor_default_mode uses the default."""
        monkeypatch.setenv("SENTINEL_CURSOR_DEFAULT_MODE", "invalid")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.cursor.default_mode == "agent"
        assert "Invalid SENTINEL_CURSOR_DEFAULT_MODE" in caplog.text

    def test_default_agent_type_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_agent_type parsing is case-insensitive."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "CURSOR")

        config = load_config()

        assert config.cursor.default_agent_type == "cursor"

    def test_default_agent_type_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid default_agent_type uses the default."""
        monkeypatch.setenv("SENTINEL_DEFAULT_AGENT_TYPE", "invalid")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.cursor.default_agent_type == "claude"
        assert "Invalid SENTINEL_DEFAULT_AGENT_TYPE" in caplog.text


class TestCodexConfig:
    """Tests for Codex CLI configuration."""

    def test_codex_path_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that codex_path defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CODEX_PATH", raising=False)

        config = load_config()

        assert config.codex.path == ""

    def test_codex_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that codex_path loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CODEX_PATH", "/usr/local/bin/codex")

        config = load_config()

        assert config.codex.path == "/usr/local/bin/codex"

    def test_codex_default_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that codex_default_model defaults to empty string."""
        monkeypatch.delenv("SENTINEL_CODEX_DEFAULT_MODEL", raising=False)

        config = load_config()

        assert config.codex.default_model == ""

    def test_codex_default_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that codex_default_model loads from environment variable."""
        monkeypatch.setenv("SENTINEL_CODEX_DEFAULT_MODEL", "o3-mini")

        config = load_config()

        assert config.codex.default_model == "o3-mini"


class TestInterMessageTimesThresholdConfig:
    """Tests for inter_message_times_threshold configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that inter_message_times_threshold defaults to 100."""
        monkeypatch.delenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", raising=False)

        config = load_config()

        assert config.execution.inter_message_times_threshold == 100

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that inter_message_times_threshold loads from environment variable."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "200")

        config = load_config()

        assert config.execution.inter_message_times_threshold == 200

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.inter_message_times_threshold == 100
        assert "Invalid SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD" in caplog.text

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.inter_message_times_threshold == 100
        assert "not positive" in caplog.text


class TestToggleCooldownConfig:
    """Tests for toggle_cooldown_seconds configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that toggle_cooldown_seconds defaults to 2.0."""
        monkeypatch.delenv("SENTINEL_TOGGLE_COOLDOWN", raising=False)

        config = load_config()

        assert config.dashboard.toggle_cooldown_seconds == 2.0

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that toggle_cooldown_seconds loads from environment variable."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "5.0")

        config = load_config()

        assert config.dashboard.toggle_cooldown_seconds == 5.0

    def test_zero_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that zero is allowed (to disable cooldown)."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "0")

        config = load_config()

        assert config.dashboard.toggle_cooldown_seconds == 0.0

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative values use the default."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "-1.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.toggle_cooldown_seconds == 2.0
        assert "must be >= 0" in caplog.text

    def test_invalid_value_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid values use the default."""
        monkeypatch.setenv("SENTINEL_TOGGLE_COOLDOWN", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.toggle_cooldown_seconds == 2.0
        assert "Invalid SENTINEL_TOGGLE_COOLDOWN" in caplog.text


class TestRateLimitCacheTtlConfig:
    """Tests for rate_limit_cache_ttl configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_ttl defaults to 3600."""
        monkeypatch.delenv("SENTINEL_RATE_LIMIT_CACHE_TTL", raising=False)

        config = load_config()

        assert config.dashboard.rate_limit_cache_ttl == 3600

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_ttl loads from environment variable."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_TTL", "7200")

        config = load_config()

        assert config.dashboard.rate_limit_cache_ttl == 7200

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_TTL", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.rate_limit_cache_ttl == 3600
        assert "not positive" in caplog.text


class TestRateLimitCacheMaxsizeConfig:
    """Tests for rate_limit_cache_maxsize configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_maxsize defaults to 10000."""
        monkeypatch.delenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", raising=False)

        config = load_config()

        assert config.dashboard.rate_limit_cache_maxsize == 10000

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rate_limit_cache_maxsize loads from environment variable."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", "50000")

        config = load_config()

        assert config.dashboard.rate_limit_cache_maxsize == 50000

    def test_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.rate_limit_cache_maxsize == 10000
        assert "not positive" in caplog.text


class TestValidateBranchName:
    """Tests for _validate_branch_name helper function."""

    def test_valid_branch_names(self) -> None:
        """Test that valid branch names are accepted."""
        assert _validate_branch_name("main") == "main"
        assert _validate_branch_name("master") == "master"
        assert _validate_branch_name("develop") == "develop"
        assert _validate_branch_name("feature/test") == "feature/test"
        assert _validate_branch_name("release-1.0") == "release-1.0"
        assert _validate_branch_name("hotfix_123") == "hotfix_123"

    def test_empty_string_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that empty string uses the default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("")
        assert result == "main"
        assert "cannot be empty" in caplog.text

    def test_whitespace_only_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that whitespace-only string uses the default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("   ")
        assert result == "main"
        assert "cannot be empty" in caplog.text

    def test_starts_with_hyphen_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name starting with hyphen uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("-feature")
        assert result == "main"
        assert "cannot start with '-' or '.'" in caplog.text

    def test_starts_with_period_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name starting with period uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name(".hidden")
        assert result == "main"
        assert "cannot start with '-' or '.'" in caplog.text

    def test_ends_with_period_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name ending with period uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature.")
        assert result == "main"
        assert "cannot end with '.' or '/'" in caplog.text

    def test_ends_with_slash_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name ending with slash uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature/")
        assert result == "main"
        assert "cannot end with '.' or '/'" in caplog.text

    def test_contains_space_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing space uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_tilde_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing tilde uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature~test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_caret_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing caret uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature^test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_colon_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing colon uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature:test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_question_mark_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing question mark uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature?test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_asterisk_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing asterisk uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature*test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_open_bracket_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing open bracket uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature[test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_backslash_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name containing backslash uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature\\test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_contains_at_brace_sequence_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that branch name containing @{ sequence uses default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature@{test")
        assert result == "main"
        assert "invalid characters" in caplog.text

    def test_at_without_brace_is_valid(self) -> None:
        """Test that @ without following { is valid."""
        assert _validate_branch_name("feature@test") == "feature@test"

    def test_consecutive_periods_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that consecutive periods use default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature..test")
        assert result == "main"
        assert "consecutive periods" in caplog.text

    def test_consecutive_slashes_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that consecutive slashes use default."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature//test")
        assert result == "main"
        assert "consecutive" in caplog.text and "slashes" in caplog.text

    def test_custom_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that custom default is used when provided."""
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("", default="develop")
        assert result == "develop"

    def test_ends_with_lock_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that branch name ending with .lock uses default.

        Git disallows branch names ending in .lock as they conflict
        with git's internal lock files.
        """
        with caplog.at_level(logging.WARNING):
            result = _validate_branch_name("feature.lock")
        assert result == "main"
        assert ".lock" in caplog.text

    def test_lock_in_middle_is_valid(self) -> None:
        """Test that .lock in the middle of a branch name is valid."""
        assert _validate_branch_name("feature.lock.test") == "feature.lock.test"


class TestDefaultBaseBranchConfig:
    """Tests for default_base_branch configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_base_branch defaults to 'main'."""
        monkeypatch.delenv("SENTINEL_DEFAULT_BASE_BRANCH", raising=False)

        config = load_config()

        assert config.execution.default_base_branch == "main"

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default_base_branch loads from environment variable."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "develop")

        config = load_config()

        assert config.execution.default_base_branch == "develop"

    def test_master_branch_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that 'master' is a valid branch name."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "master")

        config = load_config()

        assert config.execution.default_base_branch == "master"

    def test_feature_branch_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that feature branch names are valid."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "feature/test")

        config = load_config()

        assert config.execution.default_base_branch == "feature/test"

    def test_invalid_branch_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid branch names use the default."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "feature..test")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.default_base_branch == "main"
        assert "Invalid SENTINEL_DEFAULT_BASE_BRANCH" in caplog.text

    def test_empty_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that empty value uses the default."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.execution.default_base_branch == "main"
        assert "cannot be empty" in caplog.text

    def test_whitespace_trimmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that whitespace is trimmed from branch names."""
        monkeypatch.setenv("SENTINEL_DEFAULT_BASE_BRANCH", "  develop  ")

        config = load_config()

        assert config.execution.default_base_branch == "develop"


class TestJiraEpicLinkFieldConfig:
    """Tests for jira_epic_link_field configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that jira_epic_link_field defaults to 'customfield_10014'."""
        monkeypatch.delenv("JIRA_EPIC_LINK_FIELD", raising=False)

        config = load_config()

        assert config.jira.epic_link_field == "customfield_10014"

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that jira_epic_link_field loads from environment variable."""
        monkeypatch.setenv("JIRA_EPIC_LINK_FIELD", "customfield_12345")

        config = load_config()

        assert config.jira.epic_link_field == "customfield_12345"

    def test_custom_field_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that any custom field ID can be used."""
        monkeypatch.setenv("JIRA_EPIC_LINK_FIELD", "customfield_99999")

        config = load_config()

        assert config.jira.epic_link_field == "customfield_99999"

    def test_empty_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty value falls back to default."""
        monkeypatch.setenv("JIRA_EPIC_LINK_FIELD", "")

        config = load_config()

        # Empty string is valid, just means no fallback to classic epic link
        assert config.jira.epic_link_field == ""


class TestSuccessRateGreenThreshold:
    """Tests for success_rate_green_threshold configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that success_rate_green_threshold defaults to DEFAULT_GREEN_THRESHOLD."""
        monkeypatch.delenv("SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", raising=False)

        config = load_config()

        assert config.dashboard.success_rate_green_threshold == DEFAULT_GREEN_THRESHOLD

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that success_rate_green_threshold loads from environment variable."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", "95.0")

        config = load_config()

        assert config.dashboard.success_rate_green_threshold == 95.0

    def test_zero_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that zero is a valid threshold value."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", "0.0")

        config = load_config()

        assert config.dashboard.success_rate_green_threshold == 0.0

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative value uses the default."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", "-10.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.success_rate_green_threshold == DEFAULT_GREEN_THRESHOLD
        assert "must be >= 0" in caplog.text

    def test_invalid_string_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid string uses the default."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.success_rate_green_threshold == DEFAULT_GREEN_THRESHOLD
        assert "not a valid number" in caplog.text


class TestSuccessRateYellowThreshold:
    """Tests for success_rate_yellow_threshold configuration."""

    def test_default_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that success_rate_yellow_threshold defaults to DEFAULT_YELLOW_THRESHOLD."""
        monkeypatch.delenv("SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", raising=False)

        config = load_config()

        assert config.dashboard.success_rate_yellow_threshold == DEFAULT_YELLOW_THRESHOLD

    def test_loads_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that success_rate_yellow_threshold loads from environment variable."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", "80.0")

        config = load_config()

        assert config.dashboard.success_rate_yellow_threshold == 80.0

    def test_zero_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that zero is a valid threshold value."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", "0.0")

        config = load_config()

        assert config.dashboard.success_rate_yellow_threshold == 0.0

    def test_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative value uses the default."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", "-5.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.success_rate_yellow_threshold == DEFAULT_YELLOW_THRESHOLD
        assert "must be >= 0" in caplog.text

    def test_invalid_string_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid string uses the default."""
        monkeypatch.setenv("SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", "abc")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.dashboard.success_rate_yellow_threshold == DEFAULT_YELLOW_THRESHOLD
        assert "not a valid number" in caplog.text


class TestServiceHealthGateConfig:
    """Tests for ServiceHealthGateConfig load_config integration."""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that all service health gate fields default correctly."""
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_ENABLED", raising=False)
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", raising=False)
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", raising=False)
        monkeypatch.delenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", raising=False)

        config = load_config()

        assert config.service_health_gate.enabled is True
        assert config.service_health_gate.failure_threshold == 3
        assert config.service_health_gate.initial_probe_interval == 30.0
        assert config.service_health_gate.max_probe_interval == 300.0
        assert config.service_health_gate.probe_backoff_factor == 2.0
        assert config.service_health_gate.probe_timeout == 5.0

    def test_enabled_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that enabled loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_ENABLED", "false")

        config = load_config()

        assert config.service_health_gate.enabled is False

    def test_enabled_true_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that enabled=true loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_ENABLED", "true")

        config = load_config()

        assert config.service_health_gate.enabled is True

    def test_failure_threshold_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that failure_threshold loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", "5")

        config = load_config()

        assert config.service_health_gate.failure_threshold == 5

    def test_failure_threshold_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid failure_threshold uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", "not-a-number")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.failure_threshold == 3
        assert "Invalid SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD" in caplog.text

    def test_failure_threshold_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that zero failure_threshold uses the default (must be positive)."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", "0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.failure_threshold == 3
        assert "not positive" in caplog.text

    def test_initial_probe_interval_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that initial_probe_interval loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "60.0")

        config = load_config()

        assert config.service_health_gate.initial_probe_interval == 60.0

    def test_initial_probe_interval_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid initial_probe_interval uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "abc")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 30.0
        assert "Invalid SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL" in caplog.text

    def test_initial_probe_interval_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative initial_probe_interval uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "-10.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 30.0
        assert "must be >= 0" in caplog.text

    def test_max_probe_interval_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that max_probe_interval loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "600.0")

        config = load_config()

        assert config.service_health_gate.max_probe_interval == 600.0

    def test_max_probe_interval_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid max_probe_interval uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "xyz")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.max_probe_interval == 300.0
        assert "Invalid SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL" in caplog.text

    def test_probe_backoff_factor_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that probe_backoff_factor loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "3.0")

        config = load_config()

        assert config.service_health_gate.probe_backoff_factor == 3.0

    def test_probe_backoff_factor_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid probe_backoff_factor uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "bad")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.probe_backoff_factor == 2.0
        assert "Invalid SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR" in caplog.text

    def test_probe_backoff_factor_below_minimum_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that probe_backoff_factor below 1.0 uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "0.5")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.probe_backoff_factor == 2.0
        assert "Invalid SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR" in caplog.text

    def test_probe_backoff_factor_zero_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that probe_backoff_factor of 0.0 uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "0.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.probe_backoff_factor == 2.0
        assert "Invalid SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR" in caplog.text

    def test_probe_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that probe_timeout loads from environment variable."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", "10.0")

        config = load_config()

        assert config.service_health_gate.probe_timeout == 10.0

    def test_probe_timeout_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid probe_timeout uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", "nope")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.probe_timeout == 5.0
        assert "Invalid SENTINEL_HEALTH_GATE_PROBE_TIMEOUT" in caplog.text

    def test_probe_timeout_negative_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that negative probe_timeout uses the default."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", "-1.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.probe_timeout == 5.0
        assert "must be >= 0" in caplog.text

    def test_all_fields_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading all service health gate fields from environment variables."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_ENABLED", "false")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", "10")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "15.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "120.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "1.5")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", "3.0")

        config = load_config()

        assert config.service_health_gate.enabled is False
        assert config.service_health_gate.failure_threshold == 10
        assert config.service_health_gate.initial_probe_interval == 15.0
        assert config.service_health_gate.max_probe_interval == 120.0
        assert config.service_health_gate.probe_backoff_factor == 1.5
        assert config.service_health_gate.probe_timeout == 3.0

    def test_initial_probe_interval_exceeds_max_swaps_values(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that initial > max probe interval triggers swap and warning."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "500.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "100.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        # Values should be swapped
        assert config.service_health_gate.initial_probe_interval == 100.0
        assert config.service_health_gate.max_probe_interval == 500.0
        assert "SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL (500) exceeds" in caplog.text
        assert "SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL (100)" in caplog.text
        assert "swapping values" in caplog.text

    def test_initial_probe_interval_equals_max_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that initial == max probe interval does not trigger a warning."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "100.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "100.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 100.0
        assert config.service_health_gate.max_probe_interval == 100.0
        assert "swapping values" not in caplog.text

    def test_initial_probe_interval_less_than_max_no_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that initial < max probe interval does not trigger a warning."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "10.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "600.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 10.0
        assert config.service_health_gate.max_probe_interval == 600.0
        assert "swapping values" not in caplog.text

    def test_cross_field_validation_with_decimal_values(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test cross-field validation with non-whole decimal values."""
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "45.5")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "20.5")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        # Values should be swapped
        assert config.service_health_gate.initial_probe_interval == 20.5
        assert config.service_health_gate.max_probe_interval == 45.5
        assert "SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL (45.5) exceeds" in caplog.text
        assert "SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL (20.5)" in caplog.text

    def test_small_delta_initial_exceeds_max_triggers_swap(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test floating-point comparison with very small delta near boundary.

        When initial_probe_interval exceeds max_probe_interval by a tiny amount
        (e.g., 30.01 vs 30.0), the cross-field validation should still detect
        the violation and swap the values.
        """
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "30.01")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "30.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        # Values should be swapped since 30.01 > 30.0
        assert config.service_health_gate.initial_probe_interval == 30.0
        assert config.service_health_gate.max_probe_interval == 30.01
        assert "swapping values" in caplog.text

    def test_very_large_values_no_overflow(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that very large probe interval values do not cause overflow.

        Ensures cross-field validation handles large floating-point values
        correctly without arithmetic overflow or precision loss.
        """
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "1e10")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "1e15")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        # Both values are valid and initial < max, so no swap should occur
        assert config.service_health_gate.initial_probe_interval == 1e10
        assert config.service_health_gate.max_probe_interval == 1e15
        assert "swapping values" not in caplog.text

    def test_very_large_values_swapped_when_initial_exceeds_max(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that very large values are correctly swapped when initial > max.

        Validates that the cross-field comparison works correctly with large
        floating-point numbers.
        """
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "1e15")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "1e10")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        # Values should be swapped
        assert config.service_health_gate.initial_probe_interval == 1e10
        assert config.service_health_gate.max_probe_interval == 1e15
        assert "swapping values" in caplog.text

    def test_minimum_valid_values_both_zero(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test cross-field validation with both values at minimum (0.0).

        When both initial_probe_interval and max_probe_interval are 0.0,
        they are equal, so no swap should occur.
        """
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "0.0")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "0.0")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 0.0
        assert config.service_health_gate.max_probe_interval == 0.0
        assert "swapping values" not in caplog.text

    def test_minimum_valid_values_near_zero(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test cross-field validation with values very close to zero.

        Ensures that very small positive values near the minimum boundary
        are handled correctly in cross-field comparison.
        """
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "0.001")
        monkeypatch.setenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "0.002")

        with caplog.at_level(logging.WARNING):
            config = load_config()

        assert config.service_health_gate.initial_probe_interval == 0.001
        assert config.service_health_gate.max_probe_interval == 0.002
        assert "swapping values" not in caplog.text
