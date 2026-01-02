"""Tests for configuration module."""

import logging
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from sentinel.config import (
    VALID_LOG_LEVELS,
    Config,
    _parse_bool,
    _parse_positive_int,
    _validate_log_level,
    load_config,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self) -> None:
        config = Config()
        assert config.poll_interval == 60
        assert config.max_issues_per_poll == 50
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
