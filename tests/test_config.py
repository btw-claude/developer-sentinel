"""Tests for configuration loading."""

import os
from unittest.mock import patch

from sentinel.config import Config, load_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = Config()
        assert config.log_level == "INFO"
        assert config.poll_interval_seconds == 60
        assert config.max_retries == 3


class TestLoadConfig:
    """Tests for load_config function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_config(self) -> None:
        """Should load config with defaults when no env vars set."""
        config = load_config()

        assert config.log_level == "INFO"
        assert config.poll_interval_seconds == 60
        assert config.max_retries == 3

    @patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "debug",
            "POLL_INTERVAL_SECONDS": "120",
            "MAX_RETRIES": "5",
        },
        clear=True,
    )
    def test_custom_config(self) -> None:
        """Should load custom values from environment."""
        config = load_config()

        assert config.log_level == "DEBUG"  # Should be uppercased
        assert config.poll_interval_seconds == 120
        assert config.max_retries == 5

    @patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "WARNING",
        },
        clear=True,
    )
    def test_partial_config(self) -> None:
        """Should use defaults for missing values."""
        config = load_config()

        assert config.log_level == "WARNING"
        assert config.poll_interval_seconds == 60  # default
        assert config.max_retries == 3  # default
