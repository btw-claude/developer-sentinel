"""Tests for configuration module."""

from pathlib import Path

import pytest

from sentinel.config import Config, load_config


class TestConfig:
    """Tests for Config dataclass."""

    def test_defaults(self) -> None:
        config = Config()
        assert config.poll_interval == 60
        assert config.max_issues_per_poll == 50
        assert config.jira_url == ""
        assert config.jira_user == ""
        assert config.jira_api_token == ""
        assert config.log_level == "INFO"
        assert config.orchestrations_dir == Path("./orchestrations")

    def test_custom_values(self) -> None:
        config = Config(
            poll_interval=120,
            max_issues_per_poll=100,
            jira_url="https://jira.example.com",
            log_level="DEBUG",
        )
        assert config.poll_interval == 120
        assert config.max_issues_per_poll == 100
        assert config.jira_url == "https://jira.example.com"
        assert config.log_level == "DEBUG"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)
        monkeypatch.delenv("SENTINEL_MAX_ISSUES", raising=False)
        monkeypatch.delenv("JIRA_URL", raising=False)
        monkeypatch.delenv("SENTINEL_LOG_LEVEL", raising=False)

        config = load_config()

        assert config.poll_interval == 60
        assert config.max_issues_per_poll == 50
        assert config.log_level == "INFO"

    def test_loads_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_POLL_INTERVAL", "30")
        monkeypatch.setenv("SENTINEL_MAX_ISSUES", "25")
        monkeypatch.setenv("JIRA_URL", "https://jira.test.com")
        monkeypatch.setenv("JIRA_USER", "testuser")
        monkeypatch.setenv("JIRA_API_TOKEN", "secret123")
        monkeypatch.setenv("SENTINEL_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("SENTINEL_ORCHESTRATIONS_DIR", "/custom/path")

        config = load_config()

        assert config.poll_interval == 30
        assert config.max_issues_per_poll == 25
        assert config.jira_url == "https://jira.test.com"
        assert config.jira_user == "testuser"
        assert config.jira_api_token == "secret123"
        assert config.log_level == "DEBUG"
        assert config.orchestrations_dir == Path("/custom/path")

    def test_loads_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear existing env vars
        monkeypatch.delenv("SENTINEL_POLL_INTERVAL", raising=False)

        # Create a test .env file
        env_file = tmp_path / ".env"
        env_file.write_text("SENTINEL_POLL_INTERVAL=45\nSENTINEL_LOG_LEVEL=WARNING\n")

        config = load_config(env_file)

        assert config.poll_interval == 45
        assert config.log_level == "WARNING"
