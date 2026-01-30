"""Tests for health check functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sentinel.health import (
    HealthCheckConfig,
    HealthCheckContext,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    ServiceHealth,
)


class TestHealthCheckConfig:
    """Tests for HealthCheckConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HealthCheckConfig()
        assert config.timeout == 5.0
        assert config.enabled is True

    def test_from_config_with_app_config(self) -> None:
        """Test creating HealthCheckConfig from application Config."""
        from sentinel.config import Config

        app_config = Config(health_check_timeout=10.0, health_check_enabled=False)
        health_config = HealthCheckConfig.from_config(app_config)

        assert health_config.timeout == 10.0
        assert health_config.enabled is False

    def test_from_config_with_default_app_config(self) -> None:
        """Test creating HealthCheckConfig from default application Config."""
        from sentinel.config import Config

        app_config = Config()
        health_config = HealthCheckConfig.from_config(app_config)

        assert health_config.timeout == 5.0
        assert health_config.enabled is True

    def test_from_env_defaults(self) -> None:
        """Test loading from environment with no variables set."""
        with patch.dict("os.environ", {}, clear=True):
            config = HealthCheckConfig.from_env()
            assert config.timeout == 5.0
            assert config.enabled is True

    def test_from_env_custom_values(self) -> None:
        """Test loading from environment with custom values."""
        with patch.dict(
            "os.environ",
            {
                "SENTINEL_HEALTH_CHECK_TIMEOUT": "10.0",
                "SENTINEL_HEALTH_CHECK_ENABLED": "false",
            },
        ):
            config = HealthCheckConfig.from_env()
            assert config.timeout == 10.0
            assert config.enabled is False

    def test_from_env_invalid_timeout(self) -> None:
        """Test loading from environment with invalid timeout."""
        with patch.dict(
            "os.environ",
            {"SENTINEL_HEALTH_CHECK_TIMEOUT": "invalid"},
        ):
            config = HealthCheckConfig.from_env()
            assert config.timeout == 5.0  # Falls back to default

    def test_from_env_negative_timeout(self) -> None:
        """Test loading from environment with negative timeout."""
        with patch.dict(
            "os.environ",
            {"SENTINEL_HEALTH_CHECK_TIMEOUT": "-1.0"},
        ):
            config = HealthCheckConfig.from_env()
            assert config.timeout == 5.0  # Falls back to default


class TestServiceHealth:
    """Tests for ServiceHealth."""

    def test_to_dict_basic(self) -> None:
        """Test basic to_dict conversion."""
        health = ServiceHealth(
            status=HealthStatus.UP,
            latency_ms=45.123,
        )
        result = health.to_dict()
        assert result == {
            "status": "up",
            "latency_ms": 45.12,
        }

    def test_to_dict_with_error(self) -> None:
        """Test to_dict conversion with error."""
        health = ServiceHealth(
            status=HealthStatus.DOWN,
            latency_ms=0.0,
            error="Connection refused",
        )
        result = health.to_dict()
        assert result == {
            "status": "down",
            "latency_ms": 0.0,
            "error": "Connection refused",
        }


class TestHealthCheckContext:
    """Tests for HealthCheckContext."""

    def test_elapsed_ms_calculation(self) -> None:
        """Test that elapsed_ms calculates time correctly."""
        ctx = HealthCheckContext("test")
        with ctx.handle_exceptions():
            pass  # No exception, just test timing
        # elapsed_ms should return a non-negative value
        assert ctx.elapsed_ms() >= 0

    def test_handle_exceptions_no_error(self) -> None:
        """Test context manager with no exception."""
        ctx = HealthCheckContext("test")
        with ctx.handle_exceptions():
            pass  # No exception
        assert ctx.result is None

    def test_handle_exceptions_attribute_error(self) -> None:
        """Test context manager handles AttributeError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise AttributeError("'NoneType' object has no attribute 'foo'")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Configuration error" in ctx.result.error

    def test_handle_exceptions_type_error(self) -> None:
        """Test context manager handles TypeError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise TypeError("expected str, got int")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Data error" in ctx.result.error

    def test_handle_exceptions_value_error(self) -> None:
        """Test context manager handles ValueError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise ValueError("invalid value")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Data error" in ctx.result.error

    def test_handle_exceptions_key_error(self) -> None:
        """Test context manager handles KeyError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise KeyError("missing_key")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Data error" in ctx.result.error

    def test_handle_exceptions_os_error(self) -> None:
        """Test context manager handles OSError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise OSError("Network unreachable")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "OS error" in ctx.result.error

    def test_handle_exceptions_runtime_error(self) -> None:
        """Test context manager handles RuntimeError."""
        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            raise RuntimeError("Event loop is closed")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Runtime error" in ctx.result.error

    def test_handle_exceptions_generic_exception(self) -> None:
        """Test context manager handles unexpected exceptions."""
        ctx = HealthCheckContext("TestService")

        class CustomError(Exception):
            pass

        with ctx.handle_exceptions():
            raise CustomError("Something unexpected")

        assert ctx.result is not None
        assert ctx.result.status == HealthStatus.DOWN
        assert "Unexpected error" in ctx.result.error
        assert "CustomError" in ctx.result.error

    def test_handle_exceptions_latency_is_tracked(self) -> None:
        """Test that latency is tracked even when exceptions occur."""
        import time

        ctx = HealthCheckContext("TestService")
        with ctx.handle_exceptions():
            time.sleep(0.01)  # Sleep for 10ms
            raise ValueError("test error")

        assert ctx.result is not None
        # Latency should be at least 10ms (we slept for 10ms)
        assert ctx.result.latency_ms >= 10.0


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_to_dict_empty_checks(self) -> None:
        """Test to_dict with no checks."""
        result = HealthCheckResult(
            status="healthy",
            checks={},
            timestamp=1706472123.456,
        )
        output = result.to_dict()
        assert output["status"] == "healthy"
        assert output["timestamp"] == 1706472123.456
        assert output["checks"] == {}

    def test_to_dict_with_checks(self) -> None:
        """Test to_dict with multiple checks."""
        result = HealthCheckResult(
            status="degraded",
            checks={
                "jira": ServiceHealth(status=HealthStatus.UP, latency_ms=45.0),
                "github": ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=0.0,
                    error="Connection refused",
                ),
            },
            timestamp=1706472123.456,
        )
        output = result.to_dict()
        assert output["status"] == "degraded"
        assert output["checks"]["jira"] == {"status": "up", "latency_ms": 45.0}
        assert output["checks"]["github"] == {
            "status": "down",
            "latency_ms": 0.0,
            "error": "Connection refused",
        }


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_timeout_buffer_constant_is_defined(self) -> None:
        """Test that TIMEOUT_BUFFER_SECONDS constant is defined on HealthChecker."""
        assert hasattr(HealthChecker, "TIMEOUT_BUFFER_SECONDS")
        assert HealthChecker.TIMEOUT_BUFFER_SECONDS == 1.0

    def test_timeout_buffer_constant_is_documented(self) -> None:
        """Test that TIMEOUT_BUFFER_SECONDS usage is evident in class docstring or comments.

        The constant should be used in check_readiness to add buffer to individual
        check timeouts when using asyncio.wait_for().
        """
        import inspect

        source = inspect.getsource(HealthChecker)
        # Verify the constant is used in the source
        assert "TIMEOUT_BUFFER_SECONDS" in source
        # Verify it's used in the wait_for call
        assert "self.TIMEOUT_BUFFER_SECONDS" in source

    def test_from_config_factory_method(self) -> None:
        """Test creating HealthChecker from application Config."""
        from sentinel.config import Config

        app_config = Config(health_check_timeout=10.0, health_check_enabled=False)
        checker = HealthChecker.from_config(app_config)

        assert checker.config.timeout == 10.0
        assert checker.config.enabled is False

    def test_from_config_with_clients(self) -> None:
        """Test creating HealthChecker from Config with optional clients."""
        from sentinel.config import Config

        mock_jira_client = MagicMock()
        mock_github_client = MagicMock()

        app_config = Config()
        checker = HealthChecker.from_config(
            app_config,
            jira_client=mock_jira_client,
            github_client=mock_github_client,
            claude_api_key="test-key",
        )

        assert checker.jira_client is mock_jira_client
        assert checker.github_client is mock_github_client
        assert checker.claude_api_key == "test-key"

    def test_from_config_uses_config_values(self) -> None:
        """Test that from_config creates checker using Config health check settings."""
        from sentinel.config import Config

        app_config = Config(health_check_timeout=15.0, health_check_enabled=True)
        checker = HealthChecker.from_config(app_config)

        # The checker should have a config with values from the app config
        assert checker.config.timeout == 15.0
        assert checker.config.enabled is True

    def test_check_liveness(self) -> None:
        """Test liveness check returns healthy status."""
        checker = HealthChecker()
        result = checker.check_liveness()
        assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_check_readiness_no_clients(self) -> None:
        """Test readiness check with no clients configured."""
        checker = HealthChecker()
        result = await checker.check_readiness()
        assert result.status == "healthy"
        assert result.checks == {}

    @pytest.mark.asyncio
    async def test_check_readiness_disabled(self) -> None:
        """Test readiness check when disabled."""
        config = HealthCheckConfig(enabled=False)
        checker = HealthChecker(config=config)
        result = await checker.check_readiness()
        assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_check_jira_success(self) -> None:
        """Test Jira health check success."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        checker = HealthChecker(jira_client=mock_jira_client)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker._check_jira()

            assert result.status == HealthStatus.UP
            assert result.latency_ms >= 0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_check_jira_timeout(self) -> None:
        """Test Jira health check timeout."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        checker = HealthChecker(jira_client=mock_jira_client)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker._check_jira()

            assert result.status == HealthStatus.DOWN
            assert result.error == "Connection timed out"

    @pytest.mark.asyncio
    async def test_check_jira_not_configured(self) -> None:
        """Test Jira health check when not configured."""
        checker = HealthChecker(jira_client=None)
        result = await checker._check_jira()
        assert result.status == HealthStatus.DOWN
        assert result.error == "Jira client not configured"

    @pytest.mark.asyncio
    async def test_check_github_success(self) -> None:
        """Test GitHub health check success."""
        mock_github_client = MagicMock()
        mock_github_client.base_url = "https://api.github.com"
        mock_github_client.token = "ghp_test_token"

        checker = HealthChecker(github_client=mock_github_client)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker._check_github()

            assert result.status == HealthStatus.UP
            assert result.latency_ms >= 0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_check_github_not_configured(self) -> None:
        """Test GitHub health check when not configured."""
        checker = HealthChecker(github_client=None)
        result = await checker._check_github()
        assert result.status == HealthStatus.DOWN
        assert result.error == "GitHub client not configured"

    @pytest.mark.asyncio
    async def test_check_claude_success(self) -> None:
        """Test Claude health check success."""
        checker = HealthChecker(claude_api_key="sk-ant-test-key")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker._check_claude()

            assert result.status == HealthStatus.UP
            assert result.latency_ms >= 0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_check_claude_not_configured(self) -> None:
        """Test Claude health check when not configured."""
        with patch.dict("os.environ", {}, clear=True):
            checker = HealthChecker(claude_api_key="")
            result = await checker._check_claude()
            assert result.status == HealthStatus.DOWN
            assert result.error == "Claude API key not configured"

    @pytest.mark.asyncio
    async def test_check_readiness_all_healthy(self) -> None:
        """Test readiness check with all services healthy."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        mock_github_client = MagicMock()
        mock_github_client.base_url = "https://api.github.com"
        mock_github_client.token = "ghp_test_token"

        checker = HealthChecker(
            jira_client=mock_jira_client,
            github_client=mock_github_client,
            claude_api_key="sk-ant-test-key",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker.check_readiness()

            assert result.status == "healthy"
            assert "jira" in result.checks
            assert "github" in result.checks
            assert "claude" in result.checks
            assert result.checks["jira"].status == HealthStatus.UP
            assert result.checks["github"].status == HealthStatus.UP
            assert result.checks["claude"].status == HealthStatus.UP

    @pytest.mark.asyncio
    async def test_check_readiness_degraded(self) -> None:
        """Test readiness check with some services unhealthy."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        mock_github_client = MagicMock()
        mock_github_client.base_url = "https://api.github.com"
        mock_github_client.token = "ghp_test_token"

        checker = HealthChecker(
            jira_client=mock_jira_client,
            github_client=mock_github_client,
            claude_api_key="sk-ant-test-key",
        )

        # Mock _check_jira to return healthy
        async def mock_check_jira() -> ServiceHealth:
            return ServiceHealth(status=HealthStatus.UP, latency_ms=50.0)

        # Mock _check_github to return unhealthy
        async def mock_check_github() -> ServiceHealth:
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=0.0,
                error="Connection refused",
            )

        # Mock _check_claude to return healthy
        async def mock_check_claude() -> ServiceHealth:
            return ServiceHealth(status=HealthStatus.UP, latency_ms=100.0)

        with (
            patch.object(checker, "_check_jira", mock_check_jira),
            patch.object(checker, "_check_github", mock_check_github),
            patch.object(checker, "_check_claude", mock_check_claude),
        ):
            result = await checker.check_readiness()

            assert result.status == "degraded"
            assert result.checks["jira"].status == HealthStatus.UP
            assert result.checks["github"].status == HealthStatus.DOWN
            assert result.checks["claude"].status == HealthStatus.UP

    @pytest.mark.asyncio
    async def test_check_readiness_all_unhealthy(self) -> None:
        """Test readiness check with all services unhealthy."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        mock_github_client = MagicMock()
        mock_github_client.base_url = "https://api.github.com"
        mock_github_client.token = "ghp_test_token"

        checker = HealthChecker(
            jira_client=mock_jira_client,
            github_client=mock_github_client,
            claude_api_key="sk-ant-test-key",
        )

        # Mock all checks to return unhealthy
        async def mock_check_down() -> ServiceHealth:
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=0.0,
                error="Connection refused",
            )

        with (
            patch.object(checker, "_check_jira", mock_check_down),
            patch.object(checker, "_check_github", mock_check_down),
            patch.object(checker, "_check_claude", mock_check_down),
        ):
            result = await checker.check_readiness()

            assert result.status == "unhealthy"
            assert all(check.status == HealthStatus.DOWN for check in result.checks.values())

    @pytest.mark.asyncio
    async def test_check_http_status_error(self) -> None:
        """Test health check handling of HTTP status errors."""
        mock_jira_client = MagicMock()
        mock_jira_client.base_url = "https://test.atlassian.net"
        mock_jira_client.auth = ("user@example.com", "token")

        checker = HealthChecker(jira_client=mock_jira_client)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 503

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Service Unavailable",
                    request=MagicMock(),
                    response=mock_response,
                )
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await checker._check_jira()

            assert result.status == HealthStatus.DOWN
            assert result.error == "HTTP 503"
