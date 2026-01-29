"""Tests for health check functionality."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from sentinel.health import (
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
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
