"""Health check endpoints for external service dependencies.

This module provides health check functionality for the Sentinel application,
including liveness probes (basic health) and readiness probes (dependency checks).

Health Check Types:
- Liveness: Basic check to confirm the service is running
- Readiness: Verifies connectivity to external dependencies (Jira, GitHub, Claude)

Configuration via environment variables:
- SENTINEL_HEALTH_CHECK_TIMEOUT: Timeout in seconds for individual health checks (default: 5.0)
- SENTINEL_HEALTH_CHECK_ENABLED: Enable/disable health checks (default: true)

Usage:
    from sentinel.health import HealthChecker, HealthCheckConfig

    config = HealthCheckConfig.from_env()
    checker = HealthChecker(
        config=config,
        jira_client=jira_rest_client,
        github_client=github_rest_client,
    )

    # Liveness check
    liveness = checker.check_liveness()

    # Readiness check with dependency verification
    readiness = await checker.check_readiness()
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.github_rest_client import GitHubRestClient
    from sentinel.rest_clients import JiraRestClient

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status values for service checks."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"


@dataclass
class ServiceHealth:
    """Health status for an individual service.

    Attributes:
        status: The health status (up, down, degraded).
        latency_ms: Latency in milliseconds for the health check.
        error: Optional error message if the check failed.
    """

    status: HealthStatus
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class HealthCheckResult:
    """Result of a health check operation.

    Attributes:
        status: Overall health status ("healthy", "unhealthy", "degraded").
        checks: Dictionary of individual service health checks.
        timestamp: Unix timestamp of the health check.
    """

    status: str
    checks: dict[str, ServiceHealth] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
        }


@dataclass
class HealthCheckConfig:
    """Configuration for health check behavior.

    Attributes:
        timeout: Timeout in seconds for individual health checks (default: 5.0).
        enabled: Whether health checks are enabled (default: True).
    """

    timeout: float = 5.0
    enabled: bool = True

    @classmethod
    def from_env(cls) -> HealthCheckConfig:
        """Load configuration from environment variables.

        Returns:
            HealthCheckConfig with values from environment or defaults.
        """
        enabled = os.getenv("SENTINEL_HEALTH_CHECK_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        timeout_str = os.getenv("SENTINEL_HEALTH_CHECK_TIMEOUT", "5.0")
        try:
            timeout = float(timeout_str)
            if timeout <= 0:
                logger.warning(
                    f"Invalid SENTINEL_HEALTH_CHECK_TIMEOUT: {timeout} must be "
                    "positive, using default 5.0"
                )
                timeout = 5.0
        except ValueError:
            logger.warning(
                f"Invalid SENTINEL_HEALTH_CHECK_TIMEOUT: '{timeout_str}' is not "
                "a valid number, using default 5.0"
            )
            timeout = 5.0

        return cls(timeout=timeout, enabled=enabled)

    @classmethod
    def from_config(cls, config: Config) -> HealthCheckConfig:
        """Create HealthCheckConfig from application Config.

        Args:
            config: Application configuration.

        Returns:
            HealthCheckConfig with appropriate values.
        """
        return cls(
            timeout=config.health_check_timeout,
            enabled=config.health_check_enabled,
        )


class HealthChecker:
    """Health checker for external service dependencies.

    Provides liveness and readiness checks for the Sentinel application.

    Attributes:
        config: Health check configuration.
        jira_client: Optional Jira REST client for connectivity checks.
        github_client: Optional GitHub REST client for connectivity checks.
        claude_api_key: Optional Claude API key for connectivity checks.
    """

    # Claude API health check endpoint (models list is lightweight)
    CLAUDE_API_URL = "https://api.anthropic.com/v1/models"

    # Buffer added to individual check timeouts when using asyncio.wait_for().
    # This ensures that the outer timeout in check_readiness() doesn't trigger
    # before individual check timeouts, allowing proper timeout handling and
    # error reporting from each health check method.
    TIMEOUT_BUFFER_SECONDS = 1.0

    def __init__(
        self,
        config: HealthCheckConfig | None = None,
        jira_client: JiraRestClient | None = None,
        github_client: GitHubRestClient | None = None,
        claude_api_key: str | None = None,
    ) -> None:
        """Initialize the health checker.

        Args:
            config: Health check configuration. If not provided, loads from env.
            jira_client: Optional Jira REST client for connectivity checks.
            github_client: Optional GitHub REST client for connectivity checks.
            claude_api_key: Optional Claude API key for connectivity checks.
        """
        self.config = config or HealthCheckConfig.from_env()
        self.jira_client = jira_client
        self.github_client = github_client
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    @classmethod
    def from_config(
        cls,
        config: Config,
        jira_client: JiraRestClient | None = None,
        github_client: GitHubRestClient | None = None,
        claude_api_key: str | None = None,
    ) -> HealthChecker:
        """Create a HealthChecker from application Config.

        This factory method provides a consistent initialization pattern that
        aligns with other components in the application that use Config-based
        initialization, making it easier to inject test configurations.

        Args:
            config: Application configuration containing health check settings.
            jira_client: Optional Jira REST client for connectivity checks.
            github_client: Optional GitHub REST client for connectivity checks.
            claude_api_key: Optional Claude API key for connectivity checks.

        Returns:
            HealthChecker configured with settings from the application Config.
        """
        health_config = HealthCheckConfig.from_config(config)
        return cls(
            config=health_config,
            jira_client=jira_client,
            github_client=github_client,
            claude_api_key=claude_api_key,
        )

    def check_liveness(self) -> HealthCheckResult:
        """Perform a basic liveness check.

        This is a simple check to verify the service is running.
        It does not check external dependencies.

        Returns:
            HealthCheckResult with status "healthy".
        """
        return HealthCheckResult(status="healthy")

    async def check_readiness(self) -> HealthCheckResult:
        """Perform a readiness check with dependency verification.

        Checks connectivity to configured external services:
        - Jira (if configured)
        - GitHub (if configured)
        - Claude API (if API key is configured)

        Returns:
            HealthCheckResult with overall status and individual service checks.
        """
        if not self.config.enabled:
            return HealthCheckResult(status="healthy")

        checks: dict[str, ServiceHealth] = {}

        # Run all health checks concurrently
        tasks: list[tuple[str, asyncio.Task[ServiceHealth]]] = []

        if self.jira_client is not None:
            tasks.append(("jira", asyncio.create_task(self._check_jira())))

        if self.github_client is not None:
            tasks.append(("github", asyncio.create_task(self._check_github())))

        if self.claude_api_key:
            tasks.append(("claude", asyncio.create_task(self._check_claude())))

        # Wait for all tasks to complete
        for name, task in tasks:
            try:
                checks[name] = await asyncio.wait_for(
                    task, timeout=self.config.timeout + self.TIMEOUT_BUFFER_SECONDS
                )
            except TimeoutError:
                checks[name] = ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=self.config.timeout * 1000,
                    error="Health check timed out",
                )
            except asyncio.CancelledError:
                # Task was cancelled, propagate to allow clean shutdown
                raise
            except AttributeError as e:
                # Catch AttributeError explicitly for potential None access issues
                # when processing task results (e.g., accessing attributes on the
                # returned ServiceHealth object). This differs from AttributeError
                # handling in individual check methods (_check_jira, _check_github,
                # _check_claude) which catch errors during client attribute access.
                logger.error(f"None access error in {name} health check: {e}")
                checks[name] = ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=0.0,  # No timing context available for result processing errors
                    error=f"Configuration error: {e}",
                )
            except (TypeError, ValueError, KeyError) as e:
                # Catch data processing errors that may occur when handling
                # health check results (e.g., invalid types, missing keys)
                logger.error(f"Data processing error in {name} health check: {e}")
                checks[name] = ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=0.0,  # No timing context available for data errors
                    error=f"Data error: {e}",
                )
            except OSError as e:
                # Catch OS-level errors (filesystem issues, network socket errors)
                # that may occur during health check operations
                logger.error(f"OS error in {name} health check: {e}")
                checks[name] = ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=0.0,  # No timing context available for OS errors
                    error=f"OS error: {e}",
                )
            except Exception as e:
                # Broad catch intentional for health checks - they should never crash
                # Log at error level since this is unexpected
                logger.error(f"Unexpected error in {name} health check: {e}")
                checks[name] = ServiceHealth(
                    status=HealthStatus.DOWN,
                    latency_ms=0.0,  # No timing context available for unexpected errors
                    error=str(e),
                )

        # Determine overall status
        if not checks:
            # No dependencies configured
            status = "healthy"
        elif all(check.status == HealthStatus.UP for check in checks.values()):
            status = "healthy"
        elif all(check.status == HealthStatus.DOWN for check in checks.values()):
            status = "unhealthy"
        else:
            status = "degraded"

        return HealthCheckResult(status=status, checks=checks)

    async def _check_jira(self) -> ServiceHealth:
        """Check Jira API connectivity.

        Performs a lightweight API call to verify Jira connectivity.

        Returns:
            ServiceHealth with status and latency.
        """
        if self.jira_client is None:
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=0.0,
                error="Jira client not configured",
            )

        start_time = time.perf_counter()
        try:
            # Use the Jira REST API to get server info (lightweight call)
            url = f"{self.jira_client.base_url}/rest/api/3/serverInfo"
            async with httpx.AsyncClient(
                auth=self.jira_client.auth,
                timeout=httpx.Timeout(self.config.timeout),
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Jira health check succeeded in {latency_ms:.2f}ms")
            return ServiceHealth(status=HealthStatus.UP, latency_ms=latency_ms)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Jira health check timed out after {latency_ms:.2f}ms")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error="Connection timed out",
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            logger.warning(f"Jira health check failed: {error_msg}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=error_msg,
            )
        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Jira health check failed due to request error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )
        except AttributeError as e:
            # Catch AttributeError explicitly for potential None access issues
            # (e.g., accessing attributes on partially configured clients)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Jira health check failed due to None access: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Configuration error: {e}",
            )
        except (TypeError, ValueError, KeyError) as e:
            # Catch data processing errors (invalid response format, missing fields)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Jira health check failed due to data error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Data error: {e}",
            )
        except OSError as e:
            # Catch OS-level errors (DNS resolution, socket errors, etc.)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Jira health check failed due to OS error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"OS error: {e}",
            )
        except Exception as e:
            # Broad catch intentional for health checks - document justification
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Jira health check failed with unexpected error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _check_github(self) -> ServiceHealth:
        """Check GitHub API connectivity.

        Performs a lightweight API call to verify GitHub connectivity.

        Returns:
            ServiceHealth with status and latency.
        """
        if self.github_client is None:
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=0.0,
                error="GitHub client not configured",
            )

        start_time = time.perf_counter()
        try:
            # Use GitHub API rate_limit endpoint (lightweight, doesn't count)
            url = f"{self.github_client.base_url}/rate_limit"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.github_client.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"GitHub health check succeeded in {latency_ms:.2f}ms")
            return ServiceHealth(status=HealthStatus.UP, latency_ms=latency_ms)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"GitHub health check timed out after {latency_ms:.2f}ms")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error="Connection timed out",
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            logger.warning(f"GitHub health check failed: {error_msg}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=error_msg,
            )
        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"GitHub health check failed due to request error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )
        except AttributeError as e:
            # Catch AttributeError explicitly for potential None access issues
            # (e.g., accessing attributes on partially configured clients)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"GitHub health check failed due to None access: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Configuration error: {e}",
            )
        except (TypeError, ValueError, KeyError) as e:
            # Catch data processing errors (invalid response format, missing fields)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"GitHub health check failed due to data error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Data error: {e}",
            )
        except OSError as e:
            # Catch OS-level errors (DNS resolution, socket errors, etc.)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"GitHub health check failed due to OS error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"OS error: {e}",
            )
        except Exception as e:
            # Broad catch intentional for health checks - document justification
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"GitHub health check failed with unexpected error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )

    async def _check_claude(self) -> ServiceHealth:
        """Check Claude API connectivity.

        Performs a lightweight API call to verify Claude API connectivity.
        Uses the models list endpoint which is fast and lightweight.

        Returns:
            ServiceHealth with status and latency.
        """
        if not self.claude_api_key:
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=0.0,
                error="Claude API key not configured",
            )

        start_time = time.perf_counter()
        try:
            # Use the Claude models list endpoint (lightweight ping)
            headers = {
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01",
            }
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
            ) as client:
                response = await client.get(self.CLAUDE_API_URL, headers=headers)
                response.raise_for_status()

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Claude health check succeeded in {latency_ms:.2f}ms")
            return ServiceHealth(status=HealthStatus.UP, latency_ms=latency_ms)

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Claude health check timed out after {latency_ms:.2f}ms")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error="Connection timed out",
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"HTTP {e.response.status_code}"
            logger.warning(f"Claude health check failed: {error_msg}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=error_msg,
            )
        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Claude health check failed due to request error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )
        except AttributeError as e:
            # Catch AttributeError explicitly for potential None access issues
            # (e.g., accessing attributes on partially configured clients)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Claude health check failed due to None access: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Configuration error: {e}",
            )
        except (TypeError, ValueError, KeyError) as e:
            # Catch data processing errors (invalid response format, missing fields)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Claude health check failed due to data error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"Data error: {e}",
            )
        except OSError as e:
            # Catch OS-level errors (DNS resolution, socket errors, etc.)
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Claude health check failed due to OS error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=f"OS error: {e}",
            )
        except Exception as e:
            # Broad catch intentional for health checks - document justification
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Claude health check failed with unexpected error: {e}")
            return ServiceHealth(
                status=HealthStatus.DOWN,
                latency_ms=latency_ms,
                error=str(e),
            )
