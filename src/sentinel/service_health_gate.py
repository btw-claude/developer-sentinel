"""Service health gate for external service availability tracking.

This module provides orchestration-level health gating for external services
(Jira, GitHub). When a service experiences consecutive failures beyond a
configured threshold, the health gate pauses operations that depend on that
service and periodically probes the service to detect recovery.

Complementary to the circuit breaker pattern:
- Circuit breaker (circuit_breaker.py): Protects individual HTTP calls by
  failing fast when a service is unresponsive.
- Service health gate (this module): Protects the polling loop and execution
  retries at the orchestration level by gating operations based on overall
  service availability.

Configuration via environment variables:
- SENTINEL_HEALTH_GATE_ENABLED: Enable/disable health gating (default: true)
- SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD: Consecutive failures before gating (default: 3)
- SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL: Initial probe interval in seconds (default: 30)
- SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL: Maximum probe interval in seconds (default: 300)
- SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR: Backoff multiplier for probes (default: 2.0)
- SENTINEL_HEALTH_GATE_PROBE_TIMEOUT: Timeout for probe requests in seconds (default: 5.0)

Usage:
    from sentinel.service_health_gate import ServiceHealthGate, ServiceHealthGateConfig

    config = ServiceHealthGateConfig.from_env()
    gate = ServiceHealthGate(config=config)

    # Check if polling should proceed for a service
    if gate.should_poll("jira"):
        try:
            result = poll_jira()
            gate.record_poll_success("jira")
        except Exception as e:
            gate.record_poll_failure("jira", e)

    # Probe a service to check recovery
    if gate.should_probe("jira"):
        gate.probe_service("jira", base_url="https://...", auth=("user", "token"))

    # Get status snapshot for dashboard
    status = gate.get_all_status()
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from sentinel.logging import get_logger

logger = get_logger(__name__)

# Probe endpoints for external services.
# NOTE: Adding a new service requires updating these constants AND adding
# a corresponding _probe_<service>() method and branch in _execute_probe().
# See probe_service() for the dispatch logic.
JIRA_PROBE_PATH = "/rest/api/3/serverInfo"
GITHUB_PROBE_PATH = "/rate_limit"


@dataclass(frozen=True)
class ServiceHealthGateConfig:
    """Configuration for service health gate behavior.

    Controls how the system gates operations based on external service
    availability. When a service fails health checks beyond the failure
    threshold, operations depending on that service are paused until
    probe checks confirm recovery.

    Attributes:
        enabled: Enable/disable service health gating.
        failure_threshold: Number of consecutive failures before gating.
        initial_probe_interval: Initial seconds between recovery probes.
        max_probe_interval: Maximum seconds between recovery probes.
        probe_backoff_factor: Multiplier for probe interval on continued failure (>= 1.0).
        probe_timeout: Timeout in seconds for individual probe checks.
    """

    enabled: bool = True
    failure_threshold: int = 3
    initial_probe_interval: float = 30.0
    max_probe_interval: float = 300.0
    probe_backoff_factor: float = 2.0
    probe_timeout: float = 5.0

    @classmethod
    def from_env(cls) -> ServiceHealthGateConfig:
        """Load configuration from environment variables.

        Returns:
            ServiceHealthGateConfig with values from environment or defaults.
        """
        enabled = os.getenv("SENTINEL_HEALTH_GATE_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        failure_threshold = int(os.getenv("SENTINEL_HEALTH_GATE_FAILURE_THRESHOLD", "3"))
        initial_probe_interval = float(
            os.getenv("SENTINEL_HEALTH_GATE_INITIAL_PROBE_INTERVAL", "30.0")
        )
        max_probe_interval = float(
            os.getenv("SENTINEL_HEALTH_GATE_MAX_PROBE_INTERVAL", "300.0")
        )
        probe_backoff_factor = max(
            1.0,
            float(os.getenv("SENTINEL_HEALTH_GATE_PROBE_BACKOFF_FACTOR", "2.0")),
        )
        probe_timeout = float(os.getenv("SENTINEL_HEALTH_GATE_PROBE_TIMEOUT", "5.0"))

        return cls(
            enabled=enabled,
            failure_threshold=failure_threshold,
            initial_probe_interval=initial_probe_interval,
            max_probe_interval=max_probe_interval,
            probe_backoff_factor=probe_backoff_factor,
            probe_timeout=probe_timeout,
        )


@dataclass
class ServiceAvailability:
    """Mutable state tracking for a single service's availability.

    Tracks consecutive failures, availability status, and probe scheduling
    for an external service.

    Attributes:
        service_name: Name of the service being tracked.
        available: Whether the service is currently considered available.
        consecutive_failures: Number of consecutive poll failures.
        last_check_at: Timestamp of the last poll attempt.
        last_available_at: Timestamp when the service was last known available.
        last_error: Description of the most recent error, if any.
        paused_at: Timestamp when the service was gated (paused), or None.
        probe_count: Number of probe attempts since the service was gated.
    """

    service_name: str
    available: bool = True
    consecutive_failures: int = 0
    last_check_at: float | None = None
    last_available_at: float | None = None
    last_error: str | None = None
    paused_at: float | None = None
    probe_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for dashboard/logging.

        Returns:
            Dictionary with all availability state fields.
        """
        return {
            "service_name": self.service_name,
            "available": self.available,
            "consecutive_failures": self.consecutive_failures,
            "last_check_at": self.last_check_at,
            "last_available_at": self.last_available_at,
            "last_error": self.last_error,
            "paused_at": self.paused_at,
            "probe_count": self.probe_count,
        }


class ServiceHealthGate:
    """Thread-safe health gate for external service availability.

    Tracks service health and gates operations when services are unavailable.
    Periodically probes unavailable services with exponential backoff to
    detect recovery.

    This is complementary to the circuit breaker pattern: the circuit breaker
    protects individual HTTP calls, while the health gate protects the polling
    loop and execution retries at the orchestration level.

    Attributes:
        config: Health gate configuration.

    Usage:
        gate = ServiceHealthGate()

        # In polling loop
        if gate.should_poll("jira"):
            try:
                issues = poll_jira()
                gate.record_poll_success("jira")
            except Exception as e:
                gate.record_poll_failure("jira", e)
        elif gate.should_probe("jira"):
            gate.probe_service("jira", base_url=url, auth=auth)
    """

    def __init__(self, config: ServiceHealthGateConfig | None = None) -> None:
        """Initialize the service health gate.

        Args:
            config: Health gate configuration. If not provided, loads from env.
        """
        self.config = config or ServiceHealthGateConfig.from_env()
        self._services: dict[str, ServiceAvailability] = {}
        self._lock = threading.RLock()

    def _get_or_create_service(self, service_name: str) -> ServiceAvailability:
        """Get or create a ServiceAvailability tracker for a service.

        Must be called with self._lock held.

        Args:
            service_name: Name of the service.

        Returns:
            ServiceAvailability instance for the service.
        """
        if service_name not in self._services:
            self._services[service_name] = ServiceAvailability(service_name=service_name)
        return self._services[service_name]

    def should_poll(self, service_name: str) -> bool:
        """Check if polling should proceed for a service.

        Returns True if the service is available (or health gating is disabled).
        Returns False if the service has been gated due to consecutive failures.

        Args:
            service_name: Name of the service to check.

        Returns:
            True if polling should proceed, False if the service is gated.
        """
        if not self.config.enabled:
            return True

        with self._lock:
            service = self._get_or_create_service(service_name)
            return service.available

    def record_poll_success(self, service_name: str) -> None:
        """Record a successful poll for a service.

        Resets consecutive failure count and marks the service as available
        if it was previously gated.

        Args:
            service_name: Name of the service.
        """
        if not self.config.enabled:
            return

        with self._lock:
            service = self._get_or_create_service(service_name)
            now = time.time()
            was_unavailable = not service.available

            service.consecutive_failures = 0
            service.last_check_at = now
            service.last_available_at = now
            service.last_error = None

            if was_unavailable:
                service.available = True
                service.paused_at = None
                service.probe_count = 0
                logger.info(
                    "[HEALTH_GATE] %s: Service recovered and is now available",
                    service_name,
                )

    def record_poll_failure(self, service_name: str, error: BaseException | None = None) -> None:
        """Record a poll failure for a service.

        Increments the consecutive failure count. If the count reaches the
        configured threshold, the service is gated (marked unavailable).

        Args:
            service_name: Name of the service.
            error: Optional exception that caused the failure.
        """
        if not self.config.enabled:
            return

        with self._lock:
            service = self._get_or_create_service(service_name)
            now = time.time()

            service.consecutive_failures += 1
            service.last_check_at = now
            service.last_error = str(error) if error else "Unknown error"

            error_info = f": {type(error).__name__}: {error}" if error else ""
            logger.warning(
                "[HEALTH_GATE] %s: Poll failure recorded (count: %d/%d)%s",
                service_name,
                service.consecutive_failures,
                self.config.failure_threshold,
                error_info,
            )

            if (
                service.available
                and service.consecutive_failures >= self.config.failure_threshold
            ):
                service.available = False
                service.paused_at = now
                service.probe_count = 0
                logger.warning(
                    "[HEALTH_GATE] %s: Service gated after %d consecutive failures",
                    service_name,
                    service.consecutive_failures,
                )

    def should_probe(self, service_name: str) -> bool:
        """Check if a probe should be attempted for an unavailable service.

        Uses exponential backoff to determine probe timing. The probe interval
        starts at ``initial_probe_interval`` and increases by
        ``probe_backoff_factor`` after each probe, up to ``max_probe_interval``.

        Args:
            service_name: Name of the service to check.

        Returns:
            True if a probe should be attempted, False otherwise.
        """
        if not self.config.enabled:
            return False

        with self._lock:
            service = self._get_or_create_service(service_name)

            # Only probe unavailable services
            if service.available:
                return False

            if service.paused_at is None:
                return False

            now = time.time()

            # Calculate the current probe interval with exponential backoff
            interval = self.config.initial_probe_interval * (
                self.config.probe_backoff_factor ** service.probe_count
            )
            interval = min(interval, self.config.max_probe_interval)

            # Determine the reference time for when the next probe is due
            if service.last_check_at is not None and service.last_check_at > service.paused_at:
                reference_time = service.last_check_at
            else:
                reference_time = service.paused_at

            return (now - reference_time) >= interval

    def probe_service(
        self,
        service_name: str,
        *,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        token: str = "",
    ) -> bool:
        """Probe a service to check if it has recovered.

        Makes a synchronous HTTP request to the service's health endpoint.
        On success, marks the service as available. On failure, increments
        the probe count for backoff calculation.

        Probe endpoints:
        - Jira: GET /rest/api/3/serverInfo
        - GitHub: GET /rate_limit

        Args:
            service_name: Name of the service to probe ("jira" or "github").
            base_url: Base URL for the service API.
            auth: Optional (email, api_token) tuple for Jira authentication.
            token: Optional bearer token for GitHub authentication.

        Returns:
            True if the probe succeeded and the service is available,
            False if the probe failed.
        """
        if not self.config.enabled:
            return True

        success = False
        try:
            success = self._execute_probe(service_name, base_url=base_url, auth=auth, token=token)
        except Exception as e:
            # INTENTIONAL BROAD CATCH: Probes must never crash the application.
            # Unexpected errors are logged and treated as probe failures.
            logger.warning(
                "[HEALTH_GATE] %s: Probe failed with unexpected error: %s: %s",
                service_name,
                type(e).__name__,
                e,
            )
            success = False

        with self._lock:
            service = self._get_or_create_service(service_name)
            service.last_check_at = time.time()

            if success:
                service.available = True
                service.consecutive_failures = 0
                service.paused_at = None
                service.probe_count = 0
                service.last_available_at = time.time()
                service.last_error = None
                logger.info(
                    "[HEALTH_GATE] %s: Probe succeeded, service is now available",
                    service_name,
                )
            else:
                service.probe_count += 1
                logger.info(
                    "[HEALTH_GATE] %s: Probe failed (probe count: %d)",
                    service_name,
                    service.probe_count,
                )

        return success

    def _execute_probe(
        self,
        service_name: str,
        *,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        token: str = "",
    ) -> bool:
        """Execute the actual HTTP probe for a service.

        Currently dispatches to service-specific probe methods based on name.
        Future improvement: consider a ProbeStrategy protocol to allow new
        services to be added without modifying this dispatch logic.

        Args:
            service_name: Name of the service to probe.
            base_url: Base URL for the service API.
            auth: Optional (email, api_token) tuple for Jira authentication.
            token: Optional bearer token for GitHub authentication.

        Returns:
            True if the probe succeeded, False otherwise.
        """
        service_lower = service_name.lower()

        if service_lower == "jira":
            return self._probe_jira(base_url, auth)
        if service_lower == "github":
            return self._probe_github(base_url, token)

        logger.warning(
            "[HEALTH_GATE] %s: Unknown service, cannot probe",
            service_name,
        )
        return False

    def _probe_jira(self, base_url: str, auth: tuple[str, str] | None) -> bool:
        """Probe Jira service for availability.

        Args:
            base_url: Jira base URL.
            auth: (email, api_token) tuple for authentication.

        Returns:
            True if Jira is available, False otherwise.
        """
        if not base_url:
            logger.warning("[HEALTH_GATE] jira: No base_url provided for probe")
            return False

        url = f"{base_url.rstrip('/')}{JIRA_PROBE_PATH}"
        try:
            with httpx.Client(timeout=httpx.Timeout(self.config.probe_timeout)) as client:
                response = client.get(url, auth=auth)
                response.raise_for_status()
                return True
        except httpx.TimeoutException:
            logger.debug("[HEALTH_GATE] jira: Probe timed out")
            return False
        except httpx.HTTPStatusError as e:
            logger.debug("[HEALTH_GATE] jira: Probe got HTTP %d", e.response.status_code)
            return False
        except httpx.RequestError as e:
            logger.debug("[HEALTH_GATE] jira: Probe request error: %s", e)
            return False

    def _probe_github(self, base_url: str, token: str) -> bool:
        """Probe GitHub service for availability.

        Args:
            base_url: GitHub API base URL.
            token: Bearer token for authentication.

        Returns:
            True if GitHub is available, False otherwise.
        """
        if not base_url:
            logger.warning("[HEALTH_GATE] github: No base_url provided for probe")
            return False

        url = f"{base_url.rstrip('/')}{GITHUB_PROBE_PATH}"
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with httpx.Client(timeout=httpx.Timeout(self.config.probe_timeout)) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return True
        except httpx.TimeoutException:
            logger.debug("[HEALTH_GATE] github: Probe timed out")
            return False
        except httpx.HTTPStatusError as e:
            logger.debug("[HEALTH_GATE] github: Probe got HTTP %d", e.response.status_code)
            return False
        except httpx.RequestError as e:
            logger.debug("[HEALTH_GATE] github: Probe request error: %s", e)
            return False

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get a snapshot of all tracked service availability states.

        Returns a deep copy of the current state for each tracked service,
        suitable for dashboard display or logging.

        Returns:
            Dictionary mapping service names to their availability state.
        """
        with self._lock:
            return {name: svc.to_dict() for name, svc in self._services.items()}
