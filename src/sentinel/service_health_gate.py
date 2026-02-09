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

Extensibility:
    New services can be added by implementing the ``ProbeStrategy`` protocol and
    registering the strategy via ``ServiceHealthGate.register_probe_strategy()``.
    See ``JiraProbeStrategy`` and ``GitHubProbeStrategy`` for examples.

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

    # Register a custom probe strategy for a new service
    gate.register_probe_strategy("my_service", MyServiceProbeStrategy())
"""

from __future__ import annotations

import inspect
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from sentinel.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Probe endpoint constants
# ---------------------------------------------------------------------------
# These constants define the HTTP paths used by the built-in probe strategies
# (``JiraProbeStrategy`` and ``GitHubProbeStrategy``) to verify that an
# external service is reachable.  They are consumed by ``probe_service()``
# via the corresponding strategy's ``execute()`` method.
#
# To add a new service:
#   1. Create a class that satisfies the ``ProbeStrategy`` protocol.
#   2. Register it with ``ServiceHealthGate.register_probe_strategy()``.
#
# The built-in strategies are registered automatically in
# ``ServiceHealthGate.__init__``.
# ---------------------------------------------------------------------------

JIRA_PROBE_PATH = "/rest/api/3/serverInfo"
"""Jira health-check endpoint path.

Used by ``JiraProbeStrategy`` to issue ``GET {base_url}/rest/api/3/serverInfo``.
"""

GITHUB_PROBE_PATH = "/rate_limit"
"""GitHub health-check endpoint path.

Used by ``GitHubProbeStrategy`` to issue ``GET {base_url}/rate_limit``.
"""

GITHUB_API_VERSION = "2022-11-28"
"""GitHub API version header value.

Sent as ``X-GitHub-Api-Version`` in probe requests to GitHub."""

_PROBE_STRATEGY_REQUIRED_PARAMS: frozenset[str] = frozenset(
    {"timeout", "base_url", "auth", "token"}
)
"""Required keyword parameters for ``ProbeStrategy.execute()`` implementations.

This constant is the single source of truth for the parameter names that every
``ProbeStrategy`` must accept.  It is used by
``ServiceHealthGate._validate_probe_strategy_signature()`` to verify strategy
signatures at registration time.  If the ``ProbeStrategy`` protocol ever gains
new parameters, update this set accordingly.
"""


@runtime_checkable
class ProbeStrategy(Protocol):
    """Protocol for service probe strategies.

    Implement this protocol to add health-check probing for a new external
    service.  Each strategy encapsulates the HTTP request logic needed to
    determine whether a particular service is reachable.

    Register a strategy instance with
    ``ServiceHealthGate.register_probe_strategy()`` so that
    ``probe_service()`` can dispatch to it by service name.

    .. note:: **``@runtime_checkable`` limitation**

       The ``@runtime_checkable`` decorator enables ``isinstance()`` checks
       against this protocol, but Python's runtime protocol checking **only
       verifies that the required method names exist** on the object — it does
       **not** validate method signatures.  A class that defines
       ``execute(self)`` (with no keyword arguments) would pass an
       ``isinstance(obj, ProbeStrategy)`` check yet raise ``TypeError`` when
       called with the expected keyword arguments.

       To mitigate this, ``register_probe_strategy()`` performs an explicit
       signature inspection to verify that the ``execute`` method accepts the
       required keyword-only parameters defined in
       ``_PROBE_STRATEGY_REQUIRED_PARAMS``.  If the signature does not match,
       ``register_probe_strategy()`` raises ``TypeError`` with a descriptive
       message.

    .. note:: **Uniform interface — not all parameters are used by every strategy**

       The ``execute()`` method provides a uniform interface across all probe
       strategies.  Individual strategy implementations are not expected to
       use every parameter:

       - ``JiraProbeStrategy`` uses ``timeout``, ``base_url``, and ``auth``
         but ignores ``token``.
       - ``GitHubProbeStrategy`` uses ``timeout``, ``base_url``, and ``token``
         but ignores ``auth``.

       This is the intended design tradeoff: a consistent calling convention
       simplifies the dispatch logic in ``_execute_probe()`` at the cost of
       each strategy receiving parameters it may not need.  Custom strategy
       implementations should accept all four keyword parameters even if they
       only use a subset.

    Example::

        class MyServiceProbeStrategy:
            def execute(
                self,
                *,
                timeout: float,
                base_url: str = "",
                auth: tuple[str, str] | None = None,
                token: str = "",
            ) -> bool:
                url = f"{base_url}/health"
                resp = httpx.get(url, timeout=timeout)
                return resp.status_code == 200

        gate.register_probe_strategy("my_service", MyServiceProbeStrategy())
    """

    def execute(
        self,
        *,
        timeout: float,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        token: str = "",
    ) -> bool:
        """Execute the probe for this service.

        Args:
            timeout: HTTP request timeout in seconds.
            base_url: Base URL for the service API.
            auth: Optional (username/email, api_token) tuple for Basic auth.
                Not all strategies use this parameter — see the protocol-level
                docstring for details on the uniform interface tradeoff.
            token: Optional bearer token for token-based authentication.
                Not all strategies use this parameter — see the protocol-level
                docstring for details on the uniform interface tradeoff.

        Returns:
            True if the service is available, False otherwise.
        """
        ...  # pragma: no cover


class JiraProbeStrategy:
    """Probe strategy for Jira service availability.

    Issues ``GET {base_url}/rest/api/3/serverInfo`` with optional Basic
    authentication.  See ``JIRA_PROBE_PATH`` for the endpoint constant.
    """

    def execute(
        self,
        *,
        timeout: float,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        token: str = "",
    ) -> bool:
        """Probe Jira service for availability.

        Args:
            timeout: HTTP request timeout in seconds.
            base_url: Jira base URL.
            auth: (email, api_token) tuple for authentication.
            token: Not used for Jira probes (ignored).

        Returns:
            True if Jira is available, False otherwise.
        """
        if not base_url:
            logger.warning("[HEALTH_GATE] jira: No base_url provided for probe")
            return False

        url = f"{base_url.rstrip('/')}{JIRA_PROBE_PATH}"
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
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


class GitHubProbeStrategy:
    """Probe strategy for GitHub service availability.

    Issues ``GET {base_url}/rate_limit`` with the required GitHub API headers
    and optional Bearer token.  See ``GITHUB_PROBE_PATH`` and
    ``GITHUB_API_VERSION`` for the endpoint and version constants.
    """

    def execute(
        self,
        *,
        timeout: float,
        base_url: str = "",
        auth: tuple[str, str] | None = None,
        token: str = "",
    ) -> bool:
        """Probe GitHub service for availability.

        Args:
            timeout: HTTP request timeout in seconds.
            base_url: GitHub API base URL.
            auth: Not used for GitHub probes (ignored).
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
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
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

    Thread-safety contract:
        All public methods and properties on this class are safe to call
        concurrently from multiple threads.  Internal service state is
        protected by ``_lock`` (an ``RLock``).  Probe metric counters are
        protected by a dedicated ``_counter_lock`` (a ``Lock``) so that
        counter reads never need to acquire the heavier service-state lock.
        Use the ``probe_success_count``, ``probe_expected_failure_count``,
        and ``probe_unexpected_error_count`` properties for individual reads,
        or ``get_probe_metrics()`` for an atomic snapshot of all three.

    Attributes:
        config: Health gate configuration.
        probe_success_count: Total successful probe attempts across all services.
        probe_expected_failure_count: Probe failures from expected HTTP errors
            (timeouts, HTTP status errors, connection errors).
        probe_unexpected_error_count: Probe failures from unexpected exceptions
            (programming errors, unforeseen runtime errors).

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

    def __init__(
        self,
        config: ServiceHealthGateConfig | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the service health gate.

        Args:
            config: Health gate configuration. If not provided, loads from env.
            time_func: Optional callable returning the current time as a float
                (seconds since epoch). Defaults to ``time.time``. Providing a
                custom function decouples backoff tests from wall-clock time.
        """
        self.config = config or ServiceHealthGateConfig.from_env()
        self._time_func: Callable[[], float] = time_func or time.time
        self._services: dict[str, ServiceAvailability] = {}
        self._lock = threading.RLock()
        self._counter_lock = threading.Lock()
        self._probe_success_count: int = 0
        self._probe_expected_failure_count: int = 0
        self._probe_unexpected_error_count: int = 0
        self._probe_strategies: dict[str, ProbeStrategy] = {}
        self._register_default_strategies()

    @property
    def probe_success_count(self) -> int:
        """Total successful probe attempts across all services.

        Thread-safe: reads are protected by ``_counter_lock``.
        """
        with self._counter_lock:
            return self._probe_success_count

    @probe_success_count.setter
    def probe_success_count(self, value: int) -> None:
        if not isinstance(value, int):
            msg = f"probe_success_count must be an int, got {type(value).__name__}"
            raise TypeError(msg)
        if value < 0:
            msg = f"probe_success_count must be non-negative, got {value}"
            raise ValueError(msg)
        with self._counter_lock:
            self._probe_success_count = value

    @property
    def probe_expected_failure_count(self) -> int:
        """Probe failures from expected HTTP errors.

        Thread-safe: reads are protected by ``_counter_lock``.
        """
        with self._counter_lock:
            return self._probe_expected_failure_count

    @probe_expected_failure_count.setter
    def probe_expected_failure_count(self, value: int) -> None:
        if not isinstance(value, int):
            msg = f"probe_expected_failure_count must be an int, got {type(value).__name__}"
            raise TypeError(msg)
        if value < 0:
            msg = f"probe_expected_failure_count must be non-negative, got {value}"
            raise ValueError(msg)
        with self._counter_lock:
            self._probe_expected_failure_count = value

    @property
    def probe_unexpected_error_count(self) -> int:
        """Probe failures from unexpected exceptions.

        Thread-safe: reads are protected by ``_counter_lock``.
        """
        with self._counter_lock:
            return self._probe_unexpected_error_count

    @probe_unexpected_error_count.setter
    def probe_unexpected_error_count(self, value: int) -> None:
        if not isinstance(value, int):
            msg = f"probe_unexpected_error_count must be an int, got {type(value).__name__}"
            raise TypeError(msg)
        if value < 0:
            msg = f"probe_unexpected_error_count must be non-negative, got {value}"
            raise ValueError(msg)
        with self._counter_lock:
            self._probe_unexpected_error_count = value

    def get_probe_metrics(self) -> dict[str, int]:
        """Return an atomic snapshot of all probe metric counters.

        Acquires ``_counter_lock`` once to read all three counters, so the
        returned values are mutually consistent.

        Returns:
            Dictionary with keys ``probe_success_count``,
            ``probe_expected_failure_count``, and
            ``probe_unexpected_error_count``.
        """
        with self._counter_lock:
            return {
                "probe_success_count": self._probe_success_count,
                "probe_expected_failure_count": self._probe_expected_failure_count,
                "probe_unexpected_error_count": self._probe_unexpected_error_count,
            }

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

    def _register_default_strategies(self) -> None:
        """Register built-in probe strategies for Jira and GitHub."""
        self._probe_strategies["jira"] = JiraProbeStrategy()
        self._probe_strategies["github"] = GitHubProbeStrategy()

    @staticmethod
    def _validate_probe_strategy_signature(strategy: object) -> None:
        """Validate that a strategy's execute() method has the required signature.

        The ``@runtime_checkable`` decorator on ``ProbeStrategy`` only checks
        for the *existence* of an ``execute`` attribute — it does not verify
        that the method accepts the expected keyword-only parameters.  This
        helper performs an explicit ``inspect.signature`` check so that
        signature mismatches are caught eagerly at registration time rather
        than at probe-call time.

        The set of required parameters is defined by the module-level constant
        ``_PROBE_STRATEGY_REQUIRED_PARAMS``.

        Args:
            strategy: The candidate strategy object to validate.

        Raises:
            TypeError: If the ``execute`` method is missing any of the
                required keyword-only parameters defined in
                ``_PROBE_STRATEGY_REQUIRED_PARAMS``.
        """
        try:
            sig = inspect.signature(strategy.execute)  # type: ignore[attr-defined]
        except (ValueError, TypeError) as e:
            msg = (
                f"Cannot inspect execute() signature on "
                f"{type(strategy).__name__}: {e}"
            )
            raise TypeError(msg) from e

        # The ``name != "self"`` guard is defensive: ``inspect.signature()``
        # on a bound method normally excludes ``self``, but keeping the
        # filter protects against edge cases (e.g. unbound methods or
        # unusual descriptors).
        param_names = {
            name
            for name, param in sig.parameters.items()
            if param.kind
            in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != "self"
        }

        missing = _PROBE_STRATEGY_REQUIRED_PARAMS - param_names
        if missing:
            sorted_missing = sorted(missing)
            msg = (
                f"strategy execute() method on {type(strategy).__name__} is "
                f"missing required keyword parameters: {sorted_missing}. "
                f"See ProbeStrategy protocol docstring for the expected signature."
            )
            raise TypeError(msg)

    def register_probe_strategy(self, service_name: str, strategy: ProbeStrategy) -> None:
        """Register a probe strategy for a service.

        Use this method to add probing support for a new external service
        without modifying the ``_execute_probe`` dispatch logic.

        Performs two levels of validation:

        1. ``isinstance(strategy, ProbeStrategy)`` — verifies that the object
           has an ``execute`` attribute (runtime protocol check).
        2. ``_validate_probe_strategy_signature()`` — inspects the ``execute``
           method's signature to ensure it accepts the required keyword-only
           parameters defined in ``_PROBE_STRATEGY_REQUIRED_PARAMS``.

        This two-step validation catches both missing methods and incorrect
        signatures at registration time, preventing confusing ``TypeError``
        exceptions at probe-call time.

        Args:
            service_name: Lowercase service name (e.g. ``"jira"``, ``"my_service"``).
            strategy: An object satisfying the ``ProbeStrategy`` protocol.

        Raises:
            TypeError: If *strategy* does not satisfy the ``ProbeStrategy``
                protocol or its ``execute()`` method is missing required
                keyword parameters.
        """
        if not isinstance(strategy, ProbeStrategy):
            msg = f"strategy must satisfy ProbeStrategy protocol, got {type(strategy).__name__}"
            raise TypeError(msg)
        self._validate_probe_strategy_signature(strategy)
        self._probe_strategies[service_name.lower()] = strategy

    def unregister_probe_strategy(self, service_name: str) -> None:
        """Remove a registered probe strategy for a service.

        Args:
            service_name: Lowercase service name to remove.
        """
        self._probe_strategies.pop(service_name.lower(), None)

    def get_probe_strategy(self, service_name: str) -> ProbeStrategy | None:
        """Return the registered probe strategy for a service, or None.

        Args:
            service_name: Service name to look up.

        Returns:
            The registered ``ProbeStrategy``, or ``None`` if not found.
        """
        return self._probe_strategies.get(service_name.lower())

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
            now = self._time_func()
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
            now = self._time_func()

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

            now = self._time_func()

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
        unexpected_error = False
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
            unexpected_error = True

        with self._lock:
            service = self._get_or_create_service(service_name)
            now = self._time_func()
            service.last_check_at = now

            if success:
                with self._counter_lock:
                    self._probe_success_count += 1
                service.available = True
                service.consecutive_failures = 0
                service.paused_at = None
                service.probe_count = 0
                service.last_available_at = now
                service.last_error = None
                logger.info(
                    "[HEALTH_GATE] %s: Probe succeeded, service is now available",
                    service_name,
                )
            else:
                with self._counter_lock:
                    if unexpected_error:
                        self._probe_unexpected_error_count += 1
                    else:
                        self._probe_expected_failure_count += 1
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

        Dispatches to the registered ``ProbeStrategy`` for *service_name*.
        Strategies are looked up in ``_probe_strategies`` by lowercase name.
        If no strategy is registered, logs a warning and returns ``False``.

        To add probing for a new service, implement the ``ProbeStrategy``
        protocol and register it via ``register_probe_strategy()``.

        Args:
            service_name: Name of the service to probe.
            base_url: Base URL for the service API.
            auth: Optional (email, api_token) tuple for Jira authentication.
            token: Optional bearer token for GitHub authentication.

        Returns:
            True if the probe succeeded, False otherwise.
        """
        service_lower = service_name.lower()
        strategy = self._probe_strategies.get(service_lower)

        if strategy is None:
            logger.warning(
                "[HEALTH_GATE] %s: Unknown service, cannot probe",
                service_name,
            )
            return False

        return strategy.execute(
            timeout=self.config.probe_timeout,
            base_url=base_url,
            auth=auth,
            token=token,
        )

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get a snapshot of all tracked service availability states.

        Returns a deep copy of the current state for each tracked service,
        suitable for dashboard display or logging.

        Returns:
            Dictionary mapping service names to their availability state.
        """
        with self._lock:
            return {name: svc.to_dict() for name, svc in self._services.items()}
