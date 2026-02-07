"""Circuit breaker pattern implementation for external service calls.

This module provides a circuit breaker implementation to protect against cascading
failures when calling external services (Jira, GitHub, Claude APIs). When failures
exceed a threshold, the circuit "opens" and fails fast without attempting calls,
allowing the external service time to recover.

Circuit States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests fail immediately without calling the service
- HALF_OPEN: Testing recovery, limited requests pass through to test service health

Configuration is done via environment variables:
- SENTINEL_CIRCUIT_BREAKER_ENABLED: Enable/disable circuit breakers (default: true)
- SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD: Failures before opening (default: 5)
- SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Seconds before trying recovery (default: 30)
- SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS: Calls allowed in half-open state (default: 3)

Per-service overrides:
- SENTINEL_JIRA_CIRCUIT_BREAKER_FAILURE_THRESHOLD
- SENTINEL_JIRA_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
- SENTINEL_GITHUB_CIRCUIT_BREAKER_FAILURE_THRESHOLD
- SENTINEL_GITHUB_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
- SENTINEL_CLAUDE_CIRCUIT_BREAKER_FAILURE_THRESHOLD
- SENTINEL_CLAUDE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT
"""

from __future__ import annotations

import os
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

from sentinel.logging import get_logger

logger = get_logger(__name__)

# Type variable for generic function signatures
T = TypeVar("T")


class CircuitState(Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and call is rejected."""

    def __init__(self, service_name: str, state: CircuitState, message: str | None = None) -> None:
        self.service_name = service_name
        self.state = state
        msg = message or f"Circuit breaker for {service_name} is {state.value}"
        super().__init__(msg)


class CircuitBreakerConfigError(ValueError):
    """Raised when circuit breaker configuration is invalid."""

    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening the circuit (default: 5).
            Must be a positive integer.
        recovery_timeout: Seconds to wait before attempting recovery (default: 30.0).
            Must be a positive number.
        half_open_max_calls: Maximum calls to allow in half-open state (default: 3).
            Must be a positive integer.
        enabled: Whether the circuit breaker is enabled (default: True).

    Raises:
        CircuitBreakerConfigError: If any threshold values are not positive.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        # Check for boolean first since bool is a subclass of int in Python
        if isinstance(self.failure_threshold, bool):
            raise CircuitBreakerConfigError(
                "failure_threshold must be a positive integer, not a boolean"
            )
        if not isinstance(self.failure_threshold, int):
            raise CircuitBreakerConfigError(
                f"failure_threshold must be an integer, got {type(self.failure_threshold).__name__}"
            )
        if self.failure_threshold <= 0:
            raise CircuitBreakerConfigError(
                f"failure_threshold must be positive, got {self.failure_threshold}"
            )
        if self.recovery_timeout <= 0:
            raise CircuitBreakerConfigError(
                f"recovery_timeout must be positive, got {self.recovery_timeout}"
            )
        # Check for boolean first since bool is a subclass of int in Python
        if isinstance(self.half_open_max_calls, bool):
            raise CircuitBreakerConfigError(
                "half_open_max_calls must be a positive integer, not a boolean"
            )
        if not isinstance(self.half_open_max_calls, int):
            type_name = type(self.half_open_max_calls).__name__
            raise CircuitBreakerConfigError(
                f"half_open_max_calls must be an integer, got {type_name}"
            )
        if self.half_open_max_calls <= 0:
            raise CircuitBreakerConfigError(
                f"half_open_max_calls must be positive, got {self.half_open_max_calls}"
            )

    @classmethod
    def from_env(cls, service_name: str = "") -> CircuitBreakerConfig:
        """Load configuration from environment variables.

        Args:
            service_name: Optional service name for service-specific overrides.
                         If provided, looks for SENTINEL_{SERVICE}_CIRCUIT_BREAKER_* vars.

        Returns:
            CircuitBreakerConfig with values from environment or defaults.
        """
        # Global defaults
        enabled = os.getenv("SENTINEL_CIRCUIT_BREAKER_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        failure_threshold = int(os.getenv("SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
        recovery_timeout = float(os.getenv("SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30"))
        half_open_max_calls = int(os.getenv("SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", "3"))

        # Service-specific overrides
        if service_name:
            prefix = f"SENTINEL_{service_name.upper()}_CIRCUIT_BREAKER"
            if os.getenv(f"{prefix}_FAILURE_THRESHOLD"):
                failure_threshold = int(
                    os.getenv(f"{prefix}_FAILURE_THRESHOLD", str(failure_threshold))
                )
            if os.getenv(f"{prefix}_RECOVERY_TIMEOUT"):
                recovery_timeout = float(
                    os.getenv(f"{prefix}_RECOVERY_TIMEOUT", str(recovery_timeout))
                )
            if os.getenv(f"{prefix}_HALF_OPEN_MAX_CALLS"):
                half_open_max_calls = int(
                    os.getenv(f"{prefix}_HALF_OPEN_MAX_CALLS", str(half_open_max_calls))
                )

        return cls(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            enabled=enabled,
        )


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker state and operations.

    Attributes:
        total_calls: Total number of calls attempted.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        rejected_calls: Number of calls rejected due to open circuit.
        state_changes: Number of state transitions.
        last_failure_time: Timestamp of last failure.
        last_success_time: Timestamp of last success.
        last_state_change_time: Timestamp of last state change.
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    last_state_change_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "last_state_change_time": self.last_state_change_time,
        }


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for external service calls.

    Thread-safe circuit breaker that tracks failures and opens the circuit
    when failures exceed the configured threshold, preventing further calls
    to a failing service until recovery is attempted.

    Attributes:
        service_name: Name of the service this circuit breaker protects.
        config: Configuration for circuit breaker behavior.
        excluded_exceptions: Optional tuple of exception types that should not
            count as failures. Useful for excluding business exceptions or
            system exceptions like KeyboardInterrupt that shouldn't trigger
            circuit opening.

    Usage:
        cb = CircuitBreaker("jira")

        # Decorator usage
        @cb
        def call_jira_api():
            ...

        # Context manager usage
        with cb:
            call_jira_api()

        # Direct usage
        if cb.allow_request():
            try:
                result = call_jira_api()
                cb.record_success()
            except Exception as e:
                cb.record_failure(e)
                raise

        # With excluded exceptions
        cb = CircuitBreaker(
            "jira",
            excluded_exceptions=(KeyboardInterrupt, ValidationError)
        )
    """

    service_name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    excluded_exceptions: tuple[type[BaseException], ...] = field(default_factory=tuple)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _metrics: CircuitBreakerMetrics = field(default_factory=CircuitBreakerMetrics, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if the circuit breaker is currently in the OPEN state.

        Provides a clean, side-effect-free API for checking circuit state
        without requiring callers to import CircuitState. Unlike allow_request(),
        this property does not modify metrics or internal counters.

        Returns:
            True if the circuit is open (requests would be rejected), False otherwise.
        """
        return self.state == CircuitState.OPEN

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return self._metrics

    def _is_excluded_exception(self, exception: BaseException | None) -> bool:
        """Check if an exception type is in the excluded exceptions list.

        Args:
            exception: The exception to check, or None.

        Returns:
            True if the exception is excluded and should not count as a failure.
        """
        if exception is None or not self.excluded_exceptions:
            return False
        return isinstance(exception, self.excluded_exceptions)

    def _check_state_transition(self) -> None:
        """Check if circuit should transition state based on timeout."""
        time_since_failure = time.time() - self._last_failure_time
        if (
            self._state == CircuitState.OPEN
            and time_since_failure >= self.config.recovery_timeout
        ):
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging and metrics."""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change_time = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0

        logger.info(
            "[CIRCUIT_BREAKER] %s: State changed from %s to %s",
            self.service_name,
            old_state.value,
            new_state.value,
        )

    def allow_request(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        if not self.config.enabled:
            return True

        with self._lock:
            self._check_state_transition()
            self._metrics.total_calls += 1

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                self._metrics.rejected_calls += 1
                logger.warning(
                    "[CIRCUIT_BREAKER] %s: Request rejected, circuit is OPEN",
                    self.service_name,
                )
                return False

            # HALF_OPEN state - allow limited calls
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                logger.debug(
                    "[CIRCUIT_BREAKER] %s: HALF_OPEN call %s/%s",
                    self.service_name,
                    self._half_open_calls,
                    self.config.half_open_max_calls,
                )
                return True

            self._metrics.rejected_calls += 1
            logger.warning(
                "[CIRCUIT_BREAKER] %s: Request rejected, HALF_OPEN limit reached",
                self.service_name,
            )
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        if not self.config.enabled:
            return

        with self._lock:
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        "[CIRCUIT_BREAKER] %s: Recovery successful, circuit CLOSED",
                        self.service_name,
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def record_failure(self, exception: BaseException | None = None) -> None:
        """Record a failed call.

        If the exception type is in the excluded_exceptions list, the failure
        will not be recorded and the circuit state will not be affected.

        Args:
            exception: Optional exception that caused the failure.
        """
        if not self.config.enabled:
            return

        # Skip recording if this exception type is excluded
        if self._is_excluded_exception(exception):
            logger.debug(
                "[CIRCUIT_BREAKER] %s: Exception %s is excluded, not recording as failure",
                self.service_name,
                type(exception).__name__,
            )
            return

        with self._lock:
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = time.time()
            self._last_failure_time = time.time()

            error_info = f": {type(exception).__name__}: {exception}" if exception else ""
            logger.warning(
                "[CIRCUIT_BREAKER] %s: Failure recorded%s", self.service_name, error_info
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens the circuit
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "[CIRCUIT_BREAKER] %s: Recovery failed, circuit OPEN",
                    self.service_name,
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        "[CIRCUIT_BREAKER] %s: Failure threshold reached (%s/%s), circuit OPEN",
                        self.service_name,
                        self._failure_count,
                        self.config.failure_threshold,
                    )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state.

        Useful for testing or manual intervention.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0.0
            logger.info("[CIRCUIT_BREAKER] %s: Circuit manually reset to CLOSED", self.service_name)

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dictionary with state, config, and metrics.
        """
        with self._lock:
            self._check_state_transition()
            return {
                "service_name": self.service_name,
                "state": self._state.value,
                "enabled": self.config.enabled,
                "failure_count": self._failure_count,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "half_open_max_calls": self.config.half_open_max_calls,
                },
                "metrics": self._metrics.to_dict(),
            }

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker protection.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function that checks circuit state before execution.

        Raises:
            CircuitBreakerError: If circuit is open and request is rejected.

        Note:
            Exceptions in excluded_exceptions will be re-raised but will not
            count as failures for circuit breaker state transitions.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.allow_request():
                raise CircuitBreakerError(self.service_name, self._state)
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except BaseException as e:
                self.record_failure(e)
                raise

        return wrapper

    def __enter__(self) -> CircuitBreaker:
        """Context manager entry - check if request is allowed."""
        if not self.allow_request():
            raise CircuitBreakerError(self.service_name, self._state)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit - record success or failure.

        Note:
            Exceptions in excluded_exceptions will not count as failures
            for circuit breaker state transitions.
        """
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers across services.

    Provides a centralized way to access and manage circuit breakers for
    different services, ensuring each service has a single circuit breaker
    instance.

    Usage:
        registry = CircuitBreakerRegistry()
        jira_cb = registry.get("jira")
        github_cb = registry.get("github")
        claude_cb = registry.get("claude")
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(
        self,
        service_name: str,
        config: CircuitBreakerConfig | None = None,
        excluded_exceptions: tuple[type[BaseException], ...] | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a service.

        Args:
            service_name: Name of the service.
            config: Optional configuration. If not provided, loads from env.
            excluded_exceptions: Optional tuple of exception types that should
                not count as failures. Only used when creating a new circuit breaker.

        Returns:
            CircuitBreaker instance for the service.
        """
        with self._lock:
            if service_name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig.from_env(service_name)
                self._breakers[service_name] = CircuitBreaker(
                    service_name=service_name,
                    config=config,
                    excluded_exceptions=excluded_exceptions or (),
                )
                logger.info(
                    "[CIRCUIT_BREAKER] Created circuit breaker for %s: threshold=%s, timeout=%ss",
                    service_name,
                    config.failure_threshold,
                    config.recovery_timeout,
                )
            return self._breakers[service_name]

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered circuit breakers.

        Returns:
            Dictionary mapping service names to their status.
        """
        with self._lock:
            return {name: cb.get_status() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()
            logger.info("[CIRCUIT_BREAKER] All circuit breakers reset")
