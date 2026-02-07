"""Resilience wrapper coordinating circuit breaker and rate limiter.

This module provides a unified resilience wrapper that coordinates the CircuitBreaker
and ClaudeRateLimiter to avoid wasting rate limit tokens when the circuit breaker
is open. This prevents the scenario where:

1. Request comes in
2. Rate limiter consumes a token
3. Circuit breaker rejects the request immediately
4. Token is wasted

With the ResilienceWrapper, the circuit breaker state is checked BEFORE acquiring
a rate limit token, ensuring efficient use of rate limit capacity.

Usage:
    from sentinel.resilience import ResilienceWrapper
    from sentinel.circuit_breaker import CircuitBreakerRegistry

    # Create registry and get circuit breaker via dependency injection
    registry = CircuitBreakerRegistry()

    # Create wrapper with injected circuit breaker and rate limiter
    wrapper = ResilienceWrapper(
        circuit_breaker=registry.get("claude"),
        rate_limiter=ClaudeRateLimiter.from_config(config),
    )

    # Check circuit breaker state before acquiring rate limit token
    if wrapper.acquire(timeout=30.0):
        try:
            result = await call_claude_api()
            wrapper.record_success()
        except Exception as e:
            wrapper.record_failure(e)
            raise

    # Or use as a context manager
    with wrapper:
        result = call_claude_api()

    # Get combined metrics
    metrics = wrapper.get_metrics()
"""

from __future__ import annotations

import threading
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerError
from sentinel.logging import get_logger
from sentinel.rate_limiter import ClaudeRateLimiter, RateLimitExceededError

if TYPE_CHECKING:
    from sentinel.config import Config

logger = get_logger(__name__)


@dataclass
class ResilienceMetrics:
    """Metrics for resilience wrapper operations.

    Attributes:
        total_requests: Total number of acquire() calls.
        circuit_breaker_rejections: Requests rejected by circuit breaker before rate limiting.
        rate_limit_acquired: Requests that successfully acquired a rate limit token.
        rate_limit_rejections: Requests rejected by rate limiter after circuit breaker passed.
        successful_operations: Operations that completed successfully.
        failed_operations: Operations that failed (recorded as failures).
    """

    total_requests: int = 0
    circuit_breaker_rejections: int = 0
    rate_limit_acquired: int = 0
    rate_limit_rejections: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_circuit_breaker_rejection(self) -> None:
        """Record a request rejected by circuit breaker (token not consumed)."""
        with self._lock:
            self.total_requests += 1
            self.circuit_breaker_rejections += 1

    def record_rate_limit_acquired(self) -> None:
        """Record a successful rate limit token acquisition."""
        with self._lock:
            self.total_requests += 1
            self.rate_limit_acquired += 1

    def record_rate_limit_rejection(self) -> None:
        """Record a request rejected by rate limiter."""
        with self._lock:
            self.total_requests += 1
            self.rate_limit_rejections += 1

    def record_success(self) -> None:
        """Record a successful operation completion."""
        with self._lock:
            self.successful_operations += 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self.failed_operations += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/monitoring."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "circuit_breaker_rejections": self.circuit_breaker_rejections,
                "rate_limit_acquired": self.rate_limit_acquired,
                "rate_limit_rejections": self.rate_limit_rejections,
                "successful_operations": self.successful_operations,
                "failed_operations": self.failed_operations,
                # Tokens that would have been wasted without coordination
                "tokens_saved": self.circuit_breaker_rejections,
            }


class ResilienceWrapper:
    """Coordinated wrapper for circuit breaker and rate limiter.

    This wrapper ensures efficient coordination between the circuit breaker and
    rate limiter by checking the circuit breaker state BEFORE consuming rate
    limit tokens. This prevents token waste when the circuit is open.

    The wrapper provides:
    - Circuit breaker check before rate limit acquisition
    - Unified metrics tracking including tokens saved
    - Context manager and decorator support
    - Both synchronous and asynchronous acquire methods

    Thread Safety:
        This wrapper is designed to be thread-safe. The context manager's
        _in_context flag uses threading.local() storage, allowing the same
        ResilienceWrapper instance to be safely used as a context manager
        from multiple threads concurrently. Each thread maintains its own
        independent context state.

        Note: While the context manager is thread-safe, the underlying
        circuit breaker and rate limiter components have their own
        thread-safety characteristics. Refer to their respective
        documentation for details.

    Example:
        # Create registry for dependency injection
        registry = CircuitBreakerRegistry()

        wrapper = ResilienceWrapper(
            circuit_breaker=registry.get("claude"),
            rate_limiter=ClaudeRateLimiter.from_config(config),
        )

        # Synchronous usage
        if wrapper.acquire(timeout=30.0):
            try:
                result = call_api()
                wrapper.record_success()
            except Exception as e:
                wrapper.record_failure(e)
                raise

        # Async usage
        if await wrapper.acquire_async(timeout=30.0):
            try:
                result = await call_api_async()
                wrapper.record_success()
            except Exception as e:
                wrapper.record_failure(e)
                raise

        # Context manager usage (sync)
        with wrapper:
            result = call_api()

        # Context manager usage (async)
        async with wrapper:
            result = await call_api_async()

        # Thread-safe concurrent usage
        # Multiple threads can safely use the same wrapper as a context manager
        def worker():
            with wrapper:
                result = call_api()
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        rate_limiter: ClaudeRateLimiter,
    ) -> None:
        """Initialize the resilience wrapper.

        Args:
            circuit_breaker: CircuitBreaker instance to check before rate limiting.
            rate_limiter: ClaudeRateLimiter instance for rate limiting.
        """
        self._circuit_breaker = circuit_breaker
        self._rate_limiter = rate_limiter
        self._metrics = ResilienceMetrics()
        # Use thread-local storage for _in_context to ensure thread-safety
        # when the same wrapper instance is used concurrently from multiple threads.
        # Each thread maintains its own independent context state.
        self._local = threading.local()

    @classmethod
    def from_config(
        cls,
        config: Config,
        circuit_breaker: CircuitBreaker,
    ) -> ResilienceWrapper:
        """Create a resilience wrapper from application configuration.

        Args:
            config: Application configuration with rate limit settings.
            circuit_breaker: CircuitBreaker instance to use.

        Returns:
            Configured ResilienceWrapper instance.
        """
        rate_limiter = ClaudeRateLimiter.from_config(config)
        return cls(
            circuit_breaker=circuit_breaker,
            rate_limiter=rate_limiter,
        )

    @property
    def _in_context(self) -> bool:
        """Thread-local flag indicating if currently inside a context manager.

        Returns:
            True if the current thread is inside a context manager, False otherwise.
        """
        return getattr(self._local, "in_context", False)

    @_in_context.setter
    def _in_context(self, value: bool) -> None:
        """Set the thread-local context flag."""
        self._local.in_context = value

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the underlying circuit breaker."""
        return self._circuit_breaker

    @property
    def rate_limiter(self) -> ClaudeRateLimiter:
        """Get the underlying rate limiter."""
        return self._rate_limiter

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is in OPEN state.

        Returns:
            True if circuit is open (requests would be rejected), False otherwise.
        """
        return self._circuit_breaker.is_open

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire permission to make a request (synchronous).

        This method coordinates the circuit breaker and rate limiter:
        1. First, check if the circuit breaker allows the request
        2. Only if circuit allows, attempt to acquire a rate limit token
        3. Track metrics for both rejections and successful acquisitions

        Args:
            timeout: Maximum time to wait for rate limit token (seconds).
                    Passed to the underlying rate limiter.

        Returns:
            True if both circuit breaker allows and rate limit token acquired.
            False if circuit breaker rejects or rate limit times out.

        Raises:
            CircuitBreakerError: If circuit is open and should raise (depends on usage).
            RateLimitExceededError: If rate limiter is configured to raise on rejection.
        """
        # Check circuit breaker first - don't consume rate limit tokens if circuit is open
        if not self._circuit_breaker.allow_request():
            self._metrics.record_circuit_breaker_rejection()
            logger.debug(
                "Resilience: Request rejected by circuit breaker (state=%s), "
                "rate limit token NOT consumed",
                self._circuit_breaker.state.value,
            )
            return False

        # Circuit allows - now try to acquire rate limit token
        try:
            if self._rate_limiter.acquire(timeout=timeout):
                self._metrics.record_rate_limit_acquired()
                state = self._circuit_breaker.state.value
                logger.debug("Resilience: Request permitted (circuit=%s)", state)
                return True
            else:
                self._metrics.record_rate_limit_rejection()
                logger.debug("Resilience: Request rejected by rate limiter (timeout)")
                return False
        except RateLimitExceededError:
            self._metrics.record_rate_limit_rejection()
            raise

    async def acquire_async(self, timeout: float | None = None) -> bool:
        """Acquire permission to make a request (asynchronous).

        This method coordinates the circuit breaker and rate limiter:
        1. First, check if the circuit breaker allows the request
        2. Only if circuit allows, attempt to acquire a rate limit token
        3. Track metrics for both rejections and successful acquisitions

        Args:
            timeout: Maximum time to wait for rate limit token (seconds).
                    Passed to the underlying rate limiter.

        Returns:
            True if both circuit breaker allows and rate limit token acquired.
            False if circuit breaker rejects or rate limit times out.

        Raises:
            CircuitBreakerError: If circuit is open and should raise (depends on usage).
            RateLimitExceededError: If rate limiter is configured to raise on rejection.
        """
        # Check circuit breaker first - don't consume rate limit tokens if circuit is open
        if not self._circuit_breaker.allow_request():
            self._metrics.record_circuit_breaker_rejection()
            logger.debug(
                "Resilience: Request rejected by circuit breaker (state=%s), "
                "rate limit token NOT consumed",
                self._circuit_breaker.state.value,
            )
            return False

        # Circuit allows - now try to acquire rate limit token
        try:
            if await self._rate_limiter.acquire_async(timeout=timeout):
                self._metrics.record_rate_limit_acquired()
                state = self._circuit_breaker.state.value
                logger.debug("Resilience: Request permitted (circuit=%s)", state)
                return True
            else:
                self._metrics.record_rate_limit_rejection()
                logger.debug("Resilience: Request rejected by rate limiter (timeout)")
                return False
        except RateLimitExceededError:
            self._metrics.record_rate_limit_rejection()
            raise

    def record_success(self) -> None:
        """Record a successful operation.

        Should be called after the protected operation completes successfully.
        Updates both the circuit breaker and wrapper metrics.
        """
        self._circuit_breaker.record_success()
        self._metrics.record_success()

    def record_failure(self, exception: BaseException | None = None) -> None:
        """Record a failed operation.

        Should be called when the protected operation fails.
        Updates both the circuit breaker and wrapper metrics.

        Args:
            exception: Optional exception that caused the failure.
        """
        self._circuit_breaker.record_failure(exception)
        self._metrics.record_failure()

    def get_metrics(self) -> dict[str, Any]:
        """Get combined metrics from wrapper, circuit breaker, and rate limiter.

        Returns:
            Dictionary with combined metrics including:
            - wrapper_metrics: ResilienceMetrics data
            - circuit_breaker: CircuitBreaker status
            - rate_limiter: ClaudeRateLimiter metrics
        """
        return {
            "wrapper_metrics": self._metrics.to_dict(),
            "circuit_breaker": self._circuit_breaker.get_status(),
            "rate_limiter": self._rate_limiter.get_metrics(),
        }

    def reset_metrics(self) -> None:
        """Reset wrapper metrics to zero.

        Note: This only resets the wrapper's own metrics. To reset the underlying
        circuit breaker or rate limiter metrics, access them directly.
        """
        self._metrics = ResilienceMetrics()

    def __enter__(self) -> ResilienceWrapper:
        """Context manager entry - acquire permission.

        Raises:
            CircuitBreakerError: If circuit breaker rejects the request.
            RateLimitExceededError: If rate limiter rejects the request.
            RuntimeError: If nested context manager usage is attempted.
        """
        if self._in_context:
            raise RuntimeError(
                "ResilienceWrapper does not support nested context manager usage"
            )
        if not self.acquire():
            if self._circuit_breaker.is_open:
                raise CircuitBreakerError(
                    self._circuit_breaker.service_name,
                    self._circuit_breaker.state,
                )
            raise RateLimitExceededError("Rate limit exceeded in ResilienceWrapper")
        self._in_context = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit - record success or failure."""
        self._in_context = False
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)

    async def __aenter__(self) -> ResilienceWrapper:
        """Async context manager entry - acquire permission.

        Raises:
            CircuitBreakerError: If circuit breaker rejects the request.
            RateLimitExceededError: If rate limiter rejects the request.
            RuntimeError: If nested context manager usage is attempted.
        """
        if self._in_context:
            raise RuntimeError(
                "ResilienceWrapper does not support nested context manager usage"
            )
        if not await self.acquire_async():
            if self._circuit_breaker.is_open:
                raise CircuitBreakerError(
                    self._circuit_breaker.service_name,
                    self._circuit_breaker.state,
                )
            raise RateLimitExceededError("Rate limit exceeded in ResilienceWrapper")
        self._in_context = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit - record success or failure."""
        self._in_context = False
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
