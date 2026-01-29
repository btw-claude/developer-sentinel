"""Rate limiting for Claude API calls using token bucket algorithm.

This module provides configurable rate limiting to control Claude API call frequency,
helping prevent runaway costs and API throttling. It supports both queuing and
rejection strategies when rate limits are reached.

The implementation uses a thread-safe token bucket algorithm that supports:
- Configurable requests per minute and per hour limits
- Warning thresholds when approaching limits
- Metrics collection for monitoring
- Optional request queuing with timeout

Usage:
    # Create rate limiter from config
    limiter = ClaudeRateLimiter.from_config(config)

    # Acquire permit before making API call
    if limiter.acquire(timeout=30.0):
        # Make Claude API call
        response = await query(prompt, options)
    else:
        raise RateLimitExceededError("Claude API rate limit exceeded")

    # Check metrics
    metrics = limiter.get_metrics()
    print(f"Total requests: {metrics['total_requests']}")
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.config import Config

logger = get_logger(__name__)


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded and request cannot be queued."""

    pass


class RateLimitStrategy(Enum):
    """Strategy for handling requests when rate limit is reached."""

    REJECT = "reject"  # Immediately reject with RateLimitExceededError
    QUEUE = "queue"  # Queue the request and wait for a permit


@dataclass
class RateLimiterMetrics:
    """Metrics collected by the rate limiter.

    Attributes:
        total_requests: Total number of acquire() calls.
        successful_requests: Number of requests that got a permit.
        rejected_requests: Number of requests rejected due to rate limit.
        queued_requests: Number of requests that had to wait for a permit.
        total_wait_time: Total time spent waiting for permits (seconds).
        warnings_issued: Number of times warning threshold was crossed.
    """

    total_requests: int = 0
    successful_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    total_wait_time: float = 0.0
    warnings_issued: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_request(
        self, success: bool, wait_time: float = 0.0, was_queued: bool = False
    ) -> None:
        """Record metrics for a request.

        Args:
            success: Whether the request got a permit.
            wait_time: Time spent waiting for the permit (seconds).
            was_queued: Whether the request had to wait in queue.
        """
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                self.total_wait_time += wait_time
                if was_queued:
                    self.queued_requests += 1
            else:
                self.rejected_requests += 1

    def record_warning(self) -> None:
        """Record that a rate limit warning was issued."""
        with self._lock:
            self.warnings_issued += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/monitoring."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "rejected_requests": self.rejected_requests,
                "queued_requests": self.queued_requests,
                "total_wait_time": self.total_wait_time,
                "warnings_issued": self.warnings_issued,
                "avg_wait_time": (
                    self.total_wait_time / self.successful_requests
                    if self.successful_requests > 0
                    else 0.0
                ),
            }


class TokenBucket:
    """Thread-safe token bucket implementation for rate limiting.

    Tokens are added at a steady rate up to a maximum capacity. Each request
    consumes one token. If no tokens are available, the request can either
    wait or be rejected.

    The bucket supports two rate limits:
    - Per-minute rate (higher granularity, lower capacity)
    - Per-hour rate (lower granularity, higher capacity)

    Both limits must be satisfied for a request to proceed.
    """

    def __init__(
        self,
        requests_per_minute: int,
        requests_per_hour: int,
        warning_threshold: float = 0.2,
    ) -> None:
        """Initialize the token bucket.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
            requests_per_hour: Maximum requests allowed per hour.
            warning_threshold: Fraction of tokens remaining that triggers warning (0.0-1.0).
        """
        self._lock = threading.Lock()

        # Per-minute bucket
        self._minute_capacity = float(requests_per_minute)
        self._minute_tokens = float(requests_per_minute)
        self._minute_refill_rate = requests_per_minute / 60.0  # tokens per second

        # Per-hour bucket
        self._hour_capacity = float(requests_per_hour)
        self._hour_tokens = float(requests_per_hour)
        self._hour_refill_rate = requests_per_hour / 3600.0  # tokens per second

        self._warning_threshold = warning_threshold
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time. Must be called with lock held."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill minute bucket
        self._minute_tokens = min(
            self._minute_capacity,
            self._minute_tokens + elapsed * self._minute_refill_rate,
        )

        # Refill hour bucket
        self._hour_tokens = min(
            self._hour_capacity,
            self._hour_tokens + elapsed * self._hour_refill_rate,
        )

    def try_acquire(self) -> tuple[bool, float, bool]:
        """Try to acquire a token without waiting.

        Returns:
            Tuple of (success, wait_time_hint, warning_triggered).
            - success: True if token was acquired.
            - wait_time_hint: Estimated seconds until a token is available (if not acquired).
            - warning_triggered: True if token levels are below warning threshold.
        """
        with self._lock:
            self._refill()

            warning_triggered = False

            # Check warning threshold on both buckets
            minute_fraction = self._minute_tokens / self._minute_capacity
            hour_fraction = self._hour_tokens / self._hour_capacity
            if (
                minute_fraction <= self._warning_threshold
                or hour_fraction <= self._warning_threshold
            ):
                warning_triggered = True

            # Check if we have tokens in both buckets
            if self._minute_tokens >= 1.0 and self._hour_tokens >= 1.0:
                self._minute_tokens -= 1.0
                self._hour_tokens -= 1.0
                return True, 0.0, warning_triggered

            # Calculate wait time hint (time until at least 1 token in each bucket)
            minute_wait = (
                (1.0 - self._minute_tokens) / self._minute_refill_rate
                if self._minute_tokens < 1.0
                else 0.0
            )
            hour_wait = (
                (1.0 - self._hour_tokens) / self._hour_refill_rate
                if self._hour_tokens < 1.0
                else 0.0
            )
            wait_hint = max(minute_wait, hour_wait)

            return False, wait_hint, warning_triggered

    def get_status(self) -> dict[str, Any]:
        """Get current bucket status for monitoring.

        Returns:
            Dictionary with current token levels and capacities.
        """
        with self._lock:
            self._refill()
            return {
                "minute_tokens": self._minute_tokens,
                "minute_capacity": self._minute_capacity,
                "minute_utilization": 1.0 - (self._minute_tokens / self._minute_capacity),
                "hour_tokens": self._hour_tokens,
                "hour_capacity": self._hour_capacity,
                "hour_utilization": 1.0 - (self._hour_tokens / self._hour_capacity),
            }


class ClaudeRateLimiter:
    """Rate limiter for Claude API calls.

    Provides configurable rate limiting using a dual token bucket algorithm
    (per-minute and per-hour limits). Supports both synchronous and asynchronous
    acquire methods.

    Example:
        limiter = ClaudeRateLimiter(
            requests_per_minute=10,
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
        )

        # Synchronous usage
        if limiter.acquire(timeout=30.0):
            make_api_call()

        # Async usage
        if await limiter.acquire_async(timeout=30.0):
            await make_api_call()
    """

    # Default timeout for queued requests (in seconds).
    # Used when timeout is not specified and strategy is QUEUE.
    DEFAULT_QUEUE_TIMEOUT: float = 60.0

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        strategy: RateLimitStrategy = RateLimitStrategy.QUEUE,
        warning_threshold: float = 0.2,
        enabled: bool = True,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
            requests_per_hour: Maximum requests allowed per hour.
            strategy: How to handle requests when rate limit is reached.
            warning_threshold: Fraction of tokens remaining that triggers warning.
            enabled: Whether rate limiting is enabled. If False, all requests pass through.
        """
        self._enabled = enabled
        self._strategy = strategy
        self._bucket = TokenBucket(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            warning_threshold=warning_threshold,
        )
        self._metrics = RateLimiterMetrics()
        self._requests_per_minute = requests_per_minute
        self._requests_per_hour = requests_per_hour
        self._warning_threshold = warning_threshold

    @classmethod
    def from_config(cls, config: Config) -> ClaudeRateLimiter:
        """Create a rate limiter from application configuration.

        Args:
            config: Application configuration with rate limit settings.

        Returns:
            Configured ClaudeRateLimiter instance.
        """
        strategy = (
            RateLimitStrategy.QUEUE
            if config.claude_rate_limit_strategy == "queue"
            else RateLimitStrategy.REJECT
        )
        return cls(
            requests_per_minute=config.claude_rate_limit_per_minute,
            requests_per_hour=config.claude_rate_limit_per_hour,
            strategy=strategy,
            warning_threshold=config.claude_rate_limit_warning_threshold,
            enabled=config.claude_rate_limit_enabled,
        )

    @property
    def enabled(self) -> bool:
        """Whether rate limiting is enabled."""
        return self._enabled

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a permit to make an API call (synchronous).

        Args:
            timeout: Maximum time to wait for a permit (seconds).
                    If None, uses default based on strategy:
                    - REJECT: 0 (no wait)
                    - QUEUE: 60 seconds

        Returns:
            True if permit was acquired, False otherwise.

        Raises:
            RateLimitExceededError: If strategy is REJECT and no permit available.
        """
        if not self._enabled:
            self._metrics.record_request(success=True)
            return True

        if timeout is None:
            timeout = (
                0.0 if self._strategy == RateLimitStrategy.REJECT else self.DEFAULT_QUEUE_TIMEOUT
            )

        start_time = time.monotonic()
        deadline = start_time + timeout
        was_queued = False

        while True:
            success, wait_hint, warning = self._bucket.try_acquire()

            if warning:
                self._metrics.record_warning()
                status = self._bucket.get_status()
                logger.warning(
                    "Claude API rate limit approaching threshold - "
                    f"minute: {status['minute_tokens']:.1f}/{status['minute_capacity']:.0f}, "
                    f"hour: {status['hour_tokens']:.1f}/{status['hour_capacity']:.0f}"
                )

            if success:
                wait_time = time.monotonic() - start_time
                self._metrics.record_request(
                    success=True, wait_time=wait_time, was_queued=was_queued
                )
                if was_queued:
                    logger.debug(
                        f"Claude API rate limit permit acquired after {wait_time:.2f}s wait"
                    )
                return True

            # Check if we've exceeded timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._metrics.record_request(success=False)
                if self._strategy == RateLimitStrategy.REJECT:
                    logger.warning("Claude API rate limit exceeded - request rejected")
                    raise RateLimitExceededError(
                        f"Claude API rate limit exceeded. "
                        f"Limit: {self._requests_per_minute}/min, {self._requests_per_hour}/hr"
                    )
                logger.warning(f"Claude API rate limit timeout after {timeout:.1f}s wait")
                return False

            # Wait before retrying
            was_queued = True
            wait_time = min(wait_hint, remaining, 1.0)  # Cap at 1 second intervals
            if wait_time > 0:
                time.sleep(wait_time)

    async def acquire_async(self, timeout: float | None = None) -> bool:
        """Acquire a permit to make an API call (asynchronous).

        Args:
            timeout: Maximum time to wait for a permit (seconds).
                    If None, uses default based on strategy.

        Returns:
            True if permit was acquired, False otherwise.

        Raises:
            RateLimitExceededError: If strategy is REJECT and no permit available.
        """
        if not self._enabled:
            self._metrics.record_request(success=True)
            return True

        if timeout is None:
            timeout = (
                0.0 if self._strategy == RateLimitStrategy.REJECT else self.DEFAULT_QUEUE_TIMEOUT
            )

        start_time = time.monotonic()
        deadline = start_time + timeout
        was_queued = False

        while True:
            success, wait_hint, warning = self._bucket.try_acquire()

            if warning:
                self._metrics.record_warning()
                status = self._bucket.get_status()
                logger.warning(
                    "Claude API rate limit approaching threshold - "
                    f"minute: {status['minute_tokens']:.1f}/{status['minute_capacity']:.0f}, "
                    f"hour: {status['hour_tokens']:.1f}/{status['hour_capacity']:.0f}"
                )

            if success:
                wait_time = time.monotonic() - start_time
                self._metrics.record_request(
                    success=True, wait_time=wait_time, was_queued=was_queued
                )
                if was_queued:
                    logger.debug(
                        f"Claude API rate limit permit acquired after {wait_time:.2f}s wait"
                    )
                return True

            # Check if we've exceeded timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._metrics.record_request(success=False)
                if self._strategy == RateLimitStrategy.REJECT:
                    logger.warning("Claude API rate limit exceeded - request rejected")
                    raise RateLimitExceededError(
                        f"Claude API rate limit exceeded. "
                        f"Limit: {self._requests_per_minute}/min, {self._requests_per_hour}/hr"
                    )
                logger.warning(f"Claude API rate limit timeout after {timeout:.1f}s wait")
                return False

            # Wait before retrying (async)
            was_queued = True
            wait_time = min(wait_hint, remaining, 1.0)  # Cap at 1 second intervals
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    def get_metrics(self) -> dict[str, Any]:
        """Get current rate limiter metrics.

        Returns:
            Dictionary with metrics including request counts and timing.
        """
        metrics = self._metrics.to_dict()
        metrics["bucket_status"] = self._bucket.get_status()
        metrics["enabled"] = self._enabled
        metrics["strategy"] = self._strategy.value
        metrics["limits"] = {
            "requests_per_minute": self._requests_per_minute,
            "requests_per_hour": self._requests_per_hour,
            "warning_threshold": self._warning_threshold,
        }
        return metrics

    def reset_metrics(self) -> None:
        """Reset collected metrics to zero.

        Note:
            This method should only be called when no requests are currently
            in flight to avoid potential race conditions with metrics recording.
            Calling this while requests are being processed may result in
            inconsistent metrics state.

            For safer resets, consider using the `pause_metrics()` context manager:

            ```python
            with limiter.pause_metrics():
                limiter.reset_metrics()
            ```
        """
        self._metrics = RateLimiterMetrics()

    @contextmanager
    def pause_metrics(self) -> Generator[None, None, None]:
        """Context manager that pauses metrics recording during the reset operation.

        This provides a safer API for resetting metrics by ensuring no new metrics
        are recorded during the reset operation. While inside this context manager,
        any acquire() or acquire_async() calls will still function normally but
        will not record metrics.

        Example:
            ```python
            with limiter.pause_metrics():
                # Metrics recording is paused
                limiter.reset_metrics()
            # Metrics recording resumes
            ```

        Yields:
            None

        Note:
            This context manager achieves thread-safety through object reference
            swapping rather than an explicit lock. The atomicity of this operation
            relies on CPython's Global Interpreter Lock (GIL), which ensures that
            attribute assignment is atomic. Any requests that start while metrics
            are paused will not have their metrics recorded. Requests that were
            already in progress when pause_metrics() was entered may still record
            metrics for their completion.

            This implementation may not be thread-safe on non-CPython interpreters
            that lack a GIL (e.g., Jython, IronPython, or future GIL-free CPython).
            If cross-interpreter compatibility is required, an explicit lock should
            be added.
        """
        # Store the current metrics object reference
        old_metrics = self._metrics

        # Create a "paused" metrics object that doesn't actually record anything
        paused_metrics = _PausedMetrics()

        # Swap in the paused metrics
        self._metrics = paused_metrics

        try:
            yield
        finally:
            # If reset_metrics() was called, self._metrics will be a new RateLimiterMetrics
            # If not called, self._metrics will still be paused_metrics
            # In either case, if it's still the paused one, restore the old metrics
            if self._metrics is paused_metrics:
                self._metrics = old_metrics


class _PausedMetrics(RateLimiterMetrics):
    """A metrics object that discards all recordings.

    Used by pause_metrics() context manager to prevent metrics recording
    during the pause period.
    """

    def record_request(
        self, success: bool, wait_time: float = 0.0, was_queued: bool = False
    ) -> None:
        """Discard the request recording (no-op)."""
        pass

    def record_warning(self) -> None:
        """Discard the warning recording (no-op)."""
        pass
