"""Unit tests for resilience wrapper coordinating circuit breaker and rate limiter."""

from __future__ import annotations

import time

import pytest

from sentinel.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)
from sentinel.rate_limiter import ClaudeRateLimiter, RateLimitExceededError, RateLimitStrategy
from sentinel.resilience import ResilienceMetrics, ResilienceWrapper


class TestResilienceMetrics:
    """Tests for ResilienceMetrics class."""

    def test_initial_values(self) -> None:
        """Test initial metric values are zero."""
        metrics = ResilienceMetrics()
        assert metrics.total_requests == 0
        assert metrics.circuit_breaker_rejections == 0
        assert metrics.rate_limit_acquired == 0
        assert metrics.rate_limit_rejections == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0

    def test_record_circuit_breaker_rejection(self) -> None:
        """Test recording circuit breaker rejection."""
        metrics = ResilienceMetrics()
        metrics.record_circuit_breaker_rejection()
        assert metrics.total_requests == 1
        assert metrics.circuit_breaker_rejections == 1
        assert metrics.rate_limit_acquired == 0

    def test_record_rate_limit_acquired(self) -> None:
        """Test recording successful rate limit acquisition."""
        metrics = ResilienceMetrics()
        metrics.record_rate_limit_acquired()
        assert metrics.total_requests == 1
        assert metrics.rate_limit_acquired == 1
        assert metrics.circuit_breaker_rejections == 0

    def test_record_rate_limit_rejection(self) -> None:
        """Test recording rate limit rejection."""
        metrics = ResilienceMetrics()
        metrics.record_rate_limit_rejection()
        assert metrics.total_requests == 1
        assert metrics.rate_limit_rejections == 1

    def test_record_success(self) -> None:
        """Test recording successful operation."""
        metrics = ResilienceMetrics()
        metrics.record_success()
        assert metrics.successful_operations == 1

    def test_record_failure(self) -> None:
        """Test recording failed operation."""
        metrics = ResilienceMetrics()
        metrics.record_failure()
        assert metrics.failed_operations == 1

    def test_to_dict(self) -> None:
        """Test metrics conversion to dictionary."""
        metrics = ResilienceMetrics()
        metrics.record_circuit_breaker_rejection()
        metrics.record_circuit_breaker_rejection()
        metrics.record_rate_limit_acquired()
        metrics.record_success()

        result = metrics.to_dict()
        assert result["total_requests"] == 3
        assert result["circuit_breaker_rejections"] == 2
        assert result["rate_limit_acquired"] == 1
        assert result["successful_operations"] == 1
        assert result["tokens_saved"] == 2  # Same as circuit_breaker_rejections


class TestResilienceWrapperBasic:
    """Basic tests for ResilienceWrapper."""

    def test_acquire_when_circuit_closed_and_tokens_available(self) -> None:
        """Test acquire succeeds when circuit is closed and tokens available."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        result = wrapper.acquire(timeout=0)
        assert result is True
        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["rate_limit_acquired"] == 1
        assert metrics["wrapper_metrics"]["circuit_breaker_rejections"] == 0

    def test_acquire_rejected_when_circuit_open(self) -> None:
        """Test acquire is rejected when circuit is open, token NOT consumed."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Open the circuit
        cb.record_failure(Exception("test error"))
        assert cb.state == CircuitState.OPEN

        # Acquire should fail without consuming rate limit token
        result = wrapper.acquire(timeout=0)
        assert result is False

        # Verify circuit breaker rejection was recorded, not rate limit
        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["circuit_breaker_rejections"] == 1
        assert metrics["wrapper_metrics"]["rate_limit_acquired"] == 0

        # Verify rate limiter token was NOT consumed
        limiter_metrics = limiter.get_metrics()
        assert limiter_metrics["bucket_status"]["minute_tokens"] == 10.0  # Full bucket


class TestResilienceWrapperTokenSaving:
    """Tests verifying that tokens are saved when circuit is open."""

    def test_no_token_consumed_when_circuit_open(self) -> None:
        """Core test: Verify rate limit token is NOT consumed when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        # Create rate limiter with very limited tokens to clearly see consumption
        limiter = ClaudeRateLimiter(requests_per_minute=2, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Get initial token count
        initial_tokens = limiter.get_metrics()["bucket_status"]["minute_tokens"]
        assert initial_tokens == 2.0

        # Open the circuit
        cb.record_failure(Exception("test error"))
        assert cb.state == CircuitState.OPEN

        # Make multiple requests - all should be rejected by circuit breaker
        for _ in range(5):
            result = wrapper.acquire(timeout=0)
            assert result is False

        # Verify NO tokens were consumed (all 2 tokens still available)
        final_tokens = limiter.get_metrics()["bucket_status"]["minute_tokens"]
        assert final_tokens == pytest.approx(2.0, abs=0.1)

        # Verify metrics show circuit breaker rejections
        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["circuit_breaker_rejections"] == 5
        assert metrics["wrapper_metrics"]["tokens_saved"] == 5

    def test_token_consumed_when_circuit_closed(self) -> None:
        """Verify rate limit token IS consumed when circuit is closed."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=5, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Get initial token count
        initial_tokens = limiter.get_metrics()["bucket_status"]["minute_tokens"]
        assert initial_tokens == 5.0

        # Make a request - should consume a token
        result = wrapper.acquire(timeout=0)
        assert result is True

        # Verify token was consumed
        final_tokens = limiter.get_metrics()["bucket_status"]["minute_tokens"]
        assert final_tokens == pytest.approx(4.0, abs=0.1)

    def test_mixed_scenario_circuit_transitions(self) -> None:
        """Test token consumption across circuit state transitions."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Phase 1: Circuit CLOSED - token should be consumed
        result = wrapper.acquire(timeout=0)
        assert result is True
        assert limiter.get_metrics()["bucket_status"]["minute_tokens"] == pytest.approx(9.0, abs=0.1)

        # Phase 2: Open the circuit
        cb.record_failure(Exception("test"))
        assert cb.state == CircuitState.OPEN

        # Phase 2: Circuit OPEN - token should NOT be consumed
        result = wrapper.acquire(timeout=0)
        assert result is False
        # Token count should still be ~9
        assert limiter.get_metrics()["bucket_status"]["minute_tokens"] == pytest.approx(9.0, abs=0.1)

        # Phase 3: Wait for half-open
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Phase 3: Circuit HALF_OPEN - token should be consumed (allows limited requests)
        result = wrapper.acquire(timeout=0)
        assert result is True
        assert limiter.get_metrics()["bucket_status"]["minute_tokens"] == pytest.approx(8.0, abs=0.1)


class TestResilienceWrapperRateLimiting:
    """Tests for rate limiter integration."""

    def test_rate_limit_rejection_when_tokens_exhausted(self) -> None:
        """Test rate limit rejection when tokens exhausted (circuit closed)."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(
            requests_per_minute=2,
            requests_per_hour=100,
            strategy=RateLimitStrategy.REJECT,
        )
        wrapper = ResilienceWrapper(cb, limiter)

        # Exhaust tokens
        wrapper.acquire(timeout=0)
        wrapper.acquire(timeout=0)

        # Next request should fail due to rate limit, not circuit breaker
        with pytest.raises(RateLimitExceededError):
            wrapper.acquire(timeout=0)

        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["rate_limit_acquired"] == 2
        assert metrics["wrapper_metrics"]["rate_limit_rejections"] == 1
        assert metrics["wrapper_metrics"]["circuit_breaker_rejections"] == 0

    def test_rate_limit_timeout_returns_false(self) -> None:
        """Test rate limit timeout returns False (queue strategy)."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
        )
        wrapper = ResilienceWrapper(cb, limiter)

        # Exhaust the one token
        wrapper.acquire(timeout=0)

        # Next request should timeout
        result = wrapper.acquire(timeout=0.1)
        assert result is False

        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["rate_limit_rejections"] == 1


class TestResilienceWrapperContextManager:
    """Tests for context manager usage."""

    def test_context_manager_success(self) -> None:
        """Test context manager records success on normal exit."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        with wrapper:
            pass  # Simulate successful operation

        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["successful_operations"] == 1
        assert cb.metrics.successful_calls == 1

    def test_context_manager_failure(self) -> None:
        """Test context manager records failure on exception."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        with pytest.raises(ValueError), wrapper:
            raise ValueError("test error")

        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["failed_operations"] == 1
        assert cb.metrics.failed_calls == 1

    def test_context_manager_raises_when_circuit_open(self) -> None:
        """Test context manager raises CircuitBreakerError when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Open the circuit
        cb.record_failure(Exception("test"))

        with pytest.raises(CircuitBreakerError) as exc_info:
            with wrapper:
                pass

        assert exc_info.value.service_name == "test"
        assert exc_info.value.state == CircuitState.OPEN

    def test_context_manager_raises_when_rate_limited(self) -> None:
        """Test context manager raises RateLimitExceededError when rate limited."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        # Use REJECT strategy so it raises immediately instead of timing out
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.REJECT,
        )
        wrapper = ResilienceWrapper(cb, limiter)

        # Exhaust tokens
        wrapper.acquire(timeout=0)

        with pytest.raises(RateLimitExceededError):
            with wrapper:
                pass


class TestResilienceWrapperRecording:
    """Tests for success/failure recording."""

    def test_record_success_updates_circuit_breaker(self) -> None:
        """Test record_success updates circuit breaker."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        wrapper.acquire(timeout=0)
        wrapper.record_success()

        assert cb.metrics.successful_calls == 1
        assert wrapper.get_metrics()["wrapper_metrics"]["successful_operations"] == 1

    def test_record_failure_updates_circuit_breaker(self) -> None:
        """Test record_failure updates circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        wrapper.acquire(timeout=0)
        wrapper.record_failure(ValueError("test"))

        assert cb.metrics.failed_calls == 1
        assert wrapper.get_metrics()["wrapper_metrics"]["failed_operations"] == 1


class TestResilienceWrapperAsync:
    """Tests for async acquire method."""

    @pytest.mark.asyncio
    async def test_acquire_async_success(self) -> None:
        """Test async acquire succeeds when circuit closed and tokens available."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        result = await wrapper.acquire_async(timeout=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_async_rejected_when_circuit_open(self) -> None:
        """Test async acquire rejected when circuit is open, token NOT consumed."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        # Open the circuit
        cb.record_failure(Exception("test"))

        result = await wrapper.acquire_async(timeout=0)
        assert result is False

        # Verify token was NOT consumed
        assert limiter.get_metrics()["bucket_status"]["minute_tokens"] == 10.0

    @pytest.mark.asyncio
    async def test_acquire_async_rate_limit_rejection(self) -> None:
        """Test async rate limit rejection."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.REJECT,
        )
        wrapper = ResilienceWrapper(cb, limiter)

        await wrapper.acquire_async(timeout=0)

        with pytest.raises(RateLimitExceededError):
            await wrapper.acquire_async(timeout=0)


class TestResilienceWrapperMetrics:
    """Tests for combined metrics."""

    def test_get_metrics_includes_all_components(self) -> None:
        """Test get_metrics includes wrapper, circuit breaker, and rate limiter metrics."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        wrapper.acquire(timeout=0)
        wrapper.record_success()

        metrics = wrapper.get_metrics()

        assert "wrapper_metrics" in metrics
        assert "circuit_breaker" in metrics
        assert "rate_limiter" in metrics

        assert metrics["wrapper_metrics"]["total_requests"] == 1
        assert metrics["circuit_breaker"]["service_name"] == "test"
        assert "bucket_status" in metrics["rate_limiter"]

    def test_reset_metrics(self) -> None:
        """Test reset_metrics clears wrapper metrics."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        wrapper.acquire(timeout=0)
        wrapper.record_success()

        wrapper.reset_metrics()

        metrics = wrapper.get_metrics()
        assert metrics["wrapper_metrics"]["total_requests"] == 0
        assert metrics["wrapper_metrics"]["successful_operations"] == 0


class TestResilienceWrapperHelpers:
    """Tests for helper methods."""

    def test_is_circuit_open(self) -> None:
        """Test is_circuit_open helper method."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        assert wrapper.is_circuit_open() is False

        cb.record_failure(Exception("test"))
        assert wrapper.is_circuit_open() is True

    def test_circuit_breaker_property(self) -> None:
        """Test circuit_breaker property returns underlying circuit breaker."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        assert wrapper.circuit_breaker is cb

    def test_rate_limiter_property(self) -> None:
        """Test rate_limiter property returns underlying rate limiter."""
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100)
        wrapper = ResilienceWrapper(cb, limiter)

        assert wrapper.rate_limiter is limiter
