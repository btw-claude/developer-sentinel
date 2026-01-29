"""Tests for Claude API rate limiter."""

import threading
import time
from typing import Any

import pytest

from sentinel.config import Config
from sentinel.rate_limiter import (
    ClaudeRateLimiter,
    RateLimiterMetrics,
    RateLimitExceededError,
    RateLimitStrategy,
    TokenBucket,
    _PausedMetrics,
)


class TestTokenBucket:
    """Tests for TokenBucket class."""

    def test_initial_tokens(self) -> None:
        """Token bucket starts with full capacity."""
        bucket = TokenBucket(requests_per_minute=10, requests_per_hour=100)
        status = bucket.get_status()
        assert status["minute_tokens"] == 10.0
        assert status["hour_tokens"] == 100.0

    def test_acquire_consumes_token(self) -> None:
        """Acquiring a token decrements both buckets."""
        bucket = TokenBucket(requests_per_minute=10, requests_per_hour=100)
        success, _, _ = bucket.try_acquire()
        assert success is True
        status = bucket.get_status()
        # Use approx due to potential token refill between operations
        assert status["minute_tokens"] == pytest.approx(9.0, abs=0.1)
        assert status["hour_tokens"] == pytest.approx(99.0, abs=0.1)

    def test_acquire_when_empty_fails(self) -> None:
        """Cannot acquire when bucket is empty."""
        bucket = TokenBucket(requests_per_minute=1, requests_per_hour=100)
        # First acquire should succeed
        success, _, _ = bucket.try_acquire()
        assert success is True
        # Second acquire should fail (minute bucket empty)
        success, wait_hint, _ = bucket.try_acquire()
        assert success is False
        assert wait_hint > 0  # Should suggest wait time

    def test_token_refill_over_time(self) -> None:
        """Tokens refill over time."""
        bucket = TokenBucket(requests_per_minute=60, requests_per_hour=1000)
        # Consume all minute tokens
        for _ in range(60):
            bucket.try_acquire()
        status = bucket.get_status()
        assert status["minute_tokens"] < 1.0

        # Wait for refill (60 req/min = 1 token/sec)
        time.sleep(0.1)
        status = bucket.get_status()
        # Should have some tokens back
        assert status["minute_tokens"] > 0.0

    def test_warning_threshold(self) -> None:
        """Warning is triggered when below threshold."""
        bucket = TokenBucket(requests_per_minute=10, requests_per_hour=100, warning_threshold=0.3)
        # Consume 8 tokens (leaving 2, which is 20% - below 30% threshold)
        for _ in range(8):
            bucket.try_acquire()
        _, _, warning = bucket.try_acquire()
        assert warning is True

    def test_no_warning_above_threshold(self) -> None:
        """No warning when above threshold."""
        bucket = TokenBucket(requests_per_minute=10, requests_per_hour=100, warning_threshold=0.2)
        # Consume 5 tokens (leaving 5, which is 50% - above 20% threshold)
        for _ in range(5):
            bucket.try_acquire()
        _, _, warning = bucket.try_acquire()
        assert warning is False

    def test_utilization_calculation(self) -> None:
        """Utilization is correctly calculated."""
        bucket = TokenBucket(requests_per_minute=10, requests_per_hour=100)
        # Consume 5 minute tokens
        for _ in range(5):
            bucket.try_acquire()
        status = bucket.get_status()
        assert status["minute_utilization"] == pytest.approx(0.5, abs=0.01)


class TestRateLimiterMetrics:
    """Tests for RateLimiterMetrics class."""

    def test_initial_metrics(self) -> None:
        """Metrics start at zero."""
        metrics = RateLimiterMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.rejected_requests == 0

    def test_record_successful_request(self) -> None:
        """Recording successful request updates counts."""
        metrics = RateLimiterMetrics()
        metrics.record_request(success=True, wait_time=0.5, was_queued=True)
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.queued_requests == 1
        assert metrics.total_wait_time == 0.5

    def test_record_rejected_request(self) -> None:
        """Recording rejected request updates counts."""
        metrics = RateLimiterMetrics()
        metrics.record_request(success=False)
        assert metrics.total_requests == 1
        assert metrics.rejected_requests == 1

    def test_to_dict(self) -> None:
        """to_dict returns all metrics."""
        metrics = RateLimiterMetrics()
        metrics.record_request(success=True, wait_time=1.0)
        metrics.record_request(success=True, wait_time=2.0, was_queued=True)
        metrics.record_warning()
        result = metrics.to_dict()
        assert result["total_requests"] == 2
        assert result["successful_requests"] == 2
        assert result["queued_requests"] == 1
        assert result["total_wait_time"] == 3.0
        assert result["warnings_issued"] == 1
        assert result["avg_wait_time"] == 1.5

    def test_thread_safety(self) -> None:
        """Metrics are thread-safe."""
        metrics = RateLimiterMetrics()
        num_threads = 10
        requests_per_thread = 100

        def record_requests() -> None:
            for _ in range(requests_per_thread):
                metrics.record_request(success=True)

        threads = [threading.Thread(target=record_requests) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.total_requests == num_threads * requests_per_thread


class TestClaudeRateLimiter:
    """Tests for ClaudeRateLimiter class."""

    def test_disabled_limiter_always_allows(self) -> None:
        """Disabled rate limiter always grants permits."""
        limiter = ClaudeRateLimiter(enabled=False)
        # Should always succeed even with exhausted theoretical limits
        for _ in range(100):
            assert limiter.acquire(timeout=0) is True

    def test_acquire_success(self) -> None:
        """Acquire succeeds when tokens available."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        assert limiter.acquire(timeout=0) is True
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1

    def test_acquire_reject_strategy(self) -> None:
        """Reject strategy raises exception when limit exceeded."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.REJECT,
            enabled=True,
        )
        # First request succeeds
        assert limiter.acquire(timeout=0) is True
        # Second request should raise
        with pytest.raises(RateLimitExceededError):
            limiter.acquire(timeout=0)

    def test_acquire_queue_strategy_waits(self) -> None:
        """Queue strategy waits for permit."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=60,  # 1 token per second refill
            requests_per_hour=1000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )
        # Exhaust minute bucket
        for _ in range(60):
            limiter.acquire(timeout=0.1)

        # Next acquire should wait and eventually succeed
        start = time.monotonic()
        result = limiter.acquire(timeout=2.0)  # Wait up to 2 seconds
        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed > 0.5  # Should have waited

    def test_acquire_queue_strategy_timeout(self) -> None:
        """Queue strategy returns False on timeout."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )
        limiter.acquire(timeout=0)  # Exhaust token
        # Short timeout should fail
        result = limiter.acquire(timeout=0.1)
        assert result is False

    def test_from_config(self) -> None:
        """Can create limiter from Config."""
        config = Config(
            claude_rate_limit_enabled=True,
            claude_rate_limit_per_minute=30,
            claude_rate_limit_per_hour=500,
            claude_rate_limit_strategy="reject",
            claude_rate_limit_warning_threshold=0.3,
        )
        limiter = ClaudeRateLimiter.from_config(config)
        assert limiter.enabled is True
        metrics = limiter.get_metrics()
        assert metrics["limits"]["requests_per_minute"] == 30
        assert metrics["limits"]["requests_per_hour"] == 500
        assert metrics["strategy"] == "reject"

    def test_get_metrics_includes_bucket_status(self) -> None:
        """get_metrics includes bucket status."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        limiter.acquire(timeout=0)
        metrics = limiter.get_metrics()
        assert "bucket_status" in metrics
        # Use approx due to potential token refill between operations
        assert metrics["bucket_status"]["minute_tokens"] == pytest.approx(9.0, abs=0.1)
        assert metrics["bucket_status"]["hour_tokens"] == pytest.approx(99.0, abs=0.1)

    def test_reset_metrics(self) -> None:
        """reset_metrics clears all counts."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        limiter.acquire(timeout=0)
        limiter.reset_metrics()
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 0

    def test_pause_metrics_context_manager_basic(self) -> None:
        """pause_metrics context manager pauses metrics recording."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        # Record some initial requests
        limiter.acquire(timeout=0)
        limiter.acquire(timeout=0)
        initial_metrics = limiter.get_metrics()
        assert initial_metrics["total_requests"] == 2

        # Requests inside pause_metrics should not be recorded
        with limiter.pause_metrics():
            limiter.acquire(timeout=0)
            limiter.acquire(timeout=0)
            limiter.acquire(timeout=0)

        # Metrics should still show only the original 2 requests
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 2

    def test_pause_metrics_with_reset(self) -> None:
        """pause_metrics allows safe reset of metrics."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        # Record some requests
        limiter.acquire(timeout=0)
        limiter.acquire(timeout=0)

        # Reset metrics within the pause context
        with limiter.pause_metrics():
            limiter.reset_metrics()

        # Metrics should be reset
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 0

        # New requests after reset should be recorded
        limiter.acquire(timeout=0)
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 1

    def test_pause_metrics_restores_on_exit_without_reset(self) -> None:
        """pause_metrics restores original metrics if reset is not called."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        # Record some requests
        limiter.acquire(timeout=0)
        limiter.acquire(timeout=0)

        # Enter pause context but don't reset
        with limiter.pause_metrics():
            pass  # Don't call reset_metrics()

        # Original metrics should be preserved
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 2

    def test_pause_metrics_exception_safety(self) -> None:
        """pause_metrics restores metrics on exception."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        # Record some requests
        limiter.acquire(timeout=0)
        limiter.acquire(timeout=0)

        # Enter pause context and raise exception
        try:
            with limiter.pause_metrics():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Original metrics should be preserved
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 2

    def test_pause_metrics_nested_not_recommended_but_safe(self) -> None:
        """Nested pause_metrics calls work safely (though not recommended)."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        limiter.acquire(timeout=0)

        with limiter.pause_metrics():
            with limiter.pause_metrics():
                limiter.reset_metrics()
            # Inner context exits, but outer context still paused

        # Metrics were reset, should be 0
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == 0


class TestPausedMetrics:
    """Tests for _PausedMetrics class."""

    def test_paused_metrics_discards_requests(self) -> None:
        """_PausedMetrics discards all request recordings."""
        metrics = _PausedMetrics()
        metrics.record_request(success=True, wait_time=1.0, was_queued=True)
        metrics.record_request(success=False)
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.rejected_requests == 0

    def test_paused_metrics_discards_warnings(self) -> None:
        """_PausedMetrics discards all warning recordings."""
        metrics = _PausedMetrics()
        metrics.record_warning()
        metrics.record_warning()
        assert metrics.warnings_issued == 0


class TestClaudeRateLimiterAsync:
    """Tests for async acquire method."""

    @pytest.mark.asyncio
    async def test_acquire_async_success(self) -> None:
        """Async acquire succeeds when tokens available."""
        limiter = ClaudeRateLimiter(requests_per_minute=10, requests_per_hour=100, enabled=True)
        result = await limiter.acquire_async(timeout=0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_async_reject_strategy(self) -> None:
        """Async reject strategy raises exception."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.REJECT,
            enabled=True,
        )
        await limiter.acquire_async(timeout=0)
        with pytest.raises(RateLimitExceededError):
            await limiter.acquire_async(timeout=0)

    @pytest.mark.asyncio
    async def test_acquire_async_queue_strategy_waits(self) -> None:
        """Async queue strategy waits using asyncio.sleep."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=60,  # 1 token per second refill
            requests_per_hour=1000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )
        # Exhaust minute bucket
        for _ in range(60):
            await limiter.acquire_async(timeout=0.1)

        # Next acquire should wait and eventually succeed
        start = time.monotonic()
        result = await limiter.acquire_async(timeout=2.0)
        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed > 0.5

    @pytest.mark.asyncio
    async def test_acquire_async_disabled(self) -> None:
        """Disabled limiter always allows in async mode."""
        limiter = ClaudeRateLimiter(enabled=False)
        for _ in range(100):
            assert await limiter.acquire_async(timeout=0) is True


class TestRateLimiterIntegration:
    """Integration tests for rate limiter behavior."""

    def test_sustained_rate_under_limit(self) -> None:
        """Sustained rate under limit succeeds without waiting."""
        # Allow 10 requests per minute
        limiter = ClaudeRateLimiter(
            requests_per_minute=10,
            requests_per_hour=1000,
            strategy=RateLimitStrategy.REJECT,
            enabled=True,
        )
        # Make 10 requests quickly - all should succeed
        for _ in range(10):
            assert limiter.acquire(timeout=0) is True

    def test_burst_then_throttle(self) -> None:
        """Burst exhausts tokens, then gets throttled."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=5,
            requests_per_hour=1000,
            strategy=RateLimitStrategy.REJECT,
            enabled=True,
        )
        # Burst 5 requests
        for _ in range(5):
            limiter.acquire(timeout=0)
        # 6th request should fail
        with pytest.raises(RateLimitExceededError):
            limiter.acquire(timeout=0)

    def test_warning_logged_when_approaching_limit(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning is logged when approaching rate limit."""
        import logging

        limiter = ClaudeRateLimiter(
            requests_per_minute=10,
            requests_per_hour=100,
            warning_threshold=0.3,
            enabled=True,
        )
        with caplog.at_level(logging.WARNING):
            # Consume 8 tokens (leaving 2, which is 20% - below 30% threshold)
            for _ in range(8):
                limiter.acquire(timeout=0)
            # This one should trigger warning
            limiter.acquire(timeout=0)
        assert "rate limit approaching threshold" in caplog.text.lower()


class TestConfigIntegration:
    """Tests for rate limiter configuration integration."""

    def test_load_config_with_rate_limit_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Config correctly loads rate limit settings from environment."""
        from sentinel.config import load_config

        # Set environment variables
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_MINUTE", "30")
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_HOUR", "500")
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY", "reject")
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD", "0.25")

        config = load_config()
        assert config.claude_rate_limit_enabled is True
        assert config.claude_rate_limit_per_minute == 30
        assert config.claude_rate_limit_per_hour == 500
        assert config.claude_rate_limit_strategy == "reject"
        assert config.claude_rate_limit_warning_threshold == 0.25

    def test_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config has sensible defaults for rate limiting."""
        from sentinel.config import load_config

        # Clear any existing env vars
        monkeypatch.delenv("SENTINEL_CLAUDE_RATE_LIMIT_ENABLED", raising=False)
        monkeypatch.delenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_MINUTE", raising=False)
        monkeypatch.delenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_HOUR", raising=False)
        monkeypatch.delenv("SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY", raising=False)
        monkeypatch.delenv("SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD", raising=False)

        config = load_config()
        assert config.claude_rate_limit_enabled is True
        assert config.claude_rate_limit_per_minute == 60
        assert config.claude_rate_limit_per_hour == 1000
        assert config.claude_rate_limit_strategy == "queue"
        assert config.claude_rate_limit_warning_threshold == 0.2

    def test_invalid_strategy_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid strategy value uses default."""
        import logging

        from sentinel.config import load_config

        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY", "invalid")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config.claude_rate_limit_strategy == "queue"
        assert "Invalid SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY" in caplog.text

    def test_invalid_threshold_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid threshold value uses default."""
        import logging

        from sentinel.config import load_config

        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD", "1.5")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config.claude_rate_limit_warning_threshold == 0.2
        assert "Invalid SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD" in caplog.text
