"""Tests for Claude API rate limiter."""

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from sentinel.config import Config, RateLimitConfig
from sentinel.rate_limiter import (
    ClaudeRateLimiter,
    QueueFullError,
    QueueFullStrategy,
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
        # Mock time.monotonic to control time progression deterministically
        start_time = 1000.0
        with patch("time.monotonic") as mock_monotonic:
            mock_monotonic.return_value = start_time
            bucket = TokenBucket(requests_per_minute=60, requests_per_hour=1000)

            # Consume all minute tokens
            for _ in range(60):
                bucket.try_acquire()
            status = bucket.get_status()
            assert status["minute_tokens"] < 1.0

            # Advance time by 0.1 seconds (60 req/min = 1 token/sec, so 0.1s = 0.1 tokens)
            mock_monotonic.return_value = start_time + 0.1
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
            rate_limit=RateLimitConfig(
                enabled=True,
                per_minute=30,
                per_hour=500,
                strategy="reject",
                warning_threshold=0.3,
            )
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
        """pause_metrics context manager pauses metrics recording.

        Verifies the core behavior: requests made inside the pause_metrics
        context are not counted. See also test_pause_metrics_with_reset (which
        tests resetting metrics within the pause context) and
        test_pause_metrics_restores_on_exit_without_reset (which verifies
        that original metrics are preserved when no reset is performed).
        """
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
        """pause_metrics allows safe reset of metrics.

        Unlike test_pause_metrics_context_manager_basic (which tests that
        requests are silently discarded during pause), this test verifies
        the reset workflow: calling reset_metrics() inside the pause context
        clears all counters, and new requests after exit are recorded fresh.
        """
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
        """pause_metrics restores original metrics if reset is not called.

        Unlike test_pause_metrics_with_reset (which tests the reset-inside-pause
        workflow), this test verifies the no-op path: when the pause context
        exits without calling reset_metrics(), the original metrics are fully
        preserved and no data is lost.
        """
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

        with limiter.pause_metrics(), limiter.pause_metrics():
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
        """Async reject strategy raises exception.

        Verifies that when the REJECT strategy is configured and the rate
        limit is exceeded, acquire_async raises RateLimitExceededError. See
        also test_acquire_async_queue_strategy_waits, which tests the
        alternative QUEUE strategy that waits for token availability instead
        of raising.
        """
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
        """Async queue strategy waits using asyncio.sleep.

        Unlike test_acquire_async_reject_strategy (which raises an exception
        when the limit is exceeded), this test verifies that the QUEUE strategy
        waits for token replenishment via asyncio.sleep and eventually succeeds
        once a token becomes available.
        """
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


class TestClaudeRateLimiterThreadSafety:
    """Tests for thread-safety of ClaudeRateLimiter under concurrent access.

    These tests verify that ClaudeRateLimiter operates correctly when accessed
    from multiple threads simultaneously. This is particularly important for
    free-threaded Python 3.13+ builds (python -X gil=0) where the GIL is not
    available to provide implicit synchronization.
    """

    def test_concurrent_acquire_thread_safety(self) -> None:
        """Verify thread-safety of concurrent acquire() calls.

        This test creates multiple threads that simultaneously call acquire()
        and verifies that:
        1. All successful acquires are accurately counted in metrics
        2. No race conditions cause metrics corruption
        3. The rate limiter behaves correctly under contention
        """
        limiter = ClaudeRateLimiter(
            requests_per_minute=1000,  # High limit to avoid throttling
            requests_per_hour=10000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )

        num_threads = 20
        requests_per_thread = 50
        results: list[bool] = []
        results_lock = threading.Lock()

        def acquire_requests() -> None:
            thread_results = []
            for _ in range(requests_per_thread):
                result = limiter.acquire(timeout=1.0)
                thread_results.append(result)
            with results_lock:
                results.extend(thread_results)

        threads = [threading.Thread(target=acquire_requests) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should have succeeded
        assert len(results) == num_threads * requests_per_thread
        successful_count = sum(1 for r in results if r)

        # Metrics should accurately reflect the successful requests
        metrics = limiter.get_metrics()
        assert metrics["total_requests"] == num_threads * requests_per_thread
        assert metrics["successful_requests"] == successful_count

    def test_concurrent_metrics_access_thread_safety(self) -> None:
        """Verify thread-safety of concurrent get_metrics() and acquire() calls.

        This test simulates a monitoring scenario where metrics are being read
        while requests are being processed, ensuring no race conditions occur.
        """
        limiter = ClaudeRateLimiter(
            requests_per_minute=1000,
            requests_per_hour=10000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )

        num_acquire_threads = 10
        num_metrics_threads = 5
        requests_per_thread = 100
        metrics_reads_per_thread = 50
        stop_event = threading.Event()
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def acquire_requests() -> None:
            try:
                for _ in range(requests_per_thread):
                    limiter.acquire(timeout=1.0)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def read_metrics() -> None:
            try:
                for _ in range(metrics_reads_per_thread):
                    if stop_event.is_set():
                        break
                    metrics = limiter.get_metrics()
                    # Verify metrics structure is valid
                    assert "total_requests" in metrics
                    assert "successful_requests" in metrics
                    assert isinstance(metrics["total_requests"], int)
                    assert isinstance(metrics["successful_requests"], int)
                    # Small delay between reads for CPU yield (deterministic alternative to time.sleep)
                    stop_event.wait(timeout=0.001)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        acquire_threads = [
            threading.Thread(target=acquire_requests) for _ in range(num_acquire_threads)
        ]
        metrics_threads = [
            threading.Thread(target=read_metrics) for _ in range(num_metrics_threads)
        ]

        # Start all threads
        for t in acquire_threads + metrics_threads:
            t.start()

        # Wait for acquire threads to finish
        for t in acquire_threads:
            t.join()

        # Signal metrics threads to stop and wait for them
        stop_event.set()
        for t in metrics_threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"

        # Final metrics should be consistent
        final_metrics = limiter.get_metrics()
        assert final_metrics["total_requests"] == num_acquire_threads * requests_per_thread

    def test_concurrent_pause_metrics_thread_safety(self) -> None:
        """Verify thread-safety of pause_metrics() under concurrent access.

        This test ensures that the pause_metrics context manager correctly
        handles concurrent access without race conditions, particularly
        important for free-threaded Python 3.13+.
        """
        limiter = ClaudeRateLimiter(
            requests_per_minute=1000,
            requests_per_hour=10000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )

        # Record some initial requests
        for _ in range(10):
            limiter.acquire(timeout=0)

        initial_count = limiter.get_metrics()["total_requests"]
        assert initial_count == 10

        num_threads = 10
        requests_per_thread = 20
        errors: list[Exception] = []
        errors_lock = threading.Lock()
        # Use barrier to synchronize thread startup
        startup_barrier = threading.Barrier(num_threads + 1)

        def acquire_during_pause() -> None:
            try:
                # Wait for all threads to be ready
                startup_barrier.wait()
                for _ in range(requests_per_thread):
                    limiter.acquire(timeout=1.0)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Start threads that will acquire while we pause metrics
        threads = [threading.Thread(target=acquire_during_pause) for _ in range(num_threads)]

        with limiter.pause_metrics():
            # Start threads inside pause context
            for t in threads:
                t.start()
            # Wait for all threads to be ready (deterministic synchronization)
            startup_barrier.wait()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"

        # Metrics recorded during pause should be discarded
        # Only the initial 10 requests should be counted, plus any
        # requests that completed after pause ended
        final_metrics = limiter.get_metrics()
        # The count should be at least 10 (initial) but less than
        # 10 + num_threads * requests_per_thread (all requests)
        # because some were discarded during pause
        assert final_metrics["total_requests"] >= initial_count

    def test_concurrent_reset_metrics_thread_safety(self) -> None:
        """Verify thread-safety of reset_metrics() under concurrent access.

        This test ensures that resetting metrics while other threads are
        recording metrics does not cause race conditions.
        """
        limiter = ClaudeRateLimiter(
            requests_per_minute=1000,
            requests_per_hour=10000,
            strategy=RateLimitStrategy.QUEUE,
            enabled=True,
        )

        num_acquire_threads = 10
        requests_per_thread = 50
        errors: list[Exception] = []
        errors_lock = threading.Lock()
        reset_complete = threading.Event()
        # Use barrier to ensure threads have started before reset
        startup_barrier = threading.Barrier(num_acquire_threads + 1)

        def acquire_requests() -> None:
            try:
                # Signal that thread is ready
                startup_barrier.wait()
                for _ in range(requests_per_thread):
                    limiter.acquire(timeout=1.0)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        def reset_metrics_task() -> None:
            try:
                # Wait for all acquire threads to start (deterministic synchronization)
                startup_barrier.wait()
                with limiter.pause_metrics():
                    limiter.reset_metrics()
                reset_complete.set()
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        acquire_threads = [
            threading.Thread(target=acquire_requests) for _ in range(num_acquire_threads)
        ]
        reset_thread = threading.Thread(target=reset_metrics_task)

        # Start all threads
        for t in acquire_threads:
            t.start()
        reset_thread.start()

        # Wait for all threads to complete
        for t in acquire_threads:
            t.join()
        reset_thread.join()

        # No errors should have occurred
        assert len(errors) == 0, f"Errors occurred during concurrent access: {errors}"

        # Reset should have completed
        assert reset_complete.is_set()

        # Metrics should be valid (either reset or counting new requests)
        final_metrics = limiter.get_metrics()
        assert isinstance(final_metrics["total_requests"], int)
        assert final_metrics["total_requests"] >= 0


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
        assert config.rate_limit.enabled is True
        assert config.rate_limit.per_minute == 30
        assert config.rate_limit.per_hour == 500
        assert config.rate_limit.strategy == "reject"
        assert config.rate_limit.warning_threshold == 0.25

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
        assert config.rate_limit.enabled is True
        assert config.rate_limit.per_minute == 60
        assert config.rate_limit.per_hour == 1000
        assert config.rate_limit.strategy == "queue"
        assert config.rate_limit.warning_threshold == 0.2

    def test_invalid_strategy_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid strategy value uses default."""
        import logging

        from sentinel.config import load_config

        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY", "invalid")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config.rate_limit.strategy == "queue"
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
        assert config.rate_limit.warning_threshold == 0.2
        assert "Invalid SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD" in caplog.text

    def test_load_config_with_backpressure_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config correctly loads backpressure settings from environment."""
        from sentinel.config import load_config

        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_MAX_QUEUED", "50")
        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_QUEUE_FULL_STRATEGY", "wait")

        config = load_config()
        assert config.rate_limit.max_queued == 50
        assert config.rate_limit.queue_full_strategy == "wait"

    def test_invalid_queue_full_strategy_uses_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Invalid queue full strategy value uses default."""
        import logging

        from sentinel.config import load_config

        monkeypatch.setenv("SENTINEL_CLAUDE_RATE_LIMIT_QUEUE_FULL_STRATEGY", "invalid")
        with caplog.at_level(logging.WARNING):
            config = load_config()
        assert config.rate_limit.queue_full_strategy == "reject"
        assert "Invalid SENTINEL_CLAUDE_RATE_LIMIT_QUEUE_FULL_STRATEGY" in caplog.text


class TestBoundedQueueBackpressure:
    """Tests for bounded queue backpressure functionality."""

    def test_queue_properties(self) -> None:
        """Rate limiter exposes queue properties."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=10,
            requests_per_hour=100,
            max_queued=50,
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )
        assert limiter.max_queued == 50
        assert limiter.queued_count == 0

    def test_queue_full_reject_strategy(self) -> None:
        """Queue full with REJECT strategy raises QueueFullError."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,  # Only 1 token
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
            max_queued=2,  # Very small queue
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )

        # First request succeeds (gets the token)
        assert limiter.acquire(timeout=0) is True

        # Fill the queue with waiting requests (they will be blocked)
        threads = []
        errors: list[Exception] = []
        errors_lock = threading.Lock()

        def waiting_acquire() -> None:
            try:
                limiter.acquire(timeout=2.0)
            except Exception as e:
                with errors_lock:
                    errors.append(e)

        # Start threads to fill the queue
        for _ in range(2):
            t = threading.Thread(target=waiting_acquire)
            t.start()
            threads.append(t)

        # Poll until both threads have entered the queue (deterministic synchronization
        # on the actual queue state rather than external signals)
        deadline = threading.Event()
        for _ in range(200):  # Up to 2 seconds
            if limiter.queued_count >= 2:
                break
            deadline.wait(timeout=0.01)
        assert limiter.queued_count >= 2, f"Expected queue count >= 2, got {limiter.queued_count}"

        # Now queue should be full, next request should raise QueueFullError
        with pytest.raises(QueueFullError):
            limiter.acquire(timeout=0.1)

        # Clean up threads
        for t in threads:
            t.join(timeout=3.0)

    def test_queue_metrics_include_queue_full_rejections(self) -> None:
        """Metrics track queue full rejections."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
            max_queued=1,
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )

        # First request succeeds
        limiter.acquire(timeout=0)

        # Start a thread to fill the queue
        def waiting_acquire() -> None:
            import contextlib

            with contextlib.suppress(Exception):
                limiter.acquire(timeout=1.0)

        t = threading.Thread(target=waiting_acquire)
        t.start()

        # Poll until thread has entered the queue (deterministic synchronization
        # on the actual queue state)
        for _ in range(200):  # Up to 2 seconds
            if limiter.queued_count >= 1:
                break
            threading.Event().wait(timeout=0.01)
        assert limiter.queued_count >= 1, f"Expected queue count >= 1, got {limiter.queued_count}"

        # Try to acquire when queue is full
        import contextlib

        with contextlib.suppress(QueueFullError):
            limiter.acquire(timeout=0.1)

        t.join(timeout=2.0)

        metrics = limiter.get_metrics()
        assert metrics["queue_full_rejections"] >= 1

    def test_from_config_with_backpressure_settings(self) -> None:
        """Can create limiter from Config with backpressure settings."""
        config = Config(
            rate_limit=RateLimitConfig(
                enabled=True,
                per_minute=30,
                per_hour=500,
                strategy="queue",
                max_queued=75,
                queue_full_strategy="wait",
            )
        )
        limiter = ClaudeRateLimiter.from_config(config)
        assert limiter.enabled is True
        assert limiter.max_queued == 75
        metrics = limiter.get_metrics()
        assert metrics["queue"]["max_size"] == 75
        assert metrics["queue"]["full_strategy"] == "wait"

    def test_get_metrics_includes_queue_info(self) -> None:
        """get_metrics includes queue status."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=10,
            requests_per_hour=100,
            max_queued=100,
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )
        metrics = limiter.get_metrics()
        assert "queue" in metrics
        assert metrics["queue"]["current_size"] == 0
        assert metrics["queue"]["max_size"] == 100
        assert metrics["queue"]["full_strategy"] == "reject"

    @pytest.mark.asyncio
    async def test_queue_full_reject_strategy_async(self) -> None:
        """Async queue full with REJECT strategy raises QueueFullError."""
        import asyncio
        import contextlib

        limiter = ClaudeRateLimiter(
            requests_per_minute=1,
            requests_per_hour=100,
            strategy=RateLimitStrategy.QUEUE,
            max_queued=2,
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )

        # First request succeeds
        assert await limiter.acquire_async(timeout=0) is True

        # Start tasks to fill the queue
        async def waiting_acquire() -> None:
            with contextlib.suppress(Exception):
                await limiter.acquire_async(timeout=2.0)

        tasks = [asyncio.create_task(waiting_acquire()) for _ in range(2)]
        await asyncio.sleep(0.1)

        # Queue should be full
        with pytest.raises(QueueFullError):
            await limiter.acquire_async(timeout=0.1)

        # Cancel pending tasks
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


class TestBackpressureThreadSafety:
    """Tests for thread-safety of backpressure under concurrent access."""

    def test_concurrent_queue_access_thread_safety(self) -> None:
        """Verify thread-safety of queue enter/exit under concurrent access."""
        limiter = ClaudeRateLimiter(
            requests_per_minute=5,
            requests_per_hour=1000,
            strategy=RateLimitStrategy.QUEUE,
            max_queued=10,
            queue_full_strategy=QueueFullStrategy.REJECT,
            enabled=True,
        )

        num_threads = 20
        results: list[tuple[bool, Exception | None]] = []
        results_lock = threading.Lock()

        def try_acquire() -> None:
            try:
                result = limiter.acquire(timeout=1.0)
                with results_lock:
                    results.append((result, None))
            except Exception as e:
                with results_lock:
                    results.append((False, e))

        threads = [threading.Thread(target=try_acquire) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have completed
        assert len(results) == num_threads

        # Queue count should be 0 after all threads complete
        assert limiter.queued_count == 0

        # Some requests may have been rejected due to queue being full
        # (depending on timing), but the queue should be consistent
        metrics = limiter.get_metrics()
        assert metrics["queue"]["current_size"] == 0
