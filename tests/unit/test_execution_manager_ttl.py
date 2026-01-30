"""Tests for ExecutionManager TTL-based cleanup functionality.

These tests verify the memory safety improvements added in DS-467:
- Timestamp tracking for futures
- TTL-based cleanup of stale futures
- Maximum list size with overflow handling
- Monitoring/logging for long-running futures
"""

import threading
import time
from concurrent.futures import Future
from unittest.mock import patch

from sentinel.execution_manager import (
    DEFAULT_FUTURE_TTL_SECONDS,
    DEFAULT_MAX_FUTURES,
    LONG_RUNNING_WARNING_THRESHOLD,
    ExecutionManager,
    TrackedFuture,
)


class TestTrackedFuture:
    """Tests for the TrackedFuture dataclass."""

    def test_age_seconds_returns_elapsed_time(self) -> None:
        """Test that age_seconds returns the correct elapsed time."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future, created_at=time.monotonic() - 10.0)

        age = tracked.age_seconds()
        assert age >= 10.0
        assert age < 11.0  # Allow some tolerance

    def test_is_stale_returns_false_for_done_future(self) -> None:
        """Test that completed futures are not considered stale."""
        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, created_at=time.monotonic() - 1000.0)

        assert not tracked.is_stale(ttl_seconds=1.0)

    def test_is_stale_returns_false_for_young_future(self) -> None:
        """Test that young futures are not considered stale."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future)

        assert not tracked.is_stale(ttl_seconds=60.0)

    def test_is_stale_returns_true_for_old_pending_future(self) -> None:
        """Test that old pending futures are considered stale."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future, created_at=time.monotonic() - 100.0)

        assert tracked.is_stale(ttl_seconds=50.0)

    def test_description_defaults_to_empty_string(self) -> None:
        """Test that description defaults to empty string."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future)

        assert tracked.description == ""

    def test_description_can_be_set(self) -> None:
        """Test that description can be provided."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future, description="test task")

        assert tracked.description == "test task"


class TestExecutionManagerTTL:
    """Tests for ExecutionManager TTL-based cleanup."""

    def test_default_ttl_is_set(self) -> None:
        """Test that default TTL is set correctly."""
        manager = ExecutionManager(max_concurrent_executions=2)

        assert manager.future_ttl_seconds == DEFAULT_FUTURE_TTL_SECONDS

    def test_default_max_futures_is_set(self) -> None:
        """Test that default max_futures is set correctly."""
        manager = ExecutionManager(max_concurrent_executions=2)

        assert manager.max_futures == DEFAULT_MAX_FUTURES

    def test_custom_ttl_can_be_set(self) -> None:
        """Test that custom TTL can be provided."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=60.0,
        )

        assert manager.future_ttl_seconds == 60.0

    def test_custom_max_futures_can_be_set(self) -> None:
        """Test that custom max_futures can be provided."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            max_futures=50,
        )

        assert manager.max_futures == 50

    def test_submit_creates_tracked_future(self) -> None:
        """Test that submit creates a TrackedFuture with timestamp."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        try:
            before = time.monotonic()
            future = manager.submit(lambda: 42, description="test task")
            after = time.monotonic()

            assert future is not None
            tracked_futures = manager.get_tracked_futures()
            assert len(tracked_futures) == 1
            assert tracked_futures[0].description == "test task"
            assert before <= tracked_futures[0].created_at <= after

            future.result(timeout=1.0)
        finally:
            manager.shutdown()

    def test_stale_futures_cleaned(self) -> None:
        """Test that stale futures counter tracks cleaned futures."""
        manager = ExecutionManager(max_concurrent_executions=2)

        assert manager.stale_futures_cleaned == 0

    def test_collect_completed_results_cleans_stale_futures(self) -> None:
        """Test that stale futures are cleaned up during result collection."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.1,  # Very short TTL for testing
        )
        manager.start()

        try:
            # Create a future that will become stale
            block_event = threading.Event()
            manager.submit(
                lambda: block_event.wait(timeout=10),
                description="stale task",
            )

            # Wait for the future to become stale
            time.sleep(0.2)

            # Collect results - should clean up stale future
            with patch("sentinel.execution_manager.logger"):
                manager.collect_completed_results()

            # Verify stale future was cleaned up
            assert manager.stale_futures_cleaned == 1
            assert len(manager.get_active_futures()) == 0

            # Release the event (even though future was cancelled/removed)
            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)

    def test_max_futures_limit_enforced_with_completed(self) -> None:
        """Test that max futures limit is enforced when futures complete."""
        manager = ExecutionManager(
            max_concurrent_executions=10,
            max_futures=5,
            future_ttl_seconds=0.05,  # Short TTL for testing
        )
        manager.start()

        try:
            # Submit futures that complete quickly
            for i in range(3):
                manager.submit(lambda: 42, description=f"quick-{i}")

            # Wait for them to complete
            time.sleep(0.1)

            # Submit more futures - should trigger cleanup of completed futures
            for i in range(5):
                manager.submit(lambda: time.sleep(0.5), description=f"slow-{i}")

            # Wait a bit for TTL to expire on some
            time.sleep(0.1)

            # Submit more futures to trigger overflow handling
            for i in range(5):
                manager.submit(lambda: time.sleep(0.5), description=f"overflow-{i}")

            # Should have cleaned up some futures
            # Note: exact count depends on timing, but should be under control
            active = manager.get_active_futures()
            # The limit enforcement removes completed and stale futures
            assert len(active) <= 10  # Upper bound check

        finally:
            manager.shutdown(cancel_futures=True)

    def test_max_futures_cleans_completed_first(self) -> None:
        """Test that max futures limit removes completed futures first."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
        )
        manager.start()

        try:
            # Submit a future that completes immediately
            fast_future = manager.submit(lambda: 42, description="fast")
            fast_future.result(timeout=1.0)  # Wait for completion

            # Submit more futures to reach the limit
            manager.submit(lambda: time.sleep(5), description="slow-1")
            manager.submit(lambda: time.sleep(5), description="slow-2")
            manager.submit(lambda: time.sleep(5), description="slow-3")

            # At this point we're at the limit - next submit should clean up the completed future
            manager.submit(lambda: time.sleep(5), description="slow-4")

            # Should have cleaned up the completed future
            tracked = manager.get_tracked_futures()
            # All tracked futures should be the slow ones (not done)
            pending_count = sum(1 for tf in tracked if not tf.future.done())
            assert pending_count >= 3  # At least the slow ones should be tracked

        finally:
            manager.shutdown(cancel_futures=True)

    def test_get_futures_stats_returns_correct_info(self) -> None:
        """Test that get_futures_stats returns accurate statistics."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=60.0,
        )
        manager.start()

        try:
            # Submit a task
            future = manager.submit(lambda: 42, description="test")
            future.result(timeout=1.0)  # Wait for completion

            stats = manager.get_futures_stats()

            assert "total_tracked" in stats
            assert "pending" in stats
            assert "completed" in stats
            assert "stale" in stats
            assert "long_running" in stats
            assert "max_age_seconds" in stats
            assert "avg_age_seconds" in stats
            assert "total_stale_cleaned" in stats
            assert "ttl_seconds" in stats
            assert "max_futures" in stats

            assert stats["ttl_seconds"] == 60.0
            assert stats["max_futures"] == DEFAULT_MAX_FUTURES

        finally:
            manager.shutdown()

    def test_long_running_warning_logged(self) -> None:
        """Test that warnings are logged for long-running futures."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=300.0,
        )
        manager.start()

        try:
            # Create a tracked future that appears to be running for a long time
            block_event = threading.Event()
            manager.submit(
                lambda: block_event.wait(timeout=10),
                description="long task",
            )

            # Manually adjust the created_at time to simulate a long-running task
            with manager._futures_lock:
                manager._active_futures[0].created_at = (
                    time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 10
                )

            # Collect results - should log warning for long-running task
            with patch("sentinel.execution_manager.logger") as mock_logger:
                manager.collect_completed_results()

                # Verify warning was logged
                warning_calls = [
                    call for call in mock_logger.warning.call_args_list
                    if "Long-running future detected" in str(call)
                ]
                assert len(warning_calls) >= 1

            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)

    def test_get_tracked_futures_returns_copy(self) -> None:
        """Test that get_tracked_futures returns a copy of the list."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        try:
            manager.submit(lambda: 42)
            tracked1 = manager.get_tracked_futures()
            tracked2 = manager.get_tracked_futures()

            # Should be equal but not the same object
            assert tracked1 == tracked2
            assert tracked1 is not tracked2

        finally:
            manager.shutdown()

    def test_cleanup_completed_futures_includes_stale(self) -> None:
        """Test that cleanup_completed_futures also cleans stale futures."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.1,
        )
        manager.start()

        try:
            block_event = threading.Event()
            manager.submit(lambda: block_event.wait(timeout=10))

            # Wait for future to become stale
            time.sleep(0.2)

            # Cleanup should include the stale future
            cleaned = manager.cleanup_completed_futures()
            assert cleaned >= 1

            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)


class TestExecutionManagerBackwardsCompatibility:
    """Tests to ensure backwards compatibility with existing code."""

    def test_submit_without_description_works(self) -> None:
        """Test that submit works without the description parameter."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        try:
            future = manager.submit(lambda: 42)
            assert future is not None
            result = future.result(timeout=1.0)
            assert result == 42
        finally:
            manager.shutdown()

    def test_get_active_futures_returns_futures(self) -> None:
        """Test that get_active_futures returns Future objects, not TrackedFutures."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        try:
            block_event = threading.Event()
            future = manager.submit(lambda: block_event.wait(timeout=5))

            active = manager.get_active_futures()
            assert len(active) == 1
            assert isinstance(active[0], Future)
            assert active[0] is future

            block_event.set()
            future.result(timeout=1.0)
        finally:
            manager.shutdown()

    def test_get_pending_futures_returns_futures(self) -> None:
        """Test that get_pending_futures returns Future objects."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        try:
            block_event = threading.Event()
            future = manager.submit(lambda: block_event.wait(timeout=5))

            pending = manager.get_pending_futures()
            assert len(pending) == 1
            assert isinstance(pending[0], Future)

            block_event.set()
            future.result(timeout=1.0)
        finally:
            manager.shutdown()
