"""Tests for ExecutionManager TTL-based cleanup functionality.

These tests verify the memory safety improvements added in DS-467:
- Timestamp tracking for futures
- TTL-based cleanup of stale futures
- Maximum list size with overflow handling
- Monitoring/logging for long-running futures

DS-481 additions:
- Warning deduplication for long-running futures via last_warned_at tracking
- Configurable stale_removal_fraction for max_futures limit enforcement
"""

import threading
import time
from concurrent.futures import Future
from unittest.mock import patch

from sentinel.execution_manager import (
    DEFAULT_FUTURE_TTL_SECONDS,
    DEFAULT_MAX_FUTURES,
    DEFAULT_STALE_REMOVAL_FRACTION,
    LONG_RUNNING_WARNING_INTERVAL,
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

            # Make the future stale by backdating its created_at timestamp
            # instead of sleeping (deterministic approach - DS-933)
            with manager._futures_lock:
                manager._active_futures[0].created_at = time.monotonic() - 100

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
            futures = []
            for i in range(3):
                f = manager.submit(lambda: 42, description=f"quick-{i}")
                futures.append(f)

            # Wait for them to complete
            for f in futures:
                if f:
                    f.result(timeout=1.0)

            # Submit more futures - should trigger cleanup of completed futures
            # Submit slow futures using Event.wait for interruptible blocking (DS-943)
            slow_event = threading.Event()
            slow_futures = []
            for i in range(5):
                f = manager.submit(lambda: slow_event.wait(timeout=5), description=f"slow-{i}")
                slow_futures.append(f)

            # Make some futures stale by backdating their created_at timestamps
            # instead of sleeping (deterministic approach - DS-933)
            with manager._futures_lock:
                for tracked in manager._active_futures[:3]:
                    tracked.created_at = time.monotonic() - 100

            # Submit more futures to trigger overflow handling
            overflow_event = threading.Event()
            for i in range(5):
                manager.submit(lambda: overflow_event.wait(timeout=5), description=f"overflow-{i}")

            # Should have cleaned up some futures
            # Note: exact count depends on timing, but should be under control
            active = manager.get_active_futures()
            # The limit enforcement removes completed and stale futures
            assert len(active) <= 10  # Upper bound check

        finally:
            slow_event.set()
            overflow_event.set()
            manager.shutdown(cancel_futures=True)

    def test_max_futures_cleans_completed_first(self) -> None:
        """Test that max futures limit removes completed futures first."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
        )
        manager.start()

        # Use a shared event so slow futures block interruptibly (DS-943)
        block_event = threading.Event()
        try:
            # Submit a future that completes immediately
            fast_future = manager.submit(lambda: 42, description="fast")
            fast_future.result(timeout=1.0)  # Wait for completion

            # Submit more futures to reach the limit
            manager.submit(lambda: block_event.wait(timeout=10), description="slow-1")
            manager.submit(lambda: block_event.wait(timeout=10), description="slow-2")
            manager.submit(lambda: block_event.wait(timeout=10), description="slow-3")

            # At this point we're at the limit - next submit should clean up the completed future
            manager.submit(lambda: block_event.wait(timeout=10), description="slow-4")

            # Should have cleaned up the completed future
            tracked = manager.get_tracked_futures()
            # All tracked futures should be the slow ones (not done)
            pending_count = sum(1 for tf in tracked if not tf.future.done())
            assert pending_count >= 3  # At least the slow ones should be tracked

        finally:
            block_event.set()
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

            # Make the future stale by backdating its created_at timestamp
            # instead of sleeping (deterministic approach - DS-933)
            with manager._futures_lock:
                manager._active_futures[0].created_at = time.monotonic() - 100

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


class TestDS481WarningDeduplication:
    """Tests for DS-481: Long-running future warning deduplication."""

    def test_tracked_future_last_warned_at_defaults_to_none(self) -> None:
        """Test that last_warned_at defaults to None."""
        future: Future[int] = Future()
        tracked = TrackedFuture(future=future)

        assert tracked.last_warned_at is None

    def test_tracked_future_last_warned_at_can_be_set(self) -> None:
        """Test that last_warned_at can be provided."""
        future: Future[int] = Future()
        warned_time = time.monotonic()
        tracked = TrackedFuture(future=future, last_warned_at=warned_time)

        assert tracked.last_warned_at == warned_time

    def test_first_warning_logs_at_warning_level(self) -> None:
        """Test that first warning for a long-running future logs at WARNING level."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=300.0,
        )
        manager.start()

        try:
            block_event = threading.Event()
            manager.submit(
                lambda: block_event.wait(timeout=10),
                description="long task",
            )

            # Simulate a long-running future
            with manager._futures_lock:
                manager._active_futures[0].created_at = (
                    time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 10
                )
                # Ensure last_warned_at is None (first warning)
                assert manager._active_futures[0].last_warned_at is None

            with patch("sentinel.execution_manager.logger") as mock_logger:
                manager.collect_completed_results()

                # Verify WARNING was logged (not just DEBUG)
                warning_calls = [
                    call for call in mock_logger.warning.call_args_list
                    if "Long-running future detected" in str(call)
                ]
                assert len(warning_calls) >= 1

            # Verify last_warned_at was updated
            with manager._futures_lock:
                assert manager._active_futures[0].last_warned_at is not None

            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)

    def test_subsequent_warning_logs_at_debug_level(self) -> None:
        """Test that subsequent warnings within interval log at DEBUG level."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=300.0,
        )
        manager.start()

        try:
            block_event = threading.Event()
            manager.submit(
                lambda: block_event.wait(timeout=10),
                description="long task",
            )

            # Simulate a long-running future that was recently warned
            current_time = time.monotonic()
            with manager._futures_lock:
                manager._active_futures[0].created_at = (
                    current_time - LONG_RUNNING_WARNING_THRESHOLD - 10
                )
                # Set last_warned_at to recent time (within interval)
                manager._active_futures[0].last_warned_at = current_time - 10

            with patch("sentinel.execution_manager.logger") as mock_logger:
                manager.collect_completed_results()

                # Verify DEBUG was logged (not WARNING)
                debug_calls = [
                    call for call in mock_logger.debug.call_args_list
                    if "Long-running future detected" in str(call)
                ]
                # Should have at least one DEBUG call
                assert len(debug_calls) >= 1

                # Should NOT have a new WARNING call for long-running
                warning_calls = [
                    call for call in mock_logger.warning.call_args_list
                    if "Long-running future detected" in str(call)
                ]
                assert len(warning_calls) == 0

            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)

    def test_warning_after_interval_logs_at_warning_level(self) -> None:
        """Test that warning after interval has elapsed logs at WARNING level."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=300.0,
        )
        manager.start()

        try:
            block_event = threading.Event()
            manager.submit(
                lambda: block_event.wait(timeout=10),
                description="long task",
            )

            # Simulate a long-running future that was warned long ago
            current_time = time.monotonic()
            with manager._futures_lock:
                manager._active_futures[0].created_at = (
                    current_time - LONG_RUNNING_WARNING_THRESHOLD - 10
                )
                # Set last_warned_at to old time (beyond interval)
                manager._active_futures[0].last_warned_at = (
                    current_time - LONG_RUNNING_WARNING_INTERVAL - 10
                )

            with patch("sentinel.execution_manager.logger") as mock_logger:
                manager.collect_completed_results()

                # Verify WARNING was logged
                warning_calls = [
                    call for call in mock_logger.warning.call_args_list
                    if "Long-running future detected" in str(call)
                ]
                assert len(warning_calls) >= 1

            block_event.set()

        finally:
            manager.shutdown(cancel_futures=True)


class TestDS481StaleRemovalFraction:
    """Tests for DS-481: Configurable stale_removal_fraction."""

    def test_default_stale_removal_fraction_is_set(self) -> None:
        """Test that default stale_removal_fraction is set correctly."""
        manager = ExecutionManager(max_concurrent_executions=2)

        assert manager.stale_removal_fraction == DEFAULT_STALE_REMOVAL_FRACTION

    def test_custom_stale_removal_fraction_can_be_set(self) -> None:
        """Test that custom stale_removal_fraction can be provided."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            stale_removal_fraction=0.75,
        )

        assert manager.stale_removal_fraction == 0.75

    def test_stale_removal_fraction_clamped_to_valid_range(self) -> None:
        """Test that stale_removal_fraction is clamped to 0.0-1.0."""
        # Test value below 0
        manager_low = ExecutionManager(
            max_concurrent_executions=2,
            stale_removal_fraction=-0.5,
        )
        assert manager_low.stale_removal_fraction == 0.0

        # Test value above 1
        manager_high = ExecutionManager(
            max_concurrent_executions=2,
            stale_removal_fraction=1.5,
        )
        assert manager_high.stale_removal_fraction == 1.0

    def test_get_futures_stats_includes_stale_removal_fraction(self) -> None:
        """Test that get_futures_stats includes stale_removal_fraction."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            stale_removal_fraction=0.3,
        )
        manager.start()

        try:
            stats = manager.get_futures_stats()
            assert "stale_removal_fraction" in stats
            assert stats["stale_removal_fraction"] == 0.3
        finally:
            manager.shutdown()

    def test_stale_removal_uses_configured_fraction(self) -> None:
        """Test that stale removal uses the configured fraction."""
        # Use a high fraction to remove more stale futures
        manager = ExecutionManager(
            max_concurrent_executions=10,
            max_futures=5,
            future_ttl_seconds=0.01,  # Very short TTL
            stale_removal_fraction=0.8,  # Remove 80%
        )
        manager.start()

        try:
            # Create stale futures
            events = []
            for i in range(5):
                event = threading.Event()
                events.append(event)
                manager.submit(lambda e=event: e.wait(timeout=10), description=f"task-{i}")

            # Make them stale by backdating their created_at timestamps
            # instead of sleeping (deterministic approach - DS-933)
            with manager._futures_lock:
                for tracked in manager._active_futures:
                    tracked.created_at = time.monotonic() - 100

            # Submit one more to trigger max_futures enforcement
            with patch("sentinel.execution_manager.logger"):
                manager.submit(lambda: 42, description="trigger")

                # Should have removed some stale futures using the configured fraction
                assert manager.stale_futures_cleaned > 0

            # Clean up
            for event in events:
                event.set()

        finally:
            manager.shutdown(cancel_futures=True)
