"""Dedicated tests for ExecutionManager to increase coverage from 65.4% to >85%.

This file targets the uncovered code paths in execution_manager.py that are not
exercised by test_execution_manager_ttl.py. Focus areas include:
- Lifecycle methods (start/shutdown edge cases, is_running)
- Slot and active count calculations with mixed future states
- Submit when pool not running
- _enforce_max_futures_limit with stale removal path
- collect_completed_results error branches (OSError, RuntimeError, KeyError, ValueError, None)
- _log_long_running_futures warning deduplication paths
- _cleanup_stale_futures cancel-and-remove flow
- wait_for_completion / wait_for_all_completion
- execute_synchronously error branches
- cleanup_completed_futures callback error branches
- get_futures_stats comprehensive validation
"""

# NOTE: This test module intentionally accesses private attributes (e.g.,
# _active_futures, _futures_lock, _future_ttl_seconds) for white-box testing
# of internal ExecutionManager state. This is deliberate and provides coverage
# of internal error handling and state management paths that cannot be tested
# through the public API alone.

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from sentinel.execution_manager import (
    DEFAULT_FUTURE_TTL_SECONDS,
    DEFAULT_MAX_FUTURES,
    DEFAULT_STALE_REMOVAL_FRACTION,
    LONG_RUNNING_WARNING_INTERVAL,
    LONG_RUNNING_WARNING_THRESHOLD,
    ExecutionManager,
    TrackedFuture,
)


def _make_manager(
    max_concurrent_executions: int = 2,
    future_ttl_seconds: float = 9999,
    max_futures: int = 1000,
    stale_removal_fraction: float = 0.5,
) -> ExecutionManager:
    """Create an ExecutionManager with sensible test defaults.

    Centralizes manager creation to reduce boilerplate across test classes.
    Uses a high TTL by default to avoid stale-cleanup interference.

    Note: Callers needing TTL-sensitive behavior (e.g., stale-future cleanup
    or enforce-limit tests) should pass an explicit ``future_ttl_seconds``
    value rather than relying on the default, which is intentionally set
    high (9999 s) to prevent accidental stale-cleanup interference.
    """
    return ExecutionManager(
        max_concurrent_executions=max_concurrent_executions,
        future_ttl_seconds=future_ttl_seconds,
        max_futures=max_futures,
        stale_removal_fraction=stale_removal_fraction,
    )

# ---------------------------------------------------------------------------
# 1. Lifecycle: start(), shutdown(), is_running()
# ---------------------------------------------------------------------------


class TestExecutionManagerLifecycle:
    """Tests for ExecutionManager start/shutdown/is_running lifecycle."""

    def test_start_creates_thread_pool(self) -> None:
        """Calling start() should create the internal thread pool."""
        manager = ExecutionManager(max_concurrent_executions=2)
        assert not manager.is_running()

        manager.start()
        try:
            assert manager.is_running()
        finally:
            manager.shutdown()

    def test_start_when_already_started_logs_warning(self) -> None:
        """Calling start() when the pool is already running logs a warning and returns."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()
        try:
            with patch("sentinel.execution_manager.logger") as mock_logger:
                manager.start()
                mock_logger.warning.assert_called_once_with("Thread pool already started")
        finally:
            manager.shutdown()

    def test_shutdown_when_not_started_is_noop(self) -> None:
        """Calling shutdown() when pool is None should be a no-op (no error)."""
        manager = ExecutionManager(max_concurrent_executions=2)
        # Should not raise
        manager.shutdown()
        assert not manager.is_running()

    def test_shutdown_sets_pool_to_none(self) -> None:
        """After shutdown the pool should be None and is_running returns False."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()
        assert manager.is_running()

        manager.shutdown()
        assert not manager.is_running()

    def test_shutdown_with_block_false(self) -> None:
        """Shutdown with block=False should return immediately without waiting."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        event = threading.Event()
        manager.submit(lambda: event.wait(timeout=10), description="blocking task")

        try:
            manager.shutdown(block=False)
            assert not manager.is_running()
        finally:
            event.set()

    def test_shutdown_with_cancel_futures(self) -> None:
        """Shutdown with cancel_futures=True should cancel pending futures."""
        manager = ExecutionManager(max_concurrent_executions=1)
        manager.start()

        event = threading.Event()
        # Submit a blocking task to fill the single worker
        manager.submit(lambda: event.wait(timeout=10), description="blocker")
        # Submit another that will be pending (queued, not running)
        manager.submit(lambda: 99, description="pending")

        try:
            manager.shutdown(block=False, cancel_futures=True)
            assert not manager.is_running()
        finally:
            event.set()

    def test_is_running_returns_false_initially(self) -> None:
        """is_running should return False before start() is called."""
        manager = ExecutionManager(max_concurrent_executions=2)
        assert manager.is_running() is False

    def test_is_running_returns_true_after_start(self) -> None:
        """is_running should return True after start() is called."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()
        try:
            assert manager.is_running() is True
        finally:
            manager.shutdown()


# ---------------------------------------------------------------------------
# 2. Slots and active count
# ---------------------------------------------------------------------------


class TestExecutionManagerSlots:
    """Tests for get_available_slots and get_active_count with mixed states."""

    def test_get_available_slots_all_idle(self) -> None:
        """With no futures submitted, all slots should be available."""
        manager = ExecutionManager(max_concurrent_executions=4)
        assert manager.get_available_slots() == 4

    def test_get_available_slots_with_done_and_pending(self) -> None:
        """Done futures should not count against available slots."""
        manager = ExecutionManager(max_concurrent_executions=4)
        manager.start()

        event = threading.Event()
        try:
            # Submit a future that completes immediately
            done_future = manager.submit(lambda: 42, description="done")
            done_future.result(timeout=2.0)

            # Submit a future that blocks
            manager.submit(lambda: event.wait(timeout=10), description="pending")

            # 1 pending (the blocking one) -> 3 available
            slots = manager.get_available_slots()
            assert slots == 3
        finally:
            event.set()
            manager.shutdown(cancel_futures=True)

    def test_get_active_count_mixed(self) -> None:
        """get_active_count should only count non-done futures."""
        manager = ExecutionManager(max_concurrent_executions=4)
        manager.start()

        event = threading.Event()
        try:
            # Submit a future that completes immediately
            done_future = manager.submit(lambda: 42, description="done")
            done_future.result(timeout=2.0)

            # Submit a future that blocks
            manager.submit(lambda: event.wait(timeout=10), description="pending")

            active = manager.get_active_count()
            assert active == 1
        finally:
            event.set()
            manager.shutdown(cancel_futures=True)

    def test_get_active_count_no_futures(self) -> None:
        """get_active_count should return 0 when there are no futures."""
        manager = ExecutionManager(max_concurrent_executions=2)
        assert manager.get_active_count() == 0


# ---------------------------------------------------------------------------
# 3. Submit edge cases
# ---------------------------------------------------------------------------


class TestExecutionManagerSubmit:
    """Tests for submit() edge cases."""

    def test_submit_when_pool_not_running_returns_none(self) -> None:
        """submit() should return None and log a warning when pool is not running."""
        manager = ExecutionManager(max_concurrent_executions=2)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.submit(lambda: 42, description="no pool")
            assert result is None
            mock_logger.warning.assert_called_once_with(
                "Cannot submit task: thread pool not running"
            )

    def test_submit_at_max_futures_triggers_enforce_limit(self) -> None:
        """When at max_futures, submit should trigger _enforce_max_futures_limit."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
            future_ttl_seconds=0.01,
        )
        manager.start()

        events = []
        try:
            # Fill to max_futures with blocking tasks
            for i in range(3):
                ev = threading.Event()
                events.append(ev)
                manager.submit(lambda e=ev: e.wait(timeout=10), description=f"task-{i}")

            # Make them stale by backdating their created_at timestamps
            # instead of sleeping (deterministic approach - DS-933)
            with manager._futures_lock:
                for tracked in manager._active_futures:
                    tracked.created_at = time.monotonic() - 100

            with patch("sentinel.execution_manager.logger"):
                # This submit should trigger _enforce_max_futures_limit
                future = manager.submit(lambda: "new", description="trigger")
                assert future is not None

            # Stale futures should have been cleaned
            assert manager.stale_futures_cleaned > 0
        finally:
            for ev in events:
                ev.set()
            manager.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------
# 4. collect_completed_results error branches
# ---------------------------------------------------------------------------


class TestCollectCompletedResultsErrors:
    """Tests for error handling branches in collect_completed_results."""

    def test_future_raises_os_error(self) -> None:
        """An OSError from future.result() should be logged, not propagated."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_exception(OSError("disk full"))
        tracked = TrackedFuture(future=future, description="os-error task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            results = manager.collect_completed_results()
            assert results == []
            io_calls = [
                c for c in mock_logger.error.call_args_list
                if "I/O or timeout" in c.args[0]
            ]
            assert len(io_calls) == 1
            assert "disk full" in str(io_calls[0].args[1])

    def test_future_raises_timeout_error(self) -> None:
        """A TimeoutError from future.result() should be caught by the OSError/TimeoutError branch."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_exception(TimeoutError("timed out"))
        tracked = TrackedFuture(future=future, description="timeout task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            results = manager.collect_completed_results()
            assert results == []
            io_calls = [
                c for c in mock_logger.error.call_args_list
                if "I/O or timeout" in c.args[0]
            ]
            assert len(io_calls) == 1
            assert "timed out" in str(io_calls[0].args[1])

    def test_future_raises_runtime_error(self) -> None:
        """A RuntimeError from future.result() should be logged."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_exception(RuntimeError("bad state"))
        tracked = TrackedFuture(future=future, description="runtime-error task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            results = manager.collect_completed_results()
            assert results == []
            rt_calls = [
                c for c in mock_logger.error.call_args_list
                if "runtime error" in c.args[0]
            ]
            assert len(rt_calls) == 1
            assert "bad state" in str(rt_calls[0].args[1])

    def test_future_raises_key_error(self) -> None:
        """A KeyError from future.result() should be logged as data error."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_exception(KeyError("missing_key"))
        tracked = TrackedFuture(future=future, description="key-error task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            results = manager.collect_completed_results()
            assert results == []
            # Verify any error call mentioning data error was made
            data_error_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_error_calls) >= 1

    def test_future_raises_value_error(self) -> None:
        """A ValueError from future.result() should be logged as data error."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_exception(ValueError("bad value"))
        tracked = TrackedFuture(future=future, description="value-error task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            results = manager.collect_completed_results()
            assert results == []
            data_error_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_error_calls) >= 1

    def test_future_returns_none_result(self) -> None:
        """A future that returns None should not add anything to results."""
        manager = _make_manager()

        future: Future[None] = Future()
        future.set_result(None)
        tracked = TrackedFuture(future=future, description="none-result task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        results = manager.collect_completed_results()
        assert results == []

    def test_future_returns_valid_result(self) -> None:
        """A future that returns a non-None result should be included in results."""
        manager = _make_manager()

        sentinel_result = MagicMock()
        future: Future[MagicMock] = Future()
        future.set_result(sentinel_result)
        tracked = TrackedFuture(future=future, description="good task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        results = manager.collect_completed_results()
        assert results == [sentinel_result]


# ---------------------------------------------------------------------------
# 5. execute_synchronously error branches
# ---------------------------------------------------------------------------


class TestExecuteSynchronously:
    """Tests for execute_synchronously success and error branches."""

    def test_success(self) -> None:
        """Successful synchronous execution should return the result."""
        manager = ExecutionManager(max_concurrent_executions=2)
        result = manager.execute_synchronously(lambda: 42)
        assert result == 42

    def test_os_error_returns_none(self) -> None:
        """OSError during synchronous execution should return None and log."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def raise_os_error() -> None:
            raise OSError("disk failure")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.execute_synchronously(raise_os_error)
            assert result is None
            io_calls = [
                c for c in mock_logger.error.call_args_list
                if "I/O or timeout" in c.args[0]
            ]
            assert len(io_calls) == 1
            assert "disk failure" in str(io_calls[0].args[1])

    def test_timeout_error_returns_none(self) -> None:
        """TimeoutError during synchronous execution should return None and log."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def raise_timeout() -> None:
            raise TimeoutError("timed out")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.execute_synchronously(raise_timeout)
            assert result is None
            io_calls = [
                c for c in mock_logger.error.call_args_list
                if "I/O or timeout" in c.args[0]
            ]
            assert len(io_calls) == 1
            assert "timed out" in str(io_calls[0].args[1])

    def test_runtime_error_returns_none(self) -> None:
        """RuntimeError during synchronous execution should return None and log."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def raise_runtime() -> None:
            raise RuntimeError("runtime issue")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.execute_synchronously(raise_runtime)
            assert result is None
            rt_calls = [
                c for c in mock_logger.error.call_args_list
                if "runtime error" in c.args[0]
            ]
            assert len(rt_calls) == 1
            assert "runtime issue" in str(rt_calls[0].args[1])

    def test_key_error_returns_none(self) -> None:
        """KeyError during synchronous execution should return None and log."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def raise_key_error() -> None:
            raise KeyError("missing")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.execute_synchronously(raise_key_error)
            assert result is None
            data_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_calls) == 1

    def test_value_error_returns_none(self) -> None:
        """ValueError during synchronous execution should return None and log."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def raise_value_error() -> None:
            raise ValueError("bad input")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            result = manager.execute_synchronously(raise_value_error)
            assert result is None
            data_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_calls) == 1

    def test_passes_args_and_kwargs(self) -> None:
        """execute_synchronously should forward positional and keyword args."""
        manager = ExecutionManager(max_concurrent_executions=2)

        def add(a: int, b: int, offset: int = 0) -> int:
            return a + b + offset

        result = manager.execute_synchronously(add, 3, 4, offset=10)
        assert result == 17


# ---------------------------------------------------------------------------
# 6. cleanup_completed_futures callback error branches
# ---------------------------------------------------------------------------


class TestCleanupCompletedFutures:
    """Tests for cleanup_completed_futures with and without callbacks."""

    def test_no_callback(self) -> None:
        """Without a callback, completed futures should still be counted."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="done task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        cleaned = manager.cleanup_completed_futures()
        assert cleaned == 1
        assert len(manager.get_active_futures()) == 0

    def test_callback_success(self) -> None:
        """A successful callback should not affect the cleaned count."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="done task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        callback = MagicMock()
        cleaned = manager.cleanup_completed_futures(on_future_done=callback)
        assert cleaned == 1
        callback.assert_called_once_with(future)

    def test_callback_raises_os_error(self) -> None:
        """OSError from callback should be logged and not prevent cleanup."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="os-cb task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        def bad_callback(f: Future[int]) -> None:
            raise OSError("callback IO error")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            cleaned = manager.cleanup_completed_futures(on_future_done=bad_callback)
            assert cleaned == 1
            runtime_io_calls = [
                c for c in mock_logger.error.call_args_list
                if "runtime/I/O error" in c.args[0]
            ]
            assert len(runtime_io_calls) >= 1

    def test_callback_raises_runtime_error(self) -> None:
        """RuntimeError from callback should be logged and not prevent cleanup."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="rt-cb task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        def bad_callback(f: Future[int]) -> None:
            raise RuntimeError("callback runtime error")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            cleaned = manager.cleanup_completed_futures(on_future_done=bad_callback)
            assert cleaned == 1
            runtime_io_calls = [
                c for c in mock_logger.error.call_args_list
                if "runtime/I/O error" in c.args[0]
            ]
            assert len(runtime_io_calls) >= 1

    def test_callback_raises_key_error(self) -> None:
        """KeyError from callback should be logged as data error."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="ke-cb task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        def bad_callback(f: Future[int]) -> None:
            raise KeyError("callback key error")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            cleaned = manager.cleanup_completed_futures(on_future_done=bad_callback)
            assert cleaned == 1
            data_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_calls) >= 1

    def test_callback_raises_value_error(self) -> None:
        """ValueError from callback should be logged as data error."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="ve-cb task")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        def bad_callback(f: Future[int]) -> None:
            raise ValueError("callback value error")

        with patch("sentinel.execution_manager.logger") as mock_logger:
            cleaned = manager.cleanup_completed_futures(on_future_done=bad_callback)
            assert cleaned == 1
            data_calls = [
                c for c in mock_logger.error.call_args_list
                if "data error" in c.args[0]
            ]
            assert len(data_calls) >= 1

    def test_includes_stale_futures_in_count(self) -> None:
        """cleanup_completed_futures should also clean stale futures."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.01,
        )

        # Create a stale future (not done, old creation time)
        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="stale task",
            created_at=time.monotonic() - 100,
        )
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger"):
            cleaned = manager.cleanup_completed_futures()
            # The stale future should have been cleaned
            assert cleaned >= 1
            assert manager.stale_futures_cleaned >= 1


# ---------------------------------------------------------------------------
# 7. wait_for_completion
# ---------------------------------------------------------------------------


class TestWaitForCompletion:
    """Tests for wait_for_completion with various arguments."""

    def test_no_pending_returns_empty_sets(self) -> None:
        """When there are no pending futures, return two empty sets."""
        manager = ExecutionManager(max_concurrent_executions=2)
        done, not_done = manager.wait_for_completion()
        assert done == set()
        assert not_done == set()

    def test_with_timeout(self) -> None:
        """wait_for_completion should respect the timeout parameter."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        event = threading.Event()
        try:
            manager.submit(lambda: event.wait(timeout=10), description="blocking")

            start_time = time.monotonic()
            done, not_done = manager.wait_for_completion(timeout=0.1)
            elapsed = time.monotonic() - start_time

            # Should have returned quickly (around 0.1s, not 10s)
            assert elapsed < 1.0
            # The blocking future should be in not_done
            assert len(not_done) == 1
        finally:
            event.set()
            manager.shutdown(cancel_futures=True)

    def test_with_all_completed(self) -> None:
        """wait_for_completion with ALL_COMPLETED should wait for all futures."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        event1 = threading.Event()
        event2 = threading.Event()
        try:
            manager.submit(lambda: event1.wait(timeout=10), description="task-1")
            manager.submit(lambda: event2.wait(timeout=10), description="task-2")

            # Release both so they complete
            event1.set()
            event2.set()

            done, not_done = manager.wait_for_completion(
                timeout=5.0, return_when="ALL_COMPLETED"
            )

            assert len(done) == 2
            assert len(not_done) == 0
        finally:
            event1.set()
            event2.set()
            manager.shutdown(cancel_futures=True)

    def test_with_first_exception(self) -> None:
        """wait_for_completion with FIRST_EXCEPTION should return on first error."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        gate_event = threading.Event()
        block_event = threading.Event()
        try:
            def raise_after_gate() -> None:
                # Wait briefly then raise - the gate ensures this future is still
                # "pending" when get_pending_futures is called, but the error fires
                # quickly enough to be caught by wait().
                gate_event.wait(timeout=10)
                raise RuntimeError("boom")

            manager.submit(raise_after_gate, description="error-task")
            manager.submit(lambda: block_event.wait(timeout=10), description="blocking")

            # Schedule the gate to open shortly, so the error occurs during wait()
            # Use a shorter delay to avoid unnecessary waiting (deterministic approach - DS-933)
            def release_gate() -> None:
                # Brief real-time delay is intentional here: gate_event must be
                # set *after* wait_for_completion enters its wait() call so the
                # FIRST_EXCEPTION path is exercised.  An Event-based approach
                # is impractical because there is no hook to detect that the
                # caller has entered wait().  The 10 ms sleep is short enough to
                # keep tests fast while providing the necessary ordering (DS-943).
                time.sleep(0.01)
                gate_event.set()

            t = threading.Thread(target=release_gate)
            t.start()

            done, not_done = manager.wait_for_completion(
                timeout=5.0, return_when="FIRST_EXCEPTION"
            )

            # The errored future should be in done
            assert len(done) >= 1
            t.join(timeout=2)
        finally:
            gate_event.set()
            block_event.set()
            manager.shutdown(cancel_futures=True)

    def test_with_first_completed_default(self) -> None:
        """wait_for_completion with default FIRST_COMPLETED returns when any completes."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        release_event = threading.Event()
        block_event = threading.Event()
        try:
            manager.submit(lambda: release_event.wait(timeout=10), description="fast")
            manager.submit(lambda: block_event.wait(timeout=10), description="slow")

            # Release the fast task
            release_event.set()

            done, not_done = manager.wait_for_completion(timeout=5.0)

            # At least the fast one should be done
            assert len(done) >= 1
        finally:
            release_event.set()
            block_event.set()
            manager.shutdown(cancel_futures=True)

    def test_unknown_return_when_falls_back_to_first_completed(self) -> None:
        """An unrecognized return_when string should fall back to FIRST_COMPLETED."""
        manager = ExecutionManager(max_concurrent_executions=2)
        manager.start()

        release_event = threading.Event()
        try:
            manager.submit(lambda: release_event.wait(timeout=10), description="task")

            # Release the task
            release_event.set()

            done, not_done = manager.wait_for_completion(
                timeout=5.0, return_when="BOGUS_VALUE"
            )

            # Should still work (falls back to FC)
            assert len(done) >= 1
        finally:
            release_event.set()
            manager.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------
# 8. wait_for_all_completion
# ---------------------------------------------------------------------------


class TestWaitForAllCompletion:
    """Tests for wait_for_all_completion."""

    def test_waits_until_all_done_and_returns_results(self) -> None:
        """Should block until all futures complete, then collect results."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=9999,
        )
        manager.start()

        try:
            sentinel_result = MagicMock()
            manager.submit(lambda: sentinel_result, description="task-1")
            manager.submit(lambda: sentinel_result, description="task-2")

            results = manager.wait_for_all_completion(poll_interval=0.01)
            assert len(results) == 2
        finally:
            manager.shutdown(cancel_futures=True)

    def test_returns_empty_when_no_futures(self) -> None:
        """With no pending futures, should return quickly with empty list."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        results = manager.wait_for_all_completion(poll_interval=0.01)
        assert results == []

    def test_waits_for_slow_futures(self) -> None:
        """Should wait for futures that take a bit of time to finish."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=9999,
        )
        manager.start()

        result_value = MagicMock()

        try:
            # Use an event for synchronization instead of sleep (deterministic approach - DS-933)
            task_event = threading.Event()

            def slow_task() -> MagicMock:
                task_event.wait(timeout=10)
                return result_value

            manager.submit(slow_task, description="slow-1")

            # Release the task to complete
            task_event.set()

            results = manager.wait_for_all_completion(poll_interval=0.01)
            assert len(results) == 1
            assert results[0] is result_value
        finally:
            manager.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------
# 9. _enforce_max_futures_limit
# ---------------------------------------------------------------------------


class TestEnforceMaxFuturesLimit:
    """Tests for _enforce_max_futures_limit internal method."""

    def test_removes_completed_futures_first(self) -> None:
        """Completed futures should be removed before considering stale ones."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
            future_ttl_seconds=9999,
        )

        # Create 3 completed futures to fill to the limit
        for i in range(3):
            f: Future[int] = Future()
            f.set_result(i)
            tracked = TrackedFuture(future=f, description=f"done-{i}")
            manager._active_futures.append(tracked)

        assert len(manager._active_futures) == 3

        # Enforce limit - should remove all completed futures
        with manager._futures_lock:
            manager._enforce_max_futures_limit()

        assert len(manager._active_futures) == 0

    def test_removes_stale_when_still_over_limit_after_completed_removal(self) -> None:
        """If still over limit after removing completed, stale futures are removed."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
            future_ttl_seconds=0.01,
            stale_removal_fraction=0.5,
        )

        # Create 3 stale (not done, old) futures
        for i in range(3):
            f: Future[int] = Future()
            tracked = TrackedFuture(
                future=f,
                description=f"stale-{i}",
                created_at=time.monotonic() - 100 + i,  # varied ages, oldest first
            )
            manager._active_futures.append(tracked)

        assert len(manager._active_futures) == 3

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._enforce_max_futures_limit()

            # Should have removed some stale futures
            assert manager.stale_futures_cleaned > 0
            # Warning about max futures limit should have been logged
            stale_warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "Max futures limit" in c.args[0]
            ]
            assert len(stale_warning_calls) >= 1

    def test_stale_removal_fraction_applied(self) -> None:
        """The configured stale_removal_fraction controls how many stale futures are removed."""
        manager = ExecutionManager(
            max_concurrent_executions=10,
            max_futures=5,
            future_ttl_seconds=0.01,
            stale_removal_fraction=0.5,
        )

        # Create 5 stale futures (all not done, all old)
        for i in range(5):
            f: Future[int] = Future()
            tracked = TrackedFuture(
                future=f,
                description=f"stale-{i}",
                created_at=time.monotonic() - 200 + i,
            )
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger"):
            with manager._futures_lock:
                manager._enforce_max_futures_limit()

        # With fraction=0.5 and 5 stale: int(5 * 0.5) + 1 = 3 removed
        assert manager.stale_futures_cleaned == 3
        assert len(manager._active_futures) == 2

    def test_stale_removal_oldest_first(self) -> None:
        """Stale futures should be removed oldest first."""
        manager = ExecutionManager(
            max_concurrent_executions=10,
            max_futures=4,
            future_ttl_seconds=0.01,
            stale_removal_fraction=0.5,
        )

        now = time.monotonic()
        descriptions_by_age = []
        for i in range(4):
            f: Future[int] = Future()
            desc = f"stale-age-{i}"
            descriptions_by_age.append(desc)
            tracked = TrackedFuture(
                future=f,
                description=desc,
                created_at=now - 400 + i * 100,  # stale-age-0 is oldest
            )
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger"):
            with manager._futures_lock:
                manager._enforce_max_futures_limit()

        # int(4 * 0.5) + 1 = 3 removed (the 3 oldest)
        remaining = [tf.description for tf in manager._active_futures]
        # Only the newest should remain
        assert "stale-age-3" in remaining
        assert "stale-age-0" not in remaining

    def test_no_stale_futures_leaves_list_unchanged(self) -> None:
        """If no stale futures exist and all are pending, list stays unchanged."""
        manager = ExecutionManager(
            max_concurrent_executions=5,
            max_futures=3,
            future_ttl_seconds=9999,
        )

        # Create 3 non-done, non-stale futures (recent creation)
        for i in range(3):
            f: Future[int] = Future()
            tracked = TrackedFuture(future=f, description=f"fresh-{i}")
            manager._active_futures.append(tracked)

        with manager._futures_lock:
            manager._enforce_max_futures_limit()

        # Nothing was stale, nothing completed, so nothing removed (except completed filter)
        # All are still pending and fresh -> still 3
        assert len(manager._active_futures) == 3
        assert manager.stale_futures_cleaned == 0


# ---------------------------------------------------------------------------
# 10. _log_long_running_futures
# ---------------------------------------------------------------------------


class TestLogLongRunningFutures:
    """Tests for _log_long_running_futures warning deduplication."""

    def test_first_warning_logs_at_warning_level(self) -> None:
        """First time a long-running future is detected, log at WARNING and set last_warned_at."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="long task",
            created_at=time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 10,
        )
        assert tracked.last_warned_at is None

        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._log_long_running_futures()

            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "Long-running future detected" in c.args[0]
            ]
            assert len(warning_calls) == 1

        # last_warned_at should now be set
        assert tracked.last_warned_at is not None

    def test_subsequent_within_interval_logs_debug(self) -> None:
        """Within the warning interval, subsequent checks should log at DEBUG."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        current_time = time.monotonic()
        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="long task",
            created_at=current_time - LONG_RUNNING_WARNING_THRESHOLD - 10,
            last_warned_at=current_time - 5,  # warned 5 seconds ago (within interval)
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._log_long_running_futures()

            # Should have DEBUG, not WARNING
            debug_calls = [
                c for c in mock_logger.debug.call_args_list
                if "Long-running future detected" in c.args[0]
            ]
            assert len(debug_calls) == 1

            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "Long-running future detected" in c.args[0]
            ]
            assert len(warning_calls) == 0

    def test_after_interval_logs_warning_again(self) -> None:
        """After the warning interval elapses, WARNING level should be used again."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        current_time = time.monotonic()
        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="long task",
            created_at=current_time - LONG_RUNNING_WARNING_THRESHOLD - 500,
            last_warned_at=current_time - LONG_RUNNING_WARNING_INTERVAL - 10,
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._log_long_running_futures()

            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "Long-running future detected" in c.args[0]
            ]
            assert len(warning_calls) == 1

        # last_warned_at should have been updated
        assert tracked.last_warned_at > current_time - LONG_RUNNING_WARNING_INTERVAL

    def test_done_futures_are_skipped(self) -> None:
        """Completed futures should not trigger long-running warnings."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(
            future=future,
            description="done task",
            created_at=time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 100,
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._log_long_running_futures()

            # No long-running warnings should be logged for done futures
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "Long-running future detected" in c.args[0]
            ]
            assert len(warning_calls) == 0

    def test_unnamed_task_uses_default_description(self) -> None:
        """Future with empty description should use 'unnamed task' in log message."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="",
            created_at=time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 10,
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._log_long_running_futures()

            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "unnamed task" in c.args[0]
            ]
            assert len(warning_calls) == 1


# ---------------------------------------------------------------------------
# 11. _cleanup_stale_futures
# ---------------------------------------------------------------------------


class TestCleanupStaleFutures:
    """Tests for _cleanup_stale_futures internal method."""

    def test_cancels_and_removes_stale_futures(self) -> None:
        """Stale futures should be cancelled and removed from the active list."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.01,
        )

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="stale task",
            created_at=time.monotonic() - 100,
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger"):
            with manager._futures_lock:
                count = manager._cleanup_stale_futures()

        assert count == 1
        assert manager.stale_futures_cleaned == 1
        assert len(manager._active_futures) == 0

    def test_non_stale_futures_are_kept(self) -> None:
        """Non-stale futures should remain in the active list."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=9999,
        )

        future: Future[int] = Future()
        tracked = TrackedFuture(future=future, description="fresh task")
        manager._active_futures.append(tracked)

        with manager._futures_lock:
            count = manager._cleanup_stale_futures()

        assert count == 0
        assert len(manager._active_futures) == 1

    def test_stale_counter_increments(self) -> None:
        """The stale_futures_cleaned counter should increment for each cleanup."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.01,
        )

        for i in range(3):
            f: Future[int] = Future()
            tracked = TrackedFuture(
                future=f,
                description=f"stale-{i}",
                created_at=time.monotonic() - 100,
            )
            manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger"):
            with manager._futures_lock:
                count = manager._cleanup_stale_futures()

        assert count == 3
        assert manager.stale_futures_cleaned == 3

    def test_logs_error_for_each_stale_future(self) -> None:
        """Each stale future should have an error logged with its description and age."""
        manager = ExecutionManager(
            max_concurrent_executions=2,
            future_ttl_seconds=0.01,
        )

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="hung task",
            created_at=time.monotonic() - 100,
        )
        manager._active_futures.append(tracked)

        with patch("sentinel.execution_manager.logger") as mock_logger:
            with manager._futures_lock:
                manager._cleanup_stale_futures()

            error_calls = [
                c for c in mock_logger.error.call_args_list
                if "Future exceeded TTL" in c.args[0] and "hung task" in c.args[0]
            ]
            assert len(error_calls) == 1


# ---------------------------------------------------------------------------
# 12. get_futures_stats comprehensive
# ---------------------------------------------------------------------------


class TestGetFuturesStatsComprehensive:
    """Tests for get_futures_stats with various future states."""

    def test_stats_with_no_futures(self) -> None:
        """Stats with no futures should show zeros."""
        manager = ExecutionManager(max_concurrent_executions=2)
        stats = manager.get_futures_stats()

        assert stats["total_tracked"] == 0
        assert stats["pending"] == 0
        assert stats["completed"] == 0
        assert stats["stale"] == 0
        assert stats["long_running"] == 0
        assert stats["max_age_seconds"] == 0.0
        assert stats["avg_age_seconds"] == 0.0
        assert stats["total_stale_cleaned"] == 0
        assert stats["ttl_seconds"] == DEFAULT_FUTURE_TTL_SECONDS
        assert stats["max_futures"] == DEFAULT_MAX_FUTURES
        assert stats["stale_removal_fraction"] == DEFAULT_STALE_REMOVAL_FRACTION

    def test_stats_with_completed_future(self) -> None:
        """Stats should correctly count a completed future."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        future.set_result(42)
        tracked = TrackedFuture(future=future, description="done")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        stats = manager.get_futures_stats()
        assert stats["total_tracked"] == 1
        assert stats["completed"] == 1
        assert stats["pending"] == 0

    def test_stats_with_pending_future(self) -> None:
        """Stats should correctly count a pending future."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        tracked = TrackedFuture(future=future, description="pending")
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        stats = manager.get_futures_stats()
        assert stats["total_tracked"] == 1
        assert stats["pending"] == 1
        assert stats["completed"] == 0

    def test_stats_with_stale_future(self) -> None:
        """Stats should correctly count a stale future."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=0.01)

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="stale",
            created_at=time.monotonic() - 100,
        )
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        stats = manager.get_futures_stats()
        assert stats["stale"] == 1
        assert stats["pending"] == 1

    def test_stats_with_long_running_future(self) -> None:
        """Stats should correctly count a long-running (but not stale) future."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=9999)

        future: Future[int] = Future()
        tracked = TrackedFuture(
            future=future,
            description="long-running",
            created_at=time.monotonic() - LONG_RUNNING_WARNING_THRESHOLD - 10,
        )
        with manager._futures_lock:
            manager._active_futures.append(tracked)

        stats = manager.get_futures_stats()
        assert stats["long_running"] == 1
        assert stats["max_age_seconds"] > LONG_RUNNING_WARNING_THRESHOLD

    def test_stats_with_mixed_states(self) -> None:
        """Stats should correctly handle a mix of completed, pending, stale, long-running."""
        manager = ExecutionManager(
            max_concurrent_executions=10,
            future_ttl_seconds=500,
        )

        now = time.monotonic()

        # 1 completed future
        f_done: Future[int] = Future()
        f_done.set_result(1)
        manager._active_futures.append(
            TrackedFuture(future=f_done, description="done")
        )

        # 1 fresh pending future
        f_fresh: Future[int] = Future()
        manager._active_futures.append(
            TrackedFuture(future=f_fresh, description="fresh", created_at=now - 1)
        )

        # 1 long-running pending future (not stale because TTL=500s, but > 60s threshold)
        f_long: Future[int] = Future()
        manager._active_futures.append(
            TrackedFuture(
                future=f_long,
                description="long-running",
                created_at=now - LONG_RUNNING_WARNING_THRESHOLD - 10,
            )
        )

        # 1 stale pending future (age > TTL of 500s)
        f_stale: Future[int] = Future()
        manager._active_futures.append(
            TrackedFuture(
                future=f_stale,
                description="stale",
                created_at=now - 600,  # 600s > 500s TTL
            )
        )

        stats = manager.get_futures_stats()
        assert stats["total_tracked"] == 4
        assert stats["completed"] == 1
        assert stats["pending"] == 3  # fresh + long-running + stale
        assert stats["stale"] == 1  # Only the 600s old one
        assert stats["long_running"] == 2  # 70s + 600s both > 60s threshold
        assert stats["max_age_seconds"] > 500
        assert stats["avg_age_seconds"] > 0
        assert stats["total_stale_cleaned"] == 0

    def test_stats_reflects_stale_cleaned_counter(self) -> None:
        """total_stale_cleaned should reflect the cumulative counter."""
        manager = ExecutionManager(max_concurrent_executions=2, future_ttl_seconds=0.01)

        # Simulate some stale cleanup having happened
        manager._stale_futures_cleaned = 42

        stats = manager.get_futures_stats()
        assert stats["total_stale_cleaned"] == 42

    def test_stats_avg_and_max_age_with_multiple_pending(self) -> None:
        """avg_age and max_age should be computed only from pending (not done) futures."""
        manager = ExecutionManager(max_concurrent_executions=5, future_ttl_seconds=9999)

        now = time.monotonic()

        # Pending future aged 10s
        f1: Future[int] = Future()
        manager._active_futures.append(
            TrackedFuture(future=f1, description="p1", created_at=now - 10)
        )

        # Pending future aged 20s
        f2: Future[int] = Future()
        manager._active_futures.append(
            TrackedFuture(future=f2, description="p2", created_at=now - 20)
        )

        # Completed future aged 100s (should NOT affect age stats)
        f3: Future[int] = Future()
        f3.set_result(99)
        manager._active_futures.append(
            TrackedFuture(future=f3, description="done", created_at=now - 100)
        )

        stats = manager.get_futures_stats()
        # max_age should be ~20s (not 100s from the done future)
        assert stats["max_age_seconds"] >= 19.0
        assert stats["max_age_seconds"] < 30.0
        # avg_age should be ~15s
        assert stats["avg_age_seconds"] >= 14.0
        assert stats["avg_age_seconds"] < 20.0
