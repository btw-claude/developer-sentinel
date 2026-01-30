"""Execution manager for the Sentinel orchestrator.

This module provides the ExecutionManager class which manages:
- Thread pool lifecycle
- Future tracking and collection
- Execution submission and result collection
- Graceful shutdown handling

This is part of the Sentinel refactoring to split the God Object into focused,
composable components (DS-384).

Component Boundaries
--------------------
ExecutionManager is responsible for all thread pool and concurrent execution
concerns. It is the only component that directly interacts with
``concurrent.futures.ThreadPoolExecutor``.

The Sentinel class delegates to ExecutionManager for:
- ``_execution_manager.start()`` / ``shutdown()`` : Thread pool lifecycle
- ``_execution_manager.submit()`` : Submitting tasks for async execution
- ``_execution_manager.get_available_slots()`` : Checking execution capacity
- ``_execution_manager.get_active_count()`` : Getting number of running tasks
- ``_execution_manager.collect_completed_results()`` : Harvesting results

See Also
--------
- docs/architecture.md : Full architecture documentation
- sentinel.state_tracker : State and metrics management
- sentinel.orchestration_registry : Orchestration loading and hot-reload
- sentinel.poll_coordinator : Polling coordination
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.executor import ExecutionResult

logger = get_logger(__name__)

T = TypeVar("T")

# Default TTL for futures in seconds (5 minutes)
DEFAULT_FUTURE_TTL_SECONDS = 300

# Default maximum number of futures to track
DEFAULT_MAX_FUTURES = 1000

# Warning threshold for long-running futures (seconds)
LONG_RUNNING_WARNING_THRESHOLD = 60

# Minimum interval between repeated warnings for the same long-running future (seconds)
# After the first warning, subsequent warnings are logged at DEBUG level until
# this interval has elapsed, then a new WARNING is logged. This prevents log
# flooding while still providing visibility into long-running tasks (DS-481).
LONG_RUNNING_WARNING_INTERVAL = 300

# Default fraction of stale futures to remove when max_futures limit is reached.
# When the limit is hit, we remove this fraction of stale futures (oldest first)
# to create headroom for new work while avoiding thrashing (DS-481).
DEFAULT_STALE_REMOVAL_FRACTION = 0.5


@dataclass
class TrackedFuture:
    """A future with associated metadata for TTL tracking.

    Attributes:
        future: The actual Future object.
        created_at: Timestamp when the future was created (monotonic time).
        description: Optional description for logging purposes.
        last_warned_at: Timestamp when a long-running warning was last logged for this
            future (monotonic time). None if never warned. Used to deduplicate warnings
            and avoid log noise for legitimately long-running tasks (DS-481).
    """

    future: Future[Any]
    created_at: float = field(default_factory=time.monotonic)
    description: str = ""
    last_warned_at: float | None = field(default=None)

    def age_seconds(self) -> float:
        """Get the age of this future in seconds."""
        return time.monotonic() - self.created_at

    def is_stale(self, ttl_seconds: float) -> bool:
        """Check if this future has exceeded the TTL threshold.

        Args:
            ttl_seconds: The TTL threshold in seconds.

        Returns:
            True if the future is older than the TTL and not done.
        """
        return not self.future.done() and self.age_seconds() > ttl_seconds


class ExecutionManager:
    """Manages thread pool and execution futures for the Sentinel orchestrator.

    This class is responsible for:
    - Creating and managing the ThreadPoolExecutor
    - Tracking active futures with TTL-based cleanup
    - Collecting completed results
    - Providing slot availability information
    - Managing graceful shutdown
    - Monitoring and logging long-running futures

    Thread Safety:
        All public methods that modify shared state use internal locks.

    Memory Safety:
        Futures are tracked with timestamps and cleaned up based on TTL.
        A maximum list size prevents unbounded memory growth.
    """

    def __init__(
        self,
        max_concurrent_executions: int,
        future_ttl_seconds: float = DEFAULT_FUTURE_TTL_SECONDS,
        max_futures: int = DEFAULT_MAX_FUTURES,
        stale_removal_fraction: float = DEFAULT_STALE_REMOVAL_FRACTION,
    ) -> None:
        """Initialize the execution manager.

        Args:
            max_concurrent_executions: Maximum number of concurrent executions allowed.
            future_ttl_seconds: TTL in seconds for futures before they are considered stale.
                Stale futures are logged and cleaned up. Default is 300 seconds (5 minutes).
            max_futures: Maximum number of futures to track. When exceeded, oldest
                completed futures are removed first, then oldest stale futures.
                Default is 1000.
            stale_removal_fraction: Fraction of stale futures to remove when max_futures
                limit is reached (0.0 to 1.0). Default is 0.5 (50%). This creates headroom
                for new work while avoiding excessive cleanup thrashing. Higher values
                remove more futures at once (fewer cleanups, more aggressive), while lower
                values remove fewer (more frequent cleanups, gentler). Added in DS-481.
        """
        self._max_concurrent_executions = max_concurrent_executions
        self._future_ttl_seconds = future_ttl_seconds
        self._max_futures = max_futures
        self._stale_removal_fraction = max(0.0, min(1.0, stale_removal_fraction))
        self._thread_pool: ThreadPoolExecutor | None = None
        self._active_futures: list[TrackedFuture] = []
        self._futures_lock = threading.Lock()
        self._stale_futures_cleaned = 0  # Counter for monitoring

    @property
    def max_concurrent_executions(self) -> int:
        """Get the maximum number of concurrent executions."""
        return self._max_concurrent_executions

    @property
    def future_ttl_seconds(self) -> float:
        """Get the TTL for futures in seconds."""
        return self._future_ttl_seconds

    @property
    def max_futures(self) -> int:
        """Get the maximum number of futures to track."""
        return self._max_futures

    @property
    def stale_removal_fraction(self) -> float:
        """Get the fraction of stale futures to remove when limit is reached."""
        return self._stale_removal_fraction

    @property
    def stale_futures_cleaned(self) -> int:
        """Get the total count of stale futures that have been cleaned up."""
        return self._stale_futures_cleaned

    def start(self) -> None:
        """Start the thread pool.

        Creates a new ThreadPoolExecutor with the configured max workers.
        """
        if self._thread_pool is not None:
            logger.warning("Thread pool already started")
            return

        self._thread_pool = ThreadPoolExecutor(
            max_workers=self._max_concurrent_executions,
            thread_name_prefix="sentinel-exec-",
        )
        logger.info(f"Started thread pool with max {self._max_concurrent_executions} workers")

    def shutdown(self, block: bool = True, cancel_futures: bool = False) -> None:
        """Shutdown the thread pool.

        Args:
            block: If True, wait for all pending futures to complete.
            cancel_futures: If True, cancel pending futures.
        """
        if self._thread_pool is not None:
            logger.info("Shutting down thread pool...")
            self._thread_pool.shutdown(wait=block, cancel_futures=cancel_futures)
            self._thread_pool = None
            logger.info("Thread pool shutdown complete")

    def is_running(self) -> bool:
        """Check if the thread pool is running.

        Returns:
            True if the thread pool is active, False otherwise.
        """
        return self._thread_pool is not None

    def get_available_slots(self) -> int:
        """Get the number of available execution slots.

        Returns:
            Number of execution slots available for new work.
        """
        with self._futures_lock:
            # Count active futures without removing them
            active_count = sum(1 for tf in self._active_futures if not tf.future.done())
            return self._max_concurrent_executions - active_count

    def get_active_count(self) -> int:
        """Get the count of currently active executions.

        Returns:
            Number of active executions.
        """
        with self._futures_lock:
            return sum(1 for tf in self._active_futures if not tf.future.done())

    def get_active_futures(self) -> list[Future[Any]]:
        """Get a copy of the list of active futures.

        Returns:
            List of currently active Future objects.
        """
        with self._futures_lock:
            return [tf.future for tf in self._active_futures]

    def get_tracked_futures(self) -> list[TrackedFuture]:
        """Get a copy of the list of tracked futures with metadata.

        Returns:
            List of TrackedFuture objects with timing information.
        """
        with self._futures_lock:
            return list(self._active_futures)

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        description: str = "",
        **kwargs: Any,
    ) -> Future[T] | None:
        """Submit a task to the thread pool.

        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            description: Optional description for logging and monitoring.
            **kwargs: Keyword arguments for the callable.

        Returns:
            The Future representing the execution, or None if pool not running
            or max futures limit reached.
        """
        if self._thread_pool is None:
            logger.warning("Cannot submit task: thread pool not running")
            return None

        future = self._thread_pool.submit(fn, *args, **kwargs)
        tracked = TrackedFuture(future=future, description=description)

        with self._futures_lock:
            # Enforce maximum futures limit
            if len(self._active_futures) >= self._max_futures:
                self._enforce_max_futures_limit()

            self._active_futures.append(tracked)

        return future

    def _enforce_max_futures_limit(self) -> None:
        """Enforce the maximum futures limit by removing old futures.

        This method should be called while holding _futures_lock.
        Removes completed futures first, then stale futures if needed.

        Stale Future Removal Strategy (DS-481):
            When the max_futures limit is reached and we still have too many futures
            after removing completed ones, we remove a configurable fraction of stale
            futures (controlled by stale_removal_fraction, default 50%).

            Rationale for removing a fraction rather than all stale futures:
            1. **Headroom Creation**: Removing a fraction creates sufficient headroom
               for new work without being overly aggressive. If we only removed one
               future at a time, we'd trigger cleanup on every submit.
            2. **Thrashing Prevention**: Removing too few futures would cause frequent
               cleanups (thrashing). Removing approximately half strikes a balance.
            3. **Graceful Degradation**: Stale futures may complete eventually. By
               keeping some, we give them a chance to finish before being evicted.
            4. **Predictable Behavior**: A consistent fraction makes the cleanup
               behavior predictable and easier to tune via configuration.

            The fraction is configurable via the stale_removal_fraction parameter
            to allow tuning based on workload characteristics.
        """
        # First, remove completed futures
        original_count = len(self._active_futures)
        self._active_futures = [tf for tf in self._active_futures if not tf.future.done()]

        # If still over limit, remove stale futures (oldest first)
        if len(self._active_futures) >= self._max_futures:
            stale = [tf for tf in self._active_futures if tf.is_stale(self._future_ttl_seconds)]
            if stale:
                # Sort by age (oldest first) and remove a fraction of stale futures.
                # We use the configured fraction (default 50%) plus 1 to ensure we
                # always remove at least one future when over the limit.
                stale.sort(key=lambda tf: tf.created_at)
                num_to_remove = int(len(stale) * self._stale_removal_fraction) + 1
                stale_to_remove = {id(tf) for tf in stale[:num_to_remove]}
                removed_count = len(stale_to_remove)
                self._active_futures = [
                    tf for tf in self._active_futures if id(tf) not in stale_to_remove
                ]
                self._stale_futures_cleaned += removed_count
                logger.warning(
                    f"Max futures limit ({self._max_futures}) reached. "
                    f"Removed {removed_count} stale futures "
                    f"({self._stale_removal_fraction:.0%} of {len(stale)} stale)."
                )

        removed = original_count - len(self._active_futures)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} futures to enforce limit")

    def collect_completed_results(self) -> list[ExecutionResult]:
        """Collect results from completed futures and clean up stale ones.

        This method also performs TTL-based cleanup:
        - Removes completed futures after collecting their results
        - Logs and removes stale futures that have exceeded the TTL
        - Logs warnings for long-running futures

        Returns:
            List of execution results from completed futures.
        """
        results: list[ExecutionResult] = []
        with self._futures_lock:
            completed = [tf for tf in self._active_futures if tf.future.done()]
            for tracked in completed:
                try:
                    result = tracked.future.result()
                    if result is not None:
                        results.append(result)
                except (OSError, TimeoutError) as e:
                    logger.error(
                        f"Error collecting result from future due to I/O or timeout: {e}",
                        extra={"description": tracked.description, "error_type": type(e).__name__},
                    )
                except RuntimeError as e:
                    logger.error(
                        f"Error collecting result from future due to runtime error: {e}",
                        extra={"description": tracked.description, "error_type": type(e).__name__},
                    )
                except (KeyError, ValueError) as e:
                    logger.error(
                        f"Error collecting result from future due to data error: {e}",
                        extra={"description": tracked.description, "error_type": type(e).__name__},
                    )

            # Log warnings for long-running futures
            self._log_long_running_futures()

            # Identify and clean up stale futures
            stale_count = self._cleanup_stale_futures()
            if stale_count > 0:
                logger.warning(
                    f"Cleaned up {stale_count} stale futures that exceeded "
                    f"TTL of {self._future_ttl_seconds}s"
                )

            # Remove completed futures
            self._active_futures = [tf for tf in self._active_futures if not tf.future.done()]

        return results

    def _log_long_running_futures(self) -> None:
        """Log warnings for futures that have been running for a long time.

        This method should be called while holding _futures_lock.

        Warning Deduplication Strategy (DS-481):
            To avoid log flooding for legitimately long-running tasks, we track
            when each future was last warned about via the `last_warned_at` field.

            - First warning: Logged at WARNING level, updates `last_warned_at`
            - Subsequent checks within LONG_RUNNING_WARNING_INTERVAL: Logged at DEBUG
            - After LONG_RUNNING_WARNING_INTERVAL has elapsed: Logged at WARNING again

            This ensures visibility into long-running tasks without creating excessive
            log noise when collect_completed_results() is called frequently.
        """
        current_time = time.monotonic()
        for tracked in self._active_futures:
            if not tracked.future.done():
                age = tracked.age_seconds()
                if age > LONG_RUNNING_WARNING_THRESHOLD:
                    desc = tracked.description or "unnamed task"
                    message = (
                        f"Long-running future detected: {desc} "
                        f"(running for {age:.1f}s)"
                    )

                    # Determine whether to log at WARNING or DEBUG level
                    if tracked.last_warned_at is None:
                        # First warning for this future
                        logger.warning(message)
                        tracked.last_warned_at = current_time
                    elif current_time - tracked.last_warned_at >= LONG_RUNNING_WARNING_INTERVAL:
                        # Enough time has passed since last warning
                        logger.warning(message)
                        tracked.last_warned_at = current_time
                    else:
                        # Recently warned, log at DEBUG to reduce noise
                        logger.debug(message)

    def _cleanup_stale_futures(self) -> int:
        """Clean up futures that have exceeded the TTL threshold.

        This method should be called while holding _futures_lock.
        Stale futures are cancelled if possible and removed from tracking.

        Returns:
            Number of stale futures cleaned up.
        """
        stale_futures = [
            tf for tf in self._active_futures
            if tf.is_stale(self._future_ttl_seconds)
        ]

        for tracked in stale_futures:
            desc = tracked.description or "unnamed task"
            age = tracked.age_seconds()
            logger.error(
                f"Future exceeded TTL: {desc} (age: {age:.1f}s, TTL: {self._future_ttl_seconds}s). "
                "This may indicate a hung task or network issue."
            )
            # Attempt to cancel the future (may not work if already running)
            tracked.future.cancel()

        # Remove stale futures from tracking
        stale_ids = {id(tf) for tf in stale_futures}
        self._active_futures = [
            tf for tf in self._active_futures if id(tf) not in stale_ids
        ]
        self._stale_futures_cleaned += len(stale_futures)

        return len(stale_futures)

    def get_pending_futures(self) -> list[Future[Any]]:
        """Get futures that are still pending.

        Returns:
            List of futures that haven't completed yet.
        """
        with self._futures_lock:
            return [tf.future for tf in self._active_futures if not tf.future.done()]

    def wait_for_completion(
        self,
        timeout: float | None = None,
        return_when: str = "FIRST_COMPLETED",
    ) -> tuple[set[Future[Any]], set[Future[Any]]]:
        """Wait for futures to complete.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.
            return_when: When to return. One of FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED.

        Returns:
            Tuple of (done_futures, not_done_futures).
        """
        pending = self.get_pending_futures()
        if not pending:
            return set(), set()

        from concurrent.futures import ALL_COMPLETED as AC
        from concurrent.futures import FIRST_COMPLETED as FC
        from concurrent.futures import FIRST_EXCEPTION as FE

        when_map = {
            "FIRST_COMPLETED": FC,
            "ALL_COMPLETED": AC,
            "FIRST_EXCEPTION": FE,
        }

        return wait(pending, timeout=timeout, return_when=when_map.get(return_when, FC))

    def wait_for_all_completion(self, poll_interval: float = 0.1) -> list[ExecutionResult]:
        """Wait for all pending futures to complete and collect results.

        This is useful for --once mode where we want to wait for all work to finish.

        Args:
            poll_interval: How often to check for completion in seconds.

        Returns:
            List of all execution results.
        """
        all_results: list[ExecutionResult] = []

        while True:
            with self._futures_lock:
                pending = [tf for tf in self._active_futures if not tf.future.done()]

            if not pending:
                break

            time.sleep(poll_interval)

        # Collect final results
        all_results.extend(self.collect_completed_results())
        return all_results

    def execute_synchronously(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T | None:
        """Execute a task synchronously (fallback when pool not running).

        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            The result of the callable, or None if an error occurred.
        """
        try:
            return fn(*args, **kwargs)
        except (OSError, TimeoutError) as e:
            logger.error(
                f"Error in synchronous execution due to I/O or timeout: {e}",
                extra={"error_type": type(e).__name__},
            )
            return None
        except RuntimeError as e:
            logger.error(
                f"Error in synchronous execution due to runtime error: {e}",
                extra={"error_type": type(e).__name__},
            )
            return None
        except (KeyError, ValueError) as e:
            logger.error(
                f"Error in synchronous execution due to data error: {e}",
                extra={"error_type": type(e).__name__},
            )
            return None

    def cleanup_completed_futures(
        self, on_future_done: Callable[[Future[Any]], None] | None = None
    ) -> int:
        """Clean up completed futures from the tracking list.

        Args:
            on_future_done: Optional callback for each completed future.

        Returns:
            Number of futures cleaned up (completed + stale).
        """
        cleaned_count = 0
        with self._futures_lock:
            completed = [tf for tf in self._active_futures if tf.future.done()]
            for tracked in completed:
                if on_future_done:
                    try:
                        on_future_done(tracked.future)
                    except (OSError, RuntimeError) as e:
                        logger.error(
                            f"Error in future done callback due to runtime/I/O error: {e}",
                            extra={
                                "description": tracked.description,
                                "error_type": type(e).__name__,
                            },
                        )
                    except (KeyError, ValueError) as e:
                        logger.error(
                            f"Error in future done callback due to data error: {e}",
                            extra={
                                "description": tracked.description,
                                "error_type": type(e).__name__,
                            },
                        )
                cleaned_count += 1

            # Also clean up stale futures
            stale_count = self._cleanup_stale_futures()
            cleaned_count += stale_count

            self._active_futures = [tf for tf in self._active_futures if not tf.future.done()]
        return cleaned_count

    def get_futures_stats(self) -> dict[str, Any]:
        """Get statistics about tracked futures for monitoring.

        Returns:
            Dictionary with futures statistics including counts, ages, and stale info.
        """
        with self._futures_lock:
            total = len(self._active_futures)
            pending = sum(1 for tf in self._active_futures if not tf.future.done())
            completed = total - pending
            stale = sum(
                1 for tf in self._active_futures if tf.is_stale(self._future_ttl_seconds)
            )
            long_running = sum(
                1 for tf in self._active_futures
                if not tf.future.done() and tf.age_seconds() > LONG_RUNNING_WARNING_THRESHOLD
            )

            ages = [tf.age_seconds() for tf in self._active_futures if not tf.future.done()]
            max_age = max(ages) if ages else 0.0
            avg_age = sum(ages) / len(ages) if ages else 0.0

            return {
                "total_tracked": total,
                "pending": pending,
                "completed": completed,
                "stale": stale,
                "long_running": long_running,
                "max_age_seconds": max_age,
                "avg_age_seconds": avg_age,
                "total_stale_cleaned": self._stale_futures_cleaned,
                "ttl_seconds": self._future_ttl_seconds,
                "max_futures": self._max_futures,
                "stale_removal_fraction": self._stale_removal_fraction,
            }
