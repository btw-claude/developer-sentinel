"""Execution manager for the Sentinel orchestrator.

This module provides the ExecutionManager class which manages:
- Thread pool lifecycle
- Future tracking and collection
- Execution submission and result collection
- Graceful shutdown handling

This is part of the Sentinel refactoring to split the God Object into focused,
composable components (DS-384).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, TypeVar

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.executor import ExecutionResult

logger = get_logger(__name__)

T = TypeVar("T")


class ExecutionManager:
    """Manages thread pool and execution futures for the Sentinel orchestrator.

    This class is responsible for:
    - Creating and managing the ThreadPoolExecutor
    - Tracking active futures
    - Collecting completed results
    - Providing slot availability information
    - Managing graceful shutdown

    Thread Safety:
        All public methods that modify shared state use internal locks.
    """

    def __init__(self, max_concurrent_executions: int) -> None:
        """Initialize the execution manager.

        Args:
            max_concurrent_executions: Maximum number of concurrent executions allowed.
        """
        self._max_concurrent_executions = max_concurrent_executions
        self._thread_pool: ThreadPoolExecutor | None = None
        self._active_futures: list[Future[Any]] = []
        self._futures_lock = threading.Lock()

    @property
    def max_concurrent_executions(self) -> int:
        """Get the maximum number of concurrent executions."""
        return self._max_concurrent_executions

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
            active_count = sum(1 for f in self._active_futures if not f.done())
            return self._max_concurrent_executions - active_count

    def get_active_count(self) -> int:
        """Get the count of currently active executions.

        Returns:
            Number of active executions.
        """
        with self._futures_lock:
            return sum(1 for f in self._active_futures if not f.done())

    def get_active_futures(self) -> list[Future[Any]]:
        """Get a copy of the list of active futures.

        Returns:
            List of currently active Future objects.
        """
        with self._futures_lock:
            return list(self._active_futures)

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> Future[T] | None:
        """Submit a task to the thread pool.

        Args:
            fn: The callable to execute.
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            The Future representing the execution, or None if pool not running.
        """
        if self._thread_pool is None:
            logger.warning("Cannot submit task: thread pool not running")
            return None

        future = self._thread_pool.submit(fn, *args, **kwargs)
        with self._futures_lock:
            self._active_futures.append(future)
        return future

    def collect_completed_results(self) -> list[ExecutionResult]:
        """Collect results from completed futures.

        Returns:
            List of execution results from completed futures.
        """

        results: list[ExecutionResult] = []
        with self._futures_lock:
            completed = [f for f in self._active_futures if f.done()]
            for future in completed:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error collecting result from future: {e}")
            self._active_futures = [f for f in self._active_futures if not f.done()]
        return results

    def get_pending_futures(self) -> list[Future[Any]]:
        """Get futures that are still pending.

        Returns:
            List of futures that haven't completed yet.
        """
        with self._futures_lock:
            return [f for f in self._active_futures if not f.done()]

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
        import time

        all_results: list[ExecutionResult] = []

        while True:
            with self._futures_lock:
                pending = [f for f in self._active_futures if not f.done()]

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
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return None

    def cleanup_completed_futures(
        self, on_future_done: Callable[[Future[Any]], None] | None = None
    ) -> int:
        """Clean up completed futures from the tracking list.

        Args:
            on_future_done: Optional callback for each completed future.

        Returns:
            Number of futures cleaned up.
        """
        cleaned_count = 0
        with self._futures_lock:
            completed = [f for f in self._active_futures if f.done()]
            for future in completed:
                if on_future_done:
                    try:
                        on_future_done(future)
                    except Exception as e:
                        logger.error(f"Error in future done callback: {e}")
                cleaned_count += 1
            self._active_futures = [f for f in self._active_futures if not f.done()]
        return cleaned_count
