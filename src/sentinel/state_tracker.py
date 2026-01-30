"""State tracking for the Sentinel orchestrator.

This module provides the StateTracker class which manages:
- Attempt count tracking with TTL-based cleanup
- Running step metadata for dashboard display
- Issue queue management
- Per-orchestration execution counts

This is part of the Sentinel refactoring to split the God Object into focused,
composable components (DS-384).

Component Boundaries
--------------------
StateTracker is responsible for all runtime state that needs to persist across
polling cycles but is not related to orchestration definitions or thread pool
management.

The Sentinel class delegates to StateTracker for:
- ``get_running_steps()`` -> ``StateTracker.get_running_steps()``
- ``get_issue_queue()`` -> ``StateTracker.get_issue_queue()``
- ``get_per_orch_count()`` -> ``StateTracker.get_per_orch_count()``
- ``get_all_per_orch_counts()`` -> ``StateTracker.get_all_per_orch_counts()``
- ``_get_available_slots_for_orchestration()`` ->
  ``StateTracker.get_available_slots_for_orchestration()``

See Also
--------
- docs/architecture.md : Full architecture documentation
- sentinel.execution_manager : Thread pool and future management
- sentinel.orchestration_registry : Orchestration loading and hot-reload
- sentinel.poll_coordinator : Polling coordination
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sentinel.logging import get_logger

if TYPE_CHECKING:
    from concurrent.futures import Future

    from sentinel.orchestration import Orchestration

logger = get_logger(__name__)


@dataclass
class AttemptCountEntry:
    """Entry in the attempt counts dictionary tracking count and last access time.

    This class tracks both the attempt count and the last time the entry was accessed,
    enabling TTL-based cleanup to prevent unbounded memory growth.
    """

    count: int
    last_access: float  # time.monotonic() timestamp


@dataclass
class RunningStepInfo:
    """Metadata about a currently running execution step.

    This class tracks information about active agent executions for dashboard display.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    issue_url: str  # URL to Jira or GitHub issue


@dataclass
class QueuedIssueInfo:
    """Metadata about an issue waiting in queue for an execution slot.

    This class tracks information about issues that matched orchestration triggers
    but couldn't be executed immediately due to all execution slots being full.
    """

    issue_key: str
    orchestration_name: str
    queued_at: datetime


@dataclass
class CompletedExecutionInfo:
    """Metadata about a completed execution with usage data.

    This class tracks information about completed agent executions for dashboard display.
    It captures the execution context, timing, success/failure status, and usage data
    (token counts and cost) from the Claude Agent SDK.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    completed_at: datetime
    status: str  # "success" or "failure"
    input_tokens: int
    output_tokens: int
    total_cost_usd: float
    issue_url: str


class StateTracker:
    """Tracks execution state, metrics, and queues for the Sentinel orchestrator.

    This class is responsible for:
    - Tracking attempt counts per (issue_key, orchestration_name) pair
    - Tracking running step metadata for dashboard display
    - Managing the issue queue for issues waiting for execution slots
    - Tracking per-orchestration active execution counts

    Thread Safety:
        All public methods are thread-safe and use internal locks.

    Lock Ordering Discipline:
        This class uses multiple locks to protect different data structures.
        To prevent deadlocks, the following lock ordering MUST be followed
        if multiple locks need to be acquired simultaneously:

            1. _attempt_counts_lock (highest priority)
            2. _running_steps_lock
            3. _queue_lock
            4. _per_orch_counts_lock
            5. _completed_executions_lock (lowest priority)

        Rules:
        - When acquiring multiple locks, always acquire them in the order above.
        - Never acquire a higher-priority lock while holding a lower-priority lock.
        - Currently, all methods acquire only a single lock at a time, which is
          the preferred pattern. This ordering is documented for future maintenance
          in case methods need to be added that require multiple locks.
        - If you need to call a method that acquires a different lock, release
          your current lock first to avoid potential deadlock.

        Example (if multiple locks were needed):
            # CORRECT: Acquire locks in order
            with self._attempt_counts_lock:
                with self._running_steps_lock:
                    # ... do work ...

            # INCORRECT: Would violate lock ordering
            with self._running_steps_lock:
                with self._attempt_counts_lock:  # NEVER DO THIS
                    # ... do work ...
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        attempt_counts_ttl: float = 86400.0,
        max_completed_executions: int = 50,
    ) -> None:
        """Initialize the state tracker.

        Args:
            max_queue_size: Maximum size of the issue queue. When full, oldest items
                are evicted to make room for new ones.
            attempt_counts_ttl: Time-to-live in seconds for attempt count entries.
                Entries older than this are cleaned up to prevent unbounded memory growth.
            max_completed_executions: Maximum number of completed executions to track.
                When full, oldest entries are evicted to make room for new ones.
        """
        self._max_queue_size = max_queue_size
        self._attempt_counts_ttl = attempt_counts_ttl
        self._max_completed_executions = max_completed_executions

        # Track attempt counts per (issue_key, orchestration_name) pair
        self._attempt_counts: dict[tuple[str, str], AttemptCountEntry] = {}
        # Lock ordering priority: 1 (highest) - see class docstring for details
        self._attempt_counts_lock = threading.Lock()

        # Track running step metadata for dashboard display
        # Maps future id() to RunningStepInfo for active executions
        self._running_steps: dict[int, RunningStepInfo] = {}
        # Lock ordering priority: 2 - see class docstring for details
        self._running_steps_lock = threading.Lock()

        # Track queued issues waiting for execution slots
        self._issue_queue: deque[QueuedIssueInfo] = deque(maxlen=max_queue_size)
        # Lock ordering priority: 3 - see class docstring for details
        self._queue_lock = threading.Lock()

        # Track per-orchestration active execution counts
        self._per_orch_active_counts: defaultdict[str, int] = defaultdict(int)
        # Lock ordering priority: 4 - see class docstring for details
        self._per_orch_counts_lock = threading.Lock()

        # Track completed executions for dashboard display
        # Uses appendleft for most recent first, maxlen for automatic eviction
        self._completed_executions: deque[CompletedExecutionInfo] = deque(
            maxlen=max_completed_executions
        )
        # Lock ordering priority: 5 (lowest) - see class docstring for details
        self._completed_executions_lock = threading.Lock()

        # Track process start time
        self._start_time: datetime = datetime.now()

        # Track last poll times
        self._last_jira_poll: datetime | None = None
        self._last_github_poll: datetime | None = None

    @property
    def start_time(self) -> datetime:
        """Get the process start time."""
        return self._start_time

    @property
    def last_jira_poll(self) -> datetime | None:
        """Get the last Jira poll time."""
        return self._last_jira_poll

    @last_jira_poll.setter
    def last_jira_poll(self, value: datetime) -> None:
        """Set the last Jira poll time."""
        self._last_jira_poll = value

    @property
    def last_github_poll(self) -> datetime | None:
        """Get the last GitHub poll time."""
        return self._last_github_poll

    @last_github_poll.setter
    def last_github_poll(self, value: datetime) -> None:
        """Set the last GitHub poll time."""
        self._last_github_poll = value

    # =========================================================================
    # Attempt Count Tracking
    # =========================================================================

    def get_and_increment_attempt_count(self, issue_key: str, orchestration_name: str) -> int:
        """Get and increment the attempt count for an issue/orchestration pair.

        This method atomically increments the attempt count and returns the new value.
        It tracks how many times an issue has been processed for a given orchestration
        across different polling cycles, providing accurate retry attempt numbers
        in the Running Steps dashboard.

        Also updates the last_access time to support TTL-based cleanup.

        Args:
            issue_key: The key of the issue being processed.
            orchestration_name: The name of the orchestration being executed.

        Returns:
            The new attempt number (1 for first attempt, 2 for second, etc.).
        """
        key = (issue_key, orchestration_name)
        current_time = time.monotonic()
        with self._attempt_counts_lock:
            entry = self._attempt_counts.get(key)
            new_count = 1 if entry is None else entry.count + 1
            self._attempt_counts[key] = AttemptCountEntry(count=new_count, last_access=current_time)
            return new_count

    def cleanup_stale_attempt_counts(self) -> int:
        """Clean up stale attempt count entries based on TTL.

        Removes entries from _attempt_counts that haven't been accessed within
        the configured TTL period. This prevents unbounded memory growth for
        long-running processes.

        Returns:
            Number of entries cleaned up.
        """
        current_time = time.monotonic()
        cleaned_count = 0

        with self._attempt_counts_lock:
            stale_keys = [
                key
                for key, entry in self._attempt_counts.items()
                if (current_time - entry.last_access) > self._attempt_counts_ttl
            ]
            for key in stale_keys:
                del self._attempt_counts[key]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale attempt count entries "
                f"(TTL: {self._attempt_counts_ttl}s)"
            )
        else:
            logger.debug(
                f"Attempt counts cleanup: no stale entries found (TTL: {self._attempt_counts_ttl}s)"
            )

        return cleaned_count

    # =========================================================================
    # Running Steps Tracking
    # =========================================================================

    def add_running_step(
        self,
        future_id: int,
        issue_key: str,
        orchestration_name: str,
        attempt_number: int,
        issue_url: str,
    ) -> None:
        """Add a running step entry for a future.

        Args:
            future_id: The id() of the Future object.
            issue_key: The key of the issue being processed.
            orchestration_name: The name of the orchestration.
            attempt_number: The attempt number for this execution.
            issue_url: URL to the issue (Jira or GitHub).
        """
        with self._running_steps_lock:
            self._running_steps[future_id] = RunningStepInfo(
                issue_key=issue_key,
                orchestration_name=orchestration_name,
                attempt_number=attempt_number,
                started_at=datetime.now(),
                issue_url=issue_url,
            )

    def remove_running_step(self, future_id: int) -> RunningStepInfo | None:
        """Remove a running step entry for a completed future.

        Args:
            future_id: The id() of the Future object.

        Returns:
            The removed RunningStepInfo, or None if not found.
        """
        with self._running_steps_lock:
            return self._running_steps.pop(future_id, None)

    def get_running_steps(self, active_futures: list[Future[Any]]) -> list[RunningStepInfo]:
        """Get information about currently running execution steps.

        Args:
            active_futures: List of currently active Future objects.

        Returns:
            List of RunningStepInfo for active executions.
        """
        with self._running_steps_lock:
            running = []
            for future in active_futures:
                if not future.done():
                    future_id = id(future)
                    if future_id in self._running_steps:
                        running.append(self._running_steps[future_id])
            return running

    # =========================================================================
    # Issue Queue Management
    # =========================================================================

    def clear_issue_queue(self) -> None:
        """Clear the issue queue at the start of a new polling cycle.

        The queue is cleared each cycle because issues will be re-polled
        and re-added if they still match and slots are still unavailable.
        """
        with self._queue_lock:
            self._issue_queue.clear()

    def add_to_issue_queue(self, issue_key: str, orchestration_name: str) -> None:
        """Add an issue to the queue when no execution slot is available.

        Queue uses collections.deque with maxlen for automatic oldest-item
        eviction when full. This provides FIFO behavior - when the queue is at
        capacity, adding a new item automatically evicts the oldest item instead
        of dropping the new one.

        Args:
            issue_key: The key of the issue being queued.
            orchestration_name: The name of the orchestration the issue matched.
        """
        with self._queue_lock:
            evicted_item: QueuedIssueInfo | None = None
            # Capture the item that will be evicted before appending
            if len(self._issue_queue) == self._issue_queue.maxlen:
                evicted_item = self._issue_queue[0]  # Oldest item (leftmost)

            self._issue_queue.append(
                QueuedIssueInfo(
                    issue_key=issue_key,
                    orchestration_name=orchestration_name,
                    queued_at=datetime.now(),
                )
            )

            if evicted_item is not None:
                logger.debug(
                    f"Issue queue at capacity ({self._issue_queue.maxlen}), "
                    f"evicted '{evicted_item.issue_key}' "
                    f"(orch: '{evicted_item.orchestration_name}') "
                    f"to add {issue_key} for '{orchestration_name}'"
                )

    def get_issue_queue(self) -> list[QueuedIssueInfo]:
        """Get information about issues waiting in queue for execution slots.

        Returns:
            List of QueuedIssueInfo for queued issues.
        """
        with self._queue_lock:
            return list(self._issue_queue)

    # =========================================================================
    # Per-Orchestration Execution Counts
    # =========================================================================

    def increment_per_orch_count(self, orchestration_name: str) -> int:
        """Increment the active execution count for an orchestration.

        Args:
            orchestration_name: The name of the orchestration being executed.

        Returns:
            The new active count for this orchestration after incrementing.
        """
        with self._per_orch_counts_lock:
            self._per_orch_active_counts[orchestration_name] += 1
            new_count = self._per_orch_active_counts[orchestration_name]
            logger.debug(
                f"Incremented per-orch count for '{orchestration_name}': "
                f"{new_count - 1} -> {new_count}"
            )
            return new_count

    def decrement_per_orch_count(self, orchestration_name: str) -> int:
        """Decrement the active execution count for an orchestration.

        Args:
            orchestration_name: The name of the orchestration that completed.

        Returns:
            The new active count for this orchestration after decrementing.
        """
        with self._per_orch_counts_lock:
            current_count = self._per_orch_active_counts[orchestration_name]
            if current_count == 0:
                logger.warning(
                    f"Attempted to decrement per-orch count for '{orchestration_name}' "
                    f"but count was already 0"
                )
                return 0
            new_count = current_count - 1
            if new_count == 0:
                # Clean up entry when count reaches 0
                del self._per_orch_active_counts[orchestration_name]
            else:
                self._per_orch_active_counts[orchestration_name] = new_count
            logger.debug(
                f"Decremented per-orch count for '{orchestration_name}': "
                f"{current_count} -> {new_count}"
            )
            return new_count

    def get_per_orch_count(self, orchestration_name: str) -> int:
        """Get the active execution count for a specific orchestration.

        Args:
            orchestration_name: The name of the orchestration to query.

        Returns:
            The current active execution count for the orchestration.
        """
        with self._per_orch_counts_lock:
            return self._per_orch_active_counts.get(orchestration_name, 0)

    def get_all_per_orch_counts(self) -> dict[str, int]:
        """Get all per-orchestration active execution counts.

        Returns:
            A dictionary mapping orchestration names to their active execution counts.
        """
        with self._per_orch_counts_lock:
            return dict(self._per_orch_active_counts)

    def get_available_slots_for_orchestration(
        self,
        orchestration: Orchestration,
        global_available: int,
    ) -> int:
        """Get available slots for a specific orchestration considering both limits.

        Args:
            orchestration: The orchestration to check available slots for.
            global_available: The number of globally available slots.

        Returns:
            The number of available execution slots for this orchestration.
        """
        # If orchestration has no per-orchestration limit, use global only
        if orchestration.max_concurrent is None:
            return global_available

        # Calculate per-orchestration available slots
        with self._per_orch_counts_lock:
            current_orch_count = self._per_orch_active_counts.get(orchestration.name, 0)
            per_orch_available = orchestration.max_concurrent - current_orch_count

        # Return the minimum of global and per-orchestration limits
        available = min(global_available, per_orch_available)

        logger.debug(
            f"Available slots for '{orchestration.name}': {available} "
            f"(global: {global_available}, per-orch: {per_orch_available}, "
            f"max_concurrent: {orchestration.max_concurrent})"
        )

        return max(0, available)

    # =========================================================================
    # Completed Executions Tracking
    # =========================================================================

    def add_completed_execution(self, info: CompletedExecutionInfo) -> None:
        """Add a completed execution entry for dashboard display.

        Uses appendleft to add entries at the front of the deque, ensuring
        most recent executions are first. The deque's maxlen automatically
        evicts the oldest entry when the maximum size is exceeded.

        Args:
            info: The completed execution information to add.
        """
        with self._completed_executions_lock:
            self._completed_executions.appendleft(info)
            logger.debug(
                f"Recorded completed execution for '{info.issue_key}' "
                f"(orchestration: '{info.orchestration_name}', status: {info.status})"
            )

    def get_completed_executions(self) -> list[CompletedExecutionInfo]:
        """Get the list of completed executions for dashboard display.

        Returns:
            A list of CompletedExecutionInfo entries, ordered with most recent first.
        """
        with self._completed_executions_lock:
            return list(self._completed_executions)
