"""Deduplication utilities for preventing duplicate agent spawns.

This module provides shared deduplication logic used by both Jira and GitHub
trigger polling to prevent spawning multiple agents for the same issue/orchestration
combination within a single polling cycle.

DS-138: Refactored from inline implementations in _poll_jira_triggers() and
_poll_github_triggers() (DS-130, DS-131) into a shared utility.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from sentinel.logging import get_logger

logger = get_logger(__name__)


class DeduplicationManager:
    """Manages deduplication of issue/orchestration pairs within a polling cycle.

    This class tracks (issue_key, orchestration_name) tuples to prevent duplicate
    agent spawns when the same issue matches multiple overlapping triggers before
    tag updates propagate.

    The manager is designed to be:
    1. Created fresh for each polling cycle (via create_cycle_tracker)
    2. Thread-safe for concurrent access during task submission
    3. Reusable across different trigger types (Jira, GitHub, etc.)

    Usage:
        # In run_once():
        dedup_manager = DeduplicationManager()

        # In each polling function:
        with dedup_manager.cycle_tracker() as tracker:
            # ... poll and route issues ...
            if tracker.should_submit(issue_key, orchestration_name):
                # Submit the task
                pass

    Or using the simpler set-based interface:
        submitted_pairs = dedup_manager.create_cycle_set()
        if dedup_manager.check_and_mark(submitted_pairs, issue_key, orch_name):
            # Submit the task
            pass
    """

    def __init__(self) -> None:
        """Initialize the deduplication manager."""
        self._lock = threading.Lock()
        # Global set tracking all submitted pairs across all trigger types in current cycle
        self._current_cycle_pairs: set[tuple[str, str]] = set()

    def reset_cycle(self) -> None:
        """Reset the deduplication state for a new polling cycle.

        Call this at the start of each polling cycle to clear the tracking
        of previously submitted pairs.
        """
        with self._lock:
            self._current_cycle_pairs.clear()

    def create_cycle_set(self) -> set[tuple[str, str]]:
        """Create a new set for tracking submitted pairs in a polling cycle.

        This is a convenience method for the existing pattern where polling
        functions pass a set to _submit_execution_tasks().

        Returns:
            An empty set to track (issue_key, orchestration_name) tuples.
        """
        return set()

    def check_and_mark(
        self,
        submitted_pairs: set[tuple[str, str]],
        issue_key: str,
        orchestration_name: str,
    ) -> bool:
        """Check if a pair should be submitted and mark it as submitted.

        This method atomically checks if the (issue_key, orchestration_name) pair
        has already been submitted in this polling cycle. If not, it marks it as
        submitted and returns True. If already submitted, returns False.

        Args:
            submitted_pairs: Set tracking submitted pairs for this cycle.
            issue_key: The issue key (e.g., "PROJ-123" or "org/repo#123").
            orchestration_name: The name of the orchestration.

        Returns:
            True if this pair should be submitted (was not previously seen),
            False if this pair was already submitted (should be skipped).
        """
        pair = (issue_key, orchestration_name)
        if pair in submitted_pairs:
            logger.debug(
                f"Skipping duplicate submission of '{orchestration_name}' "
                f"for {issue_key} (already submitted this cycle)"
            )
            return False
        submitted_pairs.add(pair)
        return True

    def is_duplicate(
        self,
        submitted_pairs: set[tuple[str, str]],
        issue_key: str,
        orchestration_name: str,
    ) -> bool:
        """Check if a pair has already been submitted this cycle.

        Unlike check_and_mark(), this method only checks without marking.
        Use this when you need to check before deciding whether to proceed.

        Args:
            submitted_pairs: Set tracking submitted pairs for this cycle.
            issue_key: The issue key (e.g., "PROJ-123" or "org/repo#123").
            orchestration_name: The name of the orchestration.

        Returns:
            True if this pair was already submitted (is a duplicate),
            False if this pair has not been submitted yet.
        """
        return (issue_key, orchestration_name) in submitted_pairs

    def mark_submitted(
        self,
        submitted_pairs: set[tuple[str, str]],
        issue_key: str,
        orchestration_name: str,
    ) -> None:
        """Mark a pair as submitted without checking.

        Use this after successfully submitting a task to ensure it's tracked.

        Args:
            submitted_pairs: Set tracking submitted pairs for this cycle.
            issue_key: The issue key (e.g., "PROJ-123" or "org/repo#123").
            orchestration_name: The name of the orchestration.
        """
        submitted_pairs.add((issue_key, orchestration_name))

    @contextmanager
    def cycle_tracker(self) -> Iterator[CycleTracker]:
        """Context manager for tracking submissions within a polling cycle.

        This provides a more object-oriented interface for tracking duplicates
        within a single polling cycle.

        Yields:
            A CycleTracker instance for checking and marking submissions.
        """
        tracker = CycleTracker(self)
        try:
            yield tracker
        finally:
            # Merge tracker's pairs into the global set for cross-trigger deduplication
            with self._lock:
                self._current_cycle_pairs.update(tracker._pairs)


class CycleTracker:
    """Tracks submitted pairs within a single polling cycle.

    This class provides a clean interface for deduplication within a polling
    cycle, encapsulating the set operations.

    Usage:
        with dedup_manager.cycle_tracker() as tracker:
            if tracker.should_submit(issue_key, orch_name):
                # Submit the task
                pass
    """

    def __init__(self, manager: DeduplicationManager) -> None:
        """Initialize the cycle tracker.

        Args:
            manager: The parent DeduplicationManager.
        """
        self._manager = manager
        self._pairs: set[tuple[str, str]] = set()

    def should_submit(self, issue_key: str, orchestration_name: str) -> bool:
        """Check if this pair should be submitted and mark it if so.

        Args:
            issue_key: The issue key (e.g., "PROJ-123" or "org/repo#123").
            orchestration_name: The name of the orchestration.

        Returns:
            True if this pair should be submitted (first occurrence),
            False if this pair has already been seen (duplicate).
        """
        return self._manager.check_and_mark(self._pairs, issue_key, orchestration_name)

    def is_duplicate(self, issue_key: str, orchestration_name: str) -> bool:
        """Check if this pair has already been submitted.

        Args:
            issue_key: The issue key (e.g., "PROJ-123" or "org/repo#123").
            orchestration_name: The name of the orchestration.

        Returns:
            True if this pair was already submitted (duplicate),
            False if this pair has not been seen yet.
        """
        return self._manager.is_duplicate(self._pairs, issue_key, orchestration_name)

    @property
    def submitted_count(self) -> int:
        """Get the number of unique pairs submitted in this cycle."""
        return len(self._pairs)

    def get_submitted_pairs(self) -> set[tuple[str, str]]:
        """Get a copy of all submitted pairs in this cycle.

        Returns:
            A copy of the set of (issue_key, orchestration_name) tuples.
        """
        return self._pairs.copy()
