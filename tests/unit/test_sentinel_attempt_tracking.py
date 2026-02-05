"""Tests for Sentinel retry attempt tracking functionality."""

import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch

import pytest

from sentinel.executor import AgentRunResult
from sentinel.main import Sentinel

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockAgentClient,
    MockJiraPoller,
    MockTagClient,
    make_config,
    make_issue,
    make_orchestration,
)
from tests.mocks import MockAgentClientFactory

# Polling configuration constants for test coordination
POLLING_INTERVAL = 0.05  # seconds between polls
POLLING_MAX_WAIT = 10  # seconds total max wait time
POLLING_ITERATIONS = int(POLLING_MAX_WAIT / POLLING_INTERVAL)


class TestAttemptCountTracking:
    """Tests for retry attempt number tracking in Running Steps dashboard."""

    def test_get_and_increment_attempt_count_starts_at_one(self) -> None:
        """Test that first attempt for an issue/orchestration pair returns 1."""
        jira_poller = MockJiraPoller(issues=[])
        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        count = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")
        assert count == 1

    def test_get_and_increment_attempt_count_increments(self) -> None:
        """Test that subsequent calls increment the attempt count."""
        jira_poller = MockJiraPoller(issues=[])
        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        count1 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")
        count2 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")
        count3 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3

    def test_get_and_increment_tracks_separately_by_issue_key(self) -> None:
        """Test that different issues have separate attempt counts."""
        jira_poller = MockJiraPoller(issues=[])
        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")
        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "my-orchestration")

        count = sentinel._state_tracker.get_and_increment_attempt_count("TEST-2", "my-orchestration")
        assert count == 1

    def test_get_and_increment_tracks_separately_by_orchestration(self) -> None:
        """Test that different orchestrations have separate attempt counts."""
        jira_poller = MockJiraPoller(issues=[])
        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "orch-a")
        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "orch-a")

        count = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "orch-b")
        assert count == 1

    def test_get_and_increment_is_thread_safe(self) -> None:
        """Test that attempt count incrementing is thread-safe."""
        jira_poller = MockJiraPoller(issues=[])
        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        results: list[int] = []
        num_threads = 10

        def increment() -> None:
            count = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "orch")
            results.append(count)

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(1, num_threads + 1))

    def test_running_steps_use_tracked_attempt_number(self) -> None:
        """Test that Running Steps dashboard uses the tracked attempt number."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS", "SUCCESS"])
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        count1 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")
        count2 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")
        count3 = sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")

        assert count1 == 1, "First attempt should be 1"
        assert count2 == 2, "Second attempt should be 2"
        assert count3 == 3, "Third attempt should be 3"

        key = ("TEST-1", "test-orch")
        with sentinel._state_tracker._attempt_counts_lock:
            entry = sentinel._state_tracker._attempt_counts.get(key)
            stored_count = entry.count if entry else 0
        assert stored_count == 3, "Stored count should be 3 after 3 increments"

    def test_running_step_info_contains_attempt_number(self) -> None:
        """Test that RunningStepInfo is created with correct attempt_number."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[
                make_issue(key="TEST-1", summary="Issue 1", labels=["review"]),
            ],
            tag_client=tag_client,
        )
        # Use threading.Event for cross-thread signaling (test coordination)
        # Use asyncio.Event for async consistency within run_agent method
        should_unblock = threading.Event()

        class BlockingAgentClient(MockAgentClient):
            async def run_agent(self, *args: Any, **kwargs: Any) -> AgentRunResult:
                # Use asyncio.Event for async-consistent blocking within the async method
                blocking_event = asyncio.Event()

                async def wait_for_unblock() -> None:
                    # Poll the threading.Event in an async-friendly way
                    while not should_unblock.is_set():
                        await asyncio.sleep(0.01)
                    blocking_event.set()

                # Start the polling task and wait for the blocking event
                unblock_task = asyncio.create_task(wait_for_unblock())
                try:
                    await asyncio.wait_for(blocking_event.wait(), timeout=POLLING_MAX_WAIT)
                except TimeoutError:
                    pass
                finally:
                    unblock_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await unblock_task
                return await super().run_agent(*args, **kwargs)

        agent_client = BlockingAgentClient(responses=["SUCCESS"])
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")
        sentinel._execution_manager._thread_pool = ThreadPoolExecutor(max_workers=2)

        try:
            run_thread = threading.Thread(target=sentinel.run_once)
            run_thread.start()

            # Wait for the running step to be registered.
            # The running step is added in _submit_execution_tasks after the future
            # is created, which happens very quickly after run_once starts. We poll
            # with sufficient timeout to handle CI environments with high load.
            # Using POLLING_ITERATIONS * POLLING_INTERVAL = POLLING_MAX_WAIT provides ample margin.
            running_steps = []
            for _ in range(POLLING_ITERATIONS):
                running_steps = sentinel.get_running_steps()
                if len(running_steps) == 1:
                    break
                time.sleep(POLLING_INTERVAL)

            assert len(running_steps) == 1, (
                f"Should have one running step, got {len(running_steps)}. "
                "This may indicate a race condition or the task was not submitted."
            )
            step = running_steps[0]
            assert step.attempt_number == 2, "Attempt number should be 2 (second attempt)"
            assert step.issue_key == "TEST-1"
            assert step.orchestration_name == "test-orch"
        finally:
            should_unblock.set()
            run_thread.join(timeout=POLLING_MAX_WAIT)
            sentinel._execution_manager._thread_pool.shutdown(wait=True)
            sentinel._execution_manager._thread_pool = None

    def test_cleanup_stale_attempt_counts(self) -> None:
        """Test that stale attempt count entries are cleaned up based on TTL."""
        import time

        from sentinel.state_tracker import AttemptCountEntry

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        agent_factory = MockAgentClientFactory(agent_client)

        config = make_config(attempt_counts_ttl=1)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        current_time = time.monotonic()
        with sentinel._state_tracker._attempt_counts_lock:
            sentinel._state_tracker._attempt_counts[("OLD-1", "test-orch")] = AttemptCountEntry(
                count=5, last_access=current_time - 100
            )
            sentinel._state_tracker._attempt_counts[("RECENT-1", "test-orch")] = AttemptCountEntry(
                count=3, last_access=current_time - 0.5
            )

        assert len(sentinel._state_tracker._attempt_counts) == 2

        cleaned = sentinel._state_tracker.cleanup_stale_attempt_counts()

        assert cleaned == 1
        assert len(sentinel._state_tracker._attempt_counts) == 1
        assert ("OLD-1", "test-orch") not in sentinel._state_tracker._attempt_counts
        assert ("RECENT-1", "test-orch") in sentinel._state_tracker._attempt_counts

    def test_cleanup_attempt_counts_called_in_run_once(self) -> None:
        """Test that _cleanup_stale_attempt_counts is called during run_once."""

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config()
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        with patch.object(sentinel._state_tracker, "cleanup_stale_attempt_counts") as mock_cleanup:
            mock_cleanup.return_value = 0
            sentinel.run_once()
            mock_cleanup.assert_called_once()

    def test_attempt_count_entry_updates_last_access(self) -> None:
        """Test that accessing an attempt count updates its last_access time."""

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config()
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        time_before = time.monotonic()
        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")
        time_after = time.monotonic()

        key = ("TEST-1", "test-orch")
        with sentinel._state_tracker._attempt_counts_lock:
            entry = sentinel._state_tracker._attempt_counts[key]

        assert entry.count == 1
        assert entry.last_access >= time_before
        assert entry.last_access <= time_after

        time.sleep(0.01)
        time_before_second = time.monotonic()

        sentinel._state_tracker.get_and_increment_attempt_count("TEST-1", "test-orch")

        with sentinel._state_tracker._attempt_counts_lock:
            entry = sentinel._state_tracker._attempt_counts[key]

        assert entry.count == 2
        assert entry.last_access >= time_before_second

    def test_cleanup_logs_debug_when_no_stale_entries(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that cleanup logs at debug level when no stale entries are found."""

        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        agent_factory = MockAgentClientFactory(agent_client)
        config = make_config(attempt_counts_ttl=3600)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_factory,
            tag_client=tag_client,
        )

        assert len(sentinel._state_tracker._attempt_counts) == 0

        with caplog.at_level(logging.DEBUG, logger="sentinel.state_tracker"):
            cleaned = sentinel._state_tracker.cleanup_stale_attempt_counts()

        assert cleaned == 0

        debug_messages = [
            r.message
            for r in caplog.records
            if r.levelno == logging.DEBUG and r.name == "sentinel.state_tracker"
        ]
        assert any(
            "no stale entries found" in msg for msg in debug_messages
        ), f"Expected debug log about no stale entries, got: {debug_messages}"

        ttl_value = config.execution.attempt_counts_ttl
        all_debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(
            f"TTL: {ttl_value}" in msg for msg in all_debug_messages
        ), f"Expected TTL value ({ttl_value}s) in debug message, got: {all_debug_messages}"
