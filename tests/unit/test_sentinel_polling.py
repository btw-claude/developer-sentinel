"""Tests for Sentinel polling and core functionality."""

import logging
import signal
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest

from sentinel.main import Sentinel, parse_args, setup_logging

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockJiraClient,
    MockAgentClient,
    MockTagClient,
    SignalHandler,
    make_config,
    make_orchestration,
)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_args(self) -> None:
        args = parse_args([])
        assert args.config_dir is None
        assert args.once is False
        assert args.interval is None
        assert args.log_level is None

    def test_config_dir(self) -> None:
        args = parse_args(["--config-dir", "/path/to/orchestrations"])
        assert args.config_dir == Path("/path/to/orchestrations")

    def test_once_flag(self) -> None:
        args = parse_args(["--once"])
        assert args.once is True

    def test_interval(self) -> None:
        args = parse_args(["--interval", "30"])
        assert args.interval == 30

    def test_log_level(self) -> None:
        args = parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_env_file(self) -> None:
        args = parse_args(["--env-file", "/path/to/.env"])
        assert args.env_file == Path("/path/to/.env")

    def test_all_args(self) -> None:
        args = parse_args(
            [
                "--config-dir",
                "/configs",
                "--once",
                "--interval",
                "120",
                "--log-level",
                "WARNING",
                "--env-file",
                "/custom/.env",
            ]
        )
        assert args.config_dir == Path("/configs")
        assert args.once is True
        assert args.interval == 120
        assert args.log_level == "WARNING"
        assert args.env_file == Path("/custom/.env")


class TestSetupLogging:
    """Tests for setup_logging function (re-exported from sentinel.logging)."""

    def test_sets_log_level(self) -> None:
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_handles_invalid_level(self) -> None:
        setup_logging("INVALID")
        root = logging.getLogger()
        assert root.level == logging.INFO  # default


class TestSentinelRunOnce:
    """Tests for Sentinel.run_once method."""

    def test_polls_and_executes(self) -> None:
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Test issue", "labels": ["review"]}},
            ]
        )
        agent_client = MockAgentClient(responses=["SUCCESS: Completed"])
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results, submitted_count = sentinel.run_once()

        assert len(results) == 1
        assert results[0].succeeded is True
        assert len(jira_client.search_calls) == 1
        assert len(agent_client.calls) == 1

    def test_no_issues_returns_empty(self) -> None:
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results, submitted_count = sentinel.run_once()

        assert len(results) == 0
        assert len(agent_client.calls) == 0

    def test_handles_multiple_orchestrations(self) -> None:
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
            ]
        )
        agent_client = MockAgentClient(responses=["SUCCESS"])
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [
            make_orchestration(name="orch1", tags=["review"]),
            make_orchestration(name="orch2", tags=["review"]),
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results, submitted_count = sentinel.run_once()

        # Issue matches both orchestrations
        assert len(results) == 2

    def test_respects_shutdown_request(self) -> None:
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
            ]
        )
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Request shutdown before running
        sentinel.request_shutdown()
        results, submitted_count = sentinel.run_once()

        # Should return early without executing all issues
        assert len(results) == 0


class TestSentinelEagerPolling:
    """Tests for Sentinel eager polling feature."""

    def test_run_once_returns_submitted_count(self) -> None:
        """Test that run_once returns the number of submitted tasks."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS"])
        config = make_config()
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results, submitted_count = sentinel.run_once()

        # Without thread pool, execution is synchronous
        assert len(results) == 2
        # submitted_count represents tasks submitted (0 in sync mode)
        assert isinstance(submitted_count, int)

    def test_run_once_returns_zero_submitted_when_no_work(self) -> None:
        """Test that run_once returns 0 submitted when no issues found."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results, submitted_count = sentinel.run_once()

        assert len(results) == 0
        assert submitted_count == 0

    def test_run_once_returns_zero_submitted_when_slots_busy(self) -> None:
        """Test that run_once returns 0 submitted when all slots are busy."""
        from concurrent.futures import ThreadPoolExecutor

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[{"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}}],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS"])
        config = make_config(max_concurrent_executions=1)
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Manually set up the thread pool with a busy slot
        sentinel._thread_pool = ThreadPoolExecutor(max_workers=1)
        block_event = threading.Event()
        future = sentinel._thread_pool.submit(lambda: block_event.wait(timeout=5))
        sentinel._active_futures = [future]

        # run_once should return 0 submitted since slot is busy
        results, submitted_count = sentinel.run_once()

        assert len(results) == 0
        assert submitted_count == 0

        # Clean up
        block_event.set()
        future.result()
        sentinel._thread_pool.shutdown(wait=True)
        sentinel._thread_pool = None

    def test_completion_driven_polling_waits_for_task_completion(self) -> None:
        """Integration test: verify polling waits for task completion.

        This test verifies the completion-driven polling behavior:
        1. When tasks are submitted, the sentinel waits for at least one to complete
        2. After a task completes, the sentinel immediately polls for more work
        3. Only when no work is found and no tasks are pending does it sleep

        The test uses the full run() method with mocked sleep and signal.
        """
        from concurrent.futures import Future
        from unittest.mock import MagicMock

        # Save original sleep function before patching
        real_sleep = time.sleep

        # Track poll cycles and waits
        poll_cycles = 0
        wait_calls: list[int] = []  # Track when wait() is called with pending future count
        max_poll_cycles = 6

        class ControlledWorkJiraClient(MockJiraClient):
            """Jira client that returns work for first few polls, then stops."""

            def __init__(self, tag_client: MockTagClient) -> None:
                super().__init__(issues=[], tag_client=tag_client)
                self.issue_counter = 0

            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_cycles
                self.search_calls.append((jql, max_results))
                poll_cycles += 1

                # Stop returning work after max_poll_cycles
                if poll_cycles > max_poll_cycles:
                    return []

                # Return a new issue to trigger task submission
                self.issue_counter += 1
                return [
                    {
                        "key": f"TEST-{self.issue_counter}",
                        "fields": {"summary": f"Issue {self.issue_counter}", "labels": ["review"]},
                    }
                ]

        tag_client = MockTagClient()
        jira_client = ControlledWorkJiraClient(tag_client=tag_client)
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])

        config = make_config(
            poll_interval=2,
            max_concurrent_executions=1,
        )
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        def mock_sleep(seconds: float) -> None:
            real_sleep(0.001)

        def mock_signal(signum: int, handler: SignalHandler) -> None:
            """Mock signal.signal to avoid threading issues."""
            pass

        # Patch the wait function to track calls
        original_wait = __import__("concurrent.futures").futures.wait

        def mock_wait(
            futures: set[Future[object]],
            timeout: float | None = None,
            return_when: str = "ALL_COMPLETED",
        ) -> tuple[set[Future[object]], set[Future[object]]]:
            wait_calls.append(len(futures) if futures else 0)
            return original_wait(futures, timeout=timeout, return_when=return_when)

        sentinel_exception: Exception | None = None

        def run_sentinel() -> None:
            nonlocal sentinel_exception
            try:
                with patch("time.sleep", side_effect=mock_sleep), \
                     patch("signal.signal", side_effect=mock_signal), \
                     patch("sentinel.main.wait", side_effect=mock_wait):
                    sentinel.run()
            except Exception as e:
                sentinel_exception = e

        sentinel_thread = threading.Thread(target=run_sentinel, daemon=True)
        sentinel_thread.start()

        # Wait for enough poll cycles
        for _ in range(50):
            if poll_cycles >= max_poll_cycles:
                break
            real_sleep(0.1)

        sentinel.request_shutdown()
        sentinel_thread.join(timeout=5)

        assert sentinel_exception is None, f"Sentinel raised exception: {sentinel_exception}"
        assert poll_cycles >= 4, f"Expected at least 4 poll cycles, got {poll_cycles}"

        # Verify that wait() was called when there were pending tasks
        # The wait_calls list should show non-zero values when tasks were pending
        assert any(w > 0 for w in wait_calls), (
            f"Expected wait() to be called with pending futures, "
            f"but wait_calls were: {wait_calls}"
        )

    def test_polling_sleeps_when_no_work_found(self) -> None:
        """Integration test: verify polling sleeps when no work is found.

        This verifies that when a poll cycle returns no work and there are
        no pending tasks, the sentinel sleeps for poll_interval before polling again.
        """
        real_sleep = time.sleep
        poll_cycles = 0
        sleep_intervals: list[float] = []

        class EmptyJiraClient(MockJiraClient):
            """Jira client that never returns work."""

            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_cycles
                self.search_calls.append((jql, max_results))
                poll_cycles += 1
                return []  # No work

        tag_client = MockTagClient()
        jira_client = EmptyJiraClient(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient()

        config = make_config(
            poll_interval=2,
            max_concurrent_executions=1,
        )
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        def mock_sleep(seconds: float) -> None:
            sleep_intervals.append(seconds)
            real_sleep(0.001)

        def mock_signal(signum: int, handler: SignalHandler) -> None:
            pass

        sentinel_exception: Exception | None = None

        def run_sentinel() -> None:
            nonlocal sentinel_exception
            try:
                with patch("time.sleep", side_effect=mock_sleep), \
                     patch("signal.signal", side_effect=mock_signal):
                    sentinel.run()
            except Exception as e:
                sentinel_exception = e

        sentinel_thread = threading.Thread(target=run_sentinel, daemon=True)
        sentinel_thread.start()

        # Wait for a few poll cycles
        for _ in range(30):
            if poll_cycles >= 3:
                break
            real_sleep(0.1)

        sentinel.request_shutdown()
        sentinel_thread.join(timeout=5)

        assert sentinel_exception is None, f"Sentinel raised exception: {sentinel_exception}"
        assert poll_cycles >= 2, f"Expected at least 2 poll cycles, got {poll_cycles}"

        # When no work is found, the sentinel should sleep for poll_interval (in 1-second chunks)
        # Count total 1-second sleeps which indicates normal poll intervals
        one_second_sleeps = sum(1 for s in sleep_intervals if s == 1)
        assert one_second_sleeps >= config.poll_interval, (
            f"Expected at least {config.poll_interval} 1-second sleeps (one full poll interval), "
            f"got {one_second_sleeps}. Sleep intervals: {sleep_intervals}"
        )


class TestSentinelRun:
    """Tests for Sentinel.run method."""

    def test_runs_until_shutdown(self) -> None:
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(poll_interval=1)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Request shutdown after a short delay
        def shutdown_after_delay() -> None:
            time.sleep(0.1)
            sentinel.request_shutdown()

        shutdown_thread = threading.Thread(target=shutdown_after_delay)
        shutdown_thread.start()

        sentinel.run()

        shutdown_thread.join()
        # Should have completed at least one cycle
        assert len(jira_client.search_calls) >= 1

    def test_handles_errors_gracefully(self) -> None:
        from unittest.mock import MagicMock

        jira_client = MockJiraClient()
        jira_client.search_issues = MagicMock(side_effect=Exception("API error"))
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(poll_interval=1)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Request immediate shutdown
        sentinel.request_shutdown()
        # Should not raise despite the error
        sentinel.run()
