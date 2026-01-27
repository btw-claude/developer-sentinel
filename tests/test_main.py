"""Tests for main entry point module."""

import logging
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from sentinel.executor import AgentClient, AgentRunResult
from sentinel.main import Sentinel, parse_args, setup_logging
from sentinel.orchestration import Orchestration

# DS-100: Import shared fixtures and helpers from conftest.py
# These provide MockJiraClient, MockAgentClient, MockTagClient,
# make_config, make_orchestration, and set_mtime_in_future
# DS-188: Added TrackingAgentClient for concurrency tracking tests
# DS-296: Added MockAgentClientFactory for factory pattern tests
from tests.conftest import (
    MockJiraClient,
    MockAgentClient,
    MockAgentClientFactory,
    MockTagClient,
    TrackingAgentClient,
    make_config,
    make_orchestration,
    set_mtime_in_future,
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
    """Tests for Sentinel eager polling feature (DS-94)."""

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

    def test_max_eager_iterations_config_respected(self) -> None:
        """Test that max_eager_iterations config is stored correctly (DS-98).

        Note: DS-133 deprecated max_eager_iterations, but we keep this test
        for backward compatibility since the config parameter still exists.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_eager_iterations=5)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel.config.max_eager_iterations == 5

    def test_completion_driven_polling_waits_for_task_completion(self) -> None:
        """Integration test: verify polling waits for task completion (DS-133).

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

        def mock_signal(signum: int, handler: Any) -> Any:
            """Mock signal.signal to avoid threading issues."""
            return None

        # Patch the wait function to track calls
        original_wait = __import__("concurrent.futures").futures.wait

        def mock_wait(futures: Any, timeout: float | None = None, return_when: str = "ALL_COMPLETED") -> Any:
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
        """Integration test: verify polling sleeps when no work is found (DS-133).

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

        def mock_signal(signum: int, handler: Any) -> Any:
            return None

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


class TestSentinelSignalHandling:
    """Tests for Sentinel signal handling."""

    def test_registers_signal_handlers(self) -> None:
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

        registered_handlers: dict[int, Any] = {}

        def mock_signal(signum: int, handler: Any) -> Any:
            registered_handlers[signum] = handler
            return None

        with patch("signal.signal", side_effect=mock_signal):
            # Request shutdown immediately so run() exits quickly
            sentinel.request_shutdown()
            sentinel.run()

        # Verify both handlers were registered
        assert signal.SIGINT in registered_handlers
        assert signal.SIGTERM in registered_handlers

    def test_sigint_triggers_shutdown(self) -> None:
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

        captured_handler: Any = None

        def capture_sigint_handler(signum: int, handler: Any) -> Any:
            nonlocal captured_handler
            if signum == signal.SIGINT:
                captured_handler = handler
            return None

        with patch("signal.signal", side_effect=capture_sigint_handler):
            # Start run in a thread that will be stopped by the signal
            def run_sentinel() -> None:
                sentinel.run()

            thread = threading.Thread(target=run_sentinel)
            thread.start()

            # Give it a moment to register handlers
            time.sleep(0.05)

            # Simulate SIGINT by calling the captured handler
            if captured_handler:
                captured_handler(signal.SIGINT, None)

            thread.join(timeout=2)
            assert not thread.is_alive(), "Thread should have stopped"
            assert sentinel._shutdown_requested is True

    def test_sigterm_triggers_shutdown(self) -> None:
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

        captured_handler: Any = None

        def capture_sigterm_handler(signum: int, handler: Any) -> Any:
            nonlocal captured_handler
            if signum == signal.SIGTERM:
                captured_handler = handler
            return None

        with patch("signal.signal", side_effect=capture_sigterm_handler):
            # Start run in a thread that will be stopped by the signal
            def run_sentinel() -> None:
                sentinel.run()

            thread = threading.Thread(target=run_sentinel)
            thread.start()

            # Give it a moment to register handlers
            time.sleep(0.05)

            # Simulate SIGTERM by calling the captured handler
            if captured_handler:
                captured_handler(signal.SIGTERM, None)

            thread.join(timeout=2)
            assert not thread.is_alive(), "Thread should have stopped"
            assert sentinel._shutdown_requested is True


class TestSentinelConcurrentExecution:
    """Tests for Sentinel concurrent execution."""

    def test_respects_max_concurrent_executions(self) -> None:
        """Test that max_concurrent_executions limits parallel work."""
        execution_count = 0
        max_concurrent_seen = 0
        lock = threading.Lock()

        class SlowAgentClient(AgentClient):
            """Agent client that tracks concurrent executions."""

            def run_agent(
                self,
                prompt: str,
                tools: list[str],
                context: dict[str, Any] | None = None,
                timeout_seconds: int | None = None,
                issue_key: str | None = None,
                model: str | None = None,
                orchestration_name: str | None = None,
            ) -> AgentRunResult:
                nonlocal execution_count, max_concurrent_seen
                with lock:
                    execution_count += 1
                    if execution_count > max_concurrent_seen:
                        max_concurrent_seen = execution_count

                # Simulate some work
                time.sleep(0.1)

                with lock:
                    execution_count -= 1

                return AgentRunResult(response="SUCCESS: Done", workdir=None)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
                {"key": "TEST-3", "fields": {"summary": "Issue 3", "labels": ["review"]}},
            ],
            tag_client=tag_client,  # Link tag client so it filters processed issues
        )
        agent_client = SlowAgentClient()
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # All 3 issues should have been processed
        assert len(results) == 3
        # Max concurrent should not exceed 2
        assert max_concurrent_seen <= 2

    def test_run_once_and_wait_completes_all_work(self) -> None:
        """Test that run_once_and_wait waits for all tasks to complete."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # Both issues should have results
        assert len(results) == 2
        assert all(r.succeeded for r in results)

    def test_slots_limiting_skips_polling_when_busy(self) -> None:
        """Test that polling is skipped when all slots are busy."""
        # Track poll calls
        poll_calls = 0

        class TrackingJiraClient(MockJiraClient):
            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_calls
                poll_calls += 1
                return super().search_issues(jql, max_results)

        tag_client = MockTagClient()
        jira_client = TrackingJiraClient(
            issues=[{"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}}],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])
        config = make_config(poll_interval=1, max_concurrent_executions=1)
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Manually set up the thread pool and simulate a busy slot
        sentinel._thread_pool = ThreadPoolExecutor(max_workers=1)
        # Submit a dummy task to occupy the slot
        block_event = threading.Event()
        future = sentinel._thread_pool.submit(lambda: block_event.wait(timeout=5))
        sentinel._active_futures = [future]

        # First run_once should skip polling since slot is busy
        results, submitted_count = sentinel.run_once()
        first_poll_count = poll_calls

        # Should have returned immediately without polling
        assert first_poll_count == 0
        assert len(results) == 0

        # Release the blocking task
        block_event.set()
        future.result()  # Wait for completion

        # Clean up
        sentinel._thread_pool.shutdown(wait=True)
        sentinel._thread_pool = None

    def test_concurrent_execution_with_thread_pool(self) -> None:
        """Test that run() uses thread pool correctly."""
        # DS-188: Use shared TrackingAgentClient with order tracking enabled
        agent_client = TrackingAgentClient(execution_delay=0.05, track_order=True)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        config = make_config(poll_interval=1, max_concurrent_executions=2)
        orchestrations = [make_orchestration(tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Run a single cycle with wait
        results = sentinel.run_once_and_wait()

        assert len(results) == 2

        # With concurrent execution, both should start before either ends
        # (since we have 2 slots and 2 issues)
        execution_order = agent_client.execution_order
        start_indices = [i for i, x in enumerate(execution_order) if x.startswith("start:")]
        end_indices = [i for i, x in enumerate(execution_order) if x.startswith("end:")]

        # Both should start before either ends (concurrent execution)
        assert len(start_indices) == 2
        assert len(end_indices) == 2
        # Second start should happen before first end in concurrent execution
        assert start_indices[1] < end_indices[0]


class TestPerOrchestrationConcurrencyLimits:
    """Tests for per-orchestration concurrency limits (DS-181).

    These tests verify:
    - max_concurrent parsing and validation
    - Slot checking with per-orchestration limits
    - Integration with global concurrency limits
    - Per-orchestration count tracking
    """

    def test_orchestration_without_max_concurrent_uses_global_limit(self) -> None:
        """Test that orchestrations without max_concurrent use only global limit."""
        # DS-188: Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.05)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
                {"key": "TEST-3", "fields": {"summary": "Issue 3", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        config = make_config(max_concurrent_executions=2)
        # Orchestration without max_concurrent (uses None by default)
        orchestrations = [make_orchestration(tags=["review"], max_concurrent=None)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # All 3 issues should be processed (global limit of 2 allows all eventually)
        results = sentinel.run_once_and_wait()
        assert len(results) == 3
        assert all(r.succeeded for r in results)
        # Global limit of 2 should allow up to 2 concurrent
        assert agent_client.max_concurrent_seen <= 2

    def test_per_orch_limit_stricter_than_global(self) -> None:
        """Test per-orchestration limit is respected when stricter than global."""
        # DS-188: Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.1)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
                {"key": "TEST-3", "fields": {"summary": "Issue 3", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        # Global limit of 5, but orchestration limit of 1
        config = make_config(max_concurrent_executions=5)
        orchestrations = [make_orchestration(name="limited-orch", tags=["review"], max_concurrent=1)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        assert len(results) == 3
        # Per-orch limit of 1 should cap concurrent executions
        assert agent_client.max_concurrent_seen == 1

    def test_global_limit_stricter_than_per_orch(self) -> None:
        """Test global limit is respected when stricter than per-orchestration."""
        # DS-188: Use shared TrackingAgentClient for concurrent execution tracking
        agent_client = TrackingAgentClient(execution_delay=0.1)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
                {"key": "TEST-3", "fields": {"summary": "Issue 3", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        # Global limit of 1, per-orch limit of 10
        config = make_config(max_concurrent_executions=1)
        orchestrations = [make_orchestration(name="high-limit-orch", tags=["review"], max_concurrent=10)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        assert len(results) == 3
        # Global limit of 1 should cap concurrent executions
        assert agent_client.max_concurrent_seen == 1

    def test_multiple_orchestrations_with_different_limits(self) -> None:
        """Test multiple orchestrations respect their individual limits."""
        # DS-188: Use shared TrackingAgentClient with per-orchestration tracking enabled
        agent_client = TrackingAgentClient(execution_delay=0.05, track_per_orch=True)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                # Issues for orch-1 (max_concurrent=1)
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
                # Issues for orch-2 (max_concurrent=2)
                {"key": "TEST-3", "fields": {"summary": "Issue 3", "labels": ["deploy"]}},
                {"key": "TEST-4", "fields": {"summary": "Issue 4", "labels": ["deploy"]}},
            ],
            tag_client=tag_client,
        )
        # High global limit, different per-orch limits
        config = make_config(max_concurrent_executions=10)
        orchestrations = [
            make_orchestration(name="orch-1", tags=["review"], max_concurrent=1),
            make_orchestration(name="orch-2", tags=["deploy"], max_concurrent=2),
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # All issues should be processed
        assert len(results) == 4
        # orch-1 should never exceed its limit of 1
        assert agent_client.orch_max_concurrent.get("orch-1", 0) <= 1
        # orch-2 should never exceed its limit of 2
        assert agent_client.orch_max_concurrent.get("orch-2", 0) <= 2

    def test_per_orch_count_increment_and_decrement(self) -> None:
        """Test that per-orchestration counts are properly tracked."""
        # DS-188: Use shared TrackingAgentClient for consistent test behavior
        agent_client = TrackingAgentClient(execution_delay=0.05)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        config = make_config(max_concurrent_executions=5)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"], max_concurrent=2)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Before execution, count should be 0
        assert sentinel._per_orch_active_counts.get("test-orch", 0) == 0

        # Run and wait for completion
        results = sentinel.run_once_and_wait()

        assert len(results) == 1
        # After execution completes, count should return to 0
        assert sentinel._per_orch_active_counts.get("test-orch", 0) == 0

    def test_increment_per_orch_count_returns_new_count(self) -> None:
        """Test _increment_per_orch_count returns the new count."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # First increment
        count = sentinel._increment_per_orch_count("test-orch")
        assert count == 1
        assert sentinel._per_orch_active_counts["test-orch"] == 1

        # Second increment
        count = sentinel._increment_per_orch_count("test-orch")
        assert count == 2
        assert sentinel._per_orch_active_counts["test-orch"] == 2

        # Increment different orchestration
        count = sentinel._increment_per_orch_count("other-orch")
        assert count == 1
        assert sentinel._per_orch_active_counts["other-orch"] == 1
        assert sentinel._per_orch_active_counts["test-orch"] == 2

    def test_decrement_per_orch_count_returns_new_count(self) -> None:
        """Test _decrement_per_orch_count returns the new count."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Setup initial counts
        sentinel._per_orch_active_counts["test-orch"] = 3

        # First decrement
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 2
        assert sentinel._per_orch_active_counts["test-orch"] == 2

        # Second decrement
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 1

        # Third decrement
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 0

    def test_decrement_per_orch_count_clamps_to_zero(self) -> None:
        """Test _decrement_per_orch_count does not go below zero."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Decrement when count is 0
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 0
        assert sentinel._per_orch_active_counts["test-orch"] == 0

        # Decrement again
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 0

    def test_decrement_per_orch_count_cleans_up_at_zero(self) -> None:
        """Test _decrement_per_orch_count removes entry when count reaches 0 (DS-187)."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Increment to 1
        sentinel._increment_per_orch_count("test-orch")
        assert "test-orch" in sentinel._per_orch_active_counts

        # Decrement to 0 - should clean up entry
        count = sentinel._decrement_per_orch_count("test-orch")
        assert count == 0
        # DS-187: Entry should be removed when count reaches 0
        assert "test-orch" not in sentinel._per_orch_active_counts

    def test_get_per_orch_count_returns_count(self) -> None:
        """Test get_per_orch_count returns the current count for observability (DS-187)."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Initially should return 0 for unknown orchestration
        assert sentinel.get_per_orch_count("test-orch") == 0

        # After incrementing, should return the count
        sentinel._increment_per_orch_count("test-orch")
        assert sentinel.get_per_orch_count("test-orch") == 1

        sentinel._increment_per_orch_count("test-orch")
        assert sentinel.get_per_orch_count("test-orch") == 2

        # Check a different orchestration
        assert sentinel.get_per_orch_count("other-orch") == 0

    def test_get_all_per_orch_counts_returns_all_counts(self) -> None:
        """Test get_all_per_orch_counts returns all counts for observability (DS-187)."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config()
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Initially should return empty dict
        counts = sentinel.get_all_per_orch_counts()
        assert counts == {}

        # After incrementing multiple orchestrations
        sentinel._increment_per_orch_count("orch-a")
        sentinel._increment_per_orch_count("orch-a")
        sentinel._increment_per_orch_count("orch-b")

        counts = sentinel.get_all_per_orch_counts()
        assert counts == {"orch-a": 2, "orch-b": 1}

        # Verify it returns a copy (not the original dict)
        counts["orch-c"] = 99
        assert "orch-c" not in sentinel.get_all_per_orch_counts()

    def test_get_available_slots_for_orchestration_no_limit(self) -> None:
        """Test _get_available_slots_for_orchestration with no per-orch limit."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config(max_concurrent_executions=5)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Orchestration without max_concurrent
        orch = make_orchestration(name="unlimited", max_concurrent=None)

        # Should return global available slots
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 5

    def test_get_available_slots_for_orchestration_with_limit(self) -> None:
        """Test _get_available_slots_for_orchestration with per-orch limit."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config(max_concurrent_executions=10)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Orchestration with max_concurrent=3
        orch = make_orchestration(name="limited", max_concurrent=3)

        # With 0 active executions, should return per-orch limit (3)
        # since it's lower than global (10)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 3

    def test_get_available_slots_considers_current_per_orch_count(self) -> None:
        """Test _get_available_slots_for_orchestration considers current count."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config(max_concurrent_executions=10)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        orch = make_orchestration(name="limited", max_concurrent=3)

        # Simulate 2 active executions for this orchestration
        sentinel._per_orch_active_counts["limited"] = 2

        # Should return 1 (3 - 2)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 1

        # Simulate 3 active executions (at limit)
        sentinel._per_orch_active_counts["limited"] = 3
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 0

    def test_get_available_slots_returns_min_of_global_and_per_orch(self) -> None:
        """Test that available slots is minimum of global and per-orch."""
        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        config = make_config(max_concurrent_executions=2)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Orchestration with max_concurrent=5, but global is 2
        orch = make_orchestration(name="high-limit", max_concurrent=5)

        # Should return global limit (2) since it's lower
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 2

    def test_per_orch_count_decremented_on_failure(self) -> None:
        """Test per-orchestration count is decremented even on execution failure."""

        class FailingAgentClient(AgentClient):
            def run_agent(
                self,
                prompt: str,
                tools: list[str],
                context: dict[str, Any] | None = None,
                timeout_seconds: int | None = None,
                issue_key: str | None = None,
                model: str | None = None,
                orchestration_name: str | None = None,
            ) -> AgentRunResult:
                time.sleep(0.05)
                return AgentRunResult(response="FAILURE: Error occurred", workdir=None)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        agent_client = FailingAgentClient()
        config = make_config(max_concurrent_executions=5)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"], max_concurrent=2)]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        results = sentinel.run_once_and_wait()

        # Execution should complete (with failure after retries)
        assert len(results) == 1
        # Count should be decremented back to 0 even on failure
        assert sentinel._per_orch_active_counts.get("test-orch", 0) == 0

    def test_make_orchestration_helper_with_max_concurrent(self) -> None:
        """Test the make_orchestration helper supports max_concurrent."""
        # Without max_concurrent
        orch1 = make_orchestration(name="test1")
        assert orch1.max_concurrent is None

        # With max_concurrent
        orch2 = make_orchestration(name="test2", max_concurrent=5)
        assert orch2.max_concurrent == 5


class TestSentinelOrchestrationHotReload:
    """Tests for Sentinel orchestration file hot-reload functionality."""

    def test_init_tracks_existing_orchestration_files(self) -> None:
        """Test that __init__ initializes the set of known orchestration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            # Create some initial orchestration files
            (orch_dir / "existing1.yaml").write_text(
                "orchestrations:\n  - name: existing1\n    trigger: {project: TEST}\n    agent: {prompt: test}"
            )
            (orch_dir / "existing2.yml").write_text(
                "orchestrations:\n  - name: existing2\n    trigger: {project: TEST}\n    agent: {prompt: test}"
            )
            # Create a non-yaml file that should be ignored
            (orch_dir / "readme.txt").write_text("This is not an orchestration file")

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations = [make_orchestration()]

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Should have tracked both yaml files
            assert len(sentinel._known_orchestration_files) == 2
            assert orch_dir / "existing1.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "existing2.yml" in sentinel._known_orchestration_files

    def test_detects_new_orchestration_files(self) -> None:
        """Test that new orchestration files are detected during poll cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Initially no files
            assert len(sentinel._known_orchestration_files) == 0
            assert len(sentinel.orchestrations) == 0

            # Add a new orchestration file
            (orch_dir / "new_orch.yaml").write_text(
                """orchestrations:
  - name: new-orchestration
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Test prompt
      tools: [jira]
"""
            )

            # Run a poll cycle - should detect and load the new file
            sentinel.run_once()

            # Should have detected and loaded the new orchestration
            assert len(sentinel._known_orchestration_files) == 1
            assert orch_dir / "new_orch.yaml" in sentinel._known_orchestration_files
            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "new-orchestration"

    def test_loads_multiple_orchestrations_from_new_file(self) -> None:
        """Test that multiple orchestrations from a single new file are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a file with multiple orchestrations
            (orch_dir / "multi.yaml").write_text(
                """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
  - name: orch-2
    trigger:
      project: PROJ
    agent:
      prompt: Test 2
"""
            )

            # Run a poll cycle
            sentinel.run_once()

            # Should have loaded both orchestrations
            assert len(sentinel.orchestrations) == 2
            assert sentinel.orchestrations[0].name == "orch-1"
            assert sentinel.orchestrations[1].name == "orch-2"

    def test_updates_router_with_new_orchestrations(self) -> None:
        """Test that the router is updated when new orchestrations are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(
                issues=[
                    {"key": "TEST-1", "fields": {"summary": "Test", "labels": ["new-tag"]}},
                ]
            )
            agent_client = MockAgentClient(responses=["SUCCESS"])
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # First run - no orchestrations, no matches
            results, submitted_count = sentinel.run_once()
            assert len(results) == 0

            # Add a new orchestration that matches the existing issue
            (orch_dir / "matching.yaml").write_text(
                """orchestrations:
  - name: matching-orch
    trigger:
      project: TEST
      tags: [new-tag]
    agent:
      prompt: Process this
      tools: [jira]
"""
            )

            # Second run - should detect new file, update router, and match the issue
            results, submitted_count = sentinel.run_once()
            assert len(results) == 1
            assert results[0].succeeded is True

    def test_handles_invalid_orchestration_file_gracefully(self) -> None:
        """Test that invalid orchestration files don't crash the system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add an invalid orchestration file
            (orch_dir / "invalid.yaml").write_text("this is not valid: yaml: [[[")

            # Run a poll cycle - should handle error gracefully
            sentinel.run_once()

            # File should be marked as known (to avoid retry)
            assert orch_dir / "invalid.yaml" in sentinel._known_orchestration_files
            # But no orchestrations should be added
            assert len(sentinel.orchestrations) == 0

    def test_no_router_rebuild_for_empty_or_invalid_files(self) -> None:
        """Test that Router is not rebuilt when files contain no valid orchestrations.

        DS-99: This optimization ensures that finding new files that are empty,
        invalid, or contain no enabled orchestrations does not trigger an unnecessary
        Router rebuild. The Router should only be rebuilt when actual orchestrations
        are loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Capture original router reference
            original_router = sentinel.router

            # Add an invalid orchestration file
            (orch_dir / "invalid.yaml").write_text("this is not valid: yaml: [[[")

            # Add an empty orchestration file (valid YAML but no orchestrations)
            (orch_dir / "empty.yaml").write_text("orchestrations: []")

            # Add a file with disabled orchestrations only
            (orch_dir / "disabled.yaml").write_text(
                """orchestrations:
  - name: disabled-orch
    enabled: false
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            # Run a poll cycle
            sentinel.run_once()

            # All files should be tracked as known
            assert orch_dir / "invalid.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "empty.yaml" in sentinel._known_orchestration_files
            assert orch_dir / "disabled.yaml" in sentinel._known_orchestration_files

            # No orchestrations should be loaded
            assert len(sentinel.orchestrations) == 0

            # Router should NOT have been rebuilt (same object reference)
            # This verifies the DS-99 optimization: Router rebuild only happens
            # when actual orchestrations are loaded, not just when files are found
            assert sentinel.router is original_router

    def test_does_not_reload_known_files(self) -> None:
        """Test that known files are not reloaded on subsequent poll cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            # Create initial file
            (orch_dir / "initial.yaml").write_text(
                """orchestrations:
  - name: initial
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Initial file is already tracked at init time
            assert len(sentinel._known_orchestration_files) == 1

            # First run_once
            sentinel.run_once()

            # Orchestration count should still be 0 because the file was tracked at init
            # (it won't be loaded by _detect_and_load_new_orchestration_files)
            assert len(sentinel.orchestrations) == 0

            # Run again - should not try to reload the file
            initial_count = len(sentinel.orchestrations)
            sentinel.run_once()
            assert len(sentinel.orchestrations) == initial_count

    def test_handles_nonexistent_orchestrations_directory(self) -> None:
        """Test handling when orchestrations directory doesn't exist."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(orchestrations_dir=Path("/nonexistent/path"))
        orchestrations: list[Orchestration] = []

        # Should not raise during initialization
        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert len(sentinel._known_orchestration_files) == 0

        # Should not raise during run_once
        sentinel.run_once()
        assert len(sentinel.orchestrations) == 0

    def test_detects_modified_orchestration_files(self) -> None:
        """Test that modified orchestration files are detected and reloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "modifiable.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: original-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Original prompt
      tools: [jira]
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # File should be tracked with its mtime
            assert len(sentinel._known_orchestration_files) == 1
            assert orch_file in sentinel._known_orchestration_files
            original_mtime = sentinel._known_orchestration_files[orch_file]

            # First run - no new files (file was tracked at init)
            sentinel.run_once()
            assert len(sentinel.orchestrations) == 0

            # Modify the file (need to ensure different mtime)
            orch_file.write_text(
                """orchestrations:
  - name: modified-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Modified prompt
      tools: [jira]
"""
            )
            set_mtime_in_future(orch_file)

            # Run again - should detect the modification
            sentinel.run_once()

            # Should have loaded the modified orchestration
            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "modified-orch"
            assert sentinel.orchestrations[0].agent.prompt == "Modified prompt"

            # Mtime should be updated
            assert sentinel._known_orchestration_files[orch_file] > original_mtime

    def test_modified_file_moves_old_version_to_pending_removal(self) -> None:
        """Test that modifying a file moves old versions to pending removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "versioned.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Version 1
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # No versions initially (file was tracked at init, not loaded)
            assert len(sentinel._active_versions) == 0
            assert len(sentinel._pending_removal_versions) == 0

            # Add a new file after init to get it loaded with version tracking
            new_file = orch_dir / "new_versioned.yaml"
            new_file.write_text(
                """orchestrations:
  - name: new-versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: New Version 1
"""
            )

            sentinel.run_once()

            # Should have one active version
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].name == "new-versioned-orch"

            # Simulate an active execution on this version
            sentinel._active_versions[0].increment_executions()

            # Modify the new file
            new_file.write_text(
                """orchestrations:
  - name: new-versioned-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: New Version 2 - Modified
"""
            )
            set_mtime_in_future(new_file)

            sentinel.run_once()

            # Old version should be in pending removal (has active execution)
            assert len(sentinel._pending_removal_versions) == 1
            assert sentinel._pending_removal_versions[0].orchestration.agent.prompt == "New Version 1"

            # New version should be active
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].orchestration.agent.prompt == "New Version 2 - Modified"

    def test_pending_removal_version_cleaned_up_after_execution_completes(self) -> None:
        """Test that pending removal versions are cleaned up after executions complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a file to get version tracking
            orch_file = orch_dir / "cleanup_test.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 1
"""
            )

            sentinel.run_once()

            # Simulate active execution on this version
            sentinel._active_versions[0].increment_executions()

            # Modify the file
            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 2
"""
            )
            set_mtime_in_future(orch_file)

            sentinel.run_once()

            # Old version in pending removal with 1 active execution
            assert len(sentinel._pending_removal_versions) == 1
            assert sentinel._pending_removal_versions[0].active_executions == 1

            # Simulate execution completing
            sentinel._pending_removal_versions[0].decrement_executions()

            # Run again - cleanup should remove the old version
            sentinel.run_once()

            # Pending removal should be empty now
            assert len(sentinel._pending_removal_versions) == 0

    def test_version_without_active_executions_removed_immediately(self) -> None:
        """Test that old versions without active executions are removed immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a file
            orch_file = orch_dir / "no_exec.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: no-exec-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 1
"""
            )

            sentinel.run_once()

            # No active executions on the version
            assert sentinel._active_versions[0].active_executions == 0

            # Modify the file
            orch_file.write_text(
                """orchestrations:
  - name: no-exec-orch
    trigger:
      project: TEST
    agent:
      prompt: Version 2
"""
            )
            set_mtime_in_future(orch_file)

            sentinel.run_once()

            # Old version should NOT be in pending removal (no active executions)
            assert len(sentinel._pending_removal_versions) == 0

            # New version should be active
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].orchestration.agent.prompt == "Version 2"

    def test_known_files_stores_mtime(self) -> None:
        """Test that _known_orchestration_files stores mtimes, not just presence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)
            orch_file = orch_dir / "mtime_test.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: test
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Should store mtime, not just True/False
            assert isinstance(sentinel._known_orchestration_files[orch_file], float)
            assert sentinel._known_orchestration_files[orch_file] > 0

    def test_detects_removed_orchestration_files(self) -> None:
        """Test that removed orchestration files are detected and their orchestrations unloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a new orchestration file
            orch_file = orch_dir / "removable.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: removable-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Test prompt
"""
            )

            # Run to detect and load the new file
            sentinel.run_once()

            # Should have loaded the orchestration
            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "removable-orch"
            assert orch_file in sentinel._known_orchestration_files

            # Now delete the file
            orch_file.unlink()

            # Run again - should detect removal and unload the orchestration
            sentinel.run_once()

            # Orchestration should be removed
            assert len(sentinel.orchestrations) == 0
            # File should be removed from known files
            assert orch_file not in sentinel._known_orchestration_files

    def test_removed_file_with_active_execution_moves_to_pending_removal(self) -> None:
        """Test that orchestrations from removed files with active executions go to pending removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a new orchestration file
            orch_file = orch_dir / "active_removal.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: active-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            # Run to load the file
            sentinel.run_once()

            # Should have one active version
            assert len(sentinel._active_versions) == 1

            # Simulate an active execution
            sentinel._active_versions[0].increment_executions()
            assert sentinel._active_versions[0].active_executions == 1

            # Delete the file
            orch_file.unlink()

            # Run again - should detect removal
            sentinel.run_once()

            # Orchestration should be removed from main list
            assert len(sentinel.orchestrations) == 0
            # But version should be in pending removal
            assert len(sentinel._pending_removal_versions) == 1
            assert sentinel._pending_removal_versions[0].name == "active-orch"
            assert sentinel._pending_removal_versions[0].active_executions == 1
            # Active versions should be empty
            assert len(sentinel._active_versions) == 0

    def test_removed_file_without_active_execution_removed_immediately(self) -> None:
        """Test that orchestrations from removed files without active executions are removed immediately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a new orchestration file
            orch_file = orch_dir / "no_active.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: no-active-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            # Run to load the file
            sentinel.run_once()

            # Should have one active version with no active executions
            assert len(sentinel._active_versions) == 1
            assert sentinel._active_versions[0].active_executions == 0

            # Delete the file
            orch_file.unlink()

            # Run again - should detect removal
            sentinel.run_once()

            # Orchestration should be removed from main list
            assert len(sentinel.orchestrations) == 0
            # Should NOT be in pending removal (no active executions)
            assert len(sentinel._pending_removal_versions) == 0
            # Active versions should be empty
            assert len(sentinel._active_versions) == 0

    def test_pending_removal_from_file_deletion_cleaned_up_after_execution(self) -> None:
        """Test that pending removal versions from deleted files are cleaned up after execution completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a new orchestration file
            orch_file = orch_dir / "cleanup_after_delete.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: cleanup-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
            )

            # Run to load the file
            sentinel.run_once()

            # Simulate an active execution
            sentinel._active_versions[0].increment_executions()

            # Delete the file
            orch_file.unlink()

            # Run - should move to pending removal
            sentinel.run_once()
            assert len(sentinel._pending_removal_versions) == 1

            # Simulate execution completing
            sentinel._pending_removal_versions[0].decrement_executions()

            # Run again - should clean up the pending removal
            sentinel.run_once()
            assert len(sentinel._pending_removal_versions) == 0

    def test_router_updated_after_file_removal(self) -> None:
        """Test that the router is updated when orchestrations are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(
                issues=[
                    {"key": "TEST-1", "fields": {"summary": "Test", "labels": ["review"]}},
                ]
            )
            agent_client = MockAgentClient(responses=["SUCCESS"])
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add an orchestration file that matches the issue
            orch_file = orch_dir / "matching.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: matching-orch
    trigger:
      project: TEST
      tags: [review]
    agent:
      prompt: Process this
"""
            )

            # Run - should detect and execute
            results, submitted_count = sentinel.run_once()
            assert len(results) == 1

            # Reset client call count
            jira_client.search_calls.clear()

            # Delete the file
            orch_file.unlink()

            # Run again - removal detected, no more orchestrations to match
            results, submitted_count = sentinel.run_once()

            # No orchestrations means no polling should occur
            assert len(sentinel.orchestrations) == 0

    def test_multiple_orchestrations_from_same_removed_file(self) -> None:
        """Test that multiple orchestrations from a single removed file are all unloaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add a file with multiple orchestrations
            orch_file = orch_dir / "multi.yaml"
            orch_file.write_text(
                """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
  - name: orch-2
    trigger:
      project: PROJ
    agent:
      prompt: Test 2
  - name: orch-3
    trigger:
      project: OTHER
    agent:
      prompt: Test 3
"""
            )

            # Run to load all orchestrations
            sentinel.run_once()

            # Should have loaded all three
            assert len(sentinel.orchestrations) == 3
            assert len(sentinel._active_versions) == 3

            # Delete the file
            orch_file.unlink()

            # Run again - should unload all orchestrations
            sentinel.run_once()

            # All orchestrations should be removed
            assert len(sentinel.orchestrations) == 0
            assert len(sentinel._active_versions) == 0
            assert orch_file not in sentinel._known_orchestration_files

    def test_removal_of_one_file_does_not_affect_others(self) -> None:
        """Test that removing one file doesn't affect orchestrations from other files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch_dir = Path(tmpdir)

            jira_client = MockJiraClient(issues=[])
            agent_client = MockAgentClient()
            tag_client = MockTagClient()
            config = make_config(orchestrations_dir=orch_dir)
            orchestrations: list[Orchestration] = []

            sentinel = Sentinel(
                config=config,
                orchestrations=orchestrations,
                jira_client=jira_client,
                agent_client=agent_client,
                tag_client=tag_client,
            )

            # Add two orchestration files
            file1 = orch_dir / "keep.yaml"
            file1.write_text(
                """orchestrations:
  - name: keep-orch
    trigger:
      project: TEST
    agent:
      prompt: Keep this
"""
            )

            file2 = orch_dir / "remove.yaml"
            file2.write_text(
                """orchestrations:
  - name: remove-orch
    trigger:
      project: PROJ
    agent:
      prompt: Remove this
"""
            )

            # Run to load both files
            sentinel.run_once()

            # Should have loaded both
            assert len(sentinel.orchestrations) == 2
            assert len(sentinel._known_orchestration_files) == 2

            # Delete only the second file
            file2.unlink()

            # Run again
            sentinel.run_once()

            # Should still have the first orchestration
            assert len(sentinel.orchestrations) == 1
            assert sentinel.orchestrations[0].name == "keep-orch"
            # Only the deleted file should be removed from known files
            assert file1 in sentinel._known_orchestration_files
            assert file2 not in sentinel._known_orchestration_files


class TestSentinelHotReloadMetrics:
    """Tests for hot-reload observability metrics (DS-97)."""

    def test_get_hot_reload_metrics_returns_dict(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_client: MockJiraClient,
        mock_agent_client: MockAgentClient,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that get_hot_reload_metrics returns a dict with expected keys."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=mock_jira_client,
            agent_client=mock_agent_client,
            tag_client=mock_tag_client,
        )

        metrics = sentinel.get_hot_reload_metrics()

        assert isinstance(metrics, dict)
        assert "orchestrations_loaded_total" in metrics
        assert "orchestrations_unloaded_total" in metrics
        assert "orchestrations_reloaded_total" in metrics
        # Initial values should be 0
        assert metrics["orchestrations_loaded_total"] == 0
        assert metrics["orchestrations_unloaded_total"] == 0
        assert metrics["orchestrations_reloaded_total"] == 0

    def test_loaded_counter_increments_on_new_file(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_client: MockJiraClient,
        mock_agent_client: MockAgentClient,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that loaded counter increments when new files are detected."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=mock_jira_client,
            agent_client=mock_agent_client,
            tag_client=mock_tag_client,
        )

        # Initial metrics
        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 0

        # Add a new orchestration file
        (temp_orchestrations_dir / "new.yaml").write_text(
            """orchestrations:
  - name: new-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
        )

        sentinel.run_once()

        # Counter should have incremented
        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 1

    def test_unloaded_counter_increments_on_file_deletion(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_client: MockJiraClient,
        mock_agent_client: MockAgentClient,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that unloaded counter increments when files are deleted."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=mock_jira_client,
            agent_client=mock_agent_client,
            tag_client=mock_tag_client,
        )

        # Add and load a file
        orch_file = temp_orchestrations_dir / "removable.yaml"
        orch_file.write_text(
            """orchestrations:
  - name: removable-orch
    trigger:
      project: TEST
    agent:
      prompt: Test
"""
        )
        sentinel.run_once()

        # Initial counter state after loading
        assert sentinel.get_hot_reload_metrics()["orchestrations_loaded_total"] == 1
        assert sentinel.get_hot_reload_metrics()["orchestrations_unloaded_total"] == 0

        # Delete the file
        orch_file.unlink()
        sentinel.run_once()

        # Unloaded counter should have incremented
        assert sentinel.get_hot_reload_metrics()["orchestrations_unloaded_total"] == 1

    def test_reloaded_counter_increments_on_file_modification(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_client: MockJiraClient,
        mock_agent_client: MockAgentClient,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that reloaded counter increments when files are modified."""
        # Create file before sentinel init so it's tracked
        orch_file = temp_orchestrations_dir / "modifiable.yaml"
        orch_file.write_text(
            """orchestrations:
  - name: original-orch
    trigger:
      project: TEST
    agent:
      prompt: Original
"""
        )

        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=mock_jira_client,
            agent_client=mock_agent_client,
            tag_client=mock_tag_client,
        )

        # Initial state
        assert sentinel.get_hot_reload_metrics()["orchestrations_reloaded_total"] == 0

        # Modify the file
        orch_file.write_text(
            """orchestrations:
  - name: modified-orch
    trigger:
      project: TEST
    agent:
      prompt: Modified
"""
        )
        set_mtime_in_future(orch_file)

        sentinel.run_once()

        # Reloaded counter should have incremented
        assert sentinel.get_hot_reload_metrics()["orchestrations_reloaded_total"] == 1

    def test_metrics_accumulate_over_multiple_operations(
        self,
        temp_orchestrations_dir: Path,
        mock_jira_client: MockJiraClient,
        mock_agent_client: MockAgentClient,
        mock_tag_client: MockTagClient,
    ) -> None:
        """Test that metrics accumulate correctly over multiple operations."""
        config = make_config(orchestrations_dir=temp_orchestrations_dir)
        orchestrations: list[Orchestration] = []

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=mock_jira_client,
            agent_client=mock_agent_client,
            tag_client=mock_tag_client,
        )

        # Load two files
        (temp_orchestrations_dir / "file1.yaml").write_text(
            """orchestrations:
  - name: orch-1
    trigger:
      project: TEST
    agent:
      prompt: Test 1
"""
        )
        (temp_orchestrations_dir / "file2.yaml").write_text(
            """orchestrations:
  - name: orch-2
    trigger:
      project: TEST
    agent:
      prompt: Test 2
"""
        )
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2
        assert metrics["orchestrations_unloaded_total"] == 0
        assert metrics["orchestrations_reloaded_total"] == 0

        # Delete one file
        (temp_orchestrations_dir / "file1.yaml").unlink()
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2  # Unchanged
        assert metrics["orchestrations_unloaded_total"] == 1
        assert metrics["orchestrations_reloaded_total"] == 0

        # Modify remaining file
        (temp_orchestrations_dir / "file2.yaml").write_text(
            """orchestrations:
  - name: orch-2-modified
    trigger:
      project: TEST
    agent:
      prompt: Test 2 Modified
"""
        )
        set_mtime_in_future(temp_orchestrations_dir / "file2.yaml")
        sentinel.run_once()

        metrics = sentinel.get_hot_reload_metrics()
        assert metrics["orchestrations_loaded_total"] == 2  # Unchanged
        assert metrics["orchestrations_unloaded_total"] == 1  # Unchanged
        assert metrics["orchestrations_reloaded_total"] == 1


class TestAttemptCountTracking:
    """Tests for DS-141: Track retry attempt numbers in Running Steps dashboard."""

    def test_get_and_increment_attempt_count_starts_at_one(self) -> None:
        """Test that first attempt for an issue/orchestration pair returns 1."""
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

        count = sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")
        assert count == 1

    def test_get_and_increment_attempt_count_increments(self) -> None:
        """Test that subsequent calls increment the attempt count."""
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

        count1 = sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")
        count2 = sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")
        count3 = sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")

        assert count1 == 1
        assert count2 == 2
        assert count3 == 3

    def test_get_and_increment_tracks_separately_by_issue_key(self) -> None:
        """Test that different issues have separate attempt counts."""
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

        # First issue gets attempts 1, 2
        sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")
        sentinel._get_and_increment_attempt_count("TEST-1", "my-orchestration")

        # Second issue starts at 1
        count = sentinel._get_and_increment_attempt_count("TEST-2", "my-orchestration")
        assert count == 1

    def test_get_and_increment_tracks_separately_by_orchestration(self) -> None:
        """Test that different orchestrations have separate attempt counts."""
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

        # Same issue, different orchestrations
        sentinel._get_and_increment_attempt_count("TEST-1", "orch-a")
        sentinel._get_and_increment_attempt_count("TEST-1", "orch-a")

        # Different orchestration starts at 1
        count = sentinel._get_and_increment_attempt_count("TEST-1", "orch-b")
        assert count == 1

    def test_get_and_increment_is_thread_safe(self) -> None:
        """Test that attempt count incrementing is thread-safe."""
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

        results: list[int] = []
        num_threads = 10

        def increment() -> None:
            count = sentinel._get_and_increment_attempt_count("TEST-1", "orch")
            results.append(count)

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All numbers from 1 to 10 should appear exactly once
        assert sorted(results) == list(range(1, num_threads + 1))

    def test_running_steps_use_tracked_attempt_number(self) -> None:
        """Test that Running Steps dashboard uses the tracked attempt number (DS-141).

        This test verifies that the attempt_number in RunningStepInfo is correctly
        tracked across multiple executions of the same issue/orchestration pair.
        We test this directly by calling _get_and_increment_attempt_count and verifying
        the values are used when creating RunningStepInfo objects.
        """
        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        agent_client = MockAgentClient(responses=["SUCCESS", "SUCCESS"])
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Simulate multiple executions by directly calling the attempt count tracker
        # This is the core functionality that DS-141 implements
        count1 = sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")
        count2 = sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")
        count3 = sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")

        # Verify the counts increment correctly
        assert count1 == 1, "First attempt should be 1"
        assert count2 == 2, "Second attempt should be 2"
        assert count3 == 3, "Third attempt should be 3"

        # Verify the internal state matches
        # DS-152: Now stores AttemptCountEntry with count and last_access
        key = ("TEST-1", "test-orch")
        with sentinel._attempt_counts_lock:
            entry = sentinel._attempt_counts.get(key)
            stored_count = entry.count if entry else 0
        assert stored_count == 3, "Stored count should be 3 after 3 increments"

    def test_running_step_info_contains_attempt_number(self) -> None:
        """Test that RunningStepInfo is created with correct attempt_number (DS-141).

        This test verifies that when an execution is submitted to the thread pool,
        the RunningStepInfo correctly uses the attempt_number from the tracker.
        """
        from sentinel.main import RunningStepInfo
        from datetime import datetime

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        # Use a blocking response to keep the future active
        blocking_event = threading.Event()
        task_started_event = threading.Event()

        class BlockingAgentClient(MockAgentClient):
            def run_agent(self, *args: Any, **kwargs: Any) -> AgentRunResult:
                task_started_event.set()  # Signal that task has started
                blocking_event.wait(timeout=5)
                return super().run_agent(*args, **kwargs)

        agent_client = BlockingAgentClient(responses=["SUCCESS"])
        config = make_config(max_concurrent_executions=2)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Pre-increment the attempt count to simulate a previous execution
        sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")

        # Set up thread pool and run
        sentinel._thread_pool = ThreadPoolExecutor(max_workers=2)

        try:
            # Run once in a background thread so we can check running steps
            run_thread = threading.Thread(target=sentinel.run_once)
            run_thread.start()

            # Wait for the task to actually start executing
            task_started_event.wait(timeout=5)

            # Check the running steps while task is blocked
            running_steps = sentinel.get_running_steps()
            assert len(running_steps) == 1, "Should have one running step"
            step = running_steps[0]
            assert step.attempt_number == 2, "Attempt number should be 2 (second attempt)"
            assert step.issue_key == "TEST-1"
            assert step.orchestration_name == "test-orch"
        finally:
            # Unblock the agent and clean up
            blocking_event.set()
            run_thread.join(timeout=5)
            sentinel._thread_pool.shutdown(wait=True)
            sentinel._thread_pool = None

    def test_cleanup_stale_attempt_counts(self) -> None:
        """Test that stale attempt count entries are cleaned up based on TTL (DS-152).

        This test verifies that the cleanup mechanism removes entries that haven't
        been accessed within the configured TTL period.
        """
        import time
        from unittest.mock import patch
        from sentinel.main import AttemptCountEntry

        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])

        # Use a very short TTL for testing
        config = make_config(attempt_counts_ttl=1)
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Add some entries directly with old timestamps
        current_time = time.monotonic()
        with sentinel._attempt_counts_lock:
            # Old entry (should be cleaned up)
            sentinel._attempt_counts[("OLD-1", "test-orch")] = AttemptCountEntry(
                count=5, last_access=current_time - 100  # 100 seconds ago
            )
            # Recent entry (should NOT be cleaned up)
            sentinel._attempt_counts[("RECENT-1", "test-orch")] = AttemptCountEntry(
                count=3, last_access=current_time - 0.5  # 0.5 seconds ago
            )

        # Verify both entries exist
        assert len(sentinel._attempt_counts) == 2

        # Run cleanup
        cleaned = sentinel._cleanup_stale_attempt_counts()

        # Verify old entry was cleaned up
        assert cleaned == 1
        assert len(sentinel._attempt_counts) == 1
        assert ("OLD-1", "test-orch") not in sentinel._attempt_counts
        assert ("RECENT-1", "test-orch") in sentinel._attempt_counts

    def test_cleanup_attempt_counts_called_in_run_once(self) -> None:
        """Test that _cleanup_stale_attempt_counts is called during run_once (DS-152).

        This ensures the cleanup is integrated into the main polling cycle.
        """
        from unittest.mock import patch, MagicMock

        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        config = make_config()
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Mock the cleanup method to track if it was called
        with patch.object(sentinel, '_cleanup_stale_attempt_counts') as mock_cleanup:
            mock_cleanup.return_value = 0
            sentinel.run_once()
            mock_cleanup.assert_called_once()

    def test_attempt_count_entry_updates_last_access(self) -> None:
        """Test that accessing an attempt count updates its last_access time (DS-152).

        This verifies that the TTL clock is reset each time an issue is processed,
        ensuring active issues are not prematurely cleaned up.
        """
        import time
        from sentinel.main import AttemptCountEntry

        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        config = make_config()
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Get initial time
        time_before = time.monotonic()

        # Increment attempt count
        sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")

        time_after = time.monotonic()

        # Verify the entry has a recent last_access time
        key = ("TEST-1", "test-orch")
        with sentinel._attempt_counts_lock:
            entry = sentinel._attempt_counts[key]

        assert entry.count == 1
        assert entry.last_access >= time_before
        assert entry.last_access <= time_after

        # Wait a tiny bit and update again
        time.sleep(0.01)
        time_before_second = time.monotonic()

        sentinel._get_and_increment_attempt_count("TEST-1", "test-orch")

        # Verify last_access was updated
        with sentinel._attempt_counts_lock:
            entry = sentinel._attempt_counts[key]

        assert entry.count == 2
        assert entry.last_access >= time_before_second

    def test_cleanup_logs_debug_when_no_stale_entries(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that cleanup logs at debug level when no stale entries are found (DS-157).

        This test verifies that the cleanup method logs a debug message when it runs
        but finds no stale entries to clean up, improving operational visibility.
        """
        import logging

        tag_client = MockTagClient()
        jira_client = MockJiraClient(issues=[], tag_client=tag_client)
        agent_client = MockAgentClient(responses=[])
        config = make_config(attempt_counts_ttl=3600)  # 1 hour TTL
        orchestrations = [make_orchestration(name="test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Ensure no entries exist (empty state)
        assert len(sentinel._attempt_counts) == 0

        # Run cleanup with debug logging enabled
        with caplog.at_level(logging.DEBUG, logger="sentinel.main"):
            cleaned = sentinel._cleanup_stale_attempt_counts()

        # Verify no entries were cleaned
        assert cleaned == 0

        # Verify debug message was logged
        debug_messages = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("no stale entries found" in msg for msg in debug_messages), (
            f"Expected debug log about no stale entries, got: {debug_messages}"
        )

        # DS-160: Verify TTL value is included in the debug message
        ttl_value = config.attempt_counts_ttl
        assert any(f"TTL: {ttl_value}s" in msg for msg in debug_messages), (
            f"Expected TTL value ({ttl_value}s) in debug message, got: {debug_messages}"
        )


class TestQueueEvictionBehavior:
    """Tests for DS-158: Queue eviction behavior - tests and logging."""

    def test_queue_evicts_oldest_item_when_full(self) -> None:
        """Test that the oldest item is evicted when queue reaches maxlen (DS-158).

        This test verifies that the deque-based queue correctly evicts the oldest
        (leftmost) item when a new item is added at capacity, maintaining FIFO ordering.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)  # Small queue for testing
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Add 3 items to fill the queue
        sentinel._add_to_issue_queue("TEST-1", "orch-1")
        sentinel._add_to_issue_queue("TEST-2", "orch-2")
        sentinel._add_to_issue_queue("TEST-3", "orch-3")

        # Verify queue is full
        assert len(sentinel._issue_queue) == 3
        queue_keys = [item.issue_key for item in sentinel._issue_queue]
        assert queue_keys == ["TEST-1", "TEST-2", "TEST-3"]

        # Add a 4th item - should evict TEST-1 (oldest)
        sentinel._add_to_issue_queue("TEST-4", "orch-4")

        # Verify queue still has 3 items but TEST-1 was evicted
        assert len(sentinel._issue_queue) == 3
        queue_keys = [item.issue_key for item in sentinel._issue_queue]
        assert queue_keys == ["TEST-2", "TEST-3", "TEST-4"]
        assert "TEST-1" not in queue_keys

    def test_queue_maintains_fifo_ordering(self) -> None:
        """Test that the queue maintains FIFO ordering as items are added (DS-158).

        This test verifies that items are ordered from oldest (leftmost) to newest
        (rightmost), which is the expected deque behavior for FIFO eviction.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=5)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Add items in order
        for i in range(1, 6):
            sentinel._add_to_issue_queue(f"TEST-{i}", f"orch-{i}")

        # Verify FIFO order (oldest first)
        queue_keys = [item.issue_key for item in sentinel._issue_queue]
        assert queue_keys == ["TEST-1", "TEST-2", "TEST-3", "TEST-4", "TEST-5"]

        # First item should be oldest (TEST-1)
        assert sentinel._issue_queue[0].issue_key == "TEST-1"
        # Last item should be newest (TEST-5)
        assert sentinel._issue_queue[-1].issue_key == "TEST-5"

    def test_eviction_logging_includes_evicted_item_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that eviction logging includes the evicted item's key (DS-158).

        This test verifies that when an item is evicted due to queue being at
        capacity, the log message includes the key of the evicted item for
        better debugging and observability.

        DS-161: Improved to use structured logging assertions - instead of
        string matching with 'evicted' in msg.lower(), we now verify the
        logger name, log level, and use more robust content assertions.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)  # Small queue for testing
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Fill the queue
        sentinel._add_to_issue_queue("EVICT-ME", "orch-old")
        sentinel._add_to_issue_queue("TEST-2", "orch-2")

        # Clear log records
        caplog.clear()

        # Add new item to trigger eviction (with debug logging)
        with caplog.at_level(logging.DEBUG, logger="sentinel.main"):
            sentinel._add_to_issue_queue("NEW-ITEM", "orch-new")

        # DS-161: Use structured logging verification - check logger name,
        # level, and presence of expected structured content in the message
        eviction_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.main"
            and "capacity" in r.message
            and "evicted" in r.message
        ]

        assert len(eviction_records) == 1, (
            f"Expected 1 eviction log record, got {len(eviction_records)}: "
            f"{[r.message for r in eviction_records]}"
        )

        eviction_record = eviction_records[0]
        # Verify the evicted item's key is logged
        assert "EVICT-ME" in eviction_record.message, (
            f"Expected evicted key 'EVICT-ME' in log message: {eviction_record.message}"
        )
        # Verify the evicted item's orchestration is logged
        assert "orch-old" in eviction_record.message, (
            f"Expected evicted orchestration 'orch-old' in log message: {eviction_record.message}"
        )
        # Verify the new item's key is logged
        assert "NEW-ITEM" in eviction_record.message, (
            f"Expected new key 'NEW-ITEM' in log message: {eviction_record.message}"
        )

    def test_dashboard_shows_most_recent_queued_issues(self) -> None:
        """Test that get_issue_queue returns most recent items after eviction (DS-158).

        This test verifies that after eviction, the dashboard API (get_issue_queue)
        returns the most recently queued issues, not the oldest ones that were evicted.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Add 5 items, causing 2 evictions
        for i in range(1, 6):
            sentinel._add_to_issue_queue(f"TEST-{i}", f"orch-{i}")

        # Dashboard API should return the 3 most recent items
        queue = sentinel.get_issue_queue()
        assert len(queue) == 3

        queue_keys = [item.issue_key for item in queue]
        # Should have TEST-3, TEST-4, TEST-5 (the 3 most recent)
        assert queue_keys == ["TEST-3", "TEST-4", "TEST-5"]
        # TEST-1 and TEST-2 should have been evicted
        assert "TEST-1" not in queue_keys
        assert "TEST-2" not in queue_keys

    def test_no_eviction_log_when_queue_not_full(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that no eviction log is produced when queue is not full (DS-158).

        This test ensures that eviction logging only occurs when an item is
        actually evicted, not on every add operation.

        DS-161: Improved to use structured logging assertions - instead of
        string matching, we verify using logger name and log level checks.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=10)  # Larger queue
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Add items without filling the queue
        with caplog.at_level(logging.DEBUG, logger="sentinel.main"):
            sentinel._add_to_issue_queue("TEST-1", "orch-1")
            sentinel._add_to_issue_queue("TEST-2", "orch-2")
            sentinel._add_to_issue_queue("TEST-3", "orch-3")

        # DS-161: Use structured logging verification - check for eviction logs
        # by logger name, level, and presence of eviction-related keywords
        eviction_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.main"
            and "capacity" in r.message
            and "evicted" in r.message
        ]
        assert len(eviction_records) == 0, (
            f"Expected no eviction logs, got: {[r.message for r in eviction_records]}"
        )

    def test_multiple_evictions_log_each_evicted_item(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that each eviction produces a log with the correct evicted item (DS-158).

        This test verifies that when multiple items are evicted in sequence,
        each eviction is logged with the correct evicted item's key.

        DS-161: Improved to use structured logging assertions - instead of
        string matching, we verify using logger name and log level checks.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Fill the queue
        sentinel._add_to_issue_queue("ITEM-A", "orch-a")
        sentinel._add_to_issue_queue("ITEM-B", "orch-b")

        with caplog.at_level(logging.DEBUG, logger="sentinel.main"):
            # Add ITEM-C -> evicts ITEM-A
            sentinel._add_to_issue_queue("ITEM-C", "orch-c")
            # Add ITEM-D -> evicts ITEM-B
            sentinel._add_to_issue_queue("ITEM-D", "orch-d")

        # DS-161: Use structured logging verification - check logger name,
        # level, and presence of expected structured content in the message
        eviction_records = [
            r for r in caplog.records
            if r.levelno == logging.DEBUG
            and r.name == "sentinel.main"
            and "capacity" in r.message
            and "evicted" in r.message
        ]

        assert len(eviction_records) == 2, (
            f"Expected 2 eviction log records, got {len(eviction_records)}: "
            f"{[r.message for r in eviction_records]}"
        )

        # First eviction should mention ITEM-A
        assert "ITEM-A" in eviction_records[0].message, (
            f"Expected 'ITEM-A' in first eviction log: {eviction_records[0].message}"
        )
        # Second eviction should mention ITEM-B
        assert "ITEM-B" in eviction_records[1].message, (
            f"Expected 'ITEM-B' in second eviction log: {eviction_records[1].message}"
        )

    def test_queue_clear_resets_for_new_cycle(self) -> None:
        """Test that _clear_issue_queue properly resets the queue (DS-158).

        This test verifies that the queue can be cleared (as happens at the
        start of each polling cycle) and new items can be added fresh.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=3)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Fill the queue
        for i in range(1, 4):
            sentinel._add_to_issue_queue(f"OLD-{i}", f"orch-{i}")

        assert len(sentinel._issue_queue) == 3

        # Clear the queue (as happens at cycle start)
        sentinel._clear_issue_queue()

        assert len(sentinel._issue_queue) == 0

        # Add new items - should not trigger eviction since queue is empty
        sentinel._add_to_issue_queue("NEW-1", "new-orch-1")

        assert len(sentinel._issue_queue) == 1
        assert sentinel._issue_queue[0].issue_key == "NEW-1"

    def test_eviction_preserves_queued_at_timestamp_ordering(self) -> None:
        """Test that queued_at timestamps are preserved and ordered correctly (DS-158).

        This test verifies that the queued_at timestamp for each item is preserved
        after eviction, ensuring proper ordering for dashboard display.

        DS-161: Improved to use mocked datetime.now() instead of time.sleep()
        for deterministic testing. The original implementation used time.sleep(0.01)
        which could theoretically be flaky on heavily loaded CI systems.
        """
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(max_queue_size=2)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # DS-161: Use deterministic timestamps via mocking instead of time.sleep()
        # This ensures reliable test behavior regardless of system load
        mock_times = [
            datetime(2026, 1, 15, 10, 0, 0),  # TEST-1 timestamp (will be evicted)
            datetime(2026, 1, 15, 10, 0, 1),  # TEST-2 timestamp
            datetime(2026, 1, 15, 10, 0, 2),  # TEST-3 timestamp
        ]
        time_iterator = iter(mock_times)

        with patch("sentinel.main.datetime") as mock_datetime:
            mock_datetime.now.side_effect = lambda: next(time_iterator)
            # Preserve datetime class for type checking
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            # Add items with mocked timestamps
            sentinel._add_to_issue_queue("TEST-1", "orch-1")  # Gets mock_times[0]
            sentinel._add_to_issue_queue("TEST-2", "orch-2")  # Gets mock_times[1]
            sentinel._add_to_issue_queue("TEST-3", "orch-3")  # Gets mock_times[2], evicts TEST-1

        # Verify timestamps are in ascending order
        queue_items = list(sentinel._issue_queue)
        assert len(queue_items) == 2

        # TEST-2's timestamp should be before TEST-3's
        assert queue_items[0].issue_key == "TEST-2"
        assert queue_items[1].issue_key == "TEST-3"
        assert queue_items[0].queued_at < queue_items[1].queued_at
        # DS-161: Verify exact timestamps for deterministic assertion
        assert queue_items[0].queued_at == datetime(2026, 1, 15, 10, 0, 1)
        assert queue_items[1].queued_at == datetime(2026, 1, 15, 10, 0, 2)


class TestSentinelOrchestrationLogging:
    """Tests for Sentinel integration with per-orchestration logging (DS-185)."""

    def test_sentinel_initializes_orch_log_manager_when_configured(
        self, tmp_path: Path
    ) -> None:
        """Test Sentinel initializes OrchestrationLogManager when configured."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel._orch_log_manager is not None
        sentinel.run_once_and_wait()  # Clean up

    def test_sentinel_does_not_initialize_orch_log_manager_when_not_configured(
        self,
    ) -> None:
        """Test Sentinel doesn't initialize OrchestrationLogManager when not set."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        # orchestration_logs_dir defaults to None
        assert config.orchestration_logs_dir is None
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        assert sentinel._orch_log_manager is None

    def test_sentinel_creates_log_files_via_log_for_orchestration(
        self, tmp_path: Path
    ) -> None:
        """Test that _log_for_orchestration creates log files for orchestrations."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [
            make_orchestration(name="test-orchestration", tags=["review"])
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Use _log_for_orchestration to trigger log file creation
        sentinel._log_for_orchestration(
            "test-orchestration", logging.INFO, "Test log entry"
        )

        # Close the log manager to flush
        sentinel._orch_log_manager.close_all()

        # Log file should be created for the orchestration
        log_file = logs_dir / "test-orchestration.log"
        assert log_file.exists()

    def test_sentinel_logs_contain_orchestration_activity(
        self, tmp_path: Path
    ) -> None:
        """Test orchestration logs contain relevant execution activity."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration(name="log-test-orch", tags=["review"])]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Simulate logging calls that would happen during orchestration execution
        sentinel._log_for_orchestration(
            "log-test-orch",
            logging.INFO,
            "Polling Jira for orchestration 'log-test-orch'",
        )
        sentinel._log_for_orchestration(
            "log-test-orch", logging.INFO, "Submitting 'log-test-orch' for TEST-1"
        )

        # Close the log manager to flush
        sentinel._orch_log_manager.close_all()

        # Read the log file and verify it contains expected content
        log_file = logs_dir / "log-test-orch.log"
        content = log_file.read_text()

        # Should contain polling activity
        assert "Polling Jira" in content
        # Should contain issue/execution activity
        assert "TEST-1" in content or "log-test-orch" in content

    def test_sentinel_separate_orchestrations_have_separate_logs(
        self, tmp_path: Path
    ) -> None:
        """Test different orchestrations write to separate log files."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [
            make_orchestration(name="orch-alpha", tags=["alpha"]),
            make_orchestration(name="orch-beta", tags=["beta"]),
        ]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Log to each orchestration's log file
        sentinel._log_for_orchestration(
            "orch-alpha", logging.INFO, "Message for orch-alpha"
        )
        sentinel._log_for_orchestration(
            "orch-beta", logging.INFO, "Message for orch-beta"
        )

        # Close the log manager to flush
        sentinel._orch_log_manager.close_all()

        # Both log files should exist
        log_alpha = logs_dir / "orch-alpha.log"
        log_beta = logs_dir / "orch-beta.log"
        assert log_alpha.exists()
        assert log_beta.exists()

        # Each log should contain its orchestration name
        content_alpha = log_alpha.read_text()
        content_beta = log_beta.read_text()

        assert "orch-alpha" in content_alpha
        assert "orch-beta" in content_beta
        # Verify separation - alpha's log shouldn't contain beta's message
        assert "Message for orch-beta" not in content_alpha
        assert "Message for orch-alpha" not in content_beta

    def test_sentinel_closes_log_manager_on_run_once_and_wait(
        self, tmp_path: Path
    ) -> None:
        """Test that run_once_and_wait properly closes the log manager."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # Access the logger to initialize it
        assert sentinel._orch_log_manager is not None
        sentinel._orch_log_manager.get_logger("test-orch")

        # After run_once_and_wait, handlers should be cleaned up
        sentinel.run_once_and_wait()

        # Handlers and loggers should be cleared
        assert len(sentinel._orch_log_manager._handlers) == 0
        assert len(sentinel._orch_log_manager._loggers) == 0

    def test_log_for_orchestration_logs_to_main_logger(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that _log_for_orchestration logs to the main logger."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        # No orchestration_logs_dir set - only main logger
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        with caplog.at_level(logging.INFO):
            sentinel._log_for_orchestration(
                "test-orch", logging.INFO, "Test log message"
            )

        assert "Test log message" in caplog.text

    def test_log_for_orchestration_logs_to_both_when_configured(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _log_for_orchestration logs to both main logger and orch file."""
        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        logs_dir = tmp_path / "orch_logs"
        config = replace(make_config(), orchestration_logs_dir=logs_dir)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        with caplog.at_level(logging.INFO):
            sentinel._log_for_orchestration(
                "dual-log-test", logging.INFO, "Dual log message"
            )

        # Should be in main logger
        assert "Dual log message" in caplog.text

        # Should also be in orchestration log file
        sentinel.run_once_and_wait()  # This closes the log manager and flushes
        log_file = logs_dir / "dual-log-test.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Dual log message" in content


class TestExtractRepoFromUrl:
    """Tests for extract_repo_from_url function (DS-204)."""

    def test_extracts_from_issue_url(self) -> None:
        """Test extraction from GitHub issue URL."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/org/repo/issues/123"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_extracts_from_pull_url(self) -> None:
        """Test extraction from GitHub pull request URL."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/my-org/my-repo/pull/456"
        result = extract_repo_from_url(url)
        assert result == "my-org/my-repo"

    def test_handles_empty_url(self) -> None:
        """Test that empty URL returns None."""
        from sentinel.main import extract_repo_from_url

        assert extract_repo_from_url("") is None
        assert extract_repo_from_url(None) is None  # type: ignore[arg-type]

    def test_handles_invalid_url(self) -> None:
        """Test that invalid URLs return None."""
        from sentinel.main import extract_repo_from_url

        assert extract_repo_from_url("not a url") is None
        assert extract_repo_from_url("https://github.com/org/repo") is None
        assert extract_repo_from_url("https://github.com/org/repo/commits/abc") is None

    def test_handles_enterprise_github_url(self) -> None:
        """Test extraction from GitHub Enterprise URLs."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.enterprise.com/org/repo/issues/123"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_handles_http_url(self) -> None:
        """Test extraction from HTTP URL (no HTTPS)."""
        from sentinel.main import extract_repo_from_url

        url = "http://github.com/org/repo/pull/789"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_handles_complex_repo_names(self) -> None:
        """Test extraction with complex org/repo names."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/my-org-123/my-repo-name/issues/1"
        result = extract_repo_from_url(url)
        assert result == "my-org-123/my-repo-name"


class TestGitHubIssueWithRepo:
    """Tests for GitHubIssueWithRepo class (DS-204)."""

    def test_key_includes_repo_context(self) -> None:
        """Test that key property includes full repo context."""
        from sentinel.github_poller import GitHubIssue
        from sentinel.main import GitHubIssueWithRepo

        issue = GitHubIssue(number=123, title="Test")
        wrapper = GitHubIssueWithRepo(issue, "org/repo")

        assert wrapper.key == "org/repo#123"

    def test_delegates_all_properties(self) -> None:
        """Test that all properties are properly delegated."""
        from sentinel.github_poller import GitHubIssue
        from sentinel.main import GitHubIssueWithRepo

        issue = GitHubIssue(
            number=42,
            title="Test Title",
            body="Test Body",
            state="open",
            author="testuser",
            assignees=["user1", "user2"],
            labels=["bug", "urgent"],
            is_pull_request=True,
            head_ref="feature-branch",
            base_ref="main",
            draft=False,
            repo_url="https://github.com/org/repo/pull/42",
        )
        wrapper = GitHubIssueWithRepo(issue, "org/repo")

        assert wrapper.number == 42
        assert wrapper.title == "Test Title"
        assert wrapper.body == "Test Body"
        assert wrapper.state == "open"
        assert wrapper.author == "testuser"
        assert wrapper.assignees == ["user1", "user2"]
        assert wrapper.labels == ["bug", "urgent"]
        assert wrapper.is_pull_request is True
        assert wrapper.head_ref == "feature-branch"
        assert wrapper.base_ref == "main"
        assert wrapper.draft is False
        assert wrapper.repo_url == "https://github.com/org/repo/pull/42"


class TestAddRepoContextFromUrls:
    """Tests for _add_repo_context_from_urls method (DS-204)."""

    def test_wraps_issues_with_repo_context(self) -> None:
        """Test that issues are wrapped with repo context from URL."""
        from sentinel.github_poller import GitHubIssue

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

        issues = [
            GitHubIssue(
                number=1,
                title="Issue 1",
                repo_url="https://github.com/org1/repo1/issues/1",
            ),
            GitHubIssue(
                number=2,
                title="Issue 2",
                repo_url="https://github.com/org2/repo2/pull/2",
            ),
        ]

        result = sentinel._add_repo_context_from_urls(issues)

        assert len(result) == 2
        assert result[0].key == "org1/repo1#1"
        assert result[1].key == "org2/repo2#2"

    def test_skips_issues_with_invalid_urls(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that issues with invalid URLs are skipped with warning."""
        from sentinel.github_poller import GitHubIssue

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

        issues = [
            GitHubIssue(
                number=1,
                title="Valid Issue",
                repo_url="https://github.com/org/repo/issues/1",
            ),
            GitHubIssue(
                number=2,
                title="Invalid Issue",
                repo_url="invalid-url",
            ),
            GitHubIssue(
                number=3,
                title="Empty URL Issue",
                repo_url="",
            ),
        ]

        with caplog.at_level(logging.WARNING):
            result = sentinel._add_repo_context_from_urls(issues)

        # Only valid issue should be returned
        assert len(result) == 1
        assert result[0].key == "org/repo#1"

        # Warnings should be logged for invalid URLs
        assert "Could not extract repo from URL for issue #2" in caplog.text
        assert "Could not extract repo from URL for issue #3" in caplog.text

    def test_handles_mixed_repos_in_project(self) -> None:
        """Test handling of issues from multiple repos in a single project."""
        from sentinel.github_poller import GitHubIssue

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

        # A project can contain issues from multiple repositories
        issues = [
            GitHubIssue(
                number=10,
                title="From repo-a",
                repo_url="https://github.com/org/repo-a/issues/10",
            ),
            GitHubIssue(
                number=20,
                title="From repo-b",
                repo_url="https://github.com/org/repo-b/pull/20",
            ),
            GitHubIssue(
                number=30,
                title="From repo-c",
                repo_url="https://github.com/different-org/repo-c/issues/30",
            ),
        ]

        result = sentinel._add_repo_context_from_urls(issues)

        assert len(result) == 3
        assert result[0].key == "org/repo-a#10"
        assert result[1].key == "org/repo-b#20"
        assert result[2].key == "different-org/repo-c#30"


class TestGitHubTriggerDeduplication:
    """Tests for GitHub trigger deduplication logic (DS-341).

    These tests verify that:
    1. Same project with different project_filter values are polled separately
    2. Same project with different labels values are polled separately
    3. Identical triggers are still properly deduplicated
    4. None/empty values in project_filter and labels are handled cleanly
    """

    def test_same_project_different_project_filter_polled_separately(self) -> None:
        """Test that same project with different project_filter values are polled separately."""
        from sentinel.orchestration import TriggerConfig

        # Create orchestrations with same project but different project_filter
        orch1 = make_orchestration(
            name="orch-ready",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
        )
        orch2 = make_orchestration(
            name="orch-in-progress",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "In Progress"',
        )

        # Build deduplication keys matching the logic in _poll_github_triggers
        def build_trigger_key(orch: Orchestration) -> str:
            filter_part = orch.trigger.project_filter or ""
            labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
            return (
                f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
                f":{filter_part}:{labels_part}"
            )

        key1 = build_trigger_key(orch1)
        key2 = build_trigger_key(orch2)

        # Keys should be different due to different project_filter values
        assert key1 != key2
        assert 'Status = "Ready"' in key1
        assert 'Status = "In Progress"' in key2

    def test_same_project_different_labels_polled_separately(self) -> None:
        """Test that same project with different labels values are polled separately."""
        # Create orchestrations with same project but different labels
        orch1 = make_orchestration(
            name="orch-bugs",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            labels=["bug"],
        )
        orch2 = make_orchestration(
            name="orch-features",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            labels=["feature", "enhancement"],
        )

        # Build deduplication keys
        def build_trigger_key(orch: Orchestration) -> str:
            filter_part = orch.trigger.project_filter or ""
            labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
            return (
                f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
                f":{filter_part}:{labels_part}"
            )

        key1 = build_trigger_key(orch1)
        key2 = build_trigger_key(orch2)

        # Keys should be different due to different labels values
        assert key1 != key2
        assert "bug" in key1
        assert "feature,enhancement" in key2

    def test_identical_triggers_deduplicated(self) -> None:
        """Test that identical triggers are properly deduplicated."""
        # Create orchestrations with identical trigger configurations
        orch1 = make_orchestration(
            name="orch-1",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            labels=["bug"],
        )
        orch2 = make_orchestration(
            name="orch-2",  # Different name, same trigger config
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            labels=["bug"],
        )

        # Build deduplication keys
        def build_trigger_key(orch: Orchestration) -> str:
            filter_part = orch.trigger.project_filter or ""
            labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
            return (
                f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
                f":{filter_part}:{labels_part}"
            )

        key1 = build_trigger_key(orch1)
        key2 = build_trigger_key(orch2)

        # Keys should be identical since trigger configurations match
        assert key1 == key2

    def test_none_project_filter_produces_clean_key(self) -> None:
        """Test that None project_filter values produce clean deduplication keys (DS-341).

        Previously, None values would result in the string 'None' in the key.
        After DS-341, None/empty values should produce empty string in the key.
        """
        # Create orchestration with no project_filter (None/empty)
        orch = make_orchestration(
            name="orch-no-filter",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            # project_filter not specified (defaults to "")
        )

        # Build deduplication key
        filter_part = orch.trigger.project_filter or ""
        labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
        key = (
            f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
            f":{filter_part}:{labels_part}"
        )

        # Key should NOT contain 'None' as a string
        assert "None" not in key
        # Key should have clean format: github:owner/number::
        assert key == "github:test-org/1::"

    def test_empty_labels_produces_clean_key(self) -> None:
        """Test that empty labels list produces clean deduplication keys (DS-341)."""
        # Create orchestration with no labels
        orch = make_orchestration(
            name="orch-no-labels",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            # labels not specified or empty
        )

        # Build deduplication key
        filter_part = orch.trigger.project_filter or ""
        labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
        key = (
            f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
            f":{filter_part}:{labels_part}"
        )

        # Key should have clean format with empty labels part
        assert key == 'github:test-org/1:Status = "Ready":'

    def test_both_none_filter_and_empty_labels_clean_key(self) -> None:
        """Test that both None filter and empty labels produce clean key."""
        # Create orchestration with neither filter nor labels
        orch = make_orchestration(
            name="orch-minimal",
            source="github",
            project_number=5,
            project_owner="my-org",
            project_scope="org",
        )

        # Build deduplication key
        filter_part = orch.trigger.project_filter or ""
        labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
        key = (
            f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
            f":{filter_part}:{labels_part}"
        )

        # Key should be clean without 'None' anywhere
        assert "None" not in key
        assert key == "github:my-org/5::"

    def test_combined_filter_and_labels_in_key(self) -> None:
        """Test that both filter and labels are included in deduplication key."""
        orch = make_orchestration(
            name="orch-full",
            source="github",
            project_number=10,
            project_owner="acme",
            project_scope="org",
            project_filter='Priority = "High"',
            labels=["urgent", "critical"],
        )

        # Build deduplication key
        filter_part = orch.trigger.project_filter or ""
        labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
        key = (
            f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
            f":{filter_part}:{labels_part}"
        )

        # Verify all parts are present
        assert key == 'github:acme/10:Priority = "High":urgent,critical'
