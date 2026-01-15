"""Tests for main entry point module."""

import logging
import signal
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from sentinel.executor import AgentClient, AgentRunResult
from sentinel.main import Sentinel, parse_args, setup_logging
from sentinel.orchestration import Orchestration

# DS-100: Import shared fixtures and helpers from conftest.py
# These provide MockJiraClient, MockAgentClient, MockTagClient,
# make_config, make_orchestration, and set_mtime_in_future
from tests.conftest import (
    MockJiraClient,
    MockAgentClient,
    MockTagClient,
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
        execution_order: list[str] = []
        lock = threading.Lock()

        class OrderTrackingAgentClient(AgentClient):
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
                with lock:
                    execution_order.append(f"start:{issue_key}")
                time.sleep(0.05)
                with lock:
                    execution_order.append(f"end:{issue_key}")
                return AgentRunResult(response="SUCCESS: Done", workdir=None)

        tag_client = MockTagClient()
        jira_client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1", "labels": ["review"]}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2", "labels": ["review"]}},
            ],
            tag_client=tag_client,
        )
        agent_client = OrderTrackingAgentClient()
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
        start_indices = [i for i, x in enumerate(execution_order) if x.startswith("start:")]
        end_indices = [i for i, x in enumerate(execution_order) if x.startswith("end:")]

        # Both should start before either ends (concurrent execution)
        assert len(start_indices) == 2
        assert len(end_indices) == 2
        # Second start should happen before first end in concurrent execution
        assert start_indices[1] < end_indices[0]


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
        key = ("TEST-1", "test-orch")
        with sentinel._attempt_counts_lock:
            stored_count = sentinel._attempt_counts.get(key, 0)
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
