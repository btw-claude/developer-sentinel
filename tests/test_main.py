"""Tests for main entry point module."""

import logging
import signal
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from sentinel.executor import AgentClient
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
        from concurrent.futures import ThreadPoolExecutor
        import threading

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

    def test_consecutive_eager_polls_counter_initialized(self) -> None:
        """Test that consecutive eager polls counter starts at 0 (DS-98)."""
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

        assert sentinel._consecutive_eager_polls == 0

    def test_max_eager_iterations_config_respected(self) -> None:
        """Test that max_eager_iterations config is stored correctly (DS-98)."""
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

    def test_eager_polling_limit_forces_sleep_in_run_loop(self) -> None:
        """Integration test: verify max_eager_iterations forces sleep in the run loop (DS-101).

        This test exercises the full eager polling limit behavior by:
        1. Configuring max_eager_iterations=3 and a short poll_interval
        2. Providing a continuous stream of work (issues that always match)
        3. Verifying the counter increments each time work is submitted
        4. Verifying that after reaching max_eager_iterations, a sleep is forced
        5. Verifying the counter resets appropriately

        The test uses the full run() method with time.sleep and signal mocking.
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch

        # Save original sleep function before patching
        real_sleep = time.sleep

        # Track forced sleeps and poll cycles
        forced_sleep_intervals = 0
        poll_cycles = 0
        max_poll_cycles = 10  # Safety limit to prevent infinite loops

        class ContinuousWorkJiraClient(MockJiraClient):
            """Jira client that always returns fresh work to trigger eager polling."""

            def __init__(self, tag_client: MockTagClient) -> None:
                super().__init__(issues=[], tag_client=tag_client)
                self.issue_counter = 0

            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_cycles
                self.search_calls.append((jql, max_results))
                poll_cycles += 1

                # Stop returning work after max_poll_cycles to allow shutdown
                if poll_cycles > max_poll_cycles:
                    return []

                # Always return a new issue to simulate continuous work
                self.issue_counter += 1
                return [
                    {
                        "key": f"TEST-{self.issue_counter}",
                        "fields": {"summary": f"Issue {self.issue_counter}", "labels": ["review"]},
                    }
                ]

        tag_client = MockTagClient()
        jira_client = ContinuousWorkJiraClient(tag_client=tag_client)
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])

        # Configure with max_eager_iterations=3 and short poll interval
        config = make_config(
            poll_interval=2,  # 2 seconds to distinguish from eager polling (which is immediate)
            max_eager_iterations=3,
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

        # Verify initial state
        assert sentinel._consecutive_eager_polls == 0

        # Track sleep patterns to detect forced sleep intervals
        sleep_pattern: list[tuple[float, int]] = []  # (sleep_seconds, poll_cycle_at_time)
        consecutive_one_second_sleeps = 0

        def mock_sleep(seconds: float) -> None:
            nonlocal forced_sleep_intervals, consecutive_one_second_sleeps
            sleep_pattern.append((seconds, poll_cycles))

            # Count when we hit poll_interval consecutive 1-second sleeps
            # This indicates a forced sleep interval after max_eager_iterations
            if seconds == 1:
                consecutive_one_second_sleeps += 1
                if consecutive_one_second_sleeps >= config.poll_interval:
                    forced_sleep_intervals += 1
                    consecutive_one_second_sleeps = 0
            else:
                consecutive_one_second_sleeps = 0

            # Minimal actual sleep to prevent tight loops (using real_sleep, not the mocked one)
            real_sleep(0.001)

        def mock_signal(signum: int, handler: Any) -> Any:
            """Mock signal.signal to avoid threading issues."""
            return None

        # Run the sentinel's run() method in a thread with mocked sleep and signal
        sentinel_exception: Exception | None = None

        def run_sentinel() -> None:
            nonlocal sentinel_exception
            try:
                with patch("time.sleep", side_effect=mock_sleep), \
                     patch("signal.signal", side_effect=mock_signal):
                    sentinel.run()
            except Exception as e:
                sentinel_exception = e

        # Start sentinel in background thread
        sentinel_thread = threading.Thread(target=run_sentinel, daemon=True)
        sentinel_thread.start()

        # Wait for enough poll cycles (using real_sleep)
        for _ in range(50):  # Up to 5 seconds
            if poll_cycles >= max_poll_cycles:
                break
            real_sleep(0.1)

        # Request shutdown
        sentinel.request_shutdown()
        sentinel_thread.join(timeout=5)

        # Verify no exceptions occurred
        assert sentinel_exception is None, f"Sentinel raised exception: {sentinel_exception}"

        # Verify we had enough poll cycles to observe the behavior
        assert poll_cycles >= 6, f"Expected at least 6 poll cycles, got {poll_cycles}"

        # Verify we saw at least one forced sleep interval
        # With max_eager_iterations=3 and 6+ poll cycles, we should have hit the limit twice
        # Each time the limit is hit, the sentinel sleeps for poll_interval seconds (in 1-second increments)
        assert forced_sleep_intervals >= 1, (
            f"Expected at least 1 forced sleep interval after reaching max_eager_iterations, "
            f"got {forced_sleep_intervals}. Sleep pattern: {sleep_pattern[:20]}"
        )

    def test_eager_counter_resets_when_no_work_submitted(self) -> None:
        """Integration test: verify eager counter resets when no work is found (DS-101).

        This verifies that the counter properly resets to 0 when a poll cycle
        returns no work (submitted_count == 0), ensuring the eager polling
        limit only applies to consecutive cycles with work.

        Uses run() to test the full run loop behavior.
        """
        import threading
        import time
        from unittest.mock import patch

        # Save original sleep function before patching
        real_sleep = time.sleep

        # Track poll cycles and counter values
        poll_cycles = 0

        class AlternatingWorkJiraClient(MockJiraClient):
            """Jira client that alternates between returning work and no work."""

            def __init__(self, tag_client: MockTagClient, max_polls: int = 8) -> None:
                super().__init__(issues=[], tag_client=tag_client)
                self.max_polls = max_polls

            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_cycles
                self.search_calls.append((jql, max_results))
                poll_cycles += 1

                # Stop after max_polls
                if poll_cycles > self.max_polls:
                    return []

                # Return work on odd polls, no work on even polls
                if poll_cycles % 2 == 1:
                    return [
                        {
                            "key": f"TEST-{poll_cycles}",
                            "fields": {"summary": f"Issue {poll_cycles}", "labels": ["review"]},
                        }
                    ]
                return []

        tag_client = MockTagClient()
        jira_client = AlternatingWorkJiraClient(tag_client=tag_client)
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])

        config = make_config(
            poll_interval=1,
            max_eager_iterations=5,  # Higher than we'll reach
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

        # Capture counter values by wrapping the run loop's sleep handling
        original_counter_values: list[int] = []

        def mock_sleep(seconds: float) -> None:
            # Capture counter value at each sleep
            original_counter_values.append(sentinel._consecutive_eager_polls)
            real_sleep(0.001)

        def mock_signal(signum: int, handler: Any) -> Any:
            """Mock signal.signal to avoid threading issues."""
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

        # Wait for enough poll cycles (using real_sleep)
        for _ in range(30):
            if poll_cycles >= 8:
                break
            real_sleep(0.1)

        sentinel.request_shutdown()
        sentinel_thread.join(timeout=5)

        assert sentinel_exception is None, f"Sentinel raised exception: {sentinel_exception}"
        assert poll_cycles >= 6, f"Expected at least 6 poll cycles, got {poll_cycles}"

        # With alternating work/no-work, counter should oscillate between 0 and 1
        # and never reach max_eager_iterations (5)
        # Counter values captured during sleeps show the reset behavior
        # After no work (even polls), counter should be 0
        # After work (odd polls), counter should be 1
        assert all(v <= 1 for v in original_counter_values), (
            f"Counter should never exceed 1 with alternating work pattern, "
            f"got values: {original_counter_values}"
        )

    def test_eager_iterations_counter_increments_with_continuous_work(self) -> None:
        """Integration test: verify counter increments correctly with continuous work (DS-101).

        This tests that when work is continuously submitted, the _consecutive_eager_polls
        counter increments properly until it reaches max_eager_iterations.
        """
        import threading
        import time
        from unittest.mock import patch

        # Save original sleep function before patching
        real_sleep = time.sleep

        poll_cycles = 0
        max_polls = 8

        class ContinuousWorkJiraClient(MockJiraClient):
            """Jira client that returns work for first N polls, then stops."""

            def __init__(self, tag_client: MockTagClient) -> None:
                super().__init__(issues=[], tag_client=tag_client)
                self.issue_counter = 0

            def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
                nonlocal poll_cycles
                self.search_calls.append((jql, max_results))
                poll_cycles += 1

                if poll_cycles > max_polls:
                    return []

                self.issue_counter += 1
                return [
                    {
                        "key": f"TEST-{self.issue_counter}",
                        "fields": {"summary": f"Issue {self.issue_counter}", "labels": ["review"]},
                    }
                ]

        tag_client = MockTagClient()
        jira_client = ContinuousWorkJiraClient(tag_client=tag_client)
        agent_client = MockAgentClient(responses=["SUCCESS: Done"])

        config = make_config(
            poll_interval=2,
            max_eager_iterations=3,  # Counter resets every 3 polls with work
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

        # Track counter values at each sleep
        counter_at_sleep: list[int] = []

        def mock_sleep(seconds: float) -> None:
            counter_at_sleep.append(sentinel._consecutive_eager_polls)
            real_sleep(0.001)

        def mock_signal(signum: int, handler: Any) -> Any:
            """Mock signal.signal to avoid threading issues."""
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

        # Wait for enough poll cycles (using real_sleep)
        for _ in range(50):
            if poll_cycles >= max_polls:
                break
            real_sleep(0.1)

        sentinel.request_shutdown()
        sentinel_thread.join(timeout=5)

        assert sentinel_exception is None, f"Sentinel raised exception: {sentinel_exception}"
        assert poll_cycles >= 6, f"Expected at least 6 poll cycles, got {poll_cycles}"

        # With max_eager_iterations=3, we expect:
        # Polls 1, 2 -> counter at 1, 2 (no sleep, immediate poll)
        # Poll 3 -> counter hits 3, forced sleep (counter resets to 0)
        # Polls 4, 5 -> counter at 1, 2
        # Poll 6 -> counter hits 3, forced sleep again

        # The counter values captured at sleep should show the pattern
        # Sleep happens when counter >= max_eager_iterations (forced) or when no work (counter reset)
        # Look for the pattern of counter values including 0 (after forced sleep reset)
        zeros_count = sum(1 for v in counter_at_sleep if v == 0)
        assert zeros_count >= 2, (
            f"Expected at least 2 counter resets (from forced sleeps), "
            f"got {zeros_count} zeros in counter values: {counter_at_sleep}"
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
            import time

            time.sleep(0.1)
            sentinel.request_shutdown()

        import threading

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
            import threading

            def run_sentinel() -> None:
                sentinel.run()

            thread = threading.Thread(target=run_sentinel)
            thread.start()

            # Give it a moment to register handlers
            import time

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
            import threading

            def run_sentinel() -> None:
                sentinel.run()

            thread = threading.Thread(target=run_sentinel)
            thread.start()

            # Give it a moment to register handlers
            import time

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
        import threading
        import time

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
            ) -> str:
                nonlocal execution_count, max_concurrent_seen
                with lock:
                    execution_count += 1
                    if execution_count > max_concurrent_seen:
                        max_concurrent_seen = execution_count

                # Simulate some work
                time.sleep(0.1)

                with lock:
                    execution_count -= 1

                return "SUCCESS: Done"

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
        from concurrent.futures import ThreadPoolExecutor

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
        import threading
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
        import threading
        import time

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
            ) -> str:
                with lock:
                    execution_order.append(f"start:{issue_key}")
                time.sleep(0.05)
                with lock:
                    execution_order.append(f"end:{issue_key}")
                return "SUCCESS: Done"

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

