"""Tests for main entry point module."""

import logging
import signal
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from sentinel.config import Config
from sentinel.executor import AgentClient
from sentinel.main import Sentinel, parse_args, setup_logging
from sentinel.orchestration import AgentConfig, Orchestration, TriggerConfig
from sentinel.poller import JiraClient
from sentinel.tag_manager import JiraTagClient


class MockJiraClient(JiraClient):
    """Mock Jira client for testing."""

    def __init__(
        self,
        issues: list[dict[str, Any]] | None = None,
        tag_client: "MockTagClient | None" = None,
    ) -> None:
        self.issues = issues or []
        self.search_calls: list[tuple[str, int]] = []
        self.tag_client = tag_client

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.search_calls.append((jql, max_results))
        # If we have a tag client, filter out issues whose trigger tags have been removed
        if self.tag_client:
            result = []
            for issue in self.issues:
                issue_key = issue.get("key", "")
                removed = self.tag_client.remove_calls
                # Check if any label in this issue was removed
                labels = issue.get("fields", {}).get("labels", [])
                # If any label was removed for this issue, skip it
                if any(key == issue_key for key, _ in removed):
                    continue
                result.append(issue)
            return result
        return self.issues


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["SUCCESS: Done"]
        self.call_count = 0
        self.calls: list[
            tuple[str, list[str], dict[str, Any] | None, int | None, str | None, str | None, str | None]
        ] = []

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
        self.calls.append((prompt, tools, context, timeout_seconds, issue_key, model, orchestration_name))
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class MockTagClient(JiraTagClient):
    """Mock tag client for testing."""

    def __init__(self) -> None:
        self.labels: dict[str, list[str]] = {}
        self.add_calls: list[tuple[str, str]] = []
        self.remove_calls: list[tuple[str, str]] = []

    def add_label(self, issue_key: str, label: str) -> None:
        self.add_calls.append((issue_key, label))
        if issue_key not in self.labels:
            self.labels[issue_key] = []
        self.labels[issue_key].append(label)

    def remove_label(self, issue_key: str, label: str) -> None:
        self.remove_calls.append((issue_key, label))


def make_config(
    poll_interval: int = 60,
    max_issues: int = 50,
    max_concurrent_executions: int = 1,
    orchestrations_dir: Path | None = None,
) -> Config:
    """Helper to create a Config for testing."""
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent_executions,
        orchestrations_dir=orchestrations_dir or Path("orchestrations"),
    )


def make_orchestration(
    name: str = "test-orch",
    project: str = "TEST",
    tags: list[str] | None = None,
) -> Orchestration:
    """Helper to create an Orchestration for testing."""
    return Orchestration(
        name=name,
        trigger=TriggerConfig(project=project, tags=tags or []),
        agent=AgentConfig(prompt="Test prompt", tools=["jira"]),
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

        results = sentinel.run_once()

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

        results = sentinel.run_once()

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

        results = sentinel.run_once()

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
        results = sentinel.run_once()

        # Should return early without executing all issues
        assert len(results) == 0


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
        results = sentinel.run_once()
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
            results = sentinel.run_once()
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
            results = sentinel.run_once()
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
