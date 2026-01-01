"""Tests for main entry point module."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from sentinel.config import Config
from sentinel.executor import AgentClient
from sentinel.main import Sentinel, parse_args, setup_logging
from sentinel.orchestration import AgentConfig, Orchestration, TriggerConfig
from sentinel.poller import JiraClient
from sentinel.tag_manager import JiraTagClient


class MockJiraClient(JiraClient):
    """Mock Jira client for testing."""

    def __init__(self, issues: list[dict[str, Any]] | None = None) -> None:
        self.issues = issues or []
        self.search_calls: list[tuple[str, int]] = []

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.search_calls.append((jql, max_results))
        return self.issues


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["SUCCESS: Done"]
        self.call_count = 0
        self.calls: list[tuple[str, list[str], dict[str, Any] | None, int | None]] = []

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        self.calls.append((prompt, tools, context, timeout_seconds))
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

    def get_labels(self, issue_key: str) -> list[str]:
        return self.labels.get(issue_key, [])


def make_config(
    poll_interval: int = 60,
    max_issues: int = 50,
) -> Config:
    """Helper to create a Config for testing."""
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
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
