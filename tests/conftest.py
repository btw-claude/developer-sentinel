"""Shared pytest fixtures for Developer Sentinel tests.

This module provides reusable test fixtures extracted from test_main.py to reduce
code duplication, particularly in the hot-reload tests.

DS-97: Refactored test setup code into shared fixtures.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from sentinel.config import Config
from sentinel.executor import AgentClient, AgentRunResult
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
    ) -> AgentRunResult:
        self.calls.append((prompt, tools, context, timeout_seconds, issue_key, model, orchestration_name))
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return AgentRunResult(response=response, workdir=None)


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
    max_eager_iterations: int = 10,
    orchestrations_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
) -> Config:
    """Helper to create a Config for testing."""
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent_executions,
        max_eager_iterations=max_eager_iterations,
        orchestrations_dir=orchestrations_dir or Path("orchestrations"),
        attempt_counts_ttl=attempt_counts_ttl,
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


def set_mtime_in_future(file_path: Path, seconds_offset: float = 1.0) -> None:
    """Set a file's mtime to a future time to ensure mtime difference detection.

    This helper explicitly sets the file's modification time using os.utime()
    rather than relying on time.sleep() which can be flaky on fast filesystems
    or under heavy load.

    Args:
        file_path: Path to the file to modify.
        seconds_offset: Number of seconds to add to current time (default: 1.0).
    """
    current_stat = file_path.stat()
    new_mtime = current_stat.st_mtime + seconds_offset
    os.utime(file_path, (current_stat.st_atime, new_mtime))


# Pytest fixtures


@pytest.fixture
def mock_jira_client() -> MockJiraClient:
    """Provide a fresh MockJiraClient instance."""
    return MockJiraClient(issues=[])


@pytest.fixture
def mock_agent_client() -> MockAgentClient:
    """Provide a fresh MockAgentClient instance."""
    return MockAgentClient()


@pytest.fixture
def mock_tag_client() -> MockTagClient:
    """Provide a fresh MockTagClient instance."""
    return MockTagClient()


@pytest.fixture
def temp_orchestrations_dir():
    """Provide a temporary directory for orchestration files.

    Yields the Path to the temporary directory and cleans it up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hot_reload_sentinel(
    temp_orchestrations_dir: Path,
    mock_jira_client: MockJiraClient,
    mock_agent_client: MockAgentClient,
    mock_tag_client: MockTagClient,
):
    """Provide a Sentinel instance configured for hot-reload testing.

    This fixture creates a Sentinel with:
    - A temporary orchestrations directory
    - Mock Jira, Agent, and Tag clients
    - No initial orchestrations (empty list)

    This is the standard setup for most hot-reload tests.
    """
    from sentinel.main import Sentinel

    config = make_config(orchestrations_dir=temp_orchestrations_dir)
    orchestrations: list[Orchestration] = []

    sentinel = Sentinel(
        config=config,
        orchestrations=orchestrations,
        jira_client=mock_jira_client,
        agent_client=mock_agent_client,
        tag_client=mock_tag_client,
    )

    return sentinel
