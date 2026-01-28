"""Shared pytest fixtures for Developer Sentinel tests.

This module provides reusable test fixtures extracted from test_main.py to reduce
code duplication, particularly in the hot-reload tests.

DS-97: Refactored test setup code into shared fixtures.

TrackingAgentClient Usage Guidelines (DS-196)
=============================================

The TrackingAgentClient can be instantiated in two ways:

1. **Direct instantiation** (PREFERRED for most tests)::

       from tests.conftest import TrackingAgentClient
       agent_client = TrackingAgentClient(execution_delay=0.05, track_order=True)

2. **Factory fixture** (for dependency injection scenarios)::

       def test_example(tracking_agent_client_factory):
           client = tracking_agent_client_factory(track_order=True)

**Recommended Approach**: Use direct instantiation for most tests.

Rationale:

- Direct instantiation is explicit and makes test setup clearer
- Tests don't need to rely on pytest's fixture dependency injection
- The class parameters (execution_delay, track_order, track_per_orch) are
  immediately visible at the call site
- Existing tests in test_main.py follow this pattern consistently

The factory fixture is available for cases where:

- You need pytest's automatic cleanup/teardown behavior
- You're building higher-level fixtures that compose multiple components
- You prefer dependency injection patterns in your test suite
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal

import pytest

from sentinel.agent_clients.base import AgentType
from sentinel.agent_clients.factory import AgentClientFactory
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

    def __init__(
        self,
        responses: list[str] | None = None,
        agent_type_value: AgentType = "claude",
    ) -> None:
        self.responses = responses or ["SUCCESS: Done"]
        self.call_count = 0
        self.calls: list[
            tuple[str, list[str], dict[str, Any] | None, int | None, str | None, str | None, str | None]
        ] = []
        self._agent_type = agent_type_value

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return self._agent_type

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
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


class TrackingAgentClient(AgentClient):
    """Agent client that tracks concurrent executions for testing.

    DS-188: Extracted from duplicate inner classes in TestPerOrchestrationConcurrencyLimits
    to reduce code duplication and improve maintainability.

    This class provides flexible tracking of concurrent executions with support for:
    - Global execution count tracking
    - Per-orchestration execution count tracking
    - Configurable execution delay
    - Optional execution order tracking

    Attributes:
        execution_count: Current number of concurrent executions (global).
        max_concurrent_seen: Maximum concurrent executions observed (global).
        orch_execution_counts: Per-orchestration current execution counts.
        orch_max_concurrent: Per-orchestration maximum concurrent executions observed.
        execution_order: List of execution events (if track_order is True).
        execution_delay: Time to sleep during each execution (simulates work).
        track_order: Whether to track execution order events.
        track_per_orch: Whether to track per-orchestration counts.
        lock: Thread lock for safe concurrent access.
    """

    def __init__(
        self,
        execution_delay: float = 0.05,
        track_order: bool = False,
        track_per_orch: bool = False,
        agent_type_value: AgentType = "claude",
    ) -> None:
        """Initialize the tracking agent client.

        Args:
            execution_delay: Time in seconds to sleep during each execution.
            track_order: If True, track start/end events in execution_order list.
            track_per_orch: If True, track per-orchestration execution counts.
            agent_type_value: The agent type this client simulates.
        """
        self.lock = threading.Lock()
        self.execution_count = 0
        self.max_concurrent_seen = 0
        self.orch_execution_counts: dict[str, int] = {}
        self.orch_max_concurrent: dict[str, int] = {}
        self.execution_order: list[str] = []
        self.execution_delay = execution_delay
        self.track_order = track_order
        self.track_per_orch = track_per_orch
        self._agent_type = agent_type_value

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return self._agent_type

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        with self.lock:
            # Track global execution count
            self.execution_count += 1
            if self.execution_count > self.max_concurrent_seen:
                self.max_concurrent_seen = self.execution_count

            # Track per-orchestration count if enabled
            if self.track_per_orch:
                orch_name = orchestration_name or "unknown"
                self.orch_execution_counts[orch_name] = (
                    self.orch_execution_counts.get(orch_name, 0) + 1
                )
                current = self.orch_execution_counts[orch_name]
                if current > self.orch_max_concurrent.get(orch_name, 0):
                    self.orch_max_concurrent[orch_name] = current

            # Track execution order if enabled
            if self.track_order:
                self.execution_order.append(f"start:{issue_key}")

        # Simulate work
        time.sleep(self.execution_delay)

        with self.lock:
            # Track execution order if enabled
            if self.track_order:
                self.execution_order.append(f"end:{issue_key}")

            # Decrement global execution count
            self.execution_count -= 1

            # Decrement per-orchestration count if enabled
            if self.track_per_orch:
                orch_name = orchestration_name or "unknown"
                self.orch_execution_counts[orch_name] = (
                    self.orch_execution_counts.get(orch_name, 0) - 1
                )

        return AgentRunResult(response="SUCCESS: Done", workdir=None)


class MockAgentClientFactory(AgentClientFactory):
    """Mock agent client factory for testing.

    DS-296: This factory allows tests to control which agent client is returned
    for different orchestrations without needing to set up real agent backends.

    The factory always returns the same mock client instance, making it easy
    to verify that the factory pattern is being used correctly.
    """

    def __init__(self, mock_client: AgentClient) -> None:
        """Initialize with a mock client to return for all requests.

        Args:
            mock_client: The mock client to return from create methods.
        """
        super().__init__()
        self._mock_client = mock_client

    def create(self, agent_type: AgentType | None, config: Config) -> AgentClient:
        """Return the mock client regardless of agent type."""
        return self._mock_client

    def get_or_create(
        self, agent_type: AgentType | None, config: Config
    ) -> AgentClient:
        """Return the mock client regardless of agent type."""
        return self._mock_client

    def create_for_orchestration(
        self,
        orch_agent_type: AgentType | None,
        config: Config,
        **kwargs: Any,
    ) -> AgentClient:
        """Return the mock client regardless of orchestration config."""
        return self._mock_client


def make_config(
    poll_interval: int = 60,
    max_issues: int = 50,
    max_concurrent_executions: int = 1,
    max_eager_iterations: int = 10,
    orchestrations_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
    max_queue_size: int = 100,
) -> Config:
    """Helper to create a Config for testing."""
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent_executions,
        max_eager_iterations=max_eager_iterations,
        orchestrations_dir=orchestrations_dir or Path("orchestrations"),
        attempt_counts_ttl=attempt_counts_ttl,
        max_queue_size=max_queue_size,
    )


def build_github_trigger_key(orch: Orchestration) -> str:
    """Build a deduplication key for a GitHub trigger.

    This helper mirrors the key-building logic from Sentinel._poll_github_triggers
    to ensure test assertions stay in sync with production code.

    DS-347: Extracted from duplicated definitions in TestGitHubTriggerDeduplication
    to improve test maintainability.

    The key format is: github:{project_owner}/{project_number}:{project_filter}:{labels}

    Args:
        orch: The orchestration containing a GitHub trigger configuration.

    Returns:
        A string key for deduplicating GitHub triggers.
    """
    filter_part = orch.trigger.project_filter or ""
    labels_part = ",".join(orch.trigger.labels) if orch.trigger.labels else ""
    return (
        f"github:{orch.trigger.project_owner}/{orch.trigger.project_number}"
        f":{filter_part}:{labels_part}"
    )


def make_orchestration(
    name: str = "test-orch",
    project: str = "TEST",
    tags: list[str] | None = None,
    max_concurrent: int | None = None,
    # DS-341: Added GitHub-specific fields for trigger deduplication tests
    source: Literal["jira", "github"] = "jira",
    project_number: int | None = None,
    project_owner: str = "",
    project_scope: Literal["org", "user"] = "org",
    project_filter: str = "",
    labels: list[str] | None = None,
) -> Orchestration:
    """Helper to create an Orchestration for testing.

    Args:
        name: The orchestration name.
        project: The Jira project key (for Jira triggers).
        tags: List of tags to trigger on (for Jira triggers).
        max_concurrent: Optional per-orchestration concurrency limit (DS-181).
        source: Trigger source, either "jira" or "github" (DS-341).
        project_number: GitHub project number (for GitHub triggers).
        project_owner: GitHub project owner (org or user name).
        project_scope: GitHub project scope ("org" or "user").
        project_filter: GitHub project filter expression.
        labels: List of GitHub labels to filter by.

    Returns:
        An Orchestration instance configured for testing.
    """
    # DS-341: Support both Jira and GitHub triggers
    if source == "github":
        trigger = TriggerConfig(
            source="github",
            project_number=project_number,
            project_owner=project_owner,
            project_scope=project_scope,
            project_filter=project_filter,
            labels=labels or [],
        )
    else:
        trigger = TriggerConfig(project=project, tags=tags or [])

    return Orchestration(
        name=name,
        trigger=trigger,
        agent=AgentConfig(prompt="Test prompt", tools=["jira"]),
        max_concurrent=max_concurrent,
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


@pytest.fixture
def tracking_agent_client_factory():
    """Factory fixture for creating TrackingAgentClient instances.

    DS-188: Provides a factory function to create TrackingAgentClient instances
    with different configurations, allowing tests to customize tracking behavior.

    Returns:
        A factory function that accepts optional parameters:
        - execution_delay: Time in seconds to sleep during each execution (default: 0.05)
        - track_order: If True, track start/end events in execution_order list
        - track_per_orch: If True, track per-orchestration execution counts

    Example:
        def test_concurrent_execution(tracking_agent_client_factory):
            client = tracking_agent_client_factory(track_order=True)
            # ... use client in test ...
            assert client.max_concurrent_seen <= expected_limit
    """

    def factory(
        execution_delay: float = 0.05,
        track_order: bool = False,
        track_per_orch: bool = False,
    ) -> TrackingAgentClient:
        return TrackingAgentClient(
            execution_delay=execution_delay,
            track_order=track_order,
            track_per_orch=track_per_orch,
        )

    return factory
