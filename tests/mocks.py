"""Mock classes for Developer Sentinel tests.

This module provides reusable mock implementations of core interfaces used in testing.
These mocks enable isolated unit testing without requiring real API connections.

Usage Guidelines
================

**Direct instantiation** is the preferred approach for most tests::

    from tests.mocks import MockJiraClient, MockAgentClient, MockTagClient

    def test_example():
        jira_client = MockJiraClient(issues=[...])
        agent_client = MockAgentClient(responses=["SUCCESS"])
        tag_client = MockTagClient()
        # ... use in test ...

Rationale:

- Direct instantiation is explicit and makes test setup clearer
- Tests don't need to rely on pytest's fixture dependency injection
- The class parameters are immediately visible at the call site
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from sentinel.agent_clients.base import AgentType
from sentinel.agent_clients.factory import AgentClientFactory
from sentinel.config import Config
from sentinel.executor import AgentClient, AgentRunResult
from sentinel.poller import JiraClient
from sentinel.tag_manager import JiraTagClient


class MockJiraClient(JiraClient):
    """Mock Jira client for testing.

    Simulates Jira API responses for search operations without making real API calls.

    Args:
        issues: List of issue dicts to return from search_issues.
        tag_client: Optional MockTagClient to coordinate label removal filtering.

    Attributes:
        issues: The issues that will be returned by search_issues.
        search_calls: List of (jql, max_results) tuples recording each search call.
        tag_client: Reference to MockTagClient for filtering logic.
    """

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
    """Mock agent client for testing.

    Implements the async AgentClient interface for use in tests. Returns
    pre-configured responses without executing actual agent operations.

    Supports error and timeout simulation for testing error handling paths.

    Args:
        responses: List of response strings to return. Cycles through if needed.
        workdir: Optional workdir path to return in results.
        agent_type_value: The agent type this mock represents.

    Attributes:
        responses: The responses to return from run_agent.
        workdir: The workdir to return in AgentRunResult.
        call_count: Number of times run_agent was called.
        calls: List of call arguments for verification.
        should_error: If True, raises AgentClientError (up to max_errors times).
        error_count: Number of errors raised so far.
        max_errors: Maximum number of errors to raise before succeeding.
        should_timeout: If True, raises AgentTimeoutError (up to max_timeouts times).
        timeout_count: Number of timeouts raised so far.
        max_timeouts: Maximum number of timeouts to raise before succeeding.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        workdir: "Path | None" = None,
        agent_type_value: AgentType = "claude",
    ) -> None:
        self.responses = responses or ["SUCCESS: Task completed"]
        self.workdir = workdir
        self.call_count = 0
        self.calls: list[
            tuple[str, list[str], dict[str, Any] | None, int | None, str | None, str | None]
        ] = []
        self.should_error = False
        self.error_count = 0
        self.max_errors = 0
        self.should_timeout = False
        self.timeout_count = 0
        self.max_timeouts = 0
        self._agent_type = agent_type_value

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return self._agent_type

    async def run_agent(
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
        """Async mock implementation of run_agent."""
        from sentinel.executor import AgentClientError, AgentTimeoutError

        self.calls.append((prompt, tools, context, timeout_seconds, issue_key, model))

        if self.should_timeout and self.timeout_count < self.max_timeouts:
            self.timeout_count += 1
            raise AgentTimeoutError(f"Agent timed out after {timeout_seconds}s")

        if self.should_error and self.error_count < self.max_errors:
            self.error_count += 1
            raise AgentClientError("Mock agent error")

        # Response cycling: returns responses in order until exhausted, then repeats the last one.
        # The min() ensures we don't index past the end of the responses list.
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return AgentRunResult(response=response, workdir=self.workdir)


class MockTagClient(JiraTagClient):
    """Mock tag client for testing.

    Tracks label add/remove operations without making real API calls.

    Attributes:
        labels: Dict mapping issue keys to their current labels.
        add_calls: List of (issue_key, label) tuples for add operations.
        remove_calls: List of (issue_key, label) tuples for remove operations.
    """

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

    Provides flexible tracking of concurrent executions with support for:
    - Global execution count tracking
    - Per-orchestration execution count tracking
    - Configurable execution delay
    - Optional execution order tracking

    Args:
        execution_delay: Time to sleep during each execution (simulates work).
        track_order: Whether to track execution order events.
        track_per_orch: Whether to track per-orchestration counts.
        agent_type_value: The agent type this client simulates.

    Attributes:
        execution_count: Current number of concurrent executions (global).
        max_concurrent_seen: Maximum concurrent executions observed (global).
        orch_execution_counts: Per-orchestration current execution counts.
        orch_max_concurrent: Per-orchestration maximum concurrent executions observed.
        execution_order: List of execution events (if track_order is True).
        lock: Thread lock for safe concurrent access.
    """

    def __init__(
        self,
        execution_delay: float = 0.05,
        track_order: bool = False,
        track_per_orch: bool = False,
        agent_type_value: AgentType = "claude",
    ) -> None:
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

    async def run_agent(
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
        """Async mock implementation of run_agent with tracking."""
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

    This factory allows tests to control which agent client is returned
    for different orchestrations without needing to set up real agent backends.

    The factory always returns the same mock client instance, making it easy
    to verify that the factory pattern is being used correctly.

    Args:
        mock_client: The mock client to return from create methods.
    """

    def __init__(self, mock_client: AgentClient) -> None:
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
