"""Test helper functions for Developer Sentinel tests.

This module also provides helper functions for creating Sentinel instances
with dependency injection, making it easier to write tests that don't rely
on deprecated constructor parameters.

This module provides utility functions for creating test fixtures and
managing test data. These helpers simplify test setup by providing
sensible defaults while allowing customization.

Usage
=====

Import and use helpers directly in tests::

    from tests.helpers import (
        assert_call_args_length,
        make_config,
        make_orchestration,
        make_issue,
        set_mtime_in_future,
    )

    def test_example():
        config = make_config(max_concurrent_executions=5)
        orch = make_orchestration(name="test", tags=["review"])
        issue = make_issue(key="TEST-1", summary="Test issue")
        # ... use in test ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from unittest.mock import _Call

    from sentinel.router import Router
    from tests.mocks import MockJiraPoller, MockRouter

from sentinel.config import Config
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    RetryConfig,
    TriggerConfig,
)
from sentinel.poller import JiraIssue


def make_config(
    poll_interval: int = 60,
    max_issues: int = 50,
    max_concurrent_executions: int = 1,
    orchestrations_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
    max_queue_size: int = 100,
) -> Config:
    """Create a Config instance for testing.

    Provides sensible defaults for all config options, with the ability
    to override any parameter.

    Args:
        poll_interval: Seconds between polling cycles.
        max_issues: Maximum issues to fetch per poll.
        max_concurrent_executions: Global concurrency limit.
        orchestrations_dir: Directory containing orchestration files.
        attempt_counts_ttl: TTL for attempt counts cache.
        max_queue_size: Maximum size of the execution queue.

    Returns:
        A Config instance with the specified parameters.
    """
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent_executions,
        orchestrations_dir=orchestrations_dir or Path("orchestrations"),
        attempt_counts_ttl=attempt_counts_ttl,
        max_queue_size=max_queue_size,
    )


def make_issue(
    key: str = "TEST-1",
    summary: str = "Test issue",
    description: str = "",
    status: str = "",
    assignee: str | None = None,
    labels: list[str] | None = None,
    comments: list[str] | None = None,
    links: list[str] | None = None,
    epic_key: str | None = None,
    parent_key: str | None = None,
) -> JiraIssue:
    """Create a JiraIssue instance for testing.

    Provides sensible defaults for all issue fields, with the ability
    to override any parameter.

    Args:
        key: The Jira issue key (e.g., "TEST-1").
        summary: Issue summary/title.
        description: Issue description text.
        status: Issue status (e.g., "Open", "In Progress").
        assignee: Display name of the assignee.
        labels: List of labels on the issue.
        comments: List of comment texts.
        links: List of linked issue keys.
        epic_key: Parent epic key (if issue is linked to an epic).
        parent_key: Parent issue key (for sub-tasks).

    Returns:
        A JiraIssue instance with the specified parameters.
    """
    return JiraIssue(
        key=key,
        summary=summary,
        description=description,
        status=status,
        assignee=assignee,
        labels=labels or [],
        comments=comments or [],
        links=links or [],
        epic_key=epic_key,
        parent_key=parent_key,
    )


def make_orchestration(
    name: str = "test-orch",
    project: str = "TEST",
    tags: list[str] | None = None,
    max_concurrent: int | None = None,
    source: Literal["jira", "github"] = "jira",
    project_number: int | None = None,
    project_owner: str = "",
    project_scope: Literal["org", "user"] = "org",
    project_filter: str = "",
    labels: list[str] | None = None,
    prompt: str = "Test prompt",
    tools: list[str] | None = None,
    github: GitHubContext | None = None,
    max_attempts: int = 3,
    success_patterns: list[str] | None = None,
    failure_patterns: list[str] | None = None,
) -> Orchestration:
    """Create an Orchestration instance for testing.

    Supports both Jira and GitHub trigger sources, with optional retry
    configuration for executor tests.

    Args:
        name: The orchestration name.
        project: The Jira project key (for Jira triggers).
        tags: List of tags to trigger on (for Jira triggers).
        max_concurrent: Optional per-orchestration concurrency limit.
        source: Trigger source, either "jira" or "github".
        project_number: GitHub project number (for GitHub triggers).
        project_owner: GitHub project owner (org or user name).
        project_scope: GitHub project scope ("org" or "user").
        project_filter: GitHub project filter expression.
        labels: List of GitHub labels to filter by.
        prompt: The agent prompt template.
        tools: List of tools to enable for the agent.
        github: GitHub context for the agent (host, org, repo, branch).
        max_attempts: Maximum retry attempts for the agent.
        success_patterns: Patterns that indicate success in agent response.
        failure_patterns: Patterns that indicate failure in agent response.

    Returns:
        An Orchestration instance configured for testing.
    """
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
        agent=AgentConfig(
            prompt=prompt,
            tools=tools or ["jira"],
            github=github,
        ),
        max_concurrent=max_concurrent,
        retry=RetryConfig(
            max_attempts=max_attempts,
            success_patterns=success_patterns or ["SUCCESS", "completed successfully"],
            failure_patterns=failure_patterns or ["FAILURE", "failed", "error"],
        ),
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


def assert_call_args_length(call_args: _Call, min_length: int) -> None:
    """Assert that call_args tuple has at least min_length positional arguments.

    This helper provides clearer error messages when checking mock call arguments
    by including the expected and actual lengths in the assertion message.

    Args:
        call_args: The call_args from a mock, typically mock.call_args.
        min_length: The minimum number of positional arguments expected.

    Raises:
        AssertionError: If call_args[0] has fewer than min_length elements.

    Example::

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args
        assert_call_args_length(call_args, 5)
        assert "message format" in call_args[0][0]
    """
    assert (
        len(call_args[0]) >= min_length
    ), f"Expected at least {min_length} positional args, got {len(call_args[0])}"


def make_jira_poller(
    issues: list[JiraIssue] | None = None,
) -> MockJiraPoller:
    """Create a MockJiraPoller for testing.

    Args:
        issues: List of JiraIssue objects to return from poll.

    Returns:
        A MockJiraPoller configured with the specified issues.
    """
    from tests.mocks import MockJiraPoller

    return MockJiraPoller(issues=issues)


def make_router(orchestrations: list[Orchestration] | None = None) -> Router:
    """Create a Router for testing.

    Args:
        orchestrations: List of orchestrations to configure the router with.

    Returns:
        A Router configured with the specified orchestrations.
    """
    from sentinel.router import Router

    return Router(orchestrations or [])


def make_mock_router(
    orchestrations: list[Orchestration] | None = None,
    route_results: dict[str, list[Orchestration]] | None = None,
) -> MockRouter:
    """Create a MockRouter for testing with more control over routing behavior.

    Unlike make_router() which creates a real Router, this function creates a
    MockRouter that allows tests to:
    - Track which issues were routed via route_calls
    - Override routing results for specific issue keys via route_results
    - Still delegate to real routing logic when no override is specified

    Args:
        orchestrations: List of orchestrations to configure the router with.
        route_results: Optional dict mapping issue keys to lists of orchestrations
            that should be returned as matches. When an issue key is in this dict,
            the MockRouter returns those orchestrations instead of performing
            real routing logic.

    Returns:
        A MockRouter configured with the specified orchestrations and route results.

    Example::

        # Create a mock router that returns specific orchestrations for TEST-1
        orch: Orchestration = make_orchestration(name="test")
        router: MockRouter = make_mock_router(route_results={"TEST-1": [orch]})

        # Route an issue - will use custom result
        issue: JiraIssue = make_issue(key="TEST-1")
        result: RoutingResult = router.route(issue)
        assert result.orchestrations == [orch]

        # Verify routing was called
        assert len(router.route_calls) == 1
    """
    from tests.mocks import MockRouter

    return MockRouter(orchestrations=orchestrations, route_results=route_results)
