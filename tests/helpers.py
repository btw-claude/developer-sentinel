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

    from sentinel.agent_clients.base import UsageInfo
    from sentinel.router import Router
    from tests.mocks import MockAgentClient, MockAgentClientFactory, MockJiraPoller, MockRouter

from sentinel.config import (
    CircuitBreakerConfig,
    CodexConfig,
    Config,
    CursorConfig,
    DashboardConfig,
    ExecutionConfig,
    GitHubConfig,
    HealthCheckConfig,
    JiraConfig,
    LoggingConfig,
    PollingConfig,
    RateLimitConfig,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    RetryConfig,
    TriggerConfig,
)
from sentinel.poller import JiraIssue
from sentinel.types import AgentType


def make_config(
    # Polling settings
    poll_interval: int = 60,
    max_issues: int = 50,
    max_issues_per_poll: int | None = None,  # Alias for max_issues
    # Execution settings
    max_concurrent_executions: int = 1,
    orchestrations_dir: Path | None = None,
    agent_workdir: Path | None = None,
    agent_logs_dir: Path | None = None,
    orchestration_logs_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
    max_queue_size: int = 100,
    cleanup_workdir_on_success: bool = True,
    disable_streaming_logs: bool = False,
    subprocess_timeout: float = 60.0,
    default_base_branch: str = "main",
    inter_message_times_threshold: int = 100,
    # Logging settings
    log_level: str = "INFO",
    log_json: bool = False,
    # Dashboard settings
    dashboard_enabled: bool = False,
    dashboard_port: int = 8080,
    dashboard_host: str = "127.0.0.1",
    toggle_cooldown_seconds: float = 2.0,
    rate_limit_cache_ttl: int = 3600,
    rate_limit_cache_maxsize: int = 10000,
    # Jira settings
    jira_base_url: str = "",
    jira_email: str = "",
    jira_api_token: str = "",
    jira_epic_link_field: str = "customfield_10014",
    # GitHub settings
    github_token: str = "",
    github_api_url: str = "",
    # Agent settings
    default_agent_type: str = "claude",
    # Cursor settings
    cursor_path: str = "",
    cursor_default_model: str = "",
    cursor_default_mode: str = "agent",
    # Codex settings
    codex_path: str = "",
    codex_default_model: str = "",
    # Rate limit settings
    claude_rate_limit_enabled: bool = True,
    claude_rate_limit_per_minute: int = 60,
    claude_rate_limit_per_hour: int = 1000,
    claude_rate_limit_strategy: str = "queue",
    claude_rate_limit_warning_threshold: float = 0.2,
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_recovery_timeout: float = 30.0,
    circuit_breaker_half_open_max_calls: int = 3,
    # Health check settings
    health_check_enabled: bool = True,
    health_check_timeout: float = 5.0,
    # Shutdown settings
    shutdown_timeout_seconds: float = 300.0,
) -> Config:
    """Create a Config instance for testing with sensible defaults.

    This test helper constructs Config directly with sub-config objects,
    providing convenient defaults for all configuration options. Use this
    function in tests when you need a Config object but only care about
    specific settings.

    The function is organized into parameter groups matching Config's sub-configs:

    **Polling Settings** - Control how often and how many issues are fetched:
        poll_interval: Seconds between polling cycles (default: 60).
        max_issues: Maximum issues to fetch per poll (default: 50).
        max_issues_per_poll: Alias for max_issues (takes precedence if set).

    **Execution Settings** - Control agent execution behavior:
        max_concurrent_executions: Parallel orchestration limit (default: 1).
        orchestrations_dir: Directory for orchestration files.
        agent_workdir: Base directory for agent working directories.
        agent_logs_dir: Base directory for agent execution logs.
        orchestration_logs_dir: Per-orchestration log directory.
        attempt_counts_ttl: TTL for retry attempt tracking (default: 3600s).
        max_queue_size: Maximum queued issues (default: 100).
        cleanup_workdir_on_success: Remove workdir after success (default: True).
        disable_streaming_logs: Disable real-time log writes (default: False).
        subprocess_timeout: Default subprocess timeout (default: 60.0s).
        default_base_branch: Default git base branch (default: "main").
        inter_message_times_threshold: Timing metrics threshold (default: 100).

    **Logging Settings** - Control log output:
        log_level: Log level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO).
        log_json: Output logs in JSON format (default: False).

    **Dashboard Settings** - Control the web dashboard:
        dashboard_enabled: Enable web dashboard (default: False).
        dashboard_port: Dashboard server port (default: 8080).
        dashboard_host: Dashboard server host (default: "127.0.0.1").
        toggle_cooldown_seconds: Cooldown between file toggle writes (default: 2.0).
        rate_limit_cache_ttl: Rate limit cache TTL (default: 3600s).
        rate_limit_cache_maxsize: Rate limit cache max entries (default: 10000).

    **Jira Settings** - Jira REST API configuration:
        jira_base_url: Jira instance URL (default: "").
        jira_email: Authentication email (default: "").
        jira_api_token: Authentication token (default: "").
        jira_epic_link_field: Epic link custom field ID (default: "customfield_10014").

    **GitHub Settings** - GitHub REST API configuration:
        github_token: Personal access token (default: "").
        github_api_url: GitHub Enterprise API URL (default: "").

    **Agent Settings** - Default agent type configuration:
        default_agent_type: Agent type - "claude", "codex", or "cursor" (default: "claude").

    **Cursor Settings** - Cursor CLI configuration:
        cursor_path: Path to Cursor CLI executable (default: "").
        cursor_default_model: Default Cursor model (default: "").
        cursor_default_mode: Cursor mode - agent, plan, ask (default: "agent").

    **Codex Settings** - Codex CLI configuration:
        codex_path: Path to Codex CLI executable (default: "").
        codex_default_model: Default Codex model (default: "").

    **Rate Limit Settings** - Claude API rate limiting:
        claude_rate_limit_enabled: Enable rate limiting (default: True).
        claude_rate_limit_per_minute: Max requests/minute (default: 60).
        claude_rate_limit_per_hour: Max requests/hour (default: 1000).
        claude_rate_limit_strategy: Strategy - "queue" or "reject" (default: "queue").
        claude_rate_limit_warning_threshold: Warning threshold 0.0-1.0 (default: 0.2).

    **Circuit Breaker Settings** - External service circuit breakers:
        circuit_breaker_enabled: Enable circuit breakers (default: True).
        circuit_breaker_failure_threshold: Failures to open circuit (default: 5).
        circuit_breaker_recovery_timeout: Recovery wait time (default: 30.0s).
        circuit_breaker_half_open_max_calls: Half-open state max calls (default: 3).

    **Health Check Settings** - External dependency health checks:
        health_check_enabled: Enable health checks (default: True).
        health_check_timeout: Health check timeout (default: 5.0s).

    **Shutdown Settings** - Graceful shutdown configuration:
        shutdown_timeout_seconds: Timeout in seconds for graceful shutdown (default: 300.0).
            If executions don't complete within this time, they will be forcefully
            terminated. Set to 0 to wait indefinitely.

    Returns:
        A Config instance with the specified parameters organized into sub-configs.

    Example::

        # Create config with all defaults
        config = make_config()

        # Override specific settings
        config = make_config(
            max_concurrent_executions=5,
            log_level="DEBUG",
            dashboard_enabled=True,
        )

        # Configure for integration testing with real services
        config = make_config(
            jira_base_url="https://test.atlassian.net",
            jira_email="test@example.com",
            jira_api_token="test-token",
        )
    """
    # Support both max_issues and max_issues_per_poll parameter names
    effective_max_issues = max_issues_per_poll if max_issues_per_poll is not None else max_issues

    return Config(
        polling=PollingConfig(
            interval=poll_interval,
            max_issues_per_poll=effective_max_issues,
        ),
        execution=ExecutionConfig(
            max_concurrent_executions=max_concurrent_executions,
            orchestrations_dir=orchestrations_dir or Path("./orchestrations"),
            agent_workdir=agent_workdir or Path("./workdir"),
            agent_logs_dir=agent_logs_dir or Path("./logs"),
            orchestration_logs_dir=orchestration_logs_dir,
            attempt_counts_ttl=attempt_counts_ttl,
            max_queue_size=max_queue_size,
            cleanup_workdir_on_success=cleanup_workdir_on_success,
            disable_streaming_logs=disable_streaming_logs,
            subprocess_timeout=subprocess_timeout,
            default_base_branch=default_base_branch,
            inter_message_times_threshold=inter_message_times_threshold,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        ),
        logging_config=LoggingConfig(
            level=log_level,
            json=log_json,
        ),
        dashboard=DashboardConfig(
            enabled=dashboard_enabled,
            port=dashboard_port,
            host=dashboard_host,
            toggle_cooldown_seconds=toggle_cooldown_seconds,
            rate_limit_cache_ttl=rate_limit_cache_ttl,
            rate_limit_cache_maxsize=rate_limit_cache_maxsize,
        ),
        jira=JiraConfig(
            base_url=jira_base_url,
            email=jira_email,
            api_token=jira_api_token,
            epic_link_field=jira_epic_link_field,
        ),
        github=GitHubConfig(
            token=github_token,
            api_url=github_api_url,
        ),
        cursor=CursorConfig(
            default_agent_type=default_agent_type,
            path=cursor_path,
            default_model=cursor_default_model,
            default_mode=cursor_default_mode,
        ),
        codex=CodexConfig(
            path=codex_path,
            default_model=codex_default_model,
        ),
        rate_limit=RateLimitConfig(
            enabled=claude_rate_limit_enabled,
            per_minute=claude_rate_limit_per_minute,
            per_hour=claude_rate_limit_per_hour,
            strategy=claude_rate_limit_strategy,
            warning_threshold=claude_rate_limit_warning_threshold,
        ),
        circuit_breaker=CircuitBreakerConfig(
            enabled=circuit_breaker_enabled,
            failure_threshold=circuit_breaker_failure_threshold,
            recovery_timeout=circuit_breaker_recovery_timeout,
            half_open_max_calls=circuit_breaker_half_open_max_calls,
        ),
        health_check=HealthCheckConfig(
            enabled=health_check_enabled,
            timeout=health_check_timeout,
        ),
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
    github: GitHubContext | None = None,
    max_attempts: int = 3,
    success_patterns: list[str] | None = None,
    failure_patterns: list[str] | None = None,
    strict_template_variables: bool = False,
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
        github: GitHub context for the agent (host, org, repo, branch).
        max_attempts: Maximum retry attempts for the agent.
        success_patterns: Patterns that indicate success in agent response.
        failure_patterns: Patterns that indicate failure in agent response.
        strict_template_variables: If True, raise ValueError for unknown template
            variables instead of logging a warning.

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
            github=github,
            strict_template_variables=strict_template_variables,
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


def make_agent_factory(
    responses: list[str] | None = None,
    workdir: Path | None = None,
    agent_type_value: AgentType = AgentType.CLAUDE,
    usage: UsageInfo | None = None,
) -> tuple[MockAgentClientFactory, MockAgentClient]:
    """Create a MockAgentClientFactory with a wrapped MockAgentClient for testing.

    This helper consolidates the common 2-line pattern::

        agent_client = MockAgentClient()
        agent_factory = MockAgentClientFactory(agent_client)

    Into a single call::

        agent_factory, agent_client = make_agent_factory()

    For tests that don't need access to the underlying client::

        agent_factory, _ = make_agent_factory()

    Args:
        responses: List of response strings for MockAgentClient.
        workdir: Optional workdir path for MockAgentClient.
        agent_type_value: The agent type to use (default: AgentType.CLAUDE).
        usage: Optional UsageInfo for MockAgentClient.

    Returns:
        A tuple of (MockAgentClientFactory, MockAgentClient) for use in tests.
    """
    from tests.mocks import MockAgentClient, MockAgentClientFactory

    agent_client = MockAgentClient(
        responses=responses,
        workdir=workdir,
        agent_type_value=agent_type_value,
        usage=usage,
    )
    agent_factory = MockAgentClientFactory(agent_client)
    return agent_factory, agent_client
