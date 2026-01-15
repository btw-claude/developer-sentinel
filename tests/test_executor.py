"""Tests for Claude Agent SDK executor module."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sentinel.executor import (
    AgentClient,
    AgentClientError,
    AgentExecutor,
    AgentRunResult,
    AgentTimeoutError,
    ExecutionResult,
    ExecutionStatus,
    cleanup_workdir,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    Outcome,
    RetryConfig,
    TriggerConfig,
)
from sentinel.poller import JiraIssue


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(
        self,
        responses: list[str] | None = None,
        workdir: Path | None = None,
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
        self.calls.append((prompt, tools, context, timeout_seconds, issue_key, model))

        if self.should_timeout and self.timeout_count < self.max_timeouts:
            self.timeout_count += 1
            raise AgentTimeoutError(f"Agent timed out after {timeout_seconds}s")

        if self.should_error and self.error_count < self.max_errors:
            self.error_count += 1
            raise AgentClientError("Mock agent error")

        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return AgentRunResult(response=response, workdir=self.workdir)


def make_issue(
    key: str = "TEST-1",
    summary: str = "Test issue",
    description: str = "",
    status: str = "",
    assignee: str | None = None,
    labels: list[str] | None = None,
    comments: list[str] | None = None,
    links: list[str] | None = None,
) -> JiraIssue:
    """Helper to create a JiraIssue for testing."""
    return JiraIssue(
        key=key,
        summary=summary,
        description=description,
        status=status,
        assignee=assignee,
        labels=labels or [],
        comments=comments or [],
        links=links or [],
    )


def make_orchestration(
    name: str = "test-orch",
    prompt: str = "Process this issue",
    tools: list[str] | None = None,
    github: GitHubContext | None = None,
    max_attempts: int = 3,
    success_patterns: list[str] | None = None,
    failure_patterns: list[str] | None = None,
) -> Orchestration:
    """Helper to create an Orchestration for testing."""
    return Orchestration(
        name=name,
        trigger=TriggerConfig(),
        agent=AgentConfig(
            prompt=prompt,
            tools=tools or ["jira"],
            github=github,
        ),
        retry=RetryConfig(
            max_attempts=max_attempts,
            success_patterns=success_patterns or ["SUCCESS", "completed successfully"],
            failure_patterns=failure_patterns or ["FAILURE", "failed", "error"],
        ),
    )


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_succeeded_when_success(self) -> None:
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            response="Done",
            attempts=1,
            issue_key="TEST-1",
            orchestration_name="test",
        )
        assert result.succeeded is True

    def test_not_succeeded_when_failure(self) -> None:
        result = ExecutionResult(
            status=ExecutionStatus.FAILURE,
            response="Failed",
            attempts=1,
            issue_key="TEST-1",
            orchestration_name="test",
        )
        assert result.succeeded is False

    def test_not_succeeded_when_error(self) -> None:
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            response="Error",
            attempts=1,
            issue_key="TEST-1",
            orchestration_name="test",
        )
        assert result.succeeded is False


class TestAgentExecutorBuildPrompt:
    """Tests for AgentExecutor.build_prompt with template variables."""

    def test_returns_prompt_without_variables_unchanged(self) -> None:
        """Prompt without template variables should be returned as-is."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(prompt="Review this code without any variables")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review this code without any variables"

    def test_substitutes_jira_issue_key(self) -> None:
        """Should substitute {jira_issue_key} with actual issue key."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-123")
        orch = make_orchestration(prompt="Review issue {jira_issue_key}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review issue PROJ-123"

    def test_substitutes_jira_summary(self) -> None:
        """Should substitute {jira_summary} with issue summary."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(summary="Fix the authentication bug")
        orch = make_orchestration(prompt="Task: {jira_summary}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Task: Fix the authentication bug"

    def test_substitutes_jira_description(self) -> None:
        """Should substitute {jira_description} with issue description."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(description="Detailed description here")
        orch = make_orchestration(prompt="Description:\n{jira_description}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Description:\nDetailed description here"

    def test_substitutes_jira_status(self) -> None:
        """Should substitute {jira_status} with issue status."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(status="In Progress")
        orch = make_orchestration(prompt="Status: {jira_status}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Status: In Progress"

    def test_substitutes_jira_assignee(self) -> None:
        """Should substitute {jira_assignee} with assignee name."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(assignee="John Doe")
        orch = make_orchestration(prompt="Assigned to: {jira_assignee}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Assigned to: John Doe"

    def test_substitutes_jira_labels(self) -> None:
        """Should substitute {jira_labels} with comma-separated labels."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(labels=["bug", "urgent", "security"])
        orch = make_orchestration(prompt="Labels: {jira_labels}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Labels: bug, urgent, security"

    def test_substitutes_jira_comments(self) -> None:
        """Should substitute {jira_comments} with formatted comments."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(comments=["First comment", "Second comment"])
        orch = make_orchestration(prompt="Comments:\n{jira_comments}")

        prompt = executor.build_prompt(issue, orch)

        assert "1. First comment" in prompt
        assert "2. Second comment" in prompt

    def test_substitutes_jira_links(self) -> None:
        """Should substitute {jira_links} with comma-separated linked issues."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(links=["TEST-2", "TEST-3"])
        orch = make_orchestration(prompt="Related: {jira_links}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Related: TEST-2, TEST-3"

    def test_substitutes_github_variables(self) -> None:
        """Should substitute GitHub context variables."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(
            prompt="Repo: {github_org}/{github_repo} on {github_host}",
            github=GitHubContext(host="github.example.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Repo: myorg/myrepo on github.example.com"

    def test_substitutes_multiple_variables(self) -> None:
        """Should substitute multiple variables in one prompt."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-456", summary="Update docs", status="Open")
        orch = make_orchestration(
            prompt="Review {jira_issue_key}: {jira_summary} (Status: {jira_status})"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review DS-456: Update docs (Status: Open)"

    def test_empty_values_substituted_as_empty_string(self) -> None:
        """Missing optional values should be substituted as empty strings."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(description="", assignee=None)
        orch = make_orchestration(prompt="Desc: [{jira_description}] Assignee: [{jira_assignee}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Desc: [] Assignee: []"

    def test_preserves_unknown_variables(self) -> None:
        """Unknown variables should be preserved as-is."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="TEST-1")
        orch = make_orchestration(prompt="Issue {jira_issue_key} with {unknown_var}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Issue TEST-1 with {unknown_var}"

    def test_handles_literal_braces(self) -> None:
        """Should handle text that looks like variables but isn't."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="TEST-1")
        orch = make_orchestration(prompt='Use JSON format: {"key": "value"}')

        prompt = executor.build_prompt(issue, orch)

        # JSON-like braces are preserved since they don't match {word} pattern
        assert '{"key": "value"}' in prompt

    def test_truncates_long_comments(self) -> None:
        """Should truncate comments longer than 500 characters."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        long_comment = "x" * 600
        issue = make_issue(comments=[long_comment])
        orch = make_orchestration(prompt="{jira_comments}")

        prompt = executor.build_prompt(issue, orch)

        assert len(prompt) < 600
        assert "..." in prompt

    def test_limits_to_last_three_comments(self) -> None:
        """Should only include last 3 comments."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(comments=["First", "Second", "Third", "Fourth", "Fifth"])
        orch = make_orchestration(prompt="{jira_comments}")

        prompt = executor.build_prompt(issue, orch)

        assert "First" not in prompt
        assert "Second" not in prompt
        assert "Third" in prompt
        assert "Fourth" in prompt
        assert "Fifth" in prompt


class TestAgentExecutorMatchesPattern:
    """Tests for AgentExecutor._matches_pattern."""

    def test_matches_simple_substring(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("Task completed SUCCESS", ["SUCCESS"]) is True

    def test_matches_case_insensitive(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("task success", ["SUCCESS"]) is True

    def test_no_match_returns_false(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("Task completed", ["SUCCESS"]) is False

    def test_matches_any_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("Done!", ["SUCCESS", "DONE"]) is True

    def test_matches_regex_pattern_with_prefix(self) -> None:
        """Regex patterns require explicit 'regex:' prefix."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        # With regex: prefix, pattern is treated as regex
        assert executor._matches_pattern("error code 123", ["regex:error.*\\d+"]) is True

    def test_regex_pattern_anchors(self) -> None:
        """Test regex patterns with anchors."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        # Start anchor
        assert executor._matches_pattern("Task completed", ["regex:^Task"]) is True
        assert executor._matches_pattern("My Task completed", ["regex:^Task"]) is False

        # End anchor
        assert executor._matches_pattern("Task completed", ["regex:completed$"]) is True
        assert executor._matches_pattern("Task completed!", ["regex:completed$"]) is False

    def test_pattern_without_prefix_is_substring(self) -> None:
        """Patterns without 'regex:' prefix are treated as substrings."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        # Without prefix, '*' is matched literally as a substring
        assert executor._matches_pattern("rating: 5*", ["5*"]) is True
        # This would NOT match as regex since there's no 'regex:' prefix
        assert executor._matches_pattern("555", ["5*"]) is False

    def test_invalid_regex_falls_back_to_substring(self) -> None:
        """Invalid regex patterns fall back to substring matching."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        # Invalid regex (unbalanced bracket) falls back to substring
        assert executor._matches_pattern("error [unclosed", ["regex:[unclosed"]) is True


class TestAgentExecutorDetermineStatus:
    """Tests for AgentExecutor._determine_status."""

    def test_returns_success_on_success_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status, outcome = executor._determine_status("Task SUCCESS", retry_config)

        assert status == ExecutionStatus.SUCCESS
        assert outcome is None  # No outcomes configured

    def test_returns_failure_on_failure_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status, outcome = executor._determine_status("Task FAILURE", retry_config)

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_success_takes_precedence(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        # Response contains both patterns
        status, outcome = executor._determine_status("SUCCESS but had FAILURE", retry_config)

        assert status == ExecutionStatus.SUCCESS
        assert outcome is None

    def test_defaults_to_success_when_no_match(self) -> None:
        """Default behavior: returns SUCCESS when no patterns match."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status, outcome = executor._determine_status("Task done", retry_config)

        assert status == ExecutionStatus.SUCCESS
        assert outcome is None

    def test_default_status_success_explicit(self) -> None:
        """Explicitly configured default_status='success' returns SUCCESS."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
            default_status="success",
        )

        status, outcome = executor._determine_status("Task done", retry_config)

        assert status == ExecutionStatus.SUCCESS
        assert outcome is None

    def test_default_status_failure(self) -> None:
        """Configured default_status='failure' returns FAILURE when no patterns match."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
            default_status="failure",
        )

        status, outcome = executor._determine_status("Task done", retry_config)

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_default_status_only_used_when_no_match(self) -> None:
        """default_status is ignored when patterns actually match."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        # Even with default_status='failure', explicit success pattern wins
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
            default_status="failure",
        )

        status, outcome = executor._determine_status("Task SUCCESS", retry_config)

        assert status == ExecutionStatus.SUCCESS
        assert outcome is None


class TestAgentExecutorDetermineStatusWithOutcomes:
    """Tests for AgentExecutor._determine_status with outcomes configured."""

    def test_matches_outcome_pattern(self) -> None:
        """Should match outcome pattern and return outcome name."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"])
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED", "LGTM"]),
            Outcome(name="changes-requested", patterns=["CHANGES REQUESTED"]),
        ]

        status, outcome = executor._determine_status(
            "Code review: APPROVED", retry_config, outcomes
        )

        assert status == ExecutionStatus.SUCCESS
        assert outcome == "approved"

    def test_matches_second_outcome(self) -> None:
        """Should match second outcome when first doesn't match."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"])
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
            Outcome(name="changes-requested", patterns=["CHANGES REQUESTED"]),
        ]

        status, outcome = executor._determine_status(
            "Review: CHANGES REQUESTED", retry_config, outcomes
        )

        assert status == ExecutionStatus.SUCCESS
        assert outcome == "changes-requested"

    def test_outcome_takes_precedence_over_failure_patterns(self) -> None:
        """Outcome patterns should take precedence over failure patterns.

        This is important because agents often mention words like 'error' in context
        (e.g., 'fixed the error', 'Error reference ID') while still succeeding.
        Explicit outcome keywords like 'SUCCESS' or 'APPROVED' indicate the agent's
        deliberate signal about the result.
        """
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["error", "failed"])
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED", "SUCCESS"]),
        ]

        # Response mentions "error" in context but ends with explicit SUCCESS
        status, outcome = executor._determine_status(
            "Fixed the error in the code. All tests pass. SUCCESS", retry_config, outcomes
        )

        assert status == ExecutionStatus.SUCCESS
        assert outcome == "approved"

    def test_failure_patterns_trigger_when_no_outcome_matches(self) -> None:
        """Failure patterns should trigger when no outcome pattern matches."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR", "FAILED"])
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
        ]

        # Response has failure pattern but no outcome pattern
        status, outcome = executor._determine_status(
            "ERROR: Could not access repo", retry_config, outcomes
        )

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_default_outcome_when_no_match(self) -> None:
        """Should use default_outcome when no outcome patterns match."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            failure_patterns=["ERROR"],
            default_outcome="approved",
        )
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
            Outcome(name="changes-requested", patterns=["CHANGES REQUESTED"]),
        ]

        status, outcome = executor._determine_status(
            "Review complete with no issues", retry_config, outcomes
        )

        assert status == ExecutionStatus.SUCCESS
        assert outcome == "approved"

    def test_returns_failure_when_no_match_and_no_default(self) -> None:
        """Should return FAILURE when no patterns match and no default_outcome."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"])
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
            Outcome(name="changes-requested", patterns=["CHANGES REQUESTED"]),
        ]

        status, outcome = executor._determine_status("Review complete", retry_config, outcomes)

        # No match and no default = failure (triggers retry/on_failure)
        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_default_outcome_failure_keyword(self) -> None:
        """Should return FAILURE when default_outcome is 'failure'."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"], default_outcome="failure")
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
        ]

        status, outcome = executor._determine_status("Review complete", retry_config, outcomes)

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_default_outcome_failure_case_insensitive(self) -> None:
        """Should handle 'FAILURE' keyword case-insensitively."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"], default_outcome="FAILURE")
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
        ]

        status, outcome = executor._determine_status("Review complete", retry_config, outcomes)

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_invalid_default_outcome_returns_failure(self) -> None:
        """Should return FAILURE when default_outcome doesn't exist in outcomes."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"], default_outcome="nonexistent")
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED"]),
        ]

        status, outcome = executor._determine_status("Review complete", retry_config, outcomes)

        assert status == ExecutionStatus.FAILURE
        assert outcome is None

    def test_outcome_with_regex_pattern(self) -> None:
        """Should match regex patterns in outcomes."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"])
        outcomes = [
            Outcome(name="approved", patterns=["regex:^APPROVED"]),
        ]

        status, outcome = executor._determine_status(
            "APPROVED: Code looks good", retry_config, outcomes
        )

        assert status == ExecutionStatus.SUCCESS
        assert outcome == "approved"

    def test_outcome_regex_no_match(self) -> None:
        """Regex pattern should not match if anchor doesn't align."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(failure_patterns=["ERROR"], default_outcome="other")
        outcomes = [
            Outcome(name="approved", patterns=["regex:^APPROVED"]),
            Outcome(name="other", patterns=["regex:.*"]),
        ]

        status, outcome = executor._determine_status("Result: APPROVED", retry_config, outcomes)

        # ^APPROVED won't match because APPROVED is not at start
        # Falls back to "other" which matches anything
        assert status == ExecutionStatus.SUCCESS
        assert outcome == "other"


class TestAgentExecutorExecute:
    """Tests for AgentExecutor.execute."""

    def test_successful_execution(self) -> None:
        client = MockAgentClient(responses=["SUCCESS: Task completed"])
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.status == ExecutionStatus.SUCCESS
        assert result.attempts == 1
        assert "SUCCESS" in result.response

    def test_passes_tools_to_client(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(tools=["jira", "github", "confluence"])

        executor.execute(issue, orch)

        assert client.calls[0][1] == ["jira", "github", "confluence"]

    def test_passes_github_context_to_client(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(
            github=GitHubContext(host="github.example.com", org="myorg", repo="myrepo")
        )

        executor.execute(issue, orch)

        context = client.calls[0][2]
        assert context is not None
        assert context["github"]["host"] == "github.example.com"
        assert context["github"]["org"] == "myorg"
        assert context["github"]["repo"] == "myrepo"

    def test_retries_on_failure(self) -> None:
        client = MockAgentClient(
            responses=["FAILURE: First attempt", "FAILURE: Second", "SUCCESS: Done"]
        )
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 3
        assert client.call_count == 3

    def test_stops_retrying_after_success(self) -> None:
        client = MockAgentClient(responses=["FAILURE: First", "SUCCESS: Done", "Should not reach"])
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 2
        assert client.call_count == 2

    def test_returns_failure_after_max_attempts(self) -> None:
        client = MockAgentClient(
            responses=["FAILURE: Attempt 1", "FAILURE: Attempt 2", "FAILURE: Attempt 3"]
        )
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is False
        assert result.status == ExecutionStatus.FAILURE
        assert result.attempts == 3

    def test_handles_client_error(self) -> None:
        client = MockAgentClient()
        client.should_error = True
        client.max_errors = 10
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=2)

        result = executor.execute(issue, orch)

        assert result.succeeded is False
        assert result.status == ExecutionStatus.ERROR
        assert result.attempts == 2

    def test_retries_on_client_error_then_succeeds(self) -> None:
        client = MockAgentClient(responses=["SUCCESS: Done"])
        client.should_error = True
        client.max_errors = 1  # Fail once, then succeed
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 2

    def test_result_includes_issue_key(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-999")
        orch = make_orchestration()

        result = executor.execute(issue, orch)

        assert result.issue_key == "PROJ-999"

    def test_result_includes_orchestration_name(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(name="code-review")

        result = executor.execute(issue, orch)

        assert result.orchestration_name == "code-review"

    def test_passes_timeout_to_client(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()
        orch.agent.timeout_seconds = 300

        executor.execute(issue, orch)

        assert client.calls[0][3] == 300

    def test_handles_timeout_error(self) -> None:
        client = MockAgentClient()
        client.should_timeout = True
        client.max_timeouts = 10
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=2)

        result = executor.execute(issue, orch)

        assert result.succeeded is False
        assert result.status == ExecutionStatus.ERROR
        assert result.attempts == 2

    def test_retries_on_timeout_then_succeeds(self) -> None:
        client = MockAgentClient(responses=["SUCCESS: Done"])
        client.should_timeout = True
        client.max_timeouts = 1  # Timeout once, then succeed
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 2

    def test_timeout_none_by_default(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()

        executor.execute(issue, orch)

        assert client.calls[0][3] is None

    def test_passes_model_to_client(self) -> None:
        """Model from orchestration should be passed to client."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()
        orch.agent.model = "claude-opus-4-5-20251101"

        executor.execute(issue, orch)

        # Index 5 is model in the calls tuple
        assert client.calls[0][5] == "claude-opus-4-5-20251101"

    def test_model_none_by_default(self) -> None:
        """Model should be None when not specified in orchestration."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()

        executor.execute(issue, orch)

        # Index 5 is model in the calls tuple
        assert client.calls[0][5] is None


class TestAgentExecutorExecuteWithOutcomes:
    """Tests for AgentExecutor.execute with outcomes configured."""

    def test_successful_execution_with_outcome(self) -> None:
        """Execute should return matched outcome in result."""
        client = MockAgentClient(responses=["APPROVED: Code looks good"])
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = Orchestration(
            name="code-review",
            trigger=TriggerConfig(),
            agent=AgentConfig(prompt="Review code", tools=["github"]),
            retry=RetryConfig(failure_patterns=["ERROR"]),
            outcomes=[
                Outcome(name="approved", patterns=["APPROVED"], add_tag="code-reviewed"),
                Outcome(
                    name="changes-requested",
                    patterns=["CHANGES REQUESTED"],
                    add_tag="changes-requested",
                ),
            ],
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.matched_outcome == "approved"

    def test_execute_matches_second_outcome(self) -> None:
        """Execute should match correct outcome from list."""
        client = MockAgentClient(responses=["CHANGES REQUESTED: Please fix the bug"])
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = Orchestration(
            name="code-review",
            trigger=TriggerConfig(),
            agent=AgentConfig(prompt="Review code", tools=["github"]),
            retry=RetryConfig(failure_patterns=["ERROR"]),
            outcomes=[
                Outcome(name="approved", patterns=["APPROVED"]),
                Outcome(name="changes-requested", patterns=["CHANGES REQUESTED"]),
            ],
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.matched_outcome == "changes-requested"

    def test_execute_retries_on_failure_with_outcomes(self) -> None:
        """Execute should retry on failure patterns even with outcomes configured."""
        client = MockAgentClient(
            responses=["ERROR: GitHub API failed", "APPROVED: Code looks good"]
        )
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = Orchestration(
            name="code-review",
            trigger=TriggerConfig(),
            agent=AgentConfig(prompt="Review code", tools=["github"]),
            retry=RetryConfig(failure_patterns=["ERROR"], max_attempts=3),
            outcomes=[
                Outcome(name="approved", patterns=["APPROVED"]),
            ],
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.matched_outcome == "approved"
        assert result.attempts == 2

    def test_execute_no_outcome_without_outcomes_config(self) -> None:
        """Execute should return None for matched_outcome when outcomes not configured."""
        client = MockAgentClient(responses=["SUCCESS: Task completed"])
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()  # No outcomes configured

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.matched_outcome is None


class TestAgentExecutorBuildPromptGitHubIssue:
    """Tests for AgentExecutor.build_prompt with GitHub Issues."""

    def test_substitutes_github_issue_number(self) -> None:
        """Should substitute {github_issue_number} with issue number."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        orch = make_orchestration(prompt="Review issue #{github_issue_number}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review issue #42"

    def test_substitutes_github_issue_title(self) -> None:
        """Should substitute {github_issue_title} with issue title."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Fix authentication bug")
        orch = make_orchestration(prompt="Task: {github_issue_title}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Task: Fix authentication bug"

    def test_substitutes_github_issue_body(self) -> None:
        """Should substitute {github_issue_body} with issue body."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", body="Detailed description here")
        orch = make_orchestration(prompt="Description:\n{github_issue_body}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Description:\nDetailed description here"

    def test_substitutes_github_issue_state(self) -> None:
        """Should substitute {github_issue_state} with issue state."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", state="open")
        orch = make_orchestration(prompt="State: {github_issue_state}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "State: open"

    def test_substitutes_github_issue_author(self) -> None:
        """Should substitute {github_issue_author} with author username."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", author="octocat")
        orch = make_orchestration(prompt="Author: {github_issue_author}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Author: octocat"

    def test_substitutes_github_issue_assignees(self) -> None:
        """Should substitute {github_issue_assignees} with comma-separated assignees."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", assignees=["alice", "bob"])
        orch = make_orchestration(prompt="Assignees: {github_issue_assignees}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Assignees: alice, bob"

    def test_substitutes_github_issue_labels(self) -> None:
        """Should substitute {github_issue_labels} with comma-separated labels."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", labels=["bug", "urgent"])
        orch = make_orchestration(prompt="Labels: {github_issue_labels}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Labels: bug, urgent"

    def test_substitutes_github_issue_url(self) -> None:
        """Should substitute {github_issue_url} with full URL."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.com/myorg/myrepo/issues/42"

    def test_substitutes_github_pr_url(self) -> None:
        """Should substitute {github_issue_url} with PR URL for pull requests."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test PR", is_pull_request=True)
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.com/myorg/myrepo/pull/42"

    def test_substitutes_github_is_pr(self) -> None:
        """Should substitute {github_is_pr} with 'true' or 'false'."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)

        issue = GitHubIssue(number=1, title="Test Issue", is_pull_request=False)
        orch = make_orchestration(prompt="Is PR: {github_is_pr}")
        prompt = executor.build_prompt(issue, orch)
        assert prompt == "Is PR: false"

        pr = GitHubIssue(number=1, title="Test PR", is_pull_request=True)
        prompt = executor.build_prompt(pr, orch)
        assert prompt == "Is PR: true"

    def test_substitutes_github_pr_head(self) -> None:
        """Should substitute {github_pr_head} with head branch reference."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=1, title="Test PR", is_pull_request=True, head_ref="feature-branch"
        )
        orch = make_orchestration(prompt="Head: {github_pr_head}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: feature-branch"

    def test_substitutes_github_pr_base(self) -> None:
        """Should substitute {github_pr_base} with base branch reference."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test PR", is_pull_request=True, base_ref="main")
        orch = make_orchestration(prompt="Base: {github_pr_base}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Base: main"

    def test_substitutes_github_pr_draft(self) -> None:
        """Should substitute {github_pr_draft} with 'true' or 'false'."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)

        draft_pr = GitHubIssue(number=1, title="Draft PR", is_pull_request=True, draft=True)
        orch = make_orchestration(prompt="Draft: {github_pr_draft}")
        prompt = executor.build_prompt(draft_pr, orch)
        assert prompt == "Draft: true"

        ready_pr = GitHubIssue(number=1, title="Ready PR", is_pull_request=True, draft=False)
        prompt = executor.build_prompt(ready_pr, orch)
        assert prompt == "Draft: false"

    def test_github_issue_jira_variables_empty(self) -> None:
        """Jira variables should be empty for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="Jira: [{jira_issue_key}] [{jira_summary}] [{jira_status}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Jira: [] [] []"

    def test_jira_issue_github_variables_empty(self) -> None:
        """GitHub Issue variables should be empty for Jira issues."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-123", summary="Test")
        orch = make_orchestration(
            prompt="GitHub: [{github_issue_number}] [{github_issue_title}] [{github_issue_state}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "GitHub: [] [] []"

    def test_substitutes_multiple_github_variables(self) -> None:
        """Should substitute multiple GitHub variables in one prompt."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=42,
            title="Fix authentication",
            state="open",
            author="octocat",
            labels=["bug", "security"],
        )
        orch = make_orchestration(
            prompt="#{github_issue_number}: {github_issue_title} by {github_issue_author} ({github_issue_labels})"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "#42: Fix authentication by octocat (bug, security)"

    def test_substitutes_github_context_and_issue_variables(self) -> None:
        """Should substitute both GitHub repo context and issue variables."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        orch = make_orchestration(
            prompt="Review {github_org}/{github_repo}#{github_issue_number}: {github_issue_title}",
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Review myorg/myrepo#42: Test Issue"


class TestAgentExecutorExecuteGitHubIssue:
    """Tests for AgentExecutor.execute with GitHub Issues."""

    def test_successful_execution_with_github_issue(self) -> None:
        """Execute should work with GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient(responses=["SUCCESS: Review completed"])
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test PR", is_pull_request=True)
        orch = make_orchestration(prompt="Review GitHub PR #{github_issue_number}")

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.status == ExecutionStatus.SUCCESS
        assert result.issue_key == "#42"
        assert "SUCCESS" in result.response

    def test_github_issue_key_in_result(self) -> None:
        """Result should include GitHub issue key format."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=123, title="Test")
        orch = make_orchestration()

        result = executor.execute(issue, orch)

        assert result.issue_key == "#123"

    def test_github_issue_retries_on_failure(self) -> None:
        """Execute should retry on failure for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient(
            responses=["FAILURE: First attempt", "SUCCESS: Done"]
        )
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(max_attempts=3)

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.attempts == 2


class TestGitHubTemplateVariables:
    """Additional tests for GitHub Issue template variable handling in executor.

    These tests ensure comprehensive coverage of GitHub template variables,
    particularly edge cases and interactions between Jira and GitHub contexts.
    Added as follow-up from DS-55 code review.
    """

    def test_github_issue_url_empty_when_no_github_context(self) -> None:
        """URL should be empty when GitHub context is not configured."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test Issue")
        # No GitHub context configured
        orch = make_orchestration(prompt="URL: [{github_issue_url}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: []"

    def test_github_pr_fields_empty_for_regular_issue(self) -> None:
        """PR-specific fields should be empty strings for regular issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        # Regular issue, not a PR
        issue = GitHubIssue(number=42, title="Regular Issue", is_pull_request=False)
        orch = make_orchestration(
            prompt="Head: [{github_pr_head}] Base: [{github_pr_base}] Draft: [{github_pr_draft}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: [] Base: [] Draft: []"

    def test_github_pr_fields_populated_for_pull_request(self) -> None:
        """PR-specific fields should be populated for pull requests."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=42,
            title="Feature PR",
            is_pull_request=True,
            head_ref="feature/new-thing",
            base_ref="main",
            draft=True,
        )
        orch = make_orchestration(
            prompt="Head: {github_pr_head} Base: {github_pr_base} Draft: {github_pr_draft}"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Head: feature/new-thing Base: main Draft: true"

    def test_github_issue_empty_body_substitutes_empty_string(self) -> None:
        """Empty issue body should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", body="")
        orch = make_orchestration(prompt="Body: [{github_issue_body}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Body: []"

    def test_github_issue_none_body_substitutes_empty_string(self) -> None:
        """Empty body should substitute as empty string (None is normalized to empty)."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        # GitHubIssue uses empty string default, but test the flow
        issue = GitHubIssue(number=1, title="Test", body="")
        orch = make_orchestration(prompt="Body: [{github_issue_body}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Body: []"

    def test_github_issue_empty_assignees_substitutes_empty_string(self) -> None:
        """Empty assignees list should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", assignees=[])
        orch = make_orchestration(prompt="Assignees: [{github_issue_assignees}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Assignees: []"

    def test_github_issue_empty_labels_substitutes_empty_string(self) -> None:
        """Empty labels list should substitute as empty string."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test", labels=[])
        orch = make_orchestration(prompt="Labels: [{github_issue_labels}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Labels: []"

    def test_github_issue_url_with_custom_host(self) -> None:
        """URL construction should work with custom GitHub host."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.enterprise.com", org="corp", repo="internal-app"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.enterprise.com/corp/internal-app/issues/42"

    def test_github_pr_url_with_custom_host(self) -> None:
        """PR URL construction should work with custom GitHub host."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=99, title="Enterprise PR", is_pull_request=True)
        orch = make_orchestration(
            prompt="URL: {github_issue_url}",
            github=GitHubContext(host="github.enterprise.com", org="corp", repo="internal-app"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "URL: https://github.enterprise.com/corp/internal-app/pull/99"

    def test_all_jira_variables_empty_for_github_issue(self) -> None:
        """All Jira variables should be empty strings for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test")
        orch = make_orchestration(
            prompt=(
                "key:[{jira_issue_key}] summary:[{jira_summary}] "
                "desc:[{jira_description}] status:[{jira_status}] "
                "assignee:[{jira_assignee}] labels:[{jira_labels}] "
                "comments:[{jira_comments}] links:[{jira_links}]"
            )
        )

        prompt = executor.build_prompt(issue, orch)

        expected = (
            "key:[] summary:[] desc:[] status:[] assignee:[] labels:[] comments:[] links:[]"
        )
        assert prompt == expected

    def test_all_github_issue_variables_empty_for_jira(self) -> None:
        """All GitHub Issue variables should be empty strings for Jira issues."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-1", summary="Test")
        orch = make_orchestration(
            prompt=(
                "num:[{github_issue_number}] title:[{github_issue_title}] "
                "body:[{github_issue_body}] state:[{github_issue_state}] "
                "author:[{github_issue_author}] assignees:[{github_issue_assignees}] "
                "labels:[{github_issue_labels}] url:[{github_issue_url}] "
                "is_pr:[{github_is_pr}] head:[{github_pr_head}] "
                "base:[{github_pr_base}] draft:[{github_pr_draft}]"
            )
        )

        prompt = executor.build_prompt(issue, orch)

        expected = (
            "num:[] title:[] body:[] state:[] author:[] assignees:[] "
            "labels:[] url:[] is_pr:[] head:[] base:[] draft:[]"
        )
        assert prompt == expected

    def test_github_is_pr_false_lowercase(self) -> None:
        """github_is_pr should be lowercase 'false' for regular issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Issue", is_pull_request=False)
        orch = make_orchestration(prompt="{github_is_pr}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "false"

    def test_github_is_pr_true_lowercase(self) -> None:
        """github_is_pr should be lowercase 'true' for PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="PR", is_pull_request=True)
        orch = make_orchestration(prompt="{github_is_pr}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "true"

    def test_github_pr_draft_false_lowercase(self) -> None:
        """github_pr_draft should be lowercase 'false' for non-draft PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="PR", is_pull_request=True, draft=False)
        orch = make_orchestration(prompt="{github_pr_draft}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "false"

    def test_github_pr_draft_true_lowercase(self) -> None:
        """github_pr_draft should be lowercase 'true' for draft PRs."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Draft PR", is_pull_request=True, draft=True)
        orch = make_orchestration(prompt="{github_pr_draft}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "true"

    def test_github_context_variables_still_work_for_github_issues(self) -> None:
        """GitHub repo context variables should work with GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            prompt="Repo: {github_org}/{github_repo} on {github_host}",
            github=GitHubContext(host="github.com", org="acme", repo="widgets"),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Repo: acme/widgets on github.com"

    def test_github_context_variables_empty_when_not_configured(self) -> None:
        """GitHub repo context should be empty when not configured."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=1, title="Test")
        # No GitHub context in orchestration
        orch = make_orchestration(
            prompt="Host:[{github_host}] Org:[{github_org}] Repo:[{github_repo}]"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Host:[] Org:[] Repo:[]"

    def test_combined_github_context_and_issue_variables(self) -> None:
        """Test combining repo context with issue-specific variables."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(
            number=123,
            title="Add feature X",
            state="open",
            author="developer",
            is_pull_request=True,
            head_ref="feature-x",
            base_ref="develop",
        )
        orch = make_orchestration(
            prompt=(
                "PR #{github_issue_number} '{github_issue_title}' by {github_issue_author} "
                "in {github_org}/{github_repo} from {github_pr_head} to {github_pr_base}"
            ),
            github=GitHubContext(host="github.com", org="company", repo="project"),
        )

        prompt = executor.build_prompt(issue, orch)

        expected = (
            "PR #123 'Add feature X' by developer "
            "in company/project from feature-x to develop"
        )
        assert prompt == expected


class TestCleanupWorkdir:
    """Tests for the cleanup_workdir function."""

    def test_cleanup_workdir_removes_directory(self) -> None:
        """cleanup_workdir should remove the directory and return True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            assert workdir.exists()
            result = cleanup_workdir(workdir)

            assert result is True
            assert not workdir.exists()

    def test_cleanup_workdir_none_returns_true(self) -> None:
        """cleanup_workdir should return True when workdir is None."""
        result = cleanup_workdir(None)
        assert result is True

    def test_cleanup_workdir_nonexistent_returns_true(self) -> None:
        """cleanup_workdir should return True for non-existent directory."""
        workdir = Path("/nonexistent/path/that/does/not/exist")
        result = cleanup_workdir(workdir)
        assert result is True

    def test_cleanup_workdir_permission_error_returns_false(self) -> None:
        """cleanup_workdir should return False on PermissionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = PermissionError("Permission denied")
                result = cleanup_workdir(workdir)

            assert result is False

    def test_cleanup_workdir_os_error_returns_false(self) -> None:
        """cleanup_workdir should return False on OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = OSError("OS error")
                result = cleanup_workdir(workdir)

            assert result is False

    def test_cleanup_workdir_unexpected_error_returns_false(self) -> None:
        """cleanup_workdir should return False on unexpected exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = RuntimeError("Unexpected error")
                result = cleanup_workdir(workdir)

            assert result is False

    # Tests for force parameter

    def test_cleanup_workdir_force_removes_directory(self) -> None:
        """cleanup_workdir with force=True should remove the directory on first attempt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            assert workdir.exists()
            result = cleanup_workdir(workdir, force=True)

            assert result is True
            assert not workdir.exists()

    def test_cleanup_workdir_force_none_returns_true(self) -> None:
        """cleanup_workdir with force=True should return True when workdir is None."""
        result = cleanup_workdir(None, force=True)
        assert result is True

    def test_cleanup_workdir_force_nonexistent_returns_true(self) -> None:
        """cleanup_workdir with force=True should return True for non-existent directory."""
        workdir = Path("/nonexistent/path/that/does/not/exist")
        result = cleanup_workdir(workdir, force=True)
        assert result is True

    def test_cleanup_workdir_force_retries_on_permission_error(self) -> None:
        """cleanup_workdir with force=True should retry on PermissionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise PermissionError("Permission denied")
                # Successful removal on second attempt - actually remove it
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert call_count == 2

    def test_cleanup_workdir_force_retries_on_os_error(self) -> None:
        """cleanup_workdir with force=True should retry on OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise OSError("Resource busy")
                # Successful removal on second attempt
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert call_count == 2

    def test_cleanup_workdir_force_uses_ignore_errors_on_final_attempt(self) -> None:
        """cleanup_workdir with force=True should use ignore_errors on final attempt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            rmtree_calls: list[tuple[Any, dict[str, Any]]] = []
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                rmtree_calls.append((path, kwargs))
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # When ignore_errors=True, actually remove the directory
                original_rmtree(path, ignore_errors=True)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert len(rmtree_calls) == 3
            # First two calls should not have ignore_errors
            assert rmtree_calls[0][1].get("ignore_errors", False) is False
            assert rmtree_calls[1][1].get("ignore_errors", False) is False
            # Final call should have ignore_errors=True
            assert rmtree_calls[2][1].get("ignore_errors") is True

    def test_cleanup_workdir_force_returns_false_if_directory_remains(self) -> None:
        """cleanup_workdir with force=True should return False if directory still exists after all attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                # Always fail (even with ignore_errors, directory persists)
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # With ignore_errors, just do nothing (directory remains)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is False
            assert workdir.exists()

    def test_cleanup_workdir_force_no_retry_on_unexpected_error(self) -> None:
        """cleanup_workdir with force=True should not retry on unexpected exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Unexpected error")

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is False
            assert call_count == 1  # Should not retry

    def test_cleanup_workdir_force_custom_max_retries(self) -> None:
        """cleanup_workdir with force=True should respect custom max_retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # With ignore_errors, just do nothing (directory remains)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=5)

            assert result is False
            assert call_count == 5  # Should retry 5 times

    def test_cleanup_workdir_force_respects_retry_delay(self) -> None:
        """cleanup_workdir with force=True should sleep between retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise PermissionError("Permission denied")
                # Successful on third attempt
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep") as mock_sleep:
                    result = cleanup_workdir(
                        workdir, force=True, max_retries=3, retry_delay=1.5
                    )

            assert result is True
            # Should have slept twice (after first and second failures)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_called_with(1.5)


class TestAgentExecutorWorkdirCleanup:
    """Tests for AgentExecutor workdir cleanup behavior.

    Design rationale for workdir preservation on failure:
        When an agent execution fails (whether through explicit FAILURE response,
        client ERROR, or timeout), the workdir is intentionally preserved rather
        than cleaned up. This design decision supports debugging by allowing
        developers to inspect:
        - Cloned repositories and their state at failure time
        - Partial work completed before the failure
        - Log files or artifacts generated during execution
        - Any modifications made to files that might explain the failure

        The tests in this class verify both the success-path cleanup AND the
        failure-path preservation behavior. Tests that verify preservation on
        failure are critical - they ensure we don't accidentally remove valuable
        debugging context when something goes wrong.
    """

    def test_cleanup_on_success_when_enabled(self) -> None:
        """Should cleanup workdir on successful execution when cleanup is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Task completed"],
                workdir=workdir,
            )
            # cleanup_workdir_on_success=True is the default
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration()

            assert workdir.exists()
            result = executor.execute(issue, orch)

            assert result.succeeded is True
            assert not workdir.exists()

    def test_no_cleanup_when_disabled(self, caplog: Any) -> None:
        """Should preserve workdir when cleanup_workdir_on_success is False.

        Also verifies that the appropriate debug log message is emitted
        indicating the workdir was preserved.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Task completed"],
                workdir=workdir,
            )
            executor = AgentExecutor(client, cleanup_workdir_on_success=False)
            issue = make_issue()
            orch = make_orchestration()

            with caplog.at_level(logging.DEBUG):
                result = executor.execute(issue, orch)

            assert result.succeeded is True
            assert workdir.exists()  # Workdir should be preserved

            # Verify the debug log message was emitted
            assert any(
                "Workdir preserved at" in record.message
                and "(cleanup_workdir_on_success=False)" in record.message
                and record.levelno == logging.DEBUG
                for record in caplog.records
            ), "Expected debug log message about workdir preservation was not emitted"

    def test_no_cleanup_on_failure(self) -> None:
        """Should preserve workdir on failed execution for debugging.

        When an agent reports FAILURE, we intentionally keep the workdir intact.
        This allows debugging what went wrong by examining the file state,
        partial work, and any artifacts the agent created before failing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["FAILURE: Task failed"],
                workdir=workdir,
            )
            # Even with cleanup enabled, failures should preserve workdir
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.FAILURE
            assert workdir.exists()  # Workdir preserved for debugging

    def test_cleanup_handles_none_workdir(self) -> None:
        """Should handle None workdir gracefully when no workdir is provided.

        This test verifies the executor handles the case where the agent client
        returns None for workdir (e.g., when the agent doesn't create a working
        directory). Graceful handling prevents crashes and ensures the cleanup
        logic safely skips cleanup when there's nothing to clean up.

        Expected behavior: execution completes successfully without raising
        any errors, even when workdir is None.
        """
        client = MockAgentClient(
            responses=["SUCCESS: Task completed"],
            workdir=None,  # No workdir
        )
        executor = AgentExecutor(client, cleanup_workdir_on_success=True)
        issue = make_issue()
        orch = make_orchestration()

        # Should not raise any errors
        result = executor.execute(issue, orch)

        assert result.succeeded is True

    def test_cleanup_default_is_enabled(self) -> None:
        """Should have cleanup enabled by default.

        Verifies that cleanup_workdir_on_success defaults to True when not
        explicitly specified. This ensures workdirs are cleaned up after
        successful executions unless explicitly disabled.
        """
        client = MockAgentClient()
        executor = AgentExecutor(client)  # No explicit cleanup_workdir_on_success

        assert executor.cleanup_workdir_on_success is True

    def test_no_cleanup_on_error(self) -> None:
        """Should preserve workdir on agent client error.

        When the agent client raises an error (e.g., API failure, network issue),
        we preserve the workdir to aid debugging. The workdir may contain partial
        state or cloned repos that help diagnose why the agent errored.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Done"],
                workdir=workdir,
            )
            client.should_error = True
            client.max_errors = 10  # Always error
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.ERROR
            assert workdir.exists()  # Workdir preserved

    def test_no_cleanup_on_timeout(self) -> None:
        """Should preserve workdir on agent timeout.

        When an agent times out, the workdir is preserved to support debugging.
        Timeout scenarios often leave partial work in progress, and examining
        the workdir state can reveal what the agent was working on when time
        ran out.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Done"],
                workdir=workdir,
            )
            client.should_timeout = True
            client.max_timeouts = 10  # Always timeout
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.ERROR
            assert workdir.exists()  # Workdir preserved


class TestAgentRunResult:
    """Tests for AgentRunResult dataclass."""

    def test_agent_run_result_with_response_only(self) -> None:
        """AgentRunResult can be created with just response."""
        result = AgentRunResult(response="Task completed")
        assert result.response == "Task completed"
        assert result.workdir is None

    def test_agent_run_result_with_workdir(self) -> None:
        """AgentRunResult can include a workdir path."""
        workdir = Path("/tmp/test-workdir")
        result = AgentRunResult(response="Task completed", workdir=workdir)
        assert result.response == "Task completed"
        assert result.workdir == workdir

    def test_agent_run_result_workdir_default_none(self) -> None:
        """AgentRunResult workdir should default to None."""
        result = AgentRunResult(response="Done")
        assert result.workdir is None
