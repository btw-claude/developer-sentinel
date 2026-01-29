"""Tests for Claude Agent SDK executor status determination and execution.

This module contains executor tests for:
- determine_status tests (with and without outcomes)
- execute method tests
"""

from sentinel.executor import AgentExecutor, ExecutionStatus
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    Outcome,
    RetryConfig,
    TriggerConfig,
)
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


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
        assert outcome is None

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
        client.max_errors = 1
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
        client.max_timeouts = 1
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

        assert client.calls[0][5] == "claude-opus-4-5-20251101"

    def test_model_none_by_default(self) -> None:
        """Model should be None when not specified in orchestration."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()

        executor.execute(issue, orch)

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
        orch = make_orchestration()

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        assert result.matched_outcome is None
