"""Tests for Claude Agent SDK executor module."""

from typing import Any

from sentinel.executor import (
    AgentClient,
    AgentClientError,
    AgentExecutor,
    ExecutionResult,
    ExecutionStatus,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    RetryConfig,
    TriggerConfig,
)
from sentinel.poller import JiraIssue


class MockAgentClient(AgentClient):
    """Mock agent client for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["SUCCESS: Task completed"]
        self.call_count = 0
        self.calls: list[tuple[str, list[str], dict[str, Any] | None]] = []
        self.should_error = False
        self.error_count = 0
        self.max_errors = 0

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append((prompt, tools, context))

        if self.should_error and self.error_count < self.max_errors:
            self.error_count += 1
            raise AgentClientError("Mock agent error")

        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


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
    """Tests for AgentExecutor.build_prompt."""

    def test_includes_base_prompt(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(prompt="Review this code")

        prompt = executor.build_prompt(issue, orch)

        assert "Review this code" in prompt

    def test_includes_issue_key_and_summary(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="PROJ-123", summary="Fix the bug")
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "PROJ-123" in prompt
        assert "Fix the bug" in prompt

    def test_includes_description(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(description="Detailed description here")
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "Detailed description here" in prompt

    def test_includes_status(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(status="In Progress")
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "In Progress" in prompt

    def test_includes_assignee(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(assignee="John Doe")
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "John Doe" in prompt

    def test_includes_labels(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(labels=["bug", "urgent"])
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "bug" in prompt
        assert "urgent" in prompt

    def test_includes_comments(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(comments=["First comment", "Second comment"])
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "First comment" in prompt
        assert "Second comment" in prompt

    def test_includes_links(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(links=["TEST-2", "TEST-3"])
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "TEST-2" in prompt
        assert "TEST-3" in prompt

    def test_includes_github_context(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration(
            github=GitHubContext(org="myorg", repo="myrepo")
        )

        prompt = executor.build_prompt(issue, orch)

        assert "myorg/myrepo" in prompt

    def test_includes_response_format_instructions(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue()
        orch = make_orchestration()

        prompt = executor.build_prompt(issue, orch)

        assert "SUCCESS" in prompt
        assert "FAILURE" in prompt


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

    def test_matches_regex_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("error code 123", ["error.*\\d+"]) is True


class TestAgentExecutorDetermineStatus:
    """Tests for AgentExecutor._determine_status."""

    def test_returns_success_on_success_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status = executor._determine_status("Task SUCCESS", retry_config)

        assert status == ExecutionStatus.SUCCESS

    def test_returns_failure_on_failure_pattern(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status = executor._determine_status("Task FAILURE", retry_config)

        assert status == ExecutionStatus.FAILURE

    def test_success_takes_precedence(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        # Response contains both patterns
        status = executor._determine_status("SUCCESS but had FAILURE", retry_config)

        assert status == ExecutionStatus.SUCCESS

    def test_defaults_to_success_when_no_match(self) -> None:
        client = MockAgentClient()
        executor = AgentExecutor(client)
        retry_config = RetryConfig(
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
        )

        status = executor._determine_status("Task done", retry_config)

        assert status == ExecutionStatus.SUCCESS


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
        client = MockAgentClient(
            responses=["FAILURE: First", "SUCCESS: Done", "Should not reach"]
        )
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
