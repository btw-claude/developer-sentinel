"""Core tests for Claude Agent SDK executor module.

This module contains core executor tests including:
- ExecutionResult tests
- build_prompt tests
- expand_branch_pattern tests
- matches_pattern tests
"""

from sentinel.executor import AgentExecutor, ExecutionResult, ExecutionStatus
from sentinel.orchestration import GitHubContext
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


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


class TestAgentExecutorExpandBranchPattern:
    """Tests for AgentExecutor._expand_branch_pattern."""

    def test_returns_none_when_no_github_context(self) -> None:
        """Should return None when no GitHub context is configured."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration()

        result = executor._expand_branch_pattern(issue, orch)

        assert result is None

    def test_returns_none_when_no_branch_configured(self) -> None:
        """Should return None when GitHub context exists but no branch pattern."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration(
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo")
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result is None

    def test_returns_none_when_branch_is_empty_string(self) -> None:
        """Should return None when branch pattern is an empty string."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration(
            github=GitHubContext(host="github.com", org="myorg", repo="myrepo", branch="")
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result is None

    def test_expands_jira_issue_key(self) -> None:
        """Should expand {jira_issue_key} in branch pattern."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-290")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", branch="feature/{jira_issue_key}"
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/DS-290"

    def test_expands_github_issue_number(self) -> None:
        """Should expand {github_issue_number} in branch pattern."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=123, title="Test Issue")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", branch="fix/{github_issue_number}"
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "fix/123"

    def test_expands_jira_summary_rejects_spaces(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should reject {jira_summary} that contains spaces.

        Raw summary text with spaces produces invalid branch names.
        Users should use {jira_summary_slug} for branch-safe names.
        Runtime validation now correctly rejects these patterns.
        """
        import logging

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Add login button")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", branch="feature/{jira_summary}"
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Spaces in branch names are invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "invalid characters" in caplog.text

    def test_expands_github_issue_title_rejects_spaces(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should reject {github_issue_title} that contains spaces.

        Raw issue title text with spaces produces invalid branch names.
        Users should use {github_issue_title_slug} for branch-safe names.
        Runtime validation now correctly rejects these patterns.
        """
        import logging

        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Fix authentication bug")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", branch="fix/{github_issue_title}"
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Spaces in branch names are invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "invalid characters" in caplog.text

    def test_expands_multiple_variables_valid(self) -> None:
        """Should expand multiple variables in branch pattern when result is valid."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        # Use a summary without spaces to produce a valid branch name
        issue = make_issue(key="DS-456", summary="update-docs")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="{jira_issue_key}-{jira_summary}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "DS-456-update-docs"

    def test_preserves_unknown_variables(self) -> None:
        """Should preserve unknown variables as-is."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_issue_key}/{unknown_var}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/DS-123/{unknown_var}"

    def test_returns_pattern_unchanged_when_no_variables(self) -> None:
        """Should return pattern unchanged when no template variables present."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", branch="static-branch-name"
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "static-branch-name"

    def test_github_issue_jira_variables_empty_rejected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should reject pattern when Jira variables produce trailing slash.

        When using Jira-specific variables with GitHub issues, they produce
        empty strings which can result in trailing slashes - invalid branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Test")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_issue_key}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Trailing slash is invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text

    def test_jira_issue_github_variables_empty_rejected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should reject pattern when GitHub variables produce trailing slash.

        When using GitHub-specific variables with Jira issues, they produce
        empty strings which can result in trailing slashes - invalid branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{github_issue_number}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Trailing slash is invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text


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

        assert executor._matches_pattern("error code 123", ["regex:error.*\\d+"]) is True

    def test_regex_pattern_anchors(self) -> None:
        """Test regex patterns with anchors."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("Task completed", ["regex:^Task"]) is True
        assert executor._matches_pattern("My Task completed", ["regex:^Task"]) is False

        assert executor._matches_pattern("Task completed", ["regex:completed$"]) is True
        assert executor._matches_pattern("Task completed!", ["regex:completed$"]) is False

    def test_pattern_without_prefix_is_substring(self) -> None:
        """Patterns without 'regex:' prefix are treated as substrings."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("rating: 5*", ["5*"]) is True
        assert executor._matches_pattern("555", ["5*"]) is False

    def test_invalid_regex_falls_back_to_substring(self) -> None:
        """Invalid regex patterns fall back to substring matching."""
        client = MockAgentClient()
        executor = AgentExecutor(client)

        assert executor._matches_pattern("error [unclosed", ["regex:[unclosed"]) is True
