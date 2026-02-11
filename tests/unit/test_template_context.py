"""Tests for TemplateContext dataclass in executor module.

This module tests the data-driven template variable generation using
dataclass introspection, as implemented in DS-396.
"""

from sentinel.executor import (
    AgentExecutor,
    TemplateContext,
    _compute_slug,
    _format_comments,
    _format_list,
)
from sentinel.github_poller import GitHubIssue
from sentinel.orchestration import GitHubContext
from sentinel.poller import JiraIssue
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


class TestHelperFunctions:
    """Tests for helper functions used by TemplateContext."""

    def test_compute_slug_converts_to_branch_safe(self) -> None:
        """Should convert text to branch-safe slug."""
        assert _compute_slug("Add login button") == "add-login-button"
        assert _compute_slug("Fix: Critical Bug!") == "fix-critical-bug"

    def test_compute_slug_empty_string(self) -> None:
        """Should handle empty string."""
        assert _compute_slug("") == ""

    def test_format_comments_empty_list(self) -> None:
        """Should return empty string for empty list."""
        assert _format_comments([]) == ""

    def test_format_comments_single_comment(self) -> None:
        """Should format single comment."""
        result = _format_comments(["First comment"])
        assert result == "1. First comment"

    def test_format_comments_multiple_comments(self) -> None:
        """Should format multiple comments with numbers."""
        result = _format_comments(["First", "Second", "Third"])
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result

    def test_format_comments_truncates_long_comments(self) -> None:
        """Should truncate comments longer than max_length."""
        long_comment = "x" * 600
        result = _format_comments([long_comment], max_length=500)
        assert "..." in result
        assert len(result.split("...")[0]) <= 503  # "1. " + 500 chars

    def test_format_comments_limits_to_last_n(self) -> None:
        """Should only include last N comments."""
        comments = ["First", "Second", "Third", "Fourth", "Fifth"]
        result = _format_comments(comments, limit=3)
        assert "First" not in result
        assert "Second" not in result
        assert "Third" in result
        assert "Fourth" in result
        assert "Fifth" in result

    def test_format_list_empty(self) -> None:
        """Should return empty string for empty list."""
        assert _format_list([]) == ""
        assert _format_list(None) == ""

    def test_format_list_single_item(self) -> None:
        """Should return item without comma."""
        assert _format_list(["item"]) == "item"

    def test_format_list_multiple_items(self) -> None:
        """Should join items with comma and space."""
        assert _format_list(["a", "b", "c"]) == "a, b, c"


class TestTemplateContextFromJiraIssue:
    """Tests for TemplateContext.from_jira_issue factory method."""

    def test_creates_context_with_basic_fields(self) -> None:
        """Should populate basic Jira fields."""
        issue = JiraIssue(
            key="DS-123",
            summary="Test issue",
            description="Test description",
            status="In Progress",
            assignee="developer",
        )

        context = TemplateContext.from_jira_issue(issue)

        assert context.jira_issue_key == "DS-123"
        assert context.jira_summary == "Test issue"
        assert context.jira_description == "Test description"
        assert context.jira_status == "In Progress"
        assert context.jira_assignee == "developer"

    def test_creates_context_with_epic_and_parent(self) -> None:
        """Should populate epic and parent key fields."""
        issue = JiraIssue(
            key="DS-456",
            summary="Sub-task",
            epic_key="EPIC-100",
            parent_key="DS-400",
        )

        context = TemplateContext.from_jira_issue(issue)

        assert context.jira_epic_key == "EPIC-100"
        assert context.jira_parent_key == "DS-400"

    def test_creates_context_with_lists(self) -> None:
        """Should store list fields for formatting."""
        issue = JiraIssue(
            key="DS-789",
            summary="Test",
            labels=["bug", "urgent"],
            comments=["Comment 1", "Comment 2"],
            links=["DS-100", "DS-200"],
        )

        context = TemplateContext.from_jira_issue(issue)

        assert context._jira_labels_list == ["bug", "urgent"]
        assert context._jira_comments_list == ["Comment 1", "Comment 2"]
        assert context._jira_links_list == ["DS-100", "DS-200"]

    def test_github_fields_empty_for_jira_issue(self) -> None:
        """Should have empty GitHub fields for Jira issues."""
        issue = JiraIssue(key="DS-123", summary="Test")

        context = TemplateContext.from_jira_issue(issue)

        assert context.github_issue_number == ""
        assert context.github_issue_title == ""
        assert context.github_issue_body == ""
        assert context.github_issue_state == ""
        assert context.github_issue_author == ""

    def test_includes_github_repo_context(self) -> None:
        """Should include GitHub repository context when provided."""
        issue = JiraIssue(key="DS-123", summary="Test")
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_jira_issue(issue, github)

        assert context.github_host == "github.com"
        assert context.github_org == "myorg"
        assert context.github_repo == "myrepo"


class TestTemplateContextFromGitHubIssue:
    """Tests for TemplateContext.from_github_issue factory method."""

    def test_creates_context_with_basic_fields(self) -> None:
        """Should populate basic GitHub fields."""
        issue = GitHubIssue(
            number=42,
            title="Fix bug",
            body="Bug description",
            state="open",
            author="developer",
        )

        context = TemplateContext.from_github_issue(issue)

        assert context.github_issue_number == "42"
        assert context.github_issue_title == "Fix bug"
        assert context.github_issue_body == "Bug description"
        assert context.github_issue_state == "open"
        assert context.github_issue_author == "developer"

    def test_creates_context_with_pr_fields(self) -> None:
        """Should populate PR-specific fields."""
        issue = GitHubIssue(
            number=99,
            title="Add feature",
            is_pull_request=True,
            head_ref="feature-branch",
            base_ref="main",
            draft=True,
        )

        context = TemplateContext.from_github_issue(issue)

        assert context.github_is_pr == "true"
        assert context.github_pr_head == "feature-branch"
        assert context.github_pr_base == "main"
        assert context.github_pr_draft == "true"

    def test_creates_context_with_parent_issue(self) -> None:
        """Should populate parent issue number."""
        issue = GitHubIssue(
            number=55,
            title="Sub-issue",
            parent_issue_number=20,
        )

        context = TemplateContext.from_github_issue(issue)

        assert context.github_parent_issue_number == "20"

    def test_creates_context_with_lists(self) -> None:
        """Should store list fields for formatting."""
        issue = GitHubIssue(
            number=42,
            title="Test",
            assignees=["user1", "user2"],
            labels=["bug", "critical"],
        )

        context = TemplateContext.from_github_issue(issue)

        assert context._github_issue_assignees_list == ["user1", "user2"]
        assert context._github_issue_labels_list == ["bug", "critical"]

    def test_jira_fields_empty_for_github_issue(self) -> None:
        """Should have empty Jira fields for GitHub issues."""
        issue = GitHubIssue(number=42, title="Test")

        context = TemplateContext.from_github_issue(issue)

        assert context.jira_issue_key == ""
        assert context.jira_summary == ""
        assert context.jira_description == ""
        assert context.jira_epic_key == ""
        assert context.jira_parent_key == ""

    def test_builds_github_issue_url(self) -> None:
        """Should build correct GitHub issue URL."""
        issue = GitHubIssue(number=42, title="Test", is_pull_request=False)
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_github_issue(issue, github)

        assert context.github_issue_url == "https://github.com/myorg/myrepo/issues/42"

    def test_builds_github_pr_url(self) -> None:
        """Should build correct GitHub PR URL."""
        issue = GitHubIssue(number=99, title="Test PR", is_pull_request=True)
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_github_issue(issue, github)

        assert context.github_issue_url == "https://github.com/myorg/myrepo/pull/99"


class TestTemplateContextToDict:
    """Tests for TemplateContext.to_dict method using dataclass introspection."""

    def test_includes_all_public_fields(self) -> None:
        """Should include all non-private fields in dict."""
        context = TemplateContext(
            github_host="github.com",
            github_org="myorg",
            github_repo="myrepo",
            jira_issue_key="DS-123",
            jira_summary="Test summary",
        )

        result = context.to_dict()

        assert result["github_host"] == "github.com"
        assert result["github_org"] == "myorg"
        assert result["github_repo"] == "myrepo"
        assert result["jira_issue_key"] == "DS-123"
        assert result["jira_summary"] == "Test summary"

    def test_excludes_private_fields(self) -> None:
        """Should not include private fields (prefixed with _) in dict."""
        context = TemplateContext(
            jira_issue_key="DS-123",
            _jira_labels_list=["bug", "urgent"],
            _jira_comments_list=["Comment 1"],
        )

        result = context.to_dict()

        assert "_jira_labels_list" not in result
        assert "_jira_comments_list" not in result

    def test_includes_computed_slug_fields(self) -> None:
        """Should include computed slug fields."""
        context = TemplateContext(
            jira_summary="Add Login Button",
            github_issue_title="Fix Authentication Bug",
        )

        result = context.to_dict()

        assert result["jira_summary_slug"] == "add-login-button"
        assert result["github_issue_title_slug"] == "fix-authentication-bug"

    def test_includes_formatted_list_fields(self) -> None:
        """Should include formatted list fields."""
        context = TemplateContext(
            _jira_labels_list=["bug", "urgent"],
            _jira_comments_list=["First comment", "Second comment"],
            _jira_links_list=["DS-100", "DS-200"],
            _github_issue_assignees_list=["user1", "user2"],
            _github_issue_labels_list=["enhancement"],
        )

        result = context.to_dict()

        assert result["jira_labels"] == "bug, urgent"
        assert "1. First comment" in result["jira_comments"]
        assert "2. Second comment" in result["jira_comments"]
        assert result["jira_links"] == "DS-100, DS-200"
        assert result["github_issue_assignees"] == "user1, user2"
        assert result["github_issue_labels"] == "enhancement"

    def test_all_values_are_strings(self) -> None:
        """Should convert all values to strings."""
        context = TemplateContext(
            jira_issue_key="DS-123",
            github_issue_number="42",
        )

        result = context.to_dict()

        for key, value in result.items():
            assert isinstance(value, str), f"{key} should be str, got {type(value)}"

    def test_empty_fields_are_empty_strings(self) -> None:
        """Should have empty strings for unset fields."""
        context = TemplateContext()

        result = context.to_dict()

        assert result["jira_issue_key"] == ""
        assert result["github_issue_number"] == ""
        assert result["jira_labels"] == ""
        assert result["github_issue_assignees"] == ""


class TestTemplateContextIntegration:
    """Integration tests for TemplateContext with real issue objects."""

    def test_jira_issue_full_workflow(self) -> None:
        """Should correctly process a fully populated Jira issue."""
        issue = JiraIssue(
            key="DS-123",
            summary="Implement Feature X",
            description="Full description here",
            status="In Progress",
            assignee="developer",
            labels=["feature", "v2.0"],
            comments=["First review", "Second review", "Approved"],
            links=["DS-100", "DS-101"],
            epic_key="EPIC-50",
            parent_key="DS-100",
        )
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_jira_issue(issue, github)
        result = context.to_dict()

        # Verify all fields are present
        assert result["jira_issue_key"] == "DS-123"
        assert result["jira_summary"] == "Implement Feature X"
        assert result["jira_summary_slug"] == "implement-feature-x"
        assert result["jira_labels"] == "feature, v2.0"
        assert result["jira_epic_key"] == "EPIC-50"
        assert result["jira_parent_key"] == "DS-100"
        assert result["github_host"] == "github.com"
        # GitHub issue fields should be empty
        assert result["github_issue_number"] == ""

    def test_github_issue_full_workflow(self) -> None:
        """Should correctly process a fully populated GitHub issue."""
        issue = GitHubIssue(
            number=42,
            title="Fix Critical Bug",
            body="Bug details here",
            state="open",
            author="contributor",
            assignees=["reviewer1", "reviewer2"],
            labels=["bug", "priority-high"],
            is_pull_request=True,
            head_ref="fix/critical-bug",
            base_ref="main",
            draft=False,
            parent_issue_number=10,
        )
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_github_issue(issue, github)
        result = context.to_dict()

        # Verify all fields are present
        assert result["github_issue_number"] == "42"
        assert result["github_issue_title"] == "Fix Critical Bug"
        assert result["github_issue_title_slug"] == "fix-critical-bug"
        assert result["github_issue_assignees"] == "reviewer1, reviewer2"
        assert result["github_issue_labels"] == "bug, priority-high"
        assert result["github_is_pr"] == "true"
        assert result["github_pr_head"] == "fix/critical-bug"
        assert result["github_parent_issue_number"] == "10"
        assert result["github_issue_url"] == "https://github.com/myorg/myrepo/pull/42"
        # Jira fields should be empty
        assert result["jira_issue_key"] == ""


class TestBaseBranchTemplateVariable:
    """Tests for {base_branch} template variable (DS-905).

    Verifies that base_branch is available in to_dict() output,
    substituted in branch patterns and prompts, and defaults
    to "main" when github context is absent.
    """

    def test_base_branch_appears_in_to_dict(self) -> None:
        """Should include base_branch in to_dict() output."""
        context = TemplateContext(base_branch="develop")

        result = context.to_dict()

        assert "base_branch" in result
        assert result["base_branch"] == "develop"

    def test_base_branch_default_value_in_to_dict(self) -> None:
        """Should default base_branch to 'main' in to_dict() output."""
        context = TemplateContext()

        result = context.to_dict()

        assert result["base_branch"] == "main"

    def test_base_branch_from_jira_issue_with_github_context(self) -> None:
        """Should populate base_branch from github.base_branch for Jira issues."""
        issue = JiraIssue(key="DS-123", summary="Test issue")
        github = GitHubContext(
            host="github.com", org="myorg", repo="myrepo", base_branch="develop"
        )

        context = TemplateContext.from_jira_issue(issue, github)

        assert context.base_branch == "develop"
        assert context.to_dict()["base_branch"] == "develop"

    def test_base_branch_from_jira_issue_without_github_context(self) -> None:
        """Should default base_branch to 'main' when github context is absent for Jira issues."""
        issue = JiraIssue(key="DS-123", summary="Test issue")

        context = TemplateContext.from_jira_issue(issue)

        assert context.base_branch == "main"
        assert context.to_dict()["base_branch"] == "main"

    def test_base_branch_from_github_issue_with_github_context(self) -> None:
        """Should populate base_branch from github.base_branch for GitHub issues."""
        issue = GitHubIssue(number=42, title="Fix bug")
        github = GitHubContext(
            host="github.com", org="myorg", repo="myrepo", base_branch="release"
        )

        context = TemplateContext.from_github_issue(issue, github)

        assert context.base_branch == "release"
        assert context.to_dict()["base_branch"] == "release"

    def test_base_branch_from_github_issue_without_github_context(self) -> None:
        """Should default base_branch to 'main' when github context is absent for GitHub issues."""
        issue = GitHubIssue(number=42, title="Fix bug")

        context = TemplateContext.from_github_issue(issue)

        assert context.base_branch == "main"
        assert context.to_dict()["base_branch"] == "main"

    def test_base_branch_substituted_in_prompts(self) -> None:
        """Should substitute {base_branch} in prompts."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Test")
        orch = make_orchestration(
            prompt="Create a PR targeting {base_branch}",
            github=GitHubContext(
                host="github.com", org="myorg", repo="myrepo", base_branch="develop"
            ),
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Create a PR targeting develop"

    def test_base_branch_substituted_in_branch_patterns(self) -> None:
        """Should substitute {base_branch} in branch patterns."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Test")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{base_branch}/{jira_issue_key}",
                base_branch="develop",
            ),
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/develop/DS-123"

    def test_base_branch_default_main_in_prompt_without_github(self) -> None:
        """Should use default 'main' for {base_branch} in prompts when no github context."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Test")
        orch = make_orchestration(prompt="Target branch: {base_branch}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Target branch: main"

    def test_base_branch_with_github_default_base_branch(self) -> None:
        """Should use github default base_branch 'main' when not explicitly set."""
        issue = JiraIssue(key="DS-123", summary="Test issue")
        github = GitHubContext(host="github.com", org="myorg", repo="myrepo")

        context = TemplateContext.from_jira_issue(issue, github)

        # GitHubContext defaults base_branch to "main"
        assert context.base_branch == "main"
        assert context.to_dict()["base_branch"] == "main"
