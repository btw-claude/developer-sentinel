"""Template variable tests for Claude Agent SDK executor module.

This module contains tests for advanced template variable features:
- Parent/epic template variables
- Parent/epic branch pattern expansion
- Parent/epic integration tests
- Slugify utility function
- Slug template variables
"""

import pytest

from sentinel.executor import AgentExecutor, slugify
from sentinel.orchestration import GitHubContext
from sentinel.poller import JiraIssue
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


class TestParentEpicTemplateVariables:
    """Tests for parent/epic template variables.

    These tests verify that the new template variables for parent and epic
    relationships are correctly resolved in both prompts and branch patterns:
    - {jira_epic_key}: The epic key for Jira issues
    - {jira_parent_key}: The parent key for Jira sub-tasks
    - {github_parent_issue_number}: The parent issue number for GitHub sub-issues
    """

    @pytest.fixture
    def mock_client(self) -> MockAgentClient:
        """Create a MockAgentClient for testing."""
        return MockAgentClient()

    @pytest.fixture
    def executor(self, mock_client: MockAgentClient) -> AgentExecutor:
        """Create an AgentExecutor with a mock client."""
        return AgentExecutor(mock_client)

    # Tests for Jira epic/parent template variables in prompts

    def test_substitutes_jira_epic_key(self, executor: AgentExecutor) -> None:
        """Should substitute {jira_epic_key} with epic key value."""
        issue = JiraIssue(key="DS-123", summary="Test", epic_key="EPIC-100")
        orch = make_orchestration(prompt="Epic: {jira_epic_key}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Epic: EPIC-100"

    def test_substitutes_jira_parent_key(self, executor: AgentExecutor) -> None:
        """Should substitute {jira_parent_key} with parent key value."""
        issue = JiraIssue(key="DS-456", summary="Sub-task", parent_key="DS-400")
        orch = make_orchestration(prompt="Parent: {jira_parent_key}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Parent: DS-400"

    def test_jira_epic_key_empty_when_none(self, executor: AgentExecutor) -> None:
        """Should substitute {jira_epic_key} with empty string when None."""
        issue = JiraIssue(key="DS-789", summary="No epic", epic_key=None)
        orch = make_orchestration(prompt="Epic: [{jira_epic_key}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Epic: []"

    def test_jira_parent_key_empty_when_none(self, executor: AgentExecutor) -> None:
        """Should substitute {jira_parent_key} with empty string when None."""
        issue = JiraIssue(key="DS-999", summary="No parent", parent_key=None)
        orch = make_orchestration(prompt="Parent: [{jira_parent_key}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Parent: []"

    def test_combined_epic_and_parent_key_variables(self, executor: AgentExecutor) -> None:
        """Should substitute both epic and parent key in same prompt."""
        issue = JiraIssue(key="DS-123", summary="Sub-task", epic_key="EPIC-50", parent_key="DS-100")
        orch = make_orchestration(
            prompt="Issue {jira_issue_key} in epic {jira_epic_key} under {jira_parent_key}"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Issue DS-123 in epic EPIC-50 under DS-100"

    # Tests for GitHub parent issue number template variable

    def test_substitutes_github_parent_issue_number(self, executor: AgentExecutor) -> None:
        """Should substitute {github_parent_issue_number} with parent issue number."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Sub-issue", parent_issue_number=10)
        orch = make_orchestration(prompt="Parent: #{github_parent_issue_number}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Parent: #10"

    def test_github_parent_issue_number_empty_when_none(self, executor: AgentExecutor) -> None:
        """Should substitute {github_parent_issue_number} with empty string when None."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Top-level issue", parent_issue_number=None)
        orch = make_orchestration(prompt="Parent: [{github_parent_issue_number}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Parent: []"

    def test_github_parent_combined_with_other_variables(self, executor: AgentExecutor) -> None:
        """Should substitute parent_issue_number along with other GitHub variables."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(
            number=42, title="Fix sub-issue", parent_issue_number=10, author="developer"
        )
        orch = make_orchestration(
            prompt="#{github_issue_number}: {github_issue_title} (parent: #{github_parent_issue_number}) by {github_issue_author}"
        )

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "#42: Fix sub-issue (parent: #10) by developer"

    # Tests for cross-source variable isolation

    def test_jira_epic_parent_empty_for_github_issues(self, executor: AgentExecutor) -> None:
        """Jira epic/parent variables should be empty for GitHub issues."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Test", parent_issue_number=10)
        orch = make_orchestration(prompt="Epic: [{jira_epic_key}] Parent: [{jira_parent_key}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Epic: [] Parent: []"

    def test_github_parent_empty_for_jira_issues(self, executor: AgentExecutor) -> None:
        """GitHub parent variable should be empty for Jira issues."""
        issue = JiraIssue(key="DS-123", summary="Test", epic_key="EPIC-100", parent_key="DS-100")
        orch = make_orchestration(prompt="GitHub Parent: [{github_parent_issue_number}]")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "GitHub Parent: []"


class TestParentEpicBranchPatternExpansion:
    """Tests for parent/epic template variables in branch patterns.

    These tests verify branch pattern expansion with the new parent/epic
    template variables, enabling patterns like:
    - feature/{jira_epic_key}/{jira_issue_key}
    - fix/{github_parent_issue_number}/{github_issue_number}
    """

    @pytest.fixture
    def mock_client(self) -> MockAgentClient:
        """Create a MockAgentClient for testing."""
        return MockAgentClient()

    @pytest.fixture
    def executor(self, mock_client: MockAgentClient) -> AgentExecutor:
        """Create an AgentExecutor with a mock client."""
        return AgentExecutor(mock_client)

    def test_expand_branch_pattern_with_jira_epic_key(self, executor: AgentExecutor) -> None:
        """Should expand {jira_epic_key} in branch pattern."""
        issue = JiraIssue(key="DS-123", summary="Test", epic_key="EPIC-100")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_epic_key}/{jira_issue_key}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/EPIC-100/DS-123"

    def test_expand_branch_pattern_with_jira_parent_key(self, executor: AgentExecutor) -> None:
        """Should expand {jira_parent_key} in branch pattern."""
        issue = JiraIssue(key="DS-456", summary="Sub-task", parent_key="DS-400")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{jira_parent_key}/{jira_issue_key}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "fix/DS-400/DS-456"

    def test_expand_branch_pattern_with_github_parent_issue_number(
        self, executor: AgentExecutor
    ) -> None:
        """Should expand {github_parent_issue_number} in branch pattern."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Sub-issue fix", parent_issue_number=10)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{github_parent_issue_number}/{github_issue_number}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "fix/10/42"

    def test_expand_branch_pattern_epic_empty_when_none(
        self, executor: AgentExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Branch pattern should be rejected when empty epic produces consecutive slashes.

        When epic_key is None, it produces an empty string which results in
        'feature//DS-123' - consecutive slashes are invalid git branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        issue = JiraIssue(key="DS-123", summary="No epic", epic_key=None)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_epic_key}/{jira_issue_key}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Empty epic_key results in consecutive slashes which is invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "consecutive" in caplog.text

    def test_expand_branch_pattern_github_parent_empty_when_none(
        self, executor: AgentExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Branch pattern should be rejected when empty parent produces consecutive slashes.

        When parent_issue_number is None, it produces an empty string which results in
        'fix//42' - consecutive slashes are invalid git branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Top-level", parent_issue_number=None)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{github_parent_issue_number}/{github_issue_number}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Empty parent results in consecutive slashes which is invalid
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "consecutive" in caplog.text

    def test_expand_branch_pattern_multiple_parent_variables(self, executor: AgentExecutor) -> None:
        """Should expand multiple parent/epic variables in complex patterns."""
        issue = JiraIssue(
            key="DS-789", summary="Work item", epic_key="EPIC-50", parent_key="DS-700"
        )
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="{jira_epic_key}/{jira_parent_key}/{jira_issue_key}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "EPIC-50/DS-700/DS-789"

    def test_expand_branch_pattern_fallback_with_missing_parent(
        self, executor: AgentExecutor
    ) -> None:
        """Test pattern handles missing parent/epic gracefully for branching."""
        # Issue without epic or parent - simulates standalone issue
        issue = JiraIssue(key="DS-999", summary="Standalone", epic_key=None, parent_key=None)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                # Pattern using only issue key as fallback when no parent
                branch="feature/{jira_issue_key}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        # Works fine since jira_issue_key is always present
        assert result == "feature/DS-999"

    def test_expand_branch_pattern_github_parent_jira_variables_empty(
        self, executor: AgentExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Branch pattern should be rejected when Jira variable is empty for GitHub issues.

        When using Jira-specific variables with GitHub issues, they produce empty strings
        which can result in consecutive slashes - invalid git branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(number=42, title="Fix", parent_issue_number=10)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_epic_key}/{github_issue_number}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # jira_epic_key is empty for GitHub issues, producing consecutive slashes
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "consecutive" in caplog.text

    def test_expand_branch_pattern_runtime_validation_rejects_invalid(
        self, executor: AgentExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that runtime validation rejects invalid expanded branch names.

        When template substitution produces an invalid branch name
        (e.g., from a summary with invalid characters), the function
        should return None and log a warning.
        """
        import logging

        # Create an issue with a summary that contains invalid characters
        issue = JiraIssue(
            key="DS-123",
            summary="Add feature with space",  # Space is invalid in branch names
        )
        # The jira_summary_slug should sanitize this, but test with direct summary
        # Actually, let's test with a pattern that would produce ".." (consecutive dots)
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature..{jira_issue_key}",  # Contains .. which is invalid
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Should return None due to invalid branch name
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert "consecutive" in caplog.text

    def test_expand_branch_pattern_runtime_validation_rejects_lock_suffix(
        self, executor: AgentExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that runtime validation rejects branch names ending with .lock.

        Git disallows branch names ending in .lock as they conflict with
        git's internal lock files. This test verifies the new .lock validation.
        """
        import logging

        issue = JiraIssue(key="DS-123", summary="Test lock issue")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_issue_key}.lock",  # Invalid: ends with .lock
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # Should return None due to .lock suffix
        assert result is None
        assert "Invalid expanded branch name" in caplog.text
        assert ".lock" in caplog.text

    def test_expand_branch_pattern_runtime_validation_valid_passes(
        self, executor: AgentExecutor
    ) -> None:
        """Test that valid expanded branch names pass runtime validation."""
        issue = JiraIssue(key="DS-123", summary="Add login button")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_issue_key}/{jira_summary_slug}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        # Should succeed with a valid branch name
        assert result is not None
        assert result == "feature/DS-123/add-login-button"


class TestParentEpicIntegration:
    """Integration tests for parent/epic template variables.

    These tests verify end-to-end integration of parent/epic template
    variables with the executor, ensuring they work correctly in
    realistic orchestration scenarios.
    """

    @pytest.fixture
    def mock_client(self) -> MockAgentClient:
        """Create a MockAgentClient with success response for testing."""
        return MockAgentClient(responses=["SUCCESS: Completed"])

    @pytest.fixture
    def executor(self, mock_client: MockAgentClient) -> AgentExecutor:
        """Create an AgentExecutor with a mock client."""
        return AgentExecutor(mock_client)

    def test_execute_with_jira_epic_in_prompt(
        self, mock_client: MockAgentClient, executor: AgentExecutor
    ) -> None:
        """Execute should properly substitute Jira epic key in prompt."""
        issue = JiraIssue(
            key="DS-123", summary="Fix bug in epic", epic_key="EPIC-100", parent_key="DS-100"
        )
        orch = make_orchestration(
            prompt="Fix issue {jira_issue_key} in epic {jira_epic_key} (parent: {jira_parent_key})"
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        # Verify the prompt was correctly built
        prompt = mock_client.calls[0][0]
        assert prompt == "Fix issue DS-123 in epic EPIC-100 (parent: DS-100)"

    def test_execute_with_github_parent_in_prompt(
        self, mock_client: MockAgentClient, executor: AgentExecutor
    ) -> None:
        """Execute should properly substitute GitHub parent issue number in prompt."""
        from sentinel.github_poller import GitHubIssue

        issue = GitHubIssue(
            number=42,
            title="Fix for parent issue",
            parent_issue_number=10,
            author="developer",
        )
        orch = make_orchestration(
            prompt="Implement fix for #{github_issue_number} (parent: #{github_parent_issue_number})"
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        prompt = mock_client.calls[0][0]
        assert prompt == "Implement fix for #42 (parent: #10)"

    def test_full_workflow_jira_with_epic_branch(
        self, mock_client: MockAgentClient, executor: AgentExecutor
    ) -> None:
        """Test complete workflow: Jira issue with epic-based branch pattern."""
        mock_client.responses = ["SUCCESS: All changes committed"]
        issue = JiraIssue(
            key="DS-456",
            summary="Implement feature",
            description="Add new feature under epic",
            epic_key="FEAT-100",
            labels=["feature"],
        )
        orch = make_orchestration(
            prompt="Implement {jira_summary} for {jira_issue_key} under epic {jira_epic_key}",
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_epic_key}/{jira_issue_key}",
            ),
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        # Verify prompt substitution
        prompt = mock_client.calls[0][0]
        assert prompt == "Implement Implement feature for DS-456 under epic FEAT-100"

    def test_full_workflow_github_with_parent_branch(
        self, mock_client: MockAgentClient, executor: AgentExecutor
    ) -> None:
        """Test complete workflow: GitHub issue with parent-based branch pattern."""
        from sentinel.github_poller import GitHubIssue

        mock_client.responses = ["SUCCESS: PR created"]
        issue = GitHubIssue(
            number=55,
            title="Sub-task implementation",
            body="Implement part of parent issue",
            parent_issue_number=20,
            author="contributor",
            labels=["enhancement"],
        )
        orch = make_orchestration(
            prompt="Complete #{github_issue_number}: {github_issue_title} (child of #{github_parent_issue_number})",
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{github_parent_issue_number}/{github_issue_number}",
            ),
        )

        result = executor.execute(issue, orch)

        assert result.succeeded is True
        prompt = mock_client.calls[0][0]
        assert prompt == "Complete #55: Sub-task implementation (child of #20)"


class TestSlugify:
    """Tests for the slugify utility function."""

    def test_converts_spaces_to_hyphens(self) -> None:
        """Should convert spaces to hyphens."""
        assert slugify("Add login button") == "add-login-button"

    def test_converts_to_lowercase(self) -> None:
        """Should convert to lowercase."""
        assert slugify("Fix Authentication Bug") == "fix-authentication-bug"

    def test_removes_special_characters(self) -> None:
        """Should remove special characters invalid in git branch names."""
        assert slugify("Fix: bug!") == "fix-bug"
        assert slugify("Feature [WIP]") == "feature-wip"
        assert slugify("Test?") == "test"
        assert slugify("Hello*World") == "helloworld"

    def test_normalizes_unicode(self) -> None:
        """Should normalize unicode characters to ASCII equivalents."""
        assert slugify("Éclair résumé") == "eclair-resume"
        assert slugify("Café") == "cafe"
        assert slugify("naïve") == "naive"

    def test_collapses_multiple_hyphens(self) -> None:
        """Should collapse multiple consecutive hyphens into one."""
        assert slugify("foo  bar") == "foo-bar"
        assert slugify("foo---bar") == "foo-bar"
        assert slugify("foo - bar") == "foo-bar"

    def test_strips_leading_trailing_hyphens(self) -> None:
        """Should strip leading and trailing hyphens."""
        assert slugify("-hello-") == "hello"
        assert slugify("--test--") == "test"
        assert slugify(" hello ") == "hello"

    def test_handles_underscores(self) -> None:
        """Should convert underscores to hyphens."""
        assert slugify("foo_bar") == "foo-bar"
        assert slugify("hello_world_test") == "hello-world-test"

    def test_handles_empty_string(self) -> None:
        """Should return empty string for empty input."""
        assert slugify("") == ""

    def test_handles_only_special_chars(self) -> None:
        """Should return empty string when input is only special chars."""
        assert slugify("!@#$%") == ""

    def test_preserves_dots(self) -> None:
        """Should preserve single dots (valid in git branch names)."""
        assert slugify("v1.0.0") == "v1.0.0"

    def test_collapses_consecutive_dots(self) -> None:
        """Should collapse consecutive dots (invalid in git branch names)."""
        assert slugify("test..case") == "test.case"

    def test_preserves_forward_slashes(self) -> None:
        """Should preserve forward slashes (valid in git branch names)."""
        # Note: slashes in input are typically from path-like structures
        # but slugify doesn't add slashes, it just processes the text
        assert slugify("feature/test") == "feature/test"

    def test_complex_example(self) -> None:
        """Should handle complex real-world examples."""
        assert slugify("Add user authentication [URGENT]") == "add-user-authentication-urgent"
        assert slugify("Fix: Login bug (critical!)") == "fix-login-bug-critical"
        assert slugify("JIRA-123: Implement feature") == "jira-123-implement-feature"


class TestSlugTemplateVariables:
    """Tests for slugified template variables in branch patterns."""

    def test_jira_summary_slug_in_branch_pattern(self) -> None:
        """Should expand {jira_summary_slug} with slugified summary."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Add login button")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_issue_key}-{jira_summary_slug}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/DS-123-add-login-button"

    def test_jira_summary_slug_with_special_chars(self) -> None:
        """Should handle special characters in Jira summary when slugified."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-456", summary="Fix: Critical bug [URGENT]!")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{jira_summary_slug}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "fix/fix-critical-bug-urgent"

    def test_github_issue_title_slug_in_branch_pattern(self) -> None:
        """Should expand {github_issue_title_slug} with slugified title."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Add authentication flow")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{github_issue_number}-{github_issue_title_slug}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "feature/42-add-authentication-flow"

    def test_github_issue_title_slug_with_special_chars(self) -> None:
        """Should handle special characters in GitHub title when slugified."""
        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=99, title="Bug: Login fails [priority: high]")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="fix/{github_issue_title_slug}",
            )
        )

        result = executor._expand_branch_pattern(issue, orch)

        assert result == "fix/bug-login-fails-priority-high"

    def test_raw_vs_slug_comparison(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should show difference between raw and slug variables.

        Raw variables with spaces produce invalid branch names.
        Slugified variables produce valid branch-safe names.
        Runtime validation correctly distinguishes between these.
        """
        import logging

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-789", summary="Add User Profile Page")

        # Test raw (non-slugified) - should be rejected due to spaces
        orch_raw = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_summary}",
            )
        )
        with caplog.at_level(logging.WARNING):
            result_raw = executor._expand_branch_pattern(issue, orch_raw)
        assert result_raw is None  # Rejected due to spaces
        assert "Invalid expanded branch name" in caplog.text

        caplog.clear()

        # Test slugified - should succeed
        orch_slug = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_summary_slug}",
            )
        )
        result_slug = executor._expand_branch_pattern(issue, orch_slug)
        assert result_slug == "feature/add-user-profile-page"  # Branch-safe

    def test_cross_source_jira_slug_with_github_issue(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should reject pattern when Jira slug produces trailing slash.

        When using Jira-specific slug variables with GitHub issues, they produce
        empty strings which result in trailing slashes - invalid branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        from sentinel.github_poller import GitHubIssue

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = GitHubIssue(number=42, title="Some title")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{jira_summary_slug}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # jira_summary_slug is empty for GitHub issues, producing trailing slash
        assert result is None
        assert "Invalid expanded branch name" in caplog.text

    def test_cross_source_github_slug_with_jira_issue(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should reject pattern when GitHub slug produces trailing slash.

        When using GitHub-specific slug variables with Jira issues, they produce
        empty strings which result in trailing slashes - invalid branch names.
        Runtime validation now correctly rejects this pattern.
        """
        import logging

        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-123", summary="Some summary")
        orch = make_orchestration(
            github=GitHubContext(
                host="github.com",
                org="myorg",
                repo="myrepo",
                branch="feature/{github_issue_title_slug}",
            )
        )

        with caplog.at_level(logging.WARNING):
            result = executor._expand_branch_pattern(issue, orch)

        # github_issue_title_slug is empty for Jira issues, producing trailing slash
        assert result is None
        assert "Invalid expanded branch name" in caplog.text

    def test_slug_in_prompt_substitution(self) -> None:
        """Should substitute slug variables in prompts as well."""
        client = MockAgentClient()
        executor = AgentExecutor(client)
        issue = make_issue(key="DS-100", summary="Add Login Button")
        orch = make_orchestration(prompt="Create branch: feature/{jira_summary_slug}")

        prompt = executor.build_prompt(issue, orch)

        assert prompt == "Create branch: feature/add-login-button"
