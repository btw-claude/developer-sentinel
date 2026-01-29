"""Tests for Sentinel GitHub integration functionality."""

import logging

import pytest

from sentinel.main import Sentinel

# Import shared fixtures and helpers from conftest.py
from tests.conftest import (
    MockAgentClient,
    MockJiraClient,
    MockTagClient,
    build_github_trigger_key,
    make_config,
    make_orchestration,
)


class TestExtractRepoFromUrl:
    """Tests for extract_repo_from_url function."""

    def test_extracts_from_issue_url(self) -> None:
        """Test extraction from GitHub issue URL."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/org/repo/issues/123"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_extracts_from_pull_url(self) -> None:
        """Test extraction from GitHub pull request URL."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/my-org/my-repo/pull/456"
        result = extract_repo_from_url(url)
        assert result == "my-org/my-repo"

    def test_handles_empty_url(self) -> None:
        """Test that empty URL returns None."""
        from sentinel.main import extract_repo_from_url

        assert extract_repo_from_url("") is None
        assert extract_repo_from_url(None) is None  # type: ignore[arg-type]

    def test_handles_invalid_url(self) -> None:
        """Test that invalid URLs return None."""
        from sentinel.main import extract_repo_from_url

        assert extract_repo_from_url("not a url") is None
        assert extract_repo_from_url("https://github.com/org/repo") is None
        assert extract_repo_from_url("https://github.com/org/repo/commits/abc") is None

    def test_handles_enterprise_github_url(self) -> None:
        """Test extraction from GitHub Enterprise URLs."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.enterprise.com/org/repo/issues/123"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_handles_http_url(self) -> None:
        """Test extraction from HTTP URL (no HTTPS)."""
        from sentinel.main import extract_repo_from_url

        url = "http://github.com/org/repo/pull/789"
        result = extract_repo_from_url(url)
        assert result == "org/repo"

    def test_handles_complex_repo_names(self) -> None:
        """Test extraction with complex org/repo names."""
        from sentinel.main import extract_repo_from_url

        url = "https://github.com/my-org-123/my-repo-name/issues/1"
        result = extract_repo_from_url(url)
        assert result == "my-org-123/my-repo-name"


class TestGitHubIssueWithRepo:
    """Tests for GitHubIssueWithRepo class."""

    def test_key_includes_repo_context(self) -> None:
        """Test that key property includes full repo context."""
        from sentinel.github_poller import GitHubIssue
        from sentinel.main import GitHubIssueWithRepo

        issue = GitHubIssue(number=123, title="Test")
        wrapper = GitHubIssueWithRepo(issue, "org/repo")

        assert wrapper.key == "org/repo#123"

    def test_delegates_all_properties(self) -> None:
        """Test that all properties are properly delegated via __getattr__."""
        from sentinel.github_poller import GitHubIssue
        from sentinel.main import GitHubIssueWithRepo

        issue = GitHubIssue(
            number=42,
            title="Test Title",
            body="Test Body",
            state="open",
            author="testuser",
            assignees=["user1", "user2"],
            labels=["bug", "urgent"],
            is_pull_request=True,
            head_ref="feature-branch",
            base_ref="main",
            draft=False,
            repo_url="https://github.com/org/repo/pull/42",
            parent_issue_number=100,
        )
        wrapper = GitHubIssueWithRepo(issue, "org/repo")

        # Verify all GitHubIssueProtocol properties are delegated
        assert wrapper.number == 42
        assert wrapper.title == "Test Title"
        assert wrapper.body == "Test Body"
        assert wrapper.state == "open"
        assert wrapper.author == "testuser"
        assert wrapper.assignees == ["user1", "user2"]
        assert wrapper.labels == ["bug", "urgent"]
        assert wrapper.is_pull_request is True
        assert wrapper.head_ref == "feature-branch"
        assert wrapper.base_ref == "main"
        assert wrapper.draft is False
        assert wrapper.repo_url == "https://github.com/org/repo/pull/42"
        assert wrapper.parent_issue_number == 100

    def test_raises_attribute_error_for_invalid_attribute(self) -> None:
        """Test that AttributeError is raised for non-existent attributes."""
        from sentinel.github_poller import GitHubIssue
        from sentinel.main import GitHubIssueWithRepo

        issue = GitHubIssue(number=123, title="Test")
        wrapper = GitHubIssueWithRepo(issue, "org/repo")

        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attribute


class TestAddRepoContextFromUrls:
    """Tests for _add_repo_context_from_urls method."""

    def test_wraps_issues_with_repo_context(self) -> None:
        """Test that issues are wrapped with repo context from URL."""
        from sentinel.github_poller import GitHubIssue

        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        issues = [
            GitHubIssue(
                number=1,
                title="Issue 1",
                repo_url="https://github.com/org1/repo1/issues/1",
            ),
            GitHubIssue(
                number=2,
                title="Issue 2",
                repo_url="https://github.com/org2/repo2/pull/2",
            ),
        ]

        result = sentinel._add_repo_context_from_urls(issues)

        assert len(result) == 2
        assert result[0].key == "org1/repo1#1"
        assert result[1].key == "org2/repo2#2"

    def test_skips_issues_with_invalid_urls(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that issues with invalid URLs are skipped with warning."""
        from sentinel.github_poller import GitHubIssue

        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        issues = [
            GitHubIssue(
                number=1,
                title="Valid Issue",
                repo_url="https://github.com/org/repo/issues/1",
            ),
            GitHubIssue(
                number=2,
                title="Invalid Issue",
                repo_url="invalid-url",
            ),
            GitHubIssue(
                number=3,
                title="Empty URL Issue",
                repo_url="",
            ),
        ]

        with caplog.at_level(logging.WARNING):
            result = sentinel._add_repo_context_from_urls(issues)

        # Only valid issue should be returned
        assert len(result) == 1
        assert result[0].key == "org/repo#1"

        # Warnings should be logged for invalid URLs
        assert "Could not extract repo from URL for issue #2" in caplog.text
        assert "Could not extract repo from URL for issue #3" in caplog.text

    def test_handles_mixed_repos_in_project(self) -> None:
        """Test handling of issues from multiple repos in a single project."""
        from sentinel.github_poller import GitHubIssue

        jira_client = MockJiraClient(issues=[])
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config()
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_client=jira_client,
            agent_client=agent_client,
            tag_client=tag_client,
        )

        # A project can contain issues from multiple repositories
        issues = [
            GitHubIssue(
                number=10,
                title="From repo-a",
                repo_url="https://github.com/org/repo-a/issues/10",
            ),
            GitHubIssue(
                number=20,
                title="From repo-b",
                repo_url="https://github.com/org/repo-b/pull/20",
            ),
            GitHubIssue(
                number=30,
                title="From repo-c",
                repo_url="https://github.com/different-org/repo-c/issues/30",
            ),
        ]

        result = sentinel._add_repo_context_from_urls(issues)

        assert len(result) == 3
        assert result[0].key == "org/repo-a#10"
        assert result[1].key == "org/repo-b#20"
        assert result[2].key == "different-org/repo-c#30"


class TestGitHubTriggerDeduplication:
    """Tests for GitHub trigger deduplication logic.

    These tests verify that:
    1. Same project with different project_filter values are polled separately
    2. Same project with different labels values are polled separately
    3. Identical triggers are still properly deduplicated
    4. None/empty values in project_filter and labels are handled cleanly
    """

    def test_same_project_different_project_filter_polled_separately(self) -> None:
        """Test that same project with different project_filter values are polled separately."""

        # Create orchestrations with same project but different project_filter
        orch1 = make_orchestration(
            name="orch-ready",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
        )
        orch2 = make_orchestration(
            name="orch-in-progress",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "In Progress"',
        )

        # Use shared helper for deduplication keys
        key1 = build_github_trigger_key(orch1)
        key2 = build_github_trigger_key(orch2)

        # Keys should be different due to different project_filter values
        assert key1 != key2
        assert 'Status = "Ready"' in key1
        assert 'Status = "In Progress"' in key2

    def test_same_project_different_labels_polled_separately(self) -> None:
        """Test that same project with different labels values are polled separately."""
        # Create orchestrations with same project but different labels
        orch1 = make_orchestration(
            name="orch-bugs",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            labels=["bug"],
        )
        orch2 = make_orchestration(
            name="orch-features",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            labels=["feature", "enhancement"],
        )

        # Use shared helper for deduplication keys
        key1 = build_github_trigger_key(orch1)
        key2 = build_github_trigger_key(orch2)

        # Keys should be different due to different labels values
        assert key1 != key2
        assert "bug" in key1
        assert "feature,enhancement" in key2

    def test_identical_triggers_deduplicated(self) -> None:
        """Test that identical triggers are properly deduplicated."""
        # Create orchestrations with identical trigger configurations
        orch1 = make_orchestration(
            name="orch-1",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            labels=["bug"],
        )
        orch2 = make_orchestration(
            name="orch-2",  # Different name, same trigger config
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            labels=["bug"],
        )

        # Use shared helper for deduplication keys
        key1 = build_github_trigger_key(orch1)
        key2 = build_github_trigger_key(orch2)

        # Keys should be identical since trigger configurations match
        assert key1 == key2

    def test_none_project_filter_produces_clean_key(self) -> None:
        """Test that None project_filter values produce clean deduplication keys.

        Previously, None values would result in the string 'None' in the key.
        None/empty values should now produce empty string in the key.
        """
        # Create orchestration with no project_filter (None/empty)
        orch = make_orchestration(
            name="orch-no-filter",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            # project_filter not specified (defaults to "")
        )

        # Use shared helper for deduplication key
        key = build_github_trigger_key(orch)

        # Key should NOT contain 'None' as a string
        assert "None" not in key
        # Key should have clean format: github:owner/number::
        assert key == "github:test-org/1::"

    def test_empty_labels_produces_clean_key(self) -> None:
        """Test that empty labels list produces clean deduplication keys."""
        # Create orchestration with no labels
        orch = make_orchestration(
            name="orch-no-labels",
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
            # labels not specified or empty
        )

        # Use shared helper for deduplication key
        key = build_github_trigger_key(orch)

        # Key should have clean format with empty labels part
        assert key == 'github:test-org/1:Status = "Ready":'

    def test_both_none_filter_and_empty_labels_clean_key(self) -> None:
        """Test that both None filter and empty labels produce clean key."""
        # Create orchestration with neither filter nor labels
        orch = make_orchestration(
            name="orch-minimal",
            source="github",
            project_number=5,
            project_owner="my-org",
            project_scope="org",
        )

        # Use shared helper for deduplication key
        key = build_github_trigger_key(orch)

        # Key should be clean without 'None' anywhere
        assert "None" not in key
        assert key == "github:my-org/5::"

    def test_combined_filter_and_labels_in_key(self) -> None:
        """Test that both filter and labels are included in deduplication key."""
        orch = make_orchestration(
            name="orch-full",
            source="github",
            project_number=10,
            project_owner="acme",
            project_scope="org",
            project_filter='Priority = "High"',
            labels=["urgent", "critical"],
        )

        # Use shared helper for deduplication key
        key = build_github_trigger_key(orch)

        # Verify all parts are present
        assert key == 'github:acme/10:Priority = "High":urgent,critical'
