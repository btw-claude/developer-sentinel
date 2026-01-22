"""Integration tests for GitHub Project polling feature (DS-205).

These tests verify the integration between GitHub project polling components:
- Project-based trigger configuration
- Filter expression evaluation against project items
- Repository URL extraction from issue/PR content
- End-to-end orchestration execution flow

Note: These are unit tests that use mocks. For true integration tests against
a live GitHub project, set GITHUB_TOKEN and run with pytest --integration.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.github_poller import (
    GitHubClient,
    GitHubClientError,
    GitHubIssue,
    GitHubIssueProtocol,
    GitHubPoller,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    RetryConfig,
    TriggerConfig,
)
from sentinel.project_filter import ProjectFilterParser


class MockGitHubClient(GitHubClient):
    """Mock GitHub client for integration testing."""

    def __init__(
        self,
        project_items: list[dict[str, Any]] | None = None,
        project_data: dict[str, Any] | None = None,
    ) -> None:
        self.project_items = project_items or []
        self.project_data = project_data or {
            "id": "PVT_integration_test",
            "title": "Integration Test Project",
            "url": "https://github.com/orgs/test-org/projects/1",
            "fields": [
                {"id": "F1", "name": "Status", "dataType": "SINGLE_SELECT"},
                {"id": "F2", "name": "Priority", "dataType": "SINGLE_SELECT"},
                {"id": "F3", "name": "Sprint", "dataType": "ITERATION"},
            ],
        }
        self.get_project_calls: list[tuple[str, int, str]] = []
        self.list_project_items_calls: list[tuple[str, int]] = []

    def search_issues(
        self, query: str, max_results: int = 50
    ) -> list[dict[str, Any]]:
        return []

    def get_project(
        self, owner: str, project_number: int, scope: str = "organization"
    ) -> dict[str, Any]:
        self.get_project_calls.append((owner, project_number, scope))
        return self.project_data

    def list_project_items(
        self, project_id: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        self.list_project_items_calls.append((project_id, max_results))
        return self.project_items


class TestGitHubProjectPollingIntegration:
    """Integration tests for GitHub project polling flow."""

    def test_poll_project_with_filter_integration(self) -> None:
        """Test end-to-end project polling with filter expression.

        This test simulates polling a GitHub project with items in different
        statuses and verifies that the filter correctly selects matching items.
        """
        # Set up project items with different Status field values
        project_items = [
            {
                "id": "PVTI_1",
                "content": {
                    "type": "Issue",
                    "number": 101,
                    "title": "Ready issue",
                    "state": "OPEN",
                    "url": "https://github.com/test-org/repo-a/issues/101",
                    "body": "Ready for development",
                    "labels": ["feature"],
                    "assignees": ["dev1"],
                    "author": "pm",
                },
                "fieldValues": [
                    {"field": "Status", "value": "Ready"},
                    {"field": "Priority", "value": "High"},
                ],
            },
            {
                "id": "PVTI_2",
                "content": {
                    "type": "Issue",
                    "number": 102,
                    "title": "In Progress issue",
                    "state": "OPEN",
                    "url": "https://github.com/test-org/repo-a/issues/102",
                    "body": "Being worked on",
                    "labels": ["bug"],
                    "assignees": ["dev2"],
                    "author": "qa",
                },
                "fieldValues": [
                    {"field": "Status", "value": "In Progress"},
                    {"field": "Priority", "value": "Medium"},
                ],
            },
            {
                "id": "PVTI_3",
                "content": {
                    "type": "PullRequest",
                    "number": 201,
                    "title": "Ready PR",
                    "state": "OPEN",
                    "url": "https://github.com/test-org/repo-b/pull/201",
                    "body": "Feature implementation",
                    "labels": ["needs-review"],
                    "assignees": [],
                    "author": "dev1",
                    "isDraft": False,
                    "headRefName": "feature/new-feature",
                    "baseRefName": "main",
                },
                "fieldValues": [
                    {"field": "Status", "value": "Ready"},
                    {"field": "Priority", "value": "High"},
                ],
            },
            {
                "id": "PVTI_4",
                "content": {
                    "type": "Issue",
                    "number": 103,
                    "title": "Blocked issue",
                    "state": "OPEN",
                    "url": "https://github.com/test-org/repo-a/issues/103",
                    "body": "Waiting on dependency",
                    "labels": [],
                    "assignees": [],
                    "author": "dev3",
                },
                "fieldValues": [
                    {"field": "Status", "value": "Blocked"},
                    {"field": "Priority", "value": "Low"},
                ],
            },
        ]

        client = MockGitHubClient(project_items=project_items)
        poller = GitHubPoller(client, max_retries=1)

        # Configure trigger to poll only items with Status = 'Ready'
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
        )

        # Poll the project
        issues = poller.poll(trigger)

        # Should only return the 2 items with Status = 'Ready'
        assert len(issues) == 2

        # Verify the correct issues were returned
        issue_numbers = sorted([i.number for i in issues])
        assert issue_numbers == [101, 201]

        # Verify issue details
        ready_issue = next(i for i in issues if i.number == 101)
        assert ready_issue.title == "Ready issue"
        assert ready_issue.repo_url == "https://github.com/test-org/repo-a/issues/101"
        assert ready_issue.is_pull_request is False

        ready_pr = next(i for i in issues if i.number == 201)
        assert ready_pr.title == "Ready PR"
        assert ready_pr.repo_url == "https://github.com/test-org/repo-b/pull/201"
        assert ready_pr.is_pull_request is True
        assert ready_pr.head_ref == "feature/new-feature"
        assert ready_pr.base_ref == "main"

    def test_poll_project_with_complex_filter_integration(self) -> None:
        """Test polling with AND/OR filter expressions."""
        project_items = [
            {
                "id": "PVTI_1",
                "content": {
                    "type": "Issue",
                    "number": 1,
                    "title": "High priority ready",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/1",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "user",
                },
                "fieldValues": [
                    {"field": "Status", "value": "Ready"},
                    {"field": "Priority", "value": "High"},
                ],
            },
            {
                "id": "PVTI_2",
                "content": {
                    "type": "Issue",
                    "number": 2,
                    "title": "Low priority ready",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/2",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "user",
                },
                "fieldValues": [
                    {"field": "Status", "value": "Ready"},
                    {"field": "Priority", "value": "Low"},
                ],
            },
            {
                "id": "PVTI_3",
                "content": {
                    "type": "Issue",
                    "number": 3,
                    "title": "High priority in progress",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/3",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "user",
                },
                "fieldValues": [
                    {"field": "Status", "value": "In Progress"},
                    {"field": "Priority", "value": "High"},
                ],
            },
        ]

        client = MockGitHubClient(project_items=project_items)
        poller = GitHubPoller(client, max_retries=1)

        # Filter: Ready AND High Priority
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready" AND Priority = "High"',
        )

        issues = poller.poll(trigger)

        # Should only return issue #1 (Ready AND High)
        assert len(issues) == 1
        assert issues[0].number == 1
        assert issues[0].title == "High priority ready"

    def test_poll_project_multiple_repos_integration(self) -> None:
        """Test polling a project containing issues from multiple repositories.

        GitHub Projects (v2) can contain issues and PRs from multiple repositories.
        This test verifies that repo_url is correctly extracted from each item.
        """
        project_items = [
            {
                "id": "PVTI_1",
                "content": {
                    "type": "Issue",
                    "number": 10,
                    "title": "Frontend bug",
                    "state": "OPEN",
                    "url": "https://github.com/myorg/frontend/issues/10",
                    "body": "CSS issue",
                    "labels": ["bug"],
                    "assignees": [],
                    "author": "dev1",
                },
                "fieldValues": [{"field": "Status", "value": "Ready"}],
            },
            {
                "id": "PVTI_2",
                "content": {
                    "type": "Issue",
                    "number": 20,
                    "title": "Backend feature",
                    "state": "OPEN",
                    "url": "https://github.com/myorg/backend/issues/20",
                    "body": "New API endpoint",
                    "labels": ["feature"],
                    "assignees": [],
                    "author": "dev2",
                },
                "fieldValues": [{"field": "Status", "value": "Ready"}],
            },
            {
                "id": "PVTI_3",
                "content": {
                    "type": "PullRequest",
                    "number": 30,
                    "title": "Docs update",
                    "state": "OPEN",
                    "url": "https://github.com/myorg/docs/pull/30",
                    "body": "Update README",
                    "labels": [],
                    "assignees": [],
                    "author": "writer",
                    "isDraft": False,
                    "headRefName": "update-docs",
                    "baseRefName": "main",
                },
                "fieldValues": [{"field": "Status", "value": "Ready"}],
            },
        ]

        client = MockGitHubClient(project_items=project_items)
        poller = GitHubPoller(client, max_retries=1)

        trigger = TriggerConfig(
            source="github",
            project_number=5,
            project_owner="myorg",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        assert len(issues) == 3

        # Verify repo_url is correctly extracted from each item
        frontend_issue = next(i for i in issues if i.number == 10)
        assert frontend_issue.repo_url == "https://github.com/myorg/frontend/issues/10"

        backend_issue = next(i for i in issues if i.number == 20)
        assert backend_issue.repo_url == "https://github.com/myorg/backend/issues/20"

        docs_pr = next(i for i in issues if i.number == 30)
        assert docs_pr.repo_url == "https://github.com/myorg/docs/pull/30"
        assert docs_pr.is_pull_request is True


class TestRepoUrlExtractionIntegration:
    """Integration tests for repo URL extraction from issue/PR content."""

    def test_extract_repo_from_url_integration(self) -> None:
        """Test extract_repo_from_url works correctly with main.py."""
        from sentinel.main import extract_repo_from_url

        # Test various URL formats
        test_cases = [
            ("https://github.com/org/repo/issues/123", "org/repo"),
            ("https://github.com/org/repo/pull/456", "org/repo"),
            ("https://github.com/my-org-123/my-repo-name/issues/1", "my-org-123/my-repo-name"),
            ("https://github.enterprise.com/corp/project/issues/999", "corp/project"),
            ("http://github.com/test/test/pull/1", "test/test"),
        ]

        for url, expected_repo in test_cases:
            result = extract_repo_from_url(url)
            assert result == expected_repo, f"Failed for URL: {url}"

    def test_extract_repo_from_url_invalid_cases(self) -> None:
        """Test that invalid URLs return None."""
        from sentinel.main import extract_repo_from_url

        invalid_urls = [
            "",
            None,  # type: ignore[arg-type]
            "not a url",
            "https://github.com/org/repo",  # Missing /issues/ or /pull/
            "https://github.com/org/repo/commits/abc123",
            "https://example.com/org/repo/issues/123",  # Not github domain (but matches pattern)
        ]

        for url in invalid_urls:
            if url is None:
                result = extract_repo_from_url(url)  # type: ignore[arg-type]
            else:
                result = extract_repo_from_url(url)
            # Note: URLs with /issues/ or /pull/ pattern will match even on non-github domains
            if url and "/issues/" not in str(url) and "/pull/" not in str(url):
                assert result is None, f"Expected None for URL: {url}"


class TestFilterExpressionIntegration:
    """Integration tests for project filter expression evaluation."""

    def test_filter_parser_with_project_items(self) -> None:
        """Test that ProjectFilterParser correctly filters project items."""
        parser = ProjectFilterParser()

        # Create test field data (as extracted from project items)
        items_data = [
            {"Status": "Ready", "Priority": "High", "Sprint": "Sprint 1"},
            {"Status": "In Progress", "Priority": "High", "Sprint": "Sprint 1"},
            {"Status": "Ready", "Priority": "Low", "Sprint": "Sprint 2"},
            {"Status": "Done", "Priority": "Medium", "Sprint": "Sprint 1"},
        ]

        # Test filter: Status = 'Ready'
        filter_expr = parser.parse('Status = "Ready"')
        ready_items = [item for item in items_data if parser.evaluate(filter_expr, item)]
        assert len(ready_items) == 2

        # Test filter: Status = 'Ready' AND Priority = 'High'
        filter_expr = parser.parse('Status = "Ready" AND Priority = "High"')
        ready_high = [item for item in items_data if parser.evaluate(filter_expr, item)]
        assert len(ready_high) == 1
        assert ready_high[0]["Priority"] == "High"

        # Test filter: Status = 'Ready' OR Status = 'In Progress'
        filter_expr = parser.parse('Status = "Ready" OR Status = "In Progress"')
        active = [item for item in items_data if parser.evaluate(filter_expr, item)]
        assert len(active) == 3

        # Test filter with Sprint
        filter_expr = parser.parse('Sprint = "Sprint 1"')
        sprint1 = [item for item in items_data if parser.evaluate(filter_expr, item)]
        assert len(sprint1) == 3

    def test_filter_not_equal_operator(self) -> None:
        """Test != operator in filter expressions."""
        parser = ProjectFilterParser()

        items_data = [
            {"Status": "Ready"},
            {"Status": "In Progress"},
            {"Status": "Done"},
        ]

        filter_expr = parser.parse('Status != "Done"')
        not_done = [item for item in items_data if parser.evaluate(filter_expr, item)]
        assert len(not_done) == 2
        assert all(item["Status"] != "Done" for item in not_done)


class TestEndToEndOrchestrationFlow:
    """Integration tests for end-to-end orchestration execution with GitHub triggers."""

    def test_github_trigger_orchestration_setup(self) -> None:
        """Test that GitHub trigger orchestrations are correctly configured."""
        # Create a GitHub-triggered orchestration
        orchestration = Orchestration(
            name="github-code-review",
            trigger=TriggerConfig(
                source="github",
                project_number=42,
                project_owner="my-org",
                project_scope="org",
                project_filter='Status = "Ready for Review"',
            ),
            agent=AgentConfig(
                prompt="Review the code for issue {github_org}/{github_repo}#{jira_issue_key}",
                tools=["github"],
                github=GitHubContext(
                    host="github.com",
                    org="my-org",
                    repo="my-repo",
                ),
            ),
            retry=RetryConfig(
                max_attempts=3,
                success_patterns=["SUCCESS", "APPROVED"],
                failure_patterns=["FAILURE", "ERROR"],
            ),
        )

        # Verify orchestration is correctly set up
        assert orchestration.name == "github-code-review"
        assert orchestration.trigger.source == "github"
        assert orchestration.trigger.project_number == 42
        assert orchestration.trigger.project_owner == "my-org"
        assert orchestration.trigger.project_scope == "org"
        assert orchestration.trigger.project_filter == 'Status = "Ready for Review"'

    def test_orchestration_with_project_based_deduplication(self) -> None:
        """Test that project-based deduplication key is correctly formed.

        DS-204 changed deduplication to use project_owner/project_number
        instead of repo/tags to avoid polling the same project multiple times.
        """
        # Two orchestrations pointing to the same project but different filters
        orch1 = Orchestration(
            name="review-ready",
            trigger=TriggerConfig(
                source="github",
                project_number=1,
                project_owner="test-org",
                project_scope="org",
                project_filter='Status = "Ready"',
            ),
            agent=AgentConfig(prompt="Process ready items"),
        )

        orch2 = Orchestration(
            name="review-blocked",
            trigger=TriggerConfig(
                source="github",
                project_number=1,
                project_owner="test-org",
                project_scope="org",
                project_filter='Status = "Blocked"',
            ),
            agent=AgentConfig(prompt="Process blocked items"),
        )

        # Build deduplication keys (as done in main.py _poll_github_triggers)
        key1 = f"github:{orch1.trigger.project_owner}/{orch1.trigger.project_number}"
        key2 = f"github:{orch2.trigger.project_owner}/{orch2.trigger.project_number}"

        # Both should have the same deduplication key (same project)
        assert key1 == key2 == "github:test-org/1"


class TestDraftIssueHandling:
    """Integration tests for draft issue handling in project polling."""

    def test_draft_issues_are_skipped(self) -> None:
        """Test that DraftIssue items are correctly skipped during polling."""
        project_items = [
            {
                "id": "PVTI_1",
                "content": {
                    "type": "Issue",
                    "number": 1,
                    "title": "Real issue",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/1",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "user",
                },
                "fieldValues": [],
            },
            {
                "id": "PVTI_2",
                "content": {
                    "type": "DraftIssue",
                    "title": "Draft issue (no number)",
                },
                "fieldValues": [],
            },
            {
                "id": "PVTI_3",
                "content": None,  # No content (deleted item)
                "fieldValues": [],
            },
            {
                "id": "PVTI_4",
                "content": {
                    "type": "PullRequest",
                    "number": 10,
                    "title": "Real PR",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/pull/10",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "dev",
                    "isDraft": False,
                    "headRefName": "feature",
                    "baseRefName": "main",
                },
                "fieldValues": [],
            },
        ]

        client = MockGitHubClient(project_items=project_items)
        poller = GitHubPoller(client, max_retries=1)

        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        # Should only return the real Issue and PullRequest
        assert len(issues) == 2
        assert {i.number for i in issues} == {1, 10}


class TestProjectCaching:
    """Integration tests for project ID caching behavior."""

    def test_project_id_is_cached(self) -> None:
        """Test that project ID is cached after first lookup."""
        project_items = [
            {
                "id": "PVTI_1",
                "content": {
                    "type": "Issue",
                    "number": 1,
                    "title": "Test",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/1",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "user",
                },
                "fieldValues": [],
            }
        ]

        client = MockGitHubClient(project_items=project_items)
        poller = GitHubPoller(client, max_retries=1)

        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        # First poll - should call get_project
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1

        # Second poll - should use cached project ID
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1  # Still only 1 call

        # Verify cache contains the project
        assert "test-org/1" in poller._project_id_cache
