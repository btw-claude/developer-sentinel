"""Tests for GitHub poller module."""

import warnings
from typing import Any
from unittest.mock import patch

import pytest

from sentinel.github_poller import (
    GitHubClient,
    GitHubClientError,
    GitHubIssue,
    GitHubPoller,
)
from sentinel.orchestration import TriggerConfig


class MockGitHubClient(GitHubClient):
    """Mock GitHub client for testing."""

    def __init__(
        self,
        issues: list[dict[str, Any]] | None = None,
        project_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self.issues = issues or []
        self.project_items = project_items or []
        self.search_calls: list[tuple[str, int]] = []
        self.get_project_calls: list[tuple[str, int, str]] = []
        self.list_project_items_calls: list[tuple[str, int]] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0
        # Mock project data
        self.mock_project: dict[str, Any] = {
            "id": "PVT_test123",
            "title": "Test Project",
            "url": "https://github.com/orgs/test-org/projects/1",
            "fields": [],
        }

    def search_issues(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.search_calls.append((query, max_results))
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise GitHubClientError("API error")
        return self.issues

    def get_project(
        self, owner: str, project_number: int, scope: str = "organization"
    ) -> dict[str, Any]:
        self.get_project_calls.append((owner, project_number, scope))
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise GitHubClientError("API error")
        return self.mock_project

    def list_project_items(
        self, project_id: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        self.list_project_items_calls.append((project_id, max_results))
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise GitHubClientError("API error")
        return self.project_items


class TestGitHubIssueKey:
    """Tests for GitHubIssue.key property."""

    def test_key_format(self) -> None:
        issue = GitHubIssue(number=42, title="Test")
        assert issue.key == "#42"

    def test_key_with_pr(self) -> None:
        issue = GitHubIssue(number=123, title="Test PR", is_pull_request=True)
        assert issue.key == "#123"


class TestGitHubIssueFromApiResponse:
    """Tests for GitHubIssue.from_api_response."""

    def test_basic_fields(self) -> None:
        data = {
            "number": 123,
            "title": "Test issue",
            "body": "Issue description",
            "state": "open",
            "labels": [{"name": "bug"}, {"name": "urgent"}],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.number == 123
        assert issue.title == "Test issue"
        assert issue.body == "Issue description"
        assert issue.state == "open"
        assert issue.labels == ["bug", "urgent"]
        assert issue.is_pull_request is False

    def test_author_extraction(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "user": {"login": "testuser"},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.author == "testuser"

    def test_no_author(self) -> None:
        data = {"number": 1, "title": "Test"}
        issue = GitHubIssue.from_api_response(data)
        assert issue.author == ""

    def test_assignees_extraction(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "assignees": [
                {"login": "user1"},
                {"login": "user2"},
            ],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.assignees == ["user1", "user2"]

    def test_empty_assignees(self) -> None:
        data = {"number": 1, "title": "Test", "assignees": []}
        issue = GitHubIssue.from_api_response(data)
        assert issue.assignees == []

    def test_labels_as_strings(self) -> None:
        """Test handling labels when they're strings instead of objects."""
        data = {
            "number": 1,
            "title": "Test",
            "labels": ["bug", "feature"],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.labels == ["bug", "feature"]

    def test_labels_as_objects(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "labels": [{"name": "bug"}, {"name": "feature"}],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.labels == ["bug", "feature"]

    def test_pull_request_detection_from_key(self) -> None:
        """Test PR detection via pull_request key presence."""
        data = {
            "number": 1,
            "title": "Test PR",
            "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/1"},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is True

    def test_pull_request_with_branches(self) -> None:
        """Test PR with head and base branch info."""
        data = {
            "number": 1,
            "title": "Test PR",
            "pull_request": {},
            "head": {"ref": "feature-branch"},
            "base": {"ref": "main"},
            "draft": True,
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is True
        assert issue.head_ref == "feature-branch"
        assert issue.base_ref == "main"
        assert issue.draft is True

    def test_regular_issue(self) -> None:
        """Test that regular issues don't have PR fields."""
        data = {
            "number": 1,
            "title": "Test Issue",
            "state": "open",
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is False
        assert issue.head_ref == ""
        assert issue.base_ref == ""
        assert issue.draft is False

    def test_null_body(self) -> None:
        """Test handling of null body."""
        data = {
            "number": 1,
            "title": "Test",
            "body": None,
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.body == ""

    def test_empty_data(self) -> None:
        data: dict[str, Any] = {}
        issue = GitHubIssue.from_api_response(data)
        assert issue.number == 0
        assert issue.title == ""


class TestGitHubPollerBuildQuery:
    """Tests for GitHubPoller.build_query (deprecated)."""

    def test_repo_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert "repo:org/repo" in query

    def test_tags_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", tags=["needs-review", "urgent"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert 'label:"needs-review"' in query
        assert 'label:"urgent"' in query

    def test_custom_query_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="is:pr is:draft")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert "is:pr is:draft" in query

    def test_default_state_open(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert "state:open" in query

    def test_no_state_when_specified_in_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="state:closed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        # Should not add default state:open
        assert query.count("state:") == 1

    def test_no_state_when_is_open_in_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="is:open")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert "state:open" not in query

    def test_combined_filters(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            repo="org/repo",
            tags=["review"],
            query_filter="is:pr",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        assert "repo:org/repo" in query
        assert 'label:"review"' in query
        assert "is:pr" in query

    def test_empty_trigger(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            query = poller.build_query(trigger)
        # Should only have default state:open
        assert query == "state:open"


class TestGitHubPollerPoll:
    """Tests for GitHubPoller.poll with project-based polling."""

    def test_successful_poll(self) -> None:
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Issue 1",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author1",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "Issue 2",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author2",
                    },
                    "fieldValues": [],
                },
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2

    def test_poll_calls_client_with_project_id(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        poller.poll(trigger, max_results=25)

        assert len(client.get_project_calls) == 1
        assert client.get_project_calls[0] == ("org", 1, "organization")
        assert len(client.list_project_items_calls) == 1
        project_id, max_results = client.list_project_items_calls[0]
        assert project_id == "PVT_test123"
        assert max_results == 25

    def test_poll_empty_project_returns_empty(self) -> None:
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        assert issues == []

    def test_poll_retries_on_failure(self) -> None:
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Test",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                }
            ]
        )
        client.should_fail = True
        client.max_fails = 2  # Fail twice, then succeed

        poller = GitHubPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        with patch("sentinel.github_poller.time.sleep"):
            issues = poller.poll(trigger)

        assert len(issues) == 1
        assert len(client.get_project_calls) == 3  # 2 failures + 1 success

    def test_poll_raises_after_max_retries(self) -> None:
        client = MockGitHubClient()
        client.should_fail = True
        client.max_fails = 10  # Always fail

        poller = GitHubPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
        )

        with (
            patch("sentinel.github_poller.time.sleep"),
            pytest.raises(GitHubClientError, match="Failed to poll GitHub after 3 attempts"),
        ):
            poller.poll(trigger)

        assert len(client.get_project_calls) == 3

    def test_poll_uses_exponential_backoff(self) -> None:
        client = MockGitHubClient()
        client.should_fail = True
        client.max_fails = 2

        poller = GitHubPoller(client, max_retries=3, retry_delay=1.0)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        with patch("sentinel.github_poller.time.sleep") as mock_sleep:
            # Need to set up client to eventually succeed
            client.project_items = [
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Test",
                        "state": "OPEN",
                        "url": "https://github.com/test-org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "testuser",
                    },
                    "fieldValues": [],
                }
            ]
            poller.poll(trigger)

        # Should have called sleep with exponential delays: 1.0, 2.0
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1.0 * (2^0)
        mock_sleep.assert_any_call(2.0)  # Second retry: 1.0 * (2^1)


class TestGitHubIssueFromProjectItem:
    """Tests for GitHubIssue.from_project_item."""

    def test_basic_issue(self) -> None:
        """Test conversion of a basic issue project item."""
        item = {
            "id": "PVTI_123",
            "content": {
                "type": "Issue",
                "number": 42,
                "title": "Test issue",
                "state": "OPEN",
                "url": "https://github.com/org/repo/issues/42",
                "body": "Issue description",
                "labels": ["bug", "urgent"],
                "assignees": ["user1", "user2"],
                "author": "testuser",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.number == 42
        assert issue.title == "Test issue"
        assert issue.body == "Issue description"
        assert issue.state == "open"
        assert issue.labels == ["bug", "urgent"]
        assert issue.assignees == ["user1", "user2"]
        assert issue.author == "testuser"
        assert issue.is_pull_request is False

    def test_pull_request(self) -> None:
        """Test conversion of a pull request project item."""
        item = {
            "id": "PVTI_456",
            "content": {
                "type": "PullRequest",
                "number": 123,
                "title": "Test PR",
                "state": "OPEN",
                "url": "https://github.com/org/repo/pull/123",
                "body": "PR description",
                "labels": ["enhancement"],
                "assignees": ["reviewer"],
                "author": "prauthor",
                "isDraft": False,
                "headRefName": "feature-branch",
                "baseRefName": "main",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.number == 123
        assert issue.title == "Test PR"
        assert issue.is_pull_request is True
        assert issue.head_ref == "feature-branch"
        assert issue.base_ref == "main"
        assert issue.draft is False

    def test_draft_pull_request(self) -> None:
        """Test conversion of a draft pull request."""
        item = {
            "id": "PVTI_789",
            "content": {
                "type": "PullRequest",
                "number": 456,
                "title": "Draft PR",
                "state": "OPEN",
                "url": "https://github.com/org/repo/pull/456",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "author",
                "isDraft": True,
                "headRefName": "wip-branch",
                "baseRefName": "main",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.draft is True

    def test_draft_issue_returns_none(self) -> None:
        """Test that DraftIssue items return None."""
        item = {
            "id": "PVTI_draft",
            "content": {
                "type": "DraftIssue",
                "title": "Draft idea",
                "body": "Some notes",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is None

    def test_no_content_returns_none(self) -> None:
        """Test that items with no content return None."""
        item = {
            "id": "PVTI_empty",
            "content": None,
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is None

    def test_merged_state_normalized(self) -> None:
        """Test that MERGED state is normalized to closed."""
        item = {
            "id": "PVTI_merged",
            "content": {
                "type": "PullRequest",
                "number": 789,
                "title": "Merged PR",
                "state": "MERGED",
                "url": "https://github.com/org/repo/pull/789",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "author",
                "isDraft": False,
                "headRefName": "merged-branch",
                "baseRefName": "main",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.state == "closed"

    def test_closed_state(self) -> None:
        """Test that CLOSED state is normalized to lowercase."""
        item = {
            "id": "PVTI_closed",
            "content": {
                "type": "Issue",
                "number": 100,
                "title": "Closed issue",
                "state": "CLOSED",
                "url": "https://github.com/org/repo/issues/100",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "author",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.state == "closed"


class TestGitHubPollerProjectBased:
    """Tests for project-based polling in GitHubPoller."""

    def test_poll_with_project_config(self) -> None:
        """Test polling with project configuration."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Issue 1",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author1",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "Issue 2",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author2",
                    },
                    "fieldValues": [],
                },
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2
        # Verify client calls
        assert len(client.get_project_calls) == 1
        assert client.get_project_calls[0] == ("test-org", 1, "organization")
        assert len(client.list_project_items_calls) == 1

    def test_poll_caches_project_id(self) -> None:
        """Test that project ID is cached between polls."""
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=5,
            project_owner="cached-org",
            project_scope="org",
        )

        # First poll
        poller.poll(trigger)
        # Second poll
        poller.poll(trigger)

        # get_project should only be called once due to caching
        assert len(client.get_project_calls) == 1
        # list_project_items should be called twice
        assert len(client.list_project_items_calls) == 2

    def test_poll_with_user_scope(self) -> None:
        """Test polling with user-scoped project."""
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=2,
            project_owner="testuser",
            project_scope="user",
        )

        poller.poll(trigger)

        assert client.get_project_calls[0] == ("testuser", 2, "user")

    def test_poll_with_filter(self) -> None:
        """Test polling with project_filter expression."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Ready issue",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [{"field": "Status", "value": "Ready"}],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "In Progress issue",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [{"field": "Status", "value": "In Progress"}],
                },
                {
                    "id": "item3",
                    "content": {
                        "type": "Issue",
                        "number": 3,
                        "title": "Done issue",
                        "state": "CLOSED",
                        "url": "https://github.com/org/repo/issues/3",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [{"field": "Status", "value": "Done"}],
                },
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready"',
        )

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].number == 1
        assert issues[0].title == "Ready issue"

    def test_poll_skips_draft_issues(self) -> None:
        """Test that draft issues are skipped during conversion."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Real issue",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "DraftIssue",
                        "title": "Draft idea",
                        "body": "Notes",
                    },
                    "fieldValues": [],
                },
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].number == 1

    def test_poll_missing_project_number_raises(self) -> None:
        """Test that polling without project_number raises an error."""
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_owner="test-org",
            # project_number not set
        )

        with pytest.raises(GitHubClientError, match="requires project_number"):
            poller.poll(trigger)

    def test_poll_missing_project_owner_raises(self) -> None:
        """Test that polling without project_owner raises an error."""
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            # project_owner not set
        )

        with pytest.raises(GitHubClientError, match="requires project_owner"):
            poller.poll(trigger)

    def test_poll_invalid_filter_raises(self) -> None:
        """Test that invalid project_filter raises an error."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Test",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                }
            ]
        )
        # Use max_retries=1 to avoid retry loop for validation errors
        poller = GitHubPoller(client, max_retries=1, retry_delay=0.01)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
            project_filter='Status = "Ready',  # Missing closing quote
        )

        # The error gets wrapped, so we match the final error message
        with pytest.raises(GitHubClientError, match="Failed to poll GitHub"):
            with patch("sentinel.github_poller.time.sleep"):
                poller.poll(trigger)

    def test_poll_retries_on_failure(self) -> None:
        """Test that polling retries on API failures."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Test",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                }
            ]
        )
        client.should_fail = True
        client.max_fails = 2  # Fail twice, then succeed

        poller = GitHubPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        with patch("sentinel.github_poller.time.sleep"):
            issues = poller.poll(trigger)

        assert len(issues) == 1
        # get_project is called 3 times due to retries (failures reset fail_count)
        assert len(client.get_project_calls) == 3


class TestGitHubIssueRepoUrl:
    """Tests for repo_url field in GitHubIssue (DS-204)."""

    def test_from_project_item_extracts_url(self) -> None:
        """Test that from_project_item extracts repo_url from content."""
        item = {
            "id": "PVTI_123",
            "content": {
                "type": "Issue",
                "number": 42,
                "title": "Test issue",
                "state": "OPEN",
                "url": "https://github.com/org/repo/issues/42",
                "body": "Issue description",
                "labels": [],
                "assignees": [],
                "author": "testuser",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.repo_url == "https://github.com/org/repo/issues/42"

    def test_from_project_item_handles_missing_url(self) -> None:
        """Test that from_project_item handles missing URL gracefully."""
        item = {
            "id": "PVTI_456",
            "content": {
                "type": "Issue",
                "number": 99,
                "title": "No URL issue",
                "state": "OPEN",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "author",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.repo_url == ""

    def test_from_project_item_pr_url(self) -> None:
        """Test that from_project_item extracts URL for pull requests."""
        item = {
            "id": "PVTI_789",
            "content": {
                "type": "PullRequest",
                "number": 123,
                "title": "Test PR",
                "state": "OPEN",
                "url": "https://github.com/my-org/my-repo/pull/123",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "prauthor",
                "isDraft": False,
                "headRefName": "feature",
                "baseRefName": "main",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.repo_url == "https://github.com/my-org/my-repo/pull/123"


class TestGitHubPollerBuildQueryDeprecation:
    """Tests for build_query deprecation."""

    def test_build_query_emits_deprecation_warning(self) -> None:
        """Test that build_query emits a deprecation warning."""
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            poller.build_query(trigger)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            assert "project-based" in str(w[0].message).lower()
