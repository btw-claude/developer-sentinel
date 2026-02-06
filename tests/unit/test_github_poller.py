"""Tests for GitHub poller module."""

from typing import Any
from unittest.mock import patch

import pytest

from sentinel.github_poller import GitHubClient, GitHubClientError, GitHubIssue, GitHubPoller
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

    def list_project_items(self, project_id: str, max_results: int = 100) -> list[dict[str, Any]]:
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
    """Tests for repo_url field in GitHubIssue."""

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


class TestGitHubPollerClearProjectIdCache:
    """Tests for GitHubPoller.clear_project_id_cache."""

    def test_cache_is_cleared_after_calling_method(self) -> None:
        """Test that cache is cleared after calling clear_project_id_cache."""
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        # First poll to populate the cache
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1

        # Second poll should use cached project ID
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1  # Still 1, using cache

        # Clear the cache
        poller.clear_project_id_cache()

        # Third poll should fetch project ID again since cache was cleared
        poller.poll(trigger)
        assert len(client.get_project_calls) == 2  # Now 2, cache was cleared

    def test_method_can_be_called_multiple_times_without_error(self) -> None:
        """Test that clear_project_id_cache can be called multiple times without error."""
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)

        # Should not raise any errors when called multiple times
        poller.clear_project_id_cache()
        poller.clear_project_id_cache()
        poller.clear_project_id_cache()

        # Also test calling on empty cache after populating and clearing
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )
        poller.poll(trigger)  # Populate cache
        poller.clear_project_id_cache()
        poller.clear_project_id_cache()  # Clear already empty cache

    def test_cache_is_properly_repopulated_after_clearing(self) -> None:
        """Test that cache is properly repopulated after clearing."""
        client = MockGitHubClient(project_items=[])
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="test-org",
            project_scope="org",
        )

        # First poll to populate the cache
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1

        # Verify cache is being used
        poller.poll(trigger)
        assert len(client.get_project_calls) == 1  # Still 1

        # Clear cache
        poller.clear_project_id_cache()

        # Poll again to repopulate cache
        poller.poll(trigger)
        assert len(client.get_project_calls) == 2  # Fetched again

        # Verify repopulated cache is being used
        poller.poll(trigger)
        assert len(client.get_project_calls) == 2  # Still 2, using repopulated cache

        # Poll once more to confirm cache is stable
        poller.poll(trigger)
        assert len(client.get_project_calls) == 2  # Still 2


class TestGitHubPollerLabelFiltering:
    """Tests for label filtering in GitHubPoller.poll().

    These tests verify the labels field filtering functionality:
    - Test poll() with labels filter
    - Test AND logic (must have ALL labels)
    - Test case-insensitive matching
    - Test combination of labels + project_filter
    - Test empty labels list (no filtering)
    """

    def test_poll_with_label_filter(self) -> None:
        """Test polling with label filtering returns only matching issues."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Bug with urgent label",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": ["bug", "urgent"],
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
                        "title": "Feature request",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": ["feature"],
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
            labels=["bug", "urgent"],
        )

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].number == 1
        assert issues[0].title == "Bug with urgent label"

    def test_poll_labels_and_logic_must_have_all_labels(self) -> None:
        """Test that issues must have ALL specified labels (AND logic)."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Has both labels",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": ["bug", "urgent", "needs-review"],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "Only has bug label",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": ["bug"],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item3",
                    "content": {
                        "type": "Issue",
                        "number": 3,
                        "title": "Only has urgent label",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/3",
                        "body": "",
                        "labels": ["urgent"],
                        "assignees": [],
                        "author": "author",
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
            labels=["bug", "urgent"],
        )

        issues = poller.poll(trigger)

        # Only the first issue has BOTH "bug" AND "urgent"
        assert len(issues) == 1
        assert issues[0].number == 1

    def test_poll_labels_case_insensitive_matching(self) -> None:
        """Test that label matching is case-insensitive."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Mixed case labels",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": ["BUG", "Urgent"],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "Lowercase labels",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": ["bug", "urgent"],
                        "assignees": [],
                        "author": "author",
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
            labels=["bug", "URGENT"],  # Mixed case in filter
        )

        issues = poller.poll(trigger)

        # Both issues should match due to case-insensitive comparison
        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2

    def test_poll_labels_combined_with_project_filter(self) -> None:
        """Test polling with both labels and project_filter."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Ready bug",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": ["bug"],
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
                        "title": "In Progress bug",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": ["bug"],
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
                        "title": "Ready feature",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/3",
                        "body": "",
                        "labels": ["feature"],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [{"field": "Status", "value": "Ready"}],
                },
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
            project_filter='Status = "Ready"',
            labels=["bug"],
        )

        issues = poller.poll(trigger)

        # Only item1 matches both the project_filter AND labels
        assert len(issues) == 1
        assert issues[0].number == 1
        assert issues[0].title == "Ready bug"

    def test_poll_empty_labels_list_no_filtering(self) -> None:
        """Test that empty labels list returns all issues (no filtering)."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "Issue with labels",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": ["bug", "urgent"],
                        "assignees": [],
                        "author": "author",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "Issue",
                        "number": 2,
                        "title": "Issue without labels",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/2",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
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
            labels=[],  # Empty labels - no filtering
        )

        issues = poller.poll(trigger)

        # All issues should be returned
        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2

    def test_poll_labels_none_vs_empty_list_handled_identically(self) -> None:
        """Test that labels=None and labels=[] both result in no filtering.

        This ensures consistent behavior whether the labels parameter is
        explicitly set to an empty list or left as None (the default).
        """
        project_items = [
            {
                "id": "item1",
                "content": {
                    "type": "Issue",
                    "number": 1,
                    "title": "Issue with labels",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/1",
                    "body": "",
                    "labels": ["bug", "urgent"],
                    "assignees": [],
                    "author": "author",
                },
                "fieldValues": [],
            },
            {
                "id": "item2",
                "content": {
                    "type": "Issue",
                    "number": 2,
                    "title": "Issue without labels",
                    "state": "OPEN",
                    "url": "https://github.com/org/repo/issues/2",
                    "body": "",
                    "labels": [],
                    "assignees": [],
                    "author": "author",
                },
                "fieldValues": [],
            },
        ]

        # Test with labels=None (default)
        client_none = MockGitHubClient(project_items=project_items)
        poller_none = GitHubPoller(client_none)
        trigger_none = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
            # labels not specified, defaults to None
        )
        issues_none = poller_none.poll(trigger_none)

        # Test with labels=[]
        client_empty = MockGitHubClient(project_items=project_items)
        poller_empty = GitHubPoller(client_empty)
        trigger_empty = TriggerConfig(
            source="github",
            project_number=1,
            project_owner="org",
            project_scope="org",
            labels=[],  # Explicitly empty list
        )
        issues_empty = poller_empty.poll(trigger_empty)

        # Both should return all issues (no filtering applied)
        assert len(issues_none) == 2, "labels=None should return all issues"
        assert len(issues_empty) == 2, "labels=[] should return all issues"
        assert issues_none[0].number == issues_empty[0].number
        assert issues_none[1].number == issues_empty[1].number

    def test_poll_labels_filter_issue_with_no_labels_rejected(self) -> None:
        """Test that issues without labels are rejected when filter requires labels."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "Issue",
                        "number": 1,
                        "title": "No labels issue",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/issues/1",
                        "body": "",
                        "labels": [],
                        "assignees": [],
                        "author": "author",
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
            labels=["bug"],
        )

        issues = poller.poll(trigger)

        # Issue without labels should not match when labels filter is set
        assert len(issues) == 0

    def test_poll_labels_filter_with_pull_request(self) -> None:
        """Test label filtering works with pull requests too."""
        client = MockGitHubClient(
            project_items=[
                {
                    "id": "item1",
                    "content": {
                        "type": "PullRequest",
                        "number": 10,
                        "title": "PR with review label",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/pull/10",
                        "body": "",
                        "labels": ["needs-review", "enhancement"],
                        "assignees": [],
                        "author": "prauthor",
                        "isDraft": False,
                        "headRefName": "feature",
                        "baseRefName": "main",
                    },
                    "fieldValues": [],
                },
                {
                    "id": "item2",
                    "content": {
                        "type": "PullRequest",
                        "number": 11,
                        "title": "PR without review label",
                        "state": "OPEN",
                        "url": "https://github.com/org/repo/pull/11",
                        "body": "",
                        "labels": ["enhancement"],
                        "assignees": [],
                        "author": "prauthor",
                        "isDraft": False,
                        "headRefName": "feature2",
                        "baseRefName": "main",
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
            labels=["needs-review"],
        )

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].number == 10
        assert issues[0].is_pull_request is True


class TestGitHubIssueParentIssueNumber:
    """Tests for GitHubIssue parent_issue_number field.

    These tests verify that the parent_issue_number field is correctly
    populated from the GitHub API response, supporting the
    {github_parent_issue_number} template variable.
    """

    def test_parent_issue_number_from_api_response(self) -> None:
        """Test parent_issue_number is populated from API response."""
        data = {
            "number": 42,
            "title": "Sub-issue",
            "parent": {"number": 10},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number == 10

    def test_parent_issue_number_none_when_no_parent(self) -> None:
        """Test parent_issue_number is None when no parent field."""
        data = {
            "number": 42,
            "title": "Top-level issue",
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number is None

    def test_parent_issue_number_none_when_parent_empty(self) -> None:
        """Test parent_issue_number is None when parent field is empty dict."""
        data = {
            "number": 42,
            "title": "Issue with empty parent",
            "parent": {},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number is None

    def test_parent_issue_number_none_when_parent_not_dict(self) -> None:
        """Test parent_issue_number is None when parent is not a dict."""
        data = {
            "number": 42,
            "title": "Issue with invalid parent",
            "parent": None,
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number is None

    def test_parent_issue_number_none_when_parent_missing_number(self) -> None:
        """Test parent_issue_number is None when parent field present but number missing.

        This is a defensive test case for malformed GitHub API responses where
        the parent field is present (as a dict) but doesn't contain the expected
        'number' key. This could happen due to API inconsistencies or partial data.
        """
        data = {
            "number": 42,
            "title": "Issue with malformed parent",
            "parent": {"id": "some_id", "url": "https://example.com"},  # number key missing
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number is None

    def test_parent_issue_number_none_when_parent_number_is_none(self) -> None:
        """Test parent_issue_number is None when parent.number is explicitly None.

        This is a defensive test case where the parent dict has a 'number' key
        but its value is None instead of an integer.
        """
        data = {
            "number": 42,
            "title": "Issue with null parent number",
            "parent": {"number": None},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.parent_issue_number is None

    def test_parent_issue_number_invalid_type_returns_none(self) -> None:
        """Test that invalid type for parent.number returns None.

        This test verifies that when the parent dict has a 'number' key but its
        value is an unexpected type (not int), the method returns None and logs
        a warning. This ensures parent_issue_number is always either None or an int.
        """
        data = {
            "number": 42,
            "title": "Issue with string parent number",
            "parent": {"number": "not-a-number"},
        }
        issue = GitHubIssue.from_api_response(data)
        # Invalid type should be treated as None
        assert issue.parent_issue_number is None

    def test_parent_issue_number_from_project_item(self) -> None:
        """Test parent_issue_number is populated from project item."""
        item = {
            "id": "PVTI_123",
            "content": {
                "type": "Issue",
                "number": 42,
                "title": "Sub-issue from project",
                "state": "OPEN",
                "url": "https://github.com/org/repo/issues/42",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "testuser",
                "parent": {"number": 5},
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.parent_issue_number == 5

    def test_parent_issue_number_none_from_project_item_no_parent(self) -> None:
        """Test parent_issue_number is None when project item has no parent."""
        item = {
            "id": "PVTI_456",
            "content": {
                "type": "Issue",
                "number": 99,
                "title": "Top-level issue from project",
                "state": "OPEN",
                "url": "https://github.com/org/repo/issues/99",
                "body": "",
                "labels": [],
                "assignees": [],
                "author": "testuser",
            },
            "fieldValues": [],
        }
        issue = GitHubIssue.from_project_item(item)
        assert issue is not None
        assert issue.parent_issue_number is None

    def test_parent_issue_number_with_pull_request(self) -> None:
        """Test parent_issue_number works with pull requests too."""
        data = {
            "number": 123,
            "title": "Fix for sub-issue",
            "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/123"},
            "parent": {"number": 50},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is True
        assert issue.parent_issue_number == 50

    def test_github_issue_dataclass_default(self) -> None:
        """Test GitHubIssue dataclass has correct default for parent_issue_number."""
        issue = GitHubIssue(number=1, title="Test")
        assert issue.parent_issue_number is None

    def test_github_issue_with_explicit_parent_issue_number(self) -> None:
        """Test GitHubIssue can be created with explicit parent_issue_number."""
        issue = GitHubIssue(number=42, title="Sub-issue", parent_issue_number=10)
        assert issue.parent_issue_number == 10
