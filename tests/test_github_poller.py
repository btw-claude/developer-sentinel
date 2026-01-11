"""Tests for GitHub poller module."""

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

    def __init__(self, issues: list[dict[str, Any]] | None = None) -> None:
        self.issues = issues or []
        self.search_calls: list[tuple[str, int]] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0

    def search_issues(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.search_calls.append((query, max_results))
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise GitHubClientError("API error")
        return self.issues


class TestGitHubIssueFromApiResponse:
    """Tests for GitHubIssue.from_api_response."""

    def test_basic_fields(self) -> None:
        data = {
            "number": 123,
            "title": "Test issue",
            "body": "Issue body",
            "state": "open",
            "labels": [{"name": "bug"}, {"name": "urgent"}],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.number == 123
        assert issue.title == "Test issue"
        assert issue.body == "Issue body"
        assert issue.state == "open"
        assert issue.labels == ["bug", "urgent"]
        assert issue.is_pull_request is False

    def test_issue_key_property(self) -> None:
        data = {"number": 456, "title": "Test"}
        issue = GitHubIssue.from_api_response(data)
        assert issue.key == "#456"

    def test_author_extraction(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "user": {"login": "johndoe"},
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.author == "johndoe"

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

    def test_labels_as_strings(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "labels": ["bug", "enhancement"],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.labels == ["bug", "enhancement"]

    def test_pull_request_detection(self) -> None:
        data = {
            "number": 42,
            "title": "Fix bug",
            "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/42"},
            "head": {"ref": "feature-branch"},
            "base": {"ref": "main"},
            "draft": True,
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is True
        assert issue.head_ref == "feature-branch"
        assert issue.base_ref == "main"
        assert issue.draft is True

    def test_regular_issue_no_pr_fields(self) -> None:
        data = {
            "number": 10,
            "title": "Bug report",
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.is_pull_request is False
        assert issue.head_ref == ""
        assert issue.base_ref == ""
        assert issue.draft is False

    def test_null_body(self) -> None:
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
        assert issue.body == ""

    def test_assignees_with_empty_login(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "assignees": [
                {"login": ""},
                {"login": "valid_user"},
            ],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.assignees == ["valid_user"]

    def test_labels_with_empty_name(self) -> None:
        data = {
            "number": 1,
            "title": "Test",
            "labels": [
                {"name": ""},
                {"name": "valid_label"},
            ],
        }
        issue = GitHubIssue.from_api_response(data)
        assert issue.labels == ["valid_label"]


class TestGitHubPollerBuildQuery:
    """Tests for GitHubPoller.build_query."""

    def test_repo_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")
        query = poller.build_query(trigger)
        assert "repo:org/repo" in query

    def test_tags_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", tags=["needs-review", "urgent"])
        query = poller.build_query(trigger)
        assert 'label:"needs-review"' in query
        assert 'label:"urgent"' in query

    def test_custom_query_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="is:pr author:octocat")
        query = poller.build_query(trigger)
        assert "is:pr author:octocat" in query

    def test_default_state_open(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")
        query = poller.build_query(trigger)
        assert "state:open" in query

    def test_no_default_state_when_state_in_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="state:closed")
        query = poller.build_query(trigger)
        assert query.count("state:") == 1  # Only the custom one

    def test_no_default_state_when_is_in_filter(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", query_filter="is:open is:pr")
        query = poller.build_query(trigger)
        assert "state:open" not in query

    def test_combined_filters(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(
            source="github",
            repo="myorg/myrepo",
            tags=["review-needed"],
            query_filter="is:pr",
        )
        query = poller.build_query(trigger)
        assert "repo:myorg/myrepo" in query
        assert 'label:"review-needed"' in query
        assert "is:pr" in query

    def test_empty_trigger(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github")
        query = poller.build_query(trigger)
        # Should only have default state:open
        assert query == "state:open"


class TestGitHubPollerPoll:
    """Tests for GitHubPoller.poll."""

    def test_successful_poll(self) -> None:
        client = MockGitHubClient(
            issues=[
                {"number": 1, "title": "Issue 1"},
                {"number": 2, "title": "Issue 2"},
            ]
        )
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo")

        issues = poller.poll(trigger)

        assert len(issues) == 2
        assert issues[0].number == 1
        assert issues[1].number == 2

    def test_poll_calls_client_with_query(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/repo", tags=["review"])

        poller.poll(trigger, max_results=25)

        assert len(client.search_calls) == 1
        query, max_results = client.search_calls[0]
        assert "repo:org/repo" in query
        assert 'label:"review"' in query
        assert max_results == 25

    def test_poll_empty_query_returns_empty(self) -> None:
        client = MockGitHubClient()
        poller = GitHubPoller(client)
        trigger = TriggerConfig(source="github", repo="org/empty")

        issues = poller.poll(trigger)

        assert issues == []

    def test_poll_retries_on_failure(self) -> None:
        client = MockGitHubClient(issues=[{"number": 1, "title": "Test"}])
        client.should_fail = True
        client.max_fails = 2  # Fail twice, then succeed

        poller = GitHubPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(source="github", repo="org/repo")

        with patch("sentinel.github_poller.time.sleep"):
            issues = poller.poll(trigger)

        assert len(issues) == 1
        assert len(client.search_calls) == 3  # 2 failures + 1 success

    def test_poll_raises_after_max_retries(self) -> None:
        client = MockGitHubClient()
        client.should_fail = True
        client.max_fails = 10  # Always fail

        poller = GitHubPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(source="github", repo="org/repo")

        with (
            patch("sentinel.github_poller.time.sleep"),
            pytest.raises(GitHubClientError, match="Failed to poll GitHub after 3 attempts"),
        ):
            poller.poll(trigger)

        assert len(client.search_calls) == 3

    def test_exponential_backoff(self) -> None:
        client = MockGitHubClient()
        client.should_fail = True
        client.max_fails = 10  # Always fail

        poller = GitHubPoller(client, max_retries=3, retry_delay=1.0)
        trigger = TriggerConfig(source="github", repo="org/repo")

        sleep_times: list[float] = []
        with patch("sentinel.github_poller.time.sleep", side_effect=lambda x: sleep_times.append(x)):
            with pytest.raises(GitHubClientError):
                poller.poll(trigger)

        # Check exponential backoff: 1.0, 2.0 (only 2 sleeps for 3 retries)
        assert len(sleep_times) == 2
        assert sleep_times[0] == 1.0
        assert sleep_times[1] == 2.0
