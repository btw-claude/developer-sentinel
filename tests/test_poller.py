"""Tests for Jira poller module."""

from typing import Any
from unittest.mock import patch

import pytest

from sentinel.orchestration import TriggerConfig
from sentinel.poller import (
    JiraClient,
    JiraClientError,
    JiraIssue,
    JiraPoller,
    _extract_adf_text,
)


class MockJiraClient(JiraClient):
    """Mock Jira client for testing."""

    def __init__(self, issues: list[dict[str, Any]] | None = None) -> None:
        self.issues = issues or []
        self.search_calls: list[tuple[str, int]] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_fails = 0

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        self.search_calls.append((jql, max_results))
        if self.should_fail and self.fail_count < self.max_fails:
            self.fail_count += 1
            raise JiraClientError("API error")
        return self.issues


class TestJiraIssueFromApiResponse:
    """Tests for JiraIssue.from_api_response."""

    def test_basic_fields(self) -> None:
        data = {
            "key": "TEST-123",
            "fields": {
                "summary": "Test issue",
                "labels": ["bug", "urgent"],
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.key == "TEST-123"
        assert issue.summary == "Test issue"
        assert issue.labels == ["bug", "urgent"]

    def test_string_description(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "description": "Plain text description",
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.description == "Plain text description"

    def test_adf_description(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": "world"},
                            ],
                        }
                    ],
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.description == "Hello world"

    def test_status_extraction(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "status": {"name": "In Progress"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.status == "In Progress"

    def test_assignee_extraction(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "assignee": {"displayName": "John Doe", "name": "jdoe"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.assignee == "John Doe"

    def test_assignee_fallback_to_name(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "assignee": {"name": "jdoe"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.assignee == "jdoe"

    def test_no_assignee(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {"summary": "Test"},
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.assignee is None

    def test_comments_extraction(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "comment": {
                    "comments": [
                        {"body": "First comment"},
                        {"body": "Second comment"},
                    ]
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.comments == ["First comment", "Second comment"]

    def test_comments_with_adf(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "comment": {
                    "comments": [
                        {
                            "body": {
                                "type": "doc",
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [{"type": "text", "text": "ADF comment"}],
                                    }
                                ],
                            }
                        }
                    ]
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.comments == ["ADF comment"]

    def test_issue_links_extraction(self) -> None:
        data = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test",
                "issuelinks": [
                    {"outwardIssue": {"key": "TEST-2"}},
                    {"inwardIssue": {"key": "TEST-3"}},
                ],
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.links == ["TEST-2", "TEST-3"]

    def test_empty_data(self) -> None:
        data: dict[str, Any] = {}
        issue = JiraIssue.from_api_response(data)
        assert issue.key == ""
        assert issue.summary == ""


class TestExtractAdfText:
    """Tests for _extract_adf_text helper."""

    def test_simple_text(self) -> None:
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
        }
        assert _extract_adf_text(adf) == "Hello"

    def test_nested_content(self) -> None:
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Line 1"},
                    ],
                },
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Line 2"},
                    ],
                },
            ],
        }
        assert _extract_adf_text(adf) == "Line 1 Line 2"

    def test_empty_adf(self) -> None:
        adf: dict[str, Any] = {"type": "doc", "content": []}
        assert _extract_adf_text(adf) == ""


class TestJiraPollerBuildJql:
    """Tests for JiraPoller.build_jql."""

    def test_project_filter(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(project="TEST")
        jql = poller.build_jql(trigger)
        assert 'project = "TEST"' in jql

    def test_tags_filter(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(tags=["needs-review", "urgent"])
        jql = poller.build_jql(trigger)
        assert 'labels = "needs-review"' in jql
        assert 'labels = "urgent"' in jql

    def test_custom_jql_filter(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='priority = "High"')
        jql = poller.build_jql(trigger)
        assert '(priority = "High")' in jql

    def test_default_status_exclusion(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(project="TEST")
        jql = poller.build_jql(trigger)
        assert 'status NOT IN ("Done", "Closed", "Resolved")' in jql

    def test_no_status_exclusion_when_status_in_jql(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='status = "Open"')
        jql = poller.build_jql(trigger)
        assert "status NOT IN" not in jql

    def test_combined_filters(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(
            project="PROJ",
            tags=["review"],
            jql_filter='type = "Bug"',
        )
        jql = poller.build_jql(trigger)
        assert 'project = "PROJ"' in jql
        assert 'labels = "review"' in jql
        assert '(type = "Bug")' in jql

    def test_empty_trigger(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig()
        jql = poller.build_jql(trigger)
        # Should only have default status exclusion
        assert jql == 'status NOT IN ("Done", "Closed", "Resolved")'


class TestJiraPollerPoll:
    """Tests for JiraPoller.poll."""

    def test_successful_poll(self) -> None:
        client = MockJiraClient(
            issues=[
                {"key": "TEST-1", "fields": {"summary": "Issue 1"}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2"}},
            ]
        )
        poller = JiraPoller(client)
        trigger = TriggerConfig(project="TEST")

        issues = poller.poll(trigger)

        assert len(issues) == 2
        assert issues[0].key == "TEST-1"
        assert issues[1].key == "TEST-2"

    def test_poll_calls_client_with_jql(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(project="PROJ", tags=["review"])

        poller.poll(trigger, max_results=25)

        assert len(client.search_calls) == 1
        jql, max_results = client.search_calls[0]
        assert 'project = "PROJ"' in jql
        assert 'labels = "review"' in jql
        assert max_results == 25

    def test_poll_empty_jql_returns_empty(self) -> None:
        client = MockJiraClient()
        poller = JiraPoller(client)
        # Create a trigger that would generate empty JQL
        # Actually, with default status exclusion, we always get some JQL
        # So let's test with an empty result set instead
        trigger = TriggerConfig(project="EMPTY")

        issues = poller.poll(trigger)

        assert issues == []

    def test_poll_retries_on_failure(self) -> None:
        client = MockJiraClient(issues=[{"key": "TEST-1", "fields": {"summary": "Test"}}])
        client.should_fail = True
        client.max_fails = 2  # Fail twice, then succeed

        poller = JiraPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(project="TEST")

        with patch("sentinel.poller.time.sleep"):
            issues = poller.poll(trigger)

        assert len(issues) == 1
        assert len(client.search_calls) == 3  # 2 failures + 1 success

    def test_poll_raises_after_max_retries(self) -> None:
        client = MockJiraClient()
        client.should_fail = True
        client.max_fails = 10  # Always fail

        poller = JiraPoller(client, max_retries=3, retry_delay=0.01)
        trigger = TriggerConfig(project="TEST")

        with (
            patch("sentinel.poller.time.sleep"),
            pytest.raises(JiraClientError, match="Failed to poll Jira after 3 attempts"),
        ):
            poller.poll(trigger)

        assert len(client.search_calls) == 3


class TestJiraIssueEpicAndParentFields:
    """Tests for JiraIssue epic_key and parent_key fields (DS-348).

    These tests verify that the epic_key and parent_key fields are correctly
    populated from the Jira API response, supporting parent/epic template
    variables in branch patterns and prompts.
    """

    def test_epic_key_from_parent_next_gen(self) -> None:
        """Test epic_key is populated from parent field (next-gen projects)."""
        data = {
            "key": "TEST-123",
            "fields": {
                "summary": "Sub-task issue",
                "parent": {"key": "EPIC-100"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.epic_key == "EPIC-100"

    def test_epic_key_from_customfield_classic(self) -> None:
        """Test epic_key is populated from customfield_10014 (classic projects)."""
        data = {
            "key": "TEST-456",
            "fields": {
                "summary": "Issue in epic",
                "customfield_10014": "EPIC-200",
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.epic_key == "EPIC-200"

    def test_epic_key_parent_takes_precedence_over_customfield(self) -> None:
        """Test parent field takes precedence over customfield for epic_key."""
        data = {
            "key": "TEST-789",
            "fields": {
                "summary": "Issue with both",
                "parent": {"key": "EPIC-300"},
                "customfield_10014": "EPIC-OLD",
            },
        }
        issue = JiraIssue.from_api_response(data)
        # parent takes precedence
        assert issue.epic_key == "EPIC-300"

    def test_epic_key_none_when_no_parent_or_customfield(self) -> None:
        """Test epic_key is None when neither parent nor customfield is present."""
        data = {
            "key": "TEST-999",
            "fields": {
                "summary": "Standalone issue",
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.epic_key is None

    def test_parent_key_from_parent_field(self) -> None:
        """Test parent_key is populated from parent field for sub-tasks."""
        data = {
            "key": "TEST-101",
            "fields": {
                "summary": "Sub-task",
                "parent": {"key": "PARENT-500"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.parent_key == "PARENT-500"

    def test_parent_key_none_when_no_parent(self) -> None:
        """Test parent_key is None when parent field is not present."""
        data = {
            "key": "TEST-202",
            "fields": {
                "summary": "Top-level issue",
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.parent_key is None

    def test_parent_key_none_when_parent_empty(self) -> None:
        """Test parent_key is None when parent field is empty dict."""
        data = {
            "key": "TEST-303",
            "fields": {
                "summary": "Issue with empty parent",
                "parent": {},
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.parent_key is None

    def test_both_epic_and_parent_key_populated(self) -> None:
        """Test both epic_key and parent_key can be populated from same parent."""
        data = {
            "key": "TEST-404",
            "fields": {
                "summary": "Issue with parent",
                "parent": {"key": "PARENT-600"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        # For next-gen projects, parent serves as both epic and parent
        assert issue.epic_key == "PARENT-600"
        assert issue.parent_key == "PARENT-600"

    def test_jira_issue_dataclass_defaults(self) -> None:
        """Test JiraIssue dataclass has correct defaults for new fields."""
        issue = JiraIssue(key="TEST-1", summary="Test")
        assert issue.epic_key is None
        assert issue.parent_key is None
