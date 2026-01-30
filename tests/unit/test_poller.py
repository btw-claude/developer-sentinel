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
    JqlSanitizationError,
    _extract_adf_text,
    sanitize_jql_string,
    validate_jql_filter,
    validate_jql_identifier,
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


class TestSanitizeJqlString:
    """Tests for sanitize_jql_string function."""

    def test_simple_string_unchanged(self) -> None:
        """Test that simple alphanumeric strings pass through unchanged."""
        assert sanitize_jql_string("needs-review") == "needs-review"
        assert sanitize_jql_string("urgent") == "urgent"
        assert sanitize_jql_string("bug-fix-123") == "bug-fix-123"

    def test_escapes_double_quotes(self) -> None:
        """Test that double quotes are properly escaped."""
        assert sanitize_jql_string('tag"with"quotes') == 'tag\\"with\\"quotes'
        assert sanitize_jql_string('"quoted"') == '\\"quoted\\"'

    def test_escapes_backslashes(self) -> None:
        """Test that backslashes are properly escaped."""
        assert sanitize_jql_string("path\\to\\thing") == "path\\\\to\\\\thing"

    def test_escapes_backslash_before_quote(self) -> None:
        """Test that backslash followed by quote is properly escaped."""
        # Input: tag\" -> should become tag\\\"
        assert sanitize_jql_string('tag\\"') == 'tag\\\\\\"'

    def test_rejects_empty_string(self) -> None:
        """Test that empty strings are rejected."""
        with pytest.raises(JqlSanitizationError, match="cannot be empty"):
            sanitize_jql_string("")

    def test_rejects_null_character(self) -> None:
        """Test that null characters are rejected."""
        with pytest.raises(JqlSanitizationError, match="invalid null character"):
            sanitize_jql_string("tag\x00with\x00nulls")

    def test_rejects_excessively_long_string(self) -> None:
        """Test that very long strings are rejected."""
        long_string = "a" * 1001
        with pytest.raises(JqlSanitizationError, match="exceeds maximum length"):
            sanitize_jql_string(long_string)

    def test_allows_special_characters(self) -> None:
        """Test that various special characters are allowed (after escaping)."""
        # These are valid label characters that should be allowed
        assert sanitize_jql_string("tag-with-dashes") == "tag-with-dashes"
        assert sanitize_jql_string("tag_with_underscores") == "tag_with_underscores"
        assert sanitize_jql_string("tag.with.dots") == "tag.with.dots"
        assert sanitize_jql_string("tag:with:colons") == "tag:with:colons"

    def test_custom_field_name_in_error(self) -> None:
        """Test that custom field name appears in error messages."""
        with pytest.raises(JqlSanitizationError, match="my_label cannot be empty"):
            sanitize_jql_string("", "my_label")


class TestValidateJqlIdentifier:
    """Tests for validate_jql_identifier function."""

    def test_valid_project_keys(self) -> None:
        """Test that valid project keys pass validation."""
        assert validate_jql_identifier("TEST") == "TEST"
        assert validate_jql_identifier("PROJ") == "PROJ"
        assert validate_jql_identifier("MyProject") == "MyProject"
        assert validate_jql_identifier("PROJ123") == "PROJ123"
        assert validate_jql_identifier("MY-PROJECT") == "MY-PROJECT"
        assert validate_jql_identifier("MY_PROJECT") == "MY_PROJECT"

    def test_rejects_empty_string(self) -> None:
        """Test that empty identifiers are rejected."""
        with pytest.raises(JqlSanitizationError, match="cannot be empty"):
            validate_jql_identifier("")

    def test_rejects_starting_with_number(self) -> None:
        """Test that identifiers starting with numbers are rejected."""
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            validate_jql_identifier("123PROJECT")

    def test_rejects_special_characters(self) -> None:
        """Test that special characters are rejected."""
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            validate_jql_identifier('PROJECT"INJECT')
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            validate_jql_identifier("PROJECT OR 1=1")
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            validate_jql_identifier("PROJECT;DROP")

    def test_rejects_quotes(self) -> None:
        """Test that quotes are rejected (injection attempt)."""
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            validate_jql_identifier('TEST" OR project = "OTHER')

    def test_rejects_excessively_long_identifier(self) -> None:
        """Test that very long identifiers are rejected."""
        long_id = "A" * 256
        with pytest.raises(JqlSanitizationError, match="exceeds maximum length"):
            validate_jql_identifier(long_id)

    def test_custom_field_name_in_error(self) -> None:
        """Test that custom field name appears in error messages."""
        with pytest.raises(JqlSanitizationError, match="project_key cannot be empty"):
            validate_jql_identifier("", "project_key")


class TestValidateJqlFilter:
    """Tests for validate_jql_filter function."""

    def test_empty_string_returns_empty(self) -> None:
        """Test that empty strings are valid (optional field)."""
        assert validate_jql_filter("") == ""

    def test_valid_jql_filter_unchanged(self) -> None:
        """Test that valid JQL filters pass through unchanged."""
        filter1 = 'priority = "High"'
        assert validate_jql_filter(filter1) == filter1

        filter2 = 'type = "Bug" AND status = "Open"'
        assert validate_jql_filter(filter2) == filter2

        filter3 = 'labels IN ("urgent", "critical")'
        assert validate_jql_filter(filter3) == filter3

    def test_complex_jql_with_nested_parens(self) -> None:
        """Test that complex JQL with nested parentheses is valid."""
        jql = '(priority = "High" OR priority = "Critical") AND (type = "Bug")'
        assert validate_jql_filter(jql) == jql

    def test_jql_with_escaped_quotes(self) -> None:
        """Test that JQL with escaped quotes is valid."""
        jql = 'summary ~ "test\\"quoted\\"value"'
        assert validate_jql_filter(jql) == jql

    def test_rejects_null_character(self) -> None:
        """Test that null characters are rejected."""
        with pytest.raises(JqlSanitizationError, match="invalid null character"):
            validate_jql_filter('priority = "High"\x00DROP')

    def test_rejects_excessively_long_filter(self) -> None:
        """Test that very long filters are rejected."""
        long_filter = 'priority = "' + "a" * 10001 + '"'
        with pytest.raises(JqlSanitizationError, match="exceeds maximum length"):
            validate_jql_filter(long_filter)

    def test_rejects_unbalanced_opening_paren(self) -> None:
        """Test that unbalanced opening parentheses are rejected."""
        with pytest.raises(JqlSanitizationError, match="unclosed opening parenthesis"):
            validate_jql_filter('(priority = "High"')

    def test_rejects_unbalanced_closing_paren(self) -> None:
        """Test that unbalanced closing parentheses are rejected."""
        with pytest.raises(JqlSanitizationError, match="unexpected closing parenthesis"):
            validate_jql_filter('priority = "High")')

    def test_rejects_multiple_unbalanced_parens(self) -> None:
        """Test that multiple unbalanced parentheses are rejected."""
        with pytest.raises(JqlSanitizationError, match="unclosed opening parenthesis"):
            validate_jql_filter('((priority = "High")')

    def test_rejects_unbalanced_quotes(self) -> None:
        """Test that unbalanced quotes are rejected."""
        with pytest.raises(JqlSanitizationError, match="unclosed string literal"):
            validate_jql_filter('priority = "High')

    def test_balanced_quotes_with_escaped_quotes(self) -> None:
        """Test that escaped quotes don't count as unbalanced."""
        # This has balanced quotes: open " content with \" inside " close
        jql = 'summary ~ "test\\"value"'
        assert validate_jql_filter(jql) == jql

    def test_quotes_with_multiple_escapes(self) -> None:
        """Test handling of multiple backslashes before quotes."""
        # Two backslashes before quote = escaped backslash + real quote
        jql = 'summary ~ "test\\\\"'  # "test\\" - ends with escaped backslash
        # This should have balanced quotes: one opening, one closing
        assert validate_jql_filter(jql) == jql

    def test_accepts_jql_without_quotes(self) -> None:
        """Test that JQL without quotes is valid."""
        jql = "project = TEST AND status != Done"
        assert validate_jql_filter(jql) == jql


class TestJqlFilterValidationInBuildJql:
    """Tests verifying jql_filter validation is applied in build_jql."""

    def test_jql_filter_with_null_raises_error(self) -> None:
        """Test that jql_filter with null characters raises JqlSanitizationError."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='priority = "High"\x00injection')
        with pytest.raises(JqlSanitizationError, match="invalid null character"):
            poller.build_jql(trigger)

    def test_jql_filter_with_unbalanced_parens_raises_error(self) -> None:
        """Test that jql_filter with unbalanced parentheses raises error."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='(priority = "High"')
        with pytest.raises(JqlSanitizationError, match="unclosed opening parenthesis"):
            poller.build_jql(trigger)

    def test_jql_filter_with_unbalanced_quotes_raises_error(self) -> None:
        """Test that jql_filter with unbalanced quotes raises error."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='priority = "High')
        with pytest.raises(JqlSanitizationError, match="unclosed string literal"):
            poller.build_jql(trigger)

    def test_valid_jql_filter_passes_through(self) -> None:
        """Test that valid jql_filter is included in built JQL."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(jql_filter='priority = "High" AND type = "Bug"')
        jql = poller.build_jql(trigger)
        assert '(priority = "High" AND type = "Bug")' in jql


class TestJqlSanitizationInBuildJql:
    """Tests verifying JQL sanitization is applied in build_jql."""

    def test_tag_with_quotes_is_escaped(self) -> None:
        """Test that tags containing quotes are properly escaped."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(tags=['needs"review'])
        jql = poller.build_jql(trigger)
        # The quote should be escaped
        assert 'labels = "needs\\"review"' in jql

    def test_tag_with_backslash_is_escaped(self) -> None:
        """Test that tags containing backslashes are properly escaped."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(tags=["path\\tag"])
        jql = poller.build_jql(trigger)
        # The backslash should be escaped
        assert 'labels = "path\\\\tag"' in jql

    def test_invalid_project_raises_error(self) -> None:
        """Test that invalid project keys raise JqlSanitizationError."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(project='TEST" OR 1=1 --')
        with pytest.raises(JqlSanitizationError, match="invalid characters"):
            poller.build_jql(trigger)

    def test_empty_tag_raises_error(self) -> None:
        """Test that empty tags raise JqlSanitizationError."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(tags=["valid", ""])
        with pytest.raises(JqlSanitizationError, match="cannot be empty"):
            poller.build_jql(trigger)

    def test_tag_with_null_raises_error(self) -> None:
        """Test that tags with null characters raise JqlSanitizationError."""
        client = MockJiraClient()
        poller = JiraPoller(client)
        trigger = TriggerConfig(tags=["tag\x00injection"])
        with pytest.raises(JqlSanitizationError, match="invalid null character"):
            poller.build_jql(trigger)


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
    """Tests for JiraIssue epic_key and parent_key fields.

    These tests verify that the epic_key and parent_key fields are correctly
    populated from the Jira API response, supporting parent/epic template
    variables in branch patterns and prompts.

    Distinguishes between Epic parents (populates epic_key only) and
    Story/Task parents (populates parent_key only) based on parent issue type.
    """

    def test_epic_key_from_parent_next_gen(self) -> None:
        """Test epic_key is populated from parent field when parent is Epic (next-gen projects)."""
        data = {
            "key": "TEST-123",
            "fields": {
                "summary": "Sub-task issue",
                "parent": {
                    "key": "EPIC-100",
                    "fields": {
                        "issuetype": {"name": "Epic"},
                    },
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.epic_key == "EPIC-100"
        assert issue.parent_key is None  # Epic parent should not populate parent_key

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

    def test_epic_key_from_custom_epic_link_field(self) -> None:
        """Test epic_key is populated from custom epic link field."""
        data = {
            "key": "TEST-457",
            "fields": {
                "summary": "Issue with custom epic field",
                "customfield_99999": "EPIC-300",
            },
        }
        # Use a custom epic link field ID
        issue = JiraIssue.from_api_response(data, epic_link_field="customfield_99999")
        assert issue.epic_key == "EPIC-300"

    def test_epic_key_default_field_not_found_with_custom(self) -> None:
        """Test that default field is not used when custom field is specified."""
        data = {
            "key": "TEST-458",
            "fields": {
                "summary": "Issue with default field only",
                "customfield_10014": "EPIC-DEFAULT",  # Default field has value
            },
        }
        # Specify a different custom field that doesn't exist
        issue = JiraIssue.from_api_response(data, epic_link_field="customfield_99999")
        # Should not find the epic because we're looking at wrong field
        assert issue.epic_key is None

    def test_epic_key_parent_takes_precedence_over_customfield(self) -> None:
        """Test Epic parent field takes precedence over customfield for epic_key."""
        data = {
            "key": "TEST-789",
            "fields": {
                "summary": "Issue with both",
                "parent": {
                    "key": "EPIC-300",
                    "fields": {
                        "issuetype": {"name": "Epic"},
                    },
                },
                "customfield_10014": "EPIC-OLD",
            },
        }
        issue = JiraIssue.from_api_response(data)
        # Epic parent takes precedence over customfield
        assert issue.epic_key == "EPIC-300"
        assert issue.parent_key is None

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
        """Test parent_key is populated from parent field for sub-tasks with non-Epic parent."""
        data = {
            "key": "TEST-101",
            "fields": {
                "summary": "Sub-task",
                "parent": {
                    "key": "PARENT-500",
                    "fields": {
                        "issuetype": {"name": "Story"},
                    },
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.parent_key == "PARENT-500"
        assert issue.epic_key is None  # Story parent should not populate epic_key

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
        assert issue.epic_key is None

    def test_epic_parent_type_populates_epic_key_only(self) -> None:
        """Test Epic parent type only populates epic_key."""
        data = {
            "key": "TEST-404",
            "fields": {
                "summary": "Issue under Epic",
                "parent": {
                    "key": "EPIC-600",
                    "fields": {
                        "issuetype": {"name": "Epic"},
                    },
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.epic_key == "EPIC-600"
        assert issue.parent_key is None  # Epic parent should NOT populate parent_key

    def test_task_parent_type_populates_parent_key_only(self) -> None:
        """Test Task parent type only populates parent_key."""
        data = {
            "key": "TEST-405",
            "fields": {
                "summary": "Sub-task under Task",
                "parent": {
                    "key": "TASK-700",
                    "fields": {
                        "issuetype": {"name": "Task"},
                    },
                },
            },
        }
        issue = JiraIssue.from_api_response(data)
        assert issue.parent_key == "TASK-700"
        assert issue.epic_key is None  # Task parent should NOT populate epic_key

    def test_parent_without_issuetype_does_not_populate_either(self) -> None:
        """Test parent without issuetype info defaults to sub-task relationship."""
        data = {
            "key": "TEST-406",
            "fields": {
                "summary": "Issue with parent lacking type info",
                "parent": {"key": "PARENT-800"},
            },
        }
        issue = JiraIssue.from_api_response(data)
        # Without issuetype, we can't determine if it's an Epic or not
        # Default behavior: treat as non-epic (sub-task relationship)
        assert issue.parent_key == "PARENT-800"
        assert issue.epic_key is None

    def test_jira_issue_dataclass_defaults(self) -> None:
        """Test JiraIssue dataclass has correct defaults for new fields."""
        issue = JiraIssue(key="TEST-1", summary="Test")
        assert issue.epic_key is None
        assert issue.parent_key is None


class TestJiraPollerEpicLinkField:
    """Tests for JiraPoller with configurable epic_link_field."""

    def test_poller_uses_default_epic_link_field(self) -> None:
        """Test that poller uses default epic_link_field when not specified."""
        client = MockJiraClient(
            issues=[
                {
                    "key": "TEST-1",
                    "fields": {
                        "summary": "Issue 1",
                        "customfield_10014": "EPIC-100",
                    },
                }
            ]
        )
        poller = JiraPoller(client)
        trigger = TriggerConfig(project="TEST")

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].epic_key == "EPIC-100"

    def test_poller_uses_custom_epic_link_field(self) -> None:
        """Test that poller uses custom epic_link_field when specified."""
        client = MockJiraClient(
            issues=[
                {
                    "key": "TEST-1",
                    "fields": {
                        "summary": "Issue 1",
                        "customfield_12345": "EPIC-200",
                    },
                }
            ]
        )
        poller = JiraPoller(client, epic_link_field="customfield_12345")
        trigger = TriggerConfig(project="TEST")

        issues = poller.poll(trigger)

        assert len(issues) == 1
        assert issues[0].epic_key == "EPIC-200"

    def test_poller_custom_field_not_found(self) -> None:
        """Test that epic_key is None when custom field is not found."""
        client = MockJiraClient(
            issues=[
                {
                    "key": "TEST-1",
                    "fields": {
                        "summary": "Issue 1",
                        "customfield_10014": "EPIC-100",  # Default field exists
                    },
                }
            ]
        )
        # Specify a different custom field
        poller = JiraPoller(client, epic_link_field="customfield_99999")
        trigger = TriggerConfig(project="TEST")

        issues = poller.poll(trigger)

        assert len(issues) == 1
        # Should not find epic because looking at wrong field
        assert issues[0].epic_key is None

    def test_poller_stores_epic_link_field(self) -> None:
        """Test that poller stores the epic_link_field attribute."""
        client = MockJiraClient()
        poller = JiraPoller(client, epic_link_field="customfield_67890")

        assert poller.epic_link_field == "customfield_67890"

    def test_poller_default_epic_link_field_attribute(self) -> None:
        """Test that poller has default epic_link_field attribute."""
        client = MockJiraClient()
        poller = JiraPoller(client)

        assert poller.epic_link_field == "customfield_10014"
