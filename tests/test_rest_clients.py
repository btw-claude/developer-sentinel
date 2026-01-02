"""Tests for REST-based Jira clients."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.poller import JiraClientError
from sentinel.rest_clients import (
    DEFAULT_TIMEOUT,
    JiraRestClient,
    JiraRestTagClient,
)
from sentinel.tag_manager import JiraTagClientError


class TestJiraRestClient:
    """Tests for JiraRestClient."""

    def test_init_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net/",
            email="test@example.com",
            api_token="token123",
        )
        assert client.base_url == "https://example.atlassian.net"

    def test_init_uses_default_timeout(self) -> None:
        """Test that default timeout is used when not specified."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )
        assert client.timeout == DEFAULT_TIMEOUT

    def test_init_uses_custom_timeout(self) -> None:
        """Test that custom timeout is used when specified."""
        custom_timeout = httpx.Timeout(5.0)
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
            timeout=custom_timeout,
        )
        assert client.timeout == custom_timeout

    def test_search_issues_success(self) -> None:
        """Test successful issue search."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "issues": [
                {"key": "PROJ-1", "fields": {"summary": "Issue 1"}},
                {"key": "PROJ-2", "fields": {"summary": "Issue 2"}},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            issues = client.search_issues("project = PROJ", max_results=50)

            assert len(issues) == 2
            assert issues[0]["key"] == "PROJ-1"
            assert issues[1]["key"] == "PROJ-2"

            # Verify the request was made correctly
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert call_args[0][0] == "https://example.atlassian.net/rest/api/3/search"
            assert call_args[1]["params"]["jql"] == "project = PROJ"
            assert call_args[1]["params"]["maxResults"] == 50

    def test_search_issues_empty_result(self) -> None:
        """Test search returning no issues."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {"issues": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            issues = client.search_issues("project = EMPTY")

            assert issues == []

    def test_search_issues_timeout_error(self) -> None:
        """Test that timeout raises JiraClientError."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client_class.return_value = mock_client

            with pytest.raises(JiraClientError) as exc_info:
                client.search_issues("project = PROJ")

            assert "timed out" in str(exc_info.value)

    def test_search_issues_http_status_error(self) -> None:
        """Test that HTTP errors raise JiraClientError with error messages."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"errorMessages": ["Invalid JQL"]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Bad Request", request=mock_request, response=mock_response
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(JiraClientError) as exc_info:
                client.search_issues("invalid jql")

            assert "400" in str(exc_info.value)
            assert "Invalid JQL" in str(exc_info.value)

    def test_search_issues_request_error(self) -> None:
        """Test that request errors raise JiraClientError."""
        client = JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.RequestError("Connection failed")
            mock_client_class.return_value = mock_client

            with pytest.raises(JiraClientError) as exc_info:
                client.search_issues("project = PROJ")

            assert "request failed" in str(exc_info.value)


class TestJiraRestTagClient:
    """Tests for JiraRestTagClient."""

    def test_init_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net/",
            email="test@example.com",
            api_token="token123",
        )
        assert client.base_url == "https://example.atlassian.net"

    def test_add_label_success(self) -> None:
        """Test successfully adding a label."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        # Mock get labels response
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["existing-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        # Mock put response
        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            client.add_label("PROJ-123", "new-label")

            # Verify get was called to fetch current labels
            mock_client.get.assert_called()

            # Verify put was called with new label added
            mock_client.put.assert_called()
            put_call = mock_client.put.call_args
            assert put_call[1]["json"]["fields"]["labels"] == ["existing-label", "new-label"]

    def test_add_label_already_exists(self) -> None:
        """Test adding a label that already exists (no-op)."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["existing-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client_class.return_value = mock_client

            client.add_label("PROJ-123", "existing-label")

            # Verify put was NOT called since label already exists
            mock_client.put.assert_not_called()

    def test_remove_label_success(self) -> None:
        """Test successfully removing a label."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {
            "fields": {"labels": ["label-to-remove", "other-label"]}
        }
        mock_get_response.raise_for_status = MagicMock()

        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            client.remove_label("PROJ-123", "label-to-remove")

            # Verify put was called with label removed
            put_call = mock_client.put.call_args
            assert put_call[1]["json"]["fields"]["labels"] == ["other-label"]

    def test_remove_label_not_found(self) -> None:
        """Test removing a label that doesn't exist (no-op)."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["other-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client_class.return_value = mock_client

            client.remove_label("PROJ-123", "nonexistent-label")

            # Verify put was NOT called since label doesn't exist
            mock_client.put.assert_not_called()

    def test_add_label_timeout_error(self) -> None:
        """Test that timeout on get labels raises JiraTagClientError."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("Timed out")
            mock_client_class.return_value = mock_client

            with pytest.raises(JiraTagClientError) as exc_info:
                client.add_label("PROJ-123", "new-label")

            assert "timed out" in str(exc_info.value)

    def test_add_label_http_status_error(self) -> None:
        """Test that HTTP error on update raises JiraTagClientError."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": []}}
        mock_get_response.raise_for_status = MagicMock()

        mock_request = MagicMock()
        mock_put_response = MagicMock()
        mock_put_response.status_code = 403
        mock_put_response.json.return_value = {"errorMessages": ["Permission denied"]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client.put.side_effect = httpx.HTTPStatusError(
                "Forbidden", request=mock_request, response=mock_put_response
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(JiraTagClientError) as exc_info:
                client.add_label("PROJ-123", "new-label")

            assert "403" in str(exc_info.value)
            assert "Permission denied" in str(exc_info.value)

    def test_get_current_labels_empty_fields(self) -> None:
        """Test handling of empty fields in get labels response."""
        client = JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        )

        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {}}
        mock_get_response.raise_for_status = MagicMock()

        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            # Should not raise, should treat empty labels as []
            client.add_label("PROJ-123", "new-label")

            put_call = mock_client.put.call_args
            assert put_call[1]["json"]["fields"]["labels"] == ["new-label"]
