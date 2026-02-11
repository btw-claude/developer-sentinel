"""Tests for REST-based Jira clients."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.poller import JiraClientError
from sentinel.rest_clients import (
    DEFAULT_RETRY_CONFIG,
    DEFAULT_TIMEOUT,
    BaseJiraHttpClient,
    JiraAuth,
    JiraRestClient,
    JiraRestTagClient,
    RetryConfig,
    _calculate_backoff_delay,
    _check_rate_limit_warning,
    _execute_with_retry,
    _get_retry_after,
)
from sentinel.tag_manager import JiraTagClientError


class TestJiraRestClient:
    """Tests for JiraRestClient."""

    def test_init_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        with JiraRestClient(
            base_url="https://example.atlassian.net/",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert client.base_url == "https://example.atlassian.net"

    def test_init_uses_default_timeout(self) -> None:
        """Test that default timeout is used when not specified."""
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert client.timeout == DEFAULT_TIMEOUT

    def test_init_uses_custom_timeout(self) -> None:
        """Test that custom timeout is used when specified."""
        custom_timeout = httpx.Timeout(5.0)
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
            timeout=custom_timeout,
        ) as client:
            assert client.timeout == custom_timeout

    def test_init_stores_auth_as_basic_auth(self) -> None:
        """Test that auth is stored as httpx.BasicAuth for extensibility."""
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert isinstance(client.auth, httpx.BasicAuth)

    def test_search_issues_success(self) -> None:
        """Test successful issue search."""
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
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                issues = client.search_issues("project = PROJ", max_results=50)

                assert len(issues) == 2
                assert issues[0]["key"] == "PROJ-1"
                assert issues[1]["key"] == "PROJ-2"

                # Verify the request was made correctly
                mock_client.get.assert_called_once()
                call_args = mock_client.get.call_args
                assert (
                    call_args[0][0]
                    == "https://example.atlassian.net/rest/api/3/search/jql"
                )
                assert call_args[1]["params"]["jql"] == "project = PROJ"
                assert call_args[1]["params"]["maxResults"] == 50

    def test_search_issues_empty_result(self) -> None:
        """Test search returning no issues."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"issues": []}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                issues = client.search_issues("project = EMPTY")

                assert issues == []

    def test_search_issues_timeout_error(self) -> None:
        """Test that timeout raises JiraClientError."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                with pytest.raises(JiraClientError) as exc_info:
                    client.search_issues("project = PROJ")

                assert "timed out" in str(exc_info.value)

    def test_search_issues_http_status_error(self) -> None:
        """Test that HTTP errors raise JiraClientError with error messages."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"errorMessages": ["Invalid JQL"]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Bad Request", request=mock_request, response=mock_response
            )
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                with pytest.raises(JiraClientError) as exc_info:
                    client.search_issues("invalid jql")

                assert "400" in str(exc_info.value)
                assert "Invalid JQL" in str(exc_info.value)

    def test_search_issues_request_error(self) -> None:
        """Test that request errors raise JiraClientError."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.RequestError("Connection failed")
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                with pytest.raises(JiraClientError) as exc_info:
                    client.search_issues("project = PROJ")

                assert "request failed" in str(exc_info.value)


class TestJiraRestTagClient:
    """Tests for JiraRestTagClient."""

    def test_init_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        with JiraRestTagClient(
            base_url="https://example.atlassian.net/",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert client.base_url == "https://example.atlassian.net"

    def test_init_stores_auth_as_basic_auth(self) -> None:
        """Test that auth is stored as httpx.BasicAuth for extensibility."""
        with JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert isinstance(client.auth, httpx.BasicAuth)

    def test_add_label_success(self) -> None:
        """Test successfully adding a label."""
        # Mock get labels response
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["existing-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        # Mock put response
        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                client.add_label("PROJ-123", "new-label")

                # Verify get was called to fetch current labels
                mock_client.get.assert_called()

                # Verify put was called with new label added
                mock_client.put.assert_called()
                put_call = mock_client.put.call_args
                assert put_call[1]["json"]["fields"]["labels"] == [
                    "existing-label",
                    "new-label",
                ]

    def test_add_label_already_exists(self) -> None:
        """Test adding a label that already exists (no-op)."""
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["existing-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                client.add_label("PROJ-123", "existing-label")

                # Verify put was NOT called since label already exists
                mock_client.put.assert_not_called()

    def test_remove_label_success(self) -> None:
        """Test successfully removing a label."""
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {
            "fields": {"labels": ["label-to-remove", "other-label"]}
        }
        mock_get_response.raise_for_status = MagicMock()

        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                client.remove_label("PROJ-123", "label-to-remove")

                # Verify put was called with label removed
                put_call = mock_client.put.call_args
                assert put_call[1]["json"]["fields"]["labels"] == ["other-label"]

    def test_remove_label_not_found(self) -> None:
        """Test removing a label that doesn't exist (no-op)."""
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": ["other-label"]}}
        mock_get_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                client.remove_label("PROJ-123", "nonexistent-label")

                # Verify put was NOT called since label doesn't exist
                mock_client.put.assert_not_called()

    def test_add_label_timeout_error(self) -> None:
        """Test that timeout on get labels raises JiraTagClientError."""
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = httpx.TimeoutException("Timed out")
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                with pytest.raises(JiraTagClientError) as exc_info:
                    client.add_label("PROJ-123", "new-label")

                assert "timed out" in str(exc_info.value)

    def test_add_label_http_status_error(self) -> None:
        """Test that HTTP error on update raises JiraTagClientError."""
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {"labels": []}}
        mock_get_response.raise_for_status = MagicMock()

        mock_request = MagicMock()
        mock_put_response = MagicMock()
        mock_put_response.status_code = 403
        mock_put_response.json.return_value = {"errorMessages": ["Permission denied"]}

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client.put.side_effect = httpx.HTTPStatusError(
                "Forbidden", request=mock_request, response=mock_put_response
            )
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                with pytest.raises(JiraTagClientError) as exc_info:
                    client.add_label("PROJ-123", "new-label")

                assert "403" in str(exc_info.value)
                assert "Permission denied" in str(exc_info.value)

    def test_get_current_labels_empty_fields(self) -> None:
        """Test handling of empty fields in get labels response."""
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"fields": {}}
        mock_get_response.raise_for_status = MagicMock()

        mock_put_response = MagicMock()
        mock_put_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_get_response
            mock_client.put.return_value = mock_put_response
            mock_client_class.return_value = mock_client

            with JiraRestTagClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
            ) as client:
                # Should not raise, should treat empty labels as []
                client.add_label("PROJ-123", "new-label")

                put_call = mock_client.put.call_args
                assert put_call[1]["json"]["fields"]["labels"] == ["new-label"]


class TestBaseJiraHttpClient:
    """Tests for BaseJiraHttpClient base class."""

    def test_lazy_client_initialization(self) -> None:
        """Test that httpx.Client is not created until _get_client() is called."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("test@example.com", "token123")

        with TestClient() as client:
            assert client._client is None

    def test_client_reuse(self) -> None:
        """Test that _get_client() returns the same instance on subsequent calls."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("test@example.com", "token123")

        with patch("httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()
            with TestClient() as client:
                client1 = client._get_client()
                client2 = client._get_client()

                assert client1 is client2
                mock_client_class.assert_called_once()

    def test_close_clears_client(self) -> None:
        """Test that close() releases the httpx.Client."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("test@example.com", "token123")

        with patch("httpx.Client") as mock_client_class:
            mock_inner = MagicMock()
            mock_client_class.return_value = mock_inner

            with TestClient() as client:
                client._get_client()
                assert client._client is not None

                client.close()
                assert client._client is None
                mock_inner.close.assert_called_once()

    def test_context_manager(self) -> None:
        """Test that context manager calls close on exit."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("test@example.com", "token123")

        with patch("httpx.Client") as mock_client_class:
            mock_inner = MagicMock()
            mock_client_class.return_value = mock_inner

            with TestClient() as client:
                client._get_client()
                assert client._client is not None

            assert client._client is None
            mock_inner.close.assert_called_once()

    def test_close_without_init_is_noop(self) -> None:
        """Test that close() does nothing if client was never initialized."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("test@example.com", "token123")

        with TestClient() as client:
            client.close()  # Should not raise
            assert client._client is None

    def test_missing_timeout_raises_runtime_error(self) -> None:
        """Test RuntimeError when timeout is not set before _get_client()."""

        class BadClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.auth: JiraAuth = ("test@example.com", "token123")

        with BadClient() as client:
            with pytest.raises(RuntimeError, match="must set self.timeout"):
                client._get_client()

    def test_missing_auth_raises_runtime_error(self) -> None:
        """Test RuntimeError when auth is not set before _get_client()."""

        class BadClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)

        with BadClient() as client:
            with pytest.raises(RuntimeError, match="must set self.auth"):
                client._get_client()

    def test_client_created_with_auth_and_timeout(self) -> None:
        """Test that httpx.Client is created with correct auth and timeout."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(5.0)
                self.auth: JiraAuth = ("user@example.com", "mytoken")

        with patch("httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()
            with TestClient() as client:
                client._get_client()

                mock_client_class.assert_called_once_with(
                    auth=("user@example.com", "mytoken"),
                    timeout=httpx.Timeout(5.0),
                )

    def test_client_created_with_basic_auth(self) -> None:
        """Test that httpx.Client works with httpx.BasicAuth instances."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(5.0)
                self.auth: JiraAuth = httpx.BasicAuth("user@example.com", "mytoken")

        with patch("httpx.Client") as mock_client_class:
            mock_client_class.return_value = MagicMock()
            with TestClient() as client:
                client._get_client()

                call_kwargs = mock_client_class.call_args[1]
                assert isinstance(call_kwargs["auth"], httpx.BasicAuth)
                assert call_kwargs["timeout"] == httpx.Timeout(5.0)

    def test_jira_rest_client_inherits_base(self) -> None:
        """Test that JiraRestClient inherits from BaseJiraHttpClient."""
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert isinstance(client, BaseJiraHttpClient)
            assert client._client is None

    def test_jira_rest_tag_client_inherits_base(self) -> None:
        """Test that JiraRestTagClient inherits from BaseJiraHttpClient."""
        with JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert isinstance(client, BaseJiraHttpClient)
            assert client._client is None


class TestJiraAuthType:
    """Tests for JiraAuth type alias extensibility."""

    def test_tuple_auth_accepted(self) -> None:
        """Test that tuple-based auth is accepted by the JiraAuth type."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = ("user@example.com", "token")

        with TestClient() as client:
            assert client.auth == ("user@example.com", "token")

    def test_basic_auth_accepted(self) -> None:
        """Test that httpx.BasicAuth is accepted by the JiraAuth type."""

        class TestClient(BaseJiraHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self.auth: JiraAuth = httpx.BasicAuth("user@example.com", "token")

        with TestClient() as client:
            assert isinstance(client.auth, httpx.BasicAuth)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 4
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0
        assert config.jitter_min == 0.7
        assert config.jitter_max == 1.3

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=2,
            initial_delay=0.5,
            max_delay=10.0,
            jitter_min=0.8,
            jitter_max=1.2,
        )
        assert config.max_retries == 2
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.jitter_min == 0.8
        assert config.jitter_max == 1.2

    def test_default_retry_config_exists(self) -> None:
        """Test that DEFAULT_RETRY_CONFIG is available."""
        assert DEFAULT_RETRY_CONFIG is not None
        assert DEFAULT_RETRY_CONFIG.max_retries == 4


class TestCalculateBackoffDelay:
    """Tests for _calculate_backoff_delay."""

    def test_uses_retry_after_when_provided(self) -> None:
        """Test that Retry-After value is used when provided."""
        config = RetryConfig(jitter_min=1.0, jitter_max=1.0)  # No jitter
        delay = _calculate_backoff_delay(0, config, retry_after=5.0)
        assert delay == 5.0

    def test_exponential_backoff_attempt_0(self) -> None:
        """Test exponential backoff for first attempt."""
        config = RetryConfig(initial_delay=1.0, jitter_min=1.0, jitter_max=1.0)
        delay = _calculate_backoff_delay(0, config)
        assert delay == 1.0  # 1.0 * 2^0 = 1.0

    def test_exponential_backoff_attempt_1(self) -> None:
        """Test exponential backoff for second attempt."""
        config = RetryConfig(initial_delay=1.0, jitter_min=1.0, jitter_max=1.0)
        delay = _calculate_backoff_delay(1, config)
        assert delay == 2.0  # 1.0 * 2^1 = 2.0

    def test_exponential_backoff_attempt_2(self) -> None:
        """Test exponential backoff for third attempt."""
        config = RetryConfig(initial_delay=1.0, jitter_min=1.0, jitter_max=1.0)
        delay = _calculate_backoff_delay(2, config)
        assert delay == 4.0  # 1.0 * 2^2 = 4.0

    def test_respects_max_delay(self) -> None:
        """Test that delay is capped at max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter_min=1.0, jitter_max=1.0)
        delay = _calculate_backoff_delay(10, config)  # Would be 1024 without cap
        assert delay == 5.0

    def test_applies_jitter(self) -> None:
        """Test that jitter is applied within bounds."""
        config = RetryConfig(initial_delay=10.0, jitter_min=0.5, jitter_max=1.5)
        delays = [_calculate_backoff_delay(0, config) for _ in range(100)]
        # All delays should be between 5.0 (10 * 0.5) and 15.0 (10 * 1.5)
        assert all(5.0 <= d <= 15.0 for d in delays)
        # With 100 samples, we should see some variation
        assert len(set(delays)) > 1


class TestGetRetryAfter:
    """Tests for _get_retry_after."""

    def test_returns_float_when_present(self) -> None:
        """Test extraction of Retry-After header as float."""
        response = MagicMock()
        response.headers = {"Retry-After": "5"}
        assert _get_retry_after(response) == 5.0

    def test_returns_none_when_absent(self) -> None:
        """Test returns None when header is missing."""
        response = MagicMock()
        response.headers = {}
        assert _get_retry_after(response) is None

    def test_returns_none_for_invalid_value(self) -> None:
        """Test returns None for non-numeric header value."""
        response = MagicMock()
        response.headers = {"Retry-After": "invalid"}
        assert _get_retry_after(response) is None

    def test_handles_float_value(self) -> None:
        """Test handles float string values."""
        response = MagicMock()
        response.headers = {"Retry-After": "2.5"}
        assert _get_retry_after(response) == 2.5


class TestCheckRateLimitWarning:
    """Tests for _check_rate_limit_warning."""

    def test_logs_warning_when_near_limit(self) -> None:
        """Test that warning is logged when X-RateLimit-NearLimit is true."""
        response = MagicMock()
        response.headers = {
            "X-RateLimit-NearLimit": "true",
            "X-RateLimit-Remaining": "100",
            "X-RateLimit-Reset": "2024-01-01T00:00:00Z",
        }

        with patch("sentinel.rest_clients.logger") as mock_logger:
            _check_rate_limit_warning(response)
            mock_logger.warning.assert_called_once()
            assert "near exhaustion" in mock_logger.warning.call_args[0][0]

    def test_no_warning_when_not_near_limit(self) -> None:
        """Test no warning when X-RateLimit-NearLimit is false."""
        response = MagicMock()
        response.headers = {"X-RateLimit-NearLimit": "false"}

        with patch("sentinel.rest_clients.logger") as mock_logger:
            _check_rate_limit_warning(response)
            mock_logger.warning.assert_not_called()

    def test_no_warning_when_header_missing(self) -> None:
        """Test no warning when header is missing."""
        response = MagicMock()
        response.headers = {}

        with patch("sentinel.rest_clients.logger") as mock_logger:
            _check_rate_limit_warning(response)
            mock_logger.warning.assert_not_called()


class TestExecuteWithRetry:
    """Tests for _execute_with_retry."""

    def test_returns_result_on_success(self) -> None:
        """Test that successful operation returns result."""
        operation = MagicMock(return_value="success")
        result = _execute_with_retry(operation)
        assert result == "success"
        operation.assert_called_once()

    def test_retries_on_429_and_succeeds(self) -> None:
        """Test that 429 triggers retry and succeeds on subsequent attempt."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "0.01", "RateLimit-Reason": "test"}

        operation = MagicMock(
            side_effect=[
                httpx.HTTPStatusError(
                    "Rate limited", request=mock_request, response=mock_response
                ),
                "success",
            ]
        )

        config = RetryConfig(initial_delay=0.01, jitter_min=1.0, jitter_max=1.0)
        result = _execute_with_retry(operation, config)

        assert result == "success"
        assert operation.call_count == 2

    def test_raises_after_max_retries(self) -> None:
        """Test that exception is raised after max retries exhausted."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "0.01", "RateLimit-Reason": "quota-exceeded"}

        operation = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Rate limited", request=mock_request, response=mock_response
            )
        )

        config = RetryConfig(max_retries=2, initial_delay=0.01, jitter_min=1.0, jitter_max=1.0)

        with pytest.raises(JiraClientError) as exc_info:
            _execute_with_retry(operation, config, JiraClientError)

        assert "Rate limit exceeded" in str(exc_info.value)
        assert "2 retries" in str(exc_info.value)
        assert operation.call_count == 3  # Initial + 2 retries

    def test_does_not_retry_non_429_errors(self) -> None:
        """Test that non-429 HTTP errors are not retried."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 400

        operation = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Bad Request", request=mock_request, response=mock_response
            )
        )

        with pytest.raises(httpx.HTTPStatusError):
            _execute_with_retry(operation)

        operation.assert_called_once()

    def test_uses_custom_error_class(self) -> None:
        """Test that custom error class is used for final exception."""
        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "0.01", "RateLimit-Reason": "test"}

        operation = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Rate limited", request=mock_request, response=mock_response
            )
        )

        config = RetryConfig(max_retries=0, initial_delay=0.01)

        with pytest.raises(JiraTagClientError):
            _execute_with_retry(operation, config, JiraTagClientError)


class TestJiraRestClientRetry:
    """Tests for JiraRestClient retry behavior."""

    def test_uses_default_retry_config(self) -> None:
        """Test that default retry config is used."""
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert client.retry_config == DEFAULT_RETRY_CONFIG

    def test_uses_custom_retry_config(self) -> None:
        """Test that custom retry config is used."""
        custom_config = RetryConfig(max_retries=2)
        with JiraRestClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
            retry_config=custom_config,
        ) as client:
            assert client.retry_config == custom_config

    def test_retries_on_rate_limit(self) -> None:
        """Test that search_issues retries on 429."""
        mock_request = MagicMock()
        mock_429_response = MagicMock()
        mock_429_response.status_code = 429
        mock_429_response.headers = {"Retry-After": "0.01", "RateLimit-Reason": "test"}

        mock_success_response = MagicMock()
        mock_success_response.json.return_value = {"issues": [{"key": "PROJ-1"}]}
        mock_success_response.raise_for_status = MagicMock()
        mock_success_response.headers = {}

        call_count = 0

        def get_side_effect(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPStatusError(
                    "Rate limited", request=mock_request, response=mock_429_response
                )
            return mock_success_response

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.get.side_effect = get_side_effect
            mock_client_class.return_value = mock_client

            with JiraRestClient(
                base_url="https://example.atlassian.net",
                email="test@example.com",
                api_token="token123",
                retry_config=RetryConfig(
                    max_retries=1, initial_delay=0.01, jitter_min=1.0, jitter_max=1.0
                ),
            ) as client:
                issues = client.search_issues("project = PROJ")

                assert len(issues) == 1
                assert call_count == 2


class TestJiraRestTagClientRetry:
    """Tests for JiraRestTagClient retry behavior."""

    def test_uses_default_retry_config(self) -> None:
        """Test that default retry config is used."""
        with JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
        ) as client:
            assert client.retry_config == DEFAULT_RETRY_CONFIG

    def test_uses_custom_retry_config(self) -> None:
        """Test that custom retry config is used."""
        custom_config = RetryConfig(max_retries=2)
        with JiraRestTagClient(
            base_url="https://example.atlassian.net",
            email="test@example.com",
            api_token="token123",
            retry_config=custom_config,
        ) as client:
            assert client.retry_config == custom_config
