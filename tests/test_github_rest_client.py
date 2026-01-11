"""Tests for GitHub REST API client implementations."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.github_rest_client import (
    DEFAULT_GITHUB_API_URL,
    DEFAULT_RETRY_CONFIG,
    BaseGitHubHttpClient,
    GitHubRateLimitError,
    GitHubRestClient,
    GitHubRestTagClient,
    GitHubRetryConfig,
    GitHubTagClientError,
    _calculate_backoff_delay,
    _check_rate_limit_warning,
    _execute_with_retry,
    _get_retry_after,
)


class TestGitHubRetryConfig:
    """Tests for GitHubRetryConfig dataclass."""

    def test_default_values(self) -> None:
        config = GitHubRetryConfig()
        assert config.max_retries == 4
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter_min == 0.7
        assert config.jitter_max == 1.3

    def test_custom_values(self) -> None:
        config = GitHubRetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            jitter_min=0.8,
            jitter_max=1.2,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter_min == 0.8
        assert config.jitter_max == 1.2

    def test_is_frozen(self) -> None:
        config = GitHubRetryConfig()
        with pytest.raises(FrozenInstanceError):
            config.max_retries = 10  # type: ignore[misc]


class TestCalculateBackoffDelay:
    """Tests for _calculate_backoff_delay function."""

    def test_exponential_backoff(self) -> None:
        config = GitHubRetryConfig(initial_delay=1.0, jitter_min=1.0, jitter_max=1.0)
        # Without jitter, should be pure exponential
        assert _calculate_backoff_delay(0, config) == 1.0  # 1 * 2^0
        assert _calculate_backoff_delay(1, config) == 2.0  # 1 * 2^1
        assert _calculate_backoff_delay(2, config) == 4.0  # 1 * 2^2
        assert _calculate_backoff_delay(3, config) == 8.0  # 1 * 2^3

    def test_respects_max_delay(self) -> None:
        config = GitHubRetryConfig(
            initial_delay=1.0, max_delay=5.0, jitter_min=1.0, jitter_max=1.0
        )
        # 2^10 = 1024, but should be capped at max_delay
        assert _calculate_backoff_delay(10, config) == 5.0

    def test_uses_retry_after_header(self) -> None:
        config = GitHubRetryConfig(initial_delay=1.0, jitter_min=1.0, jitter_max=1.0)
        # When retry_after is provided, it should be used as base
        delay = _calculate_backoff_delay(0, config, retry_after=30.0)
        assert delay == 30.0

    def test_jitter_applied(self) -> None:
        config = GitHubRetryConfig(initial_delay=10.0, jitter_min=0.5, jitter_max=1.5)
        # Run multiple times to check jitter range
        delays = [_calculate_backoff_delay(0, config) for _ in range(100)]
        # All delays should be within jitter range
        assert all(5.0 <= d <= 15.0 for d in delays)
        # With random jitter, we should see variation (not all same)
        assert len(set(delays)) > 1


class TestCheckRateLimitWarning:
    """Tests for _check_rate_limit_warning function."""

    def test_logs_warning_when_remaining_low(self) -> None:
        response = MagicMock()
        response.headers = {"X-RateLimit-Remaining": "5", "X-RateLimit-Reset": "1234567890"}

        with patch("sentinel.github_rest_client.logger") as mock_logger:
            _check_rate_limit_warning(response)
            mock_logger.warning.assert_called_once()
            assert "5" in str(mock_logger.warning.call_args)

    def test_no_warning_when_remaining_high(self) -> None:
        response = MagicMock()
        response.headers = {"X-RateLimit-Remaining": "100"}

        with patch("sentinel.github_rest_client.logger") as mock_logger:
            _check_rate_limit_warning(response)
            mock_logger.warning.assert_not_called()

    def test_handles_missing_header(self) -> None:
        response = MagicMock()
        response.headers = {}

        # Should not raise
        _check_rate_limit_warning(response)

    def test_handles_invalid_remaining_value(self) -> None:
        response = MagicMock()
        response.headers = {"X-RateLimit-Remaining": "not-a-number"}

        # Should not raise
        _check_rate_limit_warning(response)


class TestGetRetryAfter:
    """Tests for _get_retry_after function."""

    def test_uses_retry_after_header(self) -> None:
        response = MagicMock()
        response.headers = {"Retry-After": "30"}

        result = _get_retry_after(response)
        assert result == 30.0

    def test_uses_ratelimit_reset_header(self) -> None:
        response = MagicMock()
        current_time = 1000
        reset_time = 1060  # 60 seconds from now
        response.headers = {"X-RateLimit-Reset": str(reset_time)}

        with patch("sentinel.github_rest_client.time.time", return_value=current_time):
            result = _get_retry_after(response)
            assert result == 60.0

    def test_retry_after_takes_precedence(self) -> None:
        response = MagicMock()
        response.headers = {
            "Retry-After": "30",
            "X-RateLimit-Reset": "9999999999",
        }

        result = _get_retry_after(response)
        assert result == 30.0

    def test_returns_none_when_no_headers(self) -> None:
        response = MagicMock()
        response.headers = {}

        result = _get_retry_after(response)
        assert result is None

    def test_returns_none_for_past_reset_time(self) -> None:
        response = MagicMock()
        current_time = 1000
        reset_time = 500  # In the past
        response.headers = {"X-RateLimit-Reset": str(reset_time)}

        with patch("sentinel.github_rest_client.time.time", return_value=current_time):
            result = _get_retry_after(response)
            assert result is None

    def test_handles_invalid_retry_after(self) -> None:
        response = MagicMock()
        response.headers = {"Retry-After": "not-a-number"}

        with patch("sentinel.github_rest_client.logger"):
            result = _get_retry_after(response)
            assert result is None


class TestExecuteWithRetry:
    """Tests for _execute_with_retry function."""

    def test_success_on_first_attempt(self) -> None:
        operation = MagicMock(return_value="success")
        result = _execute_with_retry(operation)
        assert result == "success"
        assert operation.call_count == 1

    def test_retries_on_rate_limit_403(self) -> None:
        response = MagicMock()
        response.status_code = 403
        response.headers = {"X-RateLimit-Remaining": "0"}

        error = httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=response)

        call_count = 0

        def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise error
            return "success"

        config = GitHubRetryConfig(max_retries=3, initial_delay=0.01)
        result = _execute_with_retry(operation, config)
        assert result == "success"
        assert call_count == 3

    def test_retries_on_429(self) -> None:
        response = MagicMock()
        response.status_code = 429
        response.headers = {}

        error = httpx.HTTPStatusError("Too many requests", request=MagicMock(), response=response)

        call_count = 0

        def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise error
            return "success"

        config = GitHubRetryConfig(max_retries=3, initial_delay=0.01)
        result = _execute_with_retry(operation, config)
        assert result == "success"
        assert call_count == 2

    def test_no_retry_on_403_with_remaining_quota(self) -> None:
        response = MagicMock()
        response.status_code = 403
        response.headers = {"X-RateLimit-Remaining": "100"}  # Has quota, so permission error

        error = httpx.HTTPStatusError("Forbidden", request=MagicMock(), response=response)

        operation = MagicMock(side_effect=error)
        config = GitHubRetryConfig(max_retries=3, initial_delay=0.01)

        with pytest.raises(httpx.HTTPStatusError):
            _execute_with_retry(operation, config)

        assert operation.call_count == 1

    def test_raises_after_max_retries(self) -> None:
        response = MagicMock()
        response.status_code = 429
        response.headers = {}

        error = httpx.HTTPStatusError("Too many requests", request=MagicMock(), response=response)
        operation = MagicMock(side_effect=error)

        config = GitHubRetryConfig(max_retries=2, initial_delay=0.01)

        with pytest.raises(GitHubRateLimitError, match="Rate limit exceeded"):
            _execute_with_retry(operation, config)

        assert operation.call_count == 3  # Initial + 2 retries


class TestBaseGitHubHttpClient:
    """Tests for BaseGitHubHttpClient base class."""

    def test_lazy_client_initialization(self) -> None:
        """Test that HTTP client is lazily initialized."""

        class TestClient(BaseGitHubHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self._headers = {"Test": "Header"}

        client = TestClient()
        assert client._client is None

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_get_client_creates_client(self, mock_client_class: MagicMock) -> None:
        """Test that _get_client creates the httpx.Client instance."""

        class TestClient(BaseGitHubHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self._headers = {"Test": "Header"}

        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = TestClient()
        result = client._get_client()

        assert result == mock_http_client
        mock_client_class.assert_called_once()

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_get_client_reuses_client(self, mock_client_class: MagicMock) -> None:
        """Test that _get_client reuses the same client instance."""

        class TestClient(BaseGitHubHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self._headers = {"Test": "Header"}

        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = TestClient()
        client._get_client()
        client._get_client()

        # Should only create one client
        assert mock_client_class.call_count == 1

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_close_closes_client(self, mock_client_class: MagicMock) -> None:
        """Test that close() properly closes the HTTP client."""

        class TestClient(BaseGitHubHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self._headers = {"Test": "Header"}

        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = TestClient()
        client._get_client()  # Create the client
        client.close()

        mock_http_client.close.assert_called_once()
        assert client._client is None

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_context_manager(self, mock_client_class: MagicMock) -> None:
        """Test context manager support."""

        class TestClient(BaseGitHubHttpClient):
            def __init__(self) -> None:
                super().__init__()
                self.timeout = httpx.Timeout(10.0)
                self._headers = {"Test": "Header"}

        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        with TestClient() as client:
            client._get_client()

        mock_http_client.close.assert_called_once()


class TestGitHubRestClient:
    """Tests for GitHubRestClient class."""

    def test_init_with_defaults(self) -> None:
        client = GitHubRestClient(token="test-token")
        assert client.base_url == DEFAULT_GITHUB_API_URL
        assert client.token == "test-token"
        assert client.retry_config == DEFAULT_RETRY_CONFIG

    def test_init_with_custom_base_url(self) -> None:
        client = GitHubRestClient(
            token="test-token", base_url="https://github.example.com/api/v3"
        )
        assert client.base_url == "https://github.example.com/api/v3"

    def test_init_strips_trailing_slash(self) -> None:
        client = GitHubRestClient(
            token="test-token", base_url="https://api.github.com/"
        )
        assert client.base_url == "https://api.github.com"

    def test_headers_set_correctly(self) -> None:
        client = GitHubRestClient(token="test-token")
        assert client._headers["Authorization"] == "Bearer test-token"
        assert client._headers["Accept"] == "application/vnd.github+json"
        assert "X-GitHub-Api-Version" in client._headers

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_search_issues_success(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"X-RateLimit-Remaining": "100"}
        mock_response.json.return_value = {
            "items": [
                {"number": 1, "title": "Issue 1"},
                {"number": 2, "title": "Issue 2"},
            ],
            "total_count": 2,
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        client = GitHubRestClient(token="test-token")
        results = client.search_issues("repo:org/repo label:bug")

        assert len(results) == 2
        assert results[0]["number"] == 1
        assert results[1]["number"] == 2

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_search_issues_respects_max_results(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"items": [], "total_count": 0}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        client = GitHubRestClient(token="test-token")
        client.search_issues("query", max_results=25)

        # Check that per_page was set correctly
        call_args = mock_http_client.get.call_args
        assert call_args[1]["params"]["per_page"] == 25

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_search_issues_caps_max_results_at_100(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"items": [], "total_count": 0}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.get.return_value = mock_response

        client = GitHubRestClient(token="test-token")
        client.search_issues("query", max_results=500)

        # GitHub max is 100 per page
        call_args = mock_http_client.get.call_args
        assert call_args[1]["params"]["per_page"] == 100


class TestGitHubRestTagClient:
    """Tests for GitHubRestTagClient class."""

    def test_init_with_defaults(self) -> None:
        client = GitHubRestTagClient(token="test-token")
        assert client.base_url == DEFAULT_GITHUB_API_URL
        assert client.token == "test-token"

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_add_label_success(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        client = GitHubRestTagClient(token="test-token")
        client.add_label("owner", "repo", 123, "bug")

        # Verify the API was called correctly
        call_args = mock_http_client.post.call_args
        assert "owner/repo/issues/123/labels" in call_args[0][0]
        assert call_args[1]["json"] == {"labels": ["bug"]}

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_remove_label_success(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_http_client.delete.return_value = mock_response

        client = GitHubRestTagClient(token="test-token")
        client.remove_label("owner", "repo", 123, "bug")

        # Verify the API was called correctly
        call_args = mock_http_client.delete.call_args
        assert "owner/repo/issues/123/labels/" in call_args[0][0]

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_remove_label_handles_404(self, mock_client_class: MagicMock) -> None:
        """Remove should not error if label already gone (404)."""
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_http_client.delete.return_value = mock_response

        client = GitHubRestTagClient(token="test-token")
        # Should not raise
        client.remove_label("owner", "repo", 123, "nonexistent-label")

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_add_label_raises_on_error(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.return_value = {"message": "Internal server error"}
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )
        mock_http_client.post.return_value = mock_response

        client = GitHubRestTagClient(token="test-token")
        with pytest.raises(GitHubTagClientError, match="status 500"):
            client.add_label("owner", "repo", 123, "bug")

    @patch("sentinel.github_rest_client.httpx.Client")
    def test_add_label_handles_timeout(self, mock_client_class: MagicMock) -> None:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        mock_http_client.post.side_effect = httpx.TimeoutException("Timeout")

        client = GitHubRestTagClient(token="test-token")
        with pytest.raises(GitHubTagClientError, match="timed out"):
            client.add_label("owner", "repo", 123, "bug")
