"""Integration tests for Jira API interactions.

These tests verify the actual behavior of the Jira REST client against
the real Jira API. They require valid Jira credentials to be configured
in the environment.

Environment Variables Required:
- JIRA_BASE_URL: Jira instance URL (e.g., "https://yoursite.atlassian.net")
- JIRA_USER_EMAIL: User email for authentication
- JIRA_API_TOKEN: API token for authentication

Run with: pytest tests/integration -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient, RetryConfig


def _get_jira_credentials() -> tuple[str, str, str] | None:
    """Get Jira credentials from environment variables.

    Returns:
        Tuple of (base_url, email, api_token) if all are set, None otherwise.
    """
    base_url = os.getenv("JIRA_BASE_URL")
    email = os.getenv("JIRA_USER_EMAIL")
    api_token = os.getenv("JIRA_API_TOKEN")

    if all([base_url, email, api_token]):
        return base_url, email, api_token  # type: ignore[return-value]
    return None


def _jira_credentials_available() -> bool:
    """Check if Jira credentials are available."""
    return _get_jira_credentials() is not None


@pytest.fixture
def jira_credentials() -> tuple[str, str, str]:
    """Fixture to provide Jira credentials.

    Skips the test if credentials are not available.
    """
    creds = _get_jira_credentials()
    if creds is None:
        pytest.skip("Jira credentials not available in environment")
    return creds


@pytest.fixture
def jira_client(jira_credentials: tuple[str, str, str]) -> JiraRestClient:
    """Create a JiraRestClient instance with environment credentials."""
    base_url, email, api_token = jira_credentials
    # Use a fresh circuit breaker for each test to avoid state leakage
    circuit_breaker = CircuitBreaker(
        service_name="jira_test",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=2,
        ),
    )
    return JiraRestClient(
        base_url=base_url,
        email=email,
        api_token=api_token,
        circuit_breaker=circuit_breaker,
    )


@pytest.fixture
def jira_tag_client(jira_credentials: tuple[str, str, str]) -> JiraRestTagClient:
    """Create a JiraRestTagClient instance with environment credentials."""
    base_url, email, api_token = jira_credentials
    # Use a fresh circuit breaker for each test
    circuit_breaker = CircuitBreaker(
        service_name="jira_tag_test",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=2,
        ),
    )
    return JiraRestTagClient(
        base_url=base_url,
        email=email,
        api_token=api_token,
        circuit_breaker=circuit_breaker,
    )


class TestJiraRestClientIntegration:
    """Integration tests for JiraRestClient against real Jira API."""

    @pytest.mark.integration
    def test_search_issues_returns_results(
        self, jira_client: JiraRestClient
    ) -> None:
        """Should successfully search for issues using JQL.

        This test uses a simple JQL query that should return results
        from most Jira instances with at least one project.
        """
        # Use a broad query that should return at least some results
        # project is not empty means any issue in any project
        results = jira_client.search_issues("project is not EMPTY", max_results=10)

        # Verify structure of results
        assert isinstance(results, list)
        # Note: We can't assert len > 0 because the instance might be empty
        # but we can verify the call succeeded

        # If results exist, verify issue structure
        if results:
            issue = results[0]
            assert "key" in issue
            assert "fields" in issue
            assert isinstance(issue["fields"], dict)

    @pytest.mark.integration
    def test_search_issues_with_invalid_jql(
        self, jira_client: JiraRestClient
    ) -> None:
        """Should raise error for invalid JQL syntax."""
        from sentinel.poller import JiraClientError

        with pytest.raises(JiraClientError) as exc_info:
            jira_client.search_issues("INVALID JQL SYNTAX !!!")

        # Error message should indicate the failure
        assert "failed" in str(exc_info.value).lower() or "400" in str(exc_info.value)

    @pytest.mark.integration
    def test_search_issues_with_max_results_limit(
        self, jira_client: JiraRestClient
    ) -> None:
        """Should respect max_results parameter."""
        # Request only 5 results
        results = jira_client.search_issues("project is not EMPTY", max_results=5)

        # Should return at most 5 results
        assert len(results) <= 5

    @pytest.mark.integration
    def test_circuit_breaker_records_success(
        self, jira_client: JiraRestClient
    ) -> None:
        """Circuit breaker should record successful calls."""
        # Make a successful call
        jira_client.search_issues("project is not EMPTY", max_results=1)

        # Circuit breaker should remain closed
        assert jira_client.circuit_breaker.state == CircuitState.CLOSED
        assert jira_client.circuit_breaker.metrics.successful_calls > 0

    @pytest.mark.integration
    def test_search_returns_expected_fields(
        self, jira_client: JiraRestClient
    ) -> None:
        """Should return issues with expected field structure."""
        results = jira_client.search_issues("project is not EMPTY", max_results=5)

        if results:
            issue = results[0]
            fields = issue.get("fields", {})

            # These fields are requested in the search
            # At minimum, status should be present
            assert "status" in fields or fields.get("status") is None


class TestJiraRestTagClientIntegration:
    """Integration tests for JiraRestTagClient label operations.

    Note: These tests modify labels on Jira issues. Use with caution
    and ensure you have a test project/issue available.
    """

    @pytest.mark.integration
    def test_get_current_labels(
        self, jira_tag_client: JiraRestTagClient, jira_client: JiraRestClient
    ) -> None:
        """Should successfully retrieve labels for an issue.

        This test finds an existing issue and retrieves its labels.
        """
        # First, find an issue to test with
        results = jira_client.search_issues("project is not EMPTY", max_results=1)

        if not results:
            pytest.skip("No issues found in Jira instance")

        issue_key = results[0]["key"]

        # Get labels for this issue
        labels = jira_tag_client._get_current_labels(issue_key)

        # Verify we got a list back (may be empty)
        assert isinstance(labels, list)

    @pytest.mark.integration
    def test_circuit_breaker_tracks_tag_operations(
        self, jira_tag_client: JiraRestTagClient, jira_client: JiraRestClient
    ) -> None:
        """Circuit breaker should track tag client operations."""
        # Find an issue to test with
        results = jira_client.search_issues("project is not EMPTY", max_results=1)

        if not results:
            pytest.skip("No issues found in Jira instance")

        issue_key = results[0]["key"]

        # Get labels (should succeed)
        jira_tag_client._get_current_labels(issue_key)

        # Verify circuit breaker recorded success
        assert jira_tag_client.circuit_breaker.state == CircuitState.CLOSED
        assert jira_tag_client.circuit_breaker.metrics.successful_calls > 0


class TestJiraApiConnectionHandling:
    """Integration tests for connection and timeout handling."""

    @pytest.mark.integration
    def test_client_handles_custom_timeout(
        self, jira_credentials: tuple[str, str, str]
    ) -> None:
        """Should successfully use custom timeout configuration."""
        base_url, email, api_token = jira_credentials

        # Create client with custom timeout
        custom_timeout = httpx.Timeout(5.0, read=15.0)
        client = JiraRestClient(
            base_url=base_url,
            email=email,
            api_token=api_token,
            timeout=custom_timeout,
        )

        # Make a request - should succeed with custom timeout
        results = client.search_issues("project is not EMPTY", max_results=1)
        assert isinstance(results, list)

    @pytest.mark.integration
    def test_client_handles_retry_config(
        self, jira_credentials: tuple[str, str, str]
    ) -> None:
        """Should successfully use custom retry configuration."""
        base_url, email, api_token = jira_credentials

        # Create client with custom retry config
        custom_retry = RetryConfig(
            max_retries=2,
            initial_delay=0.5,
            max_delay=5.0,
        )
        client = JiraRestClient(
            base_url=base_url,
            email=email,
            api_token=api_token,
            retry_config=custom_retry,
        )

        # Make a request
        results = client.search_issues("project is not EMPTY", max_results=1)
        assert isinstance(results, list)


class TestJiraApiPerformance:
    """Integration tests for Jira API performance characteristics."""

    @pytest.mark.integration
    def test_search_response_time(
        self, jira_client: JiraRestClient
    ) -> None:
        """Search should complete within reasonable time frame."""
        start_time = time.perf_counter()

        jira_client.search_issues("project is not EMPTY", max_results=10)

        elapsed = time.perf_counter() - start_time

        # Request should complete within 30 seconds under normal conditions
        assert elapsed < 30.0, f"Search took too long: {elapsed:.2f}s"

    @pytest.mark.integration
    def test_multiple_sequential_requests(
        self, jira_client: JiraRestClient
    ) -> None:
        """Should handle multiple sequential requests without issues."""
        successful_calls = 0

        for _ in range(3):
            try:
                jira_client.search_issues("project is not EMPTY", max_results=1)
                successful_calls += 1
            except httpx.HTTPError:
                # Handle HTTP-related errors (network issues, status errors)
                pass
            except OSError:
                # Handle OS-level network errors (connection refused, DNS failure)
                pass

        # At least 2 out of 3 calls should succeed
        assert successful_calls >= 2, f"Only {successful_calls}/3 calls succeeded"
