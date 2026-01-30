"""Integration tests for GitHub API interactions.

These tests verify the actual behavior of the GitHub REST client against
the real GitHub API. They require a valid GitHub token to be configured
in the environment.

Environment Variables Required:
- GITHUB_TOKEN: Personal access token or app token for authentication
- GITHUB_TEST_REPO: Optional repository for testing (format: "owner/repo")
  If not set, tests will use a public repository for read-only tests.

Run with: pytest tests/integration -m integration
Skip with: pytest -m "not integration"
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator

import httpx
import pytest

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from sentinel.github_poller import GitHubClientError
from sentinel.github_rest_client import (
    GitHubRestClient,
    GitHubRestTagClient,
    GitHubRetryConfig,
    GitHubTagClientError,
)


def _get_github_token() -> str | None:
    """Get GitHub token from environment variables."""
    return os.getenv("GITHUB_TOKEN")


def _github_token_available() -> bool:
    """Check if GitHub token is available."""
    return _get_github_token() is not None


def _get_test_repo() -> tuple[str, str] | None:
    """Get test repository from environment.

    Returns:
        Tuple of (owner, repo) if GITHUB_TEST_REPO is set, None otherwise.
    """
    repo = os.getenv("GITHUB_TEST_REPO")
    if repo and "/" in repo:
        owner, name = repo.split("/", 1)
        return owner, name
    return None


@pytest.fixture
def github_token() -> str:
    """Fixture to provide GitHub token.

    Skips the test if token is not available.
    """
    token = _get_github_token()
    if token is None:
        pytest.skip("GITHUB_TOKEN not available in environment")
    return token


@pytest.fixture
def github_client(github_token: str) -> Generator[GitHubRestClient, None, None]:
    """Create a GitHubRestClient instance with environment credentials."""
    # Use a fresh circuit breaker for each test
    circuit_breaker = CircuitBreaker(
        service_name="github_test",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=2,
        ),
    )
    client = GitHubRestClient(
        token=github_token,
        circuit_breaker=circuit_breaker,
    )
    yield client
    # Cleanup: close the HTTP client
    client.close()


@pytest.fixture
def github_tag_client(github_token: str) -> Generator[GitHubRestTagClient, None, None]:
    """Create a GitHubRestTagClient instance with environment credentials."""
    circuit_breaker = CircuitBreaker(
        service_name="github_tag_test",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0,
            half_open_max_calls=2,
        ),
    )
    client = GitHubRestTagClient(
        token=github_token,
        circuit_breaker=circuit_breaker,
    )
    yield client
    client.close()


class TestGitHubRestClientIntegration:
    """Integration tests for GitHubRestClient against real GitHub API."""

    @pytest.mark.integration
    def test_search_issues_returns_results(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should successfully search for issues using GitHub search syntax.

        This test searches for open issues in a well-known public repository.
        """
        # Search for issues in a popular public repository
        results = github_client.search_issues(
            "repo:microsoft/vscode is:issue is:open",
            max_results=10,
        )

        # Verify structure of results
        assert isinstance(results, list)
        # VSCode repo should have issues
        assert len(results) > 0

        # Verify issue structure
        issue = results[0]
        assert "number" in issue
        assert "title" in issue
        assert "html_url" in issue
        assert "state" in issue

    @pytest.mark.integration
    def test_search_issues_empty_results(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should handle searches that return no results."""
        # Search for something that won't exist
        results = github_client.search_issues(
            "repo:microsoft/vscode nonexistent-random-string-xyz123",
            max_results=10,
        )

        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.integration
    def test_search_issues_with_max_results_limit(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should respect max_results parameter."""
        results = github_client.search_issues(
            "repo:microsoft/vscode is:issue",
            max_results=5,
        )

        assert len(results) <= 5

    @pytest.mark.integration
    def test_search_pull_requests(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should search for pull requests using is:pr filter."""
        results = github_client.search_issues(
            "repo:microsoft/vscode is:pr is:open",
            max_results=5,
        )

        assert isinstance(results, list)
        if results:
            pr = results[0]
            assert "pull_request" in pr  # PRs have this field in search results

    @pytest.mark.integration
    def test_circuit_breaker_records_success(
        self, github_client: GitHubRestClient
    ) -> None:
        """Circuit breaker should record successful calls."""
        github_client.search_issues("repo:microsoft/vscode is:issue", max_results=1)

        assert github_client.circuit_breaker.state == CircuitState.CLOSED
        assert github_client.circuit_breaker.metrics.successful_calls > 0


class TestGitHubProjectsIntegration:
    """Integration tests for GitHub Projects (v2) GraphQL operations.

    Note: These tests require access to a GitHub organization or user
    with at least one project. They are skipped if no test repo is configured.
    """

    @pytest.mark.integration
    def test_get_project_with_invalid_owner(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should raise error for non-existent organization."""
        with pytest.raises(GitHubClientError) as exc_info:
            github_client.get_project(
                owner="nonexistent-org-xyz123",
                project_number=1,
                scope="organization",
            )

        # Error message may vary: "not found" or "could not resolve"
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "could not resolve" in error_msg

    @pytest.mark.integration
    def test_get_project_with_invalid_scope(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should raise error for invalid scope parameter."""
        with pytest.raises(GitHubClientError) as exc_info:
            github_client.get_project(
                owner="test",
                project_number=1,
                scope="invalid",
            )

        assert "invalid scope" in str(exc_info.value).lower()


class TestGitHubApiConnectionHandling:
    """Integration tests for connection and timeout handling."""

    @pytest.mark.integration
    def test_client_handles_custom_timeout(
        self, github_token: str
    ) -> None:
        """Should successfully use custom timeout configuration."""
        custom_timeout = httpx.Timeout(5.0, read=15.0)
        client = GitHubRestClient(
            token=github_token,
            timeout=custom_timeout,
        )

        try:
            results = client.search_issues("repo:microsoft/vscode is:issue", max_results=1)
            assert isinstance(results, list)
        finally:
            client.close()

    @pytest.mark.integration
    def test_client_handles_retry_config(
        self, github_token: str
    ) -> None:
        """Should successfully use custom retry configuration."""
        custom_retry = GitHubRetryConfig(
            max_retries=2,
            initial_delay=0.5,
            max_delay=5.0,
        )
        client = GitHubRestClient(
            token=github_token,
            retry_config=custom_retry,
        )

        try:
            results = client.search_issues("repo:microsoft/vscode is:issue", max_results=1)
            assert isinstance(results, list)
        finally:
            client.close()

    @pytest.mark.integration
    def test_client_context_manager(
        self, github_token: str
    ) -> None:
        """Should work correctly as context manager."""
        with GitHubRestClient(token=github_token) as client:
            results = client.search_issues("repo:microsoft/vscode is:issue", max_results=1)
            assert isinstance(results, list)

        # After context exit, client should be closed
        assert client._client is None


class TestGitHubRestTagClientIntegration:
    """Integration tests for GitHubRestTagClient label operations.

    Note: Tests that modify labels require GITHUB_TEST_REPO to be set
    and the token to have write permissions on that repository.
    Read-only tests can run against any public repository.
    """

    @pytest.mark.integration
    def test_circuit_breaker_tracks_operations(
        self, github_tag_client: GitHubRestTagClient
    ) -> None:
        """Circuit breaker should track tag client operations (including failures)."""
        # Tag client only has write operations (add_label, remove_label)
        # Without write permission, these will fail with 403
        # But the circuit breaker should still track these attempts
        initial_failures = github_tag_client.circuit_breaker.metrics.failed_calls

        with pytest.raises(GitHubTagClientError):
            github_tag_client.add_label(
                owner="microsoft",
                repo="vscode",
                issue_number=1,
                label="test-label-xyz123",
            )

        # Circuit breaker should have recorded the failure
        assert github_tag_client.circuit_breaker.metrics.failed_calls > initial_failures
        # But circuit should still be closed (single failure doesn't trip it)
        assert github_tag_client.circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.integration
    def test_add_label_requires_write_permission(
        self, github_tag_client: GitHubRestTagClient
    ) -> None:
        """Should fail gracefully when lacking write permission."""
        # Attempt to add a label to a repo we don't have write access to
        with pytest.raises(GitHubTagClientError) as exc_info:
            github_tag_client.add_label(
                owner="microsoft",
                repo="vscode",
                issue_number=1,
                label="test-label",
            )

        # Should fail with 403 (forbidden) or 404 (not found/hidden)
        error_str = str(exc_info.value)
        assert "403" in error_str or "404" in error_str or "failed" in error_str.lower()


class TestGitHubApiPerformance:
    """Integration tests for GitHub API performance characteristics."""

    @pytest.mark.integration
    def test_search_response_time(
        self, github_client: GitHubRestClient
    ) -> None:
        """Search should complete within reasonable time frame."""
        start_time = time.perf_counter()

        github_client.search_issues("repo:microsoft/vscode is:issue", max_results=10)

        elapsed = time.perf_counter() - start_time

        # Request should complete within 30 seconds under normal conditions
        assert elapsed < 30.0, f"Search took too long: {elapsed:.2f}s"

    @pytest.mark.integration
    def test_multiple_sequential_requests(
        self, github_client: GitHubRestClient
    ) -> None:
        """Should handle multiple sequential requests without issues."""
        successful_calls = 0

        for _ in range(3):
            try:
                github_client.search_issues("repo:microsoft/vscode is:issue", max_results=1)
                successful_calls += 1
            except (httpx.HTTPError, GitHubClientError):
                # Handle HTTP-related errors (network issues, status errors)
                # and GitHub-specific client errors (rate limits, API errors)
                pass
            except OSError:
                # Handle OS-level network errors (connection refused, DNS failure)
                pass

        assert successful_calls >= 2, f"Only {successful_calls}/3 calls succeeded"

    @pytest.mark.integration
    def test_connection_pooling_performance(
        self, github_token: str
    ) -> None:
        """Connection pooling should improve performance of multiple requests."""
        with GitHubRestClient(token=github_token) as client:
            times: list[float] = []

            for _ in range(3):
                start = time.perf_counter()
                client.search_issues("repo:microsoft/vscode is:issue", max_results=1)
                times.append(time.perf_counter() - start)

            # Subsequent requests should benefit from connection reuse
            # (first request may be slower due to connection setup)
            # We don't assert specific times, just that all completed
            assert len(times) == 3
            assert all(t < 30.0 for t in times)
