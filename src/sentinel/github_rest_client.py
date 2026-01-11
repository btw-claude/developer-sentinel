"""REST-based client implementations for GitHub API.

These clients use direct HTTP calls to the GitHub REST API for fast,
cost-effective polling and label operations without Claude invocations.

Implements exponential backoff with jitter for rate limiting per:
https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, TypeVar

import httpx

from sentinel.github_poller import GitHubClient, GitHubClientError
from sentinel.logging import get_logger

logger = get_logger(__name__)

# Default timeout for HTTP requests (connect, read, write, pool)
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)

# GitHub API default base URL
DEFAULT_GITHUB_API_URL = "https://api.github.com"

# Type variable for generic retry function
T = TypeVar("T")


@dataclass(frozen=True)
class GitHubRetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 4).
        initial_delay: Initial delay in seconds before first retry (default: 1.0).
        max_delay: Maximum delay in seconds between retries (default: 60.0).
        jitter_min: Minimum jitter multiplier (default: 0.7).
        jitter_max: Maximum jitter multiplier (default: 1.3).
    """

    max_retries: int = 4
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter_min: float = 0.7
    jitter_max: float = 1.3


# Default retry configuration per GitHub recommendations
DEFAULT_RETRY_CONFIG = GitHubRetryConfig()


class GitHubRateLimitError(GitHubClientError):
    """Raised when rate limit is exceeded and all retries are exhausted.

    This exception is raised by _execute_with_retry when all retry attempts
    are exhausted due to rate limiting (HTTP 403 with rate limit headers or
    HTTP 429). It indicates the caller should wait before making additional
    requests to the GitHub API.
    """

    pass


class GitHubTagClientError(Exception):
    """Raised when a GitHub tag/label operation fails."""

    pass


def _calculate_backoff_delay(
    attempt: int,
    config: GitHubRetryConfig,
    retry_after: float | None = None,
) -> float:
    """Calculate the delay before the next retry attempt.

    Args:
        attempt: Current retry attempt number (0-indexed).
        config: Retry configuration.
        retry_after: Optional Retry-After header value in seconds.

    Returns:
        Delay in seconds before next retry.
    """
    # Use Retry-After header if provided
    if retry_after is not None:
        base_delay = retry_after
    else:
        # Exponential backoff: initial_delay * 2^attempt
        base_delay = min(config.initial_delay * (2**attempt), config.max_delay)

    # Apply jitter
    jitter = random.uniform(config.jitter_min, config.jitter_max)
    return base_delay * jitter


def _check_rate_limit_warning(response: httpx.Response) -> None:
    """Log a warning if rate limit is near exhaustion.

    Args:
        response: HTTP response to check for rate limit headers.
    """
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        try:
            remaining_int = int(remaining)
            if remaining_int <= 10:
                reset_time = response.headers.get("X-RateLimit-Reset", "unknown")
                logger.warning(
                    f"GitHub rate limit near exhaustion. Remaining: {remaining}, Reset: {reset_time}"
                )
        except ValueError:
            pass


def _get_retry_after(response: httpx.Response) -> float | None:
    """Extract Retry-After value from response headers.

    GitHub uses either Retry-After header or x-ratelimit-reset timestamp.

    Args:
        response: HTTP response to check.

    Returns:
        Retry-After value in seconds, or None if not present.
    """
    # Check standard Retry-After header first
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            logger.warning(f"Invalid Retry-After header value: {retry_after}")

    # Check GitHub-specific x-ratelimit-reset (Unix timestamp)
    reset_time = response.headers.get("X-RateLimit-Reset")
    if reset_time is not None:
        try:
            reset_timestamp = int(reset_time)
            delay = reset_timestamp - int(time.time())
            if delay > 0:
                return float(delay)
        except ValueError:
            logger.warning(f"Invalid X-RateLimit-Reset header value: {reset_time}")

    return None


def _execute_with_retry(
    operation: Callable[[], T],
    config: GitHubRetryConfig = DEFAULT_RETRY_CONFIG,
    error_class: type[Exception] = Exception,
) -> T:
    """Execute an operation with retry logic for rate limiting.

    Args:
        operation: Callable that performs the HTTP operation and returns a result.
                  Should raise httpx.HTTPStatusError on rate limit responses.
        config: Retry configuration.
        error_class: Exception class to raise on final failure.

    Returns:
        Result from the operation.

    Raises:
        error_class: If all retries are exhausted or a non-retryable error occurs.
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return operation()
        except httpx.HTTPStatusError as e:
            # GitHub uses 403 for rate limiting (primary and secondary)
            # and 429 for abuse detection
            if e.response.status_code not in (403, 429):
                # Not a rate limit error, don't retry
                raise

            # Check if it's actually a rate limit (403 can have other meanings)
            if e.response.status_code == 403:
                # Check for rate limit headers
                remaining = e.response.headers.get("X-RateLimit-Remaining")
                if remaining is not None:
                    try:
                        if int(remaining) > 0:
                            # Not rate limited, it's a permission error
                            raise
                    except ValueError:
                        pass

            last_exception = e

            if attempt >= config.max_retries:
                # All retries exhausted - raise GitHubRateLimitError specifically
                # for rate limit failures to distinguish from other errors
                raise GitHubRateLimitError(
                    f"Rate limit exceeded after {config.max_retries} retries"
                ) from e

            # Calculate delay and wait
            retry_after = _get_retry_after(e.response)
            delay = _calculate_backoff_delay(attempt, config, retry_after)

            logger.warning(
                f"Rate limited (attempt {attempt + 1}/{config.max_retries + 1}). "
                f"Retrying in {delay:.2f}s"
            )

            time.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise error_class(f"Retry failed: {last_exception}") from last_exception
    raise error_class("Retry failed with no exception")


class BaseGitHubHttpClient:
    """Base class providing HTTP client connection pooling for GitHub API clients.

    This class encapsulates the shared connection pooling functionality including:
    - Lazy initialization of httpx.Client
    - Resource cleanup via close() method
    - Context manager support (__enter__/__exit__)

    Subclasses must set self.timeout and self._headers before using _get_client().

    Example subclass implementation::

        class MyGitHubClient(BaseGitHubHttpClient):
            def __init__(self, token: str, timeout: httpx.Timeout | None = None) -> None:
                super().__init__()
                self.timeout = timeout or DEFAULT_TIMEOUT
                self._headers = {
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                }

            def my_api_method(self) -> dict:
                client = self._get_client()
                response = client.get("https://api.github.com/...")
                return response.json()
    """

    # Instance attribute type stubs - subclasses MUST set these in __init__
    # before calling _get_client(). These are declared here to support static
    # type checking and IDE autocompletion for the abstract contract.
    timeout: httpx.Timeout
    _headers: dict[str, str]

    def __init__(self) -> None:
        """Initialize the base HTTP client.

        Note: Subclasses must call super().__init__() and then set
        self.timeout and self._headers before using _get_client().
        """
        # Reusable HTTP client for connection pooling - lazily initialized
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the reusable HTTP client.

        Returns:
            A configured httpx.Client instance with connection pooling.

        Raises:
            RuntimeError: If subclass has not set required timeout or _headers attributes.
        """
        if self._client is None:
            # Runtime safety check: ensure subclass has properly initialized required attributes
            if not hasattr(self, "timeout") or self.timeout is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must set self.timeout before calling _get_client(). "
                    "See BaseGitHubHttpClient docstring for implementation example."
                )
            if not hasattr(self, "_headers") or self._headers is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must set self._headers before calling _get_client(). "
                    "See BaseGitHubHttpClient docstring for implementation example."
                )
            self._client = httpx.Client(timeout=self.timeout, headers=self._headers)
        return self._client

    def close(self) -> None:
        """Close the HTTP client and release resources.

        Should be called when the client is no longer needed.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - closes the HTTP client."""
        self.close()


class GitHubRestClient(BaseGitHubHttpClient, GitHubClient):
    """GitHub client that uses direct REST API calls for searching issues.

    Supports both GitHub.com and GitHub Enterprise via configurable base URL.
    Uses connection pooling via a reusable httpx.Client for better performance
    in high-volume scenarios.
    """

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
        retry_config: GitHubRetryConfig | None = None,
    ) -> None:
        """Initialize the GitHub REST client.

        Args:
            token: GitHub personal access token or app token.
            base_url: Optional custom API base URL for GitHub Enterprise.
                     Defaults to "https://api.github.com".
                     For GitHub Enterprise: "https://your-ghe-host/api/v3"
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
        """
        super().__init__()
        self.base_url = (base_url or DEFAULT_GITHUB_API_URL).rstrip("/")
        self.token = token
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def search_issues(
        self, query: str, max_results: int = 50
    ) -> list[dict[str, Any]]:
        """Search for issues and pull requests using GitHub search syntax.

        Args:
            query: GitHub search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue/PR data from GitHub API.

        Raises:
            GitHubClientError: If the search fails or rate limit is exhausted.
        """
        url = f"{self.base_url}/search/issues"
        params: dict[str, str | int] = {
            "q": query,
            "per_page": min(max_results, 100),  # GitHub max is 100 per page
        }

        logger.debug(f"Searching GitHub: {query}")

        def do_search() -> list[dict[str, Any]]:
            client = self._get_client()
            response = client.get(url, params=params)
            _check_rate_limit_warning(response)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = data.get("items", [])
            total_count = data.get("total_count", 0)
            logger.info(
                f"GitHub search returned {len(items)} items (total: {total_count})"
            )
            return items

        try:
            return _execute_with_retry(do_search, self.retry_config, GitHubClientError)
        except httpx.TimeoutException as e:
            raise GitHubClientError(f"GitHub search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_msg = f"GitHub search failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except Exception:
                pass
            raise GitHubClientError(error_msg) from e
        except httpx.RequestError as e:
            raise GitHubClientError(f"GitHub search request failed: {e}") from e


class GitHubTagClient(ABC):
    """Abstract interface for GitHub label operations.

    This allows the tag manager to work with different implementations:
    - Real GitHub API client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def add_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Add a label to a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to add.

        Raises:
            GitHubTagClientError: If the operation fails.
        """
        pass

    @abstractmethod
    def remove_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Remove a label from a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to remove.

        Raises:
            GitHubTagClientError: If the operation fails.
        """
        pass


class GitHubRestTagClient(BaseGitHubHttpClient, GitHubTagClient):
    """GitHub tag client that uses direct REST API calls for label operations.

    Uses connection pooling via a reusable httpx.Client for better performance
    in high-volume scenarios.
    """

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
        retry_config: GitHubRetryConfig | None = None,
    ) -> None:
        """Initialize the GitHub REST tag client.

        Args:
            token: GitHub personal access token or app token.
            base_url: Optional custom API base URL for GitHub Enterprise.
                     Defaults to "https://api.github.com".
                     For GitHub Enterprise: "https://your-ghe-host/api/v3"
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
        """
        super().__init__()
        self.base_url = (base_url or DEFAULT_GITHUB_API_URL).rstrip("/")
        self.token = token
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def add_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Add a label to a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to add.

        Raises:
            GitHubTagClientError: If the operation fails.
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/labels"
        payload = {"labels": [label]}

        def do_add() -> None:
            client = self._get_client()
            response = client.post(url, json=payload)
            _check_rate_limit_warning(response)
            response.raise_for_status()

        try:
            _execute_with_retry(do_add, self.retry_config, GitHubTagClientError)
            logger.info(f"Added label '{label}' to {owner}/{repo}#{issue_number}")
        except httpx.TimeoutException as e:
            raise GitHubTagClientError(f"Add label timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_msg = f"Add label failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except Exception:
                pass
            raise GitHubTagClientError(error_msg) from e
        except httpx.RequestError as e:
            raise GitHubTagClientError(f"Add label request failed: {e}") from e

    def remove_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Remove a label from a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to remove.

        Raises:
            GitHubTagClientError: If the operation fails.
        """
        # URL-encode the label name for path safety. Using httpx.URL(path=label).path
        # leverages httpx's built-in URL encoding to safely handle labels containing
        # special characters (e.g., spaces, slashes, unicode) that would otherwise
        # break the URL path or cause incorrect API calls.
        encoded_label = httpx.URL(path=label).path
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/labels/{encoded_label}"

        def do_remove() -> None:
            client = self._get_client()
            response = client.delete(url)
            _check_rate_limit_warning(response)
            # 404 is acceptable - label might already be removed
            if response.status_code == 404:
                logger.debug(
                    f"Label '{label}' not found on {owner}/{repo}#{issue_number}"
                )
                return
            response.raise_for_status()

        try:
            _execute_with_retry(do_remove, self.retry_config, GitHubTagClientError)
            logger.info(f"Removed label '{label}' from {owner}/{repo}#{issue_number}")
        except httpx.TimeoutException as e:
            raise GitHubTagClientError(f"Remove label timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_msg = f"Remove label failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except Exception:
                pass
            raise GitHubTagClientError(error_msg) from e
        except httpx.RequestError as e:
            raise GitHubTagClientError(f"Remove label request failed: {e}") from e
