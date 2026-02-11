"""REST-based client implementations for Jira API.

These clients use direct HTTP calls to the Jira REST API for fast,
cost-effective polling and label operations without Claude invocations.

Implements exponential backoff with jitter for rate limiting per:
https://developer.atlassian.com/cloud/jira/platform/rate-limiting/

Circuit breaker pattern is implemented to prevent cascading failures
when the Jira API is experiencing issues.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import httpx

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# Default timeout for HTTP requests (connect, read, write, pool)
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 4).
        initial_delay: Initial delay in seconds before first retry (default: 1.0).
        max_delay: Maximum delay in seconds between retries (default: 30.0).
        jitter_min: Minimum jitter multiplier (default: 0.7).
        jitter_max: Maximum jitter multiplier (default: 1.3).
    """

    max_retries: int = 4
    initial_delay: float = 1.0
    max_delay: float = 30.0
    jitter_min: float = 0.7
    jitter_max: float = 1.3


# Default retry configuration per Atlassian recommendations
DEFAULT_RETRY_CONFIG = RetryConfig()


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and all retries are exhausted."""

    pass


def _calculate_backoff_delay(
    attempt: int,
    config: RetryConfig,
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
    near_limit = response.headers.get("X-RateLimit-NearLimit", "").lower()
    if near_limit == "true":
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        reset_time = response.headers.get("X-RateLimit-Reset", "unknown")
        logger.warning(
            "Jira rate limit near exhaustion. Remaining: %s, Reset: %s", remaining, reset_time
        )


def _get_retry_after(response: httpx.Response) -> float | None:
    """Extract Retry-After value from response headers.

    Args:
        response: HTTP response to check.

    Returns:
        Retry-After value in seconds, or None if not present.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            logger.warning("Invalid Retry-After header value: %s", retry_after)
    return None


def _execute_with_retry[T](
    operation: Callable[[], T],
    config: RetryConfig = DEFAULT_RETRY_CONFIG,
    error_class: type[Exception] = Exception,
) -> T:
    """Execute an operation with retry logic for rate limiting.

    Args:
        operation: Callable that performs the HTTP operation and returns a result.
                  Should raise httpx.HTTPStatusError on 429 responses.
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
            if e.response.status_code != 429:
                # Not a rate limit error, don't retry
                raise

            last_exception = e

            if attempt >= config.max_retries:
                # All retries exhausted
                reason = e.response.headers.get("RateLimit-Reason", "unknown")
                raise error_class(
                    f"Rate limit exceeded after {config.max_retries} retries. Reason: {reason}"
                ) from e

            # Calculate delay and wait
            retry_after = _get_retry_after(e.response)
            delay = _calculate_backoff_delay(attempt, config, retry_after)

            reason = e.response.headers.get("RateLimit-Reason", "unknown")
            logger.warning(
                "Rate limited (attempt %s/%s). "
                "Reason: %s. Retrying in %.2fs",
                attempt + 1,
                config.max_retries + 1,
                reason,
                delay,
            )

            time.sleep(delay)

    # Should not reach here, but satisfy type checker
    if last_exception:
        raise error_class(f"Retry failed: {last_exception}") from last_exception
    raise error_class("Retry failed with no exception")


"""Type alias for authentication credentials used by Jira HTTP clients.

Supports basic auth (email, token) tuples and httpx.BasicAuth for extensibility.
Token-based or custom auth schemes can be added to this union in the future
without changing the BaseJiraHttpClient interface or its subclasses.
"""
JiraAuth = tuple[str, str] | httpx.BasicAuth


class BaseJiraHttpClient:
    """Base class providing HTTP client connection pooling for Jira API clients.

    This class encapsulates the shared connection pooling functionality including:
    - Lazy initialization of httpx.Client
    - Resource cleanup via close() method
    - Context manager support (__enter__/__exit__)

    Subclasses must set self.timeout and self.auth before using _get_client().

    MRO note:
        Concrete subclasses (e.g., JiraRestClient, JiraRestTagClient) use cooperative
        multiple inheritance, inheriting from both BaseJiraHttpClient and an abstract
        interface (JiraClient or JiraTagClient). Currently the abstract interfaces do
        not define ``__init__`` methods, so the MRO chain works without explicit
        ``super().__init__()`` forwarding in every class.

        If JiraClient or JiraTagClient ever gain ``__init__`` methods, all classes
        in the MRO chain must use cooperative ``super().__init__(**kwargs)`` calls to
        ensure every ``__init__`` in the chain is invoked. Failing to do so would
        silently skip initialization of parent classes that appear later in the MRO.
        See the Python docs on `cooperative multiple inheritance
        <https://docs.python.org/3/tutorial/classes.html#multiple-inheritance>`_ for
        details.

    Example subclass implementation::

        class MyJiraClient(BaseJiraHttpClient):
            def __init__(
                self,
                base_url: str,
                email: str,
                api_token: str,
                timeout: httpx.Timeout | None = None,
            ) -> None:
                super().__init__()
                self.base_url = base_url.rstrip("/")
                self.auth = httpx.BasicAuth(email, api_token)
                self.timeout = timeout or DEFAULT_TIMEOUT

            def my_api_method(self) -> dict:
                client = self._get_client()
                response = client.get("https://yoursite.atlassian.net/rest/api/3/...")
                return response.json()
    """

    # Instance attribute type stubs - subclasses MUST set these in __init__
    # before calling _get_client(). These are declared here to support static
    # type checking and IDE autocompletion for the abstract contract.
    timeout: httpx.Timeout
    auth: JiraAuth

    def __init__(self) -> None:
        """Initialize the base HTTP client.

        Note: Subclasses must call super().__init__() and then set
        self.timeout and self.auth before using _get_client().
        """
        # Reusable HTTP client for connection pooling - lazily initialized
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the reusable HTTP client.

        Returns:
            A configured httpx.Client instance with connection pooling.

        Raises:
            RuntimeError: If subclass has not set required timeout or auth attributes.
        """
        if self._client is None:
            # Runtime safety check: ensure subclass has properly initialized required attributes
            if not hasattr(self, "timeout") or self.timeout is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must set self.timeout before "
                    "calling _get_client(). See BaseJiraHttpClient docstring."
                )
            if not hasattr(self, "auth") or self.auth is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must set self.auth before "
                    "calling _get_client(). See BaseJiraHttpClient docstring."
                )
            self._client = httpx.Client(auth=self.auth, timeout=self.timeout)
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


class JiraRestClient(BaseJiraHttpClient, JiraClient):
    """Jira client that uses direct REST API calls for searching issues.

    Uses connection pooling via a reusable httpx.Client for better performance
    in high-volume scenarios.

    Implements circuit breaker pattern to prevent cascading failures when
    Jira API is unavailable or experiencing issues.
    """

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        timeout: httpx.Timeout | None = None,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the Jira REST client.

        Args:
            base_url: Jira base URL (e.g., "https://yoursite.atlassian.net").
            email: User email for authentication.
            api_token: API token for authentication.
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "jira" service.
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.auth: JiraAuth = httpx.BasicAuth(email, api_token)
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="jira",
            config=CircuitBreakerConfig.from_env("jira"),
        )

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL via REST API.

        Args:
            jql: JQL query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue data from Jira API.

        Raises:
            JiraClientError: If the search fails, rate limit is exhausted,
                or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise JiraClientError(
                f"Jira circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        # Use the new /search/jql endpoint (the old /search was deprecated Aug 2025)
        # See: https://developer.atlassian.com/changelog/#CHANGE-2046
        url = f"{self.base_url}/rest/api/3/search/jql"
        params: dict[str, str | int] = {
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,description,status,assignee,labels,comment,issuelinks",
        }

        logger.debug("Searching Jira: %s", jql)

        def do_search() -> list[dict[str, Any]]:
            client = self._get_client()
            response = client.get(url, params=params)
            _check_rate_limit_warning(response)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            issues: list[dict[str, Any]] = data.get("issues", [])
            logger.info("JQL search returned %s issues", len(issues))
            return issues

        try:
            result = _execute_with_retry(do_search, self.retry_config, JiraClientError)
            self._circuit_breaker.record_success()
            return result
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise JiraClientError(f"Jira search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"Jira search failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "errorMessages" in error_data:
                    error_msg += f": {', '.join(error_data['errorMessages'])}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise JiraClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise JiraClientError(f"Jira search request failed: {e}") from e


class JiraRestTagClient(BaseJiraHttpClient, JiraTagClient):
    """Jira tag client that uses direct REST API calls for label operations.

    Uses connection pooling via a reusable httpx.Client for better performance
    in high-volume scenarios.

    Implements circuit breaker pattern to prevent cascading failures when
    Jira API is unavailable or experiencing issues.
    """

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        timeout: httpx.Timeout | None = None,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the Jira REST tag client.

        Args:
            base_url: Jira base URL (e.g., "https://yoursite.atlassian.net").
            email: User email for authentication.
            api_token: API token for authentication.
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "jira" service.
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.auth: JiraAuth = httpx.BasicAuth(email, api_token)
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="jira",
            config=CircuitBreakerConfig.from_env("jira"),
        )

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def _get_current_labels(self, issue_key: str) -> list[str]:
        """Get current labels for an issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").

        Returns:
            List of current labels.

        Raises:
            JiraTagClientError: If the operation fails or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise JiraTagClientError(
                f"Jira circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        params = {"fields": "labels"}

        def do_get() -> list[str]:
            client = self._get_client()
            response = client.get(url, params=params)
            _check_rate_limit_warning(response)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            labels: list[str] = data.get("fields", {}).get("labels", [])
            return labels

        try:
            result = _execute_with_retry(do_get, self.retry_config, JiraTagClientError)
            self._circuit_breaker.record_success()
            return result
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise JiraTagClientError(f"Get labels timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            raise JiraTagClientError(
                f"Get labels failed with status {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise JiraTagClientError(f"Get labels request failed: {e}") from e

    def _update_labels(self, issue_key: str, labels: list[str]) -> None:
        """Update labels for an issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            labels: New list of labels.

        Raises:
            JiraTagClientError: If the operation fails or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise JiraTagClientError(
                f"Jira circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        payload = {"fields": {"labels": labels}}

        def do_update() -> None:
            client = self._get_client()
            response = client.put(url, json=payload)
            _check_rate_limit_warning(response)
            response.raise_for_status()

        try:
            _execute_with_retry(do_update, self.retry_config, JiraTagClientError)
            self._circuit_breaker.record_success()
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise JiraTagClientError(f"Update labels timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"Update labels failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "errorMessages" in error_data:
                    error_msg += f": {', '.join(error_data['errorMessages'])}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise JiraTagClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise JiraTagClientError(f"Update labels request failed: {e}") from e

    def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to a Jira issue via REST API.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to add.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        current_labels = self._get_current_labels(issue_key)
        if label not in current_labels:
            new_labels = current_labels + [label]
            self._update_labels(issue_key, new_labels)
            logger.info("Added label '%s' to %s", label, issue_key)
        else:
            logger.debug("Label '%s' already exists on %s", label, issue_key)

    def remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from a Jira issue via REST API.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to remove.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        current_labels = self._get_current_labels(issue_key)
        if label in current_labels:
            new_labels = [lbl for lbl in current_labels if lbl != label]
            self._update_labels(issue_key, new_labels)
            logger.info("Removed label '%s' from %s", label, issue_key)
        else:
            logger.debug("Label '%s' not found on %s", label, issue_key)
