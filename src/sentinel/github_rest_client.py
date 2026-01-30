"""REST-based client implementations for GitHub API.

These clients use direct HTTP calls to the GitHub REST API for fast,
cost-effective polling and label operations without Claude invocations.

Implements exponential backoff with jitter for rate limiting per:
https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api

Circuit breaker pattern is implemented to prevent cascading failures
when the GitHub API is experiencing issues.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, TypeVar

import httpx

from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from sentinel.github_poller import GitHubClient, GitHubClientError
from sentinel.logging import get_logger

logger = get_logger(__name__)

# Default timeout for HTTP requests (connect, read, write, pool)
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)

# GitHub API default base URL
DEFAULT_GITHUB_API_URL = "https://api.github.com"

# Type variable for generic retry function
T = TypeVar("T")

# GraphQL query for fetching organization-owned GitHub Projects (v2)
# Returns project metadata and field definitions including single-select options
GRAPHQL_QUERY_ORG_PROJECT = """
query($owner: String!, $number: Int!) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      title
      url
      fields(first: 50) {
        nodes {
          ... on ProjectV2Field {
            id
            name
            dataType
          }
          ... on ProjectV2IterationField {
            id
            name
            dataType
          }
          ... on ProjectV2SingleSelectField {
            id
            name
            dataType
            options {
              id
              name
            }
          }
        }
      }
    }
  }
}
"""

# GraphQL query for fetching user-owned GitHub Projects (v2)
# Returns project metadata and field definitions including single-select options
GRAPHQL_QUERY_USER_PROJECT = """
query($owner: String!, $number: Int!) {
  user(login: $owner) {
    projectV2(number: $number) {
      id
      title
      url
      fields(first: 50) {
        nodes {
          ... on ProjectV2Field {
            id
            name
            dataType
          }
          ... on ProjectV2IterationField {
            id
            name
            dataType
          }
          ... on ProjectV2SingleSelectField {
            id
            name
            dataType
            options {
              id
              name
            }
          }
        }
      }
    }
  }
}
"""

# GraphQL query for paginating through GitHub Project (v2) items
# Returns Issues, PRs, and DraftIssues with their field values
GRAPHQL_QUERY_PROJECT_ITEMS = """
query($projectId: ID!, $cursor: String) {
  node(id: $projectId) {
    ... on ProjectV2 {
      items(first: 50, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          id
          content {
            ... on Issue {
              __typename
              number
              title
              state
              url
              body
              labels(first: 20) {
                nodes {
                  name
                }
              }
              assignees(first: 10) {
                nodes {
                  login
                }
              }
              author {
                login
              }
            }
            ... on PullRequest {
              __typename
              number
              title
              state
              url
              body
              isDraft
              headRefName
              baseRefName
              labels(first: 20) {
                nodes {
                  name
                }
              }
              assignees(first: 10) {
                nodes {
                  login
                }
              }
              author {
                login
              }
            }
            ... on DraftIssue {
              __typename
              title
              body
            }
          }
          fieldValues(first: 20) {
            nodes {
              ... on ProjectV2ItemFieldTextValue {
                text
                field {
                  ... on ProjectV2Field {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldNumberValue {
                number
                field {
                  ... on ProjectV2Field {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldDateValue {
                date
                field {
                  ... on ProjectV2Field {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldSingleSelectValue {
                name
                field {
                  ... on ProjectV2SingleSelectField {
                    name
                  }
                }
              }
              ... on ProjectV2ItemFieldIterationValue {
                title
                field {
                  ... on ProjectV2IterationField {
                    name
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


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
                    "GitHub rate limit near exhaustion. "
                    "Remaining: %s, Reset: %s",
                    remaining,
                    reset_time,
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
            logger.warning("Invalid Retry-After header value: %s", retry_after)

    # Check GitHub-specific x-ratelimit-reset (Unix timestamp)
    reset_time = response.headers.get("X-RateLimit-Reset")
    if reset_time is not None:
        try:
            reset_timestamp = int(reset_time)
            delay = reset_timestamp - int(time.time())
            if delay > 0:
                return float(delay)
        except ValueError:
            logger.warning("Invalid X-RateLimit-Reset header value: %s", reset_time)

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
                "Rate limited (attempt %s/%s). "
                "Retrying in %.2fs",
                attempt + 1,
                config.max_retries + 1,
                delay,
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
                    f"{self.__class__.__name__} must set self.timeout before "
                    "calling _get_client(). See BaseGitHubHttpClient docstring."
                )
            if not hasattr(self, "_headers") or self._headers is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must set self._headers before "
                    "calling _get_client(). See BaseGitHubHttpClient docstring."
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

    Also supports GitHub Projects (v2) via GraphQL API for project-based polling.

    Implements circuit breaker pattern to prevent cascading failures when
    GitHub API is unavailable or experiencing issues.
    """

    # GitHub GraphQL API endpoint
    GRAPHQL_URL = "https://api.github.com/graphql"

    # Default maximum number of results for paginated queries
    DEFAULT_MAX_RESULTS = 100

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
        retry_config: GitHubRetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the GitHub REST client.

        Args:
            token: GitHub personal access token or app token.
            base_url: Optional custom API base URL for GitHub Enterprise.
                     Defaults to "https://api.github.com".
                     For GitHub Enterprise: "https://your-ghe-host/api/v3"
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "github" service.
        """
        super().__init__()
        self.base_url = (base_url or DEFAULT_GITHUB_API_URL).rstrip("/")
        self.token = token
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="github",
            config=CircuitBreakerConfig.from_env("github"),
        )
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        # Derive GraphQL URL from base_url for GitHub Enterprise support.
        # GitHub.com uses separate domains for REST (api.github.com) and GraphQL
        # (api.github.com/graphql), while GitHub Enterprise Server (GHE) uses a
        # different URL structure: REST API is at <host>/api/v3 and GraphQL is at
        # <host>/api/graphql. We detect GHE by checking if the base_url is NOT the
        # standard api.github.com, then strip the "/api/v3" suffix to get the host
        # and append "/api/graphql" for the GraphQL endpoint.
        if base_url and "api.github.com" not in base_url:
            ghe_host = base_url.replace("/api/v3", "").rstrip("/")
            self._graphql_url = f"{ghe_host}/api/graphql"
        else:
            self._graphql_url = self.GRAPHQL_URL

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def search_issues(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues and pull requests using GitHub search syntax.

        Args:
            query: GitHub search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue/PR data from GitHub API.

        Raises:
            GitHubClientError: If the search fails, rate limit is exhausted,
                or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise GitHubClientError(
                f"GitHub circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        url = f"{self.base_url}/search/issues"
        params: dict[str, str | int] = {
            "q": query,
            "per_page": min(max_results, 100),  # GitHub max is 100 per page
        }

        logger.debug("Searching GitHub: %s", query)

        def do_search() -> list[dict[str, Any]]:
            client = self._get_client()
            response = client.get(url, params=params)
            _check_rate_limit_warning(response)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            items: list[dict[str, Any]] = data.get("items", [])
            total_count = data.get("total_count", 0)
            logger.info("GitHub search returned %s items (total: %s)", len(items), total_count)
            return items

        try:
            result = _execute_with_retry(do_search, self.retry_config, GitHubClientError)
            self._circuit_breaker.record_success()
            return result
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubClientError(f"GitHub search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"GitHub search failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise GitHubClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubClientError(f"GitHub search request failed: {e}") from e

    def _execute_graphql(
        self, query: str, variables: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a GraphQL query against the GitHub API.

        Args:
            query: The GraphQL query string.
            variables: Optional variables for the query.

        Returns:
            The 'data' portion of the GraphQL response.

        Raises:
            GitHubClientError: If the query fails, returns errors, or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise GitHubClientError(
                f"GitHub circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        def do_graphql() -> dict[str, Any]:
            client = self._get_client()
            response = client.post(self._graphql_url, json=payload)
            _check_rate_limit_warning(response)
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            # Check for GraphQL errors
            if "errors" in result:
                errors = result["errors"]
                error_messages = [e.get("message", str(e)) for e in errors]
                raise GitHubClientError(f"GraphQL errors: {'; '.join(error_messages)}")

            data: dict[str, Any] = result.get("data", {})
            return data

        try:
            result = _execute_with_retry(do_graphql, self.retry_config, GitHubClientError)
            self._circuit_breaker.record_success()
            return result
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubClientError(f"GraphQL request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"GraphQL request failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise GitHubClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubClientError(f"GraphQL request failed: {e}") from e

    def get_project(
        self, owner: str, project_number: int, scope: str = "organization"
    ) -> dict[str, Any]:
        """Query a GitHub Project (v2) by number.

        Args:
            owner: The organization or user that owns the project.
            project_number: The project number (visible in project URL).
            scope: Either "organization" or "user" depending on project ownership.

        Returns:
            Dictionary containing:
                - id: The project's global node ID (used for other operations)
                - title: The project title
                - url: The project URL
                - fields: List of field definitions with id, name, and dataType

        Raises:
            GitHubClientError: If the query fails or project is not found.
        """
        # Select appropriate GraphQL query based on scope (org vs user)
        if scope == "organization":
            query = GRAPHQL_QUERY_ORG_PROJECT
            owner_key = "organization"
        elif scope == "user":
            query = GRAPHQL_QUERY_USER_PROJECT
            owner_key = "user"
        else:
            raise GitHubClientError(f"Invalid scope: {scope}. Must be 'organization' or 'user'.")

        variables = {"owner": owner, "number": project_number}

        logger.debug("Fetching %s project: %s/project/%s", scope, owner, project_number)

        data = self._execute_graphql(query, variables)

        # Extract project data from response
        owner_data = data.get(owner_key)
        if not owner_data:
            raise GitHubClientError(f"{scope.capitalize()} '{owner}' not found")

        project = owner_data.get("projectV2")
        if not project:
            raise GitHubClientError(f"Project #{project_number} not found for {scope} '{owner}'")

        # Normalize the response
        fields = []
        for field_node in project.get("fields", {}).get("nodes", []):
            if field_node:  # Skip null nodes
                field_info: dict[str, Any] = {
                    "id": field_node.get("id"),
                    "name": field_node.get("name"),
                    "dataType": field_node.get("dataType"),
                }
                # Include options for single select fields
                if "options" in field_node:
                    field_info["options"] = field_node["options"]
                fields.append(field_info)

        result: dict[str, Any] = {
            "id": project.get("id"),
            "title": project.get("title"),
            "url": project.get("url"),
            "fields": fields,
        }

        logger.info("Retrieved project '%s' with %s fields", result['title'], len(fields))
        return result

    def list_project_items(
        self, project_id: str, max_results: int | None = None
    ) -> list[dict[str, Any]]:
        """List items in a GitHub Project (v2) with pagination.

        Args:
            project_id: The project's global node ID (from get_project).
            max_results: Maximum number of items to retrieve. Defaults to
                DEFAULT_MAX_RESULTS (100) if not specified.

        Returns:
            List of project items, each containing:
                - id: Item node ID
                - content: The linked Issue or PR with details (number, title,
                  state, url, labels, assignees) or None for draft items
                - fieldValues: List of field values for this item

        Raises:
            GitHubClientError: If the query fails.
        """
        if max_results is None:
            max_results = self.DEFAULT_MAX_RESULTS

        all_items: list[dict[str, Any]] = []
        cursor: str | None = None

        logger.debug("Fetching project items for project ID: %s", project_id)

        while len(all_items) < max_results:
            variables: dict[str, Any] = {"projectId": project_id}
            if cursor:
                variables["cursor"] = cursor

            data = self._execute_graphql(GRAPHQL_QUERY_PROJECT_ITEMS, variables)

            node = data.get("node")
            if not node:
                raise GitHubClientError(f"Project with ID '{project_id}' not found")

            items_data = node.get("items", {})
            nodes = items_data.get("nodes", [])

            for item_node in nodes:
                if len(all_items) >= max_results:
                    break
                if item_node:  # Skip null nodes
                    all_items.append(self._normalize_project_item(item_node))

            # Check for more pages
            page_info = items_data.get("pageInfo", {})
            if not page_info.get("hasNextPage") or len(all_items) >= max_results:
                break

            cursor = page_info.get("endCursor")

        logger.info("Retrieved %s project items", len(all_items))
        return all_items

    def _normalize_project_item(self, item_node: dict[str, Any]) -> dict[str, Any]:
        """Normalize a project item node from GraphQL response.

        Args:
            item_node: Raw item node from GraphQL response.

        Returns:
            Normalized item dictionary with consistent structure.
        """
        item: dict[str, Any] = {
            "id": item_node.get("id"),
            "content": None,
            "fieldValues": [],
        }

        # Process content (Issue, PR, or DraftIssue)
        content = item_node.get("content")
        if content:
            content_type = content.get("__typename")
            normalized_content: dict[str, Any] = {
                "type": content_type,
                "title": content.get("title"),
                "body": content.get("body", ""),
            }

            if content_type in ("Issue", "PullRequest"):
                normalized_content["number"] = content.get("number")
                normalized_content["state"] = content.get("state")
                normalized_content["url"] = content.get("url")

                # Extract labels
                labels_data = content.get("labels", {}).get("nodes", [])
                normalized_content["labels"] = [label.get("name") for label in labels_data if label]

                # Extract assignees
                assignees_data = content.get("assignees", {}).get("nodes", [])
                normalized_content["assignees"] = [
                    assignee.get("login") for assignee in assignees_data if assignee
                ]

                # Extract author
                author_data = content.get("author")
                normalized_content["author"] = author_data.get("login") if author_data else ""

                # PR-specific fields
                if content_type == "PullRequest":
                    normalized_content["isDraft"] = content.get("isDraft", False)
                    normalized_content["headRefName"] = content.get("headRefName", "")
                    normalized_content["baseRefName"] = content.get("baseRefName", "")

            item["content"] = normalized_content

        # Process field values
        field_values = item_node.get("fieldValues", {}).get("nodes", [])
        for fv in field_values:
            if not fv:
                continue

            field_data = fv.get("field", {})
            field_name = field_data.get("name")
            if not field_name:
                continue

            # Field values can be text, numbers, dates, or single select names
            value: str | int | float | None = None
            if "text" in fv:
                value = fv["text"]
            elif "number" in fv:
                value = fv["number"]
            elif "date" in fv:
                value = fv["date"]
            elif "name" in fv:  # Single select
                value = fv["name"]
            elif "title" in fv:  # Iteration
                value = fv["title"]

            if value is not None:
                item["fieldValues"].append({"field": field_name, "value": value})

        return item


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

    Implements circuit breaker pattern to prevent cascading failures when
    GitHub API is unavailable or experiencing issues.
    """

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        timeout: httpx.Timeout | None = None,
        retry_config: GitHubRetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the GitHub REST tag client.

        Args:
            token: GitHub personal access token or app token.
            base_url: Optional custom API base URL for GitHub Enterprise.
                     Defaults to "https://api.github.com".
                     For GitHub Enterprise: "https://your-ghe-host/api/v3"
            timeout: Optional custom timeout configuration.
            retry_config: Optional retry configuration for rate limiting.
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "github" service.
        """
        super().__init__()
        self.base_url = (base_url or DEFAULT_GITHUB_API_URL).rstrip("/")
        self.token = token
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="github",
            config=CircuitBreakerConfig.from_env("github"),
        )
        self._headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def add_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Add a label to a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to add.

        Raises:
            GitHubTagClientError: If the operation fails or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise GitHubTagClientError(
                f"GitHub circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/labels"
        payload = {"labels": [label]}

        def do_add() -> None:
            client = self._get_client()
            response = client.post(url, json=payload)
            _check_rate_limit_warning(response)
            response.raise_for_status()

        try:
            _execute_with_retry(do_add, self.retry_config, GitHubTagClientError)
            self._circuit_breaker.record_success()
            logger.info("Added label '%s' to %s/%s#%s", label, owner, repo, issue_number)
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubTagClientError(f"Add label timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"Add label failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise GitHubTagClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubTagClientError(f"Add label request failed: {e}") from e

    def remove_label(self, owner: str, repo: str, issue_number: int, label: str) -> None:
        """Remove a label from a GitHub issue or pull request.

        Args:
            owner: Repository owner (user or organization).
            repo: Repository name.
            issue_number: Issue or PR number.
            label: The label to remove.

        Raises:
            GitHubTagClientError: If the operation fails or circuit breaker is open.
        """
        # Check circuit breaker before attempting the request
        if not self._circuit_breaker.allow_request():
            raise GitHubTagClientError(
                f"GitHub circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

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
                logger.debug("Label '%s' not found on %s/%s#%s", label, owner, repo, issue_number)
                return
            response.raise_for_status()

        try:
            _execute_with_retry(do_remove, self.retry_config, GitHubTagClientError)
            self._circuit_breaker.record_success()
            logger.info("Removed label '%s' from %s/%s#%s", label, owner, repo, issue_number)
        except httpx.TimeoutException as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubTagClientError(f"Remove label timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            self._circuit_breaker.record_failure(e)
            error_msg = f"Remove label failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_msg += f": {error_data['message']}"
            except (ValueError, KeyError, TypeError):
                # JSON parsing or data extraction failed - continue with base error message
                pass
            raise GitHubTagClientError(error_msg) from e
        except httpx.RequestError as e:
            self._circuit_breaker.record_failure(e)
            raise GitHubTagClientError(f"Remove label request failed: {e}") from e
