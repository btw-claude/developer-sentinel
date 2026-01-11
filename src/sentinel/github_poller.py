"""GitHub poller module for querying issues and PRs matching orchestration triggers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sentinel.logging import get_logger
from sentinel.orchestration import TriggerConfig

logger = get_logger(__name__)


@dataclass
class GitHubIssue:
    """Represents a GitHub issue or pull request with relevant fields for orchestration.

    Attributes:
        number: The issue/PR number.
        title: The issue/PR title.
        body: The issue/PR body/description.
        state: The state of the issue/PR (e.g., "open", "closed").
        author: The username of the author.
        assignees: List of assigned usernames.
        labels: List of label names.
        is_pull_request: Whether this is a pull request (vs a regular issue).
        head_ref: The head branch reference (for PRs).
        base_ref: The base branch reference (for PRs).
        draft: Whether this PR is a draft (for PRs).
    """

    number: int
    title: str
    body: str = ""
    state: str = "open"
    author: str = ""
    assignees: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    is_pull_request: bool = False
    head_ref: str = ""
    base_ref: str = ""
    draft: bool = False

    @property
    def key(self) -> str:
        """Return a unique key for this issue/PR in the format 'org/repo#number'.

        This is used as a consistent identifier across the system.
        """
        return f"#{self.number}"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GitHubIssue:
        """Create a GitHubIssue from GitHub API response data.

        Args:
            data: Raw issue/PR data from GitHub API.

        Returns:
            GitHubIssue instance.
        """
        # Extract author
        author = ""
        user_data = data.get("user")
        if user_data and isinstance(user_data, dict):
            author = user_data.get("login", "")

        # Extract assignees
        assignees: list[str] = []
        assignees_data = data.get("assignees", [])
        for assignee in assignees_data:
            if isinstance(assignee, dict) and "login" in assignee:
                assignees.append(assignee["login"])

        # Extract labels
        labels: list[str] = []
        labels_data = data.get("labels", [])
        for label in labels_data:
            if isinstance(label, dict) and "name" in label:
                labels.append(label["name"])
            elif isinstance(label, str):
                labels.append(label)

        # Determine if this is a pull request
        is_pull_request = "pull_request" in data or data.get("pull_request") is not None

        # Extract PR-specific fields
        head_ref = ""
        base_ref = ""
        draft = False

        if is_pull_request:
            # For search results, PR data might be nested
            pr_data = data.get("pull_request", {})
            if isinstance(pr_data, dict):
                # Search API doesn't include head/base, but PR API does
                pass

            # Direct PR fields (from PR API or if included in search)
            head_data = data.get("head")
            if head_data and isinstance(head_data, dict):
                head_ref = head_data.get("ref", "")

            base_data = data.get("base")
            if base_data and isinstance(base_data, dict):
                base_ref = base_data.get("ref", "")

            draft = data.get("draft", False)

        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body") or "",
            state=data.get("state", "open"),
            author=author,
            assignees=assignees,
            labels=labels,
            is_pull_request=is_pull_request,
            head_ref=head_ref,
            base_ref=base_ref,
            draft=draft,
        )


class GitHubClientError(Exception):
    """Raised when a GitHub API operation fails."""

    pass


class GitHubClient(ABC):
    """Abstract interface for GitHub operations.

    This allows the poller to work with different implementations:
    - REST API client (production)
    - Mock client (testing)
    """

    @abstractmethod
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
            GitHubClientError: If the search fails.
        """
        pass


class GitHubPoller:
    """Polls GitHub for issues and PRs matching orchestration triggers."""

    def __init__(
        self,
        client: GitHubClient,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the GitHub poller.

        Args:
            client: GitHub client implementation.
            max_retries: Maximum number of retry attempts for API calls.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def build_query(self, trigger: TriggerConfig) -> str:
        """Build a GitHub search query from trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.

        Returns:
            GitHub search query string.
        """
        conditions: list[str] = []

        # Repository filter
        if trigger.repo:
            conditions.append(f"repo:{trigger.repo}")

        # Tag/label filter - must have ALL specified tags
        for tag in trigger.tags:
            conditions.append(f'label:"{tag}"')

        # Custom query filter (appended as-is)
        if trigger.query_filter:
            conditions.append(trigger.query_filter)

        # Default: only open issues/PRs
        if not any("state:" in c.lower() or "is:open" in c.lower() or "is:closed" in c.lower() for c in conditions):
            conditions.append("state:open")

        return " ".join(conditions) if conditions else ""

    def poll(self, trigger: TriggerConfig, max_results: int = 50) -> list[GitHubIssue]:
        """Poll GitHub for issues/PRs matching the trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.
            max_results: Maximum number of issues to return.

        Returns:
            List of matching GitHubIssue objects.

        Raises:
            GitHubClientError: If polling fails after retries.
        """
        query = self.build_query(trigger)
        if not query:
            logger.warning("Empty query generated from trigger config")
            return []

        logger.info(f"Polling GitHub with query: {query}")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw_issues = self.client.search_issues(query, max_results)
                issues = [GitHubIssue.from_api_response(issue) for issue in raw_issues]
                logger.info(f"Found {len(issues)} matching issues/PRs")
                return issues
            except GitHubClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"GitHub API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"GitHub API error after {self.max_retries} attempts: {e}")

        raise GitHubClientError(
            f"Failed to poll GitHub after {self.max_retries} attempts"
        ) from last_error
