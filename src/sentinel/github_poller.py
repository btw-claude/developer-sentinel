"""GitHub poller module for querying issues/PRs matching orchestration triggers."""

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
        body: The issue/PR body content.
        state: The state (open, closed).
        author: The author's login username.
        assignees: List of assignee login usernames.
        labels: List of label names.
        is_pull_request: True if this is a pull request, False if it's an issue.
        head_ref: For PRs, the head branch name.
        base_ref: For PRs, the base branch name.
        draft: For PRs, whether this is a draft PR.
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
        """Return a unique key for this issue/PR in 'org/repo#number' format.

        This mirrors the Jira issue key concept for consistent orchestration handling.
        """
        # Note: The actual org/repo prefix should be added by the caller
        # since we don't store that info in the dataclass
        return f"#{self.number}"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GitHubIssue:
        """Create a GitHubIssue from GitHub API response data.

        Args:
            data: Raw issue/PR data from GitHub API.

        Returns:
            GitHubIssue instance.
        """
        # Determine if this is a PR (has pull_request field)
        is_pr = "pull_request" in data

        # Extract assignees
        assignees: list[str] = []
        for assignee in data.get("assignees", []):
            if isinstance(assignee, dict):
                login = assignee.get("login", "")
                if login:
                    assignees.append(login)

        # Extract labels
        labels: list[str] = []
        for label in data.get("labels", []):
            if isinstance(label, dict):
                name = label.get("name", "")
                if name:
                    labels.append(name)
            elif isinstance(label, str):
                labels.append(label)

        # Extract author
        author = ""
        user = data.get("user")
        if isinstance(user, dict):
            author = user.get("login", "")

        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=data.get("body") or "",
            state=data.get("state", "open"),
            author=author,
            assignees=assignees,
            labels=labels,
            is_pull_request=is_pr,
            head_ref=data.get("head", {}).get("ref", "") if is_pr else "",
            base_ref=data.get("base", {}).get("ref", "") if is_pr else "",
            draft=data.get("draft", False) if is_pr else False,
        )


class GitHubClientError(Exception):
    """Raised when a GitHub API operation fails."""

    pass


class GitHubClient(ABC):
    """Abstract interface for GitHub operations.

    This allows the poller to work with different implementations:
    - REST-based client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def search_issues(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues/PRs using GitHub search syntax.

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
    """Polls GitHub for issues/PRs matching orchestration triggers."""

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

        # Repository filter (required for GitHub triggers)
        if trigger.repo:
            conditions.append(f"repo:{trigger.repo}")

        # Tag/label filter - each label is a separate condition
        for tag in trigger.tags:
            conditions.append(f'label:"{tag}"')

        # Custom query filter (appended as-is)
        if trigger.query_filter:
            conditions.append(trigger.query_filter)

        # Default: only open issues/PRs
        if not any("state:" in c.lower() or "is:" in c.lower() for c in conditions):
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
