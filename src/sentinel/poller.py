"""Jira poller module for querying issues matching orchestration triggers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sentinel.logging import get_logger
from sentinel.orchestration import TriggerConfig

logger = get_logger(__name__)


@dataclass
class JiraIssue:
    """Represents a Jira issue with relevant fields for orchestration."""

    key: str
    summary: str
    description: str = ""
    status: str = ""
    assignee: str | None = None
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    epic_key: str | None = None  # Parent epic key
    parent_key: str | None = None  # Parent issue key (for sub-tasks)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> JiraIssue:
        """Create a JiraIssue from Jira API response data.

        Args:
            data: Raw issue data from Jira API.

        Returns:
            JiraIssue instance.
        """
        fields = data.get("fields", {})

        # Extract description text
        description = ""
        desc_data = fields.get("description")
        if desc_data and isinstance(desc_data, dict):
            # Atlassian Document Format - extract text content
            description = _extract_adf_text(desc_data)
        elif isinstance(desc_data, str):
            description = desc_data

        # Extract assignee
        assignee = None
        assignee_data = fields.get("assignee")
        if assignee_data:
            assignee = assignee_data.get("displayName") or assignee_data.get("name")

        # Extract status
        status = ""
        status_data = fields.get("status")
        if status_data:
            status = status_data.get("name", "")

        # Extract comments
        comments: list[str] = []
        comment_data = fields.get("comment", {})
        if isinstance(comment_data, dict):
            for comment in comment_data.get("comments", []):
                body = comment.get("body")
                if isinstance(body, dict):
                    comments.append(_extract_adf_text(body))
                elif isinstance(body, str):
                    comments.append(body)

        # Extract issue links
        links: list[str] = []
        for link in fields.get("issuelinks", []):
            if "outwardIssue" in link:
                links.append(link["outwardIssue"]["key"])
            if "inwardIssue" in link:
                links.append(link["inwardIssue"]["key"])

        # Extract epic link and parent key, distinguishing by parent issue type
        # For next-gen projects, the parent field can represent either an epic or a story/task
        epic_key = None
        parent_key = None
        parent_data = fields.get("parent")
        if parent_data:
            parent_issue_key = parent_data.get("key")
            parent_type_data = parent_data.get("fields", {}).get("issuetype", {})
            parent_type_name = parent_type_data.get("name", "").lower()

            # Check if parent is an Epic type
            if parent_type_name == "epic":
                epic_key = parent_issue_key
            else:
                # Non-epic parent (Story, Task, etc.) - this is a sub-task relationship
                parent_key = parent_issue_key

        # Fallback to classic epic link field if no epic found from parent
        if not epic_key:
            epic_key = fields.get("customfield_10014")  # Classic epic link field

        return cls(
            key=data.get("key", ""),
            summary=fields.get("summary", ""),
            description=description,
            status=status,
            assignee=assignee,
            labels=fields.get("labels", []),
            comments=comments,
            links=links,
            epic_key=epic_key,
            parent_key=parent_key,
        )


def _extract_adf_text(adf: dict[str, Any]) -> str:
    """Extract plain text from Atlassian Document Format.

    Args:
        adf: Atlassian Document Format data.

    Returns:
        Plain text content.
    """
    texts: list[str] = []

    def extract(node: dict[str, Any]) -> None:
        if node.get("type") == "text":
            texts.append(node.get("text", ""))
        for child in node.get("content", []):
            if isinstance(child, dict):
                extract(child)

    extract(adf)
    return " ".join(texts)


class JiraClientError(Exception):
    """Raised when a Jira API operation fails."""

    pass


class JiraClient(ABC):
    """Abstract interface for Jira operations.

    This allows the poller to work with different implementations:
    - SDK-based client (Claude Agent SDK)
    - REST client (direct HTTP)
    - Mock client (testing)
    """

    @abstractmethod
    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL.

        Args:
            jql: JQL query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue data from Jira API.

        Raises:
            JiraClientError: If the search fails.
        """
        pass


class JiraPoller:
    """Polls Jira for issues matching orchestration triggers."""

    def __init__(
        self,
        client: JiraClient,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the Jira poller.

        Args:
            client: Jira client implementation.
            max_retries: Maximum number of retry attempts for API calls.
            retry_delay: Base delay between retries (exponential backoff).
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def build_jql(self, trigger: TriggerConfig) -> str:
        """Build a JQL query from trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.

        Returns:
            JQL query string.
        """
        conditions: list[str] = []

        # Project filter
        if trigger.project:
            conditions.append(f'project = "{trigger.project}"')

        # Tag/label filter - must have ALL specified tags
        for tag in trigger.tags:
            conditions.append(f'labels = "{tag}"')

        # Custom JQL filter (appended as-is)
        if trigger.jql_filter:
            conditions.append(f"({trigger.jql_filter})")

        # Default: exclude resolved/closed issues
        if not any("status" in c.lower() for c in conditions):
            conditions.append('status NOT IN ("Done", "Closed", "Resolved")')

        return " AND ".join(conditions) if conditions else ""

    def poll(self, trigger: TriggerConfig, max_results: int = 50) -> list[JiraIssue]:
        """Poll Jira for issues matching the trigger configuration.

        Args:
            trigger: Trigger configuration from orchestration.
            max_results: Maximum number of issues to return.

        Returns:
            List of matching JiraIssue objects.

        Raises:
            JiraClientError: If polling fails after retries.
        """
        jql = self.build_jql(trigger)
        if not jql:
            logger.warning("Empty JQL query generated from trigger config")
            return []

        logger.info(f"Polling Jira with JQL: {jql}")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw_issues = self.client.search_issues(jql, max_results)
                issues = [JiraIssue.from_api_response(issue) for issue in raw_issues]
                logger.info(f"Found {len(issues)} matching issues")
                return issues
            except JiraClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Jira API error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Jira API error after {self.max_retries} attempts: {e}")

        raise JiraClientError(
            f"Failed to poll Jira after {self.max_retries} attempts"
        ) from last_error
