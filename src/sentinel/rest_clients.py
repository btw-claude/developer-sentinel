"""REST-based client implementations for Jira API.

These clients use direct HTTP calls to the Jira REST API for fast,
cost-effective polling and label operations without Claude invocations.
"""

from __future__ import annotations

from typing import Any

import httpx

from sentinel.logging import get_logger
from sentinel.poller import JiraClient, JiraClientError
from sentinel.tag_manager import JiraTagClient, JiraTagClientError

logger = get_logger(__name__)

# Default timeout for HTTP requests (connect, read, write, pool)
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)


class JiraRestClient(JiraClient):
    """Jira client that uses direct REST API calls for searching issues."""

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """Initialize the Jira REST client.

        Args:
            base_url: Jira base URL (e.g., "https://yoursite.atlassian.net").
            email: User email for authentication.
            api_token: API token for authentication.
            timeout: Optional custom timeout configuration.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.timeout = timeout or DEFAULT_TIMEOUT

    def search_issues(self, jql: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Search for issues using JQL via REST API.

        Args:
            jql: JQL query string.
            max_results: Maximum number of results to return.

        Returns:
            List of raw issue data from Jira API.

        Raises:
            JiraClientError: If the search fails.
        """
        url = f"{self.base_url}/rest/api/3/search"
        params: dict[str, str | int] = {
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,description,status,assignee,labels,comment,issuelinks",
        }

        logger.debug(f"Searching Jira: {jql}")

        try:
            with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                issues: list[dict[str, Any]] = data.get("issues", [])
                logger.info(f"JQL search returned {len(issues)} issues")
                return issues

        except httpx.TimeoutException as e:
            raise JiraClientError(f"Jira search timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_msg = f"Jira search failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "errorMessages" in error_data:
                    error_msg += f": {', '.join(error_data['errorMessages'])}"
            except Exception:
                pass
            raise JiraClientError(error_msg) from e
        except httpx.RequestError as e:
            raise JiraClientError(f"Jira search request failed: {e}") from e


class JiraRestTagClient(JiraTagClient):
    """Jira tag client that uses direct REST API calls for label operations."""

    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """Initialize the Jira REST tag client.

        Args:
            base_url: Jira base URL (e.g., "https://yoursite.atlassian.net").
            email: User email for authentication.
            api_token: API token for authentication.
            timeout: Optional custom timeout configuration.
        """
        self.base_url = base_url.rstrip("/")
        self.auth = (email, api_token)
        self.timeout = timeout or DEFAULT_TIMEOUT

    def _get_current_labels(self, issue_key: str) -> list[str]:
        """Get current labels for an issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").

        Returns:
            List of current labels.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        params = {"fields": "labels"}

        try:
            with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                labels: list[str] = data.get("fields", {}).get("labels", [])
                return labels

        except httpx.TimeoutException as e:
            raise JiraTagClientError(f"Get labels timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise JiraTagClientError(
                f"Get labels failed with status {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise JiraTagClientError(f"Get labels request failed: {e}") from e

    def _update_labels(self, issue_key: str, labels: list[str]) -> None:
        """Update labels for an issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            labels: New list of labels.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        payload = {"fields": {"labels": labels}}

        try:
            with httpx.Client(auth=self.auth, timeout=self.timeout) as client:
                response = client.put(url, json=payload)
                response.raise_for_status()

        except httpx.TimeoutException as e:
            raise JiraTagClientError(f"Update labels timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            error_msg = f"Update labels failed with status {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "errorMessages" in error_data:
                    error_msg += f": {', '.join(error_data['errorMessages'])}"
            except Exception:
                pass
            raise JiraTagClientError(error_msg) from e
        except httpx.RequestError as e:
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
            logger.info(f"Added label '{label}' to {issue_key}")
        else:
            logger.debug(f"Label '{label}' already exists on {issue_key}")

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
            logger.info(f"Removed label '{label}' from {issue_key}")
        else:
            logger.debug(f"Label '{label}' not found on {issue_key}")
