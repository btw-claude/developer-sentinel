"""GitHub poller module for querying issues and PRs matching orchestration triggers."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sentinel.logging import get_logger
from sentinel.orchestration import TriggerConfig
from sentinel.project_filter import ProjectFilterParser

logger = get_logger(__name__)


@runtime_checkable
class GitHubIssueProtocol(Protocol):
    """Protocol defining the interface for GitHub issue objects.

    This protocol enables type-safe handling of both GitHubIssue and wrapper classes
    like GitHubIssueWithRepo (which adds repository context to the issue key).

    Both GitHubIssue and any wrapper classes that delegate to GitHubIssue should
    satisfy this protocol, enabling proper type annotations without '# type: ignore'.
    """

    number: int
    title: str
    body: str
    state: str
    author: str
    assignees: list[str]
    labels: list[str]
    is_pull_request: bool
    head_ref: str
    base_ref: str
    draft: bool
    repo_url: str
    parent_issue_number: int | None

    @property
    def key(self) -> str:
        """Return a unique key for this issue/PR."""
        ...


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
        repo_url: The full URL to the issue/PR (used to extract repo context).
        parent_issue_number: The parent issue number (if this issue has a parent).
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
    repo_url: str = ""
    parent_issue_number: int | None = None

    @property
    def key(self) -> str:
        """Return a unique key for this issue/PR in the format '#number'.

        This is used as a consistent identifier across the system.
        """
        return f"#{self.number}"

    @staticmethod
    def _extract_parent_issue_number(data: dict[str, Any]) -> int | None:
        """Extract parent issue number from data dictionary.

        Args:
            data: Dictionary containing potential parent issue data.

        Returns:
            The parent issue number if present and valid, None otherwise.
            If an unexpected type is encountered, logs a warning and returns None.
        """
        parent_data = data.get("parent")
        if parent_data and isinstance(parent_data, dict):
            number = parent_data.get("number")
            if number is None:
                return None
            if isinstance(number, int):
                return number
            # Log warning for unexpected type and treat as None
            logger.warning(
                f"Unexpected type for parent.number: {type(number).__name__}, "
                f"expected int. Treating as None."
            )
            return None
        return None

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
            parent_issue_number=cls._extract_parent_issue_number(data),
        )

    @classmethod
    def from_project_item(cls, item: dict[str, Any]) -> GitHubIssue | None:
        """Create a GitHubIssue from a GitHub Project item.

        Parses the project item structure returned by the GraphQL API,
        extracting content details (Issue or PullRequest) and converting
        to a GitHubIssue object.

        Args:
            item: Normalized project item from list_project_items(), containing:
                - id: Item node ID
                - content: The linked Issue/PR with details, or None for drafts
                - fieldValues: List of field values for this item

        Returns:
            GitHubIssue instance if the item contains an Issue or PullRequest,
            None if the item is a DraftIssue or has no content.
        """
        content = item.get("content")
        if not content:
            # No content - possibly a draft or deleted item
            return None

        content_type = content.get("type")

        # Skip DraftIssue items - they don't have issue numbers
        if content_type == "DraftIssue":
            return None

        # Determine if this is a pull request
        is_pull_request = content_type == "PullRequest"

        # Extract labels from nested nodes structure
        labels: list[str] = content.get("labels", [])

        # Extract assignees from nested nodes structure
        assignees: list[str] = content.get("assignees", [])

        # Extract author
        author = content.get("author", "")

        # Map state - GraphQL returns OPEN/CLOSED/MERGED for PRs
        state = content.get("state", "OPEN").lower()
        if state == "merged":
            state = "closed"  # Normalize to REST API convention

        # Extract PR-specific fields
        head_ref = ""
        base_ref = ""
        draft = False

        if is_pull_request:
            head_ref = content.get("headRefName", "")
            base_ref = content.get("baseRefName", "")
            draft = content.get("isDraft", False)

        # Extract URL for repo context extraction
        repo_url = content.get("url", "")

        return cls(
            number=content.get("number", 0),
            title=content.get("title", ""),
            body=content.get("body", "") or "",
            state=state,
            author=author,
            assignees=assignees,
            labels=labels,
            is_pull_request=is_pull_request,
            head_ref=head_ref,
            base_ref=base_ref,
            draft=draft,
            repo_url=repo_url,
            parent_issue_number=cls._extract_parent_issue_number(content),
        )


class GitHubClientError(Exception):
    """Raised when a GitHub API operation fails."""

    pass


class GitHubClient(ABC):
    """Abstract interface for GitHub operations.

    This allows the poller to work with different implementations:
    - REST API client (production)
    - Mock client (testing)

    Includes both REST API methods (search_issues) and GraphQL methods
    (get_project, list_project_items) for GitHub Projects (v2) support.
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

    @abstractmethod
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
        pass

    @abstractmethod
    def list_project_items(
        self, project_id: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """List items in a GitHub Project (v2) with pagination.

        Args:
            project_id: The project's global node ID (from get_project).
            max_results: Maximum number of items to retrieve.

        Returns:
            List of project items, each containing:
                - id: Item node ID
                - content: The linked Issue or PR with details (number, title,
                  state, url, labels, assignees) or None for draft items
                - fieldValues: List of field values for this item

        Raises:
            GitHubClientError: If the query fails.
        """
        pass


class GitHubPoller:
    """Polls GitHub for issues and PRs matching orchestration triggers.

    Supports project-based polling using GitHub Projects (v2) GraphQL API.
    The poll() method uses the trigger's project configuration to fetch
    items from a GitHub Project and optionally filter them using a
    JQL-like project_filter expression.
    """

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
        # Cache for project IDs: key is "owner/project_number", value is project_id
        self._project_id_cache: dict[str, str] = {}
        # Filter parser for project_filter expressions
        self._filter_parser = ProjectFilterParser()

    def _get_project_id(self, owner: str, project_number: int, scope: str) -> str:
        """Get the project ID, using cache to avoid repeated lookups.

        Args:
            owner: The organization or user that owns the project.
            project_number: The project number.
            scope: Either "org" or "user" depending on project ownership.

        Returns:
            The project's global node ID.

        Raises:
            GitHubClientError: If the project lookup fails.
        """
        cache_key = f"{owner}/{project_number}"
        if cache_key in self._project_id_cache:
            logger.debug(f"Using cached project ID for {cache_key}")
            return self._project_id_cache[cache_key]

        # Map scope from config format to API format
        api_scope = "organization" if scope == "org" else "user"

        logger.debug(f"Fetching project ID for {cache_key} (scope={api_scope})")
        project = self.client.get_project(owner, project_number, api_scope)
        project_id = project["id"]

        self._project_id_cache[cache_key] = project_id
        logger.info(f"Cached project ID for {cache_key}: {project_id[:20]}...")
        return project_id

    def clear_project_id_cache(self) -> None:
        """Clear the cached project IDs.

        Useful for testing scenarios or long-running processes that may need
        to refresh the project ID cache (e.g., if a project is deleted and
        recreated with a new ID).
        """
        self._project_id_cache.clear()
        logger.debug("Project ID cache cleared")

    def _extract_field_values(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extract field values from a project item into a flat dictionary.

        Args:
            item: Project item with fieldValues list.

        Returns:
            Dictionary mapping field names to their values.
        """
        fields: dict[str, Any] = {}
        for fv in item.get("fieldValues", []):
            field_name = fv.get("field")
            value = fv.get("value")
            if field_name:
                fields[field_name] = value
        return fields

    def _issue_has_all_labels(self, issue: GitHubIssue, required_labels: list[str]) -> bool:
        """Check if issue has all required labels (case-insensitive AND logic).

        Args:
            issue: GitHubIssue to check labels on.
            required_labels: List of labels that must all be present.

        Returns:
            True if issue has all required labels, False otherwise.
        """
        issue_labels_lower = {label.lower() for label in issue.labels}
        return all(label.lower() in issue_labels_lower for label in required_labels)

    def _apply_filter(
        self, items: list[dict[str, Any]], filter_expr: str
    ) -> list[dict[str, Any]]:
        """Apply a project_filter expression to filter items.

        Args:
            items: List of project items to filter.
            filter_expr: JQL-like filter expression.

        Returns:
            Filtered list of items matching the expression.
        """
        if not filter_expr:
            return items

        try:
            parsed_filter = self._filter_parser.parse(filter_expr)
        except ValueError as e:
            logger.error(f"Invalid project_filter expression: {e}")
            raise GitHubClientError(f"Invalid project_filter: {e}") from e

        filtered: list[dict[str, Any]] = []
        for item in items:
            fields = self._extract_field_values(item)
            if self._filter_parser.evaluate(parsed_filter, fields):
                filtered.append(item)

        logger.debug(
            f"Filter '{filter_expr}' matched {len(filtered)}/{len(items)} items"
        )
        return filtered

    def build_query(self, trigger: TriggerConfig) -> str:
        """Build a GitHub search query from trigger configuration.

        .. deprecated::
            This method is deprecated for GitHub triggers. Use project-based
            configuration (project_number, project_scope, project_owner,
            project_filter) instead. This method will be removed in a future version.

        Args:
            trigger: Trigger configuration from orchestration.

        Returns:
            GitHub search query string.
        """
        warnings.warn(
            "build_query() is deprecated for GitHub triggers. "
            "Use project-based polling with project_number, project_scope, "
            "project_owner, and project_filter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        Uses project-based polling via GitHub Projects (v2) GraphQL API.
        Fetches items from the configured project and filters them using
        the optional project_filter expression.

        Args:
            trigger: Trigger configuration from orchestration. Must have
                project_number, project_scope, and project_owner set.
            max_results: Maximum number of issues to return.

        Returns:
            List of matching GitHubIssue objects. Draft issues are skipped.

        Raises:
            GitHubClientError: If polling fails after retries.
        """
        # Validate project configuration
        if trigger.project_number is None:
            raise GitHubClientError(
                "GitHub trigger requires project_number to be configured"
            )
        if not trigger.project_owner:
            raise GitHubClientError(
                "GitHub trigger requires project_owner to be configured"
            )

        logger.info(
            f"Polling GitHub project: {trigger.project_owner}/project/{trigger.project_number}"
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                # Get project ID (cached)
                project_id = self._get_project_id(
                    trigger.project_owner,
                    trigger.project_number,
                    trigger.project_scope,
                )

                # Fetch project items
                raw_items = self.client.list_project_items(project_id, max_results)
                logger.debug(f"Fetched {len(raw_items)} items from project")

                # Apply filter if configured
                filtered_items = self._apply_filter(raw_items, trigger.project_filter)

                # Convert to GitHubIssue objects, skipping drafts
                issues: list[GitHubIssue] = []
                for item in filtered_items:
                    issue = GitHubIssue.from_project_item(item)
                    if issue is not None:
                        if trigger.labels and not self._issue_has_all_labels(issue, trigger.labels):
                            continue
                        issues.append(issue)

                logger.info(
                    f"Found {len(issues)} matching issues/PRs "
                    f"(filtered from {len(raw_items)} project items)"
                )
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
