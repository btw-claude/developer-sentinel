"""Post-processing tag management for Jira and GitHub issues."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sentinel.executor import ExecutionResult, ExecutionStatus
from sentinel.github_rest_client import GitHubTagClient, GitHubTagClientError
from sentinel.logging import get_logger
from sentinel.orchestration import Orchestration

logger = get_logger(__name__)

# Regex patterns for issue key detection
# GitHub format: org/repo#123
GITHUB_ISSUE_PATTERN = re.compile(r"^([^/]+)/([^#]+)#(\d+)$")
# Jira format: PROJ-123
JIRA_ISSUE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]+-\d+$")


class JiraTagClientError(Exception):
    """Raised when a Jira tag operation fails."""

    pass


class JiraTagClient(ABC):
    """Abstract interface for Jira label/tag operations.

    This allows the tag manager to work with different implementations:
    - Real Jira API client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def add_label(self, issue_key: str, label: str) -> None:
        """Add a label to a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to add.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        pass

    @abstractmethod
    def remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
            label: The label to remove.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        pass


@dataclass
class TagUpdateResult:
    """Result of updating tags on a Jira or GitHub issue."""

    issue_key: str
    added_tags: list[str]
    removed_tags: list[str]
    errors: list[str]

    @property
    def success(self) -> bool:
        """Return True if all tag operations succeeded."""
        return len(self.errors) == 0

    @property
    def partial_success(self) -> bool:
        """Return True if some operations succeeded but there were also errors."""
        has_errors = len(self.errors) > 0
        has_successes = len(self.added_tags) > 0 or len(self.removed_tags) > 0
        return has_errors and has_successes


class TagManager:
    """Manages tag updates for Jira and GitHub issues throughout the processing lifecycle.

    Handles tags at two points:
    - On start: removes trigger tags, adds in-progress tag (prevents duplicate processing)
    - On complete: removes in-progress tag, adds completion/failure tags

    Supports both Jira issues (PROJ-123 format) and GitHub issues (org/repo#123 format).
    """

    def __init__(
        self,
        client: JiraTagClient,
        github_client: GitHubTagClient | None = None,
    ) -> None:
        """Initialize the tag manager with Jira and optional GitHub clients.

        Args:
            client: Jira tag client implementation.
            github_client: Optional GitHub tag client for GitHub issue support.
        """
        self.client = client
        self.github_client = github_client

    def _get_client_for_issue(
        self, issue_key: str
    ) -> tuple[str, JiraTagClient | GitHubTagClient | None, tuple[str, str, int] | None]:
        """Detect the platform from issue key format and return appropriate client.

        Args:
            issue_key: The issue identifier.
                      GitHub format: org/repo#123
                      Jira format: PROJ-123

        Returns:
            A tuple of (platform, client, parsed_info):
            - platform: "github" or "jira"
            - client: The appropriate tag client, or None if not available
            - parsed_info: For GitHub, a tuple of (owner, repo, issue_number).
                          For Jira, None.
        """
        # Try GitHub format first: org/repo#123
        github_match = GITHUB_ISSUE_PATTERN.match(issue_key)
        if github_match:
            owner, repo, issue_num = github_match.groups()
            return ("github", self.github_client, (owner, repo, int(issue_num)))

        # Try Jira format: PROJ-123
        if JIRA_ISSUE_PATTERN.match(issue_key):
            return ("jira", self.client, None)

        # Default to Jira for backwards compatibility
        logger.warning(
            "Issue key '%s' doesn't match known formats. "
            "Assuming Jira format for backwards compatibility.",
            issue_key
        )
        return ("jira", self.client, None)

    def _add_label(self, issue_key: str, label: str) -> None:
        """Add a label to an issue using the appropriate client.

        Args:
            issue_key: The issue identifier (Jira or GitHub format).
            label: The label to add.

        Raises:
            JiraTagClientError: If the Jira operation fails.
            GitHubTagClientError: If the GitHub operation fails.
            ValueError: If the client for the platform is not available.
        """
        platform, client, github_info = self._get_client_for_issue(issue_key)

        if client is None:
            raise ValueError(
                f"No client available for {platform} platform. "
                f"Cannot add label to {issue_key}."
            )

        if platform == "github" and github_info:
            owner, repo, issue_number = github_info
            if isinstance(client, GitHubTagClient):
                client.add_label(owner, repo, issue_number, label)
            else:
                raise ValueError(f"Invalid GitHub client type for {issue_key}")
        else:
            # Jira client
            if isinstance(client, JiraTagClient):
                client.add_label(issue_key, label)
            else:
                raise ValueError(f"Invalid Jira client type for {issue_key}")

    def _remove_label(self, issue_key: str, label: str) -> None:
        """Remove a label from an issue using the appropriate client.

        Args:
            issue_key: The issue identifier (Jira or GitHub format).
            label: The label to remove.

        Raises:
            JiraTagClientError: If the Jira operation fails.
            GitHubTagClientError: If the GitHub operation fails.
            ValueError: If the client for the platform is not available.
        """
        platform, client, github_info = self._get_client_for_issue(issue_key)

        if client is None:
            raise ValueError(
                f"No client available for {platform} platform. "
                f"Cannot remove label from {issue_key}."
            )

        if platform == "github" and github_info:
            owner, repo, issue_number = github_info
            if isinstance(client, GitHubTagClient):
                client.remove_label(owner, repo, issue_number, label)
            else:
                raise ValueError(f"Invalid GitHub client type for {issue_key}")
        else:
            # Jira client
            if isinstance(client, JiraTagClient):
                client.remove_label(issue_key, label)
            else:
                raise ValueError(f"Invalid Jira client type for {issue_key}")

    def start_processing(
        self,
        issue_key: str,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Update tags when an issue is picked up for processing.

        Removes trigger tags and adds in-progress tag to prevent duplicate processing.

        Args:
            issue_key: The issue key (Jira "PROJ-123" or GitHub "org/repo#123").
            orchestration: The orchestration configuration.

        Returns:
            TagUpdateResult with details of what was changed.
        """
        added_tags: list[str] = []
        removed_tags: list[str] = []
        errors: list[str] = []

        # Remove trigger tags
        for tag in orchestration.trigger.tags:
            try:
                self._remove_label(issue_key, tag)
                removed_tags.append(tag)
                logger.info("Removed trigger tag '%s' from %s", tag, issue_key)
            except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                error_msg = f"Failed to remove trigger tag '{tag}' from {issue_key}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Add in-progress tag if configured
        on_start = orchestration.on_start
        if on_start.add_tag:
            try:
                self._add_label(issue_key, on_start.add_tag)
                added_tags.append(on_start.add_tag)
                logger.info("Added in-progress tag '%s' to %s", on_start.add_tag, issue_key)
            except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                error_msg = (
                    f"Failed to add in-progress tag '{on_start.add_tag}' to {issue_key}: {e}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

        result = TagUpdateResult(
            issue_key=issue_key,
            added_tags=added_tags,
            removed_tags=removed_tags,
            errors=errors,
        )

        if result.success:
            logger.info(
                "Start processing tags updated for %s: added=%s, removed=%s",
                issue_key, added_tags, removed_tags
            )
        else:
            logger.warning("Start processing tag update had errors for %s: %s", issue_key, errors)

        return result

    def update_tags(
        self,
        result: ExecutionResult,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Update tags on an issue based on execution result.

        Removes in-progress tag (if configured) and applies completion/failure tags.

        Args:
            result: The result of the agent execution.
            orchestration: The orchestration configuration.

        Returns:
            TagUpdateResult with details of what was changed.
        """
        added_tags: list[str] = []
        removed_tags: list[str] = []
        errors: list[str] = []

        # Always remove in-progress tag if configured (regardless of success/failure)
        on_start = orchestration.on_start
        if on_start.add_tag:
            try:
                self._remove_label(result.issue_key, on_start.add_tag)
                removed_tags.append(on_start.add_tag)
                logger.info(
                    "Removed in-progress tag '%s' from %s", on_start.add_tag, result.issue_key
                )
            except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                error_msg = (
                    f"Failed to remove in-progress tag '{on_start.add_tag}' "
                    f"from {result.issue_key}: {e}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

        if result.status == ExecutionStatus.SUCCESS:
            # Handle successful execution
            # Determine which tag to add based on matched_outcome or on_complete
            tag_to_add = ""

            if result.matched_outcome and orchestration.outcomes:
                # Use outcome-based tag
                for outcome in orchestration.outcomes:
                    if outcome.name == result.matched_outcome:
                        tag_to_add = outcome.add_tag
                        break
            else:
                # Fall back to legacy on_complete behavior
                on_complete = orchestration.on_complete

                # Remove trigger tag if specified (legacy behavior)
                if on_complete.remove_tag:
                    try:
                        self._remove_label(result.issue_key, on_complete.remove_tag)
                        removed_tags.append(on_complete.remove_tag)
                        logger.info(
                            "Removed tag '%s' from %s",
                            on_complete.remove_tag, result.issue_key
                        )
                    except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                        error_msg = (
                            f"Failed to remove tag '{on_complete.remove_tag}' "
                            f"from {result.issue_key}: {e}"
                        )
                        errors.append(error_msg)
                        logger.error(error_msg)

                tag_to_add = on_complete.add_tag

            # Add the determined tag
            if tag_to_add:
                try:
                    self._add_label(result.issue_key, tag_to_add)
                    added_tags.append(tag_to_add)
                    logger.info("Added tag '%s' to %s", tag_to_add, result.issue_key)
                except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                    error_msg = f"Failed to add tag '{tag_to_add}' to {result.issue_key}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        elif result.status in (ExecutionStatus.FAILURE, ExecutionStatus.ERROR):
            # Handle failed execution - add failure tag but keep trigger tags
            on_failure = orchestration.on_failure

            if on_failure.add_tag:
                try:
                    self._add_label(result.issue_key, on_failure.add_tag)
                    added_tags.append(on_failure.add_tag)
                    logger.info(
                        "Added failure tag '%s' to %s", on_failure.add_tag, result.issue_key
                    )
                except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                    error_msg = (
                        f"Failed to add failure tag '{on_failure.add_tag}' "
                        f"to {result.issue_key}: {e}"
                    )
                    errors.append(error_msg)
                    logger.error(error_msg)

        update_result = TagUpdateResult(
            issue_key=result.issue_key,
            added_tags=added_tags,
            removed_tags=removed_tags,
            errors=errors,
        )

        if update_result.success:
            logger.info(
                "Tag update completed for %s: added=%s, removed=%s",
                result.issue_key, added_tags, removed_tags
            )
        else:
            logger.warning("Tag update had errors for %s: %s", result.issue_key, errors)

        return update_result

    def apply_failure_tags(
        self,
        issue_key: str,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Apply failure tags to an issue when processing is interrupted.

        This is used when a process is terminated (e.g., by Ctrl-C) and we need
        to mark the issue as failed. Removes in-progress tag and adds failure tag.

        Args:
            issue_key: The issue key (Jira "PROJ-123" or GitHub "org/repo#123").
            orchestration: The orchestration configuration.

        Returns:
            TagUpdateResult with details of what was changed.
        """
        added_tags: list[str] = []
        removed_tags: list[str] = []
        errors: list[str] = []

        # Remove in-progress tag if configured
        on_start = orchestration.on_start
        if on_start.add_tag:
            try:
                self._remove_label(issue_key, on_start.add_tag)
                removed_tags.append(on_start.add_tag)
                logger.info("Removed in-progress tag '%s' from %s", on_start.add_tag, issue_key)
            except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                error_msg = (
                    f"Failed to remove in-progress tag '{on_start.add_tag}' from {issue_key}: {e}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

        # Add failure tag if configured
        on_failure = orchestration.on_failure
        if on_failure.add_tag:
            try:
                self._add_label(issue_key, on_failure.add_tag)
                added_tags.append(on_failure.add_tag)
                logger.info("Added failure tag '%s' to %s", on_failure.add_tag, issue_key)
            except (JiraTagClientError, GitHubTagClientError, ValueError) as e:
                error_msg = f"Failed to add failure tag '{on_failure.add_tag}' to {issue_key}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        result = TagUpdateResult(
            issue_key=issue_key,
            added_tags=added_tags,
            removed_tags=removed_tags,
            errors=errors,
        )

        if result.success:
            logger.info(
                "Failure tags applied for %s: added=%s, removed=%s",
                issue_key, added_tags, removed_tags
            )
        else:
            logger.warning("Failure tag update had errors for %s: %s", issue_key, errors)

        return result

    def update_tags_batch(
        self,
        results: list[tuple[ExecutionResult, Orchestration]],
    ) -> list[TagUpdateResult]:
        """Update tags for multiple execution results.

        Args:
            results: List of (ExecutionResult, Orchestration) tuples.

        Returns:
            List of TagUpdateResults, one per input.
        """
        update_results = []
        for result, orchestration in results:
            update_result = self.update_tags(result, orchestration)
            update_results.append(update_result)

        success_count = sum(1 for r in update_results if r.success)
        logger.info("Batch tag update completed: %s/%s successful", success_count, len(results))

        return update_results
