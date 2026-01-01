"""Post-processing tag management for Jira issues."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sentinel.executor import ExecutionResult, ExecutionStatus
from sentinel.logging import get_logger
from sentinel.orchestration import Orchestration

logger = get_logger(__name__)


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

    @abstractmethod
    def get_labels(self, issue_key: str) -> list[str]:
        """Get all labels for a Jira issue.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").

        Returns:
            List of label names.

        Raises:
            JiraTagClientError: If the operation fails.
        """
        pass


@dataclass
class TagUpdateResult:
    """Result of updating tags on a Jira issue."""

    issue_key: str
    added_tags: list[str]
    removed_tags: list[str]
    errors: list[str]

    @property
    def success(self) -> bool:
        """Return True if all tag operations succeeded."""
        return len(self.errors) == 0


class TagManager:
    """Manages post-processing tag updates for Jira issues.

    After agent execution, updates issue tags based on the orchestration config:
    - On success: removes trigger tags, adds completion tags
    - On failure: adds failure tags, keeps trigger tags for investigation
    """

    def __init__(self, client: JiraTagClient) -> None:
        """Initialize the tag manager with a Jira client.

        Args:
            client: Jira tag client implementation.
        """
        self.client = client

    def update_tags(
        self,
        result: ExecutionResult,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Update tags on a Jira issue based on execution result.

        Args:
            result: The result of the agent execution.
            orchestration: The orchestration configuration.

        Returns:
            TagUpdateResult with details of what was changed.
        """
        added_tags: list[str] = []
        removed_tags: list[str] = []
        errors: list[str] = []

        if result.status == ExecutionStatus.SUCCESS:
            # Handle successful execution
            on_complete = orchestration.on_complete

            # Remove trigger tag if specified
            if on_complete.remove_tag:
                try:
                    self.client.remove_label(result.issue_key, on_complete.remove_tag)
                    removed_tags.append(on_complete.remove_tag)
                    logger.info(f"Removed tag '{on_complete.remove_tag}' from {result.issue_key}")
                except JiraTagClientError as e:
                    error_msg = (
                        f"Failed to remove tag '{on_complete.remove_tag}' "
                        f"from {result.issue_key}: {e}"
                    )
                    errors.append(error_msg)
                    logger.error(error_msg)

            # Add completion tag if specified
            if on_complete.add_tag:
                try:
                    self.client.add_label(result.issue_key, on_complete.add_tag)
                    added_tags.append(on_complete.add_tag)
                    logger.info(f"Added tag '{on_complete.add_tag}' to {result.issue_key}")
                except JiraTagClientError as e:
                    error_msg = (
                        f"Failed to add tag '{on_complete.add_tag}' to {result.issue_key}: {e}"
                    )
                    errors.append(error_msg)
                    logger.error(error_msg)

        elif result.status in (ExecutionStatus.FAILURE, ExecutionStatus.ERROR):
            # Handle failed execution - add failure tag but keep trigger tags
            on_failure = orchestration.on_failure

            if on_failure.add_tag:
                try:
                    self.client.add_label(result.issue_key, on_failure.add_tag)
                    added_tags.append(on_failure.add_tag)
                    logger.info(f"Added failure tag '{on_failure.add_tag}' to {result.issue_key}")
                except JiraTagClientError as e:
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
                f"Tag update completed for {result.issue_key}: "
                f"added={added_tags}, removed={removed_tags}"
            )
        else:
            logger.warning(f"Tag update had errors for {result.issue_key}: {errors}")

        return update_result

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
        logger.info(f"Batch tag update completed: {success_count}/{len(results)} successful")

        return update_results
