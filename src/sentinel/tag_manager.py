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

    @property
    def partial_success(self) -> bool:
        """Return True if some operations succeeded but there were also errors."""
        has_errors = len(self.errors) > 0
        has_successes = len(self.added_tags) > 0 or len(self.removed_tags) > 0
        return has_errors and has_successes


class TagManager:
    """Manages tag updates for Jira issues throughout the processing lifecycle.

    Handles tags at two points:
    - On start: removes trigger tags, adds in-progress tag (prevents duplicate processing)
    - On complete: removes in-progress tag, adds completion/failure tags
    """

    def __init__(self, client: JiraTagClient) -> None:
        """Initialize the tag manager with a Jira client.

        Args:
            client: Jira tag client implementation.
        """
        self.client = client

    def start_processing(
        self,
        issue_key: str,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Update tags when an issue is picked up for processing.

        Removes trigger tags and adds in-progress tag to prevent duplicate processing.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
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
                self.client.remove_label(issue_key, tag)
                removed_tags.append(tag)
                logger.info(f"Removed trigger tag '{tag}' from {issue_key}")
            except JiraTagClientError as e:
                error_msg = f"Failed to remove trigger tag '{tag}' from {issue_key}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Add in-progress tag if configured
        on_start = orchestration.on_start
        if on_start.add_tag:
            try:
                self.client.add_label(issue_key, on_start.add_tag)
                added_tags.append(on_start.add_tag)
                logger.info(f"Added in-progress tag '{on_start.add_tag}' to {issue_key}")
            except JiraTagClientError as e:
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
                f"Start processing tags updated for {issue_key}: "
                f"added={added_tags}, removed={removed_tags}"
            )
        else:
            logger.warning(f"Start processing tag update had errors for {issue_key}: {errors}")

        return result

    def update_tags(
        self,
        result: ExecutionResult,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Update tags on a Jira issue based on execution result.

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
                self.client.remove_label(result.issue_key, on_start.add_tag)
                removed_tags.append(on_start.add_tag)
                logger.info(f"Removed in-progress tag '{on_start.add_tag}' from {result.issue_key}")
            except JiraTagClientError as e:
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
                        self.client.remove_label(result.issue_key, on_complete.remove_tag)
                        removed_tags.append(on_complete.remove_tag)
                        logger.info(
                            f"Removed tag '{on_complete.remove_tag}' from {result.issue_key}"
                        )
                    except JiraTagClientError as e:
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
                    self.client.add_label(result.issue_key, tag_to_add)
                    added_tags.append(tag_to_add)
                    logger.info(f"Added tag '{tag_to_add}' to {result.issue_key}")
                except JiraTagClientError as e:
                    error_msg = f"Failed to add tag '{tag_to_add}' to {result.issue_key}: {e}"
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

    def apply_failure_tags(
        self,
        issue_key: str,
        orchestration: Orchestration,
    ) -> TagUpdateResult:
        """Apply failure tags to an issue when processing is interrupted.

        This is used when a process is terminated (e.g., by Ctrl-C) and we need
        to mark the issue as failed. Removes in-progress tag and adds failure tag.

        Args:
            issue_key: The Jira issue key (e.g., "PROJ-123").
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
                self.client.remove_label(issue_key, on_start.add_tag)
                removed_tags.append(on_start.add_tag)
                logger.info(f"Removed in-progress tag '{on_start.add_tag}' from {issue_key}")
            except JiraTagClientError as e:
                error_msg = (
                    f"Failed to remove in-progress tag '{on_start.add_tag}' from {issue_key}: {e}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

        # Add failure tag if configured
        on_failure = orchestration.on_failure
        if on_failure.add_tag:
            try:
                self.client.add_label(issue_key, on_failure.add_tag)
                added_tags.append(on_failure.add_tag)
                logger.info(f"Added failure tag '{on_failure.add_tag}' to {issue_key}")
            except JiraTagClientError as e:
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
                f"Failure tags applied for {issue_key}: added={added_tags}, removed={removed_tags}"
            )
        else:
            logger.warning(f"Failure tag update had errors for {issue_key}: {errors}")

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
        logger.info(f"Batch tag update completed: {success_count}/{len(results)} successful")

        return update_results
