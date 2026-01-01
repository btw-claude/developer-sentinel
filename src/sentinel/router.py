"""Tag-based router for matching Jira issues to orchestrations."""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.logging import get_logger
from sentinel.orchestration import Orchestration
from sentinel.poller import JiraIssue

logger = get_logger(__name__)


@dataclass
class RoutingResult:
    """Result of routing a Jira issue to orchestrations."""

    issue: JiraIssue
    orchestrations: list[Orchestration]

    @property
    def matched(self) -> bool:
        """Return True if the issue matched at least one orchestration."""
        return len(self.orchestrations) > 0


class Router:
    """Routes Jira issues to matching orchestrations based on tags.

    The router matches issues to orchestrations by checking if the issue's
    labels contain all the tags specified in an orchestration's trigger config.
    """

    def __init__(self, orchestrations: list[Orchestration]) -> None:
        """Initialize the router with orchestration configurations.

        Args:
            orchestrations: List of orchestration configurations to route to.
        """
        self.orchestrations = orchestrations
        logger.info(f"Router initialized with {len(orchestrations)} orchestrations")

    def _matches_trigger(self, issue: JiraIssue, orchestration: Orchestration) -> bool:
        """Check if an issue matches an orchestration's trigger.

        An issue matches if:
        1. The issue has ALL tags specified in the trigger (if any)
        2. The issue is in the specified project (if specified)

        Args:
            issue: The Jira issue to check.
            orchestration: The orchestration to match against.

        Returns:
            True if the issue matches the orchestration's trigger.
        """
        trigger = orchestration.trigger

        # Check project filter
        if trigger.project:
            # Extract project key from issue key (e.g., "PROJ-123" -> "PROJ")
            issue_project = issue.key.split("-")[0] if "-" in issue.key else ""
            if issue_project.upper() != trigger.project.upper():
                return False

        # Check tag/label filter - issue must have ALL specified tags
        if trigger.tags:
            issue_labels_lower = {label.lower() for label in issue.labels}
            for tag in trigger.tags:
                if tag.lower() not in issue_labels_lower:
                    return False

        return True

    def route(self, issue: JiraIssue) -> RoutingResult:
        """Route a single issue to matching orchestrations.

        Args:
            issue: The Jira issue to route.

        Returns:
            RoutingResult containing the issue and matching orchestrations.
        """
        matching: list[Orchestration] = []

        for orchestration in self.orchestrations:
            if self._matches_trigger(issue, orchestration):
                matching.append(orchestration)
                logger.debug(f"Issue {issue.key} matched orchestration '{orchestration.name}'")

        if matching:
            logger.info(
                f"Issue {issue.key} routed to {len(matching)} orchestration(s): "
                f"{[o.name for o in matching]}"
            )
        else:
            logger.debug(f"Issue {issue.key} did not match any orchestrations")

        return RoutingResult(issue=issue, orchestrations=matching)

    def route_all(self, issues: list[JiraIssue]) -> list[RoutingResult]:
        """Route multiple issues to matching orchestrations.

        Args:
            issues: List of Jira issues to route.

        Returns:
            List of RoutingResults, one per issue.
        """
        results = [self.route(issue) for issue in issues]

        matched_count = sum(1 for r in results if r.matched)
        logger.info(
            f"Routed {len(issues)} issues: {matched_count} matched, "
            f"{len(issues) - matched_count} unmatched"
        )

        return results

    def route_matched_only(self, issues: list[JiraIssue]) -> list[RoutingResult]:
        """Route multiple issues and return only those that matched.

        Args:
            issues: List of Jira issues to route.

        Returns:
            List of RoutingResults for issues that matched at least one orchestration.
        """
        return [result for result in self.route_all(issues) if result.matched]
