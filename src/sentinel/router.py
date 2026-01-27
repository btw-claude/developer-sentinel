"""Tag-based router for matching issues to orchestrations.

Supports routing both Jira issues and GitHub issues/PRs to matching orchestrations
based on tags/labels and source-specific filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from sentinel.github_poller import GitHubIssue, GitHubIssueProtocol
from sentinel.logging import get_logger
from sentinel.orchestration import Orchestration
from sentinel.poller import JiraIssue

logger = get_logger(__name__)

# Type alias for issues from any supported source
# Uses GitHubIssueProtocol to support both GitHubIssue and GitHubIssueWithRepo
AnyIssue = Union[JiraIssue, GitHubIssueProtocol]


@dataclass
class RoutingResult:
    """Result of routing an issue to orchestrations.

    Works with both Jira issues and GitHub issues/PRs.
    """

    issue: AnyIssue
    orchestrations: list[Orchestration]

    @property
    def matched(self) -> bool:
        """Return True if the issue matched at least one orchestration."""
        return len(self.orchestrations) > 0


class Router:
    """Routes issues to matching orchestrations based on tags.

    The router matches issues to orchestrations by checking if the issue's
    labels contain all the tags specified in an orchestration's trigger config.
    Supports both Jira issues and GitHub issues/PRs.
    """

    def __init__(self, orchestrations: list[Orchestration]) -> None:
        """Initialize the router with orchestration configurations.

        Args:
            orchestrations: List of orchestration configurations to route to.
        """
        self.orchestrations = orchestrations
        logger.info(f"Router initialized with {len(orchestrations)} orchestrations")

    def _matches_trigger(self, issue: AnyIssue, orchestration: Orchestration) -> bool:
        """Check if an issue matches an orchestration's trigger.

        An issue matches if:
        1. The issue source type matches the trigger source
        2. For Jira: The issue is in the specified project (if specified)
        3. For GitHub: The issue is in the specified repo (if specified)
        4. The issue has ALL tags specified in the trigger (if any)

        Tag matching is case-insensitive for both platforms.

        Args:
            issue: The issue to check (Jira or GitHub).
            orchestration: The orchestration to match against.

        Returns:
            True if the issue matches the orchestration's trigger.
        """
        trigger = orchestration.trigger

        # Check source type matches
        if isinstance(issue, JiraIssue):
            if trigger.source != "jira":
                return False

            # Check project filter for Jira
            if trigger.project:
                # Extract project key from issue key (e.g., "PROJ-123" -> "PROJ")
                issue_project = issue.key.split("-")[0] if "-" in issue.key else ""
                if issue_project.upper() != trigger.project.upper():
                    return False
        elif isinstance(issue, GitHubIssueProtocol):
            if trigger.source != "github":
                return False

            # Check repo filter for GitHub
            # Note: repo filter matching is handled during polling via search query,
            # but we validate here for direct routing if repo info is available
            # The GitHubIssue doesn't store repo info directly, so we skip repo
            # validation here - it's enforced during polling via the search query

            # For GitHub triggers, check labels field (in addition to existing tags check)
            if trigger.labels:
                issue_labels_lower = {label.lower() for label in issue.labels}
                for label in trigger.labels:
                    if label.lower() not in issue_labels_lower:
                        return False

        # Check tag/label filter - issue must have ALL specified tags (case-insensitive)
        if trigger.tags:
            issue_labels_lower = {label.lower() for label in issue.labels}
            for tag in trigger.tags:
                if tag.lower() not in issue_labels_lower:
                    return False

        return True

    def route(self, issue: AnyIssue) -> RoutingResult:
        """Route a single issue to matching orchestrations.

        Args:
            issue: The issue to route (Jira or GitHub).

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

    def route_all(self, issues: list[AnyIssue]) -> list[RoutingResult]:
        """Route multiple issues to matching orchestrations.

        Args:
            issues: List of issues to route (Jira or GitHub).

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

    def route_matched_only(self, issues: list[AnyIssue]) -> list[RoutingResult]:
        """Route multiple issues and return only those that matched.

        Args:
            issues: List of issues to route (Jira or GitHub).

        Returns:
            List of RoutingResults for issues that matched at least one orchestration.
        """
        return [result for result in self.route_all(issues) if result.matched]
