"""Polling coordinator for the Sentinel orchestrator.

This module provides the PollCoordinator class which manages:
- Coordinating polling cycles for Jira and GitHub
- Routing issues to orchestrations
- Managing deduplication of submissions
- Constructing issue URLs for dashboard display

This is part of the Sentinel refactoring to split the God Object into focused,
composable components (DS-384).

Component Boundaries
--------------------
PollCoordinator is responsible for all polling logic and issue routing. It
coordinates with JiraPoller and GitHubPoller to fetch issues, then routes
them through the Router to find matching orchestrations.

The Sentinel class delegates to PollCoordinator for:
- ``poll_jira_triggers()`` : Poll Jira for matching issues
- ``poll_github_triggers()`` : Poll GitHub for matching issues/PRs
- ``group_orchestrations_by_source()`` : Separate Jira vs GitHub orchestrations
- ``create_cycle_dedup_set()`` : Create deduplication set for polling cycle
- ``check_and_mark_submitted()`` : Deduplication check and marking
- ``construct_issue_url()`` : Build URLs for dashboard display

See Also
--------
- docs/architecture.md : Full architecture documentation
- sentinel.poller : JiraPoller for Jira issue fetching
- sentinel.github_poller : GitHubPoller for GitHub issue/PR fetching
- sentinel.router : Issue to orchestration routing
- sentinel.state_tracker : State and metrics management
- sentinel.execution_manager : Thread pool and future management
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from sentinel.deduplication import DeduplicationManager, build_github_trigger_key
from sentinel.github_poller import GitHubIssue, GitHubIssueProtocol
from sentinel.logging import get_logger
from sentinel.orchestration import Orchestration, TriggerConfig
from sentinel.poller import JiraIssue

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.github_poller import GitHubPoller
    from sentinel.poller import JiraPoller
    from sentinel.router import Router, RoutingResult

logger = get_logger(__name__)

# Module-level constant for GitHub issue/PR URL parsing
GITHUB_ISSUE_PR_URL_PATTERN = re.compile(r"https?://[^/]+/([^/]+/[^/]+)/(?:issues|pull)/\d+")


def extract_repo_from_url(url: str) -> str | None:
    """Extract owner/repo from a GitHub issue or PR URL.

    Args:
        url: GitHub URL, e.g., "https://github.com/org/repo/issues/123"

    Returns:
        Repository in "owner/repo" format, or None if URL is invalid.
    """
    if not url:
        return None

    match = GITHUB_ISSUE_PR_URL_PATTERN.match(url)
    if match:
        return match.group(1)
    return None


class GitHubIssueWithRepo:
    """Wrapper for GitHubIssue that provides full key with repo context.

    The GitHubIssue.key property returns "#123" but tag operations need "org/repo#123".
    This wrapper provides the full key while delegating all other GitHubIssueProtocol
    properties to the wrapped issue via __getattr__.
    """

    __slots__ = ("_issue", "_repo")

    def __init__(self, issue: GitHubIssue, repo: str) -> None:
        """Initialize the wrapper.

        Args:
            issue: The underlying GitHubIssue object.
            repo: Repository in "org/repo" format.
        """
        self._issue = issue
        self._repo = repo

    @property
    def key(self) -> str:
        """Return the full issue key including repo context."""
        return f"{self._repo}#{self._issue.number}"

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped GitHubIssue."""
        return getattr(self._issue, name)


class PollingResult:
    """Result of a polling operation."""

    def __init__(
        self,
        issues_found: int = 0,
        submitted_count: int = 0,
        queued_count: int = 0,
    ) -> None:
        """Initialize the polling result.

        Args:
            issues_found: Number of issues found from polling.
            submitted_count: Number of tasks successfully submitted.
            queued_count: Number of tasks queued due to slot limits.
        """
        self.issues_found = issues_found
        self.submitted_count = submitted_count
        self.queued_count = queued_count


class PollCoordinator:
    """Coordinates polling for Jira and GitHub issues.

    This class is responsible for:
    - Polling Jira and GitHub for issues matching orchestration triggers
    - Routing issues to matching orchestrations
    - Managing deduplication of submissions within a polling cycle
    - Constructing issue URLs for dashboard display

    Thread Safety:
        This class is designed to be called from a single thread (the main
        polling loop). The deduplication manager handles thread safety internally.
    """

    def __init__(
        self,
        config: Config,
        jira_poller: JiraPoller | None = None,
        github_poller: GitHubPoller | None = None,
    ) -> None:
        """Initialize the poll coordinator.

        Args:
            config: Application configuration.
            jira_poller: Optional Jira poller for polling Jira issues.
            github_poller: Optional GitHub poller for polling GitHub issues/PRs.
        """
        self._config = config
        self._jira_poller = jira_poller
        self._github_poller = github_poller
        self._dedup_manager = DeduplicationManager()

    def create_cycle_dedup_set(self) -> set[tuple[str, str]]:
        """Create a new deduplication set for a polling cycle.

        Returns:
            An empty set to track (issue_key, orchestration_name) tuples.
        """
        return self._dedup_manager.create_cycle_set()

    def check_and_mark_submitted(
        self,
        submitted_pairs: set[tuple[str, str]],
        issue_key: str,
        orchestration_name: str,
    ) -> bool:
        """Check if a pair should be submitted and mark it.

        Args:
            submitted_pairs: Set tracking submitted pairs for this cycle.
            issue_key: The issue key.
            orchestration_name: The name of the orchestration.

        Returns:
            True if this pair should be submitted, False if duplicate.
        """
        return self._dedup_manager.check_and_mark(submitted_pairs, issue_key, orchestration_name)

    def poll_jira_triggers(
        self,
        orchestrations: list[Orchestration],
        router: Router,
        shutdown_requested: bool = False,
        log_callback: Any | None = None,
    ) -> tuple[list[RoutingResult], int]:
        """Poll Jira for issues matching orchestration triggers.

        Args:
            orchestrations: List of Jira-triggered orchestrations.
            router: Router for matching issues to orchestrations.
            shutdown_requested: Whether shutdown has been requested.
            log_callback: Optional callback for logging (orchestration_name, level, message).

        Returns:
            Tuple of (routing_results, total_issues_found).
        """
        if self._jira_poller is None:
            return [], 0

        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, TriggerConfig]] = []

        for orch in orchestrations:
            trigger_key = f"jira:{orch.trigger.project}:{','.join(orch.trigger.tags)}"
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        all_routing_results: list[RoutingResult] = []
        total_issues_found = 0

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            if shutdown_requested:
                logger.info("Shutdown requested, stopping polling")
                return all_routing_results, total_issues_found

            # Log polling
            if log_callback:
                log_callback(
                    orch.name,
                    logging.INFO,
                    f"Polling Jira for orchestration '{orch.name}'",
                )

            try:
                issues = self._jira_poller.poll(trigger, self._config.max_issues_per_poll)
                total_issues_found += len(issues)

                if log_callback:
                    log_callback(
                        orch.name,
                        logging.INFO,
                        f"Found {len(issues)} Jira issues for '{orch.name}'",
                    )
            except (OSError, TimeoutError) as e:
                logger.error(
                    f"Failed to poll Jira for '{orch.name}': {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    f"Failed to poll Jira for '{orch.name}' due to data error: {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue
            except RuntimeError as e:
                logger.error(
                    f"Failed to poll Jira for '{orch.name}' due to runtime error: {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue

            # Route issues to matching orchestrations
            routing_results = router.route_matched_only(issues)
            all_routing_results.extend(routing_results)

        return all_routing_results, total_issues_found

    def poll_github_triggers(
        self,
        orchestrations: list[Orchestration],
        router: Router,
        shutdown_requested: bool = False,
        log_callback: Any | None = None,
    ) -> tuple[list[RoutingResult], int]:
        """Poll GitHub for issues/PRs matching orchestration triggers.

        Args:
            orchestrations: List of GitHub-triggered orchestrations.
            router: Router for matching issues to orchestrations.
            shutdown_requested: Whether shutdown has been requested.
            log_callback: Optional callback for logging (orchestration_name, level, message).

        Returns:
            Tuple of (routing_results, total_issues_found).
        """
        if self._github_poller is None:
            return [], 0

        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, TriggerConfig]] = []

        for orch in orchestrations:
            trigger_key = build_github_trigger_key(orch)
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        all_routing_results: list[RoutingResult] = []
        total_issues_found = 0

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            if shutdown_requested:
                logger.info("Shutdown requested, stopping polling")
                return all_routing_results, total_issues_found

            # Log polling
            if log_callback:
                log_callback(
                    orch.name,
                    logging.INFO,
                    f"Polling GitHub for orchestration '{orch.name}'",
                )

            try:
                issues = self._github_poller.poll(trigger, self._config.max_issues_per_poll)
                total_issues_found += len(issues)

                if log_callback:
                    log_callback(
                        orch.name,
                        logging.INFO,
                        f"Found {len(issues)} GitHub issues/PRs for '{orch.name}'",
                    )
            except (OSError, TimeoutError) as e:
                logger.error(
                    f"Failed to poll GitHub for '{orch.name}': {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue
            except (KeyError, TypeError, ValueError) as e:
                logger.error(
                    f"Failed to poll GitHub for '{orch.name}' due to data error: {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue
            except RuntimeError as e:
                logger.error(
                    f"Failed to poll GitHub for '{orch.name}' due to runtime error: {e}",
                    extra={"orchestration": orch.name, "error_type": type(e).__name__},
                )
                continue

            # Convert GitHub issues to include repo context
            issues_with_context = self._add_repo_context_from_urls(issues)

            # Route issues to matching orchestrations
            routing_results = router.route_matched_only(issues_with_context)
            all_routing_results.extend(routing_results)

        return all_routing_results, total_issues_found

    def _add_repo_context_from_urls(
        self,
        issues: list[GitHubIssue],
    ) -> list[GitHubIssueProtocol]:
        """Add repository context to GitHub issues by extracting repo from URLs.

        Args:
            issues: List of GitHubIssue objects from the poller.

        Returns:
            List of GitHubIssueWithRepo objects with updated keys.
        """
        result: list[GitHubIssueProtocol] = []
        for issue in issues:
            repo = extract_repo_from_url(issue.repo_url)
            if repo:
                result.append(GitHubIssueWithRepo(issue, repo))
            else:
                logger.warning(
                    f"Could not extract repo from URL for issue #{issue.number}: "
                    f"{issue.repo_url!r}"
                )
        return result

    def construct_issue_url(
        self,
        issue: JiraIssue | GitHubIssueProtocol,
        orchestration: Orchestration,
    ) -> str:
        """Construct a URL to the issue based on the trigger source.

        Args:
            issue: The issue (Jira or GitHub) to construct a URL for.
            orchestration: The orchestration containing trigger source info.

        Returns:
            The URL to the issue, or empty string if URL cannot be constructed.
        """
        trigger_source = getattr(orchestration.trigger, "source", "jira")

        if trigger_source == "jira":
            # Construct Jira URL from config base URL and issue key
            if self._config.jira_base_url:
                base_url = self._config.jira_base_url.rstrip("/")
                return f"{base_url}/browse/{issue.key}"
            return ""

        elif trigger_source == "github":
            # For GitHub, use the repo_url from the issue which contains the full URL
            if hasattr(issue, "repo_url") and issue.repo_url:
                return issue.repo_url
            # Fallback: construct from agent.github context if available
            github_ctx = orchestration.agent.github
            if github_ctx and github_ctx.org and github_ctx.repo:
                issue_num = str(issue.key).split("#")[-1] if "#" in str(issue.key) else ""
                if issue_num:
                    host = github_ctx.host or "github.com"
                    return f"https://{host}/{github_ctx.org}/{github_ctx.repo}/issues/{issue_num}"
            return ""

        return ""

    def group_orchestrations_by_source(
        self,
        orchestrations: list[Orchestration],
    ) -> tuple[list[Orchestration], list[Orchestration]]:
        """Group orchestrations by their trigger source.

        Args:
            orchestrations: List of all orchestrations.

        Returns:
            Tuple of (jira_orchestrations, github_orchestrations).
        """
        jira_orchestrations: list[Orchestration] = []
        github_orchestrations: list[Orchestration] = []

        for orch in orchestrations:
            if orch.trigger.source == "github":
                github_orchestrations.append(orch)
            else:
                jira_orchestrations.append(orch)

        return jira_orchestrations, github_orchestrations
