"""Claude Agent SDK executor for running agents on matched issues."""

from __future__ import annotations

import re
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentinel.github_poller import GitHubIssue
from sentinel.logging import get_logger, log_agent_summary
from sentinel.orchestration import Orchestration, Outcome, RetryConfig
from sentinel.poller import JiraIssue

# Type alias for issues from any supported source
AnyIssue = JiraIssue | GitHubIssue

if TYPE_CHECKING:
    from sentinel.agent_logger import AgentLogger

logger = get_logger(__name__)


@dataclass
class AgentRunResult:
    """Result of running a Claude agent.

    Attributes:
        response: The agent's response text.
        workdir: Path to the agent's working directory, if one was created.
    """

    response: str
    workdir: Path | None = None


def cleanup_workdir(
    workdir: Path | None,
    force: bool = False,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> bool:
    """Clean up an agent's working directory.

    Safely removes the working directory and all its contents using shutil.rmtree().
    Handles errors gracefully and logs any issues encountered.

    Args:
        workdir: Path to the working directory to clean up. If None, returns True.
        force: If True, enables aggressive cleanup with retries on recoverable errors
            (PermissionError, OSError) and uses ignore_errors on final attempt.
            Useful for handling transient file locks or permission issues.
        max_retries: Maximum number of retry attempts when force=True. Default is 3.
            Only applies when force=True.
        retry_delay: Delay in seconds between retry attempts when force=True.
            Default is 0.5 seconds. Only applies when force=True.

    Returns:
        True if cleanup was successful or workdir was None, False if an error occurred.
    """
    if workdir is None:
        return True

    if not force:
        # Original non-force behavior: single attempt with specific error handling
        try:
            if workdir.exists():
                shutil.rmtree(workdir)
                logger.debug(f"Cleaned up working directory: {workdir}")
            return True
        except PermissionError as e:
            logger.warning(f"Permission denied while cleaning up workdir {workdir}: {e}")
            return False
        except OSError as e:
            logger.warning(f"OS error while cleaning up workdir {workdir}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error while cleaning up workdir {workdir}: {e}")
            return False

    # Force mode: retry on recoverable errors, use ignore_errors on final attempt
    for attempt in range(1, max_retries + 1):
        try:
            if not workdir.exists():
                return True

            is_final_attempt = attempt == max_retries
            if is_final_attempt:
                # On final attempt, use ignore_errors to remove as much as possible
                shutil.rmtree(workdir, ignore_errors=True)
                if workdir.exists():
                    logger.warning(
                        f"Force cleanup could not fully remove workdir {workdir} "
                        f"after {max_retries} attempts (some files may remain)"
                    )
                    return False
            else:
                shutil.rmtree(workdir)

            logger.debug(f"Cleaned up working directory: {workdir} (force mode, attempt {attempt})")
            return True

        except (PermissionError, OSError) as e:
            if attempt < max_retries:
                logger.debug(
                    f"Recoverable error on cleanup attempt {attempt}/{max_retries} "
                    f"for {workdir}: {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                # Final attempt already uses ignore_errors, so this shouldn't happen
                # but handle it just in case
                logger.warning(
                    f"Force cleanup failed for workdir {workdir} after {max_retries} "
                    f"attempts: {e}"
                )
                return False
        except Exception as e:
            # Non-recoverable error, don't retry
            logger.warning(f"Unexpected error while cleaning up workdir {workdir}: {e}")
            return False

    return False


class ExecutionStatus(Enum):
    """Status of an agent execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of executing an agent on an issue.

    Attributes:
        status: The execution status (SUCCESS, FAILURE, or ERROR).
        response: The agent's response text.
        attempts: Number of attempts made.
        issue_key: The Jira issue key.
        orchestration_name: Name of the orchestration.
        matched_outcome: Name of the matched outcome (if outcomes are configured).
            None if no outcomes matched or outcomes are not configured.
    """

    status: ExecutionStatus
    response: str
    attempts: int
    issue_key: str
    orchestration_name: str
    matched_outcome: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return True if the execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


# Import exception classes from agent_clients.base for backward compatibility
# These exceptions are now defined in agent_clients.base but re-exported here
# to maintain existing import paths used throughout the codebase
from sentinel.agent_clients.base import AgentClientError, AgentTimeoutError


class AgentClient(ABC):
    """Abstract interface for Claude Agent SDK operations.

    This allows the executor to work with different implementations:
    - Real Claude Agent SDK client (production)
    - Mock client (testing)
    """

    @abstractmethod
    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        """Run a Claude agent with the given prompt and tools.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use.
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the CLI's default model.
            orchestration_name: Optional orchestration name for streaming log files.
            branch: Optional branch name to checkout/create before running the agent.
            create_branch: If True and branch doesn't exist, create it from base_branch.
            base_branch: Base branch to create new branches from. Defaults to "main".

        Returns:
            AgentRunResult containing the agent's response text and optional working directory path.

        Raises:
            AgentClientError: If the agent execution fails.
            AgentTimeoutError: If the agent execution times out.
        """
        pass


class AgentExecutor:
    """Executes Claude agents on Jira issues with retry logic."""

    def __init__(
        self,
        client: AgentClient,
        agent_logger: AgentLogger | None = None,
        cleanup_workdir_on_success: bool = True,
    ) -> None:
        """Initialize the executor with an agent client.

        Args:
            client: Agent client implementation.
            agent_logger: Optional logger for writing agent execution logs.
            cleanup_workdir_on_success: Whether to cleanup workdir after successful execution.
                Defaults to True. When False, workdirs are preserved for inspection.
                Failed executions always preserve workdirs regardless of this setting.
        """
        self.client = client
        self.agent_logger = agent_logger
        self.cleanup_workdir_on_success = cleanup_workdir_on_success

    def _build_template_variables(
        self, issue: AnyIssue, orchestration: Orchestration
    ) -> dict[str, str]:
        """Build the dictionary of template variables for prompt substitution.

        Args:
            issue: The issue to process (Jira or GitHub).
            orchestration: The orchestration configuration.

        Returns:
            Dictionary mapping variable names to their values.
        """
        # Get GitHub context from orchestration
        github = orchestration.agent.github

        # Build base variables dict with GitHub repository context
        variables: dict[str, str] = {
            # GitHub repository context
            "github_host": github.host if github else "",
            "github_org": github.org if github else "",
            "github_repo": github.repo if github else "",
        }

        if isinstance(issue, JiraIssue):
            # Format comments - last 3 comments, each truncated to 500 chars
            comments_text = ""
            if issue.comments:
                comment_parts = []
                for i, comment in enumerate(issue.comments[-3:], 1):
                    if len(comment) > 500:
                        comment_parts.append(f"{i}. {comment[:500]}...")
                    else:
                        comment_parts.append(f"{i}. {comment}")
                comments_text = "\n".join(comment_parts)

            # Add Jira-specific variables
            variables.update({
                # Jira context
                "jira_issue_key": issue.key,
                "jira_summary": issue.summary,
                "jira_description": issue.description or "",
                "jira_status": issue.status or "",
                "jira_assignee": issue.assignee or "",
                "jira_labels": ", ".join(issue.labels) if issue.labels else "",
                "jira_comments": comments_text,
                "jira_links": ", ".join(issue.links) if issue.links else "",
                # GitHub Issue variables (empty for Jira issues)
                "github_issue_number": "",
                "github_issue_title": "",
                "github_issue_body": "",
                "github_issue_state": "",
                "github_issue_author": "",
                "github_issue_assignees": "",
                "github_issue_labels": "",
                "github_issue_url": "",
                "github_is_pr": "",
                "github_pr_head": "",
                "github_pr_base": "",
                "github_pr_draft": "",
            })
        elif isinstance(issue, GitHubIssue):
            # Build GitHub issue URL if we have the necessary info
            github_issue_url = ""
            if github and github.host and github.org and github.repo:
                github_issue_url = (
                    f"https://{github.host}/{github.org}/{github.repo}/"
                    f"{'pull' if issue.is_pull_request else 'issues'}/{issue.number}"
                )

            # Add GitHub Issue-specific variables
            variables.update({
                # GitHub Issue context
                "github_issue_number": str(issue.number),
                "github_issue_title": issue.title,
                "github_issue_body": issue.body or "",
                "github_issue_state": issue.state,
                "github_issue_author": issue.author,
                "github_issue_assignees": ", ".join(issue.assignees) if issue.assignees else "",
                "github_issue_labels": ", ".join(issue.labels) if issue.labels else "",
                "github_issue_url": github_issue_url,
                # PR-specific fields
                "github_is_pr": str(issue.is_pull_request).lower(),
                "github_pr_head": issue.head_ref if issue.is_pull_request else "",
                "github_pr_base": issue.base_ref if issue.is_pull_request else "",
                "github_pr_draft": str(issue.draft).lower() if issue.is_pull_request else "",
                # Jira variables (empty for GitHub issues)
                "jira_issue_key": "",
                "jira_summary": "",
                "jira_description": "",
                "jira_status": "",
                "jira_assignee": "",
                "jira_labels": "",
                "jira_comments": "",
                "jira_links": "",
            })

        return variables

    def _expand_branch_pattern(
        self, issue: AnyIssue, orchestration: Orchestration
    ) -> str | None:
        """Expand template variables in the branch pattern.

        Returns None if no branch pattern is configured.

        Template variables available in the branch pattern:
        - {jira_issue_key}: The Jira issue key (e.g., "DS-290")
        - {github_issue_number}: The GitHub issue/PR number (e.g., "123")
        - {jira_summary}: Issue summary (use with caution - may need slugification)
        - {github_issue_title}: Issue/PR title (use with caution - may need slugification)

        Example:
        - Pattern: "feature/{jira_issue_key}"
        - Result: "feature/DS-290"

        Args:
            issue: The issue to process (Jira or GitHub).
            orchestration: The orchestration configuration.

        Returns:
            The expanded branch name, or None if no branch pattern is configured.
        """
        github = orchestration.agent.github
        if not github or not github.branch:
            return None

        variables = self._build_template_variables(issue, orchestration)

        # Expand variables in branch pattern
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return variables[var_name]
            return match.group(0)

        return re.sub(r"\{(\w+)\}", replace_var, github.branch)

    def build_prompt(self, issue: AnyIssue, orchestration: Orchestration) -> str:
        """Build the agent prompt by substituting template variables.

        Template variables available in the prompt:

        Jira context (populated for Jira issues):
        - {jira_issue_key}: The Jira issue key (e.g., "DS-123")
        - {jira_summary}: Issue summary/title
        - {jira_description}: Full issue description
        - {jira_status}: Current issue status
        - {jira_assignee}: Assignee display name
        - {jira_labels}: Comma-separated list of labels
        - {jira_comments}: Recent comments (last 3)
        - {jira_links}: Comma-separated list of linked issue keys

        GitHub repository context (from orchestration config):
        - {github_host}: GitHub host (e.g., "github.com")
        - {github_org}: GitHub organization
        - {github_repo}: GitHub repository name

        GitHub Issue context (populated for GitHub issues/PRs):
        - {github_issue_number}: The issue/PR number
        - {github_issue_title}: Issue/PR title
        - {github_issue_body}: Issue/PR body/description
        - {github_issue_state}: State (e.g., "open", "closed")
        - {github_issue_author}: Username of the author
        - {github_issue_assignees}: Comma-separated list of assignees
        - {github_issue_labels}: Comma-separated list of labels
        - {github_issue_url}: Full URL to the issue/PR
        - {github_is_pr}: "true" if this is a pull request, "false" otherwise
        - {github_pr_head}: Head branch reference (for PRs)
        - {github_pr_base}: Base branch reference (for PRs)
        - {github_pr_draft}: "true" if draft PR, "false" otherwise (for PRs)

        Args:
            issue: The issue to process (Jira or GitHub).
            orchestration: The orchestration configuration.

        Returns:
            The prompt with all template variables substituted.
        """
        template = orchestration.agent.prompt
        variables = self._build_template_variables(issue, orchestration)

        # Use a custom substitution to handle missing/unknown variables gracefully
        # This preserves any {unknown_var} that isn't in our variables dict
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return variables[var_name]
            # Keep unknown variables as-is (user might have literal braces)
            return match.group(0)

        # Match {variable_name} patterns
        prompt = re.sub(r"\{(\w+)\}", replace_var, template)

        return prompt

    def _matches_pattern(self, response: str, patterns: list[str]) -> bool:
        """Check if the response matches any of the patterns.

        Patterns can be:
        - Simple substring matches (default): "SUCCESS", "error"
        - Regex patterns (with prefix): "regex:error.*\\d+", "regex:^Task completed"

        The "regex:" prefix explicitly indicates a regex pattern. Without
        the prefix, patterns are matched as case-insensitive substrings.

        Args:
            response: The agent's response text.
            patterns: List of patterns to match. Use "regex:" prefix for regex.

        Returns:
            True if any pattern matches.
        """
        response_lower = response.lower()
        for pattern in patterns:
            if pattern.startswith("regex:"):
                # Explicit regex pattern
                regex_pattern = pattern[6:]  # Strip "regex:" prefix
                try:
                    if re.search(regex_pattern, response, re.IGNORECASE):
                        return True
                except re.error:
                    # If invalid regex, fall back to substring match
                    if regex_pattern.lower() in response_lower:
                        return True
            else:
                # Simple substring match (case-insensitive)
                if pattern.lower() in response_lower:
                    return True
        return False

    def _log_execution(
        self,
        issue_key: str,
        orchestration_name: str,
        prompt: str,
        response: str,
        status: ExecutionStatus,
        attempts: int,
        start_time: datetime,
    ) -> None:
        """Log the agent execution if a logger is configured.

        Args:
            issue_key: The Jira issue key.
            orchestration_name: Name of the orchestration.
            prompt: The prompt sent to the agent.
            response: The agent's response.
            status: The execution status.
            attempts: Number of attempts made.
            start_time: When execution started.
        """
        if self.agent_logger is None:
            return

        try:
            self.agent_logger.log_execution(
                issue_key=issue_key,
                orchestration_name=orchestration_name,
                prompt=prompt,
                response=response,
                status=status,
                attempts=attempts,
                start_time=start_time,
                end_time=datetime.now(),
            )
        except Exception as e:
            # Log but don't fail the execution due to logging errors
            logger.warning(f"Failed to write agent execution log: {e}")

    def _determine_outcome(self, response: str, outcomes: list[Outcome]) -> Outcome | None:
        """Determine which outcome matches the response.

        Args:
            response: The agent's response text.
            outcomes: List of outcome configurations.

        Returns:
            The matched Outcome, or None if no outcome patterns match.
        """
        for outcome in outcomes:
            if self._matches_pattern(response, outcome.patterns):
                return outcome
        return None

    def _determine_status(
        self,
        response: str,
        retry_config: RetryConfig,
        outcomes: list[Outcome] | None = None,
    ) -> tuple[ExecutionStatus, str | None]:
        """Determine the execution status and matched outcome from the agent response.

        Args:
            response: The agent's response text.
            retry_config: Retry configuration with success/failure patterns and default_status.
            outcomes: Optional list of outcome configurations.

        Returns:
            Tuple of (ExecutionStatus, matched_outcome_name or None).
        """
        # If outcomes are configured, use outcome-based logic
        if outcomes:
            # Check outcome patterns FIRST - explicit success markers like "SUCCESS"
            # take precedence over generic words like "error" in context
            matched_outcome = self._determine_outcome(response, outcomes)
            if matched_outcome:
                return ExecutionStatus.SUCCESS, matched_outcome.name

            # No outcome matched - check failure patterns (these trigger retries)
            if self._matches_pattern(response, retry_config.failure_patterns):
                return ExecutionStatus.FAILURE, None

            # No outcome or failure matched - check default_outcome
            if retry_config.default_outcome:
                # Special "failure" keyword triggers failure mechanism
                if retry_config.default_outcome.lower() == "failure":
                    logger.warning(
                        "Response did not match any outcome patterns, "
                        "using default_outcome 'failure' - triggering failure mechanism"
                    )
                    return ExecutionStatus.FAILURE, None

                # Find the named default outcome
                for outcome in outcomes:
                    if outcome.name == retry_config.default_outcome:
                        logger.warning(
                            "Response did not match any outcome patterns, "
                            "using default_outcome: %s",
                            retry_config.default_outcome,
                        )
                        return ExecutionStatus.SUCCESS, outcome.name

                # Default outcome name not found in outcomes - treat as failure
                logger.warning(
                    "default_outcome '%s' not found in outcomes, treating as failure",
                    retry_config.default_outcome,
                )
                return ExecutionStatus.FAILURE, None

            # No default_outcome configured - treat as failure
            # This is safer than silently succeeding with an arbitrary outcome
            logger.warning(
                "Response did not match any outcome patterns and no default_outcome "
                "configured, treating as failure"
            )
            return ExecutionStatus.FAILURE, None

        # Legacy logic when outcomes are not configured
        # Check success patterns first
        if self._matches_pattern(response, retry_config.success_patterns):
            return ExecutionStatus.SUCCESS, None

        # Check failure patterns
        if self._matches_pattern(response, retry_config.failure_patterns):
            return ExecutionStatus.FAILURE, None

        # Use configured default status when no patterns match
        default = retry_config.default_status.upper()
        logger.warning(
            "Response did not match success or failure patterns, using default_status: %s",
            default,
        )
        status = ExecutionStatus.SUCCESS if default == "SUCCESS" else ExecutionStatus.FAILURE
        return status, None

    def execute(
        self,
        issue: AnyIssue,
        orchestration: Orchestration,
    ) -> ExecutionResult:
        """Execute an agent on an issue with retry logic.

        Args:
            issue: The issue to process (Jira or GitHub).
            orchestration: The orchestration configuration.

        Returns:
            ExecutionResult with status, response, and attempt count.
        """
        retry_config = orchestration.retry
        max_attempts = retry_config.max_attempts
        tools = orchestration.agent.tools
        timeout_seconds = orchestration.agent.timeout_seconds
        model = orchestration.agent.model

        # Build context for the agent
        context: dict[str, Any] = {}
        github = orchestration.agent.github
        if github:
            context["github"] = {
                "host": github.host,
                "org": github.org,
                "repo": github.repo,
            }

        prompt = self.build_prompt(issue, orchestration)
        last_response = ""
        last_status = ExecutionStatus.ERROR
        last_matched_outcome: str | None = None
        last_workdir: Path | None = None
        start_time = datetime.now()

        # Get outcomes if configured
        outcomes = orchestration.outcomes if orchestration.outcomes else None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                f"Executing agent for {issue.key} with orchestration "
                f"'{orchestration.name}' (attempt {attempt}/{max_attempts})",
                extra={
                    "issue_key": issue.key,
                    "orchestration": orchestration.name,
                    "attempt": attempt,
                },
            )

            try:
                # Get branch configuration
                branch = self._expand_branch_pattern(issue, orchestration)

                run_result = self.client.run_agent(
                    prompt,
                    tools,
                    context,
                    timeout_seconds,
                    issue_key=issue.key,
                    model=model,
                    orchestration_name=orchestration.name,
                    branch=branch,
                    create_branch=github.create_branch if github else False,
                    base_branch=github.base_branch if github else "main",
                )
                response = run_result.response
                last_response = response
                last_workdir = run_result.workdir
                status, matched_outcome = self._determine_status(response, retry_config, outcomes)
                last_status = status
                last_matched_outcome = matched_outcome

                # Log agent summary
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status=status.value.upper(),
                    response=response,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )

                if status == ExecutionStatus.SUCCESS:
                    result = ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        response=response,
                        attempts=attempt,
                        issue_key=issue.key,
                        orchestration_name=orchestration.name,
                        matched_outcome=matched_outcome,
                    )
                    self._log_execution(
                        issue.key,
                        orchestration.name,
                        prompt,
                        response,
                        status,
                        attempt,
                        start_time,
                    )
                    # Cleanup workdir on success if enabled
                    if self.cleanup_workdir_on_success and last_workdir:
                        logger.debug(
                            f"Cleaning up workdir after successful execution: {last_workdir}"
                        )
                        cleanup_workdir(last_workdir)
                    elif not self.cleanup_workdir_on_success and last_workdir:
                        logger.debug(
                            f"Workdir preserved at {last_workdir} (cleanup_workdir_on_success=False)"
                        )
                    return result

                if status == ExecutionStatus.FAILURE and attempt < max_attempts:
                    logger.warning(
                        f"Agent execution failed for {issue.key} on attempt {attempt}, retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                        },
                    )

            except AgentTimeoutError as e:
                last_response = f"Timeout: {e}"
                last_status = ExecutionStatus.ERROR
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status="TIMEOUT",
                    response=str(e),
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent timed out for {issue.key} on attempt {attempt}, retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
            except AgentClientError as e:
                last_response = str(e)
                last_status = ExecutionStatus.ERROR
                log_agent_summary(
                    logger=logger,
                    issue_key=issue.key,
                    orchestration=orchestration.name,
                    status="ERROR",
                    response=str(e),
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent client error for {issue.key} on attempt {attempt}: {e}, "
                        f"retrying...",
                        extra={
                            "issue_key": issue.key,
                            "orchestration": orchestration.name,
                            "attempt": attempt,
                        },
                    )

        # All attempts exhausted
        result = ExecutionResult(
            status=last_status,
            response=last_response,
            attempts=max_attempts,
            issue_key=issue.key,
            orchestration_name=orchestration.name,
            matched_outcome=last_matched_outcome,
        )
        self._log_execution(
            issue.key,
            orchestration.name,
            prompt,
            last_response,
            last_status,
            max_attempts,
            start_time,
        )
        return result
