"""Claude Agent SDK executor for running agents on matched issues."""

from __future__ import annotations

import re
import shutil
import time
import unicodedata
from collections.abc import Coroutine
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from sentinel.agent_clients.base import AgentClient, AgentClientError, AgentTimeoutError
from sentinel.branch_validation import validate_runtime_branch_name
from sentinel.github_poller import GitHubIssue
from sentinel.logging import get_logger, log_agent_summary
from sentinel.orchestration import (
    GitHubContext,
    Orchestration,
    Outcome,
    RetryConfig,
    get_effective_timeout,
)
from sentinel.poller import JiraIssue

# Type alias for issues from any supported source
AnyIssue = JiraIssue | GitHubIssue

if TYPE_CHECKING:
    from sentinel.agent_logger import AgentLogger

logger = get_logger(__name__)


def slugify(text: str) -> str:
    """Convert text to a branch-safe slug format.

    Transforms text to lowercase, normalizes unicode, replaces spaces/underscores
    with hyphens, removes git-invalid characters, and strips leading/trailing hyphens.

    Example: "Add login button" -> "add-login-button"
    """
    if not text:
        return ""

    # Normalize unicode characters (e.g., é -> e)
    normalized = unicodedata.normalize("NFKD", text)
    # Remove combining characters (accents, etc.)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    slug = ascii_text.lower()

    # Replace spaces and underscores with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)

    # Remove characters that are invalid in git branch names
    # Keep only alphanumeric, hyphens, dots, and forward slashes
    # Git disallows: ~ ^ : ? * [ \ space and control characters
    # Remove invalid git branch characters including @ { }
    slug = re.sub(r"[~^:?*\[\]\\@{}!\"'#$%&()+,;<>=|`]", "", slug)

    # Replace consecutive dots with single dot
    slug = re.sub(r"\.{2,}", ".", slug)

    # Collapse multiple consecutive hyphens into one
    slug = re.sub(r"-{2,}", "-", slug)

    # Strip leading/trailing hyphens and dots
    slug = slug.strip("-.")

    return slug


def _compute_slug(value: str) -> str:
    """Compute a slug for computed fields."""
    return slugify(value)


def _format_comments(comments: list[str], limit: int = 3, max_length: int = 500) -> str:
    """Format comments for template variable.

    Args:
        comments: List of comment strings.
        limit: Maximum number of comments to include (from the end).
        max_length: Maximum length per comment before truncation.

    Returns:
        Formatted string with numbered comments.
    """
    if not comments:
        return ""

    comment_parts = []
    for i, comment in enumerate(comments[-limit:], 1):
        if len(comment) > max_length:
            comment_parts.append(f"{i}. {comment[:max_length]}...")
        else:
            comment_parts.append(f"{i}. {comment}")
    return "\n".join(comment_parts)


def _format_list(items: list[str] | None) -> str:
    """Format a list as a comma-separated string."""
    return ", ".join(items) if items else ""


@dataclass
class TemplateContext:
    """Data-driven template context using dataclass introspection.

    This dataclass holds all template variables available for prompt and branch
    pattern expansion. Using dataclass introspection, we can automatically
    generate the variables dict without manual mapping.

    Field metadata:
    - source: "jira", "github", or "common" - indicates which source type the field belongs to.
              Used by factory methods for auto-initialization of non-applicable fields.

    Computed fields (marked with metadata) are derived from other fields:
    - jira_summary_slug: Derived from jira_summary
    - github_issue_title_slug: Derived from github_issue_title
    - jira_labels (formatted): Derived from _jira_labels_list
    - jira_comments (formatted): Derived from _jira_comments_list
    - jira_links (formatted): Derived from _jira_links_list
    - github_issue_assignees (formatted): Derived from _github_issue_assignees_list
    - github_issue_labels (formatted): Derived from _github_issue_labels_list
    """

    # GitHub repository context (always available)
    github_host: str = field(default="", metadata={"source": "common"})
    github_org: str = field(default="", metadata={"source": "common"})
    github_repo: str = field(default="", metadata={"source": "common"})

    # Jira context
    jira_issue_key: str = field(default="", metadata={"source": "jira"})
    jira_summary: str = field(default="", metadata={"source": "jira"})
    jira_description: str = field(default="", metadata={"source": "jira"})
    jira_status: str = field(default="", metadata={"source": "jira"})
    jira_assignee: str = field(default="", metadata={"source": "jira"})
    jira_epic_key: str = field(default="", metadata={"source": "jira"})
    jira_parent_key: str = field(default="", metadata={"source": "jira"})

    # Jira computed/formatted fields (source lists stored privately)
    _jira_labels_list: list[str] = field(
        default_factory=list, repr=False, metadata={"source": "jira"}
    )
    _jira_comments_list: list[str] = field(
        default_factory=list, repr=False, metadata={"source": "jira"}
    )
    _jira_links_list: list[str] = field(
        default_factory=list, repr=False, metadata={"source": "jira"}
    )

    # GitHub Issue context
    github_issue_number: str = field(default="", metadata={"source": "github"})
    github_issue_title: str = field(default="", metadata={"source": "github"})
    github_issue_body: str = field(default="", metadata={"source": "github"})
    github_issue_state: str = field(default="", metadata={"source": "github"})
    github_issue_author: str = field(default="", metadata={"source": "github"})
    github_issue_url: str = field(default="", metadata={"source": "github"})
    github_is_pr: str = field(default="", metadata={"source": "github"})
    github_pr_head: str = field(default="", metadata={"source": "github"})
    github_pr_base: str = field(default="", metadata={"source": "github"})
    github_pr_draft: str = field(default="", metadata={"source": "github"})
    github_parent_issue_number: str = field(default="", metadata={"source": "github"})

    # GitHub computed/formatted fields (source lists stored privately)
    _github_issue_assignees_list: list[str] = field(
        default_factory=list, repr=False, metadata={"source": "github"}
    )
    _github_issue_labels_list: list[str] = field(
        default_factory=list, repr=False, metadata={"source": "github"}
    )

    def to_dict(self) -> dict[str, str]:
        """Convert to template variables dict using dataclass introspection.

        Automatically iterates over all fields and includes:
        - Regular string fields (excluding private _ prefixed ones)
        - Computed slug fields
        - Formatted list fields

        Returns:
            Dict mapping template variable names to string values.
        """
        variables: dict[str, str] = {}

        for f in fields(self):
            # Skip private fields (they're source data for computed fields)
            if f.name.startswith("_"):
                continue

            value = getattr(self, f.name)
            variables[f.name] = value if isinstance(value, str) else str(value)

        # Add computed slug fields
        variables["jira_summary_slug"] = _compute_slug(self.jira_summary)
        variables["github_issue_title_slug"] = _compute_slug(self.github_issue_title)

        # Add formatted list fields
        variables["jira_labels"] = _format_list(self._jira_labels_list)
        variables["jira_comments"] = _format_comments(self._jira_comments_list)
        variables["jira_links"] = _format_list(self._jira_links_list)
        variables["github_issue_assignees"] = _format_list(self._github_issue_assignees_list)
        variables["github_issue_labels"] = _format_list(self._github_issue_labels_list)

        return variables

    @classmethod
    def from_jira_issue(
        cls, issue: JiraIssue, github: GitHubContext | None = None
    ) -> TemplateContext:
        """Create a TemplateContext from a Jira issue.

        Uses field metadata to auto-initialize GitHub-specific fields to their
        default values (empty strings/lists), reducing boilerplate.

        Args:
            issue: The Jira issue to extract data from.
            github: Optional GitHub context from orchestration.

        Returns:
            TemplateContext populated with Jira issue data.
        """
        # Only specify Jira-specific and common fields - GitHub fields use defaults via metadata
        return cls(
            # GitHub repository context (common fields)
            github_host=github.host if github else "",
            github_org=github.org if github else "",
            github_repo=github.repo if github else "",
            # Jira context
            jira_issue_key=issue.key,
            jira_summary=issue.summary,
            jira_description=issue.description or "",
            jira_status=issue.status or "",
            jira_assignee=issue.assignee or "",
            jira_epic_key=issue.epic_key or "",
            jira_parent_key=issue.parent_key or "",
            _jira_labels_list=issue.labels or [],
            _jira_comments_list=issue.comments or [],
            _jira_links_list=issue.links or [],
            # GitHub Issue fields auto-initialized to defaults via field metadata
        )

    @classmethod
    def from_github_issue(
        cls, issue: GitHubIssue, github: GitHubContext | None = None
    ) -> TemplateContext:
        """Create a TemplateContext from a GitHub issue.

        Uses field metadata to auto-initialize Jira-specific fields to their
        default values (empty strings/lists), reducing boilerplate.

        Args:
            issue: The GitHub issue to extract data from.
            github: Optional GitHub context from orchestration.

        Returns:
            TemplateContext populated with GitHub issue data.
        """
        # Build GitHub issue URL if we have the necessary info
        github_issue_url = ""
        if github and github.host and github.org and github.repo:
            github_issue_url = (
                f"https://{github.host}/{github.org}/{github.repo}/"
                f"{'pull' if issue.is_pull_request else 'issues'}/{issue.number}"
            )

        # Only specify GitHub-specific and common fields - Jira fields use defaults via metadata
        return cls(
            # GitHub repository context (common fields)
            github_host=github.host if github else "",
            github_org=github.org if github else "",
            github_repo=github.repo if github else "",
            # GitHub Issue context
            github_issue_number=str(issue.number),
            github_issue_title=issue.title,
            github_issue_body=issue.body or "",
            github_issue_state=issue.state,
            github_issue_author=issue.author,
            github_issue_url=github_issue_url,
            github_is_pr=str(issue.is_pull_request).lower(),
            github_pr_head=issue.head_ref if issue.is_pull_request else "",
            github_pr_base=issue.base_ref if issue.is_pull_request else "",
            github_pr_draft=str(issue.draft).lower() if issue.is_pull_request else "",
            github_parent_issue_number=str(issue.parent_issue_number)
            if issue.parent_issue_number
            else "",
            _github_issue_assignees_list=issue.assignees or [],
            _github_issue_labels_list=issue.labels or [],
            # Jira fields auto-initialized to defaults via field metadata
        )


def cleanup_workdir(
    workdir: Path | None,
    force: bool = False,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> bool:
    """Clean up an agent's working directory.

    Returns True if cleanup succeeded or workdir is None. When force=True,
    retries on PermissionError/OSError up to max_retries times with retry_delay
    between attempts, using ignore_errors on final attempt.
    """
    if workdir is None:
        return True

    if not force:
        # Original non-force behavior: single attempt with specific error handling
        try:
            if workdir.exists():
                shutil.rmtree(workdir)
                logger.debug("Cleaned up working directory: %s", workdir)
            return True
        except PermissionError as e:
            logger.warning("Permission denied while cleaning up workdir %s: %s", workdir, e)
            return False
        except OSError as e:
            logger.warning("OS error while cleaning up workdir %s: %s", workdir, e)
            return False
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            # Handle unexpected argument/runtime errors that shouldn't occur in normal operation
            # - TypeError/ValueError: invalid arguments
            # - AttributeError: path object missing expected methods
            # - RuntimeError: unexpected runtime issues
            logger.error(
                "Unexpected %s while cleaning up workdir %s: %s",
                type(e).__name__,
                workdir,
                e,
            )
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
                        "Force cleanup could not fully remove workdir %s "
                        "after %s attempts (some files may remain)",
                        workdir,
                        max_retries,
                    )
                    return False
            else:
                shutil.rmtree(workdir)

            logger.debug(
                "Cleaned up working directory: %s (force mode, attempt %s)", workdir, attempt
            )
            return True

        except (PermissionError, OSError) as e:
            if attempt < max_retries:
                logger.debug(
                    "Recoverable error on cleanup attempt %s/%s "
                    "for %s: %s. Retrying in %ss...",
                    attempt,
                    max_retries,
                    workdir,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
            else:
                # Final attempt already uses ignore_errors, so this shouldn't happen
                # but handle it just in case
                logger.warning(
                    "Force cleanup failed for workdir %s after %s "
                    "attempts: %s",
                    workdir,
                    max_retries,
                    e,
                )
                return False
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            # Non-recoverable argument/runtime error, don't retry
            # - TypeError/ValueError: invalid arguments
            # - AttributeError: path object missing expected methods
            # - RuntimeError: unexpected runtime issues
            logger.error(
                "Unexpected %s while cleaning up workdir %s: %s",
                type(e).__name__,
                workdir,
                e,
            )
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
        input_tokens: Number of input tokens consumed by the agent.
        output_tokens: Number of output tokens generated by the agent.
        total_cost_usd: Total cost in USD for the agent execution.
    """

    status: ExecutionStatus
    response: str
    attempts: int
    issue_key: str
    orchestration_name: str
    matched_outcome: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Return True if the execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


# Explicit async return type alias for executor methods (DS-533).
# This type alias makes the async nature of the execute method more discoverable.
ExecutionCoroutine: TypeAlias = Coroutine[Any, Any, ExecutionResult]
"""Coroutine type for async execution methods returning ExecutionResult."""

__all__ = [
    "AgentExecutor",
    "ExecutionCoroutine",
    "ExecutionResult",
    "ExecutionStatus",
]


class AgentExecutor:
    """Executes Claude agents on Jira issues with retry logic.

    This executor is async-native to enable proper async composition. The execute()
    method is async and should be awaited. Callers that need sync execution should
    use asyncio.run() at the top level of their application.

    Architecture Decision (DS-509):
    The AgentClient.run_agent() method is async because it interacts with async APIs
    (Claude SDK). Rather than creating a new event loop per execution (which is
    inefficient and prevents async composition), the executor is designed to be
    async-native. This means:
    - AgentExecutor.execute() is an async method
    - Callers should use asyncio.run() only at application entry points
    - Multiple executions can be composed with asyncio.gather() etc.
    """

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
        """Build template variables dict from issue and orchestration context.

        Uses TemplateContext dataclass with introspection to automatically
        generate the variables dict. This reduces code duplication and makes
        it easier to add new template variables.

        Returns a dict with keys for Jira context (jira_issue_key, jira_summary,
        jira_summary_slug, etc.) and GitHub context (github_issue_number,
        github_issue_title, github_host, github_org, github_repo, etc.).
        Variables for the non-matching source type are set to empty strings.
        """
        github = orchestration.agent.github

        if isinstance(issue, JiraIssue):
            context = TemplateContext.from_jira_issue(issue, github)
        elif isinstance(issue, GitHubIssue):
            context = TemplateContext.from_github_issue(issue, github)
        else:
            # Fallback for unknown issue types - return empty context
            context = TemplateContext(
                github_host=github.host if github else "",
                github_org=github.org if github else "",
                github_repo=github.repo if github else "",
            )

        return context.to_dict()

    def _expand_branch_pattern(
        self,
        issue: AnyIssue,
        orchestration: Orchestration,
        strict: bool = False,
    ) -> str | None:
        """Expand template variables in the branch pattern.

        Returns None if no branch pattern is configured. Uses variables from
        _build_template_variables(). See GitHubContext docstring for available
        template variables and usage notes.

        After template substitution, the fully-resolved branch name is validated
        against git branch naming rules. If validation fails, a warning is logged
        and None is returned to prevent using an invalid branch name.

        Args:
            issue: The issue (Jira or GitHub) to extract variables from.
            orchestration: The orchestration configuration.
            strict: If True, raise ValueError for unknown template variables.
                If False (default), log a warning and preserve unknown variables.

        Raises:
            ValueError: If strict=True and an unknown template variable is found.
        """
        github = orchestration.agent.github
        if not github or not github.branch:
            return None

        variables = self._build_template_variables(issue, orchestration)
        unknown_vars: list[str] = []

        # Expand variables in branch pattern
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return variables[var_name]
            unknown_vars.append(var_name)
            return match.group(0)

        expanded_branch = re.sub(r"\{(\w+)\}", replace_var, github.branch)

        # Handle unknown template variables
        if unknown_vars:
            available_vars = sorted(variables.keys())
            if strict:
                raise ValueError(
                    f"Unknown template variable(s) in branch pattern: {unknown_vars}. "
                    f"Available variables: {available_vars}"
                )
            logger.warning(
                "Unknown template variable(s) in branch pattern '%s': %s. "
                "Available variables: %s. "
                "Variable(s) will be preserved as-is in the output.",
                github.branch,
                unknown_vars,
                available_vars,
            )

        # Validate the fully-resolved branch name at runtime
        validation_result = validate_runtime_branch_name(expanded_branch)
        if not validation_result.is_valid:
            logger.warning(
                "Invalid expanded branch name '%s' (from pattern '%s'): %s. "
                "Branch will not be used.",
                expanded_branch,
                github.branch,
                validation_result.error_message,
            )
            return None

        return expanded_branch

    def build_prompt(
        self,
        issue: AnyIssue,
        orchestration: Orchestration,
        strict: bool = False,
    ) -> str:
        """Build the agent prompt by substituting template variables.

        Uses variables from _build_template_variables(). Unknown variables
        are preserved as-is in the output with a warning logged. See GitHubContext
        docstring for available template variables.

        Args:
            issue: The issue (Jira or GitHub) to extract variables from.
            orchestration: The orchestration configuration.
            strict: If True, raise ValueError for unknown template variables.
                If False (default), log a warning and preserve unknown variables.

        Returns:
            The prompt with template variables substituted.

        Raises:
            ValueError: If strict=True and an unknown template variable is found.
        """
        template = orchestration.agent.prompt
        variables = self._build_template_variables(issue, orchestration)
        unknown_vars: list[str] = []

        # Use a custom substitution to handle missing/unknown variables gracefully
        # This preserves any {unknown_var} that isn't in our variables dict
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return variables[var_name]
            # Track unknown variable and keep as-is
            unknown_vars.append(var_name)
            return match.group(0)

        # Match {variable_name} patterns
        prompt = re.sub(r"\{(\w+)\}", replace_var, template)

        # Handle unknown template variables
        if unknown_vars:
            available_vars = sorted(variables.keys())
            if strict:
                raise ValueError(
                    f"Unknown template variable(s) in prompt: {unknown_vars}. "
                    f"Available variables: {available_vars}"
                )
            logger.warning(
                "Unknown template variable(s) in prompt: %s. "
                "Available variables: %s. "
                "Variable(s) will be preserved as-is in the output.",
                unknown_vars,
                available_vars,
            )

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
                except re.error as e:
                    # If invalid regex, fall back to substring match
                    logger.warning(
                        "Invalid regex pattern '%s': %s. Falling back to substring match.",
                        regex_pattern,
                        e,
                    )
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
        except OSError as e:
            # Log but don't fail the execution due to file I/O errors
            logger.error("Failed to write agent execution log due to I/O error: %s", e)
        except (TypeError, ValueError) as e:
            # Log but don't fail the execution due to serialization errors
            logger.error("Failed to write agent execution log due to data error: %s", e)

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

    async def execute(
        self,
        issue: AnyIssue,
        orchestration: Orchestration,
    ) -> ExecutionResult:
        """Execute an agent on an issue with retry logic.

        This is an async method to enable proper async composition and avoid
        creating new event loops per call. Callers should use asyncio.run()
        at their entry points when needed.

        Note:
            This method returns a coroutine (``Coroutine[Any, Any, ExecutionResult]``).
            The ``ExecutionCoroutine`` type alias is provided for explicit type
            annotations where needed (DS-533).

        Args:
            issue: The issue to process (Jira or GitHub).
            orchestration: The orchestration configuration.

        Returns:
            ExecutionResult with status, response, and attempt count.
        """
        retry_config = orchestration.retry
        max_attempts = retry_config.max_attempts
        # Apply Agent Teams timeout multiplier/minimum when applicable (DS-697)
        timeout_seconds = get_effective_timeout(orchestration.agent)
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

        # Build prompt with strict mode from orchestration config
        strict = orchestration.agent.strict_template_variables
        prompt = self.build_prompt(issue, orchestration, strict=strict)
        last_response = ""
        last_status = ExecutionStatus.ERROR
        last_matched_outcome: str | None = None
        last_workdir: Path | None = None
        last_input_tokens = 0
        last_output_tokens = 0
        last_total_cost_usd = 0.0
        start_time = datetime.now()

        # Get outcomes if configured
        outcomes = orchestration.outcomes if orchestration.outcomes else None

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "Executing agent for %s with orchestration "
                "'%s' (attempt %s/%s)",
                issue.key,
                orchestration.name,
                attempt,
                max_attempts,
                extra={
                    "issue_key": issue.key,
                    "orchestration": orchestration.name,
                    "attempt": attempt,
                },
            )

            try:
                # Get branch configuration with strict mode from orchestration config
                branch = self._expand_branch_pattern(issue, orchestration, strict=strict)

                # Await the async agent execution directly
                # The executor is async-native, avoiding the need to create
                # new event loops per execution (DS-509)
                run_result = await self.client.run_agent(
                    prompt,
                    context,
                    timeout_seconds,
                    issue_key=issue.key,
                    model=model,
                    orchestration_name=orchestration.name,
                    branch=branch,
                    create_branch=github.create_branch if github else False,
                    base_branch=github.base_branch if github else "main",
                    agent_teams=orchestration.agent.agent_teams,
                )
                response = run_result.response
                last_response = response
                last_workdir = run_result.workdir

                # Extract usage data from AgentRunResult
                if run_result.usage is not None:
                    last_input_tokens = run_result.usage.input_tokens
                    last_output_tokens = run_result.usage.output_tokens
                    last_total_cost_usd = run_result.usage.total_cost_usd

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
                        input_tokens=last_input_tokens,
                        output_tokens=last_output_tokens,
                        total_cost_usd=last_total_cost_usd,
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
                            "Cleaning up workdir after successful execution: %s", last_workdir
                        )
                        cleanup_workdir(last_workdir)
                    elif not self.cleanup_workdir_on_success and last_workdir:
                        logger.debug(
                            "Workdir preserved at %s (cleanup_workdir_on_success=False)",
                            last_workdir,
                        )
                    return result

                if status == ExecutionStatus.FAILURE and attempt < max_attempts:
                    logger.warning(
                        "Agent execution failed for %s on attempt %s, retrying...",
                        issue.key,
                        attempt,
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
                        "Agent timed out for %s on attempt %s, retrying...",
                        issue.key,
                        attempt,
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
                        "Agent client error for %s on attempt %s: %s, "
                        "retrying...",
                        issue.key,
                        attempt,
                        e,
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
            input_tokens=last_input_tokens,
            output_tokens=last_output_tokens,
            total_cost_usd=last_total_cost_usd,
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
