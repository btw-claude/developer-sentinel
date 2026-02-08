"""Orchestration configuration schema and loading."""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml

from sentinel.branch_validation import validate_branch_name_core
from sentinel.logging import get_logger
from sentinel.types import (
    AgentType,
    AgentTypeLiteral,
    CursorMode,
    CursorModeLiteral,
    TriggerSource,
    TriggerSourceLiteral,
)

logger = get_logger(__name__)

# Agent Teams timeout constants (DS-697, DS-701)
# Agent Teams orchestrations involve multiple Claude Code processes (team lead +
# teammates) coordinating via shared task lists and mailboxes, which takes
# significantly longer than single-agent runs.
#
# Both constants can be overridden via environment variables so operators can
# tune timeout behaviour per deployment without code changes (DS-701).

_DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER: int = 3
_DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS: int = 900


def _parse_env_int(name: str, default: int, *, min_value: int = 1) -> int:
    """Parse an integer from an environment variable, falling back to *default*.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is unset or empty.
        min_value: Minimum accepted value (inclusive). Defaults to 1.

    Returns:
        The parsed integer value.

    Raises:
        ValueError: If the environment variable is set to a non-integer value
            or to a value below *min_value*.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {raw!r}"
        ) from None

    if value < min_value:
        raise ValueError(
            f"Environment variable {name} must be >= {min_value}, got {value}"
        )

    logger.info(
        "Using %s=%d from environment (default: %d)", name, value, default
    )
    return value


AGENT_TEAMS_TIMEOUT_MULTIPLIER: int = _parse_env_int(
    "AGENT_TEAMS_TIMEOUT_MULTIPLIER", _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER
)
"""Multiplier applied to timeout_seconds when agent_teams is enabled.

When an orchestration has ``agent_teams: true``, the effective timeout is
``timeout_seconds * AGENT_TEAMS_TIMEOUT_MULTIPLIER`` to account for the
coordination overhead of multiple Claude Code processes.

Override via the ``AGENT_TEAMS_TIMEOUT_MULTIPLIER`` environment variable.
Default: 3.
"""

AGENT_TEAMS_MIN_TIMEOUT_SECONDS: int = _parse_env_int(
    "AGENT_TEAMS_MIN_TIMEOUT_SECONDS", _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS
)
"""Minimum recommended timeout (in seconds) for agent_teams-enabled orchestrations.

If a configured ``timeout_seconds`` (before multiplier) falls below this
threshold, a warning is logged advising the user to increase it.  The value
of 900 seconds (15 minutes) reflects the typical coordination overhead of
Agent Teams orchestrations.

Override via the ``AGENT_TEAMS_MIN_TIMEOUT_SECONDS`` environment variable.
Default: 900.
"""

# Note: Branch validation is now handled by the shared branch_validation module


class ValidationResult(NamedTuple):
    """Result of a validation operation with is_valid flag and error_message."""

    is_valid: bool
    error_message: str

    @classmethod
    def success(cls) -> ValidationResult:
        """Create a successful validation result."""
        return cls(is_valid=True, error_message="")

    @classmethod
    def failure(cls, message: str) -> ValidationResult:
        """Create a failed validation result with the given error message."""
        return cls(is_valid=False, error_message=message)


@dataclass
class TriggerConfig:
    """Configuration for what triggers an orchestration.

    Supports Jira triggers (source=TriggerSource.JIRA) with project, jql_filter,
    tags fields, and GitHub triggers (source=TriggerSource.GITHUB) with
    project_number, project_scope, project_owner, project_filter, and labels fields.

    Multiple tags/labels use AND logic (issue must have all specified).

    Attributes:
        source: Trigger source type (TriggerSource.JIRA or TriggerSource.GITHUB).
        project: Jira project key (validated for safe characters: alphanumeric,
            underscores, hyphens, must start with letter).
        jql_filter: Custom JQL fragment for advanced filtering. Security validated
            for structural integrity (balanced parentheses/quotes, no null bytes,
            length limits). Should only be configured by trusted administrators.
            Full JQL syntax validation is performed by the Jira API.
        tags: List of Jira labels the issue must have (sanitized for safe use in JQL).

    Security Note:
        The jql_filter field accepts raw JQL fragments and provides defense-in-depth
        validation. However, it should only be configured by trusted administrators
        as it directly influences the JQL query sent to Jira.
    """

    source: TriggerSourceLiteral = "jira"
    project: str = ""
    jql_filter: str = ""
    tags: list[str] = field(default_factory=list)
    # GitHub Project-based fields
    project_number: int | None = None
    project_scope: Literal["org", "user"] = "org"
    project_owner: str = ""
    project_filter: str = ""
    labels: list[str] = field(default_factory=list)


@dataclass
class GitHubContext:
    """GitHub repository context for agent operations.

    Attributes:
        host: GitHub host (e.g., "github.com" or enterprise host).
        org: GitHub organization or user name.
        repo: Repository name.
        branch: Branch pattern supporting template variables (e.g., "feature/{jira_issue_key}").
        create_branch: Whether to auto-create the branch if it doesn't exist.
            Requires a non-empty branch pattern when True.
        base_branch: Base branch for new branch creation (default: "main").

    Template Variables:
        Use ``{variable_name}`` syntax in branch patterns and prompts.
        This is the authoritative reference for all available template variables.

        Jira Variables (populated when source is Jira, empty for GitHub triggers):
            - ``{jira_issue_key}``: The Jira issue key (e.g., "PROJ-123").
            - ``{jira_summary}``: Issue title/summary text.
            - ``{jira_summary_slug}``: Branch-safe version of summary (computed, see Slug
              Transformation below).
            - ``{jira_description}``: Full issue description.
            - ``{jira_status}``: Current status name.
            - ``{jira_assignee}``: Assignee display name.
            - ``{jira_epic_key}``: Parent epic key if linked.
            - ``{jira_parent_key}``: Parent issue key if subtask.
            - ``{jira_labels}``: Comma-separated list of labels (see List Formatting below).
            - ``{jira_comments}``: Formatted comment thread (see List Formatting below).
            - ``{jira_links}``: Comma-separated issue links (see List Formatting below).

        GitHub Variables (populated when source is GitHub, empty for Jira triggers):
            - ``{github_host}``: GitHub host (e.g., "github.com").
            - ``{github_org}``: Organization or user name.
            - ``{github_repo}``: Repository name.
            - ``{github_issue_number}``: Issue or PR number.
            - ``{github_issue_title}``: Issue or PR title.
            - ``{github_issue_title_slug}``: Branch-safe version of title (computed, see Slug
              Transformation below).
            - ``{github_issue_body}``: Issue or PR body/description.
            - ``{github_issue_state}``: State (open, closed).
            - ``{github_issue_author}``: Author username.
            - ``{github_issue_url}``: Full URL to issue or PR.
            - ``{github_is_pr}``: "true" if PR, "false" if issue.
            - ``{github_pr_head}``: PR head branch (empty if not a PR).
            - ``{github_pr_base}``: PR base branch (empty if not a PR).
            - ``{github_pr_draft}``: "true" if draft PR, "false" otherwise.
            - ``{github_parent_issue_number}``: Parent issue number if linked.
            - ``{github_issue_assignees}``: Comma-separated assignee usernames (see List
              Formatting below).
            - ``{github_issue_labels}``: Comma-separated label names (see List Formatting
              below).

        Slug Transformation:
            Variables with ``_slug`` suffix are transformed to be branch-safe. The
            transformation algorithm (implemented in ``executor.py:slugify()``):

            1. Normalize unicode characters (e.g., "café" -> "cafe")
            2. Convert to lowercase
            3. Replace spaces and underscores with hyphens
            4. Remove git-invalid characters (~, ^, :, ?, *, [, ], \\, @, {, }, etc.)
            5. Collapse consecutive hyphens/dots into single characters
            6. Strip leading/trailing hyphens and dots

            Note: The slug transformation does not truncate long titles. Very long
            issue titles will produce correspondingly long branch names. If you need
            shorter branch names, consider using a pattern like
            ``feature/{jira_issue_key}`` instead of including the full title slug.

            Examples:
                - "Add User Authentication" -> "add-user-authentication"
                - "Fix bug #123" -> "fix-bug-123"
                - "Update README.md" -> "update-readme.md"
                - "Résumé Upload Feature" -> "resume-upload-feature"
                - "API v2.0 Migration" -> "api-v2.0-migration"
                - "A Very Long Issue Title That Exceeds Normal Length" ->
                  "a-very-long-issue-title-that-exceeds-normal-length" (no truncation)

        List Formatting:
            Variables marked as "formatted" contain list data rendered as comma-separated
            strings. For example, if an issue has labels ["bug", "priority-high"], the
            ``{jira_labels}`` or ``{github_issue_labels}`` variable will contain:
            "bug, priority-high".

            Comments are formatted as numbered entries with truncation for length.

        Notes:
            - Use slug variables (``_slug`` suffix) for branch-safe formatting.
            - Cross-source variables result in empty strings (e.g., Jira vars with
              GitHub triggers).
    """

    host: str = "github.com"
    org: str = ""
    repo: str = ""
    branch: str = ""
    create_branch: bool = False
    base_branch: str = "main"


@dataclass
class AgentConfig:
    """Configuration for the Claude agent.

    Attributes:
        prompt: The prompt template for the agent.
        github: Optional GitHub repository context.
        timeout_seconds: Optional timeout in seconds for agent execution.
            If None, no timeout is applied.

            **Agent Teams timeout behaviour (DS-697):** When ``agent_teams`` is
            ``True``, the effective timeout used at execution time is
            ``timeout_seconds * AGENT_TEAMS_TIMEOUT_MULTIPLIER`` (default 3x),
            with a recommended minimum of ``AGENT_TEAMS_MIN_TIMEOUT_SECONDS``
            (default 900 s / 15 min).  A warning is logged at parse time if the
            configured value falls below the recommended minimum.
        model: Optional model identifier to use for this agent.
            If None, uses the Claude CLI's default model.
            Examples: "claude-opus-4-5-20251101", "claude-sonnet-4-20250514"
        agent_type: Optional agent type to use for this agent.
            If None, defaults to config.default_agent_type.
            Valid values: "claude", "codex", "cursor"
        cursor_mode: Optional cursor mode when agent_type is "cursor".
            Only valid when agent_type is "cursor".
            Not valid when agent_type is "claude" or "codex".
            Valid values: "agent", "plan", "ask"
        agent_teams: Whether to enable Claude Code's experimental Agent Teams
            feature for this orchestration step. Agent Teams spawns a team of
            Claude Code agents (e.g., developer + code reviewers) that coordinate
            via shared task lists and mailboxes. Only valid when agent_type is
            "claude" (Agent Teams is a Claude Code feature). Defaults to False.

            When enabled, the executor automatically applies a timeout multiplier
            (see ``AGENT_TEAMS_TIMEOUT_MULTIPLIER``) to account for the additional
            coordination overhead of multiple Claude Code processes.
        strict_template_variables: If True, raise ValueError for unknown template
            variables instead of logging a warning. Useful for catching typos in
            prompts and branch patterns during development/testing.
            Defaults to False for backwards compatibility.
    """

    prompt: str = ""
    github: GitHubContext | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    agent_type: AgentTypeLiteral | None = None
    cursor_mode: CursorModeLiteral | None = None
    agent_teams: bool = False
    strict_template_variables: bool = False


@dataclass
class Outcome:
    """Configuration for a specific execution outcome.

    Outcomes allow different success patterns to trigger different tags.
    For example, a code review could have "approved" and "changes-requested"
    outcomes, each adding a different label.

    Attributes:
        name: Unique name for this outcome (e.g., "approved", "changes-requested").
        patterns: Patterns that indicate this outcome (substring or "regex:..." prefix).
        add_tag: Label to add to the Jira issue when this outcome is matched.
    """

    name: str = ""
    patterns: list[str] = field(default_factory=list)
    add_tag: str = ""


@dataclass
class RetryConfig:
    """Configuration for retry logic based on agent response patterns.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        success_patterns: Patterns indicating success (substring or "regex:..." prefix).
            Deprecated: Use outcomes instead for more control.
        failure_patterns: Patterns indicating failure (substring or "regex:..." prefix).
            These trigger retries - use for actual errors (API failures, etc.).
        default_status: Status to use when no patterns match ("success" or "failure").
            Defaults to "success" for backwards compatibility.
        default_outcome: Name of outcome to use when no outcome patterns match.
            Only used when outcomes are configured.
    """

    max_attempts: int = 3
    success_patterns: list[str] = field(
        default_factory=lambda: ["SUCCESS", "completed successfully"]
    )
    failure_patterns: list[str] = field(default_factory=lambda: ["FAILURE", "failed", "error"])
    default_status: Literal["success", "failure"] = "success"
    default_outcome: str = ""


@dataclass
class OnStartConfig:
    """Actions to take when an issue is picked up for processing.

    Use this to add an in-progress tag to prevent duplicate processing
    if the poll interval is shorter than the processing time.
    """

    add_tag: str = ""


@dataclass
class OnCompleteConfig:
    """Actions to take after successful orchestration completion."""

    remove_tag: str = ""
    add_tag: str = ""


@dataclass
class OnFailureConfig:
    """Actions to take after failed orchestration (all retries exhausted)."""

    add_tag: str = ""


@dataclass
class Orchestration:
    """A single orchestration configuration.

    Attributes:
        name: Unique identifier for this orchestration.
        trigger: Configuration for what triggers this orchestration.
        agent: Configuration for the Claude agent.
        retry: Configuration for retry logic.
        outcomes: Optional list of outcome configurations for different success types.
            When defined, these replace success_patterns and on_complete for tag handling.
        on_start: Actions to take when processing starts.
        on_complete: Actions to take after successful processing (deprecated if outcomes used).
        on_failure: Actions to take after failed processing.
        enabled: Whether this orchestration is enabled. Defaults to True for backwards
            compatibility. When False, this orchestration will be skipped during loading.
        max_concurrent: Maximum number of concurrent slots this orchestration can use.
            If None, no per-orchestration limit is applied (uses global limit only).
            Must be a positive integer if provided.
    """

    name: str
    trigger: TriggerConfig
    agent: AgentConfig
    retry: RetryConfig = field(default_factory=RetryConfig)
    outcomes: list[Outcome] = field(default_factory=list)
    on_start: OnStartConfig = field(default_factory=OnStartConfig)
    on_complete: OnCompleteConfig = field(default_factory=OnCompleteConfig)
    on_failure: OnFailureConfig = field(default_factory=OnFailureConfig)
    enabled: bool = True
    max_concurrent: int | None = None


class OrchestrationError(Exception):
    """Raised when orchestration configuration is invalid."""

    pass


@dataclass
class OrchestrationVersion:
    """A versioned wrapper around an Orchestration for hot-reload support.

    When orchestration files are modified, new versions are created while
    old versions remain active until all their running executions complete.

    Attributes:
        version_id: Unique identifier for this version (UUID).
        orchestration: The underlying orchestration configuration.
        source_file: Path to the file this orchestration was loaded from.
        mtime: Modification time of the source file when loaded.
        loaded_at: Datetime when this version was loaded.
        active_executions: Count of currently running executions using this version.

    Thread Safety:
        The increment_executions() and decrement_executions() methods are internally
        thread-safe and can be called from multiple threads without external locking.

    Usage Note:
        Do not directly assign to ``active_executions`` in production code as this
        bypasses the internal thread-safety lock. Always use increment_executions()
        and decrement_executions() methods instead. Direct assignment is acceptable
        only in test setup where thread safety is not a concern.
    """

    version_id: str
    orchestration: Orchestration
    source_file: Path
    mtime: float
    loaded_at: datetime
    active_executions: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        orchestration: Orchestration,
        source_file: Path,
        mtime: float,
    ) -> OrchestrationVersion:
        """Create a new OrchestrationVersion with a unique version ID.

        Args:
            orchestration: The orchestration configuration.
            source_file: Path to the source file.
            mtime: Modification time of the source file.

        Returns:
            A new OrchestrationVersion instance.
        """
        return cls(
            version_id=str(uuid.uuid4()),
            orchestration=orchestration,
            source_file=source_file,
            mtime=mtime,
            loaded_at=datetime.now(),
        )

    @property
    def name(self) -> str:
        """Return the orchestration name."""
        return self.orchestration.name

    def increment_executions(self) -> None:
        """Increment the active execution count.

        This method is thread-safe and can be called from multiple threads
        without external locking.
        """
        with self._lock:
            self.active_executions += 1

    def decrement_executions(self) -> None:
        """Decrement the active execution count.

        This method is thread-safe and can be called from multiple threads
        without external locking. The count will not go below zero.
        """
        with self._lock:
            self.active_executions = max(0, self.active_executions - 1)

    @property
    def has_active_executions(self) -> bool:
        """Return True if there are active executions using this version.

        This is a point-in-time snapshot; the value may change immediately
        after reading if other threads are modifying the execution count.
        """
        with self._lock:
            return self.active_executions > 0


def _validate_string_field(
    value: Any,
    field_name: str,
    *,
    reject_empty: bool = True,
) -> None:
    """Validate that a value is a string and not whitespace-only.

    Centralises the isinstance check and whitespace-only rejection that was
    previously repeated for host, org, and repo in _parse_github_context().

    Args:
        value: The raw value to validate (may be None, a string, or another type).
        field_name: Dot-qualified field name used in error messages
            (e.g. ``"github.host"``).
        reject_empty: When ``True`` (the default) an empty string ``""`` is
            treated the same as a whitespace-only string and rejected.
            When ``False`` the empty string is accepted without error.

    Raises:
        OrchestrationError: If *value* is not ``None`` and fails validation.
            ``None`` is always accepted (callers fall back to a default).
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise OrchestrationError(
            f"Invalid {field_name} '{value}': must be a string. "
            f"Please provide a valid value or omit the field."
        )

    # Reject whitespace-only strings (and empty strings when reject_empty=True).
    if not reject_empty and value == "":
        return
    if value.strip() == "":
        label = "empty string" if value == "" else "whitespace-only string"
        raise OrchestrationError(
            f"Invalid {field_name}: {label} is not a valid value. "
            f"Please provide a valid value or omit the field to use the default."
        )


def _validate_string_list(value: Any, field_name: str) -> None:
    """Validate that a value is a list of non-empty strings.

    Args:
        value: The value to validate.
        field_name: The name of the field for error messages.

    Raises:
        OrchestrationError: If validation fails.
    """
    if value:
        if not isinstance(value, list):
            raise OrchestrationError(f"{field_name} must be a list")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise OrchestrationError(f"{field_name} must contain non-empty strings")


def _validate_branch_name(branch: str) -> ValidationResult:
    """Validate that a branch name follows git branch naming rules.

    This function wraps the shared validate_branch_name_core() function,
    adapting the BranchValidationResult to the local ValidationResult type
    used throughout this module.

    Git branch naming rules enforced:
    - Cannot start with a hyphen (-) or period (.)
    - Cannot end with a period (.), forward slash (/), or .lock
    - Cannot contain: space, ~, ^, :, ?, *, [, ], \\, @{
    - Cannot contain consecutive periods (..) or forward slashes (//)
    - Cannot be empty (but empty strings are allowed as "not configured")

    Note: Branch names with template variables like {jira_issue_key} are validated
    for the static parts only. The dynamic parts are validated at runtime.

    Returns ValidationResult.success() for valid or empty strings,
    ValidationResult.failure(message) otherwise.
    """
    result = validate_branch_name_core(branch, allow_empty=True, allow_template_variables=True)

    if result.is_valid:
        return ValidationResult.success()
    return ValidationResult.failure(result.error_message)


# ---------------------------------------------------------------------------
# Shared validation helpers (DS-763)
#
# Each helper validates a single field or cross-field invariant and returns
# an error message string when validation fails, or ``None`` when the value
# is valid.  Both the ``_collect_*_errors`` functions (which append to a
# list) and the ``_parse_*`` functions (which raise OrchestrationError) call
# these helpers so that the validation logic is defined in exactly one place.
# ---------------------------------------------------------------------------


def _validate_trigger_source(source: str) -> str | None:
    """Validate trigger source value.

    Args:
        source: The trigger source string to validate.

    Returns:
        An error message if *source* is not a recognised trigger source,
        or ``None`` if valid.
    """
    if not TriggerSource.is_valid(source):
        valid_sources = ", ".join(f"'{s}'" for s in sorted(TriggerSource.values()))
        return f"Invalid trigger source '{source}': must be {valid_sources}"
    return None


def _validate_project_number(project_number: Any) -> str | None:
    """Validate project_number for GitHub triggers.

    Checks that *project_number* is present (not ``None``) and is a positive
    integer.

    Args:
        project_number: The project_number value to validate.

    Returns:
        An error message if validation fails, or ``None`` if valid.
    """
    if project_number is None:
        return (
            "GitHub triggers require 'project_number' to be set. "
            "Please configure project_number, project_scope, and project_owner "
            "for GitHub Project-based polling."
        )
    if not isinstance(project_number, int) or project_number <= 0:
        return f"Invalid project_number '{project_number}': must be a positive integer"
    return None


def _validate_project_scope(project_scope: str) -> str | None:
    """Validate project_scope for GitHub triggers.

    Args:
        project_scope: The project_scope value to validate.

    Returns:
        An error message if *project_scope* is not ``'org'`` or ``'user'``,
        or ``None`` if valid.
    """
    if project_scope not in ("org", "user"):
        return f"Invalid project_scope '{project_scope}': must be 'org' or 'user'"
    return None


def _validate_project_owner(project_owner: str) -> str | None:
    """Validate project_owner for GitHub triggers.

    Args:
        project_owner: The project_owner value to validate.

    Returns:
        An error message if *project_owner* is empty/falsy, or ``None``
        if valid.
    """
    if not project_owner:
        return (
            "GitHub triggers require 'project_owner' to be set "
            "(organization name or username)"
        )
    return None


def _validate_timeout_seconds(timeout: Any) -> str | None:
    """Validate timeout_seconds for agent configuration.

    Args:
        timeout: The timeout_seconds value to validate.

    Returns:
        An error message if *timeout* is not ``None`` and is not a positive
        integer, or ``None`` if valid.
    """
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        return f"Invalid timeout_seconds '{timeout}': must be a positive integer"
    return None


def _validate_model(model: Any) -> str | None:
    """Validate model for agent configuration.

    Args:
        model: The model value to validate.

    Returns:
        An error message if *model* is not ``None`` and is not a string,
        or ``None`` if valid.
    """
    if model is not None and not isinstance(model, str):
        return f"Invalid model '{model}': must be a string"
    return None


def _validate_agent_type(agent_type: Any) -> str | None:
    """Validate agent_type for agent configuration.

    Args:
        agent_type: The agent_type value to validate.

    Returns:
        An error message if *agent_type* is not ``None`` and is not a
        recognised agent type, or ``None`` if valid.
    """
    if agent_type is not None and not AgentType.is_valid(agent_type):
        valid_types = ", ".join(f"'{t}'" for t in sorted(AgentType.values()))
        return f"Invalid agent_type '{agent_type}': must be {valid_types}"
    return None


def _validate_cursor_mode(cursor_mode: Any, agent_type: Any) -> str | None:
    """Validate cursor_mode for agent configuration.

    Checks both value validity and compatibility with *agent_type*.

    Args:
        cursor_mode: The cursor_mode value to validate.
        agent_type: The agent_type value (used for compatibility check).

    Returns:
        An error message if *cursor_mode* is invalid or incompatible with
        *agent_type*, or ``None`` if valid.
    """
    if cursor_mode is None:
        return None
    if not CursorMode.is_valid(cursor_mode):
        valid_modes = ", ".join(f"'{m}'" for m in sorted(CursorMode.values()))
        return f"Invalid cursor_mode '{cursor_mode}': must be {valid_modes}"
    # cursor_mode is only valid when agent_type is 'cursor'.
    # NOTE: This uses an exclude-list approach (DS-646). Any agent type
    # NOT listed in the tuple below will silently accept cursor_mode
    # without validation. When adding new AgentType values that do NOT
    # support cursor_mode, you must add them to the tuple below so the
    # validation rejects them.
    if agent_type is not None and agent_type in (
        AgentType.CLAUDE.value,
        AgentType.CODEX.value,
    ):
        return (
            f"cursor_mode '{cursor_mode}' is only valid when agent_type is "
            f"'{AgentType.CURSOR.value}'"
        )
    return None


def _validate_agent_teams(agent_teams: Any, agent_type: Any) -> str | None:
    """Validate agent_teams for agent configuration.

    Checks both type validity and compatibility with *agent_type*.

    Args:
        agent_teams: The agent_teams value to validate.
        agent_type: The agent_type value (used for compatibility check).

    Returns:
        An error message if *agent_teams* is not a boolean or is
        incompatible with *agent_type*, or ``None`` if valid.
    """
    if not isinstance(agent_teams, bool):
        return f"Invalid agent_teams '{agent_teams}': must be a boolean"
    # agent_teams is only valid when agent_type is "claude" (Agent Teams is a
    # Claude Code feature, not available in Cursor or Codex).
    # NOTE: This uses an exclude-list approach. When adding new AgentType values
    # that do NOT support agent_teams, you must add them to the tuple below so
    # the validation rejects them.
    if agent_teams and agent_type is not None and agent_type in (
        AgentType.CURSOR.value,
        AgentType.CODEX.value,
    ):
        return (
            f"agent_teams is only valid when agent_type is "
            f"'{AgentType.CLAUDE.value}', got agent_type='{agent_type}'"
        )
    return None


def _validate_strict_template_variables(strict_template_variables: Any) -> str | None:
    """Validate strict_template_variables for agent configuration.

    Args:
        strict_template_variables: The value to validate.

    Returns:
        An error message if *strict_template_variables* is not a boolean,
        or ``None`` if valid.
    """
    if not isinstance(strict_template_variables, bool):
        return (
            f"Invalid strict_template_variables '{strict_template_variables}': "
            f"must be a boolean"
        )
    return None


def _collect_trigger_errors(data: dict[str, Any]) -> list[str]:
    """Collect all validation errors from trigger configuration (DS-734).

    Unlike ``_parse_trigger`` which raises on the first error, this function
    collects every validation error so the API can report them all at once.

    Args:
        data: The trigger configuration dict to validate.

    Returns:
        A list of validation error messages. Empty list means no errors.
    """
    errors: list[str] = []

    source = data.get("source", TriggerSource.JIRA.value)
    error = _validate_trigger_source(source)
    if error:
        errors.append(error)

    tags = data.get("tags", [])
    labels = data.get("labels", [])

    # Validate labels field (used by GitHub triggers)
    try:
        _validate_string_list(labels, "labels")
    except OrchestrationError as e:
        errors.append(str(e))

    # Validate tags field (used by Jira triggers)
    try:
        _validate_string_list(tags, "tags")
    except OrchestrationError as e:
        errors.append(str(e))

    # Validation for GitHub triggers
    project_number = data.get("project_number")
    project_scope = data.get("project_scope", "org")
    project_owner = data.get("project_owner", "")

    if source == TriggerSource.GITHUB.value:
        error = _validate_project_number(project_number)
        if error:
            errors.append(error)

        error = _validate_project_scope(project_scope)
        if error:
            errors.append(error)

        error = _validate_project_owner(project_owner)
        if error:
            errors.append(error)

    return errors


def _parse_trigger(data: dict[str, Any]) -> TriggerConfig:
    """Parse trigger configuration from dict.

    Supports both Jira and GitHub triggers:
    - Jira triggers use: source="jira", project, jql_filter, tags
    - GitHub triggers use: source="github", project_number, project_scope,
      project_owner, project_filter

    Raises:
        OrchestrationError: If trigger configuration is invalid.
    """
    source = data.get("source", TriggerSource.JIRA.value)
    error = _validate_trigger_source(source)
    if error:
        raise OrchestrationError(error)

    tags = data.get("tags", [])

    # GitHub Project-based fields
    project_number = data.get("project_number")
    project_scope = data.get("project_scope", "org")
    project_owner = data.get("project_owner", "")
    project_filter = data.get("project_filter", "")
    labels = data.get("labels", [])

    # Validate labels field (used by GitHub triggers)
    _validate_string_list(labels, "labels")

    # Validate tags field (used by Jira triggers)
    _validate_string_list(tags, "tags")

    # Validation for GitHub triggers
    if source == TriggerSource.GITHUB.value:
        error = _validate_project_number(project_number)
        if error:
            raise OrchestrationError(error)

        error = _validate_project_scope(project_scope)
        if error:
            raise OrchestrationError(error)

        error = _validate_project_owner(project_owner)
        if error:
            raise OrchestrationError(error)

    return TriggerConfig(
        source=source,
        project=data.get("project", ""),
        jql_filter=data.get("jql_filter", ""),
        tags=tags,
        project_number=project_number,
        project_scope=project_scope,
        project_owner=project_owner,
        project_filter=project_filter,
        labels=labels,
    )


def _parse_github_context(data: dict[str, Any] | None) -> GitHubContext | None:
    """Parse GitHub context from dict.

    Raises:
        OrchestrationError: If branch or base_branch names are invalid.
    """
    if not data:
        return None

    # Validate branch pattern if provided
    branch = data.get("branch") or ""
    if branch.strip():
        result = _validate_branch_name(branch)
        if not result.is_valid:
            raise OrchestrationError(f"Invalid branch pattern '{branch}': {result.error_message}")

    # Validate base_branch if provided
    base_branch = data.get("base_branch") or "main"
    if base_branch.strip():
        result = _validate_branch_name(base_branch)
        if not result.is_valid:
            raise OrchestrationError(f"Invalid base_branch '{base_branch}': {result.error_message}")

    # Validate that create_branch=True requires a non-empty branch pattern
    create_branch = data.get("create_branch", False)
    if create_branch and not branch.strip():
        raise OrchestrationError(
            "create_branch is True but no branch pattern is specified. "
            "Please provide a branch pattern (e.g., 'feature/{jira_issue_key}') "
            "when create_branch is enabled."
        )

    # Use explicit None checks instead of or-coalescing so that empty string
    # is preserved as a distinct value from None (future-proofing).
    host = data.get("host")
    org = data.get("org")
    repo = data.get("repo")

    # Validate host, org, and repo using the shared helper (DS-634).
    # None is always acceptable (falls back to a field-specific default).
    # Host rejects both empty and whitespace-only strings (DS-631, DS-632).
    # Org and repo allow empty strings but reject whitespace-only (DS-633).
    _validate_string_field(host, "github.host")
    _validate_string_field(org, "github.org", reject_empty=False)
    _validate_string_field(repo, "github.repo", reject_empty=False)
    resolved_host = host if host is not None else "github.com"

    return GitHubContext(
        host=resolved_host,
        org=org if org is not None else "",
        repo=repo if repo is not None else "",
        branch=branch,
        create_branch=create_branch,
        base_branch=base_branch,
    )


def _collect_agent_errors(data: dict[str, Any]) -> list[str]:
    """Collect all validation errors from agent configuration (DS-734).

    Unlike ``_parse_agent`` which raises on the first error, this function
    collects every validation error so the API can report them all at once.

    Args:
        data: The agent configuration dict to validate.

    Returns:
        A list of validation error messages. Empty list means no errors.
    """
    errors: list[str] = []

    timeout = data.get("timeout_seconds")
    error = _validate_timeout_seconds(timeout)
    if error:
        errors.append(error)

    model = data.get("model")
    error = _validate_model(model)
    if error:
        errors.append(error)

    agent_type = data.get("agent_type")
    error = _validate_agent_type(agent_type)
    if error:
        errors.append(error)

    cursor_mode = data.get("cursor_mode")
    error = _validate_cursor_mode(cursor_mode, agent_type)
    if error:
        errors.append(error)

    # Parse agent_teams (defaults to False)
    agent_teams = data.get("agent_teams", False)
    error = _validate_agent_teams(agent_teams, agent_type)
    if error:
        errors.append(error)

    # Parse strict_template_variables (defaults to False for backwards compatibility)
    strict_template_variables = data.get("strict_template_variables", False)
    error = _validate_strict_template_variables(strict_template_variables)
    if error:
        errors.append(error)

    # Validate github context if present
    github_data = data.get("github")
    if github_data:
        try:
            _parse_github_context(github_data)
        except OrchestrationError as e:
            errors.append(str(e))

    return errors


def _parse_agent(data: dict[str, Any]) -> AgentConfig:
    """Parse agent configuration from dict."""
    timeout = data.get("timeout_seconds")
    error = _validate_timeout_seconds(timeout)
    if error:
        raise OrchestrationError(error)

    model = data.get("model")
    error = _validate_model(model)
    if error:
        raise OrchestrationError(error)

    agent_type = data.get("agent_type")
    error = _validate_agent_type(agent_type)
    if error:
        raise OrchestrationError(error)

    cursor_mode = data.get("cursor_mode")
    error = _validate_cursor_mode(cursor_mode, agent_type)
    if error:
        raise OrchestrationError(error)

    # Parse agent_teams (defaults to False)
    agent_teams = data.get("agent_teams", False)
    error = _validate_agent_teams(agent_teams, agent_type)
    if error:
        raise OrchestrationError(error)

    # Warn if agent_teams is enabled but timeout_seconds is below the recommended
    # minimum (DS-697).  This is a parse-time warning so operators see it as
    # early as possible.
    if agent_teams and timeout is not None and timeout < AGENT_TEAMS_MIN_TIMEOUT_SECONDS:
        logger.warning(
            "agent_teams is enabled but timeout_seconds=%d is below the "
            "recommended minimum of %d seconds for Agent Teams orchestrations. "
            "The effective timeout will be %d seconds (timeout_seconds * %d). "
            "Consider setting timeout_seconds >= %d to avoid premature timeouts.",
            timeout,
            AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
            timeout * AGENT_TEAMS_TIMEOUT_MULTIPLIER,
            AGENT_TEAMS_TIMEOUT_MULTIPLIER,
            AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
        )

    # Parse strict_template_variables (defaults to False for backwards compatibility)
    strict_template_variables = data.get("strict_template_variables", False)
    error = _validate_strict_template_variables(strict_template_variables)
    if error:
        raise OrchestrationError(error)

    return AgentConfig(
        prompt=data.get("prompt", ""),
        github=_parse_github_context(data.get("github")),
        timeout_seconds=timeout,
        model=model,
        agent_type=agent_type,
        cursor_mode=cursor_mode,
        agent_teams=agent_teams,
        strict_template_variables=strict_template_variables,
    )


def get_effective_timeout(agent_config: AgentConfig) -> int | None:
    """Compute the effective timeout for an agent configuration.

    When ``agent_teams`` is enabled the configured ``timeout_seconds`` is
    multiplied by ``AGENT_TEAMS_TIMEOUT_MULTIPLIER`` to account for the
    coordination overhead of multiple Claude Code processes.  If the result
    is still below ``AGENT_TEAMS_MIN_TIMEOUT_SECONDS`` the minimum is used
    instead.

    When ``agent_teams`` is ``False`` the configured ``timeout_seconds`` is
    returned unchanged.

    Args:
        agent_config: The agent configuration to compute the timeout for.

    Returns:
        The effective timeout in seconds, or ``None`` if no timeout is
        configured.
    """
    timeout = agent_config.timeout_seconds
    if timeout is None:
        return None

    if not agent_config.agent_teams:
        return timeout

    # Apply multiplier and enforce minimum for Agent Teams orchestrations
    effective = timeout * AGENT_TEAMS_TIMEOUT_MULTIPLIER
    return max(effective, AGENT_TEAMS_MIN_TIMEOUT_SECONDS)


def _parse_retry(data: dict[str, Any] | None) -> RetryConfig:
    """Parse retry configuration from dict."""
    if not data:
        return RetryConfig()

    # Validate default_status if provided
    default_status = data.get("default_status", "success")
    if default_status not in ("success", "failure"):
        raise OrchestrationError(
            f"Invalid default_status '{default_status}': must be 'success' or 'failure'"
        )

    return RetryConfig(
        max_attempts=data.get("max_attempts", 3),
        success_patterns=data.get("success_patterns", ["SUCCESS", "completed successfully"]),
        failure_patterns=data.get("failure_patterns", ["FAILURE", "failed", "error"]),
        default_status=default_status,
        default_outcome=data.get("default_outcome", ""),
    )


def _parse_outcome(data: dict[str, Any]) -> Outcome:
    """Parse a single outcome configuration from dict."""
    name = data.get("name", "")
    if not name:
        raise OrchestrationError("Outcome must have a 'name' field")

    patterns = data.get("patterns", [])
    if not patterns:
        raise OrchestrationError(f"Outcome '{name}' must have at least one pattern")

    return Outcome(
        name=name,
        patterns=patterns,
        add_tag=data.get("add_tag", ""),
    )


def _parse_outcomes(data: list[dict[str, Any]] | None) -> list[Outcome]:
    """Parse outcomes configuration from list."""
    if not data:
        return []

    outcomes = [_parse_outcome(item) for item in data]

    # Validate unique names
    names = [o.name for o in outcomes]
    if len(names) != len(set(names)):
        raise OrchestrationError("Outcome names must be unique")

    return outcomes


def _parse_on_start(data: dict[str, Any] | None) -> OnStartConfig:
    """Parse on_start configuration from dict."""
    if not data:
        return OnStartConfig()
    return OnStartConfig(
        add_tag=data.get("add_tag", ""),
    )


def _parse_on_complete(data: dict[str, Any] | None) -> OnCompleteConfig:
    """Parse on_complete configuration from dict."""
    if not data:
        return OnCompleteConfig()
    return OnCompleteConfig(
        remove_tag=data.get("remove_tag", ""),
        add_tag=data.get("add_tag", ""),
    )


def _parse_on_failure(data: dict[str, Any] | None) -> OnFailureConfig:
    """Parse on_failure configuration from dict."""
    if not data:
        return OnFailureConfig()
    return OnFailureConfig(
        add_tag=data.get("add_tag", ""),
    )


def _parse_orchestration(data: dict[str, Any]) -> Orchestration:
    """Parse a single orchestration from dict."""
    name = data.get("name")
    if not name:
        raise OrchestrationError("Orchestration must have a 'name' field")

    trigger_data = data.get("trigger")
    if not trigger_data:
        raise OrchestrationError(f"Orchestration '{name}' must have a 'trigger' field")

    agent_data = data.get("agent")
    if not agent_data:
        raise OrchestrationError(f"Orchestration '{name}' must have an 'agent' field")

    # Parse enabled field (defaults to True for backwards compatibility)
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise OrchestrationError(
            f"Orchestration '{name}' has invalid 'enabled' value: must be a boolean"
        )

    # Parse max_concurrent field (defaults to None for no per-orchestration limit)
    max_concurrent = data.get("max_concurrent")
    if max_concurrent is not None and (
        not isinstance(max_concurrent, int) or max_concurrent <= 0
    ):
        raise OrchestrationError(
            f"Orchestration '{name}' has invalid 'max_concurrent' value: "
            "must be a positive integer"
        )

    return Orchestration(
        name=name,
        trigger=_parse_trigger(trigger_data),
        agent=_parse_agent(agent_data),
        retry=_parse_retry(data.get("retry")),
        outcomes=_parse_outcomes(data.get("outcomes")),
        on_start=_parse_on_start(data.get("on_start")),
        on_complete=_parse_on_complete(data.get("on_complete")),
        on_failure=_parse_on_failure(data.get("on_failure")),
        enabled=enabled,
        max_concurrent=max_concurrent,
    )


def _load_orchestration_file_with_counts(file_path: Path) -> tuple[list[Orchestration], int]:
    """Load orchestrations from a file and return both enabled list and total count.

    This is the internal implementation that performs all YAML parsing and filtering.
    It returns both the enabled orchestrations and the total count for logging purposes.

    Supports file-level and orchestration-level enabled flags:
    - File-level `enabled: false` disables all orchestrations in the file
    - Orchestration-level `enabled: false` disables just that orchestration
    - File-level takes precedence over orchestration-level
    - Both default to True for backwards compatibility

    Args:
        file_path: Path to the YAML file.

    Returns:
        A tuple of (enabled_orchestrations, total_count) where:
        - enabled_orchestrations: List of enabled Orchestration objects
        - total_count: Total number of orchestrations defined in the file

    Raises:
        OrchestrationError: If the file is invalid.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise OrchestrationError(f"Invalid YAML in {file_path}: {e}") from e
    except FileNotFoundError:
        raise OrchestrationError(f"Orchestration file not found: {file_path}") from None

    if not data:
        return [], 0

    # Check file-level enabled flag (defaults to True for backwards compatibility)
    file_enabled = data.get("enabled", True)
    if not isinstance(file_enabled, bool):
        raise OrchestrationError(f"File-level 'enabled' must be a boolean in {file_path}")

    orchestrations_data = data.get("orchestrations", [])
    if not isinstance(orchestrations_data, list):
        raise OrchestrationError(f"'orchestrations' must be a list in {file_path}")

    total_count = len(orchestrations_data)

    # If file-level enabled is False, all orchestrations are filtered
    if not file_enabled:
        logger.debug(
            "Skipping all orchestrations in %s: file-level enabled is false",
            file_path,
        )
        return [], total_count

    # Parse all orchestrations and filter out disabled ones
    orchestrations = [_parse_orchestration(orch) for orch in orchestrations_data]
    enabled_orchestrations = []
    for orch in orchestrations:
        if orch.enabled:
            enabled_orchestrations.append(orch)
        else:
            logger.debug(
                "Skipping orchestration '%s' in %s: orchestration-level enabled is false",
                orch.name,
                file_path,
            )

    return enabled_orchestrations, total_count


def load_orchestration_file(file_path: Path) -> list[Orchestration]:
    """Load orchestrations from a single YAML file.

    Supports file-level and orchestration-level enabled flags:
    - File-level `enabled: false` disables all orchestrations in the file
    - Orchestration-level `enabled: false` disables just that orchestration
    - File-level takes precedence over orchestration-level
    - Both default to True for backwards compatibility

    This function delegates to _load_orchestration_file_with_counts() internally
    and discards the count information for callers who only need the orchestration list.

    Args:
        file_path: Path to the YAML file.

    Returns:
        List of enabled Orchestration objects. Disabled orchestrations are filtered out.

    Raises:
        OrchestrationError: If the file is invalid.
    """
    enabled_orchestrations, _ = _load_orchestration_file_with_counts(file_path)
    return enabled_orchestrations


def load_orchestrations(directory: Path) -> list[Orchestration]:
    """Load all orchestrations from a directory.

    Loads all .yaml and .yml files from the given directory.

    Args:
        directory: Path to the orchestrations directory.

    Returns:
        List of all Orchestration objects from all files.

    Raises:
        OrchestrationError: If any file is invalid.
    """
    if not directory.exists():
        return []

    if not directory.is_dir():
        raise OrchestrationError(f"Orchestrations path is not a directory: {directory}")

    orchestrations: list[Orchestration] = []
    total_count = 0

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix in (".yaml", ".yml"):
            enabled_orchestrations, file_total = _load_orchestration_file_with_counts(file_path)
            orchestrations.extend(enabled_orchestrations)
            total_count += file_total

    filtered_count = total_count - len(orchestrations)
    logger.info(
        "Loaded %d orchestrations (%d disabled/filtered)",
        len(orchestrations),
        filtered_count,
    )

    return orchestrations
