"""Orchestration configuration schema and loading."""

from __future__ import annotations

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
        model: Optional model identifier to use for this agent.
            If None, uses the Claude CLI's default model.
            Examples: "claude-opus-4-5-20251101", "claude-sonnet-4-20250514"
        agent_type: Optional agent type to use for this agent.
            If None, defaults to config.default_agent_type.
            Valid values: "claude", "cursor"
        cursor_mode: Optional cursor mode when agent_type is "cursor".
            Only valid when agent_type is "cursor".
            Valid values: "agent", "plan", "ask"
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
    if not TriggerSource.is_valid(source):
        valid_sources = ", ".join(f"'{s}'" for s in sorted(TriggerSource.values()))
        raise OrchestrationError(f"Invalid trigger source '{source}': must be {valid_sources}")

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
        # Validate project_number is set for GitHub triggers
        if project_number is None:
            raise OrchestrationError(
                "GitHub triggers require 'project_number' to be set. "
                "Please configure project_number, project_scope, and project_owner "
                "for GitHub Project-based polling."
            )

        # Validate project_number is a positive integer
        if not isinstance(project_number, int) or project_number <= 0:
            raise OrchestrationError(
                f"Invalid project_number '{project_number}': must be a positive integer"
            )

        # Validate project_scope
        if project_scope not in ("org", "user"):
            raise OrchestrationError(
                f"Invalid project_scope '{project_scope}': must be 'org' or 'user'"
            )

        # Validate project_owner is set
        if not project_owner:
            raise OrchestrationError(
                "GitHub triggers require 'project_owner' to be set "
                "(organization name or username)"
            )

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
    branch = data.get("branch", "")
    if branch:
        result = _validate_branch_name(branch)
        if not result.is_valid:
            raise OrchestrationError(f"Invalid branch pattern '{branch}': {result.error_message}")

    # Validate base_branch if provided
    base_branch = data.get("base_branch", "main")
    if base_branch:
        result = _validate_branch_name(base_branch)
        if not result.is_valid:
            raise OrchestrationError(f"Invalid base_branch '{base_branch}': {result.error_message}")

    return GitHubContext(
        host=data.get("host", "github.com"),
        org=data.get("org", ""),
        repo=data.get("repo", ""),
        branch=branch,
        create_branch=data.get("create_branch", False),
        base_branch=base_branch,
    )


def _parse_agent(data: dict[str, Any]) -> AgentConfig:
    """Parse agent configuration from dict."""
    timeout = data.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise OrchestrationError(f"Invalid timeout_seconds '{timeout}': must be a positive integer")

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise OrchestrationError(f"Invalid model '{model}': must be a string")

    agent_type = data.get("agent_type")
    if agent_type is not None and not AgentType.is_valid(agent_type):
        valid_types = ", ".join(f"'{t}'" for t in sorted(AgentType.values()))
        raise OrchestrationError(f"Invalid agent_type '{agent_type}': must be {valid_types}")

    cursor_mode = data.get("cursor_mode")
    if cursor_mode is not None:
        if not CursorMode.is_valid(cursor_mode):
            valid_modes = ", ".join(f"'{m}'" for m in sorted(CursorMode.values()))
            raise OrchestrationError(f"Invalid cursor_mode '{cursor_mode}': must be {valid_modes}")
        # cursor_mode is only valid when agent_type is 'cursor'
        if agent_type is not None and agent_type != AgentType.CURSOR.value:
            raise OrchestrationError(
                f"cursor_mode '{cursor_mode}' is only valid when agent_type is "
                f"'{AgentType.CURSOR.value}'"
            )

    # Parse strict_template_variables (defaults to False for backwards compatibility)
    strict_template_variables = data.get("strict_template_variables", False)
    if not isinstance(strict_template_variables, bool):
        raise OrchestrationError(
            f"Invalid strict_template_variables '{strict_template_variables}': must be a boolean"
        )

    return AgentConfig(
        prompt=data.get("prompt", ""),
        github=_parse_github_context(data.get("github")),
        timeout_seconds=timeout,
        model=model,
        agent_type=agent_type,
        cursor_mode=cursor_mode,
        strict_template_variables=strict_template_variables,
    )


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
