"""Orchestration configuration schema and loading."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml

from sentinel.branch_validation import validate_branch_name_core
from sentinel.logging import get_logger

logger = get_logger(__name__)


# Deprecation warning message template for GitHub trigger fields
_GITHUB_FIELD_DEPRECATION_MSG = (
    "The '{field}' field is deprecated for GitHub triggers. "
    "Use project-based configuration (project_number, project_scope, project_owner, "
    "project_filter) instead. This field will be removed in a future version."
)


# Pre-compiled regex patterns for GitHub repository validation
# These are compiled at module level to avoid overhead on each validation call
# See _validate_github_repo_format() for usage and format documentation

# Owner pattern: starts with alphanumeric, ends with alphanumeric,
# middle can have alphanumeric or single hyphens (no consecutive hyphens)
_GITHUB_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9]|-(?!-))*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")

# Repo name pattern: alphanumeric, hyphens, underscores, periods
_GITHUB_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$|^[a-zA-Z0-9]$")

# Note: Branch validation is now handled by the shared branch_validation module


class ValidationResult(NamedTuple):
    """Result of a validation operation with is_valid flag and error_message."""

    is_valid: bool
    error_message: str

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, error_message="")

    @classmethod
    def failure(cls, message: str) -> "ValidationResult":
        """Create a failed validation result with the given error message."""
        return cls(is_valid=False, error_message=message)


@dataclass
class TriggerConfig:
    """Configuration for what triggers an orchestration.

    Supports Jira triggers (source="jira") with project, jql_filter, tags fields,
    and GitHub triggers (source="github") with project_number, project_scope,
    project_owner, project_filter, and labels fields.

    Multiple tags/labels use AND logic (issue must have all specified).

    Note: repo, query_filter, and tags are DEPRECATED for GitHub triggers.
    """

    source: Literal["jira", "github"] = "jira"
    project: str = ""
    jql_filter: str = ""
    tags: list[str] = field(default_factory=list)
    # Deprecated GitHub fields (still supported for backwards compatibility)
    repo: str = ""
    query_filter: str = ""
    # New GitHub Project-based fields
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
            - ``{jira_summary_slug}``: Branch-safe version of summary (computed).
            - ``{jira_description}``: Full issue description.
            - ``{jira_status}``: Current status name.
            - ``{jira_assignee}``: Assignee display name.
            - ``{jira_epic_key}``: Parent epic key if linked.
            - ``{jira_parent_key}``: Parent issue key if subtask.
            - ``{jira_labels}``: Comma-separated list of labels (formatted).
            - ``{jira_comments}``: Formatted comment thread (formatted).
            - ``{jira_links}``: Comma-separated issue links (formatted).

        GitHub Variables (populated when source is GitHub, empty for Jira triggers):
            - ``{github_host}``: GitHub host (e.g., "github.com").
            - ``{github_org}``: Organization or user name.
            - ``{github_repo}``: Repository name.
            - ``{github_issue_number}``: Issue or PR number.
            - ``{github_issue_title}``: Issue or PR title.
            - ``{github_issue_title_slug}``: Branch-safe version of title (computed).
            - ``{github_issue_body}``: Issue or PR body/description.
            - ``{github_issue_state}``: State (open, closed).
            - ``{github_issue_author}``: Author username.
            - ``{github_issue_url}``: Full URL to issue or PR.
            - ``{github_is_pr}``: "true" if PR, "false" if issue.
            - ``{github_pr_head}``: PR head branch (empty if not a PR).
            - ``{github_pr_base}``: PR base branch (empty if not a PR).
            - ``{github_pr_draft}``: "true" if draft PR, "false" otherwise.
            - ``{github_parent_issue_number}``: Parent issue number if linked.
            - ``{github_issue_assignees}``: Comma-separated assignee usernames (formatted).
            - ``{github_issue_labels}``: Comma-separated label names (formatted).

        Notes:
            - Use slug variables (``_slug`` suffix) for branch-safe formatting.
            - Cross-source variables result in empty strings (e.g., Jira vars with
              GitHub triggers).
            - Formatted list variables use comma separation with proper escaping.
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
        tools: List of tool names the agent can use.
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
    """

    prompt: str = ""
    tools: list[str] = field(default_factory=list)
    github: GitHubContext | None = None
    timeout_seconds: int | None = None
    model: str | None = None
    agent_type: Literal["claude", "cursor"] | None = None
    cursor_mode: Literal["agent", "plan", "ask"] | None = None


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
    """

    version_id: str
    orchestration: Orchestration
    source_file: Path
    mtime: float
    loaded_at: datetime
    active_executions: int = 0

    @classmethod
    def create(
        cls,
        orchestration: Orchestration,
        source_file: Path,
        mtime: float,
    ) -> "OrchestrationVersion":
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

        Thread Safety Note:
            This method modifies active_executions without internal synchronization.
            Callers must hold the appropriate lock (e.g., _versions_lock in main.py)
            when calling this method to ensure thread safety.
        """
        self.active_executions += 1

    def decrement_executions(self) -> None:
        """Decrement the active execution count.

        Thread Safety Note:
            This method modifies active_executions without internal synchronization.
            Callers must hold the appropriate lock (e.g., _versions_lock in main.py)
            when calling this method to ensure thread safety.
        """
        self.active_executions = max(0, self.active_executions - 1)

    @property
    def has_active_executions(self) -> bool:
        """Return True if there are active executions using this version."""
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


def _validate_github_repo_format(repo: str) -> ValidationResult:
    """Validate that a GitHub repo string is in 'owner/repo-name' format.

    Enforces GitHub naming rules:
    - Owner: max 39 chars, alphanumeric/hyphens, no leading/trailing/consecutive hyphens
    - Repo: max 100 chars, alphanumeric/hyphens/underscores/periods, no leading period,
      cannot end with .git

    Returns ValidationResult.success() for valid or empty strings,
    ValidationResult.failure(message) otherwise.
    """
    if not repo:
        return ValidationResult.success()  # Empty is valid (optional field)

    parts = repo.split("/")
    if len(parts) != 2:
        return ValidationResult.failure(
            "must be in 'owner/repo-name' format (e.g., 'octocat/hello-world')"
        )

    owner, repo_name = parts

    # Basic check for non-empty parts
    if not owner.strip():
        return ValidationResult.failure("owner cannot be empty")

    if not repo_name.strip():
        return ValidationResult.failure("repository name cannot be empty")

    # Reserved names check (both owner and repo cannot be "." or "..")
    if owner in (".", ".."):
        return ValidationResult.failure(f"owner '{owner}' is a reserved name")

    if repo_name in (".", ".."):
        return ValidationResult.failure(f"repository name '{repo_name}' is a reserved name")

    # Validate owner (username/organization)
    # - Max 39 characters
    # - Alphanumeric and hyphens only
    # - Cannot start or end with hyphen
    # - Cannot have consecutive hyphens
    if len(owner) > 39:
        return ValidationResult.failure(
            f"owner exceeds maximum length of 39 characters (got {len(owner)})"
        )

    if not _GITHUB_OWNER_PATTERN.match(owner):
        return ValidationResult.failure(
            f"owner '{owner}' contains invalid characters or format "
            "(must be alphanumeric with single hyphens, cannot start/end with hyphen)"
        )

    # Validate repo name
    # - Max 100 characters
    # - Alphanumeric, hyphens, underscores, periods
    # - Cannot start with a period
    # - Cannot end with .git
    if len(repo_name) > 100:
        return ValidationResult.failure(
            f"repository name exceeds maximum length of 100 characters (got {len(repo_name)})"
        )

    if repo_name.startswith("."):
        return ValidationResult.failure("repository name cannot start with a period")

    if repo_name.lower().endswith(".git"):
        return ValidationResult.failure("repository name cannot end with '.git'")

    if not _GITHUB_REPO_PATTERN.match(repo_name):
        return ValidationResult.failure(
            f"repository name '{repo_name}' contains invalid characters "
            "(allowed: alphanumeric, hyphens, underscores, periods)"
        )

    return ValidationResult.success()


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
    result = validate_branch_name_core(
        branch, allow_empty=True, allow_template_variables=True
    )

    if result.is_valid:
        return ValidationResult.success()
    return ValidationResult.failure(result.error_message)


def _parse_trigger(data: dict[str, Any]) -> TriggerConfig:
    """Parse trigger configuration from dict.

    Supports both Jira and GitHub triggers:
    - Jira triggers use: source="jira", project, jql_filter, tags
    - GitHub triggers use: source="github", project_number, project_scope,
      project_owner, project_filter (preferred)
    - Legacy GitHub triggers use: source="github", repo, query_filter, tags
      (deprecated, will log warnings)

    Raises:
        OrchestrationError: If trigger configuration is invalid.
    """
    source = data.get("source", "jira")
    if source not in ("jira", "github"):
        raise OrchestrationError(f"Invalid trigger source '{source}': must be 'jira' or 'github'")

    repo = data.get("repo", "")
    query_filter = data.get("query_filter", "")
    tags = data.get("tags", [])

    # New GitHub Project-based fields
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
    if source == "github":
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

        # Log deprecation warnings for legacy GitHub fields
        if repo:
            logger.warning(_GITHUB_FIELD_DEPRECATION_MSG.format(field="repo"))
        if query_filter:
            logger.warning(_GITHUB_FIELD_DEPRECATION_MSG.format(field="query_filter"))
        if tags:
            logger.warning(_GITHUB_FIELD_DEPRECATION_MSG.format(field="tags"))

        # Validate repo format if provided (for backwards compatibility)
        if repo:
            is_valid, error_message = _validate_github_repo_format(repo)
            if not is_valid:
                raise OrchestrationError(f"Invalid GitHub repo format '{repo}': {error_message}")

    return TriggerConfig(
        source=source,
        project=data.get("project", ""),
        jql_filter=data.get("jql_filter", ""),
        tags=tags,
        repo=repo,
        query_filter=query_filter,
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
    if agent_type is not None and agent_type not in ("claude", "cursor"):
        raise OrchestrationError(
            f"Invalid agent_type '{agent_type}': must be 'claude' or 'cursor'"
        )

    cursor_mode = data.get("cursor_mode")
    if cursor_mode is not None:
        if cursor_mode not in ("agent", "plan", "ask"):
            raise OrchestrationError(
                f"Invalid cursor_mode '{cursor_mode}': must be 'agent', 'plan', or 'ask'"
            )
        # cursor_mode is only valid when agent_type is 'cursor'
        if agent_type is not None and agent_type != "cursor":
            raise OrchestrationError(
                f"cursor_mode '{cursor_mode}' is only valid when agent_type is 'cursor'"
            )

    return AgentConfig(
        prompt=data.get("prompt", ""),
        tools=data.get("tools", []),
        github=_parse_github_context(data.get("github")),
        timeout_seconds=timeout,
        model=model,
        agent_type=agent_type,
        cursor_mode=cursor_mode,
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
    if max_concurrent is not None:
        if not isinstance(max_concurrent, int) or max_concurrent <= 0:
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
        raise OrchestrationError(
            f"File-level 'enabled' must be a boolean in {file_path}"
        )

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
