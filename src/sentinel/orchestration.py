"""Orchestration configuration schema and loading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml

from sentinel.logging import get_logger

logger = get_logger(__name__)


# Pre-compiled regex patterns for GitHub repository validation
# These are compiled at module level to avoid overhead on each validation call
# See _validate_github_repo_format() for usage and format documentation

# Owner pattern: starts with alphanumeric, ends with alphanumeric,
# middle can have alphanumeric or single hyphens (no consecutive hyphens)
_GITHUB_OWNER_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9]|-(?!-))*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")

# Repo name pattern: alphanumeric, hyphens, underscores, periods
_GITHUB_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$|^[a-zA-Z0-9]$")


class ValidationResult(NamedTuple):
    """Result of a validation operation.

    Provides a structured result with named fields for better semantics
    and self-documenting code.

    Attributes:
        is_valid: Whether the validation passed.
        error_message: Error message if validation failed, empty string if valid.

    Example usage:
        # Using factory methods (preferred):
        return ValidationResult.success()
        return ValidationResult.failure("error details")

        # Using direct instantiation:
        return ValidationResult(is_valid=True, error_message="")
        return ValidationResult(is_valid=False, error_message="error details")
    """

    is_valid: bool
    error_message: str

    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result.

        Returns:
            A ValidationResult with is_valid=True and empty error_message.
        """
        return cls(is_valid=True, error_message="")

    @classmethod
    def failure(cls, message: str) -> "ValidationResult":
        """Create a failed validation result.

        Args:
            message: The error message describing why validation failed.

        Returns:
            A ValidationResult with is_valid=False and the provided error_message.
        """
        return cls(is_valid=False, error_message=message)


@dataclass
class TriggerConfig:
    """Configuration for what triggers an orchestration.

    This class supports both Jira and GitHub triggers. Fields are source-specific:

    **Common fields (used by both sources):**
        - source: The source system ("jira" or "github")
        - tags: List of tags/labels to filter by (case-insensitive matching)

    **Jira-specific fields (ignored when source="github"):**
        - project: Jira project key (e.g., "PROJ")
        - jql_filter: Additional JQL filter conditions

    **GitHub-specific fields (ignored when source="jira"):**
        - repo: Repository in "owner/repo-name" format (e.g., "octocat/hello-world")
        - query_filter: Additional GitHub search syntax (e.g., "is:pr draft:false")

    **Tag matching behavior:**
        When multiple tags are specified, issues must have ALL tags to match (AND logic).
        For example, tags: ["needs-review", "priority-high"] will only match issues
        that have both labels applied.

    Attributes:
        source: The source system for triggers ("jira" or "github").
        project: Jira project key. Only used when source is "jira".
            Ignored when source is "github".
        jql_filter: JQL filter for Jira queries. Only used when source is "jira".
            Ignored when source is "github".
        tags: List of tags/labels to filter by. Used by both Jira and GitHub.
            Issues must have ALL specified tags to match (AND logic).
        repo: GitHub repository in "owner/repo-name" format. Only used when
            source is "github". Ignored when source is "jira".
        query_filter: GitHub search syntax filter. Only used when source is "github".
            Ignored when source is "jira". Supports GitHub search qualifiers like
            "is:pr", "is:issue", "draft:false", "author:username", etc.
    """

    source: Literal["jira", "github"] = "jira"
    project: str = ""
    jql_filter: str = ""
    tags: list[str] = field(default_factory=list)
    repo: str = ""
    query_filter: str = ""


@dataclass
class GitHubContext:
    """GitHub repository context for agent operations."""

    host: str = "github.com"
    org: str = ""
    repo: str = ""


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
    """

    prompt: str = ""
    tools: list[str] = field(default_factory=list)
    github: GitHubContext | None = None
    timeout_seconds: int | None = None
    model: str | None = None


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


class OrchestrationError(Exception):
    """Raised when orchestration configuration is invalid."""

    pass


def _validate_github_repo_format(repo: str) -> ValidationResult:
    """Validate that a GitHub repo string is in 'owner/repo-name' format.

    Enforces GitHub-specific character restrictions:

    **Username/Organization rules (owner):**
        - Maximum 39 characters
        - Alphanumeric characters and hyphens only
        - Cannot start or end with a hyphen
        - Cannot have consecutive hyphens
        - Cannot be a reserved name ("." or "..")

    **Repository name rules:**
        - Maximum 100 characters
        - Alphanumeric characters, hyphens, underscores, and periods
        - Cannot start with a period
        - Cannot end with .git (case-insensitive)
        - Cannot be a reserved name ("." or "..")

    Args:
        repo: The repository string to validate.

    Returns:
        A ValidationResult with is_valid and error_message fields.
        If valid, error_message is empty.

    Valid formats:
        - "owner/repo"
        - "owner/repo-name"
        - "org-name/repo_name"
        - "user123/my.project"
    Invalid formats:
        - "repo" (missing owner)
        - "owner/" (missing repo name)
        - "/repo" (missing owner)
        - "owner/repo/extra" (too many parts)
        - "-owner/repo" (owner starts with hyphen)
        - "owner-/repo" (owner ends with hyphen)
        - "owner/repo.git" (repo ends with .git)
        - "a" * 40 + "/repo" (owner too long)
        - "./repo" (owner is reserved name)
        - "owner/." (repo is reserved name)
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


def _parse_trigger(data: dict[str, Any]) -> TriggerConfig:
    """Parse trigger configuration from dict.

    Supports both Jira and GitHub triggers:
    - Jira triggers use: source="jira", project, jql_filter, tags
    - GitHub triggers use: source="github", repo, query_filter, tags

    Raises:
        OrchestrationError: If trigger configuration is invalid.
    """
    source = data.get("source", "jira")
    if source not in ("jira", "github"):
        raise OrchestrationError(f"Invalid trigger source '{source}': must be 'jira' or 'github'")

    repo = data.get("repo", "")

    # Validate repo format for GitHub triggers
    if source == "github" and repo:
        is_valid, error_message = _validate_github_repo_format(repo)
        if not is_valid:
            raise OrchestrationError(f"Invalid GitHub repo format '{repo}': {error_message}")

    return TriggerConfig(
        source=source,
        project=data.get("project", ""),
        jql_filter=data.get("jql_filter", ""),
        tags=data.get("tags", []),
        repo=repo,
        query_filter=data.get("query_filter", ""),
    )


def _parse_github_context(data: dict[str, Any] | None) -> GitHubContext | None:
    """Parse GitHub context from dict."""
    if not data:
        return None
    return GitHubContext(
        host=data.get("host", "github.com"),
        org=data.get("org", ""),
        repo=data.get("repo", ""),
    )


def _parse_agent(data: dict[str, Any]) -> AgentConfig:
    """Parse agent configuration from dict."""
    timeout = data.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise OrchestrationError(f"Invalid timeout_seconds '{timeout}': must be a positive integer")

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise OrchestrationError(f"Invalid model '{model}': must be a string")

    return AgentConfig(
        prompt=data.get("prompt", ""),
        tools=data.get("tools", []),
        github=_parse_github_context(data.get("github")),
        timeout_seconds=timeout,
        model=model,
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
    )


def _load_orchestration_file_with_counts(file_path: Path) -> tuple[list[Orchestration], int]:
    """Load orchestrations from a file and return both enabled list and total count.

    This is the internal implementation that performs all YAML parsing and filtering.
    It returns both the enabled orchestrations and the total count for logging purposes.

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
