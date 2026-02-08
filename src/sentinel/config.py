"""Configuration loading from environment variables.

This module provides a hierarchical configuration system organized into focused
sub-configs for each subsystem:

- JiraConfig: Jira REST API settings
- GitHubConfig: GitHub REST API settings
- DashboardConfig: Dashboard server settings
- RateLimitConfig: Claude API rate limiting settings
- CircuitBreakerConfig: Circuit breaker settings for external services
- ExecutionConfig: Agent execution settings (paths, concurrency, timeouts)
- CodexConfig: Codex CLI settings

The main Config class composes these sub-configs into a single configuration object.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from sentinel.branch_validation import validate_branch_name_core
from sentinel.types import (
    VALID_AGENT_TYPES,
    VALID_CURSOR_MODES,
    VALID_QUEUE_FULL_STRATEGIES,
    VALID_RATE_LIMIT_STRATEGIES,
    AgentType,
    CursorMode,
    QueueFullStrategy,
    RateLimitStrategy,
)

# Valid log levels
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Port validation bounds
MIN_PORT = 1
MAX_PORT = 65535

# Default success rate thresholds for dashboard display coloring
DEFAULT_GREEN_THRESHOLD = 90.0
DEFAULT_YELLOW_THRESHOLD = 70.0


@dataclass(frozen=True)
class JiraConfig:
    """Jira REST API configuration.

    Attributes:
        base_url: Jira instance URL (e.g., "https://yoursite.atlassian.net").
        email: User email for authentication.
        api_token: API token for authentication.
        epic_link_field: Custom field ID for epic links.
    """

    base_url: str = ""
    email: str = ""
    api_token: str = ""
    epic_link_field: str = "customfield_10014"

    @property
    def configured(self) -> bool:
        """Check if Jira REST API is configured."""
        return bool(self.base_url and self.email and self.api_token)


@dataclass(frozen=True)
class GitHubConfig:
    """GitHub REST API configuration.

    Attributes:
        token: Personal access token or app token.
        api_url: Custom API URL for GitHub Enterprise (empty = github.com).
    """

    token: str = ""
    api_url: str = ""

    @property
    def configured(self) -> bool:
        """Check if GitHub REST API is configured."""
        return bool(self.token)


@dataclass(frozen=True)
class DashboardConfig:
    """Dashboard server configuration.

    Attributes:
        enabled: Whether to enable the web dashboard.
        port: Port for the dashboard server.
        host: Host to bind the dashboard server.
        toggle_cooldown_seconds: Cooldown period between writes to the same file.
        rate_limit_cache_ttl: TTL in seconds for rate limit cache entries.
        rate_limit_cache_maxsize: Maximum number of entries in the rate limit cache.
        success_rate_green_threshold: Minimum success rate (%) for green display.
        success_rate_yellow_threshold: Minimum success rate (%) for yellow display.
    """

    enabled: bool = False
    port: int = 8080
    host: str = "127.0.0.1"
    toggle_cooldown_seconds: float = 2.0
    rate_limit_cache_ttl: int = 3600
    rate_limit_cache_maxsize: int = 10000
    success_rate_green_threshold: float = DEFAULT_GREEN_THRESHOLD
    success_rate_yellow_threshold: float = DEFAULT_YELLOW_THRESHOLD


@dataclass(frozen=True)
class RateLimitConfig:
    """Claude API rate limiting configuration.

    Attributes:
        enabled: Enable/disable rate limiting for Claude API calls.
        per_minute: Maximum requests per minute.
        per_hour: Maximum requests per hour.
        strategy: Strategy when limit reached ("queue" or "reject").
        warning_threshold: Fraction of remaining capacity that triggers warning.
        max_queued: Maximum number of requests that can wait in the queue.
            When the queue is full, new requests are handled according to
            queue_full_strategy. Default is 100.
        queue_full_strategy: Strategy when queue is full ("reject" or "wait").
            "reject" immediately rejects new requests when queue is full.
            "wait" waits for queue space (subject to timeout). Default is "reject".
    """

    enabled: bool = True
    per_minute: int = 60
    per_hour: int = 1000
    strategy: str = "queue"
    warning_threshold: float = 0.2
    max_queued: int = 100
    queue_full_strategy: str = "reject"


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Circuit breaker configuration for external service calls.

    Attributes:
        enabled: Enable/disable circuit breakers.
        failure_threshold: Number of consecutive failures before opening circuit.
        recovery_timeout: Seconds to wait before attempting recovery.
        half_open_max_calls: Maximum calls to allow in half-open state.
    """

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


@dataclass(frozen=True)
class HealthCheckConfig:
    """Health check configuration.

    Attributes:
        enabled: Enable/disable health checks for external dependencies.
        timeout: Timeout in seconds for individual health checks.
    """

    enabled: bool = True
    timeout: float = 5.0


@dataclass(frozen=True)
class ExecutionConfig:
    """Agent execution configuration.

    Attributes:
        orchestrations_dir: Directory containing orchestration files.
        agent_workdir: Base directory for agent working directories.
        agent_logs_dir: Base directory for agent execution logs.
        orchestration_logs_dir: Per-orchestration log directory (optional).
        max_concurrent_executions: Number of orchestrations to run in parallel.
        cleanup_workdir_on_success: Whether to cleanup workdir after success.
        disable_streaming_logs: Whether to disable streaming log writes.
        subprocess_timeout: Default timeout for subprocess calls.
        default_base_branch: Default base branch for new branch creation.
        attempt_counts_ttl: TTL for attempt counts cache.
        max_queue_size: Maximum number of issues in the queue.
        inter_message_times_threshold: Threshold for summarizing timing metrics.
        shutdown_timeout_seconds: Timeout in seconds for graceful shutdown.
            If executions don't complete within this time, they will be forcefully
            terminated. Set to 0 to wait indefinitely (not recommended).
        max_recent_executions: Maximum number of recent completed executions to retain.
    """

    orchestrations_dir: Path = field(default_factory=lambda: Path("./orchestrations"))
    agent_workdir: Path = field(default_factory=lambda: Path("./workdir"))
    agent_logs_dir: Path = field(default_factory=lambda: Path("./logs"))
    orchestration_logs_dir: Path | None = None
    max_concurrent_executions: int = 1
    cleanup_workdir_on_success: bool = True
    disable_streaming_logs: bool = False
    subprocess_timeout: float = 60.0
    default_base_branch: str = "main"
    attempt_counts_ttl: int = 3600
    max_queue_size: int = 100
    inter_message_times_threshold: int = 100
    shutdown_timeout_seconds: float = 300.0
    max_recent_executions: int = 10


@dataclass(frozen=True)
class CursorConfig:
    """Cursor CLI configuration.

    Attributes:
        default_agent_type: Default agent type (claude or cursor).
        path: Path to Cursor CLI executable.
        default_model: Default model for Cursor agent.
        default_mode: Default mode (agent, plan, or ask).
    """

    default_agent_type: str = "claude"
    path: str = ""
    default_model: str = ""
    default_mode: str = "agent"


@dataclass(frozen=True)
class CodexConfig:
    """Codex CLI configuration.

    Attributes:
        path: Path to Codex CLI executable.
        default_model: Default model for Codex agent.
    """

    path: str = ""
    default_model: str = ""


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json: Whether to output logs in JSON format.
    """

    level: str = "INFO"
    json: bool = False


@dataclass(frozen=True)
class PollingConfig:
    """Polling configuration.

    Attributes:
        interval: Seconds between polling cycles.
        max_issues_per_poll: Maximum issues to fetch per poll.
    """

    interval: int = 60
    max_issues_per_poll: int = 50


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment.

    This dataclass composes focused sub-configs for each subsystem.

    Sub-configs:
        jira: Jira REST API settings
        github: GitHub REST API settings
        dashboard: Dashboard server settings
        rate_limit: Claude API rate limiting settings
        circuit_breaker: Circuit breaker settings
        health_check: Health check settings
        execution: Agent execution settings
        cursor: Cursor CLI settings
        codex: Codex CLI settings
        logging_config: Logging settings
        polling: Polling settings
    """

    # Sub-configs
    jira: JiraConfig = field(default_factory=JiraConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)


def _parse_positive_int(value: str, name: str, default: int) -> int:
    """Parse a string as a positive integer with validation.

    Args:
        value: The string value to parse.
        name: The name of the setting (for error messages).
        default: The default value to use if parsing fails.

    Returns:
        The parsed positive integer, or the default if invalid.

    Logs a warning if the value is invalid.
    """
    try:
        parsed = int(value)
        if parsed <= 0:
            logging.warning(
                "Invalid %s: %d is not positive, using default %d",
                name,
                parsed,
                default,
            )
            return default
        return parsed
    except ValueError:
        logging.warning(
            "Invalid %s: '%s' is not a valid integer, using default %d",
            name,
            value,
            default,
        )
        return default


def _parse_port(value: str, name: str, default: int) -> int:
    """Parse a string as a valid TCP port number with range validation.

    Args:
        value: The string value to parse.
        name: The name of the setting (for error messages).
        default: The default value to use if parsing fails.

    Returns:
        The parsed port number (MIN_PORT-MAX_PORT), or the default if invalid.

    Logs a warning if the value is invalid or out of range.
    """
    try:
        parsed = int(value)
        if parsed < MIN_PORT or parsed > MAX_PORT:
            logging.warning(
                "Invalid %s: %d is not a valid port (must be %d-%d), using default %d",
                name,
                parsed,
                MIN_PORT,
                MAX_PORT,
                default,
            )
            return default
        return parsed
    except ValueError:
        logging.warning(
            "Invalid %s: '%s' is not a valid integer, using default %d",
            name,
            value,
            default,
        )
        return default


def _validate_log_level(value: str, default: str = "INFO") -> str:
    """Validate and normalize a log level string.

    Args:
        value: The log level string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated log level (uppercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.upper()
    if normalized not in VALID_LOG_LEVELS:
        logging.warning(
            "Invalid SENTINEL_LOG_LEVEL: '%s' is not valid, using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_LOG_LEVELS)),
        )
        return default
    return normalized


def _parse_bool(value: str) -> bool:
    """Parse a string as a boolean.

    Args:
        value: The string value to parse.

    Returns:
        True if value is "true", "1", or "yes" (case-insensitive), False otherwise.
    """
    return value.lower() in ("true", "1", "yes")


def _format_bound_message(
    min_val: float | None = None,
    max_val: float | None = None,
) -> str:
    """Format a human-readable bound description for warning messages.

    Args:
        min_val: Optional minimum allowed value (inclusive). None means no lower bound.
        max_val: Optional maximum allowed value (inclusive). None means no upper bound.

    Returns:
        A descriptive string such as "is not in range 0.0-1.0", "must be >= 0.0",
        or "must be <= 1.0".
    """
    if min_val is not None and max_val is not None:
        return f"is not in range {min_val}-{max_val}"
    if min_val is not None:
        return f"must be >= {min_val}"
    if max_val is not None:
        return f"must be <= {max_val}"
    return "is out of range"


def _parse_bounded_float(
    value: str,
    name: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Parse a string as a float with optional bounds validation.

    This is a shared helper for parsing float values with range checks.
    Both ``_parse_non_negative_float`` and ``_parse_warning_threshold``
    delegate to this function.

    Args:
        value: The string value to parse.
        name: The name of the setting (for error messages).
        default: The default value to use if parsing fails.
        min_val: Optional minimum allowed value (inclusive). None means no lower bound.
        max_val: Optional maximum allowed value (inclusive). None means no upper bound.

    Returns:
        The parsed float within the specified bounds, or the default if invalid.

    Logs a warning if the value is invalid or out of range.
    """
    try:
        parsed = float(value)
        if min_val is not None and parsed < min_val:
            logging.warning(
                "Invalid %s: %f %s, using default %f",
                name,
                parsed,
                _format_bound_message(min_val, max_val),
                default,
            )
            return default
        if max_val is not None and parsed > max_val:
            logging.warning(
                "Invalid %s: %f %s, using default %f",
                name,
                parsed,
                _format_bound_message(min_val, max_val),
                default,
            )
            return default
        return parsed
    except ValueError:
        logging.warning(
            "Invalid %s: '%s' is not a valid number, using default %f",
            name,
            value,
            default,
        )
        return default


def _parse_non_negative_float(value: str, name: str, default: float) -> float:
    """Parse a string as a non-negative float with validation.

    Args:
        value: The string value to parse.
        name: The name of the setting (for error messages).
        default: The default value to use if parsing fails.

    Returns:
        The parsed non-negative float, or the default if invalid.

    Logs a warning if the value is invalid.
    """
    return _parse_bounded_float(value, name, default, min_val=0.0)


def _validate_cursor_mode(value: str, default: str = CursorMode.AGENT.value) -> str:
    """Validate and normalize a Cursor mode string.

    Args:
        value: The cursor mode string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated cursor mode (lowercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.lower()
    if not CursorMode.is_valid(normalized):
        logging.warning(
            "Invalid SENTINEL_CURSOR_DEFAULT_MODE: '%s' is not valid, "
            "using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_CURSOR_MODES)),
        )
        return default
    return normalized


def _validate_agent_type(value: str) -> str:
    """Validate and normalize an agent type string.

    Args:
        value: The agent type string to validate.

    Returns:
        The validated agent type (lowercase), or "claude" if invalid.

    Logs a warning if the value is invalid.
    """
    default = AgentType.CLAUDE.value
    normalized = value.lower()
    if not AgentType.is_valid(normalized):
        logging.warning(
            "Invalid SENTINEL_DEFAULT_AGENT_TYPE: '%s' is not valid, "
            "using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_AGENT_TYPES)),
        )
        return default
    return normalized


def _validate_rate_limit_strategy(value: str, default: str = RateLimitStrategy.QUEUE.value) -> str:
    """Validate and normalize a rate limit strategy string.

    Args:
        value: The rate limit strategy string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated strategy (lowercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.lower()
    if not RateLimitStrategy.is_valid(normalized):
        logging.warning(
            "Invalid SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY: '%s' is not valid, "
            "using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_RATE_LIMIT_STRATEGIES)),
        )
        return default
    return normalized


def _validate_queue_full_strategy(
    value: str, default: str = QueueFullStrategy.REJECT.value
) -> str:
    """Validate and normalize a queue full strategy string.

    Args:
        value: The queue full strategy string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated strategy (lowercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.lower()
    if not QueueFullStrategy.is_valid(normalized):
        logging.warning(
            "Invalid SENTINEL_CLAUDE_RATE_LIMIT_QUEUE_FULL_STRATEGY: '%s' is not valid, "
            "using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_QUEUE_FULL_STRATEGIES)),
        )
        return default
    return normalized


def _parse_warning_threshold(value: str, name: str, default: float) -> float:
    """Parse a string as a warning threshold float (0.0-1.0) with validation.

    Args:
        value: The string value to parse.
        name: The name of the setting (for error messages).
        default: The default value to use if parsing fails.

    Returns:
        The parsed threshold value (0.0-1.0), or the default if invalid.

    Logs a warning if the value is invalid or out of range.
    """
    return _parse_bounded_float(value, name, default, min_val=0.0, max_val=1.0)


def _validate_branch_name(value: str, default: str = "main") -> str:
    """Validate and normalize a git branch name string for config.

    This function wraps the shared validate_branch_name_core() function,
    providing config-specific behavior: logging warnings and returning
    a default value when validation fails.

    Git branch naming rules enforced:
    - Cannot start with a hyphen (-) or period (.)
    - Cannot end with a period (.), forward slash (/), or .lock
    - Cannot contain: space, ~, ^, :, ?, *, [, ], \\, @{
    - Cannot contain consecutive periods (..) or forward slashes (//)
    - Cannot be empty

    Args:
        value: The branch name string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated branch name, or the default if invalid.

    Logs a warning if the value is invalid.
    """
    result = validate_branch_name_core(value, allow_empty=False, allow_template_variables=False)

    if result.is_valid:
        return value.strip() if value else default

    # Log a warning with the specific error message
    logging.warning(
        "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' - %s, using default '%s'",
        value,
        result.error_message,
        default,
    )
    return default


def load_config(env_file: Path | None = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_file: Optional path to .env file. If not provided,
                  looks for .env in current directory.

    Returns:
        Config object with loaded values.

    Values are validated and defaults are used for invalid inputs:
    - Integer values must be valid positive integers
    - LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Parse and validate integer settings
    poll_interval = _parse_positive_int(
        os.getenv("SENTINEL_POLL_INTERVAL", "60"),
        "SENTINEL_POLL_INTERVAL",
        60,
    )

    max_issues = _parse_positive_int(
        os.getenv("SENTINEL_MAX_ISSUES", "50"),
        "SENTINEL_MAX_ISSUES",
        50,
    )

    max_concurrent = _parse_positive_int(
        os.getenv("SENTINEL_MAX_CONCURRENT_EXECUTIONS", "1"),
        "SENTINEL_MAX_CONCURRENT_EXECUTIONS",
        1,
    )

    # Validate log level
    log_level = _validate_log_level(
        os.getenv("SENTINEL_LOG_LEVEL", "INFO"),
    )

    # Parse log JSON format option
    log_json = _parse_bool(os.getenv("SENTINEL_LOG_JSON", ""))

    # Parse workdir cleanup option
    # (SENTINEL_KEEP_WORKDIR=true means cleanup_workdir_on_success=False)
    keep_workdir = _parse_bool(os.getenv("SENTINEL_KEEP_WORKDIR", ""))
    cleanup_workdir_on_success = not keep_workdir

    # Parse dashboard configuration
    dashboard_enabled = _parse_bool(os.getenv("SENTINEL_DASHBOARD_ENABLED", ""))
    dashboard_port = _parse_port(
        os.getenv("SENTINEL_DASHBOARD_PORT", "8080"),
        "SENTINEL_DASHBOARD_PORT",
        8080,
    )
    dashboard_host = os.getenv("SENTINEL_DASHBOARD_HOST", "127.0.0.1")

    # Parse attempt counts TTL
    attempt_counts_ttl = _parse_positive_int(
        os.getenv("SENTINEL_ATTEMPT_COUNTS_TTL", "3600"),
        "SENTINEL_ATTEMPT_COUNTS_TTL",
        3600,
    )

    # Parse max queue size
    max_queue_size = _parse_positive_int(
        os.getenv("SENTINEL_MAX_QUEUE_SIZE", "100"),
        "SENTINEL_MAX_QUEUE_SIZE",
        100,
    )

    # Parse disable streaming logs option
    disable_streaming_logs = _parse_bool(os.getenv("SENTINEL_DISABLE_STREAMING_LOGS", ""))

    # Parse orchestration logs dir
    orchestration_logs_dir_str = os.getenv("SENTINEL_ORCHESTRATION_LOGS_DIR", "")
    orchestration_logs_dir = (
        Path(orchestration_logs_dir_str) if orchestration_logs_dir_str else None
    )

    # Parse Cursor CLI configuration
    default_agent_type = _validate_agent_type(
        os.getenv("SENTINEL_DEFAULT_AGENT_TYPE", "claude"),
    )
    cursor_path = os.getenv("SENTINEL_CURSOR_PATH", "")
    cursor_default_model = os.getenv("SENTINEL_CURSOR_DEFAULT_MODEL", "")
    cursor_default_mode = _validate_cursor_mode(
        os.getenv("SENTINEL_CURSOR_DEFAULT_MODE", "agent"),
    )

    # Parse Codex CLI configuration
    codex_path = os.getenv("SENTINEL_CODEX_PATH", "")
    codex_default_model = os.getenv("SENTINEL_CODEX_DEFAULT_MODEL", "")

    # Parse timing metrics threshold
    inter_message_times_threshold = _parse_positive_int(
        os.getenv("SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD", "100"),
        "SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD",
        100,
    )

    # Parse dashboard rate limiting configuration
    toggle_cooldown_seconds = _parse_non_negative_float(
        os.getenv("SENTINEL_TOGGLE_COOLDOWN", "2.0"),
        "SENTINEL_TOGGLE_COOLDOWN",
        2.0,
    )
    rate_limit_cache_ttl = _parse_positive_int(
        os.getenv("SENTINEL_RATE_LIMIT_CACHE_TTL", "3600"),
        "SENTINEL_RATE_LIMIT_CACHE_TTL",
        3600,
    )
    rate_limit_cache_maxsize = _parse_positive_int(
        os.getenv("SENTINEL_RATE_LIMIT_CACHE_MAXSIZE", "10000"),
        "SENTINEL_RATE_LIMIT_CACHE_MAXSIZE",
        10000,
    )
    max_recent_executions = _parse_positive_int(
        os.getenv("SENTINEL_MAX_RECENT_EXECUTIONS", "10"),
        "SENTINEL_MAX_RECENT_EXECUTIONS",
        10,
    )

    # Parse success rate threshold configuration
    success_rate_green_threshold = _parse_non_negative_float(
        os.getenv(
            "SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD", str(DEFAULT_GREEN_THRESHOLD)
        ),
        "SENTINEL_SUCCESS_RATE_GREEN_THRESHOLD",
        DEFAULT_GREEN_THRESHOLD,
    )
    success_rate_yellow_threshold = _parse_non_negative_float(
        os.getenv(
            "SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD", str(DEFAULT_YELLOW_THRESHOLD)
        ),
        "SENTINEL_SUCCESS_RATE_YELLOW_THRESHOLD",
        DEFAULT_YELLOW_THRESHOLD,
    )

    # Parse Claude API rate limiting configuration
    claude_rate_limit_enabled = _parse_bool(os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_ENABLED", "true"))
    claude_rate_limit_per_minute = _parse_positive_int(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_MINUTE", "60"),
        "SENTINEL_CLAUDE_RATE_LIMIT_PER_MINUTE",
        60,
    )
    claude_rate_limit_per_hour = _parse_positive_int(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_PER_HOUR", "1000"),
        "SENTINEL_CLAUDE_RATE_LIMIT_PER_HOUR",
        1000,
    )
    claude_rate_limit_strategy = _validate_rate_limit_strategy(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY", "queue"),
    )
    claude_rate_limit_warning_threshold = _parse_warning_threshold(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD", "0.2"),
        "SENTINEL_CLAUDE_RATE_LIMIT_WARNING_THRESHOLD",
        0.2,
    )
    claude_rate_limit_max_queued = _parse_positive_int(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_MAX_QUEUED", "100"),
        "SENTINEL_CLAUDE_RATE_LIMIT_MAX_QUEUED",
        100,
    )
    claude_rate_limit_queue_full_strategy = _validate_queue_full_strategy(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_QUEUE_FULL_STRATEGY", "reject"),
    )

    # Parse circuit breaker configuration
    circuit_breaker_enabled = _parse_bool(os.getenv("SENTINEL_CIRCUIT_BREAKER_ENABLED", "true"))
    circuit_breaker_failure_threshold = _parse_positive_int(
        os.getenv("SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"),
        "SENTINEL_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        5,
    )
    circuit_breaker_recovery_timeout = _parse_non_negative_float(
        os.getenv("SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "30.0"),
        "SENTINEL_CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
        30.0,
    )
    circuit_breaker_half_open_max_calls = _parse_positive_int(
        os.getenv("SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS", "3"),
        "SENTINEL_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS",
        3,
    )

    # Parse health check configuration
    health_check_enabled = _parse_bool(os.getenv("SENTINEL_HEALTH_CHECK_ENABLED", "true"))
    health_check_timeout = _parse_non_negative_float(
        os.getenv("SENTINEL_HEALTH_CHECK_TIMEOUT", "5.0"),
        "SENTINEL_HEALTH_CHECK_TIMEOUT",
        5.0,
    )

    # Parse default base branch
    default_base_branch = _validate_branch_name(
        os.getenv("SENTINEL_DEFAULT_BASE_BRANCH", "main"),
    )

    # Parse subprocess timeout
    subprocess_timeout = _parse_non_negative_float(
        os.getenv("SENTINEL_SUBPROCESS_TIMEOUT", "60.0"),
        "SENTINEL_SUBPROCESS_TIMEOUT",
        60.0,
    )

    # Parse shutdown timeout
    shutdown_timeout_seconds = _parse_non_negative_float(
        os.getenv("SENTINEL_SHUTDOWN_TIMEOUT_SECONDS", "300.0"),
        "SENTINEL_SHUTDOWN_TIMEOUT_SECONDS",
        300.0,
    )

    # Create sub-configs
    jira_config = JiraConfig(
        base_url=os.getenv("JIRA_BASE_URL", ""),
        email=os.getenv("JIRA_EMAIL", ""),
        api_token=os.getenv("JIRA_API_TOKEN", ""),
        epic_link_field=os.getenv("JIRA_EPIC_LINK_FIELD", "customfield_10014"),
    )

    github_config = GitHubConfig(
        token=os.getenv("GITHUB_TOKEN", ""),
        api_url=os.getenv("GITHUB_API_URL", ""),
    )

    dashboard_config = DashboardConfig(
        enabled=dashboard_enabled,
        port=dashboard_port,
        host=dashboard_host,
        toggle_cooldown_seconds=toggle_cooldown_seconds,
        rate_limit_cache_ttl=rate_limit_cache_ttl,
        rate_limit_cache_maxsize=rate_limit_cache_maxsize,
        success_rate_green_threshold=success_rate_green_threshold,
        success_rate_yellow_threshold=success_rate_yellow_threshold,
    )

    rate_limit_config = RateLimitConfig(
        enabled=claude_rate_limit_enabled,
        per_minute=claude_rate_limit_per_minute,
        per_hour=claude_rate_limit_per_hour,
        strategy=claude_rate_limit_strategy,
        warning_threshold=claude_rate_limit_warning_threshold,
        max_queued=claude_rate_limit_max_queued,
        queue_full_strategy=claude_rate_limit_queue_full_strategy,
    )

    circuit_breaker_config = CircuitBreakerConfig(
        enabled=circuit_breaker_enabled,
        failure_threshold=circuit_breaker_failure_threshold,
        recovery_timeout=circuit_breaker_recovery_timeout,
        half_open_max_calls=circuit_breaker_half_open_max_calls,
    )

    health_check_config = HealthCheckConfig(
        enabled=health_check_enabled,
        timeout=health_check_timeout,
    )

    execution_config = ExecutionConfig(
        orchestrations_dir=Path(os.getenv("SENTINEL_ORCHESTRATIONS_DIR", "./orchestrations")),
        agent_workdir=Path(os.getenv("SENTINEL_AGENT_WORKDIR", "./workdir")),
        agent_logs_dir=Path(os.getenv("SENTINEL_AGENT_LOGS_DIR", "./logs")),
        orchestration_logs_dir=orchestration_logs_dir,
        max_concurrent_executions=max_concurrent,
        cleanup_workdir_on_success=cleanup_workdir_on_success,
        disable_streaming_logs=disable_streaming_logs,
        subprocess_timeout=subprocess_timeout,
        default_base_branch=default_base_branch,
        attempt_counts_ttl=attempt_counts_ttl,
        max_queue_size=max_queue_size,
        inter_message_times_threshold=inter_message_times_threshold,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        max_recent_executions=max_recent_executions,
    )

    cursor_config = CursorConfig(
        default_agent_type=default_agent_type,
        path=cursor_path,
        default_model=cursor_default_model,
        default_mode=cursor_default_mode,
    )

    codex_config = CodexConfig(
        path=codex_path,
        default_model=codex_default_model,
    )

    logging_cfg = LoggingConfig(
        level=log_level,
        json=log_json,
    )

    polling_config = PollingConfig(
        interval=poll_interval,
        max_issues_per_poll=max_issues,
    )

    return Config(
        jira=jira_config,
        github=github_config,
        dashboard=dashboard_config,
        rate_limit=rate_limit_config,
        circuit_breaker=circuit_breaker_config,
        health_check=health_check_config,
        execution=execution_config,
        cursor=cursor_config,
        codex=codex_config,
        logging_config=logging_cfg,
        polling=polling_config,
    )
