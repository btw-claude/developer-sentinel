"""Configuration loading from environment variables.

This module provides a hierarchical configuration system organized into focused
sub-configs for each subsystem:

- JiraConfig: Jira REST API settings
- GitHubConfig: GitHub REST API settings
- DashboardConfig: Dashboard server settings
- RateLimitConfig: Claude API rate limiting settings
- CircuitBreakerConfig: Circuit breaker settings for external services
- ExecutionConfig: Agent execution settings (paths, concurrency, timeouts)

The main Config class composes these sub-configs while maintaining backward
compatibility by exposing all fields as top-level properties.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from sentinel.branch_validation import validate_branch_name_core
from sentinel.types import (
    VALID_AGENT_TYPES,
    VALID_CURSOR_MODES,
    VALID_RATE_LIMIT_STRATEGIES,
    AgentType,
    CursorMode,
    RateLimitStrategy,
)

if TYPE_CHECKING:
    pass

# Valid log levels
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Port validation bounds
MIN_PORT = 1
MAX_PORT = 65535


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
    """

    enabled: bool = False
    port: int = 8080
    host: str = "127.0.0.1"
    toggle_cooldown_seconds: float = 2.0
    rate_limit_cache_ttl: int = 3600
    rate_limit_cache_maxsize: int = 10000


@dataclass(frozen=True)
class RateLimitConfig:
    """Claude API rate limiting configuration.

    Attributes:
        enabled: Enable/disable rate limiting for Claude API calls.
        per_minute: Maximum requests per minute.
        per_hour: Maximum requests per hour.
        strategy: Strategy when limit reached ("queue" or "reject").
        warning_threshold: Fraction of remaining capacity that triggers warning.
    """

    enabled: bool = True
    per_minute: int = 60
    per_hour: int = 1000
    strategy: str = "queue"
    warning_threshold: float = 0.2


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

    This dataclass composes focused sub-configs for each subsystem while
    maintaining backward compatibility by exposing all fields as properties.

    Sub-configs:
        jira: Jira REST API settings
        github: GitHub REST API settings
        dashboard: Dashboard server settings
        rate_limit: Claude API rate limiting settings
        circuit_breaker: Circuit breaker settings
        health_check: Health check settings
        execution: Agent execution settings
        cursor: Cursor CLI settings
        logging: Logging settings
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
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    polling: PollingConfig = field(default_factory=PollingConfig)

    # ==========================================================================
    # Backward compatibility properties - Polling
    # ==========================================================================

    @property
    def poll_interval(self) -> int:
        """Seconds between polling cycles."""
        return self.polling.interval

    @property
    def max_issues_per_poll(self) -> int:
        """Maximum issues to fetch per poll."""
        return self.polling.max_issues_per_poll

    # ==========================================================================
    # Backward compatibility properties - Logging
    # ==========================================================================

    @property
    def log_level(self) -> str:
        """Log level."""
        return self.logging_config.level

    @property
    def log_json(self) -> bool:
        """Whether to output logs in JSON format."""
        return self.logging_config.json

    # ==========================================================================
    # Backward compatibility properties - Execution
    # ==========================================================================

    @property
    def max_concurrent_executions(self) -> int:
        """Number of orchestrations to run in parallel."""
        return self.execution.max_concurrent_executions

    @property
    def orchestrations_dir(self) -> Path:
        """Directory containing orchestration files."""
        return self.execution.orchestrations_dir

    @property
    def agent_workdir(self) -> Path:
        """Base directory for agent working directories."""
        return self.execution.agent_workdir

    @property
    def agent_logs_dir(self) -> Path:
        """Base directory for agent execution logs."""
        return self.execution.agent_logs_dir

    @property
    def orchestration_logs_dir(self) -> Path | None:
        """Per-orchestration log directory (optional)."""
        return self.execution.orchestration_logs_dir

    @property
    def cleanup_workdir_on_success(self) -> bool:
        """Whether to cleanup workdir after successful execution."""
        return self.execution.cleanup_workdir_on_success

    @property
    def disable_streaming_logs(self) -> bool:
        """Whether to disable streaming log writes."""
        return self.execution.disable_streaming_logs

    @property
    def subprocess_timeout(self) -> float:
        """Default timeout for subprocess calls."""
        return self.execution.subprocess_timeout

    @property
    def default_base_branch(self) -> str:
        """Default base branch for new branch creation."""
        return self.execution.default_base_branch

    @property
    def attempt_counts_ttl(self) -> int:
        """TTL for attempt counts cache."""
        return self.execution.attempt_counts_ttl

    @property
    def max_queue_size(self) -> int:
        """Maximum number of issues in the queue."""
        return self.execution.max_queue_size

    @property
    def inter_message_times_threshold(self) -> int:
        """Threshold for summarizing timing metrics."""
        return self.execution.inter_message_times_threshold

    # ==========================================================================
    # Backward compatibility properties - Jira
    # ==========================================================================

    @property
    def jira_base_url(self) -> str:
        """Jira instance URL."""
        return self.jira.base_url

    @property
    def jira_email(self) -> str:
        """User email for Jira authentication."""
        return self.jira.email

    @property
    def jira_api_token(self) -> str:
        """API token for Jira authentication."""
        return self.jira.api_token

    @property
    def jira_epic_link_field(self) -> str:
        """Custom field ID for epic links."""
        return self.jira.epic_link_field

    @property
    def jira_configured(self) -> bool:
        """Check if Jira REST API is configured."""
        return self.jira.configured

    # ==========================================================================
    # Backward compatibility properties - GitHub
    # ==========================================================================

    @property
    def github_token(self) -> str:
        """GitHub personal access token or app token."""
        return self.github.token

    @property
    def github_api_url(self) -> str:
        """Custom API URL for GitHub Enterprise."""
        return self.github.api_url

    @property
    def github_configured(self) -> bool:
        """Check if GitHub REST API is configured."""
        return self.github.configured

    # ==========================================================================
    # Backward compatibility properties - Dashboard
    # ==========================================================================

    @property
    def dashboard_enabled(self) -> bool:
        """Whether to enable the web dashboard."""
        return self.dashboard.enabled

    @property
    def dashboard_port(self) -> int:
        """Port for the dashboard server."""
        return self.dashboard.port

    @property
    def dashboard_host(self) -> str:
        """Host to bind the dashboard server."""
        return self.dashboard.host

    @property
    def toggle_cooldown_seconds(self) -> float:
        """Cooldown period between writes to the same file."""
        return self.dashboard.toggle_cooldown_seconds

    @property
    def rate_limit_cache_ttl(self) -> int:
        """TTL in seconds for rate limit cache entries."""
        return self.dashboard.rate_limit_cache_ttl

    @property
    def rate_limit_cache_maxsize(self) -> int:
        """Maximum number of entries in the rate limit cache."""
        return self.dashboard.rate_limit_cache_maxsize

    # ==========================================================================
    # Backward compatibility properties - Rate Limiting
    # ==========================================================================

    @property
    def claude_rate_limit_enabled(self) -> bool:
        """Enable/disable rate limiting for Claude API calls."""
        return self.rate_limit.enabled

    @property
    def claude_rate_limit_per_minute(self) -> int:
        """Maximum requests per minute."""
        return self.rate_limit.per_minute

    @property
    def claude_rate_limit_per_hour(self) -> int:
        """Maximum requests per hour."""
        return self.rate_limit.per_hour

    @property
    def claude_rate_limit_strategy(self) -> str:
        """Strategy when rate limit is reached."""
        return self.rate_limit.strategy

    @property
    def claude_rate_limit_warning_threshold(self) -> float:
        """Fraction of remaining capacity that triggers warning."""
        return self.rate_limit.warning_threshold

    # ==========================================================================
    # Backward compatibility properties - Circuit Breaker
    # ==========================================================================

    @property
    def circuit_breaker_enabled(self) -> bool:
        """Enable/disable circuit breakers."""
        return self.circuit_breaker.enabled

    @property
    def circuit_breaker_failure_threshold(self) -> int:
        """Number of consecutive failures before opening circuit."""
        return self.circuit_breaker.failure_threshold

    @property
    def circuit_breaker_recovery_timeout(self) -> float:
        """Seconds to wait before attempting recovery."""
        return self.circuit_breaker.recovery_timeout

    @property
    def circuit_breaker_half_open_max_calls(self) -> int:
        """Maximum calls to allow in half-open state."""
        return self.circuit_breaker.half_open_max_calls

    # ==========================================================================
    # Backward compatibility properties - Health Check
    # ==========================================================================

    @property
    def health_check_enabled(self) -> bool:
        """Enable/disable health checks."""
        return self.health_check.enabled

    @property
    def health_check_timeout(self) -> float:
        """Timeout in seconds for individual health checks."""
        return self.health_check.timeout

    # ==========================================================================
    # Backward compatibility properties - Cursor
    # ==========================================================================

    @property
    def default_agent_type(self) -> str:
        """Default agent type (claude or cursor)."""
        return self.cursor.default_agent_type

    @property
    def cursor_path(self) -> str:
        """Path to Cursor CLI executable."""
        return self.cursor.path

    @property
    def cursor_default_model(self) -> str:
        """Default model for Cursor agent."""
        return self.cursor.default_model

    @property
    def cursor_default_mode(self) -> str:
        """Default mode (agent, plan, or ask)."""
        return self.cursor.default_mode


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
    try:
        parsed = float(value)
        if parsed < 0:
            logging.warning(
                "Invalid %s: %f is negative, using default %f",
                name,
                parsed,
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
    try:
        parsed = float(value)
        if parsed < 0.0 or parsed > 1.0:
            logging.warning(
                "Invalid %s: %f is not in range 0.0-1.0, using default %f",
                name,
                parsed,
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

    # Parse MCP server args (comma-separated)
    def parse_args(env_var: str) -> list[str]:
        value = os.getenv(env_var, "")
        return [arg.strip() for arg in value.split(",") if arg.strip()] if value else []

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
    )

    rate_limit_config = RateLimitConfig(
        enabled=claude_rate_limit_enabled,
        per_minute=claude_rate_limit_per_minute,
        per_hour=claude_rate_limit_per_hour,
        strategy=claude_rate_limit_strategy,
        warning_threshold=claude_rate_limit_warning_threshold,
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
    )

    cursor_config = CursorConfig(
        default_agent_type=default_agent_type,
        path=cursor_path,
        default_model=cursor_default_model,
        default_mode=cursor_default_mode,
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
        logging_config=logging_cfg,
        polling=polling_config,
    )


def create_config(
    # Polling settings (backward-compatible with flat params)
    poll_interval: int = 60,
    max_issues_per_poll: int = 50,
    # Execution settings
    max_concurrent_executions: int = 1,
    orchestrations_dir: Path | None = None,
    agent_workdir: Path | None = None,
    agent_logs_dir: Path | None = None,
    orchestration_logs_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
    max_queue_size: int = 100,
    cleanup_workdir_on_success: bool = True,
    disable_streaming_logs: bool = False,
    subprocess_timeout: float = 60.0,
    default_base_branch: str = "main",
    inter_message_times_threshold: int = 100,
    # Logging settings
    log_level: str = "INFO",
    log_json: bool = False,
    # Dashboard settings
    dashboard_enabled: bool = False,
    dashboard_port: int = 8080,
    dashboard_host: str = "127.0.0.1",
    toggle_cooldown_seconds: float = 2.0,
    rate_limit_cache_ttl: int = 3600,
    rate_limit_cache_maxsize: int = 10000,
    # Jira settings
    jira_base_url: str = "",
    jira_email: str = "",
    jira_api_token: str = "",
    jira_epic_link_field: str = "customfield_10014",
    # GitHub settings
    github_token: str = "",
    github_api_url: str = "",
    # Cursor settings
    default_agent_type: str = "claude",
    cursor_path: str = "",
    cursor_default_model: str = "",
    cursor_default_mode: str = "agent",
    # Rate limit settings
    claude_rate_limit_enabled: bool = True,
    claude_rate_limit_per_minute: int = 60,
    claude_rate_limit_per_hour: int = 1000,
    claude_rate_limit_strategy: str = "queue",
    claude_rate_limit_warning_threshold: float = 0.2,
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True,
    circuit_breaker_failure_threshold: int = 5,
    circuit_breaker_recovery_timeout: float = 30.0,
    circuit_breaker_half_open_max_calls: int = 3,
    # Health check settings
    health_check_enabled: bool = True,
    health_check_timeout: float = 5.0,
) -> Config:
    """Create a Config instance from flat parameters (backward-compatible factory).

    This function provides backward compatibility for code that used to create
    Config instances with flat parameters before the sub-config refactoring.
    It accepts the old-style flat parameters and creates the appropriate
    sub-config objects internally.

    Args:
        poll_interval: Seconds between polling cycles.
        max_issues_per_poll: Maximum issues to fetch per poll.
        max_concurrent_executions: Number of orchestrations to run in parallel.
        orchestrations_dir: Directory containing orchestration files.
        agent_workdir: Base directory for agent working directories.
        agent_logs_dir: Base directory for agent execution logs.
        orchestration_logs_dir: Per-orchestration log directory (optional).
        attempt_counts_ttl: TTL for attempt counts cache.
        max_queue_size: Maximum number of issues in the queue.
        cleanup_workdir_on_success: Whether to cleanup workdir after success.
        disable_streaming_logs: Whether to disable streaming log writes.
        subprocess_timeout: Default timeout for subprocess calls.
        default_base_branch: Default base branch for new branch creation.
        inter_message_times_threshold: Threshold for summarizing timing metrics.
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_json: Whether to output logs in JSON format.
        dashboard_enabled: Whether to enable the web dashboard.
        dashboard_port: Port for the dashboard server.
        dashboard_host: Host to bind the dashboard server.
        toggle_cooldown_seconds: Cooldown period for dashboard toggles.
        rate_limit_cache_ttl: TTL for rate limit cache entries.
        rate_limit_cache_maxsize: Maximum rate limit cache entries.
        jira_base_url: Jira instance URL.
        jira_email: User email for Jira authentication.
        jira_api_token: API token for Jira authentication.
        jira_epic_link_field: Custom field ID for epic links.
        github_token: GitHub personal access token.
        github_api_url: Custom API URL for GitHub Enterprise.
        default_agent_type: Default agent type (claude or cursor).
        cursor_path: Path to Cursor CLI executable.
        cursor_default_model: Default model for Cursor agent.
        cursor_default_mode: Default mode (agent, plan, or ask).
        claude_rate_limit_enabled: Enable/disable rate limiting.
        claude_rate_limit_per_minute: Maximum requests per minute.
        claude_rate_limit_per_hour: Maximum requests per hour.
        claude_rate_limit_strategy: Strategy when limit reached.
        claude_rate_limit_warning_threshold: Warning threshold fraction.
        circuit_breaker_enabled: Enable/disable circuit breakers.
        circuit_breaker_failure_threshold: Failures before opening circuit.
        circuit_breaker_recovery_timeout: Recovery timeout in seconds.
        circuit_breaker_half_open_max_calls: Max calls in half-open state.
        health_check_enabled: Enable/disable health checks.
        health_check_timeout: Health check timeout in seconds.

    Returns:
        A Config instance with the specified parameters organized into sub-configs.
    """
    return Config(
        polling=PollingConfig(
            interval=poll_interval,
            max_issues_per_poll=max_issues_per_poll,
        ),
        execution=ExecutionConfig(
            max_concurrent_executions=max_concurrent_executions,
            orchestrations_dir=orchestrations_dir or Path("./orchestrations"),
            agent_workdir=agent_workdir or Path("./workdir"),
            agent_logs_dir=agent_logs_dir or Path("./logs"),
            orchestration_logs_dir=orchestration_logs_dir,
            attempt_counts_ttl=attempt_counts_ttl,
            max_queue_size=max_queue_size,
            cleanup_workdir_on_success=cleanup_workdir_on_success,
            disable_streaming_logs=disable_streaming_logs,
            subprocess_timeout=subprocess_timeout,
            default_base_branch=default_base_branch,
            inter_message_times_threshold=inter_message_times_threshold,
        ),
        logging_config=LoggingConfig(
            level=log_level,
            json=log_json,
        ),
        dashboard=DashboardConfig(
            enabled=dashboard_enabled,
            port=dashboard_port,
            host=dashboard_host,
            toggle_cooldown_seconds=toggle_cooldown_seconds,
            rate_limit_cache_ttl=rate_limit_cache_ttl,
            rate_limit_cache_maxsize=rate_limit_cache_maxsize,
        ),
        jira=JiraConfig(
            base_url=jira_base_url,
            email=jira_email,
            api_token=jira_api_token,
            epic_link_field=jira_epic_link_field,
        ),
        github=GitHubConfig(
            token=github_token,
            api_url=github_api_url,
        ),
        cursor=CursorConfig(
            default_agent_type=default_agent_type,
            path=cursor_path,
            default_model=cursor_default_model,
            default_mode=cursor_default_mode,
        ),
        rate_limit=RateLimitConfig(
            enabled=claude_rate_limit_enabled,
            per_minute=claude_rate_limit_per_minute,
            per_hour=claude_rate_limit_per_hour,
            strategy=claude_rate_limit_strategy,
            warning_threshold=claude_rate_limit_warning_threshold,
        ),
        circuit_breaker=CircuitBreakerConfig(
            enabled=circuit_breaker_enabled,
            failure_threshold=circuit_breaker_failure_threshold,
            recovery_timeout=circuit_breaker_recovery_timeout,
            half_open_max_calls=circuit_breaker_half_open_max_calls,
        ),
        health_check=HealthCheckConfig(
            enabled=health_check_enabled,
            timeout=health_check_timeout,
        ),
    )
