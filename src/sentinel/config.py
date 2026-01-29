"""Configuration loading from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Valid log levels
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# Valid Cursor modes
VALID_CURSOR_MODES = frozenset({"agent", "plan", "ask"})

# Valid agent types
VALID_AGENT_TYPES = frozenset({"claude", "cursor"})

# Valid rate limit strategies
VALID_RATE_LIMIT_STRATEGIES = frozenset({"queue", "reject"})

# Port validation bounds
MIN_PORT = 1
MAX_PORT = 65535


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment.

    This dataclass is frozen (immutable) to prevent accidental modification
    after creation.
    """

    # Polling configuration
    poll_interval: int = 60  # seconds
    max_issues_per_poll: int = 50

    # Concurrent execution
    max_concurrent_executions: int = 1  # Number of orchestrations to run in parallel

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Paths
    orchestrations_dir: Path = Path("./orchestrations")
    agent_workdir: Path = Path("./workdir")  # Base directory for agent working directories
    agent_logs_dir: Path = Path("./logs")  # Base directory for agent execution logs
    orchestration_logs_dir: Path | None = None  # Per-orchestration log directory (optional)

    # Jira REST API configuration
    jira_base_url: str = ""  # e.g., "https://yoursite.atlassian.net"
    jira_email: str = ""  # User email for authentication
    jira_api_token: str = ""  # API token for authentication

    # GitHub REST API configuration
    github_token: str = ""  # Personal access token or app token
    github_api_url: str = ""  # Custom API URL for GitHub Enterprise (empty = github.com)

    # Workdir cleanup configuration
    # Via SENTINEL_KEEP_WORKDIR (inverted)
    cleanup_workdir_on_success: bool = True  # Whether to cleanup workdir after successful execution

    # Dashboard configuration
    dashboard_enabled: bool = False  # Whether to enable the web dashboard
    dashboard_port: int = 8080  # Port for the dashboard server
    dashboard_host: str = "127.0.0.1"  # Host to bind the dashboard server

    # Attempt counts cleanup configuration
    # Time in seconds after which inactive attempt count entries are cleaned up
    attempt_counts_ttl: int = 3600  # 1 hour default

    # Issue queue configuration
    # Maximum number of issues that can be held in the queue
    # When the queue is full, oldest items are evicted to make room for new ones
    max_queue_size: int = 100

    # Streaming logs configuration
    # When True, disables streaming log writes during agent execution
    # Uses _run_simple() path and writes full response after completion
    disable_streaming_logs: bool = False

    # Cursor CLI configuration
    default_agent_type: str = "claude"  # Default agent type: claude or cursor
    cursor_path: str = ""  # Path to Cursor CLI executable
    cursor_default_model: str = ""  # Default model for Cursor agent
    cursor_default_mode: str = "agent"  # Default mode: agent, plan, or ask

    # Timing metrics configuration
    # Threshold for summarizing inter_message_times in TimingMetrics
    # When message count exceeds this, store statistical summary instead of raw data
    inter_message_times_threshold: int = 100

    # Dashboard rate limiting configuration
    # Cooldown period in seconds between writes to the same orchestration file
    toggle_cooldown_seconds: float = 2.0
    # TTL in seconds for rate limit cache entries
    rate_limit_cache_ttl: int = 3600  # 1 hour
    # Maximum number of entries in the rate limit cache
    rate_limit_cache_maxsize: int = 10000  # 10k entries

    # Claude API rate limiting configuration
    # Enable/disable rate limiting for Claude API calls
    claude_rate_limit_enabled: bool = True
    # Maximum requests per minute (default: 60)
    claude_rate_limit_per_minute: int = 60
    # Maximum requests per hour (default: 1000)
    claude_rate_limit_per_hour: int = 1000
    # Strategy when rate limit is reached: "queue" (wait) or "reject" (fail immediately)
    claude_rate_limit_strategy: str = "queue"
    # Warning threshold as fraction of remaining capacity (0.0-1.0)
    # When tokens fall below this fraction, warnings are logged
    claude_rate_limit_warning_threshold: float = 0.2

    # Circuit breaker configuration
    # Enable/disable circuit breakers for external service calls (Jira, GitHub, Claude)
    circuit_breaker_enabled: bool = True
    # Number of consecutive failures before opening the circuit (default: 5)
    circuit_breaker_failure_threshold: int = 5
    # Seconds to wait before attempting recovery after circuit opens (default: 30)
    circuit_breaker_recovery_timeout: float = 30.0
    # Maximum calls to allow in half-open state for recovery testing (default: 3)
    circuit_breaker_half_open_max_calls: int = 3

    # Health check configuration
    # Enable/disable health checks for external service dependencies
    health_check_enabled: bool = True
    # Timeout in seconds for individual health checks (default: 5.0)
    health_check_timeout: float = 5.0

    # Base branch configuration
    # Default base branch for new branch creation (default: "main")
    # Can be overridden per-orchestration in GitHubContext.base_branch
    default_base_branch: str = "main"

    @property
    def jira_configured(self) -> bool:
        """Check if Jira REST API is configured."""
        return bool(self.jira_base_url and self.jira_email and self.jira_api_token)

    @property
    def github_configured(self) -> bool:
        """Check if GitHub REST API is configured."""
        return bool(self.github_token)


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


def _validate_cursor_mode(value: str, default: str = "agent") -> str:
    """Validate and normalize a Cursor mode string.

    Args:
        value: The cursor mode string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated cursor mode (lowercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.lower()
    if normalized not in VALID_CURSOR_MODES:
        logging.warning(
            "Invalid SENTINEL_CURSOR_DEFAULT_MODE: '%s' is not valid, using default '%s'. Valid values: %s",
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
    default = "claude"
    normalized = value.lower()
    if normalized not in VALID_AGENT_TYPES:
        logging.warning(
            "Invalid SENTINEL_DEFAULT_AGENT_TYPE: '%s' is not valid, using default '%s'. Valid values: %s",
            value,
            default,
            ", ".join(sorted(VALID_AGENT_TYPES)),
        )
        return default
    return normalized


def _validate_rate_limit_strategy(value: str, default: str = "queue") -> str:
    """Validate and normalize a rate limit strategy string.

    Args:
        value: The rate limit strategy string to validate.
        default: The default value to use if invalid.

    Returns:
        The validated strategy (lowercase), or the default if invalid.

    Logs a warning if the value is invalid.
    """
    normalized = value.lower()
    if normalized not in VALID_RATE_LIMIT_STRATEGIES:
        logging.warning(
            "Invalid SENTINEL_CLAUDE_RATE_LIMIT_STRATEGY: '%s' is not valid, using default '%s'. Valid values: %s",
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
    """Validate and normalize a git branch name string.

    Git branch naming rules enforced:
    - Cannot start with a hyphen (-) or period (.)
    - Cannot end with a period (.) or forward slash (/)
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
    if not value or not value.strip():
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: empty branch name, using default '%s'",
            default,
        )
        return default

    value = value.strip()

    # Check for invalid starting characters
    if value.startswith("-") or value.startswith("."):
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' cannot start with '-' or '.', "
            "using default '%s'",
            value,
            default,
        )
        return default

    # Check for invalid ending characters
    if value.endswith(".") or value.endswith("/"):
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' cannot end with '.' or '/', "
            "using default '%s'",
            value,
            default,
        )
        return default

    # Check for invalid characters
    invalid_chars = set(" ~^:?*[]\\")
    found_invalid = [c for c in value if c in invalid_chars]
    if found_invalid:
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' contains invalid characters %s, "
            "using default '%s'",
            value,
            found_invalid,
            default,
        )
        return default

    # Check for @{ sequence (reflog syntax)
    if "@{" in value:
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' cannot contain '@{', "
            "using default '%s'",
            value,
            default,
        )
        return default

    # Check for consecutive periods or slashes
    if ".." in value or "//" in value:
        logging.warning(
            "Invalid SENTINEL_DEFAULT_BASE_BRANCH: '%s' cannot contain '..' or '//', "
            "using default '%s'",
            value,
            default,
        )
        return default

    return value


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

    # Parse workdir cleanup option (SENTINEL_KEEP_WORKDIR=true means cleanup_workdir_on_success=False)
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
    orchestration_logs_dir = Path(orchestration_logs_dir_str) if orchestration_logs_dir_str else None

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
    claude_rate_limit_enabled = _parse_bool(
        os.getenv("SENTINEL_CLAUDE_RATE_LIMIT_ENABLED", "true")
    )
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
    circuit_breaker_enabled = _parse_bool(
        os.getenv("SENTINEL_CIRCUIT_BREAKER_ENABLED", "true")
    )
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
    health_check_enabled = _parse_bool(
        os.getenv("SENTINEL_HEALTH_CHECK_ENABLED", "true")
    )
    health_check_timeout = _parse_non_negative_float(
        os.getenv("SENTINEL_HEALTH_CHECK_TIMEOUT", "5.0"),
        "SENTINEL_HEALTH_CHECK_TIMEOUT",
        5.0,
    )

    # Parse default base branch
    default_base_branch = _validate_branch_name(
        os.getenv("SENTINEL_DEFAULT_BASE_BRANCH", "main"),
    )

    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent,
        log_level=log_level,
        log_json=log_json,
        orchestrations_dir=Path(os.getenv("SENTINEL_ORCHESTRATIONS_DIR", "./orchestrations")),
        agent_workdir=Path(os.getenv("SENTINEL_AGENT_WORKDIR", "./workdir")),
        agent_logs_dir=Path(os.getenv("SENTINEL_AGENT_LOGS_DIR", "./logs")),
        orchestration_logs_dir=orchestration_logs_dir,
        jira_base_url=os.getenv("JIRA_BASE_URL", ""),
        jira_email=os.getenv("JIRA_EMAIL", ""),
        jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
        github_token=os.getenv("GITHUB_TOKEN", ""),
        github_api_url=os.getenv("GITHUB_API_URL", ""),
        cleanup_workdir_on_success=cleanup_workdir_on_success,
        dashboard_enabled=dashboard_enabled,
        dashboard_port=dashboard_port,
        dashboard_host=dashboard_host,
        attempt_counts_ttl=attempt_counts_ttl,
        max_queue_size=max_queue_size,
        disable_streaming_logs=disable_streaming_logs,
        default_agent_type=default_agent_type,
        cursor_path=cursor_path,
        cursor_default_model=cursor_default_model,
        cursor_default_mode=cursor_default_mode,
        inter_message_times_threshold=inter_message_times_threshold,
        toggle_cooldown_seconds=toggle_cooldown_seconds,
        rate_limit_cache_ttl=rate_limit_cache_ttl,
        rate_limit_cache_maxsize=rate_limit_cache_maxsize,
        claude_rate_limit_enabled=claude_rate_limit_enabled,
        claude_rate_limit_per_minute=claude_rate_limit_per_minute,
        claude_rate_limit_per_hour=claude_rate_limit_per_hour,
        claude_rate_limit_strategy=claude_rate_limit_strategy,
        claude_rate_limit_warning_threshold=claude_rate_limit_warning_threshold,
        circuit_breaker_enabled=circuit_breaker_enabled,
        circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
        circuit_breaker_recovery_timeout=circuit_breaker_recovery_timeout,
        circuit_breaker_half_open_max_calls=circuit_breaker_half_open_max_calls,
        health_check_enabled=health_check_enabled,
        health_check_timeout=health_check_timeout,
        default_base_branch=default_base_branch,
    )
