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
    max_eager_iterations: int = 10  # Deprecated: No longer used, kept for backward compatibility

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

    max_eager = _parse_positive_int(
        os.getenv("SENTINEL_MAX_EAGER_ITERATIONS", "10"),
        "SENTINEL_MAX_EAGER_ITERATIONS",
        10,
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

    # Parse MCP server args (comma-separated)
    def parse_args(env_var: str) -> list[str]:
        value = os.getenv(env_var, "")
        return [arg.strip() for arg in value.split(",") if arg.strip()] if value else []

    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_eager_iterations=max_eager,
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
    )
