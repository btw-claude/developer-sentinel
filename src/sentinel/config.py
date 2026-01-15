"""Configuration loading from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Valid log levels
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment.

    This dataclass is frozen (immutable) to prevent accidental modification
    after creation.
    """

    # Polling configuration
    poll_interval: int = 60  # seconds
    max_issues_per_poll: int = 50
    max_eager_iterations: int = 10  # Limit consecutive immediate polls

    # Concurrent execution
    max_concurrent_executions: int = 1  # Number of orchestrations to run in parallel

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Paths
    orchestrations_dir: Path = Path("./orchestrations")
    agent_workdir: Path = Path("./workdir")  # Base directory for agent working directories
    agent_logs_dir: Path = Path("./logs")  # Base directory for agent execution logs

    # Jira REST API configuration
    jira_base_url: str = ""  # e.g., "https://yoursite.atlassian.net"
    jira_email: str = ""  # User email for authentication
    jira_api_token: str = ""  # API token for authentication

    # MCP Server Configuration
    mcp_jira_command: str = ""
    mcp_jira_args: list[str] = field(default_factory=list)
    mcp_confluence_command: str = ""
    mcp_confluence_args: list[str] = field(default_factory=list)
    mcp_github_command: str = ""
    mcp_github_args: list[str] = field(default_factory=list)

    # GitHub REST API configuration
    github_token: str = ""  # Personal access token or app token
    github_api_url: str = ""  # Custom API URL for GitHub Enterprise (empty = github.com)

    # Workdir cleanup configuration
    # Via SENTINEL_KEEP_WORKDIR (inverted)
    cleanup_workdir_on_success: bool = True  # Whether to cleanup workdir after successful execution

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
        jira_base_url=os.getenv("JIRA_BASE_URL", ""),
        jira_email=os.getenv("JIRA_EMAIL", ""),
        jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
        mcp_jira_command=os.getenv("MCP_JIRA_COMMAND", ""),
        mcp_jira_args=parse_args("MCP_JIRA_ARGS"),
        mcp_confluence_command=os.getenv("MCP_CONFLUENCE_COMMAND", ""),
        mcp_confluence_args=parse_args("MCP_CONFLUENCE_ARGS"),
        mcp_github_command=os.getenv("MCP_GITHUB_COMMAND", ""),
        mcp_github_args=parse_args("MCP_GITHUB_ARGS"),
        github_token=os.getenv("GITHUB_TOKEN", ""),
        github_api_url=os.getenv("GITHUB_API_URL", ""),
        cleanup_workdir_on_success=cleanup_workdir_on_success,
    )
