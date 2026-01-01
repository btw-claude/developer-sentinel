"""Configuration loading from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Valid log levels
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment.

    This dataclass is frozen (immutable) to prevent accidental modification
    after creation.
    """

    # Polling configuration
    poll_interval: int = 60  # seconds
    max_issues_per_poll: int = 50

    # Jira configuration
    jira_url: str = ""
    jira_user: str = ""
    jira_api_token: str = ""

    # Logging
    log_level: str = "INFO"

    # Paths
    orchestrations_dir: Path = Path("./orchestrations")


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

    # Validate log level
    log_level = _validate_log_level(
        os.getenv("SENTINEL_LOG_LEVEL", "INFO"),
    )

    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        jira_url=os.getenv("JIRA_URL", ""),
        jira_user=os.getenv("JIRA_USER", ""),
        jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
        log_level=log_level,
        orchestrations_dir=Path(os.getenv("SENTINEL_ORCHESTRATIONS_DIR", "./orchestrations")),
    )
