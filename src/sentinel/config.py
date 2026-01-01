"""Configuration loading from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from environment."""

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


def load_config(env_file: Path | None = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_file: Optional path to .env file. If not provided,
                  looks for .env in current directory.

    Returns:
        Config object with loaded values.
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Config(
        poll_interval=int(os.getenv("SENTINEL_POLL_INTERVAL", "60")),
        max_issues_per_poll=int(os.getenv("SENTINEL_MAX_ISSUES", "50")),
        jira_url=os.getenv("JIRA_URL", ""),
        jira_user=os.getenv("JIRA_USER", ""),
        jira_api_token=os.getenv("JIRA_API_TOKEN", ""),
        log_level=os.getenv("SENTINEL_LOG_LEVEL", "INFO"),
        orchestrations_dir=Path(os.getenv("SENTINEL_ORCHESTRATIONS_DIR", "./orchestrations")),
    )
