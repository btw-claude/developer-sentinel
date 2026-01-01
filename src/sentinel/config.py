"""Configuration loading for .env and YAML orchestration files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration."""

    log_level: str = "INFO"
    poll_interval_seconds: int = 60
    max_retries: int = 3


def load_config(env_file: Path | None = None) -> Config:
    """Load configuration from environment variables.

    Args:
        env_file: Optional path to .env file. If not provided, will look for
                  .env in the current directory.

    Returns:
        Config object with all settings loaded.
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Config(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL_SECONDS", "60")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
    )
