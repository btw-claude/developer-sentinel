"""Test helper functions for Developer Sentinel tests.

This module provides utility functions for creating test fixtures and
managing test data. These helpers simplify test setup by providing
sensible defaults while allowing customization.

Usage
=====

Import and use helpers directly in tests::

    from tests.helpers import make_config, make_orchestration, set_mtime_in_future

    def test_example():
        config = make_config(max_concurrent_executions=5)
        orch = make_orchestration(name="test", tags=["review"])
        # ... use in test ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from sentinel.config import Config
from sentinel.orchestration import AgentConfig, Orchestration, TriggerConfig


def make_config(
    poll_interval: int = 60,
    max_issues: int = 50,
    max_concurrent_executions: int = 1,
    orchestrations_dir: Path | None = None,
    attempt_counts_ttl: int = 3600,
    max_queue_size: int = 100,
) -> Config:
    """Create a Config instance for testing.

    Provides sensible defaults for all config options, with the ability
    to override any parameter.

    Args:
        poll_interval: Seconds between polling cycles.
        max_issues: Maximum issues to fetch per poll.
        max_concurrent_executions: Global concurrency limit.
        orchestrations_dir: Directory containing orchestration files.
        attempt_counts_ttl: TTL for attempt counts cache.
        max_queue_size: Maximum size of the execution queue.

    Returns:
        A Config instance with the specified parameters.
    """
    return Config(
        poll_interval=poll_interval,
        max_issues_per_poll=max_issues,
        max_concurrent_executions=max_concurrent_executions,
        orchestrations_dir=orchestrations_dir or Path("orchestrations"),
        attempt_counts_ttl=attempt_counts_ttl,
        max_queue_size=max_queue_size,
    )


def make_orchestration(
    name: str = "test-orch",
    project: str = "TEST",
    tags: list[str] | None = None,
    max_concurrent: int | None = None,
    source: Literal["jira", "github"] = "jira",
    project_number: int | None = None,
    project_owner: str = "",
    project_scope: Literal["org", "user"] = "org",
    project_filter: str = "",
    labels: list[str] | None = None,
) -> Orchestration:
    """Create an Orchestration instance for testing.

    Supports both Jira and GitHub trigger sources.

    Args:
        name: The orchestration name.
        project: The Jira project key (for Jira triggers).
        tags: List of tags to trigger on (for Jira triggers).
        max_concurrent: Optional per-orchestration concurrency limit.
        source: Trigger source, either "jira" or "github".
        project_number: GitHub project number (for GitHub triggers).
        project_owner: GitHub project owner (org or user name).
        project_scope: GitHub project scope ("org" or "user").
        project_filter: GitHub project filter expression.
        labels: List of GitHub labels to filter by.

    Returns:
        An Orchestration instance configured for testing.
    """
    if source == "github":
        trigger = TriggerConfig(
            source="github",
            project_number=project_number,
            project_owner=project_owner,
            project_scope=project_scope,
            project_filter=project_filter,
            labels=labels or [],
        )
    else:
        trigger = TriggerConfig(project=project, tags=tags or [])

    return Orchestration(
        name=name,
        trigger=trigger,
        agent=AgentConfig(prompt="Test prompt", tools=["jira"]),
        max_concurrent=max_concurrent,
    )


def set_mtime_in_future(file_path: Path, seconds_offset: float = 1.0) -> None:
    """Set a file's mtime to a future time to ensure mtime difference detection.

    This helper explicitly sets the file's modification time using os.utime()
    rather than relying on time.sleep() which can be flaky on fast filesystems
    or under heavy load.

    Args:
        file_path: Path to the file to modify.
        seconds_offset: Number of seconds to add to current time (default: 1.0).
    """
    current_stat = file_path.stat()
    new_mtime = current_stat.st_mtime + seconds_offset
    os.utime(file_path, (current_stat.st_atime, new_mtime))
