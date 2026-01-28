"""Shared pytest fixtures for Developer Sentinel tests.

This module provides pytest fixtures and re-exports commonly used mocks and helpers.

Test Setup Guidelines
=====================

**Direct instantiation** is the preferred approach for most tests::

    from tests.mocks import MockJiraClient, MockAgentClient, MockTagClient
    from tests.helpers import make_config, make_orchestration

    def test_example():
        jira_client = MockJiraClient(issues=[...])
        agent_client = MockAgentClient()
        # ... explicit setup visible at call site ...

Pytest fixtures are available for cases where dependency injection is preferred,
particularly for shared test resources like temporary directories.

Module Organization
===================

- ``tests/mocks.py`` - Mock implementations of core interfaces
- ``tests/helpers.py`` - Test helper functions (make_config, make_orchestration, etc.)
- ``tests/conftest.py`` - Pytest fixtures and re-exports for backwards compatibility
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Re-export mocks for backwards compatibility with existing test imports
from tests.mocks import (
    MockJiraClient,
    MockAgentClient,
    MockTagClient,
    TrackingAgentClient,
    MockAgentClientFactory,
)

# Re-export helpers for backwards compatibility with existing test imports
from tests.helpers import (
    make_config,
    make_orchestration,
    set_mtime_in_future,
)

# Re-export build_github_trigger_key from deduplication module
# This provides backwards compatibility for tests that imported from conftest
from sentinel.deduplication import build_github_trigger_key

__all__ = [
    # Mocks
    "MockJiraClient",
    "MockAgentClient",
    "MockTagClient",
    "TrackingAgentClient",
    "MockAgentClientFactory",
    # Helpers
    "make_config",
    "make_orchestration",
    "set_mtime_in_future",
    "build_github_trigger_key",
]


# Pytest fixtures


@pytest.fixture
def mock_jira_client() -> MockJiraClient:
    """Provide a fresh MockJiraClient instance."""
    return MockJiraClient(issues=[])


@pytest.fixture
def mock_agent_client() -> MockAgentClient:
    """Provide a fresh MockAgentClient instance."""
    return MockAgentClient()


@pytest.fixture
def mock_tag_client() -> MockTagClient:
    """Provide a fresh MockTagClient instance."""
    return MockTagClient()


@pytest.fixture
def temp_orchestrations_dir():
    """Provide a temporary directory for orchestration files.

    Yields the Path to the temporary directory and cleans it up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
