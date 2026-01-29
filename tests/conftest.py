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
- ``tests/conftest.py`` - Pytest fixtures, type aliases, and re-exports for backwards compatibility

Dashboard Test Helpers
======================

- ``create_test_app()`` - Create a test FastAPI app with dashboard routes
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI

from sentinel.dashboard.routes import create_routes

if TYPE_CHECKING:
    from sentinel.config import Config
    from sentinel.dashboard.state import SentinelStateAccessor

# Re-export mocks for backwards compatibility with existing test imports
# Re-export build_github_trigger_key from deduplication module
# This provides backwards compatibility for tests that imported from conftest
from sentinel.deduplication import build_github_trigger_key

# Re-export helpers for backwards compatibility with existing test imports
from tests.helpers import (
    assert_call_args_length,
    make_config,
    make_issue,
    make_orchestration,
    set_mtime_in_future,
)
from tests.mocks import (
    MockAgentClient,
    MockAgentClientFactory,
    MockJiraClient,
    MockTagClient,
    TrackingAgentClient,
)

# Type alias for signal handlers to avoid Any type annotations
# Used by tests that mock signal handling behavior
SignalHandler = Callable[[int, FrameType | None], None]

__all__ = [
    # Type aliases (alphabetized)
    "SignalHandler",
    "TestAppFactory",
    # Mocks (alphabetized)
    "MockAgentClient",
    "MockAgentClientFactory",
    "MockJiraClient",
    "MockTagClient",
    "TrackingAgentClient",
    # Helpers (alphabetized)
    "assert_call_args_length",
    "build_github_trigger_key",
    "create_test_app",
    "make_config",
    "make_issue",
    "make_orchestration",
    "set_mtime_in_future",
]

# Type alias for the test app factory fixture
TestAppFactory = Callable[["SentinelStateAccessor", "Config | None"], FastAPI]


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
def temp_orchestrations_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for orchestration files.

    Yields the Path to the temporary directory and cleans it up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_app_factory() -> TestAppFactory:
    """Provide a factory for creating test FastAPI apps with dashboard routes.

    This fixture wraps the create_test_app() helper function, allowing tests
    to request the factory as a parameter instead of importing the helper directly.

    Returns:
        A factory function that creates FastAPI test apps.

    Example:
        def test_example(test_app_factory):
            accessor = SentinelStateAccessor(mock_sentinel)
            app = test_app_factory(accessor)
            with TestClient(app) as client:
                response = client.get("/health/live")
                assert response.status_code == 200

        With optional config parameter for rate limiting settings::

            app = test_app_factory(accessor, config=make_config())
    """
    return create_test_app


# Dashboard test helpers


def create_test_app(
    accessor: SentinelStateAccessor,
    config: Config | None = None,
) -> FastAPI:
    """Create a test FastAPI app with dashboard routes.

    This is a shared helper function for creating test apps with dashboard routes.
    It reduces code duplication across test classes.

    Args:
        accessor: The state accessor for the test.
        config: Optional config for rate limiting settings.

    Returns:
        A FastAPI app with dashboard routes configured.
    """
    app = FastAPI()
    router = create_routes(accessor, config=config)
    app.include_router(router)
    return app
