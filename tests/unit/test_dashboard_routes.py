"""Tests for dashboard routes.

Tests for the log file discovery API endpoint.
Tests for the SSE log streaming endpoint.
Tests for orchestration toggle API endpoints.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.config import Config, DashboardConfig, ExecutionConfig
from sentinel.dashboard.routes import HEALTH_ENDPOINT_SUNSET_DATE, create_routes
from sentinel.dashboard.state import (
    ExecutionStateSnapshot,
    OrchestrationInfo,
    OrchestrationVersionSnapshot,
    SentinelStateAccessor,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    Orchestration,
    RetryConfig,
    TriggerConfig,
)
from sentinel.state_tracker import CompletedExecutionInfo
from tests.conftest import create_test_app
from tests.helpers import assert_call_args_length


@pytest.fixture(autouse=True)
def reset_sse_app_status() -> Generator[None, None, None]:
    """Reset sse_starlette AppStatus to avoid event loop pollution between tests.

    The sse_starlette library uses a singleton AppStatus with an asyncio.Event
    that can get bound to one event loop. When tests run with different event
    loops, this causes RuntimeError. This fixture resets the AppStatus before
    each test.
    """
    import asyncio

    from sse_starlette.sse import AppStatus

    # Reset the should_exit_event to a new Event
    AppStatus.should_exit_event = asyncio.Event()
    AppStatus.should_exit = False
    yield


@pytest.fixture
def temp_logs_dir() -> Generator[Path, None, None]:
    """Fixture that provides a temporary directory for logs.

    This fixture creates a temporary directory that is automatically
    cleaned up after the test completes.

    Yields:
        Path to the temporary logs directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class MockSentinel:
    """Mock Sentinel instance for testing dashboard state accessor.

    This mock implements the SentinelStateProvider protocol, providing all the
    public methods that the dashboard needs to access state. It does not expose
    any private attributes, demonstrating the decoupled interface.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._shutdown_requested = False
        self._start_time = datetime.now(tz=UTC)
        self._last_jira_poll: datetime | None = None
        self._last_github_poll: datetime | None = None

    @property
    def orchestrations(self) -> list[Any]:
        return []

    def get_hot_reload_metrics(self) -> dict[str, int]:
        return {
            "orchestrations_loaded_total": 0,
            "orchestrations_unloaded_total": 0,
            "orchestrations_reloaded_total": 0,
        }

    def get_running_steps(self) -> list[Any]:
        return []

    def get_issue_queue(self) -> list[Any]:
        return []

    def get_start_time(self) -> datetime:
        return self._start_time

    def get_last_jira_poll(self) -> datetime | None:
        return self._last_jira_poll

    def get_last_github_poll(self) -> datetime | None:
        return self._last_github_poll

    def get_active_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return empty list of active version snapshots."""
        return []

    def get_pending_removal_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return empty list of pending removal version snapshots."""
        return []

    def get_execution_state(self) -> ExecutionStateSnapshot:
        """Return execution state snapshot with zero active executions."""
        return ExecutionStateSnapshot(active_count=0)

    def get_completed_executions(self) -> list[CompletedExecutionInfo]:
        """Return list of completed executions.

        Returns an empty list by default. Override in subclasses or tests
        to provide test data.
        """
        return []

    def is_shutdown_requested(self) -> bool:
        """Return whether shutdown has been requested."""
        return self._shutdown_requested

    def get_service_health_status(self) -> dict[str, dict[str, Any]]:
        """Return service health status.

        Returns an empty dict by default. Override in subclasses or tests
        to provide test data.
        """
        return {}


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_legacy_health_endpoint_returns_deprecation_header(self, temp_logs_dir: Path) -> None:
        """Test that legacy /health endpoint returns Deprecation header per RFC 8594."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
            # Check for Deprecation header per RFC 8594 with HTTP-date format
            assert "Deprecation" in response.headers
            assert response.headers["Deprecation"] == HEALTH_ENDPOINT_SUNSET_DATE

    def test_health_dashboard_endpoint_returns_status(self, temp_logs_dir: Path) -> None:
        """Test that /health/dashboard endpoint returns dashboard status."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health/dashboard")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "up"
            assert data["component"] == "dashboard"
            assert "timestamp" in data
            assert data["message"] == "Dashboard is operational"

    def test_health_dashboard_endpoint_no_deprecation_header(self, temp_logs_dir: Path) -> None:
        """Test that /health/dashboard endpoint does not have Deprecation header."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health/dashboard")

            assert response.status_code == 200
            assert "Deprecation" not in response.headers

    def test_legacy_health_endpoint_returns_sunset_header(self, temp_logs_dir: Path) -> None:
        """Test that legacy /health endpoint returns Sunset header with removal date."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            # Check for Sunset header with planned removal date
            assert "Sunset" in response.headers
            assert response.headers["Sunset"] == HEALTH_ENDPOINT_SUNSET_DATE

    def test_legacy_health_endpoint_returns_link_header(self, temp_logs_dir: Path) -> None:
        """Test that legacy /health endpoint returns Link header pointing to successors."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            # Check for Link header pointing to successor endpoints
            assert "Link" in response.headers
            link_header = response.headers["Link"]
            assert "/health/live" in link_header
            assert "/health/ready" in link_header
            assert 'rel="successor-version"' in link_header

    def test_health_live_endpoint_does_not_have_deprecation_header(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that /health/live endpoint does not have Deprecation header."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health/live")

            assert response.status_code == 200
            assert "Deprecation" not in response.headers

    def test_health_ready_endpoint_does_not_have_deprecation_header(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that /health/ready endpoint does not have Deprecation header."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health/ready")

            assert response.status_code == 200
            assert "Deprecation" not in response.headers


class TestApiLogsFilesEndpoint:
    """Tests for GET /api/logs/files endpoint."""

    def test_returns_empty_list_when_no_logs_dir(self) -> None:
        """Test that endpoint returns empty list when logs directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "nonexistent_logs"
            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert result == []

    def test_returns_empty_list_when_logs_dir_empty(self) -> None:
        """Test that endpoint returns empty list when logs directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert result == []

    def test_discovers_log_files_grouped_by_orchestration(self) -> None:
        """Test that log files are discovered and grouped by orchestration name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Create orchestration directories with log files
            orch1_dir = logs_dir / "my-orchestration"
            orch1_dir.mkdir()
            (orch1_dir / "20260114_100000.log").write_text("Log content 1")
            (orch1_dir / "20260114_110000.log").write_text("Log content 2")

            orch2_dir = logs_dir / "another-orch"
            orch2_dir.mkdir()
            (orch2_dir / "20260114_120000.log").write_text("Log content 3")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            # Should have 2 orchestrations
            assert len(result) == 2

            # Orchestrations should be sorted alphabetically
            assert result[0]["orchestration"] == "another-orch"
            assert result[1]["orchestration"] == "my-orchestration"

            # Check files are present
            assert len(result[0]["files"]) == 1
            assert len(result[1]["files"]) == 2

    def test_log_files_sorted_by_modification_time_newest_first(self) -> None:
        """Test that log files are sorted with most recent first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Create orchestration directory with log files
            orch_dir = logs_dir / "test-orch"
            orch_dir.mkdir()

            # Create files with different modification times
            old_file = orch_dir / "20260114_080000.log"
            new_file = orch_dir / "20260114_120000.log"

            old_file.write_text("Old log")
            new_file.write_text("New log")

            # Ensure modification times are different
            import os
            import time

            old_stat = old_file.stat()
            os.utime(old_file, (old_stat.st_atime, time.time() - 3600))  # 1 hour ago

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            # Should have 1 orchestration with 2 files
            assert len(result) == 1
            files = result[0]["files"]
            assert len(files) == 2

            # Files should be sorted with newest first
            assert files[0]["filename"] == "20260114_120000.log"
            assert files[1]["filename"] == "20260114_080000.log"

    def test_log_file_info_contains_required_fields(self) -> None:
        """Test that each log file entry contains all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_dir = logs_dir / "test-orch"
            orch_dir.mkdir()
            (orch_dir / "20260114_153045.log").write_text("Test content")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert len(result) == 1
            files = result[0]["files"]
            assert len(files) == 1

            file_info = files[0]
            assert "filename" in file_info
            assert "display_name" in file_info
            assert "size" in file_info
            assert "modified" in file_info

            assert file_info["filename"] == "20260114_153045.log"
            assert file_info["display_name"] == "2026-01-14 15:30:45"
            assert file_info["size"] > 0

    def test_directory_structure_matches_expected_format(self) -> None:
        """Test that expected directory structure {base_dir}/{orchestration_name}/{timestamp}.log works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Create the expected directory structure
            # {base_dir}/{orchestration_name}/{timestamp}.log
            orch_name = "my-workflow"
            timestamp = "20260114_143022"

            orch_dir = logs_dir / orch_name
            orch_dir.mkdir()
            log_file = orch_dir / f"{timestamp}.log"
            log_file.write_text("Execution log content")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert len(result) == 1
            assert result[0]["orchestration"] == "my-workflow"
            assert result[0]["files"][0]["filename"] == "20260114_143022.log"

    def test_ignores_non_log_files(self) -> None:
        """Test that non-.log files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_dir = logs_dir / "test-orch"
            orch_dir.mkdir()
            (orch_dir / "20260114_100000.log").write_text("Log content")
            (orch_dir / "readme.txt").write_text("Not a log file")
            (orch_dir / "config.json").write_text("{}")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert len(result) == 1
            files = result[0]["files"]
            # Should only have the .log file
            assert len(files) == 1
            assert files[0]["filename"] == "20260114_100000.log"

    def test_ignores_files_at_root_level(self) -> None:
        """Test that files directly in logs_dir (not in orchestration subdirs) are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # File at root level (should be ignored)
            (logs_dir / "root_file.log").write_text("Root level log")

            # File in orchestration subdirectory (should be included)
            orch_dir = logs_dir / "test-orch"
            orch_dir.mkdir()
            (orch_dir / "20260114_100000.log").write_text("Orchestration log")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            # Should only have the orchestration directory
            assert len(result) == 1
            assert result[0]["orchestration"] == "test-orch"

    def test_skips_empty_orchestration_directories(self) -> None:
        """Test that orchestration directories with no log files are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Empty orchestration directory
            empty_dir = logs_dir / "empty-orch"
            empty_dir.mkdir()

            # Orchestration directory with logs
            with_logs_dir = logs_dir / "has-logs"
            with_logs_dir.mkdir()
            (with_logs_dir / "20260114_100000.log").write_text("Log content")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            # Should only have the directory with logs
            assert len(result) == 1
            assert result[0]["orchestration"] == "has-logs"


class TestSseLogStreamingEndpoint:
    """Tests for GET /api/logs/stream/{orchestration}/{filename} endpoint.

    Tests the Server-Sent Events log streaming endpoint that tails log files
    and streams new lines in real-time.

    Note: Some tests use httpx directly with stream() to handle SSE streaming
    properly, as the TestClient may not handle long-lived connections well.
    Tests for error cases (which exit immediately) use TestClient normally.
    """

    def test_stream_returns_error_when_log_file_not_found(self, temp_logs_dir: Path) -> None:
        """Test that streaming returns error event when log file doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get(
                "/api/logs/stream/nonexistent-orch/20260114_100000.log",
                headers={"Accept": "text/event-stream"},
            )

            assert response.status_code == 200
            content = response.text

            # Should contain an error event
            assert "event: error" in content
            assert "Log file not found" in content

    def test_stream_error_event_contains_json_data(self, temp_logs_dir: Path) -> None:
        """Test that error event data is valid JSON with error field."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get(
                "/api/logs/stream/missing/file.log",
                headers={"Accept": "text/event-stream"},
            )

            content = response.text

            # Parse SSE data line for error
            for line in content.split("\n"):
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    data = json.loads(data_str)
                    assert "error" in data
                    break

    def test_stream_handles_path_traversal_attack(self, temp_logs_dir: Path) -> None:
        """Test that path traversal attempts are blocked."""
        # Create a file outside the logs directory
        secret_file = temp_logs_dir.parent / "secret.txt"
        secret_file.write_text("secret content")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Attempt path traversal - FastAPI may return 404 for invalid paths
            # or 200 with error event, both are valid security responses
            response = client.get(
                "/api/logs/stream/..%2F/secret.txt",
                headers={"Accept": "text/event-stream"},
            )

            # Should NOT contain the secret content regardless of status
            assert "secret content" not in response.text

            # Either 404 or 200 with error is acceptable
            if response.status_code == 200:
                assert "error" in response.text.lower() or "not found" in response.text.lower()

    def test_get_log_file_path_returns_valid_path(self) -> None:
        """Test that get_log_file_path returns correct path for valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_dir = logs_dir / "my-orch"
            orch_dir.mkdir()
            log_file = orch_dir / "20260114_100000.log"
            log_file.write_text("Log content")

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_file_path("my-orch", "20260114_100000.log")

            assert result is not None
            assert result == log_file
            assert result.exists()

    def test_get_log_file_path_returns_none_for_missing_file(self) -> None:
        """Test that get_log_file_path returns None for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_file_path("missing-orch", "missing.log")

            assert result is None

    def test_get_log_file_path_blocks_path_traversal(self) -> None:
        """Test that get_log_file_path blocks path traversal attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Test path traversal blocking - we can't create a file in parent,
            # but we can test the path check

            config = Config(execution=ExecutionConfig(agent_logs_dir=logs_dir))
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            # Try various path traversal attempts
            result1 = accessor.get_log_file_path("..", "etc/passwd")
            result2 = accessor.get_log_file_path("../../../", "secret.txt")
            result3 = accessor.get_log_file_path("orch", "../../../secret.txt")

            assert result1 is None
            assert result2 is None
            assert result3 is None

    def test_stream_endpoint_uses_sse_starlette(self) -> None:
        """Test that the streaming endpoint imports and uses sse-starlette."""
        # Verify that sse_starlette.sse.EventSourceResponse is used
        from sse_starlette.sse import EventSourceResponse as SSEResponse

        from sentinel.dashboard.routes import EventSourceResponse

        assert EventSourceResponse is SSEResponse

    def test_stream_endpoint_exists_with_correct_path(self, temp_logs_dir: Path) -> None:
        """Test that the stream endpoint exists at /api/logs/stream/{orchestration}/{filename}."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        # Verify the route exists
        route_paths = [route.path for route in app.routes]
        assert "/api/logs/stream/{orchestration}/{filename}" in route_paths


class MockSentinelWithOrchestrations(MockSentinel):
    """Mock Sentinel with configurable orchestrations for toggle endpoint tests."""

    def __init__(self, config: Config, orchestrations_data: list[OrchestrationInfo]) -> None:
        super().__init__(config)
        self._orchestrations_data = orchestrations_data

    @property
    def orchestrations(self) -> list[Any]:
        """Return mock orchestrations for state accessor."""
        # Return simple mock objects that the state accessor can convert
        return []


class MockStateAccessorWithOrchestrations(SentinelStateAccessor):
    """Mock state accessor that returns configured orchestration info."""

    def __init__(self, sentinel: Any, orchestrations_data: list[OrchestrationInfo]) -> None:
        self._sentinel = sentinel
        self._orchestrations_data = orchestrations_data

    def get_state(self) -> Any:
        """Return a mock state with the configured orchestrations."""
        from sentinel.dashboard.state import DashboardState

        return DashboardState(
            poll_interval=60,
            max_concurrent_executions=5,
            max_issues_per_poll=10,
            orchestrations=self._orchestrations_data,
        )


class TestToggleOrchestrationEndpoint:
    """Tests for POST /api/orchestrations/{name}/toggle endpoint."""

    def test_toggle_orchestration_success(self, temp_logs_dir: Path) -> None:
        """Test successful toggle of a single orchestration."""
        # Create an orchestration YAML file
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        # Create orchestration info pointing to the real file
        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["enabled"] is False
            assert data["name"] == "test-orch"

            # Verify file was updated
            updated_content = orch_file.read_text()
            assert "enabled: false" in updated_content

    def test_toggle_orchestration_enable(self, temp_logs_dir: Path) -> None:
        """Test enabling a disabled orchestration."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: false
    trigger:
      source: jira
    agent:
      prompt: "Test"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=False,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": True},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["enabled"] is True

            # Verify file was updated
            updated_content = orch_file.read_text()
            assert "enabled: true" in updated_content

    def test_toggle_orchestration_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        # No orchestrations configured
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/nonexistent/toggle",
                json={"enabled": False},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_toggle_orchestration_file_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration file doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        # Orchestration info pointing to non-existent file
        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file="/nonexistent/path.yaml",
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": False},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_toggle_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the toggle endpoint exists at the correct path."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/{name}/toggle" in route_paths


class TestBulkToggleOrchestrationEndpoint:
    """Tests for POST /api/orchestrations/bulk-toggle endpoint."""

    def test_bulk_toggle_jira_orchestrations_success(self, temp_logs_dir: Path) -> None:
        """Test bulk toggling Jira orchestrations by project."""
        # Create an orchestration YAML file with multiple orchestrations
        orch_file = temp_logs_dir / "jira-orchestrations.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "jira-orch-1"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "First"
  - name: "jira-orch-2"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Second"
  - name: "other-project"
    enabled: true
    trigger:
      source: jira
      project: "OTHER"
    agent:
      prompt: "Other project"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="jira-orch-1",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="jira-orch-2",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="Second",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="other-project",
                enabled=True,
                trigger_source="jira",
                trigger_project="OTHER",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="Other project",
                source_file=str(orch_file),
            ),
        ]

        accessor = MockStateAccessorWithOrchestrations(sentinel, orchestrations)
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "jira", "identifier": "TEST", "enabled": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["toggled_count"] == 2

            # Verify file was updated
            updated_content = orch_file.read_text()
            # The TEST project orchestrations should be disabled
            # but we can't easily verify order, just count
            assert updated_content.count("enabled: false") == 2
            # OTHER project should still be enabled
            assert "enabled: true" in updated_content

    def test_bulk_toggle_github_orchestrations_success(self, temp_logs_dir: Path) -> None:
        """Test bulk toggling GitHub orchestrations by project owner."""
        orch_file = temp_logs_dir / "github-orchestrations.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "github-orch-1"
    enabled: true
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "First"
  - name: "github-orch-2"
    enabled: true
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "Second"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="github-orch-1",
                enabled=True,
                trigger_source="github",
                trigger_project=None,
                trigger_project_owner="my-org",
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="github-orch-2",
                enabled=True,
                trigger_source="github",
                trigger_project=None,
                trigger_project_owner="my-org",
                trigger_tags=[],
                agent_prompt_preview="Second",
                source_file=str(orch_file),
            ),
        ]

        accessor = MockStateAccessorWithOrchestrations(sentinel, orchestrations)
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "github", "identifier": "my-org", "enabled": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["toggled_count"] == 2

    def test_bulk_toggle_no_matching_orchestrations(self, temp_logs_dir: Path) -> None:
        """Test 404 when no orchestrations match the identifier."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        # No matching orchestrations
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "jira", "identifier": "NONEXISTENT", "enabled": False},
            )

            assert response.status_code == 404
            assert "No orchestrations found" in response.json()["detail"]

    def test_bulk_toggle_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the bulk toggle endpoint exists at the correct path."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/bulk-toggle" in route_paths

    def test_bulk_toggle_invalid_source(self, temp_logs_dir: Path) -> None:
        """Test validation error for invalid source type."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "invalid", "identifier": "TEST", "enabled": False},
            )

            # Should return 422 for validation error
            assert response.status_code == 422


class TestDeleteOrchestrationEndpoint:
    """Tests for DELETE /api/orchestrations/{name} endpoint."""

    def test_delete_orchestration_success(self, temp_logs_dir: Path) -> None:
        """Test successful deletion of an orchestration."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
  - name: "other-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Other prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.delete("/api/orchestrations/test-orch")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "test-orch"

            # Verify file was updated - orchestration should be removed
            updated_content = orch_file.read_text()
            assert "test-orch" not in updated_content
            assert "other-orch" in updated_content

    def test_delete_orchestration_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.delete("/api/orchestrations/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_delete_orchestration_file_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration file doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file="/nonexistent/path.yaml",
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.delete("/api/orchestrations/test-orch")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_delete_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the delete endpoint exists at the correct path."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/{name}" in route_paths

    def test_delete_orchestration_rate_limited(self, temp_logs_dir: Path) -> None:
        """Test that rapid deletes are rate limited."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "orch-1"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "First"
  - name: "orch-2"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Second"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="orch-1",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="orch-2",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="Second",
                source_file=str(orch_file),
            ),
        ]

        accessor = MockStateAccessorWithOrchestrations(sentinel, orchestrations)
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # First delete should succeed
            response1 = client.delete("/api/orchestrations/orch-1")
            assert response1.status_code == 200

            # Immediate second delete should be rate limited (same file)
            response2 = client.delete("/api/orchestrations/orch-2")
            assert response2.status_code == 429
            assert "Rate limit exceeded" in response2.json()["detail"]


class TestDeleteOrchestrationDebugLogging:
    """Tests for enriched debug logging in delete_orchestration endpoint (DS-771).

    Verifies that the delete_orchestration endpoint emits a debug-level log
    containing request metadata (client info, user-agent, query params, method,
    URL) for detailed troubleshooting when debug logging is enabled.
    """

    def test_delete_orchestration_debug_log_contains_request_metadata(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that delete_orchestration logs enriched debug with request context."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            with TestClient(app) as client:
                client.delete("/api/orchestrations/test-orch")

            # Find the debug call for delete_orchestration
            debug_calls = mock_logger.debug.call_args_list
            delete_debug_call = None
            for call in debug_calls:
                if call[0] and "delete_orchestration called" in str(call[0][0]):
                    delete_debug_call = call
                    break

            assert delete_debug_call is not None, (
                "Expected a debug log for delete_orchestration with request metadata"
            )
            fmt_str = delete_debug_call[0][0]
            assert "client=" in fmt_str
            assert "user_agent=" in fmt_str
            assert "query_params=" in fmt_str
            assert "method=" in fmt_str
            assert "url=" in fmt_str
            # Verify the orchestration name is passed
            assert delete_debug_call[0][1] == "test-orch"

    def test_delete_orchestration_debug_log_method_is_delete(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that the debug log captures DELETE as the HTTP method."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            with TestClient(app) as client:
                client.delete("/api/orchestrations/test-orch")

            debug_calls = mock_logger.debug.call_args_list
            delete_debug_call = None
            for call in debug_calls:
                if call[0] and "delete_orchestration called" in str(call[0][0]):
                    delete_debug_call = call
                    break

            assert delete_debug_call is not None
            # Render the full log message and assert on the output
            # instead of using fragile positional indexing into call args
            rendered = delete_debug_call[0][0] % delete_debug_call[0][1:]
            assert "method=DELETE" in rendered

    def test_delete_orchestration_debug_log_not_found_still_logs(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that debug log is NOT emitted when orchestration is not found.

        The debug log occurs after the info log but before the orchestration
        lookup, so it should still be emitted even for missing orchestrations.
        """
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            with TestClient(app) as client:
                response = client.delete("/api/orchestrations/nonexistent")

            assert response.status_code == 404

            # Debug log should still be emitted before the 404
            debug_calls = mock_logger.debug.call_args_list
            delete_debug_call = None
            for call in debug_calls:
                if call[0] and "delete_orchestration called" in str(call[0][0]):
                    delete_debug_call = call
                    break

            assert delete_debug_call is not None, (
                "Debug log should be emitted even when orchestration is not found"
            )
            assert delete_debug_call[0][1] == "nonexistent"


class TestEditOrchestrationEndpoint:
    """Tests for PUT /api/orchestrations/{name} endpoint (DS-727)."""

    def test_edit_orchestration_success(self, temp_logs_dir: Path) -> None:
        """Test successful edit of an orchestration field via PUT."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Original prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Original prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.put(
                "/api/orchestrations/test-orch",
                json={"agent": {"prompt": "Updated prompt"}},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "test-orch"

            # Verify file was updated
            updated_content = orch_file.read_text()
            assert "Updated prompt" in updated_content

    def test_edit_orchestration_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.put(
                "/api/orchestrations/nonexistent",
                json={"enabled": False},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_edit_orchestration_validation_error(self, temp_logs_dir: Path) -> None:
        """Test 422 when edit produces invalid configuration."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Send invalid max_concurrent (must be positive integer)
            response = client.put(
                "/api/orchestrations/test-orch",
                json={"max_concurrent": -1},
            )

            assert response.status_code == 422

    def test_edit_orchestration_empty_update(self, temp_logs_dir: Path) -> None:
        """Test that empty update returns success without file modification."""
        orch_file = temp_logs_dir / "test-orch.yaml"
        original_content = """
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
"""
        orch_file.write_text(original_content)

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            # Send request with no fields set (all None)
            response = client.put(
                "/api/orchestrations/test-orch",
                json={},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "test-orch"

    def test_edit_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the edit endpoint exists at the correct path."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/{name}" in route_paths

    def test_edit_orchestration_file_not_found(self, temp_logs_dir: Path) -> None:
        """Test 404 when orchestration file doesn't exist."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file="/nonexistent/path.yaml",
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.put(
                "/api/orchestrations/test-orch",
                json={"enabled": False},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()


class TestToggleRateLimiting:
    """Tests for rate limiting on toggle endpoints.

    Rate limiting now uses Config values injected into create_routes().
    Tests configure the rate limiter by passing a Config with custom values.
    """

    def test_toggle_rate_limit_enforced(self, temp_logs_dir: Path) -> None:
        """Test that rapid toggles are rate limited."""
        # Create an orchestration YAML file
        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test prompt"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test prompt",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        # Use default config which has 2.0 second cooldown
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # First toggle should succeed
            response1 = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": False},
            )
            assert response1.status_code == 200

            # Immediate second toggle should be rate limited
            response2 = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": True},
            )
            assert response2.status_code == 429
            assert "Rate limit exceeded" in response2.json()["detail"]

    def test_rate_limit_allows_after_cooldown(self, temp_logs_dir: Path) -> None:
        """Test that toggles are allowed after cooldown period."""
        import time

        orch_file = temp_logs_dir / "test-orch.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test"
""")

        # Use a very short cooldown for testing
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(toggle_cooldown_seconds=0.1),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file=str(orch_file),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # First toggle should succeed
            response1 = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": False},
            )
            assert response1.status_code == 200

            # Wait for cooldown
            time.sleep(0.15)

            # Second toggle should now succeed
            response2 = client.post(
                "/api/orchestrations/test-orch/toggle",
                json={"enabled": True},
            )
            assert response2.status_code == 200

    def test_bulk_toggle_rate_limit_enforced(self, temp_logs_dir: Path) -> None:
        """Test that rapid bulk toggles are rate limited."""
        orch_file = temp_logs_dir / "jira-orchestrations.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "jira-orch-1"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "First"
""")

        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="jira-orch-1",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_project_owner=None,
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
        ]

        accessor = MockStateAccessorWithOrchestrations(sentinel, orchestrations)
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # First bulk toggle should succeed
            response1 = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "jira", "identifier": "TEST", "enabled": False},
            )
            assert response1.status_code == 200

            # Immediate second bulk toggle should be rate limited
            response2 = client.post(
                "/api/orchestrations/bulk-toggle",
                json={"source": "jira", "identifier": "TEST", "enabled": True},
            )
            assert response2.status_code == 429
            assert "Rate limit exceeded" in response2.json()["detail"]


class TestToggleRateLimitConfiguration:
    """Tests for rate limiting configuration.

    Rate limiting configuration is now controlled via Config class.
    Config values are passed to create_routes() and used by the RateLimiter.
    Tests for environment variable parsing are in test_config.py.
    """

    def test_rate_limiter_uses_config_toggle_cooldown(self) -> None:
        """Test that RateLimiter uses toggle_cooldown_seconds from Config."""
        from sentinel.dashboard.routes import RateLimiter

        # Create config with custom cooldown
        config = Config(dashboard=DashboardConfig(toggle_cooldown_seconds=5.0))
        rate_limiter = RateLimiter(config)

        # The rate limiter should use the config value
        assert rate_limiter._toggle_cooldown_seconds == config.dashboard.toggle_cooldown_seconds

    def test_rate_limiter_uses_config_cache_settings(self) -> None:
        """Test that RateLimiter uses cache TTL and maxsize from Config."""
        from cachetools import TTLCache

        from sentinel.dashboard.routes import RateLimiter

        # Create config with custom cache settings
        config = Config(
            dashboard=DashboardConfig(
                rate_limit_cache_ttl=7200,
                rate_limit_cache_maxsize=5000,
            ),
        )
        rate_limiter = RateLimiter(config)

        # The internal cache should use the config values
        assert isinstance(rate_limiter._last_write_times, TTLCache)
        assert rate_limiter._last_write_times.ttl == config.dashboard.rate_limit_cache_ttl
        assert rate_limiter._last_write_times.maxsize == config.dashboard.rate_limit_cache_maxsize

    def test_rate_limiter_defaults_match_config_defaults(self) -> None:
        """Test that default Config values match expected defaults."""
        from sentinel.dashboard.routes import RateLimiter

        # Create config with defaults
        config = Config()
        rate_limiter = RateLimiter(config)

        # Check defaults
        assert rate_limiter._toggle_cooldown_seconds == config.dashboard.toggle_cooldown_seconds
        assert rate_limiter._last_write_times.ttl == config.dashboard.rate_limit_cache_ttl
        assert rate_limiter._last_write_times.maxsize == config.dashboard.rate_limit_cache_maxsize

    def test_create_routes_uses_config_when_provided(self) -> None:
        """Test that create_routes uses provided Config for rate limiting."""
        from sentinel.dashboard.state import SentinelStateAccessor

        config = Config(dashboard=DashboardConfig(toggle_cooldown_seconds=0.5))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        # Creating routes with config should not raise
        router = create_routes(accessor, config=config)
        assert router is not None

    def test_create_routes_uses_default_config_when_not_provided(self) -> None:
        """Test that create_routes creates default Config when not provided."""
        from sentinel.dashboard.state import SentinelStateAccessor

        config = Config()
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        # Creating routes without config should use defaults
        router = create_routes(accessor)
        assert router is not None


class TestToggleOpenApiDocs:
    """Tests for OpenAPI documentation on toggle endpoints."""

    def test_toggle_endpoint_has_openapi_summary(self, temp_logs_dir: Path) -> None:
        """Test that single toggle endpoint has OpenAPI summary."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        openapi_schema = app.openapi()
        toggle_path = openapi_schema["paths"].get("/api/orchestrations/{name}/toggle")

        assert toggle_path is not None
        assert "post" in toggle_path
        assert "summary" in toggle_path["post"]
        assert toggle_path["post"]["summary"] == "Toggle orchestration enabled status"

    def test_toggle_endpoint_has_openapi_description(self, temp_logs_dir: Path) -> None:
        """Test that single toggle endpoint has OpenAPI description."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        openapi_schema = app.openapi()
        toggle_path = openapi_schema["paths"].get("/api/orchestrations/{name}/toggle")

        assert toggle_path is not None
        assert "description" in toggle_path["post"]
        assert "rate limited" in toggle_path["post"]["description"].lower()

    def test_bulk_toggle_endpoint_has_openapi_summary(self, temp_logs_dir: Path) -> None:
        """Test that bulk toggle endpoint has OpenAPI summary."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        openapi_schema = app.openapi()
        bulk_toggle_path = openapi_schema["paths"].get("/api/orchestrations/bulk-toggle")

        assert bulk_toggle_path is not None
        assert "post" in bulk_toggle_path
        assert "summary" in bulk_toggle_path["post"]
        assert bulk_toggle_path["post"]["summary"] == "Bulk toggle orchestrations by source"

    def test_bulk_toggle_endpoint_has_openapi_description(self, temp_logs_dir: Path) -> None:
        """Test that bulk toggle endpoint has OpenAPI description."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        openapi_schema = app.openapi()
        bulk_toggle_path = openapi_schema["paths"].get("/api/orchestrations/bulk-toggle")

        assert bulk_toggle_path is not None
        assert "description" in bulk_toggle_path["post"]
        assert "rate limited" in bulk_toggle_path["post"]["description"].lower()


class TestCreateRoutesLogging:
    """Tests for debug logging in create_routes function.

    These tests verify that create_routes logs appropriate debug messages
    when using provided Config vs. default Config values. Tests use mock
    logger to check call arguments directly for more robust assertions.
    """

    def test_create_routes_logs_debug_with_provided_config(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs debug message when Config is provided."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(
                toggle_cooldown_seconds=5.0,
                rate_limit_cache_ttl=7200,
                rate_limit_cache_maxsize=5000,
            ),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor, config=config)

            # Verify debug was called with correct arguments
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 5)
            # Check format string contains expected placeholders
            assert "create_routes using %s Config" in call_args[0][0]
            # Check positional arguments: config_source, toggle_cooldown, cache_ttl, cache_maxsize
            assert call_args[0][1] == "provided"
            assert call_args[0][2] == 5.0
            assert call_args[0][3] == 7200
            assert call_args[0][4] == 5000

    def test_create_routes_logs_debug_with_default_config(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs debug message when using default Config."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor)  # No config provided

            # Verify debug was called with correct arguments
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 5)
            # Check format string contains expected placeholders
            assert "create_routes using %s Config" in call_args[0][0]
            # Check positional arguments: config_source, toggle_cooldown, cache_ttl, cache_maxsize
            assert call_args[0][1] == "default"
            assert call_args[0][2] == 2.0  # default toggle_cooldown
            assert call_args[0][3] == 3600  # default cache_ttl
            assert call_args[0][4] == 10000  # default cache_maxsize

    def test_create_routes_logs_toggle_cooldown_value(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs toggle_cooldown_seconds value."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(toggle_cooldown_seconds=3.5),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor, config=config)

            # Verify the toggle_cooldown value is passed correctly
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 3)
            assert call_args[0][2] == 3.5

    def test_create_routes_logs_cache_ttl_value(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs rate_limit_cache_ttl value."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(rate_limit_cache_ttl=1800),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor, config=config)

            # Verify the cache_ttl value is passed correctly
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 4)
            assert call_args[0][3] == 1800

    def test_create_routes_logs_cache_maxsize_value(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs rate_limit_cache_maxsize value."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
            dashboard=DashboardConfig(rate_limit_cache_maxsize=20000),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor, config=config)

            # Verify the cache_maxsize value is passed correctly
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 5)
            assert call_args[0][4] == 20000


class MockSentinelForEditForm(MockSentinel):
    """Mock Sentinel with real orchestrations for edit form endpoint tests.

    Extends MockSentinel to return actual Orchestration instances, which is
    required by SentinelStateAccessor.get_orchestration_detail(). This enables
    testing the GET /partials/orchestration_edit/{name} endpoint that renders
    the edit form template.
    """

    def __init__(
        self,
        config: Config,
        orchestration_list: list[Orchestration] | None = None,
        active_versions: list[OrchestrationVersionSnapshot] | None = None,
    ) -> None:
        super().__init__(config)
        self._orchestration_list = orchestration_list or []
        self._active_versions = active_versions or []

    @property
    def orchestrations(self) -> list[Any]:
        """Return the list of active orchestrations."""
        return self._orchestration_list

    def get_active_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return active version snapshots."""
        return self._active_versions


class TestEditFormPartialEndpoint:
    """Tests for GET /partials/orchestration_edit/{name} endpoint (DS-728).

    Tests the edit form partial that renders a pre-populated HTMX form
    for inline orchestration editing.
    """

    @staticmethod
    def _create_app_with_templates(
        accessor: SentinelStateAccessor,
        config: Config | None = None,
    ) -> FastAPI:
        """Create a test FastAPI app with Jinja2 templates configured.

        The edit form endpoint requires Jinja2 templates, so this helper
        configures the template environment the same way the production
        create_app() does.
        """
        from jinja2 import Environment, FileSystemLoader, select_autoescape

        from sentinel.dashboard.app import TemplateEnvironmentWrapper, format_duration

        app = create_test_app(accessor, config)

        templates_dir = Path(__file__).parent.parent.parent / "src" / "sentinel" / "dashboard" / "templates"
        template_env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            enable_async=True,
        )
        template_env.filters["format_duration"] = format_duration
        app.state.templates = TemplateEnvironmentWrapper(template_env)

        return app

    def _make_orchestration(
        self,
        name: str = "test-orch",
        source: str = "jira",
        project: str = "TEST",
        prompt: str = "Test prompt",
        github: GitHubContext | None = None,
        agent_type: str | None = None,
    ) -> Orchestration:
        """Create a test Orchestration instance."""
        return Orchestration(
            name=name,
            trigger=TriggerConfig(source=source, project=project),
            agent=AgentConfig(
                prompt=prompt,
                github=github,
                agent_type=agent_type,
            ),
            retry=RetryConfig(),
        )

    def test_edit_form_returns_200_for_existing_orchestration(self, temp_logs_dir: Path) -> None:
        """Test that edit form partial returns 200 for an existing orchestration."""
        orch = self._make_orchestration(name="my-orch", prompt="Review the code")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/my-orch")

            assert response.status_code == 200

    def test_edit_form_returns_404_for_nonexistent_orchestration(
        self, temp_logs_dir: Path
    ) -> None:
        """Test that edit form partial returns 404 for unknown orchestration name."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/nonexistent")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_edit_form_contains_orchestration_name(self, temp_logs_dir: Path) -> None:
        """Test that edit form HTML contains the orchestration name."""
        orch = self._make_orchestration(name="review-orch", prompt="Review code")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/review-orch")

            assert response.status_code == 200
            assert "review-orch" in response.text

    def test_edit_form_contains_prompt_text(self, temp_logs_dir: Path) -> None:
        """Test that edit form HTML contains the agent prompt text."""
        orch = self._make_orchestration(name="test-orch", prompt="Analyze and fix the bug")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/test-orch")

            assert response.status_code == 200
            assert "Analyze and fix the bug" in response.text

    def test_edit_form_contains_form_element(self, temp_logs_dir: Path) -> None:
        """Test that edit form HTML contains a form element with correct ID."""
        orch = self._make_orchestration(name="form-orch")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/form-orch")

            assert response.status_code == 200
            assert 'id="edit-form-form-orch"' in response.text

    def test_edit_form_contains_save_and_cancel_buttons(self, temp_logs_dir: Path) -> None:
        """Test that edit form HTML contains Save and Cancel action buttons."""
        orch = self._make_orchestration(name="btn-orch")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/btn-orch")

            assert response.status_code == 200
            assert "Save" in response.text
            assert "Cancel" in response.text

    def test_edit_form_contains_trigger_source_radio(self, temp_logs_dir: Path) -> None:
        """Test that edit form contains trigger source radio buttons."""
        orch = self._make_orchestration(name="radio-orch", source="jira")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/radio-orch")

            assert response.status_code == 200
            assert 'name="trigger_source"' in response.text
            assert 'value="jira"' in response.text
            assert 'value="github"' in response.text

    def test_edit_form_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the edit form endpoint exists at the correct path."""
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/partials/orchestration_edit/{name}" in route_paths

    def test_edit_form_returns_html_content_type(self, temp_logs_dir: Path) -> None:
        """Test that edit form endpoint returns HTML content type."""
        orch = self._make_orchestration(name="html-orch")
        config = Config(execution=ExecutionConfig(agent_logs_dir=temp_logs_dir))
        sentinel = MockSentinelForEditForm(config, orchestration_list=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = self._create_app_with_templates(accessor)

        with TestClient(app) as client:
            response = client.get("/partials/orchestration_edit/html-orch")

            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


class TestCreateOrchestrationEndpoint:
    """Tests for POST /api/orchestrations endpoint (DS-729)."""

    def test_create_orchestration_success(self, temp_logs_dir: Path) -> None:
        """Test successful creation of a new orchestration."""
        # Create orchestrations directory
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Get CSRF token first (DS-736)
            csrf_resp = client.get("/api/csrf-token")
            csrf_token = csrf_resp.json()["csrf_token"]

            response = client.post(
                "/api/orchestrations",
                json={
                    "name": "new-orch",
                    "target_file": "new-file.yaml",
                    "file_trigger": {"source": "jira", "project": "TEST"},
                    "agent": {"prompt": "Test prompt"},
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["name"] == "new-orch"

            # Verify file was created
            created_file = orch_dir / "new-file.yaml"
            assert created_file.exists()
            content = created_file.read_text()
            assert "new-orch" in content

    def test_create_orchestration_duplicate_name(self, temp_logs_dir: Path) -> None:
        """Test 422 when orchestration name already exists."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])

        # Create existing orchestration
        orch_info = OrchestrationInfo(
            name="existing-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_project_owner=None,
            trigger_tags=[],
            agent_prompt_preview="Test",
            source_file=str(orch_dir / "existing.yaml"),
        )

        accessor = MockStateAccessorWithOrchestrations(sentinel, [orch_info])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Get CSRF token first (DS-736)
            csrf_resp = client.get("/api/csrf-token")
            csrf_token = csrf_resp.json()["csrf_token"]

            response = client.post(
                "/api/orchestrations",
                json={
                    "name": "existing-orch",
                    "target_file": "new-file.yaml",
                    "trigger": {"source": "jira", "project": "TEST"},
                    "agent": {"prompt": "Test"},
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 422
            assert "already exists" in response.json()["detail"]

    def test_create_orchestration_path_traversal(self, temp_logs_dir: Path) -> None:
        """Test 422 when target file is outside orchestrations directory."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Get CSRF token first (DS-736)
            csrf_resp = client.get("/api/csrf-token")
            csrf_token = csrf_resp.json()["csrf_token"]

            response = client.post(
                "/api/orchestrations",
                json={
                    "name": "new-orch",
                    "target_file": "../outside.yaml",
                    "trigger": {"source": "jira", "project": "TEST"},
                    "agent": {"prompt": "Test"},
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 422
            assert "not within" in response.json()["detail"]

    def test_create_orchestration_appends_to_existing_file(self, temp_logs_dir: Path) -> None:
        """Test that creation appends to an existing YAML file."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        # Create an existing YAML file with an orchestration
        existing_file = orch_dir / "existing.yaml"
        existing_file.write_text("""
orchestrations:
  - name: "first-orch"
    enabled: true
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "First"
""")

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Get CSRF token first (DS-736)
            csrf_resp = client.get("/api/csrf-token")
            csrf_token = csrf_resp.json()["csrf_token"]

            response = client.post(
                "/api/orchestrations",
                json={
                    "name": "second-orch",
                    "target_file": "existing.yaml",
                    "file_trigger": {"source": "jira", "project": "TEST"},
                    "agent": {"prompt": "Second"},
                },
                headers={"X-CSRF-Token": csrf_token},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

            # Verify both orchestrations exist in the file
            content = existing_file.read_text()
            assert "first-orch" in content
            assert "second-orch" in content

    def test_create_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the create endpoint exists at the correct path."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations" in route_paths


class TestOrchestrationFilesEndpoint:
    """Tests for GET /api/orchestrations/files endpoint (DS-729)."""

    def test_returns_yaml_files(self, temp_logs_dir: Path) -> None:
        """Test that endpoint returns list of YAML files."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()
        (orch_dir / "file1.yaml").write_text("orchestrations: []")
        (orch_dir / "file2.yml").write_text("orchestrations: []")

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            response = client.get("/api/orchestrations/files")

            assert response.status_code == 200
            files = response.json()
            assert "file1.yaml" in files
            assert "file2.yml" in files

    def test_returns_empty_when_no_files(self, temp_logs_dir: Path) -> None:
        """Test that endpoint returns empty list when no YAML files exist."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            response = client.get("/api/orchestrations/files")

            assert response.status_code == 200
            assert response.json() == []

    def test_returns_sorted_files(self, temp_logs_dir: Path) -> None:
        """Test that files are returned in sorted order."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()
        (orch_dir / "c-file.yaml").write_text("orchestrations: []")
        (orch_dir / "a-file.yaml").write_text("orchestrations: []")
        (orch_dir / "b-file.yaml").write_text("orchestrations: []")

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            response = client.get("/api/orchestrations/files")

            assert response.status_code == 200
            files = response.json()
            assert files == ["a-file.yaml", "b-file.yaml", "c-file.yaml"]

    def test_files_endpoint_exists(self, temp_logs_dir: Path) -> None:
        """Test that the files endpoint exists at the correct path."""
        orch_dir = temp_logs_dir / "orchestrations"
        orch_dir.mkdir()

        config = Config(
            execution=ExecutionConfig(
                agent_logs_dir=temp_logs_dir,
                orchestrations_dir=orch_dir,
            ),
        )
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor, config=config)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/files" in route_paths


class TestCsrfTokenRateLimiting:
    """Tests for CSRF token endpoint rate limiting (DS-737)."""

    def test_csrf_token_returns_token(self, temp_logs_dir: Path) -> None:
        """Test that the CSRF token endpoint returns a token."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            response = client.get("/api/csrf-token")

            assert response.status_code == 200
            data = response.json()
            assert "csrf_token" in data
            assert len(data["csrf_token"]) > 0

    def test_csrf_token_rate_limited_after_30_requests(self, temp_logs_dir: Path) -> None:
        """Test that CSRF token endpoint returns 429 after 30 requests per minute."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # Make 30 requests - should all succeed
            for i in range(30):
                response = client.get("/api/csrf-token")
                assert response.status_code == 200, f"Request {i+1} should succeed"

            # 31st request should be rate limited
            response = client.get("/api/csrf-token")
            assert response.status_code == 429
            assert "rate limit" in response.json()["detail"].lower()

    def test_csrf_token_rate_limit_per_client_host(self, temp_logs_dir: Path) -> None:
        """Test that CSRF rate limiting is per-client host."""
        config = Config(
            execution=ExecutionConfig(agent_logs_dir=temp_logs_dir),
        )
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)
        app = create_test_app(accessor, config=config)

        with TestClient(app) as client:
            # A single TestClient IP can make 30 requests
            for _ in range(30):
                response = client.get("/api/csrf-token")
                assert response.status_code == 200

            # Verify the 31st is blocked
            response = client.get("/api/csrf-token")
            assert response.status_code == 429
