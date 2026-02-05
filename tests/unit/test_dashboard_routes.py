"""Tests for dashboard routes.

Tests for the log file discovery API endpoint.
Tests for the SSE log streaming endpoint.
Tests for orchestration toggle API endpoints.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sentinel.config import Config, create_config
from sentinel.dashboard.routes import HEALTH_ENDPOINT_SUNSET_DATE, create_routes
from sentinel.dashboard.state import (
    ExecutionStateSnapshot,
    OrchestrationInfo,
    OrchestrationVersionSnapshot,
    SentinelStateAccessor,
)
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
        self._start_time = datetime.now()
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

    def is_shutdown_requested(self) -> bool:
        """Return whether shutdown has been requested."""
        return self._shutdown_requested


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_legacy_health_endpoint_returns_deprecation_header(self, temp_logs_dir: Path) -> None:
        """Test that legacy /health endpoint returns Deprecation header per RFC 8594."""
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
        app = create_test_app(accessor)

        with TestClient(app) as client:
            response = client.get("/health/dashboard")

            assert response.status_code == 200
            assert "Deprecation" not in response.headers

    def test_legacy_health_endpoint_returns_sunset_header(self, temp_logs_dir: Path) -> None:
        """Test that legacy /health endpoint returns Sunset header with removal date."""
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
            config = create_config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert result == []

    def test_returns_empty_list_when_logs_dir_empty(self) -> None:
        """Test that endpoint returns empty list when logs directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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

        config = create_config(agent_logs_dir=temp_logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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
            config = create_config(agent_logs_dir=logs_dir)
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

            config = create_config(agent_logs_dir=logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        # Create orchestration info pointing to the real file
        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_repo=None,
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

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=False,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_repo=None,
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        # Orchestration info pointing to non-existent file
        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_repo=None,
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="jira-orch-1",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_repo=None,
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="jira-orch-2",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_repo=None,
                trigger_tags=[],
                agent_prompt_preview="Second",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="other-project",
                enabled=True,
                trigger_source="jira",
                trigger_project="OTHER",
                trigger_repo=None,
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
        """Test bulk toggling GitHub orchestrations by repo."""
        orch_file = temp_logs_dir / "github-orchestrations.yaml"
        orch_file.write_text("""
orchestrations:
  - name: "github-orch-1"
    enabled: true
    trigger:
      source: github
      repo: "org/repo"
    agent:
      prompt: "First"
  - name: "github-orch-2"
    enabled: true
    trigger:
      source: github
      repo: "org/repo"
    agent:
      prompt: "Second"
""")

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="github-orch-1",
                enabled=True,
                trigger_source="github",
                trigger_project=None,
                trigger_repo="org/repo",
                trigger_tags=[],
                agent_prompt_preview="First",
                source_file=str(orch_file),
            ),
            OrchestrationInfo(
                name="github-orch-2",
                enabled=True,
                trigger_source="github",
                trigger_project=None,
                trigger_repo="org/repo",
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
                json={"source": "github", "identifier": "org/repo", "enabled": False},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["toggled_count"] == 2

    def test_bulk_toggle_no_matching_orchestrations(self, temp_logs_dir: Path) -> None:
        """Test 404 when no orchestrations match the identifier."""
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        route_paths = [route.path for route in app.routes]
        assert "/api/orchestrations/bulk-toggle" in route_paths

    def test_bulk_toggle_invalid_source(self, temp_logs_dir: Path) -> None:
        """Test validation error for invalid source type."""
        config = create_config(agent_logs_dir=temp_logs_dir)
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

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_repo=None,
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
        config = create_config(agent_logs_dir=temp_logs_dir, toggle_cooldown_seconds=0.1)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orch_info = OrchestrationInfo(
            name="test-orch",
            enabled=True,
            trigger_source="jira",
            trigger_project="TEST",
            trigger_repo=None,
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

        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])

        orchestrations = [
            OrchestrationInfo(
                name="jira-orch-1",
                enabled=True,
                trigger_source="jira",
                trigger_project="TEST",
                trigger_repo=None,
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
        config = create_config(toggle_cooldown_seconds=5.0)
        rate_limiter = RateLimiter(config)

        # The rate limiter should use the config value
        assert rate_limiter._toggle_cooldown_seconds == config.dashboard.toggle_cooldown_seconds

    def test_rate_limiter_uses_config_cache_settings(self) -> None:
        """Test that RateLimiter uses cache TTL and maxsize from Config."""
        from cachetools import TTLCache

        from sentinel.dashboard.routes import RateLimiter

        # Create config with custom cache settings
        config = create_config(rate_limit_cache_ttl=7200, rate_limit_cache_maxsize=5000)
        rate_limiter = RateLimiter(config)

        # The internal cache should use the config values
        assert isinstance(rate_limiter._last_write_times, TTLCache)
        assert rate_limiter._last_write_times.ttl == config.dashboard.rate_limit_cache_ttl
        assert rate_limiter._last_write_times.maxsize == config.dashboard.rate_limit_cache_maxsize

    def test_rate_limiter_defaults_match_config_defaults(self) -> None:
        """Test that default Config values match expected defaults."""
        from sentinel.dashboard.routes import RateLimiter

        # Create config with defaults
        config = create_config()
        rate_limiter = RateLimiter(config)

        # Check defaults
        assert rate_limiter._toggle_cooldown_seconds == config.dashboard.toggle_cooldown_seconds
        assert rate_limiter._last_write_times.ttl == config.dashboard.rate_limit_cache_ttl
        assert rate_limiter._last_write_times.maxsize == config.dashboard.rate_limit_cache_maxsize

    def test_create_routes_uses_config_when_provided(self) -> None:
        """Test that create_routes uses provided Config for rate limiting."""
        from sentinel.dashboard.state import SentinelStateAccessor

        config = create_config(toggle_cooldown_seconds=0.5)
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        # Creating routes with config should not raise
        router = create_routes(accessor, config=config)
        assert router is not None

    def test_create_routes_uses_default_config_when_not_provided(self) -> None:
        """Test that create_routes creates default Config when not provided."""
        from sentinel.dashboard.state import SentinelStateAccessor

        config = create_config()
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        # Creating routes without config should use defaults
        router = create_routes(accessor)
        assert router is not None


class TestToggleOpenApiDocs:
    """Tests for OpenAPI documentation on toggle endpoints."""

    def test_toggle_endpoint_has_openapi_summary(self, temp_logs_dir: Path) -> None:
        """Test that single toggle endpoint has OpenAPI summary."""
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir)
        sentinel = MockSentinelWithOrchestrations(config, [])
        accessor = MockStateAccessorWithOrchestrations(sentinel, [])
        app = create_test_app(accessor)

        openapi_schema = app.openapi()
        bulk_toggle_path = openapi_schema["paths"].get("/api/orchestrations/bulk-toggle")

        assert bulk_toggle_path is not None
        assert "description" in bulk_toggle_path["post"]
        assert "rate limited" in bulk_toggle_path["post"]["description"].lower()


class TestDeprecatedModuleLevelConstants:
    """Tests for deprecated module-level constants.

    These tests verify that accessing the legacy module-level constants
    (_DEFAULT_TOGGLE_COOLDOWN, _DEFAULT_RATE_LIMIT_CACHE_TTL,
    _DEFAULT_RATE_LIMIT_CACHE_MAXSIZE) emits DeprecationWarning while
    still returning the correct values for backward compatibility.
    """

    def test_default_toggle_cooldown_emits_deprecation_warning(self) -> None:
        """Test that accessing _DEFAULT_TOGGLE_COOLDOWN emits DeprecationWarning."""
        import warnings

        from sentinel.dashboard import routes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = routes._DEFAULT_TOGGLE_COOLDOWN

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_DEFAULT_TOGGLE_COOLDOWN" in str(w[0].message)
            assert "Config.toggle_cooldown_seconds" in str(w[0].message)
            assert value == 2.0

    def test_default_rate_limit_cache_ttl_emits_deprecation_warning(self) -> None:
        """Test that accessing _DEFAULT_RATE_LIMIT_CACHE_TTL emits DeprecationWarning."""
        import warnings

        from sentinel.dashboard import routes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = routes._DEFAULT_RATE_LIMIT_CACHE_TTL

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_DEFAULT_RATE_LIMIT_CACHE_TTL" in str(w[0].message)
            assert "Config.rate_limit_cache_ttl" in str(w[0].message)
            assert value == 3600

    def test_default_rate_limit_cache_maxsize_emits_deprecation_warning(self) -> None:
        """Test that accessing _DEFAULT_RATE_LIMIT_CACHE_MAXSIZE emits DeprecationWarning."""
        import warnings

        from sentinel.dashboard import routes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = routes._DEFAULT_RATE_LIMIT_CACHE_MAXSIZE

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_DEFAULT_RATE_LIMIT_CACHE_MAXSIZE" in str(w[0].message)
            assert "Config.rate_limit_cache_maxsize" in str(w[0].message)
            assert value == 10000

    def test_deprecation_warning_mentions_future_removal(self) -> None:
        """Test that deprecation warning mentions future removal."""
        import warnings

        from sentinel.dashboard import routes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = routes._DEFAULT_TOGGLE_COOLDOWN

            assert len(w) == 1
            assert "will be removed in a future release" in str(w[0].message)

    def test_accessing_non_deprecated_attribute_raises_attribute_error(self) -> None:
        """Test that accessing non-existent attribute raises AttributeError."""
        from sentinel.dashboard import routes

        with pytest.raises(AttributeError) as exc_info:
            _ = routes._NONEXISTENT_CONSTANT

        assert "has no attribute '_NONEXISTENT_CONSTANT'" in str(exc_info.value)


class TestCreateRoutesLogging:
    """Tests for debug logging in create_routes function.

    These tests verify that create_routes logs appropriate debug messages
    when using provided Config vs. default Config values. Tests use mock
    logger to check call arguments directly for more robust assertions.
    """

    def test_create_routes_logs_debug_with_provided_config(self, temp_logs_dir: Path) -> None:
        """Test that create_routes logs debug message when Config is provided."""
        config = create_config(
            agent_logs_dir=temp_logs_dir,
            toggle_cooldown_seconds=5.0,
            rate_limit_cache_ttl=7200,
            rate_limit_cache_maxsize=5000,
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
        config = create_config(agent_logs_dir=temp_logs_dir)
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
        config = create_config(agent_logs_dir=temp_logs_dir, toggle_cooldown_seconds=3.5)
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
        config = create_config(agent_logs_dir=temp_logs_dir, rate_limit_cache_ttl=1800)
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
        config = create_config(agent_logs_dir=temp_logs_dir, rate_limit_cache_maxsize=20000)
        sentinel = MockSentinel(config)
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        with patch("sentinel.dashboard.routes.logger") as mock_logger:
            _ = create_routes(accessor, config=config)

            # Verify the cache_maxsize value is passed correctly
            call_args = mock_logger.debug.call_args
            assert_call_args_length(call_args, 5)
            assert call_args[0][4] == 20000
