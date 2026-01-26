"""Tests for dashboard routes.

DS-125: Tests for the log file discovery API endpoint.
DS-126: Tests for the SSE log streaming endpoint.
DS-250: Tests for orchestration toggle API endpoints.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sentinel.config import Config
from sentinel.dashboard.routes import create_routes
from sentinel.dashboard.state import OrchestrationInfo, SentinelStateAccessor


class MockSentinel:
    """Mock Sentinel instance for testing dashboard state accessor."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._shutdown_requested = False
        self._versions_lock = MagicMock()
        self._futures_lock = MagicMock()
        self._active_versions: list[Any] = []
        self._pending_removal_versions: list[Any] = []
        self._active_futures: list[Any] = []
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


class TestApiLogsFilesEndpoint:
    """Tests for GET /api/logs/files endpoint (DS-125)."""

    def test_returns_empty_list_when_no_logs_dir(self) -> None:
        """Test that endpoint returns empty list when logs directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "nonexistent_logs"
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            assert result == []

    def test_returns_empty_list_when_logs_dir_empty(self) -> None:
        """Test that endpoint returns empty list when logs directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
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

            config = Config(agent_logs_dir=logs_dir)
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
        """Test that log files are sorted with most recent first (DS-125 requirement)."""
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

            config = Config(agent_logs_dir=logs_dir)
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

            config = Config(agent_logs_dir=logs_dir)
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

            # Create the exact structure specified in DS-125
            # {base_dir}/{orchestration_name}/{timestamp}.log
            orch_name = "my-workflow"
            timestamp = "20260114_143022"

            orch_dir = logs_dir / orch_name
            orch_dir.mkdir()
            log_file = orch_dir / f"{timestamp}.log"
            log_file.write_text("Execution log content")

            config = Config(agent_logs_dir=logs_dir)
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

            config = Config(agent_logs_dir=logs_dir)
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

            config = Config(agent_logs_dir=logs_dir)
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

            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_files()

            # Should only have the directory with logs
            assert len(result) == 1
            assert result[0]["orchestration"] == "has-logs"


class TestSseLogStreamingEndpoint:
    """Tests for GET /api/logs/stream/{orchestration}/{filename} endpoint (DS-126).

    Tests the Server-Sent Events log streaming endpoint that tails log files
    and streams new lines in real-time.

    Note: Some tests use httpx directly with stream() to handle SSE streaming
    properly, as the TestClient may not handle long-lived connections well.
    Tests for error cases (which exit immediately) use TestClient normally.
    """

    def _create_test_app(self, accessor: SentinelStateAccessor) -> FastAPI:
        """Create a test FastAPI app with dashboard routes."""
        app = FastAPI()
        router = create_routes(accessor)
        app.include_router(router)
        return app

    def test_stream_returns_error_when_log_file_not_found(self) -> None:
        """Test that streaming returns error event when log file doesn't exist (DS-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
            app = self._create_test_app(accessor)

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

    def test_stream_error_event_contains_json_data(self) -> None:
        """Test that error event data is valid JSON with error field (DS-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
            app = self._create_test_app(accessor)

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

    def test_stream_handles_path_traversal_attack(self) -> None:
        """Test that path traversal attempts are blocked (DS-126 security)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            logs_dir.mkdir(exist_ok=True)

            # Create a file outside the logs directory
            secret_file = Path(tmpdir) / "secret.txt"
            secret_file.write_text("secret content")

            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
            app = self._create_test_app(accessor)

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
        """Test that get_log_file_path returns correct path for valid file (DS-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_dir = logs_dir / "my-orch"
            orch_dir.mkdir()
            log_file = orch_dir / "20260114_100000.log"
            log_file.write_text("Log content")

            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_file_path("my-orch", "20260114_100000.log")

            assert result is not None
            assert result == log_file
            assert result.exists()

    def test_get_log_file_path_returns_none_for_missing_file(self) -> None:
        """Test that get_log_file_path returns None for non-existent file (DS-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

            result = accessor.get_log_file_path("missing-orch", "missing.log")

            assert result is None

    def test_get_log_file_path_blocks_path_traversal(self) -> None:
        """Test that get_log_file_path blocks path traversal attempts (DS-126 security)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Test path traversal blocking - we can't create a file in parent,
            # but we can test the path check

            config = Config(agent_logs_dir=logs_dir)
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
        """Test that the streaming endpoint imports and uses sse-starlette (DS-126)."""
        # Verify that sse_starlette.sse.EventSourceResponse is used
        from sentinel.dashboard.routes import EventSourceResponse
        from sse_starlette.sse import EventSourceResponse as SSEResponse

        assert EventSourceResponse is SSEResponse

    def test_stream_endpoint_exists_with_correct_path(self) -> None:
        """Test that the stream endpoint exists at /api/logs/stream/{orchestration}/{filename} (DS-126)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinel(config)
            accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]
            app = self._create_test_app(accessor)

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
    """Tests for POST /api/orchestrations/{name}/toggle endpoint (DS-250)."""

    def _create_test_app(self, accessor: SentinelStateAccessor) -> FastAPI:
        """Create a test FastAPI app with dashboard routes."""
        app = FastAPI()
        router = create_routes(accessor)
        app.include_router(router)
        return app

    def test_toggle_orchestration_success(self) -> None:
        """Test successful toggle of a single orchestration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Create an orchestration YAML file
            orch_file = logs_dir / "test-orch.yaml"
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

            config = Config(agent_logs_dir=logs_dir)
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
            app = self._create_test_app(accessor)

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

    def test_toggle_orchestration_enable(self) -> None:
        """Test enabling a disabled orchestration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_file = logs_dir / "test-orch.yaml"
            orch_file.write_text("""
orchestrations:
  - name: "test-orch"
    enabled: false
    trigger:
      source: jira
    agent:
      prompt: "Test"
""")

            config = Config(agent_logs_dir=logs_dir)
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
            app = self._create_test_app(accessor)

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

    def test_toggle_orchestration_not_found(self) -> None:
        """Test 404 when orchestration doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinelWithOrchestrations(config, [])

            # No orchestrations configured
            accessor = MockStateAccessorWithOrchestrations(sentinel, [])
            app = self._create_test_app(accessor)

            with TestClient(app) as client:
                response = client.post(
                    "/api/orchestrations/nonexistent/toggle",
                    json={"enabled": False},
                )

                assert response.status_code == 404
                assert "not found" in response.json()["detail"].lower()

    def test_toggle_orchestration_file_not_found(self) -> None:
        """Test 404 when orchestration file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
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
            app = self._create_test_app(accessor)

            with TestClient(app) as client:
                response = client.post(
                    "/api/orchestrations/test-orch/toggle",
                    json={"enabled": False},
                )

                assert response.status_code == 404
                assert "not found" in response.json()["detail"].lower()

    def test_toggle_endpoint_exists(self) -> None:
        """Test that the toggle endpoint exists at the correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinelWithOrchestrations(config, [])
            accessor = MockStateAccessorWithOrchestrations(sentinel, [])
            app = self._create_test_app(accessor)

            route_paths = [route.path for route in app.routes]
            assert "/api/orchestrations/{name}/toggle" in route_paths


class TestBulkToggleOrchestrationEndpoint:
    """Tests for POST /api/orchestrations/bulk-toggle endpoint (DS-250)."""

    def _create_test_app(self, accessor: SentinelStateAccessor) -> FastAPI:
        """Create a test FastAPI app with dashboard routes."""
        app = FastAPI()
        router = create_routes(accessor)
        app.include_router(router)
        return app

    def test_bulk_toggle_jira_orchestrations_success(self) -> None:
        """Test bulk toggling Jira orchestrations by project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            # Create an orchestration YAML file with multiple orchestrations
            orch_file = logs_dir / "jira-orchestrations.yaml"
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

            config = Config(agent_logs_dir=logs_dir)
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
            app = self._create_test_app(accessor)

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

    def test_bulk_toggle_github_orchestrations_success(self) -> None:
        """Test bulk toggling GitHub orchestrations by repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)

            orch_file = logs_dir / "github-orchestrations.yaml"
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

            config = Config(agent_logs_dir=logs_dir)
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
            app = self._create_test_app(accessor)

            with TestClient(app) as client:
                response = client.post(
                    "/api/orchestrations/bulk-toggle",
                    json={"source": "github", "identifier": "org/repo", "enabled": False},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["toggled_count"] == 2

    def test_bulk_toggle_no_matching_orchestrations(self) -> None:
        """Test 404 when no orchestrations match the identifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinelWithOrchestrations(config, [])

            # No matching orchestrations
            accessor = MockStateAccessorWithOrchestrations(sentinel, [])
            app = self._create_test_app(accessor)

            with TestClient(app) as client:
                response = client.post(
                    "/api/orchestrations/bulk-toggle",
                    json={"source": "jira", "identifier": "NONEXISTENT", "enabled": False},
                )

                assert response.status_code == 404
                assert "No orchestrations found" in response.json()["detail"]

    def test_bulk_toggle_endpoint_exists(self) -> None:
        """Test that the bulk toggle endpoint exists at the correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinelWithOrchestrations(config, [])
            accessor = MockStateAccessorWithOrchestrations(sentinel, [])
            app = self._create_test_app(accessor)

            route_paths = [route.path for route in app.routes]
            assert "/api/orchestrations/bulk-toggle" in route_paths

    def test_bulk_toggle_invalid_source(self) -> None:
        """Test validation error for invalid source type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            config = Config(agent_logs_dir=logs_dir)
            sentinel = MockSentinelWithOrchestrations(config, [])
            accessor = MockStateAccessorWithOrchestrations(sentinel, [])
            app = self._create_test_app(accessor)

            with TestClient(app) as client:
                response = client.post(
                    "/api/orchestrations/bulk-toggle",
                    json={"source": "invalid", "identifier": "TEST", "enabled": False},
                )

                # Should return 422 for validation error
                assert response.status_code == 422
