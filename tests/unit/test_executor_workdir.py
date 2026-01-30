"""Workdir cleanup tests for Claude Agent SDK executor module.

This module contains tests for workdir cleanup functionality:
- cleanup_workdir function tests
- AgentExecutor workdir cleanup behavior
- AgentRunResult dataclass tests
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sentinel.executor import AgentExecutor, AgentRunResult, ExecutionStatus, cleanup_workdir
from tests.helpers import make_issue, make_orchestration
from tests.mocks import MockAgentClient


class TestCleanupWorkdir:
    """Tests for the cleanup_workdir function."""

    def test_cleanup_workdir_removes_directory(self) -> None:
        """cleanup_workdir should remove the directory and return True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            assert workdir.exists()
            result = cleanup_workdir(workdir)

            assert result is True
            assert not workdir.exists()

    def test_cleanup_workdir_none_returns_true(self) -> None:
        """cleanup_workdir should return True when workdir is None."""
        result = cleanup_workdir(None)
        assert result is True

    def test_cleanup_workdir_nonexistent_returns_true(self) -> None:
        """cleanup_workdir should return True for non-existent directory."""
        workdir = Path("/nonexistent/path/that/does/not/exist")
        result = cleanup_workdir(workdir)
        assert result is True

    def test_cleanup_workdir_permission_error_returns_false(self) -> None:
        """cleanup_workdir should return False on PermissionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = PermissionError("Permission denied")
                result = cleanup_workdir(workdir)

            assert result is False

    def test_cleanup_workdir_os_error_returns_false(self) -> None:
        """cleanup_workdir should return False on OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = OSError("OS error")
                result = cleanup_workdir(workdir)

            assert result is False

    def test_cleanup_workdir_unexpected_error_returns_false(self) -> None:
        """cleanup_workdir should return False on unexpected exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = RuntimeError("Unexpected error")
                result = cleanup_workdir(workdir)

            assert result is False

    # Tests for force parameter

    def test_cleanup_workdir_force_removes_directory(self) -> None:
        """cleanup_workdir with force=True should remove the directory on first attempt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            assert workdir.exists()
            result = cleanup_workdir(workdir, force=True)

            assert result is True
            assert not workdir.exists()

    def test_cleanup_workdir_force_none_returns_true(self) -> None:
        """cleanup_workdir with force=True should return True when workdir is None."""
        result = cleanup_workdir(None, force=True)
        assert result is True

    def test_cleanup_workdir_force_nonexistent_returns_true(self) -> None:
        """cleanup_workdir with force=True should return True for non-existent directory."""
        workdir = Path("/nonexistent/path/that/does/not/exist")
        result = cleanup_workdir(workdir, force=True)
        assert result is True

    def test_cleanup_workdir_force_retries_on_permission_error(self) -> None:
        """cleanup_workdir with force=True should retry on PermissionError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise PermissionError("Permission denied")
                # Successful removal on second attempt - actually remove it
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert call_count == 2

    def test_cleanup_workdir_force_retries_on_os_error(self) -> None:
        """cleanup_workdir with force=True should retry on OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise OSError("Resource busy")
                # Successful removal on second attempt
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert call_count == 2

    def test_cleanup_workdir_force_uses_ignore_errors_on_final_attempt(self) -> None:
        """cleanup_workdir with force=True should use ignore_errors on final attempt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            rmtree_calls: list[tuple[Any, dict[str, Any]]] = []
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                rmtree_calls.append((path, kwargs))
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # When ignore_errors=True, actually remove the directory
                original_rmtree(path, ignore_errors=True)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is True
            assert len(rmtree_calls) == 3
            # First two calls should not have ignore_errors
            assert rmtree_calls[0][1].get("ignore_errors", False) is False
            assert rmtree_calls[1][1].get("ignore_errors", False) is False
            # Final call should have ignore_errors=True
            assert rmtree_calls[2][1].get("ignore_errors") is True

    def test_cleanup_workdir_force_returns_false_if_directory_remains(self) -> None:
        """cleanup_workdir with force=True should return False if directory still exists after all attempts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                # Always fail (even with ignore_errors, directory persists)
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # With ignore_errors, just do nothing (directory remains)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is False
            assert workdir.exists()

    def test_cleanup_workdir_force_no_retry_on_unexpected_error(self) -> None:
        """cleanup_workdir with force=True should not retry on unexpected exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Unexpected error")

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                result = cleanup_workdir(workdir, force=True, max_retries=3)

            assert result is False
            assert call_count == 1  # Should not retry

    def test_cleanup_workdir_force_custom_max_retries(self) -> None:
        """cleanup_workdir with force=True should respect custom max_retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if not kwargs.get("ignore_errors", False):
                    raise PermissionError("Permission denied")
                # With ignore_errors, just do nothing (directory remains)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep"):  # Speed up test
                    result = cleanup_workdir(workdir, force=True, max_retries=5)

            assert result is False
            assert call_count == 5  # Should retry 5 times

    def test_cleanup_workdir_force_respects_retry_delay(self) -> None:
        """cleanup_workdir with force=True should sleep between retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()

            call_count = 0
            original_rmtree = shutil.rmtree

            def mock_rmtree_side_effect(path: Any, **kwargs: Any) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise PermissionError("Permission denied")
                # Successful on third attempt
                original_rmtree(path, **kwargs)

            with patch("shutil.rmtree", side_effect=mock_rmtree_side_effect):
                with patch("time.sleep") as mock_sleep:
                    result = cleanup_workdir(workdir, force=True, max_retries=3, retry_delay=1.5)

            assert result is True
            # Should have slept twice (after first and second failures)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_called_with(1.5)


class TestAgentExecutorWorkdirCleanup:
    """Tests for AgentExecutor workdir cleanup behavior.

    Design rationale for workdir preservation on failure:
        When an agent execution fails (whether through explicit FAILURE response,
        client ERROR, or timeout), the workdir is intentionally preserved rather
        than cleaned up. This design decision supports debugging by allowing
        developers to inspect:
        - Cloned repositories and their state at failure time
        - Partial work completed before the failure
        - Log files or artifacts generated during execution
        - Any modifications made to files that might explain the failure

        The tests in this class verify both the success-path cleanup AND the
        failure-path preservation behavior. Tests that verify preservation on
        failure are critical - they ensure we don't accidentally remove valuable
        debugging context when something goes wrong.
    """

    def test_cleanup_on_success_when_enabled(self) -> None:
        """Should cleanup workdir on successful execution when cleanup is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Task completed"],
                workdir=workdir,
            )
            # cleanup_workdir_on_success=True is the default
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration()

            assert workdir.exists()
            result = executor.execute(issue, orch)

            assert result.succeeded is True
            assert not workdir.exists()

    def test_no_cleanup_when_disabled(self, caplog: Any) -> None:
        """Should preserve workdir when cleanup_workdir_on_success is False.

        Also verifies that the appropriate debug log message is emitted
        indicating the workdir was preserved.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Task completed"],
                workdir=workdir,
            )
            executor = AgentExecutor(client, cleanup_workdir_on_success=False)
            issue = make_issue()
            orch = make_orchestration()

            with caplog.at_level(logging.DEBUG, logger="sentinel.executor"):
                result = executor.execute(issue, orch)

            assert result.succeeded is True
            assert workdir.exists()  # Workdir should be preserved

            # Verify the debug log message was emitted
            assert any(
                "Workdir preserved at" in record.message
                and "(cleanup_workdir_on_success=False)" in record.message
                and record.levelno == logging.DEBUG
                for record in caplog.records
            ), "Expected debug log message about workdir preservation was not emitted"

    def test_no_cleanup_on_failure(self) -> None:
        """Should preserve workdir on failed execution for debugging.

        When an agent reports FAILURE, we intentionally keep the workdir intact.
        This allows debugging what went wrong by examining the file state,
        partial work, and any artifacts the agent created before failing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["FAILURE: Task failed"],
                workdir=workdir,
            )
            # Even with cleanup enabled, failures should preserve workdir
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.FAILURE
            assert workdir.exists()  # Workdir preserved for debugging

    def test_cleanup_handles_none_workdir(self) -> None:
        """Should handle None workdir gracefully when no workdir is provided.

        This test covers the scenario where an agent execution completes
        successfully but no workdir was created or returned by the agent client.
        This can happen when the agent doesn't need to clone a repository or
        create temporary files during execution.

        Graceful handling of None workdir is important because:
        - Not all agent tasks require file system operations
        - The cleanup logic should not crash or raise exceptions when workdir is None
        - The execution should complete successfully regardless of workdir presence

        Expected behavior: The executor should skip cleanup operations entirely
        and return a successful result without any errors.
        """
        client = MockAgentClient(
            responses=["SUCCESS: Task completed"],
            workdir=None,  # No workdir
        )
        executor = AgentExecutor(client, cleanup_workdir_on_success=True)
        issue = make_issue()
        orch = make_orchestration()

        # Should not raise any errors
        result = executor.execute(issue, orch)

        assert result.succeeded is True

    def test_cleanup_default_is_enabled(self) -> None:
        """Should have cleanup enabled by default.

        Verifies that cleanup_workdir_on_success defaults to True when not
        explicitly specified. This ensures workdirs are cleaned up after
        successful executions unless explicitly disabled.
        """
        client = MockAgentClient()
        executor = AgentExecutor(client)  # No explicit cleanup_workdir_on_success

        assert executor.cleanup_workdir_on_success is True

    def test_no_cleanup_on_error(self) -> None:
        """Should preserve workdir on agent client error.

        When the agent client raises an error (e.g., API failure, network issue),
        we preserve the workdir to aid debugging. The workdir may contain partial
        state or cloned repos that help diagnose why the agent errored.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Done"],
                workdir=workdir,
            )
            client.should_error = True
            client.max_errors = 10  # Always error
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.ERROR
            assert workdir.exists()  # Workdir preserved

    def test_no_cleanup_on_timeout(self) -> None:
        """Should preserve workdir on agent timeout.

        When an agent times out, the workdir is preserved to support debugging.
        Timeout scenarios often leave partial work in progress, and examining
        the workdir state can reveal what the agent was working on when time
        ran out.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "test_workdir"
            workdir.mkdir()
            (workdir / "test_file.txt").write_text("test content")

            client = MockAgentClient(
                responses=["SUCCESS: Done"],
                workdir=workdir,
            )
            client.should_timeout = True
            client.max_timeouts = 10  # Always timeout
            executor = AgentExecutor(client, cleanup_workdir_on_success=True)
            issue = make_issue()
            orch = make_orchestration(max_attempts=1)

            result = executor.execute(issue, orch)

            assert result.succeeded is False
            assert result.status == ExecutionStatus.ERROR
            assert workdir.exists()  # Workdir preserved


class TestAgentRunResult:
    """Tests for AgentRunResult dataclass."""

    def test_agent_run_result_with_response_only(self) -> None:
        """AgentRunResult can be created with just response."""
        result = AgentRunResult(response="Task completed")
        assert result.response == "Task completed"
        assert result.workdir is None

    def test_agent_run_result_with_workdir(self) -> None:
        """AgentRunResult can include a workdir path."""
        workdir = Path("/tmp/test-workdir")
        result = AgentRunResult(response="Task completed", workdir=workdir)
        assert result.response == "Task completed"
        assert result.workdir == workdir

    def test_agent_run_result_workdir_default_none(self) -> None:
        """AgentRunResult workdir should default to None."""
        result = AgentRunResult(response="Done")
        assert result.workdir is None
