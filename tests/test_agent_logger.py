"""Tests for the agent execution logger."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sentinel.agent_logger import AgentLogger
from sentinel.executor import ExecutionStatus


class TestAgentLogger:
    """Tests for AgentLogger."""

    def test_creates_orchestration_directory(self, tmp_path: Path) -> None:
        """Should create orchestration-specific directory."""
        logger = AgentLogger(base_dir=tmp_path)

        logger.log_execution(
            issue_key="DS-123",
            orchestration_name="code-review",
            prompt="Test prompt",
            response="Test response",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime(2024, 1, 15, 10, 30, 0),
            end_time=datetime(2024, 1, 15, 10, 31, 0),
        )

        orch_dir = tmp_path / "code-review"
        assert orch_dir.exists()
        assert orch_dir.is_dir()

    def test_creates_log_file_with_timestamp(self, tmp_path: Path) -> None:
        """Should create log file named with timestamp."""
        logger = AgentLogger(base_dir=tmp_path)
        start_time = datetime(2024, 1, 15, 10, 30, 45)

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="code-review",
            prompt="Test prompt",
            response="Test response",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=start_time,
            end_time=datetime(2024, 1, 15, 10, 31, 0),
        )

        assert log_path.name == "20240115_103045.log"
        assert log_path.exists()

    def test_log_contains_metadata(self, tmp_path: Path) -> None:
        """Should include execution metadata in log."""
        logger = AgentLogger(base_dir=tmp_path)
        start_time = datetime(2024, 1, 15, 10, 30, 0)
        end_time = datetime(2024, 1, 15, 10, 31, 30)

        log_path = logger.log_execution(
            issue_key="DS-456",
            orchestration_name="update-docs",
            prompt="Test prompt",
            response="Test response",
            status=ExecutionStatus.FAILURE,
            attempts=3,
            start_time=start_time,
            end_time=end_time,
        )

        content = log_path.read_text()
        assert "Issue Key:      DS-456" in content
        assert "Orchestration:  update-docs" in content
        assert "Status:         FAILURE" in content
        assert "Attempts:       3" in content
        assert "Duration:       90.00s" in content

    def test_log_contains_prompt(self, tmp_path: Path) -> None:
        """Should include the prompt in the log."""
        logger = AgentLogger(base_dir=tmp_path)
        prompt = "This is a test prompt\nwith multiple lines"

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="test",
            prompt=prompt,
            response="Response",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        content = log_path.read_text()
        assert "PROMPT" in content
        assert prompt in content

    def test_log_contains_response(self, tmp_path: Path) -> None:
        """Should include the response in the log."""
        logger = AgentLogger(base_dir=tmp_path)
        response = "This is the agent response\nSUCCESS: Task completed"

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="test",
            prompt="Prompt",
            response=response,
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        content = log_path.read_text()
        assert "RESPONSE" in content
        assert response in content

    def test_handles_special_characters_in_orchestration_name(self, tmp_path: Path) -> None:
        """Should handle orchestration names that are valid directory names."""
        logger = AgentLogger(base_dir=tmp_path)

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="my-orch-v2",
            prompt="Prompt",
            response="Response",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        assert log_path.exists()
        assert "my-orch-v2" in str(log_path)

    def test_multiple_logs_same_orchestration(self, tmp_path: Path) -> None:
        """Should create multiple log files in the same orchestration directory."""
        logger = AgentLogger(base_dir=tmp_path)

        log1 = logger.log_execution(
            issue_key="DS-1",
            orchestration_name="review",
            prompt="P1",
            response="R1",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime(2024, 1, 15, 10, 0, 0),
            end_time=datetime(2024, 1, 15, 10, 1, 0),
        )

        log2 = logger.log_execution(
            issue_key="DS-2",
            orchestration_name="review",
            prompt="P2",
            response="R2",
            status=ExecutionStatus.FAILURE,
            attempts=2,
            start_time=datetime(2024, 1, 15, 11, 0, 0),
            end_time=datetime(2024, 1, 15, 11, 2, 0),
        )

        assert log1.parent == log2.parent
        assert log1.name != log2.name
        assert len(list((tmp_path / "review").glob("*.log"))) == 2

    def test_returns_log_path(self, tmp_path: Path) -> None:
        """Should return the path to the created log file."""
        logger = AgentLogger(base_dir=tmp_path)

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="test",
            prompt="Prompt",
            response="Response",
            status=ExecutionStatus.SUCCESS,
            attempts=1,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        assert isinstance(log_path, Path)
        assert log_path.exists()
        assert log_path.suffix == ".log"

    def test_logs_error_status(self, tmp_path: Path) -> None:
        """Should correctly log ERROR status."""
        logger = AgentLogger(base_dir=tmp_path)

        log_path = logger.log_execution(
            issue_key="DS-123",
            orchestration_name="test",
            prompt="Prompt",
            response="Timeout: Agent timed out",
            status=ExecutionStatus.ERROR,
            attempts=3,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        content = log_path.read_text()
        assert "Status:         ERROR" in content
