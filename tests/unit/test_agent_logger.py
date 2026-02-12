"""Tests for the agent execution logger."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sentinel.agent_logger import AgentLogger, StreamingLogWriter
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
            attempt=1,
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
            attempt=1,
            start_time=start_time,
            end_time=datetime(2024, 1, 15, 10, 31, 0),
        )

        assert log_path.name == "DS-123_20240115-103045_a1.log"
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
            attempt=3,
            start_time=start_time,
            end_time=end_time,
        )

        content = log_path.read_text()
        assert "Issue Key:      DS-456" in content
        assert "Orchestration:  update-docs" in content
        assert "Status:         FAILURE" in content
        assert "Attempt:        3" in content
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
            attempt=1,
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
            attempt=1,
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
            attempt=1,
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
            attempt=1,
            start_time=datetime(2024, 1, 15, 10, 0, 0),
            end_time=datetime(2024, 1, 15, 10, 1, 0),
        )

        log2 = logger.log_execution(
            issue_key="DS-2",
            orchestration_name="review",
            prompt="P2",
            response="R2",
            status=ExecutionStatus.FAILURE,
            attempt=2,
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
            attempt=1,
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
            attempt=3,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        content = log_path.read_text()
        assert "Status:         ERROR" in content


class TestStreamingLogWriter:
    """Tests for StreamingLogWriter."""

    def test_creates_log_file_on_enter(self, tmp_path: Path) -> None:
        """Should create log file immediately when entering context."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test-orch",
            issue_key="DS-123",
            prompt="Test prompt",
        ) as writer:
            assert writer.log_path is not None
            assert writer.log_path.exists()

    def test_creates_orchestration_directory(self, tmp_path: Path) -> None:
        """Should create orchestration-specific directory."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="my-orchestration",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            assert (tmp_path / "my-orchestration").exists()
            assert writer.log_path is not None
            assert writer.log_path.parent.name == "my-orchestration"

    def test_log_path_has_timestamp_filename(self, tmp_path: Path) -> None:
        """Should create log file with timestamp-based filename."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            assert writer.log_path is not None
            assert writer.log_path.suffix == ".log"
            # Filename should match {issue_key}_{YYYYMMDD-HHMMSS}_a{N}.log pattern (DS-960)
            name = writer.log_path.stem
            assert "DS-123" in name
            assert "_a1" in name

    def test_header_contains_metadata(self, tmp_path: Path) -> None:
        """Should write header with metadata when entering context."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="code-review",
            issue_key="DS-456",
            prompt="Review this code",
        ) as writer:
            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "Issue Key:      DS-456" in content
            assert "Orchestration:  code-review" in content
            assert "PROMPT" in content
            assert "Review this code" in content

    def test_write_line_appends_to_file(self, tmp_path: Path) -> None:
        """Should append lines to log file immediately with timestamps."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("First output line")
            writer.write_line("Second output line")

            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "First output line" in content
            assert "Second output line" in content
            # Verify timestamps are present in [HH:MM:SS.mmm] format
            timestamp_pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]"
            assert re.search(timestamp_pattern, content) is not None

    def test_write_line_adds_newline_if_missing(self, tmp_path: Path) -> None:
        """Should add newline if not present in the line."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Line without newline")

            assert writer.log_path is not None
            content = writer.log_path.read_text()
            # Should end with newline
            lines = content.split("\n")
            assert any("Line without newline" in line for line in lines)

    def test_write_appends_with_timestamps(self, tmp_path: Path) -> None:
        """Should append text with timestamp prefix."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write("partial")
            writer.write(" text")

            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "partial" in content
            assert " text" in content
            # Verify timestamps are present in [HH:MM:SS.mmm] format
            timestamp_pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]"
            assert re.search(timestamp_pattern, content) is not None

    def test_get_response_returns_accumulated_output(self, tmp_path: Path) -> None:
        """Should return all accumulated output from write/write_line calls."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Line 1")
            writer.write("partial")
            writer.write_line("Line 2")

            response = writer.get_response()
            assert "Line 1\n" in response
            assert "partial" in response
            assert "Line 2\n" in response

    def test_finalize_writes_summary(self, tmp_path: Path) -> None:
        """Should write execution summary when finalized."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Some output")
            writer.finalize(status="COMPLETED", attempt=1)

            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "EXECUTION SUMMARY" in content
            assert "Status:         COMPLETED" in content
            assert "Attempt:        1" in content
            assert "Duration:" in content
            assert "END OF LOG" in content

    def test_finalize_with_multiple_attempts(self, tmp_path: Path) -> None:
        """Should record attempt count in summary."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.finalize(status="FAILED", attempt=3)

            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "Status:         FAILED" in content
            assert "Attempt:        3" in content

    def test_duration_calculation(self, tmp_path: Path) -> None:
        """Should calculate duration between start and finalize."""
        # Mock datetime.now to control the duration instead of sleeping (deterministic approach - DS-933)
        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        end_time = start_time + timedelta(seconds=0.1)

        with patch("sentinel.agent_logger.datetime") as mock_datetime:
            # Set up the mock to return start_time on enter, end_time on finalize
            mock_datetime.now.side_effect = [start_time, end_time]
            # Keep other datetime methods working
            mock_datetime.fromisoformat = datetime.fromisoformat
            mock_datetime.fromtimestamp = datetime.fromtimestamp

            with StreamingLogWriter(
                base_dir=tmp_path,
                orchestration_name="test",
                issue_key="DS-123",
                prompt="Test",
            ) as writer:
                writer.finalize(status="COMPLETED", attempt=1)

                assert writer.log_path is not None
                content = writer.log_path.read_text()
                # Duration should be 0.10 seconds (as mocked)
                assert "Duration:       0.10s" in content

    def test_context_manager_closes_file(self, tmp_path: Path) -> None:
        """Should close file when exiting context."""
        writer = StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        )

        with writer:
            writer.write_line("Output")
            assert writer._file is not None

        # File should be closed after exit
        assert writer._file is None

    def test_write_after_exit_is_noop(self, tmp_path: Path) -> None:
        """Should ignore writes after context is exited."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Before exit")
            log_path = writer.log_path

        # This should not raise or write
        writer.write_line("After exit")
        writer.write("More after exit")

        assert log_path is not None
        content = log_path.read_text()
        assert "Before exit" in content
        assert "After exit" not in content

    def test_streaming_header_label(self, tmp_path: Path) -> None:
        """Should indicate streaming in header."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            assert writer.log_path is not None
            content = writer.log_path.read_text()
            assert "STREAMING" in content or "streaming" in content

    def test_log_path_none_before_enter(self, tmp_path: Path) -> None:
        """log_path should be None before entering context."""
        writer = StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        )
        assert writer.log_path is None

    def test_timestamp_format_for_tail_monitoring(self, tmp_path: Path) -> None:
        """Should add [HH:MM:SS.mmm] timestamps for tail -f monitoring."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Test message for monitoring")

            assert writer.log_path is not None
            content = writer.log_path.read_text()

            # Verify timestamp format matches [HH:MM:SS.mmm]
            timestamp_pattern = r"\[\d{2}:\d{2}:\d{2}\.\d{3}\] Test message for monitoring"
            assert re.search(timestamp_pattern, content) is not None

    def test_get_response_excludes_timestamps(self, tmp_path: Path) -> None:
        """get_response should return original content without timestamp prefixes."""
        with StreamingLogWriter(
            base_dir=tmp_path,
            orchestration_name="test",
            issue_key="DS-123",
            prompt="Test",
        ) as writer:
            writer.write_line("Line 1")
            writer.write("partial")

            response = writer.get_response()
            # Response should contain original text without timestamps
            assert response == "Line 1\npartial"
