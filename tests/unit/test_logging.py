"""Tests for logging module."""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from sentinel.logging import (
    ContextAdapter,
    DiagnosticFilter,
    JSONFormatter,
    OrchestrationLogManager,
    SentinelLogger,
    StructuredFormatter,
    get_logger,
    log_agent_summary,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="sentinel.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test" in result  # component extracted from logger name
        assert "Test message" in result

    def test_format_with_context(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="sentinel.executor",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing issue",
            args=(),
            exc_info=None,
        )
        record.issue_key = "TEST-123"
        record.orchestration = "review"
        record.attempt = 1

        result = formatter.format(record)

        assert "issue_key=TEST-123" in result
        assert "orchestration=review" in result
        assert "attempt=1" in result
        assert "Processing issue" in result

    def test_format_extracts_component_from_name(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="sentinel.poller",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "[poller" in result

    def test_format_handles_simple_name(self) -> None:
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="mylogger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "[mylogger" in result


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinel.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["component"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_format_with_context(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinel.executor",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Executing",
            args=(),
            exc_info=None,
        )
        record.issue_key = "PROJ-456"
        record.orchestration = "deploy"
        record.status = "SUCCESS"
        record.response_summary = "Task completed"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["issue_key"] == "PROJ-456"
        assert data["orchestration"] == "deploy"
        assert data["status"] == "SUCCESS"
        assert data["response_summary"] == "Task completed"

    def test_format_valid_json(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="sentinel",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug info",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestContextAdapter:
    """Tests for ContextAdapter."""

    def test_adds_context_to_logs(self) -> None:
        base_logger = logging.getLogger("test.context")
        adapter = ContextAdapter(base_logger, {"issue_key": "TEST-1"})

        msg, kwargs = adapter.process("Test message", {})

        assert kwargs["extra"]["issue_key"] == "TEST-1"

    def test_merges_with_existing_extra(self) -> None:
        base_logger = logging.getLogger("test.context2")
        adapter = ContextAdapter(base_logger, {"orchestration": "review"})

        msg, kwargs = adapter.process("Test", {"extra": {"attempt": 2}})

        assert kwargs["extra"]["orchestration"] == "review"
        assert kwargs["extra"]["attempt"] == 2


class TestSentinelLogger:
    """Tests for SentinelLogger."""

    def test_with_context_returns_adapter(self) -> None:
        logger = get_logger("test.sentinel")

        ctx_logger = logger.with_context(issue_key="ABC-123", orchestration="test")

        assert isinstance(ctx_logger, ContextAdapter)

    def test_get_logger_returns_sentinel_logger(self) -> None:
        logger = get_logger("test.module")

        assert isinstance(logger, SentinelLogger)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_sets_log_level(self) -> None:
        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_json_format_adds_json_formatter(self) -> None:
        setup_logging(level="INFO", json_format=True)

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_structured_format_by_default(self) -> None:
        setup_logging(level="INFO", json_format=False)

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, StructuredFormatter)

    def test_handles_invalid_level_gracefully(self) -> None:
        # Should not raise, should default to INFO
        setup_logging(level="INVALID")

        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_replace_handlers_true_removes_existing(self) -> None:
        """Default behavior removes existing handlers."""
        root = logging.getLogger()

        # Add a custom handler
        existing_handler = logging.StreamHandler(StringIO())
        root.addHandler(existing_handler)

        # Setup with default replace_handlers=True
        setup_logging(level="INFO")

        # Existing handler should be removed
        assert existing_handler not in root.handlers
        # Should have exactly one handler (our new one)
        assert len(root.handlers) == 1

    def test_replace_handlers_false_preserves_existing(self) -> None:
        """replace_handlers=False preserves existing handlers."""
        root = logging.getLogger()

        # First, clear and setup with replace_handlers=True
        setup_logging(level="INFO", replace_handlers=True)
        initial_handler_count = len(root.handlers)

        # Add a custom handler
        existing_handler = logging.StreamHandler(StringIO())
        root.addHandler(existing_handler)

        # Setup with replace_handlers=False
        setup_logging(level="DEBUG", replace_handlers=False)

        # Existing handler should be preserved
        assert existing_handler in root.handlers
        # Should have initial handlers + existing + new one
        assert len(root.handlers) == initial_handler_count + 2

        # Clean up
        root.removeHandler(existing_handler)


class TestLogAgentSummary:
    """Tests for log_agent_summary function."""

    def test_logs_success_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.summary")

        with caplog.at_level(logging.INFO):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-100",
                orchestration="build",
                status="SUCCESS",
                response="Build completed successfully.",
                attempt=1,
                max_attempts=3,
            )

        assert "SUCCESS" in caplog.text
        assert "TEST-100" in caplog.text
        assert "attempt 1/3" in caplog.text

    def test_truncates_long_response(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.truncate")
        long_response = "A" * 300  # More than 200 chars

        with caplog.at_level(logging.INFO):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-101",
                orchestration="test",
                status="SUCCESS",
                response=long_response,
                attempt=1,
                max_attempts=1,
            )

        # Should be truncated with ...
        assert "..." in caplog.text
        # Should not contain full 300 chars
        assert "A" * 300 not in caplog.text

    def test_removes_newlines_from_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.newlines")
        response_with_newlines = "Line 1\nLine 2\nLine 3"

        with caplog.at_level(logging.INFO):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-102",
                orchestration="test",
                status="FAILURE",
                response=response_with_newlines,
                attempt=2,
                max_attempts=3,
            )

        # Newlines should be replaced with spaces
        assert "Line 1 Line 2 Line 3" in caplog.text

    def test_logs_success_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.level.success")

        with caplog.at_level(logging.DEBUG):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-200",
                orchestration="test",
                status="SUCCESS",
                response="Done",
                attempt=1,
                max_attempts=1,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO

    def test_logs_failure_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.level.failure")

        with caplog.at_level(logging.DEBUG):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-201",
                orchestration="test",
                status="FAILURE",
                response="Failed",
                attempt=1,
                max_attempts=1,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_logs_error_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.level.error")

        with caplog.at_level(logging.DEBUG):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-202",
                orchestration="test",
                status="ERROR",
                response="Error occurred",
                attempt=1,
                max_attempts=1,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR

    def test_logs_timeout_at_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.level.timeout")

        with caplog.at_level(logging.DEBUG):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-203",
                orchestration="test",
                status="TIMEOUT",
                response="Timed out",
                attempt=1,
                max_attempts=1,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING

    def test_logs_unknown_status_at_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger("test.level.unknown")

        with caplog.at_level(logging.DEBUG):
            log_agent_summary(
                logger=logger,
                issue_key="TEST-204",
                orchestration="test",
                status="UNKNOWN",
                response="Unknown status",
                attempt=1,
                max_attempts=1,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING


class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_full_logging_flow_structured(self) -> None:
        # Setup with structured format
        setup_logging(level="DEBUG", json_format=False)

        # Get a logger and add context
        logger = get_logger("sentinel.integration")
        ctx_logger = logger.with_context(issue_key="INT-1", orchestration="test")

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

        try:
            ctx_logger.info("Integration test message")
            output = stream.getvalue()

            assert "INT-1" in output or "issue_key=INT-1" in output
            assert "Integration test message" in output
        finally:
            logger.removeHandler(handler)

    def test_full_logging_flow_json(self) -> None:
        # Setup with JSON format
        setup_logging(level="INFO", json_format=True)

        logger = get_logger("sentinel.json_integration")

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

        try:
            logger.info(
                "JSON test",
                extra={
                    "issue_key": "JSON-1",
                    "orchestration": "deploy",
                    "status": "SUCCESS",
                },
            )
            output = stream.getvalue()
            data = json.loads(output)

            assert data["message"] == "JSON test"
            assert data["issue_key"] == "JSON-1"
        finally:
            logger.removeHandler(handler)


class TestOrchestrationLogManager:
    """Tests for OrchestrationLogManager."""

    def test_creates_base_directory(self, tmp_path: Path) -> None:
        """Test that base directory is created when getting a logger."""
        base_dir = tmp_path / "logs" / "orchestrations"
        assert not base_dir.exists()

        manager = OrchestrationLogManager(base_dir)
        manager.get_logger("test-orch")

        assert base_dir.exists()
        manager.close_all()

    def test_creates_log_file_for_orchestration(self, tmp_path: Path) -> None:
        """Test that a log file is created for each orchestration."""
        manager = OrchestrationLogManager(tmp_path)

        manager.get_logger("my-orchestration")

        log_file = tmp_path / "my-orchestration.log"
        assert log_file.exists()
        manager.close_all()

    def test_returns_same_logger_for_same_orchestration(self, tmp_path: Path) -> None:
        """Test that requesting the same orchestration returns the same logger."""
        manager = OrchestrationLogManager(tmp_path)

        logger1 = manager.get_logger("orch-a")
        logger2 = manager.get_logger("orch-a")

        assert logger1 is logger2
        manager.close_all()

    def test_returns_different_loggers_for_different_orchestrations(self, tmp_path: Path) -> None:
        """Test that different orchestrations get different loggers."""
        manager = OrchestrationLogManager(tmp_path)

        logger1 = manager.get_logger("orch-a")
        logger2 = manager.get_logger("orch-b")

        assert logger1 is not logger2
        manager.close_all()

    def test_logger_writes_to_correct_file(self, tmp_path: Path) -> None:
        """Test that logs are written to the correct orchestration log file."""
        manager = OrchestrationLogManager(tmp_path)

        logger = manager.get_logger("write-test")
        logger.info("Test message for write-test")

        # Flush and close to ensure write
        manager.close_all()

        log_file = tmp_path / "write-test.log"
        content = log_file.read_text()
        assert "Test message for write-test" in content

    def test_separate_orchestrations_have_separate_logs(self, tmp_path: Path) -> None:
        """Test that different orchestrations write to separate files."""
        manager = OrchestrationLogManager(tmp_path)

        logger_a = manager.get_logger("orch-a")
        logger_b = manager.get_logger("orch-b")

        logger_a.info("Message for A")
        logger_b.info("Message for B")

        manager.close_all()

        file_a = tmp_path / "orch-a.log"
        file_b = tmp_path / "orch-b.log"

        content_a = file_a.read_text()
        content_b = file_b.read_text()

        assert "Message for A" in content_a
        assert "Message for B" not in content_a
        assert "Message for B" in content_b
        assert "Message for A" not in content_b

    def test_close_all_clears_handlers(self, tmp_path: Path) -> None:
        """Test that close_all clears all handlers and loggers."""
        manager = OrchestrationLogManager(tmp_path)

        manager.get_logger("orch-1")
        manager.get_logger("orch-2")

        manager.close_all()

        assert len(manager._handlers) == 0
        assert len(manager._loggers) == 0

    def test_logger_uses_structured_formatter(self, tmp_path: Path) -> None:
        """Test that loggers use StructuredFormatter."""
        manager = OrchestrationLogManager(tmp_path)

        manager.get_logger("format-test")

        handler = manager._handlers["format-test"]
        assert isinstance(handler.formatter, StructuredFormatter)
        manager.close_all()

    def test_logger_is_sentinel_logger_type(self, tmp_path: Path) -> None:
        """Test that returned logger is a SentinelLogger."""
        manager = OrchestrationLogManager(tmp_path)

        logger = manager.get_logger("type-test")

        assert isinstance(logger, SentinelLogger)
        manager.close_all()

    def test_logger_supports_with_context(self, tmp_path: Path) -> None:
        """Test that the logger supports with_context for adding context."""
        manager = OrchestrationLogManager(tmp_path)

        logger = manager.get_logger("context-test")
        ctx_logger = logger.with_context(issue_key="TEST-123")
        ctx_logger.info("Contextual message")

        manager.close_all()

        log_file = tmp_path / "context-test.log"
        content = log_file.read_text()
        assert "issue_key=TEST-123" in content
        assert "Contextual message" in content

    def test_can_reopen_after_close_all(self, tmp_path: Path) -> None:
        """Test that new loggers can be created after close_all."""
        manager = OrchestrationLogManager(tmp_path)

        logger1 = manager.get_logger("reopen-test")
        logger1.info("First message")
        manager.close_all()

        logger2 = manager.get_logger("reopen-test")
        logger2.info("Second message")
        manager.close_all()

        log_file = tmp_path / "reopen-test.log"
        content = log_file.read_text()
        # Both messages should be in the file (append mode is default for FileHandler)
        assert "First message" in content
        assert "Second message" in content

    def test_context_manager_basic_usage(self, tmp_path: Path) -> None:
        """Test that OrchestrationLogManager works as a context manager."""
        with OrchestrationLogManager(tmp_path) as manager:
            logger = manager.get_logger("context-manager-test")
            logger.info("Message inside context")

        # After exiting context, handlers should be closed
        assert len(manager._handlers) == 0
        assert len(manager._loggers) == 0

        # But the log file should contain the message
        log_file = tmp_path / "context-manager-test.log"
        content = log_file.read_text()
        assert "Message inside context" in content

    def test_context_manager_returns_self(self, tmp_path: Path) -> None:
        """Test that __enter__ returns the manager instance."""
        manager = OrchestrationLogManager(tmp_path)
        with manager as ctx_manager:
            assert ctx_manager is manager
        manager.close_all()

    def test_context_manager_closes_on_exception(self, tmp_path: Path) -> None:
        """Test that handlers are closed even when an exception occurs."""
        try:
            with OrchestrationLogManager(tmp_path) as manager:
                logger = manager.get_logger("exception-test")
                logger.info("Message before exception")
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Handlers should still be closed after exception
        assert len(manager._handlers) == 0
        assert len(manager._loggers) == 0

        # Log file should contain the message written before exception
        log_file = tmp_path / "exception-test.log"
        content = log_file.read_text()
        assert "Message before exception" in content

    def test_context_manager_multiple_loggers(self, tmp_path: Path) -> None:
        """Test context manager with multiple orchestration loggers."""
        with OrchestrationLogManager(tmp_path) as manager:
            logger_a = manager.get_logger("orch-a")
            logger_b = manager.get_logger("orch-b")
            logger_a.info("Message A")
            logger_b.info("Message B")

        # All handlers should be closed
        assert len(manager._handlers) == 0
        assert len(manager._loggers) == 0

        # Both log files should exist with correct content
        file_a = tmp_path / "orch-a.log"
        file_b = tmp_path / "orch-b.log"
        assert "Message A" in file_a.read_text()
        assert "Message B" in file_b.read_text()


class TestDiagnosticFilter:
    """Tests for DiagnosticFilter (DS-972)."""

    def _make_record(
        self,
        level: int = logging.DEBUG,
        diagnostic_tag: str | None = None,
    ) -> logging.LogRecord:
        """Create a LogRecord with an optional diagnostic_tag extra field."""
        record = logging.LogRecord(
            name="sentinel.test",
            level=level,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        if diagnostic_tag is not None:
            record.diagnostic_tag = diagnostic_tag  # type: ignore[attr-defined]
        return record

    # ------------------------------------------------------------------
    # 1. DiagnosticFilter with no enabled tags
    # ------------------------------------------------------------------

    def test_no_enabled_tags_suppresses_tagged_debug(self) -> None:
        """Tagged DEBUG records are suppressed when no tags are enabled."""
        f = DiagnosticFilter()
        record = self._make_record(level=logging.DEBUG, diagnostic_tag="polling")
        assert f.filter(record) is False

    def test_no_enabled_tags_passes_untagged_debug(self) -> None:
        """Untagged DEBUG records always pass through."""
        f = DiagnosticFilter()
        record = self._make_record(level=logging.DEBUG)
        assert f.filter(record) is True

    def test_no_enabled_tags_passes_non_debug(self) -> None:
        """Non-DEBUG records always pass regardless of tags."""
        f = DiagnosticFilter()
        for level in (logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            record = self._make_record(level=level, diagnostic_tag="polling")
            assert f.filter(record) is True, f"Level {level} should always pass"

    def test_no_enabled_tags_passes_non_debug_without_tag(self) -> None:
        """Non-DEBUG records without a tag pass through."""
        f = DiagnosticFilter()
        record = self._make_record(level=logging.INFO)
        assert f.filter(record) is True

    # ------------------------------------------------------------------
    # 2. DiagnosticFilter with specific tags
    # ------------------------------------------------------------------

    def test_specific_tags_matching_tag_passes(self) -> None:
        """DEBUG records with a matching tag pass through."""
        f = DiagnosticFilter(frozenset({"polling"}))
        record = self._make_record(level=logging.DEBUG, diagnostic_tag="polling")
        assert f.filter(record) is True

    def test_specific_tags_non_matching_tag_suppressed(self) -> None:
        """DEBUG records with a non-matching tag are suppressed."""
        f = DiagnosticFilter(frozenset({"polling"}))
        record = self._make_record(level=logging.DEBUG, diagnostic_tag="execution")
        assert f.filter(record) is False

    def test_specific_tags_multiple_enabled(self) -> None:
        """Multiple enabled tags: matching tags pass, others are suppressed."""
        f = DiagnosticFilter(frozenset({"polling", "execution"}))
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="polling")) is True
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="execution")) is True
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="other")) is False

    def test_specific_tags_untagged_debug_passes(self) -> None:
        """Untagged DEBUG records pass even when specific tags are configured."""
        f = DiagnosticFilter(frozenset({"polling"}))
        record = self._make_record(level=logging.DEBUG)
        assert f.filter(record) is True

    # ------------------------------------------------------------------
    # 3. DiagnosticFilter with wildcard "*"
    # ------------------------------------------------------------------

    def test_wildcard_passes_all_tagged_debug(self) -> None:
        """Wildcard '*' enables all tagged DEBUG records."""
        f = DiagnosticFilter(frozenset({"*"}))
        assert f.allow_all is True
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="polling")) is True
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="execution")) is True
        assert f.filter(self._make_record(level=logging.DEBUG, diagnostic_tag="unknown")) is True

    def test_wildcard_passes_untagged_debug(self) -> None:
        """Wildcard mode still passes untagged DEBUG records."""
        f = DiagnosticFilter(frozenset({"*"}))
        record = self._make_record(level=logging.DEBUG)
        assert f.filter(record) is True

    def test_wildcard_passes_non_debug(self) -> None:
        """Wildcard mode passes non-DEBUG records."""
        f = DiagnosticFilter(frozenset({"*"}))
        record = self._make_record(level=logging.INFO, diagnostic_tag="polling")
        assert f.filter(record) is True

    # ------------------------------------------------------------------
    # 4. from_config_string() edge cases
    # ------------------------------------------------------------------

    def test_from_config_string_empty_string(self) -> None:
        """Empty string produces a filter with no enabled tags."""
        f = DiagnosticFilter.from_config_string("")
        assert f.enabled_tags == frozenset()
        assert f.allow_all is False

    def test_from_config_string_single_tag(self) -> None:
        """Single tag is parsed correctly."""
        f = DiagnosticFilter.from_config_string("polling")
        assert f.enabled_tags == frozenset({"polling"})

    def test_from_config_string_multiple_tags(self) -> None:
        """Multiple comma-separated tags are parsed correctly."""
        f = DiagnosticFilter.from_config_string("polling,execution")
        assert f.enabled_tags == frozenset({"polling", "execution"})

    def test_from_config_string_whitespace_handling(self) -> None:
        """Whitespace around tags is stripped."""
        f = DiagnosticFilter.from_config_string("  polling , execution  , routing  ")
        assert f.enabled_tags == frozenset({"polling", "execution", "routing"})

    def test_from_config_string_wildcard(self) -> None:
        """Wildcard '*' sets allow_all flag."""
        f = DiagnosticFilter.from_config_string("*")
        assert f.allow_all is True
        assert "*" in f.enabled_tags

    def test_from_config_string_only_whitespace(self) -> None:
        """Whitespace-only string is treated as empty."""
        f = DiagnosticFilter.from_config_string("   ")
        assert f.enabled_tags == frozenset()

    def test_from_config_string_empty_tokens_stripped(self) -> None:
        """Consecutive commas and empty tokens are ignored."""
        f = DiagnosticFilter.from_config_string("polling,,execution,")
        assert f.enabled_tags == frozenset({"polling", "execution"})

    def test_from_config_string_whitespace_only_tokens_stripped(self) -> None:
        """Tokens that are only whitespace are ignored."""
        f = DiagnosticFilter.from_config_string("polling,  , execution")
        assert f.enabled_tags == frozenset({"polling", "execution"})

    # ------------------------------------------------------------------
    # 5. Constructor edge cases
    # ------------------------------------------------------------------

    def test_none_enabled_tags_defaults_to_empty(self) -> None:
        """Passing None for enabled_tags defaults to empty frozenset."""
        f = DiagnosticFilter(None)
        assert f.enabled_tags == frozenset()
        assert f.allow_all is False


class TestDiagnosticFilterSetupLoggingIntegration:
    """Integration tests for DiagnosticFilter with setup_logging (DS-972)."""

    def test_setup_logging_installs_diagnostic_filter_on_handler(self) -> None:
        """Verify DiagnosticFilter is installed on handler when diagnostic_tags is passed."""
        setup_logging(level="DEBUG", diagnostic_tags="polling,execution")

        root = logging.getLogger()
        assert len(root.handlers) > 0

        handler = root.handlers[0]
        diagnostic_filters = [f for f in handler.filters if isinstance(f, DiagnosticFilter)]
        assert len(diagnostic_filters) == 1

        df = diagnostic_filters[0]
        assert df.enabled_tags == frozenset({"polling", "execution"})

    def test_setup_logging_installs_diagnostic_filter_empty_tags(self) -> None:
        """Verify DiagnosticFilter is installed even with empty diagnostic_tags."""
        setup_logging(level="DEBUG", diagnostic_tags="")

        root = logging.getLogger()
        handler = root.handlers[0]
        diagnostic_filters = [f for f in handler.filters if isinstance(f, DiagnosticFilter)]
        assert len(diagnostic_filters) == 1

        df = diagnostic_filters[0]
        assert df.enabled_tags == frozenset()

    def test_setup_logging_diagnostic_filter_suppresses_tagged_debug(self) -> None:
        """End-to-end: tagged DEBUG messages are suppressed when tag not enabled."""
        setup_logging(level="DEBUG", diagnostic_tags="")

        test_logger = get_logger("sentinel.diagnostic_integration_test")

        # Capture output on the root handler
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter())
        # Install a DiagnosticFilter on our test handler too
        handler.addFilter(DiagnosticFilter.from_config_string(""))
        test_logger.addHandler(handler)

        try:
            # Tagged DEBUG should be suppressed
            test_logger.debug(
                "This should be suppressed",
                extra={"diagnostic_tag": "polling"},
            )
            # Untagged DEBUG should pass
            test_logger.debug("This should pass through")

            output = stream.getvalue()
            assert "This should be suppressed" not in output
            assert "This should pass through" in output
        finally:
            test_logger.removeHandler(handler)

    def test_setup_logging_diagnostic_filter_passes_enabled_tagged_debug(self) -> None:
        """End-to-end: tagged DEBUG messages pass when tag is enabled."""
        setup_logging(level="DEBUG", diagnostic_tags="polling")

        test_logger = get_logger("sentinel.diagnostic_enabled_test")

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(StructuredFormatter())
        handler.addFilter(DiagnosticFilter.from_config_string("polling"))
        test_logger.addHandler(handler)

        try:
            test_logger.debug(
                "Polling diagnostic message",
                extra={"diagnostic_tag": "polling"},
            )

            output = stream.getvalue()
            assert "Polling diagnostic message" in output
        finally:
            test_logger.removeHandler(handler)


class TestDiagnosticFilterNotOnOrchestrationFileHandlers:
    """Integration tests verifying DiagnosticFilter is absent from OrchestrationLogManager FileHandlers (DS-983).

    OrchestrationLogManager creates per-orchestration FileHandlers that should
    write all DEBUG-level messages unconditionally.  DiagnosticFilter is only
    intended for the root StreamHandler created by setup_logging().  These tests
    codify that documented behaviour and protect against accidental regressions.
    """

    def test_file_handlers_have_no_diagnostic_filter(self, tmp_path: Path) -> None:
        """Verify that no FileHandler created by OrchestrationLogManager has a DiagnosticFilter."""
        with OrchestrationLogManager(tmp_path) as manager:
            manager.get_logger("orch-alpha")
            manager.get_logger("orch-beta")

            for name, handler in manager._handlers.items():
                diagnostic_filters = [
                    f for f in handler.filters if isinstance(f, DiagnosticFilter)
                ]
                assert diagnostic_filters == [], (
                    f"FileHandler for '{name}' unexpectedly has a DiagnosticFilter installed"
                )

    def test_single_orchestration_handler_has_no_diagnostic_filter(self, tmp_path: Path) -> None:
        """Verify a single orchestration's FileHandler has no DiagnosticFilter."""
        manager = OrchestrationLogManager(tmp_path)

        manager.get_logger("solo-orch")

        handler = manager._handlers["solo-orch"]
        assert isinstance(handler, logging.FileHandler)
        diagnostic_filters = [
            f for f in handler.filters if isinstance(f, DiagnosticFilter)
        ]
        assert diagnostic_filters == [], (
            "FileHandler for 'solo-orch' should not have a DiagnosticFilter"
        )
        manager.close_all()

    def test_tagged_debug_written_unconditionally_to_orchestration_log(
        self, tmp_path: Path
    ) -> None:
        """Verify tagged DEBUG messages are written unconditionally to orchestration log files.

        Because OrchestrationLogManager FileHandlers do not have a
        DiagnosticFilter, tagged DEBUG messages must appear in the log file
        regardless of any global diagnostic-tag configuration.
        """
        with OrchestrationLogManager(tmp_path) as manager:
            logger = manager.get_logger("diagnostics-passthrough")

            # Write a tagged DEBUG message (would be suppressed by DiagnosticFilter
            # with empty enabled_tags on a StreamHandler)
            logger.debug(
                "Tagged debug message",
                extra={"diagnostic_tag": "polling"},
            )
            # Also write an untagged DEBUG message for comparison
            logger.debug("Untagged debug message")

        log_file = tmp_path / "diagnostics-passthrough.log"
        content = log_file.read_text()

        assert "Tagged debug message" in content, (
            "Tagged DEBUG message should be written unconditionally to orchestration log"
        )
        assert "Untagged debug message" in content, (
            "Untagged DEBUG message should be written to orchestration log"
        )

    def test_multiple_tagged_messages_all_written(self, tmp_path: Path) -> None:
        """Verify multiple differently-tagged DEBUG messages all appear in orchestration logs."""
        with OrchestrationLogManager(tmp_path) as manager:
            logger = manager.get_logger("multi-tag-test")

            logger.debug("Polling data", extra={"diagnostic_tag": "polling"})
            logger.debug("Execution trace", extra={"diagnostic_tag": "execution"})
            logger.debug("Router decision", extra={"diagnostic_tag": "routing"})
            logger.info("Regular info message")

        log_file = tmp_path / "multi-tag-test.log"
        content = log_file.read_text()

        assert "Polling data" in content
        assert "Execution trace" in content
        assert "Router decision" in content
        assert "Regular info message" in content
