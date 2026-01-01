"""Tests for logging module."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from sentinel.logging import (
    ContextAdapter,
    JSONFormatter,
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
