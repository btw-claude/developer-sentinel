"""Structured logging configuration for Developer Sentinel."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured log messages.

    Includes timestamp, level, component, message, and any extra context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with structured output.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string.
        """
        # Extract component from logger name (e.g., "sentinel.executor" -> "executor")
        component = record.name.split(".")[-1] if "." in record.name else record.name

        # Build base message
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]

        # Start with base format
        parts = [
            f"{timestamp}",
            f"[{record.levelname:8}]",
            f"[{component:12}]",
        ]

        # Add context fields if present
        context_parts = []
        for key in ("issue_key", "orchestration", "attempt"):
            if hasattr(record, key):
                value = getattr(record, key)
                context_parts.append(f"{key}={value}")

        if context_parts:
            parts.append(f"[{' '.join(context_parts)}]")

        # Add the main message
        parts.append(record.getMessage())

        # Add exception info if present
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return " ".join(parts)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON log messages for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        component = record.name.split(".")[-1] if "." in record.name else record.name

        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "component": component,
            "message": record.getMessage(),
        }

        # Add context fields if present
        for key in ("issue_key", "orchestration", "attempt", "status", "response_summary"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that adds context to all log messages.

    Usage:
        logger = get_logger(__name__)
        ctx_logger = logger.with_context(issue_key="TEST-123", orchestration="review")
        ctx_logger.info("Processing issue")
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """Add context to the log record.

        Args:
            msg: The log message.
            kwargs: Keyword arguments for the log call.

        Returns:
            Tuple of (message, kwargs) with context added.
        """
        extra = kwargs.get("extra", {})
        if self.extra is not None:
            extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


class SentinelLogger(logging.Logger):
    """Custom logger with context support."""

    def with_context(self, **context: Any) -> ContextAdapter:
        """Create a logger adapter with additional context.

        Args:
            **context: Context fields to add to all log messages.

        Returns:
            ContextAdapter with the specified context.
        """
        return ContextAdapter(self, context)


# Register our custom logger class
logging.setLoggerClass(SentinelLogger)


def get_logger(name: str) -> SentinelLogger:
    """Get a logger with the custom SentinelLogger class.

    Args:
        name: Logger name (typically __name__).

    Returns:
        SentinelLogger instance.
    """
    return logging.getLogger(name)  # type: ignore[return-value]


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON-formatted logs.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)

    # Set formatter based on format preference
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StructuredFormatter())

    root_logger.addHandler(handler)

    # Set level for sentinel loggers
    logging.getLogger("sentinel").setLevel(numeric_level)


def log_agent_summary(
    logger: logging.Logger,
    issue_key: str,
    orchestration: str,
    status: str,
    response: str,
    attempt: int,
    max_attempts: int,
) -> None:
    """Log a summary of an agent execution.

    Args:
        logger: Logger to use.
        issue_key: The Jira issue key.
        orchestration: The orchestration name.
        status: Execution status (SUCCESS, FAILURE, ERROR, TIMEOUT).
        response: The agent's response text.
        attempt: Current attempt number.
        max_attempts: Maximum attempts configured.
    """
    # Truncate response for summary
    response_summary = response[:200] + "..." if len(response) > 200 else response
    response_summary = response_summary.replace("\n", " ")

    # Select appropriate log level based on status
    if status in ("ERROR", "FAILURE"):
        log_method = logger.error
    elif status == "SUCCESS":
        log_method = logger.info
    else:
        # TIMEOUT and other statuses use warning
        log_method = logger.warning

    log_method(
        f"Agent execution {status} for {issue_key} "
        f"(attempt {attempt}/{max_attempts}): {response_summary}",
        extra={
            "issue_key": issue_key,
            "orchestration": orchestration,
            "status": status,
            "attempt": attempt,
            "response_summary": response_summary,
        },
    )
