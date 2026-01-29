"""Structured logging configuration for Developer Sentinel."""

from __future__ import annotations

import json
import logging
import sys
import types
from collections.abc import MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Log filename format constant
# Used across the codebase for consistent log file naming: YYYYMMDD_HHMMSS.log
LOG_FILENAME_FORMAT = "%Y%m%d_%H%M%S"
LOG_FILENAME_EXTENSION = ".log"


def generate_log_filename(timestamp: datetime) -> str:
    """Generate a log filename from a timestamp.

    Creates a consistent log filename format (YYYYMMDD_HHMMSS.log) used
    throughout the codebase for agent execution logs.

    Args:
        timestamp: The datetime to use for the filename.

    Returns:
        Log filename string in format YYYYMMDD_HHMMSS.log
    """
    return timestamp.strftime(LOG_FILENAME_FORMAT) + LOG_FILENAME_EXTENSION


def parse_log_filename(filename: str) -> datetime | None:
    """Parse a log filename back to a datetime.

    Args:
        filename: Log filename in format YYYYMMDD_HHMMSS.log

    Returns:
        Parsed datetime, or None if the filename doesn't match the expected format.
    """
    try:
        # Remove .log extension using explicit suffix removal
        name = filename.removesuffix(LOG_FILENAME_EXTENSION)
        return datetime.strptime(name, LOG_FILENAME_FORMAT)
    except (ValueError, IndexError):
        return None


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
    replace_handlers: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON-formatted logs.
        replace_handlers: If True, remove existing handlers before adding new ones.
            Set to False to preserve existing handlers (e.g., from third-party libraries).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers only if requested
    if replace_handlers:
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


class OrchestrationLogManager:
    """Manager for per-orchestration log files.

    Creates and manages separate log files for each orchestration,
    allowing for better log organization and isolation.

    Usage:
        log_manager = OrchestrationLogManager(Path("./logs"))
        logger = log_manager.get_logger("my-orchestration")
        logger.info("Processing started")
        # ... later ...
        log_manager.close_all()

    Context manager usage:
        with OrchestrationLogManager(Path("./logs")) as manager:
            logger = manager.get_logger("my-orchestration")
            logger.info("Processing started")
            # ... logs automatically closed on exit
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize the orchestration log manager.

        Args:
            base_dir: Base directory where orchestration log files will be created.
                     Each orchestration gets its own log file in this directory.
        """
        self._base_dir = base_dir
        self._loggers: dict[str, SentinelLogger] = {}
        self._handlers: dict[str, logging.FileHandler] = {}

    def get_logger(self, orchestration_name: str) -> SentinelLogger:
        """Get or create a logger for the specified orchestration.

        Creates a logger that writes to a dedicated log file for the orchestration.
        The log file is created at {base_dir}/{orchestration_name}.log.

        Args:
            orchestration_name: Name of the orchestration (used for logger name
                               and log file name).

        Returns:
            SentinelLogger instance configured to write to the orchestration's
            log file.
        """
        if orchestration_name in self._loggers:
            return self._loggers[orchestration_name]

        # Ensure base directory exists
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Create logger with a unique name
        logger_name = f"sentinel.orchestration.{orchestration_name}"
        logger: SentinelLogger = logging.getLogger(logger_name)  # type: ignore[assignment]

        # Create file handler for this orchestration
        log_file = self._base_dir / f"{orchestration_name}.log"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(StructuredFormatter())
        handler.setLevel(logging.DEBUG)

        # Configure logger
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

        # Store references for cleanup
        self._loggers[orchestration_name] = logger
        self._handlers[orchestration_name] = handler

        return logger

    def close_all(self) -> None:
        """Close all log file handlers and release resources.

        Should be called when shutting down to ensure all log files are
        properly flushed and closed.
        """
        for orchestration_name, handler in self._handlers.items():
            handler.flush()
            handler.close()
            # Remove handler from logger
            if orchestration_name in self._loggers:
                self._loggers[orchestration_name].removeHandler(handler)

        self._handlers.clear()
        self._loggers.clear()

    def __enter__(self) -> OrchestrationLogManager:
        """Enter the context manager.

        Returns:
            Self for use in with statements.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context manager and close all handlers.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        self.close_all()
