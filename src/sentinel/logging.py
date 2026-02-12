"""Structured logging configuration for Developer Sentinel."""

from __future__ import annotations

import json
import logging
import re
import sys
import types
from collections.abc import MutableMapping
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple


@lru_cache(maxsize=128)
def _parse_log_timestamp(timestamp: str) -> datetime:
    """Parse a log timestamp string to a UTC-aware datetime (cached).

    Module-level cache avoids redundant ``strptime`` calls when the same
    timestamp string is parsed multiple times (e.g., from both
    ``parse_log_filename`` and ``_format_log_display_name``).

    Args:
        timestamp: Timestamp in ``YYYYMMDD-HHMMSS`` format.

    Returns:
        A UTC-aware ``datetime``.

    Raises:
        ValueError: If *timestamp* does not conform to the expected format.
    """
    return datetime.strptime(timestamp, LOG_TIMESTAMP_FORMAT).replace(tzinfo=UTC)

# Log filename format constants
# New format: {issue_key}_{YYYYMMDD-HHMMSS}_a{attempt}.log
# - _ separates issue key from timestamp (issue keys contain - but not _)
# - - separates date from time within timestamp (disambiguates from _ delimiter)
# - _a{N} suffix guarantees uniqueness across retries
LOG_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
LOG_FILENAME_EXTENSION = ".log"

# Legacy format for backward compatibility: YYYYMMDD_HHMMSS.log
_LEGACY_LOG_FILENAME_FORMAT = "%Y%m%d_%H%M%S"

# Regex for parsing log filenames (DS-966)
# Captures: (issue_key)?, timestamp, attempt_number
# Pattern: {issue_key}_{YYYYMMDD-HHMMSS}_a{N} or {YYYYMMDD-HHMMSS}_a{N}
# Note: The greedy (.+) for issue_key is safe because backtracking is anchored
# by the strict \d{8}-\d{6} timestamp pattern that follows the underscore delimiter.
_LOG_FILENAME_REGEX = re.compile(r"^(?:(.+)_)?(\d{8}-\d{6})_a(\d+)$")


def generate_log_filename(
    timestamp: datetime,
    issue_key: str | None = None,
    attempt: int = 1,
) -> str:
    """Generate a log filename from a timestamp, issue key, and attempt number.

    Creates a consistent log filename format used throughout the codebase
    for agent execution logs.

    Format: {issue_key}_{YYYYMMDD-HHMMSS}_a{attempt}.log
    Fallback (no issue_key): {YYYYMMDD-HHMMSS}_a{attempt}.log

    Args:
        timestamp: The datetime to use for the filename.
        issue_key: Optional issue key to include in the filename.
        attempt: Attempt number (1-based). Defaults to 1.

    Returns:
        Log filename string.
    """
    ts = timestamp.strftime(LOG_TIMESTAMP_FORMAT)
    if issue_key:
        return f"{issue_key}_{ts}_a{attempt}{LOG_FILENAME_EXTENSION}"
    return f"{ts}_a{attempt}{LOG_FILENAME_EXTENSION}"


def parse_log_filename(filename: str) -> datetime | None:
    """Parse a log filename back to a datetime.

    Supports both new format ({issue_key}_{YYYYMMDD-HHMMSS}_a{N}.log)
    and legacy format (YYYYMMDD_HHMMSS.log).

    Delegates to :func:`parse_log_filename_parts` for new-format parsing,
    keeping a single source of truth for the ``.removesuffix()`` /
    regex-match / ``strptime`` pipeline.

    See Also:
        :func:`parse_log_filename_parts` — returns all components (issue key,
        timestamp string, attempt number) instead of just the datetime.

    Args:
        filename: Log filename to parse.

    Returns:
        Parsed datetime, or None if the filename doesn't match any expected format.
    """
    # Delegate to parse_log_filename_parts for the new format, keeping a
    # single source of truth for .removesuffix() and regex parsing (DS-987).
    parts = parse_log_filename_parts(filename)
    if parts is not None:
        return parts.to_datetime()

    # Fall back to legacy format: YYYYMMDD_HHMMSS
    try:
        name = filename.removesuffix(LOG_FILENAME_EXTENSION)
        return datetime.strptime(name, _LEGACY_LOG_FILENAME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


class LogFilenameParts(NamedTuple):
    """Parsed components of a log filename in the new format.

    Provides named attribute access (e.g., ``parts.issue_key``) instead of
    positional tuple indexing (``parts[0]``), improving readability at call
    sites like ``_format_log_display_name``.

    .. warning:: Direct construction safety (DS-986)

        ``NamedTuple`` does **not** enforce field types at runtime.  The
        ``int`` conversion for ``attempt`` is guaranteed by the factory
        function :func:`parse_log_filename_parts`, which calls ``int()``
        on the regex capture group before creating this tuple.  Callers
        constructing ``LogFilenameParts`` directly — rather than via
        :func:`parse_log_filename_parts` — **must** ensure that:

        * ``attempt`` is passed as an ``int`` (not a raw ``str``).
        * ``timestamp`` conforms to the ``YYYYMMDD-HHMMSS`` format
          expected by :meth:`to_datetime`; otherwise ``to_datetime``
          will raise :class:`ValueError`.

        Prefer using :func:`parse_log_filename_parts` to obtain instances
        whenever possible, as it enforces these invariants automatically.

    Attributes:
        issue_key: The issue key prefix, or None if the filename has no
            issue key (e.g., ``"DS-123"`` or ``None``).
        timestamp: The timestamp string in ``YYYYMMDD-HHMMSS`` format
            (always non-None when a match is found).
        attempt: The attempt number as an int, converted from the string
            capture group at construction time for API ergonomics.
    """

    issue_key: str | None
    timestamp: str
    attempt: int

    def to_datetime(self) -> datetime:
        """Convert the timestamp string to a timezone-aware datetime.

        Convenience method to avoid repeating the ``strptime``/``replace``
        pattern across call sites (e.g., ``_format_log_display_name`` and
        ``parse_log_filename``).

        Delegates to the module-level :func:`_parse_log_timestamp` cache
        so that repeated calls with the same timestamp string (e.g., from
        both ``parse_log_filename`` and ``_format_log_display_name``)
        avoid redundant ``strptime`` parsing (DS-994).

        Returns:
            A UTC-aware ``datetime`` parsed from :attr:`timestamp`.

        Raises:
            ValueError: If :attr:`timestamp` does not conform to the
                ``YYYYMMDD-HHMMSS`` format.  This can happen when a
                ``LogFilenameParts`` is constructed directly with a
                malformed timestamp string rather than via
                :func:`parse_log_filename_parts`.
        """
        return _parse_log_timestamp(self.timestamp)

    def __repr__(self) -> str:
        """Return an unambiguous representation for debugging.

        Produces a string like::

            LogFilenameParts(issue_key='DS-123', timestamp='20240115-103045', attempt=1)

        This complements :meth:`__str__`, which reconstructs the filename
        stem for logging contexts.  ``__repr__`` is designed for debugging
        (e.g., in interactive sessions or log output where the type and
        field values should be immediately visible) (DS-994).

        Returns:
            A string in ``LogFilenameParts(...)`` format showing all fields.
        """
        return (
            f"LogFilenameParts(issue_key={self.issue_key!r}, "
            f"timestamp={self.timestamp!r}, attempt={self.attempt!r})"
        )

    def __str__(self) -> str:
        """Reconstruct the original filename pattern for debugging/logging.

        Returns the canonical filename (without ``.log`` extension) that
        ``generate_log_filename`` would produce for the same components.
        This is useful in debugging and logging contexts where a human-readable
        representation of the parsed parts is needed.

        Returns:
            A string in the format ``{issue_key}_{YYYYMMDD-HHMMSS}_a{attempt}``
            or ``{YYYYMMDD-HHMMSS}_a{attempt}`` if ``issue_key`` is ``None``.
        """
        if self.issue_key:
            return f"{self.issue_key}_{self.timestamp}_a{self.attempt}"
        return f"{self.timestamp}_a{self.attempt}"


def parse_log_filename_parts(
    filename: str,
) -> LogFilenameParts | None:
    """Parse a log filename into its component parts.

    Extracts the issue key, timestamp string, and attempt number from a log
    filename in the new format ({issue_key}_{YYYYMMDD-HHMMSS}_a{N}.log).

    This function exposes the regex match groups as a public API, avoiding
    the need for external modules to import the private _LOG_FILENAME_REGEX.

    On a successful match the returned :class:`LogFilenameParts` always contains a valid
    :class:`~datetime.datetime` (UTC) and an ``int`` attempt number;
    only *issue_key* may be ``None``.  The greedy ``rsplit("_", 1)``
    used internally to separate the issue key from the timestamp is safe
    because the anchored ``YYYYMMDD-HHMMSS`` timestamp pattern uses
    hyphens — not underscores — internally, so even issue keys that
    themselves contain underscores (e.g. ``MY_PROJ-123``) are correctly
    preserved.

    .. note:: Extensionless filenames (intentional API behavior)

        The ``.log`` extension is **not** required.  Internally,
        ``str.removesuffix`` strips ``LOG_FILENAME_EXTENSION`` when present
        and is a no-op when the suffix is absent, so bare stems such as
        ``"DS-123_20240115-103045_a1"`` are matched identically to their
        ``.log``-suffixed counterparts.  This is intentional — it allows
        callers that have already stripped the extension (or that operate on
        stems directly) to use this function without re-appending it.  See
        ``test_parses_filename_without_log_extension`` for coverage.

    Args:
        filename: Log filename to parse.  May include or omit the ``.log``
            extension — both forms are accepted (see note above).

    Returns:
        A :class:`LogFilenameParts` named tuple if the filename matches the
        new format, or None if it doesn't match. ``issue_key`` may be None
        if the filename has no issue key prefix. ``timestamp`` and
        ``attempt`` are always non-None when a match is found, since both
        the timestamp and attempt regex groups are non-optional. The
        ``attempt`` value is converted to ``int`` at construction time.
    """
    name = filename.removesuffix(LOG_FILENAME_EXTENSION)
    match = _LOG_FILENAME_REGEX.match(name)
    if match:
        return LogFilenameParts(
            issue_key=match.group(1),
            timestamp=match.group(2),
            attempt=int(match.group(3)),
        )
    return None


class DiagnosticFilter(logging.Filter):
    """Filter that gates debug log messages based on diagnostic tags.

    When installed on a handler, this filter examines each DEBUG-level log
    record for a ``diagnostic_tag`` attribute (set via the ``extra`` dict).
    Records whose tag is **not** in the set of enabled tags are suppressed.
    Records at levels above DEBUG, or without a ``diagnostic_tag``, always
    pass through.

    This allows developers to add tagged diagnostic log lines that are
    silent by default but can be enabled at runtime via the
    ``SENTINEL_DIAGNOSTIC_TAGS`` environment variable (e.g.
    ``SENTINEL_DIAGNOSTIC_TAGS=polling,execution``).  Setting the value
    to ``"*"`` enables all tagged diagnostics.

    .. note::

        This filter is only installed on the root logger's
        ``StreamHandler`` (created by :func:`setup_logging`).  It does
        **not** apply to the per-orchestration ``FileHandler`` instances
        created by :class:`OrchestrationLogManager`, which write all
        DEBUG-level messages unconditionally.  If you need diagnostic
        filtering on per-orchestration file logs, you must add a
        ``DiagnosticFilter`` instance to those handlers explicitly.

    Usage in application code::

        logger.debug(
            "Slot availability: %s", available,
            extra={"diagnostic_tag": "polling"},
        )

    Configuration::

        SENTINEL_DIAGNOSTIC_TAGS=polling,execution  # enable specific tags
        SENTINEL_DIAGNOSTIC_TAGS=*                   # enable all tags

    Attributes:
        enabled_tags: Frozenset of tag strings that are allowed through.
        allow_all: If ``True``, all tagged diagnostics are emitted.
    """

    def __init__(self, enabled_tags: frozenset[str] | None = None) -> None:
        """Initialize the diagnostic filter.

        Args:
            enabled_tags: Set of tag strings to allow.  Pass ``None`` or an
                empty frozenset to suppress all tagged diagnostics.  A
                frozenset containing ``"*"`` enables all tags.
        """
        super().__init__()
        self.enabled_tags: frozenset[str] = enabled_tags or frozenset()
        self.allow_all: bool = "*" in self.enabled_tags

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether the log record should be emitted.

        Args:
            record: The log record to evaluate.

        Returns:
            ``True`` if the record should be emitted, ``False`` otherwise.
        """
        # Non-DEBUG records always pass through.
        if record.levelno != logging.DEBUG:
            return True

        tag: str | None = getattr(record, "diagnostic_tag", None)

        # Records without a diagnostic tag always pass through.
        if tag is None:
            return True

        # If all tags are enabled, pass through.
        if self.allow_all:
            return True

        return tag in self.enabled_tags

    @classmethod
    def from_config_string(cls, tags_csv: str) -> DiagnosticFilter:
        """Create a filter from a comma-separated configuration string.

        Args:
            tags_csv: Comma-separated list of tags (e.g. ``"polling,execution"``).
                Whitespace around tags is stripped.  ``"*"`` enables all tags.
                An empty string means no tagged diagnostics are emitted.

        Returns:
            A configured ``DiagnosticFilter`` instance.
        """
        if not tags_csv.strip():
            return cls(frozenset())
        tags = frozenset(t.strip() for t in tags_csv.split(",") if t.strip())
        return cls(tags)


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
    diagnostic_tags: str = "",
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON-formatted logs.
        replace_handlers: If True, remove existing handlers before adding new ones.
            Set to False to preserve existing handlers (e.g., from third-party libraries).
        diagnostic_tags: Comma-separated list of diagnostic tags to enable.
            When set, only debug log messages whose ``diagnostic_tag`` extra
            field matches one of the enabled tags will be emitted.  Messages
            without a ``diagnostic_tag`` are always emitted.  Set to ``"*"``
            to enable all tagged diagnostics.  Empty string (default) means
            no tagged diagnostics are emitted.
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

    # Install diagnostic filter for tagged debug messages (DS-965).
    diagnostic_filter = DiagnosticFilter.from_config_string(diagnostic_tags)
    handler.addFilter(diagnostic_filter)

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
