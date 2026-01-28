"""Agent execution logging to orchestration-specific directories.

Logs agent executions to structured directories:
    {base_dir}/{orchestration_name}/{datetime}.log

Each log file contains:
- Execution metadata (issue key, orchestration, timestamps)
- Agent prompt
- Agent response (streamed in real-time when using StreamingLogWriter)
- Execution status
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from sentinel.logging import generate_log_filename, get_logger

if TYPE_CHECKING:
    from sentinel.executor import ExecutionStatus

logger = get_logger(__name__)


class StreamingLogWriter:
    """Writes agent execution output to a log file in real-time.

    Usage:
        with StreamingLogWriter(base_dir, orchestration_name, issue_key, prompt) as writer:
            # Stream output lines as they arrive
            writer.write_line("Agent output line...")
            writer.write_line("Another line...")

        # After context exit, writer.log_path contains the log file path
    """

    def __init__(
        self,
        base_dir: Path,
        orchestration_name: str,
        issue_key: str,
        prompt: str,
    ) -> None:
        """Initialize the streaming log writer.

        Args:
            base_dir: Base directory for logs (e.g., ./logs).
            orchestration_name: Name of the orchestration.
            issue_key: The Jira issue key.
            prompt: The prompt sent to the agent.
        """
        self.base_dir = base_dir
        self.orchestration_name = orchestration_name
        self.issue_key = issue_key
        self.prompt = prompt
        self.start_time = datetime.now()
        self._file: IO[str] | None = None
        self._log_path: Path | None = None
        self._response_lines: list[str] = []

    @property
    def log_path(self) -> Path | None:
        """Return the log file path, or None if not yet created."""
        return self._log_path

    def _get_log_path(self) -> Path:
        """Get the log file path for this execution."""
        orch_dir = self.base_dir / self.orchestration_name
        orch_dir.mkdir(parents=True, exist_ok=True)
        filename = generate_log_filename(self.start_time)
        return orch_dir / filename

    def __enter__(self) -> StreamingLogWriter:
        """Open the log file and write the header."""
        self._log_path = self._get_log_path()
        self._file = open(self._log_path, "w", encoding="utf-8")

        separator = "=" * 80
        header = f"""{separator}
AGENT EXECUTION LOG (STREAMING)
{separator}

Issue Key:      {self.issue_key}
Orchestration:  {self.orchestration_name}
Start Time:     {self.start_time.isoformat()}

{separator}
PROMPT
{separator}

{self.prompt}

{separator}
AGENT OUTPUT (streaming)
{separator}

"""
        self._file.write(header)
        self._file.flush()

        logger.info(f"Streaming log started: {self._log_path}")
        return self

    def _get_timestamp(self) -> str:
        """Get the current timestamp in [HH:MM:SS.mmm] format."""
        now = datetime.now()
        return now.strftime("[%H:%M:%S.") + f"{now.microsecond // 1000:03d}]"

    def write_line(self, line: str) -> None:
        """Write a line of output to the log file with timestamp.

        Args:
            line: The line to write (newline will be added if not present).
        """
        if self._file is None:
            return

        if not line.endswith("\n"):
            line = line + "\n"

        timestamped_line = f"{self._get_timestamp()} {line}"
        self._file.write(timestamped_line)
        self._file.flush()
        self._response_lines.append(line)

    def write(self, text: str) -> None:
        """Write text to the log file with timestamp, without adding newline.

        Each call to this method adds its own timestamp prefix. This is intentional
        for monitoring purposes (e.g., tail -f), as each write represents a distinct
        event in the agent output stream.

        Args:
            text: The text to write.
        """
        if self._file is None:
            return

        timestamped_text = f"{self._get_timestamp()} {text}"
        self._file.write(timestamped_text)
        self._file.flush()
        self._response_lines.append(text)

    def get_response(self) -> str:
        """Get the accumulated response text."""
        return "".join(self._response_lines)

    def finalize(
        self,
        status: str,
        attempts: int = 1,
    ) -> None:
        """Write the final metadata to the log file.

        Args:
            status: The execution status (e.g., "SUCCESS", "FAILURE", "ERROR").
            attempts: Number of attempts made.
        """
        if self._file is None:
            return

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        separator = "=" * 80

        footer = f"""
{separator}
EXECUTION SUMMARY
{separator}

Status:         {status}
Attempts:       {attempts}
End Time:       {end_time.isoformat()}
Duration:       {duration:.2f}s

{separator}
END OF LOG
{separator}
"""
        self._file.write(footer)
        self._file.flush()

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"Streaming log completed: {self._log_path}")


class AgentLogger:
    """Writes agent execution logs to orchestration-specific directories."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the agent logger.

        Args:
            base_dir: Base directory for logs (e.g., ./logs).
        """
        self.base_dir = base_dir

    def _get_log_path(self, orchestration_name: str, timestamp: datetime) -> Path:
        """Get the log file path for an execution.

        Args:
            orchestration_name: Name of the orchestration.
            timestamp: Execution timestamp.

        Returns:
            Path to the log file.
        """
        # Create orchestration-specific directory
        orch_dir = self.base_dir / orchestration_name
        orch_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename using centralized format
        filename = generate_log_filename(timestamp)
        return orch_dir / filename

    def log_execution(
        self,
        issue_key: str,
        orchestration_name: str,
        prompt: str,
        response: str,
        status: ExecutionStatus,
        attempts: int,
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Write an agent execution log.

        Args:
            issue_key: The Jira issue key.
            orchestration_name: Name of the orchestration.
            prompt: The prompt sent to the agent.
            response: The agent's response.
            status: The execution status.
            attempts: Number of attempts made.
            start_time: When execution started.
            end_time: When execution ended.

        Returns:
            Path to the written log file.
        """
        log_path = self._get_log_path(orchestration_name, start_time)
        duration = (end_time - start_time).total_seconds()
        separator = "=" * 80

        log_content = f"""{separator}
AGENT EXECUTION LOG
{separator}

Issue Key:      {issue_key}
Orchestration:  {orchestration_name}
Status:         {status.value.upper()}
Attempts:       {attempts}
Start Time:     {start_time.isoformat()}
End Time:       {end_time.isoformat()}
Duration:       {duration:.2f}s

{separator}
PROMPT
{separator}

{prompt}

{separator}
RESPONSE
{separator}

{response}

{separator}
END OF LOG
{separator}
"""

        log_path.write_text(log_content)
        logger.info(f"Agent execution log written to {log_path}")
        return log_path
