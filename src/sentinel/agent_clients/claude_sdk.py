"""Claude Agent SDK client implementation.

This module provides the Claude Agent SDK client for running agents,
along with supporting code for timing metrics and shutdown handling.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from claude_agent_sdk import ClaudeAgentOptions, query

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
)
from sentinel.config import Config
from sentinel.logging import generate_log_filename, get_logger

logger = get_logger(__name__)


@dataclass
class TimingMetrics:
    """Timing metrics for performance instrumentation.

    Tracks key timing phases to help identify performance bottlenecks:
    - Time from query() call to first message received
    - Time between messages in the async loop
    - Time spent in file I/O vs API wait
    - Total elapsed time

    For long-running operations (DS-189), inter_message_times can be summarized
    to statistical values (min, max, p50, p95, p99) to reduce log file size
    while preserving meaningful performance insights.
    """

    # Threshold for summarizing inter_message_times (DS-189)
    # When message count exceeds this, store statistical summary instead of raw data
    INTER_MESSAGE_TIMES_THRESHOLD: ClassVar[int] = 100

    query_start_time: float = 0.0
    first_message_time: float | None = None
    last_message_time: float | None = None
    total_end_time: float = 0.0
    message_count: int = 0
    inter_message_times: list[float] = field(default_factory=list)
    file_io_time: float = 0.0
    api_wait_time: float = 0.0

    def start_query(self) -> None:
        """Mark the start of a query call."""
        self.query_start_time = time.perf_counter()

    def record_message_received(self) -> None:
        """Record when a message is received from the API."""
        now = time.perf_counter()
        if self.first_message_time is None:
            self.first_message_time = now
        else:
            if self.last_message_time is not None:
                self.inter_message_times.append(now - self.last_message_time)
        self.last_message_time = now
        self.message_count += 1

    def add_file_io_time(self, duration: float) -> None:
        """Add time spent on file I/O operations."""
        self.file_io_time += duration

    def add_api_wait_time(self, duration: float) -> None:
        """Add time spent waiting for API responses."""
        self.api_wait_time += duration

    def finish(self) -> None:
        """Mark the end of the query execution."""
        self.total_end_time = time.perf_counter()

    @property
    def time_to_first_message(self) -> float | None:
        """Time from query start to first message received."""
        if self.first_message_time is None:
            return None
        return self.first_message_time - self.query_start_time

    @property
    def total_elapsed_time(self) -> float:
        """Total elapsed time from query start to finish."""
        return self.total_end_time - self.query_start_time

    @property
    def avg_inter_message_time(self) -> float | None:
        """Average time between messages."""
        if not self.inter_message_times:
            return None
        return sum(self.inter_message_times) / len(self.inter_message_times)

    def _calculate_percentile(self, data: list[float], percentile: float) -> float:
        """Calculate a percentile value from sorted data.

        Args:
            data: Sorted list of float values.
            percentile: Percentile to calculate (0-100). Note: This parameter
                expects values in the range 0-100 (e.g., 50 for median, 95 for
                95th percentile), NOT 0-1. This follows the common convention
                used by most statistical libraries and tools.

        Returns:
            The percentile value.

        Note:
            For very large datasets, consider using statistics.quantiles() from
            the standard library (Python 3.8+) as an alternative. The current
            implementation is optimized for the expected data sizes in typical
            agent operations but may benefit from the standard library for
            significantly larger datasets.
        """
        if not data:
            return 0.0
        n = len(data)
        idx = (percentile / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return data[lower] * (1 - weight) + data[upper] * weight

    def get_inter_message_times_summary(self) -> dict[str, Any]:
        """Get statistical summary of inter_message_times (DS-189).

        Returns a dictionary with min, max, p50, p95, p99 statistics
        for inter_message_times data.

        Returns:
            Dictionary with statistical summary.
        """
        if not self.inter_message_times:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "p50": None,
                "p95": None,
                "p99": None,
            }

        sorted_times = sorted(self.inter_message_times)
        return {
            "count": len(sorted_times),
            "min": sorted_times[0],
            "max": sorted_times[-1],
            "avg": sum(sorted_times) / len(sorted_times),
            "p50": self._calculate_percentile(sorted_times, 50),
            "p95": self._calculate_percentile(sorted_times, 95),
            "p99": self._calculate_percentile(sorted_times, 99),
        }

    def log_metrics(self, operation: str = "Query") -> None:
        """Log the collected timing metrics."""
        logger.info(f"[TIMING] {operation} Performance Metrics:")
        logger.info(f"[TIMING]   Total elapsed time: {self.total_elapsed_time:.3f}s")
        if self.time_to_first_message is not None:
            logger.info(f"[TIMING]   Time to first message: {self.time_to_first_message:.3f}s")
        logger.info(f"[TIMING]   Messages received: {self.message_count}")
        if self.avg_inter_message_time is not None:
            logger.info(f"[TIMING]   Avg inter-message time: {self.avg_inter_message_time:.3f}s")
        if self.file_io_time > 0:
            logger.info(f"[TIMING]   File I/O time: {self.file_io_time:.3f}s")
        if self.api_wait_time > 0:
            logger.info(f"[TIMING]   API wait time: {self.api_wait_time:.3f}s")
        if self.inter_message_times:
            min_time = min(self.inter_message_times)
            max_time = max(self.inter_message_times)
            logger.info(f"[TIMING]   Inter-message time range: {min_time:.3f}s - {max_time:.3f}s")

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting.

        For long-running operations (DS-189), when the number of inter_message_times
        exceeds INTER_MESSAGE_TIMES_THRESHOLD, statistical summaries are stored
        instead of the raw array to reduce log file size while preserving
        meaningful performance insights.

        The output format changes based on the data size:
        - Below threshold: "inter_message_times" contains the raw array
        - Above threshold: "inter_message_times_summary" contains statistical summary
                          (min, max, avg, p50, p95, p99, count)
        """
        result: dict[str, Any] = {
            "total_elapsed_time": self.total_elapsed_time,
            "time_to_first_message": self.time_to_first_message,
            "message_count": self.message_count,
            "avg_inter_message_time": self.avg_inter_message_time,
            "file_io_time": self.file_io_time,
            "api_wait_time": self.api_wait_time,
        }

        # DS-189: Optimize storage for long-running operations
        if len(self.inter_message_times) > self.INTER_MESSAGE_TIMES_THRESHOLD:
            result["inter_message_times_summary"] = self.get_inter_message_times_summary()
        else:
            result["inter_message_times"] = self.inter_message_times

        return result


# Module-level shutdown event for interrupting async operations
_shutdown_event: asyncio.Event | None = None
_shutdown_lock = threading.Lock()


def get_shutdown_event() -> asyncio.Event:
    """Get or create the shutdown event for async operations."""
    global _shutdown_event
    with _shutdown_lock:
        if _shutdown_event is None:
            _shutdown_event = asyncio.Event()
        return _shutdown_event


def request_shutdown() -> None:
    """Request shutdown of any running Claude agent operations."""
    global _shutdown_event
    logger.debug("Shutdown requested for Claude agent operations")
    with _shutdown_lock:
        if _shutdown_event is None:
            _shutdown_event = asyncio.Event()
        _shutdown_event.set()


def reset_shutdown() -> None:
    """Reset the shutdown flag. Used for testing."""
    global _shutdown_event
    with _shutdown_lock:
        _shutdown_event = None


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    global _shutdown_event
    with _shutdown_lock:
        return _shutdown_event is not None and _shutdown_event.is_set()


class ClaudeProcessInterruptedError(Exception):
    """Raised when a Claude agent operation is interrupted by shutdown request."""

    pass


async def _run_query(
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
    collect_metrics: bool = True,
) -> str:
    """Run a query using the Claude Agent SDK.

    Args:
        prompt: The prompt to send.
        model: Optional model identifier.
        cwd: Optional working directory.
        collect_metrics: Whether to collect and log timing metrics.

    Returns:
        The response text.

    Raises:
        ClaudeProcessInterruptedError: If shutdown was requested.
    """
    metrics = TimingMetrics() if collect_metrics else None

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=model,
        cwd=cwd,
        setting_sources=["project", "user"],  # Load skills from project and ~/.claude/skills
    )

    shutdown_event = get_shutdown_event()
    response_text = ""

    if metrics:
        metrics.start_query()
        logger.debug("[TIMING] Query started")

    api_wait_start = time.perf_counter()
    async for message in query(prompt=prompt, options=options):
        if metrics:
            api_wait_end = time.perf_counter()
            metrics.add_api_wait_time(api_wait_end - api_wait_start)
            metrics.record_message_received()

        if shutdown_event.is_set():
            raise ClaudeProcessInterruptedError(
                "Claude agent interrupted by shutdown request"
            )
        if hasattr(message, "text"):
            response_text = message.text
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text"):
                    response_text = block.text

        if metrics:
            api_wait_start = time.perf_counter()

    if metrics:
        metrics.finish()
        metrics.log_metrics("_run_query")

    return response_text


class ClaudeSdkAgentClient(AgentClient):
    """Agent client that uses Claude Agent SDK."""

    def __init__(
        self,
        config: Config,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
        disable_streaming_logs: bool | None = None,
    ) -> None:
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir
        # Use explicit parameter if provided, otherwise fall back to config
        self._disable_streaming_logs = (
            disable_streaming_logs
            if disable_streaming_logs is not None
            else config.disable_streaming_logs
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return "claude"

    def _create_workdir(self, issue_key: str) -> Path:
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")
        self.base_workdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = self.base_workdir / f"{issue_key}_{timestamp}"
        workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created agent working directory: {workdir}")
        return workdir

    def _setup_branch(
        self,
        workdir: Path,
        branch: str,
        create_branch: bool,
        base_branch: str,
    ) -> None:
        """Setup the git branch in the working directory.

        If the branch exists remotely, checks it out and pulls latest changes.
        If it doesn't exist and create_branch is True, creates it from base_branch.
        If it doesn't exist and create_branch is False, raises AgentClientError.

        Args:
            workdir: The working directory (must be a git repository).
            branch: The branch name to checkout/create.
            create_branch: If True, create the branch if it doesn't exist.
            base_branch: The base branch to create new branches from.

        Raises:
            AgentClientError: If branch doesn't exist and create_branch is False,
                or if git operations fail.
        """
        logger.info(f"Setting up branch '{branch}' in {workdir}")

        try:
            # Fetch latest from remote
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
            )

            # Check if branch exists on remote
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", branch],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
            )
            branch_exists = bool(result.stdout.strip())

            if branch_exists:
                # Check if we're already on the branch and up to date with remote
                # DS-366: Combine three git rev-parse calls into a single subprocess
                # to reduce process spawns. Uses newline-separated output format.
                #
                # DS-373: Command structure explanation:
                # The --abbrev-ref flag only affects the FIRST argument (HEAD), returning
                # the branch name instead of the SHA. The second HEAD and origin/{branch}
                # are unaffected by --abbrev-ref and return full SHA hashes. This allows
                # us to get branch name, local SHA, and remote SHA in a single call:
                #   Line 0: current branch name (from --abbrev-ref HEAD)
                #   Line 1: local HEAD SHA (from plain HEAD)
                #   Line 2: remote branch SHA (from origin/{branch})
                branch_state_result = subprocess.run(
                    [
                        "git",
                        "rev-parse",
                        "--abbrev-ref",
                        "HEAD",
                        "HEAD",
                        f"origin/{branch}",
                    ],
                    cwd=workdir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                lines = branch_state_result.stdout.strip().split("\n")
                # DS-373: Defensive validation for expected 3-line output.
                # While check=True should catch most failures, malformed output
                # could still occur in edge cases (e.g., unusual git configurations).
                if len(lines) != 3:
                    raise AgentClientError(
                        f"Unexpected git rev-parse output: expected 3 lines, got {len(lines)}. "
                        f"Output: {branch_state_result.stdout!r}"
                    )
                current_branch = lines[0]
                local_sha = lines[1]
                remote_sha = lines[2]

                if current_branch == branch and local_sha == remote_sha:
                    # Already on the correct branch and up to date
                    logger.info(
                        f"Branch '{branch}' already checked out and up to date with remote"
                    )
                else:
                    # Branch exists - checkout and pull
                    logger.info(f"Branch '{branch}' exists, checking out and pulling")
                    subprocess.run(
                        ["git", "checkout", branch],
                        cwd=workdir,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        ["git", "pull", "origin", branch],
                        cwd=workdir,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
            elif create_branch:
                # Branch doesn't exist but we should create it
                logger.info(
                    f"Branch '{branch}' does not exist, creating from origin/{base_branch}"
                )
                subprocess.run(
                    ["git", "checkout", "-b", branch, f"origin/{base_branch}"],
                    cwd=workdir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                # Branch doesn't exist and we shouldn't create it
                raise AgentClientError(
                    f"Branch '{branch}' does not exist on remote and create_branch is False"
                )

        except subprocess.CalledProcessError as e:
            raise AgentClientError(
                f"Git operation failed: {e.cmd} returned {e.returncode}. "
                f"stderr: {e.stderr}"
            ) from e

    def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        """Run a Claude agent with the given prompt and tools."""
        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key)

        # Setup branch if specified and workdir exists
        if branch and workdir:
            self._setup_branch(workdir, branch, create_branch, base_branch)

        # Build full prompt with context section
        full_prompt = prompt
        if context:
            full_prompt += "\n\nContext:\n" + "".join(f"- {k}: {v}\n" for k, v in context.items())

        # Determine whether to use streaming logs
        # Use streaming if: log_base_dir and issue_key and orchestration_name are all provided
        # AND streaming is not disabled via config (DS-170)
        can_stream = bool(self.log_base_dir and issue_key and orchestration_name)
        use_streaming = can_stream and not self._disable_streaming_logs

        if use_streaming:
            response = asyncio.run(self._run_with_log(full_prompt, tools, timeout_seconds, workdir, model, issue_key, orchestration_name))  # type: ignore
        else:
            response = asyncio.run(self._run_simple(full_prompt, tools, timeout_seconds, workdir, model))
            # When streaming is disabled but we have logging params, write full response after completion (DS-170)
            if can_stream and self._disable_streaming_logs:
                self._write_simple_log(full_prompt, response, issue_key, orchestration_name)  # type: ignore
        return AgentRunResult(response=response, workdir=workdir)

    async def _run_simple(
        self, prompt: str, tools: list[str], timeout: int | None, workdir: Path | None, model: str | None
    ) -> str:
        try:
            coro = _run_query(prompt, model, str(workdir) if workdir else None)
            response = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response
        except ClaudeProcessInterruptedError:
            raise
        except asyncio.TimeoutError:
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s")
        except Exception as e:
            raise AgentClientError(f"Agent execution failed: {e}") from e

    def _write_simple_log(
        self, prompt: str, response: str, issue_key: str, orch_name: str
    ) -> None:
        """Write a simple (non-streaming) log file after execution completes (DS-170).

        This is used when streaming logs are disabled via SENTINEL_DISABLE_STREAMING_LOGS.
        Instead of writing incrementally during execution, this writes the full response
        after completion.

        Args:
            prompt: The prompt sent to the agent.
            response: The complete response from the agent.
            issue_key: The issue key for the log file.
            orch_name: The orchestration name for organizing logs.
        """
        if self.log_base_dir is None:
            return

        start_time = datetime.now()
        log_dir = self.log_base_dir / orch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / generate_log_filename(start_time)

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        sep = "=" * 80
        log_content = (
            f"{sep}\n"
            f"AGENT EXECUTION LOG (non-streaming mode)\n"
            f"{sep}\n\n"
            f"Issue Key:      {issue_key}\n"
            f"Orchestration:  {orch_name}\n"
            f"Time:           {start_time.isoformat()}\n"
            f"Mode:           Non-streaming (SENTINEL_DISABLE_STREAMING_LOGS=true)\n\n"
            f"{sep}\n"
            f"PROMPT\n"
            f"{sep}\n\n"
            f"{prompt}\n\n"
            f"{sep}\n"
            f"AGENT OUTPUT\n"
            f"{sep}\n\n"
            f"{response}\n\n"
            f"{sep}\n"
            f"EXECUTION SUMMARY\n"
            f"{sep}\n\n"
            f"Status:         COMPLETED\n"
            f"End Time:       {end_time.isoformat()}\n"
            f"Duration:       {elapsed_time:.2f}s\n\n"
            f"{sep}\n"
            f"END OF LOG\n"
            f"{sep}\n"
        )

        try:
            log_path.write_text(log_content, encoding="utf-8")
            logger.info(f"Non-streaming log written: {log_path}")
        except Exception as e:
            logger.warning(f"Failed to write non-streaming log to {log_path}: {e}")

    async def _run_with_log(
        self, prompt: str, tools: list[str], timeout: int | None, workdir: Path | None, model: str | None, issue_key: str, orch_name: str
    ) -> str:
        assert self.log_base_dir is not None
        start_time = datetime.now()
        log_dir = self.log_base_dir / orch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / generate_log_filename(start_time)

        # Initialize timing metrics
        metrics = TimingMetrics()

        sep = "=" * 80
        io_start = time.perf_counter()
        log_path.write_text(f"{sep}\nAGENT EXECUTION LOG\n{sep}\n\nIssue Key:      {issue_key}\nOrchestration:  {orch_name}\nStart Time:     {start_time.isoformat()}\n\n{sep}\nPROMPT\n{sep}\n\n{prompt}\n\n{sep}\nAGENT OUTPUT\n{sep}\n\n", encoding="utf-8")
        metrics.add_file_io_time(time.perf_counter() - io_start)
        logger.info(f"Streaming log started: {log_path}")

        status = "ERROR"
        try:
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model=model,
                cwd=str(workdir) if workdir else None,
                setting_sources=["project", "user"],  # Load skills from project and ~/.claude/skills
            )
            shutdown_event = get_shutdown_event()

            async def run_streaming() -> str:
                response_text = ""
                metrics.start_query()
                logger.debug("[TIMING] Streaming query started")

                with open(log_path, "a", encoding="utf-8") as f:
                    api_wait_start = time.perf_counter()
                    async for message in query(prompt=prompt, options=options):
                        api_wait_end = time.perf_counter()
                        metrics.add_api_wait_time(api_wait_end - api_wait_start)
                        metrics.record_message_received()

                        if shutdown_event.is_set():
                            raise ClaudeProcessInterruptedError("Interrupted by shutdown")
                        text = ""
                        if hasattr(message, "text"):
                            text = response_text = message.text
                        elif hasattr(message, "content"):
                            for block in message.content:
                                if hasattr(block, "text"):
                                    text = response_text = block.text
                        if text:
                            io_start = time.perf_counter()
                            f.write(text)
                            f.flush()
                            metrics.add_file_io_time(time.perf_counter() - io_start)

                        api_wait_start = time.perf_counter()
                return response_text

            response = await asyncio.wait_for(run_streaming(), timeout=timeout) if timeout else await run_streaming()
            status = "COMPLETED"
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key})")
            logger.info(f"Agent execution completed, response length: {len(response)}")
            return response
        except ClaudeProcessInterruptedError:
            status = "INTERRUPTED"
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - INTERRUPTED")
            raise
        except asyncio.TimeoutError:
            status = "TIMEOUT"
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - TIMEOUT")
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s")
        except Exception as e:
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - ERROR")
            io_start = time.perf_counter()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] {e}\n")
            metrics.add_file_io_time(time.perf_counter() - io_start)
            raise AgentClientError(f"Agent execution failed: {e}") from e
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            io_start = time.perf_counter()
            with open(log_path, "a", encoding="utf-8") as f:
                # Write timing metrics to log (human-readable format)
                f.write(f"\n{sep}\nTIMING METRICS\n{sep}\n\n")
                f.write(f"Total elapsed time:     {metrics.total_elapsed_time:.3f}s\n")
                if metrics.time_to_first_message is not None:
                    f.write(f"Time to first message:  {metrics.time_to_first_message:.3f}s\n")
                f.write(f"Messages received:      {metrics.message_count}\n")
                if metrics.avg_inter_message_time is not None:
                    f.write(f"Avg inter-message time: {metrics.avg_inter_message_time:.3f}s\n")
                f.write(f"File I/O time:          {metrics.file_io_time:.3f}s\n")
                f.write(f"API wait time:          {metrics.api_wait_time:.3f}s\n")
            # Track the file I/O time for writing timing metrics BEFORE writing final summary
            # This ensures the JSON metrics export includes complete I/O timing (DS-173)
            metrics.add_file_io_time(time.perf_counter() - io_start)
            io_start = time.perf_counter()
            with open(log_path, "a", encoding="utf-8") as f:
                # Write JSON metrics export for programmatic access (DS-173)
                f.write(f"\n{sep}\nMETRICS JSON\n{sep}\n\n")
                f.write(json.dumps(metrics.to_dict(), indent=2))
                f.write("\n")
                # Write execution summary
                f.write(f"\n{sep}\nEXECUTION SUMMARY\n{sep}\n\nStatus:         {status}\nEnd Time:       {end_time.isoformat()}\nDuration:       {duration:.2f}s\n\n{sep}\nEND OF LOG\n{sep}\n")
            metrics.add_file_io_time(time.perf_counter() - io_start)
            logger.info(f"Streaming log completed: {log_path}")
