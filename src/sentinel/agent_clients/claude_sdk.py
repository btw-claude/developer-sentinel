"""Claude Agent SDK client implementation.

This module provides the Claude Agent SDK client for running agents,
along with supporting code for timing metrics and shutdown handling.

Shutdown handling uses dependency injection to avoid global mutable state.
The ShutdownController class encapsulates shutdown event management,
making the code more testable and avoiding hidden dependencies.

Circuit breaker pattern is implemented to prevent cascading failures
when the Claude API is experiencing issues.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, query

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
    UsageInfo,
)
from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from sentinel.config import Config
from sentinel.logging import generate_log_filename, get_logger
from sentinel.rate_limiter import ClaudeRateLimiter, RateLimitExceededError

logger = get_logger(__name__)

# Environment variable name for enabling Claude Code's experimental Agent Teams feature.
# Extracted as a constant to avoid duplication across code paths (DS-699).
_AGENT_TEAMS_ENV_VAR = "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"


def _build_agent_teams_env(agent_teams: bool) -> dict[str, str]:
    """Build the environment dict for the agent teams feature flag.

    Args:
        agent_teams: Whether to enable Claude Code's experimental Agent Teams feature.

    Returns:
        A dict with the agent teams env var set to "1" if enabled, or an empty dict.
    """
    return {_AGENT_TEAMS_ENV_VAR: "1"} if agent_teams else {}


@dataclass
class TimingMetrics:
    """Timing metrics for performance instrumentation.

    Tracks key timing phases to help identify performance bottlenecks:
    - Time from query() call to first message received
    - Time between messages in the async loop
    - Time spent in file I/O vs API wait
    - Total elapsed time

    For long-running operations, inter_message_times can be summarized
    to statistical values (min, max, p50, p95, p99) to reduce log file size
    while preserving meaningful performance insights.

    Args:
        inter_message_times_threshold: Threshold for summarizing inter_message_times.
            When message count exceeds this, store statistical summary instead of raw data.
            Configurable via SENTINEL_INTER_MESSAGE_TIMES_THRESHOLD environment variable.
    """

    query_start_time: float = 0.0
    first_message_time: float | None = None
    last_message_time: float | None = None
    total_end_time: float = 0.0
    message_count: int = 0
    inter_message_times: list[float] = field(default_factory=list)
    file_io_time: float = 0.0
    api_wait_time: float = 0.0
    # Threshold for summarizing inter_message_times
    # When message count exceeds this, store statistical summary instead of raw data
    # Default matches Config.execution.inter_message_times_threshold default value
    inter_message_times_threshold: int = 100

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
        """Get statistical summary of inter_message_times.

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
        logger.info("[TIMING] %s Performance Metrics:", operation)
        logger.info("[TIMING]   Total elapsed time: %.3fs", self.total_elapsed_time)
        if self.time_to_first_message is not None:
            logger.info("[TIMING]   Time to first message: %.3fs", self.time_to_first_message)
        logger.info("[TIMING]   Messages received: %s", self.message_count)
        if self.avg_inter_message_time is not None:
            logger.info("[TIMING]   Avg inter-message time: %.3fs", self.avg_inter_message_time)
        if self.file_io_time > 0:
            logger.info("[TIMING]   File I/O time: %.3fs", self.file_io_time)
        if self.api_wait_time > 0:
            logger.info("[TIMING]   API wait time: %.3fs", self.api_wait_time)
        if self.inter_message_times:
            min_time = min(self.inter_message_times)
            max_time = max(self.inter_message_times)
            logger.info("[TIMING]   Inter-message time range: %.3fs - %.3fs", min_time, max_time)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting.

        For long-running operations, when the number of inter_message_times
        exceeds inter_message_times_threshold, statistical summaries are stored
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

        # Optimize storage for long-running operations
        if len(self.inter_message_times) > self.inter_message_times_threshold:
            result["inter_message_times_summary"] = self.get_inter_message_times_summary()
        else:
            result["inter_message_times"] = self.inter_message_times

        return result


class ShutdownController:
    """Controller for managing shutdown events in agent operations.

    This class encapsulates the shutdown event and related operations,
    replacing the previous module-level global mutable state. It can be
    injected as a dependency into agent clients and query functions,
    making the code more testable and avoiding hidden dependencies.

    Usage:
        # Create a controller (or use the default shared one)
        controller = ShutdownController()

        # Pass to agent client
        client = ClaudeSdkAgentClient(config, shutdown_controller=controller)

        # Request shutdown
        controller.request_shutdown()

        # Check shutdown status
        if controller.is_shutdown_requested():
            ...

        # Reset for testing
        controller.reset()
    """

    def __init__(self) -> None:
        """Initialize the shutdown controller with no active shutdown event."""
        self._shutdown_event: asyncio.Event | None = None
        self._lock = threading.Lock()

    def get_shutdown_event(self) -> asyncio.Event:
        """Get or create the shutdown event for async operations.

        Returns:
            The asyncio.Event used for shutdown signaling.
        """
        with self._lock:
            if self._shutdown_event is None:
                self._shutdown_event = asyncio.Event()
            return self._shutdown_event

    def request_shutdown(self) -> None:
        """Request shutdown of any running Claude agent operations."""
        logger.debug("Shutdown requested for Claude agent operations")
        with self._lock:
            if self._shutdown_event is None:
                self._shutdown_event = asyncio.Event()
            self._shutdown_event.set()

    def reset(self) -> None:
        """Reset the shutdown flag. Used for testing."""
        with self._lock:
            self._shutdown_event = None

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested.

        Returns:
            True if shutdown has been requested, False otherwise.
        """
        with self._lock:
            return self._shutdown_event is not None and self._shutdown_event.is_set()

    def __repr__(self) -> str:
        with self._lock:
            if self._shutdown_event is None:
                state = "uninitialized"
            elif self._shutdown_event.is_set():
                state = "shutdown_requested"
            else:
                state = "ready"
        return f"ShutdownController(state={state})"


# Shared shutdown controller for internal use.
# External code should inject a ShutdownController instance for better testability.
_default_shutdown_controller = ShutdownController()


def request_shutdown() -> None:
    """Request shutdown of any running Claude agent operations.

    Internal function for shutdown coordination. External code should use
    ShutdownController.request_shutdown() directly via dependency injection.
    """
    _default_shutdown_controller.request_shutdown()


def reset_shutdown() -> None:
    """Reset the shutdown flag. Used for testing.

    Internal function for test utilities. External code should use
    ShutdownController.reset() directly via dependency injection.
    """
    _default_shutdown_controller.reset()


class ClaudeProcessInterruptedError(Exception):
    """Raised when a Claude agent operation is interrupted by shutdown request."""

    pass


def _extract_usage_from_message(message: Any) -> UsageInfo | None:
    """Extract usage information from a Claude SDK ResultMessage.

    The ResultMessage from Claude Agent SDK contains usage data including
    token counts and cost information. This function extracts that data
    into a UsageInfo dataclass.

    Args:
        message: A message from the Claude Agent SDK query stream.

    Returns:
        UsageInfo if the message contains usage data, None otherwise.
    """
    # Check for ResultMessage by looking for characteristic attributes
    # ResultMessage has total_cost_usd at the top level
    if not hasattr(message, "total_cost_usd"):
        return None

    usage_info = UsageInfo()

    # Extract top-level fields from ResultMessage
    if hasattr(message, "total_cost_usd"):
        usage_info.total_cost_usd = getattr(message, "total_cost_usd", 0.0) or 0.0
    if hasattr(message, "duration_ms"):
        usage_info.duration_ms = getattr(message, "duration_ms", 0.0) or 0.0
    if hasattr(message, "duration_api_ms"):
        usage_info.duration_api_ms = getattr(message, "duration_api_ms", 0.0) or 0.0
    if hasattr(message, "num_turns"):
        usage_info.num_turns = getattr(message, "num_turns", 0) or 0

    # Extract token counts from the usage dict if present
    # Note: total_tokens is now a computed property, so we only need to set
    # input_tokens and output_tokens
    if hasattr(message, "usage") and message.usage is not None:
        usage_dict = message.usage
        if isinstance(usage_dict, dict):
            usage_info.input_tokens = usage_dict.get("input_tokens", 0) or 0
            usage_info.output_tokens = usage_dict.get("output_tokens", 0) or 0
        elif hasattr(usage_dict, "input_tokens"):
            # Handle case where usage is an object with attributes
            usage_info.input_tokens = getattr(usage_dict, "input_tokens", 0) or 0
            usage_info.output_tokens = getattr(usage_dict, "output_tokens", 0) or 0

    logger.debug(
        "[USAGE] Extracted usage info: tokens=%s, cost=$%.6f, turns=%s",
        usage_info.total_tokens,
        usage_info.total_cost_usd,
        usage_info.num_turns,
    )

    return usage_info


async def _run_query(
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
    collect_metrics: bool = True,
    shutdown_controller: ShutdownController | None = None,
    agent_teams: bool = False,
) -> tuple[str, UsageInfo | None]:
    """Run a query using the Claude Agent SDK.

    Args:
        prompt: The prompt to send.
        model: Optional model identifier.
        cwd: Optional working directory.
        collect_metrics: Whether to collect and log timing metrics.
        shutdown_controller: Optional ShutdownController for shutdown handling.
            If not provided, uses the default shared controller.
        agent_teams: Whether to enable Claude Code's experimental Agent Teams feature.

    Returns:
        A tuple of (response_text, usage_info) where usage_info is extracted
        from the ResultMessage if available, or None otherwise.

    Raises:
        ClaudeProcessInterruptedError: If shutdown was requested.
    """
    metrics = TimingMetrics() if collect_metrics else None

    env = _build_agent_teams_env(agent_teams)
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model=model,
        cwd=cwd,
        setting_sources=["project", "user"],  # Load skills from project and ~/.claude/skills
        env=env,
    )

    # Use provided controller or fall back to shared default
    controller = (
        shutdown_controller if shutdown_controller is not None else _default_shutdown_controller
    )
    shutdown_event = controller.get_shutdown_event()
    response_text = ""
    usage_info: UsageInfo | None = None

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
            raise ClaudeProcessInterruptedError("Claude agent interrupted by shutdown request")

        # Check for ResultMessage and extract usage data
        extracted_usage = _extract_usage_from_message(message)
        if extracted_usage is not None:
            usage_info = extracted_usage

        # Extract response text from message
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

    return response_text, usage_info


class ClaudeSdkAgentClient(AgentClient):
    """Agent client that uses Claude Agent SDK.

    Implements circuit breaker pattern to prevent cascading failures when
    Claude API is unavailable or experiencing issues.
    """

    def __init__(
        self,
        config: Config,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
        disable_streaming_logs: bool | None = None,
        shutdown_controller: ShutdownController | None = None,
        rate_limiter: ClaudeRateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the Claude SDK agent client.

        Args:
            config: Application configuration.
            base_workdir: Optional base directory for agent working directories.
            log_base_dir: Optional base directory for streaming logs.
            disable_streaming_logs: Whether to disable streaming logs.
                If not provided, uses config.disable_streaming_logs.
            shutdown_controller: Optional ShutdownController for shutdown handling.
                If not provided, uses the default shared controller. Inject a
                custom controller for testing or isolated shutdown handling.
            rate_limiter: Optional rate limiter for Claude API calls.
                If not provided, creates one from config. Inject a custom rate
                limiter for testing or to share a limiter across clients.
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "claude" service.
        """
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir
        # Use explicit parameter if provided, otherwise fall back to config
        self._disable_streaming_logs = (
            disable_streaming_logs
            if disable_streaming_logs is not None
            else config.execution.disable_streaming_logs
        )
        # Use provided controller or fall back to shared default
        self._shutdown_controller = (
            shutdown_controller if shutdown_controller is not None else _default_shutdown_controller
        )
        # Use provided rate limiter or create from config
        self._rate_limiter = (
            rate_limiter if rate_limiter is not None else ClaudeRateLimiter.from_config(config)
        )
        # Use provided circuit breaker or create a default one for the claude service
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="claude",
            config=CircuitBreakerConfig.from_env("claude"),
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return AgentType.CLAUDE

    @property
    def rate_limiter(self) -> ClaudeRateLimiter:
        """Get the rate limiter for this client."""
        return self._rate_limiter

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def get_rate_limit_metrics(self) -> dict[str, Any]:
        """Get current rate limiter metrics.

        Returns:
            Dictionary with rate limit metrics including request counts,
            timing information, and bucket status.
        """
        return self._rate_limiter.get_metrics()

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dictionary with circuit breaker state, config, and metrics.
        """
        return self._circuit_breaker.get_status()

    async def run_agent(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
        agent_teams: bool = False,
    ) -> AgentRunResult:
        """Run a Claude agent with the given prompt.

        This is an async method that properly composes with other async code
        without creating new event loops per call.

        Args:
            prompt: The prompt to send to the agent.
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the client's default model.
            orchestration_name: Optional orchestration name for streaming log files.
            branch: Optional branch name to checkout/create before running the agent.
            create_branch: If True and branch doesn't exist, create it from base_branch.
            base_branch: Base branch to create new branches from. Defaults to "main".
            agent_teams: Whether to enable Claude Code's experimental Agent Teams feature.

        Returns:
            AgentRunResult containing the agent's response text and optional working directory path.

        Raises:
            AgentClientError: If the agent execution fails.
            AgentTimeoutError: If the agent execution times out.
        """
        # Check circuit breaker before attempting the request
        # This is done first to avoid unnecessary disk I/O from _create_workdir()
        # when the circuit is open (consistent with CodexAgentClient.run_agent)
        if not self._circuit_breaker.allow_request():
            raise AgentClientError(
                f"Claude circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key)

        # Setup branch if specified and workdir exists
        if branch and workdir:
            self._setup_branch(
                workdir, branch, create_branch, base_branch,
                subprocess_timeout=self.config.execution.subprocess_timeout,
            )

        # Build full prompt with sanitized context section (DS-675).
        full_prompt = self._build_prompt_with_context(prompt, context)

        # Determine whether to use streaming logs
        # Use streaming if: log_base_dir and issue_key and orchestration_name are all provided
        # AND streaming is not disabled via config
        can_stream = bool(self.log_base_dir and issue_key and orchestration_name)
        use_streaming = can_stream and not self._disable_streaming_logs

        if use_streaming:
            # issue_key and orchestration_name are guaranteed to be str when use_streaming is True
            # (via can_stream check above), but mypy can't infer this
            assert issue_key is not None
            assert orchestration_name is not None
            response, usage_info = await self._run_with_log(
                full_prompt, timeout_seconds, workdir, model, issue_key, orchestration_name,
                agent_teams=agent_teams,
            )
        else:
            response, usage_info = await self._run_simple(
                full_prompt, timeout_seconds, workdir, model,
                agent_teams=agent_teams,
            )
            # When streaming is disabled but we have logging params, write full
            # response after completion
            if can_stream and self._disable_streaming_logs:
                self._write_simple_log(full_prompt, response, issue_key, orchestration_name)  # type: ignore
        return AgentRunResult(response=response, workdir=workdir, usage=usage_info)

    async def _run_simple(
        self,
        prompt: str,
        timeout: int | None,
        workdir: Path | None,
        model: str | None,
        agent_teams: bool = False,
    ) -> tuple[str, UsageInfo | None]:
        """Run agent without streaming logs.

        Note: Circuit breaker is checked in run_agent() before this method is called.

        Args:
            prompt: The prompt to send to the agent.
            timeout: Optional timeout in seconds.
            workdir: Optional working directory path.
            model: Optional model identifier.
            agent_teams: Whether to enable Claude Code's experimental Agent Teams feature.

        Returns:
            A tuple of (response_text, usage_info).
        """
        assert not self._circuit_breaker.is_open, (
            "Circuit breaker is OPEN - "
            "_run_simple() must be called via run_agent(). "
            f"State: {self._circuit_breaker.state.value}"
        )

        try:
            # Acquire rate limit permit before making API call
            if not await self._rate_limiter.acquire_async(timeout=timeout):
                raise AgentClientError("Claude API rate limit timeout - could not acquire permit")

            coro = _run_query(
                prompt,
                model,
                str(workdir) if workdir else None,
                shutdown_controller=self._shutdown_controller,
                agent_teams=agent_teams,
            )
            result = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
            response, usage_info = result
            self._circuit_breaker.record_success()
            logger.info("Agent execution completed, response length: %s", len(response))
            return response, usage_info
        except RateLimitExceededError as e:
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Claude API rate limit exceeded: {e}") from e
        except ClaudeProcessInterruptedError:
            # Don't count interrupts as failures - they're intentional
            raise
        except TimeoutError as e:
            self._circuit_breaker.record_failure(e)
            logger.debug("Agent execution timeout: timeout=%ss, error=%s", timeout, e)
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s") from e
        except OSError as e:
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Agent execution failed due to I/O error: {e}") from e
        except (KeyError, ValueError) as e:
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Agent execution failed due to data error: {e}") from e
        except RuntimeError as e:
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Agent execution failed due to runtime error: {e}") from e

    def _write_simple_log(self, prompt: str, response: str, issue_key: str, orch_name: str) -> None:
        """Write a simple (non-streaming) log file after execution completes.

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

        start_time = datetime.now(tz=UTC)
        log_dir = self.log_base_dir / orch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / generate_log_filename(start_time)

        end_time = datetime.now(tz=UTC)
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
            logger.info("Non-streaming log written: %s", log_path)
        except OSError as e:
            logger.warning(
                "Failed to write non-streaming log to %s due to I/O error: %s", log_path, e
            )
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.warning(
                "Failed to write non-streaming log to %s due to encoding error: %s", log_path, e
            )

    async def _run_with_log(
        self,
        prompt: str,
        timeout: int | None,
        workdir: Path | None,
        model: str | None,
        issue_key: str,
        orch_name: str,
        agent_teams: bool = False,
    ) -> tuple[str, UsageInfo | None]:
        """Run agent with streaming logs.

        Note: Circuit breaker is checked in run_agent() before this method is called.

        Args:
            prompt: The prompt to send to the agent.
            timeout: Optional timeout in seconds.
            workdir: Optional working directory path.
            model: Optional model identifier.
            issue_key: The issue key for organizing logs.
            orch_name: The orchestration name for organizing logs.
            agent_teams: Whether to enable Claude Code's experimental Agent Teams feature.

        Returns:
            A tuple of (response_text, usage_info).
        """
        assert not self._circuit_breaker.is_open, (
            "Circuit breaker is OPEN - "
            "_run_with_log() must be called via run_agent(). "
            f"State: {self._circuit_breaker.state.value}"
        )

        assert self.log_base_dir is not None

        # Acquire rate limit permit before making API call
        try:
            if not await self._rate_limiter.acquire_async(timeout=timeout):
                raise AgentClientError("Claude API rate limit timeout - could not acquire permit")
        except RateLimitExceededError as e:
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Claude API rate limit exceeded: {e}") from e

        start_time = datetime.now(tz=UTC)
        log_dir = self.log_base_dir / orch_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / generate_log_filename(start_time)

        # Initialize timing metrics with configured threshold
        metrics = TimingMetrics(
            inter_message_times_threshold=self.config.execution.inter_message_times_threshold
        )

        sep = "=" * 80
        io_start = time.perf_counter()
        log_header = (
            f"{sep}\n"
            f"AGENT EXECUTION LOG\n"
            f"{sep}\n\n"
            f"Issue Key:      {issue_key}\n"
            f"Orchestration:  {orch_name}\n"
            f"Start Time:     {start_time.isoformat()}\n\n"
            f"{sep}\n"
            f"PROMPT\n"
            f"{sep}\n\n"
            f"{prompt}\n\n"
            f"{sep}\n"
            f"AGENT OUTPUT\n"
            f"{sep}\n\n"
        )
        log_path.write_text(
            log_header,
            encoding="utf-8",
        )
        metrics.add_file_io_time(time.perf_counter() - io_start)
        logger.info("Streaming log started: %s", log_path)

        status = "ERROR"
        usage_info: UsageInfo | None = None
        try:
            env = _build_agent_teams_env(agent_teams)
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model=model,
                cwd=str(workdir) if workdir else None,
                setting_sources=[
                    "project",
                    "user",
                ],  # Load skills from project and ~/.claude/skills
                env=env,
            )
            shutdown_event = self._shutdown_controller.get_shutdown_event()

            async def run_streaming() -> tuple[str, UsageInfo | None]:
                nonlocal usage_info
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

                        # Check for ResultMessage and extract usage data
                        extracted_usage = _extract_usage_from_message(message)
                        if extracted_usage is not None:
                            usage_info = extracted_usage

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
                return response_text, usage_info

            result = (
                await asyncio.wait_for(run_streaming(), timeout=timeout)
                if timeout
                else await run_streaming()
            )
            response, usage_info = result
            status = "COMPLETED"
            self._circuit_breaker.record_success()
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key})")
            logger.info("Agent execution completed, response length: %s", len(response))
            return response, usage_info
        except ClaudeProcessInterruptedError:
            status = "INTERRUPTED"
            # Don't count interrupts as failures - they're intentional
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - INTERRUPTED")
            raise
        except TimeoutError as e:
            status = "TIMEOUT"
            self._circuit_breaker.record_failure(e)
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - TIMEOUT")
            logger.debug(
                "Agent execution timeout: issue_key=%s, timeout=%ss, error=%s",
                issue_key, timeout, e
            )
            raise AgentTimeoutError(f"Agent execution timed out after {timeout}s") from e
        except OSError as e:
            self._circuit_breaker.record_failure(e)
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - ERROR")
            try:
                io_start = time.perf_counter()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[Error] {e}\n")
                metrics.add_file_io_time(time.perf_counter() - io_start)
            except OSError:
                pass  # Ignore errors when writing error to log
            raise AgentClientError(f"Agent execution failed due to I/O error: {e}") from e
        except (KeyError, ValueError) as e:
            self._circuit_breaker.record_failure(e)
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - ERROR")
            try:
                io_start = time.perf_counter()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[Error] {e}\n")
                metrics.add_file_io_time(time.perf_counter() - io_start)
            except OSError:
                pass  # Ignore errors when writing error to log
            raise AgentClientError(f"Agent execution failed due to data error: {e}") from e
        except RuntimeError as e:
            self._circuit_breaker.record_failure(e)
            metrics.finish()
            metrics.log_metrics(f"_run_with_log ({issue_key}) - ERROR")
            try:
                io_start = time.perf_counter()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[Error] {e}\n")
                metrics.add_file_io_time(time.perf_counter() - io_start)
            except OSError:
                pass  # Ignore errors when writing error to log
            raise AgentClientError(f"Agent execution failed due to runtime error: {e}") from e
        finally:
            end_time = datetime.now(tz=UTC)
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
            # This ensures the JSON metrics export includes complete I/O timing
            metrics.add_file_io_time(time.perf_counter() - io_start)
            io_start = time.perf_counter()
            with open(log_path, "a", encoding="utf-8") as f:
                # Write JSON metrics export for programmatic access
                f.write(f"\n{sep}\nMETRICS JSON\n{sep}\n\n")
                all_metrics: dict[str, Any] = {
                    "timing": metrics.to_dict(),
                    "rate_limit": self._rate_limiter.get_metrics(),
                    "circuit_breaker": self._circuit_breaker.get_status(),
                }
                # Include usage info if available
                if usage_info is not None:
                    all_metrics["usage"] = usage_info.to_dict()
                f.write(json.dumps(all_metrics, indent=2))
                f.write("\n")
                # Write execution summary
                exec_summary = (
                    f"\n{sep}\n"
                    f"EXECUTION SUMMARY\n"
                    f"{sep}\n\n"
                    f"Status:         {status}\n"
                    f"End Time:       {end_time.isoformat()}\n"
                    f"Duration:       {duration:.2f}s\n\n"
                    f"{sep}\n"
                    f"END OF LOG\n"
                    f"{sep}\n"
                )
                f.write(exec_summary)
            metrics.add_file_io_time(time.perf_counter() - io_start)
            logger.info("Streaming log completed: %s", log_path)
