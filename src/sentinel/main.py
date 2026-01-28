"""Main entry point for the Developer Sentinel orchestrator."""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uvicorn
    from starlette.types import ASGIApp

from sentinel.agent_clients.factory import AgentClientFactory, create_default_factory
from sentinel.agent_logger import AgentLogger
from sentinel.config import Config, load_config
from sentinel.deduplication import DeduplicationManager, build_github_trigger_key
from sentinel.executor import AgentClient, AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubClient, GitHubIssue, GitHubIssueProtocol, GitHubPoller
from sentinel.github_rest_client import GitHubRestClient, GitHubRestTagClient, GitHubTagClient
from sentinel.logging import OrchestrationLogManager, get_logger, setup_logging
from sentinel.orchestration import (
    Orchestration,
    OrchestrationVersion,
    TriggerConfig,
    load_orchestration_file,
    load_orchestrations,
)
from sentinel.poller import JiraClient, JiraIssue, JiraPoller
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient
from sentinel.router import Router, RoutingResult
from sentinel.sdk_clients import (
    ClaudeProcessInterruptedError,
    JiraSdkClient,
    JiraSdkTagClient,
    request_shutdown as request_claude_shutdown,
)
from sentinel.tag_manager import JiraTagClient, TagManager

logger = get_logger(__name__)

# Module-level constant for GitHub issue/PR URL parsing
# This pattern extracts the owner/repo from GitHub URLs like:
# - https://github.com/owner/repo/issues/123
# - https://github.com/owner/repo/pull/456
# Pre-compiled for performance - avoids re-compilation on each call
GITHUB_ISSUE_PR_URL_PATTERN = re.compile(r"https?://[^/]+/([^/]+/[^/]+)/(?:issues|pull)/\d+")


@dataclass
class AttemptCountEntry:
    """Entry in the attempt counts dictionary tracking count and last access time.

    This class tracks both the attempt count and the last time the entry was accessed,
    enabling TTL-based cleanup to prevent unbounded memory growth.
    """

    count: int
    last_access: float  # time.monotonic() timestamp


@dataclass
class RunningStepInfo:
    """Metadata about a currently running execution step.

    This class tracks information about active agent executions for dashboard display.
    """

    issue_key: str
    orchestration_name: str
    attempt_number: int
    started_at: datetime
    issue_url: str  # URL to Jira or GitHub issue


@dataclass
class QueuedIssueInfo:
    """Metadata about an issue waiting in queue for an execution slot.

    This class tracks information about issues that matched orchestration triggers
    but couldn't be executed immediately due to all execution slots being full.
    """

    issue_key: str
    orchestration_name: str
    queued_at: datetime


class GitHubIssueWithRepo:
    """Wrapper for GitHubIssue that provides full key with repo context.

    The GitHubIssue.key property returns "#123" but tag operations need "org/repo#123".
    This wrapper provides the full key while delegating all other GitHubIssueProtocol
    properties to the wrapped issue via __getattr__.

    Uses __getattr__ for automatic delegation to avoid DRY violations when implementing
    all protocol properties explicitly. The only explicit override is the `key` property.
    """

    __slots__ = ("_issue", "_repo")

    def __init__(self, issue: GitHubIssue, repo: str) -> None:
        """Initialize the wrapper.

        Args:
            issue: The underlying GitHubIssue object.
            repo: Repository in "org/repo" format.
        """
        self._issue = issue
        self._repo = repo

    @property
    def key(self) -> str:
        """Return the full issue key including repo context."""
        return f"{self._repo}#{self._issue.number}"

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the wrapped GitHubIssue.

        This enables automatic delegation of all GitHubIssueProtocol properties
        (number, title, body, state, author, assignees, labels, is_pull_request,
        head_ref, base_ref, draft, repo_url, parent_issue_number) without explicit
        property definitions.

        Args:
            name: The attribute name being accessed.

        Returns:
            The attribute value from the wrapped issue.

        Raises:
            AttributeError: If the attribute doesn't exist on the wrapped issue.
        """
        return getattr(self._issue, name)


def extract_repo_from_url(url: str) -> str | None:
    """Extract owner/repo from a GitHub issue or PR URL.

    Parses GitHub URLs to extract the repository in "owner/repo" format.
    Handles both issue URLs (/issues/) and pull request URLs (/pull/).

    Args:
        url: GitHub URL, e.g., "https://github.com/org/repo/issues/123"
            or "https://github.com/org/repo/pull/456"

    Returns:
        Repository in "owner/repo" format, or None if URL is invalid.
    """
    if not url:
        return None

    match = GITHUB_ISSUE_PR_URL_PATTERN.match(url)
    if match:
        return match.group(1)
    return None


class DashboardServer:
    """Background server for the dashboard web interface.

    This class manages a uvicorn server running in a background thread,
    allowing the dashboard to run alongside the main Sentinel polling loop.
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize the dashboard server.

        Args:
            host: The host address to bind to.
            port: The port to listen on.
        """
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self, app: ASGIApp) -> None:
        """Start the dashboard server in a background thread.

        Args:
            app: The ASGI application to serve.
        """
        import uvicorn

        config = uvicorn.Config(
            app=app,
            host=self._host,
            port=self._port,
            log_level="warning",  # Reduce uvicorn logging noise
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        server = self._server  # Capture for closure

        def run_server() -> None:
            """Run the uvicorn server."""
            server.run()

        self._thread = threading.Thread(
            target=run_server,
            name="dashboard-server",
            daemon=True,
        )
        self._thread.start()

        # Wait for uvicorn server to be ready to accept connections
        # The server.started property is set by uvicorn after the server
        # is actually listening, providing a more reliable readiness signal
        start_wait = time.monotonic()
        timeout = 5.0
        while not self._server.started:
            if time.monotonic() - start_wait > timeout:
                logger.warning("Dashboard server startup timed out, continuing anyway")
                break
            time.sleep(0.05)

        if self._server.started:
            logger.info(f"Dashboard server started at http://{self._host}:{self._port}")

    def shutdown(self) -> None:
        """Shutdown the dashboard server gracefully."""
        if self._server is not None:
            logger.info("Shutting down dashboard server...")
            self._server.should_exit = True

            # Wait for the server thread to finish
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning("Dashboard server thread did not terminate gracefully")

            logger.info("Dashboard server shutdown complete")


class Sentinel:
    """Main orchestrator that coordinates polling, routing, and execution."""

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration],
        jira_client: JiraClient,
        tag_client: JiraTagClient,
        agent_factory: AgentClientFactory | AgentClient | None = None,
        agent_logger: AgentLogger | None = None,
        github_client: GitHubClient | None = None,
        github_tag_client: GitHubTagClient | None = None,
        *,
        agent_client: AgentClient | None = None,
    ) -> None:
        """Initialize the Sentinel orchestrator.

        Args:
            config: Application configuration.
            orchestrations: List of orchestration configurations.
            jira_client: Jira client for polling issues.
            tag_client: Jira client for tag operations (required).
            agent_factory: Factory for creating agent clients per-orchestration.
                For backward compatibility with tests, an AgentClient instance
                can also be passed directly (will be wrapped in a simple factory).
            agent_logger: Optional logger for agent execution logs.
            github_client: Optional GitHub client for polling GitHub issues/PRs.
            github_tag_client: Optional GitHub client for tag/label operations.
            agent_client: Deprecated keyword argument. For backward compatibility
                with tests that use agent_client= instead of agent_factory=.
        """
        self.config = config
        self.orchestrations = orchestrations
        self.jira_poller = JiraPoller(jira_client)
        self.github_poller = GitHubPoller(github_client) if github_client else None
        self.router = Router(orchestrations)

        # Support both factory pattern and legacy single client for backward compatibility
        self._agent_logger = agent_logger

        # Handle backward compatibility: agent_client keyword argument takes precedence
        # if both are provided (shouldn't happen in normal usage)
        effective_agent: AgentClientFactory | AgentClient | None = agent_factory
        if agent_client is not None:
            effective_agent = agent_client

        if effective_agent is None:
            raise ValueError(
                "Either agent_factory or agent_client must be provided to Sentinel"
            )

        # Check if we received a factory or a legacy client
        if isinstance(effective_agent, AgentClientFactory):
            # New pattern: use the factory
            self._agent_factory: AgentClientFactory | None = effective_agent
            self._legacy_agent_client: AgentClient | None = None
            # Create a default executor using factory for default agent type
            default_client = effective_agent.create_for_orchestration(None, config)
            self.executor = AgentExecutor(default_client, agent_logger)
        else:
            # Backward compatibility: treat as a legacy agent client
            self._agent_factory = None
            self._legacy_agent_client = effective_agent
            self.executor = AgentExecutor(effective_agent, agent_logger)

        self.tag_manager = TagManager(tag_client, github_client=github_tag_client)
        self._shutdown_requested = False

        # Thread pool for concurrent execution
        self._thread_pool: ThreadPoolExecutor | None = None
        self._active_futures: list[Future[ExecutionResult]] = []
        self._futures_lock = threading.Lock()

        # Track running step metadata for dashboard display
        # Maps future id() to RunningStepInfo for active executions
        self._running_steps: dict[int, RunningStepInfo] = {}

        # Track attempt counts per (issue_key, orchestration_name) pair
        # This tracks how many times an issue has been processed for an orchestration
        # across different polling cycles, providing accurate retry attempt numbers.
        # Changed from dict[tuple[str, str], int] to track last_access time
        # for TTL-based cleanup to prevent unbounded memory growth.
        self._attempt_counts: dict[tuple[str, str], AttemptCountEntry] = {}
        self._attempt_counts_lock = threading.Lock()

        # Track queued issues waiting for execution slots
        # Use deque with maxlen for automatic oldest-item eviction when full.
        # This provides FIFO behavior and prevents unbounded memory growth while
        # keeping the most recent queued issues visible in the dashboard.
        self._issue_queue: deque[QueuedIssueInfo] = deque(maxlen=config.max_queue_size)
        self._queue_lock = threading.Lock()

        # Track known orchestration files for hot-reload detection
        # Maps file path to its last known mtime
        self._known_orchestration_files: dict[Path, float] = {}
        self._init_known_orchestration_files()

        # Versioned orchestrations for hot-reload support
        # Active versions that are used for routing new work
        self._active_versions: list[OrchestrationVersion] = []
        # Old versions pending removal - kept alive until their executions complete
        self._pending_removal_versions: list[OrchestrationVersion] = []
        self._versions_lock = threading.Lock()

        # Observability counters for hot-reload metrics
        # These counters track orchestration lifecycle events for monitoring
        self._orchestrations_loaded_total: int = 0
        self._orchestrations_unloaded_total: int = 0
        self._orchestrations_reloaded_total: int = 0

        # Polling is now completion-driven: wait for task completion, then poll immediately
        # Only sleep poll_interval when no work is found and no tasks are pending

        # Track process start time and last poll times for system status display
        self._start_time: datetime = datetime.now()
        self._last_jira_poll: datetime | None = None
        self._last_github_poll: datetime | None = None

        # Shared deduplication manager for preventing duplicate agent spawns
        self._dedup_manager = DeduplicationManager()

        # Track per-orchestration active execution counts
        # Maps orchestration_name to active execution count for that orchestration.
        # This enables per-orchestration slot limits in future work.
        # Use defaultdict(int) to simplify get-or-default pattern in
        # _increment_per_orch_count and _decrement_per_orch_count methods.
        self._per_orch_active_counts: defaultdict[str, int] = defaultdict(int)
        self._per_orch_counts_lock = threading.Lock()

        # Per-orchestration logging manager for isolated log files
        # Initialized only if config.orchestration_logs_dir is set
        self._orch_log_manager: OrchestrationLogManager | None = None
        if config.orchestration_logs_dir is not None:
            self._orch_log_manager = OrchestrationLogManager(config.orchestration_logs_dir)
            logger.info(
                f"Per-orchestration logging enabled, logs will be written to: "
                f"{config.orchestration_logs_dir}"
            )

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the polling loop."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

    def _log_for_orchestration(
        self, orchestration_name: str, level: int, message: str, **kwargs: Any
    ) -> None:
        """Log a message to the orchestration-specific log file if configured.

        This helper method writes logs to both the main logger and the per-orchestration
        log file (if enabled). This enables better log isolation and organization when
        debugging specific orchestrations.

        Args:
            orchestration_name: The name of the orchestration to log for.
            level: The logging level (e.g., logging.INFO, logging.DEBUG).
            message: The log message to write.
            **kwargs: Additional context fields to include in the log record.
        """
        # Always log to the main logger
        logger.log(level, message, extra=kwargs)

        # Additionally log to the orchestration-specific log file if enabled
        if self._orch_log_manager is not None:
            orch_logger = self._orch_log_manager.get_logger(orchestration_name)
            orch_logger.log(level, message, extra=kwargs)

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Get observability metrics for hot-reload operations.

        Returns a dictionary with the following counters:
        - orchestrations_loaded_total: Count of orchestrations loaded from new files
        - orchestrations_unloaded_total: Count of orchestrations unloaded from deleted files
        - orchestrations_reloaded_total: Count of orchestrations reloaded from modified files

        These metrics can be used for monitoring and alerting on orchestration
        lifecycle events.

        Returns:
            Dict with hot-reload metric counters.
        """
        return {
            "orchestrations_loaded_total": self._orchestrations_loaded_total,
            "orchestrations_unloaded_total": self._orchestrations_unloaded_total,
            "orchestrations_reloaded_total": self._orchestrations_reloaded_total,
        }

    def get_running_steps(self) -> list[RunningStepInfo]:
        """Get information about currently running execution steps.

        Returns a list of RunningStepInfo objects for all active executions.
        This is used by the dashboard to display running steps.

        Returns:
            List of RunningStepInfo for active executions.
        """
        with self._futures_lock:
            # Get running step info for futures that are still running
            running = []
            for future in self._active_futures:
                if not future.done():
                    future_id = id(future)
                    if future_id in self._running_steps:
                        running.append(self._running_steps[future_id])
            return running

    def get_issue_queue(self) -> list[QueuedIssueInfo]:
        """Get information about issues waiting in queue for execution slots.

        Returns a list of QueuedIssueInfo objects for issues that matched
        orchestration triggers but couldn't be executed due to lack of slots.
        This is used by the dashboard to display the issue queue.

        Returns:
            List of QueuedIssueInfo for queued issues.
        """
        with self._queue_lock:
            return list(self._issue_queue)

    def get_start_time(self) -> datetime:
        """Get the process start time.

        Returns:
            The datetime when the Sentinel instance was created.
        """
        return self._start_time

    def get_last_jira_poll(self) -> datetime | None:
        """Get the last Jira poll time.

        Returns:
            The datetime of the last Jira poll, or None if never polled.
        """
        return self._last_jira_poll

    def get_last_github_poll(self) -> datetime | None:
        """Get the last GitHub poll time.

        Returns:
            The datetime of the last GitHub poll, or None if never polled.
        """
        return self._last_github_poll

    def _clear_issue_queue(self) -> None:
        """Clear the issue queue at the start of a new polling cycle.

        The queue is cleared each cycle because issues will be re-polled
        and re-added if they still match and slots are still unavailable.
        """
        with self._queue_lock:
            self._issue_queue.clear()

    def _add_to_issue_queue(self, issue_key: str, orchestration_name: str) -> None:
        """Add an issue to the queue when no execution slot is available.

        Queue uses collections.deque with maxlen for automatic oldest-item
        eviction when full. This provides FIFO behavior - when the queue is at
        capacity, adding a new item automatically evicts the oldest item instead
        of dropping the new one. This ensures the dashboard shows the most recently
        queued issues rather than the oldest ones.

        Enhanced logging to include the evicted item's key for better
        debugging and observability in production.

        Args:
            issue_key: The key of the issue being queued.
            orchestration_name: The name of the orchestration the issue matched.
        """
        with self._queue_lock:
            evicted_item: QueuedIssueInfo | None = None
            # Capture the item that will be evicted before appending
            # since deque.append() doesn't return the evicted item
            if len(self._issue_queue) == self._issue_queue.maxlen:
                evicted_item = self._issue_queue[0]  # Oldest item (leftmost)

            self._issue_queue.append(
                QueuedIssueInfo(
                    issue_key=issue_key,
                    orchestration_name=orchestration_name,
                    queued_at=datetime.now(),
                )
            )

            if evicted_item is not None:
                logger.debug(
                    f"Issue queue at capacity ({self._issue_queue.maxlen}), "
                    f"evicted '{evicted_item.issue_key}' (orchestration: '{evicted_item.orchestration_name}') "
                    f"to add {issue_key} for '{orchestration_name}'"
                )

    def _get_and_increment_attempt_count(
        self, issue_key: str, orchestration_name: str
    ) -> int:
        """Get and increment the attempt count for an issue/orchestration pair.

        This method atomically increments the attempt count and returns the new value.
        It tracks how many times an issue has been processed for a given orchestration
        across different polling cycles, providing accurate retry attempt numbers
        in the Running Steps dashboard.

        Also updates the last_access time to support TTL-based cleanup.

        Args:
            issue_key: The key of the issue being processed.
            orchestration_name: The name of the orchestration being executed.

        Returns:
            The new attempt number (1 for first attempt, 2 for second, etc.).
        """
        key = (issue_key, orchestration_name)
        current_time = time.monotonic()
        with self._attempt_counts_lock:
            entry = self._attempt_counts.get(key)
            if entry is None:
                new_count = 1
            else:
                new_count = entry.count + 1
            self._attempt_counts[key] = AttemptCountEntry(
                count=new_count, last_access=current_time
            )
            return new_count

    def _cleanup_stale_attempt_counts(self) -> int:
        """Clean up stale attempt count entries based on TTL.

        Removes entries from _attempt_counts that haven't been accessed within
        the configured TTL period. This prevents unbounded memory growth for
        long-running processes.

        Returns:
            Number of entries cleaned up.
        """
        ttl = self.config.attempt_counts_ttl
        current_time = time.monotonic()
        cleaned_count = 0

        with self._attempt_counts_lock:
            stale_keys = [
                key
                for key, entry in self._attempt_counts.items()
                if (current_time - entry.last_access) > ttl
            ]
            for key in stale_keys:
                del self._attempt_counts[key]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} stale attempt count entries "
                f"(TTL: {ttl}s)"
            )
        else:
            logger.debug(f"Attempt counts cleanup: no stale entries found (TTL: {ttl}s)")

        return cleaned_count

    def _increment_per_orch_count(self, orchestration_name: str) -> int:
        """Increment the active execution count for an orchestration.

        This method atomically increments the count of active executions for a
        given orchestration and returns the new count. Used to track how many
        concurrent executions are running for each orchestration, enabling
        per-orchestration slot limits.

        Simplified using defaultdict(int) to eliminate the get-or-default pattern.

        Args:
            orchestration_name: The name of the orchestration being executed.

        Returns:
            The new active count for this orchestration after incrementing.
        """
        with self._per_orch_counts_lock:
            self._per_orch_active_counts[orchestration_name] += 1
            new_count = self._per_orch_active_counts[orchestration_name]
            logger.debug(
                f"Incremented per-orch count for '{orchestration_name}': {new_count - 1} -> {new_count}"
            )
            return new_count

    def _decrement_per_orch_count(self, orchestration_name: str) -> int:
        """Decrement the active execution count for an orchestration.

        This method atomically decrements the count of active executions for a
        given orchestration and returns the new count. Called when an execution
        completes (success or failure) to free up the slot for that orchestration.

        If the count would go below zero (should not happen in normal operation),
        it is clamped to zero and a warning is logged.

        Simplified using defaultdict(int) and added cleanup of entries
        when count reaches 0 to prevent unbounded memory growth.

        Args:
            orchestration_name: The name of the orchestration that completed.

        Returns:
            The new active count for this orchestration after decrementing.
        """
        with self._per_orch_counts_lock:
            current_count = self._per_orch_active_counts[orchestration_name]
            if current_count == 0:
                logger.warning(
                    f"Attempted to decrement per-orch count for '{orchestration_name}' "
                    f"but count was already 0"
                )
                return 0
            new_count = current_count - 1
            if new_count == 0:
                # Clean up entry when count reaches 0 to prevent unbounded
                # memory growth if many unique orchestration names are used over time
                del self._per_orch_active_counts[orchestration_name]
            else:
                self._per_orch_active_counts[orchestration_name] = new_count
            logger.debug(
                f"Decremented per-orch count for '{orchestration_name}': {current_count} -> {new_count}"
            )
            return new_count

    def get_per_orch_count(self, orchestration_name: str) -> int:
        """Get the active execution count for a specific orchestration.

        This method provides observability into the per-orchestration execution counts
        for debugging and monitoring purposes. It returns the current count of active
        executions for the specified orchestration.

        Args:
            orchestration_name: The name of the orchestration to query.

        Returns:
            The current active execution count for the orchestration.
            Returns 0 if no active executions exist for this orchestration.
        """
        # Note: Use .get() to avoid creating entries in defaultdict for read-only checks
        with self._per_orch_counts_lock:
            return self._per_orch_active_counts.get(orchestration_name, 0)

    def get_all_per_orch_counts(self) -> dict[str, int]:
        """Get all per-orchestration active execution counts.

        This method provides observability into all per-orchestration execution counts
        for debugging, monitoring, and dashboard display purposes. It returns a copy
        of the current counts dictionary to prevent external modification.

        Returns:
            A dictionary mapping orchestration names to their active execution counts.
            Only includes orchestrations with non-zero counts.
        """
        with self._per_orch_counts_lock:
            return dict(self._per_orch_active_counts)

    def _construct_issue_url(
        self,
        issue: JiraIssue | GitHubIssueProtocol,
        orchestration: Orchestration,
    ) -> str:
        """Construct a URL to the issue based on the trigger source.

        For Jira issues: https://{jira_host}/browse/{issue_key}
        For GitHub issues/PRs: Uses the repo_url from the issue directly.

        Args:
            issue: The issue (Jira or GitHub) to construct a URL for.
            orchestration: The orchestration containing trigger source info.

        Returns:
            The URL to the issue, or empty string if URL cannot be constructed.
        """
        trigger_source = getattr(orchestration.trigger, "source", "jira")

        if trigger_source == "jira":
            # Construct Jira URL from config base URL and issue key
            if self.config.jira_base_url:
                base_url = self.config.jira_base_url.rstrip("/")
                return f"{base_url}/browse/{issue.key}"
            return ""

        elif trigger_source == "github":
            # For GitHub, use the repo_url from the issue which contains the full URL
            if hasattr(issue, "repo_url") and issue.repo_url:
                return issue.repo_url
            # Fallback: construct from agent.github context if available
            github_ctx = orchestration.agent.github
            if github_ctx and github_ctx.org and github_ctx.repo:
                # Extract issue number from the key (e.g., "#123" -> 123 or "org/repo#123" -> 123)
                issue_num = str(issue.key).split("#")[-1] if "#" in str(issue.key) else ""
                if issue_num:
                    host = github_ctx.host or "github.com"
                    return f"https://{host}/{github_ctx.org}/{github_ctx.repo}/issues/{issue_num}"
            return ""

        return ""

    def _get_available_slots_for_orchestration(self, orchestration: Orchestration) -> int:
        """Get available slots for a specific orchestration considering both limits.

        This method calculates the number of available execution slots for a given
        orchestration by considering both:
        1. Global available slots (max_concurrent_executions - total active executions)
        2. Per-orchestration available slots (max_concurrent - orchestration active count)

        The returned value is the minimum of these two limits, ensuring that neither
        the global nor the per-orchestration limit is exceeded.

        If the orchestration has no per-orchestration limit (max_concurrent is None),
        only the global limit is considered.

        Args:
            orchestration: The orchestration to check available slots for.

        Returns:
            The number of available execution slots for this orchestration.
            Returns 0 if no slots are available.
        """
        # Get global available slots
        global_available = self._get_available_slots()

        # If orchestration has no per-orchestration limit, use global only
        if orchestration.max_concurrent is None:
            return global_available

        # Calculate per-orchestration available slots
        # Note: Use .get() to avoid creating entries in defaultdict for read-only checks
        with self._per_orch_counts_lock:
            current_orch_count = self._per_orch_active_counts.get(orchestration.name, 0)
            per_orch_available = orchestration.max_concurrent - current_orch_count

        # Return the minimum of global and per-orchestration limits
        available = min(global_available, per_orch_available)

        logger.debug(
            f"Available slots for '{orchestration.name}': {available} "
            f"(global: {global_available}, per-orch: {per_orch_available}, "
            f"max_concurrent: {orchestration.max_concurrent})"
        )

        return max(0, available)

    def _init_known_orchestration_files(self) -> None:
        """Initialize the set of known orchestration files with their mtimes.

        Scans the orchestrations directory and records all .yaml/.yml files
        with their modification times. This establishes the baseline for detecting
        new and modified files in subsequent poll cycles.
        """
        orchestrations_dir = self.config.orchestrations_dir
        if not orchestrations_dir.exists() or not orchestrations_dir.is_dir():
            return

        for file_path in orchestrations_dir.iterdir():
            if file_path.suffix in (".yaml", ".yml"):
                try:
                    mtime = file_path.stat().st_mtime
                    self._known_orchestration_files[file_path] = mtime
                except OSError as e:
                    logger.warning(f"Could not stat orchestration file {file_path}: {e}")

        logger.debug(
            f"Initialized with {len(self._known_orchestration_files)} known orchestration files"
        )

    def _detect_and_load_orchestration_changes(self) -> tuple[int, int]:
        """Detect and load new and modified orchestration files.

        Scans the orchestrations directory for:
        1. New .yaml/.yml files not in the known files dict
        2. Modified files (mtime changed since last check)

        For new files, orchestrations are added to the active list.
        For modified files, new versions are created and old versions are
        moved to pending_removal (keeping them active until their executions complete).

        Returns:
            Tuple of (new_orchestrations_count, modified_orchestrations_count).
        """
        orchestrations_dir = self.config.orchestrations_dir
        if not orchestrations_dir.exists() or not orchestrations_dir.is_dir():
            return 0, 0

        new_orchestrations_count = 0
        modified_orchestrations_count = 0

        # Scan for new and modified files
        for file_path in sorted(orchestrations_dir.iterdir()):
            if file_path.suffix not in (".yaml", ".yml"):
                continue

            try:
                current_mtime = file_path.stat().st_mtime
            except OSError as e:
                logger.warning(f"Could not stat orchestration file {file_path}: {e}")
                continue

            known_mtime = self._known_orchestration_files.get(file_path)

            if known_mtime is None:
                # New file detected
                logger.info(f"Detected new orchestration file: {file_path}")
                # Pass rebuild_router=False to defer Router rebuild until after
                # all new files are processed (optimization for multiple files)
                loaded = self._load_orchestrations_from_file(
                    file_path, current_mtime, rebuild_router=False
                )
                new_orchestrations_count += loaded
                self._known_orchestration_files[file_path] = current_mtime

            elif current_mtime > known_mtime:
                # Modified file detected
                logger.info(
                    f"Detected modified orchestration file: {file_path} "
                    f"(mtime: {known_mtime} -> {current_mtime})"
                )
                reloaded = self._reload_modified_file(file_path, current_mtime)
                modified_orchestrations_count += reloaded
                self._known_orchestration_files[file_path] = current_mtime

        # Rebuild Router once after all new files are processed (optimization)
        # Modified files already rebuild the Router in _reload_modified_file()
        # Only rebuild if actual orchestrations were loaded, not just files found.
        # This avoids unnecessary rebuilds when files are empty, invalid, or contain
        # only disabled orchestrations. new_orchestrations_count tracks the actual
        # number of orchestrations loaded (sum of _load_orchestrations_from_file() returns),
        # not just the number of files found.
        if new_orchestrations_count > 0:
            self.router = Router(self.orchestrations)

        if new_orchestrations_count > 0 or modified_orchestrations_count > 0:
            logger.info(
                f"Hot-reload complete: {new_orchestrations_count} new, "
                f"{modified_orchestrations_count} reloaded, "
                f"total active: {len(self.orchestrations)}"
            )

        return new_orchestrations_count, modified_orchestrations_count

    def _load_orchestrations_from_file(
        self, file_path: Path, mtime: float, rebuild_router: bool = True
    ) -> int:
        """Load orchestrations from a new file.

        Args:
            file_path: Path to the orchestration file.
            mtime: Modification time of the file.
            rebuild_router: Whether to rebuild the router after loading.
                Set to False when loading multiple files in a batch to avoid
                redundant rebuilds; caller should rebuild router once after all
                files are processed.

        Returns:
            Number of orchestrations loaded.
        """
        try:
            new_orchestrations = load_orchestration_file(file_path)
            if new_orchestrations:
                self.orchestrations.extend(new_orchestrations)
                # Update the router with the new orchestrations
                if rebuild_router:
                    self.router = Router(self.orchestrations)

                # Create versioned entries for tracking
                with self._versions_lock:
                    for orch in new_orchestrations:
                        version = OrchestrationVersion.create(orch, file_path, mtime)
                        self._active_versions.append(version)

                # Update observability counter
                self._orchestrations_loaded_total += len(new_orchestrations)

                logger.info(
                    f"Loaded {len(new_orchestrations)} orchestration(s) from {file_path.name}"
                )
                return len(new_orchestrations)
            else:
                logger.debug(f"No enabled orchestrations in {file_path.name}")
                return 0
        except Exception as e:
            logger.error(f"Failed to load orchestration file {file_path}: {e}")
            return 0

    def _reload_modified_file(self, file_path: Path, new_mtime: float) -> int:
        """Reload orchestrations from a modified file.

        Old versions from this file are moved to pending_removal and kept active
        until their running executions complete. New versions are added to the
        active list.

        Args:
            file_path: Path to the modified orchestration file.
            new_mtime: New modification time of the file.

        Returns:
            Number of orchestrations reloaded.
        """
        try:
            new_orchestrations = load_orchestration_file(file_path)
        except Exception as e:
            logger.error(f"Failed to reload modified orchestration file {file_path}: {e}")
            return 0

        with self._versions_lock:
            # Move old versions from this file to pending_removal
            old_versions = [v for v in self._active_versions if v.source_file == file_path]
            for old_version in old_versions:
                self._active_versions.remove(old_version)
                if old_version.has_active_executions:
                    self._pending_removal_versions.append(old_version)
                    logger.info(
                        f"Orchestration '{old_version.name}' (version {old_version.version_id[:8]}) "
                        f"moved to pending removal with {old_version.active_executions} active execution(s)"
                    )
                else:
                    logger.debug(
                        f"Orchestration '{old_version.name}' (version {old_version.version_id[:8]}) "
                        f"removed immediately (no active executions)"
                    )

            # Remove old orchestrations from the main list
            old_orch_names = {v.name for v in old_versions}
            self.orchestrations = [
                o for o in self.orchestrations
                if o.name not in old_orch_names or not any(
                    v.orchestration is o for v in old_versions
                )
            ]

        # Add new orchestrations
        if new_orchestrations:
            self.orchestrations.extend(new_orchestrations)

            with self._versions_lock:
                for orch in new_orchestrations:
                    version = OrchestrationVersion.create(orch, file_path, new_mtime)
                    self._active_versions.append(version)
                    logger.info(
                        f"Created new version {version.version_id[:8]} for '{orch.name}'"
                    )

        # Update the router with the updated orchestrations
        self.router = Router(self.orchestrations)

        # Update observability counter
        self._orchestrations_reloaded_total += len(new_orchestrations)

        logger.info(
            f"Reloaded {len(new_orchestrations)} orchestration(s) from {file_path.name}"
        )
        return len(new_orchestrations)

    def _cleanup_pending_removal_versions(self) -> int:
        """Clean up old orchestration versions that no longer have active executions.

        Returns:
            Number of versions cleaned up.
        """
        cleaned_count = 0
        with self._versions_lock:
            still_pending = []
            for version in self._pending_removal_versions:
                if version.has_active_executions:
                    still_pending.append(version)
                else:
                    logger.info(
                        f"Cleaned up old orchestration version '{version.name}' "
                        f"(version {version.version_id[:8]})"
                    )
                    cleaned_count += 1
            self._pending_removal_versions = still_pending

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old orchestration version(s)")

        return cleaned_count

    def _get_version_for_orchestration(
        self, orchestration: Orchestration
    ) -> OrchestrationVersion | None:
        """Get the OrchestrationVersion for an orchestration.

        Args:
            orchestration: The orchestration to find.

        Returns:
            The OrchestrationVersion if found, None otherwise.
        """
        with self._versions_lock:
            for version in self._active_versions:
                if version.orchestration is orchestration:
                    return version
            # Also check pending removal versions for in-flight executions
            for version in self._pending_removal_versions:
                if version.orchestration is orchestration:
                    return version
        return None

    def _detect_and_unload_removed_files(self) -> int:
        """Detect and unload orchestrations from removed files.

        Scans the known orchestration files to check if any have been deleted.
        When a file is deleted:
        1. Its orchestrations are removed from the active list
        2. Versions with running executions are moved to pending_removal
        3. Versions without running executions are removed immediately
        4. The file is removed from the known files dict
        5. The router is updated

        Returns:
            Number of orchestrations unloaded.
        """
        orchestrations_dir = self.config.orchestrations_dir
        if not orchestrations_dir.exists() or not orchestrations_dir.is_dir():
            return 0

        unloaded_count = 0
        removed_files: list[Path] = []

        # Check which known files no longer exist
        for file_path in list(self._known_orchestration_files.keys()):
            if not file_path.exists():
                logger.info(f"Detected removed orchestration file: {file_path}")
                removed_files.append(file_path)

        # Process each removed file
        for file_path in removed_files:
            unloaded = self._unload_orchestrations_from_file(file_path)
            unloaded_count += unloaded
            del self._known_orchestration_files[file_path]

        if unloaded_count > 0:
            logger.info(
                f"Unloaded {unloaded_count} orchestration(s) from {len(removed_files)} "
                f"removed file(s), total active: {len(self.orchestrations)}"
            )

        return unloaded_count

    def _unload_orchestrations_from_file(self, file_path: Path) -> int:
        """Unload orchestrations from a removed file.

        Orchestrations with running executions are moved to pending_removal
        and kept alive until those executions complete. Orchestrations without
        running executions are removed immediately.

        Args:
            file_path: Path to the removed orchestration file.

        Returns:
            Number of orchestrations unloaded.
        """
        unloaded_count = 0

        with self._versions_lock:
            # Find versions from this file
            versions_to_remove = [v for v in self._active_versions if v.source_file == file_path]

            for version in versions_to_remove:
                self._active_versions.remove(version)

                if version.has_active_executions:
                    # Keep alive until executions complete
                    self._pending_removal_versions.append(version)
                    logger.info(
                        f"Orchestration '{version.name}' (version {version.version_id[:8]}) "
                        f"moved to pending removal with {version.active_executions} active execution(s)"
                    )
                else:
                    logger.debug(
                        f"Orchestration '{version.name}' (version {version.version_id[:8]}) "
                        f"removed immediately (no active executions)"
                    )

                unloaded_count += 1

            # Remove orchestrations from the main list using identity comparison
            orchestrations_to_remove = [v.orchestration for v in versions_to_remove]
            self.orchestrations = [
                o for o in self.orchestrations
                if not any(o is orch for orch in orchestrations_to_remove)
            ]

        # Update the router with the remaining orchestrations
        if unloaded_count > 0:
            self.router = Router(self.orchestrations)

            # Update observability counter
            self._orchestrations_unloaded_total += unloaded_count

        return unloaded_count

    def _detect_and_load_new_orchestration_files(self) -> int:
        """Detect and load new orchestration files added since last check.

        This method is kept for backwards compatibility. It now delegates to
        _detect_and_load_orchestration_changes() and returns the sum of new
        and modified orchestrations.

        Returns:
            Number of orchestrations loaded or reloaded.
        """
        new_count, modified_count = self._detect_and_load_orchestration_changes()
        return new_count + modified_count

    def _get_available_slots(self) -> int:
        """Get the number of available execution slots.

        Returns:
            Number of execution slots available for new work.
        """
        with self._futures_lock:
            # Count active futures without removing them - results must be collected
            # by _collect_completed_results() to avoid losing execution results
            active_count = sum(1 for f in self._active_futures if not f.done())
            return self.config.max_concurrent_executions - active_count

    def _collect_completed_results(self) -> list[ExecutionResult]:
        """Collect results from completed futures.

        Returns:
            List of execution results from completed futures.
        """
        results: list[ExecutionResult] = []
        with self._futures_lock:
            completed = [f for f in self._active_futures if f.done()]
            for future in completed:
                # Clean up running step metadata
                future_id = id(future)
                self._running_steps.pop(future_id, None)
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error collecting result from future: {e}")
            self._active_futures = [f for f in self._active_futures if not f.done()]
        return results

    def _execute_orchestration_task(
        self,
        issue: JiraIssue | GitHubIssueProtocol,
        orchestration: Orchestration,
        version: OrchestrationVersion | None = None,
    ) -> ExecutionResult | None:
        """Execute a single orchestration in a thread.

        This is the task that runs in the thread pool. It tracks execution counts
        on the OrchestrationVersion to support hot-reload without affecting
        running executions.

        Uses the agent factory to create per-orchestration clients,
        allowing different orchestrations to use different agent types (claude, cursor).

        Args:
            issue: The Jira or GitHub issue to process.
            orchestration: The orchestration to execute.
            version: Optional OrchestrationVersion for execution tracking.

        Returns:
            ExecutionResult if execution completed, None if interrupted.
        """
        issue_key = issue.key
        try:
            if self._shutdown_requested:
                # Log to orchestration-specific log file
                self._log_for_orchestration(
                    orchestration.name,
                    logging.INFO,
                    f"Shutdown requested, skipping {issue_key}",
                )
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
                return None

            # Log execution start to orchestration-specific log file
            self._log_for_orchestration(
                orchestration.name,
                logging.INFO,
                f"Starting execution of '{orchestration.name}' for {issue_key}",
            )

            # Get per-orchestration agent client based on orchestration's agent_type
            # If a legacy agent client was provided, use the default executor
            # Otherwise, use the factory's cached client lookup
            # Use get_or_create_for_orchestration to cache clients by type,
            # avoiding creation of new clients for each orchestration execution
            if self._legacy_agent_client is not None or self._agent_factory is None:
                executor = self.executor
            else:
                agent_type = orchestration.agent.agent_type
                client = self._agent_factory.get_or_create_for_orchestration(
                    agent_type, self.config
                )
                executor = AgentExecutor(client, self._agent_logger)

            result = executor.execute(issue, orchestration)
            self.tag_manager.update_tags(result, orchestration)

            # Log execution result to orchestration-specific log file
            status = "succeeded" if result.succeeded else "failed"
            self._log_for_orchestration(
                orchestration.name,
                logging.INFO if result.succeeded else logging.WARNING,
                f"Execution of '{orchestration.name}' for {issue_key} {status}",
            )

            return result

        except ClaudeProcessInterruptedError:
            # Log to orchestration-specific log file
            self._log_for_orchestration(
                orchestration.name,
                logging.INFO,
                f"Claude process interrupted for {issue_key}",
            )
            try:
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
            except Exception as tag_error:
                logger.error(f"Failed to apply failure tags: {tag_error}")
            return None
        except Exception as e:
            # Log to orchestration-specific log file
            self._log_for_orchestration(
                orchestration.name,
                logging.ERROR,
                f"Failed to execute '{orchestration.name}' for {issue_key}: {e}",
            )
            try:
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
            except Exception as tag_error:
                logger.error(f"Failed to apply failure tags: {tag_error}")
            return None
        finally:
            # Decrement the version's active execution count when done
            if version is not None:
                version.decrement_executions()
            # Decrement per-orchestration active count on completion
            self._decrement_per_orch_count(orchestration.name)

    def run_once(self) -> tuple[list[ExecutionResult], int]:
        """Run a single polling cycle.

        Returns:
            Tuple of (execution results from this cycle, number of tasks submitted).
            The submitted_count is used for eager polling - when > 0, the caller
            should skip the sleep interval and poll immediately for more work.
        """
        # Clear the issue queue at the start of each cycle
        # Issues will be re-added if they still match and slots are unavailable
        self._clear_issue_queue()

        # Detect and load any new or modified orchestration files at the start of each cycle
        self._detect_and_load_new_orchestration_files()

        # Detect and unload orchestrations from removed files
        self._detect_and_unload_removed_files()

        # Clean up old orchestration versions that no longer have active executions
        self._cleanup_pending_removal_versions()

        # Clean up stale attempt count entries to prevent unbounded memory growth
        self._cleanup_stale_attempt_counts()

        # Collect any completed results from previous cycle
        all_results = self._collect_completed_results()

        # Check how many slots are available
        available_slots = self._get_available_slots()
        if available_slots <= 0:
            logger.debug(
                f"All {self.config.max_concurrent_executions} execution slots busy, "
                "skipping polling this cycle"
            )
            return all_results, 0

        # Group orchestrations by trigger source
        jira_orchestrations: list[Orchestration] = []
        github_orchestrations: list[Orchestration] = []

        for orch in self.orchestrations:
            if orch.trigger.source == "github":
                github_orchestrations.append(orch)
            else:
                jira_orchestrations.append(orch)

        # Track how many tasks we've submitted this cycle
        submitted_count = 0

        # Use shared DeduplicationManager for tracking submitted pairs.
        # This provides consistent deduplication behavior across all trigger types.
        submitted_pairs = self._dedup_manager.create_cycle_set()

        # Poll Jira triggers
        if jira_orchestrations:
            jira_submitted = self._poll_jira_triggers(
                jira_orchestrations, all_results, submitted_pairs
            )
            submitted_count += jira_submitted

        # Poll GitHub triggers (if GitHub client is configured)
        if github_orchestrations:
            if self.github_poller:
                github_submitted = self._poll_github_triggers(
                    github_orchestrations, all_results, submitted_pairs
                )
                submitted_count += github_submitted
            else:
                logger.warning(
                    f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                    "but GitHub client is not configured. Set GITHUB_TOKEN to enable."
                )

        if submitted_count > 0:
            logger.info(f"Submitted {submitted_count} execution tasks")

        return all_results, submitted_count

    def _poll_jira_triggers(
        self,
        orchestrations: list[Orchestration],
        all_results: list[ExecutionResult],
        submitted_pairs: set[tuple[str, str]],
    ) -> int:
        """Poll Jira for issues matching orchestration triggers.

        Args:
            orchestrations: List of Jira-triggered orchestrations.
            all_results: List to append synchronous results to.
            submitted_pairs: Set of (issue_key, orchestration_name) pairs already
                submitted in this polling cycle, used to prevent duplicate spawns.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, TriggerConfig]] = []

        for orch in orchestrations:
            trigger_key = f"jira:{orch.trigger.project}:{','.join(orch.trigger.tags)}"
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        submitted_count = 0

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping polling")
                return submitted_count

            # Log polling to orchestration-specific log file
            self._log_for_orchestration(
                orch.name,
                logging.INFO,
                f"Polling Jira for orchestration '{orch.name}'",
            )
            # Update last Jira poll time for system status display
            self._last_jira_poll = datetime.now()
            try:
                issues = self.jira_poller.poll(trigger, self.config.max_issues_per_poll)
                # Log issue count to orchestration-specific log file
                self._log_for_orchestration(
                    orch.name,
                    logging.INFO,
                    f"Found {len(issues)} Jira issues for '{orch.name}'",
                )
            except Exception as e:
                logger.error(f"Failed to poll Jira for '{orch.name}': {e}")
                continue

            # Route issues to matching orchestrations
            routing_results = self.router.route_matched_only(issues)

            # Submit execution tasks
            submitted = self._submit_execution_tasks(
                routing_results, all_results, submitted_pairs
            )
            submitted_count += submitted

        return submitted_count

    def _poll_github_triggers(
        self,
        orchestrations: list[Orchestration],
        all_results: list[ExecutionResult],
        submitted_pairs: set[tuple[str, str]],
    ) -> int:
        """Poll GitHub for issues/PRs matching orchestration triggers.

        Uses project-based polling via GitHub Projects (v2) GraphQL API.
        Deduplication is based on the full trigger key which includes:
        project_owner, project_number, project_filter, and labels.

        This allows different filter/label combinations on the same project to be
        polled separately. For example, two orchestrations with the same project
        but different filters (e.g., "Status=Done" vs "Status=InProgress") will
        each trigger their own polling.

        See build_github_trigger_key() in deduplication.py for the key format.

        Args:
            orchestrations: List of GitHub-triggered orchestrations.
            all_results: List to append synchronous results to.
            submitted_pairs: Set of (issue_key, orchestration_name) pairs already
                submitted in this polling cycle, used to prevent duplicate spawns.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        if not self.github_poller:
            return 0

        # Collect unique trigger configs to avoid duplicate polling
        # Use shared build_github_trigger_key() for consistent key format
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, TriggerConfig]] = []

        for orch in orchestrations:
            # Build deduplication key using shared utility
            trigger_key = build_github_trigger_key(orch)
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        submitted_count = 0

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping polling")
                return submitted_count

            # Log polling to orchestration-specific log file
            self._log_for_orchestration(
                orch.name,
                logging.INFO,
                f"Polling GitHub for orchestration '{orch.name}'",
            )
            # Update last GitHub poll time for system status display
            self._last_github_poll = datetime.now()
            try:
                issues = self.github_poller.poll(trigger, self.config.max_issues_per_poll)
                # Log issue count to orchestration-specific log file
                self._log_for_orchestration(
                    orch.name,
                    logging.INFO,
                    f"Found {len(issues)} GitHub issues/PRs for '{orch.name}'",
                )
            except Exception as e:
                logger.error(f"Failed to poll GitHub for '{orch.name}': {e}")
                continue

            # Convert GitHub issues to include repo context for tag operations
            # Extract repo from each issue's repo_url field instead of trigger.repo
            issues_with_context = self._add_repo_context_from_urls(issues)

            # Route issues to matching orchestrations
            routing_results = self.router.route_matched_only(issues_with_context)

            # Submit execution tasks
            submitted = self._submit_execution_tasks(
                routing_results, all_results, submitted_pairs
            )
            submitted_count += submitted

        return submitted_count

    def _add_repo_context_from_urls(
        self,
        issues: list[GitHubIssue],
    ) -> list[GitHubIssueProtocol]:
        """Add repository context to GitHub issues by extracting repo from URLs.

        Extract repo from each issue's repo_url field instead of using
        a single repo from the trigger config. This is necessary for project-based
        polling where a single project can contain issues from multiple repositories.

        The GitHubIssue.key property returns "#123" but tag operations need "org/repo#123".
        This wraps GitHubIssue objects with the GitHubIssueWithRepo class.

        Args:
            issues: List of GitHubIssue objects from the poller.

        Returns:
            List of GitHubIssueWithRepo objects with updated keys.
            Issues without valid repo URLs are skipped with a warning.
        """
        result: list[GitHubIssueProtocol] = []
        for issue in issues:
            repo = extract_repo_from_url(issue.repo_url)
            if repo:
                result.append(GitHubIssueWithRepo(issue, repo))
            else:
                logger.warning(
                    f"Could not extract repo from URL for issue #{issue.number}: "
                    f"{issue.repo_url!r}"
                )
        return result

    def _submit_execution_tasks(
        self,
        routing_results: list[RoutingResult],
        all_results: list[ExecutionResult],
        submitted_pairs: set[tuple[str, str]] | None = None,
    ) -> int:
        """Submit execution tasks for routed issues.

        Tracks active executions on OrchestrationVersion instances to support
        hot-reload without affecting running executions.

        Args:
            routing_results: List of routing results from the router.
            all_results: List to append synchronous results to.
            submitted_pairs: Optional set of (issue_key, orchestration_name) pairs
                already submitted in this polling cycle. If provided, duplicates
                are skipped to prevent spawning multiple agents for the same
                issue/orchestration combination.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        submitted_count = 0

        for routing_result in routing_results:
            for matched_orch in routing_result.orchestrations:
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping execution")
                    return submitted_count

                issue_key = routing_result.issue.key

                # Use DeduplicationManager for consistent deduplication logic.
                if submitted_pairs is not None:
                    if not self._dedup_manager.check_and_mark(
                        submitted_pairs, issue_key, matched_orch.name
                    ):
                        continue

                # Check if we have available slots using per-orchestration limits
                if self._get_available_slots_for_orchestration(matched_orch) <= 0:
                    # Add to queue instead of silently skipping
                    # Log to orchestration-specific log file
                    self._log_for_orchestration(
                        matched_orch.name,
                        logging.DEBUG,
                        f"No available slots for '{matched_orch.name}', queuing {issue_key}",
                    )
                    self._add_to_issue_queue(issue_key, matched_orch.name)
                    continue

                # Log submission to orchestration-specific log file
                self._log_for_orchestration(
                    matched_orch.name,
                    logging.INFO,
                    f"Submitting '{matched_orch.name}' for {issue_key}",
                )

                # Get the version for this orchestration to track executions
                version = self._get_version_for_orchestration(matched_orch)
                if version is not None:
                    version.increment_executions()

                # Increment per-orchestration active count on submission
                self._increment_per_orch_count(matched_orch.name)

                try:
                    # Mark issue as being processed (remove trigger tags, add in-progress tag)
                    self.tag_manager.start_processing(issue_key, matched_orch)

                    # Submit to thread pool
                    if self._thread_pool is not None:
                        future = self._thread_pool.submit(
                            self._execute_orchestration_task,
                            routing_result.issue,
                            matched_orch,
                            version,
                        )
                        with self._futures_lock:
                            self._active_futures.append(future)
                            # Track running step metadata for dashboard
                            # Track actual retry attempt numbers
                            attempt_number = self._get_and_increment_attempt_count(
                                issue_key, matched_orch.name
                            )
                            # Construct issue URL for dashboard link
                            issue_url = self._construct_issue_url(
                                routing_result.issue, matched_orch
                            )
                            self._running_steps[id(future)] = RunningStepInfo(
                                issue_key=issue_key,
                                orchestration_name=matched_orch.name,
                                attempt_number=attempt_number,
                                started_at=datetime.now(),
                                issue_url=issue_url,
                            )
                        submitted_count += 1
                    else:
                        # Fallback: synchronous execution (e.g., run_once without run())
                        result = self._execute_orchestration_task(
                            routing_result.issue,
                            matched_orch,
                            version,
                        )
                        if result is not None:
                            all_results.append(result)

                except Exception as e:
                    # Decrement version count on submission failure
                    if version is not None:
                        version.decrement_executions()
                    # Decrement per-orchestration count on submission failure
                    self._decrement_per_orch_count(matched_orch.name)
                    logger.error(
                        f"Failed to submit '{matched_orch.name}' for {issue_key}: {e}"
                    )
                    try:
                        self.tag_manager.apply_failure_tags(issue_key, matched_orch)
                    except Exception as tag_error:
                        logger.error(f"Failed to apply failure tags: {tag_error}")

        return submitted_count

    def run_once_and_wait(self) -> list[ExecutionResult]:
        """Run a single polling cycle with concurrent execution and wait for completion.

        This method creates a temporary thread pool for --once mode operation,
        submits all work, and waits for all executions to complete.

        Unlike run() which runs continuously, this method ensures all issues
        from the initial poll are processed by waiting for slots to free up
        and re-polling until no new issues are found.

        Returns:
            List of all execution results.
        """
        max_workers = self.config.max_concurrent_executions
        logger.info(f"Starting single cycle with max {max_workers} concurrent executions")

        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sentinel-exec-",
        )

        try:
            all_results: list[ExecutionResult] = []

            # Keep polling and processing until no new work is found
            while True:
                # Run a polling cycle to submit work
                cycle_results, _ = self.run_once()
                all_results.extend(cycle_results)

                # Check if any work was submitted (active futures)
                with self._futures_lock:
                    pending_futures = [f for f in self._active_futures if not f.done()]

                if not pending_futures:
                    # No pending work, we're done
                    break

                logger.info(f"Waiting for {len(pending_futures)} execution(s) to complete...")

                # Wait for all current work to complete
                while True:
                    with self._futures_lock:
                        pending = [f for f in self._active_futures if not f.done()]
                    if not pending:
                        break
                    time.sleep(0.1)

                # Collect completed results
                completed_results = self._collect_completed_results()
                all_results.extend(completed_results)

            return all_results

        finally:
            # Shutdown thread pool
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True, cancel_futures=False)
                self._thread_pool = None

            # Close per-orchestration log manager to ensure all logs are flushed
            if self._orch_log_manager is not None:
                self._orch_log_manager.close_all()

    def run(self) -> None:
        """Run the main polling loop until shutdown is requested."""

        # Register signal handlers for graceful shutdown
        def handle_shutdown(signum: int, frame: FrameType | None) -> None:
            signal_name = signal.Signals(signum).name
            logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self._shutdown_requested = True
            # Also signal any running Claude subprocesses to terminate
            request_claude_shutdown()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        # Create thread pool for concurrent execution
        max_workers = self.config.max_concurrent_executions
        logger.info(
            f"Starting Sentinel with {len(self.orchestrations)} orchestrations, "
            f"polling every {self.config.poll_interval}s, "
            f"max concurrent executions: {max_workers}"
        )

        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="sentinel-exec-",
        )

        try:
            while not self._shutdown_requested:
                try:
                    results, submitted_count = self.run_once()
                    success_count = sum(1 for r in results if r.succeeded)
                    if results:
                        logger.info(
                            f"Cycle completed: {success_count}/{len(results)} successful"
                        )
                except Exception as e:
                    logger.error(f"Error in polling cycle: {e}")
                    submitted_count = 0  # On error, use normal poll interval

                # Completion-driven polling - wait for task completion, not submission
                # Only sleep poll_interval when no work is found; otherwise wait for completion
                if not self._shutdown_requested:
                    with self._futures_lock:
                        pending_futures = [f for f in self._active_futures if not f.done()]

                    if pending_futures:
                        # Wait for at least one task to complete before polling again
                        logger.debug(
                            f"Waiting for completion of {len(pending_futures)} task(s) before next poll"
                        )
                        # Use wait() with FIRST_COMPLETED to efficiently wait for any completion
                        # Add a small timeout to allow periodic shutdown checks
                        done, _ = wait(pending_futures, timeout=1.0, return_when=FIRST_COMPLETED)
                        while not done and not self._shutdown_requested:
                            with self._futures_lock:
                                pending_futures = [f for f in self._active_futures if not f.done()]
                            if not pending_futures:
                                break
                            done, _ = wait(pending_futures, timeout=1.0, return_when=FIRST_COMPLETED)

                        if done:
                            logger.debug(
                                f"{len(done)} task(s) completed, polling immediately for more work"
                            )
                    elif submitted_count == 0:
                        # No work found and no pending tasks, sleep before next poll
                        logger.debug(f"Sleeping for {self.config.poll_interval}s")
                        # Sleep in small increments to allow quick shutdown
                        for _ in range(self.config.poll_interval):
                            if self._shutdown_requested:
                                break
                            time.sleep(1)
                    else:
                        # submitted_count > 0 but no pending futures - tasks completed very quickly
                        logger.debug(
                            f"Submitted {submitted_count} task(s) but all completed quickly, polling immediately"
                        )

            # Wait for active tasks to complete
            logger.info("Waiting for active executions to complete...")
            with self._futures_lock:
                active_count = len([f for f in self._active_futures if not f.done()])
            if active_count > 0:
                logger.info(f"Waiting for {active_count} active execution(s)...")

            # Collect final results
            final_results = self._collect_completed_results()
            if final_results:
                success_count = sum(1 for r in final_results if r.succeeded)
                logger.info(
                    f"Final batch: {success_count}/{len(final_results)} successful"
                )

        finally:
            # Shutdown thread pool
            if self._thread_pool is not None:
                logger.info("Shutting down thread pool...")
                self._thread_pool.shutdown(wait=True, cancel_futures=False)
                self._thread_pool = None

            # Close per-orchestration log manager to ensure all logs are flushed
            if self._orch_log_manager is not None:
                logger.info("Closing per-orchestration log files...")
                self._orch_log_manager.close_all()

        logger.info("Sentinel shutdown complete")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Developer Sentinel - Jira to Claude Agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to orchestrations directory (default: ./orchestrations)",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (no polling loop)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Poll interval in seconds (overrides SENTINEL_POLL_INTERVAL)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Log level (overrides SENTINEL_LOG_LEVEL)",
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (default: ./.env)",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parsed = parse_args(args)

    # Load configuration
    config = load_config(parsed.env_file)

    # Apply CLI overrides (Config is frozen, so we use replace())
    overrides: dict[str, Any] = {}
    if parsed.config_dir:
        overrides["orchestrations_dir"] = parsed.config_dir
    if parsed.interval:
        overrides["poll_interval"] = parsed.interval
    if parsed.log_level:
        overrides["log_level"] = parsed.log_level
    if overrides:
        config = replace(config, **overrides)

    # Setup logging
    setup_logging(config.log_level, json_format=config.log_json)

    # Load orchestrations
    logger.info(f"Loading orchestrations from {config.orchestrations_dir}")
    try:
        orchestrations = load_orchestrations(config.orchestrations_dir)
    except Exception as e:
        logger.error(f"Failed to load orchestrations: {e}")
        return 1

    if not orchestrations:
        logger.warning("No orchestrations found, exiting")
        return 0

    logger.info(f"Loaded {len(orchestrations)} orchestrations")

    # Initialize clients
    # Use REST clients for Jira when configured (faster, no Claude invocations)
    # Fall back to SDK clients if Jira REST is not configured
    jira_client: JiraClient
    tag_client: JiraTagClient

    if config.jira_configured:
        logger.info("Using Jira REST API clients (direct HTTP)")
        jira_client = JiraRestClient(
            base_url=config.jira_base_url,
            email=config.jira_email,
            api_token=config.jira_api_token,
        )
        tag_client = JiraRestTagClient(
            base_url=config.jira_base_url,
            email=config.jira_email,
            api_token=config.jira_api_token,
        )
    else:
        logger.info("Using Jira SDK clients (via Claude Agent SDK)")
        logger.warning(
            "Jira REST API not configured. Set JIRA_BASE_URL, JIRA_EMAIL, "
            "and JIRA_API_TOKEN for faster polling."
        )
        jira_client = JiraSdkClient(config)
        tag_client = JiraSdkTagClient(config)

    # Initialize GitHub clients if configured
    github_client: GitHubClient | None = None
    github_tag_client: GitHubTagClient | None = None

    if config.github_configured:
        logger.info("Using GitHub REST API clients (direct HTTP)")
        github_client = GitHubRestClient(
            token=config.github_token,
            base_url=config.github_api_url if config.github_api_url else None,
        )
        github_tag_client = GitHubRestTagClient(
            token=config.github_token,
            base_url=config.github_api_url if config.github_api_url else None,
        )
    else:
        # Check if any orchestrations use GitHub triggers
        github_orchestrations = [o for o in orchestrations if o.trigger.source == "github"]
        if github_orchestrations:
            logger.warning(
                f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                "but GitHub is not configured. Set GITHUB_TOKEN to enable GitHub polling."
            )

    # Use factory pattern for agent client creation
    # This enables per-orchestration agent type selection (claude, cursor, etc.)
    agent_factory = create_default_factory(config)
    agent_logger = AgentLogger(base_dir=config.agent_logs_dir)

    logger.info(f"Initialized agent factory with types: {agent_factory.registered_types}")

    # Create and run Sentinel
    # tag_client is now a required positional parameter for clearer API contract
    sentinel = Sentinel(
        config=config,
        orchestrations=orchestrations,
        jira_client=jira_client,
        tag_client=tag_client,
        agent_factory=agent_factory,
        agent_logger=agent_logger,
        github_client=github_client,
        github_tag_client=github_tag_client,
    )

    # Start dashboard server if enabled
    dashboard_server: DashboardServer | None = None
    if config.dashboard_enabled:
        try:
            from sentinel.dashboard import create_app

            logger.info(
                f"Starting dashboard server on {config.dashboard_host}:{config.dashboard_port}"
            )
            dashboard_app = create_app(sentinel)
            dashboard_server = DashboardServer(
                host=config.dashboard_host,
                port=config.dashboard_port,
            )
            dashboard_server.start(dashboard_app)
        except ImportError as e:
            logger.warning(
                f"Dashboard dependencies not available, skipping dashboard: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")

    try:
        if parsed.once:
            logger.info("Running single polling cycle (--once mode)")
            results = sentinel.run_once_and_wait()
            success_count = sum(1 for r in results if r.succeeded)
            logger.info(f"Completed: {success_count}/{len(results)} successful")
            return 0 if success_count == len(results) or not results else 1

        sentinel.run()
        return 0
    finally:
        # Shutdown dashboard server if it was started
        if dashboard_server is not None:
            dashboard_server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
