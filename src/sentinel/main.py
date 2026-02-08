"""Main entry point for the Developer Sentinel orchestrator.

This module contains the Sentinel class, the core orchestrator that coordinates
polling, routing, and execution of orchestrations.

The module structure follows single responsibility:
- cli.py: Command-line argument parsing
- bootstrap.py: Startup and dependency wiring
- app.py: Application runner and lifecycle
- shutdown.py: Graceful termination
- dashboard_server.py: Dashboard management
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import datetime
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentinel.dashboard.state import ExecutionStateSnapshot, OrchestrationVersionSnapshot

from sentinel.agent_clients.claude_sdk import request_shutdown as request_claude_shutdown
from sentinel.agent_clients.factory import AgentClientFactory
from sentinel.agent_logger import AgentLogger

# Import main from app module
from sentinel.app import main

# Import from refactored modules
from sentinel.cli import parse_args
from sentinel.config import Config
from sentinel.execution_manager import ExecutionManager
from sentinel.executor import AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubIssueProtocol, GitHubPoller
from sentinel.github_rest_client import GitHubTagClient
from sentinel.logging import OrchestrationLogManager, get_logger
from sentinel.orchestration import Orchestration, OrchestrationVersion
from sentinel.orchestration_registry import OrchestrationRegistry
from sentinel.poll_coordinator import PollCoordinator
from sentinel.poller import JiraIssue, JiraPoller
from sentinel.router import Router, RoutingResult
from sentinel.sdk_clients import ClaudeProcessInterruptedError
from sentinel.state_tracker import (
    CompletedExecutionInfo,
    QueuedIssueInfo,
    RunningStepInfo,
    StateTracker,
)
from sentinel.tag_manager import JiraTagClient, TagManager
from sentinel.types import ErrorType

logger = get_logger(__name__)


# Public API exports — update this list when adding new exports to this module.
__all__ = [
    "Sentinel",
    "parse_args",
    "main",
]


class Sentinel:
    """Main orchestrator that coordinates polling, routing, and execution.

    This class has been refactored to be a thin orchestrator that composes
    focused components:
    - StateTracker: metrics, queues, attempt counts
    - ExecutionManager: thread pool and futures
    - OrchestrationRegistry: hot-reload and version tracking
    - PollCoordinator: polling cycles for Jira/GitHub
    """

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration],
        tag_client: JiraTagClient,
        agent_factory: AgentClientFactory,
        jira_poller: JiraPoller,
        agent_logger: AgentLogger | None = None,
        router: Router | None = None,
        github_poller: GitHubPoller | None = None,
        github_tag_client: GitHubTagClient | None = None,
    ) -> None:
        """Initialize the Sentinel orchestrator.

        All dependencies can be injected via the DI container.

        Args:
            config: Application configuration.
            orchestrations: List of orchestration configurations.
            tag_client: Jira client for tag operations.
            agent_factory: Factory for creating agent clients per-orchestration.
            jira_poller: Poller for fetching Jira issues.
            agent_logger: Logger for agent execution logs.
            router: Router for matching issues to orchestrations.
            github_poller: Optional poller for GitHub issues/PRs.
            github_tag_client: Optional GitHub client for tag/label operations.
        """
        self.config = config
        self.orchestrations = orchestrations
        self.jira_poller = jira_poller
        self.github_poller = github_poller

        # Initialize or create router
        if router is not None:
            self.router = router
        else:
            # Create router from orchestrations
            self.router = Router(orchestrations)

        # Initialize agent factory
        self._agent_logger = agent_logger
        self._agent_factory = agent_factory
        default_client = agent_factory.create_for_orchestration(None, config)
        self.executor = AgentExecutor(default_client, agent_logger)

        # Initialize tag manager
        self.tag_manager = TagManager(tag_client, github_client=github_tag_client)

        # Initialize components
        self._state_tracker = StateTracker(
            max_queue_size=config.execution.max_queue_size,
            attempt_counts_ttl=config.execution.attempt_counts_ttl,
            max_completed_executions=config.execution.max_recent_executions,
        )
        self._execution_manager = ExecutionManager(config.execution.max_concurrent_executions)
        self._orchestration_registry = OrchestrationRegistry(
            config.execution.orchestrations_dir,
            router_factory=Router,
        )
        self._poll_coordinator = PollCoordinator(
            config,
            jira_poller=self.jira_poller,
            github_poller=self.github_poller,
        )

        # Initialize orchestration registry
        self._orchestration_registry._orchestrations = orchestrations
        self._orchestration_registry._router = self.router
        self._orchestration_registry.init_from_directory()

        # State
        self._shutdown_requested = False

        # Per-orchestration logging manager
        self._orch_log_manager: OrchestrationLogManager | None = None
        if config.execution.orchestration_logs_dir is not None:
            self._orch_log_manager = OrchestrationLogManager(
                config.execution.orchestration_logs_dir
            )
            logger.info(
                "Per-orchestration logging enabled, logs will be written to: %s",
                config.execution.orchestration_logs_dir
            )

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the polling loop."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

    # =========================================================================
    # Public Accessors (for dashboard and external components)
    # =========================================================================

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Get observability metrics for hot-reload operations."""
        return self._orchestration_registry.get_hot_reload_metrics()

    def get_running_steps(self) -> list[RunningStepInfo]:
        """Get information about currently running execution steps."""
        active_futures = self._execution_manager.get_active_futures()
        return self._state_tracker.get_running_steps(active_futures)

    def get_issue_queue(self) -> list[QueuedIssueInfo]:
        """Get information about issues waiting in queue."""
        return self._state_tracker.get_issue_queue()

    def get_start_time(self) -> datetime:
        """Get the process start time."""
        return self._state_tracker.start_time

    def get_last_jira_poll(self) -> datetime | None:
        """Get the last Jira poll time."""
        return self._state_tracker.last_jira_poll

    def get_last_github_poll(self) -> datetime | None:
        """Get the last GitHub poll time."""
        return self._state_tracker.last_github_poll

    def get_active_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Get snapshots of active orchestration versions."""
        from sentinel.dashboard.state import OrchestrationVersionSnapshot

        versions = self._orchestration_registry.get_active_versions()
        return [
            OrchestrationVersionSnapshot(
                name=v.name,
                version_id=v.version_id,
                source_file=str(v.source_file),
                loaded_at=v.loaded_at,
                active_executions=v.active_executions,
            )
            for v in versions
        ]

    def get_pending_removal_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Get snapshots of versions pending removal."""
        from sentinel.dashboard.state import OrchestrationVersionSnapshot

        versions = self._orchestration_registry.get_pending_removal_versions()
        return [
            OrchestrationVersionSnapshot(
                name=v.name,
                version_id=v.version_id,
                source_file=str(v.source_file),
                loaded_at=v.loaded_at,
                active_executions=v.active_executions,
            )
            for v in versions
        ]

    def get_execution_state(self) -> ExecutionStateSnapshot:
        """Get a snapshot of the current execution state."""
        from sentinel.dashboard.state import ExecutionStateSnapshot

        return ExecutionStateSnapshot(active_count=self._execution_manager.get_active_count())

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def get_per_orch_count(self, orchestration_name: str) -> int:
        """Get the active execution count for a specific orchestration."""
        return self._state_tracker.get_per_orch_count(orchestration_name)

    def get_all_per_orch_counts(self) -> dict[str, int]:
        """Get all per-orchestration active execution counts."""
        return self._state_tracker.get_all_per_orch_counts()

    def get_completed_executions(self) -> list[CompletedExecutionInfo]:
        """Get information about recently completed executions.

        Returns:
            List of CompletedExecutionInfo for recent executions, ordered most recent first.
        """
        return self._state_tracker.get_completed_executions()

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _log_for_orchestration(
        self, orchestration_name: str, level: int, message: str, **kwargs: Any
    ) -> None:
        """Log a message to the orchestration-specific log file if configured."""
        logger.log(level, message, extra=kwargs)
        if self._orch_log_manager is not None:
            orch_logger = self._orch_log_manager.get_logger(orchestration_name)
            orch_logger.log(level, message, extra=kwargs)

    def _get_available_slots(self) -> int:
        """Get the number of available execution slots."""
        return self._execution_manager.get_available_slots()

    def _record_completed_execution(
        self,
        result: ExecutionResult,
        running_step: RunningStepInfo,
    ) -> None:
        """Record a completed execution with usage data.

        This method creates a CompletedExecutionInfo from the execution result
        and running step metadata, then adds it to the state tracker for
        dashboard display.

        Args:
            result: The execution result containing status and usage data.
            running_step: The running step info with timing and context.
        """
        # Extract usage data from result
        # Usage data is populated from AgentRunResult.usage in executor.py
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        total_cost_usd = result.total_cost_usd

        completed_info = CompletedExecutionInfo(
            issue_key=result.issue_key,
            orchestration_name=result.orchestration_name,
            attempt_number=running_step.attempt_number,
            started_at=running_step.started_at,
            completed_at=datetime.now(),
            status="success" if result.succeeded else "failure",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost_usd,
            issue_url=running_step.issue_url,
        )
        self._state_tracker.add_completed_execution(completed_info)

    def _get_available_slots_for_orchestration(self, orchestration: Orchestration) -> int:
        """Get available slots for a specific orchestration."""
        global_available = self._get_available_slots()
        return self._state_tracker.get_available_slots_for_orchestration(
            orchestration, global_available
        )

    def _apply_failure_tags_safely(self, issue_key: str, orchestration: Orchestration) -> None:
        """Apply failure tags with standardized exception handling.

        This helper method encapsulates the common pattern of applying failure tags
        while catching and logging any errors that occur during tag application.
        This reduces code duplication across multiple exception handlers.

        Uses the ErrorType enum for consistent error categorization throughout
        the codebase, matching the pattern used in _handle_submission_failure.

        Args:
            issue_key: The issue key (e.g., "PROJ-123") being processed.
            orchestration: The orchestration that failed.
        """
        try:
            self.tag_manager.apply_failure_tags(issue_key, orchestration)
        except (OSError, TimeoutError) as tag_error:
            logger.error(
                "Failed to apply failure tags due to %s: %s",
                ErrorType.IO_ERROR.value,
                tag_error,
                extra={"issue_key": issue_key, "orchestration": orchestration.name},
            )
        except (KeyError, ValueError) as tag_error:
            logger.error(
                "Failed to apply failure tags due to %s: %s",
                ErrorType.DATA_ERROR.value,
                tag_error,
                extra={"issue_key": issue_key, "orchestration": orchestration.name},
            )

    def _handle_execution_failure(
        self,
        issue_key: str,
        orchestration: Orchestration,
        exception: Exception,
        error_type: ErrorType,
    ) -> None:
        """Handle execution failure by logging and applying failure tags.

        This helper method encapsulates the common failure handling logic that was
        previously duplicated across multiple exception handlers. It logs the error
        and attempts to apply failure tags, handling any errors that occur during
        tag application.

        Args:
            issue_key: The issue key (e.g., "PROJ-123") being processed.
            orchestration: The orchestration that failed.
            exception: The exception that caused the failure.
            error_type: The type of error for logging, using the ErrorType enum
                for type safety (e.g., ErrorType.IO_ERROR, ErrorType.RUNTIME_ERROR,
                ErrorType.DATA_ERROR).
        """
        self._log_for_orchestration(
            orchestration.name,
            logging.ERROR,
            f"Failed to execute '{orchestration.name}' for {issue_key} "
            f"due to {error_type.value}: {exception}",
        )
        self._apply_failure_tags_safely(issue_key, orchestration)

    def _handle_submission_failure(
        self,
        issue_key: str,
        orchestration: Orchestration,
        exception: Exception,
        error_type: ErrorType,
        version: OrchestrationVersion | None,
    ) -> None:
        """Handle submission failure by cleaning up state, logging, and applying failure tags.

        This helper method encapsulates the common failure handling logic for submission
        errors in _submit_execution_tasks. It handles:
        1. Decrementing version execution counts
        2. Decrementing per-orchestration counts
        3. Logging the error
        4. Applying failure tags with error handling

        Args:
            issue_key: The issue key (e.g., "PROJ-123") being processed.
            orchestration: The orchestration that failed.
            exception: The exception that caused the failure.
            error_type: The type of error for logging, using the ErrorType enum.
            version: The orchestration version to decrement, or None.
        """
        if version is not None:
            version.decrement_executions()
        self._state_tracker.decrement_per_orch_count(orchestration.name)
        logger.error(
            "Failed to submit '%s' for %s due to %s: %s",
            orchestration.name,
            issue_key,
            error_type.value,
            exception,
            extra={"issue_key": issue_key, "orchestration": orchestration.name},
        )
        self._apply_failure_tags_safely(issue_key, orchestration)

    def _execute_orchestration_task(
        self,
        issue: JiraIssue | GitHubIssueProtocol,
        orchestration: Orchestration,
        version: OrchestrationVersion | None = None,
    ) -> ExecutionResult | None:
        """Execute a single orchestration in a thread."""
        issue_key = issue.key
        try:
            if self._shutdown_requested:
                self._log_for_orchestration(
                    orchestration.name,
                    logging.INFO,
                    f"Shutdown requested, skipping {issue_key}",
                )
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
                return None

            self._log_for_orchestration(
                orchestration.name,
                logging.INFO,
                f"Starting execution of '{orchestration.name}' for {issue_key}",
            )

            # Get per-orchestration agent client
            agent_type = orchestration.agent.agent_type
            client = self._agent_factory.get_or_create_for_orchestration(
                agent_type, self.config
            )
            executor = AgentExecutor(client, self._agent_logger)

            # Use asyncio.run() at the thread entry point to drive async execution
            # This is the top-level entry point for async code in each thread,
            # avoiding nested event loops (DS-509)
            result = asyncio.run(executor.execute(issue, orchestration))
            self.tag_manager.update_tags(result, orchestration)

            status = "succeeded" if result.succeeded else "failed"
            self._log_for_orchestration(
                orchestration.name,
                logging.INFO if result.succeeded else logging.WARNING,
                f"Execution of '{orchestration.name}' for {issue_key} {status}",
            )

            return result

        except ClaudeProcessInterruptedError:
            self._log_for_orchestration(
                orchestration.name,
                logging.INFO,
                f"Claude process interrupted for {issue_key}",
            )
            self._apply_failure_tags_safely(issue_key, orchestration)
            return None
        except (OSError, TimeoutError) as e:
            self._handle_execution_failure(issue_key, orchestration, e, ErrorType.IO_ERROR)
            return None
        except RuntimeError as e:
            self._handle_execution_failure(issue_key, orchestration, e, ErrorType.RUNTIME_ERROR)
            return None
        except (KeyError, ValueError) as e:
            self._handle_execution_failure(issue_key, orchestration, e, ErrorType.DATA_ERROR)
            return None
        finally:
            if version is not None:
                version.decrement_executions()
            self._state_tracker.decrement_per_orch_count(orchestration.name)

    def _submit_execution_tasks(
        self,
        routing_results: list[RoutingResult],
        all_results: list[ExecutionResult],
        submitted_pairs: set[tuple[str, str]] | None = None,
    ) -> int:
        """Submit execution tasks for routed issues."""
        submitted_count = 0

        for routing_result in routing_results:
            for matched_orch in routing_result.orchestrations:
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping execution")
                    return submitted_count

                issue_key = routing_result.issue.key

                # Check deduplication
                if (
                    submitted_pairs is not None
                    and not self._poll_coordinator.check_and_mark_submitted(
                        submitted_pairs, issue_key, matched_orch.name
                    )
                ):
                    continue

                # Check slot availability
                if self._get_available_slots_for_orchestration(matched_orch) <= 0:
                    self._log_for_orchestration(
                        matched_orch.name,
                        logging.DEBUG,
                        f"No available slots for '{matched_orch.name}', queuing {issue_key}",
                    )
                    self._state_tracker.add_to_issue_queue(issue_key, matched_orch.name)
                    continue

                self._log_for_orchestration(
                    matched_orch.name,
                    logging.INFO,
                    f"Submitting '{matched_orch.name}' for {issue_key}",
                )

                # Track execution
                version = self._orchestration_registry.get_version_for_orchestration(matched_orch)
                if version is not None:
                    version.increment_executions()

                self._state_tracker.increment_per_orch_count(matched_orch.name)

                try:
                    self.tag_manager.start_processing(issue_key, matched_orch)

                    # Submit to thread pool
                    if self._execution_manager.is_running():
                        future = self._execution_manager.submit(
                            self._execute_orchestration_task,
                            routing_result.issue,
                            matched_orch,
                            version,
                        )
                        if future is not None:
                            # Track running step metadata
                            attempt_number = self._state_tracker.get_and_increment_attempt_count(
                                issue_key, matched_orch.name
                            )
                            issue_url = self._poll_coordinator.construct_issue_url(
                                routing_result.issue, matched_orch
                            )
                            self._state_tracker.add_running_step(
                                id(future),
                                issue_key,
                                matched_orch.name,
                                attempt_number,
                                issue_url,
                            )
                            submitted_count += 1
                    else:
                        # Fallback: synchronous execution
                        result = self._execute_orchestration_task(
                            routing_result.issue,
                            matched_orch,
                            version,
                        )
                        if result is not None:
                            all_results.append(result)

                except (OSError, TimeoutError) as e:
                    self._handle_submission_failure(
                        issue_key, matched_orch, e, ErrorType.IO_ERROR, version
                    )
                except RuntimeError as e:
                    self._handle_submission_failure(
                        issue_key, matched_orch, e, ErrorType.RUNTIME_ERROR, version
                    )
                except (KeyError, ValueError) as e:
                    self._handle_submission_failure(
                        issue_key, matched_orch, e, ErrorType.DATA_ERROR, version
                    )

        return submitted_count

    def _collect_completed_results(self) -> list[ExecutionResult]:
        """Collect results from completed futures and record completed executions."""
        # First, collect results from completed futures
        results = self._execution_manager.collect_completed_results()

        # Build a map of (issue_key, orchestration_name) -> result for recording
        # completed executions. We need to match results with their running step metadata.
        result_map: dict[tuple[str, str], ExecutionResult] = {}
        for result in results:
            key = (result.issue_key, result.orchestration_name)
            result_map[key] = result

        # Then, clean up running step metadata for any remaining completed futures
        # Note: collect_completed_results already removes completed futures from _active_futures
        # but we need to handle the metadata cleanup and record completed executions
        def on_future_done(future: Future[Any]) -> None:
            running_step = self._state_tracker.remove_running_step(id(future))
            if running_step is not None:
                # Try to find the matching result to record the completed execution
                key = (running_step.issue_key, running_step.orchestration_name)
                result = result_map.get(key)
                if result is not None:
                    self._record_completed_execution(result, running_step)

        self._execution_manager.cleanup_completed_futures(on_future_done)

        return results

    # =========================================================================
    # Main Run Methods
    # =========================================================================

    def run_once(self) -> tuple[list[ExecutionResult], int]:
        """Run a single polling cycle."""
        # Clear state for new cycle
        self._state_tracker.clear_issue_queue()

        # Hot-reload orchestrations
        new_count, modified_count = (
            self._orchestration_registry.detect_and_load_orchestration_changes()
        )
        if new_count > 0 or modified_count > 0:
            self.orchestrations = self._orchestration_registry.orchestrations
            self.router = self._orchestration_registry.router

        unloaded_count = self._orchestration_registry.detect_and_unload_removed_files()
        if unloaded_count > 0:
            self.orchestrations = self._orchestration_registry.orchestrations
            self.router = self._orchestration_registry.router

        self._orchestration_registry.cleanup_pending_removal_versions()
        self._state_tracker.cleanup_stale_attempt_counts()

        # Collect completed results
        all_results = self._collect_completed_results()

        # Check slot availability
        available_slots = self._get_available_slots()
        if available_slots <= 0:
            logger.debug(
                "All %s execution slots busy, skipping polling this cycle",
                self.config.execution.max_concurrent_executions
            )
            return all_results, 0

        # Group orchestrations by source (DS-750: NamedTuple for self-documenting access)
        grouped = self._poll_coordinator.group_orchestrations_by_source(self.orchestrations)

        submitted_count = 0
        submitted_pairs = self._poll_coordinator.create_cycle_dedup_set()

        # Poll Jira triggers
        if grouped.jira:
            self._state_tracker.last_jira_poll = datetime.now()
            routing_results, _ = self._poll_coordinator.poll_jira_triggers(
                grouped.jira,
                self.router,
                self._shutdown_requested,
                self._log_for_orchestration,
            )
            jira_submitted = self._submit_execution_tasks(
                routing_results, all_results, submitted_pairs
            )
            submitted_count += jira_submitted

        # Poll GitHub triggers
        if grouped.github:
            if self.github_poller:
                self._state_tracker.last_github_poll = datetime.now()
                routing_results, _ = self._poll_coordinator.poll_github_triggers(
                    grouped.github,
                    self.router,
                    self._shutdown_requested,
                    self._log_for_orchestration,
                )
                github_submitted = self._submit_execution_tasks(
                    routing_results, all_results, submitted_pairs
                )
                submitted_count += github_submitted
            else:
                logger.warning(
                    "Found %s GitHub-triggered orchestrations but GitHub client is not "
                    "configured. Set GITHUB_TOKEN to enable.",
                    len(grouped.github)
                )

        if submitted_count > 0:
            logger.info("Submitted %s execution tasks", submitted_count)

        return all_results, submitted_count

    def run_once_and_wait(self) -> list[ExecutionResult]:
        """Run a single polling cycle with concurrent execution and wait for completion."""
        logger.info(
            "Starting single cycle with max %s concurrent executions",
            self.config.execution.max_concurrent_executions
        )

        self._execution_manager.start()

        try:
            all_results: list[ExecutionResult] = []

            while True:
                cycle_results, submitted_count = self.run_once()
                all_results.extend(cycle_results)

                pending_futures = self._execution_manager.get_pending_futures()
                if not pending_futures:
                    # If we submitted work this cycle but no futures are pending,
                    # tasks completed very quickly. Collect their results before exiting.
                    # This fixes a race condition where fast-completing tasks on Python 3.13
                    # could finish before we check pending_futures (DS-542).
                    if submitted_count > 0:
                        logger.debug(
                            "Tasks completed before pending_futures check, collecting %s "
                            "submitted results",
                            submitted_count,
                        )
                        final_results = self._collect_completed_results()
                        all_results.extend(final_results)
                    break

                logger.info("Waiting for %s execution(s) to complete...", len(pending_futures))

                # Wait for all to complete
                while True:
                    pending = self._execution_manager.get_pending_futures()
                    if not pending:
                        break
                    time.sleep(0.1)

                completed_results = self._collect_completed_results()
                all_results.extend(completed_results)

            return all_results

        finally:
            self._execution_manager.shutdown()
            if self._orch_log_manager is not None:
                self._orch_log_manager.close_all()

    def run(self) -> None:
        """Run the main polling loop until shutdown is requested."""

        def handle_shutdown(signum: int, frame: FrameType | None) -> None:
            signal_name = signal.Signals(signum).name
            logger.info("Received %s, initiating graceful shutdown...", signal_name)
            self._shutdown_requested = True
            request_claude_shutdown()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info(
            "Starting Sentinel with %s orchestrations, polling every %ss, "
            "max concurrent executions: %s",
            len(self.orchestrations),
            self.config.polling.interval,
            self.config.execution.max_concurrent_executions
        )

        self._execution_manager.start()

        try:
            while not self._shutdown_requested:
                try:
                    results, submitted_count = self.run_once()
                    success_count = sum(1 for r in results if r.succeeded)
                    if results:
                        logger.info(
                            "Cycle completed: %s/%s successful", success_count, len(results)
                        )
                except (OSError, TimeoutError) as e:
                    logger.error(
                        "Error in polling cycle due to I/O or timeout: %s",
                        e,
                        extra={"error_type": type(e).__name__},
                    )
                    submitted_count = 0
                except RuntimeError as e:
                    logger.error(
                        "Error in polling cycle due to runtime error: %s",
                        e,
                        extra={"error_type": type(e).__name__},
                    )
                    submitted_count = 0
                except (KeyError, ValueError) as e:
                    logger.error(
                        "Error in polling cycle due to data error: %s",
                        e,
                        extra={"error_type": type(e).__name__},
                    )
                    submitted_count = 0

                # Wait for task completion or timeout
                if not self._shutdown_requested:
                    pending_futures = self._execution_manager.get_pending_futures()

                    if pending_futures:
                        logger.debug(
                            "Waiting for %s task(s) to complete",
                            len(pending_futures)
                        )
                        done, _ = wait(pending_futures, timeout=1.0, return_when=FIRST_COMPLETED)
                        while not done and not self._shutdown_requested:
                            pending_futures = self._execution_manager.get_pending_futures()
                            if not pending_futures:
                                break
                            done, _ = wait(
                                pending_futures, timeout=1.0, return_when=FIRST_COMPLETED
                            )

                        if done:
                            logger.debug(
                                "%s task(s) completed, polling immediately for more work",
                                len(done)
                            )
                    elif submitted_count == 0:
                        logger.debug("Sleeping for %ss", self.config.polling.interval)
                        for _ in range(self.config.polling.interval):
                            if self._shutdown_requested:
                                break
                            time.sleep(1)
                    else:
                        logger.debug(
                            "Submitted %s task(s) but all completed quickly, polling immediately",
                            submitted_count
                        )

            # Wait for active tasks with configurable timeout
            active_count = self._execution_manager.get_active_count()
            if active_count > 0:
                timeout = self.config.execution.shutdown_timeout_seconds
                if timeout > 0:
                    logger.info(
                        "Waiting for %s active execution(s) to complete "
                        "(timeout: %.1fs)...",
                        active_count,
                        timeout,
                    )
                    # Wait with timeout for graceful completion
                    pending_futures = self._execution_manager.get_pending_futures()
                    if pending_futures:
                        done, not_done = wait(pending_futures, timeout=timeout)
                        if not_done:
                            logger.warning(
                                "Shutdown timeout reached after %.1fs. "
                                "Forcefully terminating %s remaining execution(s)...",
                                timeout,
                                len(not_done),
                            )
                            # Cancel remaining futures
                            for future in not_done:
                                future.cancel()
                else:
                    logger.info(
                        "Waiting for %s active execution(s) to complete "
                        "(no timeout configured)...",
                        active_count,
                    )

            final_results = self._collect_completed_results()
            if final_results:
                success_count = sum(1 for r in final_results if r.succeeded)
                logger.info("Final batch: %s/%s successful", success_count, len(final_results))

        finally:
            self._execution_manager.shutdown()
            if self._orch_log_manager is not None:
                logger.info("Closing per-orchestration log files...")
                self._orch_log_manager.close_all()

        logger.info("Sentinel shutdown complete")


if __name__ == "__main__":
    import sys
    sys.exit(main())
