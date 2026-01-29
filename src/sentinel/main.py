"""Main entry point for the Developer Sentinel orchestrator."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import replace
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
from sentinel.execution_manager import ExecutionManager
from sentinel.executor import AgentClient, AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubClient, GitHubIssueProtocol, GitHubPoller
from sentinel.github_rest_client import GitHubRestClient, GitHubRestTagClient, GitHubTagClient
from sentinel.logging import OrchestrationLogManager, get_logger, setup_logging
from sentinel.orchestration import Orchestration, OrchestrationVersion, load_orchestrations
from sentinel.orchestration_registry import OrchestrationRegistry
from sentinel.poll_coordinator import GitHubIssueWithRepo, PollCoordinator, extract_repo_from_url
from sentinel.poller import JiraClient, JiraIssue, JiraPoller
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient
from sentinel.router import Router, RoutingResult
from sentinel.sdk_clients import ClaudeProcessInterruptedError, JiraSdkClient, JiraSdkTagClient
from sentinel.sdk_clients import request_shutdown as request_claude_shutdown
from sentinel.state_tracker import AttemptCountEntry, QueuedIssueInfo, RunningStepInfo, StateTracker
from sentinel.tag_manager import JiraTagClient, TagManager

logger = get_logger(__name__)


# Re-export dataclasses and functions for backward compatibility
__all__ = [
    "AttemptCountEntry",
    "RunningStepInfo",
    "QueuedIssueInfo",
    "DashboardServer",
    "Sentinel",
    "parse_args",
    "main",
    # Re-exports from poll_coordinator for backward compatibility
    "extract_repo_from_url",
    "GitHubIssueWithRepo",
]


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
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        server = self._server

        def run_server() -> None:
            """Run the uvicorn server."""
            server.run()

        self._thread = threading.Thread(
            target=run_server,
            name="dashboard-server",
            daemon=True,
        )
        self._thread.start()

        # Wait for uvicorn server to be ready
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

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning("Dashboard server thread did not terminate gracefully")

            logger.info("Dashboard server shutdown complete")


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
            tag_client: Jira client for tag operations.
            agent_factory: Factory for creating agent clients per-orchestration.
            agent_logger: Optional logger for agent execution logs.
            github_client: Optional GitHub client for polling GitHub issues/PRs.
            github_tag_client: Optional GitHub client for tag/label operations.
            agent_client: Deprecated keyword argument for backward compatibility.
        """
        self.config = config
        self.orchestrations = orchestrations

        # Initialize pollers
        self.jira_poller = JiraPoller(jira_client)
        self.github_poller = GitHubPoller(github_client) if github_client else None

        # Initialize router
        self.router = Router(orchestrations)

        # Initialize agent factory
        self._agent_logger = agent_logger
        effective_agent: AgentClientFactory | AgentClient | None = agent_factory
        if agent_client is not None:
            effective_agent = agent_client

        if effective_agent is None:
            raise ValueError("Either agent_factory or agent_client must be provided to Sentinel")

        if isinstance(effective_agent, AgentClientFactory):
            self._agent_factory: AgentClientFactory | None = effective_agent
            self._legacy_agent_client: AgentClient | None = None
            default_client = effective_agent.create_for_orchestration(None, config)
            self.executor = AgentExecutor(default_client, agent_logger)
        else:
            self._agent_factory = None
            self._legacy_agent_client = effective_agent
            self.executor = AgentExecutor(effective_agent, agent_logger)

        # Initialize tag manager
        self.tag_manager = TagManager(tag_client, github_client=github_tag_client)

        # Initialize components
        self._state_tracker = StateTracker(
            max_queue_size=config.max_queue_size,
            attempt_counts_ttl=config.attempt_counts_ttl,
        )
        self._execution_manager = ExecutionManager(config.max_concurrent_executions)
        self._orchestration_registry = OrchestrationRegistry(
            config.orchestrations_dir,
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

    # =========================================================================
    # Backward Compatibility (for tests and external access)
    # =========================================================================

    @property
    def _issue_queue(self):
        """Backward compatibility: delegate to state tracker's issue queue."""
        return self._state_tracker._issue_queue

    @property
    def _attempt_counts(self):
        """Backward compatibility: delegate to state tracker's attempt counts."""
        return self._state_tracker._attempt_counts

    @property
    def _attempt_counts_lock(self):
        """Backward compatibility: delegate to state tracker's lock."""
        return self._state_tracker._attempt_counts_lock

    @property
    def _running_steps(self):
        """Backward compatibility: delegate to state tracker's running steps."""
        return self._state_tracker._running_steps

    @property
    def _active_versions(self):
        """Backward compatibility: delegate to orchestration registry's active versions."""
        return self._orchestration_registry._active_versions

    @property
    def _pending_removal_versions(self):
        """Backward compatibility: delegate to orchestration registry's pending removal versions."""
        return self._orchestration_registry._pending_removal_versions

    @property
    def _known_orchestration_files(self):
        """Backward compatibility: delegate to orchestration registry's known files."""
        return self._orchestration_registry._known_orchestration_files

    @property
    def _thread_pool(self):
        """Backward compatibility: delegate to execution manager's thread pool."""
        return self._execution_manager._thread_pool

    @_thread_pool.setter
    def _thread_pool(self, value):
        """Backward compatibility: allow tests to set thread pool."""
        self._execution_manager._thread_pool = value

    @property
    def _active_futures(self):
        """Backward compatibility: delegate to execution manager's active futures."""
        return self._execution_manager._active_futures

    @_active_futures.setter
    def _active_futures(self, value):
        """Backward compatibility: allow tests to set active futures."""
        self._execution_manager._active_futures = value

    @property
    def _per_orch_active_counts(self):
        """Backward compatibility: delegate to state tracker's per-orch counts."""
        return self._state_tracker._per_orch_active_counts

    def _add_to_issue_queue(self, issue_key: str, orchestration_name: str) -> None:
        """Backward compatibility: delegate to state tracker."""
        self._state_tracker.add_to_issue_queue(issue_key, orchestration_name)

    def _clear_issue_queue(self) -> None:
        """Backward compatibility: delegate to state tracker."""
        self._state_tracker.clear_issue_queue()

    def _get_and_increment_attempt_count(self, issue_key: str, orchestration_name: str) -> int:
        """Backward compatibility: delegate to state tracker."""
        return self._state_tracker.get_and_increment_attempt_count(issue_key, orchestration_name)

    def _cleanup_stale_attempt_counts(self) -> int:
        """Backward compatibility: delegate to state tracker."""
        return self._state_tracker.cleanup_stale_attempt_counts()

    def _detect_and_load_orchestration_changes(self) -> tuple[int, int]:
        """Backward compatibility: delegate to orchestration registry."""
        return self._orchestration_registry.detect_and_load_orchestration_changes()

    def _detect_and_unload_removed_files(self) -> int:
        """Backward compatibility: delegate to orchestration registry."""
        return self._orchestration_registry.detect_and_unload_removed_files()

    def _cleanup_pending_removal_versions(self) -> int:
        """Backward compatibility: delegate to orchestration registry."""
        return self._orchestration_registry.cleanup_pending_removal_versions()

    def _increment_per_orch_count(self, orchestration_name: str) -> int:
        """Backward compatibility: delegate to state tracker."""
        return self._state_tracker.increment_per_orch_count(orchestration_name)

    def _decrement_per_orch_count(self, orchestration_name: str) -> int:
        """Backward compatibility: delegate to state tracker."""
        return self._state_tracker.decrement_per_orch_count(orchestration_name)

    def _add_repo_context_from_urls(
        self,
        issues: list,
    ) -> list:
        """Backward compatibility: delegate to poll coordinator."""
        return self._poll_coordinator._add_repo_context_from_urls(issues)

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

    def _get_available_slots_for_orchestration(self, orchestration: Orchestration) -> int:
        """Get available slots for a specific orchestration."""
        global_available = self._get_available_slots()
        return self._state_tracker.get_available_slots_for_orchestration(
            orchestration, global_available
        )

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
            try:
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
            except Exception as tag_error:
                logger.error(f"Failed to apply failure tags: {tag_error}")
            return None
        except Exception as e:
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
                if submitted_pairs is not None:
                    if not self._poll_coordinator.check_and_mark_submitted(
                        submitted_pairs, issue_key, matched_orch.name
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

                except Exception as e:
                    if version is not None:
                        version.decrement_executions()
                    self._state_tracker.decrement_per_orch_count(matched_orch.name)
                    logger.error(f"Failed to submit '{matched_orch.name}' for {issue_key}: {e}")
                    try:
                        self.tag_manager.apply_failure_tags(issue_key, matched_orch)
                    except Exception as tag_error:
                        logger.error(f"Failed to apply failure tags: {tag_error}")

        return submitted_count

    def _collect_completed_results(self) -> list[ExecutionResult]:
        """Collect results from completed futures."""
        # First, collect results from completed futures
        results = self._execution_manager.collect_completed_results()

        # Then, clean up running step metadata for any remaining completed futures
        # Note: collect_completed_results already removes completed futures from _active_futures
        # but we need to handle the metadata cleanup
        def on_future_done(future: Future) -> None:
            self._state_tracker.remove_running_step(id(future))

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
                f"All {self.config.max_concurrent_executions} execution slots busy, "
                "skipping polling this cycle"
            )
            return all_results, 0

        # Group orchestrations by source
        jira_orchestrations, github_orchestrations = (
            self._poll_coordinator.group_orchestrations_by_source(self.orchestrations)
        )

        submitted_count = 0
        submitted_pairs = self._poll_coordinator.create_cycle_dedup_set()

        # Poll Jira triggers
        if jira_orchestrations:
            self._state_tracker.last_jira_poll = datetime.now()
            routing_results, _ = self._poll_coordinator.poll_jira_triggers(
                jira_orchestrations,
                self.router,
                self._shutdown_requested,
                self._log_for_orchestration,
            )
            jira_submitted = self._submit_execution_tasks(
                routing_results, all_results, submitted_pairs
            )
            submitted_count += jira_submitted

        # Poll GitHub triggers
        if github_orchestrations:
            if self.github_poller:
                self._state_tracker.last_github_poll = datetime.now()
                routing_results, _ = self._poll_coordinator.poll_github_triggers(
                    github_orchestrations,
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
                    f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                    "but GitHub client is not configured. Set GITHUB_TOKEN to enable."
                )

        if submitted_count > 0:
            logger.info(f"Submitted {submitted_count} execution tasks")

        return all_results, submitted_count

    def run_once_and_wait(self) -> list[ExecutionResult]:
        """Run a single polling cycle with concurrent execution and wait for completion."""
        logger.info(
            f"Starting single cycle with max {self.config.max_concurrent_executions} "
            "concurrent executions"
        )

        self._execution_manager.start()

        try:
            all_results: list[ExecutionResult] = []

            while True:
                cycle_results, _ = self.run_once()
                all_results.extend(cycle_results)

                pending_futures = self._execution_manager.get_pending_futures()
                if not pending_futures:
                    break

                logger.info(f"Waiting for {len(pending_futures)} execution(s) to complete...")

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
            logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self._shutdown_requested = True
            request_claude_shutdown()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info(
            f"Starting Sentinel with {len(self.orchestrations)} orchestrations, "
            f"polling every {self.config.poll_interval}s, "
            f"max concurrent executions: {self.config.max_concurrent_executions}"
        )

        self._execution_manager.start()

        try:
            while not self._shutdown_requested:
                try:
                    results, submitted_count = self.run_once()
                    success_count = sum(1 for r in results if r.succeeded)
                    if results:
                        logger.info(f"Cycle completed: {success_count}/{len(results)} successful")
                except Exception as e:
                    logger.error(f"Error in polling cycle: {e}")
                    submitted_count = 0

                # Wait for task completion or timeout
                if not self._shutdown_requested:
                    pending_futures = self._execution_manager.get_pending_futures()

                    if pending_futures:
                        logger.debug(
                            f"Waiting for completion of {len(pending_futures)} task(s) before next poll"
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
                                f"{len(done)} task(s) completed, polling immediately for more work"
                            )
                    elif submitted_count == 0:
                        logger.debug(f"Sleeping for {self.config.poll_interval}s")
                        for _ in range(self.config.poll_interval):
                            if self._shutdown_requested:
                                break
                            time.sleep(1)
                    else:
                        logger.debug(
                            f"Submitted {submitted_count} task(s) but all completed quickly, "
                            "polling immediately"
                        )

            # Wait for active tasks
            logger.info("Waiting for active executions to complete...")
            active_count = self._execution_manager.get_active_count()
            if active_count > 0:
                logger.info(f"Waiting for {active_count} active execution(s)...")

            final_results = self._collect_completed_results()
            if final_results:
                success_count = sum(1 for r in final_results if r.succeeded)
                logger.info(f"Final batch: {success_count}/{len(final_results)} successful")

        finally:
            self._execution_manager.shutdown()
            if self._orch_log_manager is not None:
                logger.info("Closing per-orchestration log files...")
                self._orch_log_manager.close_all()

        logger.info("Sentinel shutdown complete")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
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
    """Main entry point."""
    parsed = parse_args(args)

    # Load configuration
    config = load_config(parsed.env_file)

    # Apply CLI overrides
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

    # Initialize GitHub clients
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
        github_orchestrations = [o for o in orchestrations if o.trigger.source == "github"]
        if github_orchestrations:
            logger.warning(
                f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                "but GitHub is not configured. Set GITHUB_TOKEN to enable GitHub polling."
            )

    # Create agent factory
    agent_factory = create_default_factory(config)
    agent_logger = AgentLogger(base_dir=config.agent_logs_dir)

    logger.info(f"Initialized agent factory with types: {agent_factory.registered_types}")

    # Create Sentinel
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

    # Start dashboard
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
            logger.warning(f"Dashboard dependencies not available, skipping dashboard: {e}")
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
        if dashboard_server is not None:
            dashboard_server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
