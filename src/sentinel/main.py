"""Main entry point for the Developer Sentinel orchestrator."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

from sentinel.agent_logger import AgentLogger
from sentinel.config import Config, load_config
from sentinel.executor import AgentClient, AgentExecutor, ExecutionResult
from sentinel.github_poller import GitHubClient, GitHubIssue, GitHubPoller
from sentinel.github_rest_client import GitHubRestClient, GitHubRestTagClient, GitHubTagClient
from sentinel.logging import get_logger, setup_logging
from sentinel.mcp_clients import (
    ClaudeMcpAgentClient,
    ClaudeProcessInterruptedError,
    JiraMcpClient,
    JiraMcpTagClient,
)
from sentinel.mcp_clients import (
    request_shutdown as request_claude_shutdown,
)
from sentinel.orchestration import Orchestration, load_orchestrations
from sentinel.poller import JiraClient, JiraIssue, JiraPoller
from sentinel.rest_clients import JiraRestClient, JiraRestTagClient
from sentinel.router import Router
from sentinel.tag_manager import JiraTagClient, TagManager

logger = get_logger(__name__)


class Sentinel:
    """Main orchestrator that coordinates polling, routing, and execution."""

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration],
        jira_client: JiraClient,
        agent_client: AgentClient,
        tag_client: JiraTagClient,
        agent_logger: AgentLogger | None = None,
        github_client: GitHubClient | None = None,
        github_tag_client: GitHubTagClient | None = None,
    ) -> None:
        """Initialize the Sentinel orchestrator.

        Args:
            config: Application configuration.
            orchestrations: List of orchestration configurations.
            jira_client: Jira client for polling issues.
            agent_client: Agent client for executing agents.
            tag_client: Jira client for tag operations.
            agent_logger: Optional logger for agent execution logs.
            github_client: Optional GitHub client for polling GitHub issues/PRs.
            github_tag_client: Optional GitHub client for tag/label operations.
        """
        self.config = config
        self.orchestrations = orchestrations
        self.jira_poller = JiraPoller(jira_client)
        self.github_poller = GitHubPoller(github_client) if github_client else None
        self.router = Router(orchestrations)
        self.executor = AgentExecutor(agent_client, agent_logger)
        self.tag_manager = TagManager(tag_client, github_client=github_tag_client)
        self._shutdown_requested = False

        # Thread pool for concurrent execution
        self._thread_pool: ThreadPoolExecutor | None = None
        self._active_futures: list[Future[ExecutionResult]] = []
        self._futures_lock = threading.Lock()

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the polling loop."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

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
        issue: Any,
        orchestration: Orchestration,
    ) -> ExecutionResult | None:
        """Execute a single orchestration in a thread.

        This is the task that runs in the thread pool.

        Args:
            issue: The Jira issue to process.
            orchestration: The orchestration to execute.

        Returns:
            ExecutionResult if execution completed, None if interrupted.
        """
        issue_key = issue.key
        try:
            if self._shutdown_requested:
                logger.info(f"Shutdown requested, skipping {issue_key}")
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
                return None

            result = self.executor.execute(issue, orchestration)
            self.tag_manager.update_tags(result, orchestration)
            return result

        except ClaudeProcessInterruptedError:
            logger.info(f"Claude process interrupted for {issue_key}")
            try:
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
            except Exception as tag_error:
                logger.error(f"Failed to apply failure tags: {tag_error}")
            return None
        except Exception as e:
            logger.error(f"Failed to execute '{orchestration.name}' for {issue_key}: {e}")
            try:
                self.tag_manager.apply_failure_tags(issue_key, orchestration)
            except Exception as tag_error:
                logger.error(f"Failed to apply failure tags: {tag_error}")
            return None

    def run_once(self) -> list[ExecutionResult]:
        """Run a single polling cycle.

        Returns:
            List of execution results from this cycle.
        """
        # Collect any completed results from previous cycle
        all_results = self._collect_completed_results()

        # Check how many slots are available
        available_slots = self._get_available_slots()
        if available_slots <= 0:
            logger.debug(
                f"All {self.config.max_concurrent_executions} execution slots busy, "
                "skipping polling this cycle"
            )
            return all_results

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

        # Poll Jira triggers
        if jira_orchestrations:
            jira_submitted = self._poll_jira_triggers(jira_orchestrations, all_results)
            submitted_count += jira_submitted

        # Poll GitHub triggers (if GitHub client is configured)
        if github_orchestrations:
            if self.github_poller:
                github_submitted = self._poll_github_triggers(github_orchestrations, all_results)
                submitted_count += github_submitted
            else:
                logger.warning(
                    f"Found {len(github_orchestrations)} GitHub-triggered orchestrations "
                    "but GitHub client is not configured. Set GITHUB_TOKEN to enable."
                )

        if submitted_count > 0:
            logger.info(f"Submitted {submitted_count} execution tasks")

        return all_results

    def _poll_jira_triggers(
        self,
        orchestrations: list[Orchestration],
        all_results: list[ExecutionResult],
    ) -> int:
        """Poll Jira for issues matching orchestration triggers.

        Args:
            orchestrations: List of Jira-triggered orchestrations.
            all_results: List to append synchronous results to.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, Any]] = []

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

            logger.info(f"Polling Jira for orchestration '{orch.name}'")
            try:
                issues = self.jira_poller.poll(trigger, self.config.max_issues_per_poll)
                logger.info(f"Found {len(issues)} Jira issues for '{orch.name}'")
            except Exception as e:
                logger.error(f"Failed to poll Jira for '{orch.name}': {e}")
                continue

            # Route issues to matching orchestrations
            routing_results = self.router.route_matched_only(issues)

            # Submit execution tasks
            submitted = self._submit_execution_tasks(routing_results, all_results)
            submitted_count += submitted

        return submitted_count

    def _poll_github_triggers(
        self,
        orchestrations: list[Orchestration],
        all_results: list[ExecutionResult],
    ) -> int:
        """Poll GitHub for issues/PRs matching orchestration triggers.

        Args:
            orchestrations: List of GitHub-triggered orchestrations.
            all_results: List to append synchronous results to.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        if not self.github_poller:
            return 0

        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, Any]] = []

        for orch in orchestrations:
            trigger_key = f"github:{orch.trigger.repo}:{','.join(orch.trigger.tags)}"
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        submitted_count = 0

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping polling")
                return submitted_count

            logger.info(f"Polling GitHub for orchestration '{orch.name}'")
            try:
                issues = self.github_poller.poll(trigger, self.config.max_issues_per_poll)
                logger.info(f"Found {len(issues)} GitHub issues/PRs for '{orch.name}'")
            except Exception as e:
                logger.error(f"Failed to poll GitHub for '{orch.name}': {e}")
                continue

            # Convert GitHub issues to include repo context for tag operations
            # The GitHubIssue.key property returns "#123" format, but we need "org/repo#123"
            # for tag operations to work correctly
            issues_with_context = self._add_repo_context_to_github_issues(issues, trigger.repo)

            # Route issues to matching orchestrations
            routing_results = self.router.route_matched_only(issues_with_context)

            # Submit execution tasks
            submitted = self._submit_execution_tasks(routing_results, all_results)
            submitted_count += submitted

        return submitted_count

    def _add_repo_context_to_github_issues(
        self,
        issues: list[GitHubIssue],
        repo: str,
    ) -> list[GitHubIssue]:
        """Add repository context to GitHub issues for proper key formatting.

        The GitHubIssue.key property returns "#123" but tag operations need "org/repo#123".
        This creates new GitHubIssue objects with the full key.

        Args:
            issues: List of GitHubIssue objects from the poller.
            repo: Repository in "org/repo" format.

        Returns:
            List of GitHubIssue objects with updated keys.
        """
        from dataclasses import replace

        result = []
        for issue in issues:
            # Create a wrapper that provides the full key
            # We can't modify the dataclass, so we create a simple wrapper
            class GitHubIssueWithRepo:
                def __init__(self, issue: GitHubIssue, repo: str):
                    self._issue = issue
                    self._repo = repo

                @property
                def key(self) -> str:
                    return f"{self._repo}#{self._issue.number}"

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._issue, name)

            result.append(GitHubIssueWithRepo(issue, repo))  # type: ignore
        return result  # type: ignore

    def _submit_execution_tasks(
        self,
        routing_results: list[Any],
        all_results: list[ExecutionResult],
    ) -> int:
        """Submit execution tasks for routed issues.

        Args:
            routing_results: List of routing results from the router.
            all_results: List to append synchronous results to.

        Returns:
            Number of tasks submitted to the thread pool.
        """
        submitted_count = 0

        for routing_result in routing_results:
            for matched_orch in routing_result.orchestrations:
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping execution")
                    return submitted_count

                # Check if we have available slots
                if self._get_available_slots() <= 0:
                    logger.debug("No available slots, will process in next cycle")
                    return submitted_count

                issue_key = routing_result.issue.key
                logger.info(f"Submitting '{matched_orch.name}' for {issue_key}")

                try:
                    # Mark issue as being processed (remove trigger tags, add in-progress tag)
                    self.tag_manager.start_processing(issue_key, matched_orch)

                    # Submit to thread pool
                    if self._thread_pool is not None:
                        future = self._thread_pool.submit(
                            self._execute_orchestration_task,
                            routing_result.issue,
                            matched_orch,
                        )
                        with self._futures_lock:
                            self._active_futures.append(future)
                        submitted_count += 1
                    else:
                        # Fallback: synchronous execution (e.g., run_once without run())
                        result = self._execute_orchestration_task(
                            routing_result.issue,
                            matched_orch,
                        )
                        if result is not None:
                            all_results.append(result)

                except Exception as e:
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
                cycle_results = self.run_once()
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
                    results = self.run_once()
                    success_count = sum(1 for r in results if r.succeeded)
                    if results:
                        logger.info(
                            f"Cycle completed: {success_count}/{len(results)} successful"
                        )
                except Exception as e:
                    logger.error(f"Error in polling cycle: {e}")

                if not self._shutdown_requested:
                    logger.debug(f"Sleeping for {self.config.poll_interval}s")
                    # Sleep in small increments to allow quick shutdown
                    for _ in range(self.config.poll_interval):
                        if self._shutdown_requested:
                            break
                        time.sleep(1)

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
    # Fall back to MCP clients if Jira REST is not configured
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
        logger.info("Using Jira MCP clients (via Claude Agent SDK)")
        logger.warning(
            "Jira REST API not configured. Set JIRA_BASE_URL, JIRA_EMAIL, "
            "and JIRA_API_TOKEN for faster polling."
        )
        jira_client = JiraMcpClient(config)
        tag_client = JiraMcpTagClient(config)

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

    # Agent client always uses Claude Agent SDK (that's the whole point)
    # Pass log_base_dir to enable streaming logs during agent execution
    agent_client = ClaudeMcpAgentClient(
        config=config,
        base_workdir=config.agent_workdir,
        log_base_dir=config.agent_logs_dir,
    )
    agent_logger = AgentLogger(base_dir=config.agent_logs_dir)

    # Create and run Sentinel
    sentinel = Sentinel(
        config=config,
        orchestrations=orchestrations,
        jira_client=jira_client,
        agent_client=agent_client,
        tag_client=tag_client,
        agent_logger=agent_logger,
        github_client=github_client,
        github_tag_client=github_tag_client,
    )

    if parsed.once:
        logger.info("Running single polling cycle (--once mode)")
        results = sentinel.run_once_and_wait()
        success_count = sum(1 for r in results if r.succeeded)
        logger.info(f"Completed: {success_count}/{len(results)} successful")
        return 0 if success_count == len(results) or not results else 1

    sentinel.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
