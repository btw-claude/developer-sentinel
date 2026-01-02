"""Main entry point for the Developer Sentinel orchestrator."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

from sentinel.agent_logger import AgentLogger
from sentinel.config import Config, load_config
from sentinel.executor import AgentClient, AgentExecutor, ExecutionResult
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
from sentinel.poller import JiraClient, JiraPoller
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
    ) -> None:
        """Initialize the Sentinel orchestrator.

        Args:
            config: Application configuration.
            orchestrations: List of orchestration configurations.
            jira_client: Jira client for polling issues.
            agent_client: Agent client for executing agents.
            tag_client: Jira client for tag operations.
            agent_logger: Optional logger for agent execution logs.
        """
        self.config = config
        self.orchestrations = orchestrations
        self.poller = JiraPoller(jira_client)
        self.router = Router(orchestrations)
        self.executor = AgentExecutor(agent_client, agent_logger)
        self.tag_manager = TagManager(tag_client)
        self._shutdown_requested = False

    def request_shutdown(self) -> None:
        """Request graceful shutdown of the polling loop."""
        logger.info("Shutdown requested")
        self._shutdown_requested = True

    def run_once(self) -> list[ExecutionResult]:
        """Run a single polling cycle.

        Returns:
            List of execution results from this cycle.
        """
        all_results: list[ExecutionResult] = []

        # Collect unique trigger configs to avoid duplicate polling
        seen_triggers: set[str] = set()
        triggers_to_poll: list[tuple[Orchestration, Any]] = []

        for orch in self.orchestrations:
            trigger_key = f"{orch.trigger.project}:{','.join(orch.trigger.tags)}"
            if trigger_key not in seen_triggers:
                seen_triggers.add(trigger_key)
                triggers_to_poll.append((orch, orch.trigger))

        # Poll for each unique trigger
        for orch, trigger in triggers_to_poll:
            logger.info(f"Polling for orchestration '{orch.name}'")
            try:
                issues = self.poller.poll(trigger, self.config.max_issues_per_poll)
                logger.info(f"Found {len(issues)} issues for '{orch.name}'")
            except Exception as e:
                logger.error(f"Failed to poll for '{orch.name}': {e}")
                continue

            # Route issues to matching orchestrations
            routing_results = self.router.route_matched_only(issues)

            # Execute agents for matched issues
            for routing_result in routing_results:
                for matched_orch in routing_result.orchestrations:
                    if self._shutdown_requested:
                        logger.info("Shutdown requested, stopping execution")
                        return all_results

                    issue_key = routing_result.issue.key
                    logger.info(f"Executing '{matched_orch.name}' for {issue_key}")

                    try:
                        # Mark issue as being processed (remove trigger tags, add in-progress tag)
                        self.tag_manager.start_processing(issue_key, matched_orch)

                        result = self.executor.execute(routing_result.issue, matched_orch)
                        all_results.append(result)

                        # Handle post-processing tag updates
                        self.tag_manager.update_tags(result, matched_orch)

                    except ClaudeProcessInterruptedError:
                        logger.info("Claude process interrupted by shutdown request")
                        # Apply failure tags for interrupted process
                        try:
                            self.tag_manager.apply_failure_tags(issue_key, matched_orch)
                        except Exception as tag_error:
                            logger.error(f"Failed to apply failure tags: {tag_error}")
                        return all_results
                    except Exception as e:
                        logger.error(
                            f"Failed to execute '{matched_orch.name}' for {issue_key}: {e}"
                        )

        return all_results

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

        logger.info(
            f"Starting Sentinel with {len(self.orchestrations)} orchestrations, "
            f"polling every {self.config.poll_interval}s"
        )

        while not self._shutdown_requested:
            try:
                results = self.run_once()
                success_count = sum(1 for r in results if r.succeeded)
                if results:
                    logger.info(f"Cycle completed: {success_count}/{len(results)} successful")
            except Exception as e:
                logger.error(f"Error in polling cycle: {e}")

            if not self._shutdown_requested:
                logger.debug(f"Sleeping for {self.config.poll_interval}s")
                # Sleep in small increments to allow quick shutdown
                for _ in range(self.config.poll_interval):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)

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

    # Initialize MCP-based clients
    jira_client = JiraMcpClient()
    agent_client = ClaudeMcpAgentClient(base_workdir=config.agent_workdir)
    tag_client = JiraMcpTagClient()
    agent_logger = AgentLogger(base_dir=config.agent_logs_dir)

    # Create and run Sentinel
    sentinel = Sentinel(
        config=config,
        orchestrations=orchestrations,
        jira_client=jira_client,
        agent_client=agent_client,
        tag_client=tag_client,
        agent_logger=agent_logger,
    )

    if parsed.once:
        logger.info("Running single polling cycle (--once mode)")
        results = sentinel.run_once()
        success_count = sum(1 for r in results if r.succeeded)
        logger.info(f"Completed: {success_count}/{len(results)} successful")
        return 0 if success_count == len(results) or not results else 1

    sentinel.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
