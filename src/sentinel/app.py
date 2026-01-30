"""Core application runner for Developer Sentinel.

This module provides the main application runner that coordinates:
- Dashboard server lifecycle
- Sentinel polling loop
- Single-run mode execution

It acts as the orchestration layer between bootstrap, Sentinel, and shutdown
components.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from sentinel.bootstrap import BootstrapContext, bootstrap, create_sentinel_from_context
from sentinel.cli import parse_args
from sentinel.dashboard_server import DashboardServer
from sentinel.logging import get_logger

if TYPE_CHECKING:
    from sentinel.main import Sentinel

logger = get_logger(__name__)


def start_dashboard(context: BootstrapContext, sentinel: Sentinel) -> DashboardServer | None:
    """Start the dashboard server if enabled.

    Args:
        context: Bootstrap context with configuration.
        sentinel: Sentinel instance for dashboard state.

    Returns:
        DashboardServer if started successfully, None otherwise.
    """
    config = context.config
    if not config.dashboard_enabled:
        return None

    try:
        from sentinel.dashboard import create_app

        logger.info(
            "Starting dashboard server on %s:%s", config.dashboard_host, config.dashboard_port
        )
        dashboard_app = create_app(sentinel)
        dashboard_server = DashboardServer(
            host=config.dashboard_host,
            port=config.dashboard_port,
        )
        dashboard_server.start(dashboard_app)
        return dashboard_server
    except ImportError as e:
        logger.warning("Dashboard dependencies not available, skipping dashboard: %s", e)
        return None
    except OSError as e:
        logger.error(
            "Failed to start dashboard server due to network/OS error: %s", e,
            extra={"host": config.dashboard_host, "port": config.dashboard_port},
        )
        return None
    except RuntimeError as e:
        logger.error(
            "Failed to start dashboard server due to runtime error: %s", e,
            extra={"host": config.dashboard_host, "port": config.dashboard_port},
        )
        return None


def run_once_mode(sentinel: Sentinel) -> int:
    """Run Sentinel in single-cycle mode.

    Args:
        sentinel: Configured Sentinel instance.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    logger.info("Running single polling cycle (--once mode)")
    results = sentinel.run_once_and_wait()
    success_count = sum(1 for r in results if r.succeeded)
    logger.info("Completed: %s/%s successful", success_count, len(results))
    return 0 if success_count == len(results) or not results else 1


def run_continuous_mode(sentinel: Sentinel) -> int:
    """Run Sentinel in continuous polling mode.

    Args:
        sentinel: Configured Sentinel instance.

    Returns:
        Exit code: 0 for success.
    """
    sentinel.run()
    return 0


def run_application(
    parsed: argparse.Namespace,
    context: BootstrapContext,
) -> int:
    """Run the main application with the given context.

    This function coordinates:
    1. Creating the Sentinel instance
    2. Starting the dashboard (if enabled)
    3. Running the appropriate mode (once or continuous)
    4. Cleanup on exit

    Args:
        parsed: Parsed command-line arguments.
        context: Bootstrap context with all dependencies.

    Returns:
        Exit code for the application.
    """
    # Create Sentinel instance
    sentinel = create_sentinel_from_context(context)

    # Start dashboard
    dashboard_server = start_dashboard(context, sentinel)

    try:
        if parsed.once:
            return run_once_mode(sentinel)
        else:
            return run_continuous_mode(sentinel)
    finally:
        if dashboard_server is not None:
            dashboard_server.shutdown()


def main(args: list[str] | None = None) -> int:
    """Main entry point for the application.

    This is the primary entry point that:
    1. Parses command-line arguments
    2. Bootstraps dependencies
    3. Runs the application

    Args:
        args: Optional list of command-line arguments.

    Returns:
        Exit code for the application.
    """
    parsed = parse_args(args)

    # Bootstrap dependencies
    context = bootstrap(parsed)
    if context is None:
        # Bootstrap failed (logged internally)
        return 1 if parsed.once else 0

    return run_application(parsed, context)


__all__ = [
    "main",
    "run_application",
    "run_continuous_mode",
    "run_once_mode",
    "start_dashboard",
]
