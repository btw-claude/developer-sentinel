"""Core application runner for Developer Sentinel.

This module provides the main application runner that coordinates:
- Dashboard server lifecycle
- Sentinel polling loop
- Single-run mode execution

It acts as the orchestration layer between bootstrap, Sentinel, and shutdown
components.

Dashboard-less Operation Mode:
    The Sentinel application can operate without the dashboard if it fails to
    start. When dashboard startup fails (due to import errors, network issues,
    runtime errors, or other exceptions), the application logs a warning and
    continues operating normally without dashboard functionality.

    This graceful degradation ensures that:
    - The core polling and execution functionality remains operational
    - External monitoring can detect dashboard unavailability via metrics
    - The application does not crash due to optional component failures

    To check dashboard status, use the DashboardServer.is_running property
    or monitor the dashboard_available flag returned by start_dashboard().
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

    This function implements graceful degradation for dashboard startup failures.
    If the dashboard fails to start for any reason, the Sentinel application
    continues to operate without dashboard functionality.

    Args:
        context: Bootstrap context with configuration.
        sentinel: Sentinel instance for dashboard state.

    Returns:
        DashboardServer if started successfully, None otherwise.
        When None is returned, Sentinel continues operating without the dashboard.

    Note:
        All exceptions during dashboard startup are caught and logged as warnings.
        This ensures the main Sentinel polling loop is never blocked by dashboard
        failures. The dashboard is considered an optional, non-critical component.
    """
    config = context.config
    if not config.dashboard.enabled:
        logger.info("Dashboard is disabled via configuration")
        return None

    try:
        from sentinel.dashboard import create_app

        logger.info(
            "Starting dashboard server on %s:%s", config.dashboard.host, config.dashboard.port
        )
        dashboard_app = create_app(sentinel)
        dashboard_server = DashboardServer(
            host=config.dashboard.host,
            port=config.dashboard.port,
        )
        dashboard_server.start(dashboard_app)
        return dashboard_server
    except ImportError as e:
        logger.warning(
            "Dashboard startup failed: dependencies not available. "
            "Sentinel will continue without dashboard functionality. Error: %s",
            e,
            extra={"host": config.dashboard.host, "port": config.dashboard.port},
        )
        return None
    except OSError as e:
        logger.warning(
            "Dashboard startup failed: network/OS error. "
            "Sentinel will continue without dashboard functionality. Error: %s",
            e,
            extra={"host": config.dashboard.host, "port": config.dashboard.port},
        )
        return None
    except RuntimeError as e:
        logger.warning(
            "Dashboard startup failed: runtime error. "
            "Sentinel will continue without dashboard functionality. Error: %s",
            e,
            extra={"host": config.dashboard.host, "port": config.dashboard.port},
        )
        return None
    except (ValueError, TypeError) as e:
        logger.warning(
            "Dashboard startup failed: configuration error. "
            "Sentinel will continue without dashboard functionality. Error: %s",
            e,
            extra={"host": config.dashboard.host, "port": config.dashboard.port},
        )
        return None
    except Exception as e:
        # INTENTIONAL BROAD CATCH: Dashboard is optional and must never crash Sentinel.
        # All known exception types are caught above; this catches truly unexpected
        # errors to ensure the main application continues operating.
        logger.warning(
            "Dashboard startup failed: unexpected error (%s). "
            "Sentinel will continue without dashboard functionality. Error: %s",
            type(e).__name__,
            e,
            extra={
                "host": config.dashboard.host,
                "port": config.dashboard.port,
                "error_type": type(e).__name__,
            },
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
