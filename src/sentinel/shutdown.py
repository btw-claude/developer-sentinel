"""Graceful shutdown handling for Developer Sentinel.

This module provides signal handling and shutdown coordination for:
- SIGINT (Ctrl+C) handling
- SIGTERM handling
- Graceful termination of running processes
"""

from __future__ import annotations

import signal
from collections.abc import Callable
from types import FrameType

from sentinel.logging import get_logger
from sentinel.sdk_clients import request_shutdown as request_claude_shutdown

logger = get_logger(__name__)


class ShutdownHandler:
    """Manages graceful shutdown of the Sentinel application.

    This class coordinates shutdown requests from signals (SIGINT, SIGTERM)
    and propagates them to the main application loop and any running Claude
    processes.
    """

    def __init__(self, on_shutdown: Callable[[], None] | None = None) -> None:
        """Initialize the shutdown handler.

        Args:
            on_shutdown: Optional callback to invoke when shutdown is requested.
                        Typically this sets a flag on the Sentinel instance.
        """
        self._shutdown_requested = False
        self._on_shutdown = on_shutdown

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def request_shutdown(self) -> None:
        """Request graceful shutdown.

        This method can be called programmatically to initiate shutdown,
        in addition to signal-based shutdown.
        """
        logger.info("Shutdown requested")
        self._shutdown_requested = True

        if self._on_shutdown is not None:
            self._on_shutdown()

        # Signal Claude processes to terminate
        request_claude_shutdown()

    def handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals (SIGINT, SIGTERM).

        Args:
            signum: The signal number received.
            frame: The current stack frame (unused).
        """
        signal_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", signal_name)
        self.request_shutdown()

    def install_signal_handlers(self) -> None:
        """Install signal handlers for SIGINT and SIGTERM.

        After calling this method, SIGINT (Ctrl+C) and SIGTERM will
        trigger graceful shutdown instead of immediate termination.
        """
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        logger.debug("Signal handlers installed for SIGINT and SIGTERM")


def create_shutdown_handler(on_shutdown: Callable[[], None] | None = None) -> ShutdownHandler:
    """Create and configure a shutdown handler.

    This is a convenience factory function that creates a ShutdownHandler
    and installs signal handlers.

    Args:
        on_shutdown: Optional callback to invoke when shutdown is requested.

    Returns:
        Configured ShutdownHandler with signal handlers installed.
    """
    handler = ShutdownHandler(on_shutdown)
    handler.install_signal_handlers()
    return handler


__all__ = [
    "ShutdownHandler",
    "create_shutdown_handler",
]
