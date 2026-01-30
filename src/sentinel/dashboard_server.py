"""Dashboard server management for Developer Sentinel.

This module provides the DashboardServer class that manages a uvicorn server
running in a background thread, allowing the dashboard to run alongside
the main Sentinel polling loop.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uvicorn
    from starlette.types import ASGIApp

from sentinel.logging import get_logger

logger = get_logger(__name__)


class DashboardServer:
    """Background server for the dashboard web interface.

    This class manages a uvicorn server running in a background thread,
    allowing the dashboard to run alongside the main Sentinel polling loop.

    Example:
        from sentinel.dashboard import create_app
        from sentinel.dashboard_server import DashboardServer

        app = create_app(sentinel)
        server = DashboardServer(host="0.0.0.0", port=8080)
        server.start(app)

        # ... run main loop ...

        server.shutdown()
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize the dashboard server.

        Args:
            host: The host address to bind to (e.g., "0.0.0.0" or "127.0.0.1").
            port: The port to listen on.
        """
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        """Get the host address."""
        return self._host

    @property
    def port(self) -> int:
        """Get the port number."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Check if the server is currently running."""
        return self._server is not None and self._server.started

    def start(self, app: ASGIApp) -> None:
        """Start the dashboard server in a background thread.

        Args:
            app: The ASGI application to serve.

        Note:
            This method blocks until the server has started (up to 5 seconds),
            then returns. The server continues running in a background thread.
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
        """Shutdown the dashboard server gracefully.

        This method signals the server to stop and waits up to 5 seconds
        for the server thread to terminate.
        """
        if self._server is not None:
            logger.info("Shutting down dashboard server...")
            self._server.should_exit = True

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=5.0)
                if self._thread.is_alive():
                    logger.warning("Dashboard server thread did not terminate gracefully")

            logger.info("Dashboard server shutdown complete")


__all__ = ["DashboardServer"]
