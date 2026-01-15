"""Dashboard module for Developer Sentinel.

This module provides a web-based dashboard for monitoring the Sentinel orchestrator.
It includes a FastAPI application factory, route handlers, and read-only state access.
"""

from sentinel.dashboard.app import create_app
from sentinel.dashboard.state import DashboardState, SentinelStateAccessor

__all__ = [
    "create_app",
    "DashboardState",
    "SentinelStateAccessor",
]
