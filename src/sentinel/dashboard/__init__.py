"""Dashboard module for Developer Sentinel.

This module provides a web-based dashboard for monitoring the Sentinel orchestrator.
It includes a FastAPI application factory, route handlers, and read-only state access.

The dashboard is decoupled from Sentinel internals through the SentinelStateProvider
protocol. This protocol defines the public interface that Sentinel implements,
allowing the dashboard to access state without depending on internal implementation
details like locks, futures, or private attributes.

Key components:
- SentinelStateProvider: Protocol defining the interface between Sentinel and dashboard
- SentinelStateAccessor: Adapter that converts Sentinel state to dashboard DTOs
- DashboardState: Immutable snapshot of state for template rendering
- OrchestrationVersionSnapshot: DTO for version information
- ExecutionStateSnapshot: DTO for execution state
"""

from sentinel.dashboard.app import create_app
from sentinel.dashboard.state import (
    DashboardState,
    ExecutionStateSnapshot,
    OrchestrationVersionSnapshot,
    SentinelStateAccessor,
    SentinelStateProvider,
)

# NOTE: Update this list when adding new exports to this module.
__all__ = [
    "create_app",
    "DashboardState",
    "ExecutionStateSnapshot",
    "OrchestrationVersionSnapshot",
    "SentinelStateAccessor",
    "SentinelStateProvider",
]
