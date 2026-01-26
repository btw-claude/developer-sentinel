"""Agent clients package for multi-agent backend support.

This package provides abstract and concrete implementations for different
agent backends (Claude Agent SDK, Cursor CLI, etc.).
"""

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
)
from sentinel.agent_clients.claude_sdk import (
    ClaudeProcessInterruptedError,
    ClaudeSdkAgentClient,
    TimingMetrics,
    get_shutdown_event,
    is_shutdown_requested,
    request_shutdown,
    reset_shutdown,
)
from sentinel.agent_clients.cursor import (
    CursorAgentClient,
    CursorMode,
)
from sentinel.agent_clients.factory import (
    AgentClientFactory,
    create_default_factory,
)

__all__ = [
    # Base classes and types
    "AgentClient",
    "AgentClientError",
    "AgentRunResult",
    "AgentTimeoutError",
    "AgentType",
    # Claude SDK client
    "ClaudeProcessInterruptedError",
    "ClaudeSdkAgentClient",
    "TimingMetrics",
    "get_shutdown_event",
    "is_shutdown_requested",
    "request_shutdown",
    "reset_shutdown",
    # Cursor CLI client
    "CursorAgentClient",
    "CursorMode",
    # Factory
    "AgentClientFactory",
    "create_default_factory",
]
