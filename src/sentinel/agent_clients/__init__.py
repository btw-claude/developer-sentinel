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

__all__ = [
    "AgentClient",
    "AgentClientError",
    "AgentRunResult",
    "AgentTimeoutError",
    "AgentType",
]
