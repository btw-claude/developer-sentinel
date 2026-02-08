"""Agent clients package for multi-agent backend support.

This package provides abstract and concrete implementations for different
agent backends (Claude Agent SDK, Codex CLI, Cursor CLI, etc.).
"""

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
    UsageInfo,
)
from sentinel.agent_clients.claude_sdk import (
    ClaudeProcessInterruptedError,
    ClaudeSdkAgentClient,
    ShutdownController,
    TimingMetrics,
)
from sentinel.agent_clients.codex import CodexAgentClient
from sentinel.agent_clients.cursor import CursorAgentClient, CursorMode
from sentinel.agent_clients.factory import AgentClientFactory, create_default_factory

# NOTE: Update this list when adding new exports to this module.
__all__ = [
    # Base classes and types
    "AgentClient",
    "AgentClientError",
    "AgentRunResult",
    "AgentTimeoutError",
    "AgentType",
    "UsageInfo",
    # Claude SDK client
    "ClaudeProcessInterruptedError",
    "ClaudeSdkAgentClient",
    "ShutdownController",
    "TimingMetrics",
    # Codex CLI client
    "CodexAgentClient",
    # Cursor CLI client
    "CursorAgentClient",
    "CursorMode",
    # Factory
    "AgentClientFactory",
    "create_default_factory",
]
