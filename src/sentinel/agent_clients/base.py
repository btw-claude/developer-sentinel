"""Base classes and types for agent client implementations.

This module provides the foundation for multi-agent backend support,
including the abstract base class for agent clients and common types.

The AgentClient interface is async-native to enable proper async composition
and avoid creating new event loops per call. Callers should use asyncio.run()
at their entry points when needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Type alias for supported agent types
AgentType = Literal["claude", "cursor"]


@dataclass
class AgentRunResult:
    """Result of running an agent.

    Attributes:
        response: The agent's response text.
        workdir: Path to the agent's working directory, if one was created.
    """

    response: str
    workdir: Path | None = None


class AgentClientError(Exception):
    """Raised when an agent client operation fails."""

    pass


class AgentTimeoutError(AgentClientError):
    """Raised when an agent execution times out."""

    pass


class AgentClient(ABC):
    """Abstract interface for agent client operations.

    This allows the executor to work with different agent implementations:
    - Claude Agent SDK client
    - Cursor CLI client
    - Mock client (testing)
    """

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements.

        Returns:
            The agent type identifier ('claude' or 'cursor').
        """
        pass

    @abstractmethod
    async def run_agent(
        self,
        prompt: str,
        tools: list[str],
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        """Run an agent with the given prompt and tools.

        This is an async method to enable proper async composition and avoid
        creating new event loops per call. Callers should use asyncio.run()
        at their entry points when needed.

        Args:
            prompt: The prompt to send to the agent.
            tools: List of tool names the agent can use.
            context: Optional context dict (e.g., GitHub repo info).
            timeout_seconds: Optional timeout in seconds. If None, no timeout is applied.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the client's default model.
            orchestration_name: Optional orchestration name for streaming log files.
            branch: Optional branch name to checkout/create before running the agent.
            create_branch: If True and branch doesn't exist, create it from base_branch.
            base_branch: Base branch to create new branches from. Defaults to "main".

        Returns:
            AgentRunResult containing the agent's response text and optional working directory path.

        Raises:
            AgentClientError: If the agent execution fails.
            AgentTimeoutError: If the agent execution times out.
        """
        pass
