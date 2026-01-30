"""Base classes and types for agent client implementations.

This module provides the foundation for multi-agent backend support,
including the abstract base class for agent clients and common types.

The AgentClient interface is async-native to enable proper async composition
and avoid creating new event loops per call. Callers should use asyncio.run()
at their entry points when needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sentinel.types import AgentTypeLiteral

# Type alias for supported agent types (re-exported for backward compatibility)
AgentType = AgentTypeLiteral


@dataclass
class UsageInfo:
    """Usage information from Claude Agent SDK ResultMessage.

    Captures token usage and cost data from the Claude Agent SDK's ResultMessage
    that is yielded at the end of each query. This data is valuable for tracking
    costs, monitoring usage patterns, and optimizing agent operations.

    The total_tokens property is computed from input_tokens + output_tokens to
    guarantee consistency. If an explicit total_tokens value is provided during
    initialization, it will be validated against the computed value.

    Attributes:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        total_cost_usd: Total cost in USD for this query.
        duration_ms: Total duration in milliseconds.
        duration_api_ms: Time spent on API calls in milliseconds.
        num_turns: Number of conversation turns in this query.
        _total_tokens_override: Internal field for backward compatibility.
            If set to a non-None value during initialization, it will be
            validated against the computed total.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    duration_api_ms: float = 0.0
    num_turns: int = 0
    _total_tokens_override: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate total_tokens consistency if explicitly provided."""
        if self._total_tokens_override is not None:
            computed = self.input_tokens + self.output_tokens
            if self._total_tokens_override != computed:
                raise ValueError(
                    f"total_tokens mismatch: provided {self._total_tokens_override}, "
                    f"but input_tokens ({self.input_tokens}) + output_tokens "
                    f"({self.output_tokens}) = {computed}"
                )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output).

        This property is computed to guarantee consistency between
        input_tokens, output_tokens, and total_tokens.

        Returns:
            Sum of input_tokens and output_tokens.
        """
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing all usage fields in a readable format.
        """
        return (
            f"UsageInfo(input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"total_tokens={self.total_tokens}, "
            f"total_cost_usd={self.total_cost_usd}, "
            f"duration_ms={self.duration_ms}, "
            f"duration_api_ms={self.duration_api_ms}, "
            f"num_turns={self.num_turns})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert usage info to dictionary for serialization.

        Returns:
            Dictionary representation of usage data for JSON export.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "duration_ms": self.duration_ms,
            "duration_api_ms": self.duration_api_ms,
            "num_turns": self.num_turns,
        }


@dataclass
class AgentRunResult:
    """Result of running an agent.

    Attributes:
        response: The agent's response text.
        workdir: Path to the agent's working directory, if one was created.
        usage: Optional usage information (tokens, cost, duration) from the agent.
            Only populated for Claude SDK client when ResultMessage is available.
            CursorAgentClient returns None for backward compatibility.
    """

    response: str
    workdir: Path | None = None
    usage: UsageInfo | None = None


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
