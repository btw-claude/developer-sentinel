"""Base classes and types for agent client implementations.

This module provides the foundation for multi-agent backend support,
including the abstract base class for agent clients and common types.

The AgentClient interface is async-native to enable proper async composition
and avoid creating new event loops per call. Callers should use asyncio.run()
at their entry points when needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.logging import get_logger
from sentinel.types import AgentType

logger = get_logger(__name__)

__all__ = [
    "AgentClient",
    "AgentClientError",
    "AgentRunCoroutine",
    "AgentRunResult",
    "AgentTimeoutError",
    "AgentType",
    "UsageInfo",
]


@dataclass
class UsageInfo:
    """Usage information from Claude Agent SDK ResultMessage.

    Captures token usage and cost data from the Claude Agent SDK's ResultMessage
    that is yielded at the end of each query. This data is valuable for tracking
    costs, monitoring usage patterns, and optimizing agent operations.

    The total_tokens property is computed from input_tokens + output_tokens to
    guarantee consistency.

    Attributes:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        total_cost_usd: Total cost in USD for this query.
        duration_ms: Total duration in milliseconds.
        duration_api_ms: Time spent on API calls in milliseconds.
        num_turns: Number of conversation turns in this query.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: float = 0.0
    duration_api_ms: float = 0.0
    num_turns: int = 0

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


# Explicit async return type alias for improved IDE support and documentation (DS-533).
# This type alias makes the async nature of run_agent more discoverable and provides
# better autocomplete and type inference in IDEs.
AgentRunCoroutine = Coroutine[Any, Any, AgentRunResult]
"""Coroutine type for async agent execution methods returning AgentRunResult."""


class AgentClient(ABC):
    """Abstract interface for agent client operations.

    This allows the executor to work with different agent implementations:
    - Claude Agent SDK client
    - Codex CLI client
    - Cursor CLI client
    - Mock client (testing)

    Subclasses must set ``base_workdir`` in their ``__init__`` to enable
    working directory creation via :meth:`_create_workdir`.
    """

    base_workdir: Path | None = None

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements.

        Returns:
            The agent type identifier ('claude' or 'cursor').
        """
        pass

    # Truncation limits for context sanitization (DS-675).
    # Keys are truncated to 200 characters; values to 2000 characters.
    _CONTEXT_KEY_MAX_LENGTH: int = 200
    _CONTEXT_VALUE_MAX_LENGTH: int = 2000

    def _build_prompt_with_context(
        self,
        prompt: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Build a full prompt by appending a sanitized context section.

        Context values are sanitized to prevent prompt injection from
        untrusted sources (DS-666): values are converted to strings,
        truncated to a safe maximum length, and control characters
        (newlines, carriage returns) are stripped.

        This shared implementation replaces identical sanitization logic
        that was previously duplicated across ClaudeSdkAgentClient,
        CodexAgentClient, and CursorAgentClient (DS-675).

        Args:
            prompt: The base prompt text.
            context: Optional context dict (e.g., GitHub repo info).
                If None or empty, the prompt is returned unchanged.

        Returns:
            The prompt, optionally followed by a sanitized Context section.
        """
        if not context:
            return prompt

        sanitized_items = []
        for k, v in context.items():
            safe_key = (
                str(k)[: self._CONTEXT_KEY_MAX_LENGTH]
                .replace("\n", " ")
                .replace("\r", "")
            )
            safe_val = (
                str(v)[: self._CONTEXT_VALUE_MAX_LENGTH]
                .replace("\n", " ")
                .replace("\r", "")
            )
            sanitized_items.append(f"- {safe_key}: {safe_val}\n")
        return prompt + "\n\nContext:\n" + "".join(sanitized_items)

    def _create_workdir(self, issue_key: str) -> Path:
        """Create a unique working directory for an agent run.

        This shared implementation is used by all agent clients (Claude SDK,
        Codex, Cursor) to create timestamped working directories for agent
        executions. Extracted to the base class to eliminate duplication
        across concrete implementations (DS-666).

        The directory name format is ``{issue_key}_{YYYYMMDD_HHMMSS}``, created
        under the ``base_workdir`` directory.

        Args:
            issue_key: The issue key to include in the directory name.

        Returns:
            Path to the created working directory.

        Raises:
            AgentClientError: If ``base_workdir`` is not configured (i.e., is None).
        """
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")

        self.base_workdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = self.base_workdir / f"{issue_key}_{timestamp}"
        workdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created agent working directory: %s", workdir)
        return workdir

    @abstractmethod
    async def run_agent(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
    ) -> AgentRunResult:
        """Run an agent with the given prompt.

        This is an async method to enable proper async composition and avoid
        creating new event loops per call. Callers should use asyncio.run()
        at their entry points when needed.

        Note:
            This method returns a coroutine (``Coroutine[Any, Any, AgentRunResult]``).
            The ``AgentRunCoroutine`` type alias is provided for explicit type
            annotations where needed (DS-533).

        Args:
            prompt: The prompt to send to the agent.
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
