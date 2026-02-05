"""Type definitions and enums for the sentinel application.

This module provides centralized type definitions including enums for
trigger sources and agent types, replacing magic strings throughout
the codebase with type-safe constants.

Usage:
    from sentinel.types import TriggerSource, AgentType

    # Enum usage - since TriggerSource inherits from StrEnum, direct comparison works
    if trigger.source == TriggerSource.JIRA:
        ...

    # String value access - .value is redundant since enum inherits from StrEnum
    # TriggerSource.JIRA == "jira" and str(TriggerSource.JIRA) == "jira"
    source_str = TriggerSource.JIRA  # Can be used directly as "jira"

    # Validation
    TriggerSource.is_valid("jira")  # True
    AgentType.is_valid("claude")  # True
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal


class TriggerSource(StrEnum):
    """Enum for trigger source types.

    Inherits from StrEnum to allow direct string comparison and
    seamless YAML serialization/deserialization.

    Values:
        JIRA: Jira issue triggers ("jira")
        GITHUB: GitHub issue/PR triggers ("github")
    """

    JIRA = "jira"
    GITHUB = "github"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid trigger source.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid trigger source.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid trigger source values as a frozenset.

        Returns:
            Frozenset of valid trigger source string values.
        """
        return frozenset(member.value for member in cls)


class AgentType(StrEnum):
    """Enum for agent types.

    Inherits from StrEnum to allow direct string comparison and
    seamless YAML serialization/deserialization.

    Values:
        CLAUDE: Claude SDK agent ("claude")
        CURSOR: Cursor CLI agent ("cursor")
    """

    CLAUDE = "claude"
    CURSOR = "cursor"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid agent type.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid agent type.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid agent type values as a frozenset.

        Returns:
            Frozenset of valid agent type string values.
        """
        return frozenset(member.value for member in cls)


class CursorMode(StrEnum):
    """Enum for Cursor CLI modes.

    Inherits from StrEnum to allow direct string comparison and
    seamless YAML serialization/deserialization.

    Values:
        AGENT: Agent mode ("agent")
        PLAN: Plan mode ("plan")
        ASK: Ask mode ("ask")
    """

    AGENT = "agent"
    PLAN = "plan"
    ASK = "ask"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid cursor mode.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid cursor mode.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid cursor mode values as a frozenset.

        Returns:
            Frozenset of valid cursor mode string values.
        """
        return frozenset(member.value for member in cls)


class RateLimitStrategy(StrEnum):
    """Enum for rate limit strategies.

    Inherits from StrEnum to allow direct string comparison and
    seamless configuration handling.

    Values:
        QUEUE: Queue requests when rate limited ("queue")
        REJECT: Reject requests when rate limited ("reject")
    """

    QUEUE = "queue"
    REJECT = "reject"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid rate limit strategy.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid rate limit strategy.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid rate limit strategy values as a frozenset.

        Returns:
            Frozenset of valid rate limit strategy string values.
        """
        return frozenset(member.value for member in cls)


class QueueFullStrategy(StrEnum):
    """Enum for queue full strategies when bounded queue is at capacity.

    Inherits from StrEnum to allow direct string comparison and
    seamless configuration handling.

    Values:
        REJECT: Reject new requests immediately when queue is full ("reject")
        WAIT: Wait for queue space (subject to timeout) ("wait")
    """

    REJECT = "reject"
    WAIT = "wait"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid queue full strategy.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid queue full strategy.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid queue full strategy values as a frozenset.

        Returns:
            Frozenset of valid queue full strategy string values.
        """
        return frozenset(member.value for member in cls)


class ErrorType(StrEnum):
    """Enum for execution failure error types.

    Inherits from StrEnum to allow direct string comparison and
    seamless logging integration.

    This enum provides type safety for the error_type parameter in
    _handle_execution_failure, preventing typos and enabling IDE
    autocomplete support.

    Values:
        IO_ERROR: I/O and network-related errors ("I/O error")
        RUNTIME_ERROR: Runtime execution errors ("runtime error")
        DATA_ERROR: Data validation and key errors ("data error")
    """

    IO_ERROR = "I/O error"
    RUNTIME_ERROR = "runtime error"
    DATA_ERROR = "data error"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string value is a valid error type.

        Args:
            value: The string value to validate.

        Returns:
            True if the value matches a valid error type.
        """
        return value in cls._value2member_map_

    @classmethod
    def values(cls) -> frozenset[str]:
        """Return all valid error type values as a frozenset.

        Returns:
            Frozenset of valid error type string values.
        """
        return frozenset(member.value for member in cls)


# Type aliases for Literal types (for backward compatibility with type hints)
TriggerSourceLiteral = Literal["jira", "github"]
AgentTypeLiteral = Literal["claude", "cursor"]
CursorModeLiteral = Literal["agent", "plan", "ask"]
RateLimitStrategyLiteral = Literal["queue", "reject"]
QueueFullStrategyLiteral = Literal["reject", "wait"]
ErrorTypeLiteral = Literal["I/O error", "runtime error", "data error"]

# Frozen sets for validation (derived from enums for single source of truth)
VALID_TRIGGER_SOURCES = TriggerSource.values()
VALID_AGENT_TYPES = AgentType.values()
VALID_CURSOR_MODES = CursorMode.values()
VALID_RATE_LIMIT_STRATEGIES = RateLimitStrategy.values()
VALID_QUEUE_FULL_STRATEGIES = QueueFullStrategy.values()
VALID_ERROR_TYPES = ErrorType.values()
