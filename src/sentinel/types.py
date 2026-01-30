"""Type definitions and enums for the sentinel application.

This module provides centralized type definitions including enums for
trigger sources and agent types, replacing magic strings throughout
the codebase with type-safe constants.

Usage:
    from sentinel.types import TriggerSource, AgentType

    # Enum usage
    if trigger.source == TriggerSource.JIRA:
        ...

    # String value access for YAML compatibility
    source_str = TriggerSource.JIRA.value  # "jira"

    # Validation
    TriggerSource.is_valid("jira")  # True
    AgentType.is_valid("claude")  # True
"""

from __future__ import annotations

from enum import Enum
from typing import Literal


class TriggerSource(str, Enum):
    """Enum for trigger source types.

    Inherits from str to allow direct string comparison and
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


class AgentType(str, Enum):
    """Enum for agent types.

    Inherits from str to allow direct string comparison and
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


class CursorMode(str, Enum):
    """Enum for Cursor CLI modes.

    Inherits from str to allow direct string comparison and
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


class RateLimitStrategy(str, Enum):
    """Enum for rate limit strategies.

    Inherits from str to allow direct string comparison and
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


# Type aliases for Literal types (for backward compatibility with type hints)
TriggerSourceLiteral = Literal["jira", "github"]
AgentTypeLiteral = Literal["claude", "cursor"]
CursorModeLiteral = Literal["agent", "plan", "ask"]
RateLimitStrategyLiteral = Literal["queue", "reject"]

# Frozen sets for validation (derived from enums for single source of truth)
VALID_TRIGGER_SOURCES = TriggerSource.values()
VALID_AGENT_TYPES = AgentType.values()
VALID_CURSOR_MODES = CursorMode.values()
VALID_RATE_LIMIT_STRATEGIES = RateLimitStrategy.values()
