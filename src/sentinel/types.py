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
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args


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
        CODEX: Codex CLI agent ("codex")
        CURSOR: Cursor CLI agent ("cursor")
    """

    CLAUDE = "claude"
    CODEX = "codex"
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


# Type aliases for Literal types (for backward compatibility with type hints).
#
# These Literal types must stay in sync with their corresponding StrEnum classes.
# Static type checkers (mypy) require explicit Literal values and cannot resolve
# dynamically-constructed types, so we keep the Literal definitions explicit here.
# The _validate_literal_matches_enum() calls below enforce at import time that
# each Literal alias matches its enum, preventing silent drift when new members
# are added to an enum. See DS-649 for context.
TriggerSourceLiteral = Literal["jira", "github"]
AgentTypeLiteral = Literal["claude", "codex", "cursor"]
CursorModeLiteral = Literal["agent", "plan", "ask"]
RateLimitStrategyLiteral = Literal["queue", "reject"]
QueueFullStrategyLiteral = Literal["reject", "wait"]
ErrorTypeLiteral = Literal["I/O error", "runtime error", "data error"]

# Type alias for Literal type forms passed to validation helpers (DS-659, DS-669).
#
# Python does not provide a clean runtime type for Literal[...] annotations;
# at runtime they are typing._GenericAlias instances, but that is a private API.
# mypy represents them as ``<typing special form>``, which is only assignable to
# ``object``.
#
# We use a dual-definition strategy (DS-669):
# - Under TYPE_CHECKING (static analysis / mypy): LiteralForm remains ``object``
#   because mypy cannot assign Literal special forms to a Protocol.
# - At runtime: LiteralForm is a @runtime_checkable Protocol requiring an
#   ``__origin__`` attribute (typed as ``object``), which all Literal forms
#   possess (set to ``typing.Literal`` — a ``typing._SpecialForm``, not a
#   ``type``; see DS-687).  This avoids depending on the private
#   ``typing._GenericAlias`` while providing structurally tighter type narrowing
#   than bare ``object``.
#
# If Python ever exposes a public type for Literal forms, this alias should be
# updated to use it.
if TYPE_CHECKING:
    LiteralForm: TypeAlias = object
else:
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class LiteralForm(Protocol):
        """Runtime protocol for Literal type forms.

        All ``Literal[...]`` annotations have an ``__origin__`` attribute set to
        ``typing.Literal`` at runtime.  This protocol captures that structural
        requirement without depending on private ``typing._GenericAlias``.

        Note: ``__origin__`` is typed as ``object`` rather than ``type`` because
        ``Literal[...].__origin__`` is ``typing.Literal`` — a
        ``typing._SpecialForm``, not a ``type`` instance (DS-687).
        """

        __origin__: object


def _validate_literal_matches_enum(
    literal_type: LiteralForm,
    enum_cls: type[StrEnum],
    alias_name: str,
) -> None:
    """Validate that a Literal type alias matches its corresponding StrEnum.

    Compares the values in a Literal type alias against the values of a StrEnum
    class, raising TypeError at import time if they diverge. This prevents
    silent drift when new members are added to an enum but the Literal alias
    is not updated.

    Args:
        literal_type: The Literal type alias to validate (e.g., AgentTypeLiteral).
            Typed as ``LiteralForm``; at static-analysis time this is ``object``
            for mypy compatibility, and at runtime it is a ``Protocol`` requiring
            ``__origin__`` (see DS-669).
        enum_cls: The StrEnum class that is the source of truth (e.g., AgentType).
        alias_name: Human-readable name of the Literal alias for error messages.

    Raises:
        TypeError: If the Literal values do not exactly match the enum values.
    """
    literal_values = set(get_args(literal_type))
    enum_values = {member.value for member in enum_cls}
    if literal_values != enum_values:
        missing_from_literal = enum_values - literal_values
        extra_in_literal = literal_values - enum_values
        parts: list[str] = [f"{alias_name} is out of sync with {enum_cls.__name__}"]
        if missing_from_literal:
            parts.append(f"missing from Literal: {sorted(missing_from_literal)}")
        if extra_in_literal:
            parts.append(f"extra in Literal: {sorted(extra_in_literal)}")
        raise TypeError("; ".join(parts))


# Import-time validation: ensure every Literal alias matches its enum (DS-649).
_validate_literal_matches_enum(TriggerSourceLiteral, TriggerSource, "TriggerSourceLiteral")
_validate_literal_matches_enum(AgentTypeLiteral, AgentType, "AgentTypeLiteral")
_validate_literal_matches_enum(CursorModeLiteral, CursorMode, "CursorModeLiteral")
_validate_literal_matches_enum(
    RateLimitStrategyLiteral, RateLimitStrategy, "RateLimitStrategyLiteral"
)
_validate_literal_matches_enum(
    QueueFullStrategyLiteral, QueueFullStrategy, "QueueFullStrategyLiteral"
)
_validate_literal_matches_enum(ErrorTypeLiteral, ErrorType, "ErrorTypeLiteral")

# Frozen sets for validation (derived from enums for single source of truth)
VALID_TRIGGER_SOURCES = TriggerSource.values()
VALID_AGENT_TYPES = AgentType.values()
VALID_CURSOR_MODES = CursorMode.values()
VALID_RATE_LIMIT_STRATEGIES = RateLimitStrategy.values()
VALID_QUEUE_FULL_STRATEGIES = QueueFullStrategy.values()
VALID_ERROR_TYPES = ErrorType.values()
