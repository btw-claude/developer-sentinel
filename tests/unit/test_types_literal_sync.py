"""Tests for Literal type alias and StrEnum synchronization (DS-649, DS-659, DS-669).

Validates that every Literal type alias in sentinel.types stays in sync
with its corresponding StrEnum class. These tests complement the import-time
validation in types.py by providing explicit, fine-grained assertions.

Also validates the LiteralForm runtime Protocol (DS-669) which provides
structurally tighter type narrowing than bare ``object`` at runtime.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, get_args

import pytest

from sentinel.types import (
    AgentType,
    AgentTypeLiteral,
    CursorMode,
    CursorModeLiteral,
    ErrorType,
    ErrorTypeLiteral,
    LiteralForm,
    QueueFullStrategy,
    QueueFullStrategyLiteral,
    RateLimitStrategy,
    RateLimitStrategyLiteral,
    TriggerSource,
    TriggerSourceLiteral,
    _validate_literal_matches_enum,
)


class TestLiteralEnumSync:
    """Ensure each Literal type alias matches its StrEnum."""

    @pytest.mark.parametrize(
        ("literal_type", "enum_cls", "alias_name"),
        [
            (TriggerSourceLiteral, TriggerSource, "TriggerSourceLiteral"),
            (AgentTypeLiteral, AgentType, "AgentTypeLiteral"),
            (CursorModeLiteral, CursorMode, "CursorModeLiteral"),
            (RateLimitStrategyLiteral, RateLimitStrategy, "RateLimitStrategyLiteral"),
            (QueueFullStrategyLiteral, QueueFullStrategy, "QueueFullStrategyLiteral"),
            (ErrorTypeLiteral, ErrorType, "ErrorTypeLiteral"),
        ],
    )
    def test_literal_values_match_enum_values(
        self,
        literal_type: LiteralForm,
        enum_cls: type[StrEnum],
        alias_name: str,
    ) -> None:
        """Literal type alias values must exactly match StrEnum member values."""
        literal_values = set(get_args(literal_type))
        enum_values = {member.value for member in enum_cls}
        assert literal_values == enum_values, (
            f"{alias_name} values {literal_values} do not match "
            f"{enum_cls.__name__} values {enum_values}"
        )

    @pytest.mark.parametrize(
        ("literal_type", "enum_cls", "alias_name"),
        [
            (TriggerSourceLiteral, TriggerSource, "TriggerSourceLiteral"),
            (AgentTypeLiteral, AgentType, "AgentTypeLiteral"),
            (CursorModeLiteral, CursorMode, "CursorModeLiteral"),
            (RateLimitStrategyLiteral, RateLimitStrategy, "RateLimitStrategyLiteral"),
            (QueueFullStrategyLiteral, QueueFullStrategy, "QueueFullStrategyLiteral"),
            (ErrorTypeLiteral, ErrorType, "ErrorTypeLiteral"),
        ],
    )
    def test_literal_count_matches_enum_count(
        self,
        literal_type: LiteralForm,
        enum_cls: type[StrEnum],
        alias_name: str,
    ) -> None:
        """Literal type alias must have the same number of values as its StrEnum."""
        literal_count = len(get_args(literal_type))
        enum_count = len(enum_cls)
        assert literal_count == enum_count, (
            f"{alias_name} has {literal_count} values but "
            f"{enum_cls.__name__} has {enum_count} members"
        )


class TestValidateLiteralMatchesEnum:
    """Tests for the _validate_literal_matches_enum helper function."""

    def test_matching_literal_and_enum_passes(self) -> None:
        """Validation passes when Literal values match enum values."""
        _validate_literal_matches_enum(AgentTypeLiteral, AgentType, "AgentTypeLiteral")

    def test_missing_value_raises_type_error(self) -> None:
        """Validation fails when Literal is missing an enum value."""
        IncompleteAgentLiteral = Literal["claude", "codex"]  # noqa: N806
        with pytest.raises(TypeError, match="missing from Literal"):
            _validate_literal_matches_enum(
                IncompleteAgentLiteral, AgentType, "IncompleteAgentLiteral"
            )

    def test_extra_value_raises_type_error(self) -> None:
        """Validation fails when Literal has a value not in the enum."""
        ExtraAgentLiteral = Literal["claude", "codex", "cursor", "gemini"]  # noqa: N806
        with pytest.raises(TypeError, match="extra in Literal"):
            _validate_literal_matches_enum(
                ExtraAgentLiteral, AgentType, "ExtraAgentLiteral"
            )

    def test_error_message_includes_alias_name(self) -> None:
        """Error message includes the alias name for easy identification."""
        BadLiteral = Literal["wrong"]  # noqa: N806
        with pytest.raises(TypeError, match="BadLiteral"):
            _validate_literal_matches_enum(BadLiteral, AgentType, "BadLiteral")

    def test_error_message_includes_enum_name(self) -> None:
        """Error message includes the enum class name."""
        BadLiteral = Literal["wrong"]  # noqa: N806
        with pytest.raises(TypeError, match="AgentType"):
            _validate_literal_matches_enum(BadLiteral, AgentType, "BadLiteral")

    def test_completely_wrong_literal_shows_both_missing_and_extra(self) -> None:
        """When Literal is completely wrong, both missing and extra are reported."""
        WrongLiteral = Literal["foo", "bar"]  # noqa: N806
        with pytest.raises(TypeError) as exc_info:
            _validate_literal_matches_enum(WrongLiteral, AgentType, "WrongLiteral")
        message = str(exc_info.value)
        assert "missing from Literal" in message
        assert "extra in Literal" in message

    def test_missing_values_are_sorted_in_error_message(self) -> None:
        """Missing values in the error message are sorted for reproducibility (DS-659)."""
        IncompleteAgentLiteral = Literal["claude"]  # noqa: N806
        with pytest.raises(TypeError) as exc_info:
            _validate_literal_matches_enum(
                IncompleteAgentLiteral, AgentType, "IncompleteAgentLiteral"
            )
        message = str(exc_info.value)
        assert "['codex', 'cursor']" in message

    def test_extra_values_are_sorted_in_error_message(self) -> None:
        """Extra values in the error message are sorted for reproducibility (DS-659)."""
        ExtraLiteral = Literal["claude", "codex", "cursor", "gemini", "aardvark"]  # noqa: N806
        with pytest.raises(TypeError) as exc_info:
            _validate_literal_matches_enum(ExtraLiteral, AgentType, "ExtraLiteral")
        message = str(exc_info.value)
        assert "['aardvark', 'gemini']" in message


class TestLiteralFormTypeAlias:
    """Tests for the LiteralForm runtime Protocol (DS-659, DS-669).

    At runtime, LiteralForm is a @runtime_checkable Protocol requiring
    ``__origin__``, providing structurally tighter type narrowing than bare
    ``object``.  Under TYPE_CHECKING (mypy), it remains ``object``.
    """

    def test_literal_form_is_runtime_checkable_protocol(self) -> None:
        """LiteralForm is a runtime-checkable Protocol at runtime (DS-669)."""
        assert hasattr(LiteralForm, "_is_runtime_protocol")
        assert LiteralForm._is_runtime_protocol  # noqa: SLF001

    def test_literal_forms_are_instances_of_protocol(self) -> None:
        """All Literal type aliases satisfy the LiteralForm Protocol."""
        literal_types = [
            TriggerSourceLiteral,
            AgentTypeLiteral,
            CursorModeLiteral,
            RateLimitStrategyLiteral,
            QueueFullStrategyLiteral,
            ErrorTypeLiteral,
        ]
        for literal_type in literal_types:
            assert isinstance(literal_type, LiteralForm), (
                f"{literal_type!r} should be an instance of LiteralForm Protocol"
            )

    def test_literal_form_rejects_plain_objects(self) -> None:
        """Plain objects without __origin__ are not instances of LiteralForm."""
        assert not isinstance("not a literal", LiteralForm)
        assert not isinstance(42, LiteralForm)
        assert not isinstance(object(), LiteralForm)

    def test_literal_forms_have_origin_attribute(self) -> None:
        """Literal forms have __origin__ set to typing.Literal."""
        assert hasattr(AgentTypeLiteral, "__origin__")
        assert AgentTypeLiteral.__origin__ is Literal
