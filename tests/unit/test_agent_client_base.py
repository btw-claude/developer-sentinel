"""Unit tests for AgentClient base class shared methods.

Tests for the shared _build_prompt_with_context() helper method
extracted from concrete agent client implementations (DS-675).
"""

from __future__ import annotations

from typing import Any

from sentinel.agent_clients.base import AgentClient, AgentRunResult
from sentinel.types import AgentType


class _ConcreteAgentClient(AgentClient):
    """Minimal concrete subclass of AgentClient for testing base-class methods."""

    @property
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        return AgentType.CLAUDE

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
        agent_teams: bool = False,
    ) -> AgentRunResult:
        """Run agent (stub)."""
        return AgentRunResult(response="stub")


class TestBuildPromptWithContext:
    """Tests for AgentClient._build_prompt_with_context()."""

    def _make_client(self) -> _ConcreteAgentClient:
        """Create a concrete agent client instance for testing."""
        return _ConcreteAgentClient()

    def test_no_context_returns_prompt_unchanged(self) -> None:
        """Prompt is returned unchanged when context is None."""
        client = self._make_client()
        assert client._build_prompt_with_context("hello", None) == "hello"

    def test_empty_context_returns_prompt_unchanged(self) -> None:
        """Prompt is returned unchanged when context is an empty dict."""
        client = self._make_client()
        assert client._build_prompt_with_context("hello", {}) == "hello"

    def test_single_context_item(self) -> None:
        """A single context entry is appended with proper formatting."""
        client = self._make_client()
        result = client._build_prompt_with_context("hello", {"repo": "sentinel"})
        assert result == "hello\n\nContext:\n- repo: sentinel\n"

    def test_multiple_context_items(self) -> None:
        """Multiple context entries are appended as bullet items."""
        client = self._make_client()
        result = client._build_prompt_with_context(
            "prompt", {"org": "acme", "repo": "app"}
        )
        assert "\n\nContext:\n" in result
        assert "- org: acme\n" in result
        assert "- repo: app\n" in result

    def test_newlines_in_keys_are_replaced(self) -> None:
        """Newline characters in context keys are replaced with spaces."""
        client = self._make_client()
        result = client._build_prompt_with_context(
            "prompt", {"key\ninjected": "value"}
        )
        assert "key injected" in result
        assert "key\ninjected" not in result.split("Context:\n")[1]

    def test_newlines_in_values_are_replaced(self) -> None:
        """Newline characters in context values are replaced with spaces."""
        client = self._make_client()
        result = client._build_prompt_with_context(
            "prompt", {"key": "line1\nline2\nline3"}
        )
        assert "line1 line2 line3" in result

    def test_carriage_returns_are_stripped(self) -> None:
        r"""Carriage return characters (\\r) are removed from keys and values."""
        client = self._make_client()
        result = client._build_prompt_with_context(
            "prompt", {"key\r\ntest": "val\r\nue"}
        )
        context_section = result.split("Context:\n")[1]
        assert "\r" not in context_section

    def test_keys_truncated_to_200_chars(self) -> None:
        """Context keys longer than 200 characters are truncated."""
        client = self._make_client()
        long_key = "k" * 500
        result = client._build_prompt_with_context("prompt", {long_key: "val"})
        context_section = result.split("Context:\n")[1]
        assert "k" * 200 in context_section
        assert "k" * 201 not in context_section

    def test_values_truncated_to_2000_chars(self) -> None:
        """Context values longer than 2000 characters are truncated."""
        client = self._make_client()
        long_val = "v" * 5000
        result = client._build_prompt_with_context("prompt", {"key": long_val})
        context_section = result.split("Context:\n")[1]
        assert "v" * 2000 in context_section
        assert "v" * 2001 not in context_section

    def test_non_string_keys_are_converted(self) -> None:
        """Non-string context keys are converted to strings."""
        client = self._make_client()
        result = client._build_prompt_with_context("prompt", {42: "val"})
        assert "- 42: val\n" in result

    def test_non_string_values_are_converted(self) -> None:
        """Non-string context values are converted to strings."""
        client = self._make_client()
        result = client._build_prompt_with_context("prompt", {"key": 12345})
        assert "- key: 12345\n" in result

    def test_combined_sanitization(self) -> None:
        """Newline stripping and truncation work together correctly."""
        client = self._make_client()
        # Value with newlines that is also over 2000 chars
        long_val_with_newlines = ("a" * 100 + "\n") * 30  # 3030 chars before sanitization
        result = client._build_prompt_with_context(
            "prompt", {"key": long_val_with_newlines}
        )
        context_section = result.split("Context:\n")[1]
        # No raw newlines in the value portion (only the trailing \n from formatting)
        value_part = context_section.split(": ", 1)[1].rstrip("\n")
        assert "\n" not in value_part
        # Value should be truncated to 2000 chars
        assert len(value_part) <= 2000

    def test_context_key_max_length_class_attribute(self) -> None:
        """Verify _CONTEXT_KEY_MAX_LENGTH class attribute default value."""
        assert AgentClient._CONTEXT_KEY_MAX_LENGTH == 200

    def test_context_value_max_length_class_attribute(self) -> None:
        """Verify _CONTEXT_VALUE_MAX_LENGTH class attribute default value."""
        assert AgentClient._CONTEXT_VALUE_MAX_LENGTH == 2000
