"""Tests for agent_clients package __init__.py exports.

Validates that all expected symbols are exported from the agent_clients
package and are importable, following the pattern in test_validation.py.
"""

from __future__ import annotations


class TestAgentClientsModuleAll:
    """Tests for __all__ definition in agent_clients package."""

    def test_all_contains_expected_symbols(self) -> None:
        """Test that __all__ contains all expected public symbols."""
        from sentinel.agent_clients import __all__

        expected_symbols = [
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

        for symbol in expected_symbols:
            assert symbol in __all__, f"Expected {symbol} to be in __all__"

    def test_all_symbols_are_importable(self) -> None:
        """Test that all symbols in __all__ can be imported."""
        from sentinel import agent_clients
        from sentinel.agent_clients import __all__

        for symbol in __all__:
            assert hasattr(agent_clients, symbol), (
                f"Symbol {symbol} in __all__ but not importable"
            )

    def test_codex_agent_client_exported(self) -> None:
        """Test that CodexAgentClient is importable from agent_clients package."""
        from sentinel.agent_clients import CodexAgentClient

        assert CodexAgentClient is not None
        assert CodexAgentClient.__name__ == "CodexAgentClient"

    def test_codex_agent_client_in_all(self) -> None:
        """Test that CodexAgentClient is listed in __all__."""
        from sentinel.agent_clients import __all__

        assert "CodexAgentClient" in __all__

    def test_codex_agent_client_matches_source_module(self) -> None:
        """Test that CodexAgentClient from package matches the source module."""
        from sentinel.agent_clients import CodexAgentClient as PackageExport
        from sentinel.agent_clients.codex import CodexAgentClient as DirectImport

        assert PackageExport is DirectImport
