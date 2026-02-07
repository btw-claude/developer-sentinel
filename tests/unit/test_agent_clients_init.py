"""Tests for agent_clients package __init__.py exports.

Validates that all expected symbols are exported from the agent_clients
package and are importable, following the pattern in test_validation.py.
"""

from __future__ import annotations

import pytest

# Expected public symbols exported from agent_clients package
EXPECTED_SYMBOLS = [
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


class TestAgentClientsModuleAll:
    """Tests for __all__ definition in agent_clients package."""

    def test_expected_symbols_matches_all(self) -> None:
        """Test that EXPECTED_SYMBOLS matches __all__ exactly (drift detection).

        Catches both symbols missing from EXPECTED_SYMBOLS and symbols
        added to EXPECTED_SYMBOLS but not present in __all__.
        """
        from sentinel.agent_clients import __all__

        expected = set(EXPECTED_SYMBOLS)
        actual = set(__all__)
        assert expected == actual, (
            f"EXPECTED_SYMBOLS and __all__ have drifted apart.\n"
            f"  In EXPECTED_SYMBOLS but not __all__: {expected - actual}\n"
            f"  In __all__ but not EXPECTED_SYMBOLS: {actual - expected}"
        )

    @pytest.mark.parametrize("symbol", EXPECTED_SYMBOLS)
    def test_symbol_is_importable(self, symbol: str) -> None:
        """Test that the symbol in __all__ can be imported."""
        from sentinel import agent_clients

        assert hasattr(agent_clients, symbol), (
            f"Symbol {symbol} in __all__ but not importable"
        )

    def test_codex_agent_client_matches_source_module(self) -> None:
        """Test that CodexAgentClient from package matches the source module."""
        from sentinel.agent_clients import CodexAgentClient as PackageExport
        from sentinel.agent_clients.codex import CodexAgentClient as DirectImport

        assert PackageExport is DirectImport
