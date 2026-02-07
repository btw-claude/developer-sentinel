"""Tests for agent_clients package __init__.py exports.

Validates that all expected symbols are exported from the agent_clients
package and are importable, following the pattern in test_validation.py.
"""

from __future__ import annotations

import importlib

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

# Mapping of each re-exported symbol to its source submodule within
# sentinel.agent_clients.  Used by the parametrized identity-check test
# to verify that the package re-export is the same object as the direct
# import from the source module.
_SYMBOL_SOURCE_MODULE: dict[str, str] = {
    # Base classes and types  (sentinel.agent_clients.base)
    "AgentClient": "sentinel.agent_clients.base",
    "AgentClientError": "sentinel.agent_clients.base",
    "AgentRunResult": "sentinel.agent_clients.base",
    "AgentTimeoutError": "sentinel.agent_clients.base",
    "AgentType": "sentinel.agent_clients.base",
    "UsageInfo": "sentinel.agent_clients.base",
    # Claude SDK client  (sentinel.agent_clients.claude_sdk)
    "ClaudeProcessInterruptedError": "sentinel.agent_clients.claude_sdk",
    "ClaudeSdkAgentClient": "sentinel.agent_clients.claude_sdk",
    "ShutdownController": "sentinel.agent_clients.claude_sdk",
    "TimingMetrics": "sentinel.agent_clients.claude_sdk",
    # Codex CLI client  (sentinel.agent_clients.codex)
    "CodexAgentClient": "sentinel.agent_clients.codex",
    # Cursor CLI client  (sentinel.agent_clients.cursor)
    "CursorAgentClient": "sentinel.agent_clients.cursor",
    "CursorMode": "sentinel.agent_clients.cursor",
    # Factory  (sentinel.agent_clients.factory)
    "AgentClientFactory": "sentinel.agent_clients.factory",
    "create_default_factory": "sentinel.agent_clients.factory",
}

# Parametrize list derived from the mapping — keeps the single source of
# truth in _SYMBOL_SOURCE_MODULE while giving pytest clear test IDs.
_IDENTITY_CHECK_PARAMS = [
    pytest.param(sym, mod, id=sym)
    for sym, mod in _SYMBOL_SOURCE_MODULE.items()
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

    def test_symbol_source_module_covers_expected_symbols(self) -> None:
        """Test that _SYMBOL_SOURCE_MODULE covers every EXPECTED_SYMBOLS entry.

        Ensures the identity-check parametrization stays in sync when new
        symbols are added to EXPECTED_SYMBOLS.
        """
        expected = set(EXPECTED_SYMBOLS)
        mapped = set(_SYMBOL_SOURCE_MODULE)
        assert expected == mapped, (
            f"_SYMBOL_SOURCE_MODULE and EXPECTED_SYMBOLS have drifted apart.\n"
            f"  In EXPECTED_SYMBOLS but not mapped: {expected - mapped}\n"
            f"  In mapped but not EXPECTED_SYMBOLS: {mapped - expected}"
        )

    @pytest.mark.parametrize("symbol", EXPECTED_SYMBOLS)
    def test_symbol_is_importable(self, symbol: str) -> None:
        """Test that the symbol in __all__ can be imported."""
        from sentinel import agent_clients

        assert hasattr(agent_clients, symbol), (
            f"Symbol {symbol} in __all__ but not importable"
        )

    @pytest.mark.parametrize("symbol, source_module", _IDENTITY_CHECK_PARAMS)
    def test_reexport_is_same_object_as_source(
        self, symbol: str, source_module: str
    ) -> None:
        """Test that each re-exported symbol is the same object as the source.

        Verifies that ``sentinel.agent_clients.<symbol>`` is identical
        (``is``) to ``<source_module>.<symbol>``, catching accidental
        shadowing or stale re-exports.
        """
        import sentinel.agent_clients as pkg

        src_mod = importlib.import_module(source_module)
        package_obj = getattr(pkg, symbol)
        source_obj = getattr(src_mod, symbol)
        assert package_obj is source_obj, (
            f"sentinel.agent_clients.{symbol} is not the same object as "
            f"{source_module}.{symbol}"
        )
