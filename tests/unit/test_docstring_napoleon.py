"""Tests for Sphinx Napoleon compatibility of Google-style docstrings.

Validates that blank-line-separated parameter groups in test helper
docstrings render correctly when processed by Sphinx Napoleon. This is
a follow-up from DS-685 (PR #683) which replaced non-standard
``# -- Group Name --`` comment separators with standard Google-style
blank-line separation.

The tests verify that:

1. Napoleon successfully parses all documented parameters
2. Every function signature parameter has a corresponding docstring entry
3. Multi-line parameter descriptions are preserved after parsing
4. Blank-line group separators do not produce parsing artifacts
5. The overall docstring structure (summary, params, returns, examples) is intact

Usage:
    pytest tests/unit/test_docstring_napoleon.py -v
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from tests.helpers import make_config

# Skip entire module if sphinx is not installed
sphinx_napoleon = pytest.importorskip(
    "sphinx.ext.napoleon",
    reason="sphinx is required for Napoleon docstring validation",
)
GoogleDocstring = sphinx_napoleon.GoogleDocstring


def _get_docstring(func: Any) -> str:
    """Retrieve the cleaned docstring for a function.

    Args:
        func: The function to inspect.

    Returns:
        The cleaned docstring text.

    Raises:
        ValueError: If the function has no docstring.
    """
    docstring = inspect.getdoc(func)
    if docstring is None:
        msg = f"{func.__name__} has no docstring"
        raise ValueError(msg)
    return docstring


def _get_signature_params(func: Any) -> set[str]:
    """Extract parameter names from a function's signature.

    Args:
        func: The function to inspect.

    Returns:
        Set of parameter names from the function signature.
    """
    sig = inspect.signature(func)
    return set(sig.parameters.keys())


def _extract_docstring_params(docstring: str) -> set[str]:
    """Extract parameter names from a Google-style Args section.

    Parses the raw docstring (before Napoleon processing) to find
    all parameter names listed in the Args section.

    Args:
        docstring: The raw docstring text.

    Returns:
        Set of parameter names found in the Args section.
    """
    params: set[str] = set()
    in_args = False
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if in_args:
            if stripped.startswith(("Returns:", "Example:", "Raises:", "Yields:", "Note:")):
                break
            # Parameter lines have the form "param_name: description"
            # Continuation lines start with whitespace
            if ":" in stripped and not stripped.startswith(" "):
                param_name = stripped.split(":")[0].strip()
                if param_name:
                    params.add(param_name)
    return params


def _extract_napoleon_params(parsed: str) -> list[str]:
    """Extract parameter names from Napoleon-parsed reST output.

    Args:
        parsed: The Napoleon-parsed reST text.

    Returns:
        List of parameter names in order of appearance.
    """
    return re.findall(r":param (\w+):", parsed)


# -- Expected parameter groups for make_config() --
# These match the blank-line-separated groups in the docstring,
# which correspond to Config sub-config sections.
MAKE_CONFIG_PARAM_GROUPS: list[list[str]] = [
    # Polling settings
    ["poll_interval", "max_issues", "max_issues_per_poll"],
    # Execution settings
    [
        "max_concurrent_executions",
        "orchestrations_dir",
        "agent_workdir",
        "agent_logs_dir",
        "orchestration_logs_dir",
        "attempt_counts_ttl",
        "max_queue_size",
        "cleanup_workdir_on_success",
        "disable_streaming_logs",
        "subprocess_timeout",
        "default_base_branch",
        "inter_message_times_threshold",
    ],
    # Logging settings
    ["log_level", "log_json"],
    # Dashboard settings
    [
        "dashboard_enabled",
        "dashboard_port",
        "dashboard_host",
        "toggle_cooldown_seconds",
        "rate_limit_cache_ttl",
        "rate_limit_cache_maxsize",
    ],
    # Jira settings
    ["jira_base_url", "jira_email", "jira_api_token", "jira_epic_link_field"],
    # GitHub settings
    ["github_token", "github_api_url"],
    # Agent settings
    ["default_agent_type"],
    # Cursor settings
    ["cursor_path", "cursor_default_model", "cursor_default_mode"],
    # Codex settings
    ["codex_path", "codex_default_model"],
    # Rate limit settings
    [
        "claude_rate_limit_enabled",
        "claude_rate_limit_per_minute",
        "claude_rate_limit_per_hour",
        "claude_rate_limit_strategy",
        "claude_rate_limit_warning_threshold",
    ],
    # Circuit breaker settings
    [
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_recovery_timeout",
        "circuit_breaker_half_open_max_calls",
    ],
    # Health check settings
    ["health_check_enabled", "health_check_timeout"],
    # Shutdown settings
    ["shutdown_timeout_seconds"],
]


class TestMakeConfigNapoleonRendering:
    """Validate make_config() docstring renders correctly with Sphinx Napoleon."""

    @pytest.fixture
    def raw_docstring(self) -> str:
        """Get the raw make_config() docstring."""
        return _get_docstring(make_config)

    @pytest.fixture
    def parsed_rest(self, raw_docstring: str) -> str:
        """Get Napoleon-parsed reST output for make_config()."""
        return str(GoogleDocstring(raw_docstring))

    def test_napoleon_parses_without_error(self, raw_docstring: str) -> None:
        """Verify Napoleon parses the docstring without raising exceptions."""
        # GoogleDocstring should not raise during parsing
        result = str(GoogleDocstring(raw_docstring))
        assert len(result) > 0

    def test_all_signature_params_documented(self) -> None:
        """Verify every function parameter has a docstring entry."""
        sig_params = _get_signature_params(make_config)
        doc_params = _extract_docstring_params(_get_docstring(make_config))

        missing = sig_params - doc_params
        assert not missing, f"Parameters missing from docstring: {missing}"

    def test_no_extra_documented_params(self) -> None:
        """Verify no docstring params are absent from the function signature."""
        sig_params = _get_signature_params(make_config)
        doc_params = _extract_docstring_params(_get_docstring(make_config))

        extra = doc_params - sig_params
        assert not extra, f"Extra parameters in docstring not in signature: {extra}"

    def test_napoleon_extracts_all_params(self, parsed_rest: str) -> None:
        """Verify Napoleon produces :param directives for all parameters."""
        sig_params = _get_signature_params(make_config)
        napoleon_params = set(_extract_napoleon_params(parsed_rest))

        missing = sig_params - napoleon_params
        assert not missing, (
            f"Napoleon failed to extract parameters: {missing}. "
            "Blank-line group separators may be causing parsing issues."
        )

    def test_napoleon_param_count_matches_signature(self, parsed_rest: str) -> None:
        """Verify the count of Napoleon-parsed params matches the signature."""
        sig_params = _get_signature_params(make_config)
        napoleon_params = _extract_napoleon_params(parsed_rest)

        assert len(napoleon_params) == len(sig_params), (
            f"Expected {len(sig_params)} :param directives, "
            f"got {len(napoleon_params)}. "
            "Some parameters may have been lost during Napoleon parsing."
        )

    def test_napoleon_preserves_param_order(self, parsed_rest: str) -> None:
        """Verify Napoleon preserves the parameter documentation order."""
        napoleon_params = _extract_napoleon_params(parsed_rest)

        # Flatten the expected groups to get the expected order
        expected_order = [p for group in MAKE_CONFIG_PARAM_GROUPS for p in group]

        assert napoleon_params == expected_order, (
            "Napoleon changed the parameter order. "
            "Expected order matches the function signature grouping."
        )

    def test_multiline_description_preserved(self, parsed_rest: str) -> None:
        """Verify multi-line param descriptions survive Napoleon parsing.

        The shutdown_timeout_seconds parameter has a two-line description
        that should be preserved as a continuation in the :param directive.
        """
        assert "shutdown_timeout_seconds" in parsed_rest
        assert "Set to 0 to wait indefinitely" in parsed_rest

    def test_no_blank_line_artifacts_in_params(self, parsed_rest: str) -> None:
        """Verify blank-line group separators don't produce artifacts.

        The blank lines between parameter groups in the Args section should
        be cleanly consumed by Napoleon without producing empty :param
        directives, stray blank lines between :param blocks, or other
        formatting artifacts.
        """
        # No empty :param directives
        assert ":param :" not in parsed_rest
        # No :param with just whitespace
        assert re.search(r":param\s+:", parsed_rest) is None

    def test_returns_section_preserved(self, parsed_rest: str) -> None:
        """Verify the Returns section is correctly parsed."""
        assert ":returns:" in parsed_rest or ":return:" in parsed_rest
        assert "Config instance" in parsed_rest

    def test_example_section_preserved(self, parsed_rest: str) -> None:
        """Verify the Example section is correctly parsed."""
        assert "Example::" in parsed_rest or "config = make_config()" in parsed_rest

    def test_summary_preserved(self, parsed_rest: str) -> None:
        """Verify the docstring summary line is preserved."""
        assert "Create a Config instance for testing with sensible defaults" in parsed_rest


class TestDocstringGroupStructure:
    """Validate the blank-line-separated parameter groups in the raw docstring."""

    @pytest.fixture
    def raw_docstring(self) -> str:
        """Get the raw make_config() docstring."""
        return _get_docstring(make_config)

    def test_args_section_has_blank_line_groups(self, raw_docstring: str) -> None:
        """Verify the Args section uses blank lines to separate param groups."""
        # Extract just the Args section
        in_args = False
        args_lines: list[str] = []
        for line in raw_docstring.split("\n"):
            if line.strip() == "Args:":
                in_args = True
                continue
            if in_args:
                if line.strip().startswith(("Returns:", "Example:", "Raises:")):
                    break
                args_lines.append(line)

        # Count blank lines (group separators)
        blank_line_count = sum(1 for line in args_lines if line.strip() == "")

        # We expect blank lines between the parameter groups
        # (number of groups - 1) + trailing blank line
        expected_groups = len(MAKE_CONFIG_PARAM_GROUPS)
        # At minimum, there should be (groups - 1) blank lines as separators
        assert blank_line_count >= expected_groups - 1, (
            f"Expected at least {expected_groups - 1} blank-line group separators "
            f"in Args section, found {blank_line_count}. "
            "Parameter groups may have lost their visual separation."
        )

    def test_param_groups_match_function_comments(self) -> None:
        """Verify parameter groups align with the function's inline comments.

        The make_config() function uses inline comments (e.g., '# Polling settings')
        to label parameter groups. The docstring's blank-line groups should contain
        the same parameters in the same order.
        """
        # Get the source of make_config to extract comment-based groups
        source = inspect.getsource(make_config)

        # Extract only the function signature (between 'def make_config(' and ') ->')
        sig_match = re.search(r"def make_config\((.*?)\)\s*->", source, re.DOTALL)
        assert sig_match is not None, "Could not find make_config function signature"
        sig_section = sig_match.group(1)

        # Extract parameters between each group comment
        group_comment_pattern = re.compile(r"#\s+(\w[\w\s]+)\s+settings", re.IGNORECASE)
        source_groups: list[list[str]] = []
        current_group: list[str] = []

        for line in sig_section.split("\n"):
            stripped = line.strip()
            # Check for group comment (e.g., "# Polling settings")
            if group_comment_pattern.search(stripped):
                if current_group:
                    source_groups.append(current_group)
                current_group = []
                continue
            # Check for parameter definition (name: type = default)
            param_match = re.match(r"(\w+)\s*:", stripped)
            if param_match:
                param_name = param_match.group(1)
                current_group.append(param_name)

        if current_group:
            source_groups.append(current_group)

        # Flatten both and verify they contain the same parameters
        expected_flat = [p for group in MAKE_CONFIG_PARAM_GROUPS for p in group]
        source_flat = [p for group in source_groups for p in group]

        assert source_flat == expected_flat, (
            "Parameter groups in MAKE_CONFIG_PARAM_GROUPS don't match "
            "the inline comment groups in make_config() function definition."
        )

    def test_no_non_standard_separators_in_docstring(self, raw_docstring: str) -> None:
        """Verify no non-standard separators remain in the docstring.

        DS-685 (PR #683) removed ``# -- Group Name --`` style separators.
        This test ensures they don't get reintroduced.
        """
        # Check for the old-style separators
        assert "# --" not in raw_docstring, (
            "Found non-standard '# --' separator in docstring. "
            "Use blank lines between parameter groups instead (DS-685)."
        )
        assert "-- #" not in raw_docstring, (
            "Found non-standard '-- #' separator in docstring. "
            "Use blank lines between parameter groups instead (DS-685)."
        )
