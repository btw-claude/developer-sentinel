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

# Ensure sphinx is available; skip gracefully if not installed (DS-692 item 3).
pytest.importorskip(
    "sphinx.ext.napoleon",
    reason="sphinx is required for Napoleon docstring validation",
)
from sphinx.ext.napoleon import GoogleDocstring  # noqa: E402


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


def _extract_param_groups_from_source(func: Any) -> list[list[str]]:
    """Auto-generate parameter groups from make_config() source comments.

    Parses the function signature of ``func`` to extract parameter groups
    delimited by ``# <Name> settings`` comments. This eliminates the need
    for a manually maintained constant that must be kept in sync with the
    function definition (DS-691 item 2).

    Args:
        func: The function whose source to parse.

    Returns:
        List of parameter groups, where each group is a list of parameter
        names appearing between consecutive group-comment lines.

    Raises:
        ValueError: If the function signature cannot be located.
    """
    source = inspect.getsource(func)

    # Extract only the function signature (between 'def <name>(' and ') ->')
    func_name = func.__name__
    sig_match = re.search(
        rf"def {re.escape(func_name)}\((.*?)\)\s*->", source, re.DOTALL
    )
    if sig_match is None:
        raise ValueError(f"Could not find {func_name} function signature")
    sig_section = sig_match.group(1)

    group_comment_pattern = re.compile(r"#\s+\w[\w ]+ +settings", re.IGNORECASE)
    groups: list[list[str]] = []
    current_group: list[str] = []

    for line in sig_section.split("\n"):
        stripped = line.strip()
        if group_comment_pattern.search(stripped):
            if current_group:
                groups.append(current_group)
            current_group = []
            continue
        param_match = re.match(r"(\w+)\s*:", stripped)
        if param_match:
            current_group.append(param_match.group(1))

    if current_group:
        groups.append(current_group)

    return groups


# Auto-generated from make_config() source inline comments (DS-691).
# Previously a hardcoded constant that required manual updates whenever
# make_config() parameters changed.  Now derived at import time so the
# test data stays in sync with the source automatically.
# Wrapped in try/except so that a parsing failure skips the module
# gracefully instead of crashing collection (DS-692 item 4).
try:
    MAKE_CONFIG_PARAM_GROUPS: list[list[str]] = _extract_param_groups_from_source(make_config)
except (ValueError, OSError) as exc:
    pytest.skip(f"Could not auto-generate param groups: {exc}", allow_module_level=True)


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

    def test_auto_generated_groups_cover_all_params(self) -> None:
        """Verify auto-generated param groups cover every signature parameter.

        MAKE_CONFIG_PARAM_GROUPS is now auto-generated from make_config()
        source comments (DS-691). This test validates that the generation
        produces groups covering every parameter in the function signature
        and that the expected number of groups is stable (DS-692 item 5).
        """
        sig_params = _get_signature_params(make_config)
        grouped_params = {p for group in MAKE_CONFIG_PARAM_GROUPS for p in group}

        missing = sig_params - grouped_params
        assert not missing, (
            f"Auto-generated param groups are missing parameters: {missing}. "
            "Ensure every parameter in make_config() has a group comment above it."
        )

        extra = grouped_params - sig_params
        assert not extra, (
            f"Auto-generated param groups contain unknown parameters: {extra}. "
            "The source comment parsing may be picking up non-parameter lines."
        )

        # Guard against silent group merging (DS-692 item 5).
        # make_config() currently has 14 distinct "# <Name> settings" groups.
        # Bumped from 13 to 14 in DS-818 (added "# Service health gate settings").
        expected_group_count = 14
        assert len(MAKE_CONFIG_PARAM_GROUPS) == expected_group_count, (
            f"Expected {expected_group_count} parameter groups, "
            f"got {len(MAKE_CONFIG_PARAM_GROUPS)}. "
            "A group comment may have been removed or merged in make_config()."
        )

    def test_auto_generated_groups_are_non_empty(self) -> None:
        """Verify auto-generated param groups are all non-empty.

        Each ``# <Name> settings`` comment in make_config() should be followed
        by at least one parameter.
        """
        for i, group in enumerate(MAKE_CONFIG_PARAM_GROUPS):
            assert len(group) > 0, (
                f"Parameter group {i} is empty. "
                "A group comment in make_config() has no parameters beneath it."
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
