"""Tests for Jinja2 template macros.

Unit tests for status_badge, stat_or_dash, and humanize_seconds_ago_detailed
macros in macros.html. Refactored to use BeautifulSoup for robust HTML parsing.
Added helper assertion functions for reduced test repetition.

This module tests the Jinja2 macros used in dashboard templates to ensure
proper badge rendering for different states, correct stat-or-dash
placeholder behavior, and detailed time formatting with sub-unit precision.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from bs4.element import Tag
from jinja2 import Environment, FileSystemLoader


# Helper assertion functions for badge testing
# Refactored helpers to use assert_badge_exists internally
def assert_badge_exists(soup: BeautifulSoup) -> Tag:
    """Assert that a badge element exists in the parsed HTML.

    Args:
        soup: BeautifulSoup object containing the rendered HTML

    Returns:
        The badge element if found

    Raises:
        AssertionError: If badge element is not found
    """
    badge = soup.find("span", class_="badge")
    assert badge is not None, "Badge element not found"
    return badge


def assert_badge_has_class(soup: BeautifulSoup, expected_class: str) -> None:
    """Assert that the badge element has the expected CSS class.

    Args:
        soup: BeautifulSoup object containing the rendered HTML
        expected_class: The CSS class expected to be present on the badge

    Raises:
        AssertionError: If badge is not found or doesn't have the expected class
    """
    badge = assert_badge_exists(soup)
    assert expected_class in badge.get(
        "class", []
    ), f"Expected class '{expected_class}' not found in badge classes: {badge.get('class', [])}"


def assert_badge_not_has_class(soup: BeautifulSoup, unexpected_class: str) -> None:
    """Assert that the badge element does NOT have the specified CSS class.

    Args:
        soup: BeautifulSoup object containing the rendered HTML
        unexpected_class: The CSS class that should NOT be present on the badge

    Raises:
        AssertionError: If badge is not found or has the unexpected class
    """
    badge = assert_badge_exists(soup)
    assert unexpected_class not in badge.get(
        "class", []
    ), f"Unexpected class '{unexpected_class}' found in badge classes: {badge.get('class', [])}"


def assert_badge_text(soup: BeautifulSoup, expected_text: str) -> None:
    """Assert that the badge element contains the expected text.

    Args:
        soup: BeautifulSoup object containing the rendered HTML
        expected_text: The text expected to be displayed in the badge

    Raises:
        AssertionError: If badge is not found or text doesn't match
    """
    badge = assert_badge_exists(soup)
    assert badge.text == expected_text, f"Expected badge text '{expected_text}', got '{badge.text}'"


class TestStatusBadgeMacro:
    """Tests for the status_badge Jinja2 macro.

    The status_badge macro renders a badge with automatic state-based styling:
    - badge--inactive: When active_count is 0 (nothing active)
    - badge--success: When active_count equals total_count (all active)
    - (default): When partially active (some but not all)

    Tests use BeautifulSoup for semantic HTML assertions instead of
    fragile string matching.
    """

    @pytest.fixture
    def jinja_env(self) -> Environment:
        """Create a Jinja2 environment with the templates directory."""
        templates_dir = (
            Path(__file__).parent.parent.parent / "src" / "sentinel" / "dashboard" / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        return env

    @pytest.fixture
    def render_status_badge(self, jinja_env: Environment) -> Callable[[int, int], str]:
        """Create a helper function to render the status_badge macro."""
        # Load the macros template
        jinja_env.get_template("macros.html")

        # Create a template that imports and uses the macro
        def render(active_count: int, total_count: int) -> str:
            template_str = """
{%- from "macros.html" import status_badge -%}
{{ status_badge(active_count, total_count) }}
"""
            template = jinja_env.from_string(template_str)
            return template.render(active_count=active_count, total_count=total_count).strip()

        return render

    @pytest.fixture
    def parse_badge(
        self, render_status_badge: Callable[[int, int], str]
    ) -> Callable[[int, int], BeautifulSoup]:
        """Create a helper function to render and parse a status badge."""

        def parse(active_count: int, total_count: int) -> BeautifulSoup:
            result = render_status_badge(active_count=active_count, total_count=total_count)
            return BeautifulSoup(result, "html.parser")

        return parse

    def test_badge_inactive_when_active_count_is_zero(self, parse_badge) -> None:
        """Test badge--inactive state when active_count is 0."""
        soup = parse_badge(active_count=0, total_count=5)

        assert_badge_has_class(soup, "badge--inactive")
        assert_badge_text(soup, "0/5")

    @pytest.mark.parametrize("total_count", [1, 10, 100])
    def test_badge_inactive_with_zero_active_and_various_totals(
        self, parse_badge, total_count: int
    ) -> None:
        """Test badge--inactive state with various total_count values."""
        soup = parse_badge(active_count=0, total_count=total_count)

        assert_badge_has_class(soup, "badge--inactive")
        assert_badge_text(soup, f"0/{total_count}")

    def test_badge_success_when_active_equals_total(self, parse_badge) -> None:
        """Test badge--success state when active_count equals total_count."""
        soup = parse_badge(active_count=5, total_count=5)

        assert_badge_has_class(soup, "badge--success")
        assert_badge_text(soup, "5/5")

    @pytest.mark.parametrize("count", [1, 10, 100])
    def test_badge_success_with_various_equal_counts(self, parse_badge, count: int) -> None:
        """Test badge--success state with various matching counts."""
        soup = parse_badge(active_count=count, total_count=count)

        assert_badge_has_class(soup, "badge--success")
        assert_badge_text(soup, f"{count}/{count}")

    def test_badge_default_when_partially_active(self, parse_badge) -> None:
        """Test default state (no modifier class) when partially active."""
        soup = parse_badge(active_count=3, total_count=5)

        # Should have 'badge' class but NOT 'badge--inactive' or 'badge--success'
        assert_badge_has_class(soup, "badge")
        assert_badge_not_has_class(soup, "badge--inactive")
        assert_badge_not_has_class(soup, "badge--success")
        assert_badge_text(soup, "3/5")

    @pytest.mark.parametrize(
        "active_count,total_count",
        [
            (1, 5),
            (4, 5),  # almost full
            (50, 100),
        ],
    )
    def test_badge_default_with_various_partial_counts(
        self, parse_badge, active_count: int, total_count: int
    ) -> None:
        """Test default state with various partial active counts."""
        soup = parse_badge(active_count=active_count, total_count=total_count)

        assert_badge_has_class(soup, "badge")
        assert_badge_not_has_class(soup, "badge--inactive")
        assert_badge_not_has_class(soup, "badge--success")
        assert_badge_text(soup, f"{active_count}/{total_count}")

    def test_edge_case_both_counts_zero(self, parse_badge) -> None:
        """Test edge case when total_count is 0.

        When both active_count and total_count are 0, the macro applies
        badge--inactive because the `active_count == 0` check comes first
        in the conditional chain. This means "0 active items" takes precedence
        over "all items are active" when there are no items at all.
        """
        soup = parse_badge(active_count=0, total_count=0)

        # When active_count is 0, badge--inactive is applied regardless of total_count
        # This is because `active_count == 0` is checked before `active_count == total_count`
        assert_badge_has_class(soup, "badge--inactive")
        assert_badge_text(soup, "0/0")

    def test_badge_renders_span_element(self, parse_badge) -> None:
        """Test that the badge is rendered as a span element."""
        soup = parse_badge(active_count=2, total_count=5)
        # Use assert_badge_exists which returns the badge, eliminating redundant soup.find call
        badge = assert_badge_exists(soup)
        assert badge.name == "span"

    def test_badge_displays_count_format(self, parse_badge) -> None:
        """Test that the badge displays counts in 'active/total' format."""
        soup = parse_badge(active_count=7, total_count=12)

        assert_badge_text(soup, "7/12")


# Helper assertion functions for stat_or_dash testing
def assert_dash_placeholder(soup: BeautifulSoup) -> None:
    """Assert that the rendered HTML contains a dash placeholder span.

    Args:
        soup: BeautifulSoup object containing the rendered HTML

    Raises:
        AssertionError: If dash placeholder is not found or has wrong content
    """
    span = soup.find("span", style="color: var(--text-secondary);")
    assert span is not None, "Dash placeholder span not found"
    assert span.text == "-", f"Expected dash '-', got '{span.text}'"


def assert_no_dash_placeholder(soup: BeautifulSoup) -> None:
    """Assert that the rendered HTML does NOT contain a dash placeholder span.

    Args:
        soup: BeautifulSoup object containing the rendered HTML

    Raises:
        AssertionError: If dash placeholder is unexpectedly found
    """
    span = soup.find("span", style="color: var(--text-secondary);")
    assert span is None, "Dash placeholder span should not be present when value is truthy"


class TestStatOrDashMacro:
    """Tests for the stat_or_dash Jinja2 macro.

    The stat_or_dash macro renders caller block content when the value is truthy,
    or a styled dash placeholder when the value is falsy. It uses the Jinja2
    call block pattern to pass the value back to the caller for rendering.

    Tests use BeautifulSoup for semantic HTML assertions instead of
    fragile string matching.
    """

    @pytest.fixture
    def jinja_env(self) -> Environment:
        """Create a Jinja2 environment with the templates directory."""
        templates_dir = (
            Path(__file__).parent.parent.parent / "src" / "sentinel" / "dashboard" / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        return env

    @pytest.fixture
    def render_stat_or_dash(self, jinja_env: Environment) -> Callable[..., str]:
        """Create a helper function to render the stat_or_dash macro with a value."""
        def render(value: object, content_template: str = "{{ v }}") -> str:
            template_str = (
                '{%- from "macros.html" import stat_or_dash -%}'
                "{% call(v) stat_or_dash(value) %}"
                + content_template
                + "{% endcall %}"
            )
            template = jinja_env.from_string(template_str)
            return template.render(value=value).strip()

        return render

    @pytest.fixture
    def parse_stat(
        self, render_stat_or_dash: Callable[..., str]
    ) -> Callable[..., BeautifulSoup]:
        """Create a helper function to render and parse a stat_or_dash result."""

        def parse(value: object, content_template: str = "{{ v }}") -> BeautifulSoup:
            result = render_stat_or_dash(value=value, content_template=content_template)
            return BeautifulSoup(result, "html.parser")

        return parse

    def test_dash_placeholder_when_value_is_none(self, parse_stat) -> None:
        """Test that a dash placeholder is shown when value is None."""
        soup = parse_stat(value=None)
        assert_dash_placeholder(soup)

    def test_dash_placeholder_when_value_is_empty_string(self, parse_stat) -> None:
        """Test that a dash placeholder is shown when value is empty string."""
        soup = parse_stat(value="")
        assert_dash_placeholder(soup)

    def test_dash_placeholder_when_value_is_false(self, parse_stat) -> None:
        """Test that a dash placeholder is shown when value is False."""
        soup = parse_stat(value=False)
        assert_dash_placeholder(soup)

    def test_renders_content_when_value_is_truthy_string(self, render_stat_or_dash) -> None:
        """Test that caller content is rendered when value is a truthy string."""
        result = render_stat_or_dash(value="hello")
        assert "hello" in result

    def test_renders_content_when_value_is_truthy_number(self, render_stat_or_dash) -> None:
        """Test that caller content is rendered when value is a truthy number."""
        result = render_stat_or_dash(value=42)
        assert "42" in result

    def test_no_dash_when_value_is_truthy(self, parse_stat) -> None:
        """Test that no dash placeholder is present when value is truthy."""
        soup = parse_stat(value="some value")
        assert_no_dash_placeholder(soup)

    def test_value_passed_to_caller_block(self, render_stat_or_dash) -> None:
        """Test that the value is correctly passed to the caller block via v."""
        result = render_stat_or_dash(value=99, content_template="Count: {{ v }}")
        assert "Count: 99" in result

    def test_caller_block_with_format_filter(self, render_stat_or_dash) -> None:
        """Test that format filters work within the caller block."""
        result = render_stat_or_dash(
            value=3.14159, content_template='{{ "%.1f" | format(v) }}s'
        )
        assert "3.1s" in result

    def test_dash_placeholder_html_structure(self, parse_stat) -> None:
        """Test the exact HTML structure of the dash placeholder."""
        soup = parse_stat(value=None)
        span = soup.find("span")
        assert span is not None, "Span element not found"
        assert span.get("style") == "color: var(--text-secondary);"
        assert span.text == "-"

    @pytest.mark.parametrize("falsy_value", [None, "", False, 0, []])
    def test_dash_shown_for_various_falsy_values(self, parse_stat, falsy_value) -> None:
        """Test that dash placeholder is shown for various falsy values."""
        soup = parse_stat(value=falsy_value)
        assert_dash_placeholder(soup)

    @pytest.mark.parametrize("truthy_value", ["text", 1, True, [1], {"key": "val"}])
    def test_content_shown_for_various_truthy_values(self, parse_stat, truthy_value) -> None:
        """Test that caller content is rendered for various truthy values."""
        soup = parse_stat(value=truthy_value)
        assert_no_dash_placeholder(soup)


class TestHumanizeSecondsAgoDetailedMacro:
    """Tests for the humanize_seconds_ago_detailed Jinja2 macro (DS-833).

    The humanize_seconds_ago_detailed macro renders a relative time string
    with sub-unit precision, complementing the compact humanize_seconds_ago
    macro. For example:
    - 119 seconds renders as "1m 59s ago" (vs "1m ago" in compact)
    - 3661 seconds renders as "1h 1m ago" (vs "1h ago" in compact)
    - Values under 60 seconds and exact boundaries remain the same

    Tests verify correct output for all time ranges and boundary transitions.
    """

    @pytest.fixture
    def jinja_env(self) -> Environment:
        """Create a Jinja2 environment with the templates directory."""
        templates_dir = (
            Path(__file__).parent.parent.parent / "src" / "sentinel" / "dashboard" / "templates"
        )
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        return env

    @pytest.fixture
    def render_detailed(self, jinja_env: Environment) -> Callable[[int], str]:
        """Create a helper function to render the humanize_seconds_ago_detailed macro."""
        jinja_env.get_template("macros.html")

        def render(seconds_ago: int) -> str:
            template_str = """
{%- from "macros.html" import humanize_seconds_ago_detailed -%}
{{ humanize_seconds_ago_detailed(seconds_ago) }}
"""
            template = jinja_env.from_string(template_str)
            return template.render(seconds_ago=seconds_ago).strip()

        return render

    def test_zero_seconds(self, render_detailed) -> None:
        """Test that 0 seconds renders as '0s ago'."""
        assert render_detailed(0) == "0s ago"

    def test_seconds_only(self, render_detailed) -> None:
        """Test that values under 60 seconds render as 'Xs ago'."""
        assert render_detailed(45) == "45s ago"

    def test_one_second(self, render_detailed) -> None:
        """Test that 1 second renders as '1s ago'."""
        assert render_detailed(1) == "1s ago"

    def test_59_seconds(self, render_detailed) -> None:
        """Test boundary just below 1 minute."""
        assert render_detailed(59) == "59s ago"

    def test_exact_one_minute(self, render_detailed) -> None:
        """Test exact 60 seconds renders as '1m ago' (no sub-unit)."""
        assert render_detailed(60) == "1m ago"

    def test_minutes_with_seconds(self, render_detailed) -> None:
        """Test that minutes with remaining seconds show sub-unit precision."""
        assert render_detailed(119) == "1m 59s ago"

    def test_minutes_with_seconds_mid_range(self, render_detailed) -> None:
        """Test mid-range minutes with seconds precision."""
        assert render_detailed(90) == "1m 30s ago"

    def test_exact_minutes_no_seconds(self, render_detailed) -> None:
        """Test that exact minute boundaries show no sub-unit."""
        assert render_detailed(120) == "2m ago"

    def test_exact_one_hour(self, render_detailed) -> None:
        """Test exact 3600 seconds renders as '1h ago'."""
        assert render_detailed(3600) == "1h ago"

    def test_hours_with_minutes(self, render_detailed) -> None:
        """Test that hours with remaining minutes show sub-unit precision."""
        assert render_detailed(3661) == "1h 1m ago"

    def test_hours_with_minutes_larger(self, render_detailed) -> None:
        """Test larger hour values with minute precision."""
        assert render_detailed(7500) == "2h 5m ago"

    def test_hours_exact_no_minutes(self, render_detailed) -> None:
        """Test that exact hour boundaries show no sub-unit."""
        assert render_detailed(7200) == "2h ago"

    def test_hours_with_minutes_and_seconds_drops_seconds(self, render_detailed) -> None:
        """Test that hours with minutes and seconds drops the seconds component.

        When the value is in the hours range, only hours and minutes are shown.
        Seconds are dropped to keep the output readable while still providing
        more precision than the compact macro.
        """
        # 1h 1m 1s -> "1h 1m ago" (seconds dropped at hour scale)
        assert render_detailed(3661) == "1h 1m ago"

    @pytest.mark.parametrize(
        "seconds_ago,expected",
        [
            (0, "0s ago"),
            (1, "1s ago"),
            (30, "30s ago"),
            (59, "59s ago"),
            (60, "1m ago"),
            (61, "1m 1s ago"),
            (90, "1m 30s ago"),
            (119, "1m 59s ago"),
            (120, "2m ago"),
            (3599, "59m 59s ago"),
            (3600, "1h ago"),
            (3601, "1h ago"),
            (3660, "1h 1m ago"),
            (7200, "2h ago"),
            (7260, "2h 1m ago"),
            (86400, "24h ago"),
        ],
    )
    def test_detailed_parametrized_boundaries(
        self, render_detailed, seconds_ago: int, expected: str
    ) -> None:
        """Test boundary values and transitions for detailed time formatting."""
        assert render_detailed(seconds_ago) == expected

    def test_float_input_is_truncated(self, render_detailed) -> None:
        """Test that float input is truncated to integer (like compact macro)."""
        assert render_detailed(90.7) == "1m 30s ago"
