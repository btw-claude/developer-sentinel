"""Tests for Jinja2 template macros.

DS-280: Unit tests for status_badge macro in macros.html.

This module tests the Jinja2 macros used in dashboard templates to ensure
proper badge rendering for different states.
"""

from __future__ import annotations

import pytest
from jinja2 import Environment, FileSystemLoader
from pathlib import Path


class TestStatusBadgeMacro:
    """Tests for the status_badge Jinja2 macro (DS-280).

    The status_badge macro renders a badge with automatic state-based styling:
    - badge--inactive: When active_count is 0 (nothing active)
    - badge--success: When active_count equals total_count (all active)
    - (default): When partially active (some but not all)
    """

    @pytest.fixture
    def jinja_env(self) -> Environment:
        """Create a Jinja2 environment with the templates directory."""
        templates_dir = Path(__file__).parent.parent / "src" / "sentinel" / "dashboard" / "templates"
        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        return env

    @pytest.fixture
    def render_status_badge(self, jinja_env: Environment):
        """Create a helper function to render the status_badge macro."""
        # Load the macros template
        macros_template = jinja_env.get_template("macros.html")

        # Create a template that imports and uses the macro
        def render(active_count: int, total_count: int) -> str:
            template_str = """
{%- from "macros.html" import status_badge -%}
{{ status_badge(active_count, total_count) }}
"""
            template = jinja_env.from_string(template_str)
            return template.render(active_count=active_count, total_count=total_count).strip()

        return render

    def test_badge_inactive_when_active_count_is_zero(self, render_status_badge) -> None:
        """Test badge--inactive state when active_count is 0 (DS-280 requirement 1)."""
        result = render_status_badge(active_count=0, total_count=5)

        assert 'class="badge badge--inactive"' in result
        assert ">0/5<" in result

    @pytest.mark.parametrize("total_count", [1, 10, 100])
    def test_badge_inactive_with_zero_active_and_various_totals(self, render_status_badge, total_count: int) -> None:
        """Test badge--inactive state with various total_count values."""
        result = render_status_badge(active_count=0, total_count=total_count)
        assert 'badge--inactive' in result
        assert f">0/{total_count}<" in result

    def test_badge_success_when_active_equals_total(self, render_status_badge) -> None:
        """Test badge--success state when active_count equals total_count (DS-280 requirement 2)."""
        result = render_status_badge(active_count=5, total_count=5)

        assert 'class="badge badge--success"' in result
        assert ">5/5<" in result

    @pytest.mark.parametrize("count", [1, 10, 100])
    def test_badge_success_with_various_equal_counts(self, render_status_badge, count: int) -> None:
        """Test badge--success state with various matching counts."""
        result = render_status_badge(active_count=count, total_count=count)
        assert 'badge--success' in result
        assert f">{count}/{count}<" in result

    def test_badge_default_when_partially_active(self, render_status_badge) -> None:
        """Test default state (no modifier class) when partially active (DS-280 requirement 3)."""
        result = render_status_badge(active_count=3, total_count=5)

        # Should have 'badge' class but NOT 'badge--inactive' or 'badge--success'
        assert 'class="badge"' in result
        assert 'badge--inactive' not in result
        assert 'badge--success' not in result
        assert ">3/5<" in result

    @pytest.mark.parametrize(
        "active_count,total_count",
        [
            (1, 5),
            (4, 5),  # almost full
            (50, 100),
        ],
    )
    def test_badge_default_with_various_partial_counts(self, render_status_badge, active_count: int, total_count: int) -> None:
        """Test default state with various partial active counts."""
        result = render_status_badge(active_count=active_count, total_count=total_count)
        assert 'class="badge"' in result
        assert 'badge--inactive' not in result
        assert 'badge--success' not in result
        assert f">{active_count}/{total_count}<" in result

    def test_edge_case_both_counts_zero(self, render_status_badge) -> None:
        """Test edge case when total_count is 0 (DS-280 requirement 4).

        When both active_count and total_count are 0, the macro applies
        badge--inactive because the `active_count == 0` check comes first
        in the conditional chain. This means "0 active items" takes precedence
        over "all items are active" when there are no items at all.
        """
        result = render_status_badge(active_count=0, total_count=0)

        # When active_count is 0, badge--inactive is applied regardless of total_count
        # This is because `active_count == 0` is checked before `active_count == total_count`
        assert 'badge--inactive' in result
        assert ">0/0<" in result

    def test_badge_renders_span_element(self, render_status_badge) -> None:
        """Test that the badge is rendered as a span element."""
        result = render_status_badge(active_count=2, total_count=5)

        assert result.startswith('<span')
        assert result.endswith('</span>')

    def test_badge_displays_count_format(self, render_status_badge) -> None:
        """Test that the badge displays counts in 'active/total' format."""
        result = render_status_badge(active_count=7, total_count=12)

        assert ">7/12<" in result
