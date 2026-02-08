"""Tests for GroupedOrchestrations and group_orchestrations_by_source (DS-751).

Unit tests covering the GroupedOrchestrations NamedTuple and the
PollCoordinator.group_orchestrations_by_source() method introduced in DS-750.
These tests guard against regressions by explicitly verifying:

1. GroupedOrchestrations NamedTuple construction and field access (.jira, .github)
2. Backward-compatible tuple unpacking of GroupedOrchestrations
3. group_orchestrations_by_source() correctly separating Jira vs GitHub orchestrations
4. Edge cases: empty list input, all-Jira, all-GitHub, mixed sources
5. Docstring note about positional unpacking impact when adding new fields.

Origin: Code review of DS-750 / PR #740.
"""

from __future__ import annotations

from sentinel.orchestration import Orchestration
from sentinel.poll_coordinator import GroupedOrchestrations, PollCoordinator
from tests.helpers import make_config, make_orchestration


class TestGroupedOrchestrationsNamedTuple:
    """Tests for GroupedOrchestrations NamedTuple construction and field access."""

    def test_construction_with_keyword_arguments(self) -> None:
        """GroupedOrchestrations can be constructed with keyword arguments."""
        jira_list: list[Orchestration] = [make_orchestration(name="jira-orch")]
        github_list: list[Orchestration] = [make_orchestration(name="gh-orch", source="github")]

        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)

        assert grouped.jira is jira_list
        assert grouped.github is github_list

    def test_construction_with_positional_arguments(self) -> None:
        """GroupedOrchestrations can be constructed with positional arguments."""
        jira_list: list[Orchestration] = [make_orchestration(name="jira-orch")]
        github_list: list[Orchestration] = [make_orchestration(name="gh-orch", source="github")]

        grouped = GroupedOrchestrations(jira_list, github_list)

        assert grouped.jira is jira_list
        assert grouped.github is github_list

    def test_field_access_jira(self) -> None:
        """GroupedOrchestrations.jira returns the jira orchestrations list."""
        jira_orchs = [make_orchestration(name="j1"), make_orchestration(name="j2")]
        grouped = GroupedOrchestrations(jira=jira_orchs, github=[])

        assert grouped.jira == jira_orchs
        assert len(grouped.jira) == 2

    def test_field_access_github(self) -> None:
        """GroupedOrchestrations.github returns the github orchestrations list."""
        github_orchs = [
            make_orchestration(name="g1", source="github"),
            make_orchestration(name="g2", source="github"),
        ]
        grouped = GroupedOrchestrations(jira=[], github=github_orchs)

        assert grouped.github == github_orchs
        assert len(grouped.github) == 2

    def test_empty_lists(self) -> None:
        """GroupedOrchestrations can hold empty lists for both fields."""
        grouped = GroupedOrchestrations(jira=[], github=[])

        assert grouped.jira == []
        assert grouped.github == []

    def test_is_named_tuple(self) -> None:
        """GroupedOrchestrations is a NamedTuple with expected field names."""
        assert hasattr(GroupedOrchestrations, "_fields")
        assert GroupedOrchestrations._fields == ("jira", "github")

    def test_index_access_matches_fields(self) -> None:
        """Index-based access matches field names (index 0 = jira, index 1 = github)."""
        jira_list: list[Orchestration] = [make_orchestration(name="j1")]
        github_list: list[Orchestration] = [make_orchestration(name="g1", source="github")]

        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)

        assert grouped[0] is jira_list
        assert grouped[1] is github_list


class TestGroupedOrchestrationsTupleUnpacking:
    """Tests for backward-compatible tuple unpacking of GroupedOrchestrations."""

    def test_tuple_unpacking_two_variables(self) -> None:
        """GroupedOrchestrations supports unpacking into two variables."""
        jira_list: list[Orchestration] = [make_orchestration(name="j1")]
        github_list: list[Orchestration] = [make_orchestration(name="g1", source="github")]

        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)
        jira_result, github_result = grouped

        assert jira_result is jira_list
        assert github_result is github_list

    def test_tuple_unpacking_preserves_order(self) -> None:
        """Tuple unpacking yields jira first, github second."""
        jira_orchs = [make_orchestration(name="jira-orch")]
        github_orchs = [make_orchestration(name="github-orch", source="github")]

        grouped = GroupedOrchestrations(jira=jira_orchs, github=github_orchs)
        first, second = grouped

        assert first is jira_orchs
        assert second is github_orchs

    def test_tuple_unpacking_with_empty_lists(self) -> None:
        """Tuple unpacking works correctly with empty lists."""
        grouped = GroupedOrchestrations(jira=[], github=[])
        jira_result, github_result = grouped

        assert jira_result == []
        assert github_result == []

    def test_len_returns_two(self) -> None:
        """GroupedOrchestrations has length 2 (number of fields)."""
        grouped = GroupedOrchestrations(jira=[], github=[])
        assert len(grouped) == 2

    def test_iteration_yields_both_lists(self) -> None:
        """Iterating over GroupedOrchestrations yields jira then github lists."""
        jira_list: list[Orchestration] = [make_orchestration(name="j1")]
        github_list: list[Orchestration] = [make_orchestration(name="g1", source="github")]

        grouped = GroupedOrchestrations(jira=jira_list, github=github_list)
        items = list(grouped)

        assert len(items) == 2
        assert items[0] is jira_list
        assert items[1] is github_list


class TestGroupOrchestrationsEdgeCases:
    """Edge case tests for group_orchestrations_by_source."""

    def _make_coordinator(self) -> PollCoordinator:
        """Create a PollCoordinator with minimal config for testing."""
        config = make_config()
        return PollCoordinator(config)

    def test_empty_list_returns_empty_groups(self) -> None:
        """Empty input list produces empty jira and github lists."""
        coordinator = self._make_coordinator()

        result = coordinator.group_orchestrations_by_source([])

        assert result.jira == []
        assert result.github == []

    def test_all_jira_orchestrations(self) -> None:
        """All Jira-source orchestrations are grouped into .jira."""
        coordinator = self._make_coordinator()
        jira_orchs = [
            make_orchestration(name="j1", source="jira"),
            make_orchestration(name="j2", source="jira"),
            make_orchestration(name="j3", source="jira"),
        ]

        result = coordinator.group_orchestrations_by_source(jira_orchs)

        assert len(result.jira) == 3
        assert result.github == []
        assert all(o.trigger.source == "jira" for o in result.jira)

    def test_all_github_orchestrations(self) -> None:
        """All GitHub-source orchestrations are grouped into .github."""
        coordinator = self._make_coordinator()
        github_orchs = [
            make_orchestration(name="g1", source="github"),
            make_orchestration(name="g2", source="github"),
        ]

        result = coordinator.group_orchestrations_by_source(github_orchs)

        assert result.jira == []
        assert len(result.github) == 2
        assert all(o.trigger.source == "github" for o in result.github)

    def test_mixed_sources_separated_correctly(self) -> None:
        """Mixed Jira and GitHub orchestrations are separated into correct groups."""
        coordinator = self._make_coordinator()
        orchestrations = [
            make_orchestration(name="j1", source="jira"),
            make_orchestration(name="g1", source="github"),
            make_orchestration(name="j2", source="jira"),
            make_orchestration(name="g2", source="github"),
            make_orchestration(name="j3", source="jira"),
        ]

        result = coordinator.group_orchestrations_by_source(orchestrations)

        assert len(result.jira) == 3
        assert len(result.github) == 2
        # Verify correct orchestrations are in each group
        jira_names = [o.name for o in result.jira]
        github_names = [o.name for o in result.github]
        assert jira_names == ["j1", "j2", "j3"]
        assert github_names == ["g1", "g2"]

    def test_preserves_orchestration_order_within_groups(self) -> None:
        """Orchestrations maintain their relative order within each group."""
        coordinator = self._make_coordinator()
        orchestrations = [
            make_orchestration(name="j-alpha", source="jira"),
            make_orchestration(name="g-first", source="github"),
            make_orchestration(name="j-beta", source="jira"),
            make_orchestration(name="g-second", source="github"),
            make_orchestration(name="j-gamma", source="jira"),
            make_orchestration(name="g-third", source="github"),
        ]

        result = coordinator.group_orchestrations_by_source(orchestrations)

        jira_names = [o.name for o in result.jira]
        github_names = [o.name for o in result.github]
        assert jira_names == ["j-alpha", "j-beta", "j-gamma"]
        assert github_names == ["g-first", "g-second", "g-third"]

    def test_result_is_grouped_orchestrations_type(self) -> None:
        """Return value is a GroupedOrchestrations instance."""
        coordinator = self._make_coordinator()

        result = coordinator.group_orchestrations_by_source([])

        assert isinstance(result, GroupedOrchestrations)

    def test_result_supports_tuple_unpacking(self) -> None:
        """Return value supports backward-compatible tuple unpacking."""
        coordinator = self._make_coordinator()
        orchestrations = [
            make_orchestration(name="j1", source="jira"),
            make_orchestration(name="g1", source="github"),
        ]

        jira_orchs, github_orchs = coordinator.group_orchestrations_by_source(orchestrations)

        assert len(jira_orchs) == 1
        assert len(github_orchs) == 1
        assert jira_orchs[0].name == "j1"
        assert github_orchs[0].name == "g1"

    def test_single_jira_orchestration(self) -> None:
        """Single Jira orchestration is correctly grouped."""
        coordinator = self._make_coordinator()
        orchestrations = [make_orchestration(name="solo-jira", source="jira")]

        result = coordinator.group_orchestrations_by_source(orchestrations)

        assert len(result.jira) == 1
        assert result.github == []
        assert result.jira[0].name == "solo-jira"

    def test_single_github_orchestration(self) -> None:
        """Single GitHub orchestration is correctly grouped."""
        coordinator = self._make_coordinator()
        orchestrations = [make_orchestration(name="solo-gh", source="github")]

        result = coordinator.group_orchestrations_by_source(orchestrations)

        assert result.jira == []
        assert len(result.github) == 1
        assert result.github[0].name == "solo-gh"

    def test_default_source_treated_as_jira(self) -> None:
        """Orchestration with default source (jira) is grouped into .jira."""
        coordinator = self._make_coordinator()
        # make_orchestration defaults to source="jira"
        orchestrations = [make_orchestration(name="default-source")]

        result = coordinator.group_orchestrations_by_source(orchestrations)

        assert len(result.jira) == 1
        assert result.github == []


class TestGroupedOrchestrationsDocstring:
    """Tests verifying the GroupedOrchestrations docstring guidance on positional unpacking.

    These tests document the contract that adding new fields to GroupedOrchestrations
    would break existing callers that use positional tuple unpacking. The docstring
    on GroupedOrchestrations should note this impact.
    """

    def test_docstring_mentions_positional_unpacking_impact(self) -> None:
        """GroupedOrchestrations docstring warns about impact of adding new fields."""
        docstring = GroupedOrchestrations.__doc__
        assert docstring is not None
        # The docstring should mention the NamedTuple's purpose and field access
        assert "NamedTuple" in docstring
        assert "trigger source" in docstring.lower() or "trigger" in docstring.lower()
        # DS-751: The docstring should warn about positional unpacking fragility
        assert "positional unpacking" in docstring.lower() or "positional" in docstring.lower()

    def test_fields_count_is_two(self) -> None:
        """GroupedOrchestrations has exactly two fields (jira, github).

        This test serves as a canary: if a new field is added,
        this test will fail, prompting a review of all callers
        that use positional tuple unpacking.
        """
        assert len(GroupedOrchestrations._fields) == 2
        assert GroupedOrchestrations._fields == ("jira", "github")
