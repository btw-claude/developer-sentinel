"""Tests for orchestration GitHub trigger configuration validation.

This module contains tests for:
- GitHub Project trigger configuration
- Validation of GitHub-specific trigger fields
"""

from pathlib import Path

import pytest

from sentinel.orchestration import OrchestrationError, load_orchestration_file


class TestGitHubProjectTrigger:
    """Tests for GitHub Project-based trigger configuration.

    These tests verify the GitHub Project-based polling fields:
    - project_number: Required positive integer for GitHub triggers
    - project_scope: "org" or "user" scope for the project
    - project_owner: Required organization name or username
    - project_filter: JQL-like query for filtering by project field values
    """

    def test_github_trigger_with_project_config(self, tmp_path: Path) -> None:
        """Should load GitHub trigger with project-based configuration."""
        yaml_content = """
orchestrations:
  - name: "github-project-trigger"
    trigger:
      source: github
      project_number: 123
      project_scope: org
      project_owner: "my-organization"
      project_filter: "Status = 'Ready for Review'"
    agent:
      prompt: "Process project item"
"""
        file_path = tmp_path / "project_trigger.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.trigger.source == "github"
        assert orch.trigger.project_number == 123
        assert orch.trigger.project_scope == "org"
        assert orch.trigger.project_owner == "my-organization"
        assert orch.trigger.project_filter == "Status = 'Ready for Review'"

    def test_github_trigger_user_scope(self, tmp_path: Path) -> None:
        """Should load GitHub trigger with user-scoped project."""
        yaml_content = """
orchestrations:
  - name: "user-project-trigger"
    trigger:
      source: github
      project_number: 5
      project_scope: user
      project_owner: "myusername"
      project_filter: "Priority = 'High'"
    agent:
      prompt: "Process user project item"
"""
        file_path = tmp_path / "user_project.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.trigger.project_scope == "user"
        assert orch.trigger.project_owner == "myusername"

    def test_github_trigger_missing_project_number_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when GitHub trigger is missing project_number."""
        yaml_content = """
orchestrations:
  - name: "missing-project-number"
    trigger:
      source: github
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "missing_project_number.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="require 'project_number' to be set"):
            load_orchestration_file(file_path)

    def test_github_trigger_invalid_project_number_zero(self, tmp_path: Path) -> None:
        """Should raise error when project_number is zero."""
        yaml_content = """
orchestrations:
  - name: "zero-project-number"
    trigger:
      source: github
      project_number: 0
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "zero_project_number.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid project_number"):
            load_orchestration_file(file_path)

    def test_github_trigger_invalid_project_number_negative(self, tmp_path: Path) -> None:
        """Should raise error when project_number is negative."""
        yaml_content = """
orchestrations:
  - name: "negative-project-number"
    trigger:
      source: github
      project_number: -5
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "negative_project_number.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid project_number"):
            load_orchestration_file(file_path)

    def test_github_trigger_invalid_project_number_string(self, tmp_path: Path) -> None:
        """Should raise error when project_number is not an integer."""
        yaml_content = """
orchestrations:
  - name: "string-project-number"
    trigger:
      source: github
      project_number: "forty-two"
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "string_project_number.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid project_number"):
            load_orchestration_file(file_path)

    def test_github_trigger_invalid_project_scope(self, tmp_path: Path) -> None:
        """Should raise error when project_scope is invalid."""
        yaml_content = """
orchestrations:
  - name: "invalid-scope"
    trigger:
      source: github
      project_number: 42
      project_scope: "team"
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_scope.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid project_scope"):
            load_orchestration_file(file_path)

    def test_github_trigger_missing_project_owner_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when GitHub trigger is missing project_owner."""
        yaml_content = """
orchestrations:
  - name: "missing-owner"
    trigger:
      source: github
      project_number: 42
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "missing_owner.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="require 'project_owner' to be set"):
            load_orchestration_file(file_path)

    def test_github_trigger_empty_project_filter_allowed(self, tmp_path: Path) -> None:
        """Should allow empty project_filter (no filtering)."""
        yaml_content = """
orchestrations:
  - name: "no-filter"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "Process all items"
"""
        file_path = tmp_path / "no_filter.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.project_filter == ""

    def test_github_trigger_default_project_scope_is_org(self, tmp_path: Path) -> None:
        """Should default project_scope to 'org' when not specified."""
        yaml_content = """
orchestrations:
  - name: "default-scope"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "default_scope.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.project_scope == "org"

    def test_jira_trigger_does_not_require_project_number(self, tmp_path: Path) -> None:
        """Jira triggers should not require GitHub project fields."""
        yaml_content = """
orchestrations:
  - name: "jira-trigger"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Process Jira issue"
"""
        file_path = tmp_path / "jira_trigger.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.source == "jira"
        assert orchestrations[0].trigger.project_number is None

    def test_github_trigger_with_complex_project_filter(self, tmp_path: Path) -> None:
        """Should support complex project_filter expressions."""
        yaml_content = """
orchestrations:
  - name: "complex-filter"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      project_filter: "Status = 'In Progress' AND Priority = 'High' AND Assignee != null"
    agent:
      prompt: "Process high priority items"
"""
        file_path = tmp_path / "complex_filter.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.project_filter == (
            "Status = 'In Progress' AND Priority = 'High' AND Assignee != null"
        )

    def test_github_trigger_project_number_one_is_valid(self, tmp_path: Path) -> None:
        """Should allow project_number of 1 (minimum valid value)."""
        yaml_content = """
orchestrations:
  - name: "min-project-number"
    trigger:
      source: github
      project_number: 1
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "min_project_number.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.project_number == 1

    def test_github_trigger_labels_field_parsed(self, tmp_path: Path) -> None:
        """Should load GitHub trigger with labels field."""
        yaml_content = """
orchestrations:
  - name: "labels-trigger"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        - "bug"
        - "urgent"
    agent:
      prompt: "Triage bugs"
"""
        file_path = tmp_path / "labels_trigger.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.labels == ["bug", "urgent"]

    def test_github_trigger_empty_labels_list_default(self, tmp_path: Path) -> None:
        """Should default labels to empty list when not specified."""
        yaml_content = """
orchestrations:
  - name: "no-labels"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "Process all"
"""
        file_path = tmp_path / "no_labels.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.labels == []

    def test_github_trigger_labels_combined_with_project_filter(self, tmp_path: Path) -> None:
        """Should support labels combined with project_filter."""
        yaml_content = """
orchestrations:
  - name: "combined-filter"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      project_filter: "Status = 'Ready'"
      labels:
        - "needs-triage"
        - "bug"
    agent:
      prompt: "Triage ready bugs"
"""
        file_path = tmp_path / "combined_filter.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.trigger.project_filter == "Status = 'Ready'"
        assert orch.trigger.labels == ["needs-triage", "bug"]
