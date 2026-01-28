"""Tests for orchestration labels/tags validation and agent type configuration.

This module contains tests for:
- Labels and tags field validation in TriggerConfig
- Agent type configuration (claude vs cursor)
- Cursor mode configuration
"""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OrchestrationError,
    TriggerConfig,
    ValidationResult,
    _validate_github_repo_format,
    load_orchestration_file,
)


class TestLabelsAndTagsValidation:
    """Tests for labels and tags field validation in TriggerConfig.

    These tests verify that labels and tags fields are properly validated:
    - Must be a list if provided
    - Each item must be a non-empty string
    """

    def test_labels_not_list_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when labels is not a list."""
        yaml_content = """
orchestrations:
  - name: "invalid-labels"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels: "not-a-list"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_labels_type.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="labels must be a list"):
            load_orchestration_file(file_path)

    def test_labels_with_empty_string_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when labels contains an empty string."""
        yaml_content = """
orchestrations:
  - name: "empty-label"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        - "bug"
        - ""
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "empty_label.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="labels must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_labels_with_whitespace_only_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when labels contains whitespace-only string."""
        yaml_content = """
orchestrations:
  - name: "whitespace-label"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        - "   "
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "whitespace_label.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="labels must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_labels_with_non_string_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when labels contains a non-string value."""
        yaml_content = """
orchestrations:
  - name: "non-string-label"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        - "bug"
        - 123
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "non_string_label.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="labels must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_valid_labels_list_loads_successfully(self, tmp_path: Path) -> None:
        """Should load orchestration with valid labels list."""
        yaml_content = """
orchestrations:
  - name: "valid-labels"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        - "bug"
        - "urgent"
        - "needs-triage"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "valid_labels.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.labels == ["bug", "urgent", "needs-triage"]

    def test_tags_not_list_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when tags is not a list."""
        yaml_content = """
orchestrations:
  - name: "invalid-tags"
    trigger:
      source: jira
      project: "TEST"
      tags: "not-a-list"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_tags_type.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="tags must be a list"):
            load_orchestration_file(file_path)

    def test_tags_with_empty_string_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when tags contains an empty string."""
        yaml_content = """
orchestrations:
  - name: "empty-tag"
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "review"
        - ""
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "empty_tag.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="tags must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_tags_with_whitespace_only_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when tags contains whitespace-only string."""
        yaml_content = """
orchestrations:
  - name: "whitespace-tag"
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "   "
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "whitespace_tag.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="tags must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_tags_with_non_string_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when tags contains a non-string value."""
        yaml_content = """
orchestrations:
  - name: "non-string-tag"
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "review"
        - 456
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "non_string_tag.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="tags must contain non-empty strings"):
            load_orchestration_file(file_path)

    def test_valid_tags_list_loads_successfully(self, tmp_path: Path) -> None:
        """Should load orchestration with valid tags list."""
        yaml_content = """
orchestrations:
  - name: "valid-tags"
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "review"
        - "priority-high"
        - "backend"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "valid_tags.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.tags == ["review", "priority-high", "backend"]

    def test_empty_labels_list_allowed(self, tmp_path: Path) -> None:
        """Should allow empty labels list."""
        yaml_content = """
orchestrations:
  - name: "empty-labels"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels: []
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "empty_labels_list.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.labels == []

    def test_empty_tags_list_allowed(self, tmp_path: Path) -> None:
        """Should allow empty tags list."""
        yaml_content = """
orchestrations:
  - name: "empty-tags"
    trigger:
      source: jira
      project: "TEST"
      tags: []
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "empty_tags_list.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.tags == []

    def test_labels_with_dict_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when labels is a dict instead of list."""
        yaml_content = """
orchestrations:
  - name: "dict-labels"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      labels:
        key: value
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "dict_labels.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="labels must be a list"):
            load_orchestration_file(file_path)

    def test_tags_with_dict_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when tags is a dict instead of list."""
        yaml_content = """
orchestrations:
  - name: "dict-tags"
    trigger:
      source: jira
      project: "TEST"
      tags:
        key: value
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "dict_tags.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="tags must be a list"):
            load_orchestration_file(file_path)


class TestAgentType:
    """Tests for agent_type and cursor_mode fields in AgentConfig.

    These tests verify the new agent type selection fields:
    - agent_type: Optional field with values 'claude' or 'cursor'
    - cursor_mode: Optional field with values 'agent', 'plan', or 'ask'
      (only valid when agent_type is 'cursor')
    """

    def test_load_file_with_agent_type_claude(self, tmp_path: Path) -> None:
        """Should load orchestration with agent_type='claude'."""
        yaml_content = """
orchestrations:
  - name: "claude-agent"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
"""
        file_path = tmp_path / "claude_agent.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type == "claude"
        assert orchestrations[0].agent.cursor_mode is None

    def test_load_file_with_agent_type_cursor(self, tmp_path: Path) -> None:
        """Should load orchestration with agent_type='cursor'."""
        yaml_content = """
orchestrations:
  - name: "cursor-agent"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: cursor
      cursor_mode: agent
"""
        file_path = tmp_path / "cursor_agent.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type == "cursor"
        assert orchestrations[0].agent.cursor_mode == "agent"

    def test_load_file_with_cursor_mode_plan(self, tmp_path: Path) -> None:
        """Should load orchestration with cursor_mode='plan'."""
        yaml_content = """
orchestrations:
  - name: "cursor-plan"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: cursor
      cursor_mode: plan
"""
        file_path = tmp_path / "cursor_plan.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.cursor_mode == "plan"

    def test_load_file_with_cursor_mode_ask(self, tmp_path: Path) -> None:
        """Should load orchestration with cursor_mode='ask'."""
        yaml_content = """
orchestrations:
  - name: "cursor-ask"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: cursor
      cursor_mode: ask
"""
        file_path = tmp_path / "cursor_ask.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.cursor_mode == "ask"

    def test_load_file_without_agent_type_uses_none(self, tmp_path: Path) -> None:
        """Should default to None when agent_type is not specified."""
        yaml_content = """
orchestrations:
  - name: "no-agent-type"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "no_agent_type.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type is None
        assert orchestrations[0].agent.cursor_mode is None

    def test_invalid_agent_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid agent_type value."""
        yaml_content = """
orchestrations:
  - name: "invalid-agent-type"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: copilot
"""
        file_path = tmp_path / "invalid_agent_type.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid agent_type"):
            load_orchestration_file(file_path)

    def test_invalid_cursor_mode_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid cursor_mode value."""
        yaml_content = """
orchestrations:
  - name: "invalid-cursor-mode"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: cursor
      cursor_mode: invalid
"""
        file_path = tmp_path / "invalid_cursor_mode.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid cursor_mode"):
            load_orchestration_file(file_path)

    def test_cursor_mode_with_claude_agent_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when cursor_mode is set with agent_type='claude'."""
        yaml_content = """
orchestrations:
  - name: "invalid-combo"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: claude
      cursor_mode: agent
"""
        file_path = tmp_path / "invalid_combo.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="cursor_mode.*only valid when agent_type is 'cursor'"):
            load_orchestration_file(file_path)

    def test_cursor_mode_without_agent_type_allowed(self, tmp_path: Path) -> None:
        """cursor_mode without agent_type should be allowed (agent_type defaults to config)."""
        yaml_content = """
orchestrations:
  - name: "cursor-mode-only"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      cursor_mode: agent
"""
        file_path = tmp_path / "cursor_mode_only.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type is None
        assert orchestrations[0].agent.cursor_mode == "agent"

    def test_agent_type_cursor_without_cursor_mode_allowed(self, tmp_path: Path) -> None:
        """agent_type='cursor' without cursor_mode should be allowed (defaults can apply)."""
        yaml_content = """
orchestrations:
  - name: "cursor-no-mode"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: cursor
"""
        file_path = tmp_path / "cursor_no_mode.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type == "cursor"
        assert orchestrations[0].agent.cursor_mode is None

    def test_agent_type_with_all_other_config_options(self, tmp_path: Path) -> None:
        """Should load agent_type and cursor_mode along with all other options."""
        yaml_content = """
orchestrations:
  - name: "full-agent-config"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review the code"
      tools:
        - jira
        - github
      timeout_seconds: 300
      model: "claude-sonnet-4-20250514"
      agent_type: cursor
      cursor_mode: agent
      github:
        host: "github.com"
        org: "test-org"
        repo: "test-repo"
"""
        file_path = tmp_path / "full_agent_config.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.prompt == "Review the code"
        assert orch.agent.tools == ["jira", "github"]
        assert orch.agent.timeout_seconds == 300
        assert orch.agent.model == "claude-sonnet-4-20250514"
        assert orch.agent.agent_type == "cursor"
        assert orch.agent.cursor_mode == "agent"
        assert orch.agent.github is not None
        assert orch.agent.github.org == "test-org"
