"""Tests for orchestration labels/tags validation and agent type configuration.

This module contains tests for:
- Labels and tags field validation in TriggerConfig
- Agent type configuration (claude vs cursor)
- Cursor mode configuration
- Agent Teams timeout handling (DS-697)
- Agent Teams timeout configurability via environment variables (DS-701)
- Error message template regression tests for _parse_env_int() (DS-744)
"""

import logging
import re
from pathlib import Path
from unittest import mock

import pytest

from sentinel.orchestration import (
    _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
    _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER,
    AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
    AGENT_TEAMS_TIMEOUT_MULTIPLIER,
    AgentConfig,
    OrchestrationError,
    _parse_env_int,
    get_effective_timeout,
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

        with pytest.raises(
            OrchestrationError, match="cursor_mode.*only valid when agent_type is 'cursor'"
        ):
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
        assert orch.agent.timeout_seconds == 300
        assert orch.agent.model == "claude-sonnet-4-20250514"
        assert orch.agent.agent_type == "cursor"
        assert orch.agent.cursor_mode == "agent"
        assert orch.agent.github is not None
        assert orch.agent.github.org == "test-org"


class TestCodexAgentType:
    """Tests for Codex agent type configuration.

    These tests verify that the codex agent type is properly supported:
    - agent_type: 'codex' loads successfully
    - cursor_mode is rejected when agent_type is 'codex'
    """

    def test_load_file_with_agent_type_codex(self, tmp_path: Path) -> None:
        """Should load orchestration with agent_type='codex'."""
        yaml_content = """
orchestrations:
  - name: "codex-agent"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: codex
"""
        file_path = tmp_path / "codex_agent.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_type == "codex"
        assert orchestrations[0].agent.cursor_mode is None

    def test_cursor_mode_with_codex_agent_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when cursor_mode is set with agent_type='codex'."""
        yaml_content = """
orchestrations:
  - name: "invalid-codex-combo"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: codex
      cursor_mode: agent
"""
        file_path = tmp_path / "invalid_codex_combo.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(
            OrchestrationError, match="cursor_mode.*only valid when agent_type is 'cursor'"
        ):
            load_orchestration_file(file_path)

    def test_agent_type_codex_with_all_options(self, tmp_path: Path) -> None:
        """Should load codex agent_type along with all other options."""
        yaml_content = """
orchestrations:
  - name: "full-codex-config"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review the code"
      timeout_seconds: 300
      model: "o3-mini"
      agent_type: codex
      github:
        host: "github.com"
        org: "test-org"
        repo: "test-repo"
"""
        file_path = tmp_path / "full_codex_config.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.prompt == "Review the code"
        assert orch.agent.timeout_seconds == 300
        assert orch.agent.model == "o3-mini"
        assert orch.agent.agent_type == "codex"
        assert orch.agent.cursor_mode is None
        assert orch.agent.github is not None
        assert orch.agent.github.org == "test-org"


class TestAgentTeams:
    """Tests for agent_teams field in AgentConfig.

    These tests verify the agent_teams boolean field:
    - Defaults to False when not specified
    - Can be set to True when agent_type is "claude"
    - Is rejected when agent_type is "cursor" or "codex"
    - Must be a boolean value
    - Is allowed when agent_type is None (default resolved at runtime)
    """

    def test_agent_teams_defaults_to_false(self, tmp_path: Path) -> None:
        """Should default to False when agent_teams is not specified."""
        yaml_content = """
orchestrations:
  - name: "no-agent-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "no_agent_teams.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_teams is False

    def test_agent_teams_true_with_claude_agent_type(self, tmp_path: Path) -> None:
        """Should load agent_teams=True when agent_type is 'claude'."""
        yaml_content = """
orchestrations:
  - name: "claude-agent-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
      agent_teams: true
"""
        file_path = tmp_path / "claude_agent_teams.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_teams is True
        assert orchestrations[0].agent.agent_type == "claude"

    def test_agent_teams_false_explicit(self, tmp_path: Path) -> None:
        """Should load agent_teams=False when explicitly set."""
        yaml_content = """
orchestrations:
  - name: "explicit-false"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
      agent_teams: false
"""
        file_path = tmp_path / "explicit_false.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_teams is False

    def test_agent_teams_with_cursor_agent_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when agent_teams=True with agent_type='cursor'."""
        yaml_content = """
orchestrations:
  - name: "invalid-cursor-teams"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: cursor
      agent_teams: true
"""
        file_path = tmp_path / "invalid_cursor_teams.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(
            OrchestrationError, match="agent_teams is only valid when agent_type is 'claude'"
        ):
            load_orchestration_file(file_path)

    def test_agent_teams_with_codex_agent_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when agent_teams=True with agent_type='codex'."""
        yaml_content = """
orchestrations:
  - name: "invalid-codex-teams"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: codex
      agent_teams: true
"""
        file_path = tmp_path / "invalid_codex_teams.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(
            OrchestrationError, match="agent_teams is only valid when agent_type is 'claude'"
        ):
            load_orchestration_file(file_path)

    def test_agent_teams_invalid_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when agent_teams is not a boolean."""
        yaml_content = """
orchestrations:
  - name: "invalid-type"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: claude
      agent_teams: "yes"
"""
        file_path = tmp_path / "invalid_type.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid agent_teams.*must be a boolean"):
            load_orchestration_file(file_path)

    def test_agent_teams_integer_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when agent_teams is an integer instead of boolean."""
        yaml_content = """
orchestrations:
  - name: "invalid-int"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      agent_type: claude
      agent_teams: 1
"""
        file_path = tmp_path / "invalid_int.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid agent_teams.*must be a boolean"):
            load_orchestration_file(file_path)

    def test_agent_teams_true_without_agent_type_allowed(self, tmp_path: Path) -> None:
        """agent_teams=True without agent_type should be allowed (resolved at runtime)."""
        yaml_content = """
orchestrations:
  - name: "teams-no-type"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_teams: true
"""
        file_path = tmp_path / "teams_no_type.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_teams is True
        assert orchestrations[0].agent.agent_type is None

    def test_agent_teams_false_with_cursor_allowed(self, tmp_path: Path) -> None:
        """agent_teams=False with agent_type='cursor' should be allowed (no conflict)."""
        yaml_content = """
orchestrations:
  - name: "cursor-no-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: cursor
      agent_teams: false
"""
        file_path = tmp_path / "cursor_no_teams.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.agent_teams is False
        assert orchestrations[0].agent.agent_type == "cursor"

    def test_agent_teams_with_all_other_config_options(self, tmp_path: Path) -> None:
        """Should load agent_teams along with all other agent config options."""
        yaml_content = """
orchestrations:
  - name: "full-teams-config"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review the code"
      timeout_seconds: 300
      model: "claude-opus-4-5-20251101"
      agent_type: claude
      agent_teams: true
      github:
        host: "github.com"
        org: "test-org"
        repo: "test-repo"
"""
        file_path = tmp_path / "full_teams_config.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.prompt == "Review the code"
        assert orch.agent.timeout_seconds == 300
        assert orch.agent.model == "claude-opus-4-5-20251101"
        assert orch.agent.agent_type == "claude"
        assert orch.agent.agent_teams is True
        assert orch.agent.github is not None
        assert orch.agent.github.org == "test-org"


class TestAgentTeamsTimeout:
    """Tests for Agent Teams timeout handling (DS-697).

    These tests verify the timeout multiplier and minimum enforcement for
    agent_teams-enabled orchestrations, including:
    - get_effective_timeout() returns multiplied timeout when agent_teams is True
    - get_effective_timeout() returns original timeout when agent_teams is False
    - get_effective_timeout() returns None when no timeout is configured
    - get_effective_timeout() enforces the minimum timeout for agent_teams
    - A warning is logged when timeout_seconds is below the recommended minimum
    - No warning is logged when timeout_seconds meets or exceeds the minimum
    - Constants AGENT_TEAMS_TIMEOUT_MULTIPLIER and AGENT_TEAMS_MIN_TIMEOUT_SECONDS
      have expected values
    """

    def test_effective_timeout_with_agent_teams_applies_multiplier(self) -> None:
        """Should apply the timeout multiplier when agent_teams is True."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=600,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        assert result == 600 * AGENT_TEAMS_TIMEOUT_MULTIPLIER

    def test_effective_timeout_without_agent_teams_unchanged(self) -> None:
        """Should return the original timeout when agent_teams is False."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=600,
            agent_teams=False,
        )
        result = get_effective_timeout(config)
        assert result == 600

    def test_effective_timeout_none_with_agent_teams(self) -> None:
        """Should return None when no timeout is configured, even with agent_teams."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=None,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        assert result is None

    def test_effective_timeout_none_without_agent_teams(self) -> None:
        """Should return None when no timeout is configured and agent_teams is False."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=None,
            agent_teams=False,
        )
        result = get_effective_timeout(config)
        assert result is None

    def test_effective_timeout_enforces_minimum_for_agent_teams(self) -> None:
        """Should enforce the minimum timeout when multiplied result is too low."""
        # A very small timeout should still result in at least the minimum
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=100,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        # 100 * 3 = 300, which is < 900, so minimum of 900 should be used
        assert result == AGENT_TEAMS_MIN_TIMEOUT_SECONDS

    def test_effective_timeout_multiplier_exceeds_minimum(self) -> None:
        """Should use the multiplied value when it exceeds the minimum."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=600,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        # 600 * 3 = 1800, which is > 900, so 1800 should be used
        assert result == 1800
        assert result > AGENT_TEAMS_MIN_TIMEOUT_SECONDS

    def test_effective_timeout_at_minimum_boundary(self) -> None:
        """Should handle timeout at exactly the minimum boundary correctly."""
        # Find a timeout where multiplied value equals the minimum
        boundary_timeout = AGENT_TEAMS_MIN_TIMEOUT_SECONDS // AGENT_TEAMS_TIMEOUT_MULTIPLIER
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=boundary_timeout,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        assert result == max(
            boundary_timeout * AGENT_TEAMS_TIMEOUT_MULTIPLIER,
            AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
        )

    def test_warning_logged_for_low_timeout_with_agent_teams(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log a warning when timeout is below the recommended minimum."""
        yaml_content = """
orchestrations:
  - name: "low-timeout-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
      agent_teams: true
      timeout_seconds: 300
"""
        file_path = tmp_path / "low_timeout_teams.yaml"
        file_path.write_text(yaml_content)

        with caplog.at_level(logging.WARNING, logger="sentinel.orchestration"):
            orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds == 300
        assert orchestrations[0].agent.agent_teams is True

        # Check that a warning was logged about the low timeout
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "timeout_seconds=300" in msg and "below the recommended minimum" in msg
            for msg in warning_messages
        ), f"Expected timeout warning in: {warning_messages}"

    def test_no_warning_for_sufficient_timeout_with_agent_teams(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log a warning when timeout meets the recommended minimum."""
        yaml_content = """
orchestrations:
  - name: "good-timeout-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
      agent_teams: true
      timeout_seconds: 900
"""
        file_path = tmp_path / "good_timeout_teams.yaml"
        file_path.write_text(yaml_content)

        with caplog.at_level(logging.WARNING, logger="sentinel.orchestration"):
            orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds == 900

        # Verify no timeout warnings were logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any(
            "below the recommended minimum" in msg for msg in warning_messages
        ), f"Unexpected timeout warning in: {warning_messages}"

    def test_no_warning_when_no_timeout_with_agent_teams(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log a warning when no timeout is configured with agent_teams."""
        yaml_content = """
orchestrations:
  - name: "no-timeout-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      agent_type: claude
      agent_teams: true
"""
        file_path = tmp_path / "no_timeout_teams.yaml"
        file_path.write_text(yaml_content)

        with caplog.at_level(logging.WARNING, logger="sentinel.orchestration"):
            orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds is None
        assert orchestrations[0].agent.agent_teams is True

        # Verify no timeout warnings were logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any(
            "below the recommended minimum" in msg for msg in warning_messages
        ), f"Unexpected timeout warning in: {warning_messages}"

    def test_no_warning_for_low_timeout_without_agent_teams(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log a warning for low timeout when agent_teams is not enabled."""
        yaml_content = """
orchestrations:
  - name: "low-timeout-no-teams"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: 100
"""
        file_path = tmp_path / "low_timeout_no_teams.yaml"
        file_path.write_text(yaml_content)

        with caplog.at_level(logging.WARNING, logger="sentinel.orchestration"):
            orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds == 100
        assert orchestrations[0].agent.agent_teams is False

        # Verify no timeout warnings were logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any(
            "below the recommended minimum" in msg for msg in warning_messages
        ), f"Unexpected timeout warning in: {warning_messages}"

    def test_constants_have_expected_default_values(self) -> None:
        """Should have expected default constant values for timeout multiplier and minimum."""
        assert _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER == 3
        assert _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS == 900

    def test_effective_timeout_with_very_large_timeout(self) -> None:
        """Should correctly multiply very large timeout values."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=3600,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        assert result == 3600 * AGENT_TEAMS_TIMEOUT_MULTIPLIER

    def test_effective_timeout_with_timeout_of_one_second(self) -> None:
        """Should enforce minimum for very small timeout values."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=1,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        # 1 * 3 = 3, which is < 900, so minimum of 900 should be used
        assert result == AGENT_TEAMS_MIN_TIMEOUT_SECONDS


class TestAgentTeamsTimeoutEnvConfig:
    """Tests for Agent Teams timeout configurability via environment variables (DS-701).

    These tests verify that AGENT_TEAMS_TIMEOUT_MULTIPLIER and
    AGENT_TEAMS_MIN_TIMEOUT_SECONDS can be overridden via environment
    variables, enabling per-deployment tuning without code changes.
    """

    def test_parse_env_int_returns_default_when_unset(self) -> None:
        """Should return default when environment variable is not set."""
        result = _parse_env_int("NONEXISTENT_VAR_FOR_TEST_DS701", 42)
        assert result == 42

    def test_parse_env_int_returns_default_when_empty(self) -> None:
        """Should return default when environment variable is empty."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": ""}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 42

    def test_parse_env_int_returns_default_when_whitespace(self) -> None:
        """Should return default when environment variable is whitespace only."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "   "}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 42

    def test_parse_env_int_parses_valid_integer(self) -> None:
        """Should parse a valid integer from the environment variable."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "5"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 5

    def test_parse_env_int_parses_with_whitespace(self) -> None:
        """Should parse an integer with surrounding whitespace."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": " 7 "}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 7

    def test_parse_env_int_raises_on_non_integer(self) -> None:
        """Should raise ValueError for a non-integer environment variable."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "not_a_number"}):
            with pytest.raises(ValueError, match="must be an integer"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_parse_env_int_raises_on_float(self) -> None:
        """Should raise ValueError for a float environment variable."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "3.5"}):
            with pytest.raises(ValueError, match="must be an integer"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_multiplier_env_override(self) -> None:
        """Should parse env override for AGENT_TEAMS_TIMEOUT_MULTIPLIER."""
        with mock.patch.dict("os.environ", {"AGENT_TEAMS_TIMEOUT_MULTIPLIER": "5"}):
            result = _parse_env_int(
                "AGENT_TEAMS_TIMEOUT_MULTIPLIER",
                _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER,
            )
            assert result == 5

    def test_min_timeout_env_override(self) -> None:
        """Should parse env override for AGENT_TEAMS_MIN_TIMEOUT_SECONDS."""
        with mock.patch.dict("os.environ", {"AGENT_TEAMS_MIN_TIMEOUT_SECONDS": "1200"}):
            result = _parse_env_int(
                "AGENT_TEAMS_MIN_TIMEOUT_SECONDS",
                _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
            )
            assert result == 1200

    def test_multiplier_uses_default_when_unset(self) -> None:
        """Should return default multiplier when env var is not set."""
        result = _parse_env_int(
            "AGENT_TEAMS_TIMEOUT_MULTIPLIER",
            _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER,
        )
        assert result == _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER

    def test_min_timeout_uses_default_when_unset(self) -> None:
        """Should return default min timeout when env var is not set."""
        result = _parse_env_int(
            "AGENT_TEAMS_MIN_TIMEOUT_SECONDS",
            _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS,
        )
        assert result == _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS

    def test_defaults_unchanged_without_env_vars(self) -> None:
        """Should use defaults when no environment variables are set."""
        assert _DEFAULT_AGENT_TEAMS_TIMEOUT_MULTIPLIER == 3
        assert _DEFAULT_AGENT_TEAMS_MIN_TIMEOUT_SECONDS == 900


class TestParseEnvIntValidationAndLogging:
    """Tests for _parse_env_int() positive-value validation and logging (DS-714).

    These tests verify the two improvements from DS-714:
    1. Positive-value validation: values of 0 or negative are rejected with
       a clear error message, preventing operators from accidentally disabling
       or inverting timeout behaviour.
    2. Info-level logging: when an environment variable override is applied,
       an info message is logged so operators can confirm their settings are
       being picked up at startup.
    """

    def test_parse_env_int_raises_on_zero(self) -> None:
        """Should raise ValueError when environment variable is set to 0."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "0"}):
            with pytest.raises(ValueError, match="must be >= 1, got 0"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_parse_env_int_raises_on_negative(self) -> None:
        """Should raise ValueError when environment variable is set to a negative value."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "-1"}):
            with pytest.raises(ValueError, match="must be >= 1, got -1"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_parse_env_int_raises_on_large_negative(self) -> None:
        """Should raise ValueError for large negative values."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "-100"}):
            with pytest.raises(ValueError, match="must be >= 1, got -100"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_parse_env_int_accepts_one(self) -> None:
        """Should accept 1 as a valid positive integer."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "1"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 1

    def test_parse_env_int_logs_info_on_override(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log an info message when an env var override is applied."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "10"}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                result = _parse_env_int("TEST_PARSE_ENV_INT", 42)

        assert result == 10

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        expected = "Using TEST_PARSE_ENV_INT=10 from environment (default: 42)"
        assert any(
            msg == expected
            for msg in info_messages
        ), f"Expected exact info log '{expected}' in: {info_messages}"

    def test_parse_env_int_no_log_when_using_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log an info message when the default value is used."""
        with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
            result = _parse_env_int("NONEXISTENT_VAR_FOR_TEST_DS714", 42)

        assert result == 42

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any(
            "NONEXISTENT_VAR_FOR_TEST_DS714" in msg for msg in info_messages
        ), f"Unexpected info log when using default: {info_messages}"

    def test_parse_env_int_no_log_when_empty(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log an info message when env var is empty (uses default)."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": ""}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                result = _parse_env_int("TEST_PARSE_ENV_INT", 42)

        assert result == 42

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any(
            "TEST_PARSE_ENV_INT" in msg for msg in info_messages
        ), f"Unexpected info log when env var is empty: {info_messages}"

    def test_parse_env_int_log_message_format(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log message matching expected format: 'Using <name>=<value> from environment (default: <default>)'."""
        with mock.patch.dict("os.environ", {"AGENT_TEAMS_TIMEOUT_MULTIPLIER": "5"}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                result = _parse_env_int("AGENT_TEAMS_TIMEOUT_MULTIPLIER", 3)

        assert result == 5

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        expected = "Using AGENT_TEAMS_TIMEOUT_MULTIPLIER=5 from environment (default: 3)"
        assert any(
            msg == expected
            for msg in info_messages
        ), f"Expected exact info log '{expected}' in: {info_messages}"

    # --- Tests for min_value parameter (DS-721) ---

    def test_parse_env_int_min_value_zero_allows_zero(self) -> None:
        """Should accept 0 when min_value=0."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "0"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42, min_value=0)
        assert result == 0

    def test_parse_env_int_min_value_zero_rejects_negative(self) -> None:
        """Should reject negative values when min_value=0."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "-1"}):
            with pytest.raises(ValueError, match="must be >= 0, got -1"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42, min_value=0)

    def test_parse_env_int_custom_min_value_rejects_below(self) -> None:
        """Should reject values below a custom min_value."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "4"}):
            with pytest.raises(ValueError, match="must be >= 5, got 4"):
                _parse_env_int("TEST_PARSE_ENV_INT", 10, min_value=5)

    def test_parse_env_int_custom_min_value_accepts_exact(self) -> None:
        """Should accept a value exactly equal to min_value."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "5"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 10, min_value=5)
        assert result == 5

    def test_parse_env_int_custom_min_value_accepts_above(self) -> None:
        """Should accept a value above the custom min_value."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "6"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 10, min_value=5)
        assert result == 6

    def test_parse_env_int_default_min_value_is_one(self) -> None:
        """Should default to min_value=1, preserving backward compatibility."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "1"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 42)
        assert result == 1

        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "0"}):
            with pytest.raises(ValueError, match="must be >= 1, got 0"):
                _parse_env_int("TEST_PARSE_ENV_INT", 42)

    def test_parse_env_int_negative_min_value(self) -> None:
        """Should support negative min_value for signed-int use cases."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "-5"}):
            result = _parse_env_int("TEST_PARSE_ENV_INT", 0, min_value=-10)
        assert result == -5

    def test_parse_env_int_negative_min_value_rejects_below(self) -> None:
        """Should reject values below a negative min_value."""
        with mock.patch.dict("os.environ", {"TEST_PARSE_ENV_INT": "-11"}):
            with pytest.raises(ValueError, match=r"must be >= -10, got -11"):
                _parse_env_int("TEST_PARSE_ENV_INT", 0, min_value=-10)


class TestParseEnvIntErrorMessageTemplates:
    """Regression tests for _parse_env_int() error message template formats (DS-744).

    These tests use ``re.fullmatch`` to verify the *exact* structure of every
    error/log message produced by ``_parse_env_int()``.  By matching the full
    template (with placeholder groups for the variable parts) rather than a
    specific instance, we catch accidental regressions in wording, punctuation,
    or field order even when the function is refactored or gains new callers.

    Follow-up from DS-721 code review (PR #733).
    """

    # -- Non-integer error template ------------------------------------------

    _NON_INTEGER_TEMPLATE = (
        r"Environment variable \S+ must be an integer, got '.+'"
    )

    def test_non_integer_error_matches_template(self) -> None:
        """Non-integer error message should match the expected template."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "abc"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 1)

        assert re.fullmatch(self._NON_INTEGER_TEMPLATE, str(exc_info.value)), (
            f"Non-integer error message does not match template "
            f"{self._NON_INTEGER_TEMPLATE!r}: {str(exc_info.value)!r}"
        )

    def test_non_integer_error_contains_variable_name(self) -> None:
        """Non-integer error message should embed the env-var name."""
        with mock.patch.dict("os.environ", {"MY_CUSTOM_VAR": "xyz"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("MY_CUSTOM_VAR", 1)

        assert "MY_CUSTOM_VAR" in str(exc_info.value)

    def test_non_integer_error_contains_raw_value(self) -> None:
        """Non-integer error message should embed the rejected raw value."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "not_a_number"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 1)

        assert "not_a_number" in str(exc_info.value)

    # -- Below-minimum error template ----------------------------------------

    _BELOW_MIN_TEMPLATE = (
        r"Environment variable \S+ must be >= -?\d+, got -?\d+"
    )

    def test_below_min_error_matches_template_default(self) -> None:
        """Below-min error (default min_value=1) should match the template."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "0"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 10)

        assert re.fullmatch(self._BELOW_MIN_TEMPLATE, str(exc_info.value)), (
            f"Below-min error message does not match template "
            f"{self._BELOW_MIN_TEMPLATE!r}: {str(exc_info.value)!r}"
        )

    def test_below_min_error_matches_template_custom_min(self) -> None:
        """Below-min error with custom min_value should match the template."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "2"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 10, min_value=5)

        assert re.fullmatch(self._BELOW_MIN_TEMPLATE, str(exc_info.value)), (
            f"Below-min error message does not match template "
            f"{self._BELOW_MIN_TEMPLATE!r}: {str(exc_info.value)!r}"
        )

    def test_below_min_error_matches_template_negative_min(self) -> None:
        """Below-min error with negative min_value should match the template."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "-20"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 0, min_value=-10)

        assert re.fullmatch(self._BELOW_MIN_TEMPLATE, str(exc_info.value)), (
            f"Below-min error message does not match template "
            f"{self._BELOW_MIN_TEMPLATE!r}: {str(exc_info.value)!r}"
        )

    def test_below_min_error_contains_variable_name(self) -> None:
        """Below-min error message should embed the env-var name."""
        with mock.patch.dict("os.environ", {"MY_CUSTOM_VAR": "0"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("MY_CUSTOM_VAR", 10)

        assert "MY_CUSTOM_VAR" in str(exc_info.value)

    def test_below_min_error_contains_min_value(self) -> None:
        """Below-min error message should embed the min_value threshold."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "2"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 10, min_value=5)

        assert ">= 5" in str(exc_info.value)

    def test_below_min_error_contains_actual_value(self) -> None:
        """Below-min error message should embed the actual rejected value."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "-3"}):
            with pytest.raises(ValueError) as exc_info:
                _parse_env_int("TEST_TMPL_VAR", 10)

        assert "got -3" in str(exc_info.value)

    # -- Info-log message template -------------------------------------------

    _LOG_MESSAGE_TEMPLATE = (
        r"Using \S+=\d+ from environment \(default: \d+\)"
    )

    def test_log_message_matches_template(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Info log message should match the expected template format."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "7"}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                _parse_env_int("TEST_TMPL_VAR", 42)

        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        matched = [
            msg for msg in info_messages if re.fullmatch(self._LOG_MESSAGE_TEMPLATE, msg)
        ]
        assert matched, (
            f"No info log matched template {self._LOG_MESSAGE_TEMPLATE!r}; "
            f"got: {info_messages}"
        )

    def test_log_message_contains_variable_name_and_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Info log should embed the variable name and override value."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "99"}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                _parse_env_int("TEST_TMPL_VAR", 1)

        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        assert any(
            "TEST_TMPL_VAR=99" in msg for msg in info_messages
        ), f"Expected 'TEST_TMPL_VAR=99' in log messages: {info_messages}"

    def test_log_message_contains_default_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Info log should embed the default value."""
        with mock.patch.dict("os.environ", {"TEST_TMPL_VAR": "5"}):
            with caplog.at_level(logging.INFO, logger="sentinel.orchestration"):
                _parse_env_int("TEST_TMPL_VAR", 42)

        info_messages = [
            r.message for r in caplog.records if r.levelno == logging.INFO
        ]
        assert any(
            "default: 42" in msg for msg in info_messages
        ), f"Expected 'default: 42' in log messages: {info_messages}"

