"""Tests for orchestration configuration validation.

This module contains tests for:
- Orchestration enabled/disabled functionality
- max_concurrent field validation
"""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OrchestrationError,
    TriggerConfig,
    load_orchestration_file,
)


class TestOrchestrationEnabled:
    """Tests for orchestration enabled/disabled functionality."""

    def test_orchestration_enabled_defaults_to_true(self, tmp_path: Path) -> None:
        """Orchestration should default to enabled when not specified."""
        yaml_content = """
orchestrations:
  - name: "default-enabled"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "default_enabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].enabled is True

    def test_orchestration_enabled_true(self, tmp_path: Path) -> None:
        """Orchestration with enabled: true should be loaded."""
        yaml_content = """
orchestrations:
  - name: "explicitly-enabled"
    enabled: true
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "explicitly_enabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].enabled is True

    def test_orchestration_enabled_false(self, tmp_path: Path) -> None:
        """Orchestration with enabled: false should be filtered out."""
        yaml_content = """
orchestrations:
  - name: "disabled-orch"
    enabled: false
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "disabled_orch.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 0

    def test_mixed_enabled_orchestrations(self, tmp_path: Path) -> None:
        """Only enabled orchestrations should be returned."""
        yaml_content = """
orchestrations:
  - name: "enabled-one"
    enabled: true
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "First"
  - name: "disabled-one"
    enabled: false
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Second"
  - name: "enabled-two"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Third"
"""
        file_path = tmp_path / "mixed_enabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 2
        names = [o.name for o in orchestrations]
        assert "enabled-one" in names
        assert "enabled-two" in names
        assert "disabled-one" not in names

    def test_invalid_orchestration_enabled_type(self, tmp_path: Path) -> None:
        """Should raise error if orchestration enabled is not a boolean."""
        yaml_content = """
orchestrations:
  - name: "invalid-enabled"
    enabled: "yes"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_enabled.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'enabled' value"):
            load_orchestration_file(file_path)

    def test_file_level_enabled_false(self, tmp_path: Path) -> None:
        """File-level enabled: false should disable all orchestrations."""
        yaml_content = """
enabled: false
orchestrations:
  - name: "orch-one"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "First"
  - name: "orch-two"
    enabled: true
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "file_disabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 0

    def test_file_level_enabled_true(self, tmp_path: Path) -> None:
        """File-level enabled: true should allow orchestrations to be loaded."""
        yaml_content = """
enabled: true
orchestrations:
  - name: "orch-one"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "First"
"""
        file_path = tmp_path / "file_enabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1

    def test_file_level_enabled_defaults_to_true(self, tmp_path: Path) -> None:
        """File-level enabled should default to true when not specified."""
        yaml_content = """
orchestrations:
  - name: "orch-one"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "First"
"""
        file_path = tmp_path / "no_file_enabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1

    def test_file_level_takes_precedence_over_orchestration_level(self, tmp_path: Path) -> None:
        """File-level enabled: false should override orchestration-level enabled: true."""
        yaml_content = """
enabled: false
orchestrations:
  - name: "enabled-orch"
    enabled: true
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "This should still be disabled"
"""
        file_path = tmp_path / "file_precedence.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        # File-level disabled should override orchestration-level enabled
        assert len(orchestrations) == 0

    def test_invalid_file_level_enabled_type(self, tmp_path: Path) -> None:
        """Should raise error if file-level enabled is not a boolean."""
        yaml_content = """
enabled: "yes"
orchestrations:
  - name: "test"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_file_enabled.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="File-level 'enabled' must be a boolean"):
            load_orchestration_file(file_path)

    def test_file_enabled_false_with_empty_orchestrations(self, tmp_path: Path) -> None:
        """File-level enabled: false with no orchestrations should return empty list."""
        yaml_content = """
enabled: false
orchestrations: []
"""
        file_path = tmp_path / "disabled_empty.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert orchestrations == []

    def test_all_orchestrations_disabled(self, tmp_path: Path) -> None:
        """Should return empty list when all orchestrations are individually disabled."""
        yaml_content = """
orchestrations:
  - name: "disabled-one"
    enabled: false
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "First"
  - name: "disabled-two"
    enabled: false
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Second"
"""
        file_path = tmp_path / "all_disabled.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 0


class TestMaxConcurrent:
    """Tests for max_concurrent field validation in _parse_orchestration().

    These tests verify the max_concurrent field handling:
    - Orchestration loads without max_concurrent field (backwards compatible)
    - Orchestration loads with valid max_concurrent value
    - OrchestrationError is raised for zero value
    - OrchestrationError is raised for negative values
    - OrchestrationError is raised for non-integer values (string, float)

    Related to per-orchestration concurrency limits.
    """

    def test_orchestration_max_concurrent_defaults_to_none(self) -> None:
        """Orchestration max_concurrent should default to None."""
        from sentinel.orchestration import Orchestration

        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        assert orch.max_concurrent is None

    def test_orchestration_max_concurrent_can_be_set(self) -> None:
        """Orchestration max_concurrent can be set to a positive integer."""
        from sentinel.orchestration import Orchestration

        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
            max_concurrent=5,
        )
        assert orch.max_concurrent == 5

    def test_load_file_with_max_concurrent(self, tmp_path: Path) -> None:
        """Should load orchestration with max_concurrent specified."""
        yaml_content = """
orchestrations:
  - name: "limited-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
    max_concurrent: 3
"""
        file_path = tmp_path / "max_concurrent.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].max_concurrent == 3

    def test_load_file_without_max_concurrent_uses_none(self, tmp_path: Path) -> None:
        """Should default to None when max_concurrent is not specified."""
        yaml_content = """
orchestrations:
  - name: "unlimited-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "no_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].max_concurrent is None

    def test_invalid_max_concurrent_zero_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is zero."""
        yaml_content = """
orchestrations:
  - name: "zero-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: 0
"""
        file_path = tmp_path / "zero_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'max_concurrent' value"):
            load_orchestration_file(file_path)

    def test_invalid_max_concurrent_negative_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is negative."""
        yaml_content = """
orchestrations:
  - name: "negative-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: -1
"""
        file_path = tmp_path / "negative_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'max_concurrent' value"):
            load_orchestration_file(file_path)

    def test_invalid_max_concurrent_string_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is not an integer."""
        yaml_content = """
orchestrations:
  - name: "string-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: "five"
"""
        file_path = tmp_path / "string_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'max_concurrent' value"):
            load_orchestration_file(file_path)

    def test_invalid_max_concurrent_float_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is a float."""
        yaml_content = """
orchestrations:
  - name: "float-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: 2.5
"""
        file_path = tmp_path / "float_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'max_concurrent' value"):
            load_orchestration_file(file_path)

    def test_max_concurrent_with_value_one(self, tmp_path: Path) -> None:
        """Should allow max_concurrent of 1 (minimum valid value)."""
        yaml_content = """
orchestrations:
  - name: "single-slot"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
    max_concurrent: 1
"""
        file_path = tmp_path / "single_slot.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].max_concurrent == 1

    def test_max_concurrent_with_large_value(self, tmp_path: Path) -> None:
        """Should allow large max_concurrent values."""
        yaml_content = """
orchestrations:
  - name: "many-slots"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
    max_concurrent: 100
"""
        file_path = tmp_path / "many_slots.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].max_concurrent == 100

    def test_multiple_orchestrations_with_different_max_concurrent(self, tmp_path: Path) -> None:
        """Should load multiple orchestrations with different max_concurrent values."""
        yaml_content = """
orchestrations:
  - name: "limited-one"
    trigger:
      source: jira
      tags: ["review"]
    agent:
      prompt: "Review"
    max_concurrent: 2
  - name: "limited-two"
    trigger:
      source: jira
      tags: ["deploy"]
    agent:
      prompt: "Deploy"
    max_concurrent: 5
  - name: "unlimited"
    trigger:
      source: jira
      tags: ["triage"]
    agent:
      prompt: "Triage"
"""
        file_path = tmp_path / "multiple.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 3
        orch_by_name = {o.name: o for o in orchestrations}
        assert orch_by_name["limited-one"].max_concurrent == 2
        assert orch_by_name["limited-two"].max_concurrent == 5
        assert orch_by_name["unlimited"].max_concurrent is None

    def test_max_concurrent_with_all_other_config_options(self, tmp_path: Path) -> None:
        """Should load max_concurrent along with all other configuration options."""
        yaml_content = """
orchestrations:
  - name: "full-config"
    enabled: true
    max_concurrent: 3
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
    retry:
      max_attempts: 5
      default_status: failure
    on_start:
      add_tag: "processing"
    on_complete:
      remove_tag: "review"
      add_tag: "reviewed"
    on_failure:
      add_tag: "review-failed"
"""
        file_path = tmp_path / "full_config.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.name == "full-config"
        assert orch.enabled is True
        assert orch.max_concurrent == 3
        assert orch.trigger.project == "TEST"
        assert orch.agent.timeout_seconds == 300
        assert orch.retry.max_attempts == 5
        assert orch.on_start.add_tag == "processing"
        assert orch.on_complete.remove_tag == "review"
        assert orch.on_failure.add_tag == "review-failed"
