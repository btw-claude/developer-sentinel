"""Tests for orchestration configuration validation.

This module contains tests for:
- Orchestration enabled/disabled functionality
- max_concurrent field validation
- timeout_seconds field validation (int, float, NaN, inf, bool edge cases)
"""

import math
from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OrchestrationError,
    TriggerConfig,
    _validate_timeout_seconds,
    get_effective_timeout,
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

    def test_invalid_max_concurrent_boolean_true_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is boolean True (DS-1031).

        In Python, bool is a subclass of int, so isinstance(True, int)
        returns True. The boolean guard ensures that YAML values like
        ``max_concurrent: true`` are rejected.
        """
        yaml_content = """
orchestrations:
  - name: "bool-true-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: true
"""
        file_path = tmp_path / "bool_true_max_concurrent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="invalid 'max_concurrent' value"):
            load_orchestration_file(file_path)

    def test_invalid_max_concurrent_boolean_false_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when max_concurrent is boolean False (DS-1031).

        In Python, bool is a subclass of int, so isinstance(False, int)
        returns True. The boolean guard ensures that YAML values like
        ``max_concurrent: false`` are rejected.
        """
        yaml_content = """
orchestrations:
  - name: "bool-false-limit"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    max_concurrent: false
"""
        file_path = tmp_path / "bool_false_max_concurrent.yaml"
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


class TestTimeoutSecondsValidation:
    """Tests for timeout_seconds validation with float support.

    These tests verify that _validate_timeout_seconds correctly handles:
    - Positive integers (valid)
    - Positive floats / fractional seconds (valid)
    - None (valid, no timeout)
    - NaN and infinity (invalid, must be rejected)
    - Boolean values (invalid, bool is subclass of int in Python)
    - Negative values, zero, and non-numeric types (invalid)
    """

    # --- Valid values ---

    def test_validate_none_is_valid(self) -> None:
        """None should be accepted (no timeout configured)."""
        assert _validate_timeout_seconds(None) is None

    def test_validate_positive_integer(self) -> None:
        """Positive integer should be accepted."""
        assert _validate_timeout_seconds(300) is None

    def test_validate_positive_float(self) -> None:
        """Positive float should be accepted for fractional seconds."""
        assert _validate_timeout_seconds(30.5) is None

    def test_validate_small_positive_float(self) -> None:
        """Small positive float (sub-second) should be accepted."""
        assert _validate_timeout_seconds(0.5) is None

    def test_validate_large_positive_float(self) -> None:
        """Large positive float should be accepted."""
        assert _validate_timeout_seconds(3600.75) is None

    # --- Invalid: NaN and infinity ---

    def test_validate_nan_rejected(self) -> None:
        """float('nan') must be rejected (IEEE 754 NaN causes undefined behavior)."""
        result = _validate_timeout_seconds(float("nan"))
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_positive_inf_rejected(self) -> None:
        """float('inf') must be rejected."""
        result = _validate_timeout_seconds(float("inf"))
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_negative_inf_rejected(self) -> None:
        """float('-inf') must be rejected."""
        result = _validate_timeout_seconds(float("-inf"))
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_math_nan_rejected(self) -> None:
        """math.nan must be rejected."""
        result = _validate_timeout_seconds(math.nan)
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_math_inf_rejected(self) -> None:
        """math.inf must be rejected."""
        result = _validate_timeout_seconds(math.inf)
        assert result is not None
        assert "must be a positive number" in result

    # --- Invalid: boolean values ---

    def test_validate_true_rejected(self) -> None:
        """True must be rejected (bool is subclass of int, True == 1)."""
        result = _validate_timeout_seconds(True)
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_false_rejected(self) -> None:
        """False must be rejected."""
        result = _validate_timeout_seconds(False)
        assert result is not None
        assert "must be a positive number" in result

    # --- Invalid: negative, zero, non-numeric ---

    def test_validate_zero_rejected(self) -> None:
        """Zero should be rejected (timeout must be positive)."""
        result = _validate_timeout_seconds(0)
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_negative_integer_rejected(self) -> None:
        """Negative integer should be rejected."""
        result = _validate_timeout_seconds(-5)
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_negative_float_rejected(self) -> None:
        """Negative float should be rejected."""
        result = _validate_timeout_seconds(-0.5)
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_string_rejected(self) -> None:
        """String should be rejected."""
        result = _validate_timeout_seconds("300")
        assert result is not None
        assert "must be a positive number" in result

    def test_validate_zero_float_rejected(self) -> None:
        """Zero as float should be rejected."""
        result = _validate_timeout_seconds(0.0)
        assert result is not None
        assert "must be a positive number" in result

    # --- YAML parsing integration tests ---

    def test_yaml_float_timeout_accepted(self, tmp_path: Path) -> None:
        """Fractional timeout_seconds from YAML should be accepted."""
        yaml_content = """
orchestrations:
  - name: "float-timeout"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: 30.5
"""
        file_path = tmp_path / "float_timeout.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds == 30.5

    def test_yaml_integer_timeout_accepted(self, tmp_path: Path) -> None:
        """Integer timeout_seconds from YAML should still be accepted."""
        yaml_content = """
orchestrations:
  - name: "int-timeout"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: 600
"""
        file_path = tmp_path / "int_timeout.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.timeout_seconds == 600

    def test_yaml_boolean_timeout_rejected(self, tmp_path: Path) -> None:
        """Boolean timeout_seconds from YAML should be rejected."""
        yaml_content = """
orchestrations:
  - name: "bool-timeout"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: true
"""
        file_path = tmp_path / "bool_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="timeout_seconds"):
            load_orchestration_file(file_path)

    def test_yaml_negative_timeout_rejected(self, tmp_path: Path) -> None:
        """Negative timeout_seconds from YAML should be rejected."""
        yaml_content = """
orchestrations:
  - name: "negative-timeout"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: -5
"""
        file_path = tmp_path / "negative_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="timeout_seconds"):
            load_orchestration_file(file_path)

    # --- get_effective_timeout with float input ---

    def test_effective_timeout_float_without_agent_teams(self) -> None:
        """get_effective_timeout should return float unchanged without agent_teams."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=100.5,
            agent_teams=False,
        )
        result = get_effective_timeout(config)
        assert result == 100.5

    def test_effective_timeout_float_with_agent_teams(self) -> None:
        """get_effective_timeout should apply multiplier to float with agent_teams."""
        config = AgentConfig(
            prompt="Test",
            timeout_seconds=600.5,
            agent_teams=True,
        )
        result = get_effective_timeout(config)
        # AGENT_TEAMS_TIMEOUT_MULTIPLIER is 3; 600.5 * 3 = 1801.5 > 900 min
        assert result == 600.5 * 3
