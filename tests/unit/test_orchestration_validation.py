"""Tests for orchestration configuration validation.

This module contains tests for:
- Orchestration enabled/disabled functionality
- max_concurrent field validation
- timeout_seconds field validation (int, float, NaN, inf, bool edge cases)
- File-level github error collection (DS-1074, DS-1078)
- DRY shared helpers _validate_github_branch_field and
  _validate_github_string_field (DS-1083)
"""

import math
from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OrchestrationError,
    TriggerConfig,
    _collect_file_github_errors,
    _validate_github_base_branch,
    _validate_github_branch,
    _validate_github_branch_field,
    _validate_github_create_branch,
    _validate_github_host,
    _validate_github_org,
    _validate_github_repo,
    _validate_github_string_field,
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


class TestValidateGithubHelpers:
    """Tests for individual _validate_github_* helpers (DS-1078).

    Verifies that each helper correctly validates its field and returns
    ``str | None`` following the shared validation helper convention.
    """

    # --- _validate_github_branch ---

    def test_branch_valid_pattern(self) -> None:
        """Valid branch pattern should return None."""
        assert _validate_github_branch("feature/{jira_issue_key}") is None

    def test_branch_empty_is_valid(self) -> None:
        """Empty branch should return None (not configured)."""
        assert _validate_github_branch("") is None

    def test_branch_whitespace_only_is_valid(self) -> None:
        """Whitespace-only branch should return None (treated as empty)."""
        assert _validate_github_branch("   ") is None

    def test_branch_invalid_returns_error(self) -> None:
        """Invalid branch pattern should return an error message."""
        result = _validate_github_branch("invalid branch!@#")
        assert result is not None
        assert "Invalid branch pattern" in result

    # --- _validate_github_base_branch ---

    def test_base_branch_valid(self) -> None:
        """Valid base_branch should return None."""
        assert _validate_github_base_branch("main") is None

    def test_base_branch_empty_is_valid(self) -> None:
        """Empty base_branch should return None."""
        assert _validate_github_base_branch("") is None

    def test_base_branch_invalid_returns_error(self) -> None:
        """Invalid base_branch should return an error message."""
        result = _validate_github_base_branch("invalid branch!@#")
        assert result is not None
        assert "Invalid base_branch" in result

    # --- _validate_github_create_branch ---

    def test_create_branch_false_no_branch_is_valid(self) -> None:
        """create_branch=False with no branch is valid."""
        assert _validate_github_create_branch(False, "") is None

    def test_create_branch_true_with_branch_is_valid(self) -> None:
        """create_branch=True with a branch pattern is valid."""
        assert _validate_github_create_branch(True, "feature/{jira_issue_key}") is None

    def test_create_branch_true_no_branch_returns_error(self) -> None:
        """create_branch=True without branch pattern should return error."""
        result = _validate_github_create_branch(True, "")
        assert result is not None
        assert "create_branch is True" in result

    # --- _validate_github_host ---

    def test_host_valid(self) -> None:
        """Valid host should return None."""
        assert _validate_github_host("github.com") is None

    def test_host_none_is_valid(self) -> None:
        """None host should return None (falls back to default)."""
        assert _validate_github_host(None) is None

    def test_host_non_string_returns_error(self) -> None:
        """Non-string host should return an error message."""
        result = _validate_github_host(123)
        assert result is not None
        assert "github.host" in result

    def test_host_empty_string_returns_error(self) -> None:
        """Empty string host should return an error message."""
        result = _validate_github_host("")
        assert result is not None
        assert "github.host" in result

    def test_host_whitespace_only_returns_error(self) -> None:
        """Whitespace-only host should return an error message."""
        result = _validate_github_host("   ")
        assert result is not None
        assert "github.host" in result

    # --- _validate_github_org ---

    def test_org_valid(self) -> None:
        """Valid org should return None."""
        assert _validate_github_org("my-org") is None

    def test_org_none_is_valid(self) -> None:
        """None org should return None (falls back to default)."""
        assert _validate_github_org(None) is None

    def test_org_empty_string_is_valid(self) -> None:
        """Empty string org should return None (allowed)."""
        assert _validate_github_org("") is None

    def test_org_non_string_returns_error(self) -> None:
        """Non-string org should return an error message."""
        result = _validate_github_org(123)
        assert result is not None
        assert "github.org" in result

    def test_org_whitespace_only_returns_error(self) -> None:
        """Whitespace-only org should return an error message."""
        result = _validate_github_org("   ")
        assert result is not None
        assert "github.org" in result

    # --- _validate_github_repo ---

    def test_repo_valid(self) -> None:
        """Valid repo should return None."""
        assert _validate_github_repo("my-repo") is None

    def test_repo_none_is_valid(self) -> None:
        """None repo should return None (falls back to default)."""
        assert _validate_github_repo(None) is None

    def test_repo_empty_string_is_valid(self) -> None:
        """Empty string repo should return None (allowed)."""
        assert _validate_github_repo("") is None

    def test_repo_non_string_returns_error(self) -> None:
        """Non-string repo should return an error message."""
        result = _validate_github_repo(123)
        assert result is not None
        assert "github.repo" in result

    def test_repo_whitespace_only_returns_error(self) -> None:
        """Whitespace-only repo should return an error message."""
        result = _validate_github_repo("   ")
        assert result is not None
        assert "github.repo" in result


class TestCollectFileGithubErrors:
    """Tests for _collect_file_github_errors() (DS-1074, DS-1078).

    Verifies that the error-collecting variant individually validates each
    field (branch, base_branch, create_branch, host, org, repo) and collects
    all errors at once, prefixed with 'File-level github:'.

    DS-1078 refactored this function to validate each field independently
    (matching _collect_file_trigger_errors) instead of wrapping
    _parse_github_context() in a single try/except that only caught the
    first error.
    """

    def test_valid_config_returns_empty(self) -> None:
        """Valid file-level github config should produce no errors."""
        data = {"host": "github.com", "org": "my-org", "repo": "my-repo"}
        errors = _collect_file_github_errors(data)
        assert errors == []

    def test_invalid_branch_returns_error(self) -> None:
        """Invalid branch pattern should produce a prefixed error."""
        data = {"branch": "invalid branch!@#"}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "Invalid branch pattern" in errors[0]

    def test_create_branch_without_branch_returns_error(self) -> None:
        """create_branch=True without branch pattern should produce an error."""
        data = {"create_branch": True}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "create_branch is True" in errors[0]

    def test_invalid_base_branch_returns_error(self) -> None:
        """Invalid base_branch should produce a prefixed error."""
        data = {"base_branch": "invalid base!@#"}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "Invalid base_branch" in errors[0]

    def test_invalid_host_returns_error(self) -> None:
        """Non-string host should produce a prefixed error."""
        data = {"host": 123}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "github.host" in errors[0]

    def test_invalid_org_returns_error(self) -> None:
        """Non-string org should produce a prefixed error."""
        data = {"org": 456}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "github.org" in errors[0]

    def test_invalid_repo_returns_error(self) -> None:
        """Non-string repo should produce a prefixed error."""
        data = {"repo": 789}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "github.repo" in errors[0]

    def test_whitespace_host_returns_error(self) -> None:
        """Whitespace-only host should produce a prefixed error."""
        data = {"host": "   "}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "github.host" in errors[0]

    def test_empty_host_returns_error(self) -> None:
        """Empty string host should produce a prefixed error."""
        data = {"host": ""}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "File-level github:" in errors[0]
        assert "github.host" in errors[0]

    # --- DS-1078: Multi-error collection tests ---

    def test_multiple_errors_collected_independently(self) -> None:
        """All field errors should be collected, not just the first (DS-1078).

        This is the key behavioral change: previously, only the first error
        from _parse_github_context() was captured. Now each field is validated
        independently.
        """
        data = {
            "branch": "invalid branch!@#",
            "base_branch": "also invalid!@#",
            "host": 123,
            "org": 456,
            "repo": 789,
        }
        errors = _collect_file_github_errors(data)
        # Should have errors for branch, base_branch, host, org, and repo
        assert len(errors) >= 5
        # All should be prefixed
        for error in errors:
            assert "File-level github:" in error

    def test_branch_and_host_errors_both_collected(self) -> None:
        """Both branch and host errors should appear in the result (DS-1078)."""
        data = {"branch": "bad branch!@#", "host": 42}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 2
        branch_errors = [e for e in errors if "branch pattern" in e]
        host_errors = [e for e in errors if "github.host" in e]
        assert len(branch_errors) == 1
        assert len(host_errors) == 1

    def test_create_branch_and_host_errors_both_collected(self) -> None:
        """create_branch and host errors should both appear (DS-1078)."""
        data = {"create_branch": True, "host": 123}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 2
        create_errors = [e for e in errors if "create_branch" in e]
        host_errors = [e for e in errors if "github.host" in e]
        assert len(create_errors) == 1
        assert len(host_errors) == 1

    def test_all_six_fields_invalid_collects_all_errors(self) -> None:
        """When all six fields are invalid, all errors should be collected (DS-1078).

        Note: create_branch triggers when branch is empty/whitespace-only, so we
        use an empty branch to trigger that error alongside the other field errors.
        """
        data = {
            "branch": "",
            "base_branch": "bad base!@#",
            "create_branch": True,  # invalid because branch is empty
            "host": 123,
            "org": 456,
            "repo": 789,
        }
        errors = _collect_file_github_errors(data)
        # base_branch, create_branch, host, org, repo = 5 errors
        # (empty branch is valid - it means "not configured")
        assert len(errors) == 5
        for error in errors:
            assert "File-level github:" in error

    def test_branch_and_base_branch_and_host_all_invalid(self) -> None:
        """Three different field errors should all be collected (DS-1078)."""
        data = {
            "branch": "bad branch!@#",
            "base_branch": "bad base!@#",
            "host": 123,
        }
        errors = _collect_file_github_errors(data)
        assert len(errors) == 3
        branch_errors = [e for e in errors if "branch pattern" in e]
        base_errors = [e for e in errors if "base_branch" in e]
        host_errors = [e for e in errors if "github.host" in e]
        assert len(branch_errors) == 1
        assert len(base_errors) == 1
        assert len(host_errors) == 1

    def test_valid_branch_with_invalid_host_collects_only_host_error(self) -> None:
        """Valid branch but invalid host should only report host error."""
        data = {"branch": "feature/{jira_issue_key}", "host": 123}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 1
        assert "github.host" in errors[0]

    def test_invalid_org_and_repo_collects_both(self) -> None:
        """Invalid org and repo should both be reported (DS-1078)."""
        data = {"org": True, "repo": False}
        errors = _collect_file_github_errors(data)
        assert len(errors) == 2
        org_errors = [e for e in errors if "github.org" in e]
        repo_errors = [e for e in errors if "github.repo" in e]
        assert len(org_errors) == 1
        assert len(repo_errors) == 1

    def test_empty_dict_returns_no_errors(self) -> None:
        """Empty dict (all defaults) should produce no errors."""
        errors = _collect_file_github_errors({})
        assert errors == []

    def test_valid_full_config_returns_no_errors(self) -> None:
        """Fully valid config with all fields should produce no errors."""
        data = {
            "host": "github.com",
            "org": "my-org",
            "repo": "my-repo",
            "branch": "feature/{jira_issue_key}",
            "base_branch": "develop",
            "create_branch": True,
        }
        errors = _collect_file_github_errors(data)
        assert errors == []


class TestValidateGithubBranchField:
    """Tests for _validate_github_branch_field() shared helper (DS-1083).

    Verifies that the consolidated branch validation helper correctly
    validates branch-type fields using a parameterised field_name, and
    that the thin wrappers (_validate_github_branch and
    _validate_github_base_branch) produce identical results.
    """

    # --- Direct shared helper tests ---

    def test_valid_branch_pattern(self) -> None:
        """Valid branch pattern should return None."""
        assert _validate_github_branch_field("feature/{jira_issue_key}", "branch pattern") is None

    def test_valid_base_branch(self) -> None:
        """Valid base branch should return None."""
        assert _validate_github_branch_field("main", "base_branch") is None

    def test_empty_string_is_valid(self) -> None:
        """Empty string should return None (not configured)."""
        assert _validate_github_branch_field("", "branch pattern") is None

    def test_whitespace_only_is_valid(self) -> None:
        """Whitespace-only string should return None (treated as empty)."""
        assert _validate_github_branch_field("   ", "base_branch") is None

    def test_invalid_returns_error_with_field_name(self) -> None:
        """Invalid value should return error containing the field_name."""
        result = _validate_github_branch_field("invalid branch!@#", "branch pattern")
        assert result is not None
        assert "Invalid branch pattern" in result

    def test_invalid_base_branch_field_name_in_error(self) -> None:
        """Error message should use the provided field_name."""
        result = _validate_github_branch_field("bad base!@#", "base_branch")
        assert result is not None
        assert "Invalid base_branch" in result

    def test_custom_field_name_in_error(self) -> None:
        """A custom field_name should appear in the error message."""
        result = _validate_github_branch_field("bad~branch", "my_custom_field")
        assert result is not None
        assert "Invalid my_custom_field" in result

    # --- Wrapper equivalence tests ---

    def test_wrapper_branch_matches_shared_helper(self) -> None:
        """_validate_github_branch should produce same result as shared helper."""
        for value in ["feature/test", "", "   ", "invalid branch!@#"]:
            wrapper_result = _validate_github_branch(value)
            direct_result = _validate_github_branch_field(value, "branch pattern")
            assert wrapper_result == direct_result, f"Mismatch for value={value!r}"

    def test_wrapper_base_branch_matches_shared_helper(self) -> None:
        """_validate_github_base_branch should produce same result as shared helper."""
        for value in ["main", "develop", "", "invalid branch!@#"]:
            wrapper_result = _validate_github_base_branch(value)
            direct_result = _validate_github_branch_field(value, "base_branch")
            assert wrapper_result == direct_result, f"Mismatch for value={value!r}"


class TestValidateGithubStringField:
    """Tests for _validate_github_string_field() shared helper (DS-1083).

    Verifies that the consolidated string-field validation helper correctly
    wraps _validate_string_field() and converts OrchestrationError to a
    plain error string, and that the thin wrappers (_validate_github_host,
    _validate_github_org, _validate_github_repo) produce identical results.
    """

    # --- Direct shared helper tests ---

    def test_valid_string(self) -> None:
        """Valid string value should return None."""
        assert _validate_github_string_field("github.com", "github.host") is None

    def test_none_is_valid(self) -> None:
        """None should return None (falls back to default)."""
        assert _validate_github_string_field(None, "github.host") is None

    def test_non_string_returns_error(self) -> None:
        """Non-string value should return an error message."""
        result = _validate_github_string_field(123, "github.host")
        assert result is not None
        assert "github.host" in result

    def test_whitespace_only_returns_error(self) -> None:
        """Whitespace-only string should return an error message."""
        result = _validate_github_string_field("   ", "github.host")
        assert result is not None
        assert "github.host" in result

    def test_empty_string_rejected_by_default(self) -> None:
        """Empty string should be rejected when reject_empty is True (default)."""
        result = _validate_github_string_field("", "github.host")
        assert result is not None
        assert "github.host" in result

    def test_empty_string_accepted_with_reject_empty_false(self) -> None:
        """Empty string should be accepted when reject_empty=False."""
        assert _validate_github_string_field("", "github.org", reject_empty=False) is None

    def test_custom_field_name_in_error(self) -> None:
        """A custom field_name should appear in the error message."""
        result = _validate_github_string_field(42, "github.custom")
        assert result is not None
        assert "github.custom" in result

    def test_kwargs_forwarded_to_validate_string_field(self) -> None:
        """reject_empty kwarg should be forwarded correctly."""
        # With reject_empty=True (default), empty string is rejected
        assert _validate_github_string_field("", "test.field") is not None
        # With reject_empty=False, empty string is accepted
        assert _validate_github_string_field("", "test.field", reject_empty=False) is None

    # --- Wrapper equivalence tests ---

    def test_wrapper_host_matches_shared_helper(self) -> None:
        """_validate_github_host should produce same result as shared helper."""
        for value in ["github.com", None, 123, "", "   "]:
            wrapper_result = _validate_github_host(value)
            direct_result = _validate_github_string_field(value, "github.host")
            assert wrapper_result == direct_result, f"Mismatch for value={value!r}"

    def test_wrapper_org_matches_shared_helper(self) -> None:
        """_validate_github_org should produce same result as shared helper."""
        for value in ["my-org", None, 123, "", "   "]:
            wrapper_result = _validate_github_org(value)
            direct_result = _validate_github_string_field(
                value, "github.org", reject_empty=False
            )
            assert wrapper_result == direct_result, f"Mismatch for value={value!r}"

    def test_wrapper_repo_matches_shared_helper(self) -> None:
        """_validate_github_repo should produce same result as shared helper."""
        for value in ["my-repo", None, 123, "", "   "]:
            wrapper_result = _validate_github_repo(value)
            direct_result = _validate_github_string_field(
                value, "github.repo", reject_empty=False
            )
            assert wrapper_result == direct_result, f"Mismatch for value={value!r}"
