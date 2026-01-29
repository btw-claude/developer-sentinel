"""Tests for orchestration configuration validation.

This module contains tests for:
- GitHub repository format validation
- Orchestration enabled/disabled functionality
- max_concurrent field validation
"""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OrchestrationError,
    TriggerConfig,
    _validate_github_repo_format,
    load_orchestration_file,
)

"""Tests for orchestration configuration validation.

This module contains tests for validating orchestration configurations including
GitHub repository format validation, GitHub project trigger configuration,
orchestration enabled/disabled functionality, max_concurrent field validation,
labels and tags validation, and agent type configuration.
"""


class TestValidateGitHubRepoFormat:
    """Tests for _validate_github_repo_format function."""

    def test_empty_string_is_valid(self) -> None:
        """Empty string should be valid as the repo field is optional."""
        result = _validate_github_repo_format("")
        assert result.is_valid is True
        assert result.error_message == ""

    # Valid repository format tests
    @pytest.mark.parametrize(
        "repo_format",
        [
            "owner/repo",
            "org-name/repo-name",
            "user123/project456",
            "owner/my.project",
            "owner/repo.name.here",
            "owner/my_repo",
            "owner/repo_name_here",
            "a/b",
            "x/repo",
            "owner/y",
            "a" * 39 + "/repo",  # Max length owner (39 chars)
            "owner/" + "a" * 100,  # Max length repo (100 chars)
        ],
    )
    def test_valid_repo_format(self, repo_format: str) -> None:
        """Valid owner/repo formats should return True with empty error."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is True
        assert result.error_message == ""

    # Invalid format structure tests
    @pytest.mark.parametrize(
        "repo_format,expected_error",
        [
            ("repo-only", "owner/repo-name"),
            ("owner/repo/extra", "owner/repo-name"),
            ("a/b/c", "owner/repo-name"),
        ],
    )
    def test_invalid_format_structure(self, repo_format: str, expected_error: str) -> None:
        """Invalid format structures should return False with appropriate error."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert expected_error in result.error_message

    # Empty component tests
    @pytest.mark.parametrize(
        "repo_format,expected_error",
        [
            ("owner/", "repository name cannot be empty"),
            ("/repo", "owner cannot be empty"),
            ("  /repo", "owner cannot be empty"),
            ("owner/  ", "repository name cannot be empty"),
        ],
    )
    def test_empty_components(self, repo_format: str, expected_error: str) -> None:
        """Empty owner or repo components should be invalid."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert expected_error in result.error_message

    # Invalid owner format tests
    @pytest.mark.parametrize(
        "repo_format",
        [
            "-owner/repo",  # Starting with hyphen
            "-/repo",  # Just hyphen
            "owner-/repo",  # Ending with hyphen
            "owner--name/repo",  # Consecutive hyphens
            "a--b/repo",  # Consecutive hyphens
            "my_org/repo",  # Underscore
            "owner_name/repo",  # Underscore
            "my.org/repo",  # Period
            "owner!/repo",  # Special char !
            "owner@name/repo",  # Special char @
            "owner#/repo",  # Special char #
        ],
    )
    def test_invalid_owner_characters(self, repo_format: str) -> None:
        """Owner with invalid characters should be invalid."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert "invalid characters" in result.error_message

    # Owner exceeding max length test
    def test_owner_exceeding_max_length_is_invalid(self) -> None:
        """Owner exceeding 39 characters should be invalid."""
        long_owner = "a" * 40
        result = _validate_github_repo_format(f"{long_owner}/repo")
        assert result.is_valid is False
        assert "exceeds maximum length of 39 characters" in result.error_message
        assert "got 40" in result.error_message

    # Invalid repo format tests
    @pytest.mark.parametrize(
        "repo_format,expected_error",
        [
            ("owner/.repo", "cannot start with a period"),
            ("owner/.hidden", "cannot start with a period"),
            ("owner/repo.git", "cannot end with '.git'"),
            ("owner/my-project.git", "cannot end with '.git'"),
            ("owner/REPO.GIT", "cannot end with '.git'"),  # Case insensitive
        ],
    )
    def test_invalid_repo_prefix_suffix(self, repo_format: str, expected_error: str) -> None:
        """Repo with invalid prefix or suffix should be invalid."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert expected_error in result.error_message

    # Invalid repo special characters tests
    @pytest.mark.parametrize(
        "repo_format",
        [
            "owner/repo!",
            "owner/repo@name",
            "owner/repo#1",
            "owner/repo name",  # Spaces
        ],
    )
    def test_invalid_repo_special_characters(self, repo_format: str) -> None:
        """Repo with invalid special characters should be invalid."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert "invalid characters" in result.error_message

    # Repo exceeding max length test
    def test_repo_exceeding_max_length_is_invalid(self) -> None:
        """Repo exceeding 100 characters should be invalid."""
        long_repo = "a" * 101
        result = _validate_github_repo_format(f"owner/{long_repo}")
        assert result.is_valid is False
        assert "exceeds maximum length of 100 characters" in result.error_message
        assert "got 101" in result.error_message

    # Reserved names tests
    @pytest.mark.parametrize(
        "repo_format",
        [
            "./repo",  # Owner is reserved name '.'
            "../repo",  # Owner is reserved name '..'
            "owner/.",  # Repo is reserved name '.'
            "owner/..",  # Repo is reserved name '..'
        ],
    )
    def test_reserved_names_are_invalid(self, repo_format: str) -> None:
        """Reserved names '.' and '..' should be invalid."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert "reserved name" in result.error_message

    # Error message content tests
    @pytest.mark.parametrize(
        "repo_format,expected_in_error",
        [
            ("my_org/repo", "my_org"),  # Owner name in error
            ("owner/repo!", "repo!"),  # Repo name in error
        ],
    )
    def test_error_message_includes_invalid_name(
        self, repo_format: str, expected_in_error: str
    ) -> None:
        """Error message should include the invalid name."""
        result = _validate_github_repo_format(repo_format)
        assert result.is_valid is False
        assert expected_in_error in result.error_message

    def test_error_message_includes_actual_length(self) -> None:
        """Error message should include the actual length for length violations."""
        long_owner = "a" * 45
        result = _validate_github_repo_format(f"{long_owner}/repo")
        assert result.is_valid is False
        assert "got 45" in result.error_message


class TestInvalidGitHubRepoFormat:
    """Tests for invalid GitHub repo format validation in trigger parsing."""

    def test_invalid_repo_format_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid GitHub repo format when repo is provided."""
        yaml_content = """
orchestrations:
  - name: "invalid-repo"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      repo: "invalid-format"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_repo.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid GitHub repo format"):
            load_orchestration_file(file_path)

    def test_repo_with_too_many_slashes_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for repo with too many path components."""
        yaml_content = """
orchestrations:
  - name: "too-many-slashes"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
      repo: "org/repo/extra"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "too_many_slashes.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid GitHub repo format"):
            load_orchestration_file(file_path)

    def test_empty_repo_is_allowed(self, tmp_path: Path) -> None:
        """Empty repo should be allowed when using project-based triggers."""
        yaml_content = """
orchestrations:
  - name: "empty-repo"
    trigger:
      source: github
      project_number: 42
      project_owner: "my-org"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "empty_repo.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.repo == ""

    def test_jira_source_ignores_repo_validation(self, tmp_path: Path) -> None:
        """Jira source should not validate repo format (repo field is ignored)."""
        yaml_content = """
orchestrations:
  - name: "jira-with-invalid-repo"
    trigger:
      source: jira
      project: "TEST"
      repo: "this-is-ignored-for-jira"
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "jira_with_repo.yaml"
        file_path.write_text(yaml_content)

        # Should not raise error - repo field is ignored for Jira triggers
        orchestrations = load_orchestration_file(file_path)
        assert len(orchestrations) == 1
        assert orchestrations[0].trigger.source == "jira"


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
