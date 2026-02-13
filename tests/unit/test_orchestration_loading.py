"""Tests for orchestration file loading.

This module contains tests for loading orchestration configurations from YAML files
and directories.
"""

from pathlib import Path

import pytest

from sentinel.orchestration import OrchestrationError, load_orchestration_file, load_orchestrations


class TestLoadOrchestrationFile:
    """Tests for load_orchestration_file function."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        """Should load a valid orchestration file."""
        yaml_content = """
orchestrations:
  - name: "test-orch"
    trigger:
      source: jira
      project: "TEST"
      tags:
        - "test-tag"
    agent:
      prompt: "Test prompt"
      tools:
        - jira
    retry:
      max_attempts: 5
    on_complete:
      remove_tag: "test-tag"
      add_tag: "processed"
    on_failure:
      add_tag: "agent-failed"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.name == "test-orch"
        assert orch.trigger.project == "TEST"
        assert orch.trigger.tags == ["test-tag"]
        assert orch.agent.prompt == "Test prompt"
        assert orch.retry.max_attempts == 5
        assert orch.on_complete.remove_tag == "test-tag"
        assert orch.on_complete.add_tag == "processed"
        assert orch.on_failure.add_tag == "agent-failed"

    def test_load_file_with_github_context(self, tmp_path: Path) -> None:
        """Should load orchestration with GitHub context."""
        yaml_content = """
orchestrations:
  - name: "github-orch"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review code"
      tools:
        - github
      github:
        host: "github.com"
        org: "test-org"
        repo: "test-repo"
"""
        file_path = tmp_path / "github.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.github is not None
        assert orch.agent.github.host == "github.com"
        assert orch.agent.github.org == "test-org"
        assert orch.agent.github.repo == "test-repo"
        # Verify defaults for new branch fields
        assert orch.agent.github.branch == ""
        assert orch.agent.github.create_branch is False
        assert orch.agent.github.base_branch == "main"

    def test_load_file_with_github_context_branch_fields(self, tmp_path: Path) -> None:
        """Should load orchestration with GitHub context branch fields."""
        yaml_content = """
orchestrations:
  - name: "github-branch-orch"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review code"
      tools:
        - github
      github:
        host: "github.com"
        org: "test-org"
        repo: "test-repo"
        branch: "feature/{jira_issue_key}"
        create_branch: true
        base_branch: "develop"
"""
        file_path = tmp_path / "github-branch.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.github is not None
        assert orch.agent.github.host == "github.com"
        assert orch.agent.github.org == "test-org"
        assert orch.agent.github.repo == "test-repo"
        assert orch.agent.github.branch == "feature/{jira_issue_key}"
        assert orch.agent.github.create_branch is True
        assert orch.agent.github.base_branch == "develop"

    def test_load_file_with_timeout_seconds(self, tmp_path: Path) -> None:
        """Should load orchestration with timeout_seconds."""
        yaml_content = """
orchestrations:
  - name: "timeout-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      timeout_seconds: 300
"""
        file_path = tmp_path / "timeout.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.timeout_seconds == 300

    def test_load_file_with_on_start(self, tmp_path: Path) -> None:
        """Should load orchestration with on_start config."""
        yaml_content = """
orchestrations:
  - name: "on-start-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
    on_start:
      add_tag: "sentinel-processing"
"""
        file_path = tmp_path / "on_start.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.on_start.add_tag == "sentinel-processing"

    def test_load_file_without_on_start_uses_defaults(self, tmp_path: Path) -> None:
        """Should use default on_start config when not specified."""
        yaml_content = """
orchestrations:
  - name: "no-on-start-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "no_on_start.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.on_start.add_tag == ""

    def test_invalid_timeout_seconds_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid timeout_seconds."""
        yaml_content = """
orchestrations:
  - name: "invalid-timeout"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      timeout_seconds: -5
"""
        file_path = tmp_path / "invalid_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid timeout_seconds"):
            load_orchestration_file(file_path)

    def test_zero_timeout_seconds_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for zero timeout_seconds."""
        yaml_content = """
orchestrations:
  - name: "zero-timeout"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      timeout_seconds: 0
"""
        file_path = tmp_path / "zero_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid timeout_seconds"):
            load_orchestration_file(file_path)

    def test_boolean_timeout_seconds_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for boolean timeout_seconds (DS-1031).

        In Python, bool is a subclass of int, so isinstance(True, int)
        returns True. The boolean guard ensures that YAML values like
        ``timeout_seconds: true`` are rejected instead of being silently
        accepted as numeric values (True == 1, False == 0).
        """
        yaml_content = """
orchestrations:
  - name: "bool-timeout"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      timeout_seconds: true
"""
        file_path = tmp_path / "bool_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid timeout_seconds"):
            load_orchestration_file(file_path)

    def test_boolean_false_timeout_seconds_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for boolean False timeout_seconds (DS-1031).

        Symmetric with test_boolean_timeout_seconds_raises_error. Since
        ``bool`` is a subclass of ``int`` in Python, ``False`` (== 0) must
        also be rejected by the explicit boolean guard rather than falling
        through to the numeric validation.
        """
        yaml_content = """
orchestrations:
  - name: "bool-false-timeout"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      timeout_seconds: false
"""
        file_path = tmp_path / "bool_false_timeout.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid timeout_seconds"):
            load_orchestration_file(file_path)

    def test_load_file_with_model(self, tmp_path: Path) -> None:
        """Should load orchestration with model specified."""
        yaml_content = """
orchestrations:
  - name: "model-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      model: "claude-opus-4-5-20251101"
"""
        file_path = tmp_path / "model.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.model == "claude-opus-4-5-20251101"

    def test_load_file_without_model_uses_none(self, tmp_path: Path) -> None:
        """Should default to None when model is not specified."""
        yaml_content = """
orchestrations:
  - name: "no-model-orch"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "no_model.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.model is None

    def test_invalid_model_type_raises_error(self, tmp_path: Path) -> None:
        """Should raise error when model is not a string."""
        yaml_content = """
orchestrations:
  - name: "invalid-model"
    trigger:
      source: jira
    agent:
      prompt: "Test"
      model: 123
"""
        file_path = tmp_path / "invalid_model.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid model"):
            load_orchestration_file(file_path)

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Should return empty list for empty file."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")

        orchestrations = load_orchestration_file(file_path)
        assert orchestrations == []

    def test_load_file_without_orchestrations_key(self, tmp_path: Path) -> None:
        """Should return empty list if no orchestrations key."""
        yaml_content = """
some_other_key: value
"""
        file_path = tmp_path / "no_orch.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)
        assert orchestrations == []

    def test_missing_name_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if orchestration has no name."""
        yaml_content = """
orchestrations:
  - trigger:
      source: jira
    agent:
      prompt: "test"
"""
        file_path = tmp_path / "no_name.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must have a 'name' field"):
            load_orchestration_file(file_path)

    def test_missing_trigger_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if orchestration has no trigger."""
        yaml_content = """
orchestrations:
  - name: "test"
    agent:
      prompt: "test"
"""
        file_path = tmp_path / "no_trigger.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must have a 'trigger' field"):
            load_orchestration_file(file_path)

    def test_missing_agent_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if orchestration has no agent."""
        yaml_content = """
orchestrations:
  - name: "test"
    trigger:
      source: jira
"""
        file_path = tmp_path / "no_agent.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must have an 'agent' field"):
            load_orchestration_file(file_path)

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if file does not exist."""
        file_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(OrchestrationError, match="not found"):
            load_orchestration_file(file_path)

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid YAML."""
        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("{ invalid yaml: [")

        with pytest.raises(OrchestrationError, match="Invalid YAML"):
            load_orchestration_file(file_path)

    def test_load_with_default_status_failure(self, tmp_path: Path) -> None:
        """Should load orchestration with default_status='failure'."""
        yaml_content = """
orchestrations:
  - name: "strict-orch"
    trigger:
      source: jira
      project: "TEST"
      tags: ["review"]
    agent:
      prompt: "Review carefully"
    retry:
      max_attempts: 3
      default_status: failure
"""
        file_path = tmp_path / "strict.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].retry.default_status == "failure"

    def test_invalid_default_status_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid default_status value."""
        yaml_content = """
orchestrations:
  - name: "invalid-orch"
    trigger:
      source: jira
      project: "TEST"
    agent:
      prompt: "Test"
    retry:
      default_status: maybe
"""
        file_path = tmp_path / "invalid_status.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid default_status"):
            load_orchestration_file(file_path)

    def test_load_file_with_github_trigger(self, tmp_path: Path) -> None:
        """Should load orchestration with GitHub trigger configuration."""
        yaml_content = """
orchestrations:
  - name: "github-issue-triage"
    trigger:
      source: github
      project_number: 42
      project_scope: org
      project_owner: "my-organization"
      project_filter: "Status = 'Needs Triage'"
    agent:
      prompt: "Triage the GitHub issue"
      tools:
        - github
"""
        file_path = tmp_path / "github_trigger.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.name == "github-issue-triage"
        assert orch.trigger.source == "github"
        assert orch.trigger.project_number == 42
        assert orch.trigger.project_scope == "org"
        assert orch.trigger.project_owner == "my-organization"
        assert orch.trigger.project_filter == "Status = 'Needs Triage'"

    def test_invalid_trigger_source_raises_error(self, tmp_path: Path) -> None:
        """Should raise error for invalid trigger source value."""
        yaml_content = """
orchestrations:
  - name: "invalid-source"
    trigger:
      source: gitlab
      project: "TEST"
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "invalid_source.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid trigger source"):
            load_orchestration_file(file_path)

    def test_load_file_with_outcomes(self, tmp_path: Path) -> None:
        """Should load orchestration with outcomes configuration."""
        yaml_content = """
orchestrations:
  - name: "code-review"
    trigger:
      source: jira
      project: "TEST"
      tags: ["needs-review"]
    agent:
      prompt: "Review the code"
    outcomes:
      - name: approved
        patterns: ["APPROVED", "LGTM"]
        add_tag: code-reviewed
      - name: changes-requested
        patterns: ["CHANGES REQUESTED", "REQUEST CHANGES"]
        add_tag: changes-requested
    retry:
      failure_patterns: ["ERROR", "FAILED"]
      default_outcome: approved
"""
        file_path = tmp_path / "outcomes.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert len(orch.outcomes) == 2
        assert orch.outcomes[0].name == "approved"
        assert orch.outcomes[0].patterns == ["APPROVED", "LGTM"]
        assert orch.outcomes[0].add_tag == "code-reviewed"
        assert orch.outcomes[1].name == "changes-requested"
        assert orch.retry.default_outcome == "approved"

    def test_outcome_missing_name_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if outcome has no name."""
        yaml_content = """
orchestrations:
  - name: "test"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    outcomes:
      - patterns: ["SUCCESS"]
        add_tag: done
"""
        file_path = tmp_path / "no_name.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must have a 'name' field"):
            load_orchestration_file(file_path)

    def test_outcome_missing_patterns_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if outcome has no patterns."""
        yaml_content = """
orchestrations:
  - name: "test"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    outcomes:
      - name: done
        add_tag: completed
"""
        file_path = tmp_path / "no_patterns.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must have at least one pattern"):
            load_orchestration_file(file_path)

    def test_duplicate_outcome_names_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if outcome names are not unique."""
        yaml_content = """
orchestrations:
  - name: "test"
    trigger:
      source: jira
    agent:
      prompt: "Test"
    outcomes:
      - name: done
        patterns: ["SUCCESS"]
      - name: done
        patterns: ["COMPLETED"]
"""
        file_path = tmp_path / "duplicate_names.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="must be unique"):
            load_orchestration_file(file_path)

    def test_load_file_with_regex_outcome_patterns(self, tmp_path: Path) -> None:
        """Should load orchestration with regex patterns in outcomes."""
        yaml_content = """
orchestrations:
  - name: "regex-test"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test"
    outcomes:
      - name: approved
        patterns:
          - "regex:^APPROVED"
          - "regex:LGTM.*approved"
        add_tag: reviewed
"""
        file_path = tmp_path / "regex_outcomes.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].outcomes[0].patterns == [
            "regex:^APPROVED",
            "regex:LGTM.*approved",
        ]


class TestLoadOrchestrations:
    """Tests for load_orchestrations function."""

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Should load all YAML files from directory."""
        # Create two orchestration files
        (tmp_path / "first.yaml").write_text("""
orchestrations:
  - name: "first"
    trigger:
      source: jira
      tags: ["tag1"]
    agent:
      prompt: "First"
""")
        (tmp_path / "second.yml").write_text("""
orchestrations:
  - name: "second"
    trigger:
      source: jira
      tags: ["tag2"]
    agent:
      prompt: "Second"
""")
        # Create a non-YAML file that should be ignored
        (tmp_path / "readme.txt").write_text("ignore me")

        orchestrations = load_orchestrations(tmp_path)

        assert len(orchestrations) == 2
        names = [o.name for o in orchestrations]
        assert "first" in names
        assert "second" in names

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Should return empty list for empty directory."""
        orchestrations = load_orchestrations(tmp_path)
        assert orchestrations == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Should return empty list for nonexistent directory."""
        orchestrations = load_orchestrations(tmp_path / "nonexistent")
        assert orchestrations == []

    def test_not_a_directory_raises_error(self, tmp_path: Path) -> None:
        """Should raise error if path is not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")

        with pytest.raises(OrchestrationError, match="not a directory"):
            load_orchestrations(file_path)


class TestLoadOrchestrationFileStepsKey:
    """Tests for steps/orchestrations key resolution in orchestration loading (DS-900).

    Ensures the shared resolve_steps_key() utility is correctly integrated
    into _load_orchestration_file_with_counts() and that the empty list
    edge case (DS-899) is handled properly at the orchestration loading level.
    """

    def test_load_file_with_steps_key(self, tmp_path: Path) -> None:
        """Should load orchestrations from 'steps' key (new format)."""
        yaml_content = """
trigger:
  source: jira
  project: TEST
steps:
  - name: "step-one"
    trigger:
      tags:
        - "tag1"
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].name == "step-one"

    def test_load_file_with_empty_steps_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty list when 'steps' key has empty list.

        Verifies the DS-899 fix at the orchestration loading level: an empty
        list [] under 'steps' must not fall through to 'orchestrations'.
        """
        yaml_content = """
trigger:
  source: jira
  project: TEST
steps: []
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert orchestrations == []

    def test_load_file_empty_steps_not_fallthrough_to_orchestrations(
        self, tmp_path: Path
    ) -> None:
        """Should not fall through to 'orchestrations' when 'steps' is empty.

        If a file has both 'steps: []' and an 'orchestrations' key, the
        loader must return empty, not load the orchestrations list.
        """
        yaml_content = """
steps: []
orchestrations:
  - name: "should-not-load"
    trigger:
      source: jira
      tags:
        - "tag1"
    agent:
      prompt: "This should not be loaded"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert orchestrations == []

    def test_load_file_with_legacy_orchestrations_key(self, tmp_path: Path) -> None:
        """Should still load from 'orchestrations' key when 'steps' is absent."""
        yaml_content = """
orchestrations:
  - name: "legacy-orch"
    trigger:
      source: jira
      tags:
        - "tag1"
    agent:
      prompt: "Legacy prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].name == "legacy-orch"

    def test_load_file_with_neither_key_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty list when neither 'steps' nor 'orchestrations' exists."""
        yaml_content = """
trigger:
  source: jira
  project: TEST
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert orchestrations == []


class TestFileGithubLoading:
    """Tests for file-level github context loading (DS-1074).

    Verifies that file-level github: blocks in orchestration YAML files are
    correctly parsed, validated, and merged into each step's agent.github
    during loading via _load_orchestration_file_with_counts().
    """

    def test_single_step_merge(self, tmp_path: Path) -> None:
        """File-level github should be merged into a single step's agent.github."""
        yaml_content = """
github:
  host: "github.com"
  org: "my-org"
  repo: "my-repo"
steps:
  - name: "test-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.github is not None
        assert orch.agent.github.host == "github.com"
        assert orch.agent.github.org == "my-org"
        assert orch.agent.github.repo == "my-repo"

    def test_multi_step_merge(self, tmp_path: Path) -> None:
        """File-level github should be merged into every step."""
        yaml_content = """
github:
  host: "github.com"
  org: "my-org"
  repo: "my-repo"
steps:
  - name: "step-one"
    trigger:
      source: jira
      tags: ["tag1"]
    agent:
      prompt: "First prompt"
  - name: "step-two"
    trigger:
      source: jira
      tags: ["tag2"]
    agent:
      prompt: "Second prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 2
        for orch in orchestrations:
            assert orch.agent.github is not None
            assert orch.agent.github.host == "github.com"
            assert orch.agent.github.org == "my-org"
            assert orch.agent.github.repo == "my-repo"

    def test_step_level_override(self, tmp_path: Path) -> None:
        """Step-level agent.github fields should override file-level values."""
        yaml_content = """
github:
  host: "github.com"
  org: "file-org"
  repo: "file-repo"
steps:
  - name: "override-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      github:
        org: "step-org"
        repo: "step-repo"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.github is not None
        # Step-level values should take precedence
        assert orch.agent.github.org == "step-org"
        assert orch.agent.github.repo == "step-repo"
        # File-level value should fill in the missing host
        assert orch.agent.github.host == "github.com"

    def test_partial_override(self, tmp_path: Path) -> None:
        """Step-level should override only specified fields; others inherit from file."""
        yaml_content = """
github:
  host: "github.com"
  org: "file-org"
  repo: "file-repo"
  branch: "feature/{jira_issue_key}"
  create_branch: true
  base_branch: "develop"
steps:
  - name: "partial-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
      github:
        org: "step-org"
        base_branch: "main"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        github = orchestrations[0].agent.github
        assert github is not None
        # Step-level overrides
        assert github.org == "step-org"
        assert github.base_branch == "main"
        # File-level defaults inherited
        assert github.host == "github.com"
        assert github.repo == "file-repo"
        assert github.branch == "feature/{jira_issue_key}"
        assert github.create_branch is True

    def test_backward_compat_without_file_github(self, tmp_path: Path) -> None:
        """Files without file-level github should work unchanged."""
        yaml_content = """
steps:
  - name: "no-github-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is None

    def test_invalid_file_level_github_raises_error(self, tmp_path: Path) -> None:
        """Invalid file-level github should raise OrchestrationError."""
        yaml_content = """
github:
  branch: "invalid branch!@#"
steps:
  - name: "test-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid branch pattern"):
            load_orchestration_file(file_path)

    def test_both_steps_and_orchestrations_keys(self, tmp_path: Path) -> None:
        """File-level github should work with both 'steps' and 'orchestrations' keys."""
        yaml_content = """
github:
  host: "github.com"
  org: "my-org"
  repo: "my-repo"
steps: []
orchestrations:
  - name: "should-not-load"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # steps: [] takes priority; orchestrations key is ignored (DS-899)
        orchestrations = load_orchestration_file(file_path)
        assert orchestrations == []

    def test_orchestrations_key_only_with_file_github(self, tmp_path: Path) -> None:
        """File-level github should merge through the 'orchestrations' loading path.

        Companion to test_both_steps_and_orchestrations_keys which uses 'steps'.
        Verifies that when only the 'orchestrations' key is present (no 'steps'
        key), file-level github is still correctly merged into each
        orchestration's agent.github.
        """
        yaml_content = """
github:
  host: "github.com"
  org: "my-org"
  repo: "my-repo"
orchestrations:
  - name: "orch-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        assert orch.agent.github is not None
        assert orch.agent.github.host == "github.com"
        assert orch.agent.github.org == "my-org"
        assert orch.agent.github.repo == "my-repo"

    def test_non_dict_file_github_ignored(self, tmp_path: Path) -> None:
        """Non-dict file-level github values should be silently ignored."""
        yaml_content = """
github: "not-a-dict"
steps:
  - name: "test-step"
    trigger:
      source: jira
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        # github should not be set since file-level was not a dict
        assert orchestrations[0].agent.github is None

    def test_combined_with_file_level_trigger(self, tmp_path: Path) -> None:
        """File-level github should work alongside file-level trigger."""
        yaml_content = """
trigger:
  source: jira
  project: "TEST"
github:
  host: "github.com"
  org: "my-org"
  repo: "my-repo"
steps:
  - name: "combined-step"
    trigger:
      tags: ["test"]
    agent:
      prompt: "Test prompt"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        orchestrations = load_orchestration_file(file_path)

        assert len(orchestrations) == 1
        orch = orchestrations[0]
        # File-level trigger should be merged
        assert orch.trigger.source == "jira"
        assert orch.trigger.project == "TEST"
        # File-level github should be merged
        assert orch.agent.github is not None
        assert orch.agent.github.host == "github.com"
        assert orch.agent.github.org == "my-org"
        assert orch.agent.github.repo == "my-repo"
