"""Tests for orchestration configuration loading."""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OnCompleteConfig,
    OnFailureConfig,
    OnStartConfig,
    Orchestration,
    OrchestrationError,
    OrchestrationVersion,
    Outcome,
    RetryConfig,
    TriggerConfig,
    ValidationResult,
    _validate_github_repo_format,
    load_orchestration_file,
    load_orchestrations,
)


class TestDataclasses:
    """Tests for orchestration dataclasses."""

    def test_trigger_config_defaults(self) -> None:
        """TriggerConfig should have sensible defaults."""
        trigger = TriggerConfig()
        assert trigger.source == "jira"
        assert trigger.project == ""
        assert trigger.jql_filter == ""
        assert trigger.tags == []
        assert trigger.repo == ""
        assert trigger.query_filter == ""

    def test_trigger_config_github_source(self) -> None:
        """TriggerConfig should support github source."""
        trigger = TriggerConfig(
            source="github",
            repo="org/repo-name",
            query_filter="is:issue is:open label:bug",
            tags=["needs-triage"],
        )
        assert trigger.source == "github"
        assert trigger.repo == "org/repo-name"
        assert trigger.query_filter == "is:issue is:open label:bug"
        assert trigger.tags == ["needs-triage"]

    def test_agent_config_defaults(self) -> None:
        """AgentConfig should have sensible defaults."""
        agent = AgentConfig()
        assert agent.prompt == ""
        assert agent.tools == []
        assert agent.github is None
        assert agent.timeout_seconds is None

    def test_retry_config_defaults(self) -> None:
        """RetryConfig should have sensible defaults."""
        retry = RetryConfig()
        assert retry.max_attempts == 3
        assert "SUCCESS" in retry.success_patterns
        assert "FAILURE" in retry.failure_patterns
        assert retry.default_status == "success"

    def test_retry_config_default_status_failure(self) -> None:
        """RetryConfig can be configured with default_status='failure'."""
        retry = RetryConfig(default_status="failure")
        assert retry.default_status == "failure"

    def test_on_complete_config_defaults(self) -> None:
        """OnCompleteConfig should have sensible defaults."""
        on_complete = OnCompleteConfig()
        assert on_complete.remove_tag == ""
        assert on_complete.add_tag == ""

    def test_on_failure_config_defaults(self) -> None:
        """OnFailureConfig should have sensible defaults."""
        on_failure = OnFailureConfig()
        assert on_failure.add_tag == ""

    def test_on_start_config_defaults(self) -> None:
        """OnStartConfig should have sensible defaults."""
        on_start = OnStartConfig()
        assert on_start.add_tag == ""

    def test_on_start_config_with_tag(self) -> None:
        """OnStartConfig can be configured with a tag."""
        on_start = OnStartConfig(add_tag="processing")
        assert on_start.add_tag == "processing"

    def test_outcome_defaults(self) -> None:
        """Outcome should have sensible defaults."""
        outcome = Outcome()
        assert outcome.name == ""
        assert outcome.patterns == []
        assert outcome.add_tag == ""

    def test_outcome_with_values(self) -> None:
        """Outcome can be configured with values."""
        outcome = Outcome(
            name="approved",
            patterns=["APPROVED", "LGTM"],
            add_tag="code-reviewed",
        )
        assert outcome.name == "approved"
        assert outcome.patterns == ["APPROVED", "LGTM"]
        assert outcome.add_tag == "code-reviewed"

    def test_retry_config_default_outcome(self) -> None:
        """RetryConfig can be configured with default_outcome."""
        retry = RetryConfig(default_outcome="approved")
        assert retry.default_outcome == "approved"

    def test_orchestration_outcomes_defaults_to_empty(self) -> None:
        """Orchestration outcomes should default to empty list."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        assert orch.outcomes == []

    def test_orchestration_enabled_defaults_to_true(self) -> None:
        """Orchestration enabled should default to True."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        assert orch.enabled is True

    def test_orchestration_enabled_can_be_set_to_false(self) -> None:
        """Orchestration enabled can be set to False."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
            enabled=False,
        )
        assert orch.enabled is False


class TestOrchestrationVersion:
    """Tests for OrchestrationVersion dataclass."""

    def test_create_generates_unique_version_id(self) -> None:
        """OrchestrationVersion.create should generate unique version IDs."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version1 = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)
        version2 = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        assert version1.version_id != version2.version_id
        assert len(version1.version_id) == 36  # UUID format

    def test_create_sets_all_fields(self) -> None:
        """OrchestrationVersion.create should set all fields correctly."""
        orch = Orchestration(
            name="test-orch",
            trigger=TriggerConfig(project="TEST"),
            agent=AgentConfig(prompt="Test prompt"),
        )
        source_file = Path("/path/to/test.yaml")
        mtime = 1234567890.5

        version = OrchestrationVersion.create(orch, source_file, mtime)

        assert version.orchestration is orch
        assert version.source_file == source_file
        assert version.mtime == mtime
        assert version.active_executions == 0

    def test_name_property_returns_orchestration_name(self) -> None:
        """name property should return the orchestration name."""
        orch = Orchestration(
            name="my-orchestration",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        assert version.name == "my-orchestration"

    def test_increment_executions(self) -> None:
        """increment_executions should increase the count."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        assert version.active_executions == 0
        version.increment_executions()
        assert version.active_executions == 1
        version.increment_executions()
        assert version.active_executions == 2

    def test_decrement_executions(self) -> None:
        """decrement_executions should decrease the count."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)
        version.active_executions = 3

        version.decrement_executions()
        assert version.active_executions == 2
        version.decrement_executions()
        assert version.active_executions == 1

    def test_decrement_executions_does_not_go_negative(self) -> None:
        """decrement_executions should not go below zero."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)
        assert version.active_executions == 0

        version.decrement_executions()
        assert version.active_executions == 0

    def test_has_active_executions_returns_true_when_active(self) -> None:
        """has_active_executions should return True when there are active executions."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        assert version.has_active_executions is False
        version.increment_executions()
        assert version.has_active_executions is True

    def test_has_active_executions_returns_false_when_none(self) -> None:
        """has_active_executions should return False when count is zero."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        assert version.has_active_executions is False


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
        assert orch.agent.tools == ["jira"]
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
      repo: "org/repo-name"
      query_filter: "is:issue is:open label:needs-triage"
      tags:
        - "auto-process"
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
        assert orch.trigger.repo == "org/repo-name"
        assert orch.trigger.query_filter == "is:issue is:open label:needs-triage"
        assert orch.trigger.tags == ["auto-process"]

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
    def test_invalid_format_structure(
        self, repo_format: str, expected_error: str
    ) -> None:
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
    def test_invalid_repo_prefix_suffix(
        self, repo_format: str, expected_error: str
    ) -> None:
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
        """Should raise error for invalid GitHub repo format."""
        yaml_content = """
orchestrations:
  - name: "invalid-repo"
    trigger:
      source: github
      repo: "invalid-format"
      tags: ["test"]
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
      repo: "org/repo/extra"
      tags: ["test"]
    agent:
      prompt: "Test"
"""
        file_path = tmp_path / "too_many_slashes.yaml"
        file_path.write_text(yaml_content)

        with pytest.raises(OrchestrationError, match="Invalid GitHub repo format"):
            load_orchestration_file(file_path)

    def test_empty_repo_is_allowed(self, tmp_path: Path) -> None:
        """Empty repo should be allowed (triggers all repos if combined with tags)."""
        yaml_content = """
orchestrations:
  - name: "empty-repo"
    trigger:
      source: github
      tags: ["needs-review"]
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
