"""Tests for orchestration configuration loading."""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    OnCompleteConfig,
    OnFailureConfig,
    OrchestrationError,
    RetryConfig,
    TriggerConfig,
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

    def test_on_complete_config_defaults(self) -> None:
        """OnCompleteConfig should have sensible defaults."""
        on_complete = OnCompleteConfig()
        assert on_complete.remove_tag == ""
        assert on_complete.add_tag == ""

    def test_on_failure_config_defaults(self) -> None:
        """OnFailureConfig should have sensible defaults."""
        on_failure = OnFailureConfig()
        assert on_failure.add_tag == ""


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
