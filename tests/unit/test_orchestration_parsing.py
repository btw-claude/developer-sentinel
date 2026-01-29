"""Tests for orchestration configuration parsing and loading.

This module contains tests for parsing orchestration files and loading orchestration
configurations from YAML files. It tests dataclasses, orchestration versions, file loading,
and directory loading functionality.
"""

from pathlib import Path

import pytest

from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    OnCompleteConfig,
    OnFailureConfig,
    OnStartConfig,
    Orchestration,
    OrchestrationError,
    OrchestrationVersion,
    Outcome,
    RetryConfig,
    TriggerConfig,
    _validate_branch_name,
    load_orchestration_file,
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
        # New GitHub Project-based fields
        assert trigger.project_number is None
        assert trigger.project_scope == "org"
        assert trigger.project_owner == ""
        assert trigger.project_filter == ""

    def test_trigger_config_github_source(self) -> None:
        """TriggerConfig should support github source with project-based config."""
        trigger = TriggerConfig(
            source="github",
            project_number=42,
            project_scope="org",
            project_owner="my-org",
            project_filter="Status = 'In Progress'",
        )
        assert trigger.source == "github"
        assert trigger.project_number == 42
        assert trigger.project_scope == "org"
        assert trigger.project_owner == "my-org"
        assert trigger.project_filter == "Status = 'In Progress'"

    def test_trigger_config_labels_defaults_to_empty_list(self) -> None:
        """TriggerConfig labels should default to empty list."""
        trigger = TriggerConfig()
        assert trigger.labels == []

    def test_trigger_config_labels_field_parsed(self) -> None:
        """TriggerConfig should parse labels field from configuration."""
        trigger = TriggerConfig(
            source="github",
            project_number=42,
            project_owner="my-org",
            labels=["bug", "urgent"],
        )
        assert trigger.labels == ["bug", "urgent"]

    def test_trigger_config_labels_combined_with_project_filter(self) -> None:
        """TriggerConfig should support labels combined with project_filter."""
        trigger = TriggerConfig(
            source="github",
            project_number=42,
            project_owner="my-org",
            project_filter="Status = 'Ready'",
            labels=["needs-triage", "bug"],
        )
        assert trigger.labels == ["needs-triage", "bug"]
        assert trigger.project_filter == "Status = 'Ready'"

    def test_trigger_config_github_user_scope(self) -> None:
        """TriggerConfig should support user-scoped GitHub projects."""
        trigger = TriggerConfig(
            source="github",
            project_number=5,
            project_scope="user",
            project_owner="myusername",
            project_filter="Priority = 'High'",
        )
        assert trigger.source == "github"
        assert trigger.project_number == 5
        assert trigger.project_scope == "user"
        assert trigger.project_owner == "myusername"
        assert trigger.project_filter == "Priority = 'High'"

    def test_trigger_config_github_legacy_fields(self) -> None:
        """TriggerConfig should still support legacy GitHub fields."""
        trigger = TriggerConfig(
            source="github",
            repo="org/repo-name",
            query_filter="is:issue is:open label:bug",
            tags=["needs-triage"],
            project_number=1,
            project_owner="org",
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
        assert agent.agent_type is None
        assert agent.cursor_mode is None

    def test_agent_config_with_agent_type_claude(self) -> None:
        """AgentConfig should support agent_type='claude'."""
        agent = AgentConfig(agent_type="claude")
        assert agent.agent_type == "claude"
        assert agent.cursor_mode is None

    def test_agent_config_with_agent_type_cursor(self) -> None:
        """AgentConfig should support agent_type='cursor' with cursor_mode."""
        agent = AgentConfig(agent_type="cursor", cursor_mode="agent")
        assert agent.agent_type == "cursor"
        assert agent.cursor_mode == "agent"

    def test_agent_config_cursor_mode_plan(self) -> None:
        """AgentConfig should support cursor_mode='plan'."""
        agent = AgentConfig(agent_type="cursor", cursor_mode="plan")
        assert agent.cursor_mode == "plan"

    def test_agent_config_cursor_mode_ask(self) -> None:
        """AgentConfig should support cursor_mode='ask'."""
        agent = AgentConfig(agent_type="cursor", cursor_mode="ask")
        assert agent.cursor_mode == "ask"

    def test_github_context_defaults(self) -> None:
        """GitHubContext should have sensible defaults."""
        github = GitHubContext()
        assert github.host == "github.com"
        assert github.org == ""
        assert github.repo == ""
        assert github.branch == ""
        assert github.create_branch is False
        assert github.base_branch == "main"

    def test_github_context_with_branch_config(self) -> None:
        """GitHubContext should support branch configuration."""
        github = GitHubContext(
            host="github.com",
            org="myorg",
            repo="myrepo",
            branch="feature/{jira_issue_key}",
            create_branch=True,
            base_branch="develop",
        )
        assert github.host == "github.com"
        assert github.org == "myorg"
        assert github.repo == "myrepo"
        assert github.branch == "feature/{jira_issue_key}"
        assert github.create_branch is True
        assert github.base_branch == "develop"

    def test_github_context_empty_branch(self) -> None:
        """GitHubContext with empty branch should work."""
        github = GitHubContext(
            host="github.com",
            org="myorg",
            repo="myrepo",
            branch="",
        )
        assert github.branch == ""
        assert github.create_branch is False  # Default
        assert github.base_branch == "main"  # Default

    def test_github_context_branch_without_create(self) -> None:
        """GitHubContext can have branch pattern without create_branch."""
        github = GitHubContext(
            host="github.com",
            org="myorg",
            repo="myrepo",
            branch="feature/existing-branch",
            create_branch=False,
        )
        assert github.branch == "feature/existing-branch"
        assert github.create_branch is False

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


class TestBranchNameValidation:
    """Tests for branch name validation in orchestration."""

    def test_valid_branch_names(self) -> None:
        """Test that valid branch names pass validation."""
        valid_names = [
            "main",
            "master",
            "develop",
            "feature/test",
            "release-1.0",
            "hotfix_123",
            "feature/DS-123-test",
            "feature/{jira_issue_key}",
            "issue-{github_issue_number}",
        ]
        for name in valid_names:
            result = _validate_branch_name(name)
            assert result.is_valid, f"Expected '{name}' to be valid"

    def test_empty_branch_is_valid(self) -> None:
        """Test that empty branch name is valid (optional field)."""
        result = _validate_branch_name("")
        assert result.is_valid

    def test_invalid_start_with_hyphen(self) -> None:
        """Test that branch starting with hyphen is invalid."""
        result = _validate_branch_name("-feature")
        assert not result.is_valid
        assert "cannot start with '-' or '.'" in result.error_message

    def test_invalid_start_with_period(self) -> None:
        """Test that branch starting with period is invalid."""
        result = _validate_branch_name(".hidden")
        assert not result.is_valid
        assert "cannot start with '-' or '.'" in result.error_message

    def test_invalid_end_with_period(self) -> None:
        """Test that branch ending with period is invalid."""
        result = _validate_branch_name("feature.")
        assert not result.is_valid
        assert "cannot end with '.' or '/'" in result.error_message

    def test_invalid_end_with_slash(self) -> None:
        """Test that branch ending with slash is invalid."""
        result = _validate_branch_name("feature/")
        assert not result.is_valid
        assert "cannot end with '.' or '/'" in result.error_message

    def test_invalid_contains_space(self) -> None:
        """Test that branch containing space is invalid."""
        result = _validate_branch_name("feature test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_contains_tilde(self) -> None:
        """Test that branch containing tilde is invalid."""
        result = _validate_branch_name("feature~test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_contains_caret(self) -> None:
        """Test that branch containing caret is invalid."""
        result = _validate_branch_name("feature^test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_contains_colon(self) -> None:
        """Test that branch containing colon is invalid."""
        result = _validate_branch_name("feature:test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_contains_question_mark(self) -> None:
        """Test that branch containing question mark is invalid."""
        result = _validate_branch_name("feature?test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_contains_asterisk(self) -> None:
        """Test that branch containing asterisk is invalid."""
        result = _validate_branch_name("feature*test")
        assert not result.is_valid
        assert "invalid characters" in result.error_message

    def test_invalid_consecutive_periods(self) -> None:
        """Test that consecutive periods are invalid."""
        result = _validate_branch_name("feature..test")
        assert not result.is_valid
        assert "consecutive periods" in result.error_message

    def test_invalid_consecutive_slashes(self) -> None:
        """Test that consecutive slashes are invalid."""
        result = _validate_branch_name("feature//test")
        assert not result.is_valid
        assert "consecutive" in result.error_message

    def test_valid_template_variable_branch(self) -> None:
        """Test that branch patterns with template variables are valid."""
        result = _validate_branch_name("feature/{jira_issue_key}/{jira_summary_slug}")
        assert result.is_valid

    def test_at_symbol_without_brace_is_valid(self) -> None:
        """Test that @ without following { is valid."""
        result = _validate_branch_name("feature@test")
        assert result.is_valid

    def test_invalid_ends_with_lock(self) -> None:
        """Test that branch name ending with .lock is invalid.

        Git disallows branch names ending in .lock as they conflict
        with git's internal lock files.
        """
        result = _validate_branch_name("feature.lock")
        assert not result.is_valid
        assert ".lock" in result.error_message

    def test_lock_in_middle_is_valid(self) -> None:
        """Test that .lock in the middle of a branch name is valid."""
        result = _validate_branch_name("feature.lock.test")
        assert result.is_valid


class TestGitHubContextBranchValidation:
    """Tests for branch validation in GitHub context parsing."""

    def test_valid_branch_and_base_branch(self) -> None:
        """Test that valid branch names are accepted in GitHubContext."""
        github = GitHubContext(
            host="github.com",
            org="myorg",
            repo="myrepo",
            branch="feature/{jira_issue_key}",
            create_branch=True,
            base_branch="develop",
        )
        assert github.branch == "feature/{jira_issue_key}"
        assert github.base_branch == "develop"

    def test_orchestration_file_with_invalid_branch_raises_error(self, tmp_path: Path) -> None:
        """Test that loading orchestration with invalid branch raises error."""
        orch_file = tmp_path / "test.yaml"
        orch_file.write_text("""
orchestrations:
  - name: test
    trigger:
      source: jira
      project: TEST
      tags:
        - test
    agent:
      prompt: "Test prompt"
      tools: []
      github:
        host: github.com
        org: myorg
        repo: myrepo
        branch: "feature..invalid"
        create_branch: true
        base_branch: main
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid branch pattern" in str(exc_info.value)
        assert "consecutive periods" in str(exc_info.value)

    def test_orchestration_file_with_invalid_base_branch_raises_error(self, tmp_path: Path) -> None:
        """Test that loading orchestration with invalid base_branch raises error."""
        orch_file = tmp_path / "test.yaml"
        orch_file.write_text("""
orchestrations:
  - name: test
    trigger:
      source: jira
      project: TEST
      tags:
        - test
    agent:
      prompt: "Test prompt"
      tools: []
      github:
        host: github.com
        org: myorg
        repo: myrepo
        branch: "feature/test"
        create_branch: true
        base_branch: "-invalid"
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid base_branch" in str(exc_info.value)
        assert "cannot start with '-' or '.'" in str(exc_info.value)

    def test_orchestration_file_with_valid_branch_pattern_succeeds(self, tmp_path: Path) -> None:
        """Test that loading orchestration with valid branch pattern succeeds."""
        orch_file = tmp_path / "test.yaml"
        orch_file.write_text("""
orchestrations:
  - name: test
    trigger:
      source: jira
      project: TEST
      tags:
        - test
    agent:
      prompt: "Test prompt"
      tools: []
      github:
        host: github.com
        org: myorg
        repo: myrepo
        branch: "feature/{jira_issue_key}"
        create_branch: true
        base_branch: develop
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.branch == "feature/{jira_issue_key}"
        assert orchestrations[0].agent.github.base_branch == "develop"
