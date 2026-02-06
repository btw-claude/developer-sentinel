"""Tests for orchestration configuration parsing and loading.

This module contains tests for parsing orchestration files and loading orchestration
configurations from YAML files. It tests dataclasses, orchestration versions, file loading,
and directory loading functionality.
"""

import threading
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
    _validate_string_field,
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
        # GitHub Project-based fields
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

    def test_agent_config_defaults(self) -> None:
        """AgentConfig should have sensible defaults."""
        agent = AgentConfig()
        assert agent.prompt == ""
        assert agent.github is None
        assert agent.timeout_seconds is None
        assert agent.agent_type is None
        assert agent.cursor_mode is None
        assert agent.strict_template_variables is False

    def test_agent_config_strict_template_variables_true(self) -> None:
        """AgentConfig should support strict_template_variables=True."""
        agent = AgentConfig(strict_template_variables=True)
        assert agent.strict_template_variables is True

    def test_agent_config_strict_template_variables_false(self) -> None:
        """AgentConfig should support strict_template_variables=False."""
        agent = AgentConfig(strict_template_variables=False)
        assert agent.strict_template_variables is False

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

    def test_concurrent_increment_executions(self) -> None:
        """increment_executions should be thread-safe under concurrent access."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        num_threads = 100
        increments_per_thread = 100

        def increment_many_times() -> None:
            for _ in range(increments_per_thread):
                version.increment_executions()

        threads = [threading.Thread(target=increment_many_times) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Without thread safety, the count would likely be less than expected
        # due to race conditions (lost updates)
        expected_count = num_threads * increments_per_thread
        assert version.active_executions == expected_count

    def test_concurrent_decrement_executions(self) -> None:
        """decrement_executions should be thread-safe under concurrent access."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        num_threads = 100
        decrements_per_thread = 100
        initial_count = num_threads * decrements_per_thread
        version.active_executions = initial_count

        def decrement_many_times() -> None:
            for _ in range(decrements_per_thread):
                version.decrement_executions()

        threads = [threading.Thread(target=decrement_many_times) for _ in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should reach exactly zero without race conditions
        assert version.active_executions == 0

    def test_concurrent_mixed_increment_decrement(self) -> None:
        """Mixed increment/decrement operations should be thread-safe."""
        orch = Orchestration(
            name="test",
            trigger=TriggerConfig(),
            agent=AgentConfig(),
        )
        version = OrchestrationVersion.create(orch, Path("/tmp/test.yaml"), 1234567890.0)

        num_threads = 50
        operations_per_thread = 100
        # Start with enough headroom to avoid hitting zero mid-operation
        version.active_executions = num_threads * operations_per_thread

        increment_count = 0
        decrement_count = 0
        count_lock = threading.Lock()

        def increment_task() -> None:
            nonlocal increment_count
            for _ in range(operations_per_thread):
                version.increment_executions()
            with count_lock:
                increment_count += operations_per_thread

        def decrement_task() -> None:
            nonlocal decrement_count
            for _ in range(operations_per_thread):
                version.decrement_executions()
            with count_lock:
                decrement_count += operations_per_thread

        # Create equal numbers of increment and decrement threads
        threads = []
        for _ in range(num_threads // 2):
            threads.append(threading.Thread(target=increment_task))
            threads.append(threading.Thread(target=decrement_task))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final count should equal: initial + increments - decrements
        expected_count = (
            num_threads * operations_per_thread + increment_count - decrement_count
        )
        assert version.active_executions == expected_count


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

    def test_create_branch_true_without_branch_pattern_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that create_branch=True without a branch pattern raises error."""
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
        create_branch: true
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "create_branch is True but no branch pattern is specified" in str(exc_info.value)

    def test_create_branch_true_with_empty_branch_raises_error(self, tmp_path: Path) -> None:
        """Test that create_branch=True with explicit empty branch raises error."""
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
        branch: ""
        create_branch: true
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "create_branch is True but no branch pattern is specified" in str(exc_info.value)

    def test_create_branch_true_with_whitespace_only_branch_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that create_branch=True with whitespace-only branch raises error."""
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
        branch: "   "
        create_branch: true
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "create_branch is True but no branch pattern is specified" in str(exc_info.value)

    def test_whitespace_only_branch_without_create_branch_skips_validation(
        self, tmp_path: Path
    ) -> None:
        """Test that whitespace-only branch skips validation when create_branch is not set."""
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
        branch: "   "
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.branch == "   "

    def test_whitespace_only_base_branch_skips_validation(
        self, tmp_path: Path
    ) -> None:
        """Test that whitespace-only base_branch skips validation."""
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
        base_branch: "   "
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.base_branch == "   "

    def test_create_branch_false_without_branch_pattern_succeeds(self, tmp_path: Path) -> None:
        """Test that create_branch=False without a branch pattern is valid."""
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
        create_branch: false
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.create_branch is False
        assert orchestrations[0].agent.github.branch == ""

    def test_no_create_branch_without_branch_pattern_uses_defaults(self, tmp_path: Path) -> None:
        """Test that omitting create_branch and branch uses defaults without error."""
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
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.create_branch is False
        assert orchestrations[0].agent.github.branch == ""
        assert orchestrations[0].agent.github.base_branch == "main"

    def test_null_branch_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """Test that explicit null branch is coalesced to empty string without error.

        When YAML sets branch to null, data.get() returns None. The None-coalescing
        logic (using 'or') should replace None with the default empty string so that
        subsequent .strip() calls don't raise AttributeError.
        """
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
        branch: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.branch == ""

    def test_null_base_branch_defaults_to_main(self, tmp_path: Path) -> None:
        """Test that explicit null base_branch is coalesced to 'main' without error.

        When YAML sets base_branch to null, data.get() returns None. The None-coalescing
        logic (using 'or') should replace None with the default 'main' so that
        subsequent .strip() calls don't raise AttributeError.
        """
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
        base_branch: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.base_branch == "main"

    def test_null_branch_and_base_branch_together(self, tmp_path: Path) -> None:
        """Test that both branch and base_branch set to null are handled correctly."""
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
        branch: null
        base_branch: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.branch == ""
        assert orchestrations[0].agent.github.base_branch == "main"

    def test_null_host_defaults_to_github_com(self, tmp_path: Path) -> None:
        """Test that explicit null host is coalesced to 'github.com' without error.

        When YAML sets host to null, data.get() returns None. The None-coalescing
        logic (using 'or') should replace None with the default 'github.com' so that
        downstream consumers don't receive None unexpectedly.
        """
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
        host: null
        org: myorg
        repo: myrepo
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.host == "github.com"

    def test_null_org_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """Test that explicit null org is coalesced to empty string without error.

        When YAML sets org to null, data.get() returns None. The None-coalescing
        logic (using 'or') should replace None with the default empty string so that
        downstream consumers don't receive None unexpectedly.
        """
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
        org: null
        repo: myrepo
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.org == ""

    def test_null_repo_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """Test that explicit null repo is coalesced to empty string without error.

        When YAML sets repo to null, data.get() returns None. The None-coalescing
        logic (using 'or') should replace None with the default empty string so that
        downstream consumers don't receive None unexpectedly.
        """
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
        repo: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.repo == ""

    def test_null_host_org_repo_together(self, tmp_path: Path) -> None:
        """Test that host, org, and repo all set to null are handled correctly."""
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
        host: null
        org: null
        repo: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.host == "github.com"
        assert orchestrations[0].agent.github.org == ""
        assert orchestrations[0].agent.github.repo == ""

    def test_empty_string_host_raises_error(self, tmp_path: Path) -> None:
        """Test that explicit empty-string host raises OrchestrationError.

        An empty string host would be invalid for downstream GitHub API calls.
        The validation added in DS-631 ensures this is caught at parse time
        rather than causing cryptic failures later.
        """
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
        host: ""
        org: myorg
        repo: myrepo
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.host" in str(exc_info.value)
        assert "empty string" in str(exc_info.value)

    def test_empty_string_org_preserved(self, tmp_path: Path) -> None:
        """Test that explicit empty-string org is preserved, not replaced with default.

        With explicit None checks, an empty string is a distinct value from None
        and should be kept as-is. This is a no-op since the default is already
        empty string, but it validates the None-check pattern.
        """
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
        org: ""
        repo: myrepo
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.org == ""

    def test_empty_string_repo_preserved(self, tmp_path: Path) -> None:
        """Test that explicit empty-string repo is preserved, not replaced with default.

        With explicit None checks, an empty string is a distinct value from None
        and should be kept as-is. This is a no-op since the default is already
        empty string, but it validates the None-check pattern.
        """
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
        repo: ""
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.repo == ""

    def test_empty_string_host_org_repo_together_raises_error(self, tmp_path: Path) -> None:
        """Test that empty-string host raises error even when org and repo are also empty.

        The validation catches the invalid empty-string host before considering
        org and repo, ensuring early failure for misconfigured host values.
        """
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
        host: ""
        org: ""
        repo: ""
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.host" in str(exc_info.value)
        assert "empty string" in str(exc_info.value)


    def test_whitespace_only_host_raises_error(self, tmp_path: Path) -> None:
        """Test that whitespace-only host raises OrchestrationError.

        A host consisting only of whitespace is functionally equivalent to an
        empty string and would be invalid for downstream GitHub API calls.
        """
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
        host: "   "
        org: myorg
        repo: myrepo
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.host" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_valid_enterprise_host_accepted(self, tmp_path: Path) -> None:
        """Test that a valid GitHub Enterprise host is accepted without error."""
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
        host: "github.enterprise.com"
        org: myorg
        repo: myrepo
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.host == "github.enterprise.com"

    def test_non_string_host_raises_error(self, tmp_path: Path) -> None:
        """Test that non-string host raises OrchestrationError.

        A non-string host (e.g., integer) indicates a misconfiguration in the
        YAML file. Added in DS-633 for consistent type validation across
        host, org, and repo fields.
        """
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
        host: 12345
        org: myorg
        repo: myrepo
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.host" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_non_string_org_raises_error(self, tmp_path: Path) -> None:
        """Test that non-string org raises OrchestrationError.

        A non-string org (e.g., integer) indicates a misconfiguration in the
        YAML file. Added in DS-633 for consistent type validation.
        """
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
        org: 12345
        repo: myrepo
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.org" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_non_string_repo_raises_error(self, tmp_path: Path) -> None:
        """Test that non-string repo raises OrchestrationError.

        A non-string repo (e.g., integer) indicates a misconfiguration in the
        YAML file. Added in DS-633 for consistent type validation.
        """
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
        repo: 12345
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.repo" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_whitespace_only_org_raises_error(self, tmp_path: Path) -> None:
        """Test that whitespace-only org raises OrchestrationError.

        An org consisting only of whitespace would be invalid for downstream
        GitHub API calls. Added in DS-633 for consistent validation.
        """
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
        org: "   "
        repo: myrepo
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.org" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_whitespace_only_repo_raises_error(self, tmp_path: Path) -> None:
        """Test that whitespace-only repo raises OrchestrationError.

        A repo consisting only of whitespace would be invalid for downstream
        GitHub API calls. Added in DS-633 for consistent validation.
        """
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
        repo: "   "
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "Invalid github.repo" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_valid_org_and_repo_accepted(self, tmp_path: Path) -> None:
        """Test that valid string org and repo are accepted without error.

        Ensures DS-633 validation does not reject legitimate org/repo values.
        """
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
        org: my-org
        repo: my-repo
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.org == "my-org"
        assert orchestrations[0].agent.github.repo == "my-repo"

    def test_null_org_and_repo_default_to_empty_string(self, tmp_path: Path) -> None:
        """Test that null org and repo default to empty string.

        Ensures DS-633 validation does not break the existing default behavior
        where None falls back to empty string.
        """
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
        org: null
        repo: null
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.org == ""
        assert orchestrations[0].agent.github.repo == ""


class TestStrictTemplateVariablesConfig:
    """Tests for strict_template_variables configuration option.

    The strict_template_variables field allows teams to opt-in to stricter
    validation during development/testing to catch typos in template variables.
    """

    def test_strict_template_variables_defaults_to_false(self, tmp_path: Path) -> None:
        """Test that strict_template_variables defaults to False."""
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
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.strict_template_variables is False

    def test_strict_template_variables_true(self, tmp_path: Path) -> None:
        """Test that strict_template_variables can be set to True."""
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
      strict_template_variables: true
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.strict_template_variables is True

    def test_strict_template_variables_false_explicit(self, tmp_path: Path) -> None:
        """Test that strict_template_variables can be explicitly set to False."""
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
      strict_template_variables: false
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.strict_template_variables is False

    def test_strict_template_variables_invalid_type_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid strict_template_variables value raises error."""
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
      strict_template_variables: "yes"
""")
        with pytest.raises(OrchestrationError) as exc_info:
            load_orchestration_file(orch_file)
        assert "strict_template_variables" in str(exc_info.value)
        assert "must be a boolean" in str(exc_info.value)

    def test_strict_template_variables_with_github_context(self, tmp_path: Path) -> None:
        """Test that strict_template_variables works with GitHub context."""
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
      prompt: "Test {jira_issue_key}"
      tools: []
      strict_template_variables: true
      github:
        host: github.com
        org: myorg
        repo: myrepo
        branch: "feature/{jira_issue_key}"
""")
        orchestrations = load_orchestration_file(orch_file)
        assert len(orchestrations) == 1
        assert orchestrations[0].agent.strict_template_variables is True
        assert orchestrations[0].agent.github is not None
        assert orchestrations[0].agent.github.branch == "feature/{jira_issue_key}"


class TestValidateStringField:
    """Tests for _validate_string_field() helper function.

    The _validate_string_field() helper was extracted in DS-634 to eliminate
    the repeated isinstance + whitespace-only validation pattern that was
    duplicated across host, org, and repo fields in _parse_github_context().
    """

    def test_none_value_is_accepted(self) -> None:
        """None should be accepted without error (callers fall back to a default)."""
        _validate_string_field(None, "test.field")  # Should not raise

    def test_valid_string_is_accepted(self) -> None:
        """A normal non-empty string should be accepted without error."""
        _validate_string_field("github.com", "test.field")  # Should not raise

    def test_non_string_raises_error(self) -> None:
        """A non-string value (e.g., integer) should raise OrchestrationError."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field(12345, "github.host")
        assert "Invalid github.host" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_non_string_boolean_raises_error(self) -> None:
        """A boolean value should raise OrchestrationError."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field(True, "github.org")
        assert "Invalid github.org" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_non_string_list_raises_error(self) -> None:
        """A list value should raise OrchestrationError."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field(["a", "b"], "github.repo")
        assert "Invalid github.repo" in str(exc_info.value)
        assert "must be a string" in str(exc_info.value)

    def test_empty_string_rejected_by_default(self) -> None:
        """An empty string should be rejected when reject_empty is True (default)."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field("", "github.host")
        assert "Invalid github.host" in str(exc_info.value)
        assert "empty string" in str(exc_info.value)

    def test_empty_string_accepted_when_reject_empty_false(self) -> None:
        """An empty string should be accepted when reject_empty is False."""
        _validate_string_field("", "github.org", reject_empty=False)  # Should not raise

    def test_whitespace_only_rejected(self) -> None:
        """A whitespace-only string should be rejected."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field("   ", "github.host")
        assert "Invalid github.host" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_whitespace_only_rejected_even_with_reject_empty_false(self) -> None:
        """A whitespace-only string should be rejected even when reject_empty is False."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field("   ", "github.org", reject_empty=False)
        assert "Invalid github.org" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_tab_whitespace_rejected(self) -> None:
        """A tab-only string should be rejected as whitespace-only."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field("\t", "github.repo")
        assert "Invalid github.repo" in str(exc_info.value)
        assert "whitespace-only string" in str(exc_info.value)

    def test_field_name_appears_in_error_message(self) -> None:
        """The field_name parameter should appear in the error message."""
        with pytest.raises(OrchestrationError) as exc_info:
            _validate_string_field(42, "custom.field")
        assert "Invalid custom.field" in str(exc_info.value)

    def test_string_with_leading_trailing_whitespace_accepted(self) -> None:
        """A string with content and surrounding whitespace should be accepted."""
        _validate_string_field("  github.com  ", "github.host")  # Should not raise

    def test_single_character_string_accepted(self) -> None:
        """A single non-whitespace character should be accepted."""
        _validate_string_field("x", "github.org")  # Should not raise
