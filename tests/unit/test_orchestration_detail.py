"""Tests for orchestration detail DTOs and get_orchestration_detail accessor.

Tests the conversion logic from Orchestration to OrchestrationDetailInfo,
including all sub-DTOs (TriggerDetailInfo, GitHubContextInfo, AgentDetailInfo,
RetryDetailInfo, OutcomeInfo, LifecycleInfo).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sentinel.config import Config, ExecutionConfig
from sentinel.dashboard.state import (
    AgentDetailInfo,
    ExecutionStateSnapshot,
    GitHubContextInfo,
    LifecycleInfo,
    OrchestrationDetailInfo,
    OrchestrationVersionSnapshot,
    OutcomeInfo,
    RetryDetailInfo,
    SentinelStateAccessor,
    TriggerDetailInfo,
)
from sentinel.orchestration import (
    AgentConfig,
    GitHubContext,
    OnCompleteConfig,
    OnFailureConfig,
    OnStartConfig,
    Orchestration,
    Outcome,
    RetryConfig,
    TriggerConfig,
)


class MockSentinelForDetail:
    """Mock Sentinel instance for testing get_orchestration_detail.

    Implements the SentinelStateProvider protocol with controllable
    orchestrations and active version snapshots.
    """

    def __init__(
        self,
        config: Config,
        orchestrations: list[Orchestration] | None = None,
        active_versions: list[OrchestrationVersionSnapshot] | None = None,
    ) -> None:
        self.config = config
        self._orchestrations = orchestrations or []
        self._active_versions = active_versions or []
        self._start_time = datetime.now()

    @property
    def orchestrations(self) -> list[Orchestration]:
        """Return the list of active orchestrations."""
        return self._orchestrations

    def get_hot_reload_metrics(self) -> dict[str, int]:
        """Return hot-reload metrics."""
        return {
            "orchestrations_loaded_total": 0,
            "orchestrations_unloaded_total": 0,
            "orchestrations_reloaded_total": 0,
        }

    def get_running_steps(self) -> list[Any]:
        """Return empty running steps."""
        return []

    def get_issue_queue(self) -> list[Any]:
        """Return empty issue queue."""
        return []

    def get_start_time(self) -> datetime:
        """Return the process start time."""
        return self._start_time

    def get_last_jira_poll(self) -> datetime | None:
        """Return None for last Jira poll."""
        return None

    def get_last_github_poll(self) -> datetime | None:
        """Return None for last GitHub poll."""
        return None

    def get_active_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return active version snapshots."""
        return self._active_versions

    def get_pending_removal_versions(self) -> list[OrchestrationVersionSnapshot]:
        """Return empty list of pending removal version snapshots."""
        return []

    def get_execution_state(self) -> ExecutionStateSnapshot:
        """Return execution state snapshot with zero active executions."""
        return ExecutionStateSnapshot(active_count=0)

    def is_shutdown_requested(self) -> bool:
        """Return False for shutdown requested."""
        return False

    def get_completed_executions(self) -> list[Any]:
        """Return empty list of completed executions."""
        return []


def _make_config() -> Config:
    """Create a minimal Config for testing."""
    return Config(execution=ExecutionConfig())


def _make_jira_orchestration(
    name: str = "test-orch",
    enabled: bool = True,
    max_concurrent: int | None = None,
    project: str = "TEST",
    jql_filter: str = "",
    tags: list[str] | None = None,
    prompt: str = "Test prompt for orchestration",
    github: GitHubContext | None = None,
    timeout_seconds: int | None = None,
    model: str | None = None,
    agent_type: str | None = None,
    cursor_mode: str | None = None,
    strict_template_variables: bool = False,
    max_attempts: int = 3,
    success_patterns: list[str] | None = None,
    failure_patterns: list[str] | None = None,
    default_status: str = "success",
    default_outcome: str = "",
    outcomes: list[Outcome] | None = None,
    on_start_add_tag: str = "",
    on_complete_remove_tag: str = "",
    on_complete_add_tag: str = "",
    on_failure_add_tag: str = "",
) -> Orchestration:
    """Create a Jira-triggered Orchestration for testing."""
    return Orchestration(
        name=name,
        enabled=enabled,
        max_concurrent=max_concurrent,
        trigger=TriggerConfig(
            source="jira",
            project=project,
            jql_filter=jql_filter,
            tags=tags or [],
        ),
        agent=AgentConfig(
            prompt=prompt,
            github=github,
            timeout_seconds=timeout_seconds,
            model=model,
            agent_type=agent_type,
            cursor_mode=cursor_mode,
            strict_template_variables=strict_template_variables,
        ),
        retry=RetryConfig(
            max_attempts=max_attempts,
            success_patterns=success_patterns or ["SUCCESS", "completed successfully"],
            failure_patterns=failure_patterns or ["FAILURE", "failed", "error"],
            default_status=default_status,
            default_outcome=default_outcome,
        ),
        outcomes=outcomes or [],
        on_start=OnStartConfig(add_tag=on_start_add_tag),
        on_complete=OnCompleteConfig(
            remove_tag=on_complete_remove_tag,
            add_tag=on_complete_add_tag,
        ),
        on_failure=OnFailureConfig(add_tag=on_failure_add_tag),
    )


class TestOrchestrationDetailDTOs:
    """Tests for the frozen dataclass DTOs."""

    def test_trigger_detail_info_is_frozen(self) -> None:
        """Test that TriggerDetailInfo is a frozen dataclass."""
        info = TriggerDetailInfo(
            source="jira",
            project="TEST",
            jql_filter="",
            tags=["tag1"],
            project_number=None,
            project_scope="org",
            project_owner="",
            project_filter="",
            labels=[],
        )
        assert info.source == "jira"
        assert info.project == "TEST"
        assert info.tags == ["tag1"]

    def test_github_context_info_is_frozen(self) -> None:
        """Test that GitHubContextInfo is a frozen dataclass."""
        info = GitHubContextInfo(
            host="github.com",
            org="my-org",
            repo="my-repo",
            branch="feature/{jira_issue_key}",
            create_branch=True,
            base_branch="main",
        )
        assert info.host == "github.com"
        assert info.org == "my-org"
        assert info.repo == "my-repo"
        assert info.create_branch is True

    def test_agent_detail_info_is_frozen(self) -> None:
        """Test that AgentDetailInfo is a frozen dataclass."""
        info = AgentDetailInfo(
            prompt="Full prompt text",
            github=None,
            timeout_seconds=300,
            model="claude-opus-4-5-20251101",
            agent_type="claude",
            cursor_mode=None,
            agent_teams=None,
            strict_template_variables=False,
        )
        assert info.prompt == "Full prompt text"
        assert info.timeout_seconds == 300
        assert info.agent_teams is None

    def test_retry_detail_info_is_frozen(self) -> None:
        """Test that RetryDetailInfo is a frozen dataclass."""
        info = RetryDetailInfo(
            max_attempts=5,
            success_patterns=["SUCCESS"],
            failure_patterns=["FAILURE"],
            default_status="success",
            default_outcome="",
        )
        assert info.max_attempts == 5
        assert info.success_patterns == ["SUCCESS"]

    def test_outcome_info_is_frozen(self) -> None:
        """Test that OutcomeInfo is a frozen dataclass."""
        info = OutcomeInfo(
            name="approved",
            patterns=["APPROVED"],
            add_tag="approved",
        )
        assert info.name == "approved"
        assert info.patterns == ["APPROVED"]
        assert info.add_tag == "approved"

    def test_lifecycle_info_is_frozen(self) -> None:
        """Test that LifecycleInfo is a frozen dataclass."""
        info = LifecycleInfo(
            on_start_add_tag="in-progress",
            on_complete_remove_tag="in-progress",
            on_complete_add_tag="done",
            on_failure_add_tag="failed",
        )
        assert info.on_start_add_tag == "in-progress"
        assert info.on_complete_remove_tag == "in-progress"
        assert info.on_complete_add_tag == "done"
        assert info.on_failure_add_tag == "failed"

    def test_orchestration_detail_info_is_frozen(self) -> None:
        """Test that OrchestrationDetailInfo is a frozen dataclass."""
        trigger = TriggerDetailInfo(
            source="jira",
            project="TEST",
            jql_filter="",
            tags=[],
            project_number=None,
            project_scope="org",
            project_owner="",
            project_filter="",
            labels=[],
        )
        agent = AgentDetailInfo(
            prompt="prompt",
            github=None,
            timeout_seconds=None,
            model=None,
            agent_type=None,
            cursor_mode=None,
            agent_teams=None,
            strict_template_variables=False,
        )
        retry = RetryDetailInfo(
            max_attempts=3,
            success_patterns=[],
            failure_patterns=[],
            default_status="success",
            default_outcome="",
        )
        lifecycle = LifecycleInfo(
            on_start_add_tag="",
            on_complete_remove_tag="",
            on_complete_add_tag="",
            on_failure_add_tag="",
        )
        info = OrchestrationDetailInfo(
            name="test",
            enabled=True,
            max_concurrent=None,
            source_file="test.yaml",
            trigger=trigger,
            agent=agent,
            retry=retry,
            outcomes=[],
            lifecycle=lifecycle,
        )
        assert info.name == "test"
        assert info.enabled is True
        assert info.max_concurrent is None


class TestGetOrchestrationDetail:
    """Tests for SentinelStateAccessor.get_orchestration_detail."""

    def test_returns_none_when_not_found(self) -> None:
        """Test that get_orchestration_detail returns None for unknown name."""
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("nonexistent")
        assert result is None

    def test_returns_detail_for_jira_orchestration(self) -> None:
        """Test basic Jira orchestration conversion."""
        orch = _make_jira_orchestration(
            name="jira-orch",
            project="PROJ",
            tags=["review", "ready"],
            prompt="Analyze the issue and provide feedback",
        )
        version = OrchestrationVersionSnapshot(
            name="jira-orch",
            version_id="v1",
            source_file="orchestrations/jira.yaml",
            loaded_at=datetime.now(),
            active_executions=0,
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(
            config, orchestrations=[orch], active_versions=[version]
        )
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("jira-orch")

        assert result is not None
        assert result.name == "jira-orch"
        assert result.enabled is True
        assert result.source_file == "orchestrations/jira.yaml"
        assert result.trigger.source == "jira"
        assert result.trigger.project == "PROJ"
        assert result.trigger.tags == ["review", "ready"]
        assert result.agent.prompt == "Analyze the issue and provide feedback"
        assert result.agent.github is None
        assert result.agent.agent_teams is None

    def test_returns_full_prompt_not_truncated(self) -> None:
        """Test that get_orchestration_detail returns the full prompt."""
        long_prompt = "A" * 500
        orch = _make_jira_orchestration(name="long-prompt", prompt=long_prompt)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("long-prompt")

        assert result is not None
        assert result.agent.prompt == long_prompt
        assert len(result.agent.prompt) == 500

    def test_converts_github_context(self) -> None:
        """Test that GitHub context is correctly converted."""
        github = GitHubContext(
            host="github.example.com",
            org="my-org",
            repo="my-repo",
            branch="feature/{jira_issue_key}",
            create_branch=True,
            base_branch="develop",
        )
        orch = _make_jira_orchestration(name="gh-orch", github=github)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("gh-orch")

        assert result is not None
        assert result.agent.github is not None
        assert result.agent.github.host == "github.example.com"
        assert result.agent.github.org == "my-org"
        assert result.agent.github.repo == "my-repo"
        assert result.agent.github.branch == "feature/{jira_issue_key}"
        assert result.agent.github.create_branch is True
        assert result.agent.github.base_branch == "develop"

    def test_converts_agent_fields(self) -> None:
        """Test that all agent fields are correctly converted."""
        orch = _make_jira_orchestration(
            name="agent-orch",
            timeout_seconds=600,
            model="claude-sonnet-4-20250514",
            agent_type="claude",
            strict_template_variables=True,
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("agent-orch")

        assert result is not None
        assert result.agent.timeout_seconds == 600
        assert result.agent.model == "claude-sonnet-4-20250514"
        assert result.agent.agent_type == "claude"
        assert result.agent.cursor_mode is None
        assert result.agent.strict_template_variables is True

    def test_converts_retry_config(self) -> None:
        """Test that retry configuration is correctly converted."""
        orch = _make_jira_orchestration(
            name="retry-orch",
            max_attempts=5,
            success_patterns=["PASS", "OK"],
            failure_patterns=["FAIL", "ERR"],
            default_status="failure",
            default_outcome="needs-review",
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("retry-orch")

        assert result is not None
        assert result.retry.max_attempts == 5
        assert result.retry.success_patterns == ["PASS", "OK"]
        assert result.retry.failure_patterns == ["FAIL", "ERR"]
        assert result.retry.default_status == "failure"
        assert result.retry.default_outcome == "needs-review"

    def test_converts_outcomes(self) -> None:
        """Test that outcomes are correctly converted."""
        outcomes = [
            Outcome(name="approved", patterns=["APPROVED", "LGTM"], add_tag="approved"),
            Outcome(name="changes-requested", patterns=["CHANGES"], add_tag="needs-changes"),
        ]
        orch = _make_jira_orchestration(name="outcome-orch", outcomes=outcomes)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("outcome-orch")

        assert result is not None
        assert len(result.outcomes) == 2
        assert result.outcomes[0].name == "approved"
        assert result.outcomes[0].patterns == ["APPROVED", "LGTM"]
        assert result.outcomes[0].add_tag == "approved"
        assert result.outcomes[1].name == "changes-requested"

    def test_converts_lifecycle(self) -> None:
        """Test that lifecycle configuration is correctly converted."""
        orch = _make_jira_orchestration(
            name="lifecycle-orch",
            on_start_add_tag="in-progress",
            on_complete_remove_tag="in-progress",
            on_complete_add_tag="completed",
            on_failure_add_tag="failed",
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("lifecycle-orch")

        assert result is not None
        assert result.lifecycle.on_start_add_tag == "in-progress"
        assert result.lifecycle.on_complete_remove_tag == "in-progress"
        assert result.lifecycle.on_complete_add_tag == "completed"
        assert result.lifecycle.on_failure_add_tag == "failed"

    def test_converts_max_concurrent(self) -> None:
        """Test that max_concurrent is correctly converted."""
        orch = _make_jira_orchestration(name="concurrent-orch", max_concurrent=5)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("concurrent-orch")

        assert result is not None
        assert result.max_concurrent == 5

    def test_converts_disabled_orchestration(self) -> None:
        """Test that disabled orchestration is correctly converted."""
        orch = _make_jira_orchestration(name="disabled-orch", enabled=False)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("disabled-orch")

        assert result is not None
        assert result.enabled is False

    def test_source_file_empty_when_no_active_version(self) -> None:
        """Test that source_file is empty when no active version exists."""
        orch = _make_jira_orchestration(name="no-version")
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch], active_versions=[])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("no-version")

        assert result is not None
        assert result.source_file == ""

    def test_github_trigger_fields(self) -> None:
        """Test that GitHub trigger fields are correctly converted."""
        orch = Orchestration(
            name="github-orch",
            trigger=TriggerConfig(
                source="github",
                project_number=42,
                project_scope="org",
                project_owner="my-org",
                project_filter="status:Todo",
                labels=["bug", "priority-high"],
            ),
            agent=AgentConfig(prompt="Fix the bug"),
            retry=RetryConfig(),
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("github-orch")

        assert result is not None
        assert result.trigger.source == "github"
        assert result.trigger.project_number == 42
        assert result.trigger.project_scope == "org"
        assert result.trigger.project_owner == "my-org"
        assert result.trigger.project_filter == "status:Todo"
        assert result.trigger.labels == ["bug", "priority-high"]

    def test_jql_filter_field(self) -> None:
        """Test that jql_filter is correctly converted."""
        orch = _make_jira_orchestration(
            name="jql-orch",
            jql_filter="priority = High",
        )
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("jql-orch")

        assert result is not None
        assert result.trigger.jql_filter == "priority = High"

    def test_multiple_orchestrations_finds_correct_one(self) -> None:
        """Test that get_orchestration_detail finds the right one among many."""
        orch1 = _make_jira_orchestration(name="first-orch", prompt="First")
        orch2 = _make_jira_orchestration(name="second-orch", prompt="Second")
        orch3 = _make_jira_orchestration(name="third-orch", prompt="Third")
        config = _make_config()
        sentinel = MockSentinelForDetail(
            config, orchestrations=[orch1, orch2, orch3]
        )
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("second-orch")

        assert result is not None
        assert result.name == "second-orch"
        assert result.agent.prompt == "Second"

    def test_empty_outcomes_list(self) -> None:
        """Test that empty outcomes list is handled correctly."""
        orch = _make_jira_orchestration(name="no-outcomes")
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("no-outcomes")

        assert result is not None
        assert result.outcomes == []

    def test_no_github_context_returns_none(self) -> None:
        """Test that missing GitHub context returns None in agent detail."""
        orch = _make_jira_orchestration(name="no-gh", github=None)
        config = _make_config()
        sentinel = MockSentinelForDetail(config, orchestrations=[orch])
        accessor = SentinelStateAccessor(sentinel)  # type: ignore[arg-type]

        result = accessor.get_orchestration_detail("no-gh")

        assert result is not None
        assert result.agent.github is None
