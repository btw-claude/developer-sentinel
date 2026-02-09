"""Tests for poll_coordinator exception handler ordering and error_count propagation.

These tests verify that:
1. JiraClientError, GitHubClientError, and CircuitBreakerError are caught
   before generic RuntimeError and increment error_count correctly.
2. The configurable error threshold in main.py produces warnings at the
   expected thresholds.

Related: DS-828, DS-820, PR #825
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from sentinel.circuit_breaker import CircuitBreakerError, CircuitState
from sentinel.config import PollingConfig
from sentinel.github_poller import GitHubClientError
from sentinel.main import Sentinel
from sentinel.poll_coordinator import PollCoordinator
from sentinel.poller import JiraClientError
from tests.helpers import make_config, make_issue, make_orchestration
from tests.mocks import (
    MockAgentClient,
    MockAgentClientFactory,
    MockGitHubPoller,
    MockJiraPoller,
    MockTagClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_poll_coordinator(
    jira_poller: MockJiraPoller | None = None,
    github_poller: MockGitHubPoller | None = None,
    error_threshold_pct: float = 1.0,
) -> PollCoordinator:
    """Create a PollCoordinator with the given pollers."""
    config = make_config(polling_error_threshold_pct=error_threshold_pct)
    return PollCoordinator(
        config=config,
        jira_poller=jira_poller,
        github_poller=github_poller,
    )


def _make_sentinel(
    jira_poller: MockJiraPoller | None = None,
    github_poller: MockGitHubPoller | None = None,
    error_threshold_pct: float = 1.0,
    orchestrations: list[Any] | None = None,
) -> Sentinel:
    """Create a Sentinel with controllable pollers."""
    tag_client = MockTagClient()
    if jira_poller is None:
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
    agent_client = MockAgentClient()
    agent_factory = MockAgentClientFactory(agent_client)
    config = make_config(polling_error_threshold_pct=error_threshold_pct)
    if orchestrations is None:
        orchestrations = [make_orchestration(tags=["review"])]
    return Sentinel(
        config=config,
        orchestrations=orchestrations,
        tag_client=tag_client,
        agent_factory=agent_factory,
        jira_poller=jira_poller,
        github_poller=github_poller,
    )


# =========================================================================
# Part 1: Exception handler ordering and error_count propagation
# =========================================================================


class TestJiraExceptionHandlerOrdering:
    """Verify Jira polling exception handlers catch errors correctly."""

    def test_jira_client_error_increments_error_count(self) -> None:
        """JiraClientError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(side_effect=JiraClientError("connection refused"))

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_circuit_breaker_error_increments_error_count(self) -> None:
        """CircuitBreakerError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(
            side_effect=CircuitBreakerError("jira", CircuitState.OPEN, "jira circuit open")
        )

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_runtime_error_increments_error_count(self) -> None:
        """RuntimeError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(
            side_effect=RuntimeError("unexpected runtime error")
        )

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_os_error_increments_error_count(self) -> None:
        """OSError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(side_effect=OSError("network unreachable"))

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_key_error_increments_error_count(self) -> None:
        """KeyError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(side_effect=KeyError("missing_field"))

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_value_error_increments_error_count(self) -> None:
        """ValueError should be caught and increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(side_effect=ValueError("bad value"))

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [make_orchestration(name="orch1", tags=["review"])]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_multiple_triggers_accumulate_errors(self) -> None:
        """Each failing trigger should independently increment error_count."""
        jira_poller = MockJiraPoller(issues=[])
        # Each call raises a different exception type
        jira_poller.poll = MagicMock(
            side_effect=[
                JiraClientError("first"),
                CircuitBreakerError("service", CircuitState.OPEN, "second"),
                RuntimeError("third"),
            ]
        )

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        # Three orchestrations with distinct triggers to avoid dedup
        orchestrations = [
            make_orchestration(name="orch1", project="PROJ1", tags=["a"]),
            make_orchestration(name="orch2", project="PROJ2", tags=["b"]),
            make_orchestration(name="orch3", project="PROJ3", tags=["c"]),
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 3
        assert found == 0

    def test_mixed_success_and_failure(self) -> None:
        """Successful triggers should count found, failed triggers count errors."""
        jira_poller = MockJiraPoller(issues=[])

        issue = make_issue(key="TEST-1", labels=["a"])
        jira_poller.poll = MagicMock(
            side_effect=[
                [issue],  # First trigger succeeds
                JiraClientError("second fails"),  # Second trigger fails
            ]
        )

        coordinator = _make_poll_coordinator(jira_poller=jira_poller)
        orchestrations = [
            make_orchestration(name="orch1", project="PROJ1", tags=["a"]),
            make_orchestration(name="orch2", project="PROJ2", tags=["b"]),
        ]
        router = MagicMock()
        router.route_matched_only = MagicMock(return_value=[])

        results, found, errors = coordinator.poll_jira_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 1


class TestGitHubExceptionHandlerOrdering:
    """Verify GitHub polling exception handlers catch errors correctly."""

    def test_github_client_error_increments_error_count(self) -> None:
        """GitHubClientError should be caught and increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=GitHubClientError("API rate limited")
        )

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch", source="github", project_number=1
            )
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_circuit_breaker_error_increments_error_count(self) -> None:
        """CircuitBreakerError should be caught and increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=CircuitBreakerError("github", CircuitState.OPEN, "github circuit open")
        )

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch", source="github", project_number=1
            )
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_runtime_error_increments_error_count(self) -> None:
        """RuntimeError should be caught and increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=RuntimeError("unexpected error")
        )

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch", source="github", project_number=1
            )
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_os_error_increments_error_count(self) -> None:
        """OSError should be caught and increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(side_effect=OSError("network error"))

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch", source="github", project_number=1
            )
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 1
        assert found == 0
        assert results == []

    def test_key_value_errors_increment_error_count(self) -> None:
        """KeyError and ValueError should be caught and increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=[
                KeyError("missing_field"),
                ValueError("bad value"),
            ]
        )

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch1",
                source="github",
                project_number=1,
                project_owner="org1",
            ),
            make_orchestration(
                name="gh-orch2",
                source="github",
                project_number=2,
                project_owner="org2",
            ),
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 2
        assert found == 0

    def test_multiple_github_triggers_accumulate_errors(self) -> None:
        """Each failing GitHub trigger should independently increment error_count."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=[
                GitHubClientError("first"),
                CircuitBreakerError("service", CircuitState.OPEN, "second"),
                RuntimeError("third"),
            ]
        )

        coordinator = _make_poll_coordinator(github_poller=github_poller)
        orchestrations = [
            make_orchestration(
                name="gh-orch1",
                source="github",
                project_number=1,
                project_owner="org1",
            ),
            make_orchestration(
                name="gh-orch2",
                source="github",
                project_number=2,
                project_owner="org2",
            ),
            make_orchestration(
                name="gh-orch3",
                source="github",
                project_number=3,
                project_owner="org3",
            ),
        ]
        router = MagicMock()

        results, found, errors = coordinator.poll_github_triggers(
            orchestrations, router
        )

        assert errors == 3
        assert found == 0


# =========================================================================
# Part 2: Configurable error threshold for polling warnings
# =========================================================================


class TestConfigurableErrorThreshold:
    """Verify the configurable error_threshold_pct works as expected."""

    def test_default_threshold_is_one(self) -> None:
        """Default PollingConfig.error_threshold_pct should be 1.0."""
        config = PollingConfig()
        assert config.error_threshold_pct == 1.0

    def test_custom_threshold_via_make_config(self) -> None:
        """make_config should propagate polling_error_threshold_pct."""
        config = make_config(polling_error_threshold_pct=0.5)
        assert config.polling.error_threshold_pct == 0.5

    def test_threshold_env_var_parsing(self) -> None:
        """SENTINEL_POLLING_ERROR_THRESHOLD_PCT should be parsed from env."""
        from sentinel.config import load_config

        with patch.dict(
            "os.environ",
            {"SENTINEL_POLLING_ERROR_THRESHOLD_PCT": "0.5"},
            clear=False,
        ):
            config = load_config()
        assert config.polling.error_threshold_pct == 0.5

    def test_threshold_env_var_invalid_value_uses_default(self) -> None:
        """Invalid threshold value falls back to the default (1.0)."""
        from sentinel.config import load_config

        with patch.dict(
            "os.environ",
            {"SENTINEL_POLLING_ERROR_THRESHOLD_PCT": "not-a-number"},
            clear=False,
        ):
            config = load_config()
        assert config.polling.error_threshold_pct == 1.0

    def test_threshold_env_var_out_of_range_uses_default(self) -> None:
        """Out-of-range threshold value falls back to the default (1.0)."""
        from sentinel.config import load_config

        with patch.dict(
            "os.environ",
            {"SENTINEL_POLLING_ERROR_THRESHOLD_PCT": "1.5"},
            clear=False,
        ):
            config = load_config()
        assert config.polling.error_threshold_pct == 1.0


class TestThresholdWarningBehaviourJira:
    """Verify Jira threshold-based warnings in Sentinel.run_once."""

    def test_all_triggers_fail_warns_with_default_threshold(self) -> None:
        """When all Jira triggers fail and threshold=1.0, emit a warning."""
        jira_poller = MockJiraPoller(issues=[])
        jira_poller.poll = MagicMock(side_effect=JiraClientError("fail"))

        sentinel = _make_sentinel(
            jira_poller=jira_poller, error_threshold_pct=1.0
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        # The "All ... trigger(s) failed" warning should fire with "Jira" in args
        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any(
            "trigger(s) failed" in msg and "Jira" in msg
            for msg in warning_messages
        )

    def test_partial_failure_no_warn_with_default_threshold(self) -> None:
        """When some triggers fail but not all, no threshold warning at default 1.0."""
        jira_poller = MockJiraPoller(issues=[])
        issue = make_issue(key="TEST-1", labels=["a"])
        # First trigger succeeds, second fails
        jira_poller.poll = MagicMock(
            side_effect=[
                [issue],
                JiraClientError("fail"),
            ]
        )

        orchestrations = [
            make_orchestration(name="orch1", project="PROJ1", tags=["a"]),
            make_orchestration(name="orch2", project="PROJ2", tags=["b"]),
        ]
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            error_threshold_pct=1.0,
            orchestrations=orchestrations,
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        # The threshold warning should NOT fire (only 50% failed, threshold is 100%)
        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert not any(
            "Jira trigger(s) failed during this polling cycle" in msg
            for msg in warning_messages
        )

    def test_partial_failure_warns_with_lower_threshold(self) -> None:
        """When 50% triggers fail and threshold=0.5, emit a warning."""
        jira_poller = MockJiraPoller(issues=[])
        issue = make_issue(key="TEST-1", labels=["a"])
        # First trigger succeeds, second fails
        jira_poller.poll = MagicMock(
            side_effect=[
                [issue],
                JiraClientError("fail"),
            ]
        )

        orchestrations = [
            make_orchestration(name="orch1", project="PROJ1", tags=["a"]),
            make_orchestration(name="orch2", project="PROJ2", tags=["b"]),
        ]
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            error_threshold_pct=0.5,
            orchestrations=orchestrations,
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        # The threshold warning SHOULD fire (50% failed, threshold is 50%)
        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any(
            "trigger(s) failed" in msg and "Jira" in msg
            for msg in warning_messages
        )

    def test_below_threshold_no_warning(self) -> None:
        """When failure rate is below threshold, no warning should fire."""
        jira_poller = MockJiraPoller(issues=[])
        issue = make_issue(key="TEST-1", labels=["a"])
        # First two succeed, third fails — 33% failure
        jira_poller.poll = MagicMock(
            side_effect=[
                [issue],
                [issue],
                JiraClientError("fail"),
            ]
        )

        orchestrations = [
            make_orchestration(name="orch1", project="PROJ1", tags=["a"]),
            make_orchestration(name="orch2", project="PROJ2", tags=["b"]),
            make_orchestration(name="orch3", project="PROJ3", tags=["c"]),
        ]
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            error_threshold_pct=0.5,
            orchestrations=orchestrations,
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        # No threshold warning should fire (33% < 50%)
        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert not any(
            "Jira trigger(s) failed during this polling cycle" in msg
            for msg in warning_messages
        )


class TestThresholdWarningBehaviourGitHub:
    """Verify GitHub threshold-based warnings in Sentinel.run_once."""

    def test_all_github_triggers_fail_warns(self) -> None:
        """When all GitHub triggers fail, emit a warning."""
        github_poller = MockGitHubPoller(issues=[])
        github_poller.poll = MagicMock(
            side_effect=GitHubClientError("API error")
        )
        jira_poller = MockJiraPoller(issues=[])

        orchestrations = [
            make_orchestration(
                name="gh-orch", source="github", project_number=1
            )
        ]
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            github_poller=github_poller,
            error_threshold_pct=1.0,
            orchestrations=orchestrations,
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any(
            "trigger(s) failed" in msg and "GitHub" in msg
            for msg in warning_messages
        )

    def test_partial_github_failure_warns_at_lower_threshold(self) -> None:
        """When 50% GitHub triggers fail and threshold=0.5, emit a warning."""
        from sentinel.github_poller import GitHubIssue

        github_poller = MockGitHubPoller(issues=[])
        gh_issue = GitHubIssue(
            number=1,
            title="Test",
            body="body",
            state="open",
            author="user",
            assignees=[],
            labels=["review"],
            is_pull_request=False,
            head_ref="",
            base_ref="",
            draft=False,
            repo_url="https://github.com/org/repo/issues/1",
            parent_issue_number=None,
        )
        github_poller.poll = MagicMock(
            side_effect=[
                [gh_issue],  # First trigger succeeds
                GitHubClientError("fail"),  # Second fails
            ]
        )
        jira_poller = MockJiraPoller(issues=[])

        orchestrations = [
            make_orchestration(
                name="gh-orch1",
                source="github",
                project_number=1,
                project_owner="org1",
            ),
            make_orchestration(
                name="gh-orch2",
                source="github",
                project_number=2,
                project_owner="org2",
            ),
        ]
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            github_poller=github_poller,
            error_threshold_pct=0.5,
            orchestrations=orchestrations,
        )

        with patch("sentinel.main.logger") as mock_logger:
            sentinel.run_once()

        warning_messages = [
            str(call) for call in mock_logger.warning.call_args_list
        ]
        assert any(
            "trigger(s) failed" in msg and "GitHub" in msg
            for msg in warning_messages
        )
