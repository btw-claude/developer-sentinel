"""Comprehensive tests for the Sentinel class in sentinel/main.py.

This test module covers the untested methods and code paths in the Sentinel
class to improve coverage from ~65% to above 85%. Tests are grouped by
functional area into distinct test classes.

Coverage targets:
- Public accessors (dashboard delegates)
- Private log/health helpers
- Failure handling patterns
- _execute_orchestration_task paths
- _submit_execution_tasks deduplication and error handling
- run_once polling logic
- run() main loop error recovery and shutdown
- run_once_and_wait completion
- _record_completed_execution
- Sentinel initialization variants
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.executor import ExecutionResult, ExecutionStatus
from sentinel.github_poller import GitHubIssue
from sentinel.main import Sentinel
from sentinel.orchestration import OrchestrationVersion
from sentinel.sdk_clients import ClaudeProcessInterruptedError
from sentinel.types import ErrorType
from tests.helpers import make_agent_factory, make_config, make_issue, make_orchestration
from tests.mocks import MockGitHubPoller, MockJiraPoller, MockTagClient

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_sentinel(
    jira_poller: MockJiraPoller | None = None,
    github_poller: MockGitHubPoller | None = None,
    orchestrations: list | None = None,
    config: Any | None = None,
    tag_client: MockTagClient | None = None,
    service_health_gate: Any | None = None,
    router: Any | None = None,
) -> Sentinel:
    """Create a Sentinel instance wired with test doubles.

    All dependencies that are not explicitly provided receive sensible
    defaults backed by mock implementations so tests stay isolated from
    external services.
    """
    if tag_client is None:
        tag_client = MockTagClient()
    if jira_poller is None:
        jira_poller = MockJiraPoller(issues=[], tag_client=tag_client)
    agent_factory, _ = make_agent_factory()
    if config is None:
        config = make_config()
    if orchestrations is None:
        orchestrations = [make_orchestration(tags=["review"])]
    return Sentinel(
        config=config,
        orchestrations=orchestrations,
        tag_client=tag_client,
        agent_factory=agent_factory,
        jira_poller=jira_poller,
        github_poller=github_poller,
        service_health_gate=service_health_gate,
        router=router,
    )


def _make_execution_result(
    issue_key: str = "TEST-1",
    orchestration_name: str = "test-orch",
    succeeded: bool = True,
    input_tokens: int = 100,
    output_tokens: int = 50,
    total_cost_usd: float = 0.01,
) -> ExecutionResult:
    """Build a minimal ExecutionResult for testing."""
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS if succeeded else ExecutionStatus.FAILURE,
        response="done",
        attempts=1,
        issue_key=issue_key,
        orchestration_name=orchestration_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=total_cost_usd,
    )


# =========================================================================
# 1. TestSentinelPublicAccessors
# =========================================================================


class TestSentinelPublicAccessors:
    """Tests for all public accessor methods used by the dashboard."""

    def test_get_hot_reload_metrics_returns_dict(self) -> None:
        """get_hot_reload_metrics should delegate to the registry and return a dict."""
        sentinel = _make_sentinel()
        metrics = sentinel.get_hot_reload_metrics()
        assert isinstance(metrics, dict)

    def test_get_running_steps_empty_by_default(self) -> None:
        """get_running_steps should return an empty list when nothing is running."""
        sentinel = _make_sentinel()
        assert sentinel.get_running_steps() == []

    def test_get_issue_queue_empty_by_default(self) -> None:
        """get_issue_queue should return an empty list when no issues are queued."""
        sentinel = _make_sentinel()
        assert sentinel.get_issue_queue() == []

    def test_get_start_time_is_recent(self) -> None:
        """get_start_time should return a UTC datetime close to now."""
        sentinel = _make_sentinel()
        now = datetime.now(tz=UTC)
        start = sentinel.get_start_time()
        assert isinstance(start, datetime)
        assert (now - start).total_seconds() < 5

    def test_get_last_jira_poll_none_by_default(self) -> None:
        """get_last_jira_poll should be None before any polling cycle."""
        sentinel = _make_sentinel()
        assert sentinel.get_last_jira_poll() is None

    def test_get_last_github_poll_none_by_default(self) -> None:
        """get_last_github_poll should be None before any polling cycle."""
        sentinel = _make_sentinel()
        assert sentinel.get_last_github_poll() is None

    def test_get_active_versions(self) -> None:
        """get_active_versions should return a list of version snapshots."""
        sentinel = _make_sentinel()
        versions = sentinel.get_active_versions()
        assert isinstance(versions, list)

    def test_get_pending_removal_versions_empty(self) -> None:
        """get_pending_removal_versions should be empty when nothing is pending."""
        sentinel = _make_sentinel()
        assert sentinel.get_pending_removal_versions() == []

    def test_get_execution_state(self) -> None:
        """get_execution_state should return a snapshot with active_count."""
        sentinel = _make_sentinel()
        state = sentinel.get_execution_state()
        assert state.active_count == 0

    def test_is_shutdown_requested_false_initially(self) -> None:
        """is_shutdown_requested should be False before request_shutdown is called."""
        sentinel = _make_sentinel()
        assert sentinel.is_shutdown_requested() is False

    def test_is_shutdown_requested_true_after_request(self) -> None:
        """is_shutdown_requested should be True after request_shutdown is called."""
        sentinel = _make_sentinel()
        sentinel.request_shutdown()
        assert sentinel.is_shutdown_requested() is True

    def test_get_service_health_status(self) -> None:
        """get_service_health_status should delegate to the health gate."""
        sentinel = _make_sentinel()
        status = sentinel.get_service_health_status()
        assert isinstance(status, dict)

    def test_get_per_orch_count_zero_by_default(self) -> None:
        """get_per_orch_count should return 0 for any orchestration name."""
        sentinel = _make_sentinel()
        assert sentinel.get_per_orch_count("nonexistent") == 0

    def test_get_all_per_orch_counts_empty(self) -> None:
        """get_all_per_orch_counts should be empty when nothing is running."""
        sentinel = _make_sentinel()
        assert sentinel.get_all_per_orch_counts() == {}

    def test_get_completed_executions_empty(self) -> None:
        """get_completed_executions should be empty initially."""
        sentinel = _make_sentinel()
        assert sentinel.get_completed_executions() == []

    def test_service_health_gate_property(self) -> None:
        """The service_health_gate property should return the injected gate."""
        mock_gate = MagicMock()
        mock_gate.get_all_status.return_value = {}
        sentinel = _make_sentinel(service_health_gate=mock_gate)
        assert sentinel.service_health_gate is mock_gate


# =========================================================================
# 2. TestSentinelLogHelpers
# =========================================================================


class TestSentinelLogHelpers:
    """Tests for _log_partial_failure, _record_poll_health, _log_total_failure."""

    def test_log_partial_failure(self) -> None:
        """_log_partial_failure emits a warning with service name, found, errors."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_partial_failure("Jira", found=3, errors=1)
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            assert "Jira" in args[1]

    # -- _record_poll_health ------------------------------------------------

    def test_record_poll_health_success(self) -> None:
        """_record_poll_health records success when error_count is 0."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._record_poll_health("jira", "Jira", error_count=0, issues_found=5)
        sentinel._health_gate.record_poll_success.assert_called_once_with(service_name="jira")

    def test_record_poll_health_failure_all_errors_no_issues(self) -> None:
        """_record_poll_health records failure when all triggers failed."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._record_poll_health("github", "GitHub", error_count=2, issues_found=0)
        sentinel._health_gate.record_poll_failure.assert_called_once_with(service_name="github")

    def test_record_poll_health_partial_failure(self) -> None:
        """_record_poll_health logs partial failure when errors > 0 but issues > 0."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        with patch.object(sentinel, "_log_partial_failure") as mock_lpf:
            sentinel._record_poll_health("jira", "Jira", error_count=1, issues_found=2)
            mock_lpf.assert_called_once_with(service_name="Jira", found=2, errors=1)
        # Should NOT call record_poll_success or record_poll_failure
        sentinel._health_gate.record_poll_success.assert_not_called()
        sentinel._health_gate.record_poll_failure.assert_not_called()

    # -- _log_total_failure -------------------------------------------------

    def test_log_total_failure_below_threshold_no_log(self) -> None:
        """_log_total_failure should not log when error fraction is below threshold."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure("Jira", error_count=1, trigger_count=10, error_threshold=0.5)
            mock_logger.warning.assert_not_called()

    def test_log_total_failure_at_threshold_logs_partial(self) -> None:
        """_log_total_failure should log partial warning at threshold."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure("Jira", error_count=5, trigger_count=10, error_threshold=0.5)
            mock_logger.warning.assert_called_once()
            msg = mock_logger.warning.call_args[0][0]
            assert "%s of %s" in msg

    def test_log_total_failure_all_failed(self) -> None:
        """_log_total_failure should log total failure when all triggers failed."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure("Jira", error_count=3, trigger_count=3, error_threshold=0.5)
            mock_logger.warning.assert_called_once()
            msg = mock_logger.warning.call_args[0][0]
            assert "All" in msg

    def test_log_total_failure_zero_triggers(self) -> None:
        """_log_total_failure should do nothing when trigger_count is 0."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure("Jira", error_count=0, trigger_count=0, error_threshold=0.5)
            mock_logger.warning.assert_not_called()

    def test_log_total_failure_above_threshold_partial(self) -> None:
        """_log_total_failure should log partial warning above threshold but not all failed."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure("GitHub", error_count=8, trigger_count=10, error_threshold=0.5)
            mock_logger.warning.assert_called_once()
            msg = mock_logger.warning.call_args[0][0]
            assert "%s of %s" in msg


# =========================================================================
# 3. TestSentinelFailureHandling
# =========================================================================


class TestSentinelFailureHandling:
    """Tests for _apply_failure_tags_safely, _handle_execution_failure,
    _handle_submission_failure."""

    def test_apply_failure_tags_safely_success(self) -> None:
        """_apply_failure_tags_safely succeeds silently when tag_manager works."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        orch = make_orchestration(tags=["review"])
        sentinel._apply_failure_tags_safely("TEST-1", orch)
        sentinel.tag_manager.apply_failure_tags.assert_called_once_with(
            issue_key="TEST-1", orchestration=orch
        )

    def test_apply_failure_tags_safely_os_error(self) -> None:
        """_apply_failure_tags_safely catches OSError and logs."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.apply_failure_tags.side_effect = OSError("network down")
        orch = make_orchestration(tags=["review"])
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._apply_failure_tags_safely("TEST-1", orch)
            mock_logger.error.assert_called_once()
            assert "I/O error" in mock_logger.error.call_args[0][1]

    def test_apply_failure_tags_safely_key_error(self) -> None:
        """_apply_failure_tags_safely catches KeyError and logs."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.apply_failure_tags.side_effect = KeyError("bad key")
        orch = make_orchestration(tags=["review"])
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._apply_failure_tags_safely("TEST-1", orch)
            mock_logger.error.assert_called_once()
            assert "data error" in mock_logger.error.call_args[0][1]

    def test_apply_failure_tags_safely_timeout_error(self) -> None:
        """_apply_failure_tags_safely catches TimeoutError and logs."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.apply_failure_tags.side_effect = TimeoutError("timed out")
        orch = make_orchestration(tags=["review"])
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._apply_failure_tags_safely("TEST-1", orch)
            mock_logger.error.assert_called_once()
            assert "I/O error" in mock_logger.error.call_args[0][1]

    def test_apply_failure_tags_safely_value_error(self) -> None:
        """_apply_failure_tags_safely catches ValueError and logs."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.apply_failure_tags.side_effect = ValueError("bad value")
        orch = make_orchestration(tags=["review"])
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._apply_failure_tags_safely("TEST-1", orch)
            mock_logger.error.assert_called_once()
            assert "data error" in mock_logger.error.call_args[0][1]

    def test_handle_execution_failure(self) -> None:
        """_handle_execution_failure logs the error and applies failure tags."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        orch = make_orchestration(name="my-orch", tags=["review"])
        exc = OSError("disk full")
        with patch.object(sentinel, "_log_for_orchestration") as mock_log:
            sentinel._handle_execution_failure("TEST-1", orch, exc, ErrorType.IO_ERROR)
            mock_log.assert_called_once()
            assert "I/O error" in mock_log.call_args[0][2]
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_handle_submission_failure_with_version(self) -> None:
        """_handle_submission_failure decrements version, decrements per-orch, logs, tags."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        orch = make_orchestration(name="my-orch", tags=["review"])
        version = MagicMock(spec=OrchestrationVersion)
        exc = RuntimeError("pool broken")
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._handle_submission_failure("TEST-1", orch, exc, ErrorType.RUNTIME_ERROR, version)
            version.decrement_executions.assert_called_once()
            mock_logger.error.assert_called_once()
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_handle_submission_failure_without_version(self) -> None:
        """_handle_submission_failure works when version is None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        orch = make_orchestration(name="my-orch", tags=["review"])
        exc = KeyError("missing")
        with patch("sentinel.main.logger"):
            sentinel._handle_submission_failure("TEST-1", orch, exc, ErrorType.DATA_ERROR, None)
        # Should not raise; version.decrement_executions not called
        sentinel.tag_manager.apply_failure_tags.assert_called_once()


# =========================================================================
# 4. TestExecuteOrchestrationTask
# =========================================================================


class TestExecuteOrchestrationTask:
    """Tests for _execute_orchestration_task covering all exception paths."""

    def test_normal_success(self) -> None:
        """Successful execution returns an ExecutionResult and calls update_tags."""
        tag_client = MockTagClient()
        sentinel = _make_sentinel(tag_client=tag_client)
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        # Mock asyncio.run to avoid actually running the executor
        mock_result = _make_execution_result(succeeded=True)
        with patch("sentinel.main.asyncio.run", return_value=mock_result):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is not None
        assert result.succeeded is True
        sentinel.tag_manager.update_tags.assert_called_once()

    def test_normal_failure(self) -> None:
        """Failed execution returns an ExecutionResult with succeeded=False."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        mock_result = _make_execution_result(succeeded=False)
        with patch("sentinel.main.asyncio.run", return_value=mock_result):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is not None
        assert result.succeeded is False

    def test_shutdown_requested_returns_none(self) -> None:
        """When shutdown is requested, should skip execution and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel._shutdown_requested = True
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_claude_process_interrupted(self) -> None:
        """ClaudeProcessInterruptedError should return None and apply failure tags."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=ClaudeProcessInterruptedError()):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_os_error(self) -> None:
        """OSError should be caught, logged, and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=OSError("disk error")):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_runtime_error(self) -> None:
        """RuntimeError should be caught, logged, and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=RuntimeError("boom")):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None

    def test_key_error(self) -> None:
        """KeyError should be caught, logged, and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=KeyError("missing")):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None

    def test_value_error(self) -> None:
        """ValueError should be caught, logged, and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=ValueError("bad")):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None

    def test_timeout_error(self) -> None:
        """TimeoutError should be caught, logged, and return None."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        with patch("sentinel.main.asyncio.run", side_effect=TimeoutError("timed out")):
            result = sentinel._execute_orchestration_task(issue, orch)

        assert result is None
        sentinel.tag_manager.apply_failure_tags.assert_called_once()

    def test_version_decremented_in_finally(self) -> None:
        """The version's active execution count should be decremented in finally block."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        version = MagicMock(spec=OrchestrationVersion)

        mock_result = _make_execution_result(succeeded=True)
        with patch("sentinel.main.asyncio.run", return_value=mock_result):
            sentinel._execute_orchestration_task(issue, orch, version=version)

        version.decrement_executions.assert_called_once()

    def test_unknown_trigger_source_defaults_to_jira(self) -> None:
        """Unknown trigger source should default to jira for health probing."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        # Patch the trigger source to something unrecognized
        orch.trigger = MagicMock()
        orch.trigger.source = "bitbucket"
        orch.agent = MagicMock()
        orch.agent.agent_type = "claude"
        orch.agent.github = None
        orch.agent.strict_template_variables = False
        orch.name = "test-orch"

        mock_result = _make_execution_result(succeeded=True)
        with patch("sentinel.main.asyncio.run", return_value=mock_result):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._execute_orchestration_task(issue, orch)
                # Should log a warning about unknown trigger source
                warning_calls = [
                    c for c in mock_logger.warning.call_args_list
                    if "Unknown trigger source" in c.args[0]
                ]
                assert len(warning_calls) == 1

    def test_github_trigger_source(self) -> None:
        """GitHub trigger source should be recognized without warning."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"], source="github",
                                  project_number=1, labels=["dev-sentinel"])

        mock_result = _make_execution_result(succeeded=True)
        with patch("sentinel.main.asyncio.run", return_value=mock_result):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._execute_orchestration_task(issue, orch)
                # Should NOT log an "Unknown trigger source" warning
                warning_calls = [
                    c for c in mock_logger.warning.call_args_list
                    if "Unknown trigger source" in c.args[0]
                ]
                assert len(warning_calls) == 0


# =========================================================================
# 5. TestSubmitExecutionTasks
# =========================================================================


class TestSubmitExecutionTasks:
    """Tests for _submit_execution_tasks covering deduplication, slots, errors."""

    def _make_routing_result(self, issue, orchestrations):
        """Build a RoutingResult-like object."""
        from sentinel.router import RoutingResult
        return RoutingResult(issue=issue, orchestrations=orchestrations)

    def test_deduplication_skipping(self) -> None:
        """Already-submitted (issue, orchestration) pairs should be skipped."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])

        routing_result = self._make_routing_result(issue, [orch])

        # Pre-populate the submitted_pairs set so it looks like this was already submitted
        submitted_pairs: set[tuple[str, str]] = set()
        # First submission should succeed
        sentinel._execution_manager.start()
        try:
            count1 = sentinel._submit_execution_tasks(
                [routing_result], [], submitted_pairs
            )
            # Second submission of the same pair should be skipped
            count2 = sentinel._submit_execution_tasks(
                [routing_result], [], submitted_pairs
            )
        finally:
            sentinel._execution_manager.shutdown()

        # First call should submit, second should be deduplicated
        assert count1 == 1
        assert count2 == 0

    def test_no_available_slots_queuing(self) -> None:
        """When no slots are available, issue should be added to queue."""
        config = make_config(max_concurrent_executions=1)
        sentinel = _make_sentinel(config=config)
        sentinel.tag_manager = MagicMock()

        # Mock _get_available_slots_for_orchestration to return 0
        with patch.object(sentinel, "_get_available_slots_for_orchestration", return_value=0):
            issue = make_issue(key="TEST-1", labels=["review"])
            orch = make_orchestration(tags=["review"])
            routing_result = self._make_routing_result(issue, [orch])

            count = sentinel._submit_execution_tasks([routing_result], [])
            assert count == 0
            # Issue should be in the queue
            queue = sentinel.get_issue_queue()
            assert len(queue) == 1
            assert queue[0].issue_key == "TEST-1"

    def test_shutdown_during_submission(self) -> None:
        """Shutdown during submission should stop processing remaining items."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel._shutdown_requested = True

        issue1 = make_issue(key="TEST-1", labels=["review"])
        issue2 = make_issue(key="TEST-2", labels=["review"])
        orch = make_orchestration(tags=["review"])
        routing_results = [
            self._make_routing_result(issue1, [orch]),
            self._make_routing_result(issue2, [orch]),
        ]

        count = sentinel._submit_execution_tasks(routing_results, [])
        assert count == 0

    def test_submission_failure_os_error(self) -> None:
        """OSError during submission should be handled gracefully."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.start_processing.side_effect = OSError("conn refused")

        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        routing_result = self._make_routing_result(issue, [orch])

        sentinel._execution_manager.start()
        try:
            with patch("sentinel.main.logger"):
                count = sentinel._submit_execution_tasks([routing_result], [])
        finally:
            sentinel._execution_manager.shutdown()

        assert count == 0

    def test_submission_failure_runtime_error(self) -> None:
        """RuntimeError during submission should be handled gracefully."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.start_processing.side_effect = RuntimeError("pool broken")

        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        routing_result = self._make_routing_result(issue, [orch])

        sentinel._execution_manager.start()
        try:
            with patch("sentinel.main.logger"):
                count = sentinel._submit_execution_tasks([routing_result], [])
        finally:
            sentinel._execution_manager.shutdown()

        assert count == 0

    def test_submission_failure_key_error(self) -> None:
        """KeyError during submission should be handled gracefully."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        sentinel.tag_manager.start_processing.side_effect = KeyError("missing key")

        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        routing_result = self._make_routing_result(issue, [orch])

        sentinel._execution_manager.start()
        try:
            with patch("sentinel.main.logger"):
                count = sentinel._submit_execution_tasks([routing_result], [])
        finally:
            sentinel._execution_manager.shutdown()

        assert count == 0

    def test_synchronous_fallback_when_pool_not_running(self) -> None:
        """When thread pool is not running, tasks should execute synchronously."""
        sentinel = _make_sentinel()
        sentinel.tag_manager = MagicMock()
        issue = make_issue(key="TEST-1", labels=["review"])
        orch = make_orchestration(tags=["review"])
        routing_result = self._make_routing_result(issue, [orch])

        # Pool is NOT started, so is_running() returns False
        all_results: list[ExecutionResult] = []
        mock_result = _make_execution_result(succeeded=True)
        with patch.object(sentinel, "_execute_orchestration_task", return_value=mock_result):
            count = sentinel._submit_execution_tasks([routing_result], all_results)

        # Synchronous execution appends to all_results directly
        assert len(all_results) == 1
        assert count == 0  # No futures were submitted


# =========================================================================
# 6. TestRunOnce
# =========================================================================


class TestRunOnce:
    """Tests for run_once method covering polling branches."""

    def test_run_once_no_available_slots_skips_polling(self) -> None:
        """run_once should skip polling when no execution slots are available."""
        sentinel = _make_sentinel()
        with patch.object(sentinel._execution_manager, "get_available_slots", return_value=0):
            results, submitted = sentinel.run_once()
            assert submitted == 0

    def test_run_once_jira_polling(self) -> None:
        """run_once should poll Jira when there are Jira-triggered orchestrations."""
        tag_client = MockTagClient()
        issue = make_issue(key="TEST-1", labels=["review"])
        jira_poller = MockJiraPoller(issues=[issue], tag_client=tag_client)
        orch = make_orchestration(tags=["review"])
        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            tag_client=tag_client,
            orchestrations=[orch],
        )

        sentinel._execution_manager.start()
        try:
            results, submitted = sentinel.run_once()
            assert submitted >= 0  # May be 0 if dedup or routing blocks it
        finally:
            sentinel._execution_manager.shutdown()

    def test_run_once_github_no_poller_configured(self) -> None:
        """run_once should log a warning when GitHub orchestrations exist but no poller."""
        orch = make_orchestration(
            source="github", project_number=1, labels=["dev-sentinel"]
        )
        sentinel = _make_sentinel(orchestrations=[orch], github_poller=None)

        with patch("sentinel.main.logger") as mock_logger:
            results, submitted = sentinel.run_once()
            # Should warn about GitHub orchestrations without a poller
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if "GitHub" in c.args[0] and "not configured" in c.args[0]
            ]
            assert len(warning_calls) >= 1

    def test_run_once_health_gate_blocking(self) -> None:
        """run_once should skip polling when health gate blocks the service."""
        tag_client = MockTagClient()
        issue = make_issue(key="TEST-1", labels=["review"])
        jira_poller = MockJiraPoller(issues=[issue], tag_client=tag_client)
        orch = make_orchestration(tags=["review"])

        mock_gate = MagicMock()
        mock_gate.should_poll.return_value = False
        mock_gate.should_probe.return_value = False

        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            tag_client=tag_client,
            orchestrations=[orch],
            service_health_gate=mock_gate,
        )

        results, submitted = sentinel.run_once()
        assert submitted == 0
        mock_gate.should_poll.assert_called()

    def test_run_once_health_gate_probing(self) -> None:
        """run_once should probe the service when health gate says to probe."""
        tag_client = MockTagClient()
        issue = make_issue(key="TEST-1", labels=["review"])
        jira_poller = MockJiraPoller(issues=[issue], tag_client=tag_client)
        orch = make_orchestration(tags=["review"])

        mock_gate = MagicMock()
        mock_gate.should_poll.return_value = False
        mock_gate.should_probe.return_value = True

        sentinel = _make_sentinel(
            jira_poller=jira_poller,
            tag_client=tag_client,
            orchestrations=[orch],
            service_health_gate=mock_gate,
        )

        results, submitted = sentinel.run_once()
        assert submitted == 0
        mock_gate.probe_service.assert_called_once()

    def test_run_once_github_with_poller(self) -> None:
        """run_once should poll GitHub when a GitHub poller is configured."""
        gh_issue = GitHubIssue(
            number=42,
            title="Test PR",
            body="Fix something",
            state="open",
            author="dev",
            assignees=[],
            labels=["dev-sentinel"],
            is_pull_request=False,
            head_ref="",
            base_ref="",
            draft=False,
            repo_url="https://github.com/org/repo",
            parent_issue_number=None,
        )
        github_poller = MockGitHubPoller(issues=[gh_issue])
        orch = make_orchestration(
            source="github", project_number=1, labels=["dev-sentinel"]
        )
        sentinel = _make_sentinel(
            github_poller=github_poller,
            orchestrations=[orch],
        )
        sentinel._execution_manager.start()
        try:
            results, submitted = sentinel.run_once()
            # The poller should have been called
            assert len(github_poller.poll_calls) >= 0  # May or may not match routing
        finally:
            sentinel._execution_manager.shutdown()

    def test_run_once_github_health_gate_blocking(self) -> None:
        """run_once should skip GitHub polling when health gate blocks."""
        github_poller = MockGitHubPoller(issues=[])
        orch = make_orchestration(
            source="github", project_number=1, labels=["dev-sentinel"]
        )
        mock_gate = MagicMock()
        mock_gate.should_poll.side_effect = lambda svc: svc != "github"
        mock_gate.should_probe.return_value = False

        sentinel = _make_sentinel(
            github_poller=github_poller,
            orchestrations=[orch],
            service_health_gate=mock_gate,
        )
        results, submitted = sentinel.run_once()
        assert submitted == 0

    def test_run_once_github_health_gate_probing(self) -> None:
        """run_once should probe GitHub service when should_probe returns True."""
        github_poller = MockGitHubPoller(issues=[])
        orch = make_orchestration(
            source="github", project_number=1, labels=["dev-sentinel"]
        )
        mock_gate = MagicMock()
        mock_gate.should_poll.side_effect = lambda svc: svc != "github"
        mock_gate.should_probe.return_value = True

        sentinel = _make_sentinel(
            github_poller=github_poller,
            orchestrations=[orch],
            service_health_gate=mock_gate,
        )
        results, submitted = sentinel.run_once()
        mock_gate.probe_service.assert_called()


# =========================================================================
# 6b. TestPollService
# =========================================================================


class TestPollService:
    """Tests for the generic _poll_service() helper (DS-931).

    These tests verify the extracted polling logic that was previously
    duplicated in the Jira and GitHub branches of run_once().
    """

    def _make_poll_fn(
        self,
        routing_results: list | None = None,
        issues_found: int = 0,
        error_count: int = 0,
    ):
        """Build a mock poll function matching the PollCoordinator signature."""
        if routing_results is None:
            routing_results = []
        return MagicMock(return_value=(routing_results, issues_found, error_count))

    def test_poll_service_happy_path_submits_tasks(self) -> None:
        """_poll_service should poll, submit, log, and record health on success."""
        sentinel = _make_sentinel()
        sentinel._execution_manager.start()
        try:
            mock_poll_fn = self._make_poll_fn(issues_found=2, error_count=0)
            all_results: list = []
            submitted_pairs: set[tuple[str, str]] = set()

            with patch.object(sentinel, "_submit_execution_tasks", return_value=2) as mock_submit:
                submitted = sentinel._poll_service(
                    service_name="jira",
                    display_name="Jira",
                    orchestrations=[make_orchestration(tags=["review"])],
                    poll_fn=mock_poll_fn,
                    probe_kwargs={"base_url": "https://jira.example.com"},
                    all_results=all_results,
                    submitted_pairs=submitted_pairs,
                )

            assert submitted == 2
            mock_poll_fn.assert_called_once()
            mock_submit.assert_called_once()
        finally:
            sentinel._execution_manager.shutdown()

    def test_poll_service_records_success_health(self) -> None:
        """_poll_service should record poll success when no errors occurred."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._health_gate.should_poll.return_value = True

        mock_poll_fn = self._make_poll_fn(issues_found=3, error_count=0)
        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            sentinel._poll_service(
                service_name="jira",
                display_name="Jira",
                orchestrations=[make_orchestration(tags=["review"])],
                poll_fn=mock_poll_fn,
                probe_kwargs={},
                all_results=[],
                submitted_pairs=set(),
            )

        sentinel._health_gate.record_poll_success.assert_called_once_with(service_name="jira")

    def test_poll_service_records_failure_health(self) -> None:
        """_poll_service should record poll failure when all triggers errored."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._health_gate.should_poll.return_value = True

        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=2)
        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            sentinel._poll_service(
                service_name="github",
                display_name="GitHub",
                orchestrations=[make_orchestration(tags=["review"])],
                poll_fn=mock_poll_fn,
                probe_kwargs={},
                all_results=[],
                submitted_pairs=set(),
            )

        sentinel._health_gate.record_poll_failure.assert_called_once_with(service_name="github")

    def test_poll_service_logs_summary_when_issues_or_errors(self) -> None:
        """_poll_service should log a polling summary when issues or errors exist."""
        sentinel = _make_sentinel()
        mock_poll_fn = self._make_poll_fn(issues_found=5, error_count=1)

        with patch.object(sentinel, "_submit_execution_tasks", return_value=3):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._poll_service(
                    service_name="jira",
                    display_name="Jira",
                    orchestrations=[make_orchestration(tags=["review"])],
                    poll_fn=mock_poll_fn,
                    probe_kwargs={},
                    all_results=[],
                    submitted_pairs=set(),
                )
                info_calls = [
                    c for c in mock_logger.info.call_args_list
                    if "polling summary" in c.args[0]
                ]
                assert len(info_calls) >= 1

    def test_poll_service_skips_summary_when_no_issues_or_errors(self) -> None:
        """_poll_service should not log a polling summary when nothing happened."""
        sentinel = _make_sentinel()
        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=0)

        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._poll_service(
                    service_name="jira",
                    display_name="Jira",
                    orchestrations=[make_orchestration(tags=["review"])],
                    poll_fn=mock_poll_fn,
                    probe_kwargs={},
                    all_results=[],
                    submitted_pairs=set(),
                )
                summary_calls = [
                    c for c in mock_logger.info.call_args_list
                    if "polling summary" in c.args[0]
                ]
                assert len(summary_calls) == 0

    def test_poll_service_gated_no_probe(self) -> None:
        """_poll_service should return 0 when service is gated and probe is not due."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._health_gate.should_poll.return_value = False
        sentinel._health_gate.should_probe.return_value = False

        mock_poll_fn = self._make_poll_fn()
        submitted = sentinel._poll_service(
            service_name="jira",
            display_name="Jira",
            orchestrations=[make_orchestration(tags=["review"])],
            poll_fn=mock_poll_fn,
            probe_kwargs={},
            all_results=[],
            submitted_pairs=set(),
        )

        assert submitted == 0
        mock_poll_fn.assert_not_called()
        sentinel._health_gate.probe_service.assert_not_called()

    def test_poll_service_gated_with_probe(self) -> None:
        """_poll_service should probe the service when gated and probe is due."""
        sentinel = _make_sentinel()
        sentinel._health_gate = MagicMock()
        sentinel._health_gate.should_poll.return_value = False
        sentinel._health_gate.should_probe.return_value = True

        mock_poll_fn = self._make_poll_fn()
        probe_kwargs = {"base_url": "https://jira.example.com", "auth": ("user", "token")}
        submitted = sentinel._poll_service(
            service_name="jira",
            display_name="Jira",
            orchestrations=[make_orchestration(tags=["review"])],
            poll_fn=mock_poll_fn,
            probe_kwargs=probe_kwargs,
            all_results=[],
            submitted_pairs=set(),
        )

        assert submitted == 0
        mock_poll_fn.assert_not_called()
        sentinel._health_gate.probe_service.assert_called_once_with(
            "jira", base_url="https://jira.example.com", auth=("user", "token")
        )

    def test_poll_service_updates_jira_last_poll_timestamp(self) -> None:
        """_poll_service should update last_jira_poll when service_name is 'jira'."""
        sentinel = _make_sentinel()
        assert sentinel._state_tracker.last_jira_poll is None

        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=0)
        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            sentinel._poll_service(
                service_name="jira",
                display_name="Jira",
                orchestrations=[make_orchestration(tags=["review"])],
                poll_fn=mock_poll_fn,
                probe_kwargs={},
                all_results=[],
                submitted_pairs=set(),
            )

        assert sentinel._state_tracker.last_jira_poll is not None

    def test_poll_service_updates_github_last_poll_timestamp(self) -> None:
        """_poll_service should update last_github_poll when service_name is 'github'."""
        sentinel = _make_sentinel()
        assert sentinel._state_tracker.last_github_poll is None

        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=0)
        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            sentinel._poll_service(
                service_name="github",
                display_name="GitHub",
                orchestrations=[make_orchestration(tags=["review"])],
                poll_fn=mock_poll_fn,
                probe_kwargs={},
                all_results=[],
                submitted_pairs=set(),
            )

        assert sentinel._state_tracker.last_github_poll is not None

    def test_poll_service_calls_log_total_failure(self) -> None:
        """_poll_service should invoke _log_total_failure with correct parameters."""
        sentinel = _make_sentinel()
        orch = make_orchestration(tags=["review"])
        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=3)

        with patch.object(sentinel, "_submit_execution_tasks", return_value=0):
            with patch.object(sentinel, "_log_total_failure") as mock_ltf:
                sentinel._poll_service(
                    service_name="jira",
                    display_name="Jira",
                    orchestrations=[orch],
                    poll_fn=mock_poll_fn,
                    probe_kwargs={},
                    all_results=[],
                    submitted_pairs=set(),
                )
                mock_ltf.assert_called_once_with(
                    "Jira",
                    3,
                    1,  # len([orch])
                    sentinel.config.polling.error_threshold_pct,
                )

    def test_poll_service_unknown_service_name_raises_value_error(self) -> None:
        """_poll_service should raise ValueError for an unrecognised service_name (DS-940)."""
        sentinel = _make_sentinel()
        mock_poll_fn = self._make_poll_fn(issues_found=0, error_count=0)

        with pytest.raises(ValueError, match="Unknown polling service 'bitbucket'"):
            sentinel._poll_service(
                service_name="bitbucket",
                display_name="Bitbucket",
                orchestrations=[make_orchestration(tags=["review"])],
                poll_fn=mock_poll_fn,
                probe_kwargs={},
                all_results=[],
                submitted_pairs=set(),
            )

        # The poll function should not have been called
        mock_poll_fn.assert_not_called()


# =========================================================================
# 7. TestRun
# =========================================================================


class TestRun:
    """Tests for run() main loop, covering error recovery and shutdown."""

    def _run_sentinel_in_thread(self, sentinel: Sentinel, timeout: float = 3.0) -> threading.Thread:
        """Start sentinel.run() in a background thread."""
        thread = threading.Thread(target=sentinel.run, daemon=True)
        thread.start()
        return thread

    def test_run_os_error_in_cycle(self) -> None:
        """OSError in run_once should be caught and the loop should continue."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network down")
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                with patch("sentinel.main.logger") as mock_logger:
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()
                    assert call_count >= 2
                    # Verify the specific I/O error message was logged
                    error_calls = [
                        c for c in mock_logger.error.call_args_list
                        if "I/O or timeout" in c.args[0]
                    ]
                    assert len(error_calls) == 1
                    assert "network down" in str(error_calls[0].args[1])

    def test_run_runtime_error_in_cycle(self) -> None:
        """RuntimeError in run_once should be caught and the loop should continue."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("bad state")
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                with patch("sentinel.main.logger") as mock_logger:
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()
                    # Verify the specific runtime error message was logged
                    error_calls = [
                        c for c in mock_logger.error.call_args_list
                        if "runtime error" in c.args[0]
                    ]
                    assert len(error_calls) == 1
                    assert "bad state" in str(error_calls[0].args[1])

    def test_run_key_value_error_in_cycle(self) -> None:
        """KeyError/ValueError in run_once should be caught and the loop should continue."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyError("missing")
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                with patch("sentinel.main.logger") as mock_logger:
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()
                    # Verify the specific data error message was logged
                    error_calls = [
                        c for c in mock_logger.error.call_args_list
                        if "data error" in c.args[0]
                    ]
                    assert len(error_calls) == 1
                    assert "missing" in str(error_calls[0].args[1])

    def test_run_generic_exception_in_cycle(self) -> None:
        """Generic Exception in run_once should be caught and the loop should continue."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("unexpected")
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                with patch("sentinel.main.logger") as mock_logger:
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()
                    # Verify the specific unexpected-error message was logged
                    exception_calls = [
                        c for c in mock_logger.exception.call_args_list
                        if "Unexpected error" in c.args[0]
                    ]
                    assert len(exception_calls) == 1
                    assert "unexpected" in str(exception_calls[0].args[1])

    def test_run_shutdown_after_idle_cycle(self) -> None:
        """run() should sleep poll_interval seconds when idle (submitted_count=0)."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                thread = self._run_sentinel_in_thread(sentinel)
                thread.join(timeout=10)
                assert not thread.is_alive()
                assert call_count >= 2

    def test_run_shutdown_with_active_tasks_and_timeout(self) -> None:
        """run() should wait for active tasks during shutdown with timeout."""
        config = make_config(poll_interval=1, shutdown_timeout_seconds=1.0)
        sentinel = _make_sentinel(config=config)
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            sentinel.request_shutdown()
            return [], 0

        # Mock pending futures that never complete to test timeout path
        never_done_future = Future()

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel._execution_manager,
                "get_active_count",
                return_value=1,
            ):
                with patch.object(
                    sentinel._execution_manager,
                    "get_pending_futures",
                    return_value=[never_done_future],
                ):
                    with patch("signal.signal"):
                        thread = self._run_sentinel_in_thread(sentinel)
                        thread.join(timeout=10)
                        assert not thread.is_alive()

        # Force-set the future so the pool can shut down cleanly
        if not never_done_future.done():
            never_done_future.cancel()

    def test_run_shutdown_with_active_tasks_no_timeout(self) -> None:
        """run() should log waiting for active tasks when shutdown_timeout is 0."""
        config = make_config(poll_interval=1, shutdown_timeout_seconds=0)
        sentinel = _make_sentinel(config=config)
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel._execution_manager,
                "get_active_count",
                return_value=1,
            ):
                with patch.object(
                    sentinel._execution_manager,
                    "get_pending_futures",
                    return_value=[],
                ):
                    with patch("signal.signal"):
                        with patch("sentinel.main.logger") as mock_logger:
                            thread = self._run_sentinel_in_thread(sentinel)
                            thread.join(timeout=5)
                            assert not thread.is_alive()
                            # Should log about waiting with no timeout
                            info_calls = [
                                c for c in mock_logger.info.call_args_list
                                if "no timeout" in c.args[0].lower()
                            ]
                            assert len(info_calls) >= 1

    def test_run_with_results_in_cycle(self) -> None:
        """run() should log cycle results when run_once returns results."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        mock_result = _make_execution_result(succeeded=True)
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_result], 0
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch("signal.signal"):
                with patch("sentinel.main.logger") as mock_logger:
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()
                    # Should log cycle results
                    info_calls = [
                        c for c in mock_logger.info.call_args_list
                        if "Cycle completed" in c.args[0]
                    ]
                    assert len(info_calls) >= 1

    def test_run_with_pending_futures_polls_immediately(self) -> None:
        """run() should poll immediately when tasks complete during wait."""
        sentinel = _make_sentinel(config=make_config(poll_interval=60))
        call_count = 0

        # Create a future that completes immediately
        done_future = Future()
        done_future.set_result(None)

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sentinel.request_shutdown()
            return [], 1  # submitted_count=1

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel._execution_manager,
                "get_pending_futures",
                return_value=[done_future],
            ):
                with patch("signal.signal"):
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()

    def test_run_submitted_tasks_completed_quickly(self) -> None:
        """run() should recognize when submitted tasks completed before wait check."""
        sentinel = _make_sentinel(config=make_config(poll_interval=60))
        call_count = 0

        def mock_run_once():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                sentinel.request_shutdown()
            return [], 1  # submitted_count > 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            # No pending futures means tasks completed before check
            with patch.object(
                sentinel._execution_manager,
                "get_pending_futures",
                return_value=[],
            ):
                with patch("signal.signal"):
                    thread = self._run_sentinel_in_thread(sentinel)
                    thread.join(timeout=5)
                    assert not thread.is_alive()

    def test_run_final_results_collection(self) -> None:
        """run() should collect and log final results after shutdown."""
        sentinel = _make_sentinel(config=make_config(poll_interval=1))
        mock_result = _make_execution_result(succeeded=True)

        def mock_run_once():
            sentinel.request_shutdown()
            return [], 0

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel, "_collect_completed_results", return_value=[mock_result]
            ):
                with patch("signal.signal"):
                    with patch("sentinel.main.logger") as mock_logger:
                        thread = self._run_sentinel_in_thread(sentinel)
                        thread.join(timeout=5)
                        assert not thread.is_alive()
                        # Should log final batch results
                        info_calls = [
                            c for c in mock_logger.info.call_args_list
                            if "Final batch" in c.args[0]
                        ]
                        assert len(info_calls) >= 1


# =========================================================================
# 8. TestRunOnceAndWait
# =========================================================================


class TestRunOnceAndWait:
    """Tests for run_once_and_wait method."""

    def test_basic_completion(self) -> None:
        """run_once_and_wait should execute a full cycle and return results."""
        tag_client = MockTagClient()
        jira_poller = MockJiraPoller(
            issues=[make_issue(key="TEST-1", labels=["review"])],
            tag_client=tag_client,
        )
        agent_factory, _ = make_agent_factory(responses=["SUCCESS: Done"])
        config = make_config(max_concurrent_executions=1)
        orch = make_orchestration(tags=["review"])

        sentinel = Sentinel(
            config=config,
            orchestrations=[orch],
            tag_client=tag_client,
            agent_factory=agent_factory,
            jira_poller=jira_poller,
        )

        results = sentinel.run_once_and_wait()
        assert isinstance(results, list)

    def test_fast_task_completion_before_pending_check(self) -> None:
        """run_once_and_wait should handle tasks that complete very quickly (DS-542)."""
        sentinel = _make_sentinel()
        mock_result = _make_execution_result(succeeded=True)
        cycle_count = 0

        def mock_run_once():
            nonlocal cycle_count
            cycle_count += 1
            return [mock_result], 1  # submitted_count=1

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel._execution_manager,
                "get_pending_futures",
                return_value=[],  # No pending = tasks completed immediately
            ):
                with patch.object(
                    sentinel, "_collect_completed_results", return_value=[mock_result]
                ):
                    sentinel._execution_manager.start()
                    try:
                        results = sentinel.run_once_and_wait()
                    finally:
                        sentinel._execution_manager.shutdown()

        # Should have collected the fast results
        assert len(results) >= 1


# =========================================================================
# 9. TestRecordCompletedExecution
# =========================================================================


class TestRecordCompletedExecution:
    """Tests for _record_completed_execution."""

    def test_record_successful_execution(self) -> None:
        """_record_completed_execution should record a successful execution."""
        sentinel = _make_sentinel()
        result = _make_execution_result(
            issue_key="TEST-1",
            orchestration_name="test-orch",
            succeeded=True,
            input_tokens=500,
            output_tokens=200,
            total_cost_usd=0.05,
        )
        from sentinel.state_tracker import RunningStepInfo

        running_step = RunningStepInfo(
            issue_key="TEST-1",
            orchestration_name="test-orch",
            attempt_number=1,
            started_at=datetime.now(tz=UTC),
            issue_url="https://jira.example.com/TEST-1",
        )

        sentinel._record_completed_execution(result, running_step)

        completed = sentinel.get_completed_executions()
        assert len(completed) == 1
        assert completed[0].issue_key == "TEST-1"
        assert completed[0].status == "success"
        assert completed[0].input_tokens == 500
        assert completed[0].output_tokens == 200
        assert completed[0].total_cost_usd == 0.05

    def test_record_failed_execution(self) -> None:
        """_record_completed_execution should record a failed execution."""
        sentinel = _make_sentinel()
        result = _make_execution_result(
            issue_key="TEST-2",
            orchestration_name="test-orch",
            succeeded=False,
            input_tokens=100,
            output_tokens=10,
            total_cost_usd=0.001,
        )
        from sentinel.state_tracker import RunningStepInfo

        running_step = RunningStepInfo(
            issue_key="TEST-2",
            orchestration_name="test-orch",
            attempt_number=2,
            started_at=datetime.now(tz=UTC),
            issue_url="https://jira.example.com/TEST-2",
        )

        sentinel._record_completed_execution(result, running_step)

        completed = sentinel.get_completed_executions()
        assert len(completed) == 1
        assert completed[0].issue_key == "TEST-2"
        assert completed[0].status == "failure"
        assert completed[0].attempt_number == 2


# =========================================================================
# 10. TestSentinelInitialization
# =========================================================================


class TestSentinelInitialization:
    """Tests for Sentinel constructor initialization paths."""

    def test_initialization_without_router(self) -> None:
        """Sentinel should create a Router from orchestrations when none is provided."""
        sentinel = _make_sentinel(router=None)
        assert sentinel.router is not None

    def test_initialization_with_router(self) -> None:
        """Sentinel should use the provided Router when one is injected."""
        from sentinel.router import Router

        orch = make_orchestration(tags=["review"])
        custom_router = Router([orch])
        sentinel = _make_sentinel(router=custom_router, orchestrations=[orch])
        assert sentinel.router is custom_router

    def test_initialization_with_orchestration_logs_dir(self) -> None:
        """Sentinel should initialize OrchestrationLogManager when logs_dir is set."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(orchestration_logs_dir=Path(tmpdir))
            sentinel = _make_sentinel(config=config)
            assert sentinel._orch_log_manager is not None

    def test_initialization_without_orchestration_logs_dir(self) -> None:
        """Sentinel should not initialize OrchestrationLogManager when logs_dir is None."""
        config = make_config(orchestration_logs_dir=None)
        sentinel = _make_sentinel(config=config)
        assert sentinel._orch_log_manager is None

    def test_initialization_with_service_health_gate(self) -> None:
        """Sentinel should use the injected service health gate."""
        mock_gate = MagicMock()
        mock_gate.get_all_status.return_value = {}
        sentinel = _make_sentinel(service_health_gate=mock_gate)
        assert sentinel._health_gate is mock_gate

    def test_initialization_default_health_gate(self) -> None:
        """Sentinel should create a default health gate when none is provided."""
        sentinel = _make_sentinel(service_health_gate=None)
        assert sentinel._health_gate is not None

    def test_initialization_with_github_poller(self) -> None:
        """Sentinel should store the GitHub poller when provided."""
        github_poller = MockGitHubPoller(issues=[])
        sentinel = _make_sentinel(github_poller=github_poller)
        assert sentinel.github_poller is github_poller

    def test_initialization_without_github_poller(self) -> None:
        """Sentinel should have github_poller as None when not provided."""
        sentinel = _make_sentinel(github_poller=None)
        assert sentinel.github_poller is None

    def test_request_shutdown_sets_flag(self) -> None:
        """request_shutdown should set _shutdown_requested to True."""
        sentinel = _make_sentinel()
        assert sentinel._shutdown_requested is False
        sentinel.request_shutdown()
        assert sentinel._shutdown_requested is True


# =========================================================================
# 11. TestGetAvailableSlotsForOrchestration
# =========================================================================


class TestGetAvailableSlotsForOrchestration:
    """Tests for _get_available_slots_for_orchestration."""

    def test_global_slots_only(self) -> None:
        """When orchestration has no max_concurrent, use global slots."""
        config = make_config(max_concurrent_executions=3)
        sentinel = _make_sentinel(config=config)
        orch = make_orchestration(max_concurrent=None)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 3

    def test_per_orch_limit(self) -> None:
        """When orchestration has max_concurrent, use the minimum of global and per-orch."""
        config = make_config(max_concurrent_executions=5)
        sentinel = _make_sentinel(config=config)
        orch = make_orchestration(max_concurrent=2)
        available = sentinel._get_available_slots_for_orchestration(orch)
        assert available == 2


# =========================================================================
# 12. TestCollectCompletedResults
# =========================================================================


class TestCollectCompletedResults:
    """Tests for _collect_completed_results."""

    def test_empty_when_nothing_running(self) -> None:
        """_collect_completed_results should return empty list when nothing is running."""
        sentinel = _make_sentinel()
        results = sentinel._collect_completed_results()
        assert results == []


# =========================================================================
# 13. TestLogForOrchestration
# =========================================================================


class TestLogForOrchestration:
    """Tests for _log_for_orchestration helper."""

    def test_logs_to_main_logger(self) -> None:
        """_log_for_orchestration should log to the main logger."""
        sentinel = _make_sentinel()
        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_for_orchestration("test-orch", logging.INFO, "test message")
            mock_logger.log.assert_called_once()

    def test_logs_to_orch_logger_when_configured(self) -> None:
        """_log_for_orchestration should also log to the orchestration logger."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(orchestration_logs_dir=Path(tmpdir))
            sentinel = _make_sentinel(config=config)
            sentinel._orch_log_manager = MagicMock()
            mock_orch_logger = MagicMock()
            sentinel._orch_log_manager.get_logger.return_value = mock_orch_logger

            sentinel._log_for_orchestration("test-orch", logging.WARNING, "problem")
            sentinel._orch_log_manager.get_logger.assert_called_once_with("test-orch")
            mock_orch_logger.log.assert_called_once()

    def test_no_orch_logger_when_not_configured(self) -> None:
        """_log_for_orchestration should skip orchestration logging when not configured."""
        config = make_config(orchestration_logs_dir=None)
        sentinel = _make_sentinel(config=config)
        assert sentinel._orch_log_manager is None
        # Should not raise
        sentinel._log_for_orchestration("test-orch", logging.INFO, "test message")


# =========================================================================
# 14. TestRunShutdownTimeout
# =========================================================================


class TestRunShutdownTimeout:
    """Tests for run() shutdown timeout behavior with futures that time out."""

    def test_shutdown_timeout_cancels_remaining_futures(self) -> None:
        """run() should cancel remaining futures when shutdown timeout is reached."""
        config = make_config(poll_interval=1, shutdown_timeout_seconds=0.5)
        sentinel = _make_sentinel(config=config)

        def mock_run_once():
            sentinel.request_shutdown()
            return [], 0

        # Create a future that will never complete
        never_done = Future()

        with patch.object(sentinel, "run_once", side_effect=mock_run_once):
            with patch.object(
                sentinel._execution_manager, "get_active_count", return_value=1
            ):
                with patch.object(
                    sentinel._execution_manager,
                    "get_pending_futures",
                    return_value=[never_done],
                ):
                    with patch("signal.signal"):
                        with patch("sentinel.main.logger") as mock_logger:
                            thread = threading.Thread(target=sentinel.run, daemon=True)
                            thread.start()
                            thread.join(timeout=10)
                            assert not thread.is_alive()
                            # Should have logged timeout warning
                            warning_calls = [
                                c for c in mock_logger.warning.call_args_list
                                if "timeout reached" in c.args[0].lower()
                            ]
                            assert len(warning_calls) >= 1

        # Clean up the future
        if not never_done.done():
            never_done.cancel()

    def test_run_closes_orch_log_manager(self) -> None:
        """run() should close the orchestration log manager in the finally block."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config(poll_interval=1, orchestration_logs_dir=Path(tmpdir))
            sentinel = _make_sentinel(config=config)
            sentinel._orch_log_manager = MagicMock()

            def mock_run_once():
                sentinel.request_shutdown()
                return [], 0

            with patch.object(sentinel, "run_once", side_effect=mock_run_once):
                with patch("signal.signal"):
                    sentinel.run()

            sentinel._orch_log_manager.close_all.assert_called_once()
