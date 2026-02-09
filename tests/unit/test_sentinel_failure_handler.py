"""Tests for Sentinel exception handling helper methods.

This module tests the extracted helper methods that encapsulate common failure
handling logic for execution and submission errors. The helper methods reduce
code duplication by centralizing logging and tag application error handling.

Helper methods tested:
- _apply_failure_tags_safely: Applies failure tags with standardized exception handling
- _handle_execution_failure: Logs errors and applies failure tags for execution failures
- _handle_submission_failure: Handles submission failures with state cleanup
- _log_total_failure: Logs threshold-based warnings when trigger failures exceed limits

Tests use pytest.mark.parametrize for cleaner test organization and reduced
code duplication where appropriate.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from sentinel.main import Sentinel
from sentinel.types import ErrorType
from tests.conftest import (
    MockAgentClient,
    MockJiraPoller,
    MockTagClient,
    make_agent_factory,
    make_config,
    make_orchestration,
)


def _create_sentinel(
    tag_client: MockTagClient | None = None,
) -> tuple[Sentinel, MockTagClient]:
    """Create a Sentinel instance for testing.

    This module-level helper replaces the duplicate ``_create_sentinel``
    methods that were previously defined in each test class.

    Args:
        tag_client: Optional pre-configured tag client.  When ``None``
            (the default) a fresh :class:`MockTagClient` is created.

    Returns:
        A ``(sentinel, tag_client)`` tuple ready for assertions.
    """
    jira_poller = MockJiraPoller(issues=[])
    agent_factory, _ = make_agent_factory()
    if tag_client is None:
        tag_client = MockTagClient()
    config = make_config(poll_interval=1)
    orchestrations = [make_orchestration()]

    sentinel = Sentinel(
        config=config,
        orchestrations=orchestrations,
        jira_poller=jira_poller,
        agent_factory=agent_factory,
        tag_client=tag_client,
    )
    return sentinel, tag_client


class TestLogPartialFailure:
    """Tests for the _log_partial_failure helper method (DS-827).

    This method extracts the duplicated ``logger.warning`` calls for partial
    polling failures in ``run_once()`` into a single reusable helper.
    """

    @pytest.mark.parametrize(
        "service_name,found,errors",
        [
            pytest.param("Jira", 3, 1, id="jira"),
            pytest.param("GitHub", 5, 2, id="github"),
            pytest.param("CustomService", 1, 4, id="custom"),
        ],
    )
    def test_logs_warning_with_correct_format(
        self, service_name: str, found: int, errors: int
    ) -> None:
        """Test that the helper emits a warning with the correct format and trigger counts."""
        sentinel, _ = _create_sentinel()

        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_partial_failure(
                service_name=service_name, found=found, errors=errors
            )

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            format_string = call_args[0][0]
            assert "Partial" in format_string
            assert "polling failure" in format_string
            assert "trigger(s) succeeded" in format_string
            assert "trigger(s) failed" in format_string
            assert call_args[0][1] == service_name
            assert call_args[0][2] == found
            assert call_args[0][3] == errors


class TestLogTotalFailure:
    """Tests for the _log_total_failure helper method (DS-834).

    This method extracts the duplicated threshold-based warning logging for
    Jira and GitHub polling in ``run_once()`` into a single reusable helper.
    """

    @pytest.mark.parametrize(
        "service_name,error_count,trigger_count",
        [
            pytest.param("Jira", 3, 3, id="jira_all_failed"),
            pytest.param("GitHub", 5, 5, id="github_all_failed"),
        ],
    )
    def test_logs_all_triggers_failed_warning(
        self, service_name: str, error_count: int, trigger_count: int
    ) -> None:
        """Test that a warning is logged when all triggers fail.

        Also verifies the message includes a connectivity investigation hint
        (consolidated from test_message_includes_connectivity_hint per DS-859).
        """
        sentinel, _ = _create_sentinel()

        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure(
                service_name=service_name,
                error_count=error_count,
                trigger_count=trigger_count,
                error_threshold=1.0,
            )

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            format_string = call_args[0][0]
            assert "All" in format_string
            assert "failed during this polling cycle" in format_string
            assert "connectivity" in format_string
            assert call_args[0][1] == error_count
            assert call_args[0][2] == service_name
            assert call_args[0][3] == service_name

    @pytest.mark.parametrize(
        "service_name,error_count,trigger_count,threshold",
        [
            pytest.param("Jira", 4, 5, 0.5, id="jira_partial_threshold"),
            pytest.param("GitHub", 3, 4, 0.5, id="github_partial_threshold"),
        ],
    )
    def test_logs_partial_threshold_warning(
        self,
        service_name: str,
        error_count: int,
        trigger_count: int,
        threshold: float,
    ) -> None:
        """Test that a warning is logged when errors exceed threshold but not all fail.

        Also verifies the threshold is formatted as an integer percentage in the
        warning arguments (consolidated from test_threshold_percentage_formatted_correctly
        per DS-859).
        """
        sentinel, _ = _create_sentinel()

        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure(
                service_name=service_name,
                error_count=error_count,
                trigger_count=trigger_count,
                error_threshold=threshold,
            )

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            format_string = call_args[0][0]
            assert "threshold" in format_string
            assert call_args[0][1] == error_count
            assert call_args[0][2] == trigger_count
            assert call_args[0][3] == service_name
            assert call_args[0][4] == int(threshold * 100)
            assert call_args[0][5] == service_name

    @pytest.mark.parametrize(
        "service_name,error_count,trigger_count,threshold",
        [
            pytest.param("Jira", 1, 10, 0.5, id="below_threshold"),
            pytest.param("Jira", 0, 0, 1.0, id="zero_triggers"),
        ],
    )
    def test_no_warning_when_conditions_not_met(
        self,
        service_name: str,
        error_count: int,
        trigger_count: int,
        threshold: float,
    ) -> None:
        """Test that no warning is logged when errors are below threshold or triggers is zero."""
        sentinel, _ = _create_sentinel()

        with patch("sentinel.main.logger") as mock_logger:
            sentinel._log_total_failure(
                service_name=service_name,
                error_count=error_count,
                trigger_count=trigger_count,
                error_threshold=threshold,
            )

            mock_logger.warning.assert_not_called()


class TestRecordPollHealth:
    """Tests for the _record_poll_health helper method (DS-835).

    This method extracts the duplicated health-gate recording blocks for
    Jira and GitHub polling in ``run_once()`` into a single reusable helper.
    """

    @pytest.mark.parametrize(
        "service_name,display_name",
        [
            pytest.param("jira", "Jira", id="jira"),
            pytest.param("github", "GitHub", id="github"),
        ],
    )
    def test_records_success_when_no_errors(
        self, service_name: str, display_name: str
    ) -> None:
        """Test that record_poll_success is called when error_count is zero."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel._health_gate, "record_poll_success") as mock_success:
            sentinel._record_poll_health(
                service_name=service_name,
                display_name=display_name,
                error_count=0,
                issues_found=5,
            )

            mock_success.assert_called_once_with(service_name)

    @pytest.mark.parametrize(
        "service_name,display_name",
        [
            pytest.param("jira", "Jira", id="jira"),
            pytest.param("github", "GitHub", id="github"),
        ],
    )
    def test_records_failure_when_all_triggers_failed(
        self, service_name: str, display_name: str
    ) -> None:
        """Test that record_poll_failure is called when errors > 0 and no issues found."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel._health_gate, "record_poll_failure") as mock_failure:
            sentinel._record_poll_health(
                service_name=service_name,
                display_name=display_name,
                error_count=3,
                issues_found=0,
            )

            mock_failure.assert_called_once_with(service_name)

    @pytest.mark.parametrize(
        "service_name,display_name",
        [
            pytest.param("jira", "Jira", id="jira"),
            pytest.param("github", "GitHub", id="github"),
        ],
    )
    def test_logs_partial_failure_when_mixed_results(
        self, service_name: str, display_name: str
    ) -> None:
        """Test that _log_partial_failure is called when errors > 0 and issues > 0."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel, "_log_partial_failure") as mock_partial:
            sentinel._record_poll_health(
                service_name=service_name,
                display_name=display_name,
                error_count=2,
                issues_found=3,
            )

            mock_partial.assert_called_once_with(
                service_name=display_name, found=3, errors=2
            )

    def test_no_failure_recorded_for_partial_success(self) -> None:
        """Test that record_poll_failure is NOT called when some issues were found."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel._health_gate, "record_poll_failure") as mock_failure:
            with patch.object(sentinel, "_log_partial_failure"):
                sentinel._record_poll_health(
                    service_name="jira",
                    display_name="Jira",
                    error_count=1,
                    issues_found=2,
                )

                mock_failure.assert_not_called()

    def test_no_success_recorded_when_errors_present(self) -> None:
        """Test that record_poll_success is NOT called when errors > 0."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel._health_gate, "record_poll_success") as mock_success:
            with patch.object(sentinel._health_gate, "record_poll_failure"):
                sentinel._record_poll_health(
                    service_name="github",
                    display_name="GitHub",
                    error_count=3,
                    issues_found=0,
                )

                mock_success.assert_not_called()

    def test_uses_display_name_for_partial_failure_logging(self) -> None:
        """Test that the display_name (not service_name) is passed to _log_partial_failure."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel, "_log_partial_failure") as mock_partial:
            sentinel._record_poll_health(
                service_name="github",
                display_name="GitHub",
                error_count=1,
                issues_found=4,
            )

            mock_partial.assert_called_once_with(
                service_name="GitHub", found=4, errors=1
            )

    def test_uses_service_name_for_health_gate_calls(self) -> None:
        """Test that the service_name (not display_name) is used for health gate calls."""
        sentinel, _ = _create_sentinel()

        with patch.object(sentinel._health_gate, "record_poll_success") as mock_success:
            sentinel._record_poll_health(
                service_name="jira",
                display_name="Jira",
                error_count=0,
                issues_found=1,
            )

            mock_success.assert_called_once_with("jira")


class TestHandleExecutionFailure:
    """Tests for the _handle_execution_failure helper method."""

    def test_logs_error_with_correct_format(self) -> None:
        """Test that the helper logs errors with the correct format."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = OSError("Connection refused")

        with patch.object(sentinel, "_log_for_orchestration") as mock_log:
            sentinel._handle_execution_failure(
                issue_key="TEST-123",
                orchestration=orchestration,
                exception=exception,
                error_type=ErrorType.IO_ERROR,
            )

        mock_log.assert_called_once_with(
            "test-orch",
            logging.ERROR,
            "Failed to execute 'test-orch' for TEST-123 due to I/O error: Connection refused",
        )

    def test_applies_failure_tags(self) -> None:
        """Test that the helper calls apply_failure_tags on the tag manager."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Unexpected error")

        with patch.object(sentinel.tag_manager, "apply_failure_tags") as mock_apply:
            sentinel._handle_execution_failure(
                issue_key="TEST-123",
                orchestration=orchestration,
                exception=exception,
                error_type=ErrorType.RUNTIME_ERROR,
            )

            # Verify apply_failure_tags was called with correct arguments
            mock_apply.assert_called_once_with("TEST-123", orchestration)

    @pytest.mark.parametrize(
        "tag_exception,expected_error_type",
        [
            pytest.param(
                OSError("Network error"),
                ErrorType.IO_ERROR,
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Request timed out"),
                ErrorType.IO_ERROR,
                id="timeout_error",
            ),
            pytest.param(
                KeyError("issue_key"),
                ErrorType.DATA_ERROR,
                id="keyerror",
            ),
            pytest.param(
                ValueError("Invalid value"),
                ErrorType.DATA_ERROR,
                id="valueerror",
            ),
        ],
    )
    def test_handles_tag_application_errors(
        self, tag_exception: Exception, expected_error_type: ErrorType
    ) -> None:
        """Test that various exceptions during tag application are caught and logged.

        This parametrized test verifies that OSError, TimeoutError, KeyError, and
        ValueError exceptions raised during apply_failure_tags are properly caught,
        don't propagate, and are logged with the appropriate ErrorType enum value.
        """
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Original error")

        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=tag_exception,
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._handle_execution_failure(
                    issue_key="TEST-123",
                    orchestration=orchestration,
                    exception=exception,
                    error_type=ErrorType.RUNTIME_ERROR,
                )

                # Verify the tag error was logged with correct ErrorType
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                # Check format string and ErrorType.value argument
                assert "Failed to apply failure tags due to %s" in call_args[0][0]
                assert expected_error_type.value in call_args[0]

    @pytest.mark.parametrize(
        "exception,error_type,expected_substring",
        [
            pytest.param(
                OSError("File not found"),
                ErrorType.IO_ERROR,
                "I/O error: File not found",
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Timed out"),
                ErrorType.IO_ERROR,
                "I/O error: Timed out",
                id="timeout_error",
            ),
            pytest.param(
                RuntimeError("Bad state"),
                ErrorType.RUNTIME_ERROR,
                "runtime error: Bad state",
                id="runtime_error",
            ),
            pytest.param(
                KeyError("missing"),
                ErrorType.DATA_ERROR,
                "data error: 'missing'",
                id="keyerror",
            ),
            pytest.param(
                ValueError("invalid"),
                ErrorType.DATA_ERROR,
                "data error: invalid",
                id="valueerror",
            ),
        ],
    )
    def test_different_error_types_logged_correctly(
        self, exception: Exception, error_type: ErrorType, expected_substring: str
    ) -> None:
        """Test that different error types are logged with correct descriptors.

        This parametrized test verifies that various exception types are logged
        with the appropriate ErrorType enum values, ensuring the log message
        contains the correct error type descriptor.
        """
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch.object(sentinel, "_log_for_orchestration") as mock_log:
            sentinel._handle_execution_failure(
                issue_key="TEST-333",
                orchestration=orchestration,
                exception=exception,
                error_type=error_type,
            )

            # Check that the logged message contains the expected substring
            call_args = mock_log.call_args
            assert expected_substring in call_args[0][2], (
                f"Expected '{expected_substring}' in log message for {type(exception).__name__}"
            )

    def test_extra_context_included_in_tag_error_logs(self) -> None:
        """Test that extra context (issue_key, orchestration) is included in tag error logs."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="my-orchestration")
        exception = OSError("Test error")

        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=OSError("Tag error"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._handle_execution_failure(
                    issue_key="PROJ-999",
                    orchestration=orchestration,
                    exception=exception,
                    error_type=ErrorType.IO_ERROR,
                )

                # Verify extra context was passed
                call_args = mock_logger.error.call_args
                extra = call_args[1].get("extra", {})
                assert extra.get("issue_key") == "PROJ-999"
                assert extra.get("orchestration") == "my-orchestration"


class TestExecuteOrchestrationTaskUsesHelper:
    """Tests to verify _execute_orchestration_task uses the helper method correctly."""

    @pytest.mark.parametrize(
        "exception,expected_error_type,issue_key",
        [
            pytest.param(
                OSError("Connection refused"),
                ErrorType.IO_ERROR,
                "TEST-123",
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Timed out"),
                ErrorType.IO_ERROR,
                "TEST-234",
                id="timeout_error",
            ),
            pytest.param(
                RuntimeError("Bad state"),
                ErrorType.RUNTIME_ERROR,
                "TEST-456",
                id="runtime_error",
            ),
            pytest.param(
                KeyError("missing_key"),
                ErrorType.DATA_ERROR,
                "TEST-789",
                id="keyerror",
            ),
            pytest.param(
                ValueError("Invalid value"),
                ErrorType.DATA_ERROR,
                "TEST-111",
                id="valueerror",
            ),
        ],
    )
    def test_exception_calls_helper_with_correct_error_type(
        self, exception: Exception, expected_error_type: ErrorType, issue_key: str
    ) -> None:
        """Test that exceptions in _execute_orchestration_task call _handle_execution_failure.

        This parametrized test verifies that various exception types (OSError,
        TimeoutError, RuntimeError, KeyError, ValueError) raised during execution
        properly trigger _handle_execution_failure with the correct ErrorType enum value.
        """
        # Patch MockAgentClient.run_agent before creating Sentinel
        # so the exception is raised during execution
        with patch.object(MockAgentClient, "run_agent", side_effect=exception):
            sentinel, _ = _create_sentinel()
            orchestration = make_orchestration(name="test-orch")

            mock_issue = MagicMock()
            mock_issue.key = issue_key

            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == issue_key
                assert call_args[0][1] == orchestration
                assert isinstance(call_args[0][2], type(exception))
                assert call_args[0][3] == expected_error_type


class TestApplyFailureTagsSafely:
    """Tests for the _apply_failure_tags_safely helper method."""

    def test_applies_failure_tags_successfully(self) -> None:
        """Test that the helper applies failure tags when no errors occur."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch.object(sentinel.tag_manager, "apply_failure_tags") as mock_apply:
            sentinel._apply_failure_tags_safely("TEST-123", orchestration)

            mock_apply.assert_called_once_with("TEST-123", orchestration)

    @pytest.mark.parametrize(
        "tag_exception,expected_error_type",
        [
            pytest.param(
                OSError("Network error"),
                ErrorType.IO_ERROR,
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Request timed out"),
                ErrorType.IO_ERROR,
                id="timeout_error",
            ),
            pytest.param(
                KeyError("issue_key"),
                ErrorType.DATA_ERROR,
                id="keyerror",
            ),
            pytest.param(
                ValueError("Invalid value"),
                ErrorType.DATA_ERROR,
                id="valueerror",
            ),
        ],
    )
    def test_handles_tag_application_errors(
        self, tag_exception: Exception, expected_error_type: ErrorType
    ) -> None:
        """Test that exceptions during tag application are caught and logged.

        This parametrized test verifies that OSError, TimeoutError, KeyError, and
        ValueError exceptions raised during apply_failure_tags are properly caught,
        don't propagate, and are logged with the appropriate ErrorType enum value.
        """
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=tag_exception,
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._apply_failure_tags_safely("TEST-123", orchestration)

                # Verify the tag error was logged with correct ErrorType
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                # Check format string and ErrorType.value argument
                assert "Failed to apply failure tags due to %s" in call_args[0][0]
                assert expected_error_type.value in call_args[0]

    def test_extra_context_included_in_error_logs(self) -> None:
        """Test that extra context (issue_key, orchestration) is included in error logs."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="my-orchestration")

        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=OSError("Tag error"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._apply_failure_tags_safely("PROJ-999", orchestration)

                # Verify extra context was passed
                call_args = mock_logger.error.call_args
                extra = call_args[1].get("extra", {})
                assert extra.get("issue_key") == "PROJ-999"
                assert extra.get("orchestration") == "my-orchestration"


class TestHandleSubmissionFailure:
    """Tests for the _handle_submission_failure helper method."""

    def test_decrements_version_executions_when_version_provided(self) -> None:
        """Test that version.decrement_executions is called when version is not None."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        mock_version = MagicMock()
        exception = OSError("Connection error")

        with patch.object(sentinel, "_apply_failure_tags_safely"):
            sentinel._handle_submission_failure(
                issue_key="TEST-123",
                orchestration=orchestration,
                exception=exception,
                error_type=ErrorType.IO_ERROR,
                version=mock_version,
            )

        mock_version.decrement_executions.assert_called_once()

    def test_skips_version_decrement_when_version_is_none(self) -> None:
        """Test that no error occurs when version is None."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = OSError("Connection error")

        with patch.object(sentinel, "_apply_failure_tags_safely"):
            # Should not raise
            sentinel._handle_submission_failure(
                issue_key="TEST-123",
                orchestration=orchestration,
                exception=exception,
                error_type=ErrorType.IO_ERROR,
                version=None,
            )

    def test_decrements_per_orch_count(self) -> None:
        """Test that per-orchestration count is decremented."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Test error")

        with patch.object(sentinel._state_tracker, "decrement_per_orch_count") as mock_decrement:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    issue_key="TEST-123",
                    orchestration=orchestration,
                    exception=exception,
                    error_type=ErrorType.RUNTIME_ERROR,
                    version=None,
                )

        mock_decrement.assert_called_once_with("test-orch")

    @pytest.mark.parametrize(
        "exception,error_type,expected_error_desc",
        [
            pytest.param(
                OSError("File not found"),
                ErrorType.IO_ERROR,
                "I/O error",
                id="io_error",
            ),
            pytest.param(
                RuntimeError("Bad state"),
                ErrorType.RUNTIME_ERROR,
                "runtime error",
                id="runtime_error",
            ),
            pytest.param(
                KeyError("missing"),
                ErrorType.DATA_ERROR,
                "data error",
                id="data_error",
            ),
        ],
    )
    def test_logs_error_with_correct_format(
        self, exception: Exception, error_type: ErrorType, expected_error_desc: str
    ) -> None:
        """Test that errors are logged with correct format including error type."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch("sentinel.main.logger") as mock_logger:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    issue_key="TEST-123",
                    orchestration=orchestration,
                    exception=exception,
                    error_type=error_type,
                    version=None,
                )

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        # Check the format string includes the error type placeholder
        assert "Failed to submit" in call_args[0][0]
        assert "%s" in call_args[0][0]  # Has placeholders for lazy formatting
        # Check the arguments passed for lazy formatting
        assert "test-orch" in call_args[0]
        assert "TEST-123" in call_args[0]
        assert expected_error_desc in call_args[0]

    def test_applies_failure_tags_safely(self) -> None:
        """Test that _apply_failure_tags_safely is called."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = ValueError("Invalid data")

        with patch.object(sentinel, "_apply_failure_tags_safely") as mock_apply:
            sentinel._handle_submission_failure(
                issue_key="TEST-456",
                orchestration=orchestration,
                exception=exception,
                error_type=ErrorType.DATA_ERROR,
                version=None,
            )

        mock_apply.assert_called_once_with("TEST-456", orchestration)

    def test_extra_context_in_log(self) -> None:
        """Test that extra context is included in log call."""
        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="my-orchestration")
        exception = OSError("Test error")

        with patch("sentinel.main.logger") as mock_logger:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    issue_key="PROJ-999",
                    orchestration=orchestration,
                    exception=exception,
                    error_type=ErrorType.IO_ERROR,
                    version=None,
                )

        call_args = mock_logger.error.call_args
        extra = call_args[1].get("extra", {})
        assert extra.get("issue_key") == "PROJ-999"
        assert extra.get("orchestration") == "my-orchestration"


class TestClaudeProcessInterruptedUsesHelper:
    """Tests to verify ClaudeProcessInterruptedError handler uses _apply_failure_tags_safely."""

    def test_claude_interrupted_calls_apply_failure_tags_safely(self) -> None:
        """Test that ClaudeProcessInterruptedError uses _apply_failure_tags_safely."""
        from sentinel.sdk_clients import ClaudeProcessInterruptedError

        # Patch MockAgentClient.run_agent before creating Sentinel
        with patch.object(
            MockAgentClient, "run_agent", side_effect=ClaudeProcessInterruptedError()
        ):
            sentinel, _ = _create_sentinel()
            orchestration = make_orchestration(name="test-orch")

            mock_issue = MagicMock()
            mock_issue.key = "TEST-999"

            with patch.object(sentinel, "_apply_failure_tags_safely") as mock_apply:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_apply.assert_called_once_with("TEST-999", orchestration)


class TestSubmitExecutionTasksUsesHelper:
    """Tests to verify _submit_execution_tasks uses _handle_submission_failure."""

    @pytest.mark.parametrize(
        "exception,expected_error_type",
        [
            pytest.param(
                OSError("Connection refused"),
                ErrorType.IO_ERROR,
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Timed out"),
                ErrorType.IO_ERROR,
                id="timeout_error",
            ),
            pytest.param(
                RuntimeError("Bad state"),
                ErrorType.RUNTIME_ERROR,
                id="runtime_error",
            ),
            pytest.param(
                KeyError("missing_key"),
                ErrorType.DATA_ERROR,
                id="keyerror",
            ),
            pytest.param(
                ValueError("Invalid value"),
                ErrorType.DATA_ERROR,
                id="valueerror",
            ),
        ],
    )
    def test_submission_exception_calls_helper(
        self, exception: Exception, expected_error_type: ErrorType
    ) -> None:
        """Test that exceptions in _submit_execution_tasks call _handle_submission_failure.

        This parametrized test verifies that various exception types raised during
        submission properly trigger _handle_submission_failure with the correct ErrorType.
        """
        from sentinel.router import RoutingResult

        sentinel, _ = _create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        mock_issue = MagicMock()
        mock_issue.key = "TEST-123"

        routing_result = RoutingResult(issue=mock_issue, orchestrations=[orchestration])

        # Make tag_manager.start_processing raise the exception
        with patch.object(sentinel.tag_manager, "start_processing", side_effect=exception):
            with patch.object(sentinel, "_handle_submission_failure") as mock_handler:
                with patch.object(
                    sentinel, "_get_available_slots_for_orchestration", return_value=1
                ):
                    sentinel._submit_execution_tasks([routing_result], [])

                    mock_handler.assert_called_once()
                    call_args = mock_handler.call_args
                    assert call_args[0][0] == "TEST-123"
                    assert call_args[0][1] == orchestration
                    assert isinstance(call_args[0][2], type(exception))
                    assert call_args[0][3] == expected_error_type
