"""Tests for Sentinel exception handling helper methods.

This module tests the extracted helper methods that encapsulate common failure
handling logic for execution and submission errors. The helper methods reduce
code duplication by centralizing logging and tag application error handling.

Helper methods tested:
- _apply_failure_tags_safely: Applies failure tags with standardized exception handling
- _handle_execution_failure: Logs errors and applies failure tags for execution failures
- _handle_submission_failure: Handles submission failures with state cleanup

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


class TestHandleExecutionFailure:
    """Tests for the _handle_execution_failure helper method."""

    def _create_sentinel(
        self, tag_client: MockTagClient | None = None
    ) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
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

    def test_logs_error_with_correct_format(self) -> None:
        """Test that the helper logs errors with the correct format."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = OSError("Connection refused")

        with patch.object(sentinel, "_log_for_orchestration") as mock_log:
            sentinel._handle_execution_failure(
                "TEST-123", orchestration, exception, ErrorType.IO_ERROR
            )

        mock_log.assert_called_once_with(
            "test-orch",
            logging.ERROR,
            "Failed to execute 'test-orch' for TEST-123 due to I/O error: Connection refused",
        )

    def test_applies_failure_tags(self) -> None:
        """Test that the helper calls apply_failure_tags on the tag manager."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Unexpected error")

        with patch.object(sentinel.tag_manager, "apply_failure_tags") as mock_apply:
            sentinel._handle_execution_failure(
                "TEST-123", orchestration, exception, ErrorType.RUNTIME_ERROR
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
        sentinel, _ = self._create_sentinel()
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
                    "TEST-123", orchestration, exception, ErrorType.RUNTIME_ERROR
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
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch.object(sentinel, "_log_for_orchestration") as mock_log:
            sentinel._handle_execution_failure(
                "TEST-333", orchestration, exception, error_type
            )

            # Check that the logged message contains the expected substring
            call_args = mock_log.call_args
            assert expected_substring in call_args[0][2], (
                f"Expected '{expected_substring}' in log message for {type(exception).__name__}"
            )

    def test_extra_context_included_in_tag_error_logs(self) -> None:
        """Test that extra context (issue_key, orchestration) is included in tag error logs."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="my-orchestration")
        exception = OSError("Test error")

        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=OSError("Tag error"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                sentinel._handle_execution_failure(
                    "PROJ-999", orchestration, exception, ErrorType.IO_ERROR
                )

                # Verify extra context was passed
                call_args = mock_logger.error.call_args
                extra = call_args[1].get("extra", {})
                assert extra.get("issue_key") == "PROJ-999"
                assert extra.get("orchestration") == "my-orchestration"


class TestExecuteOrchestrationTaskUsesHelper:
    """Tests to verify _execute_orchestration_task uses the helper method correctly."""

    def _create_sentinel(self) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
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
            sentinel, _ = self._create_sentinel()
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

    def _create_sentinel(
        self, tag_client: MockTagClient | None = None
    ) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
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

    def test_applies_failure_tags_successfully(self) -> None:
        """Test that the helper applies failure tags when no errors occur."""
        sentinel, _ = self._create_sentinel()
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
        sentinel, _ = self._create_sentinel()
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
        sentinel, _ = self._create_sentinel()
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

    def _create_sentinel(
        self, tag_client: MockTagClient | None = None
    ) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
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

    def test_decrements_version_executions_when_version_provided(self) -> None:
        """Test that version.decrement_executions is called when version is not None."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        mock_version = MagicMock()
        exception = OSError("Connection error")

        with patch.object(sentinel, "_apply_failure_tags_safely"):
            sentinel._handle_submission_failure(
                "TEST-123", orchestration, exception, ErrorType.IO_ERROR, mock_version
            )

        mock_version.decrement_executions.assert_called_once()

    def test_skips_version_decrement_when_version_is_none(self) -> None:
        """Test that no error occurs when version is None."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = OSError("Connection error")

        with patch.object(sentinel, "_apply_failure_tags_safely"):
            # Should not raise
            sentinel._handle_submission_failure(
                "TEST-123", orchestration, exception, ErrorType.IO_ERROR, None
            )

    def test_decrements_per_orch_count(self) -> None:
        """Test that per-orchestration count is decremented."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Test error")

        with patch.object(sentinel._state_tracker, "decrement_per_orch_count") as mock_decrement:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    "TEST-123", orchestration, exception, ErrorType.RUNTIME_ERROR, None
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
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        with patch("sentinel.main.logger") as mock_logger:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    "TEST-123", orchestration, exception, error_type, None
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
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = ValueError("Invalid data")

        with patch.object(sentinel, "_apply_failure_tags_safely") as mock_apply:
            sentinel._handle_submission_failure(
                "TEST-456", orchestration, exception, ErrorType.DATA_ERROR, None
            )

        mock_apply.assert_called_once_with("TEST-456", orchestration)

    def test_extra_context_in_log(self) -> None:
        """Test that extra context is included in log call."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="my-orchestration")
        exception = OSError("Test error")

        with patch("sentinel.main.logger") as mock_logger:
            with patch.object(sentinel, "_apply_failure_tags_safely"):
                sentinel._handle_submission_failure(
                    "PROJ-999", orchestration, exception, ErrorType.IO_ERROR, None
                )

        call_args = mock_logger.error.call_args
        extra = call_args[1].get("extra", {})
        assert extra.get("issue_key") == "PROJ-999"
        assert extra.get("orchestration") == "my-orchestration"


class TestClaudeProcessInterruptedUsesHelper:
    """Tests to verify ClaudeProcessInterruptedError handler uses _apply_failure_tags_safely."""

    def _create_sentinel(self) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
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

    def test_claude_interrupted_calls_apply_failure_tags_safely(self) -> None:
        """Test that ClaudeProcessInterruptedError uses _apply_failure_tags_safely."""
        from sentinel.sdk_clients import ClaudeProcessInterruptedError

        # Patch MockAgentClient.run_agent before creating Sentinel
        with patch.object(
            MockAgentClient, "run_agent", side_effect=ClaudeProcessInterruptedError()
        ):
            sentinel, _ = self._create_sentinel()
            orchestration = make_orchestration(name="test-orch")

            mock_issue = MagicMock()
            mock_issue.key = "TEST-999"

            with patch.object(sentinel, "_apply_failure_tags_safely") as mock_apply:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_apply.assert_called_once_with("TEST-999", orchestration)


class TestSubmitExecutionTasksUsesHelper:
    """Tests to verify _submit_execution_tasks uses _handle_submission_failure."""

    def _create_sentinel(self) -> tuple[Sentinel, MockTagClient]:
        """Create a Sentinel instance for testing."""
        jira_poller = MockJiraPoller(issues=[])
        agent_factory, _ = make_agent_factory()
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

        sentinel, _ = self._create_sentinel()
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
