"""Tests for Sentinel._handle_execution_failure helper method.

This module tests the extracted helper method that encapsulates the common failure
handling logic for execution errors. The helper method reduces code duplication by
centralizing logging and tag application error handling.

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
        agent_client = MockAgentClient()
        if tag_client is None:
            tag_client = MockTagClient()
        config = make_config(poll_interval=1)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_client,
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
        "tag_exception,expected_log_category",
        [
            pytest.param(
                OSError("Network error"),
                "I/O error",
                id="oserror",
            ),
            pytest.param(
                TimeoutError("Request timed out"),
                "I/O error",
                id="timeout_error",
            ),
            pytest.param(
                KeyError("issue_key"),
                "data error",
                id="keyerror",
            ),
            pytest.param(
                ValueError("Invalid value"),
                "data error",
                id="valueerror",
            ),
        ],
    )
    def test_handles_tag_application_errors(
        self, tag_exception: Exception, expected_log_category: str
    ) -> None:
        """Test that various exceptions during tag application are caught and logged.

        This parametrized test verifies that OSError, TimeoutError, KeyError, and
        ValueError exceptions raised during apply_failure_tags are properly caught,
        don't propagate, and are logged with the appropriate error category.
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

                # Verify the tag error was logged with correct category
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert f"Failed to apply failure tags due to {expected_log_category}" in call_args[0][0]
                # With lazy % formatting, the error is passed as an argument
                assert str(tag_exception) in str(call_args[0][1]) or str(tag_exception.args[0]) in str(call_args[0][1])

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
        agent_client = MockAgentClient()
        tag_client = MockTagClient()
        config = make_config(poll_interval=1)
        orchestrations = [make_orchestration()]

        sentinel = Sentinel(
            config=config,
            orchestrations=orchestrations,
            jira_poller=jira_poller,
            agent_factory=agent_client,
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
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        mock_issue = MagicMock()
        mock_issue.key = issue_key

        with patch.object(sentinel.executor, "execute", side_effect=exception):
            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == issue_key
                assert call_args[0][1] == orchestration
                assert isinstance(call_args[0][2], type(exception))
                assert call_args[0][3] == expected_error_type
