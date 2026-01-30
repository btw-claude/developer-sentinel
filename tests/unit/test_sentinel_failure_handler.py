"""Tests for Sentinel._handle_execution_failure helper method.

This module tests the extracted helper method that encapsulates the common failure
handling logic for execution errors. The helper method reduces code duplication by
centralizing logging and tag application error handling.
"""

import logging
from unittest.mock import MagicMock, patch

from sentinel.main import Sentinel
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
                "TEST-123", orchestration, exception, "I/O error"
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
                "TEST-123", orchestration, exception, "runtime error"
            )

            # Verify apply_failure_tags was called with correct arguments
            mock_apply.assert_called_once_with("TEST-123", orchestration)

    def test_handles_oserror_during_tag_application(self) -> None:
        """Test that OSError during tag application is caught and logged."""
        sentinel, tag_client = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = ValueError("Invalid data")

        # Make tag_manager.apply_failure_tags raise OSError
        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=OSError("Network error"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._handle_execution_failure(
                    "TEST-456", orchestration, exception, "data error"
                )

                # Verify the tag error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert "Failed to apply failure tags due to I/O error" in call_args[0][0]
                # With lazy % formatting, the error is passed as an argument
                assert "Network error" in str(call_args[0][1])

    def test_handles_timeout_error_during_tag_application(self) -> None:
        """Test that TimeoutError during tag application is caught and logged."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = KeyError("missing_key")

        # Make tag_manager.apply_failure_tags raise TimeoutError
        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=TimeoutError("Request timed out"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._handle_execution_failure(
                    "TEST-789", orchestration, exception, "data error"
                )

                # Verify the tag error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert "Failed to apply failure tags due to I/O error" in call_args[0][0]
                # With lazy % formatting, the error is passed as an argument
                assert "Request timed out" in str(call_args[0][1])

    def test_handles_keyerror_during_tag_application(self) -> None:
        """Test that KeyError during tag application is caught and logged."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = OSError("Disk full")

        # Make tag_manager.apply_failure_tags raise KeyError
        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=KeyError("issue_key"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._handle_execution_failure(
                    "TEST-111", orchestration, exception, "I/O error"
                )

                # Verify the tag error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert "Failed to apply failure tags due to data error" in call_args[0][0]

    def test_handles_valueerror_during_tag_application(self) -> None:
        """Test that ValueError during tag application is caught and logged."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")
        exception = RuntimeError("Runtime issue")

        # Make tag_manager.apply_failure_tags raise ValueError
        with patch.object(
            sentinel.tag_manager,
            "apply_failure_tags",
            side_effect=ValueError("Invalid value"),
        ):
            with patch("sentinel.main.logger") as mock_logger:
                # Should not raise
                sentinel._handle_execution_failure(
                    "TEST-222", orchestration, exception, "runtime error"
                )

                # Verify the tag error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args
                assert "Failed to apply failure tags due to data error" in call_args[0][0]
                # With lazy % formatting, the error is passed as an argument
                assert "Invalid value" in str(call_args[0][1])

    def test_different_error_types_logged_correctly(self) -> None:
        """Test that different error types are logged with correct descriptors."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        test_cases = [
            (OSError("File not found"), "I/O error", "I/O error: File not found"),
            (TimeoutError("Timed out"), "I/O error", "I/O error: Timed out"),
            (RuntimeError("Bad state"), "runtime error", "runtime error: Bad state"),
            (KeyError("missing"), "data error", "data error: 'missing'"),
            (ValueError("invalid"), "data error", "data error: invalid"),
        ]

        for exception, error_type, expected_substring in test_cases:
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
                    "PROJ-999", orchestration, exception, "I/O error"
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

    def test_oserror_calls_helper(self) -> None:
        """Test that OSError in _execute_orchestration_task calls _handle_execution_failure."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        # Create a mock issue
        mock_issue = MagicMock()
        mock_issue.key = "TEST-123"

        # Make executor.execute raise OSError
        with patch.object(
            sentinel.executor, "execute", side_effect=OSError("Connection refused")
        ):
            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == "TEST-123"
                assert call_args[0][1] == orchestration
                assert isinstance(call_args[0][2], OSError)
                assert call_args[0][3] == "I/O error"

    def test_runtimeerror_calls_helper(self) -> None:
        """Test that RuntimeError in _execute_orchestration_task calls _handle_execution_failure."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        mock_issue = MagicMock()
        mock_issue.key = "TEST-456"

        with patch.object(
            sentinel.executor, "execute", side_effect=RuntimeError("Bad state")
        ):
            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == "TEST-456"
                assert call_args[0][3] == "runtime error"

    def test_keyerror_calls_helper(self) -> None:
        """Test that KeyError in _execute_orchestration_task calls _handle_execution_failure."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        mock_issue = MagicMock()
        mock_issue.key = "TEST-789"

        with patch.object(
            sentinel.executor, "execute", side_effect=KeyError("missing_key")
        ):
            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == "TEST-789"
                assert call_args[0][3] == "data error"

    def test_valueerror_calls_helper(self) -> None:
        """Test that ValueError in _execute_orchestration_task calls _handle_execution_failure."""
        sentinel, _ = self._create_sentinel()
        orchestration = make_orchestration(name="test-orch")

        mock_issue = MagicMock()
        mock_issue.key = "TEST-111"

        with patch.object(
            sentinel.executor, "execute", side_effect=ValueError("Invalid value")
        ):
            with patch.object(sentinel, "_handle_execution_failure") as mock_handler:
                result = sentinel._execute_orchestration_task(mock_issue, orchestration)

                assert result is None
                mock_handler.assert_called_once()
                call_args = mock_handler.call_args
                assert call_args[0][0] == "TEST-111"
                assert call_args[0][3] == "data error"
