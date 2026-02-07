"""Tests for Claude Agent SDK client implementations."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.agent_clients.base import AgentClientError, AgentTimeoutError, UsageInfo
from sentinel.agent_clients.claude_sdk import (
    _extract_usage_from_message,
    request_shutdown,
    reset_shutdown,
)
from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from sentinel.config import Config
from sentinel.poller import JiraClientError
from sentinel.sdk_clients import (
    ClaudeProcessInterruptedError,
    ClaudeSdkAgentClient,
    JiraSdkClient,
    JiraSdkTagClient,
    ShutdownController,
)
from sentinel.tag_manager import JiraTagClientError
from tests.helpers import make_config


@pytest.fixture(autouse=True)
def reset_shutdown_state() -> None:
    """Reset the shutdown state before each test."""
    reset_shutdown()


@pytest.fixture
def mock_config() -> Config:
    """Create a mock config for testing.

    Uses the centralized make_config() helper from tests/helpers.py.
    """
    return make_config()


class MockMessage:
    """Mock message from the Claude Agent SDK query."""

    def __init__(self, text: str) -> None:
        self.text = text


class MockResultMessage:
    """Mock ResultMessage from the Claude Agent SDK with usage data.

    This simulates the ResultMessage that is yielded at the end of a query
    containing token usage and cost data.
    """

    def __init__(
        self,
        total_cost_usd: float = 0.05,
        duration_ms: float = 1500.0,
        duration_api_ms: float = 1200.0,
        num_turns: int = 3,
        input_tokens: int = 1000,
        output_tokens: int = 500,
    ) -> None:
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
        self.duration_api_ms = duration_api_ms
        self.num_turns = num_turns
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


def create_mock_query(return_value: str) -> MagicMock:
    """Create a mock query function that returns messages with the given text."""

    async def async_gen() -> AsyncIterator[MockMessage]:
        yield MockMessage(return_value)

    # Return a regular MagicMock that returns the async generator when called
    mock = MagicMock(return_value=async_gen())
    return mock


def create_mock_query_with_usage(
    return_value: str,
    total_cost_usd: float = 0.05,
    duration_ms: float = 1500.0,
    duration_api_ms: float = 1200.0,
    num_turns: int = 3,
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> MagicMock:
    """Create a mock query that returns text message followed by ResultMessage with usage data."""

    async def async_gen() -> AsyncIterator[MockMessage | MockResultMessage]:
        yield MockMessage(return_value)
        yield MockResultMessage(
            total_cost_usd=total_cost_usd,
            duration_ms=duration_ms,
            duration_api_ms=duration_api_ms,
            num_turns=num_turns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    mock = MagicMock(return_value=async_gen())
    return mock


def create_raising_mock_query(exception: Exception) -> MagicMock:
    """Create a mock query function that raises an exception when iterated."""

    async def async_gen() -> AsyncIterator[MockMessage]:
        raise exception
        yield  # Make it a generator  # noqa: B027

    mock = MagicMock(return_value=async_gen())
    return mock


def create_capturing_mock_query(capture_list: list[str]) -> MagicMock:
    """Create a mock query function that captures the prompt and returns 'done'."""

    def make_gen(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
        async def async_gen() -> AsyncIterator[MockMessage]:
            capture_list.append(prompt)
            yield MockMessage("done")

        return async_gen()

    mock = MagicMock(side_effect=make_gen)
    return mock


class TestJiraSdkClient:
    """Tests for JiraSdkClient."""

    def test_search_issues_returns_list(self, mock_config: Config) -> None:
        """Should return list of issues from search."""
        issues_json = json.dumps(
            [
                {"key": "TEST-1", "fields": {"summary": "Issue 1"}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2"}},
            ]
        )

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query(issues_json)):
            client = JiraSdkClient(mock_config)
            result = client.search_issues("project = TEST", max_results=10)

        assert len(result) == 2
        assert result[0]["key"] == "TEST-1"
        assert result[1]["key"] == "TEST-2"

    def test_search_issues_handles_markdown_json_block(self, mock_config: Config) -> None:
        """Should extract JSON from markdown code block."""
        response = """```json
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query(response)):
            client = JiraSdkClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_generic_code_block(self, mock_config: Config) -> None:
        """Should extract JSON from generic markdown code block."""
        response = """```
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query(response)):
            client = JiraSdkClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_wraps_single_issue(self, mock_config: Config) -> None:
        """Should wrap single issue object in list."""
        response = '{"key": "TEST-1", "fields": {"summary": "Issue"}}'

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query(response)):
            client = JiraSdkClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(TimeoutError()),
        ):
            client = JiraSdkClient(mock_config)
            with pytest.raises(JiraClientError, match="timed out"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_invalid_json(self, mock_config: Config) -> None:
        """Should raise JiraClientError for invalid JSON."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("not valid json")):
            client = JiraSdkClient(mock_config)
            with pytest.raises(JiraClientError, match="parse"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_generic_exception(self, mock_config: Config) -> None:
        """Should wrap generic exceptions in JiraClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = JiraSdkClient(mock_config)
            with pytest.raises(JiraClientError, match="failed"):
                client.search_issues("project = TEST")

    def test_shutdown_interrupts_search(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        # Request shutdown before calling
        request_shutdown()

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("[]")):
            client = JiraSdkClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                client.search_issues("project = TEST")


class TestJiraSdkTagClient:
    """Tests for JiraSdkTagClient."""

    def test_add_label_success(self, mock_config: Config) -> None:
        """Should add label successfully."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("SUCCESS")):
            client = JiraSdkTagClient(mock_config)
            # Should not raise
            client.add_label("TEST-123", "my-label")

    def test_add_label_success_with_other_text(self, mock_config: Config) -> None:
        """Should succeed if response contains SUCCESS."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("The operation was a SUCCESS. Label added."),
        ):
            client = JiraSdkTagClient(mock_config)
            client.add_label("TEST-123", "my-label")

    def test_add_label_error(self, mock_config: Config) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("ERROR: Label not found"),
        ):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="Failed to add label"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(TimeoutError()),
        ):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_generic_error(self, mock_config: Config) -> None:
        """Should wrap generic error in JiraTagClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="failed"):
                client.add_label("TEST-123", "my-label")

    def test_remove_label_success(self, mock_config: Config) -> None:
        """Should remove label successfully."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("SUCCESS")):
            client = JiraSdkTagClient(mock_config)
            # Should not raise
            client.remove_label("TEST-123", "my-label")

    def test_remove_label_error(self, mock_config: Config) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("ERROR: Label not found"),
        ):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="Failed to remove label"):
                client.remove_label("TEST-123", "my-label")

    def test_remove_label_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(TimeoutError()),
        ):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.remove_label("TEST-123", "my-label")

    def test_shutdown_interrupts_add_label(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        request_shutdown()

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("SUCCESS")):
            client = JiraSdkTagClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                client.add_label("TEST-123", "my-label")


class TestClaudeSdkAgentClient:
    """Tests for ClaudeSdkAgentClient.

    All run_agent calls use asyncio.run() since the method is async.
    """

    def test_run_agent_returns_response(self, mock_config: Config) -> None:
        """Should return AgentRunResult with agent response."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("Agent completed successfully"),
        ):
            client = ClaudeSdkAgentClient(mock_config)
            result = asyncio.run(client.run_agent("Do something"))

        assert result.response == "Agent completed successfully"
        assert result.workdir is None

    def test_run_agent_builds_context_section(self, mock_config: Config) -> None:
        """Should include context in prompt."""
        captured_prompt: list[str] = []

        with patch(
            "sentinel.agent_clients.claude_sdk.query", create_capturing_mock_query(captured_prompt)
        ):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(
                client.run_agent(
                    "Do something",
                    context={"repo": "test-repo", "branch": "main"},
                )
            )

        called_prompt = captured_prompt[0]
        assert "Context:" in called_prompt
        assert "repo: test-repo" in called_prompt
        assert "branch: main" in called_prompt

    def test_run_agent_handles_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in AgentTimeoutError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(TimeoutError()),
        ):
            client = ClaudeSdkAgentClient(mock_config)
            with pytest.raises(AgentTimeoutError, match="timed out"):
                asyncio.run(client.run_agent("Do something", timeout_seconds=300))

    def test_run_agent_handles_generic_exception(self, mock_config: Config) -> None:
        """Should wrap generic exceptions in AgentClientError."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = ClaudeSdkAgentClient(mock_config)
            with pytest.raises(AgentClientError, match="unexpected error"):
                asyncio.run(client.run_agent("Do something"))

    def test_constructor_accepts_base_workdir(self, tmp_path: Path, mock_config: Config) -> None:
        """Should accept base_workdir in constructor."""
        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)
        assert client.base_workdir == tmp_path

    def test_constructor_defaults_to_no_workdir(self, mock_config: Config) -> None:
        """Should default to no base_workdir."""
        client = ClaudeSdkAgentClient(mock_config)
        assert client.base_workdir is None

    def test_create_workdir_creates_unique_directory(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create unique directory with issue key and timestamp."""
        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)
        workdir = client._create_workdir("DS-123")

        assert workdir.parent == tmp_path
        assert workdir.name.startswith("DS-123_")
        assert workdir.exists()
        assert workdir.is_dir()

    def test_create_workdir_creates_base_if_missing(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create base directory if it doesn't exist."""
        base_path = tmp_path / "nonexistent" / "base"
        client = ClaudeSdkAgentClient(mock_config, base_workdir=base_path)
        workdir = client._create_workdir("DS-456")

        assert base_path.exists()
        assert workdir.exists()

    def test_create_workdir_raises_without_base_workdir(self, mock_config: Config) -> None:
        """Should raise AgentClientError if base_workdir not configured."""
        client = ClaudeSdkAgentClient(mock_config)
        with pytest.raises(AgentClientError, match="base_workdir not configured"):
            client._create_workdir("DS-789")

    def test_run_agent_creates_workdir_when_configured(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create workdir when base_workdir and issue_key provided."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)
            asyncio.run(client.run_agent("Do something", issue_key="DS-100"))

        # Check that workdir was created
        workdirs = list(tmp_path.glob("DS-100_*"))
        assert len(workdirs) == 1

    def test_run_agent_no_workdir_without_base(self, mock_config: Config) -> None:
        """Should not create workdir when base_workdir not configured."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(client.run_agent("Do something", issue_key="DS-200"))
        # No exception should be raised

    def test_run_agent_no_workdir_without_issue_key(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create workdir when issue_key not provided."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)
            asyncio.run(client.run_agent("Do something"))

        # No workdir should be created
        assert len(list(tmp_path.iterdir())) == 0

    def test_run_agent_rejects_when_circuit_open(
        self,
        mock_config: Config,
    ) -> None:
        """Test that run_agent raises error when circuit breaker is open."""
        cb = CircuitBreaker(
            service_name="claude",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        # Force circuit breaker to open state
        cb.record_failure(Exception("test failure"))

        client = ClaudeSdkAgentClient(mock_config, circuit_breaker=cb)
        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "circuit breaker is open" in str(exc_info.value)
        assert "Claude" in str(exc_info.value)

    def test_run_agent_circuit_open_skips_workdir_creation(
        self,
        tmp_path: Path,
        mock_config: Config,
    ) -> None:
        """Test that circuit breaker check happens before workdir creation.

        When the circuit breaker is open, run_agent() should reject immediately
        without creating a working directory. This avoids unnecessary disk I/O
        when the service is unavailable (DS-670).
        """
        cb = CircuitBreaker(
            service_name="claude",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(Exception("test failure"))

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path, circuit_breaker=cb)

        with pytest.raises(AgentClientError, match="circuit breaker is open"):
            asyncio.run(client.run_agent("prompt", issue_key="TEST-123"))

        # Verify no workdir was created - the directory should be empty
        assert list(tmp_path.iterdir()) == []

    def test_shutdown_interrupts_agent(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        request_shutdown()

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                asyncio.run(client.run_agent("Do something"))

    def test_shutdown_interrupts_agent_with_injected_controller(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError with injected ShutdownController.

        This test demonstrates the improved testability with dependency injection.
        By injecting a dedicated ShutdownController, tests can control shutdown
        behavior independently without affecting global state.
        """
        # Create an isolated controller for this test
        controller = ShutdownController()
        controller.request_shutdown()

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            # Inject the controller - no global state needed
            client = ClaudeSdkAgentClient(mock_config, shutdown_controller=controller)
            with pytest.raises(ClaudeProcessInterruptedError):
                asyncio.run(client.run_agent("Do something"))

    def test_controller_isolation(self, mock_config: Config) -> None:
        """Test that different controllers are truly isolated.

        Verifies that shutdown on one controller doesn't affect another,
        demonstrating the improved testability of the dependency injection approach.
        """
        controller1 = ShutdownController()
        controller2 = ShutdownController()

        # Shutdown only controller1
        controller1.request_shutdown()

        assert controller1.is_shutdown_requested()
        assert not controller2.is_shutdown_requested()

        # Reset controller1 and verify it's truly reset
        controller1.reset()
        assert not controller1.is_shutdown_requested()
        assert not controller2.is_shutdown_requested()


class TestClaudeSdkAgentClientStreaming:
    """Tests for streaming log functionality in ClaudeSdkAgentClient."""

    def test_constructor_accepts_log_base_dir(self, tmp_path: Path, mock_config: Config) -> None:
        """Should accept log_base_dir in constructor."""
        client = ClaudeSdkAgentClient(mock_config, log_base_dir=tmp_path)
        assert client.log_base_dir == tmp_path

    def test_constructor_defaults_to_no_log_dir(self, mock_config: Config) -> None:
        """Should default to no log_base_dir."""
        client = ClaudeSdkAgentClient(mock_config)
        assert client.log_base_dir is None

    def test_streaming_enabled_when_all_params_provided(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should use streaming logs when all required params provided."""
        log_dir = tmp_path / "logs"
        work_dir = tmp_path / "work"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(
                mock_config,
                base_workdir=work_dir,
                log_base_dir=log_dir,
            )
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                )
            )

        # Log directory should be created
        assert (log_dir / "test-orch").exists()
        # Log file should exist
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1

    def test_streaming_log_contains_header(self, tmp_path: Path, mock_config: Config) -> None:
        """Should write header to log file."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.agent_clients.claude_sdk.query", create_mock_query("Agent completed task")
        ):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            result = asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                )
            )

        assert result.response == "Agent completed task"
        assert result.workdir is None

        # Check log file content
        log_files = list((log_dir / "test").glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "DS-123" in content
        assert "Test prompt" in content

    def test_streaming_disabled_without_log_base_dir(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create streaming logs when log_base_dir not set."""
        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                )
            )

        # No logs directory should be created in tmp_path
        assert not (tmp_path / "test").exists()

    def test_streaming_disabled_without_issue_key(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create streaming logs without issue_key."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            asyncio.run(
                client.run_agent(
                    "Test",
                    [],
                    orchestration_name="test",
                )
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_disabled_without_orchestration_name(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create streaming logs without orchestration_name."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            asyncio.run(
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                )
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_log_finalized_on_success(self, tmp_path: Path, mock_config: Config) -> None:
        """Should finalize log with COMPLETED status on success."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            asyncio.run(
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                )
            )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         COMPLETED" in content
        assert "END OF LOG" in content

    def test_streaming_log_finalized_on_timeout(self, tmp_path: Path, mock_config: Config) -> None:
        """Should finalize log with TIMEOUT status on timeout."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(TimeoutError()),
        ):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            with pytest.raises(AgentTimeoutError):
                asyncio.run(
                    client.run_agent(
                        "Test",
                        [],
                        issue_key="DS-123",
                        orchestration_name="test",
                        timeout_seconds=60,
                    )
                )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         TIMEOUT" in content

    def test_streaming_log_finalized_on_error(self, tmp_path: Path, mock_config: Config) -> None:
        """Should finalize log with ERROR status on error."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_raising_mock_query(RuntimeError("test error")),
        ):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            with pytest.raises(AgentClientError):
                asyncio.run(
                    client.run_agent(
                        "Test",
                        [],
                        issue_key="DS-123",
                        orchestration_name="test",
                    )
                )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         ERROR" in content


class TestDisableStreamingLogs:
    """Tests for disable_streaming_logs functionality."""

    def test_constructor_accepts_disable_streaming_logs(self, mock_config: Config) -> None:
        """Should accept disable_streaming_logs parameter in constructor."""
        client = ClaudeSdkAgentClient(mock_config, disable_streaming_logs=True)
        assert client._disable_streaming_logs is True

    def test_constructor_defaults_to_config_value(self) -> None:
        """Should default to config value when not explicitly provided."""
        config = make_config(disable_streaming_logs=True)
        client = ClaudeSdkAgentClient(config)
        assert client._disable_streaming_logs is True

    def test_constructor_defaults_false_when_not_set(self, mock_config: Config) -> None:
        """Should default to False when config has default value."""
        client = ClaudeSdkAgentClient(mock_config)
        assert client._disable_streaming_logs is False

    def test_explicit_param_overrides_config(self) -> None:
        """Should use explicit parameter over config value."""
        config = make_config(disable_streaming_logs=True)
        client = ClaudeSdkAgentClient(config, disable_streaming_logs=False)
        assert client._disable_streaming_logs is False

    def test_disabled_streaming_uses_simple_path(self, tmp_path: Path, mock_config: Config) -> None:
        """Should use _run_simple when streaming is disabled."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(
                mock_config,
                log_base_dir=log_dir,
                disable_streaming_logs=True,
            )
            result = asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                )
            )

        assert result.response == "done"
        # Log file should still be created (non-streaming mode writes after completion)
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1

    def test_disabled_streaming_writes_log_after_completion(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should write complete log file after execution when streaming is disabled."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("Agent completed task successfully"),
        ):
            client = ClaudeSdkAgentClient(
                mock_config,
                log_base_dir=log_dir,
                disable_streaming_logs=True,
            )
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                )
            )

        # Check log file content
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        # Should contain non-streaming mode indicator
        assert "non-streaming mode" in content
        assert "SENTINEL_DISABLE_STREAMING_LOGS" in content
        # Should contain prompt and response
        assert "Test prompt" in content
        assert "Agent completed task successfully" in content
        # Should contain status
        assert "Status:         COMPLETED" in content
        assert "END OF LOG" in content

    def test_disabled_streaming_no_log_without_params(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not write log when streaming disabled but params missing."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(
                mock_config,
                log_base_dir=log_dir,
                disable_streaming_logs=True,
            )
            # Missing issue_key
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    orchestration_name="test-orch",
                )
            )

        # No log should be created
        assert not log_dir.exists()

    def test_enabled_streaming_uses_streaming_path(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should use streaming logs when not disabled."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(
                mock_config,
                log_base_dir=log_dir,
                disable_streaming_logs=False,
            )
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                )
            )

        # Log file should be created with streaming format
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        # Should NOT contain non-streaming mode indicator
        assert "non-streaming mode" not in content
        # Should have streaming format markers
        assert "AGENT EXECUTION LOG" in content


class TestClaudeSdkAgentClientBranchSetup:
    """Tests for ClaudeSdkAgentClient._setup_branch method."""

    def test_checkout_existing_branch(self, tmp_path: Path, mock_config: Config) -> None:
        """Should checkout and pull when branch exists on remote.

        Mock behavior expectations:
        - subprocess.run is mocked to track all git command invocations
        - All commands return success (returncode=0)
        - git ls-remote returns a SHA ref for the branch, simulating an existing remote branch
        - The test verifies that fetch, ls-remote, checkout, and pull commands are invoked
        """
        # Create a mock git repository
        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        # Mock subprocess.run calls
        call_history: list[list[str]] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            call_history.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            # ls-remote returns branch info when branch exists
            if "ls-remote" in cmd:
                result.stdout = "abc123\trefs/heads/feature/DS-123"
            # rev-parse with multiple args returns newline-separated output
            # Current branch, local SHA, and remote SHA (return different SHAs to trigger checkout)
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main\nlocal123\nremote456"

            return result

        with patch("subprocess.run", side_effect=mock_run):
            client._setup_branch(workdir, "feature/DS-123", create_branch=False, base_branch="main")

        # Verify the sequence of git commands
        assert any("fetch" in " ".join(cmd) for cmd in call_history)
        assert any(
            "ls-remote" in " ".join(cmd) and "feature/DS-123" in " ".join(cmd)
            for cmd in call_history
        )
        # Verify the optimized rev-parse call with multiple arguments
        assert any("rev-parse" in " ".join(cmd) and "--abbrev-ref" in cmd for cmd in call_history)
        assert any(
            "checkout" in " ".join(cmd) and "feature/DS-123" in " ".join(cmd)
            for cmd in call_history
        )
        assert any("pull" in " ".join(cmd) for cmd in call_history)

    def test_skip_checkout_when_already_up_to_date(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should skip checkout and pull when already on branch and up to date.

        This test verifies the optimization where we skip checkout/pull operations
        when the branch is already checked out and the local SHA matches the remote SHA.
        The combined rev-parse call returns all three values in one subprocess call.
        """
        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        call_history: list[list[str]] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            call_history.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            # ls-remote returns branch info when branch exists
            if "ls-remote" in cmd:
                result.stdout = "abc123\trefs/heads/feature/DS-123"
            # rev-parse returns current branch, local SHA, and remote SHA
            # Return same SHA for local and remote to simulate "already up to date"
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "feature/DS-123\nabc123\nabc123"

            return result

        with patch("subprocess.run", side_effect=mock_run):
            client._setup_branch(workdir, "feature/DS-123", create_branch=False, base_branch="main")

        # Verify fetch and ls-remote were called
        assert any("fetch" in " ".join(cmd) for cmd in call_history)
        assert any("ls-remote" in " ".join(cmd) for cmd in call_history)
        # Verify the optimized rev-parse call
        assert any("rev-parse" in " ".join(cmd) and "--abbrev-ref" in cmd for cmd in call_history)
        # Checkout and pull should NOT be called since branch is up to date
        assert not any("checkout" in " ".join(cmd) for cmd in call_history)
        assert not any("pull" in " ".join(cmd) for cmd in call_history)

    def test_create_new_branch_when_create_branch_true(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create branch from base_branch when branch doesn't exist and create_branch=True."""
        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        call_history: list[list[str]] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            call_history.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            # ls-remote returns empty when branch doesn't exist
            return result

        with patch("subprocess.run", side_effect=mock_run):
            client._setup_branch(
                workdir, "feature/DS-456", create_branch=True, base_branch="develop"
            )

        # Verify branch creation command
        checkout_with_b = [
            cmd for cmd in call_history if "checkout" in " ".join(cmd) and "-b" in cmd
        ]
        assert len(checkout_with_b) == 1
        assert "feature/DS-456" in checkout_with_b[0]
        assert "origin/develop" in checkout_with_b[0]

    def test_error_when_branch_missing_and_create_branch_false(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should raise AgentClientError when branch doesn't exist and create_branch=False."""
        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""  # Empty = branch doesn't exist
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AgentClientError, match="does not exist on remote"):
                client._setup_branch(
                    workdir, "nonexistent-branch", create_branch=False, base_branch="main"
                )

    def test_no_branch_operations_when_branch_is_none(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not call _setup_branch when branch is None in run_agent."""
        workdir = tmp_path / "work"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, base_workdir=workdir)
            with patch.object(client, "_setup_branch") as mock_setup:
                asyncio.run(
                    client.run_agent(
                        "Do something",
                        [],
                        issue_key="DS-123",
                        branch=None,  # No branch specified
                    )
                )

        # _setup_branch should not be called when branch is None
        mock_setup.assert_not_called()

    def test_branch_setup_called_when_branch_provided(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should call _setup_branch when branch is provided in run_agent."""
        workdir = tmp_path / "work"

        with patch("sentinel.agent_clients.claude_sdk.query", create_mock_query("done")):
            client = ClaudeSdkAgentClient(mock_config, base_workdir=workdir)
            with patch.object(client, "_setup_branch") as mock_setup:
                asyncio.run(
                    client.run_agent(
                        "Do something",
                        [],
                        issue_key="DS-123",
                        branch="feature/DS-123",
                        create_branch=True,
                        base_branch="develop",
                    )
                )

        # _setup_branch should be called with correct parameters
        mock_setup.assert_called_once()
        call_args = mock_setup.call_args
        assert call_args[0][1] == "feature/DS-123"  # branch
        assert call_args[0][2] is True  # create_branch
        assert call_args[0][3] == "develop"  # base_branch

    def test_git_fetch_failure_raises_error(self, tmp_path: Path, mock_config: Config) -> None:
        """Should raise AgentClientError when git fetch fails."""
        import subprocess

        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "fetch" in cmd:
                error = subprocess.CalledProcessError(1, cmd)
                error.stderr = "fatal: could not read from remote repository"
                raise error
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AgentClientError, match="Git operation failed"):
                client._setup_branch(
                    workdir, "feature/DS-123", create_branch=False, base_branch="main"
                )

    def test_git_checkout_failure_raises_error(self, tmp_path: Path, mock_config: Config) -> None:
        """Should raise AgentClientError when git checkout fails."""
        import subprocess

        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "checkout" in cmd:
                error = subprocess.CalledProcessError(1, cmd)
                error.stderr = "error: pathspec 'feature/DS-123' did not match any file(s)"
                raise error
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            # ls-remote returns branch info when branch exists
            if "ls-remote" in cmd:
                result.stdout = "abc123\trefs/heads/feature/DS-123"
            # rev-parse with multiple args returns newline-separated output
            # Return different SHAs to trigger the checkout path
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main\nlocal123\nremote456"
            else:
                result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AgentClientError, match="Git operation failed"):
                client._setup_branch(
                    workdir, "feature/DS-123", create_branch=False, base_branch="main"
                )

    def test_malformed_rev_parse_output_raises_error(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should raise AgentClientError when git rev-parse returns unexpected output.

        This test verifies the defensive validation that checks for
        the expected 3-line output from the combined rev-parse command. While check=True
        catches most failures, malformed output could occur in edge cases.
        """
        workdir = tmp_path / "repo"
        workdir.mkdir()

        client = ClaudeSdkAgentClient(mock_config, base_workdir=tmp_path)

        call_history: list[list[str]] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            call_history.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""

            # ls-remote returns branch info when branch exists
            if "ls-remote" in cmd:
                result.stdout = "abc123\trefs/heads/feature/DS-123"
            # Return malformed output (only 2 lines instead of 3)
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main\nlocal123"  # Missing third line
            else:
                result.stdout = ""

            return result

        with patch("subprocess.run", side_effect=mock_run), pytest.raises(
            AgentClientError, match="Unexpected git rev-parse output: expected 3 lines, got 2"
        ):
            client._setup_branch(
                workdir, "feature/DS-123", create_branch=False, base_branch="main"
            )

    def test_git_fetch_timeout_raises_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """Should raise AgentTimeoutError when git fetch times out.

        This test verifies that subprocess.TimeoutExpired exceptions from git commands
        are properly converted to AgentTimeoutError with a descriptive message.
        """
        workdir = tmp_path / "repo"
        workdir.mkdir()

        # Create config with a short timeout for testing
        config = make_config(subprocess_timeout=5.0)
        client = ClaudeSdkAgentClient(config, base_workdir=tmp_path)

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "fetch" in cmd:
                raise subprocess.TimeoutExpired(cmd, timeout=5.0)
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AgentTimeoutError, match="Git operation timed out"):
                client._setup_branch(
                    workdir, "feature/DS-123", create_branch=False, base_branch="main"
                )

    def test_git_pull_timeout_raises_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """Should raise AgentTimeoutError when git pull times out."""
        workdir = tmp_path / "repo"
        workdir.mkdir()

        config = make_config(subprocess_timeout=5.0)
        client = ClaudeSdkAgentClient(config, base_workdir=tmp_path)

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            if "pull" in cmd:
                raise subprocess.TimeoutExpired(cmd, timeout=5.0)
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""

            # ls-remote returns branch info when branch exists
            if "ls-remote" in cmd:
                result.stdout = "abc123\trefs/heads/feature/DS-123"
            # rev-parse with multiple args returns newline-separated output
            # Return different SHAs to trigger the checkout/pull path
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                result.stdout = "main\nlocal123\nremote456"
            else:
                result.stdout = ""

            return result

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AgentTimeoutError, match="Git operation timed out"):
                client._setup_branch(
                    workdir, "feature/DS-123", create_branch=False, base_branch="main"
                )

    def test_subprocess_timeout_passed_to_run(
        self, tmp_path: Path
    ) -> None:
        """Should pass subprocess_timeout from config to subprocess.run.

        This test verifies that the timeout parameter from config is correctly
        passed to all subprocess.run calls.
        """
        workdir = tmp_path / "repo"
        workdir.mkdir()

        config = make_config(subprocess_timeout=30.0)
        client = ClaudeSdkAgentClient(config, base_workdir=tmp_path)

        captured_timeouts: list[float | None] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            captured_timeouts.append(kwargs.get("timeout"))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            # ls-remote returns empty (branch doesn't exist)
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            client._setup_branch(
                workdir, "feature/new-branch", create_branch=True, base_branch="main"
            )

        # All calls should have used the configured timeout
        assert all(t == 30.0 for t in captured_timeouts)
        # Should have called fetch, ls-remote, and checkout -b
        assert len(captured_timeouts) == 3

    def test_zero_timeout_disables_timeout(
        self, tmp_path: Path
    ) -> None:
        """Should pass None as timeout when subprocess_timeout is 0.

        A timeout of 0 disables the timeout, allowing commands to run indefinitely.
        """
        workdir = tmp_path / "repo"
        workdir.mkdir()

        config = make_config(subprocess_timeout=0.0)
        client = ClaudeSdkAgentClient(config, base_workdir=tmp_path)

        captured_timeouts: list[float | None] = []

        def mock_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            captured_timeouts.append(kwargs.get("timeout"))
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            client._setup_branch(
                workdir, "feature/new-branch", create_branch=True, base_branch="main"
            )

        # All calls should have None as timeout (disabled)
        assert all(t is None for t in captured_timeouts)


class TestUsageInfo:
    """Tests for UsageInfo dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible default values."""
        usage = UsageInfo()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.total_cost_usd == 0.0
        assert usage.duration_ms == 0.0
        assert usage.duration_api_ms == 0.0
        assert usage.num_turns == 0

    def test_to_dict_returns_all_fields(self) -> None:
        """Should serialize all fields to dictionary."""
        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            total_cost_usd=0.05,
            duration_ms=1500.0,
            duration_api_ms=1200.0,
            num_turns=3,
        )
        result = usage.to_dict()

        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500  # Computed property
        assert result["total_cost_usd"] == 0.05
        assert result["duration_ms"] == 1500.0
        assert result["duration_api_ms"] == 1200.0
        assert result["num_turns"] == 3

    def test_to_dict_default_values(self) -> None:
        """Should serialize default values correctly."""
        usage = UsageInfo()
        result = usage.to_dict()

        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0
        assert result["total_cost_usd"] == 0.0

    def test_total_tokens_computed_property(self) -> None:
        """Should compute total_tokens from input_tokens + output_tokens."""
        usage = UsageInfo(input_tokens=1500, output_tokens=750)
        assert usage.total_tokens == 2250

    def test_total_tokens_updates_when_tokens_change(self) -> None:
        """Should reflect changes in input/output tokens immediately."""
        usage = UsageInfo(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

        usage.input_tokens = 200
        assert usage.total_tokens == 250

        usage.output_tokens = 100
        assert usage.total_tokens == 300

    def test_repr_includes_all_fields(self) -> None:
        """Should return a readable string representation."""
        usage = UsageInfo(
            input_tokens=1000,
            output_tokens=500,
            total_cost_usd=0.05,
            duration_ms=1500.0,
            duration_api_ms=1200.0,
            num_turns=3,
        )
        repr_str = repr(usage)

        assert "UsageInfo(" in repr_str
        assert "input_tokens=1000" in repr_str
        assert "output_tokens=500" in repr_str
        assert "total_tokens=1500" in repr_str
        assert "total_cost_usd=0.05" in repr_str
        assert "duration_ms=1500.0" in repr_str
        assert "duration_api_ms=1200.0" in repr_str
        assert "num_turns=3" in repr_str


class TestExtractUsageFromMessage:
    """Tests for _extract_usage_from_message function."""

    def test_extracts_usage_from_result_message(self) -> None:
        """Should extract usage info from ResultMessage."""
        message = MockResultMessage(
            total_cost_usd=0.10,
            duration_ms=2000.0,
            duration_api_ms=1800.0,
            num_turns=5,
            input_tokens=2000,
            output_tokens=1000,
        )
        result = _extract_usage_from_message(message)

        assert result is not None
        assert result.total_cost_usd == 0.10
        assert result.duration_ms == 2000.0
        assert result.duration_api_ms == 1800.0
        assert result.num_turns == 5
        assert result.input_tokens == 2000
        assert result.output_tokens == 1000
        assert result.total_tokens == 3000

    def test_returns_none_for_regular_message(self) -> None:
        """Should return None for messages without usage data."""
        message = MockMessage("Hello world")
        result = _extract_usage_from_message(message)

        assert result is None

    def test_handles_missing_usage_dict(self) -> None:
        """Should handle ResultMessage with no usage dict."""

        class MockResultWithoutUsage:
            total_cost_usd = 0.05
            duration_ms = 1000.0
            duration_api_ms = 800.0
            num_turns = 2
            usage = None

        message = MockResultWithoutUsage()
        result = _extract_usage_from_message(message)

        assert result is not None
        assert result.total_cost_usd == 0.05
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_handles_object_style_usage(self) -> None:
        """Should handle usage as object with attributes instead of dict."""

        class MockUsageObject:
            input_tokens = 500
            output_tokens = 250

        class MockResultWithObjectUsage:
            total_cost_usd = 0.03
            duration_ms = 800.0
            duration_api_ms = 600.0
            num_turns = 1
            usage = MockUsageObject()

        message = MockResultWithObjectUsage()
        result = _extract_usage_from_message(message)

        assert result is not None
        assert result.input_tokens == 500
        assert result.output_tokens == 250
        assert result.total_tokens == 750


class TestClaudeSdkAgentClientUsage:
    """Tests for usage extraction in ClaudeSdkAgentClient."""

    def test_run_agent_returns_usage_info(self, mock_config: Config) -> None:
        """Should include usage info in AgentRunResult when ResultMessage is received."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query_with_usage(
                "Agent completed successfully",
                total_cost_usd=0.08,
                duration_ms=2500.0,
                duration_api_ms=2000.0,
                num_turns=4,
                input_tokens=1500,
                output_tokens=800,
            ),
        ):
            client = ClaudeSdkAgentClient(mock_config)
            result = asyncio.run(client.run_agent("Do something"))

        assert result.response == "Agent completed successfully"
        assert result.usage is not None
        assert result.usage.total_cost_usd == 0.08
        assert result.usage.duration_ms == 2500.0
        assert result.usage.duration_api_ms == 2000.0
        assert result.usage.num_turns == 4
        assert result.usage.input_tokens == 1500
        assert result.usage.output_tokens == 800
        assert result.usage.total_tokens == 2300

    def test_run_agent_returns_none_usage_when_no_result_message(
        self, mock_config: Config
    ) -> None:
        """Should return None usage when no ResultMessage is received."""
        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query("Agent completed without usage"),
        ):
            client = ClaudeSdkAgentClient(mock_config)
            result = asyncio.run(client.run_agent("Do something"))

        assert result.response == "Agent completed without usage"
        assert result.usage is None

    def test_streaming_log_includes_usage_in_metrics_json(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should include usage info in metrics JSON when streaming logs."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.agent_clients.claude_sdk.query",
            create_mock_query_with_usage(
                "Agent completed task",
                total_cost_usd=0.12,
                input_tokens=3000,
                output_tokens=1500,
            ),
        ):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            result = asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                )
            )

        assert result.usage is not None
        assert result.usage.total_cost_usd == 0.12

        # Check log file contains usage in JSON metrics
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert '"usage":' in content
        assert '"input_tokens": 3000' in content
        assert '"output_tokens": 1500' in content
        assert '"total_cost_usd": 0.12' in content


class TestAgentTeamsEnvVar:
    """Tests for agent_teams env var passthrough to ClaudeAgentOptions."""

    def test_agent_teams_enabled_passes_env_var_simple(self, mock_config: Config) -> None:
        """Should pass CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS env var when agent_teams=True via _run_simple."""
        captured_options: list[Any] = []

        def capture_query(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
            async def async_gen() -> AsyncIterator[MockMessage]:
                captured_options.append(kwargs.get("options"))
                yield MockMessage("done")
            return async_gen()

        with patch("sentinel.agent_clients.claude_sdk.query", side_effect=capture_query):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(client.run_agent("Do something", agent_teams=True))

        assert len(captured_options) == 1
        options = captured_options[0]
        assert options.env == {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}

    def test_agent_teams_disabled_no_env_var_simple(self, mock_config: Config) -> None:
        """Should not pass agent teams env var when agent_teams=False via _run_simple."""
        captured_options: list[Any] = []

        def capture_query(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
            async def async_gen() -> AsyncIterator[MockMessage]:
                captured_options.append(kwargs.get("options"))
                yield MockMessage("done")
            return async_gen()

        with patch("sentinel.agent_clients.claude_sdk.query", side_effect=capture_query):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(client.run_agent("Do something", agent_teams=False))

        assert len(captured_options) == 1
        options = captured_options[0]
        assert "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" not in options.env

    def test_agent_teams_default_is_false(self, mock_config: Config) -> None:
        """Should default agent_teams to False when not specified."""
        captured_options: list[Any] = []

        def capture_query(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
            async def async_gen() -> AsyncIterator[MockMessage]:
                captured_options.append(kwargs.get("options"))
                yield MockMessage("done")
            return async_gen()

        with patch("sentinel.agent_clients.claude_sdk.query", side_effect=capture_query):
            client = ClaudeSdkAgentClient(mock_config)
            asyncio.run(client.run_agent("Do something"))

        assert len(captured_options) == 1
        options = captured_options[0]
        assert "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" not in options.env

    def test_agent_teams_enabled_passes_env_var_streaming(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should pass CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS env var when agent_teams=True via _run_with_log."""
        log_dir = tmp_path / "logs"
        captured_options: list[Any] = []

        def capture_query(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
            async def async_gen() -> AsyncIterator[MockMessage]:
                captured_options.append(kwargs.get("options"))
                yield MockMessage("done")
            return async_gen()

        with patch("sentinel.agent_clients.claude_sdk.query", side_effect=capture_query):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                    agent_teams=True,
                )
            )

        assert len(captured_options) == 1
        options = captured_options[0]
        assert options.env == {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}

    def test_agent_teams_disabled_no_env_var_streaming(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not pass agent teams env var when agent_teams=False via _run_with_log."""
        log_dir = tmp_path / "logs"
        captured_options: list[Any] = []

        def capture_query(prompt: str, **kwargs: Any) -> AsyncIterator[MockMessage]:
            async def async_gen() -> AsyncIterator[MockMessage]:
                captured_options.append(kwargs.get("options"))
                yield MockMessage("done")
            return async_gen()

        with patch("sentinel.agent_clients.claude_sdk.query", side_effect=capture_query):
            client = ClaudeSdkAgentClient(mock_config, log_base_dir=log_dir)
            asyncio.run(
                client.run_agent(
                    "Test prompt",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test-orch",
                    agent_teams=False,
                )
            )

        assert len(captured_options) == 1
        options = captured_options[0]
        assert "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS" not in options.env
