"""Tests for MCP-based client implementations using Claude Agent SDK."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from sentinel.config import Config
from sentinel.executor import AgentClientError, AgentTimeoutError
from sentinel.mcp_clients import (
    ClaudeMcpAgentClient,
    ClaudeProcessInterruptedError,
    JiraMcpClient,
    JiraMcpTagClient,
    request_shutdown,
    reset_shutdown,
)
from sentinel.poller import JiraClientError
from sentinel.tag_manager import JiraTagClientError


@pytest.fixture(autouse=True)
def reset_shutdown_state() -> None:
    """Reset the shutdown state before each test."""
    reset_shutdown()


@pytest.fixture
def mock_config() -> Config:
    """Create a mock config with MCP settings."""
    return Config(
        mcp_jira_command="npx",
        mcp_jira_args=["@anthropic/jira-mcp"],
        mcp_confluence_command="npx",
        mcp_confluence_args=["@anthropic/confluence-mcp"],
        mcp_github_command="npx",
        mcp_github_args=["@anthropic/github-mcp"],
    )


class MockMessage:
    """Mock message from the Claude Agent SDK query."""

    def __init__(self, text: str) -> None:
        self.text = text


def create_mock_query(return_value: str) -> MagicMock:
    """Create a mock query function that returns messages with the given text."""

    async def async_gen() -> AsyncIterator[MockMessage]:
        yield MockMessage(return_value)

    # Return a regular MagicMock that returns the async generator when called
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


class TestJiraMcpClient:
    """Tests for JiraMcpClient."""

    def test_search_issues_returns_list(self, mock_config: Config) -> None:
        """Should return list of issues from search."""
        issues_json = json.dumps(
            [
                {"key": "TEST-1", "fields": {"summary": "Issue 1"}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2"}},
            ]
        )

        with patch("sentinel.mcp_clients.query", create_mock_query(issues_json)):
            client = JiraMcpClient(mock_config)
            result = client.search_issues("project = TEST", max_results=10)

        assert len(result) == 2
        assert result[0]["key"] == "TEST-1"
        assert result[1]["key"] == "TEST-2"

    def test_search_issues_handles_markdown_json_block(self, mock_config: Config) -> None:
        """Should extract JSON from markdown code block."""
        response = """```json
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.mcp_clients.query", create_mock_query(response)):
            client = JiraMcpClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_generic_code_block(self, mock_config: Config) -> None:
        """Should extract JSON from generic markdown code block."""
        response = """```
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.mcp_clients.query", create_mock_query(response)):
            client = JiraMcpClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_wraps_single_issue(self, mock_config: Config) -> None:
        """Should wrap single issue object in list."""
        response = '{"key": "TEST-1", "fields": {"summary": "Issue"}}'

        with patch("sentinel.mcp_clients.query", create_mock_query(response)):
            client = JiraMcpClient(mock_config)
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(asyncio.TimeoutError()),
        ):
            client = JiraMcpClient(mock_config)
            with pytest.raises(JiraClientError, match="timed out"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_invalid_json(self, mock_config: Config) -> None:
        """Should raise JiraClientError for invalid JSON."""
        with patch("sentinel.mcp_clients.query", create_mock_query("not valid json")):
            client = JiraMcpClient(mock_config)
            with pytest.raises(JiraClientError, match="parse"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_generic_exception(self, mock_config: Config) -> None:
        """Should wrap generic exceptions in JiraClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = JiraMcpClient(mock_config)
            with pytest.raises(JiraClientError, match="failed"):
                client.search_issues("project = TEST")

    def test_shutdown_interrupts_search(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        # Request shutdown before calling
        request_shutdown()

        with patch("sentinel.mcp_clients.query", create_mock_query("[]")):
            client = JiraMcpClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                client.search_issues("project = TEST")


class TestJiraMcpTagClient:
    """Tests for JiraMcpTagClient."""

    def test_add_label_success(self, mock_config: Config) -> None:
        """Should add label successfully."""
        with patch("sentinel.mcp_clients.query", create_mock_query("SUCCESS")):
            client = JiraMcpTagClient(mock_config)
            # Should not raise
            client.add_label("TEST-123", "my-label")

    def test_add_label_success_with_other_text(self, mock_config: Config) -> None:
        """Should succeed if response contains SUCCESS."""
        with patch(
            "sentinel.mcp_clients.query",
            create_mock_query("The operation was a SUCCESS. Label added."),
        ):
            client = JiraMcpTagClient(mock_config)
            client.add_label("TEST-123", "my-label")

    def test_add_label_error(self, mock_config: Config) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.mcp_clients.query",
            create_mock_query("ERROR: Label not found"),
        ):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="Failed to add label"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(asyncio.TimeoutError()),
        ):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_generic_error(self, mock_config: Config) -> None:
        """Should wrap generic error in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="failed"):
                client.add_label("TEST-123", "my-label")

    def test_remove_label_success(self, mock_config: Config) -> None:
        """Should remove label successfully."""
        with patch("sentinel.mcp_clients.query", create_mock_query("SUCCESS")):
            client = JiraMcpTagClient(mock_config)
            # Should not raise
            client.remove_label("TEST-123", "my-label")

    def test_remove_label_error(self, mock_config: Config) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.mcp_clients.query",
            create_mock_query("ERROR: Label not found"),
        ):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="Failed to remove label"):
                client.remove_label("TEST-123", "my-label")

    def test_remove_label_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(asyncio.TimeoutError()),
        ):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.remove_label("TEST-123", "my-label")

    def test_shutdown_interrupts_add_label(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        request_shutdown()

        with patch("sentinel.mcp_clients.query", create_mock_query("SUCCESS")):
            client = JiraMcpTagClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                client.add_label("TEST-123", "my-label")


class TestClaudeMcpAgentClient:
    """Tests for ClaudeMcpAgentClient."""

    def test_run_agent_returns_response(self, mock_config: Config) -> None:
        """Should return agent response."""
        with patch(
            "sentinel.mcp_clients.query",
            create_mock_query("Agent completed successfully"),
        ):
            client = ClaudeMcpAgentClient(mock_config)
            result = client.run_agent("Do something", ["jira"])

        assert result == "Agent completed successfully"

    def test_run_agent_builds_context_section(self, mock_config: Config) -> None:
        """Should include context in prompt."""
        captured_prompt: list[str] = []

        with patch("sentinel.mcp_clients.query", create_capturing_mock_query(captured_prompt)):
            client = ClaudeMcpAgentClient(mock_config)
            client.run_agent(
                "Do something",
                ["jira"],
                context={"repo": "test-repo", "branch": "main"},
            )

        called_prompt = captured_prompt[0]
        assert "Context:" in called_prompt
        assert "repo: test-repo" in called_prompt
        assert "branch: main" in called_prompt

    def test_run_agent_builds_tools_section(self, mock_config: Config) -> None:
        """Should include available tools in prompt."""
        captured_prompt: list[str] = []

        with patch("sentinel.mcp_clients.query", create_capturing_mock_query(captured_prompt)):
            client = ClaudeMcpAgentClient(mock_config)
            client.run_agent("Do something", ["jira", "github", "confluence"])

        called_prompt = captured_prompt[0]
        assert "Available tools:" in called_prompt
        assert "Jira" in called_prompt
        assert "GitHub" in called_prompt
        assert "Confluence" in called_prompt

    def test_run_agent_handles_timeout(self, mock_config: Config) -> None:
        """Should wrap timeout in AgentTimeoutError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(asyncio.TimeoutError()),
        ):
            client = ClaudeMcpAgentClient(mock_config)
            with pytest.raises(AgentTimeoutError, match="timed out"):
                client.run_agent("Do something", [], timeout_seconds=300)

    def test_run_agent_handles_generic_exception(self, mock_config: Config) -> None:
        """Should wrap generic exceptions in AgentClientError."""
        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(RuntimeError("unexpected error")),
        ):
            client = ClaudeMcpAgentClient(mock_config)
            with pytest.raises(AgentClientError, match="unexpected error"):
                client.run_agent("Do something", [])

    def test_constructor_accepts_base_workdir(self, tmp_path: Path, mock_config: Config) -> None:
        """Should accept base_workdir in constructor."""
        client = ClaudeMcpAgentClient(mock_config, base_workdir=tmp_path)
        assert client.base_workdir == tmp_path

    def test_constructor_defaults_to_no_workdir(self, mock_config: Config) -> None:
        """Should default to no base_workdir."""
        client = ClaudeMcpAgentClient(mock_config)
        assert client.base_workdir is None

    def test_create_workdir_creates_unique_directory(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create unique directory with issue key and timestamp."""
        client = ClaudeMcpAgentClient(mock_config, base_workdir=tmp_path)
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
        client = ClaudeMcpAgentClient(mock_config, base_workdir=base_path)
        workdir = client._create_workdir("DS-456")

        assert base_path.exists()
        assert workdir.exists()

    def test_create_workdir_raises_without_base_workdir(self, mock_config: Config) -> None:
        """Should raise AgentClientError if base_workdir not configured."""
        client = ClaudeMcpAgentClient(mock_config)
        with pytest.raises(AgentClientError, match="base_workdir not configured"):
            client._create_workdir("DS-789")

    def test_run_agent_creates_workdir_when_configured(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should create workdir when base_workdir and issue_key provided."""
        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config, base_workdir=tmp_path)
            client.run_agent("Do something", [], issue_key="DS-100")

        # Check that workdir was created
        workdirs = list(tmp_path.glob("DS-100_*"))
        assert len(workdirs) == 1

    def test_run_agent_no_workdir_without_base(self, mock_config: Config) -> None:
        """Should not create workdir when base_workdir not configured."""
        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config)
            client.run_agent("Do something", [], issue_key="DS-200")
        # No exception should be raised

    def test_run_agent_no_workdir_without_issue_key(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create workdir when issue_key not provided."""
        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config, base_workdir=tmp_path)
            client.run_agent("Do something", [])

        # No workdir should be created
        assert len(list(tmp_path.iterdir())) == 0

    def test_shutdown_interrupts_agent(self, mock_config: Config) -> None:
        """Should raise ClaudeProcessInterruptedError when shutdown is requested."""
        request_shutdown()

        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config)
            with pytest.raises(ClaudeProcessInterruptedError):
                client.run_agent("Do something", [])


class TestClaudeMcpAgentClientStreaming:
    """Tests for streaming log functionality in ClaudeMcpAgentClient."""

    def test_constructor_accepts_log_base_dir(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should accept log_base_dir in constructor."""
        client = ClaudeMcpAgentClient(mock_config, log_base_dir=tmp_path)
        assert client.log_base_dir == tmp_path

    def test_constructor_defaults_to_no_log_dir(self, mock_config: Config) -> None:
        """Should default to no log_base_dir."""
        client = ClaudeMcpAgentClient(mock_config)
        assert client.log_base_dir is None

    def test_streaming_enabled_when_all_params_provided(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should use streaming logs when all required params provided."""
        log_dir = tmp_path / "logs"
        work_dir = tmp_path / "work"

        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(
                mock_config,
                base_workdir=work_dir,
                log_base_dir=log_dir,
            )
            client.run_agent(
                "Test prompt",
                [],
                issue_key="DS-123",
                orchestration_name="test-orch",
            )

        # Log directory should be created
        assert (log_dir / "test-orch").exists()
        # Log file should exist
        log_files = list((log_dir / "test-orch").glob("*.log"))
        assert len(log_files) == 1

    def test_streaming_log_contains_header(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should write header to log file."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients.query", create_mock_query("Agent completed task")):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            result = client.run_agent(
                "Test prompt",
                [],
                issue_key="DS-123",
                orchestration_name="test",
            )

        assert result == "Agent completed task"

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
        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config)
            client.run_agent(
                "Test",
                [],
                issue_key="DS-123",
                orchestration_name="test",
            )

        # No logs directory should be created in tmp_path
        assert not (tmp_path / "test").exists()

    def test_streaming_disabled_without_issue_key(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create streaming logs without issue_key."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            client.run_agent(
                "Test",
                [],
                orchestration_name="test",
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_disabled_without_orchestration_name(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should not create streaming logs without orchestration_name."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            client.run_agent(
                "Test",
                [],
                issue_key="DS-123",
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_log_finalized_on_success(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should finalize log with COMPLETED status on success."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients.query", create_mock_query("done")):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            client.run_agent(
                "Test",
                [],
                issue_key="DS-123",
                orchestration_name="test",
            )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         COMPLETED" in content
        assert "END OF LOG" in content

    def test_streaming_log_finalized_on_timeout(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should finalize log with TIMEOUT status on timeout."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(asyncio.TimeoutError()),
        ):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            with pytest.raises(AgentTimeoutError):
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                    timeout_seconds=60,
                )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         TIMEOUT" in content

    def test_streaming_log_finalized_on_error(
        self, tmp_path: Path, mock_config: Config
    ) -> None:
        """Should finalize log with ERROR status on error."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.mcp_clients.query",
            create_raising_mock_query(RuntimeError("test error")),
        ):
            client = ClaudeMcpAgentClient(mock_config, log_base_dir=log_dir)
            with pytest.raises(AgentClientError):
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         ERROR" in content
