"""Tests for MCP-based client implementations."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel.executor import AgentClientError, AgentTimeoutError
from sentinel.mcp_clients import (
    ClaudeMcpAgentClient,
    ClaudeProcessInterruptedError,
    JiraMcpClient,
    JiraMcpTagClient,
    _run_claude,
    request_shutdown,
    reset_shutdown,
)
from sentinel.poller import JiraClientError
from sentinel.tag_manager import JiraTagClientError


@pytest.fixture(autouse=True)
def reset_shutdown_state() -> None:
    """Reset the shutdown state before each test."""
    reset_shutdown()


def create_mock_process(stdout: str = "output", returncode: int = 0) -> MagicMock:
    """Create a mock Popen process with proper stream mocks.

    The new _run_claude implementation reads from stdout/stderr using
    threads that iterate via readline(), so we need to provide StringIO
    objects that support line-by-line reading.
    """
    mock_process = MagicMock()
    mock_process.poll.return_value = returncode  # Process completed

    # Create proper stream mocks using StringIO
    # The thread will iterate line-by-line via readline()
    mock_process.stdout = io.StringIO(stdout)
    mock_process.stderr = io.StringIO("")

    # communicate() is still called for error cases
    mock_process.communicate.return_value = (stdout, "")
    return mock_process


class TestRunClaude:
    """Tests for the _run_claude helper function."""

    def test_runs_claude_command(self) -> None:
        """Should run claude CLI with correct arguments."""
        mock_process = create_mock_process("test output")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            result = _run_claude("test prompt")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert cmd == ["claude", "--print", "--output-format", "text", "-p", "test prompt"]
        assert call_args[1]["start_new_session"] is True
        assert result == "test output"

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace from output."""
        mock_process = create_mock_process("  output with spaces  \n")

        with patch("subprocess.Popen", return_value=mock_process):
            result = _run_claude("test")

        assert result == "output with spaces"

    def test_respects_timeout(self) -> None:
        """Should handle timeout correctly."""
        mock_process = create_mock_process("output")

        with patch("subprocess.Popen", return_value=mock_process):
            result = _run_claude("test", timeout_seconds=120)

        assert result == "output"

    def test_raises_on_timeout(self) -> None:
        """Should raise TimeoutExpired when process exceeds timeout."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch("time.time", side_effect=[0, 0, 61]),  # Start, first check, timeout
            patch("time.sleep"),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            _run_claude("test", timeout_seconds=60)

    def test_raises_on_error(self) -> None:
        """Should raise CalledProcessError on command failure."""
        mock_process = create_mock_process("", returncode=1)

        with (
            patch("subprocess.Popen", return_value=mock_process),
            pytest.raises(subprocess.CalledProcessError),
        ):
            _run_claude("test")

    def test_includes_allowed_tools(self) -> None:
        """Should include --allowedTools flag when tools specified."""
        mock_process = create_mock_process("output")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            _run_claude("test", allowed_tools=["mcp__jira-agent__search_issues"])

        cmd = mock_popen.call_args[0][0]
        assert "--allowedTools" in cmd
        assert "mcp__jira-agent__search_issues" in cmd

    def test_multiple_allowed_tools_joined(self) -> None:
        """Should join multiple allowed tools with comma."""
        mock_process = create_mock_process("output")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            _run_claude("test", allowed_tools=["tool1", "tool2", "tool3"])

        cmd = mock_popen.call_args[0][0]
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "tool1,tool2,tool3"

    def test_passes_cwd(self) -> None:
        """Should pass cwd to subprocess."""
        mock_process = create_mock_process("output")

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            _run_claude("test", cwd="/path/to/workdir")

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs["cwd"] == "/path/to/workdir"

    def test_shutdown_interrupts_process(self) -> None:
        """Should terminate process when shutdown is requested."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = None

        # Set shutdown before calling
        request_shutdown()

        with (
            patch("subprocess.Popen", return_value=mock_process),
            pytest.raises(ClaudeProcessInterruptedError),
        ):
            _run_claude("test")

        mock_process.terminate.assert_called_once()


class TestJiraMcpClient:
    """Tests for JiraMcpClient."""

    def test_search_issues_returns_list(self) -> None:
        """Should return list of issues from search."""
        issues_json = json.dumps(
            [
                {"key": "TEST-1", "fields": {"summary": "Issue 1"}},
                {"key": "TEST-2", "fields": {"summary": "Issue 2"}},
            ]
        )

        with patch("sentinel.mcp_clients._run_claude", return_value=issues_json):
            client = JiraMcpClient()
            result = client.search_issues("project = TEST", max_results=10)

        assert len(result) == 2
        assert result[0]["key"] == "TEST-1"
        assert result[1]["key"] == "TEST-2"

    def test_search_issues_handles_markdown_json_block(self) -> None:
        """Should extract JSON from markdown code block."""
        response = """```json
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.mcp_clients._run_claude", return_value=response):
            client = JiraMcpClient()
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_generic_code_block(self) -> None:
        """Should extract JSON from generic markdown code block."""
        response = """```
[{"key": "TEST-1", "fields": {"summary": "Issue"}}]
```"""

        with patch("sentinel.mcp_clients._run_claude", return_value=response):
            client = JiraMcpClient()
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_wraps_single_issue(self) -> None:
        """Should wrap single issue object in list."""
        response = '{"key": "TEST-1", "fields": {"summary": "Issue"}}'

        with patch("sentinel.mcp_clients._run_claude", return_value=response):
            client = JiraMcpClient()
            result = client.search_issues("project = TEST")

        assert len(result) == 1
        assert result[0]["key"] == "TEST-1"

    def test_search_issues_handles_timeout(self) -> None:
        """Should wrap timeout in JiraClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.TimeoutExpired("cmd", 120),
        ):
            client = JiraMcpClient()
            with pytest.raises(JiraClientError, match="timed out"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_command_error(self) -> None:
        """Should wrap command error in JiraClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error msg"),
        ):
            client = JiraMcpClient()
            with pytest.raises(JiraClientError, match="failed"):
                client.search_issues("project = TEST")

    def test_search_issues_handles_invalid_json(self) -> None:
        """Should raise JiraClientError for invalid JSON."""
        with patch("sentinel.mcp_clients._run_claude", return_value="not valid json"):
            client = JiraMcpClient()
            with pytest.raises(JiraClientError, match="parse"):
                client.search_issues("project = TEST")


class TestJiraMcpTagClient:
    """Tests for JiraMcpTagClient."""

    def test_add_label_success(self) -> None:
        """Should add label successfully."""
        with patch("sentinel.mcp_clients._run_claude", return_value="SUCCESS"):
            client = JiraMcpTagClient()
            # Should not raise
            client.add_label("TEST-123", "my-label")

    def test_add_label_success_with_other_text(self) -> None:
        """Should succeed if response contains SUCCESS."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            return_value="The operation was a SUCCESS. Label added.",
        ):
            client = JiraMcpTagClient()
            client.add_label("TEST-123", "my-label")

    def test_add_label_error(self) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            return_value="ERROR: Label not found",
        ):
            client = JiraMcpTagClient()
            with pytest.raises(JiraTagClientError, match="Failed to add label"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_timeout(self) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.TimeoutExpired("cmd", 60),
        ):
            client = JiraMcpTagClient()
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.add_label("TEST-123", "my-label")

    def test_add_label_command_error(self) -> None:
        """Should wrap command error in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"),
        ):
            client = JiraMcpTagClient()
            with pytest.raises(JiraTagClientError, match="failed"):
                client.add_label("TEST-123", "my-label")

    def test_remove_label_success(self) -> None:
        """Should remove label successfully."""
        with patch("sentinel.mcp_clients._run_claude", return_value="SUCCESS"):
            client = JiraMcpTagClient()
            # Should not raise
            client.remove_label("TEST-123", "my-label")

    def test_remove_label_error(self) -> None:
        """Should raise JiraTagClientError on error response."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            return_value="ERROR: Label not found",
        ):
            client = JiraMcpTagClient()
            with pytest.raises(JiraTagClientError, match="Failed to remove label"):
                client.remove_label("TEST-123", "my-label")

    def test_remove_label_timeout(self) -> None:
        """Should wrap timeout in JiraTagClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.TimeoutExpired("cmd", 60),
        ):
            client = JiraMcpTagClient()
            with pytest.raises(JiraTagClientError, match="timed out"):
                client.remove_label("TEST-123", "my-label")


class TestClaudeMcpAgentClient:
    """Tests for ClaudeMcpAgentClient."""

    def test_run_agent_returns_response(self) -> None:
        """Should return agent response."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            return_value="Agent completed successfully",
        ):
            client = ClaudeMcpAgentClient()
            result = client.run_agent("Do something", ["jira"])

        assert result == "Agent completed successfully"

    def test_run_agent_builds_context_section(self) -> None:
        """Should include context in prompt."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient()
            client.run_agent(
                "Do something",
                ["jira"],
                context={"repo": "test-repo", "branch": "main"},
            )

        called_prompt = mock.call_args[0][0]
        assert "Context:" in called_prompt
        assert "repo: test-repo" in called_prompt
        assert "branch: main" in called_prompt

    def test_run_agent_builds_tools_section(self) -> None:
        """Should include available tools in prompt."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient()
            client.run_agent("Do something", ["jira", "github", "confluence"])

        called_prompt = mock.call_args[0][0]
        assert "Available tools:" in called_prompt
        assert "Jira" in called_prompt
        assert "GitHub" in called_prompt
        assert "Confluence" in called_prompt

    def test_run_agent_passes_timeout(self) -> None:
        """Should pass timeout to _run_claude."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient()
            client.run_agent("Do something", [], timeout_seconds=300)

        mock.assert_called_once()
        assert mock.call_args[1]["timeout_seconds"] == 300

    def test_run_agent_handles_timeout(self) -> None:
        """Should wrap timeout in AgentTimeoutError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.TimeoutExpired("cmd", 300),
        ):
            client = ClaudeMcpAgentClient()
            with pytest.raises(AgentTimeoutError, match="timed out"):
                client.run_agent("Do something", [], timeout_seconds=300)

    def test_run_agent_handles_command_error(self) -> None:
        """Should wrap command error in AgentClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.CalledProcessError(1, "cmd", stderr="failed"),
        ):
            client = ClaudeMcpAgentClient()
            with pytest.raises(AgentClientError, match="failed"):
                client.run_agent("Do something", [])

    def test_run_agent_handles_generic_exception(self) -> None:
        """Should wrap generic exceptions in AgentClientError."""
        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=RuntimeError("unexpected error"),
        ):
            client = ClaudeMcpAgentClient()
            with pytest.raises(AgentClientError, match="unexpected error"):
                client.run_agent("Do something", [])

    def test_constructor_accepts_base_workdir(self, tmp_path: Path) -> None:
        """Should accept base_workdir in constructor."""
        client = ClaudeMcpAgentClient(base_workdir=tmp_path)
        assert client.base_workdir == tmp_path

    def test_constructor_defaults_to_no_workdir(self) -> None:
        """Should default to no base_workdir."""
        client = ClaudeMcpAgentClient()
        assert client.base_workdir is None

    def test_create_workdir_creates_unique_directory(self, tmp_path: Path) -> None:
        """Should create unique directory with issue key and timestamp."""
        client = ClaudeMcpAgentClient(base_workdir=tmp_path)
        workdir = client._create_workdir("DS-123")

        assert workdir.parent == tmp_path
        assert workdir.name.startswith("DS-123_")
        assert workdir.exists()
        assert workdir.is_dir()

    def test_create_workdir_creates_base_if_missing(self, tmp_path: Path) -> None:
        """Should create base directory if it doesn't exist."""
        base_path = tmp_path / "nonexistent" / "base"
        client = ClaudeMcpAgentClient(base_workdir=base_path)
        workdir = client._create_workdir("DS-456")

        assert base_path.exists()
        assert workdir.exists()

    def test_create_workdir_raises_without_base_workdir(self) -> None:
        """Should raise AgentClientError if base_workdir not configured."""
        client = ClaudeMcpAgentClient()
        with pytest.raises(AgentClientError, match="base_workdir not configured"):
            client._create_workdir("DS-789")

    def test_run_agent_creates_workdir_when_configured(self, tmp_path: Path) -> None:
        """Should create workdir and pass cwd when base_workdir and issue_key provided."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient(base_workdir=tmp_path)
            client.run_agent("Do something", [], issue_key="DS-100")

        # Check that cwd was passed
        call_kwargs = mock.call_args[1]
        assert call_kwargs["cwd"] is not None
        assert "DS-100_" in call_kwargs["cwd"]

    def test_run_agent_no_workdir_without_base(self) -> None:
        """Should not create workdir when base_workdir not configured."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient()
            client.run_agent("Do something", [], issue_key="DS-200")

        call_kwargs = mock.call_args[1]
        assert call_kwargs["cwd"] is None

    def test_run_agent_no_workdir_without_issue_key(self, tmp_path: Path) -> None:
        """Should not create workdir when issue_key not provided."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done") as mock:
            client = ClaudeMcpAgentClient(base_workdir=tmp_path)
            client.run_agent("Do something", [])

        call_kwargs = mock.call_args[1]
        assert call_kwargs["cwd"] is None


class TestClaudeMcpAgentClientStreaming:
    """Tests for streaming log functionality in ClaudeMcpAgentClient."""

    def test_constructor_accepts_log_base_dir(self, tmp_path: Path) -> None:
        """Should accept log_base_dir in constructor."""
        client = ClaudeMcpAgentClient(log_base_dir=tmp_path)
        assert client.log_base_dir == tmp_path

    def test_constructor_defaults_to_no_log_dir(self) -> None:
        """Should default to no log_base_dir."""
        client = ClaudeMcpAgentClient()
        assert client.log_base_dir is None

    def test_streaming_enabled_when_all_params_provided(self, tmp_path: Path) -> None:
        """Should use streaming logs when all required params provided."""
        log_dir = tmp_path / "logs"
        work_dir = tmp_path / "work"

        with patch("sentinel.mcp_clients._run_claude", return_value="done"):
            client = ClaudeMcpAgentClient(
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

    def test_streaming_log_contains_output(self, tmp_path: Path) -> None:
        """Should stream output to log file."""
        log_dir = tmp_path / "logs"

        # Mock _run_claude to output some text
        with patch("sentinel.mcp_clients._run_claude", return_value="Agent completed task"):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
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

    def test_streaming_disabled_without_log_base_dir(self, tmp_path: Path) -> None:
        """Should not create streaming logs when log_base_dir not set."""
        with patch("sentinel.mcp_clients._run_claude", return_value="done"):
            client = ClaudeMcpAgentClient()
            client.run_agent(
                "Test",
                [],
                issue_key="DS-123",
                orchestration_name="test",
            )

        # No logs directory should be created in tmp_path
        assert not (tmp_path / "test").exists()

    def test_streaming_disabled_without_issue_key(self, tmp_path: Path) -> None:
        """Should not create streaming logs without issue_key."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients._run_claude", return_value="done"):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
            client.run_agent(
                "Test",
                [],
                orchestration_name="test",
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_disabled_without_orchestration_name(self, tmp_path: Path) -> None:
        """Should not create streaming logs without orchestration_name."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients._run_claude", return_value="done"):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
            client.run_agent(
                "Test",
                [],
                issue_key="DS-123",
            )

        # Logs directory should not be created
        assert not log_dir.exists()

    def test_streaming_log_finalized_on_success(self, tmp_path: Path) -> None:
        """Should finalize log with COMPLETED status on success."""
        log_dir = tmp_path / "logs"

        with patch("sentinel.mcp_clients._run_claude", return_value="done"):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
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

    def test_streaming_log_finalized_on_timeout(self, tmp_path: Path) -> None:
        """Should finalize log with TIMEOUT status on timeout."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.TimeoutExpired("cmd", 60),
        ):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
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

    def test_streaming_log_finalized_on_error(self, tmp_path: Path) -> None:
        """Should finalize log with FAILED status on process error."""
        log_dir = tmp_path / "logs"

        with patch(
            "sentinel.mcp_clients._run_claude",
            side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error msg"),
        ):
            client = ClaudeMcpAgentClient(log_base_dir=log_dir)
            with pytest.raises(AgentClientError):
                client.run_agent(
                    "Test",
                    [],
                    issue_key="DS-123",
                    orchestration_name="test",
                )

        log_files = list((log_dir / "test").glob("*.log"))
        content = log_files[0].read_text()
        assert "Status:         FAILED" in content


class TestOutputCallback:
    """Tests for output_callback parameter in _run_claude."""

    def test_callback_called_with_output(self) -> None:
        """Should call callback with each line of output."""
        mock_process = create_mock_process("line1\nline2\nline3\n")

        captured_lines: list[str] = []

        def callback(line: str) -> None:
            captured_lines.append(line)

        with patch("subprocess.Popen", return_value=mock_process):
            _run_claude("test", output_callback=callback)

        # Callback should have been called with the lines
        assert len(captured_lines) == 3
        assert "line1" in captured_lines[0]
        assert "line2" in captured_lines[1]
        assert "line3" in captured_lines[2]

    def test_callback_receives_stderr_with_prefix(self) -> None:
        """Should prefix stderr lines with [stderr]."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0
        mock_process.stdout = io.StringIO("stdout\n")
        mock_process.stderr = io.StringIO("error\n")
        mock_process.communicate.return_value = ("stdout\n", "error\n")

        captured_lines: list[str] = []

        def callback(line: str) -> None:
            captured_lines.append(line)

        with patch("subprocess.Popen", return_value=mock_process):
            _run_claude("test", output_callback=callback)

        # Should have both stdout and stderr lines
        stderr_lines = [line for line in captured_lines if "[stderr]" in line]
        assert len(stderr_lines) == 1
        assert "error" in stderr_lines[0]

    def test_no_callback_is_fine(self) -> None:
        """Should work without callback."""
        mock_process = create_mock_process("output\n")

        with patch("subprocess.Popen", return_value=mock_process):
            result = _run_claude("test")

        assert result == "output"
