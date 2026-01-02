"""Tests for MCP-based client implementations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel.executor import AgentClientError, AgentTimeoutError
from sentinel.mcp_clients import (
    ClaudeMcpAgentClient,
    JiraMcpClient,
    JiraMcpTagClient,
    _run_claude,
)
from sentinel.poller import JiraClientError
from sentinel.tag_manager import JiraTagClientError


class TestRunClaude:
    """Tests for the _run_claude helper function."""

    def test_runs_claude_command(self) -> None:
        """Should run claude CLI with correct arguments."""
        mock_result = MagicMock()
        mock_result.stdout = "test output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = _run_claude("test prompt")

        mock_run.assert_called_once_with(
            ["claude", "--print", "--output-format", "text", "-p", "test prompt"],
            capture_output=True,
            text=True,
            timeout=None,
            check=True,
            cwd=None,
        )
        assert result == "test output"

    def test_strips_whitespace(self) -> None:
        """Should strip whitespace from output."""
        mock_result = MagicMock()
        mock_result.stdout = "  output with spaces  \n"

        with patch("subprocess.run", return_value=mock_result):
            result = _run_claude("test")

        assert result == "output with spaces"

    def test_respects_timeout(self) -> None:
        """Should pass timeout to subprocess."""
        mock_result = MagicMock()
        mock_result.stdout = "output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _run_claude("test", timeout_seconds=120)

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_raises_on_timeout(self) -> None:
        """Should raise TimeoutExpired on timeout."""
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 60)),
            pytest.raises(subprocess.TimeoutExpired),
        ):
            _run_claude("test", timeout_seconds=60)

    def test_raises_on_error(self) -> None:
        """Should raise CalledProcessError on command failure."""
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            _run_claude("test")

    def test_includes_allowed_tools(self) -> None:
        """Should include --allowedTools flag when tools specified."""
        mock_result = MagicMock()
        mock_result.stdout = "output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _run_claude("test", allowed_tools=["mcp__jira-agent__search_issues"])

        cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        assert "mcp__jira-agent__search_issues" in cmd

    def test_multiple_allowed_tools_joined(self) -> None:
        """Should join multiple allowed tools with comma."""
        mock_result = MagicMock()
        mock_result.stdout = "output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _run_claude("test", allowed_tools=["tool1", "tool2", "tool3"])

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "tool1,tool2,tool3"

    def test_passes_cwd(self) -> None:
        """Should pass cwd to subprocess."""
        mock_result = MagicMock()
        mock_result.stdout = "output"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _run_claude("test", cwd="/path/to/workdir")

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == "/path/to/workdir"


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
