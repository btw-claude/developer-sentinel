"""Unit tests for CursorAgentClient.

Tests for the Cursor CLI agent client implementation.
Test fixture improvements and additional validation tests.

Tests use pytest-asyncio for async test support.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel.agent_clients import (
    AgentClientError,
    AgentTimeoutError,
    CursorAgentClient,
    CursorMode,
    create_default_factory,
)
from sentinel.config import Config, CursorConfig
from tests.helpers import make_config


@pytest.fixture
def cursor_config() -> Config:
    """Provide a standard Config with default cursor settings.

    Returns a Config with cursor_path="/usr/local/bin/cursor" and
    cursor_default_model="claude-3-sonnet". Tests that need custom
    cursor parameters should use make_config() directly.
    """
    return make_config(
        cursor_path="/usr/local/bin/cursor",
        cursor_default_model="claude-3-sonnet",
    )


@pytest.fixture
def test_config() -> Config:
    """Create a default test Config with standard cursor settings."""
    return make_config(
        agent_workdir=Path("/tmp/test-workdir"),
        agent_logs_dir=Path("/tmp/test-logs"),
        cursor_path="/usr/local/bin/cursor",
        cursor_default_model="claude-3-sonnet",
        cursor_default_mode="agent",
    )


@pytest.fixture
def test_config_plan_mode() -> Config:
    """Create a test Config with plan mode."""
    return make_config(
        agent_workdir=Path("/tmp/test-workdir"),
        agent_logs_dir=Path("/tmp/test-logs"),
        cursor_path="/usr/local/bin/cursor",
        cursor_default_model="claude-3-sonnet",
        cursor_default_mode="plan",
    )


class TestCursorMode:
    """Tests for CursorMode enum."""

    def test_mode_values(self) -> None:
        """Test that CursorMode has expected values."""
        assert CursorMode.AGENT.value == "agent"
        assert CursorMode.PLAN.value == "plan"
        assert CursorMode.ASK.value == "ask"

    def test_from_string_valid_modes(self) -> None:
        """Test from_string with valid mode strings."""
        assert CursorMode.from_string("agent") == CursorMode.AGENT
        assert CursorMode.from_string("plan") == CursorMode.PLAN
        assert CursorMode.from_string("ask") == CursorMode.ASK

    def test_from_string_case_insensitive(self) -> None:
        """Test that from_string is case-insensitive."""
        assert CursorMode.from_string("AGENT") == CursorMode.AGENT
        assert CursorMode.from_string("Plan") == CursorMode.PLAN
        assert CursorMode.from_string("ASK") == CursorMode.ASK

    def test_from_string_invalid_mode_raises(self) -> None:
        """Test that from_string raises ValueError for invalid modes."""
        with pytest.raises(ValueError) as exc_info:
            CursorMode.from_string("invalid")

        assert "Invalid Cursor mode: 'invalid'" in str(exc_info.value)
        assert "Valid modes:" in str(exc_info.value)

    def test_from_string_empty_string_raises(self) -> None:
        """Test that from_string raises ValueError for empty string."""
        with pytest.raises(ValueError) as exc_info:
            CursorMode.from_string("")

        assert "Invalid Cursor mode" in str(exc_info.value)


class TestCursorAgentClientInit:
    """Tests for CursorAgentClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config values."""
        config = Config(cursor=CursorConfig(default_mode="agent"))
        client = CursorAgentClient(config)

        assert client.config is config
        assert client.agent_type == "cursor"
        assert client._cursor_path == "cursor"
        assert client._default_mode == CursorMode.AGENT

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config values."""
        config = make_config(
            cursor_path="/custom/path/cursor",
            cursor_default_model="gpt-4",
            cursor_default_mode="plan",
        )
        client = CursorAgentClient(config)

        assert client._cursor_path == "/custom/path/cursor"
        assert client._default_model == "gpt-4"
        assert client._default_mode == CursorMode.PLAN

    def test_init_with_workdir_and_logs(self, cursor_config: Config) -> None:
        """Test initialization with workdir and log directories."""
        base_workdir = Path("/tmp/work")
        log_base_dir = Path("/tmp/logs")

        client = CursorAgentClient(
            cursor_config,
            base_workdir=base_workdir,
            log_base_dir=log_base_dir,
        )

        assert client.base_workdir == base_workdir
        assert client.log_base_dir == log_base_dir

    def test_init_with_empty_cursor_mode_raises_error(self) -> None:
        """Test that initialization raises AgentClientError when cursor_default_mode is empty."""
        config = Config(
            cursor=CursorConfig(
                path="/usr/local/bin/cursor",
                default_mode="",
            ),
        )

        with pytest.raises(AgentClientError) as exc_info:
            CursorAgentClient(config)

        assert "cursor_default_mode is not set" in str(exc_info.value)
        assert "SENTINEL_CURSOR_DEFAULT_MODE" in str(exc_info.value)

    def test_init_with_whitespace_cursor_mode_raises_error(self) -> None:
        """Test that initialization raises AgentClientError when cursor_default_mode is whitespace."""
        config = Config(
            cursor=CursorConfig(
                path="/usr/local/bin/cursor",
                default_mode="   ",
            ),
        )

        with pytest.raises(AgentClientError) as exc_info:
            CursorAgentClient(config)

        assert "cursor_default_mode is not set" in str(exc_info.value)

    def test_init_using_fixture(self, test_config: Config) -> None:
        """Test initialization using pytest fixture."""
        client = CursorAgentClient(test_config)

        assert client.config is test_config
        assert client.agent_type == "cursor"
        assert client._cursor_path == "/usr/local/bin/cursor"
        assert client._default_mode == CursorMode.AGENT


class TestCursorAgentClientBuildCommand:
    """Tests for command building."""

    def test_build_command_basic(self, cursor_config: Config) -> None:
        """Test building a basic command."""
        client = CursorAgentClient(cursor_config)

        cmd = client._build_command("test prompt")

        assert cmd[0] == "/usr/local/bin/cursor"
        assert "-p" in cmd
        assert "test prompt" in cmd
        assert "--output-format" in cmd
        assert "text" in cmd
        assert "--mode=agent" in cmd
        assert "--model" in cmd
        assert "claude-3-sonnet" in cmd

    def test_build_command_with_model_override(self) -> None:
        """Test building command with model override."""
        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="default-model",
        )
        client = CursorAgentClient(config)

        cmd = client._build_command("test", model="override-model")

        assert "override-model" in cmd

    def test_build_command_with_mode_override(self) -> None:
        """Test building command with mode override."""
        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="claude-3-sonnet",
            cursor_default_mode="agent",
        )
        client = CursorAgentClient(config)

        cmd = client._build_command("test", mode=CursorMode.PLAN)

        assert "--mode=plan" in cmd

    def test_build_command_no_model_when_none(self) -> None:
        """Test that --model flag is omitted when model is None/empty."""
        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="",
        )
        client = CursorAgentClient(config)

        cmd = client._build_command("test")

        assert "--model" not in cmd


class TestCursorAgentClientRunAgent:
    """Tests for run_agent method.

    All tests use asyncio.run() to call the async run_agent method.
    """

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_success(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test successful agent execution."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Agent response",
            stderr="",
        )

        client = CursorAgentClient(cursor_config)

        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Agent response"
        assert result.workdir is None
        mock_run.assert_called_once()

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_context(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test agent execution with context dict."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        client = CursorAgentClient(cursor_config)

        asyncio.run(
            client.run_agent(
                "test prompt",
                context={"key": "value", "another": "context"},
            )
        )

        # Verify the prompt was augmented with context
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        prompt_idx = cmd.index("-p") + 1
        full_prompt = cmd[prompt_idx]
        assert "Context:" in full_prompt
        assert "- key: value" in full_prompt
        assert "- another: context" in full_prompt

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_timeout(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test agent execution with timeout."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        client = CursorAgentClient(cursor_config)

        asyncio.run(client.run_agent("prompt", timeout_seconds=60))

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 60

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_model_override(self, mock_run: MagicMock) -> None:
        """Test agent execution with model override."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="default-model",
        )
        client = CursorAgentClient(config)

        asyncio.run(client.run_agent("prompt", model="override-model"))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "override-model"

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_creates_workdir(
        self, mock_run: MagicMock, tmp_path: Path, cursor_config: Config
    ) -> None:
        """Test that run_agent creates a working directory when configured."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        client = CursorAgentClient(cursor_config, base_workdir=tmp_path)

        result = asyncio.run(client.run_agent("prompt", issue_key="TEST-123"))

        assert result.workdir is not None
        assert result.workdir.exists()
        assert "TEST-123" in result.workdir.name
        # Verify subprocess was called with the workdir
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["cwd"] == str(result.workdir)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_timeout_raises_error(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test that timeout raises AgentTimeoutError."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="cursor", timeout=60)

        client = CursorAgentClient(cursor_config)

        with pytest.raises(AgentTimeoutError) as exc_info:
            asyncio.run(client.run_agent("prompt", timeout_seconds=60))

        assert "timed out after 60s" in str(exc_info.value)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_nonzero_exit_raises_error(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test that non-zero exit code raises AgentClientError."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        client = CursorAgentClient(cursor_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Cursor CLI execution failed" in str(exc_info.value)
        assert "something went wrong" in str(exc_info.value)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_file_not_found_raises_error(self, mock_run: MagicMock) -> None:
        """Test that FileNotFoundError raises AgentClientError."""
        mock_run.side_effect = FileNotFoundError("cursor not found")

        config = make_config(
            cursor_path="/nonexistent/cursor",
            cursor_default_model="claude-3-sonnet",
        )
        client = CursorAgentClient(config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Cursor CLI executable not found" in str(exc_info.value)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_os_error_raises_error(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test that OSError raises AgentClientError."""
        mock_run.side_effect = OSError("Permission denied")

        client = CursorAgentClient(cursor_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Failed to execute Cursor CLI" in str(exc_info.value)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_mode_override_enum(self, mock_run: MagicMock) -> None:
        """Test agent execution with mode override using CursorMode enum."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="claude-3-sonnet",
            cursor_default_mode="agent",
        )
        client = CursorAgentClient(config)

        asyncio.run(client.run_agent("prompt", mode=CursorMode.PLAN))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--mode=plan" in cmd

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_mode_override_string(self, mock_run: MagicMock) -> None:
        """Test agent execution with mode override using string."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="claude-3-sonnet",
            cursor_default_mode="agent",
        )
        client = CursorAgentClient(config)

        asyncio.run(client.run_agent("prompt", mode="ask"))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--mode=ask" in cmd

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_mode_override_case_insensitive(self, mock_run: MagicMock) -> None:
        """Test that mode override is case-insensitive when using string."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Response",
            stderr="",
        )

        config = make_config(
            cursor_path="/usr/local/bin/cursor",
            cursor_default_model="claude-3-sonnet",
            cursor_default_mode="agent",
        )
        client = CursorAgentClient(config)

        asyncio.run(client.run_agent("prompt", mode="PLAN"))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--mode=plan" in cmd

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_with_invalid_mode_override_raises_error(
        self, mock_run: MagicMock, cursor_config: Config
    ) -> None:
        """Test that invalid mode override string raises ValueError."""
        client = CursorAgentClient(cursor_config)

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(client.run_agent("prompt", mode="invalid_mode"))

        assert "Invalid Cursor mode: 'invalid_mode'" in str(exc_info.value)

    @patch("sentinel.agent_clients.cursor.subprocess.run")
    def test_run_agent_uses_fixture(
        self, mock_run: MagicMock, test_config: Config
    ) -> None:
        """Test run_agent using pytest fixture for config."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Fixture test response",
            stderr="",
        )

        client = CursorAgentClient(test_config)
        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Fixture test response"
        mock_run.assert_called_once()


class TestCursorAgentClientWorkdir:
    """Tests for working directory creation."""

    def test_create_workdir_success(
        self, tmp_path: Path, cursor_config: Config
    ) -> None:
        """Test successful workdir creation."""
        client = CursorAgentClient(cursor_config, base_workdir=tmp_path)

        workdir = client._create_workdir("TEST-456")

        assert workdir.exists()
        assert workdir.is_dir()
        assert "TEST-456" in workdir.name
        assert workdir.parent == tmp_path

    def test_create_workdir_no_base_raises_error(
        self, cursor_config: Config
    ) -> None:
        """Test that _create_workdir raises error when base_workdir is None."""
        client = CursorAgentClient(cursor_config)

        with pytest.raises(AgentClientError) as exc_info:
            client._create_workdir("TEST-789")

        assert "base_workdir not configured" in str(exc_info.value)


class TestCursorAgentClientFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_creates_cursor_client(
        self, cursor_config: Config
    ) -> None:
        """Test that create_default_factory registers cursor builder."""
        factory = create_default_factory(cursor_config)

        assert "cursor" in factory.registered_types

    def test_factory_creates_cursor_instance(
        self, cursor_config: Config
    ) -> None:
        """Test that factory creates CursorAgentClient instance."""
        factory = create_default_factory(cursor_config)

        client = factory.create("cursor", cursor_config)

        assert isinstance(client, CursorAgentClient)
        assert client.agent_type == "cursor"
        assert client.config is cursor_config

    def test_factory_cursor_client_has_correct_settings(self) -> None:
        """Test that factory-created client has correct settings from config."""
        config = make_config(
            cursor_path="/custom/cursor",
            cursor_default_model="custom-model",
            cursor_default_mode="plan",
        )
        factory = create_default_factory(config)

        client = factory.create("cursor", config)

        assert isinstance(client, CursorAgentClient)
        assert client._cursor_path == "/custom/cursor"
        assert client._default_model == "custom-model"
        assert client._default_mode == CursorMode.PLAN
        assert client.base_workdir == config.execution.agent_workdir
        assert client.log_base_dir == config.execution.agent_logs_dir
