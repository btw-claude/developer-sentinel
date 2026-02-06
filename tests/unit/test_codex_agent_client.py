"""Unit tests for CodexAgentClient.

Tests for the Codex CLI agent client implementation.

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
    CodexAgentClient,
    create_default_factory,
)
from sentinel.config import CodexConfig, Config
from tests.helpers import make_config


@pytest.fixture
def codex_config() -> Config:
    """Provide a standard Config with default codex settings.

    Returns a Config with codex_path="/usr/local/bin/codex" and
    codex_default_model="o3-mini". Tests that need custom
    codex parameters should use make_config() directly.
    """
    return make_config(
        codex_path="/usr/local/bin/codex",
        codex_default_model="o3-mini",
    )


@pytest.fixture
def test_config() -> Config:
    """Create a default test Config with standard codex settings."""
    return make_config(
        agent_workdir=Path("/tmp/test-workdir"),
        agent_logs_dir=Path("/tmp/test-logs"),
        codex_path="/usr/local/bin/codex",
        codex_default_model="o3-mini",
    )


@pytest.fixture
def mock_codex_subprocess():
    """Set up standard mocks for codex subprocess execution.

    Yields a tuple of (mock_mkstemp, mock_output_path, mock_run, mock_path_cls)
    with common setup for tempfile.mkstemp, os.close, Path, and subprocess.run.
    The output file defaults to existing with content "Agent response".
    Tests can customize mock_output_path and mock_run as needed.
    """
    with (
        patch("sentinel.agent_clients.codex.tempfile.mkstemp") as mock_mkstemp,
        patch("sentinel.agent_clients.codex.os.close") as _mock_os_close,
        patch("sentinel.agent_clients.codex.subprocess.run") as mock_run,
        patch("sentinel.agent_clients.codex.Path") as mock_path_cls,
    ):
        mock_mkstemp.return_value = (3, "/tmp/codex_output.txt")

        mock_output_path = MagicMock()
        mock_output_path.exists.return_value = True
        mock_output_path.stat.return_value = MagicMock(st_size=14)
        mock_output_path.read_text.return_value = "Agent response"
        mock_output_path.unlink = MagicMock()
        mock_path_cls.return_value = mock_output_path

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Agent response",
            stderr="",
        )

        yield mock_mkstemp, mock_output_path, mock_run, mock_path_cls


class TestCodexAgentClientInit:
    """Tests for CodexAgentClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config values."""
        config = Config(codex=CodexConfig())
        client = CodexAgentClient(config)

        assert client.config is config
        assert client.agent_type == "codex"
        assert client._codex_path == "codex"
        assert client._default_model is None

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config values."""
        config = make_config(
            codex_path="/custom/path/codex",
            codex_default_model="o3-mini",
        )
        client = CodexAgentClient(config)

        assert client._codex_path == "/custom/path/codex"
        assert client._default_model == "o3-mini"

    def test_init_with_workdir_and_logs(self, codex_config: Config) -> None:
        """Test initialization with workdir and log directories."""
        base_workdir = Path("/tmp/work")
        log_base_dir = Path("/tmp/logs")

        client = CodexAgentClient(
            codex_config,
            base_workdir=base_workdir,
            log_base_dir=log_base_dir,
        )

        assert client.base_workdir == base_workdir
        assert client.log_base_dir == log_base_dir

    def test_init_using_fixture(self, test_config: Config) -> None:
        """Test initialization using pytest fixture."""
        client = CodexAgentClient(test_config)

        assert client.config is test_config
        assert client.agent_type == "codex"
        assert client._codex_path == "/usr/local/bin/codex"
        assert client._default_model == "o3-mini"


class TestCodexAgentClientBuildCommand:
    """Tests for command building."""

    def test_build_command_basic(self, codex_config: Config) -> None:
        """Test building a basic command."""
        client = CodexAgentClient(codex_config)

        cmd = client._build_command("test prompt", "/tmp/output.txt")

        assert cmd[0] == "/usr/local/bin/codex"
        assert "exec" in cmd
        assert "test prompt" in cmd
        assert "--full-auto" in cmd
        assert "--output-last-message" in cmd
        assert "/tmp/output.txt" in cmd
        assert "--model" in cmd
        assert "o3-mini" in cmd

    def test_build_command_with_model_override(self) -> None:
        """Test building command with model override."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="default-model",
        )
        client = CodexAgentClient(config)

        cmd = client._build_command("test", "/tmp/out.txt", model="override-model")

        assert "override-model" in cmd

    def test_build_command_with_workdir(self, codex_config: Config) -> None:
        """Test building command with working directory."""
        client = CodexAgentClient(codex_config)

        cmd = client._build_command("test", "/tmp/out.txt", workdir=Path("/my/workdir"))

        assert "--cd" in cmd
        assert "/my/workdir" in cmd

    def test_build_command_no_model_when_none(self) -> None:
        """Test that --model flag is omitted when model is None/empty."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="",
        )
        client = CodexAgentClient(config)

        cmd = client._build_command("test", "/tmp/out.txt")

        assert "--model" not in cmd

    def test_build_command_no_workdir_when_none(self, codex_config: Config) -> None:
        """Test that --cd flag is omitted when workdir is None."""
        client = CodexAgentClient(codex_config)

        cmd = client._build_command("test", "/tmp/out.txt")

        assert "--cd" not in cmd


class TestCodexAgentClientRunAgent:
    """Tests for run_agent method.

    All tests use asyncio.run() to call the async run_agent method.
    """

    def test_run_agent_success(
        self,
        mock_codex_subprocess: tuple,
        codex_config: Config,
    ) -> None:
        """Test successful agent execution."""
        _mock_tmp, _mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Agent response"
        assert result.workdir is None
        mock_run.assert_called_once()

    def test_run_agent_with_context(
        self,
        mock_codex_subprocess: tuple,
        codex_config: Config,
    ) -> None:
        """Test agent execution with context dict."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=8)
        mock_output_path.read_text.return_value = "Response"
        mock_run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(
            client.run_agent(
                "test prompt",
                context={"key": "value", "another": "context"},
            )
        )

        # Verify the prompt was augmented with context
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        # The prompt is the third element (after codex, exec)
        full_prompt = cmd[2]
        assert "Context:" in full_prompt
        assert "- key: value" in full_prompt
        assert "- another: context" in full_prompt

    def test_run_agent_with_timeout(
        self,
        mock_codex_subprocess: tuple,
        codex_config: Config,
    ) -> None:
        """Test agent execution with explicit timeout."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=8)
        mock_output_path.read_text.return_value = "Response"
        mock_run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(client.run_agent("prompt", timeout_seconds=120))

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_run_agent_uses_config_subprocess_timeout(
        self,
        mock_codex_subprocess: tuple,
    ) -> None:
        """Test that run_agent falls back to subprocess_timeout from config.

        When timeout_seconds is not provided and config.execution.subprocess_timeout
        is set, the config value should be used as the effective timeout.
        """
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=8)
        mock_output_path.read_text.return_value = "Response"
        mock_run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
            subprocess_timeout=120.0,
        )
        client = CodexAgentClient(config)

        asyncio.run(client.run_agent("prompt"))

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_run_agent_with_model_override(
        self,
        mock_codex_subprocess: tuple,
    ) -> None:
        """Test agent execution with model override."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=8)
        mock_output_path.read_text.return_value = "Response"
        mock_run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="default-model",
        )
        client = CodexAgentClient(config)

        asyncio.run(client.run_agent("prompt", model="override-model"))

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "override-model"

    def test_run_agent_creates_workdir(
        self,
        mock_codex_subprocess: tuple,
        tmp_path: Path,
        codex_config: Config,
    ) -> None:
        """Test that run_agent creates a working directory when configured."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=8)
        mock_output_path.read_text.return_value = "Response"
        mock_run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config, base_workdir=tmp_path)

        result = asyncio.run(client.run_agent("prompt", issue_key="TEST-123"))

        assert result.workdir is not None
        assert result.workdir.exists()
        assert "TEST-123" in result.workdir.name
        # Verify --cd flag was passed in command
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--cd" in cmd

    @patch("sentinel.agent_clients.codex.os.close")
    @patch("sentinel.agent_clients.codex.tempfile.mkstemp")
    @patch("sentinel.agent_clients.codex.subprocess.run")
    def test_run_agent_timeout_raises_error(
        self,
        mock_run: MagicMock,
        mock_mkstemp: MagicMock,
        _mock_os_close: MagicMock,
        codex_config: Config,
    ) -> None:
        """Test that timeout raises AgentTimeoutError."""
        mock_mkstemp.return_value = (3, "/tmp/codex_output.txt")
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=60)

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentTimeoutError) as exc_info:
            asyncio.run(client.run_agent("prompt", timeout_seconds=60))

        assert "timed out after 60s" in str(exc_info.value)

    def test_run_agent_nonzero_exit_raises_error(
        self,
        mock_codex_subprocess: tuple,
        codex_config: Config,
    ) -> None:
        """Test that non-zero exit code raises AgentClientError."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.exists.return_value = False
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Codex CLI execution failed" in str(exc_info.value)
        assert "something went wrong" in str(exc_info.value)

    @patch("sentinel.agent_clients.codex.os.close")
    @patch("sentinel.agent_clients.codex.tempfile.mkstemp")
    @patch("sentinel.agent_clients.codex.subprocess.run")
    def test_run_agent_file_not_found_raises_error(
        self,
        mock_run: MagicMock,
        mock_mkstemp: MagicMock,
        _mock_os_close: MagicMock,
    ) -> None:
        """Test that FileNotFoundError raises AgentClientError."""
        mock_mkstemp.return_value = (3, "/tmp/codex_output.txt")
        mock_run.side_effect = FileNotFoundError("codex not found")

        config = make_config(
            codex_path="/nonexistent/codex",
            codex_default_model="o3-mini",
        )
        client = CodexAgentClient(config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Codex CLI executable not found" in str(exc_info.value)

    @patch("sentinel.agent_clients.codex.os.close")
    @patch("sentinel.agent_clients.codex.tempfile.mkstemp")
    @patch("sentinel.agent_clients.codex.subprocess.run")
    def test_run_agent_os_error_raises_error(
        self,
        mock_run: MagicMock,
        mock_mkstemp: MagicMock,
        _mock_os_close: MagicMock,
        codex_config: Config,
    ) -> None:
        """Test that OSError raises AgentClientError."""
        mock_mkstemp.return_value = (3, "/tmp/codex_output.txt")
        mock_run.side_effect = OSError("Permission denied")

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Failed to execute Codex CLI" in str(exc_info.value)

    def test_run_agent_falls_back_to_stdout(
        self,
        mock_codex_subprocess: tuple,
        codex_config: Config,
    ) -> None:
        """Test that response falls back to stdout when output file is empty."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        # Output file exists but is empty
        mock_output_path.stat.return_value = MagicMock(st_size=0)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Stdout response",
            stderr="",
        )

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("prompt"))

        assert result.response == "Stdout response"

    def test_run_agent_uses_fixture(
        self,
        mock_codex_subprocess: tuple,
        test_config: Config,
    ) -> None:
        """Test run_agent using pytest fixture for config."""
        _mock_tmp, mock_output_path, mock_run, _mock_path_cls = mock_codex_subprocess
        mock_output_path.stat.return_value = MagicMock(st_size=20)
        mock_output_path.read_text.return_value = "Fixture test response"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Fixture test response",
            stderr="",
        )

        client = CodexAgentClient(test_config)
        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Fixture test response"
        mock_run.assert_called_once()


class TestCodexAgentClientWorkdir:
    """Tests for working directory creation."""

    def test_create_workdir_success(
        self, tmp_path: Path, codex_config: Config
    ) -> None:
        """Test successful workdir creation."""
        client = CodexAgentClient(codex_config, base_workdir=tmp_path)

        workdir = client._create_workdir("TEST-456")

        assert workdir.exists()
        assert workdir.is_dir()
        assert "TEST-456" in workdir.name
        assert workdir.parent == tmp_path

    def test_create_workdir_no_base_raises_error(
        self, codex_config: Config
    ) -> None:
        """Test that _create_workdir raises error when base_workdir is None."""
        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            client._create_workdir("TEST-789")

        assert "base_workdir not configured" in str(exc_info.value)


class TestCodexAgentClientFactoryIntegration:
    """Tests for factory integration."""

    def test_factory_creates_codex_client(self) -> None:
        """Test that create_default_factory registers codex builder."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        factory = create_default_factory(config)

        assert "codex" in factory.registered_types

    def test_factory_creates_codex_instance(self) -> None:
        """Test that factory creates CodexAgentClient instance."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        factory = create_default_factory(config)

        client = factory.create("codex", config)

        assert isinstance(client, CodexAgentClient)
        assert client.agent_type == "codex"
        assert client.config is config

    def test_factory_codex_client_has_correct_settings(self) -> None:
        """Test that factory-created client has correct settings from config."""
        config = make_config(
            codex_path="/custom/codex",
            codex_default_model="custom-model",
        )
        factory = create_default_factory(config)

        client = factory.create("codex", config)

        assert isinstance(client, CodexAgentClient)
        assert client._codex_path == "/custom/codex"
        assert client._default_model == "custom-model"
        assert client.base_workdir == config.execution.agent_workdir
        assert client.log_base_dir == config.execution.agent_logs_dir
