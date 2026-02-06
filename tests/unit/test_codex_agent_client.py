"""Unit tests for CodexAgentClient.

Tests for the Codex CLI agent client implementation.
Test fixture uses CodexSubprocessMocks NamedTuple for self-documenting
mock access instead of positional tuple destructuring.

Tests use asyncio.run() for async test support.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest

from sentinel.agent_clients import (
    AgentClientError,
    AgentTimeoutError,
    CodexAgentClient,
    create_default_factory,
)
from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from sentinel.config import CodexConfig, Config
from tests.helpers import make_config


class CodexSubprocessMocks(NamedTuple):
    """Container for subprocess mocks used in codex agent tests.

    Using a NamedTuple makes fixture return values self-documenting
    compared to positional tuple destructuring.
    """

    tmpdir: MagicMock
    output_path: MagicMock
    run: MagicMock
    path_cls: MagicMock


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
def mock_codex_subprocess() -> CodexSubprocessMocks:
    """Set up standard mocks for codex subprocess execution.

    Yields a CodexSubprocessMocks NamedTuple with tmpdir, output_path, run,
    and path_cls mocks. Common setup for tempfile.TemporaryDirectory, Path,
    and subprocess.run. The output file defaults to existing with content
    "Agent response". Tests can customize output_path and run as needed.
    """
    with (
        patch("sentinel.agent_clients.codex.tempfile.TemporaryDirectory") as mock_tmpdir_cls,
        patch("sentinel.agent_clients.codex.subprocess.run") as mock_run,
        patch("sentinel.agent_clients.codex.Path") as mock_path_cls,
    ):
        mock_tmpdir_ctx = MagicMock()
        mock_tmpdir_ctx.__enter__ = MagicMock(return_value="/tmp/codex_tmpdir")
        mock_tmpdir_ctx.__exit__ = MagicMock(return_value=False)
        mock_tmpdir_cls.return_value = mock_tmpdir_ctx

        mock_output_path = MagicMock()
        mock_output_path.exists.return_value = True
        mock_output_path.stat.return_value = MagicMock(st_size=14)
        mock_output_path.read_text.return_value = "Agent response"
        mock_path_cls.return_value = mock_output_path
        # Make Path(tmp_dir) / "codex_output.txt" work
        mock_path_cls.__truediv__ = MagicMock(return_value=mock_output_path)

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Agent response",
            stderr="",
        )

        yield CodexSubprocessMocks(
            tmpdir=mock_tmpdir_cls,
            output_path=mock_output_path,
            run=mock_run,
            path_cls=mock_path_cls,
        )


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

    def test_init_logs_info_when_codex_path_defaults(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that INFO log is emitted when codex_path is not configured."""
        import logging

        config = Config(codex=CodexConfig())
        with caplog.at_level(logging.INFO, logger="sentinel.agent_clients.codex"):
            CodexAgentClient(config)

        assert any("codex_path not configured" in record.message for record in caplog.records)
        assert any("PATH lookup" in record.message for record in caplog.records)

    def test_init_no_info_log_when_codex_path_set(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that INFO log is NOT emitted when codex_path is explicitly set."""
        import logging

        config = make_config(codex_path="/usr/local/bin/codex")
        with caplog.at_level(logging.INFO, logger="sentinel.agent_clients.codex"):
            CodexAgentClient(config)

        assert not any("codex_path not configured" in record.message for record in caplog.records)


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
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test successful agent execution."""
        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Agent response"
        assert result.workdir is None
        mock_codex_subprocess.run.assert_called_once()

    def test_run_agent_with_context(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test agent execution with context dict."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(
            client.run_agent(
                "test prompt",
                context={"key": "value", "another": "context"},
            )
        )

        # Verify the prompt was augmented with context
        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        # The prompt is the third element (after codex, exec)
        full_prompt = cmd[2]
        assert "Context:" in full_prompt
        assert "- key: value" in full_prompt
        assert "- another: context" in full_prompt

    def test_run_agent_with_timeout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test agent execution with explicit timeout."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(client.run_agent("prompt", timeout_seconds=120))

        mock_codex_subprocess.run.assert_called_once()
        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_run_agent_uses_config_subprocess_timeout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test that run_agent falls back to subprocess_timeout from config.

        When timeout_seconds is not provided and config.execution.subprocess_timeout
        is set, the config value should be used as the effective timeout.
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
            subprocess_timeout=120.0,
        )
        client = CodexAgentClient(config)

        asyncio.run(client.run_agent("prompt"))

        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] == 120

    def test_run_agent_no_timeout_when_subprocess_timeout_zero(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test that no timeout is applied when subprocess_timeout is zero.

        When timeout_seconds is not provided and config.execution.subprocess_timeout
        is 0, the effective timeout should remain None (no timeout).
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
            subprocess_timeout=0,
        )
        client = CodexAgentClient(config)

        asyncio.run(client.run_agent("prompt"))

        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] is None

    def test_run_agent_with_model_override(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test agent execution with model override."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="default-model",
        )
        client = CodexAgentClient(config)

        asyncio.run(client.run_agent("prompt", model="override-model"))

        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "override-model"

    def test_run_agent_creates_workdir(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        tmp_path: Path,
        codex_config: Config,
    ) -> None:
        """Test that run_agent creates a working directory when configured."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config, base_workdir=tmp_path)

        result = asyncio.run(client.run_agent("prompt", issue_key="TEST-123"))

        assert result.workdir is not None
        assert result.workdir.exists()
        assert "TEST-123" in result.workdir.name
        # Verify --cd flag was passed in command
        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        assert "--cd" in cmd

    def test_run_agent_timeout_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that timeout raises AgentTimeoutError."""
        mock_codex_subprocess.run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=60)

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentTimeoutError) as exc_info:
            asyncio.run(client.run_agent("prompt", timeout_seconds=60))

        assert "timed out after 60s" in str(exc_info.value)

    def test_run_agent_nonzero_exit_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that non-zero exit code raises AgentClientError."""
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Codex CLI execution failed" in str(exc_info.value)
        assert "something went wrong" in str(exc_info.value)

    def test_run_agent_file_not_found_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that FileNotFoundError raises AgentClientError."""
        mock_codex_subprocess.run.side_effect = FileNotFoundError("codex not found")

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Codex CLI executable not found" in str(exc_info.value)

    def test_run_agent_os_error_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that OSError raises AgentClientError."""
        mock_codex_subprocess.run.side_effect = OSError("Permission denied")

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "Failed to execute Codex CLI" in str(exc_info.value)

    def test_run_agent_falls_back_to_stdout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that response falls back to stdout when output file is empty."""
        # Output file exists but is empty
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=0)
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="Stdout response",
            stderr="",
        )

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("prompt"))

        assert result.response == "Stdout response"

    def test_run_agent_uses_fixture(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        test_config: Config,
    ) -> None:
        """Test run_agent using pytest fixture for config."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=20)
        mock_codex_subprocess.output_path.read_text.return_value = "Fixture test response"
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="Fixture test response",
            stderr="",
        )

        client = CodexAgentClient(test_config)
        result = asyncio.run(client.run_agent("test prompt"))

        assert result.response == "Fixture test response"
        mock_codex_subprocess.run.assert_called_once()

    def test_run_agent_success_output_file_missing(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that run_agent falls back to stdout when output file does not exist.

        Covers the edge case where returncode=0 but the output file was never
        created (exists() returns False). The response should fall back to stdout.
        """
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="Stdout fallback",
            stderr="",
        )

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("prompt"))

        assert result.response == "Stdout fallback"
        mock_codex_subprocess.output_path.read_text.assert_not_called()

    def test_run_agent_empty_output_and_empty_stdout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test behavior when both output file and stdout are empty.

        Documents that when the output file is empty (st_size=0) and stdout
        is also empty, the response is an empty string.
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=0)
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client.run_agent("prompt"))

        assert result.response == ""


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


class TestCodexCircuitBreaker:
    """Tests for CodexAgentClient circuit breaker integration."""

    def test_init_creates_default_circuit_breaker(self, codex_config: Config) -> None:
        """Test that init creates a default circuit breaker when none is provided."""
        client = CodexAgentClient(codex_config)

        assert client.circuit_breaker is not None
        assert client.circuit_breaker.service_name == "codex"
        assert client.circuit_breaker.state == CircuitState.CLOSED

    def test_init_accepts_injected_circuit_breaker(self, codex_config: Config) -> None:
        """Test that init uses the injected circuit breaker."""
        cb = CircuitBreaker(
            service_name="codex-custom",
            config=CircuitBreakerConfig(failure_threshold=10),
        )
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        assert client.circuit_breaker is cb
        assert client.circuit_breaker.service_name == "codex-custom"

    def test_get_circuit_breaker_status(self, codex_config: Config) -> None:
        """Test that get_circuit_breaker_status returns status dict."""
        client = CodexAgentClient(codex_config)

        status = client.get_circuit_breaker_status()

        assert isinstance(status, dict)
        assert status["service_name"] == "codex"
        assert status["state"] == "closed"
        assert "config" in status
        assert "metrics" in status

    def test_circuit_breaker_property(self, codex_config: Config) -> None:
        """Test that the circuit_breaker property returns the circuit breaker."""
        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        assert client.circuit_breaker is cb

    def test_run_agent_rejects_when_circuit_open(
        self,
        codex_config: Config,
    ) -> None:
        """Test that run_agent raises error when circuit breaker is open."""
        cb = CircuitBreaker(
            service_name="codex",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        # Force circuit breaker to open state
        cb.record_failure(Exception("test failure"))

        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "circuit breaker is open" in str(exc_info.value)
        assert "codex" in str(exc_info.value).lower()

    def test_run_agent_circuit_open_skips_workdir_creation(
        self,
        tmp_path: Path,
        codex_config: Config,
    ) -> None:
        """Test that circuit breaker check happens before workdir creation.

        When the circuit breaker is open, run_agent() should reject immediately
        without creating a working directory. This avoids unnecessary disk I/O
        when the service is unavailable (DS-661).
        """
        cb = CircuitBreaker(
            service_name="codex",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(Exception("test failure"))

        client = CodexAgentClient(codex_config, base_workdir=tmp_path, circuit_breaker=cb)

        with pytest.raises(AgentClientError, match="circuit breaker is open"):
            asyncio.run(client.run_agent("prompt", issue_key="TEST-123"))

        # Verify no workdir was created - the directory should be empty
        assert list(tmp_path.iterdir()) == []

    def test_run_agent_records_success(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that successful execution records success on circuit breaker."""
        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        asyncio.run(client.run_agent("test prompt"))

        assert cb.metrics.successful_calls == 1
        assert cb.metrics.failed_calls == 0

    def test_run_agent_records_failure_on_nonzero_exit(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that non-zero exit code records failure on circuit breaker."""
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentClientError):
            asyncio.run(client.run_agent("prompt"))

        assert cb.metrics.failed_calls == 1
        assert cb.metrics.successful_calls == 0

    def test_run_agent_records_failure_on_timeout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that timeout records failure on circuit breaker."""
        mock_codex_subprocess.run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=60)

        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentTimeoutError):
            asyncio.run(client.run_agent("prompt", timeout_seconds=60))

        assert cb.metrics.failed_calls == 1

    def test_run_agent_records_failure_on_file_not_found(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that FileNotFoundError records failure on circuit breaker."""
        mock_codex_subprocess.run.side_effect = FileNotFoundError("codex not found")

        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentClientError):
            asyncio.run(client.run_agent("prompt"))

        assert cb.metrics.failed_calls == 1

    def test_run_agent_records_failure_on_os_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that OSError records failure on circuit breaker."""
        mock_codex_subprocess.run.side_effect = OSError("Permission denied")

        cb = CircuitBreaker(service_name="codex")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentClientError):
            asyncio.run(client.run_agent("prompt"))

        assert cb.metrics.failed_calls == 1

    def test_repeated_failures_open_circuit(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that repeated failures cause the circuit to open."""
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: service unavailable",
        )

        cb = CircuitBreaker(
            service_name="codex",
            config=CircuitBreakerConfig(failure_threshold=2),
        )
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        # First failure
        with pytest.raises(AgentClientError):
            asyncio.run(client.run_agent("prompt"))

        # Second failure - should open the circuit
        with pytest.raises(AgentClientError):
            asyncio.run(client.run_agent("prompt"))

        assert cb.state == CircuitState.OPEN

        # Third call should be rejected by circuit breaker
        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client.run_agent("prompt"))

        assert "circuit breaker is open" in str(exc_info.value)


class TestCodexFactoryCircuitBreakerIntegration:
    """Tests for factory integration with circuit breaker."""

    def test_factory_creates_codex_with_circuit_breaker_from_registry(self) -> None:
        """Test that factory injects circuit breaker from registry."""
        from sentinel.circuit_breaker import CircuitBreakerRegistry

        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        registry = CircuitBreakerRegistry()
        factory = create_default_factory(config, circuit_breaker_registry=registry)

        client = factory.create("codex", config)

        assert isinstance(client, CodexAgentClient)
        # The circuit breaker should be from the registry
        assert client.circuit_breaker is registry.get("codex")
        assert client.circuit_breaker.service_name == "codex"

    def test_factory_creates_codex_without_circuit_breaker_registry(self) -> None:
        """Test that factory creates codex with default circuit breaker when no registry."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        factory = create_default_factory(config)

        client = factory.create("codex", config)

        assert isinstance(client, CodexAgentClient)
        # Should have a default circuit breaker
        assert client.circuit_breaker is not None
        assert client.circuit_breaker.service_name == "codex"
