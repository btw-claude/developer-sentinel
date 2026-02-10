"""Unit tests for CodexAgentClient.

Tests for the Codex CLI agent client implementation.
Test fixture uses CodexSubprocessMocks NamedTuple for self-documenting
mock access instead of positional tuple destructuring.

Tests use asyncio.run() for async test support.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable
from unittest.mock import AsyncMock, MagicMock, patch

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

    Field naming follows the convention of the fixture variables they
    mirror (DS-667): ``tmpdir_cls`` matches ``mock_tmpdir_cls`` to
    clarify that it mocks the ``tempfile.TemporaryDirectory`` *class*,
    not a directory instance.
    """

    tmpdir_cls: MagicMock
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
def codex_config_with_timeout(request: pytest.FixtureRequest) -> Config:
    """Provide a Config with a custom subprocess_timeout for codex tests.

    Uses indirect parametrization to accept a timeout value via
    ``@pytest.mark.parametrize("codex_config_with_timeout", [120.0], indirect=True)``.
    The fixture creates a Config with codex_path="/usr/local/bin/codex",
    codex_default_model="o3-mini", and the requested subprocess_timeout.

    This reduces boilerplate in tests that need non-default timeout values
    while keeping the intent explicit through parametrize markers.
    """
    timeout: float = request.param
    return make_config(
        codex_path="/usr/local/bin/codex",
        codex_default_model="o3-mini",
        subprocess_timeout=timeout,
    )


@pytest.fixture
def codex_config_with_model(request: pytest.FixtureRequest) -> Config:
    """Provide a Config with a custom codex_default_model for codex tests.

    Uses indirect parametrization to accept a model name via
    ``@pytest.mark.parametrize("codex_config_with_model", ["gpt-4"], indirect=True)``.
    The fixture creates a Config with codex_path="/usr/local/bin/codex"
    and the requested codex_default_model.

    This reduces boilerplate in tests that need non-default model values
    while keeping the intent explicit through parametrize markers.
    """
    model: str = request.param
    return make_config(
        codex_path="/usr/local/bin/codex",
        codex_default_model=model,
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

    Yields a CodexSubprocessMocks NamedTuple with tmpdir_cls, output_path,
    run, and path_cls mocks. Common setup for tempfile.TemporaryDirectory,
    Path, and subprocess.run. The output file defaults to existing with
    content "Agent response". Tests can customize output_path and run as
    needed.
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
            tmpdir_cls=mock_tmpdir_cls,
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

    @pytest.mark.parametrize("codex_config_with_timeout", [120.0], indirect=True)
    def test_run_agent_uses_config_subprocess_timeout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config_with_timeout: Config,
    ) -> None:
        """Test that run_agent falls back to subprocess_timeout from config.

        When timeout_seconds is not provided and config.execution.subprocess_timeout
        is set, the config value should be used as the effective timeout. See also
        test_run_agent_no_timeout_when_subprocess_timeout_zero, which tests the
        special case where the config timeout is explicitly zero (disabled).
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config_with_timeout)

        asyncio.run(client.run_agent("prompt"))

        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] == 120

    @pytest.mark.parametrize("codex_config_with_timeout", [0.0], indirect=True)
    def test_run_agent_no_timeout_when_subprocess_timeout_zero(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config_with_timeout: Config,
    ) -> None:
        """Test that no timeout is applied when subprocess_timeout is zero.

        Unlike test_run_agent_uses_config_subprocess_timeout (which tests a
        non-zero config timeout being applied), this test verifies the edge
        case where subprocess_timeout is explicitly 0, meaning no timeout
        should be enforced (effective timeout remains None).
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config_with_timeout)

        asyncio.run(client.run_agent("prompt"))

        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] is None

    @pytest.mark.parametrize("codex_config_with_model", ["default-model"], indirect=True)
    def test_run_agent_with_model_override(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config_with_model: Config,
    ) -> None:
        """Test agent execution with model override."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config_with_model)

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
        """Test that response falls back to stdout when output file is empty.

        Covers the case where the output file exists but has zero bytes
        (st_size=0). See also test_run_agent_success_output_file_missing,
        which covers the case where the output file is never created at all.
        """
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

        Unlike test_run_agent_falls_back_to_stdout (which tests an existing but
        empty output file), this test covers the edge case where returncode=0
        but the output file was never created (exists() returns False). The
        response should fall back to stdout.
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

    def test_run_agent_context_sanitizes_newlines(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that context values with newlines are sanitized.

        Verifies that newline and carriage return characters in context
        keys and values are replaced with spaces to prevent prompt
        injection from untrusted sources (DS-666). See also
        test_run_agent_context_truncates_long_values, which tests the
        complementary length-based truncation of context keys and values.
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(
            client.run_agent(
                "test prompt",
                context={"key\ninjected": "value\r\nwith\nnewlines"},
            )
        )

        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        full_prompt = cmd[2]
        assert "\n\nContext:\n" in full_prompt
        # Newlines in keys/values should be replaced with spaces
        assert "key injected" in full_prompt
        assert "value with newlines" in full_prompt
        # No raw newlines in context values (only the structural ones)
        context_section = full_prompt.split("Context:\n")[1]
        lines = context_section.strip().split("\n")
        assert len(lines) == 1  # Only one context entry

    def test_run_agent_context_truncates_long_values(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test that overly long context values are truncated.

        Unlike test_run_agent_context_sanitizes_newlines (which tests
        character-level sanitization of newlines), this test verifies
        length-based truncation: context keys are capped at 200 characters
        and values at 2000 characters to prevent excessive prompt size
        from untrusted sources (DS-666).
        """
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        long_key = "k" * 500
        long_value = "v" * 5000

        asyncio.run(
            client.run_agent(
                "test prompt",
                context={long_key: long_value},
            )
        )

        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        full_prompt = cmd[2]
        context_section = full_prompt.split("Context:\n")[1]
        # Key should be truncated to 200 chars
        assert "k" * 200 in context_section
        assert "k" * 201 not in context_section
        # Value should be truncated to 2000 chars
        assert "v" * 2000 in context_section
        assert "v" * 2001 not in context_section


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


class _AsyncIterLines:
    """Async iterator over byte lines for mocking process.stderr.

    Simulates asyncio.subprocess.Process.stderr which yields ``bytes``
    objects (one per line) via ``async for raw_line in process.stderr``.
    """

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines
        self._index = 0

    def __aiter__(self) -> _AsyncIterLines:
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


@runtime_checkable
class MockProcess(Protocol):
    """Protocol documenting the expected interface of mock subprocess objects.

    Defines the contract that ``_make_mock_process()`` return values must
    satisfy, matching the subset of ``asyncio.subprocess.Process`` used by
    ``CodexAgentClient._run_with_log()``.  Using a Protocol instead of a
    bare MagicMock makes the expected attributes discoverable for future
    maintainers (DS-876).

    Attributes:
        returncode: Process exit code.
        stderr: Async-iterable of ``bytes`` lines (one per line).
        stdout: Object with an async ``read()`` method returning ``bytes``.
        wait: Awaitable that resolves when the process terminates.
        kill: Callable that sends SIGKILL to the process.
    """

    returncode: int
    stderr: _AsyncIterLines
    stdout: MagicMock
    wait: AsyncMock
    kill: MagicMock


class StreamingMocks(NamedTuple):
    """Container for mocks produced by ``_streaming_context()``.

    Provides named access to the mocks that every streaming-path test needs,
    eliminating positional destructuring and making intent self-documenting.
    """

    tmpdir_cls: MagicMock
    output_path: MagicMock
    path_cls: MagicMock


@contextmanager
def _streaming_context(
    mock_proc: MagicMock,
    *,
    output_exists: bool = True,
    output_size: int = 8,
    output_text: str = "Response",
) -> Iterator[StreamingMocks]:
    """Set up the TemporaryDirectory + Path mocks shared by streaming tests.

    Encapsulates the boilerplate that was previously repeated ~8 times across
    ``TestCodexStreamingLogs`` and ``TestCanStreamUseStreamingRouting`` (DS-876).

    Args:
        mock_proc: The mock process returned by ``_make_mock_process()``.
        output_exists: Whether the output file should report as existing.
        output_size: Value for ``output_path.stat().st_size``.
        output_text: Value returned by ``output_path.read_text()``.

    Yields:
        A ``StreamingMocks`` named-tuple with *tmpdir_cls*, *output_path*,
        and *path_cls* for any per-test customisation.
    """
    with (
        patch(
            "sentinel.agent_clients.codex.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=mock_proc),
        ),
        patch("sentinel.agent_clients.codex.tempfile.TemporaryDirectory") as mock_tmpdir_cls,
        patch("sentinel.agent_clients.codex.Path") as mock_path_cls,
    ):
        mock_tmpdir_ctx = MagicMock()
        mock_tmpdir_ctx.__enter__ = MagicMock(return_value="/tmp/codex_tmpdir")
        mock_tmpdir_ctx.__exit__ = MagicMock(return_value=False)
        mock_tmpdir_cls.return_value = mock_tmpdir_ctx

        mock_output_path = MagicMock()
        mock_output_path.exists.return_value = output_exists
        mock_output_path.stat.return_value = MagicMock(st_size=output_size)
        mock_output_path.read_text.return_value = output_text
        mock_path_cls.return_value = mock_output_path

        yield StreamingMocks(
            tmpdir_cls=mock_tmpdir_cls,
            output_path=mock_output_path,
            path_cls=mock_path_cls,
        )


def _make_mock_process(
    returncode: int = 0,
    stderr_lines: list[bytes] | None = None,
    stdout_data: bytes = b"",
) -> MagicMock:
    """Create an asyncio.subprocess.Process mock with configurable behaviour.

    The returned MagicMock satisfies the ``MockProcess`` protocol, ensuring
    it exposes the ``stderr``, ``stdout.read``, ``wait``, and ``kill``
    attributes consumed by ``CodexAgentClient._run_with_log()`` (DS-876).

    Args:
        returncode: Process exit code.
        stderr_lines: Byte lines to yield from ``async for line in process.stderr``.
        stdout_data: Bytes returned by ``process.stdout.read()``.

    Returns:
        A MagicMock that behaves like an asyncio subprocess Process.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = _AsyncIterLines(stderr_lines or [])
    proc.stdout = MagicMock()
    proc.stdout.read = AsyncMock(return_value=stdout_data)
    proc.wait = AsyncMock()
    proc.kill = MagicMock()
    return proc


class TestCodexStreamingLogs:
    """Tests for CodexAgentClient._run_with_log() streaming path.

    Covers the async subprocess streaming mode introduced in DS-872:
    log file creation, header writing, stderr streaming, response reading,
    timeout handling, execution summary, and circuit breaker integration.

    All tests use asyncio.run() for async test support and patch
    asyncio.create_subprocess_exec to avoid real process creation.
    """

    def test_streaming_creates_log_file(self, tmp_path: Path) -> None:
        """Verify that _run_with_log creates a log at logs/<orch>/YYYYMMDD_HHMMSS.log."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
        )

        output_content = "Streaming response"
        mock_proc = _make_mock_process(returncode=0)

        with _streaming_context(
            mock_proc,
            output_size=len(output_content),
            output_text=output_content,
        ):
            asyncio.run(
                client.run_agent(
                    "test prompt",
                    issue_key="TEST-100",
                    orchestration_name="test-orch",
                )
            )

        # The log directory structure should be logs/<orch>/<timestamp>.log
        orch_dir = log_dir / "test-orch"
        assert orch_dir.exists(), "Orchestration log directory should be created"
        log_files = list(orch_dir.glob("*.log"))
        assert len(log_files) == 1, "Exactly one log file should be created"
        assert log_files[0].name.endswith(".log")
        # Filename format: YYYYMMDD_HHMMSS.log
        stem = log_files[0].stem
        assert len(stem) == 15  # YYYYMMDD_HHMMSS
        assert stem[8] == "_"

    def test_streaming_writes_header(self, tmp_path: Path) -> None:
        """Verify log header contains issue key, orchestration name, and prompt."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        mock_proc = _make_mock_process(returncode=0)

        with _streaming_context(mock_proc):
            asyncio.run(
                client.run_agent(
                    "my test prompt",
                    issue_key="PROJ-42",
                    orchestration_name="review-orch",
                )
            )

        log_file = next((log_dir / "review-orch").glob("*.log"))
        log_content = log_file.read_text()

        assert "AGENT EXECUTION LOG" in log_content
        assert "Issue Key:      PROJ-42" in log_content
        assert "Orchestration:  review-orch" in log_content
        assert "PROMPT" in log_content
        assert "my test prompt" in log_content

    def test_streaming_writes_stderr_lines(self, tmp_path: Path) -> None:
        """Mock stderr and verify lines appear in the log file."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        stderr_lines = [
            b"Processing step 1...\n",
            b"Processing step 2...\n",
            b"Done.\n",
        ]
        mock_proc = _make_mock_process(returncode=0, stderr_lines=stderr_lines)

        with _streaming_context(mock_proc):
            asyncio.run(
                client.run_agent(
                    "test prompt",
                    issue_key="TEST-200",
                    orchestration_name="stderr-orch",
                )
            )

        log_file = next((log_dir / "stderr-orch").glob("*.log"))
        log_content = log_file.read_text()

        assert "Processing step 1..." in log_content
        assert "Processing step 2..." in log_content
        assert "Done." in log_content

    def test_streaming_reads_response_from_output_file(self, tmp_path: Path) -> None:
        """Response comes from --output-last-message file, not stderr."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        stderr_lines = [b"progress line\n"]
        mock_proc = _make_mock_process(
            returncode=0,
            stderr_lines=stderr_lines,
            stdout_data=b"stdout fallback data",
        )

        expected_response = "Output file response content"

        with _streaming_context(
            mock_proc,
            output_size=len(expected_response),
            output_text=expected_response,
        ) as mocks:
            result = asyncio.run(
                client.run_agent(
                    "test prompt",
                    issue_key="TEST-300",
                    orchestration_name="output-orch",
                )
            )

        # Response should come from the output file, NOT from stderr or stdout
        assert result.response == expected_response
        mocks.output_path.read_text.assert_called_once()

    def test_streaming_timeout_kills_process(self, tmp_path: Path) -> None:
        """On timeout, process.kill() is called and AgentTimeoutError is raised."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        mock_proc = _make_mock_process(returncode=0)

        # Make create_subprocess_exec return a process that causes timeout
        async def _slow_exec(*args: object, **kwargs: object) -> MagicMock:
            return mock_proc

        # Override stderr to be a slow iterator that causes timeout
        async def _slow_wait() -> None:
            await asyncio.sleep(100)  # Will be cancelled by wait_for

        mock_proc.wait = _slow_wait

        # Stderr needs to be an async iterator that also stalls
        class _StallIterator:
            def __aiter__(self) -> _StallIterator:
                return self

            async def __anext__(self) -> bytes:
                await asyncio.sleep(100)  # Will be cancelled by wait_for
                return b""

        mock_proc.stderr = _StallIterator()

        with (
            patch(
                "sentinel.agent_clients.codex.asyncio.create_subprocess_exec",
                new=AsyncMock(side_effect=_slow_exec),
            ),
            patch("sentinel.agent_clients.codex.tempfile.TemporaryDirectory") as mock_tmpdir_cls,
            patch("sentinel.agent_clients.codex.Path") as mock_path_cls,
        ):
            mock_tmpdir_ctx = MagicMock()
            mock_tmpdir_ctx.__enter__ = MagicMock(return_value="/tmp/codex_tmpdir")
            mock_tmpdir_ctx.__exit__ = MagicMock(return_value=False)
            mock_tmpdir_cls.return_value = mock_tmpdir_ctx

            mock_output_path = MagicMock()
            mock_path_cls.return_value = mock_output_path

            with pytest.raises(AgentTimeoutError) as exc_info:
                asyncio.run(
                    client.run_agent(
                        "test prompt",
                        issue_key="TEST-400",
                        orchestration_name="timeout-orch",
                        timeout_seconds=1,
                    )
                )

        assert "timed out" in str(exc_info.value)
        mock_proc.kill.assert_called_once()

    def test_streaming_writes_summary_on_success(self, tmp_path: Path) -> None:
        """Verify execution summary is appended on successful completion."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        mock_proc = _make_mock_process(returncode=0)

        with _streaming_context(mock_proc):
            asyncio.run(
                client.run_agent(
                    "test prompt",
                    issue_key="TEST-500",
                    orchestration_name="summary-orch",
                )
            )

        log_file = next((log_dir / "summary-orch").glob("*.log"))
        log_content = log_file.read_text()

        assert "EXECUTION SUMMARY" in log_content
        assert "Status:         COMPLETED" in log_content
        assert "END OF LOG" in log_content
        assert "Duration:" in log_content
        assert "METRICS JSON" in log_content

    def test_streaming_writes_summary_on_failure(self, tmp_path: Path) -> None:
        """Verify error summary is appended when the agent process fails."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(config, log_base_dir=log_dir)

        mock_proc = _make_mock_process(
            returncode=1,
            stdout_data=b"error detail from stdout",
        )

        with _streaming_context(mock_proc, output_exists=False):
            with pytest.raises(AgentClientError) as exc_info:
                asyncio.run(
                    client.run_agent(
                        "test prompt",
                        issue_key="TEST-600",
                        orchestration_name="failure-orch",
                    )
                )

        assert "Codex CLI execution failed" in str(exc_info.value)

        log_file = next((log_dir / "failure-orch").glob("*.log"))
        log_content = log_file.read_text()

        assert "EXECUTION SUMMARY" in log_content
        assert "Status:         ERROR" in log_content
        assert "END OF LOG" in log_content

    def test_streaming_disabled_uses_simple_path(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """When disable_streaming_logs=True, run_agent uses subprocess.run (simple path)."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = Path("/tmp/test-logs-disabled")
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            disable_streaming_logs=True,
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-700",
                orchestration_name="disabled-orch",
            )
        )

        # Should use the simple path (subprocess.run) instead of streaming
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_streaming_fallback_when_no_log_dir(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """When log_base_dir is None, run_agent falls back to the simple path."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        # No log_base_dir => cannot use streaming
        client = CodexAgentClient(config, log_base_dir=None)

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-800",
                orchestration_name="no-logdir-orch",
            )
        )

        # Should use the simple path (subprocess.run)
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_streaming_circuit_breaker_records(self, tmp_path: Path) -> None:
        """Circuit breaker records success and failure in the streaming path."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        # --- Test success recording ---
        cb_success = CircuitBreaker(service_name="codex-stream-success")
        client_success = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            circuit_breaker=cb_success,
        )

        mock_proc_ok = _make_mock_process(returncode=0)

        with _streaming_context(mock_proc_ok):
            asyncio.run(
                client_success.run_agent(
                    "test prompt",
                    issue_key="TEST-900",
                    orchestration_name="cb-success-orch",
                )
            )

        assert cb_success.metrics.successful_calls == 1
        assert cb_success.metrics.failed_calls == 0

        # --- Test failure recording ---
        cb_failure = CircuitBreaker(service_name="codex-stream-failure")
        client_failure = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            circuit_breaker=cb_failure,
        )

        mock_proc_fail = _make_mock_process(
            returncode=1,
            stdout_data=b"error output",
        )

        with _streaming_context(mock_proc_fail, output_exists=False):
            with pytest.raises(AgentClientError):
                asyncio.run(
                    client_failure.run_agent(
                        "test prompt",
                        issue_key="TEST-901",
                        orchestration_name="cb-failure-orch",
                    )
                )

        assert cb_failure.metrics.failed_calls == 1
        assert cb_failure.metrics.successful_calls == 0


class TestDisableStreamingLogsConstructor:
    """Tests for _disable_streaming_logs constructor parameter (DS-874).

    Verifies the three-way logic: explicit True, explicit False, and None
    (fall back to config.execution.disable_streaming_logs).
    """

    def test_disable_streaming_logs_explicit_true(self) -> None:
        """Test that explicit disable_streaming_logs=True overrides config."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            disable_streaming_logs=False,  # config says False
        )
        client = CodexAgentClient(config, disable_streaming_logs=True)

        assert client._disable_streaming_logs is True

    def test_disable_streaming_logs_explicit_false(self) -> None:
        """Test that explicit disable_streaming_logs=False overrides config."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            disable_streaming_logs=True,  # config says True
        )
        client = CodexAgentClient(config, disable_streaming_logs=False)

        assert client._disable_streaming_logs is False

    def test_disable_streaming_logs_none_falls_back_to_config_true(self) -> None:
        """Test that None falls back to config value (True case)."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            disable_streaming_logs=True,
        )
        client = CodexAgentClient(config, disable_streaming_logs=None)

        assert client._disable_streaming_logs is True

    def test_disable_streaming_logs_none_falls_back_to_config_false(self) -> None:
        """Test that None falls back to config value (False case)."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            disable_streaming_logs=False,
        )
        client = CodexAgentClient(config, disable_streaming_logs=None)

        assert client._disable_streaming_logs is False

    def test_disable_streaming_logs_default_omitted(self) -> None:
        """Test that omitting disable_streaming_logs falls back to config default."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            disable_streaming_logs=False,
        )
        # Don't pass disable_streaming_logs at all (defaults to None)
        client = CodexAgentClient(config)

        assert client._disable_streaming_logs is False


class TestRunSimpleDirect:
    """Tests for CodexAgentClient._run_simple() called directly (DS-874).

    These tests exercise _run_simple() in isolation, verifying its behaviour
    without going through run_agent(). This covers the subprocess execution
    path, output file reading, fallback to stdout, error handling, and
    circuit breaker recording.
    """

    def test_run_simple_success(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple returns response from output file on success."""
        client = CodexAgentClient(codex_config)

        result = asyncio.run(client._run_simple("test prompt", None, None, None))

        assert result == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_run_simple_with_timeout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple passes timeout to subprocess.run."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(client._run_simple("prompt", timeout=300, workdir=None, model=None))

        call_kwargs = mock_codex_subprocess.run.call_args[1]
        assert call_kwargs["timeout"] == 300

    def test_run_simple_with_model(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple passes model to _build_command."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)

        asyncio.run(client._run_simple("prompt", timeout=None, workdir=None, model="gpt-4o"))

        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        model_idx = cmd.index("--model") + 1
        assert cmd[model_idx] == "gpt-4o"

    def test_run_simple_with_workdir(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple passes workdir to _build_command."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=8)
        mock_codex_subprocess.output_path.read_text.return_value = "Response"
        mock_codex_subprocess.run.return_value = MagicMock(returncode=0, stdout="Response", stderr="")

        client = CodexAgentClient(codex_config)
        workdir = Path("/tmp/test-workdir")

        asyncio.run(client._run_simple("prompt", timeout=None, workdir=workdir, model=None))

        call_args = mock_codex_subprocess.run.call_args
        cmd = call_args[0][0]
        assert "--cd" in cmd
        assert "/tmp/test-workdir" in cmd

    def test_run_simple_falls_back_to_stdout(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple falls back to stdout when output file is empty."""
        mock_codex_subprocess.output_path.stat.return_value = MagicMock(st_size=0)
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="Stdout response",
            stderr="",
        )

        client = CodexAgentClient(codex_config)

        result = asyncio.run(client._run_simple("prompt", None, None, None))

        assert result == "Stdout response"

    def test_run_simple_nonzero_exit_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple raises AgentClientError on non-zero exit code."""
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: command failed",
        )

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client._run_simple("prompt", None, None, None))

        assert "Codex CLI execution failed" in str(exc_info.value)
        assert "command failed" in str(exc_info.value)

    def test_run_simple_timeout_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple raises AgentTimeoutError on subprocess timeout."""
        mock_codex_subprocess.run.side_effect = subprocess.TimeoutExpired(cmd="codex", timeout=120)

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentTimeoutError) as exc_info:
            asyncio.run(client._run_simple("prompt", timeout=120, workdir=None, model=None))

        assert "timed out after 120s" in str(exc_info.value)

    def test_run_simple_file_not_found_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple raises AgentClientError on FileNotFoundError."""
        mock_codex_subprocess.run.side_effect = FileNotFoundError("codex not found")

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client._run_simple("prompt", None, None, None))

        assert "Codex CLI executable not found" in str(exc_info.value)

    def test_run_simple_os_error_raises_error(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple raises AgentClientError on OSError."""
        mock_codex_subprocess.run.side_effect = OSError("Permission denied")

        client = CodexAgentClient(codex_config)

        with pytest.raises(AgentClientError) as exc_info:
            asyncio.run(client._run_simple("prompt", None, None, None))

        assert "Failed to execute Codex CLI" in str(exc_info.value)

    def test_run_simple_records_success_on_circuit_breaker(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple records success on the circuit breaker."""
        cb = CircuitBreaker(service_name="codex-direct")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        asyncio.run(client._run_simple("prompt", None, None, None))

        assert cb.metrics.successful_calls == 1
        assert cb.metrics.failed_calls == 0

    def test_run_simple_records_failure_on_circuit_breaker(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple records failure on the circuit breaker."""
        mock_codex_subprocess.output_path.exists.return_value = False
        mock_codex_subprocess.run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: service failure",
        )

        cb = CircuitBreaker(service_name="codex-direct")
        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AgentClientError):
            asyncio.run(client._run_simple("prompt", None, None, None))

        assert cb.metrics.failed_calls == 1
        assert cb.metrics.successful_calls == 0

    def test_run_simple_asserts_circuit_breaker_not_open(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
        codex_config: Config,
    ) -> None:
        """Test _run_simple assertion fails when circuit breaker is open.

        This validates the defensive assert that prevents _run_simple from being
        called directly when the circuit breaker is open (it should only be called
        via run_agent which checks the circuit breaker first).
        """
        cb = CircuitBreaker(
            service_name="codex",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure(Exception("test failure"))
        assert cb.state == CircuitState.OPEN

        client = CodexAgentClient(codex_config, circuit_breaker=cb)

        with pytest.raises(AssertionError, match="Circuit breaker is OPEN"):
            asyncio.run(client._run_simple("prompt", None, None, None))


class TestCanStreamUseStreamingRouting:
    """Tests for can_stream / use_streaming routing logic in run_agent() (DS-874).

    Verifies that run_agent correctly routes to _run_simple() vs _run_with_log()
    based on the combination of log_base_dir, issue_key, orchestration_name,
    and _disable_streaming_logs.
    """

    def test_routes_to_streaming_when_all_conditions_met(
        self,
        tmp_path: Path,
    ) -> None:
        """Test run_agent routes to _run_with_log when all streaming conditions met.

        Streaming requires: log_base_dir, issue_key, orchestration_name all provided
        AND _disable_streaming_logs is False.
        """
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            disable_streaming_logs=False,
        )

        mock_proc = _make_mock_process(returncode=0)

        with _streaming_context(mock_proc):
            result = asyncio.run(
                client.run_agent(
                    "test prompt",
                    issue_key="TEST-ROUTE-1",
                    orchestration_name="test-orch",
                )
            )

        # Should have used the async subprocess path (streaming)
        assert result.response == "Response"
        # Verify streaming log was created
        orch_dir = log_dir / "test-orch"
        assert orch_dir.exists()

    def test_routes_to_simple_when_no_issue_key(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test run_agent routes to _run_simple when issue_key is None."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = Path("/tmp/test-logs")
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            disable_streaming_logs=False,
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key=None,  # Missing issue_key
                orchestration_name="test-orch",
            )
        )

        # Should use simple path (subprocess.run)
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_routes_to_simple_when_no_orchestration_name(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test run_agent routes to _run_simple when orchestration_name is None."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = Path("/tmp/test-logs")
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            disable_streaming_logs=False,
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-ROUTE-2",
                orchestration_name=None,  # Missing orchestration_name
            )
        )

        # Should use simple path (subprocess.run)
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_routes_to_simple_when_no_log_dir(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test run_agent routes to _run_simple when log_base_dir is None."""
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        client = CodexAgentClient(
            config,
            log_base_dir=None,  # No log directory
            disable_streaming_logs=False,
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-ROUTE-3",
                orchestration_name="test-orch",
            )
        )

        # Should use simple path (subprocess.run)
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_routes_to_simple_when_streaming_disabled(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test run_agent routes to _run_simple when streaming is disabled.

        Even when all streaming conditions (log_base_dir, issue_key,
        orchestration_name) are met, if _disable_streaming_logs is True,
        the simple path should be used.
        """
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
        )
        log_dir = Path("/tmp/test-logs")
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            disable_streaming_logs=True,  # Streaming disabled
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-ROUTE-4",
                orchestration_name="test-orch",
            )
        )

        # Should use simple path (subprocess.run) despite all conditions met
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()

    def test_routes_to_simple_when_config_disables_streaming(
        self,
        mock_codex_subprocess: CodexSubprocessMocks,
    ) -> None:
        """Test run_agent routes to _run_simple when config disables streaming.

        When disable_streaming_logs is not passed to constructor (defaults to None),
        the config.execution.disable_streaming_logs value is used. If it's True,
        the simple path should be used.
        """
        config = make_config(
            codex_path="/usr/local/bin/codex",
            codex_default_model="o3-mini",
            disable_streaming_logs=True,  # Config disables streaming
        )
        log_dir = Path("/tmp/test-logs")
        client = CodexAgentClient(
            config,
            log_base_dir=log_dir,
            # disable_streaming_logs not passed, falls back to config
        )

        result = asyncio.run(
            client.run_agent(
                "test prompt",
                issue_key="TEST-ROUTE-5",
                orchestration_name="test-orch",
            )
        )

        # Should use simple path (subprocess.run) because config disables streaming
        assert result.response == "Agent response"
        mock_codex_subprocess.run.assert_called_once()
