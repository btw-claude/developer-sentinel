"""Codex CLI agent client implementation.

This module provides the Codex CLI client for running agents via subprocess,
supporting model selection and working directory configuration.

The run_agent method is async to conform to the AgentClient interface,
using asyncio.to_thread() for subprocess operations.

Circuit breaker pattern is implemented to prevent cascading failures
when the Codex CLI is experiencing issues.

Working directory strategy:
The ``--cd`` flag is used instead of ``subprocess(cwd=...)`` because Codex CLI
natively supports ``--cd`` to set its own working directory context. This keeps
the working-directory concern inside the child process and avoids changing the
parent process's environment. The ``cwd`` subprocess parameter would achieve the
same filesystem effect but ``--cd`` makes the intent explicit in the CLI
invocation and ensures Codex's internal path resolution is consistent with the
directory it reports. Future maintainers should prefer ``--cd`` for Codex
commands unless there is a need to override the subprocess-level working
directory independently.

Reference: https://developers.openai.com/codex/cli/reference
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
)
from sentinel.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from sentinel.config import Config
from sentinel.logging import get_logger

logger = get_logger(__name__)


class CodexAgentClient(AgentClient):
    """Agent client that uses Codex CLI via subprocess.

    This client runs the Codex CLI tool to execute agent tasks using
    ``codex exec "<prompt>" --full-auto --output-last-message <tmpfile>``
    via subprocess. It supports:

    - Model selection via ``--model`` flag
    - Working directory via ``--cd`` flag
    - Timeout handling via subprocess timeout
    - Working directory creation similar to CursorAgentClient

    Configuration is provided via the Config dataclass:

    - codex.path: Path to the Codex CLI executable
    - codex.default_model: Default model to use

    Example:
        config = Config(codex=CodexConfig(path="/usr/local/bin/codex"))
        client = CodexAgentClient(config)
        result = await client.run_agent("Write a hello world function")
    """

    def __init__(
        self,
        config: Config,
        base_workdir: Path | None = None,
        log_base_dir: Path | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize the Codex agent client.

        Args:
            config: Configuration object with Codex settings.
            base_workdir: Base directory for creating agent working directories.
            log_base_dir: Base directory for agent execution logs (reserved for future use).
            circuit_breaker: Circuit breaker instance for resilience. If not provided,
                creates a default circuit breaker for the "codex" service.
        """
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir
        # Use provided circuit breaker or create a default one for the codex service
        self._circuit_breaker = circuit_breaker or CircuitBreaker(
            service_name="codex",
            config=CircuitBreakerConfig.from_env("codex"),
        )

        # Validate and store codex path
        self._codex_path = config.codex.path or "codex"
        self._default_model = config.codex.default_model or None

        if not config.codex.path:
            logger.info(
                "codex_path not configured; falling back to 'codex' (PATH lookup). "
                "Set codex.path in config to use an explicit executable path."
            )

        logger.debug(
            "Initialized CodexAgentClient with path=%s, model=%s",
            self._codex_path,
            self._default_model,
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return AgentType.CODEX

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get the circuit breaker for this client."""
        return self._circuit_breaker

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        """Get current circuit breaker status.

        Returns:
            Dictionary with circuit breaker state, config, and metrics.
        """
        return self._circuit_breaker.get_status()

    def _build_command(
        self,
        prompt: str,
        output_file: str,
        model: str | None = None,
        workdir: Path | None = None,
    ) -> list[str]:
        """Build the Codex CLI command.

        Constructs a command of the form::

            codex exec "<prompt>" --full-auto --output-last-message <tmpfile>
                [--model <model>] [--cd <workdir>]

        Args:
            prompt: The prompt to send to the agent.
            output_file: Path to the temporary file for capturing the last message.
            model: Model identifier (overrides default if provided).
            workdir: Working directory for Codex execution.

        Returns:
            List of command arguments for subprocess.
        """
        effective_model = model if model is not None else self._default_model

        cmd = [
            self._codex_path,
            "exec",
            prompt,
            "--full-auto",
            "--output-last-message",
            output_file,
        ]

        if effective_model:
            cmd.extend(["--model", effective_model])

        if workdir is not None:
            cmd.extend(["--cd", str(workdir)])

        return cmd

    async def run_agent(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        issue_key: str | None = None,
        model: str | None = None,
        orchestration_name: str | None = None,
        branch: str | None = None,
        create_branch: bool = False,
        base_branch: str = "main",
        agent_teams: bool = False,
    ) -> AgentRunResult:
        """Run a Codex agent with the given prompt.

        This is an async method that uses asyncio.to_thread() to run the
        subprocess without blocking the event loop.

        The Codex CLI is invoked as::

            codex exec "<prompt>" --full-auto --output-last-message <tmpfile>

        The agent response is read from the temporary output file after
        execution completes.

        Args:
            prompt: The prompt to send to the agent.
            context: Optional context dict to append to prompt.
            timeout_seconds: Optional timeout in seconds. If None, uses
                subprocess_timeout from config, or no timeout if that is also unset.
            issue_key: Optional issue key for creating a unique working directory.
            model: Optional model identifier. If None, uses the client's default model.
            orchestration_name: Optional orchestration name (reserved for future logging).
            branch: Optional branch name (reserved for future implementation).
            create_branch: If True and branch doesn't exist, create it (reserved for future).
            base_branch: Base branch to create new branches from (reserved for future).
            agent_teams: Whether to enable Claude Code's experimental Agent Teams feature
                (ignored for Codex client).

        Returns:
            AgentRunResult containing the agent's response text and optional working
            directory path.

        Raises:
            AgentClientError: If the agent execution fails (non-zero exit code).
            AgentTimeoutError: If the agent execution times out.
        """
        # Check circuit breaker before attempting the request
        # This is done first to avoid unnecessary disk I/O from _create_workdir()
        # when the circuit is open (consistent with ClaudeSdkAgentClient.run_agent)
        if not self._circuit_breaker.allow_request():
            raise AgentClientError(
                f"Codex circuit breaker is open - service may be unavailable. "
                f"State: {self._circuit_breaker.state.value}"
            )

        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key)

        # Build full prompt with sanitized context section (DS-675).
        full_prompt = self._build_prompt_with_context(prompt, context)

        # Use subprocess_timeout from config as fallback when timeout_seconds is not provided
        effective_timeout = timeout_seconds
        if effective_timeout is None and self.config.execution.subprocess_timeout > 0:
            effective_timeout = int(self.config.execution.subprocess_timeout)

        logger.info(
            "Running Codex CLI: model=%s, timeout=%ss",
            model or self._default_model or "default",
            effective_timeout,
        )

        try:
            # Use TemporaryDirectory for automatic cleanup instead of manual
            # mkstemp/unlink. The directory (and everything inside it) is removed
            # when the context manager exits, even on exceptions.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = str(Path(tmp_dir) / "codex_output.txt")

                def _run_subprocess() -> subprocess.CompletedProcess[str]:
                    """Run subprocess in a thread to avoid blocking the event loop."""
                    cmd = self._build_command(
                        full_prompt, output_file=tmp_path, model=model, workdir=workdir
                    )
                    logger.debug("Codex command: %s...", " ".join(cmd[:3]))
                    return subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=effective_timeout,
                    )

                result = await asyncio.to_thread(_run_subprocess)

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or f"Exit code: {result.returncode}"
                    logger.error("Codex CLI failed: %s", error_msg)
                    error = AgentClientError(f"Codex CLI execution failed: {error_msg}")
                    self._circuit_breaker.record_failure(error)
                    raise error

                # Read the response from the output file
                output_path = Path(tmp_path)
                if output_path.exists() and output_path.stat().st_size > 0:
                    response = output_path.read_text().strip()
                else:
                    # Fall back to stdout if output file is empty
                    response = result.stdout.strip()

                self._circuit_breaker.record_success()
                logger.info("Codex execution completed, response length: %s", len(response))

                return AgentRunResult(response=response, workdir=workdir)

        except subprocess.TimeoutExpired:
            logger.debug(
                "Codex subprocess timeout: timeout=%ss, workdir=%s",
                effective_timeout,
                workdir,
            )
            logger.error("Codex CLI timed out after %ss", effective_timeout)
            msg = f"Codex agent execution timed out after {effective_timeout}s"
            timeout_error = AgentTimeoutError(msg)
            self._circuit_breaker.record_failure(timeout_error)
            raise timeout_error from None
        except FileNotFoundError:
            logger.error("Codex CLI not found at path: %s", self._codex_path)
            msg = f"Codex CLI executable not found: {self._codex_path}"
            not_found_error = AgentClientError(msg)
            self._circuit_breaker.record_failure(not_found_error)
            raise not_found_error from None
        except OSError as e:
            logger.error("Failed to execute Codex CLI: %s", e)
            self._circuit_breaker.record_failure(e)
            raise AgentClientError(f"Failed to execute Codex CLI: {e}") from e
