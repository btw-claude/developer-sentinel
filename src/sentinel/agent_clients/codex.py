"""Codex CLI agent client implementation.

This module provides the Codex CLI client for running agents via subprocess,
supporting model selection and working directory configuration.

The run_agent method is async to conform to the AgentClient interface,
using asyncio.to_thread() for subprocess operations.

Reference: https://developers.openai.com/codex/cli/reference
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.agent_clients.base import (
    AgentClient,
    AgentClientError,
    AgentRunResult,
    AgentTimeoutError,
    AgentType,
)
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
    ) -> None:
        """Initialize the Codex agent client.

        Args:
            config: Configuration object with Codex settings.
            base_workdir: Base directory for creating agent working directories.
            log_base_dir: Base directory for agent execution logs (reserved for future use).
        """
        self.config = config
        self.base_workdir = base_workdir
        self.log_base_dir = log_base_dir

        # Validate and store codex path
        self._codex_path = config.codex.path or "codex"
        self._default_model = config.codex.default_model or None

        logger.debug(
            "Initialized CodexAgentClient with path=%s, model=%s",
            self._codex_path,
            self._default_model,
        )

    @property
    def agent_type(self) -> AgentType:
        """Return the type of agent this client implements."""
        return AgentType.CODEX

    def _create_workdir(self, issue_key: str) -> Path:
        """Create a unique working directory for an agent run.

        Args:
            issue_key: The issue key to include in the directory name.

        Returns:
            Path to the created working directory.

        Raises:
            AgentClientError: If base_workdir is not configured.
        """
        if self.base_workdir is None:
            raise AgentClientError("base_workdir not configured")

        self.base_workdir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = self.base_workdir / f"{issue_key}_{timestamp}"
        workdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created agent working directory: %s", workdir)
        return workdir

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

        Returns:
            AgentRunResult containing the agent's response text and optional working
            directory path.

        Raises:
            AgentClientError: If the agent execution fails (non-zero exit code).
            AgentTimeoutError: If the agent execution times out.
        """
        workdir = None
        if self.base_workdir is not None and issue_key is not None:
            workdir = self._create_workdir(issue_key)

        # Build full prompt with context section
        full_prompt = prompt
        if context:
            full_prompt += "\n\nContext:\n" + "".join(f"- {k}: {v}\n" for k, v in context.items())

        # Use subprocess_timeout from config as fallback when timeout_seconds is not provided
        effective_timeout = timeout_seconds
        if effective_timeout is None and self.config.execution.subprocess_timeout > 0:
            effective_timeout = int(self.config.execution.subprocess_timeout)

        logger.info(
            "Running Codex CLI: model=%s, timeout=%ss",
            model or self._default_model or "default",
            effective_timeout,
        )

        def _run_subprocess(tmp_path: str) -> subprocess.CompletedProcess[str]:
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

        try:
            # Create a temporary file path for capturing the last message output
            fd, tmp_path = tempfile.mkstemp(suffix=".txt")
            os.close(fd)

            try:
                result = await asyncio.to_thread(_run_subprocess, tmp_path)

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or f"Exit code: {result.returncode}"
                    logger.error("Codex CLI failed: %s", error_msg)
                    raise AgentClientError(f"Codex CLI execution failed: {error_msg}")

                # Read the response from the output file
                output_path = Path(tmp_path)
                if output_path.exists() and output_path.stat().st_size > 0:
                    response = output_path.read_text().strip()
                else:
                    # Fall back to stdout if output file is empty
                    response = result.stdout.strip()

                logger.info("Codex execution completed, response length: %s", len(response))

                return AgentRunResult(response=response, workdir=workdir)
            finally:
                # Clean up the temporary file
                tmp_output = Path(tmp_path)
                if tmp_output.exists():
                    tmp_output.unlink()

        except subprocess.TimeoutExpired:
            logger.debug(
                "Codex subprocess timeout: timeout=%ss, workdir=%s",
                effective_timeout,
                workdir,
            )
            logger.error("Codex CLI timed out after %ss", effective_timeout)
            msg = f"Codex agent execution timed out after {effective_timeout}s"
            raise AgentTimeoutError(msg) from None
        except FileNotFoundError:
            logger.error("Codex CLI not found at path: %s", self._codex_path)
            msg = f"Codex CLI executable not found: {self._codex_path}"
            raise AgentClientError(msg) from None
        except OSError as e:
            logger.error("Failed to execute Codex CLI: %s", e)
            raise AgentClientError(f"Failed to execute Codex CLI: {e}") from e
